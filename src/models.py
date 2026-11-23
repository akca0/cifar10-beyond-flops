"""The 5 compared models (from simple to complex):

    MLP          - fully connected network, ignores spatial (neighbourhood) structure
    SmallCNN     - small CNN with 2 convolution layers
    MediumCNN    - VGG-style CNN with 6 convolution layers
    MobileNetV2  - CNN designed for efficiency (depthwise separable convolutions)
    ResNet18     - 18-layer deep CNN with skip (residual) connections

ResNet-18 and MobileNetV2 are the architectures of the original papers adapted
to the 32x32 images of CIFAR-10 (fewer downsampling steps in the first layers;
otherwise a 32x32 image would shrink to 1x1 halfway through the network).

Theoretical cost notes are in each class docstring. Abbreviations:
    C_in, C_out : input/output channels,  H x W : feature-map size,
    k : filter size (k=3 for 3x3),  MAC : one multiply + one add
"""
import torch
from torch import nn


# --------------------------------------------------------------------------- MLP
class MLP(nn.Module):
    """Multi-Layer Perceptron.

        input 3072 -> Linear -> 512 -> Linear -> 256 -> Linear -> 10

    A Linear(n_in, n_out) layer: n_in*n_out MACs and n_in*n_out + n_out parameters.
    Total cost O(sum_l n_{l-1} * n_l); 92% of the parameters are in the first layer.
    """

    def __init__(self, in_features: int = 3 * 32 * 32, hidden=(512, 256), num_classes: int = 10,
                 batchnorm: bool = False, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Flatten()]          # (B, 3, 32, 32) -> (B, 3072)
        prev = in_features
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batchnorm:   # normalises each layer's output -> more stable, faster learning
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:  # randomly switches off some neurons during training -> harder to memorise
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))  # output: a score (logit) for each of the 10 classes
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- plain CNNs
def conv_bn_relu(c_in: int, c_out: int, stride: int = 1) -> nn.Sequential:
    """3x3 convolution + BatchNorm + ReLU.

    Cost of Conv2d(C_in, C_out, k=3): H*W*C_in*C_out*9 MACs, C_in*C_out*9 parameters.
    Note: the parameter count does NOT depend on the image size, but the operation count grows with H*W.
    (Unlike the MLP, the same small filter is reused at every position of the image.)
    """
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class SmallCNN(nn.Module):
    """2 convolution layers: 3 -> 32 -> 64 channels, each followed by 2x2 pooling.

        32x32x3 -> conv -> 32x32x32 -> pool -> 16x16x32 -> conv -> 16x16x64 -> pool
        -> 8x8x64 -> global average -> 64 -> Linear -> 10
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_relu(3, 32), nn.MaxPool2d(2),
            conv_bn_relu(32, 64), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


class MediumCNN(nn.Module):
    """VGG style: 3 blocks of 2 convolutions each; the channel count doubles in every block.

        [32, 32] -> pool -> [64, 64] -> pool -> [128, 128] -> pool -> global average -> Linear
    """

    def __init__(self, num_classes: int = 10, width: int = 32):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            conv_bn_relu(3, w), conv_bn_relu(w, w), nn.MaxPool2d(2),
            conv_bn_relu(w, 2 * w), conv_bn_relu(2 * w, 2 * w), nn.MaxPool2d(2),
            conv_bn_relu(2 * w, 4 * w), conv_bn_relu(4 * w, 4 * w), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4 * w, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


# --------------------------------------------------------------------------- ResNet-18
class BasicBlock(nn.Module):
    """Residual block: output = F(x) + x.

    Thanks to the "+ x" (skip connection), signals and gradients can bypass
    layers, so very deep networks can be trained too (He et al., 2016).
    """

    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != c_out:  # if the shape changes, match x with a 1x1 conv
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                                          nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    """ResNet-18 (CIFAR variant): 1 stem conv + 4 stages x 2 blocks x 2 convs + 1 Linear = 18 layers.

        stage:     1      2      3      4
        channels:  64     128    256    512
        size:      32x32  16x16  8x8    4x4
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        layers, c_in = [], 64
        for c_out, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
            layers += [BasicBlock(c_in, c_out, stride), BasicBlock(c_out, c_out, 1)]
            c_in = c_out
        self.layers = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, num_classes))

    def forward(self, x):
        return self.head(self.layers(self.stem(x)))


# --------------------------------------------------------------------------- MobileNetV2
class InvertedResidual(nn.Module):
    """MobileNetV2 block (Sandler et al., 2018):

        expand with 1x1 conv (t times) -> 3x3 DEPTHWISE conv -> project with 1x1 conv (+ skip connection)

    A depthwise conv filters every channel SEPARATELY: cost H*W*C*9 MACs.
    A normal 3x3 conv costs H*W*C_in*C_out*9 MACs -> roughly C_out times cheaper.
    But on a GPU "fewer operations" does not always mean "less time" (memory
    access becomes dominant) - this is what the experiments test.
    """

    def __init__(self, c_in: int, c_out: int, stride: int, expand: int):
        super().__init__()
        hidden = c_in * expand
        self.use_res = stride == 1 and c_in == c_out
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(c_in, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),  # depthwise
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, c_out, 1, bias=False), nn.BatchNorm2d(c_out),        # pointwise (project)
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class MobileNetV2(nn.Module):
    """MobileNetV2 (CIFAR variant: stride 1 instead of 2 in the first conv and in stage 2)."""

    # (expansion t, output channels c, number of blocks n, stride of the first block s)
    CFG = ((1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
           (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1))

    def __init__(self, num_classes: int = 10):
        super().__init__()
        layers = [nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU6(inplace=True)]
        c_in = 32
        for t, c, n, s in self.CFG:
            for i in range(n):
                layers.append(InvertedResidual(c_in, c, s if i == 0 else 1, t))
                c_in = c
        layers += [nn.Conv2d(c_in, 1280, 1, bias=False), nn.BatchNorm2d(1280), nn.ReLU6(inplace=True)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1280, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


# --------------------------------------------------------------------------- helpers
MODELS = {
    "MLP": lambda: MLP(batchnorm=True, dropout=0.3),
    "SmallCNN": SmallCNN,
    "MediumCNN": MediumCNN,
    "MobileNetV2": MobileNetV2,
    "ResNet18": ResNet18,
}


def build_model(name: str) -> nn.Module:
    return MODELS[name]()


def count_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def count_flops(model: nn.Module, input_size=(1, 3, 32, 32)) -> int:
    """FLOPs of one forward pass for a single image (PyTorch's own counter).
    1 MAC = 2 FLOPs (one multiply + one add)."""
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with FlopCounterMode(display=False) as fc:
        model(torch.zeros(input_size, device=device))
    model.train(was_training)
    return fc.get_total_flops()
