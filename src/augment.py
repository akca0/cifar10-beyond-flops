"""Data augmentation on the GPU.

Augmentation: slightly change the images at random in every epoch so the model
sees "new" examples. Because the model never sees exactly the same image twice,
memorising the training set becomes harder (less overfitting).

The classic approach augments image by image on the CPU, which ties the timing
measurements to CPU speed. Here the whole batch is augmented at once on the GPU:
it takes a few milliseconds and adds the same small overhead for every model.

Two operations (standard for CIFAR-10):
1. Random horizontal flip: mirror the image with probability 0.5
2. Random crop: pad by 4 pixels and take a random 32x32 window
   (the object is shifted by a few pixels)
"""
import torch
import torch.nn.functional as F


def gpu_augment(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """x: (B, C, H, W) tensor on the GPU. Returns an augmented batch of the same shape."""
    b, c, h, w = x.shape
    dev = x.device

    # 1) Random horizontal flip
    flip = torch.rand(b, device=dev) < 0.5
    x = torch.where(flip.view(b, 1, 1, 1), x.flip(dims=[3]), x)

    # 2) Random crop: an individual (dy, dx) shift for every image
    padded = F.pad(x, (pad, pad, pad, pad))                    # (B, C, H+2p, W+2p), zero borders
    dy = torch.randint(0, 2 * pad + 1, (b,), device=dev)
    dx = torch.randint(0, 2 * pad + 1, (b,), device=dev)
    rows = (dy.view(b, 1) + torch.arange(h, device=dev)).view(b, 1, h, 1)
    cols = (dx.view(b, 1) + torch.arange(w, device=dev)).view(b, 1, 1, w)
    bi = torch.arange(b, device=dev).view(b, 1, 1, 1)
    ci = torch.arange(c, device=dev).view(1, c, 1, 1)
    return padded[bi, ci, rows, cols]
