"""CIFAR-10 data module.

Every experiment gets its data from here, so every model sees the SAME data,
the SAME preprocessing and the SAME subsets (fair-comparison requirement).

Design decisions:
- The data is loaded into memory once (uint8 tensor, ~150 MB), so no disk reads
  happen during an epoch and disk speed does not leak into the timings.
- The data stays on the CPU; only the current batch is sent to the GPU. Putting
  the whole dataset on the GPU would inflate the VRAM measurements.
- No augmentation happens here. Augmentation (random flip + crop) is applied to
  each batch on the GPU during training (see augment.py), so it does not put
  extra work on the CPU.
- The subsets (10%, 25%, 50%, 100%) are nested and stratified: the 10% subset
  is contained in the 25% subset and so on, and every class is represented in
  the same proportion.
"""
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

# Subsets are always drawn with this seed, so all models and all repetitions see
# the same images. (The repetition seeds change the initial weights and the
# batch order instead.)
SUBSET_SEED = 0


HF_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet"
CACHE_FILE = DATA_DIR / "cifar10.npz"


def _parquet_to_numpy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode the PNG images of a Hugging Face parquet file into an (N, 32, 32, 3) uint8 array."""
    import io

    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(path)
    imgs = table.column("img").to_pylist()
    x = np.stack([np.array(Image.open(io.BytesIO(im["bytes"])).convert("RGB")) for im in imgs])
    y = np.asarray(table.column("label").to_pylist(), dtype=np.int64)
    return x, y


def _build_cache() -> None:
    """Download the data once and save it as data/cifar10.npz.

    Source 1: Hugging Face (uoft-cs/cifar10) - a copy of the official
              University of Toronto release; fast.
    Source 2: torchvision (cs.toronto.edu) - official, but the server can be very slow.
    """
    import urllib.request

    hf_dir = DATA_DIR / "hf"
    hf_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    try:
        for split in ("train", "test"):
            p = hf_dir / f"{split}.parquet"
            if not p.exists():
                print(f"Downloading: {split} (Hugging Face)...")
                urllib.request.urlretrieve(HF_URL.format(split=split), p)
            print(f"Decoding PNGs: {split}...")
            arrays[f"x_{split}"], arrays[f"y_{split}"] = _parquet_to_numpy(p)
    except Exception as e:  # fall back to torchvision if Hugging Face is unreachable
        print(f"Hugging Face failed ({e}); downloading with torchvision...")
        for split, train in (("train", True), ("test", False)):
            ds = CIFAR10(root=DATA_DIR, train=train, download=True)
            arrays[f"x_{split}"], arrays[f"y_{split}"] = ds.data, np.asarray(ds.targets, dtype=np.int64)
    np.savez(CACHE_FILE, **arrays)
    print(f"Cache created: {CACHE_FILE}")


def load_raw(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CIFAR-10 (downloading it if needed) as (N, 3, 32, 32) uint8 images + labels."""
    if not CACHE_FILE.exists():
        _build_cache()
    split = "train" if train else "test"
    with np.load(CACHE_FILE) as d:
        x = torch.from_numpy(d[f"x_{split}"]).permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
        y = torch.from_numpy(d[f"y_{split}"])
    return x, y


def compute_mean_std(x_uint8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel (R, G, B) mean and standard deviation. Computed from the TRAINING
    data only; computing them from the test data would be data leakage."""
    x = x_uint8.float() / 255.0
    mean = x.mean(dim=(0, 2, 3))
    std = x.std(dim=(0, 2, 3))
    return mean, std


def normalize(x_uint8: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """[0,255] -> [0,1] -> (x - mean) / std. Result: every channel has ~0 mean and ~1 std."""
    x = x_uint8.float() / 255.0
    return (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)


def stratified_subset_indices(y: torch.Tensor, fraction: float, seed: int = SUBSET_SEED) -> torch.Tensor:
    """Select a `fraction` of the samples of every class.

    Each class is shuffled with a fixed seed and the first `fraction` of it is
    taken. Because the shuffle is always the same, smaller subsets are contained
    in larger ones (nested).
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    y_np = y.numpy()
    chosen = []
    for c in range(len(CLASSES)):
        idx = np.flatnonzero(y_np == c)
        idx = idx[rng.permutation(len(idx))]
        chosen.append(idx[: round(len(idx) * fraction)])
    return torch.from_numpy(np.sort(np.concatenate(chosen)))


class CIFAR10Data:
    """Loads and normalises the data once; builds loaders for the requested subset."""

    def __init__(self):
        x_train, self.y_train = load_raw(train=True)
        x_test, self.y_test = load_raw(train=False)
        self.mean, self.std = compute_mean_std(x_train)
        self.x_train = normalize(x_train, self.mean, self.std)
        self.x_test = normalize(x_test, self.mean, self.std)

    def train_loader(self, fraction: float = 1.0, batch_size: int = 128, seed: int = 0) -> "BatchLoader":
        idx = stratified_subset_indices(self.y_train, fraction)
        return BatchLoader(self.x_train[idx], self.y_train[idx], batch_size, shuffle=True, seed=seed)

    def test_loader(self, batch_size: int = 512) -> "BatchLoader":
        return BatchLoader(self.x_test, self.y_test, batch_size, shuffle=False)


class BatchLoader:
    """A simple, fast data loader.

    PyTorch's DataLoader with a TensorDataset fetches every image ONE BY ONE and
    then collates them; this keeps the CPU busy, so for small models the measured
    time reflects CPU speed rather than the model. Here a whole batch is taken
    with a single indexing operation. The batch order depends on the seed
    (reproducible).
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool, seed: int = 0):
        self.x, self.y = x, y
        self.n = len(y)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.g = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = torch.randperm(self.n, generator=self.g) if self.shuffle else None
        for i in range(0, self.n, self.batch_size):
            if order is None:
                yield self.x[i : i + self.batch_size], self.y[i : i + self.batch_size]
            else:
                idx = order[i : i + self.batch_size]
                yield self.x[idx], self.y[idx]
