"""Download CIFAR-10, inspect it and save sample images.

Run:     .venv\\Scripts\\python.exe src\\explore_data.py
Output:  figures/cifar10_samples.png and a dataset summary on the screen
"""
import matplotlib

matplotlib.use("Agg")  # draw to a file without opening a window
import matplotlib.pyplot as plt
import torch

from data import CLASSES, PROJECT_ROOT, CIFAR10Data, load_raw, stratified_subset_indices

FRACTIONS = (0.10, 0.25, 0.50, 1.00)

print("Loading CIFAR-10 (~145 MB is downloaded the first time)...")
x_train, y_train = load_raw(train=True)
x_test, y_test = load_raw(train=False)

print("\n=== Dataset summary (for the report) ===")
print(f"Training samples : {len(x_train):,}")
print(f"Test samples     : {len(x_test):,}")
print(f"Image shape      : {tuple(x_train.shape[1:])}  (channels, height, width)")
print(f"Features         : {x_train[0].numel():,}  (3 x 32 x 32 pixels)")
print(f"Classes          : {len(CLASSES)}")
print(f"Size in memory   : {x_train.numel() / 1024**2:.0f} MB (uint8), "
      f"{x_train.numel() * 4 / 1024**2:.0f} MB (float32)")

print("\nSamples per class (train / test):")
for c, name in enumerate(CLASSES):
    print(f"  {c}  {name:11s} {(y_train == c).sum().item():6,} / {(y_test == c).sum().item():5,}")

data = CIFAR10Data()
print("\nNormalisation (computed from the training data):")
print("  mean (R,G,B):", [round(v, 4) for v in data.mean.tolist()])
print("  std  (R,G,B):", [round(v, 4) for v in data.std.tolist()])
print(f"  After normalisation: mean={data.x_train.mean():.4f}, std={data.x_train.std():.4f}")

print("\nSubsets (stratified + nested):")
prev = None
for f in FRACTIONS:
    idx = stratified_subset_indices(y_train, f)
    counts = torch.bincount(y_train[idx], minlength=10)
    nested = "" if prev is None else f"  contains previous subset: {set(prev.tolist()) <= set(idx.tolist())}"
    print(f"  {int(f * 100):3d}%: {len(idx):6,} samples, {counts.min()}-{counts.max()} per class{nested}")
    prev = idx

loader = data.train_loader(fraction=0.10, batch_size=128)
xb, yb = next(iter(loader))
print(f"\nSample batch: x={tuple(xb.shape)} {xb.dtype}, y={tuple(yb.shape)}, "
      f"{len(loader)} batches per epoch for the 10% subset")

# 8 sample images from every class
fig, axes = plt.subplots(10, 8, figsize=(8, 10.5))
for c, name in enumerate(CLASSES):
    idx = torch.nonzero(y_train == c).flatten()[:8]
    for j, i in enumerate(idx):
        ax = axes[c, j]
        ax.imshow(x_train[i].permute(1, 2, 0).numpy())  # CHW -> HWC
        ax.axis("off")
    axes[c, 0].set_title(name, loc="left", fontsize=9, pad=2)
fig.suptitle("CIFAR-10: samples from every class (32x32 pixels)")
fig.tight_layout()
out = PROJECT_ROOT / "figures" / "cifar10_samples.png"
fig.savefig(out, dpi=120)
print(f"\nSample images saved: {out}")
