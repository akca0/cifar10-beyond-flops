"""The first model: train an MLP and watch how training progresses.

Run:     .venv\\Scripts\\python.exe src\\train_first_mlp.py
Output:  figures/mlp_first_run.png

This script is the plainest form of the training loop that is later wrapped by
the measurement code. It is worth reading line by line.
"""
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from data import PROJECT_ROOT, CIFAR10Data
from models import MLP, count_params

SEED = 42
EPOCHS = 15
BATCH_SIZE = 128
LR = 1e-3


def set_seed(seed: int) -> None:
    """Fix all sources of randomness -> same seed = same result."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()  # no gradients needed for evaluation -> faster, less memory
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.numel()
    return loss_sum / total, correct / total


def main():
    set_seed(SEED)
    device = torch.device("cuda")
    data = CIFAR10Data()
    train_loader = data.train_loader(fraction=1.0, batch_size=BATCH_SIZE, seed=SEED)
    test_loader = data.test_loader()

    model = MLP().to(device)
    print(model)
    print(f"Parameters: {count_params(model):,}")

    loss_fn = nn.CrossEntropyLoss()                            # classification loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)    # algorithm that updates the weights

    hist = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    t_start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)            # 1) forward pass: make a prediction
            loss = loss_fn(logits, y)    # 2) how wrong is the prediction?
            optimizer.zero_grad()        # 3) reset the previous gradients
            loss.backward()              # 4) backpropagation: each weight's contribution to the error
            optimizer.step()             # 5) update the weights in the direction that reduces the error

        tr_loss, tr_acc = evaluate(model, train_loader, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        for k, v in zip(hist, (tr_loss, tr_acc, te_loss, te_acc)):
            hist[k].append(v)
        print(f"Epoch {epoch:2d}/{EPOCHS} | train loss {tr_loss:.3f} acc {tr_acc:.1%} "
              f"| test loss {te_loss:.3f} acc {te_acc:.1%}")

    torch.cuda.synchronize()
    print(f"\nTotal time (including evaluations): {time.perf_counter() - t_start:.1f} s")
    print(f"Best test accuracy: {max(hist['test_acc']):.1%} "
          f"(epoch {int(np.argmax(hist['test_acc'])) + 1})")

    ep = range(1, EPOCHS + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(ep, hist["train_loss"], "o-", label="train")
    a1.plot(ep, hist["test_loss"], "s-", label="test")
    a1.set(title="Loss", xlabel="epoch")
    a2.plot(ep, [v * 100 for v in hist["train_acc"]], "o-", label="train")
    a2.plot(ep, [v * 100 for v in hist["test_acc"]], "s-", label="test")
    a2.set(title="Accuracy (%)", xlabel="epoch")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(f"MLP - CIFAR-10 ({count_params(model):,} parameters)")
    fig.tight_layout()
    out = PROJECT_ROOT / "figures" / "mlp_first_run.png"
    fig.savefig(out, dpi=120)
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
