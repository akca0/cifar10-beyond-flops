"""Attempt to improve MLP accuracy: 4 variants, same seed, same number of epochs.

  A) Baseline     : the first MLP (train_first_mlp.py)
  B) +BN +Dropout : BatchNorm + Dropout(0.3)
  C) B + Cosine LR: the learning rate decays smoothly during training
  D) C + Augment  : random flip + crop on the GPU

Run:     .venv\\Scripts\\python.exe src\\mlp_tuning.py
Output:  figures/mlp_tuning.png
"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from augment import gpu_augment
from data import PROJECT_ROOT, CIFAR10Data
from models import MLP, count_params
from train_first_mlp import evaluate, set_seed

SEED = 42
EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3

VARIANTS = {
    "A) Baseline": dict(bn=False, dropout=0.0, cosine=False, augment=False),
    "B) +BN +Dropout": dict(bn=True, dropout=0.3, cosine=False, augment=False),
    "C) B + Cosine LR": dict(bn=True, dropout=0.3, cosine=True, augment=False),
    "D) C + Augment": dict(bn=True, dropout=0.3, cosine=True, augment=True),
}


def run(cfg, data, device):
    set_seed(SEED)
    train_loader = data.train_loader(fraction=1.0, batch_size=BATCH_SIZE, seed=SEED)
    test_loader = data.test_loader()
    model = MLP(batchnorm=cfg["bn"], dropout=cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS) if cfg["cosine"] else None
    loss_fn = nn.CrossEntropyLoss()

    accs, train_time = [], 0.0
    for epoch in range(EPOCHS):
        model.train()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if cfg["augment"]:
                x = gpu_augment(x)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if sched:
            sched.step()
        torch.cuda.synchronize()
        train_time += time.perf_counter() - t0
        accs.append(evaluate(model, test_loader, device)[1])
    return model, accs, train_time


def main():
    device = torch.device("cuda")
    data = CIFAR10Data()
    results = {}
    for name, cfg in VARIANTS.items():
        model, accs, t = run(cfg, data, device)
        results[name] = accs
        print(f"{name:18s} | params {count_params(model):>9,} | best {max(accs):.1%} "
              f"| last epoch {accs[-1]:.1%} | training time {t:5.1f} s ({t / EPOCHS:.2f} s/epoch)")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, accs in results.items():
        ax.plot(range(1, EPOCHS + 1), [a * 100 for a in accs], label=name)
    ax.set(title="MLP improvements - test accuracy", xlabel="epoch", ylabel="test accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = PROJECT_ROOT / "figures" / "mlp_tuning.png"
    fig.savefig(out, dpi=120)
    print(f"Figure: {out}")


if __name__ == "__main__":
    main()
