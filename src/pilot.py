"""Pilot experiment: train the 5 models with the shared recipe, full data, 1 seed.

Goals: (1) does the recipe work reasonably for every model, (2) does ResNet-18
reach 90%, (3) how long does each model take -> plan the main experiment.

Run:     .venv\\Scripts\\python.exe src\\pilot.py
Output:  results/pilot_history.csv, figures/pilot_curves.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import PROJECT_ROOT, CIFAR10Data
from models import MODELS
from train import RECIPE, train_model

data = CIFAR10Data()
print("Recipe:", RECIPE)
rows = []
for name in MODELS:
    _, hist = train_model(name, data, fraction=1.0, seed=0, eval_every_epoch=True,
                          log=lambda s: print(s, flush=True))
    for h in hist:
        rows.append({"model": name, **h})
    best = max(h["test_acc"] for h in hist)
    total = sum(h["epoch_time_s"] for h in hist)
    print(f"==> {name}: last {hist[-1]['test_acc']:.1%}, best {best:.1%}, training {total / 60:.1f} min", flush=True)

df = pd.DataFrame(rows)
df.to_csv(PROJECT_ROOT / "results" / "pilot_history.csv", index=False)

summary = df.groupby("model", sort=False).agg(
    last_acc=("test_acc", "last"), best_acc=("test_acc", "max"),
    epoch_s=("epoch_time_s", "median"), total_min=("epoch_time_s", lambda s: s.sum() / 60))
print("\n", summary.round(3).to_string())

fig, ax = plt.subplots(figsize=(8, 4.5))
for name, g in df.groupby("model", sort=False):
    ax.plot(g["epoch"], g["test_acc"] * 100, label=name)
ax.axhline(90, color="gray", ls="--", lw=1)
ax.set(title="Pilot: test accuracy with the shared recipe (full data)", xlabel="epoch", ylabel="test accuracy (%)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(PROJECT_ROOT / "figures" / "pilot_curves.png", dpi=120)
