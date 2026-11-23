"""Shared training code: ALL models are trained with the same recipe (fair comparison).

The training recipe (RECIPE) is defined in one place; no model can be trained
with different settings. Measurement (time/memory/energy) is added around this
code in experiment.py.
"""
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from augment import gpu_augment
from data import CIFAR10Data
from models import build_model


@dataclass(frozen=True)
class Recipe:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 0.05            # initial SGD learning rate
    momentum: float = 0.9       # keeps part of the direction of previous steps
    weight_decay: float = 5e-4  # penalises large weights -> less overfitting
    augment: bool = True        # random flip + crop on the GPU
    amp: bool = True            # bfloat16 mixed precision (faster, less memory)


RECIPE = Recipe()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast(recipe: Recipe):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=recipe.amp)


@torch.no_grad()
def evaluate(model, loader, recipe: Recipe = RECIPE):
    """Test accuracy and predictions (for precision/recall/F1)."""
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.cuda(non_blocking=True)
        with autocast(recipe):
            logits = model(x)
        preds.append(logits.argmax(1).cpu())
        labels.append(y)
    preds, labels = torch.cat(preds), torch.cat(labels)
    return (preds == labels).float().mean().item(), preds, labels


def setup(model_name: str, data: CIFAR10Data, fraction: float, seed: int, recipe: Recipe = RECIPE):
    """Prepare the model and the training data (the part that must stay OUTSIDE the measurement)."""
    set_seed(seed)
    model = build_model(model_name).cuda()
    train_loader = data.train_loader(fraction, recipe.batch_size, seed)
    return model, train_loader


def train_model(model_name: str, data: CIFAR10Data, fraction: float = 1.0, seed: int = 0,
                recipe: Recipe = RECIPE, eval_every_epoch: bool = False, log=print):
    """Train a model from scratch and test it at the end. Returns (model, history)."""
    model, train_loader = setup(model_name, data, fraction, seed, recipe)
    history = fit(model, train_loader, recipe, data.test_loader() if eval_every_epoch else None,
                  log=lambda s: log(f"  [{model_name} {int(fraction * 100)}% seed{seed}] {s}"))
    if not eval_every_epoch:
        history[-1]["test_acc"] = evaluate(model, data.test_loader(), recipe)[0]
    return model, history


def fit(model, train_loader, recipe: Recipe = RECIPE, test_loader=None, log=print):
    """Training loop. If test_loader is given, test accuracy is also computed after every epoch."""
    opt = torch.optim.SGD(model.parameters(), lr=recipe.lr, momentum=recipe.momentum,
                          nesterov=True, weight_decay=recipe.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=recipe.epochs * len(train_loader))
    loss_fn = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, recipe.epochs + 1):
        model.train()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            if recipe.augment:
                x = gpu_augment(x)
            with autocast(recipe):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()  # decay the learning rate along a cosine curve at every step
            loss_sum += loss.detach()
        torch.cuda.synchronize()
        rec = {"epoch": epoch, "epoch_time_s": time.perf_counter() - t0,
               "train_loss": (loss_sum / len(train_loader)).item()}
        if test_loader is not None:
            rec["test_acc"] = evaluate(model, test_loader, recipe)[0]
        history.append(rec)
        log(f"epoch {epoch:2d}/{recipe.epochs} | loss {rec['train_loss']:.3f} | {rec['epoch_time_s']:.1f} s"
            + (f" | test {rec['test_acc']:.1%}" if "test_acc" in rec else ""))
    return history
