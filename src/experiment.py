"""One experiment = one model + one data size + one seed.

Each experiment consists of these phases and produces ONE ROW of the results table:

  1. Setup (not measured)       : build the model and the data
  2. TRAINING (measured)        : time, memory, energy
  3. Evaluation                 : accuracy, precision, recall, F1 (macro)
  4. BATCH INFERENCE (measured) : pass over the test set in batches of 512; repeated
                                  for at least 10 s (short measurements are unreliable)
  5. SINGLE-IMAGE INFERENCE     : batch=1, response time for one image (latency)
     (measured)

Trial run:  .venv\\Scripts\\python.exe src\\experiment.py
"""
import dataclasses
import gc
import statistics
import time
from datetime import datetime

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data import CIFAR10Data
from measure import Measure
from models import count_flops, count_params
from train import RECIPE, Recipe, autocast, evaluate, fit, setup

torch.backends.cudnn.benchmark = True  # let cuDNN pick the fastest algorithm per layer (same for all models)

INFER_MIN_SECONDS = 10.0  # batch inference is measured for at least this long
INFER_MIN_PASSES = 3
LATENCY_WARMUP = 30
LATENCY_N = 300


@torch.no_grad()
def inference_passes(model, test_loader, recipe: Recipe) -> tuple[int, int]:
    """Predict the test set repeatedly for at least INFER_MIN_SECONDS.
    Returns (number of passes, number of images processed)."""
    model.eval()
    passes = n_img = 0
    t0 = time.perf_counter()
    while passes < INFER_MIN_PASSES or time.perf_counter() - t0 < INFER_MIN_SECONDS:
        for x, _ in test_loader:
            x = x.cuda(non_blocking=True)
            with autocast(recipe):
                model(x).argmax(1)
            n_img += x.shape[0]
        passes += 1
        torch.cuda.synchronize()
    return passes, n_img


@torch.no_grad()
def latency_ms(model, x_test: torch.Tensor, recipe: Recipe) -> list[float]:
    """Single-image (batch=1) prediction times in milliseconds, including the copy to the GPU."""
    model.eval()
    times = []
    for i in range(LATENCY_WARMUP + LATENCY_N):
        x = x_test[i : i + 1]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with autocast(recipe):
            model(x.cuda()).argmax(1).item()  # .item() brings the result back to the CPU -> real waiting time
        dt = (time.perf_counter() - t0) * 1000
        if i >= LATENCY_WARMUP:
            times.append(dt)
    return times


def warmup(model_name: str, n_train: int, recipe: Recipe, steps: int = 15) -> None:
    """Warm-up: run the same architecture for a few steps on random data before measuring.

    On the first run the GPU kernels are loaded and cuDNN searches for the fastest
    algorithm for every layer/input shape (in our trial this made ResNet's first
    epoch take 18 s and the second 1.6 s). This one-off preparation is not a real
    cost of the model and must not be measured.
    Does not touch the real model's weights (a separate copy is used, and the seed
    is reset afterwards).
    """
    from augment import gpu_augment
    from models import build_model
    from torch import nn

    model = build_model(model_name).cuda().train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=recipe.momentum, nesterov=True,
                          weight_decay=recipe.weight_decay)
    train_shapes = {recipe.batch_size, n_train % recipe.batch_size or recipe.batch_size}
    for bs in train_shapes:  # cuDNN tunes separately for every input shape
        for _ in range(steps):
            x = torch.randn(bs, 3, 32, 32, device="cuda")
            y = torch.randint(0, 10, (bs,), device="cuda")
            if recipe.augment:
                x = gpu_augment(x)
            with autocast(recipe):
                loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        for bs in (512, 10000 % 512, 1):  # batch inference, last partial batch, single image
            for _ in range(steps):
                with autocast(recipe):
                    model(torch.randn(bs, 3, 32, 32, device="cuda")).argmax(1)
    torch.cuda.synchronize()
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()


def run_experiment(model_name: str, fraction: float, seed: int, data: CIFAR10Data,
                   recipe: Recipe = RECIPE, log=print) -> dict:
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "model": model_name, "fraction": fraction, "seed": seed}

    # 0) Warm-up (not measured)
    warmup(model_name, 10 * round(len(data.y_train) // 10 * fraction), recipe)  # = subset size

    # 1) Setup
    model, train_loader = setup(model_name, data, fraction, seed, recipe)
    test_loader = data.test_loader()
    row.update({"n_train": train_loader.n, "params": count_params(model),
                "flops_per_image": count_flops(model), **dataclasses.asdict(recipe)})

    # 2) Training
    with Measure("train") as m:
        history = fit(model, train_loader, recipe, log=lambda s: log(f"  [{model_name} {int(fraction*100)}% s{seed}] {s}"))
    row.update(m.result)
    row["train_final_loss"] = history[-1]["train_loss"]
    row["train_epoch_time_median_s"] = statistics.median(h["epoch_time_s"] for h in history)

    # 3) Performance metrics (this pass also serves as a warm-up for inference)
    _, preds, labels = evaluate(model, test_loader, recipe)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    row.update({"accuracy": accuracy_score(labels, preds), "precision_macro": p,
                "recall_macro": r, "f1_macro": f1})

    # 4) Batch inference
    with Measure("infer") as m:
        passes, n_img = inference_passes(model, test_loader, recipe)
    row.update(m.result)
    row.update({"infer_passes": passes, "infer_images": n_img,
                "infer_throughput_img_per_s": n_img / row["infer_time_s"],
                "infer_time_per_img_us": row["infer_time_s"] / n_img * 1e6,
                "infer_gpu_energy_per_img_mJ": row["infer_gpu_energy_J"] / n_img * 1000,
                "infer_time_per_testset_s": row["infer_time_s"] / passes,
                "infer_gpu_energy_per_testset_J": row["infer_gpu_energy_J"] / passes})

    # 5) Single-image inference (latency)
    lat = sorted(latency_ms(model, data.x_test, recipe))
    row.update({"latency_mean_ms": statistics.mean(lat), "latency_std_ms": statistics.stdev(lat),
                "latency_median_ms": statistics.median(lat), "latency_p95_ms": lat[int(0.95 * len(lat)) - 1]})

    del model, train_loader
    gc.collect()
    torch.cuda.empty_cache()
    return row


if __name__ == "__main__":
    # Short trial: 2 models, 10% data, 3 epochs. Goal: do the measurements make sense?
    from measure import idle_gpu_power, system_info

    print(system_info())
    print(idle_gpu_power(10))
    data = CIFAR10Data()
    trial = dataclasses.replace(RECIPE, epochs=3)
    keys = ("n_train", "train_time_s", "train_epoch_time_median_s", "train_gpu_energy_J",
            "train_gpu_energy_sampled_J", "train_cc_total_energy_J", "train_peak_vram_alloc_MB",
            "train_ram_delta_MB", "train_gpu_max_temp_C", "train_gpu_min_clock_MHz", "accuracy", "f1_macro",
            "infer_time_s", "infer_gpu_energy_J", "infer_gpu_energy_sampled_J", "infer_n_samples",
            "infer_max_sample_gap_s", "infer_gpu_avg_power_W", "infer_throughput_img_per_s",
            "infer_gpu_energy_per_img_mJ", "latency_mean_ms", "latency_p95_ms")
    for name in ("MLP", "ResNet18"):
        row = run_experiment(name, 0.10, 0, data, trial)
        print(f"\n===== {name} =====")
        for k in keys:
            v = row[k]
            print(f"  {k:30s} {v:.4g}" if isinstance(v, float) else f"  {k:30s} {v}")
