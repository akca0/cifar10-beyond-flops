"""Repeatability check: measure the same inference job several times in a row; are the results consistent?

Run:  .venv\\Scripts\\python.exe src\\repeat_check.py
"""
import torch

from data import CIFAR10Data
from experiment import inference_passes, warmup
from measure import Measure
from models import build_model
from train import RECIPE

torch.backends.cudnn.benchmark = True
data = CIFAR10Data()
loader = data.test_loader()
for name in ("ResNet18", "MediumCNN"):
    warmup(name, 5000, RECIPE)
    model = build_model(name).cuda()
    for i in range(3):
        with Measure("i", codecarbon=False) as m:
            _, n = inference_passes(model, loader, RECIPE)
        r = m.result
        print(f"{name:10s} repeat {i + 1}: {n / r['i_time_s']:8.0f} img/s | avg power {r['i_gpu_avg_power_W']:5.1f} W "
              f"| avg clock {r['i_gpu_mean_clock_MHz']:5.0f} MHz | temp {r['i_gpu_max_temp_C']} C "
              f"| counter/sampled {r['i_gpu_energy_J'] / r['i_gpu_energy_sampled_J']:.2f}", flush=True)
