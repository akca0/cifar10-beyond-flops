"""Main experiment: runs ALL experiments (meant to run overnight).

    5 models x 4 data sizes x 5 seeds = 100 experiments

Run (from the project folder):
    .venv\\Scripts\\python.exe src\\run_all.py            # main experiment (~4-6 hours)
    .venv\\Scripts\\python.exe src\\run_all.py --trial    # 1-2 minute trial run

Features:
- Every result is appended to results/experiments.csv as soon as the experiment
  finishes. If the script stops (power cut, error, Ctrl+C), running it again
  RESUMES where it left off.
- Seeds are the outer loop: first the 20 experiments of seed 0, then seed 1, ...
  Even if the night is cut short, the completed seeds are usable.
- Within each seed the experiment order is random (shuffled with a fixed seed),
  so slowly changing effects such as heating do not always hit the same model
  (avoids systematic error).
- Before every experiment the script waits for the GPU to cool down and measures
  the current idle power.
- At the start it checks the GPU power limit (is the performance mode on?).
- Prevents Windows from going to sleep while running.
"""
import argparse
import ctypes
import dataclasses
import json
import random
import sys
import time
import traceback
from datetime import datetime, timedelta

import pandas as pd
import pynvml
import torch

from data import PROJECT_ROOT, CIFAR10Data
from experiment import run_experiment, warmup
from measure import Measure, gpu_energy_counter_J, system_info
from models import MODELS, build_model
from train import RECIPE

FRACTIONS = (0.10, 0.25, 0.50, 1.00)
SEEDS = (0, 1, 2, 3, 4)
ORDER_SEED = 2026
COOLDOWN_TEMP_C = 60      # wait until the GPU is below this temperature...
COOLDOWN_MAX_S = 45       # ...but at most this long
IDLE_MEASURE_S = 5        # idle power measurement before every experiment
MIN_LOAD_POWER_W = 60     # under load in performance mode the GPU should draw more than this

RESULTS = PROJECT_ROOT / "results"

pynvml.nvmlInit()
_GPU = pynvml.nvmlDeviceGetHandleByIndex(0)


def gpu_temp() -> int:
    return pynvml.nvmlDeviceGetTemperature(_GPU, pynvml.NVML_TEMPERATURE_GPU)


def prevent_sleep(on: bool) -> None:
    """Tell Windows 'this program is running, do not sleep' (valid only while the script runs)."""
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:
        pass


def power_check(log) -> float:
    """Load the GPU with ResNet-18 for ~8 s and measure the average GPU power."""
    warmup("ResNet18", 5000, RECIPE, steps=5)
    model = build_model("ResNet18").cuda().eval()
    x = torch.randn(512, 3, 32, 32, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        t_end = time.perf_counter() + 3
        while time.perf_counter() < t_end:  # let the power settle
            model(x)
        torch.cuda.synchronize()
        e0, t0 = gpu_energy_counter_J(), time.perf_counter()
        while time.perf_counter() - t0 < 8:
            model(x)
            torch.cuda.synchronize()
        p = (gpu_energy_counter_J() - e0) / (time.perf_counter() - t0)
    del model, x
    torch.cuda.empty_cache()
    log(f"Power check: average GPU power under load {p:.1f} W (expected > {MIN_LOAD_POWER_W} W)")
    return p


def cooldown_and_idle() -> dict:
    """Let the GPU cool down; then measure the idle power for a few seconds (this experiment's baseline)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while gpu_temp() > COOLDOWN_TEMP_C and time.perf_counter() - t0 < COOLDOWN_MAX_S:
        time.sleep(1)
    waited = time.perf_counter() - t0
    e0, t1 = gpu_energy_counter_J(), time.perf_counter()
    time.sleep(IDLE_MEASURE_S)
    idle_w = (gpu_energy_counter_J() - e0) / (time.perf_counter() - t1)
    return {"cooldown_s": waited, "idle_gpu_power_W": idle_w, "start_gpu_temp_C": gpu_temp()}


def build_plan(models, fractions, seeds):
    rng = random.Random(ORDER_SEED)
    plan = []
    for s in seeds:
        block = [(m, f, s) for m in models for f in fractions]
        rng.shuffle(block)
        plan += block
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", action="store_true", help="1-2 minute trial run (writes to separate files)")
    ap.add_argument("--skip-power-check", action="store_true", help="skip the GPU power check")
    args = ap.parse_args()

    if args.trial:
        recipe = dataclasses.replace(RECIPE, epochs=1)
        plan = build_plan(("MLP", "SmallCNN"), (0.10, 0.25), (0,))
        out_csv, log_file = RESULTS / "trial_experiments.csv", RESULTS / "trial_log.txt"
        info_file = RESULTS / "trial_system_info.json"  # never overwrite the main run's system info
    else:
        recipe = RECIPE
        plan = build_plan(tuple(MODELS), FRACTIONS, SEEDS)
        out_csv, log_file = RESULTS / "experiments.csv", RESULTS / "run_log.txt"
        info_file = RESULTS / "system_info.json"

    def log(msg: str):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    RESULTS.mkdir(exist_ok=True)
    log("=" * 70)
    log(f"Started. Recipe: {recipe}")

    info = system_info()
    info_file.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"System: {info['gpu']} | {info['cpu']} | {info['os']}")

    if not args.skip_power_check:
        p = power_check(log)
        if p < MIN_LOAD_POWER_W:
            log("!" * 70)
            log(f"WARNING: the GPU draws only {p:.0f} W under load; its power seems to be capped.")
            log("  -> Select a performance power mode in the laptop vendor's tool (e.g. OMEN Gaming Hub)")
            log("  -> Make sure the computer is plugged in")
            log("  Then run the script again. (To run like this on purpose: --skip-power-check)")
            log("!" * 70)
            sys.exit(2)

    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = {(r.model, float(r.fraction), int(r.seed)) for r in prev.itertuples()}
    todo = [e for e in plan if (e[0], float(e[1]), e[2]) not in done]
    log(f"Plan: {len(plan)} experiments | completed: {len(plan) - len(todo)} | remaining: {len(todo)}")

    prevent_sleep(True)
    data = CIFAR10Data()
    t_start = time.perf_counter()
    try:
        for i, (model_name, fraction, seed) in enumerate(todo, 1):
            log(f"--- [{i}/{len(todo)}] {model_name} | data {int(fraction * 100)}% | seed {seed}")
            try:
                pre = cooldown_and_idle()
                row = run_experiment(model_name, fraction, seed, data, recipe, log=log)
                row.update(pre)
                row["run_order"] = len(done) + i
                df = pd.DataFrame([row])
                df.to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
                elapsed = time.perf_counter() - t_start
                eta = timedelta(seconds=int(elapsed / i * (len(todo) - i)))
                log(f"    done: accuracy {row['accuracy']:.1%} | training {row['train_time_s']:.0f} s, "
                    f"{row['train_gpu_energy_J'] / 1000:.1f} kJ | avg power {row['train_gpu_avg_power_W']:.0f} W, "
                    f"max {row['train_gpu_max_temp_C']} C | estimated time left ~{eta}")
            except Exception:
                log("    ERROR - experiment skipped, moving on:\n" + traceback.format_exc())
                torch.cuda.empty_cache()
    except KeyboardInterrupt:
        log("Stopped by the user (Ctrl+C). Running again resumes where it left off.")
    finally:
        prevent_sleep(False)
    log(f"Finished. Total time: {timedelta(seconds=int(time.perf_counter() - t_start))}")


if __name__ == "__main__":
    main()
