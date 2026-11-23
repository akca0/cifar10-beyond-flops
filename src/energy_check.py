"""Compare ways of measuring GPU energy under a constant load:
(a) the hardware energy counter, (b) the averaged power reading, (c) the instantaneous power reading.

Run:  .venv\\Scripts\\python.exe src\\energy_check.py
"""
import threading
import time

import pynvml as nv
import torch

from models import build_model

nv.nvmlInit()
h = nv.nvmlDeviceGetHandleByIndex(0)

FI_AVG = getattr(nv, "NVML_FI_DEV_POWER_AVERAGE", 185)
FI_INST = getattr(nv, "NVML_FI_DEV_POWER_INSTANT", 186)


def field_W(fid):
    v = nv.nvmlDeviceGetFieldValues(h, [fid])[0]
    if v.nvmlReturn != 0:
        return None
    return v.value.uiVal / 1000.0


print("Supported readings: average field =", field_W(FI_AVG), "W, instant field =", field_W(FI_INST), "W,",
      "PowerUsage =", nv.nvmlDeviceGetPowerUsage(h) / 1000, "W")

model = build_model("ResNet18").cuda().eval()
x = torch.randn(512, 3, 32, 32, device="cuda")
stop = threading.Event()


def load():
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        while not stop.is_set():
            model(x)
            torch.cuda.synchronize()


def measure(label, seconds, interval):
    samples = {"usage": [], "inst": []}
    e0, t0 = nv.nvmlDeviceGetTotalEnergyConsumption(h), time.perf_counter()
    last = t0
    integ = {"usage": 0.0, "inst": 0.0}
    prev = {"usage": nv.nvmlDeviceGetPowerUsage(h) / 1000, "inst": field_W(FI_INST)}
    while time.perf_counter() - t0 < seconds:
        time.sleep(interval)
        now = time.perf_counter()
        cur = {"usage": nv.nvmlDeviceGetPowerUsage(h) / 1000, "inst": field_W(FI_INST)}
        for k in integ:
            if cur[k] is not None and prev[k] is not None:
                integ[k] += (cur[k] + prev[k]) / 2 * (now - last)
                samples[k].append(cur[k])
        prev, last = cur, now
    T = time.perf_counter() - t0
    E = (nv.nvmlDeviceGetTotalEnergyConsumption(h) - e0) / 1000
    print(f"{label:28s} time {T:5.1f} s | counter {E:7.1f} J ({E / T:5.1f} W) | "
          f"PowerUsage {integ['usage']:7.1f} J ({integ['usage'] / T:5.1f} W) | "
          f"instant {integ['inst']:7.1f} J ({integ['inst'] / T:5.1f} W)")


measure("idle", 10, 0.05)
th = threading.Thread(target=load, daemon=True)
th.start()
time.sleep(3)  # let the load settle
measure("ResNet18 inference load", 20, 0.05)
measure("ResNet18 inference load (2)", 20, 0.05)
stop.set()
th.join()
time.sleep(3)
measure("load finished, idle", 10, 0.05)
