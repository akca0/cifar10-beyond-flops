"""Measurement tools: time, memory, energy.

Usage:
    with Measure("train") as m:
        ... work to be measured ...
    print(m.result)   # {"train_time_s": ..., "train_gpu_energy_J": ..., ...}

Energy is measured in 3 independent ways (so they can check each other):
  1. gpu_energy_J          : the NVIDIA card's hardware energy counter (NVML). Read at
                             the start and the end; the difference is used. The most
                             reliable GPU measurement.
  2. gpu_energy_sampled_J  : GPU power is read every 0.1 s and E = sum(P * dt)
                             (the E = P_avg x T relation). Cross-checks method 1.
  3. cc_*_energy_J         : CodeCarbon. System-wide estimate for GPU + CPU + RAM.
                             CPU power cannot be measured directly on an AMD CPU;
                             CodeCarbon ESTIMATES it from the CPU load.

Memory:
  - peak_vram_alloc_MB : peak GPU memory allocated by PyTorch (model + activations)
  - peak_vram_nvml_MB  : peak total usage seen on the card (including the CUDA context)
  - peak_ram_MB        : peak RAM of the Python process (RSS)
"""
import os
import platform
import threading
import time

import psutil
import pynvml
import torch

SAMPLE_INTERVAL_S = 0.1

pynvml.nvmlInit()
_GPU = pynvml.nvmlDeviceGetHandleByIndex(0)
_PROC = psutil.Process(os.getpid())


def gpu_energy_counter_J() -> float:
    """Total energy consumed by the GPU since the driver was loaded (Joule)."""
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(_GPU) / 1000.0  # mJ -> J


class Sampler(threading.Thread):
    """Reads GPU power/temperature/clock and RAM in the background every 0.1 s."""

    def __init__(self):
        super().__init__(daemon=True)
        self._stop_evt = threading.Event()
        self.energy_J = 0.0
        self.powers, self.temps, self.clocks = [], [], []
        self.peak_ram = 0
        self.peak_vram_nvml = 0

    def _read(self):
        p = pynvml.nvmlDeviceGetPowerUsage(_GPU) / 1000.0  # mW -> W
        self.powers.append(p)
        self.temps.append(pynvml.nvmlDeviceGetTemperature(_GPU, pynvml.NVML_TEMPERATURE_GPU))
        self.clocks.append(pynvml.nvmlDeviceGetClockInfo(_GPU, pynvml.NVML_CLOCK_SM))
        self.peak_ram = max(self.peak_ram, _PROC.memory_info().rss)
        self.peak_vram_nvml = max(self.peak_vram_nvml, pynvml.nvmlDeviceGetMemoryInfo(_GPU).used)
        return p

    def _step(self):
        t, p = time.perf_counter(), self._read()
        self.energy_J += (p + self._last_p) / 2 * (t - self._last_t)  # trapezoidal rule: mean power x time
        self.gaps.append(t - self._last_t)
        self._last_t, self._last_p = t, p

    def run(self):
        while not self._stop_evt.wait(SAMPLE_INTERVAL_S):
            self._step()

    def start(self):
        self.gaps = []
        self._last_t, self._last_p = time.perf_counter(), self._read()
        super().start()

    def stop(self):
        self._stop_evt.set()
        self.join()
        self._step()  # also add the interval from the last sample to the moment of stopping


def _codecarbon_tracker():
    # Offline: does not query the location over the internet; country = Turkey for the carbon intensity.
    from codecarbon import OfflineEmissionsTracker

    return OfflineEmissionsTracker(country_iso_code="TUR", measure_power_secs=1, save_to_file=False,
                                   log_level="error", allow_multiple_runs=True)


class Measure:
    """Context manager that measures the time, memory and energy of a code block."""

    def __init__(self, phase: str, codecarbon: bool = True):
        self.phase = phase
        self.use_cc = codecarbon
        self.result = {}

    def __enter__(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.ram_before = _PROC.memory_info().rss
        self.cc = _codecarbon_tracker() if self.use_cc else None
        if self.cc:
            self.cc.start()
        self.sampler = Sampler()
        self.sampler.start()
        self.e0 = gpu_energy_counter_J()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()  # wait for the GPU to finish (otherwise the time is wrong)
        t = time.perf_counter() - self.t0
        e = gpu_energy_counter_J() - self.e0
        self.sampler.stop()
        s = self.sampler
        ph = self.phase
        r = {
            f"{ph}_time_s": t,
            f"{ph}_gpu_energy_J": e,
            f"{ph}_gpu_energy_sampled_J": s.energy_J,
            f"{ph}_gpu_avg_power_W": e / t if t > 0 else float("nan"),
            f"{ph}_gpu_max_power_W": max(s.powers),
            f"{ph}_n_samples": len(s.powers),
            f"{ph}_max_sample_gap_s": max(s.gaps),
            f"{ph}_gpu_max_temp_C": max(s.temps),
            f"{ph}_gpu_mean_clock_MHz": sum(s.clocks) / len(s.clocks),
            f"{ph}_gpu_min_clock_MHz": min(s.clocks),
            f"{ph}_peak_vram_alloc_MB": torch.cuda.max_memory_allocated() / 1024**2,
            f"{ph}_peak_vram_nvml_MB": s.peak_vram_nvml / 1024**2,
            f"{ph}_peak_ram_MB": max(s.peak_ram, _PROC.memory_info().rss) / 1024**2,
            f"{ph}_ram_before_MB": self.ram_before / 1024**2,
        }
        # RAM added during this phase (excluding the resident dataset)
        r[f"{ph}_ram_delta_MB"] = r[f"{ph}_peak_ram_MB"] - r[f"{ph}_ram_before_MB"]
        if self.cc:
            self.cc.stop()
            d = self.cc.final_emissions_data
            kwh_to_J = 3.6e6
            r.update({
                f"{ph}_cc_total_energy_J": d.energy_consumed * kwh_to_J,
                f"{ph}_cc_gpu_energy_J": d.gpu_energy * kwh_to_J,
                f"{ph}_cc_cpu_energy_J": d.cpu_energy * kwh_to_J,
                f"{ph}_cc_ram_energy_J": d.ram_energy * kwh_to_J,
                f"{ph}_cc_co2_g": d.emissions * 1000,
            })
        self.result = r
        return False


def idle_gpu_power(seconds: float = 15.0) -> dict:
    """Power drawn by the GPU while doing nothing (idle consumption)."""
    torch.cuda.synchronize()
    time.sleep(2)
    e0, t0 = gpu_energy_counter_J(), time.perf_counter()
    time.sleep(seconds)
    e, t = gpu_energy_counter_J() - e0, time.perf_counter() - t0
    return {"idle_gpu_power_W": e / t, "idle_gpu_temp_C":
            pynvml.nvmlDeviceGetTemperature(_GPU, pynvml.NVML_TEMPERATURE_GPU)}


def system_info() -> dict:
    """Hardware and software information (to be reported in the paper)."""
    import codecarbon
    import numpy
    import sklearn

    cpu = platform.processor()
    try:
        import subprocess
        cpu = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True).stdout.strip() or cpu
    except Exception:
        pass
    drv = pynvml.nvmlSystemGetDriverVersion()
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": cpu,
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_GB": round(psutil.virtual_memory().total / 1024**3, 1),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_vram_GB": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
        "gpu_power_limit_W": pynvml.nvmlDeviceGetEnforcedPowerLimit(_GPU) / 1000,
        "nvidia_driver": drv.decode() if isinstance(drv, bytes) else drv,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "codecarbon": codecarbon.__version__,
    }
