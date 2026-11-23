"""Environment check: are Python, PyTorch and the GPU working?

Run:  .venv\\Scripts\\python.exe src\\check_env.py
"""
import platform
import time

import torch

print("Python      :", platform.python_version())
print("PyTorch     :", torch.__version__)
print("CUDA (torch):", torch.version.cuda)
print("GPU found   :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("No GPU visible! The CPU-only build of PyTorch may be installed.")

print("GPU name    :", torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print(f"VRAM        : {props.total_memory / 1024**3:.1f} GB")

# A small speed test: 4096x4096 matrix multiplication (~137 billion operations)
n = 4096
for device in ["cpu", "cuda"]:
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    _ = a @ b  # warm-up: the first run is always slower
    if device == "cuda":
        torch.cuda.synchronize()  # the GPU runs asynchronously; wait for it to finish
    t0 = time.perf_counter()
    for _ in range(5):
        c = a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 5
    gflops = 2 * n**3 / dt / 1e9
    print(f"{device.upper():4s} matrix multiply: {dt*1000:7.1f} ms  (~{gflops:,.0f} GFLOPS)")

print("\nEverything is ready!")
