"""Why is MobileNetV2 slow? Measure the time of one training step under different settings.

Variables: precision (float32 / bfloat16) and batch size (128 / 512).
Run:  .venv\\Scripts\\python.exe src\\speed_check.py
"""
import time

import torch
from torch import nn

from models import build_model, count_flops

torch.backends.cudnn.benchmark = True  # let cuDNN pick the fastest algorithm for every layer


def step_time(name: str, batch: int, bf16: bool, iters: int = 30) -> float:
    model = build_model(name).cuda().train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    x = torch.randn(batch, 3, 32, 32, device="cuda")
    y = torch.randint(0, 10, (batch,), device="cuda")

    def one():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(5):  # warm-up
        one()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        one()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000 / batch * 1000  # µs / image


print(f"{'setting':22s} {'ResNet18':>10s} {'MobileNetV2':>12s} {'ratio (MB/RN)':>13s}   (µs / image, training)")
for bf16 in (False, True):
    for batch in (128, 512):
        r = step_time("ResNet18", batch, bf16)
        m = step_time("MobileNetV2", batch, bf16)
        label = f"{'bfloat16' if bf16 else 'float32'}, batch {batch}"
        print(f"{label:22s} {r:10.1f} {m:12.1f} {m / r:13.2f}")

fr, fm = count_flops(build_model("ResNet18")), count_flops(build_model("MobileNetV2"))
print(f"\nTheoretical FLOP ratio (MobileNetV2 / ResNet18): {fm / fr:.2f}")
