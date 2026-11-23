"""The size of every model "on paper": parameter count, operation count, file size.

Run:     .venv\\Scripts\\python.exe src\\model_summary.py
Output:  results/model_summary.csv
"""
import pandas as pd
import torch

from data import PROJECT_ROOT
from models import MODELS, build_model, count_flops, count_params

rows = []
for name in MODELS:
    model = build_model(name)
    out = model(torch.zeros(2, 3, 32, 32))  # does it run? the output must be (2, 10)
    assert out.shape == (2, 10), out.shape
    params = count_params(model)
    flops = count_flops(model)
    rows.append({
        "model": name,
        "params": params,
        "params (M)": round(params / 1e6, 3),
        "FLOPs/image (M)": round(flops / 1e6, 1),
        "size (MB, float32)": round(params * 4 / 1024**2, 2),
        "FLOPs / param": round(flops / params, 1),
    })

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
out = PROJECT_ROOT / "results" / "model_summary.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
