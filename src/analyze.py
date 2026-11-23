"""Analysis of the main experiment (results/experiments.csv) -> tables + figures.

Run (from the project folder):
    .venv\\Scripts\\python.exe src\\analyze.py

Outputs:
  results/summary.csv            mean and std over the 5 seeds for every (model, data fraction)
  results/table_main.csv / .tex  main table of the paper (full data)
  results/scaling_fit.csv        how do time/energy grow with data size? (log-log slope)
  results/theory_vs_measured.csv FLOPs (theory) vs measured time/energy
  results/energy_methods.csv     comparison of the 3 energy measurement methods
  figures/fig1..fig6_*.png       paper figures
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import PROJECT_ROOT

RES, FIG = PROJECT_ROOT / "results", PROJECT_ROOT / "figures"
ORDER = ["MLP", "SmallCNN", "MediumCNN", "MobileNetV2", "ResNet18"]
# Colour + marker together: models stay distinguishable for colour-blind readers and in greyscale print
COLOR = {"MLP": "#2a78d6", "SmallCNN": "#eb6834", "MediumCNN": "#1baf7a",
         "MobileNetV2": "#6250d6", "ResNet18": "#e87ba4"}
MARKER = {"MLP": "o", "SmallCNN": "s", "MediumCNN": "^", "MobileNetV2": "D", "ResNet18": "v"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def load() -> pd.DataFrame:
    d = pd.read_csv(RES / "experiments.csv")
    d["model"] = pd.Categorical(d["model"], ORDER, ordered=True)
    # Dynamic energy: what the model really adds = total - (idle power x time)
    d["train_dyn_energy_J"] = d.train_gpu_energy_J - d.idle_gpu_power_W * d.train_time_s
    # Theoretical training work: forward + backward pass ~ 3 x forward pass (common approximation)
    d["train_flops_total"] = 3 * d.flops_per_image * d.n_train * d.epochs
    d["train_achieved_GFLOPS"] = d.train_flops_total / d.train_time_s / 1e9
    d["infer_achieved_GFLOPS"] = d.flops_per_image * d.infer_throughput_img_per_s / 1e9
    d["accuracy_pct"] = d.accuracy * 100
    d["f1_pct"] = d.f1_macro * 100
    return d


METRICS = ["accuracy_pct", "f1_pct", "precision_macro", "recall_macro", "train_time_s",
           "train_gpu_energy_J", "train_dyn_energy_J", "train_gpu_energy_sampled_J",
           "train_cc_total_energy_J", "train_cc_gpu_energy_J", "train_cc_co2_g",
           "train_gpu_avg_power_W", "train_peak_vram_alloc_MB", "train_peak_ram_MB",
           "train_achieved_GFLOPS", "infer_throughput_img_per_s", "infer_time_per_img_us",
           "infer_gpu_energy_per_img_mJ", "infer_peak_vram_alloc_MB", "infer_achieved_GFLOPS",
           "latency_median_ms", "latency_p95_ms"]


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(["model", "fraction"], observed=True)
    s = g[METRICS].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s.insert(0, "n_train", g.n_train.first())
    s.insert(1, "params", g.params.first())
    s.insert(2, "flops_per_image", g.flops_per_image.first())
    s.insert(3, "n_seeds", g.size())
    return s.reset_index()


def main_table(s: pd.DataFrame) -> pd.DataFrame:
    f = s[s.fraction == 1.0].set_index("model")
    t = pd.DataFrame({
        "Params (M)": f.params / 1e6,
        "MFLOPs/img": f.flops_per_image / 1e6,
        "Accuracy (%)": [f"{m:.2f} ± {sd:.2f}" for m, sd in zip(f.accuracy_pct_mean, f.accuracy_pct_std)],
        "Train time (s)": [f"{m:.1f} ± {sd:.1f}" for m, sd in zip(f.train_time_s_mean, f.train_time_s_std)],
        "Train GPU energy (kJ)": [f"{m / 1e3:.2f} ± {sd / 1e3:.2f}"
                                  for m, sd in zip(f.train_gpu_energy_J_mean, f.train_gpu_energy_J_std)],
        "Peak train VRAM (MB)": f.train_peak_vram_alloc_MB_mean,
        "Peak infer VRAM (MB)": f.infer_peak_vram_alloc_MB_mean,
        "Throughput (img/s)": f.infer_throughput_img_per_s_mean,
        "Infer energy (mJ/img)": [f"{m:.2f} ± {sd:.2f}" for m, sd in
                                  zip(f.infer_gpu_energy_per_img_mJ_mean, f.infer_gpu_energy_per_img_mJ_std)],
        "Latency bs=1 (ms)": f.latency_median_ms_mean,
    })
    return t


def metrics_table(s: pd.DataFrame) -> pd.DataFrame:
    """All four classification metrics on the full training set (mean ± std, %)."""
    f = s[s.fraction == 1.0].set_index("model")
    pm = lambda m, sd: [f"{a:.2f} ± {b:.2f}" for a, b in zip(m, sd)]
    return pd.DataFrame({
        "Accuracy (%)": pm(f.accuracy_pct_mean, f.accuracy_pct_std),
        "Precision (%)": pm(f.precision_macro_mean * 100, f.precision_macro_std * 100),
        "Recall (%)": pm(f.recall_macro_mean * 100, f.recall_macro_std * 100),
        "F1 (%)": pm(f.f1_pct_mean, f.f1_pct_std),
    })


def to_latex(t: pd.DataFrame) -> str:
    cols = list(t.columns)
    fmt = lambda v: f"{v:,.0f}" if isinstance(v, (float, np.floating)) and abs(v) >= 100 else (
        f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v).replace("±", r"$\pm$"))
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\hline",
             "Model & " + " & ".join(cols) + r" \\", r"\hline"]
    for m, row in t.iterrows():
        lines.append(f"{m} & " + " & ".join(fmt(v) for v in row) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines)


def loglog_slope(x, y) -> tuple[float, float]:
    """log(y) = k log(x) + c  -> k (slope) and R^2"""
    lx, ly = np.log(x), np.log(y)
    k, c = np.polyfit(lx, ly, 1)
    r2 = 1 - np.sum((ly - (k * lx + c)) ** 2) / np.sum((ly - ly.mean()) ** 2)
    return k, r2


def scaling_fit(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in ORDER:
        x = d[d.model == m]
        kt, rt = loglog_slope(x.n_train, x.train_time_s)
        ke, re_ = loglog_slope(x.n_train, x.train_gpu_energy_J)
        rows.append({"model": m, "time_exponent": kt, "time_R2": rt, "energy_exponent": ke, "energy_R2": re_})
    return pd.DataFrame(rows)


def theory_vs_measured(s: pd.DataFrame) -> pd.DataFrame:
    f = s[s.fraction == 1.0].set_index("model")
    ref = "ResNet18"
    out = pd.DataFrame({
        "flops_ratio_vs_ResNet18": f.flops_per_image / f.flops_per_image[ref],
        "train_time_ratio_vs_ResNet18": f.train_time_s_mean / f.train_time_s_mean[ref],
        "train_energy_ratio_vs_ResNet18": f.train_gpu_energy_J_mean / f.train_gpu_energy_J_mean[ref],
        "infer_time_ratio_vs_ResNet18": f.infer_time_per_img_us_mean / f.infer_time_per_img_us_mean[ref],
        "train_achieved_GFLOPS": f.train_achieved_GFLOPS_mean,
        "infer_achieved_GFLOPS": f.infer_achieved_GFLOPS_mean,
    })
    out["flops_corr_note"] = ""
    return out.drop(columns="flops_corr_note")


def energy_methods(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(["model", "fraction"], observed=True)
    e = pd.DataFrame({
        "counter_kJ": g.train_gpu_energy_J.mean() / 1e3,
        "sampled_kJ": g.train_gpu_energy_sampled_J.mean() / 1e3,
        "codecarbon_gpu_kJ": g.train_cc_gpu_energy_J.mean() / 1e3,
        "codecarbon_total_kJ": g.train_cc_total_energy_J.mean() / 1e3,
        "codecarbon_cpu_kJ": g.train_cc_cpu_energy_J.mean() / 1e3,
        "codecarbon_ram_kJ": g.train_cc_ram_energy_J.mean() / 1e3,
    })
    e["sampled/counter"] = e.sampled_kJ / e.counter_kJ
    e["cc_gpu/counter"] = e.codecarbon_gpu_kJ / e.counter_kJ
    e["cc_total/counter"] = e.codecarbon_total_kJ / e.counter_kJ
    return e.reset_index()


# ------------------------------------------------------------------ figures
def _legend(ax, **kw):
    ax.legend(frameon=False, **kw)


def fig_acc_vs_energy(s):
    """Fig.1: accuracy-energy trade-off (4 points per model: 10/25/50/100% data)."""
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    for m in ORDER:
        x = s[s.model == m]
        ax.errorbar(x.train_gpu_energy_J_mean / 1e3, x.accuracy_pct_mean, yerr=x.accuracy_pct_std,
                    color=COLOR[m], marker=MARKER[m], ms=6, capsize=2, lw=1.5, label=m,
                    mec="white", mew=1)
        last = x.iloc[-1]
        # label offsets (points) and alignment; chosen so that no label touches another label,
        # a curve or the title (ResNet18 and MediumCNN to the upper left, SmallCNN above its point)
        dx, dy, ha = {"ResNet18": (-8, 0, "right"), "MediumCNN": (-4, 8, "right"),
                      "SmallCNN": (0, 8, "center")}.get(m, (7, 0, "left"))
        ax.annotate(m, (last.train_gpu_energy_J_mean / 1e3, last.accuracy_pct_mean), xytext=(dx, dy),
                    textcoords="offset points", va="center", ha=ha, fontsize=8, color=INK)
    # Pareto front: points that reach higher accuracy with less energy than all others
    pts = s[["train_gpu_energy_J_mean", "accuracy_pct_mean"]].to_numpy()
    pts = pts[np.argsort(pts[:, 0])]
    front, best = [], -1
    for e, a in pts:
        if a > best:
            front.append((e, a)); best = a
    front = np.array(front)
    ax.step(front[:, 0] / 1e3, front[:, 1], where="post", color=INK2, ls="--", lw=1,
            label="Pareto frontier", zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("Training GPU energy (kJ, log scale)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy vs. training energy (10/25/50/100% data, 5 seeds)")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.6)
    _legend(ax, loc="lower right")
    fig.savefig(FIG / "fig1_accuracy_vs_energy.png")
    plt.close(fig)


def fig_scaling(s, fit):
    """Fig.2: as the data grows: (a) time, (b) energy, (c) memory, (d) accuracy
    (the four Time-Energy-Space-Performance dimensions)."""
    fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(7.2, 5.2))
    fi = fit.set_index("model")
    eb = dict(ms=5, capsize=2, lw=1.5, mec="white", mew=1)
    for m in ORDER:
        x = s[s.model == m]
        a.errorbar(x.n_train, x.train_time_s_mean, yerr=x.train_time_s_std, color=COLOR[m],
                   marker=MARKER[m], label=m, **eb)
        b.errorbar(x.n_train, x.train_gpu_energy_J_mean / 1e3, yerr=x.train_gpu_energy_J_std / 1e3,
                   color=COLOR[m], marker=MARKER[m], label=m, **eb)
        c.errorbar(x.n_train, x.train_peak_vram_alloc_MB_mean, yerr=x.train_peak_vram_alloc_MB_std,
                   color=COLOR[m], marker=MARKER[m], label=m, **eb)
        d.errorbar(x.n_train, x.accuracy_pct_mean, yerr=x.accuracy_pct_std, color=COLOR[m],
                   marker=MARKER[m], label=m, **eb)
    n = np.array([5000, 50000])
    a.plot(n, 3 * n / n[0], color=INK2, ls=":", lw=1, label="linear reference")
    b.plot(n, 0.1 * n / n[0], color=INK2, ls=":", lw=1, label="linear reference")
    a.set(xscale="log", yscale="log", ylabel="Training time (s)", title="(a) Time: training time ∝ n^k")
    b.set(xscale="log", yscale="log", ylabel="Training GPU energy (kJ)", title="(b) Energy: training energy ∝ n^k")
    c.set(xscale="log", yscale="log", ylabel="Peak training VRAM (MB)", title="(c) Space: peak training VRAM")
    d.set(xscale="log", ylabel="Test accuracy (%)", title="(d) Performance: accuracy")
    ticks = sorted(s.n_train.unique())
    for ax in (a, b, c, d):
        ax.set_xticks(ticks, [f"{t / 1000:g}k" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("Training set size (images)")
    c.set_ylim(30, 3000)
    # One shared legend below the panels (fitted exponents k are given in the paper text/caption)
    handles, labels = a.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=7.5)
    print("fitted exponents k:", fi[["time_exponent", "energy_exponent"]].round(3).to_dict())
    fig.savefig(FIG / "fig2_scaling.png")
    plt.close(fig)


def fig_theory_vs_measured(s):
    """Fig.3: theory (FLOPs) vs measured. If the theory held, the points would lie on the dashed line."""
    f = s[s.fraction == 1.0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for _, r in f.iterrows():
        m = r.model
        a.scatter(r.flops_per_image / 1e6, r.infer_time_per_img_us_mean, color=COLOR[m], marker=MARKER[m],
                  s=55, edgecolor="white", lw=1, zorder=3)
        left = m == "MobileNetV2"
        a.annotate(m, (r.flops_per_image / 1e6, r.infer_time_per_img_us_mean), xytext=(-7 if left else 6, 0),
                   textcoords="offset points", fontsize=7, color=INK, va="center", ha="right" if left else "left")
    # "If time were proportional to FLOPs" line (through ResNet-18)
    rr = f[f.model == "ResNet18"].iloc[0]
    xs = np.array([2, 2000])
    a.plot(xs, rr.infer_time_per_img_us_mean * xs / (rr.flops_per_image / 1e6), color=INK2, ls="--", lw=1,
           label="if time ∝ FLOPs")
    a.set(xscale="log", yscale="log", xlabel="Theoretical cost (MFLOPs / image)",
          ylabel="Measured inference time (µs / image)", title="(a) Theory vs. measured (inference)",
          xlim=(1.5, 4000))
    _legend(a, loc="lower right", fontsize=7)

    xpos = np.arange(len(ORDER))
    vals = [f[f.model == m].train_achieved_GFLOPS_mean.iloc[0] / 1000 for m in ORDER]
    b.bar(xpos, vals, color=[COLOR[m] for m in ORDER], width=0.6, edgecolor="white", lw=2)
    for i, v in enumerate(vals):
        b.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
    b.set_xticks(xpos, ORDER, rotation=20)
    b.set(ylabel="Achieved TFLOP/s (training)", title="(b) How well each model uses the GPU")
    b.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_theory_vs_measured.png")
    plt.close(fig)


def fig_inference(s):
    """Fig.4: inference - (a) throughput, (b) single-image latency, (c) energy per image."""
    f = s[s.fraction == 1.0].set_index("model").reindex(ORDER)
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.8))
    specs = [("infer_throughput_img_per_s_mean", "Images / s (batch 512)", "(a) Throughput", True),
             ("latency_median_ms_mean", "ms (batch 1, median)", "(b) Latency", False),
             ("infer_gpu_energy_per_img_mJ_mean", "mJ / image", "(c) Energy per image", True)]
    xpos = np.arange(len(ORDER))
    for ax, (col, ylab, title, log) in zip(axs, specs):
        v = f[col].to_numpy()
        ax.bar(xpos, v, color=[COLOR[m] for m in ORDER], width=0.65, edgecolor="white", lw=2)
        for i, val in enumerate(v):
            ax.text(i, val, f"{val:,.0f}" if val >= 100 else f"{val:.2f}", ha="center", va="bottom",
                    fontsize=6.5, color=INK)
        ax.set_xticks(xpos, ORDER, rotation=35, ha="right", fontsize=7)
        ax.set(ylabel=ylab, title=title)
        if log:
            ax.set_yscale("log")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_inference.png")
    plt.close(fig)


def fig_resources(s):
    """Fig.5: parameter count, FLOPs and real memory use do not follow the same order."""
    f = s[s.fraction == 1.0].set_index("model").reindex(ORDER)
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.8))
    specs = [(f.params / 1e6, "Million parameters", "(a) Model size"),
             (f.flops_per_image / 1e6, "MFLOPs / image", "(b) Theoretical compute"),
             (f.train_peak_vram_alloc_MB_mean, "MB", "(c) Peak GPU memory (training)")]
    xpos = np.arange(len(ORDER))
    for ax, (v, ylab, title) in zip(axs, specs):
        v = v.to_numpy()
        ax.bar(xpos, v, color=[COLOR[m] for m in ORDER], width=0.65, edgecolor="white", lw=2)
        for i, val in enumerate(v):
            ax.text(i, val, f"{val:,.0f}" if val >= 100 else f"{val:.2f}", ha="center", va="bottom",
                    fontsize=6.5, color=INK)
        ax.set_xticks(xpos, ORDER, rotation=35, ha="right", fontsize=7)
        ax.set(ylabel=ylab, title=title, yscale="log")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_resources.png")
    plt.close(fig)


def fig_energy_methods(e):
    """Fig.6: ratio of the 3 measurement methods to the hardware counter (full data)."""
    f = e[e.fraction == 1.0].set_index("model").reindex(ORDER)
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    xpos = np.arange(len(ORDER))
    w = 0.26
    series = [("sampled/counter", "Power sampling (10 Hz)", "#2a78d6"),
              ("cc_gpu/counter", "CodeCarbon GPU", "#eb6834"),
              ("cc_total/counter", "CodeCarbon total (GPU+CPU+RAM)", "#1baf7a")]
    for i, (col, lab, c) in enumerate(series):
        v = f[col].to_numpy()
        ax.bar(xpos + (i - 1) * w, v, w - 0.03, color=c, label=lab)
        if col == "cc_gpu/counter":
            continue  # all 1.00 (same counter); stated in the caption
        for x, val in zip(xpos + (i - 1) * w, v):
            ax.text(x, val, f"{val:.2f}", ha="center", va="bottom", fontsize=6, color=INK)
    ax.axhline(1, color=INK2, lw=1, ls="--")
    ax.set_xticks(xpos, ORDER, rotation=20, fontsize=7)
    ax.set(ylabel="Ratio to NVML energy counter", title="Energy methods vs. hardware counter (full data)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_ylim(0, max(f["cc_total/counter"]) * 1.12)
    fig.savefig(FIG / "fig6_energy_methods.png")
    plt.close(fig)


def main():
    d = load()
    assert d.groupby(["model", "fraction"], observed=True).size().eq(5).all(), "every combination must have 5 seeds"
    s = summarize(d)
    s.to_csv(RES / "summary.csv", index=False)

    t = main_table(s)
    t.to_csv(RES / "table_main.csv")
    (RES / "table_main.tex").write_text(to_latex(t), encoding="utf-8")
    tm = metrics_table(s)
    tm.to_csv(RES / "table_metrics.csv")
    (RES / "table_metrics.tex").write_text(to_latex(tm), encoding="utf-8")

    fit = scaling_fit(d)
    fit.to_csv(RES / "scaling_fit.csv", index=False)
    tv = theory_vs_measured(s)
    tv.to_csv(RES / "theory_vs_measured.csv")
    e = energy_methods(d)
    e.to_csv(RES / "energy_methods.csv", index=False)

    FIG.mkdir(exist_ok=True)
    fig_acc_vs_energy(s)
    fig_scaling(s, fit)
    fig_theory_vs_measured(s)
    fig_inference(s)
    fig_resources(s)
    fig_energy_methods(e)

    pd.set_option("display.width", 220, "display.max_columns", 20)
    print("=== Main table (full data) ===\n", t.to_string(), "\n")
    print("=== Scaling (time ∝ n^k) ===\n", fit.round(3).to_string(index=False), "\n")
    print("=== Theory vs measured (ResNet18 = 1) ===\n", tv.round(3).to_string(), "\n")
    print("=== Energy methods (full data) ===\n",
          e[e.fraction == 1.0].round(3).to_string(index=False), "\n")
    f = d[d.fraction == 1.0]
    print("Dynamic energy share (full data, training):",
          (f.groupby("model", observed=True).train_dyn_energy_J.mean()
           / f.groupby("model", observed=True).train_gpu_energy_J.mean()).round(3).to_dict())
    print("Whole run: training GPU energy %.1f kJ (%.3f kWh), CodeCarbon CO2 %.0f g"
          % (d.train_gpu_energy_J.sum() / 1e3, d.train_gpu_energy_J.sum() / 3.6e6,
             d.train_cc_co2_g.sum() + d.infer_cc_co2_g.sum()))


if __name__ == "__main__":
    main()
