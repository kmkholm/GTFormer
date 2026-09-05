import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RES = os.environ.get("RESULTS_LAMDA", "results_lamda"); FIG = os.path.join(RES, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150, "savefig.dpi": 300})
df = pd.read_csv(os.path.join(RES, "metrics_lamda.csv"))
MODELS = [("lgbm_plain", "LightGBM", "#0072B2"), ("mlp_plain", "MLP", "#999999"), ("gtformer_plain", "GT-Former", "#D55E00"), ("proto_plain", "ProtoCon-Net", "#CC79A7"),
          ("gtformer_tuned", "GT-Former (tuned)", "#009E73"), ("ens_mlp_lgbm", "MLP + LightGBM (rank-ensemble)", "#E69F00")]
have = [m for m in MODELS if (df.model == m[0]).any()]
yrs = sorted(int(r[4:]) for r in df.region.unique() if r.startswith("year"))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
for t, n, c in have:
    sub = df[(df.model == t) & df.region.str.startswith("year")].copy(); sub["yr"] = sub.region.str[4:].astype(int); sub = sub.sort_values("yr")
    if not t.startswith("ens"): axes[0].plot(sub.yr, sub["f1@0.5"], marker="o", ms=3, lw=1.5, color=c, label=n)
    axes[1].plot(sub.yr, sub["tpr@0.01"], marker="o", ms=3, lw=1.5, color=c, label=n)
axes[0].set_ylabel("F1 (threshold 0.5)"); axes[1].set_ylabel("TPR @ 1% FPR (IID-calibrated)"); [a.set_xlabel("year") for a in axes]; [a.set_ylim(0, 1) for a in axes]
axes[0].axvspan(2012.5, 2014.5, color="#0072B2", alpha=0.08); axes[1].axvspan(2012.5, 2014.5, color="#0072B2", alpha=0.08)
axes[1].legend(fontsize=6.5, frameon=False); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_lamda_drift.png"), bbox_inches="tight"); plt.close(fig)
# region bars
regs = ["IID", "NEAR", "FAR"]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
for ax, (metric, lab) in zip(axes, [("tpr@0.01", "TPR @ 1% FPR"), ("novel_tpr@0.01", "Novel-family TPR @ 1% FPR"), ("singleton_tpr@0.01", "Singleton TPR @ 1% FPR")]):
    w = 0.8 / len(have); x = np.arange(len(regs))
    for i, (t, n, c) in enumerate(have):
        vals = [df[(df.model == t) & (df.region == r)][metric].mean() for r in regs]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w * 0.92, color=c, label=n, linewidth=0)
    ax.set_xticks(x); ax.set_xticklabels(regs); ax.set_title(lab, fontsize=9); ax.set_ylim(0, 1)
axes[0].legend(fontsize=6, frameon=False); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_lamda_regions.png"), bbox_inches="tight"); plt.close(fig)
print("figures written", os.listdir(FIG))
