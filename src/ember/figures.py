"""Publication figures (matplotlib, Okabe-Ito colorblind-safe palette, single axis per panel)."""
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RES, FILETYPES

FIG = os.path.join(RES, "figures"); os.makedirs(FIG, exist_ok=True)
OI = {"lgbm": "#0072B2", "mlp": "#999999", "gtformer": "#D55E00", "moe": "#009E73", "proto": "#CC79A7", "ens": "#E69F00"}
MODELS = [("base", "LightGBM (benchmark)", OI["lgbm"]), ("mlp_mlp_plain", "MLP", OI["mlp"]), ("gtformer_plain", "GT-Former", OI["gtformer"]),
          ("moe_plain", "FR-MoE", OI["moe"]), ("gtformer_tuned", "GT-Former (tuned)", OI["proto"]), ("ens_gttuned_lgbm", "Tuned GT-Former + LightGBM", OI["ens"])]
FT_LABEL = {"Win32": "Win32", "Win64": "Win64", "Dot_Net": ".NET", "APK": "APK", "ELF": "ELF", "PDF": "PDF"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.5, "axes.axisbelow": True, "figure.dpi": 150, "savefig.dpi": 300})

df = pd.read_csv(os.path.join(RES, "metrics_all.csv"))
have = [(t, n, c) for t, n, c in MODELS if (df.variant == t).any()]

def grouped_bars(metric, ylabel, fname, ylim=(0, 1)):
    fts = [f for f in FILETYPES if (df.file_type == f).any()]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    w = 0.8 / len(have); x = np.arange(len(fts))
    for i, (t, n, c) in enumerate(have):
        vals = [df[(df.file_type == f) & (df.variant == t)][metric].mean() for f in fts]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w * 0.92, color=c, label=n, linewidth=0)
    ax.set_xticks(x); ax.set_xticklabels([FT_LABEL[f] for f in fts]); ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
    ax.legend(ncol=3, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), bbox_inches="tight"); plt.close(fig)

grouped_bars("ch_det@0.01(test-thr)", "Challenge-set detection @ 1% FPR", "fig_challenge_1pct.png")
grouped_bars("ch_det@0.001(test-thr)", "Challenge-set detection @ 0.1% FPR", "fig_challenge_01pct.png")
grouped_bars("tpr@0.001", "Test TPR @ 0.1% FPR", "fig_test_tpr_01pct.png", ylim=(0.5, 1))
grouped_bars("tpr@0.01_novel_fam", "Novel-family TPR @ 1% FPR", "fig_novel_family.png")
grouped_bars("tpr@0.01_lowconsensus_q1", "Low-consensus malware TPR @ 1% FPR", "fig_lowconsensus.png")

# weekly drift curves per format (TPR@1%FPR by test week)
wk_cols = [c for c in df.columns if c.startswith("wk") and c.endswith("_tpr@0.01")]
fts = [f for f in FILETYPES if (df.file_type == f).any()]
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), sharey=True)
for ax, f in zip(axes.ravel(), fts):
    for t, n, c in have:
        r = df[(df.file_type == f) & (df.variant == t)]
        if len(r) == 0: continue
        ax.plot(range(52, 52 + len(wk_cols)), r.iloc[0][wk_cols].values.astype(float), color=c, lw=1.6, label=n, marker="o", ms=2.5)
    ax.set_title(FT_LABEL[f], fontsize=9); ax.set_xlabel("test week"); ax.set_ylim(0.6, 1.0)
axes[0, 0].set_ylabel("TPR @ 1% FPR"); axes[1, 0].set_ylabel("TPR @ 1% FPR")
axes[0, 0].legend(fontsize=6.5, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_weekly_drift.png"), bbox_inches="tight"); plt.close(fig)

# ablation figure for GT-Former components (challenge@1% and novel family), Win32+Win64 averaged over formats
abl = [("gtformer_full", "Full"), ("gtformer_no_cam", "- CAM loss"), ("gtformer_no_aux", "- consensus head"), ("gtformer_no_grl", "- family GRL"),
       ("gtformer_plain", "- all (plain)"), ("mlp_full", "MLP + all losses"), ("mlp_mlp_plain", "MLP plain")]
abl = [(t, n) for t, n in abl if (df.variant == t).any()]
if abl:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, (metric, lab) in zip(axes, [("ch_det@0.01(test-thr)", "Challenge det. @1% FPR (mean over formats)"), ("tpr@0.01_novel_fam", "Novel-family TPR @1% FPR")]):
        vals = [df[df.variant == t][metric].mean() for t, _ in abl]
        ax.barh([n for _, n in abl][::-1], vals[::-1], color=OI["gtformer"], height=0.6); ax.set_xlabel(lab); ax.set_xlim(0, 1)
        for i, v in enumerate(vals[::-1]): ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_ablation.png"), bbox_inches="tight"); plt.close(fig)

# attention over feature groups (challenge vs malware vs benign) if available
p = os.path.join(RES, "attention_groups.csv")
if os.path.exists(p):
    at = pd.read_csv(p); groups = [c for c in at.columns if c not in ("format", "group")]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharey=True)
    for ax, f in zip(axes.ravel(), fts):
        sub = at[at.format == f]
        x = np.arange(len(groups)); w = 0.27
        for i, (g, c) in enumerate([("benign", "#0072B2"), ("malware", "#D55E00"), ("challenge", "#E69F00")]):
            r = sub[sub.group == g]
            if len(r): ax.bar(x + (i - 1) * w, r.iloc[0][groups].values.astype(float), w, color=c, label=g, linewidth=0)
        ax.set_title(FT_LABEL[f], fontsize=9); ax.set_xticks(x); ax.set_xticklabels(groups, rotation=70, fontsize=6)
    axes[0, 0].set_ylabel("CLS attention"); axes[1, 0].set_ylabel("CLS attention"); axes[0, 0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_attention_groups.png"), bbox_inches="tight"); plt.close(fig)

# dataset: detection-ratio distributions (train malware vs challenge)
ds = json.load(open(os.path.join(RES, "dataset_stats.json")))
print("figures written to", FIG, os.listdir(FIG))
