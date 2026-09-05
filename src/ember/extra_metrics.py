"""Classical metrics, confusion matrices and ROC / PR curves for the key models on all three benchmarks.
Threshold policy: operating point calibrated on the VALIDATION benign files at 1% FPR (deployment-realistic); for LAMDA also the region benign 1% FPR.
Outputs: results/extra_metrics.csv, results/figures/fig_confusion_*.png, fig_roc_*.png, fig_pr_*.png"""
import os, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix, matthews_corrcoef, f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score
from scipy.stats import rankdata
from common import RES, threshold_at_fpr
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150, "savefig.dpi": 300})
FIG = os.path.join(RES, "figures"); rows = []
COL = {"LightGBM": "#0072B2", "GT-Former (tuned)": "#D55E00", "MLP": "#999999", "Tuned GT-Former + LightGBM": "#E69F00", "MLP + LightGBM": "#009E73"}

def metrics(y, s, thr, name, ds, subset, extra=None):
    p = (s > thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    r = {"dataset": ds, "subset": subset, "model": name, "threshold_rule": "val benign 1% FPR", "TP": tp, "FP": fp, "TN": tn, "FN": fn,
         "accuracy": accuracy_score(y, p), "balanced_acc": balanced_accuracy_score(y, p), "precision": precision_score(y, p, zero_division=0), "recall_TPR": recall_score(y, p),
         "specificity_TNR": tn / (tn + fp) if tn + fp else np.nan, "FPR": fp / (fp + tn) if fp + tn else np.nan, "FNR": fn / (fn + tp) if fn + tp else np.nan,
         "F1": f1_score(y, p), "MCC": matthews_corrcoef(y, p), "ROC_AUC": roc_auc_score(y, s), "PR_AUC": average_precision_score(y, s)}
    if extra: r.update(extra)
    rows.append(r); return np.array([[tn, fp], [fn, tp]])

def cm_grid(cms, titles, fname, suptitle):
    n = len(cms); fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.6))
    for ax, cm, t in zip(np.atleast_1d(axes), cms, titles):
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({cmn[i, j]:.1%})", ha="center", va="center", fontsize=7, color="white" if cmn[i, j] > 0.6 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["benign", "malware"], fontsize=7); ax.set_yticklabels(["benign", "malware"], fontsize=7)
        ax.set_xlabel("predicted", fontsize=7); ax.set_ylabel("true", fontsize=7); ax.set_title(t, fontsize=8); ax.grid(False)
    fig.suptitle(suptitle, fontsize=9); fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), bbox_inches="tight"); plt.close(fig)

def curves(curve_sets, fname, title):
    """curve_sets: list of (label, color, y, s). ROC with log FPR axis + PR curve."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for lab, c, y, s in curve_sets:
        fpr, tpr, _ = roc_curve(y, s); pr, rc, _ = precision_recall_curve(y, s)
        axes[0].plot(np.clip(fpr, 1e-5, 1), tpr, color=c, lw=1.5, label=f"{lab} (AUC {roc_auc_score(y, s):.4f})")
        axes[1].plot(rc, pr, color=c, lw=1.5, label=f"{lab} (AP {average_precision_score(y, s):.4f})")
    axes[0].set_xscale("log"); axes[0].set_xlim(1e-5, 1); axes[0].set_xlabel("false-positive rate (log)"); axes[0].set_ylabel("true-positive rate"); axes[0].set_title("ROC", fontsize=9)
    axes[0].axvline(0.01, color="k", ls=":", lw=0.8); axes[0].axvline(0.001, color="k", ls=":", lw=0.8)
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision"); axes[1].set_title("Precision-Recall", fontsize=9)
    axes[0].legend(fontsize=6, frameon=False, loc="lower right"); fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), bbox_inches="tight"); plt.close(fig)

# ---------------- EMBER2024 ----------------
for ft in ["Win32", "Win64", "Dot_Net", "APK", "ELF", "PDF"]:
    sets = {}
    for tag, name in [("base", "LightGBM"), ("gtformer_tuned", "GT-Former (tuned)"), ("mlp_mlp_plain", "MLP"), ("ens_gttuned_lgbm", "Tuned GT-Former + LightGBM")]:
        p = os.path.join(RES, f"scores_{ft}_{tag}.npz")
        if os.path.exists(p): sets[name] = np.load(p)
    cms, titles, cs, cs_ch = [], [], [], []
    for name, d in sets.items():
        thr = threshold_at_fpr(d["s_val"][d["y_val"] == 0], 0.01)
        cm = metrics(d["y_test"], d["s_test"], thr, name, "EMBER2024", f"{ft} test")
        b = d["s_test"][d["y_test"] == 0]; ych = np.r_[np.zeros(len(b)), np.ones(len(d["s_ch"]))]; sch = np.r_[b, d["s_ch"]]
        metrics(ych, sch, thr, name, "EMBER2024", f"{ft} challenge-vs-test-benign")
        if name in ("LightGBM", "GT-Former (tuned)", "Tuned GT-Former + LightGBM"): cms.append(cm); titles.append(name)
        cs.append((name, COL[name], d["y_test"], d["s_test"])); cs_ch.append((name, COL[name], ych, sch))
    if cms: cm_grid(cms, titles, f"fig_confusion_{ft}.png", f"EMBER2024 {ft} test set, threshold = 1% FPR on validation benign")
    curves(cs, f"fig_roc_pr_{ft}_test.png", f"EMBER2024 {ft}: test set"); curves(cs_ch, f"fig_roc_pr_{ft}_challenge.png", f"EMBER2024 {ft}: challenge set vs test benign")

# ---------------- LAMDA ----------------
from lamda_exp import load as lload, RES as RL
X, M, tr, va, ev = lload(); Mev = M.iloc[ev].reset_index(drop=True); y = Mev.label.values
def r01(s): return (rankdata(s) - 1) / (len(s) - 1)
L = {n: np.load(os.path.join(RL, f"scores_{t}.npz")) for t, n in [("lgbm_plain", "LightGBM"), ("gtformer_tuned", "GT-Former (tuned)"), ("mlp_plain", "MLP")]}
cat = [np.concatenate([L[n]["s_val"], L[n]["s_ev"]]) for n in ("MLP", "LightGBM")]; r = np.mean([r01(c) for c in cat], 0); nv = len(L["MLP"]["s_val"])
L["MLP + LightGBM"] = {"s_val": r[:nv], "s_ev": r[nv:], "y_val": L["MLP"]["y_val"]}
for region in ["IID", "NEAR", "FAR"]:
    m = Mev.region.values == region; cms, titles, cs = [], [], []
    for name, d in L.items():
        thr = threshold_at_fpr(np.asarray(d["s_val"])[np.asarray(d["y_val"]) == 0], 0.01)
        cm = metrics(y[m], np.asarray(d["s_ev"])[m], thr, name, "LAMDA", region)
        if name in ("LightGBM", "MLP", "MLP + LightGBM"): cms.append(cm); titles.append(name)
        cs.append((name, COL[name], y[m], np.asarray(d["s_ev"])[m]))
    cm_grid(cms, titles, f"fig_confusion_LAMDA_{region}.png", f"LAMDA {region}, threshold = 1% FPR on validation benign"); curves(cs, f"fig_roc_pr_LAMDA_{region}.png", f"LAMDA {region}")

# ---------------- BODMAS ----------------
from bodmas_exp import load as bload, RES as RB
Xb, mb, trb, vab, evb = bload(); yb = mb.label.values[evb]
B = {n: np.load(os.path.join(RB, f"scores_{t}.npz")) for t, n in [("lgbm_plain", "LightGBM"), ("gtformer_tuned", "GT-Former (tuned)"), ("mlp_plain", "MLP")]}
cat = [np.concatenate([B[n]["s_val"], B[n]["s_ev"]]) for n in ("GT-Former (tuned)", "LightGBM")]; r = np.mean([r01(c) for c in cat], 0); nv = len(B["LightGBM"]["s_val"])
B["Tuned GT-Former + LightGBM"] = {"s_val": r[:nv], "s_ev": r[nv:], "y_val": B["LightGBM"]["y_val"]}
cms, titles, cs = [], [], []
for name, d in B.items():
    thr = threshold_at_fpr(np.asarray(d["s_val"])[np.asarray(d["y_val"]) == 0], 0.01)
    cm = metrics(yb, np.asarray(d["s_ev"]), thr, name, "BODMAS", "test Apr-Sep 2020")
    if name in ("LightGBM", "GT-Former (tuned)", "Tuned GT-Former + LightGBM"): cms.append(cm); titles.append(name)
    cs.append((name, COL[name], yb, np.asarray(d["s_ev"])))
cm_grid(cms, titles, "fig_confusion_BODMAS.png", "BODMAS test set, threshold = 1% FPR on validation benign"); curves(cs, "fig_roc_pr_BODMAS.png", "BODMAS test set")

df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "extra_metrics.csv"), index=False)
pd.set_option("display.width", 250); print(df.round(4).to_string())
