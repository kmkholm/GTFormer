"""LAMDA metrics from results_lamda/scores_*.npz: per region (IID/NEAR/FAR) and per year: ROC-AUC, F1@0.5 (paper-comparable, LightGBM prob / deep logit>0),
TPR@1%/0.1% FPR (threshold from region benign), novel-family / singleton / low-consensus strata, rank ensembles, bootstrap vs LightGBM."""
import os, glob, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, f1_score
from common import threshold_at_fpr, detection_rate
from lamda_exp import load, RES

X, M, tr, va, ev = load()
train_fams = set(M.fam.values[tr]) - {""}
Mev = M.iloc[ev].reset_index(drop=True); y = Mev.label.values
mal = y == 1; novel = mal & (Mev.fam.values != "") & ~Mev.fam.isin(train_fams).values; single = mal & Mev.is_singleton.values
known = mal & Mev.fam.isin(train_fams).values; low = mal & (Mev.vt_count.values <= 6)

def r01(s): return (rankdata(s) - 1) / (len(s) - 1)
files = {os.path.basename(p)[7:-4]: np.load(p) for p in sorted(glob.glob(os.path.join(RES, "scores_*.npz")))}
# ensembles
def ens(name, members):
    if all(m in files for m in members) and name not in files:
        cat = [np.concatenate([files[m]["s_val"], files[m]["s_ev"]]) for m in members]; r = np.mean([r01(c) for c in cat], 0); nv = len(files[members[0]]["s_val"])
        files[name] = {"s_val": r[:nv], "s_ev": r[nv:], "y_val": files[members[0]]["y_val"], "train_sec": 0}
ens("ens_gt_lgbm", ["gtformer_plain", "lgbm_plain"]); ens("ens_gttuned_lgbm", ["gtformer_tuned", "lgbm_plain"]); ens("ens_gttuned_mlp_lgbm", ["gtformer_tuned", "mlp_plain", "lgbm_plain"]); ens("ens_mlp_lgbm", ["mlp_plain", "lgbm_plain"]); ens("ens_gt_mlp_lgbm", ["gtformer_plain", "mlp_plain", "lgbm_plain"])

def is_prob(tag): return tag.startswith("lgbm") or tag.startswith("ens")
rows = []
for tag, d in files.items():
    s = np.asarray(d["s_ev"]); pred = (s > 0.5) if is_prob(tag) else (s > 0)
    for region in ["IID", "NEAR", "FAR"]:
        m = (Mev.region.values == region)
        thr1 = threshold_at_fpr(s[m & (y == 0)], 0.01); thr01 = threshold_at_fpr(s[m & (y == 0)], 0.001)
        r = {"model": tag, "region": region, "n": int(m.sum()), "n_mal": int((m & mal).sum()), "roc_auc": roc_auc_score(y[m], s[m]) if (m & mal).any() else np.nan,
             "f1@0.5": f1_score(y[m], pred[m]) if (m & mal).any() else np.nan, "fnr@0.5": 1 - pred[m & mal].mean() if (m & mal).any() else np.nan, "fpr@0.5": pred[m & (y == 0)].mean(),
             "tpr@0.01": detection_rate(s[m & mal], thr1), "tpr@0.001": detection_rate(s[m & mal], thr01),
             "novel_tpr@0.01": detection_rate(s[m & novel], thr1) if (m & novel).any() else np.nan, "singleton_tpr@0.01": detection_rate(s[m & single], thr1) if (m & single).any() else np.nan,
             "known_tpr@0.01": detection_rate(s[m & known], thr1) if (m & known).any() else np.nan, "lowcons_tpr@0.01": detection_rate(s[m & low], thr1) if (m & low).any() else np.nan,
             "n_novel": int((m & novel).sum()), "n_single": int((m & single).sum())}
        rows.append(r)
    # per-year drift at threshold fixed on IID benign (deployment-like) and per-year F1
    thr_iid = threshold_at_fpr(s[(Mev.region.values == "IID") & (y == 0)], 0.01)
    for yr in sorted(Mev.year.unique()):
        m = Mev.year.values == yr
        if (m & mal).sum() < 20: continue
        rows.append({"model": tag, "region": f"year{yr}", "n": int(m.sum()), "n_mal": int((m & mal).sum()), "roc_auc": roc_auc_score(y[m], s[m]), "f1@0.5": f1_score(y[m], pred[m]),
                     "tpr@0.01": detection_rate(s[m & mal], thr_iid), "fpr_at_iid_thr": detection_rate(s[m & (y == 0)], thr_iid)})
df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "metrics_lamda.csv"), index=False)
pd.set_option("display.width", 250)
print(df[df.region.isin(["IID", "NEAR", "FAR"])].round(4).to_string())
# paired bootstrap vs lgbm on FAR/NEAR TPR@1% and novel-family TPR
rng = np.random.default_rng(0); B = 1000; brows = []
ref = np.asarray(files["lgbm_plain"]["s_ev"])
for tag in [t for t in files if t not in ("lgbm_plain",)]:
    s = np.asarray(files[tag]["s_ev"])
    for region in ["NEAR", "FAR"]:
        m = Mev.region.values == region
        tr_ = threshold_at_fpr(ref[m & (y == 0)], 0.01); tm = threshold_at_fpr(s[m & (y == 0)], 0.01)
        for metric, sel in [("tpr@1%FPR", m & mal), ("novel_tpr@1%FPR", m & novel), ("singleton_tpr@1%FPR", m & single)]:
            a = (s[sel] > tm).astype(float); b = (ref[sel] > tr_).astype(float); n = len(a)
            if n < 20: continue
            idx = rng.integers(0, n, (B, n)); dd = a[idx].mean(1) - b[idx].mean(1)
            brows.append({"model": tag, "region": region, "metric": metric, "n": n, "lgbm": b.mean(), "model_value": a.mean(), "diff": a.mean() - b.mean(), "ci_lo": np.quantile(dd, .025), "ci_hi": np.quantile(dd, .975)})
bdf = pd.DataFrame(brows); bdf.to_csv(os.path.join(RES, "bootstrap_lamda.csv"), index=False); print(bdf.round(4).to_string())
