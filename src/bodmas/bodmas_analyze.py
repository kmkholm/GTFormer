"""BODMAS metrics: test (Apr-Sep 2020) overall and per month; TPR@1%/0.1% FPR (threshold on test benign), known/novel-family strata,
category strata; rank ensembles; seed mean±sd; paired bootstrap vs LightGBM. -> results_bodmas/metrics_bodmas.csv, bootstrap_bodmas.csv, seeds_bodmas.csv"""
import os, glob, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score
from common import threshold_at_fpr, detection_rate
from bodmas_exp import load, RES

X, m, tr, va, ev = load(); del X
train_fams = set(m.fam.values[tr]) - {""}
Mev = m.iloc[ev].reset_index(drop=True); y = Mev.label.values; mal = y == 1
known = mal & Mev.fam.isin(train_fams).values; novel = mal & (Mev.fam.values != "") & ~Mev.fam.isin(train_fams).values
cats = {c: mal & (Mev.category.values == c) for c in ["trojan", "worm", "backdoor", "downloader", "ransomware", "dropper"]}
def r01(s): return (rankdata(s) - 1) / (len(s) - 1)
files = {os.path.basename(p)[7:-4]: dict(np.load(p)) for p in sorted(glob.glob(os.path.join(RES, "scores_*.npz")))}
def ens(name, members):
    if all(k in files for k in members) and name not in files:
        cat = [np.concatenate([files[k]["s_val"], files[k]["s_ev"]]) for k in members]; r = np.mean([r01(c) for c in cat], 0); nv = len(files[members[0]]["s_val"])
        files[name] = {"s_val": r[:nv], "s_ev": r[nv:], "y_val": files[members[0]]["y_val"], "train_sec": 0}
ens("ens_gt_lgbm", ["gtformer_plain", "lgbm_plain"]); ens("ens_gttuned_lgbm", ["gtformer_tuned", "lgbm_plain"]); ens("ens_mlp_lgbm", ["mlp_plain", "lgbm_plain"])
rows = []
for tag, d in files.items():
    s = np.asarray(d["s_ev"]); b = s[y == 0]; thr1 = threshold_at_fpr(b, 0.01); thr01 = threshold_at_fpr(b, 0.001)
    r = {"model": tag, "region": "test", "n": len(s), "n_mal": int(mal.sum()), "roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s),
         "tpr@0.01": detection_rate(s[mal], thr1), "tpr@0.001": detection_rate(s[mal], thr01), "known_tpr@0.01": detection_rate(s[known], thr1),
         "novel_tpr@0.01": detection_rate(s[novel], thr1), "novel_tpr@0.001": detection_rate(s[novel], thr01), "n_novel": int(novel.sum())}
    for c, sel in cats.items(): r[f"{c}_tpr@0.01"] = detection_rate(s[sel], thr1) if sel.any() else np.nan
    rows.append(r)
    thr_val = threshold_at_fpr(np.asarray(d["s_val"])[np.asarray(d["y_val"]) == 0], 0.01)   # deployment threshold from March 2020 benign
    for ym in sorted(Mev.ym.unique()):
        mm = Mev.ym.values == ym
        rows.append({"model": tag, "region": ym, "n": int(mm.sum()), "n_mal": int((mm & mal).sum()), "roc_auc": roc_auc_score(y[mm], s[mm]),
                     "tpr@0.01": detection_rate(s[mm & mal], thr_val), "fpr_at_val_thr": detection_rate(s[mm & (y == 0)], thr_val), "novel_tpr@0.01": detection_rate(s[mm & novel], thr_val) if (mm & novel).any() else np.nan})
df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "metrics_bodmas.csv"), index=False)
pd.set_option("display.width", 250); print(df[df.region == "test"].round(4).to_string())
# seeds
sd = df[(df.region == "test") & df.model.str.contains(r"^(gtformer_plain|gtformer_tuned|mlp_plain|proto_plain)(_s\d)?$", regex=True)].copy(); sd["arch"] = sd.model.str.replace(r"_s\d$", "", regex=True)
g = sd.groupby("arch")[["roc_auc", "tpr@0.01", "tpr@0.001", "novel_tpr@0.01"]].agg(["mean", "std", "count"]); g.to_csv(os.path.join(RES, "seeds_bodmas.csv")); print(g.round(4).to_string())
# bootstrap vs lgbm
rng = np.random.default_rng(0); B = 1000; ref = np.asarray(files["lgbm_plain"]["s_ev"]); tr_ = threshold_at_fpr(ref[y == 0], 0.01); brows = []
for tag in files:
    if tag == "lgbm_plain": continue
    s = np.asarray(files[tag]["s_ev"]); tm = threshold_at_fpr(s[y == 0], 0.01)
    for metric, sel in [("tpr@1%FPR", mal), ("novel_tpr@1%FPR", novel)]:
        a = (s[sel] > tm).astype(float); bb = (ref[sel] > tr_).astype(float); n = len(a); idx = rng.integers(0, n, (B, n)); dd = a[idx].mean(1) - bb[idx].mean(1)
        brows.append({"model": tag, "metric": metric, "n": n, "lgbm": bb.mean(), "model_value": a.mean(), "diff": a.mean() - bb.mean(), "ci_lo": np.quantile(dd, .025), "ci_hi": np.quantile(dd, .975)})
bdf = pd.DataFrame(brows); bdf.to_csv(os.path.join(RES, "bootstrap_bodmas.csv"), index=False); print(bdf.round(4).to_string())
