"""SHAP (TreeExplainer) on the EMBER2024 LightGBM benchmark models: mean |SHAP| per feature group and top individual features,
for test benign, test malware and challenge (evasive) malware, per format. Output results/shap_groups.csv, results/shap_top_features.csv"""
import os, numpy as np, pandas as pd, lightgbm as lgb, shap, thrember
from common import load, RES, FILETYPES, DIM
from gtformer import GROUPS

names = []
ex = thrember.PEFeatureExtractor()
for fe in ex.features:
    names += [f"{fe.name}[{i}]" for i in range(fe.dim)]
gid = np.zeros(DIM, int); gname = [g[0] for g in GROUPS]
for i, (_, s, e) in enumerate(GROUPS): gid[s:e] = i
rng = np.random.default_rng(0); rows, tops = [], []
for ft in FILETYPES:
    mp = os.path.join(RES, f"model_{ft}_base.txt")
    if not os.path.exists(mp): continue
    bst = lgb.Booster(model_file=mp); expl = shap.TreeExplainer(bst)
    Xte, mte = load(f"{ft}_test", mmap=False); Xch, mch = load("challenge", mmap=False); Xch = Xch[(mch.file_type == ft).values]
    sets = {"benign": Xte[rng.choice(np.where(mte.label == 0)[0], min(3000, (mte.label == 0).sum()), replace=False)],
            "malware": Xte[rng.choice(np.where(mte.label == 1)[0], min(3000, (mte.label == 1).sum()), replace=False)], "challenge": Xch}
    for grp, X in sets.items():
        sv = expl.shap_values(X)
        if isinstance(sv, list): sv = sv[1]
        a = np.abs(sv).mean(0); signed = sv.mean(0)
        r = {"format": ft, "group": grp, "n": len(X)}
        tot = a.sum()
        for i, g in enumerate(gname): r[g] = float(a[gid == i].sum() / tot)
        rows.append(r)
        top = np.argsort(-a)[:15]
        for k in top: tops.append({"format": ft, "group": grp, "feature": names[k], "index": int(k), "mean_abs_shap": float(a[k]), "mean_shap": float(signed[k])})
    print(ft, "done", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(RES, "shap_groups.csv"), index=False); pd.DataFrame(tops).to_csv(os.path.join(RES, "shap_top_features.csv"), index=False)
print(pd.DataFrame(rows).round(3).to_string())
