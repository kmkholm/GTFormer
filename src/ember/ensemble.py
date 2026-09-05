"""Build ensemble score files from saved per-model scores (rank-averaged probabilities)."""
import os, glob, numpy as np
from scipy.stats import rankdata
from common import RES, FILETYPES

ENSEMBLES = {
    "ens3_deep": ["gtformer_plain", "mlp_mlp_plain", "moe_plain"],
    "ens4_deep_lgbm": ["gtformer_plain", "mlp_mlp_plain", "moe_plain", "base"],
    "ens2_gt_lgbm": ["gtformer_plain", "base"],
    "ens_mlp_lgbm": ["mlp_mlp_plain", "base"],
    "ens3_full_deep": ["gtformer_full", "moe_full", "proto_full"],
    "ens_gttuned_lgbm": ["gtformer_tuned", "base"],
}

def rank01(s):
    return (rankdata(s) - 1) / max(len(s) - 1, 1)

for ft in FILETYPES:
    for name, members in ENSEMBLES.items():
        paths = [os.path.join(RES, f"scores_{ft}_{m}.npz") for m in members]
        if not all(os.path.exists(p) for p in paths):
            print("skip", ft, name); continue
        ds = [np.load(p) for p in paths]
        out = {}
        for key in ("s_val", "s_test", "s_ch"):
            # rank-normalise each member on the concatenation of the three subsets so thresholds stay comparable
            pass
        cat = [np.concatenate([d["s_val"], d["s_test"], d["s_ch"]]) for d in ds]
        r = np.mean([rank01(c) for c in cat], 0)
        nv, nt = len(ds[0]["s_val"]), len(ds[0]["s_test"])
        np.savez_compressed(os.path.join(RES, f"scores_{ft}_{name}.npz"), s_val=r[:nv], y_val=ds[0]["y_val"], s_test=r[nv:nv + nt],
                            y_test=ds[0]["y_test"], s_ch=r[nv + nt:], best_iter=0, train_sec=sum(float(d["train_sec"]) for d in ds),
                            infer_us=sum(float(d["infer_us"]) for d in ds))
        print("wrote", ft, name)
