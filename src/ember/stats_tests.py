"""Paired bootstrap confidence intervals for the key comparisons (proposed vs LightGBM benchmark), per format.
Metrics: challenge detection rate at 1% FPR (threshold from test benign), TPR@1%FPR on test, TPR@1%FPR on novel-family test malware.
"""
import os, sys, json, numpy as np, pandas as pd
from common import RES, FILETYPES, threshold_at_fpr, fp, keep_mask, add_derived

B = 1000
rng = np.random.default_rng(0)

def meta(stem):
    m = pd.read_parquet(fp(f"meta_{stem}.parquet")); return add_derived(m[keep_mask(m)].reset_index(drop=True))

def boot_diff(a_hit, b_hit):
    """paired bootstrap of mean(a)-mean(b) over the same samples"""
    n = len(a_hit); idx = rng.integers(0, n, (B, n))
    d = a_hit[idx].mean(1) - b_hit[idx].mean(1)
    return float(np.mean(a_hit) - np.mean(b_hit)), float(np.quantile(d, .025)), float(np.quantile(d, .975)), float((d <= 0).mean())

def run(models, ref="base"):
    rows = []
    for ft in FILETYPES:
        pr = os.path.join(RES, f"scores_{ft}_{ref}.npz")
        if not os.path.exists(pr): continue
        R = np.load(pr); mte = meta(f"{ft}_test"); mtr = meta(f"{ft}_train"); mch = meta("challenge"); mch = mch[mch.file_type == ft]
        train_fams = set(mtr.loc[mtr.label == 1, "family"]) - {""}
        y = R["y_test"]; mal = y == 1
        novel = mal & (mte.family.values != "") & ~np.isin(mte.family.values, list(train_fams))
        thr_r = threshold_at_fpr(R["s_test"][y == 0], 0.01)
        for mname in models:
            pm = os.path.join(RES, f"scores_{ft}_{mname}.npz")
            if not os.path.exists(pm): continue
            M = np.load(pm); thr_m = threshold_at_fpr(M["s_test"][y == 0], 0.01)
            for metric, sel_r, sel_m in [
                ("challenge_det@1%FPR", R["s_ch"] > thr_r, M["s_ch"] > thr_m),
                ("test_TPR@1%FPR", R["s_test"][mal] > thr_r, M["s_test"][mal] > thr_m),
                ("novelfam_TPR@1%FPR", R["s_test"][novel] > thr_r, M["s_test"][novel] > thr_m)]:
                if len(sel_r) < 20: continue
                diff, lo, hi, p = boot_diff(sel_m.astype(float), sel_r.astype(float))
                rows.append({"format": ft, "model": mname, "ref": ref, "metric": metric, "n": len(sel_r), "ref_value": float(sel_r.mean()),
                             "model_value": float(sel_m.mean()), "diff": diff, "ci_lo": lo, "ci_hi": hi, "p_boot(one-sided)": p})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "bootstrap_tests.csv"), index=False)
    print(df.to_string()); return df

if __name__ == "__main__":
    run(sys.argv[1:] or ["gtformer_plain", "mlp_mlp_plain", "moe_plain", "gtformer_full", "moe_full", "proto_full", "ens3_deep", "ens4_deep_lgbm", "ens_mlp_lgbm", "ens2_gt_lgbm"])
