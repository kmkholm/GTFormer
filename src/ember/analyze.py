"""Compute all evaluation metrics from saved score files -> results/metrics_<tag>.csv"""
import sys, os, glob
import numpy as np, pandas as pd
from common import load, RES, summarize_binary, threshold_at_fpr, detection_rate, add_derived, FILETYPES, fp, keep_mask

_cache = {}
def meta(stem):
    if stem not in _cache:
        m = pd.read_parquet(fp(f"meta_{stem}.parquet"))
        _cache[stem] = add_derived(m[keep_mask(m)].reset_index(drop=True))
    return _cache[stem]


def evaluate(ft, tag, s_val, y_val, s_test, y_test, s_ch, extra=None):
    mte = meta(f"{ft}_test"); mtr = meta(f"{ft}_train"); mch = meta("challenge")
    mch = mch[mch.file_type == ft].reset_index(drop=True)
    assert len(mte) == len(s_test) and len(mch) == len(s_ch)
    r = {"file_type": ft, "variant": tag}
    r.update({k: v for k, v in summarize_binary(y_test, s_test).items() if k not in ("name",)})
    # thresholds from test benign (as in Joyce et al. / Horvath & Zsigmond: challenge mixed with test benign)
    b_test = s_test[y_test == 0]
    b_val = s_val[y_val == 0]
    for f in (0.01, 0.001):
        thr_t = threshold_at_fpr(b_test, f); thr_v = threshold_at_fpr(b_val, f)
        r[f"ch_det@{f}(test-thr)"] = detection_rate(s_ch, thr_t)
        r[f"ch_det@{f}(val-thr)"] = detection_rate(s_ch, thr_v)
        r[f"test_tpr@{f}(val-thr)"] = detection_rate(s_test[y_test == 1], thr_v)
        r[f"test_fpr@{f}(val-thr)"] = detection_rate(b_test, thr_v)  # realized FPR when threshold is set on validation
    # novel-family analysis (test malware whose family never appears in training malware)
    train_fams = set(mtr.loc[mtr.label == 1, "family"].unique()) - {""}
    fam_te = mte["family"].values; mal = (y_test == 1)
    known = mal & np.isin(fam_te, list(train_fams)); novel = mal & (fam_te != "") & ~np.isin(fam_te, list(train_fams)); nofam = mal & (fam_te == "")
    thr1 = threshold_at_fpr(b_test, 0.01)
    r["n_test_mal_known_fam"] = int(known.sum()); r["n_test_mal_novel_fam"] = int(novel.sum()); r["n_test_mal_nofam"] = int(nofam.sum())
    r["tpr@0.01_known_fam"] = detection_rate(s_test[known], thr1) if known.any() else np.nan
    r["tpr@0.01_novel_fam"] = detection_rate(s_test[novel], thr1) if novel.any() else np.nan
    r["tpr@0.01_nofam"] = detection_rate(s_test[nofam], thr1) if nofam.any() else np.nan
    # challenge novel vs known family
    fam_ch = mch["family"].values
    ch_known = np.isin(fam_ch, list(train_fams)); ch_novel = (fam_ch != "") & ~ch_known; ch_nofam = fam_ch == ""
    r["ch_det@0.01_known_fam"] = detection_rate(s_ch[ch_known], thr1) if ch_known.any() else np.nan
    r["ch_det@0.01_novel_fam"] = detection_rate(s_ch[ch_novel], thr1) if ch_novel.any() else np.nan
    r["ch_det@0.01_nofam"] = detection_rate(s_ch[ch_nofam], thr1) if ch_nofam.any() else np.nan
    # low-consensus test malware (bottom quartile of det_ratio among test malware)
    dr = mte["det_ratio"].values
    q = np.nanquantile(dr[mal], 0.25)
    low = mal & (dr <= q); high = mal & (dr > q)
    r["tpr@0.01_lowconsensus_q1"] = detection_rate(s_test[low], thr1); r["tpr@0.01_highconsensus"] = detection_rate(s_test[high], thr1)
    # weekly TPR@1%FPR (threshold fixed from whole test benign) -> drift curve
    wk = mte["week_id"].values
    for w in sorted(np.unique(wk)):
        mw = mal & (wk == w)
        r[f"wk{w}_tpr@0.01"] = detection_rate(s_test[mw], thr1)
    if extra: r.update(extra)
    return r


def run(pattern="scores_*.npz", out="metrics_all.csv"):
    rows = []
    for p in sorted(glob.glob(os.path.join(RES, pattern))):
        name = os.path.basename(p)[7:-4]
        ft = next(f for f in sorted(FILETYPES, key=len, reverse=True) if name.startswith(f + "_"))
        tag = name[len(ft) + 1:]
        d = np.load(p)
        extra = {k: float(d[k]) for k in ("best_iter", "train_sec", "infer_us") if k in d}
        rows.append(evaluate(ft, tag, d["s_val"], d["y_val"], d["s_test"], d["y_test"], d["s_ch"], extra))
        print(f"{ft:8s} {tag:10s} AUC={rows[-1]['roc_auc']:.4f} TPR@1%={rows[-1]['tpr@0.01']:.4f} TPR@0.1%={rows[-1]['tpr@0.001']:.4f} "
              f"CH@1%={rows[-1]['ch_det@0.01(test-thr)']:.4f} CH@0.1%={rows[-1]['ch_det@0.001(test-thr)']:.4f} "
              f"novelfam={rows[-1]['tpr@0.01_novel_fam']:.3f} known={rows[-1]['tpr@0.01_known_fam']:.3f} lowcons={rows[-1]['tpr@0.01_lowconsensus_q1']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, out), index=False)
    return df


if __name__ == "__main__":
    run(*(sys.argv[1:]))
