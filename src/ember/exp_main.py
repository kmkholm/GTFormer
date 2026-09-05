"""
Main experiment: per file type, train LightGBM detectors under different training-weight schemes and
save raw scores for test + challenge sets (analysis is done separately from the saved scores).

Variants
  base   : EMBER2024 benchmark config (is_unbalance) -- reproduction of Joyce et al. 2025
  cw     : consensus-weighted: malware weight = 1 + LAM * evasiveness, evasiveness = 1 - det_ratio/ref
  fb     : family-balanced: malware weight ∝ 1/sqrt(family frequency) (unknown family -> weight 1)
  cwfb   : both
  cwinv  : inverse (upweight high-consensus malware) -- sanity/ablation control
Validation for early stopping: last 4 weeks of training period (time-aware), max 500 rounds.
Usage: python exp_main.py <FileType> [variants...]
"""
import sys, os, json, time
import numpy as np, pandas as pd, lightgbm as lgb
from common import load, RES, CAT_FEATURES, Timer, save_json

LAM = 3.0
VAL_WEEKS = 4
BASE_PARAMS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lgbm_config.json")))
BASE_PARAMS.update({"verbosity": -1, "num_threads": 12, "metric": ["auc"]})
BASE_PARAMS.pop("task", None)


def make_weights(m, variant, train_mask):
    w = np.ones(len(m), dtype=np.float32)
    mal = (m["label"].values == 1)
    if variant in ("cw", "cwfb", "cwinv"):
        r = m["det_ratio"].values.astype(float)
        ref = np.nanquantile(r[mal & train_mask], 0.95)
        ev = np.clip(1.0 - r / ref, 0, 1)  # 0 = strongly detected, 1 = barely detected
        ev = np.nan_to_num(ev, nan=0.0)
        if variant == "cwinv":
            ev = 1.0 - ev
        w[mal] = 1.0 + LAM * ev[mal]
    if variant in ("fb", "cwfb"):
        fam = m["family"].values
        counts = pd.Series(fam[mal & train_mask]).value_counts()
        fw = np.ones(len(m), dtype=np.float32)
        known = mal & (fam != "")
        c = pd.Series(fam[known]).map(counts).fillna(1).values.astype(float)
        fwk = 1.0 / np.sqrt(c)
        fwk = fwk / fwk.mean()  # mean 1 over known-family malware
        fw[known] = fwk
        w = w * fw
    # keep the total malware weight equal to the number of malware samples (so is_unbalance semantics stay comparable)
    w[mal] = w[mal] * (mal.sum() / w[mal].sum())
    return w


def run(ft, variants):
    Xtr, mtr = load(f"{ft}_train", mmap=False)
    Xte, mte = load(f"{ft}_test", mmap=False)
    Xch, mch = load("challenge", mmap=False)
    chmask = (mch["file_type"] == ft).values
    Xch, mch = Xch[chmask], mch[chmask].reset_index(drop=True)
    wk = mtr["week_id"].values
    val_mask = wk >= (wk.max() - VAL_WEEKS + 1)
    tr_mask = ~val_mask
    print(f"{ft}: train {tr_mask.sum()} val {val_mask.sum()} test {len(mte)} challenge {len(mch)}", flush=True)
    ytr = mtr["label"].values.astype(int)
    for v in variants:
        out = os.path.join(RES, f"scores_{ft}_{v}.npz")
        if os.path.exists(out):
            print("skip", out); continue
        w = make_weights(mtr, v, tr_mask)
        dtr = lgb.Dataset(Xtr[tr_mask], ytr[tr_mask], weight=w[tr_mask], categorical_feature=CAT_FEATURES, free_raw_data=False)
        dva = lgb.Dataset(Xtr[val_mask], ytr[val_mask], reference=dtr, categorical_feature=CAT_FEATURES, free_raw_data=False)
        with Timer(f"{ft}/{v} train") as T:
            bst = lgb.train(BASE_PARAMS, dtr, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
        s_val = bst.predict(Xtr[val_mask], num_iteration=bst.best_iteration)
        t0 = time.time(); s_te = bst.predict(Xte, num_iteration=bst.best_iteration); inf = (time.time() - t0) / len(Xte) * 1e6
        s_ch = bst.predict(Xch, num_iteration=bst.best_iteration)
        bst.save_model(os.path.join(RES, f"model_{ft}_{v}.txt"), num_iteration=bst.best_iteration)
        np.savez_compressed(out, s_val=s_val, y_val=ytr[val_mask], s_test=s_te, y_test=mte["label"].values,
                            s_ch=s_ch, best_iter=bst.best_iteration, train_sec=T.dt, infer_us=inf)
        from sklearn.metrics import roc_auc_score
        print(f"  {ft}/{v}: best_iter={bst.best_iteration} test AUC={roc_auc_score(mte['label'], s_te):.4f} "
              f"challenge mean score={s_ch.mean():.3f} infer={inf:.1f}us/sample", flush=True)


if __name__ == "__main__":
    ft = sys.argv[1]
    variants = sys.argv[2:] or ["base", "cw", "fb", "cwfb", "cwinv"]
    run(ft, variants)
