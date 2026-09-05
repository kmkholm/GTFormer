"""Re-score a saved deep checkpoint (raw logits) without retraining. Usage: python rescore.py <tag>"""
import os, sys, json, time, numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from gtformer import GTFormer, FT2ID, PP, dev, VAL_WEEKS
from common import RES, DIM, FILETYPES, fp, keep_mask

tag = sys.argv[1]
ck = torch.load(os.path.join(RES, f"gtformer_{tag}.pt"), map_location=dev, weights_only=False)
cfg = ck["cfg"]; n_fam = len(json.load(open(os.path.join(PP, "families.json"))))
model = GTFormer(n_fam=n_fam, arch=cfg["arch"], aux=cfg["aux"], grl=cfg["grl"]).to(dev); model.load_state_dict(ck["state"]); model.eval()

def predict(Xn, ftn):
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(Xn), 8192):
            xb = torch.from_numpy(np.asarray(Xn[i:i + 8192])).to(dev).float(); fb = torch.from_numpy(np.asarray(ftn[i:i + 8192])).to(dev)
            out.append(model(xb, fb)[0].float().cpu().numpy())
    return np.concatenate(out)

Xch = np.memmap(os.path.join(PP, "P_challenge.dat"), dtype=np.float16, mode="r").reshape(-1, DIM)
mch = pd.read_parquet(fp("meta_challenge.parquet")); mch = mch[keep_mask(mch)].reset_index(drop=True)
for ft in FILETYPES:
    Xtr = np.memmap(os.path.join(PP, f"P_{ft}_train.dat"), dtype=np.float16, mode="r").reshape(-1, DIM)
    mtr = pd.read_parquet(fp(f"meta_{ft}_train.parquet")); mtr = mtr[keep_mask(mtr)].reset_index(drop=True)
    wk = mtr.week_id.values; vsel = np.where(wk >= wk.max() - VAL_WEEKS + 1)[0]
    Xte = np.memmap(os.path.join(PP, f"P_{ft}_test.dat"), dtype=np.float16, mode="r").reshape(-1, DIM)
    mte = pd.read_parquet(fp(f"meta_{ft}_test.parquet")); mte = mte[keep_mask(mte)].reset_index(drop=True)
    t1 = time.time(); s_te = predict(Xte, np.full(len(Xte), FT2ID[ft])); inf = (time.time() - t1) / len(Xte) * 1e6
    cm = (mch.file_type == ft).values
    s_ch = predict(Xch[cm], np.full(cm.sum(), FT2ID[ft]))
    s_v = predict(Xtr[vsel], np.full(len(vsel), FT2ID[ft]))
    old = os.path.join(RES, f"scores_{ft}_{tag}.npz"); prev = np.load(old) if os.path.exists(old) else None
    np.savez_compressed(old, s_val=s_v, y_val=mtr.label.values[vsel], s_test=s_te, y_test=mte.label.values, s_ch=s_ch, best_iter=0,
                        train_sec=float(prev["train_sec"]) if prev is not None else 0.0, infer_us=inf)
    y = mte.label.values
    print(f"{ft}: AUC {roc_auc_score(y, s_te):.4f}  benign logits > 17 (would tie at sigmoid=1): {(s_te[y==0] > 17).sum()}", flush=True)
print("rescored", tag)
