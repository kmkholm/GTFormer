"""
Open-set residual scoring on top of a trained deep encoder:
  for each query, k-NN cosine distances (in the encoder's embedding space) to benign training files and to malware training files,
  combined with the detector logit by a tiny logistic stacker fitted ONLY on the time-aware validation window (weeks 48-51).
Usage: python knn_score.py <tag> <FileType> [k]
Writes scores_<ft>_<tag>_knn.npz  (same layout as other score files)
"""
import os, sys, json, time, numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from gtformer import GTFormer, FT2ID, PP, dev, VAL_WEEKS
from common import RES, DIM, fp, keep_mask

tag, ft = sys.argv[1], sys.argv[2]; K = int(sys.argv[3]) if len(sys.argv) > 3 else 10
ck = torch.load(os.path.join(RES, f"gtformer_{tag}.pt"), map_location=dev, weights_only=False); cfg = ck["cfg"]
n_fam = len(json.load(open(os.path.join(PP, "families.json"))))
model = GTFormer(n_fam=n_fam, arch=cfg["arch"], aux=cfg["aux"], grl=cfg["grl"]).to(dev); model.load_state_dict(ck["state"]); model.eval()

def run(X):
    zs, ls = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), 8192):
            xb = torch.from_numpy(np.asarray(X[i:i + 8192])).to(dev).float(); fb = torch.full((len(xb),), FT2ID[ft], device=dev)
            logit, _, _, z = model(xb, fb); zs.append(torch.nn.functional.normalize(z.float(), dim=-1).half()); ls.append(logit.float().cpu().numpy())
    return torch.cat(zs), np.concatenate(ls)

def knn_dist(Q, R, k=K, bs=1024):
    """mean cosine distance to the k nearest rows of R (both L2-normalised, half precision on GPU)"""
    out = []
    for i in range(0, len(Q), bs):
        sim = Q[i:i + bs] @ R.T
        top = sim.float().topk(k, dim=1).values
        out.append((1 - top.mean(1)).cpu().numpy())
    return np.concatenate(out)

t0 = time.time()
Xtr = np.memmap(os.path.join(PP, f"P_{ft}_train.dat"), dtype=np.float16, mode="r").reshape(-1, DIM)
mtr = pd.read_parquet(fp(f"meta_{ft}_train.parquet")); mtr = mtr[keep_mask(mtr)].reset_index(drop=True)
wk = mtr.week_id.values; val = wk >= wk.max() - VAL_WEEKS + 1; y = mtr.label.values
Ztr, Ltr = run(Xtr)
Zref_b = Ztr[torch.from_numpy(np.where((~val) & (y == 0))[0]).to(dev)]
Zref_m = Ztr[torch.from_numpy(np.where((~val) & (y == 1))[0]).to(dev)]
Xte = np.memmap(os.path.join(PP, f"P_{ft}_test.dat"), dtype=np.float16, mode="r").reshape(-1, DIM)
mte = pd.read_parquet(fp(f"meta_{ft}_test.parquet")); mte = mte[keep_mask(mte)].reset_index(drop=True)
Xch = np.memmap(os.path.join(PP, "P_challenge.dat"), dtype=np.float16, mode="r").reshape(-1, DIM)
mch = pd.read_parquet(fp("meta_challenge.parquet")); mch = mch[keep_mask(mch)].reset_index(drop=True)
Zte, Lte = run(Xte); Zch, Lch = run(Xch[np.where((mch.file_type == ft).values)[0]])
vidx = torch.from_numpy(np.where(val)[0]).to(dev); Zva, Lva = Ztr[vidx], Ltr[val]

def feats(Z, L):
    db, dm = knn_dist(Z, Zref_b), knn_dist(Z, Zref_m)
    return np.column_stack([L, db, dm, db - dm, np.minimum(db, dm)])
Fva, Fte, Fch = feats(Zva, Lva), feats(Zte, Lte), feats(Zch, Lch)
print(f"features in {time.time()-t0:.0f}s; val AUC logit {roc_auc_score(y[val], Lva):.4f}  d_ben {roc_auc_score(y[val], Fva[:,1]):.4f}  d_ben-d_mal {roc_auc_score(y[val], Fva[:,3]):.4f}")
mu, sd = Fva.mean(0), Fva.std(0) + 1e-9
clf = LogisticRegression(C=1.0, max_iter=1000).fit((Fva - mu) / sd, y[val])
print("stacker coef", np.round(clf.coef_[0], 3))
s_val = clf.decision_function((Fva - mu) / sd); s_te = clf.decision_function((Fte - mu) / sd); s_ch = clf.decision_function((Fch - mu) / sd)
print(f"test AUC: logit-only {roc_auc_score(mte.label, Lte):.4f} -> stacked {roc_auc_score(mte.label, s_te):.4f}")
np.savez_compressed(os.path.join(RES, f"scores_{ft}_{tag}_knn.npz"), s_val=s_val, y_val=y[val], s_test=s_te, y_test=mte.label.values, s_ch=s_ch,
                    best_iter=0, train_sec=time.time() - t0, infer_us=0, coef=clf.coef_[0])
np.savez_compressed(os.path.join(RES, f"knnfeat_{ft}_{tag}.npz"), Fva=Fva, Fte=Fte, Fch=Fch)
print("done", ft, tag, f"{time.time()-t0:.0f}s")
