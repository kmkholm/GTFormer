"""
LAMDA (Android, 2013-2025) experiments mirroring the EMBER2024 study, using the AnoShift-style protocol of the LAMDA paper:
  TRAIN  = all samples of 2013 and 2014 except the last month of each year (Dec 2013, Dec 2014 -> IID test)
  VAL    = random 15% of TRAIN (early stopping / model selection; removed from TRAIN). IID = 2013-12 and 2014-08 (last collected months).
  NEAR   = 2016-2017, FAR = 2018-2025 (evaluation only)
Models: lgbm (EMBER2024 benchmark config), mlp, gtformer (Drebin-category tokens), proto; variants plain | full (CAM + consensus head + family GRL)
Usage: python lamda_exp.py <model> [--variant plain|full] [--seed 0] [--epochs 12]
Outputs results_lamda/scores_<tag>.npz with scores for VAL and for every evaluation row (IID+NEAR+FAR) in meta order.
"""
import os, sys, json, time, math, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from gtformer import GradReverse, supcon_loss

D = os.environ.get("LAMDA_PP", "preprocessed/lamda"); RES = os.environ.get("RESULTS_LAMDA", "results_lamda"); os.makedirs(RES, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIM = 4561; FAM_MIN = 20; VT_REF = 30.0
HP = {}   # optional hyper-parameter overrides (Optuna): d, layers, heads, drop, lr, wd, bs


def load():
    X = np.load(os.path.join(D, "X_int8.npy"), mmap_mode="r"); M = pd.read_parquet(os.path.join(D, "meta.parquet"))
    M["month"] = M.year_month.str[-2:].astype(int)
    M["is_singleton"] = M.family.str.startswith("singleton")
    M["fam"] = np.where(M.is_singleton | (M.label == 0), "", M.family)
    last = M[M.year.isin([2013, 2014])].groupby("year").month.max().to_dict()   # last collected month of each year -> IID test (2013-12, 2014-08)
    is_last = M.apply(lambda r: r.year in last and r.month == last[r.year], axis=1).values
    train = (M.year.isin([2013, 2014]).values & ~is_last)
    rng = np.random.default_rng(0); val = np.zeros(len(M), bool); ti = np.where(train)[0]; val[rng.choice(ti, int(0.15 * len(ti)), replace=False)] = True
    train = train & ~val
    ev = (~train) & (~val)
    M["region"] = np.where(train, "train", np.where(val, "val", np.where(M.year.isin([2013, 2014]), "IID", np.where(M.year.isin([2016, 2017]), "NEAR", "FAR"))))
    return X, M, np.where(train)[0], np.where(val)[0], np.where(ev)[0]


class Net(nn.Module):
    def __init__(self, arch, groups, n_fam, d=192, drop=0.1, n_proto=4, tau=0.1, layers=3, heads=4):
        super().__init__(); self.arch = arch; self.groups = groups; self.n_proto, self.tau = n_proto, tau
        if arch == "gtformer":
            self.proj = nn.ModuleList([nn.Sequential(nn.Linear(len(g), d), nn.GELU(), nn.Linear(d, d)) for g in groups])
            self.gemb = nn.Parameter(torch.randn(len(groups), d) * 0.02); self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=drop, activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(enc, layers); self.norm = nn.LayerNorm(d)
            self.gidx = [torch.tensor(g, device=dev) for g in groups]
        else:
            self.mlp = nn.Sequential(nn.Linear(DIM, 1024), nn.GELU(), nn.LayerNorm(1024), nn.Dropout(drop), nn.Linear(1024, 512), nn.GELU(),
                                     nn.LayerNorm(512), nn.Dropout(drop), nn.Linear(512, d), nn.GELU(), nn.LayerNorm(d))
        if arch == "proto":
            self.protos = nn.Parameter(torch.randn(2, n_proto, d) * 0.1); self.projz = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 128))
        self.det = nn.Linear(d, 1); self.auxh = nn.Linear(d, 1); self.fam = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_fam))

    def embed(self, x):
        if self.arch == "gtformer":
            h = torch.stack([p(x[:, gi]) + self.gemb[i] for i, (p, gi) in enumerate(zip(self.proj, self.gidx))], 1)
            h = torch.cat([self.cls.expand(len(x), -1, -1), h], 1)
            return self.norm(self.encoder(h)[:, 0])
        return self.mlp(x)

    def forward(self, x, grl_lam=0.0):
        z = self.embed(x)
        if self.arch == "proto":
            sim = torch.einsum("bd,ckd->bck", F.normalize(z, dim=-1), F.normalize(self.protos, dim=-1)) / self.tau
            cls = torch.logsumexp(sim, -1); logit = cls[:, 1] - cls[:, 0]
        else:
            logit = self.det(z).squeeze(1)
        return logit, self.auxh(z).squeeze(1), self.fam(GradReverse.apply(z, grl_lam)), z


def run_lgbm(X, M, tr, va, ev, tag, seed):
    import lightgbm as lgb
    from exp_main import BASE_PARAMS
    p = dict(BASE_PARAMS); p.update({"seed": seed, "device_type": "gpu", "num_threads": 2})
    y = M.label.values.astype(int)
    Xtr = np.asarray(X[tr], dtype=np.float32); Xva = np.asarray(X[va], dtype=np.float32)
    dtr = lgb.Dataset(Xtr, y[tr]); dva = lgb.Dataset(Xva, y[va], reference=dtr)
    t0 = time.time(); bst = lgb.train(p, dtr, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
    s_val = bst.predict(Xva, num_iteration=bst.best_iteration)
    s_ev = np.concatenate([bst.predict(np.asarray(X[ev[i:i + 100000]], dtype=np.float32), num_iteration=bst.best_iteration) for i in range(0, len(ev), 100000)])
    save(tag, s_val, s_ev, va, ev, M, time.time() - t0)


def save(tag, s_val, s_ev, va, ev, M, train_sec):
    np.savez_compressed(os.path.join(RES, f"scores_{tag}.npz"), s_val=s_val, y_val=M.label.values[va], s_ev=s_ev, y_ev=M.label.values[ev], idx_ev=ev, idx_val=va, train_sec=train_sec)
    yv = M.label.values[ev]
    print(f"{tag}: val AUC {roc_auc_score(M.label.values[va], s_val):.4f}  eval AUC {roc_auc_score(yv, s_ev):.4f}  ({train_sec:.0f}s)", flush=True)


def run_deep(X, M, tr, va, ev, arch, variant, tag, seed, epochs, bs=1024):
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    G = json.load(open(os.path.join(D, "groups.json"))); groups = [G["index"][g] for g in G["groups"]]
    famc = pd.Series(M.fam.values[tr]).value_counts(); vocab = {f: i for i, f in enumerate(sorted(f for f, c in famc.items() if f and c >= FAM_MIN))}
    y = M.label.values.astype(np.float32); famid = M.fam.map(vocab).fillna(-1).astype(np.int64).values.copy()
    r = np.clip(M.vt_count.values / VT_REF, 0, 1).astype(np.float32); ev_w = np.where(y == 1, 1 - r, 0).astype(np.float32)
    full = variant == "full"
    Xtr = torch.from_numpy(np.asarray(X[tr])).to(dev)  # int8 on GPU (~0.8 GB)
    ytr = torch.from_numpy(y[tr]).to(dev); ftr = torch.from_numpy(famid[tr]).to(dev); rtr = torch.from_numpy(r[tr]).to(dev); etr = torch.from_numpy(ev_w[tr]).to(dev)
    bs = HP.get("bs", bs)
    model = Net(arch, groups, len(vocab), d=HP.get("d", 192), drop=HP.get("drop", 0.1), layers=HP.get("layers", 3), heads=HP.get("heads", 4)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=HP.get("lr", 1e-3 if arch == "gtformer" else 5e-4), weight_decay=HP.get("wd", 1e-4))
    steps = len(tr) // bs; total = steps * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=total, pct_start=0.05)
    pos = y[tr].mean(); pw = torch.tensor((1 - pos) / pos, device=dev)

    def predict(idx):
        model.eval(); out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(idx), 8192):
                xb = torch.from_numpy(np.asarray(X[idx[i:i + 8192]])).to(dev).float(); out.append(model(xb)[0].float().cpu().numpy())
        model.train(); return np.concatenate(out)

    best, best_state, step, t0 = -1, None, 0, time.time()
    for ep in range(epochs):
        perm = torch.from_numpy(rng.permutation(len(tr))).to(dev)
        for b in range(steps):
            idx = perm[b * bs:(b + 1) * bs]; xb = Xtr[idx].float(); yb = ytr[idx]
            grl = (2 / (1 + math.exp(-10 * step / total)) - 1) if full else 0.0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit, aux, fam, z = model(xb, grl)
            logit = logit.float()
            if full:
                adj = logit - 2.0 * etr[idx] * yb; w = (1 + etr[idx] * yb) * torch.where(yb == 1, pw, torch.ones_like(yb))
                loss = (F.binary_cross_entropy_with_logits(adj, yb, reduction="none") * w).sum() / w.sum()
                mal = yb == 1
                if mal.any(): loss = loss + 0.5 * F.mse_loss(torch.sigmoid(aux.float()[mal]), rtr[idx][mal])
                has = ftr[idx] >= 0
                if has.any(): loss = loss + 0.1 * F.cross_entropy(fam.float()[has], ftr[idx][has])
            else:
                loss = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pw)
            if arch == "proto":
                grp = torch.where(yb == 1, torch.where(ftr[idx] >= 0, ftr[idx] + 1, torch.full_like(ftr[idx], -1)), torch.zeros_like(ftr[idx]))
                loss = loss + 0.3 * supcon_loss(model.projz(z.float()), grp)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step(); step += 1
        s_val = predict(va); auc = roc_auc_score(y[va], s_val)
        print(f"  ep{ep} val AUC {auc:.5f} ({time.time()-t0:.0f}s)", flush=True)
        if auc > best: best, best_state = auc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    if HP.get("tuning"): return best
    torch.save({"state": best_state, "arch": arch, "variant": variant, "vocab_size": len(vocab)}, os.path.join(RES, f"model_{tag}.pt"))
    save(tag, predict(va), predict(ev), va, ev, M, time.time() - t0); return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--variant", default="plain"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--hp", default="", help="json file with tuned hyper-parameters"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.hp: HP.update(json.load(open(a.hp)))
    tag = a.tag or (f"{a.model}_{a.variant}" + (f"_s{a.seed}" if a.seed else ""))
    if os.path.exists(os.path.join(RES, f"scores_{tag}.npz")): print("exists", tag); sys.exit()
    X, M, tr, va, ev = load()
    print(f"train {len(tr)} val {len(va)} eval {len(ev)}  regions: {M.region.value_counts().to_dict()}", flush=True)
    if a.model == "lgbm": run_lgbm(X, M, tr, va, ev, tag, a.seed)
    else: run_deep(X, M, tr, va, ev, a.model, a.variant, tag, a.seed, a.epochs)
