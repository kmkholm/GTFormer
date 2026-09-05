"""
BODMAS (Yang et al., DLS 2021): 134,435 PE files, EMBER v2 features (2,381), timestamps, 582 families; third benchmark for generalisation.
Protocol (temporal): TRAIN = all files with first-seen <= 2020-02 (benign back to 2007 + malware Aug 2019-Feb 2020), VAL = 2020-03,
TEST = 2020-04 .. 2020-09 (per month). No AV consensus in BODMAS -> only plain objectives; strata = novel family (unseen in TRAIN), category.
Usage: python bodmas_exp.py <lgbm|mlp|gtformer|proto> [--seed 0] [--epochs 15]
"""
import os, sys, json, time, math, argparse
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import lamda_exp
from lamda_exp import Net, supcon_loss, dev
D = os.environ.get("BODMAS_DATA", "data_bodmas"); RES = os.environ.get("RESULTS_BODMAS", "results_bodmas"); os.makedirs(RES, exist_ok=True)
DIM = 2381
GROUPS = {"histogram": (0, 256), "byteentropy": (256, 512), "strings": (512, 616), "general": (616, 626), "header": (626, 688), "section": (688, 943),
          "imports": (943, 2223), "exports": (2223, 2351), "datadirectories": (2351, 2381)}
PP = os.path.join(D, "X_pp.npy")


def load():
    d = np.load(os.path.join(D, "bodmas.npz")); X = d["X"].astype(np.float32); y = d["y"].astype(int)
    m = pd.read_csv(os.path.join(D, "bodmas_metadata.csv")); m["ts"] = pd.to_datetime(m.timestamp, format="mixed", utc=True); m["ym"] = m.ts.dt.strftime("%Y-%m")
    c = pd.read_csv(os.path.join(D, "bodmas_malware_category.csv")); m = m.merge(c.rename(columns={"sha256": "sha"}), on="sha", how="left")
    m["label"] = y; m["fam"] = np.where(m.label == 1, m.family.fillna(""), "")
    tr = (m.ym <= "2020-02").values; va = (m.ym == "2020-03").values; ev = (m.ym >= "2020-04").values
    m["region"] = np.where(tr, "train", np.where(va, "val", "test"))
    return X, m, np.where(tr)[0], np.where(va)[0], np.where(ev)[0]


def prep(X, tr):
    if os.path.exists(PP): return np.load(PP)
    T = np.sign(X) * np.log1p(np.abs(X)); mu = T[tr].mean(0); sd = T[tr].std(0); sd[sd < 1e-6] = 1
    P = np.clip((T - mu) / sd, -12, 12).astype(np.float32); np.save(PP, P); return P


def save(tag, s_val, s_ev, va, ev, m, sec):
    np.savez_compressed(os.path.join(RES, f"scores_{tag}.npz"), s_val=s_val, y_val=m.label.values[va], s_ev=s_ev, y_ev=m.label.values[ev], idx_ev=ev, idx_val=va, train_sec=sec)
    print(f"{tag}: val AUC {roc_auc_score(m.label.values[va], s_val):.4f}  test AUC {roc_auc_score(m.label.values[ev], s_ev):.4f} ({sec:.0f}s)", flush=True)


def run_lgbm(X, m, tr, va, ev, tag, seed):
    """EMBER2024 benchmark LightGBM configuration, run on the GPU (OpenCL device) -- no CPU training."""
    import lightgbm as lgb
    from exp_main import BASE_PARAMS
    p = dict(BASE_PARAMS); p.update({"seed": seed, "device_type": "gpu", "num_threads": 2}); y = m.label.values
    dtr = lgb.Dataset(X[tr], y[tr]); dva = lgb.Dataset(X[va], y[va], reference=dtr)
    t0 = time.time(); b = lgb.train(p, dtr, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
    save(tag, b.predict(X[va], num_iteration=b.best_iteration), b.predict(X[ev], num_iteration=b.best_iteration), va, ev, m, time.time() - t0)


def run_deep(P, m, tr, va, ev, arch, tag, seed, epochs, bs=512):
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    lamda_exp.DIM = DIM
    groups = [list(range(s, e)) for s, e in GROUPS.values()]
    famc = pd.Series(m.fam.values[tr]).value_counts(); vocab = {f: i for i, f in enumerate(sorted(f for f, c in famc.items() if f and c >= 20))}
    y = m.label.values.astype(np.float32); famid = m.fam.map(vocab).fillna(-1).astype(np.int64).values.copy()
    Xtr = torch.from_numpy(P[tr]).to(dev); ytr = torch.from_numpy(y[tr]).to(dev); ftr = torch.from_numpy(famid[tr]).to(dev)
    HP = lamda_exp.HP; bs = HP.get("bs", bs)
    model = Net(arch, groups, len(vocab), d=HP.get("d", 192), drop=HP.get("drop", 0.1), layers=HP.get("layers", 3), heads=HP.get("heads", 4)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=HP.get("lr", 1e-3 if arch == "gtformer" else 5e-4), weight_decay=HP.get("wd", 1e-4))
    steps = len(tr) // bs; total = steps * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=total, pct_start=0.05)
    pos = y[tr].mean(); pw = torch.tensor((1 - pos) / pos, device=dev)
    def predict(idx):
        model.eval(); out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(idx), 8192): out.append(model(torch.from_numpy(P[idx[i:i + 8192]]).to(dev))[0].float().cpu().numpy())
        model.train(); return np.concatenate(out)
    best, best_state, t0 = -1, None, time.time()
    for ep in range(epochs):
        perm = torch.from_numpy(rng.permutation(len(tr))).to(dev)
        for b in range(steps):
            idx = perm[b * bs:(b + 1) * bs]; xb = Xtr[idx]; yb = ytr[idx]
            with torch.autocast("cuda", dtype=torch.bfloat16): logit, aux, fam, z = model(xb, 0.0)
            loss = F.binary_cross_entropy_with_logits(logit.float(), yb, pos_weight=pw)
            if arch == "proto":
                grp = torch.where(yb == 1, torch.where(ftr[idx] >= 0, ftr[idx] + 1, torch.full_like(ftr[idx], -1)), torch.zeros_like(ftr[idx]))
                loss = loss + 0.3 * supcon_loss(model.projz(z.float()), grp)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
        auc = roc_auc_score(y[va], predict(va)); print(f"  ep{ep} val AUC {auc:.5f}", flush=True)
        if auc > best: best, best_state = auc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    if HP.get("tuning"): return best
    torch.save({"state": best_state, "arch": arch}, os.path.join(RES, f"model_{tag}.pt"))
    save(tag, predict(va), predict(ev), va, ev, m, time.time() - t0); return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--hp", default="", help="json file with tuned hyper-parameters"); ap.add_argument("--tag", default="")
    a = ap.parse_args(); tag = a.tag or (f"{a.model}_plain" + (f"_s{a.seed}" if a.seed else ""))
    if a.hp: lamda_exp.HP.update(json.load(open(a.hp)))
    if os.path.exists(os.path.join(RES, f"scores_{tag}.npz")): print("exists", tag); sys.exit()
    X, m, tr, va, ev = load(); print(f"train {len(tr)} (mal {m.label.values[tr].sum()}) val {len(va)} test {len(ev)}", flush=True)
    if a.model == "lgbm": run_lgbm(X, m, tr, va, ev, tag, a.seed)
    else: run_deep(prep(X, tr), m, tr, va, ev, a.model, tag, a.seed, a.epochs)
