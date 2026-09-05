"""
GT-Former: Group-Token Transformer for evasion-robust, family-invariant static malware detection (EMBER2024).

Components (each ablatable):
  tokens : EMBER v3 vector split into 12 semantic group tokens (+ file-type token + CLS) -> Transformer encoder
  cam    : Consensus-Aware Margin loss: malware with low AV consensus (evasive-like) must be classified with a larger
           logit margin and receives a larger weight  (uses detection_ratio metadata, train only)
  aux    : auxiliary consensus-regression head (predict AV detection ratio of malware)
  grl    : family-adversarial head through a Gradient Reversal Layer -> family-invariant representation
Trained jointly on all six file formats. Scores saved per format in the same format as exp_main.py.

Usage: python gtformer.py <variant> [--formats Win32,Win64,...] [--epochs N]
variants: full | no_cam | no_grl | no_aux | no_tokens (flat MLP w/ all losses) | mlp_plain (flat MLP, BCE only) | tokens_plain
"""
import os, sys, json, time, math, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from common import RES, DIM, FILETYPES, add_derived, fp, keep_mask

PP = os.environ.get("EMBER_PP", "preprocessed")
FEAT = os.environ.get("EMBER_FEAT", "features")
GROUPS = [("general", 0, 7), ("histogram", 7, 263), ("byteentropy", 263, 519), ("strings", 519, 696),
          ("header", 696, 770), ("section", 770, 994), ("imports", 994, 2276), ("exports", 2276, 2405),
          ("datadirectories", 2405, 2439), ("richheader", 2439, 2472), ("authenticode", 2472, 2480),
          ("pefilewarnings", 2480, 2568)]
FT2ID = {f: i for i, f in enumerate(FILETYPES)}
VAL_WEEKS = 4
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class GTFormer(nn.Module):
    """arch: 'gtformer' (group tokens + transformer) | 'mlp' | 'moe' (format-routed mixture of experts) | 'proto' (prototype head on MLP encoder)"""
    def __init__(self, d=192, layers=3, heads=4, n_fam=1000, arch="gtformer", aux=True, grl=True, drop=0.1, n_exp=6, n_proto=4, tau=0.1, imports_split=1, ffn_mult=2):
        super().__init__()
        global GROUPS
        if imports_split > 1 and not any(g[0].startswith("imports_") for g in GROUPS):   # split the 1,282-dim imports block into k sub-tokens
            new = []
            for name, s0, e0 in GROUPS:
                if name == "imports":
                    edges = np.linspace(s0, e0, imports_split + 1).astype(int)
                    new += [(f"imports_{i}", int(edges[i]), int(edges[i + 1])) for i in range(imports_split)]
                else: new.append((name, s0, e0))
            GROUPS = new
        self.arch, self.aux_on, self.grl_on = arch, aux, grl
        tokens = arch == "gtformer"
        self.tokens = tokens
        self.n_proto, self.tau = n_proto, tau
        if arch == "moe":
            self.ft_emb = nn.Embedding(len(FILETYPES), 32)
            self.trunk = nn.Sequential(nn.Linear(DIM + 32, 1024), nn.GELU(), nn.LayerNorm(1024), nn.Dropout(drop), nn.Linear(1024, 512), nn.GELU(), nn.LayerNorm(512))
            self.out_norm = nn.LayerNorm(d)
            self.experts = nn.ModuleList([nn.Sequential(nn.Linear(512, 384), nn.GELU(), nn.Dropout(drop), nn.Linear(384, d), nn.GELU()) for _ in range(n_exp)])
            self.gate = nn.Sequential(nn.Linear(512 + 32, 128), nn.GELU(), nn.Linear(128, n_exp))
            self.shared = nn.Sequential(nn.Linear(512, d), nn.GELU())
        elif tokens:
            self.proj = nn.ModuleList([nn.Sequential(nn.Linear(e - s, d), nn.GELU(), nn.Linear(d, d)) for _, s, e in GROUPS])
            self.group_emb = nn.Parameter(torch.randn(len(GROUPS), d) * 0.02)
            self.ft_emb = nn.Embedding(len(FILETYPES), d)
            self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=ffn_mult * d, dropout=drop, activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(enc, layers)
            self.norm = nn.LayerNorm(d)
        else:  # flat MLP with comparable parameter budget
            self.ft_emb = nn.Embedding(len(FILETYPES), 32)
            self.mlp = nn.Sequential(nn.Linear(DIM + 32, 1024), nn.GELU(), nn.LayerNorm(1024), nn.Dropout(drop), nn.Linear(1024, 512), nn.GELU(),
                                     nn.LayerNorm(512), nn.Dropout(drop), nn.Linear(512, d), nn.GELU(), nn.LayerNorm(d))
        if arch == "proto":
            self.protos = nn.Parameter(torch.randn(2, n_proto, d) * 0.1)   # [benign, malware] x K prototypes
            self.projz = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 128))  # contrastive projection
        self.det = nn.Linear(d, 1)
        self.auxh = nn.Linear(d, 1)
        self.fam = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_fam))

    def embed(self, x, ft):
        if self.arch == "moe":
            fe = self.ft_emb(ft)
            h = self.trunk(torch.cat([x, fe], 1))
            g = torch.softmax(self.gate(torch.cat([h, fe], 1)), -1)          # format+content routed soft gate
            e = torch.stack([ex(h) for ex in self.experts], 1)                # B x E x d
            self._gate = g
            return self.out_norm((g[:, :, None] * e).sum(1) + self.shared(h))
        if self.tokens:
            toks = [p(x[:, s:e]) + self.group_emb[i] for i, (p, (_, s, e)) in enumerate(zip(self.proj, GROUPS))]
            h = torch.stack(toks, 1)
            h = torch.cat([self.cls.expand(len(x), -1, -1), self.ft_emb(ft)[:, None, :], h], 1)
            h = self.encoder(h)
            return self.norm(h[:, 0])
        return self.mlp(torch.cat([x, self.ft_emb(ft)], 1))

    def forward(self, x, ft, grl_lam=0.0):
        z = self.embed(x, ft)
        if self.arch == "proto":
            zn = F.normalize(z, dim=-1); pn = F.normalize(self.protos, dim=-1)
            sim = torch.einsum("bd,ckd->bck", zn, pn) / self.tau                # B x 2 x K
            cls = torch.logsumexp(sim, -1)                                      # soft-max over prototypes per class
            logit = cls[:, 1] - cls[:, 0]
        else:
            logit = self.det(z).squeeze(1)
        aux = self.auxh(z).squeeze(1) if self.aux_on else None
        fam = self.fam(GradReverse.apply(z, grl_lam)) if self.grl_on else None
        return logit, aux, fam, z


def supcon_loss(z, groups, tau=0.1):
    """Supervised contrastive loss; positives share the same group id (benign=0, malware family id+1, unknown-family malware = -1 -> anchored only)."""
    z = F.normalize(z.float(), dim=-1)
    sim = z @ z.t() / tau
    n = len(z)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (groups[:, None] == groups[None, :]) & ~eye & (groups[:, None] >= 0)
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    npos = pos.sum(1)
    valid = npos > 0
    if not valid.any():
        return z.sum() * 0
    return -((logp * pos).sum(1)[valid] / npos[valid]).mean()


def load_split(formats, split):
    Xs, metas = [], []
    for ft in formats:
        p = os.path.join(PP, f"P_{ft}_{split}.dat")
        X = np.memmap(p, dtype=np.float16, mode="r").reshape(-1, DIM)
        m = pd.read_parquet(fp(f"meta_{ft}_{split}.parquet"))
        m = add_derived(m[keep_mask(m)].reset_index(drop=True))
        assert len(m) == len(X), (ft, split, len(m), len(X))
        m["ft_id"] = FT2ID[ft]
        Xs.append(X); metas.append(m)
    return Xs, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant"); ap.add_argument("--formats", default=",".join(FILETYPES)); ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=2048); ap.add_argument("--lr", type=float, default=None); ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--arch", default="gtformer", choices=["gtformer", "mlp", "moe", "proto"])
    ap.add_argument("--d", type=int, default=192); ap.add_argument("--layers", type=int, default=3); ap.add_argument("--heads", type=int, default=4); ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=1e-4); ap.add_argument("--imports_split", type=int, default=1); ap.add_argument("--ffn_mult", type=int, default=2)
    ap.add_argument("--sub", type=int, default=0, help="random subsample of training rows (for tuning)"); ap.add_argument("--no_eval", action="store_true", help="skip test/challenge scoring (tuning)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    v = a.variant
    arch = "mlp" if v in ("no_tokens", "mlp_plain") else a.arch
    cfg = dict(arch=arch, cam=v not in ("no_cam", "mlp_plain", "plain"), aux=v not in ("no_aux", "mlp_plain", "plain"),
               grl=v not in ("no_grl", "mlp_plain", "plain"), con=(arch == "proto" and v != "no_con"))
    tag = (a.tag or f"{arch}_{v}")
    if a.lr is None:
        a.lr = 1e-3 if arch == "gtformer" else 5e-4
    formats = a.formats.split(",")
    fam_vocab = json.load(open(os.path.join(PP, "families.json")))
    print("variant", v, cfg, "formats", formats, "families", len(fam_vocab), flush=True)

    # ---- data (train in RAM as fp16) ----
    Xs, metas = load_split(formats, "train")
    N = sum(len(x) for x in Xs)
    X = np.empty((N, DIM), dtype=np.float16); o = 0
    for x in Xs:
        X[o:o + len(x)] = x; o += len(x)
    M = pd.concat(metas, ignore_index=True)
    y = M["label"].values.astype(np.float32)
    ftid = M["ft_id"].values.astype(np.int64)
    dr = M["det_ratio"].values.astype(np.float32)
    ref = np.nanquantile(dr[y == 1], 0.95)
    ev = np.where(y == 1, np.clip(1 - np.nan_to_num(dr, nan=ref) / ref, 0, 1), 0).astype(np.float32)
    famid = M["family"].map(fam_vocab).fillna(-1).astype(np.int64).values.copy()
    famid[y == 0] = -1
    wk = M["week_id"].values
    val = wk >= wk.max() - VAL_WEEKS + 1
    tr_idx = np.where(~val)[0]; va_idx = np.where(val)[0]
    if a.sub and a.sub < len(tr_idx): tr_idx = np.sort(np.random.default_rng(a.seed).choice(tr_idx, a.sub, replace=False))
    pos_frac = y[tr_idx].mean()
    n_ft = np.bincount(ftid[tr_idx], minlength=len(FILETYPES)).astype(float)
    p_samp = 1.0 / np.sqrt(n_ft[ftid[tr_idx]]); p_samp /= p_samp.sum()   # format-balanced (sqrt) sampling
    print("effective format shares:", {FILETYPES[i]: round(float(p_samp[ftid[tr_idx] == i].sum()), 3) for i in range(len(FILETYPES)) if n_ft[i] > 0}, flush=True)
    print(f"train {len(tr_idx)} val {len(va_idx)} pos_frac {pos_frac:.3f} fam-labelled {np.mean(famid[tr_idx]>=0):.3f}", flush=True)

    model = GTFormer(d=a.d, layers=a.layers, heads=a.heads, drop=a.drop, imports_split=a.imports_split, ffn_mult=a.ffn_mult, n_fam=len(fam_vocab), arch=cfg["arch"], aux=cfg["aux"], grl=cfg["grl"]).to(dev)
    L_CON = 0.3
    nparams = sum(p.numel() for p in model.parameters())
    print("params", nparams, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = len(tr_idx) // a.bs
    total = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    LAM_W, MARGIN, L_AUX, L_GRL = 1.0, 2.0, 0.5, 0.1

    def predict(Xn, ftn):
        model.eval(); out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(Xn), 8192):
                xb = torch.from_numpy(np.asarray(Xn[i:i + 8192])).to(dev).float()
                fb = torch.from_numpy(np.asarray(ftn[i:i + 8192])).to(dev)
                out.append(model(xb, fb)[0].float().cpu().numpy())   # raw logits: sigmoid saturates in fp32 and creates ties at the top of the ranking
        model.train(); return np.concatenate(out)

    best_auc, best_state, step, t0 = -1, None, 0, time.time()
    rng = np.random.default_rng(a.seed)
    for ep in range(a.epochs):
        perm = rng.choice(tr_idx, size=len(tr_idx), replace=True, p=p_samp)
        model.train(); run_loss = 0
        for b in range(steps_per_epoch):
            idx = np.sort(perm[b * a.bs:(b + 1) * a.bs])
            xb = torch.from_numpy(X[idx]).to(dev, non_blocking=True).float()
            yb = torch.from_numpy(y[idx]).to(dev); fb = torch.from_numpy(ftid[idx]).to(dev)
            eb = torch.from_numpy(ev[idx]).to(dev); drb = torch.from_numpy(dr[idx]).to(dev); famb = torch.from_numpy(famid[idx]).to(dev)
            p = step / total
            grl_lam = 2 / (1 + math.exp(-10 * p)) - 1 if cfg["grl"] else 0.0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit, aux, fam, z = model(xb, fb, grl_lam)
            logit = logit.float()
            if cfg["cam"]:
                adj = logit - MARGIN * eb * yb          # evasive malware must clear a larger margin
                w = 1 + LAM_W * eb * yb
                loss = (F.binary_cross_entropy_with_logits(adj, yb, reduction="none") * w).sum() / w.sum()
            else:
                loss = F.binary_cross_entropy_with_logits(logit, yb)
            if cfg["aux"]:
                mal = yb == 1
                if mal.any():
                    loss = loss + L_AUX * F.mse_loss(torch.sigmoid(aux.float()[mal]), torch.nan_to_num(drb[mal], nan=0.5))
            if cfg["con"]:
                grp = torch.where(yb == 1, torch.where(famb >= 0, famb + 1, torch.full_like(famb, -1)), torch.zeros_like(famb))
                loss = loss + L_CON * supcon_loss(model.projz(z.float()), grp)
            if cfg["grl"]:
                has = famb >= 0
                if has.any():
                    loss = loss + L_GRL * F.cross_entropy(fam.float()[has], famb[has])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); step += 1
            run_loss += loss.item()
            if b % 300 == 0:
                print(f"ep{ep} step{b}/{steps_per_epoch} loss {run_loss/(b+1):.4f} {time.time()-t0:.0f}s", flush=True)
        s_val = predict(X[va_idx], ftid[va_idx]); auc = roc_auc_score(y[va_idx], s_val)
        print(f"== epoch {ep} val AUC {auc:.5f} ({time.time()-t0:.0f}s)", flush=True)
        if auc > best_auc:
            best_auc = auc; best_state = {k: v_.detach().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    train_sec = time.time() - t0
    print(f"BEST_VAL_AUC {best_auc:.6f}", flush=True)
    if a.no_eval: sys.exit(0)
    torch.save({"state": best_state, "cfg": cfg, "val_auc": best_auc, "params": nparams}, os.path.join(RES, f"gtformer_{tag}.pt"))

    # ---- evaluation per format ----
    Xch = np.memmap(os.path.join(PP, "P_challenge.dat"), dtype=np.float16, mode="r").reshape(-1, DIM)
    mch = pd.read_parquet(fp("meta_challenge.parquet")); mch = mch[keep_mask(mch)].reset_index(drop=True)
    for ft in formats:
        Xte = np.memmap(os.path.join(PP, f"P_{ft}_test.dat"), dtype=np.float16, mode="r").reshape(-1, DIM)
        mte = pd.read_parquet(fp(f"meta_{ft}_test.parquet")); mte = mte[keep_mask(mte)].reset_index(drop=True)
        t1 = time.time(); s_te = predict(Xte, np.full(len(Xte), FT2ID[ft])); inf = (time.time() - t1) / len(Xte) * 1e6
        cm = (mch.file_type == ft).values
        s_ch = predict(Xch[cm], np.full(cm.sum(), FT2ID[ft]))
        vsel = va_idx[ftid[va_idx] == FT2ID[ft]]
        s_v = predict(X[vsel], ftid[vsel])
        np.savez_compressed(os.path.join(RES, f"scores_{ft}_{tag}.npz"), s_val=s_v, y_val=y[vsel], s_test=s_te,
                            y_test=mte["label"].values, s_ch=s_ch, best_iter=0, train_sec=train_sec, infer_us=inf)
        print(f"{ft}: test AUC {roc_auc_score(mte['label'], s_te):.4f} challenge mean {s_ch.mean():.3f} infer {inf:.1f}us", flush=True)
    print("DONE", tag, "val_auc", best_auc, "params", nparams, flush=True)


if __name__ == "__main__":
    main()
