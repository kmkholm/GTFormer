"""Interpretability for GT-Former: CLS->group-token attention (last layer, head-averaged) for
benign test files, ordinary test malware, and challenge (evasive) malware, per format. Also gate usage for FR-MoE.
Outputs results/attention_groups.csv, results/moe_gates.csv
"""
import os, json, numpy as np, pandas as pd, torch
from gtformer import GTFormer, GROUPS, FT2ID, PP, dev
from common import RES, DIM, FILETYPES, fp, keep_mask

def load_model(tag):
    ck = torch.load(os.path.join(RES, f"gtformer_{tag}.pt"), map_location=dev, weights_only=False)
    n_fam = len(json.load(open(os.path.join(PP, "families.json"))))
    m = GTFormer(n_fam=n_fam, arch=ck["cfg"]["arch"], aux=ck["cfg"]["aux"], grl=ck["cfg"]["grl"]).to(dev)
    m.load_state_dict(ck["state"]); m.eval(); return m

def attn_groups(model, X, ft, n=4000, seed=0):
    """Return mean attention from CLS to each token (ft token + 12 groups) in the last encoder layer."""
    rng = np.random.default_rng(seed); idx = np.sort(rng.choice(len(X), min(n, len(X)), replace=False))
    xb = torch.from_numpy(np.asarray(X[idx])).to(dev).float(); fb = torch.full((len(idx),), FT2ID[ft], device=dev)
    layer = model.encoder.layers[-1]
    store = {}
    def hook(mod, inp, out):
        pass
    # recompute attention manually with the layer's self_attn to get weights
    with torch.no_grad():
        toks = [p(xb[:, s:e]) + model.group_emb[i] for i, (p, (_, s, e)) in enumerate(zip(model.proj, GROUPS))]
        h = torch.stack(toks, 1)
        h = torch.cat([model.cls.expand(len(xb), -1, -1), model.ft_emb(fb)[:, None, :], h], 1)
        for l in model.encoder.layers[:-1]:
            h = l(h)
        hn = layer.norm1(h)  # norm_first
        _, w = layer.self_attn(hn, hn, hn, need_weights=True, average_attn_weights=True)
        a = w[:, 0, :].mean(0).cpu().numpy()  # CLS row
    return a

if __name__ == "__main__":
    rows = []
    tag = "gtformer_plain"
    model = load_model(tag)
    Xch = np.memmap(os.path.join(PP, "P_challenge.dat"), dtype=np.float16, mode="r").reshape(-1, DIM)
    mch = pd.read_parquet(fp("meta_challenge.parquet")); mch = mch[keep_mask(mch)].reset_index(drop=True)
    names = ["filetype"] + [g[0] for g in GROUPS]
    for ft in FILETYPES:
        Xte = np.memmap(os.path.join(PP, f"P_{ft}_test.dat"), dtype=np.float16, mode="r").reshape(-1, DIM)
        mte = pd.read_parquet(fp(f"meta_{ft}_test.parquet")); mte = mte[keep_mask(mte)].reset_index(drop=True)
        for grp, sel in [("benign", (mte.label == 0).values), ("malware", (mte.label == 1).values)]:
            a = attn_groups(model, Xte[np.where(sel)[0]], ft)
            rows.append({"format": ft, "group": grp, **{n: float(v) for n, v in zip(names, a[1:])}})
        cm = np.where((mch.file_type == ft).values)[0]
        a = attn_groups(model, Xch[cm], ft)
        rows.append({"format": ft, "group": "challenge", **{n: float(v) for n, v in zip(names, a[1:])}})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "attention_groups.csv"), index=False); print(df.round(3).to_string())
    # MoE gates
    p = os.path.join(RES, "gtformer_moe_plain.pt")
    if os.path.exists(p):
        moe = load_model("moe_plain"); grows = []
        for ft in FILETYPES:
            Xte = np.memmap(os.path.join(PP, f"P_{ft}_test.dat"), dtype=np.float16, mode="r").reshape(-1, DIM)
            idx = np.sort(np.random.default_rng(0).choice(len(Xte), 4000, replace=False))
            with torch.no_grad():
                moe.embed(torch.from_numpy(np.asarray(Xte[idx])).to(dev).float(), torch.full((len(idx),), FT2ID[ft], device=dev))
            g = moe._gate.mean(0).cpu().numpy(); grows.append({"format": ft, **{f"expert{i}": float(v) for i, v in enumerate(g)}})
        pd.DataFrame(grows).to_csv(os.path.join(RES, "moe_gates.csv"), index=False); print(pd.DataFrame(grows).round(3).to_string())
