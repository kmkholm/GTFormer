"""Preprocess vectorized features for the deep model: signed log1p + standardization (train stats) -> fp16 files.
Also builds the family vocabulary (train malware families with >= FAM_MIN samples) shared across formats.
Usage: python prep.py [stems...]   (default: all available)
"""
import os, sys, json
import numpy as np, pandas as pd
from common import load, FEAT, DIM, FILETYPES, fp, keep_mask

PP = os.environ.get("EMBER_PP", "preprocessed")
os.makedirs(PP, exist_ok=True)
FAM_MIN = 50
STATS = os.path.join(PP, "stats.npz")


def tf(x):
    return np.sign(x) * np.log1p(np.abs(x))


def fit_stats():
    if os.path.exists(STATS):
        return np.load(STATS)
    rng = np.random.default_rng(0)
    samples = []
    for ft in FILETYPES:
        p = fp(f"X_{ft}_train.dat")
        if not os.path.exists(p): continue
        X = np.memmap(p, dtype=np.float32, mode="r").reshape(-1, DIM)
        idx = np.sort(rng.choice(len(X), size=min(60000, len(X)), replace=False))
        samples.append(tf(np.asarray(X[idx], dtype=np.float64)))
        print("stats sample from", ft, len(idx), flush=True)
    S = np.concatenate(samples)
    mu = S.mean(0); sd = S.std(0); sd[sd < 1e-6] = 1.0
    np.savez(STATS, mu=mu.astype(np.float32), sd=sd.astype(np.float32))
    return np.load(STATS)


def build_family_vocab():
    p = os.path.join(PP, "families.json")
    if os.path.exists(p): return json.load(open(p))
    cnt = {}
    for ft in FILETYPES:
        mp = fp(f"meta_{ft}_train.parquet")
        if not os.path.exists(mp): continue
        m = pd.read_parquet(mp, columns=["label", "family"])
        for f, c in m.loc[(m.label == 1) & m.family.notna() & (m.family != ""), "family"].value_counts().items():
            cnt[f] = cnt.get(f, 0) + int(c)
    fams = sorted([f for f, c in cnt.items() if c >= FAM_MIN])
    vocab = {f: i for i, f in enumerate(fams)}
    json.dump(vocab, open(p, "w"))
    print("family vocab size", len(vocab), flush=True)
    return vocab


def process(stem, st):
    out = os.path.join(PP, f"P_{stem}.dat")
    if os.path.exists(out):
        print("skip", stem); return
    X = np.memmap(fp(f"X_{stem}.dat"), dtype=np.float32, mode="r").reshape(-1, DIM)
    keep = np.where(keep_mask(pd.read_parquet(fp(f"meta_{stem}.parquet"), columns=["sha256"])))[0]
    mu, sd = st["mu"], st["sd"]
    with open(out + ".tmp", "wb") as f:
        for i in range(0, len(keep), 100000):
            blk = (tf(np.asarray(X[keep[i:i + 100000]], dtype=np.float32)) - mu) / sd
            np.clip(blk, -12, 12, out=blk)
            f.write(blk.astype(np.float16).tobytes())
    os.replace(out + ".tmp", out)
    print("DONE", stem, len(keep), "of", len(X), flush=True)


if __name__ == "__main__":
    st = fit_stats(); build_family_vocab()
    stems = sys.argv[1:] or ["challenge"] + [f"{ft}_{s}" for ft in FILETYPES for s in ("train", "test")]
    for s in stems:
        if os.path.exists(fp(f"meta_{s}.parquet")):
            process(s, st)
