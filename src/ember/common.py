"""Shared loading / metric utilities for the EMBER2024 evasion-aware study."""
import os, json, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

FEAT = os.environ.get("EMBER_FEAT", "features")
FEAT_DIRS = os.environ.get("EMBER_FEAT_DIRS", "features").split(os.pathsep)

def fp(name):
    """Resolve a feature/meta file across the feature directories (E: filled up; Win32 train lives on C:)."""
    for d in FEAT_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(FEAT_DIRS[0], name)
RES = os.environ.get("RESULTS", "results")
DIM = 2568
os.makedirs(RES, exist_ok=True)
FILETYPES = ["Win32", "Win64", "Dot_Net", "APK", "ELF", "PDF"]
CAT_FEATURES = [2, 3, 4, 5, 6, 701, 702]  # per thrember.train_model


def keep_mask(m):
    """First occurrence of each sha256 (the public EMBER2024 release lists every record twice)."""
    return (~m["sha256"].duplicated()).values


def load(stem, mmap=True):
    X = np.memmap(fp(f"X_{stem}.dat"), dtype=np.float32, mode="r").reshape(-1, DIM)
    m = pd.read_parquet(fp(f"meta_{stem}.parquet"))
    assert len(m) == X.shape[0], (stem, len(m), X.shape)
    k = keep_mask(m)
    m = add_derived(m[k].reset_index(drop=True))
    if k.all():
        return (X if mmap else np.asarray(X)), m
    return X[np.where(k)[0]], m


def add_derived(m):
    m = m.copy()
    parts = m["detection_ratio"].astype(str).str.split("/", expand=True)
    m["det"] = pd.to_numeric(parts[0], errors="coerce")
    m["tot"] = pd.to_numeric(parts[1], errors="coerce")
    m["det_ratio"] = (m["det"] / m["tot"]).astype(float)
    m["family"] = m["family"].fillna("")
    return m


def tpr_at_fpr(y, s, fpr_target):
    """TPR at a given FPR using ROC curve interpolation (conservative: largest threshold with fpr<=target)."""
    fpr, tpr, thr = roc_curve(y, s)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def threshold_at_fpr(benign_scores, fpr_target):
    """Score threshold such that at most fpr_target of the given benign scores exceed it."""
    b = np.sort(np.asarray(benign_scores))
    k = int(np.floor(fpr_target * len(b)))  # number of benign allowed above threshold
    if k <= 0:
        return float(b[-1]) + 1e-12
    return float(b[-k])  # k-th largest benign score; predicted positive if score > thr... use >=? keep > for strictness


def detection_rate(scores, thr):
    return float(np.mean(np.asarray(scores) > thr))


def summarize_binary(y, s, name=""):
    out = {"name": name, "n": int(len(y)), "roc_auc": float(roc_auc_score(y, s)),
           "pr_auc": float(average_precision_score(y, s))}
    for f in (0.01, 0.001):
        out[f"tpr@{f}"] = tpr_at_fpr(y, s, f)
    return out


def save_json(obj, name):
    with open(os.path.join(RES, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)


class Timer:
    def __init__(self, label): self.label = label
    def __enter__(self): self.t = time.time(); return self
    def __exit__(self, *a): self.dt = time.time() - self.t; print(f"[{self.label}] {self.dt:.1f}s", flush=True)
