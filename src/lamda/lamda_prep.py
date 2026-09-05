"""Assemble LAMDA (Baseline, 4,561 int8 features) into one int8 matrix + metadata parquet, and build feature-group index by Drebin prefix."""
import os, json, numpy as np, pandas as pd
B = os.environ.get("LAMDA_DATA", "data_lamda/Baseline")
OUT = os.environ.get("LAMDA_PP", "preprocessed/lamda"); os.makedirs(OUT, exist_ok=True)
fm = pd.read_csv(os.path.join(B, "feature_mapping.csv"))
feat_cols = [f"feat_{i}" for i in range(len(fm))]
groups = fm.feature_name.str.split("_").str[0].values
gnames = sorted(set(groups)); gidx = {g: np.where(groups == g)[0].tolist() for g in gnames}
json.dump({"groups": gnames, "index": gidx}, open(os.path.join(OUT, "groups.json"), "w"))
print({g: len(v) for g, v in gidx.items()})
metas, Xs = [], []
for y in sorted(d for d in os.listdir(B) if os.path.isdir(os.path.join(B, d))):
    for s in ("train", "test"):
        df = pd.read_parquet(os.path.join(B, y, f"{y}_{s}.parquet"))
        Xs.append(df[feat_cols].to_numpy(dtype=np.int8))
        metas.append(df[["hash", "label", "family", "vt_count", "year_month"]].assign(year=int(y), split=s))
        print(y, s, len(df), flush=True)
X = np.concatenate(Xs); M = pd.concat(metas, ignore_index=True)
assert len(X) == len(M)
dup = M.hash.duplicated().sum(); print("duplicate hashes:", dup)
np.save(os.path.join(OUT, "X_int8.npy"), X); M.to_parquet(os.path.join(OUT, "meta.parquet"), index=False)
print("saved", X.shape)
