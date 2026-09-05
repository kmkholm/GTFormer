"""
Stream EMBER2024 raw-feature zips (jsonl inside) -> vectorized float32 features (.dat) + metadata (.parquet)
No extraction to disk (raw jsonl would be ~125 GB). Multiprocess vectorization with thrember feature v3.
Usage: python vectorize.py <zipname> [<zipname> ...]
"""
import sys, os, zipfile, json, time, multiprocessing as mp
import numpy as np
import pandas as pd

DATA = os.environ.get("EMBER_DATA", "data")
OUT = os.environ.get("EMBER_OUT", os.environ.get("EMBER_FEAT", "features"))
META_KEYS = ["sha256", "first_submission_date", "last_analysis_date", "detection_ratio", "label",
             "file_type", "family", "family_confidence", "week_id"]
LIST_KEYS = ["behavior", "file_property", "packer", "exploit", "group"]

_ex = None
def _init():
    global _ex
    import thrember
    _ex = thrember.PEFeatureExtractor()

def _work(line):
    d = json.loads(line)
    v = np.asarray(_ex.process_raw_features(d), dtype=np.float32)
    m = {k: d.get(k) for k in META_KEYS}
    for k in LIST_KEYS:
        m[k] = "|".join(d.get(k) or [])
    m["n_caps"] = len(d.get("caps") or [])
    m["n_ttps"] = len(d.get("ttps") or [])
    m["n_mbc"] = len(d.get("mbc") or [])
    m["ttps"] = "|".join(sorted({t.get("Technique", "") for t in (d.get("ttps") or [])}))
    m["size"] = (d.get("general") or {}).get("size")
    m["entropy"] = (d.get("general") or {}).get("entropy")
    m["is_pe"] = (d.get("general") or {}).get("is_pe")
    return v.tobytes(), m

import re
_SHA = re.compile(rb'"sha256":\s*"([0-9a-f]{64})"')
def _lines(zpath):
    """Yield unique records (the public release contains every record twice; keep first occurrence by sha256)."""
    z = zipfile.ZipFile(zpath)
    seen = set(); ndup = 0
    for n in sorted(z.namelist()):
        if not n.endswith(".jsonl"):
            continue
        with z.open(n) as f:
            for line in f:
                if not line.strip():
                    continue
                mo = _SHA.search(line[:400])
                key = mo.group(1) if mo else line
                if key in seen:
                    ndup += 1; continue
                seen.add(key)
                yield line
    print(f"  {os.path.basename(zpath)}: skipped {ndup} duplicate records", flush=True)

def run(zipname):
    os.makedirs(OUT, exist_ok=True)
    stem = zipname.replace(".zip", "")
    xpath = os.path.join(OUT, f"X_{stem}.dat")
    mpath = os.path.join(OUT, f"meta_{stem}.parquet")
    if os.path.exists(mpath):
        print("skip (done)", stem, flush=True); return
    t0 = time.time(); n = 0; metas = []
    with open(xpath, "wb") as fx, mp.Pool(10, initializer=_init) as pool:
        for vb, m in pool.imap(_work, _lines(os.path.join(DATA, zipname)), chunksize=256):
            fx.write(vb); metas.append(m); n += 1
            if n % 50000 == 0:
                print(f"{stem}: {n} rows, {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(metas).to_parquet(mpath, index=False)
    print(f"DONE {stem}: {n} rows in {time.time()-t0:.0f}s -> {xpath}", flush=True)

if __name__ == "__main__":
    for zn in sys.argv[1:]:
        run(zn)
