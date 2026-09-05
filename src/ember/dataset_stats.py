"""Dataset statistics after de-duplication -> results/dataset_stats.json + tables"""
import os, json
import numpy as np, pandas as pd
from common import fp, keep_mask, add_derived, FILETYPES, RES

rows = []; fam_train = {}; det = {}
out = {"duplicates": {}, "per_format": {}, "challenge": {}, "families": {}}
for ft in FILETYPES:
    for split in ("train", "test"):
        p = fp(f"meta_{ft}_{split}.parquet")
        if not os.path.exists(p):
            continue
        m = pd.read_parquet(p)
        k = keep_mask(m); out["duplicates"][f"{ft}_{split}"] = {"rows_in_release": int(len(m)), "unique_sha256": int(k.sum())}
        m = add_derived(m[k].reset_index(drop=True))
        mal = m[m.label == 1]
        r = {"format": ft, "split": split, "n": len(m), "n_mal": int((m.label == 1).sum()), "n_ben": int((m.label == 0).sum()),
             "weeks": f"{m.week_id.min()}-{m.week_id.max()}", "n_families": int(mal.family.replace('', np.nan).nunique()),
             "frac_mal_no_family": float((mal.family == '').mean()), "det_ratio_median": float(mal.det_ratio.median()),
             "det_ratio_q25": float(mal.det_ratio.quantile(.25)), "det_ratio_q75": float(mal.det_ratio.quantile(.75))}
        if split == "train":
            fam_train[ft] = set(mal.family.unique()) - {''}
        else:
            tf = fam_train.get(ft, set())
            fams = mal.family
            r["test_mal_novel_family_frac"] = float(((fams != '') & ~fams.isin(tf)).mean())
            r["test_mal_known_family_frac"] = float(fams.isin(tf).mean())
        rows.append(r)
df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "dataset_stats.csv"), index=False); print(df.to_string())
mch = pd.read_parquet(fp("meta_challenge.parquet")); mch = add_derived(mch[keep_mask(mch)].reset_index(drop=True))
out["challenge"] = {"n": len(mch), "by_format": mch.file_type.value_counts().to_dict(), "det_ratio_median": float(mch.det_ratio.median()),
                    "det_ratio_q25": float(mch.det_ratio.quantile(.25)), "det_ratio_q75": float(mch.det_ratio.quantile(.75)),
                    "frac_no_family": float((mch.family == '').mean()), "weeks": f"{mch.week_id.min()}-{mch.week_id.max()}",
                    "frac_from_test_weeks": float((mch.week_id >= 52).mean())}
for ft in FILETYPES:
    c = mch[mch.file_type == ft]
    if len(c) and ft in fam_train:
        out["challenge"][f"{ft}_novel_family_frac"] = float(((c.family != '') & ~c.family.isin(fam_train[ft])).mean())
        out["challenge"][f"{ft}_known_family_frac"] = float(c.family.isin(fam_train[ft]).mean())
out["per_format"] = rows
json.dump(out, open(os.path.join(RES, "dataset_stats.json"), "w"), indent=2)
print(json.dumps(out["challenge"], indent=1)); print(out["duplicates"])
