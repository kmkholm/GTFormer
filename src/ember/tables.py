"""Build paper tables (markdown + csv) from results/metrics_all.csv"""
import os, numpy as np, pandas as pd
from common import RES, FILETYPES

df = pd.read_csv(os.path.join(RES, "metrics_all.csv"))
FT = {"Win32": "Win32", "Win64": "Win64", "Dot_Net": ".NET", "APK": "APK", "ELF": "ELF", "PDF": "PDF"}
NAMES = {"base": "LightGBM (EMBER2024 benchmark config)", "cw": "LightGBM + consensus weights", "fb": "LightGBM + family-balanced weights",
         "cwfb": "LightGBM + both", "cwinv": "LightGBM + inverse-consensus (control)", "mlp_mlp_plain": "MLP (plain BCE)", "mlp_full": "MLP + CAM/aux/GRL",
         "gtformer_full": "GT-Former (full)", "gtformer_plain": "GT-Former (plain BCE)", "gtformer_no_cam": "GT-Former − CAM", "gtformer_no_aux": "GT-Former − consensus head",
         "gtformer_no_grl": "GT-Former − family GRL", "moe_full": "FR-MoE (full)", "moe_plain": "FR-MoE (plain BCE)", "proto_full": "ProtoCon-Net (full)",
         "proto_plain": "ProtoCon-Net (plain BCE)", "proto_no_con": "ProtoCon-Net − SupCon", "ens3_deep": "Rank-ensemble: 3 deep", "ens4_deep_lgbm": "Rank-ensemble: 3 deep + LightGBM",
         "ens2_gt_lgbm": "Rank-ensemble: GT-Former + LightGBM", "ens_mlp_lgbm": "Rank-ensemble: MLP + LightGBM", "gtformer_full_knn": "GT-Former + open-set kNN stacker",
         "lgbm_joint": "LightGBM joint (all formats, file-type feature)", "gtformer_tuned": "GT-Former (Optuna-tuned)", "ens_gttuned_lgbm": "Rank-ensemble: tuned GT-Former + LightGBM"}
COLS = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("tpr@0.01", "TPR@1%FPR"), ("tpr@0.001", "TPR@0.1%FPR"), ("ch_det@0.01(test-thr)", "Challenge@1%FPR"),
        ("ch_det@0.001(test-thr)", "Challenge@0.1%FPR"), ("tpr@0.01_novel_fam", "Novel-fam TPR@1%"), ("tpr@0.01_known_fam", "Known-fam TPR@1%"),
        ("tpr@0.01_nofam", "No-fam TPR@1%"), ("tpr@0.01_lowconsensus_q1", "Low-consensus TPR@1%"), ("test_fpr@0.01(val-thr)", "Realized FPR (val-thr 1%)"),
        ("ch_det@0.01(val-thr)", "Challenge@val-thr 1%")]

def fmt(v):
    return "" if pd.isna(v) else f"{v:.4f}" if abs(v) < 1.5 else f"{v:.0f}"

out = []
# Table: main comparison per format
main_models = ["base", "mlp_mlp_plain", "gtformer_plain", "gtformer_tuned", "moe_plain", "proto_plain", "gtformer_full", "moe_full", "proto_full", "ens3_deep", "ens4_deep_lgbm", "ens_mlp_lgbm", "ens2_gt_lgbm", "ens_gttuned_lgbm"]
for ft in FILETYPES:
    sub = df[(df.file_type == ft) & df.variant.isin(main_models)].set_index("variant").reindex([m for m in main_models if m in set(df.variant)])
    if sub.dropna(how="all").empty: continue
    out.append(f"\n### Table: {FT[ft]} — main comparison\n")
    out.append("| Model | " + " | ".join(c[1] for c in COLS) + " |"); out.append("|" + "---|" * (len(COLS) + 1))
    for v, r in sub.iterrows():
        if pd.isna(r["roc_auc"]): continue
        out.append(f"| {NAMES.get(v, v)} | " + " | ".join(fmt(r[c[0]]) for c in COLS) + " |")
# Table: mean over formats for every variant (unweighted mean of per-format metrics)
out.append("\n### Table: mean over the six formats (unweighted)\n")
agg = df.groupby("variant")[[c[0] for c in COLS[:10]]].mean()
agg["n_formats"] = df.groupby("variant").size()
agg = agg.sort_values("ch_det@0.01(test-thr)", ascending=False)
out.append("| Model | n formats | " + " | ".join(c[1] for c in COLS[:10]) + " |"); out.append("|" + "---|" * 12)
for v, r in agg.iterrows():
    out.append(f"| {NAMES.get(v, v)} | {int(r['n_formats'])} | " + " | ".join(fmt(r[c[0]]) for c in COLS[:10]) + " |")
# Table: ablations (GT-Former, MoE, Proto, MLP) mean over formats
abl = ["gtformer_full", "gtformer_no_cam", "gtformer_no_aux", "gtformer_no_grl", "gtformer_plain", "mlp_full", "mlp_mlp_plain", "moe_full", "moe_plain", "proto_full", "proto_no_con", "proto_plain"]
out.append("\n### Table: ablations (mean over formats)\n")
out.append("| Variant | n formats | TPR@1% | TPR@0.1% | Challenge@1% | Challenge@0.1% | Novel-fam TPR@1% | Low-consensus TPR@1% |"); out.append("|---|---|---|---|---|---|---|---|")
for v in abl:
    if v in agg.index:
        r = agg.loc[v]
        out.append(f"| {NAMES.get(v, v)} | {int(r['n_formats'])} | {fmt(r['tpr@0.01'])} | {fmt(r['tpr@0.001'])} | {fmt(r['ch_det@0.01(test-thr)'])} | {fmt(r['ch_det@0.001(test-thr)'])} | {fmt(r['tpr@0.01_novel_fam'])} | {fmt(r['tpr@0.01_lowconsensus_q1'])} |")
# seeds
seeds = df[df.variant.str.contains(r"_(plain|tuned)(_s\d)?$", regex=True) & df.variant.str.contains("gtformer|mlp_mlp|moe")].copy()
if len(seeds):
    seeds["arch"] = seeds.variant.str.replace(r"_s\d$", "", regex=True)
    g = seeds.groupby(["arch", "file_type"])[["roc_auc", "tpr@0.01", "tpr@0.001", "ch_det@0.01(test-thr)", "tpr@0.01_novel_fam"]].agg(["mean", "std", "count"])
    out.append("\n### Table: seed variability (mean ± sd over seeds)\n"); out.append(g.round(4).to_string())
# efficiency
eff = df.groupby("variant")[["train_sec", "infer_us"]].mean().round(1)
out.append("\n### Table: cost\n"); out.append(eff.to_string())
open(os.path.join(RES, "tables.md"), "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
