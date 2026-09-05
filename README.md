# GT-Former: Group-Token Transformers and Hybrid Rank Fusion for Evasive and Novel-Family Malware Detection

Code, results, score files and paper source for the manuscript

> M. Tawfik, B. Al-sellami, M. Heshmat, I. S. Fathi, W. M. Shaban. *GT-Former: Group-Token Transformers and Hybrid Rank Fusion for Evasive and Novel-Family Malware Detection.* 2026 (under review).

The repository reproduces every table and figure of the paper on three public benchmarks:

| Benchmark | Files | Features | Protocol used in the paper |
|---|---|---|---|
| [EMBER2024](https://huggingface.co/datasets/joyce8/EMBER2024) | 2,626,000 train / 606,000 test / 6,315 challenge (after SHA-256 de-duplication) | EMBER v3, 2,568-d, six formats | official 52/12-week split, challenge set, time-aware validation (weeks 48–51) |
| [LAMDA](https://huggingface.co/datasets/IQSeC-Lab/LAMDA) | 1,008,381 APKs, 2013–2025 | Drebin-style, 4,561 binary | TRAIN 2013–14, IID, NEAR 2016–17, FAR 2018–25 (Haque et al., ICLR 2026) |
| [BODMAS](https://whyisyoung.github.io/BODMAS/) | 134,435 PE files | EMBER v2, 2,381-d | train ≤ Feb 2020, val Mar 2020, test Apr–Sep 2020 |

## What is in the repository

```
src/ember/                EMBER2024 pipeline: vectorisation, de-duplication, LightGBM benchmark, deep models,
                          rank fusion, bootstrap tests, tables, figures, SHAP / attention analysis
src/lamda/                LAMDA preprocessing, experiments, analysis and figures
src/bodmas/               BODMAS experiments and analysis
src/tuning_and_drivers/   Optuna study of GT-Former, five-seed driver, GPU job drivers
figures/drawio/           editable draw.io source and generator of the architecture figure (Figure 1)
results/                  all metric CSVs, bootstrap confidence intervals, SHAP and attention tables,
                          Optuna trial logs and every figure of the paper (PNG)
paper/                    LaTeX source of the manuscript (Scientific Reports template) with all figures
```

`src/ember/gtformer.py` contains the three deep architectures of the paper (`--arch gtformer | moe | proto | mlp`),
the three metadata-driven objectives (consensus-aware margin, consensus head, family gradient-reversal head) and
the training loop. `src/ember/common.py` holds the evaluation helpers (threshold at a fixed FPR, stratified
detection rates, the SHA-256 keep-masks that remove the duplicated records of the public EMBER2024 release).

Score files (`scores_*.npz`, one per model, seed and benchmark) and trained model checkpoints (`*.pt`) are
too large for git and are attached to the **GitHub Release** of this repository
(`GT-Former_release_scores.zip`, ~150 MB). Every table of the paper can be regenerated from the score files
alone without retraining.

## Installation

Python 3.11 or newer, one CUDA GPU (all deep models and the BODMAS LightGBM run were trained on a single
RTX A4000 Laptop GPU, 8 GB).

```bash
pip install -r requirements.txt
```

`thrember` (the EMBER2024 feature extractor) requires `signify==0.7.1`; both are pinned in `requirements.txt`.

## Data layout

All paths are read from environment variables with relative defaults, so the simplest layout is:

```
data/                 EMBER2024 zip archives downloaded from Hugging Face   (EMBER_DATA)
data_lamda/Baseline/  LAMDA "Baseline" parquet / npz files                  (LAMDA_DATA)
data_bodmas/          bodmas.npz and bodmas_metadata.csv                    (BODMAS_DATA)
features/             EMBER v3 feature matrices written by vectorize.py     (EMBER_FEAT, EMBER_FEAT_DIRS)
preprocessed/         fp16 preprocessed EMBER matrices, LAMDA int8 matrix   (EMBER_PP, LAMDA_PP)
results/, results_lamda/, results_bodmas/                                    (RESULTS, RESULTS_LAMDA, RESULTS_BODMAS)
```

Set the variables if you keep the data elsewhere (the EMBER feature matrices need about 30 GB).

## Reproducing the paper

Run every script from inside its own directory (the scripts import `common.py` from the same directory).

### EMBER2024

```bash
cd src/ember
python vectorize.py            # JSONL -> EMBER v3 features; de-duplicates by SHA-256 (Section 3.1.1)
python dataset_stats.py        # Table 2
python prep.py                 # sign-log / standardise / fp16 for the deep models
python exp_main.py             # LightGBM benchmark and its consensus / family weighted variants
python gtformer.py --arch gtformer --obj plain                     # GT-Former, default configuration
python gtformer.py --arch gtformer --obj plain --d 384 --layers 3 --heads 8 --ffn_mult 4 \
                   --imports_split 2 --drop 0.12 --lr 6e-4 --wd 3.9e-6 --tag tuned   # Optuna configuration
python gtformer.py --arch gtformer --obj full                      # CAM + consensus head + family GRL
python gtformer.py --arch moe   --obj plain                        # FR-MoE
python gtformer.py --arch proto --obj plain                        # ProtoCon-Net
python gtformer.py --arch mlp   --obj plain                        # MLP reference
python ensemble.py             # rank fusion (Eq. 8)
python analyze.py              # ROC/PR, TPR at 1% / 0.1% FPR, challenge, novel-family, low-consensus strata
python stats_tests.py          # paired bootstrap (1,000 resamples) against LightGBM
python extra_metrics.py        # classical metrics, confusion matrices, ROC / PR curves (Table 5, Figures 5-10)
python tables.py               # results/tables.md
python figures.py              # Figures 2-4
python shap_ember.py; python interpret.py     # Figures 18-20 (SHAP and CLS attention)
```

Five seeds: `python ../tuning_and_drivers/seeds_driver.py`. Optuna study of GT-Former:
`python ../tuning_and_drivers/optuna_deep.py --dataset ember` (20 trials on a 300 K-file, 3-epoch proxy).

### LAMDA

```bash
cd src/lamda
python lamda_prep.py           # Baseline variant -> int8 matrix, splits, groups
python lamda_exp.py --model lgbm
python lamda_exp.py --model mlp --obj plain
python lamda_exp.py --model gtformer --obj plain
python lamda_exp.py --model gtformer --obj plain --hp tuned --tag tuned
python lamda_analyze.py        # Tables 6-7, bootstrap
python lamda_figures.py        # Figures 14-15
```

### BODMAS

```bash
cd src/bodmas
python bodmas_exp.py           # LightGBM (GPU), MLP, ProtoCon-Net, GT-Former default and tuned, five seeds
python bodmas_analyze.py       # Tables 8-9, Figure 17
```

### Regenerating tables from the released score files only

Unzip `GT-Former_release_scores.zip` from the Release into `results/`, `results_lamda/` and `results_bodmas/`
and run `analyze.py`, `stats_tests.py`, `extra_metrics.py`, `tables.py`, `lamda_analyze.py` and
`bodmas_analyze.py`; no training or feature extraction is needed.

## Key results (from the paper)

* EMBER2024 Win32, 1% FPR: rank fusion of the tuned GT-Former and LightGBM raises challenge-set detection
  from 63.2% to 68.8% (95% CI +4.7 to +6.6) and novel-family detection from 89.1% to 91.8%.
* LAMDA FAR (2018–2025), 1% FPR: MLP + LightGBM fusion raises detection from 40.5% to 54.4% and singleton
  detection from 34.6% to 53.3%.
* BODMAS, 0.1% FPR: tuned GT-Former + LightGBM fusion 99.45% vs 98.92%; novel families 97.1% vs 94.7%.
* Every consensus- and family-aware objective lowers strict-FPR performance on EMBER2024 and LAMDA.
* The public EMBER2024 release lists every record twice; the SHA-256 keep-masks are produced by `vectorize.py`.

## Citation

```bibtex
@article{tawfik2026gtformer,
  title   = {GT-Former: Group-Token Transformers and Hybrid Rank Fusion for Evasive and Novel-Family Malware Detection},
  author  = {Tawfik, Mohammed and Al-sellami, Belal and Heshmat, Mohamed and Fathi, Islam S. and Shaban, Warda M.},
  year    = {2026},
  note    = {Under review}
}
```

## Licence

MIT (see `LICENSE`). EMBER2024 is released under Apache-2.0, LAMDA under MIT; BODMAS under its authors' terms.
