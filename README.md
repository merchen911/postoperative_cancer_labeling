# Deep learning for unsupervised recurrence & metastasis labeling in Korean postoverpative pathology reports

Code accompanying the manuscript **"Automated Extraction of Postoperative Cancer Recurrence and Metastasis from CT Reports: Semi-Supervised Deep Learning Study"** (submitted). This repository lets reviewers and readers reproduce the modeling pipeline and the reported experiments.

> **Data availability.** The clinical corpus contains protected health information (PHI)
> and **is not distributed with this code.** De-identified data may be available from the
> corresponding author on reasonable request, subject to IRB approval and a data-use
> agreement. See [Data & model placement](#5-data--model-placement). *(TODO: corresponding-author contact + IRB/DUA statement.)*

---

## 1. Overview

We classify the free-text **conclusion** of Korean oncology pathology/imaging reports
(column `검사결과결론내용`) into three classes — **negative (0) / uncertain (1) / positive (2)** —
for two clinical targets:

- **Recurrence** (`recur`) — final encoder: `PubMedBERT-base-uncased-sts-combined`
- **Metastasis** (`metas`) — final encoder: `MedEmbed-base-v0.1`

selected from an **8-encoder biomedical-BERT benchmark** (a linear head on the pooled
`[CLS]` embedding, **Focal loss** with class-balanced α, PyTorch Lightning). The pipeline
also includes a **weak-negative mining** method (rule labels refined with an encoder +
UMAP/DBSCAN in embedding space) and **Captum Integrated-Gradients** interpretability.

**The runnable `scripts/` are the primary interface** — each notebook was ported to a
config-driven, headless, target-parameterized script. The `notebooks/` tree is retained as
an optional bonus (exploratory originals; see §8).

## 2. Quick start (scripts)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Restore data/, model/, logs/ first (see §5), then run the whole pipeline for one target:
python run_pipeline.py --stage all --target metas
python run_pipeline.py --stage all --target recur

# Or the default (both targets, every stage, in dependency order):
python run_pipeline.py

# A single stage, and a no-op preview of the exact commands:
python run_pipeline.py --stage train --target recur
python run_pipeline.py --stage all --dry-run
```

`run_pipeline.py` (repo root) maps stage names to the numbered scripts and runs them in
dependency order, forwarding `--target` to every stage that supports it. `--stage` choices:
`preprocess split train apply aggregate performance embedding captum term_stats all`
(default `all`); `--target` is `recur | metas | both` (default `both`, which omits the flag
so each script loops over both targets). The `aggregate` stage fuses both targets into one
master workbook and is intentionally **not** per-target, so `--target` is never forwarded to
it. You can also call any stage script directly (they share the same `--target` interface).

### Per-script reference


| Stage / script                                   | What it does                                                                                                                                                                                                                                              | Key inputs                                                                        | Key outputs                                                                                                                                             |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **preprocess** `scripts/01_preprocess.py`        | Per-target weak labels: rule labeling (`Preprocessor`) + embedding-space weak-**negative** mining (encoder `[CLS]` → DBSCAN → keep homogeneous non-reviewed clusters past a 50% threshold), then attach the consolidated reviewed labels (a given input; the review-round consolidation is described in the paper).                                     | `config.RAW_REPORTS_DIR` corpus CSV; `config.REVIEWED_LABELS_DIR` reviewed labels; per-target encoder | `labeled_{target}.csv`, `non_labeled_{target}.csv` (default under `config.OUTPUT_ROOT` / `weak_labels`)                                                    |
| **split** `scripts/02_split.py`                  | Canonical train/test split: combine weak rule labels + consolidated reviewed labels into a human-covered set, embed the uncovered pool, UMAP+DBSCAN-cluster it, peel pure rule-negative clusters into training to balance positives.                      | `config.RAW_REPORTS_DIR` corpus CSV; `config.RULE_LABELS_DIR` `*labeled.csv`; `config.REVIEWED_LABELS_DIR` reviewed labels; encoder | `{target}_train_df.xlsx`, `{target}_test_df.xlsx` (default `config.SPLITS_DIR`)                                                               |
| **train** `scripts/03_train.py`                  | Fine-tune encoder + linear head (FocalLoss, class-balanced α), single stratified 90/10 val split, select best checkpoint by val macro-F1, emit val confusion matrix + test predictions. `--models all` benchmarks all 8 encoders.                         | `config.SPLITS_DIR` `{target}_{train,test}_df.xlsx`; local HF encoder                                 | checkpoints `clf-epoch=NN-val_f1_macro=X.XXXX.ckpt`; `test-with-pred.xlsx`; `val_confusion_matrix.png` (under `config.LOGS_ROOT`)                    |
| **apply** `scripts/04_apply.py`                  | Batch inference over the full corpus: best checkpoint, rule-segment text, tokenize dedup text, softmax → per-class probs, spread back to every row, join cancer-type metadata.                                                                            | `config.RAW_REPORTS_DIR` corpus CSV; `config.CANCER_TYPE_MAP`; `cancer_dict.json`; `config.LOGS_ROOT` checkpoints; encoder | `raw_format.xlsx`, `prediction_format.xlsx` (under `config.PREDICTIONS_DIR`)                                                                         |
| **aggregate** `scripts/05_aggregate.py`          | Aggregate per-target predictions + training gold onto the corpus → master `all.xlsx`; build cohort characteristics table (by sex / age band / cancer type). *(joint over both targets)*                                                                   | `config.RAW_REPORTS_DIR` corpus CSV; `config.PREDICTIONS_DIR` prediction workbooks; `config.SPLITS_DIR` train dfs | `all.xlsx`; `Table description.xlsx`; optional `postprocessed-for-review.xlsx` (`--no-review` to skip)                                              |
| **performance** `scripts/06_performance.py`      | Final-performance artifacts: category figures, per-cancer/per-age confusion matrices, Human-vs-LM agreement tables (gold `실제` vs prediction).                                                                                                            | `config.REVIEWER_GOLD_DIR` workbooks; aggregated corpus; `config.CANCER_TYPE_MAP`; reviewed labels | category figures; confusion-matrix PNGs; `result_overall.csv`, `pivot_binary_by_cancer.csv`, `human_vs_lm_{long,wide}.csv` (default `config.LOGS_ROOT`) |
| **embedding** `scripts/07_embedding_analysis.py` | **Downstream** embedding-space analysis *validating* the weak-negative mining method (runs **after** `apply`): encode with `multilingual-e5-base`, UMAP to 2-D, per-class distribution tests (Welch T² + 3 MMD variants) comparing gold vs model-labeled. | `config.PREDICTIONS_DIR` prediction workbooks; `config.REVIEWER_GOLD_DIR` workbooks; `multilingual-e5-base` | `Embedding_{target}_umap.png` (default `config.FIGURES_DIR`) + test results to stdout                                                                        |
| **captum** `scripts/08_captum.py`                | Integrated-Gradients (`LayerIntegratedGradients`) token attributions for lowest/highest-confidence gold examples; styled HTML panels.                                                                                                                     | `config.REVIEWER_GOLD_DIR` workbooks; `config.LOGS_ROOT` checkpoints; encoders                        | `{target}_{class}_attrs.html`, `{target}_{class}_index_{n}_attrs.html` (default `config.LOGS_ROOT`)                                                         |
| **term_stats** `scripts/09_term_stats.py`        | Surgery-to-exam interval stats: collapse to one row per (patient, surgery-date, exam-date), conflict→positive, signed day delta (`검사접수일자 − 수술일자`), delta-day histograms.                                                                                  | `config.PREDICTIONS_DIR` `raw_format.xlsx`                                                             | `stats_terms.xlsx`, `stats_terms_merged.xlsx`; overlap/stack histograms (default `config.PREDICTIONS_DIR`)                                           |
| *(utility)* `scripts/util_param_count.py`        | Not a pipeline stage. Prints encoder parameter counts / hidden sizes for the 8-candidate roster and demos the 10-fold CV setup + focal-α.                                                                                                                 | `CANDIDATE_MODELS` encoders; `config.SPLITS_DIR` split                                  | stdout only                                                                                                                                             |


> **Note — two different "embedding" steps.** The embedding-space *weak-negative mining*
> used during data prep (encoder `[CLS]` → UMAP/DBSCAN) is **inlined** inside `01_preprocess`
> and `02_split`; there is intentionally **no standalone upstream "embedding" stage**. An
> exploratory *prediction-embedding* EDA (E5 + regex weak-labeling) was run during development
> *between* preprocessing and split, but it fed no downstream stage and is not ported here.
> Stage 07 (`07_embedding_analysis.py`) is a **different, downstream** step: it runs **after**
> `apply` and only compares gold-vs-model embedding distributions (UMAP figure + MMD/Welch
> tests) to *validate* the mining method — it does not feed `split`.

## 3. Repository layout

```
.
├── run_pipeline.py      # top-level runner: maps --stage → scripts, runs them in order
├── config.py            # single source of truth for the input data contract, targets, encoders, hyperparameters
├── requirements.txt
├── cancer_dict.json     # 11 cancer-type id → {name_en, name_ko}
├── scripts/             # ← PRIMARY INTERFACE: 01_… through 09_… + util_param_count.py
├── src/                 # shared package
│   ├── pipeline.py      # hoisted shared helpers: set_seed, age-binning, checkpoint selection
│   ├── models/          # model.py (SupervisedPLModel/SimpleClassifier), loss.py (FocalLoss), inference.py
│   └── utils/           # preprocessing.py (Preprocessor), paths.py, stats.py, abbrevations.json
├── notebooks/           # OPTIONAL bonus: exploratory originals (00_eda … 06_stats); off the critical path
├── data/                # (git-ignored) restore the corpus here — see §5
├── model/               # (git-ignored) local HuggingFace encoder directories
└── logs/                # (git-ignored) fine-tuned Lightning checkpoints land here
```

## 4. Prerequisites

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- **GPU (CUDA)** recommended, but not required — every script auto-selects
`cuda` if available else `cpu` (no unconditional `.cuda()`).
- To render Korean labels in figures, set `PROJECT_FONT_PATH` to a local Unicode font that
covers Korean (e.g. `NanumGothic.ttf`); `config.apply_font()` registers it. If unset, plots
fall back to the matplotlib default (Korean glyphs may not render).

## 5. Input data contract

This repo does **not** prescribe a researcher-specific folder tree or run-dates. Instead it
defines an **input data contract**: each *role* below is a directory/file you point the
pipeline at. Every role has a default location under `data/` / `model/` / `logs/`; override
the roots with the `PROJECT_DATA_ROOT` / `PROJECT_MODEL_ROOT` / `PROJECT_LOGS_ROOT` /
`PROJECT_OUTPUT_ROOT` env vars, or override an individual role with the per-script `--input`
flag. All are **git-ignored** (PHI or large binaries; supplied out of band).

### Inputs

| Role               | config name                | Required format / columns                                                                                  | How to provide                                                        |
| ------------------ | -------------------------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| raw reports        | `config.RAW_REPORTS_DIR`   | corpus CSV(s) (`utf-8-sig`): text col `검사결과결론내용` + metadata `성별`, `검사나이`, `암번호`, `수술일자`, `검사접수일자`, `병원등록번호` | drop the corpus CSV in this dir (default `data/reports/`)             |
| cancer type map    | `config.CANCER_TYPE_MAP`   | xlsx with cols `암번호`, `병원등록번호`, `수술일자`                                                                     | provide the mapping workbook at this path                             |
| rule labels        | `config.RULE_LABELS_DIR`   | per-target `*labeled.csv`: cols `index`, `검사결과결론내용`, `label`                                                | per-target files in this dir (produced by `preprocess`; can be given) |
| reviewed labels    | `config.REVIEWED_LABELS_DIR` | per-target consolidated human labels: cols `index`, `text`, `label` in `negative`/`uncertain`/`positive` | per-target files in this dir (consolidated input; see note below)     |
| reviewer gold      | `config.REVIEWER_GOLD_DIR` | workbooks with sheets `negative`/`uncertain`/`positive` and gold column `실제`                                | per-target reviewer workbooks in this dir                             |
| local encoders     | `config.MODEL_ROOT`        | local HuggingFace encoder directories                                                                      | 8-candidate roster in `config.CANDIDATE_MODELS`, one dir each         |
| checkpoints        | `config.LOGS_ROOT`         | fine-tuned Lightning checkpoints `clf-epoch=NN-val_f1_macro=X.XXXX.ckpt`                                    | produced by `train`; consumed by `apply` & `captum`                   |

### Outputs (produced by the pipeline)

| Role         | config name              | Contents                                                            |
| ------------ | ------------------------ | ------------------------------------------------------------------- |
| splits       | `config.SPLITS_DIR`      | `{target}_train_df.xlsx`, `{target}_test_df.xlsx`                   |
| predictions  | `config.PREDICTIONS_DIR` | full-corpus prediction workbooks (`raw_format.xlsx`, `prediction_format.xlsx`, `all.xlsx`, …) |
| figures      | `config.FIGURES_DIR`     | UMAP / diagnostic figures                                           |

**Conclusion CSV schema (key columns):** `검사결과결론내용` (text), `수술일자`, `검사접수일자`,
`병원등록번호` (PHI), `성별`, `검사나이`, `암번호`, `검사코드`.

> **Human-review consolidation is paper-deferred.** The *consolidation* of the multiple
> human-review rounds into the consolidated **reviewed labels** (`config.REVIEWED_LABELS_DIR`)
> is a study-design step **described in the paper** and is **not reproduced in this code**.
> The code treats those consolidated labels as a **given input** and implements the
> reproducible **preprocessing → embedding → clustering weak-negative mining** pipeline given
> those labels.

## 6. Configuration

All knobs live in `[config.py](config.py)`: the input-role locations and output roots
(`RAW_REPORTS_DIR`, `CANCER_TYPE_MAP`, `RULE_LABELS_DIR`, `REVIEWED_LABELS_DIR`,
`REVIEWER_GOLD_DIR`, `SPLITS_DIR`, `PREDICTIONS_DIR`, `FIGURES_DIR`), per-target `encoder` +
`dbscan_eps`, the candidate-encoder roster, `SEED`/`BATCH_SIZE`/`LR`/`FOCAL_GAMMA`/
`MAX_EPOCHS`/`N_SPLITS`/`MAX_LENGTH`, and `FONT_PATH`. Nothing is hard-coded in the scripts;
switch target with `--target` (or edit `config.TARGETS`).

## 7. What reviewers should confirm

The scripts are faithful ports; a handful of notebook-implicit assumptions could not be
verified here (the PHI corpus and the fine-tuned checkpoints are not in the tree). Please
confirm these against the restored data:

- **Canonical raw CSV.** Stages read `sorted(glob(...))[0]` of `config.RAW_REPORTS_DIR` (the
notebooks used an *unsorted* `glob[0]`). `04_apply`/`05_aggregate` prefer a `all-report.csv`
by name. If the intended corpus is not the alphabetically-first CSV, pin it via `--raw-file`
where offered. The chosen CSV's row order defines the `index` join key used downstream.
- **File disambiguation by keyword.** Per-target inputs (rule `*labeled.csv`, reviewed-label
files, and reviewer gold workbooks) are selected by a `meta`/`recur` filename substring,
replacing the notebooks' sorted-order unpacking (which assumed metas-before-recur). Verify
your filenames carry those tokens.
- `**05_aggregate` prediction glob.** Expects **exactly 3** apply workbooks under
`config.PREDICTIONS_DIR` sorting as `[metas_a, metas_b, recur]` (the 2nd metas + the recur
file are merged; the 1st metas file feeds only the review workbook). Downstream stages read
the train/test split from `config.SPLITS_DIR` and predictions from `config.PREDICTIONS_DIR`.
- **Table-1 category completeness.** The cohort table's fixed-length `MultiIndex` assumes
`성별 ∈ {F, M}`, all six age bands, and all 11 cancer types + `Missing` are present; a
missing category would break the `set_axis` (same fragility as the notebook).
- **Checkpoint naming.** Best-checkpoint selection parses `val_f1_macro=` from Lightning
filenames `clf-epoch=NN-val_f1_macro=X.XXXX.ckpt` (falls back to epoch, then mtime). This is
now shared in `src/pipeline.py` across `train`/`apply`/`captum`.
- `**09_term_stats` input.** Reads `raw_format.xlsx` from `config.PREDICTIONS_DIR` (the
notebook used an unsorted `glob[1]`); it is the only prediction workbook carrying `검사접수일자`
alongside the prediction columns.
- **Human-review consolidation is paper-deferred.** The code does **not** merge multiple
human-review rounds/revision workbooks; it loads the already-consolidated reviewed labels
from `config.REVIEWED_LABELS_DIR`. The consolidation procedure is described in the paper.
Verify that the consolidated labels you supply match that described procedure.
- **Modeling details preserved from the notebooks (verify they match the paper):** the
`recur` negative reviewer sheet has its first 100 rows' gold `실제` force-set to `negative`
(a preserved positional patch); embedding analysis uses tokenizer `max_length=128` while the
classifier uses `256`; encoders load with `dtype='bfloat16'`; `train` uses a single 90/10
stratified split (not K-fold) matching the all-data notebook.

## 8. Notebooks (optional bonus)

`notebooks/` holds the exploratory originals (`00_eda` … `06_stats`) the scripts were ported
from, with outputs cleared. They are **not** the supported entry point and may still contain
hard-coded paths / per-cell target edits; use `scripts/` + `run_pipeline.py` for reproduction.

## 9. Known limitations / reproducibility notes

- **Non-determinism:** UMAP / DBSCAN and bfloat16 GPU paths are not fully seeded even with
`set_seed()` + `random_state=SEED`; expect small run-to-run variation in embeddings/mining.
- The per-target rule `*labeled.csv` files (`config.RULE_LABELS_DIR`) are a hard dependency of
the `split` step and must be supplied (or produced by `preprocess`) separately.

## 10. Expected outputs

Fine-tuned checkpoints; `all.xlsx` master dataset; final-performance figures & confusion
matrices; term-statistics workbooks; UMAP embedding figures; Captum attribution HTML.

## 11. Citation & license

*(TODO: add citation/BibTeX once published, and choose a license — e.g. MIT for code;
data governed separately.)*
