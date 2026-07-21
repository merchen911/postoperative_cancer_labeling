"""Central configuration for the cancer-conclusion classification verification repo.

Values that were hard-coded across the original notebooks are collected here, so switching
target (recur/metas), encoders, and data locations is a one-line change.

This repo does NOT prescribe a researcher-specific folder tree or run-dates. It defines an
INPUT DATA CONTRACT: each role below is a directory/file you point at (via its default
location, the PROJECT_DATA_ROOT / PROJECT_OUTPUT_ROOT env vars, or a per-script ``--input``
flag), and README "Input data contract" documents the required format/columns for each role.
Data, models and checkpoints are supplied out of band (PHI or large binaries; not committed).

NOTE: How the human-reviewed label set is consolidated from multiple review rounds is a
study-design step described in the paper, NOT reproduced here. This code consumes the
already-consolidated reviewed labels (REVIEWED_LABELS_DIR / REVIEWER_GOLD_DIR) as given
inputs, and implements the reproducible preprocessing -> embedding -> clustering that mines
weak negatives once those labels are provided.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Repo & configurable roots ----------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parent
DATA_ROOT   = Path(os.environ.get("PROJECT_DATA_ROOT",   REPO_ROOT / "data")).resolve()
MODEL_ROOT  = Path(os.environ.get("PROJECT_MODEL_ROOT",  REPO_ROOT / "model")).resolve()
LOGS_ROOT   = Path(os.environ.get("PROJECT_LOGS_ROOT",   REPO_ROOT / "logs")).resolve()
OUTPUT_ROOT = Path(os.environ.get("PROJECT_OUTPUT_ROOT", DATA_ROOT / "outputs")).resolve()

# --- Input roles (WHAT to provide, not a fixed folder tree) ------------------
# Defaults live under DATA_ROOT; override via PROJECT_DATA_ROOT or a per-script --input flag.
# See README "Input data contract" for the required format/columns of each role. The
# provenance of the reviewed/gold labels (how review rounds were merged) is described in
# the paper; this code treats those consolidated labels as given inputs.
RAW_REPORTS_DIR     = DATA_ROOT / "reports"               # corpus CSV(s); text col = TEXT_COL (+ metadata cols)
CANCER_TYPE_MAP     = DATA_ROOT / "cancer_type_map.xlsx"  # cols: 암번호, 병원등록번호, 수술일자
RULE_LABELS_DIR     = DATA_ROOT / "rule_labels"           # per-target *labeled.csv: index, TEXT_COL, label
REVIEWED_LABELS_DIR = DATA_ROOT / "reviewed_labels"       # per-target consolidated human labels (data-prep input)
REVIEWER_GOLD_DIR   = DATA_ROOT / "reviewer"              # reviewer workbooks: sheets negative/uncertain/positive, gold col = GOLD_COL

# --- Output locations (derived artifacts; no run-date subfolders) ------------
SPLITS_DIR      = OUTPUT_ROOT / "splits"        # {target}_train_df.xlsx / {target}_test_df.xlsx
PREDICTIONS_DIR = OUTPUT_ROOT / "predictions"   # full-corpus prediction workbooks
FIGURES_DIR     = LOGS_ROOT / "figures"         # UMAP / diagnostic figures

# --- Column names & label scheme --------------------------------------------
TEXT_COL = "검사결과결론내용"        # raw conclusion text
PREP_COL = "prep_text"               # preprocessed text used for modeling
GOLD_COL = "실제"                    # reviewer gold-label column (downstream evaluation)
LABEL_DICT = {"negative": 0, "uncertain": 1, "positive": 2}
INV_LABEL_DICT = {v: k for k, v in LABEL_DICT.items()}

# --- Weak-labeling rule patterns (src.utils.preprocessing.Preprocessor) ------
NEGATIVE_PATTERNS = ["without", "not? (sugges|compat)", "unlike", "neither", "no", "negative"]
UNCERTAIN_PATTERNS = ["rule out", "Rule Out", "Rule out"]

# --- Encoders (local HuggingFace dirs under MODEL_ROOT, NOT hub ids) ---------
CANDIDATE_MODELS = {
    "Bluebert": "bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
    "Bio_Clinical_Bert": "Bio_ClinicalBERT",
    "BioSimCSE_BioLinkBERT_base": "BioSimCSE-BioLinkBERT-BASE",
    "MS_BiomedNLP_BiomedBERT_base": "BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "ClinicalBERT": "ClinicalBERT",
    "MedEmbed_base": "MedEmbed-base-v0.1",
    "pubmedbert": "PubMedBERT-base-uncased-sts-combined",
    "Multilingual_E5_base": "multilingual-e5-base",
}
EMBEDDING_MODEL = "multilingual-e5-base"   # encoder used for embedding-space analysis (stage 07)

# --- Per-target settings -----------------------------------------------------
# `encoder` = final selected encoder per target; `dbscan_eps` = weak-negative mining eps.
TARGETS = {
    "recur": {"encoder": "PubMedBERT-base-uncased-sts-combined", "dbscan_eps": 1.0},
    "metas": {"encoder": "MedEmbed-base-v0.1", "dbscan_eps": 0.1},
}

# --- Training hyperparameters (all-data / canonical track) -------------------
SEED = 42
BATCH_SIZE = 8          # all-data track (the partial-track scripts used 32)
PATIENCE = 5
MAX_EPOCHS = 30
LR = 1e-5
FOCAL_GAMMA = 2.0
N_SPLITS = 10           # StratifiedKFold
MAX_LENGTH = 256        # tokenizer max_length

# --- Plotting ----------------------------------------------------------------
# Optional CJK/serif font for figure labels — set PROJECT_FONT_PATH to a local .ttf.
# Empty default: apply_font() then no-ops (matplotlib default), so nothing is bundled/required.
FONT_PATH = os.environ.get("PROJECT_FONT_PATH", "")


# --- Helpers -----------------------------------------------------------------
def model_path(name: str) -> Path:
    """Absolute path to a local HuggingFace encoder directory."""
    return MODEL_ROOT / name


def apply_font():
    """Register the configured font with matplotlib (call once per plotting notebook)."""
    from matplotlib import font_manager as fm, pyplot as plt
    if os.path.exists(FONT_PATH):
        fm.fontManager.addfont(FONT_PATH)
        plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams["axes.unicode_minus"] = False
