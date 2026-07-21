"""Weak-label generation for the pathology-conclusion 3-class classifier (recur / metas).

Two complementary weak-labeling steps are combined, per target, into a `labeled_`
(train) and a `non_labeled_` (to-be-predicted) CSV:
  1. Rule-based 3-class labeling of the raw conclusion text via `Preprocessor`
     (positive / negative / uncertain by the NEGATIVE/UNCERTAIN regex patterns).
  2. Weak-NEGATIVE mining in embedding space: encode the rule-labeled pool with the
     per-target encoder (CLS embeddings), cluster with DBSCAN, and keep the tail of
     homogeneous, non-reviewed negative clusters (those beyond the 50% cumulative-unique
     threshold) as extra weak negatives. The already-consolidated reviewed (gold) labels
     are folded in via `revision_checker` (unique / single-label dedup).

Inputs:
  - config.RAW_REPORTS_DIR/*  (first corpus CSV, sorted; must contain config.TEXT_COL)
  - config.REVIEWED_LABELS_DIR/*  (already-consolidated per-target reviewed labels;
    columns index, config.TEXT_COL, label in {negative, uncertain, positive})
  - per-target encoder under config.MODEL_ROOT (config.TARGETS[target]["encoder"])
Outputs (per target, into --output-dir; default config.OUTPUT_ROOT / "weak_labels"):
  - labeled_{target}.csv       (reviewed gold labels + mined weak negatives)
  - non_labeled_{target}.csv   (clusters not selected as weak negatives; predicted later)
  - optional --save-figures:   umap_clusters_{target}.png, weak_negative_threshold_{target}.png

How the raw reviewer workbooks are consolidated into the per-target reviewed labels is a
study-design step described in the paper; this script consumes those consolidated labels as
a given input (see load_reviewed_labels), and reproduces only the preprocessing -> embedding
-> clustering weak-negative mining once the labels are provided.

Dropped from the source notebook (exploratory / dead / diagnostic cells):
  - value_counts()/shape/output_dir display cells (3, 4, 17, 22, 23, 27, 30, 32, 33, 35, 36, 37, 38)
  - stale `c == 925` diagnostic (cell 24) and `most_c` (cell 26, only used in commented-out lines)
  - unused `other_cs` accumulator (cell 19) and unused `idx` variable (cell 12)
  - dead df.loc[..] recur_text/metas_text assignments never read downstream (cell 7)
  - unused helpers negative_words()/uncertain_words() and txt2label/textmapping_fn (cells 0, 10)
  - `ldf` recur load used only for a value_counts (cell 2) and duplicate reviewed loads (cell 11)
  - unused imports RandomForestClassifier / Isomap / TSNE / confusion_matrix / classification_report (cell 8)
"""
import os
import sys
import random
import argparse
from glob import glob
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import numpy as np
import pandas as pd
import config
from src.utils.preprocessing import Preprocessor, revision_checker  # NOT re-exported by src.utils

# Default weak-label output location (neutral; no run-date subfolder).
DEFAULT_OUTPUT_DIR = config.OUTPUT_ROOT / "weak_labels"

# Cumulative-unique-% threshold that separates the retained weak-negative clusters
# (the tail) from the large clusters kept as the to-be-predicted set (notebook `thr = 50`).
WEAK_NEG_THRESHOLD = 50


from src.pipeline import set_seed  # hoisted shared helper


# --- Reviewed (gold) label loading --------------------------------------------

def load_reviewed_labels(reviewed_dir, target: str) -> pd.DataFrame:
    """Load the already-consolidated per-target reviewed labels.

    Picks the file under ``reviewed_dir`` whose name contains ``target`` and returns a frame
    with columns index, config.TEXT_COL, label (a 3-class value in {negative, uncertain,
    positive}). Accepts either an Excel workbook or a CSV.

    Consolidation of the human-review rounds into these labels is described in the paper;
    treated here as a given input.
    """
    paths = sorted(glob(os.path.join(str(reviewed_dir), "*")))
    matches = [p for p in paths if target in os.path.basename(p)]
    if not matches:
        raise FileNotFoundError(
            f"No consolidated reviewed-label file for target '{target}' under {reviewed_dir}"
        )
    path = matches[0]
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, engine="openpyxl")
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")
    return df.loc[:, ["index", config.TEXT_COL, "label"]].dropna()


def check_label(series: pd.Series, label: str) -> pd.DataFrame:
    """Turn a text Series (index preserved) into a [TEXT_COL, label] frame with a constant label."""
    frame = series.to_frame()
    frame["label"] = label
    return frame


# --- Encoder embeddings (CLS token) -------------------------------------------

def embed_texts(texts: pd.Series, encoder_path, device: str, batch_size: int = 128) -> np.ndarray:
    """Return the CLS (`last_hidden_state[:, 0]`) embedding of every text, in input order."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer
    from tqdm import tqdm

    class _TextDataset(Dataset):
        def __init__(self, series):
            self.items = series.tolist()

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    tokenizer = AutoTokenizer.from_pretrained(str(encoder_path))
    model = AutoModel.from_pretrained(str(encoder_path), dtype="bfloat16")
    model.eval()
    model.to(device)  # FIX: device-aware; notebook called unconditional .cuda()

    loader = DataLoader(_TextDataset(texts), batch_size=batch_size, shuffle=False)
    outputs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="embedding"):
            token = tokenizer(
                list(batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
                padding_side="right",
            )
            token = {k: v.to(device) for k, v in token.items()}  # FIX: device instead of .cuda()
            vec = model(**token).last_hidden_state[:, 0].cpu().float().numpy()
            outputs.append(vec)
    return np.concatenate(outputs, axis=0)


# --- DBSCAN weak-negative mining ----------------------------------------------

def mine_weak_negatives(merged_df: pd.DataFrame, reviewed_df: pd.DataFrame,
                        embeddings: np.ndarray, eps: float, threshold: int = WEAK_NEG_THRESHOLD):
    """Cluster the rule-labeled pool and pick homogeneous, non-reviewed negative clusters.

    Returns (groups, sorted_c, thr_cs, temp_df, numb_cluster):
      groups      = cluster ids retained as weak negatives (tail beyond the threshold)
      sorted_c    = [c, n_all, n_unique, cum%_unique, cum%_all] per candidate cluster (or None)
      thr_cs      = index of the first cluster reaching `threshold` cumulative-unique % (or None)
      temp_df     = merged_df with a 'c' cluster column
      numb_cluster= raw DBSCAN labels aligned to merged_df row order
    """
    from sklearn.cluster import DBSCAN

    # NOTE: DBSCAN runs on the RAW encoder embeddings (not the UMAP projection), faithful to
    # the notebook; UMAP is used for visualization only (see save_figures).
    # FIX: eps comes from config.TARGETS[target] (recur=1.0, metas=0.1); the notebook hardcoded 1.
    db = DBSCAN(eps=eps)
    numb_cluster = db.fit_predict(embeddings)

    temp_df = merged_df.copy()
    temp_df["c"] = numb_cluster  # positional assignment; embeddings share merged_df's order

    reviewed_negs = (
        reviewed_df.query('label == "negative"')
        .drop_duplicates(subset=[config.TEXT_COL])[config.TEXT_COL]
        .to_list()
    )

    neg_cs = []
    for c, sdf in temp_df.groupby("c"):
        labels = sdf.label.unique()
        # Clusters touched by a positive or uncertain rule-label are not weak negatives.
        if "positive" in labels or "uncertain" in labels:
            continue
        # Homogeneous single-label cluster with no overlap into reviewed negatives -> candidate.
        if len(labels) == 1:
            unique_texts = sdf[config.TEXT_COL].drop_duplicates().unique()
            if sum(t in reviewed_negs for t in unique_texts) == 0:
                neg_cs.append([c, sdf.shape[0], sdf.drop_duplicates(subset=[config.TEXT_COL]).shape[0]])

    if len(neg_cs) == 0:
        return np.array([], dtype=int), None, None, temp_df, numb_cluster

    sorted_c = sorted(neg_cs, key=lambda x: x[2], reverse=True)  # by unique-text count, desc
    arr = np.array(sorted_c, dtype=float)
    sorted_c = np.c_[
        arr,
        arr[:, -1].cumsum() / arr[:, -1].sum() * 100,  # cumulative % of unique texts
        arr[:, 1].cumsum() / arr[:, 1].sum() * 100,     # cumulative % of all texts
    ]
    thr_cs = int(np.where(sorted_c[:, -2] >= threshold)[0][0])  # first cluster crossing the threshold
    groups = sorted_c[thr_cs:, 0].astype(int)                   # tail clusters -> mined weak negatives
    return groups, sorted_c, thr_cs, temp_df, numb_cluster


# --- Optional diagnostic figures ----------------------------------------------

def save_figures(embeddings: np.ndarray, numb_cluster: np.ndarray, sorted_c, thr_cs,
                 target: str, output_dir: str):
    """UMAP cluster jointplot + cumulative weak-negative threshold curve (visualization only)."""
    import matplotlib
    matplotlib.use("Agg")  # FIX: headless backend for non-interactive script use
    from matplotlib import pyplot as plt
    import seaborn as sbn
    from umap import UMAP

    config.apply_font()  # FIX: replaces the hardcoded system-font path

    # UMAP is used ONLY for visualization; the clustering was done on raw embeddings.
    manifold = UMAP(random_state=config.SEED)  # FIX: deterministic
    manifold_vectors = manifold.fit_transform(embeddings)
    plot_df = pd.DataFrame(
        np.concatenate([manifold_vectors, numb_cluster[:, None]], axis=1),
        columns=["x1", "x2", "c"],
    )
    g = sbn.jointplot(data=plot_df, x="x1", y="x2", hue="c", joint_kws={"s": 2, "alpha": 0.5})
    g.figure.savefig(os.path.join(output_dir, f"umap_clusters_{target}.png"), dpi=150, bbox_inches="tight")
    plt.close(g.figure)

    if sorted_c is None or thr_cs is None:
        return

    fig, ax = plt.subplots()
    tax = ax.twinx()
    tax.plot(sorted_c[:, -2])                          # cumulative % unique
    ax.plot(sorted_c[:, -1], color="tab:orange")       # cumulative % all
    ax.set_ylabel(f"# of whole texts\n{int(sorted_c[:, 1].sum())} samples")
    tax.set_ylabel(f"Cumulative % of unique texts\n{int(sorted_c[:, 2].sum())} samples")
    ax.set_xlabel("Cluster")
    tax.axvline(thr_cs, color="tab:red", linestyle="--")

    y_left = sorted_c[thr_cs, -1]   # cumulative % all (orange)
    y_right = sorted_c[thr_cs, -2]  # cumulative % unique (blue)
    ax.annotate(f"{y_left:.1f}%", xy=(thr_cs, y_left), xytext=(thr_cs + 10, y_left - 5),
                textcoords="data", arrowprops=dict(arrowstyle="->", color="tab:orange"),
                fontsize=10, color="tab:orange", va="bottom")
    tax.annotate(f"{y_right:.1f}%", xy=(thr_cs, y_right), xytext=(thr_cs + 10, y_right - 5),
                 textcoords="data", arrowprops=dict(arrowstyle="->", color="tab:blue"),
                 fontsize=10, color="tab:blue", va="bottom")
    ax.legend(handles=[
        plt.Line2D([0], [0], color="tab:blue", lw=2, label="Unique"),
        plt.Line2D([0], [0], color="tab:orange", lw=2, label="All"),
    ], loc="lower right")
    fig.savefig(os.path.join(output_dir, f"weak_negative_threshold_{target}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- Per-target pipeline ------------------------------------------------------

def process_target(target: str, raw_csv_path: str, reviewed_dir, output_dir: str,
                   device: str, save_figs: bool = False):
    cfg = config.TARGETS[target]
    encoder_path = config.model_path(cfg["encoder"])
    eps = cfg["dbscan_eps"]

    # 1. Rule-based weak labels via Preprocessor.
    raw_df = pd.read_csv(raw_csv_path, encoding="utf-8-sig", engine="c")
    text_series = raw_df[config.TEXT_COL].dropna()
    preprocessor = Preprocessor(
        df=text_series,
        negative_patterns=config.NEGATIVE_PATTERNS,
        uncertain_patterns=config.UNCERTAIN_PATTERNS,
        abbrev_path=os.path.join(config.REPO_ROOT, "src", "utils", "abbrevations.json"),
    )
    weak_dict = preprocessor.target_filtering(target)  # {positive, negative, uncertain} text Series
    merged_df = pd.concat([check_label(s, lab) for lab, s in weak_dict.items()]).sort_index()

    # 2. Already-consolidated reviewed (gold) labels for this target.
    reviewed_df = load_reviewed_labels(reviewed_dir, target)

    # 3. Drop any rule-labeled text that was already adjudicated by reviewers.
    in_reviewed = merged_df[config.TEXT_COL].isin(reviewed_df[config.TEXT_COL].drop_duplicates().unique())
    merged_df = merged_df.loc[~in_reviewed]

    # 4. Encoder embeddings (CLS) of the remaining rule-labeled pool.
    embeddings = embed_texts(merged_df[config.TEXT_COL], encoder_path, device)

    # 5. Weak-negative mining via DBSCAN.
    groups, sorted_c, thr_cs, temp_df, numb_cluster = mine_weak_negatives(
        merged_df, reviewed_df, embeddings, eps
    )
    neg_temp_df = temp_df.loc[temp_df.c.isin(groups)].reset_index().drop(columns=["c"])
    test_df = temp_df.loc[~temp_df.c.isin(groups)].reset_index().drop(columns=["c"])

    # 6. Reviewed training labels: unique + single-label multiples (drops conflicting/novel cases).
    normal_df = revision_checker(reviewed_df, drop_duplicates=True)["label_df"]

    # 7. Assemble train (labeled) and write both CSVs.
    train_df = (
        pd.concat([normal_df, neg_temp_df.drop_duplicates(config.TEXT_COL)])
        .sort_values("index")
        .reset_index(drop=True)
    )

    os.makedirs(output_dir, exist_ok=True)
    labeled_path = os.path.join(output_dir, f"labeled_{target}.csv")
    non_labeled_path = os.path.join(output_dir, f"non_labeled_{target}.csv")
    train_df.reset_index(drop=True).to_csv(labeled_path, index=False, encoding="utf-8-sig")
    test_df.reset_index(drop=True).to_csv(non_labeled_path, index=False, encoding="utf-8-sig")
    print(f"[{target}] wrote {labeled_path} ({len(train_df)} rows), "
          f"{non_labeled_path} ({len(test_df)} rows)")

    if save_figs:
        save_figures(embeddings, numb_cluster, sorted_c, thr_cs, target, output_dir)

    return labeled_path, non_labeled_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", choices=sorted(config.TARGETS), default=None,
                        help="Target to process; default = both recur and metas.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory for the labeled_/non_labeled_ CSVs.")
    parser.add_argument("--reports-dir", default=str(config.RAW_REPORTS_DIR),
                        help="Directory of corpus CSV(s); the first (sorted) is used.")
    parser.add_argument("--reviewed-dir", default=str(config.REVIEWED_LABELS_DIR),
                        help="Directory of the already-consolidated per-target reviewed labels.")
    parser.add_argument("--save-figures", action="store_true",
                        help="Also write the UMAP cluster and weak-negative threshold figures.")
    args = parser.parse_args()

    set_seed()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"  # FIX: device-aware

    # First corpus CSV (sorted for determinism); it must contain config.TEXT_COL.
    raw_files = sorted(glob(os.path.join(str(args.reports_dir), "*")))
    if not raw_files:
        raise FileNotFoundError(f"No corpus CSVs found under {args.reports_dir}")
    raw_csv_path = raw_files[0]

    targets = [args.target] if args.target else list(config.TARGETS)
    for target in targets:
        process_target(target, raw_csv_path, args.reviewed_dir, args.output_dir, device, args.save_figures)


if __name__ == "__main__":
    main()
