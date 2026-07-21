"""Build the canonical all-data train/test split for the conclusion 3-class classifier.

Starts from the already-consolidated per-target human-reviewed ground truth (loaded as a
given input), then mines extra weak-negative rows from the uncovered pool via an encoder
embedding -> UMAP -> DBSCAN clustering pipeline (so the training positives are balanced by
rule-negative clusters). Consolidation of the human-review rounds into these labels is
described in the paper; treated here as a given input. Per target it writes:
    config.SPLITS_DIR/{target}_train_df.xlsx
    config.SPLITS_DIR/{target}_test_df.xlsx
The train frame carries the resolved ``label`` column (positive/uncertain/negative); the
held-out test frame keeps its raw_label/rule_label columns for downstream gold labelling
(the notebook never assigned a single ``label`` to test rows).

Ported from the all-data CLF-dataprepare notebook.

Inputs:
  config.RAW_REPORTS_DIR/*.csv               base corpus (column 검사결과결론내용)
  config.RULE_LABELS_DIR/*labeled.csv        weak rule labels (metas / recur)
  config.REVIEWED_LABELS_DIR/*               consolidated per-target reviewed ground truth
                                             (index, prep_text/text, label, rule_label)
  config.model_path(<per-target encoder>)    local HuggingFace encoder for embedding

Outputs:
  config.SPLITS_DIR/{target}_train_df.xlsx
  config.SPLITS_DIR/{target}_test_df.xlsx

Dropped from the notebook:
  - cell 2 redefinitions of repo helpers (save_dir_setup / compute_alpha /
    get_model_predictions) — these live in src and are not needed here anyway;
  - unused evaluation helpers vis_confusion / score2dict / sub_return (never called in
    the data-prep flow; score2dict even referenced an unimported accuracy_score);
  - the dead ground-truth embedding + ``manifold.transform(gt_outputs)`` — the result
    (gt_manifold_eos_vector) is never consumed downstream, so gt embedding is skipped;
  - the UMAP jointplot (cell 26), the Counter / np.where diagnostics (cell 27), and the
    set-intersection / isna / bare-display cells (cells 23, 30-35);
  - unused imports (RandomForestClassifier, Isomap, TSNE, confusion_matrix,
    classification_report, seaborn, pytorch_lightning, matplotlib, f1_score, ...);
  - scratch variables dup_flag / labels / most_common that fed nothing;
  - the in-code merge of the human-review rounds into a covered ground-truth set — the
    consolidated reviewed labels are consumed here as a given input (see the paper).
"""
import os, sys, random, argparse
from glob import glob
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import numpy as np
import pandas as pd
import config
# NB: torch / transformers / umap / sklearn are imported lazily inside functions so the
# module stays importable (and py-compilable) without the ML stack installed.

# Embedding mini-batch size used by the notebook's llm_embedding_fn.
EMBED_BATCH_SIZE = 128

# Substring used to disambiguate the per-target label files on disk.
_TARGET_FILE_KEYWORD = {"metas": "meta", "recur": "recur"}


from src.pipeline import set_seed  # hoisted shared helper


def check_label(series: pd.Series, label: str) -> pd.DataFrame:
    """Turn a text Series into a frame with a constant ``label`` column (notebook check_label)."""
    frame = series.to_frame()
    frame["label"] = label
    return frame


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _pick_by_target(paths, target):
    """Select the single path whose basename contains the target keyword (meta/recur).

    FIX: the notebook relied on sorted-glob unpack order (meta before recur); selecting by
    keyword is equally faithful but robust to extra files landing in the directory.
    """
    key = _TARGET_FILE_KEYWORD[target]
    matches = [p for p in paths if key in os.path.basename(p).lower()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one file containing '{key}', found {matches} among {paths}"
        )
    return matches[0]


def load_base_frame(raw_file: str | None = None):
    """Load the base corpus (with an ``index`` column) and build the shared Preprocessor."""
    from src.utils.preprocessing import Preprocessor

    if raw_file:
        raw_path = raw_file if os.path.isabs(raw_file) else str(config.RAW_REPORTS_DIR / raw_file)
    else:
        # FIX: notebook used an unsorted glob(...)[0]; sort for determinism. Pass --raw-file
        # to pin the exact corpus when the directory holds more than one CSV.
        candidates = sorted(glob(str(config.RAW_REPORTS_DIR / "*.csv")))
        if not candidates:
            raise FileNotFoundError(f"no *.csv under {config.RAW_REPORTS_DIR}")
        raw_path = candidates[0]

    print(f"loading base corpus: {raw_path}")
    df = pd.read_csv(raw_path, encoding="utf-8-sig", engine="c").reset_index()
    text_df = df[config.TEXT_COL].dropna()

    preprocessor = Preprocessor(
        df=text_df,
        negative_patterns=config.NEGATIVE_PATTERNS,
        uncertain_patterns=config.UNCERTAIN_PATTERNS,
        # FIX: absolute abbrev path (Preprocessor default is cwd-relative './src/...').
        abbrev_path=os.path.join(config.REPO_ROOT, "src", "utils", "abbrevations.json"),
    )
    return df, preprocessor


def load_rule_labels():
    """Weak rule-based labels per target from RULE_LABELS_DIR/*labeled.csv."""
    paths = sorted(glob(str(config.RULE_LABELS_DIR / "*labeled.csv")))
    if not paths:
        raise FileNotFoundError(f"no *labeled.csv under {config.RULE_LABELS_DIR}")
    return {
        t: pd.read_csv(_pick_by_target(paths, t), encoding="utf-8-sig", engine="c")
        for t in config.TARGETS
    }


def _read_any(path):
    """Read a reviewed-labels file by extension (.csv or .xlsx)."""
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8-sig", engine="c")
    return pd.read_excel(path, engine="openpyxl")


def _normalize_reviewed(frame):
    """Ensure the reviewed frame exposes its modeling text under config.PREP_COL."""
    frame = frame.copy()
    if config.PREP_COL not in frame.columns:
        if "text" in frame.columns:
            frame = frame.rename(columns={"text": config.PREP_COL})
        elif config.TEXT_COL in frame.columns:
            frame = frame.rename(columns={config.TEXT_COL: config.PREP_COL})
    return frame


def load_reviewed_labels():
    """Load the consolidated per-target human-reviewed ground truth from REVIEWED_LABELS_DIR.

    Consolidation of the human-review rounds into these labels is described in the paper;
    treated here as a given input. Each per-target file provides at least: index, prep_text
    (or text), a resolved 3-class ``label`` in {negative, uncertain, positive}, and the weak
    ``rule_label``.
    """
    paths = sorted(glob(str(config.REVIEWED_LABELS_DIR / "*.xlsx")))
    paths += sorted(glob(str(config.REVIEWED_LABELS_DIR / "*.csv")))
    if not paths:
        raise FileNotFoundError(f"no *.xlsx/*.csv under {config.REVIEWED_LABELS_DIR}")
    return {
        t: _normalize_reviewed(_read_any(_pick_by_target(paths, t))) for t in config.TARGETS
    }


# ---------------------------------------------------------------------------
# Rule-labeled corpus frame + uncovered pool (notebook agg_df_load / cover cells)
# ---------------------------------------------------------------------------
def agg_df_load(target, df, preprocessor, rule_dict):
    """Weak-label the corpus for ``target`` and outer-merge the rule labels.

    Produces the aggregated corpus frame (prep_text + rule_label) from which the uncovered
    pool is reconstructed. The human-review rounds are NOT merged here: the consolidated
    reviewed ground truth is loaded separately (see load_reviewed_labels).
    """
    data_dict = preprocessor.target_filtering(target)
    merged_target_df = pd.concat([check_label(d, k) for k, d in data_dict.items()]).sort_index()

    temp_df = df.copy()
    temp_df.loc[merged_target_df.index, f"{target}_text"] = merged_target_df[config.TEXT_COL]
    temp_df.loc[merged_target_df.index, "label"] = merged_target_df["label"]
    target_df = temp_df.iloc[merged_target_df.index].copy()

    # prep_text := target-extracted text; raw_text := full conclusion; raw_label := weak rule label.
    df1 = target_df.reset_index()[["index", config.TEXT_COL, target + "_text", "label"]].set_axis(
        ["index", "raw_text", "prep_text", "raw_label"], axis=1
    )
    df2 = rule_dict[target][["index", config.TEXT_COL, "label"]].set_axis(
        ["index", "rule_text", "rule_label"], axis=1
    )

    agg_df = df1.merge(df2, on="index", how="outer")
    agg_df["n_samples"] = agg_df.groupby("prep_text")["prep_text"].transform("count")
    return agg_df


def build_uncovered_pool(agg_df, gt_df):
    """Rule-labeled corpus rows NOT present in the consolidated reviewed set (index/prep_text)."""
    un_df = agg_df.loc[
        ~agg_df["index"].isin(gt_df["index"]) & ~agg_df["prep_text"].isin(gt_df["prep_text"])
    ].drop_duplicates("prep_text")
    return un_df


# ---------------------------------------------------------------------------
# Embedding + weak-negative mining (notebook filtering cells)
# ---------------------------------------------------------------------------
def embed_texts(text_series, model, tokenizer, device, batch_size: int = EMBED_BATCH_SIZE):
    """CLS-token embeddings for a text Series (notebook llm_embedding_fn)."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm

    class _TextDataset(Dataset):
        def __init__(self, series):
            self.items = series.tolist()

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    model.eval()
    model.to(device)  # FIX: device-aware (notebook called .cuda() unconditionally).

    loader = DataLoader(_TextDataset(text_series), batch_size=batch_size, shuffle=False)
    outputs = []
    with torch.no_grad():
        for text in tqdm(loader):
            token = tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, padding_side="right"
            )
            token = {k: v.to(device) for k, v in token.items()}
            vec = model(**token).last_hidden_state[:, 0].cpu().float().numpy()
            outputs.append(vec)
    return np.concatenate(outputs, axis=0)


def mine_weak_negatives(gt_df, un_df, un_embeddings, dbscan_eps, seed: int = config.SEED):
    """UMAP + DBSCAN cluster the uncovered pool; peel pure rule-negative clusters into train."""
    from collections import Counter
    from sklearn.cluster import DBSCAN
    from umap import UMAP

    manifold = UMAP(random_state=seed)  # FIX: seed UMAP for determinism.
    manifold_vectors = manifold.fit_transform(un_embeddings)

    db = DBSCAN(eps=dbscan_eps)  # per-target eps from config.TARGETS
    numb_cluster = db.fit_predict(manifold_vectors)

    counter = Counter(numb_cluster)

    neg_texts, test_texts = [], []
    cumsum = 0
    train_label_counts = gt_df.label.value_counts()
    # Cap on mined negatives so training positives don't dwarf negatives.
    pos_thr = train_label_counts["positive"] - train_label_counts["negative"]
    for c, _ in counter.most_common():
        sub_df = un_df.iloc[np.where(numb_cluster == c)[0]]  # FIX: explicit [0] on np.where tuple.
        unique_label = sub_df["rule_label"].unique()
        if ("negative" in unique_label) and (len(unique_label) == 1) and (cumsum < pos_thr):
            neg_texts.append(sub_df)
            cumsum += sub_df.shape[0]
        else:
            test_texts.append(sub_df)

    un_agg_df = pd.concat(test_texts)
    neg_df = pd.concat(neg_texts)
    neg_df.loc[:, "label"] = "negative"

    test_df = un_agg_df.copy().drop_duplicates("prep_text")
    train_df = pd.concat([gt_df, neg_df.loc[~neg_df.prep_text.isin(test_df.prep_text)]])
    return train_df, test_df


def process_target(target, df, preprocessor, rule_dict, reviewed_dict, device):
    """Full per-target pipeline: load gt + rule corpus -> embed uncovered -> mine -> frames."""
    from transformers import AutoModel, AutoTokenizer

    print(f"[{target}] building rule-labeled corpus frame ...")
    agg_df = agg_df_load(target, df, preprocessor, rule_dict)
    gt_df = reviewed_dict[target]
    un_df = build_uncovered_pool(agg_df, gt_df)

    # FIX: encoder + eps come from config.TARGETS per target. The notebook hard-coded
    # PubMedBERT while target was 'metas' (a stale-state bug); config maps metas->MedEmbed,
    # recur->PubMedBERT and the matching DBSCAN eps.
    encoder_name = config.TARGETS[target]["encoder"]
    dbscan_eps = config.TARGETS[target]["dbscan_eps"]
    model_dir = str(config.model_path(encoder_name))

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, dtype="bfloat16")

    print(f"[{target}] embedding {un_df.shape[0]} uncovered rows with {encoder_name} ...")
    un_embeddings = embed_texts(un_df["prep_text"], model, tokenizer, device)

    print(f"[{target}] mining weak negatives (UMAP + DBSCAN eps={dbscan_eps}) ...")
    return mine_weak_negatives(gt_df, un_df, un_embeddings, dbscan_eps)


def main():
    parser = argparse.ArgumentParser(description="Build the all-data train/test split (recur & metas).")
    parser.add_argument(
        "--target", choices=list(config.TARGETS), default=None,
        help="target to build; default builds both recur and metas",
    )
    parser.add_argument(
        "--raw-file", default=None,
        help="explicit base CSV (name under RAW_REPORTS_DIR or absolute path)",
    )
    args = parser.parse_args()

    set_seed()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"  # FIX: no unconditional .cuda()
    print(f"device: {device}")

    targets = [args.target] if args.target else list(config.TARGETS)

    df, preprocessor = load_base_frame(args.raw_file)
    rule_dict = load_rule_labels()
    reviewed_dict = load_reviewed_labels()

    out_dir = config.SPLITS_DIR
    os.makedirs(out_dir, exist_ok=True)

    for target in targets:
        train_df, test_df = process_target(target, df, preprocessor, rule_dict, reviewed_dict, device)
        # Preserve the notebook's exact output filenames so paper results match.
        train_path = out_dir / f"{target}_train_df.xlsx"
        test_path = out_dir / f"{target}_test_df.xlsx"
        train_df.to_excel(train_path, index=False)
        test_df.to_excel(test_path, index=False)
        print(f"[{target}] wrote {train_df.shape[0]} train / {test_df.shape[0]} test rows -> {out_dir}")


if __name__ == "__main__":
    main()
