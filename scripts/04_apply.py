"""Batch inference of the fine-tuned 3-class conclusion classifiers over the full corpus.

For each target (recur / metas) this loads the best fine-tuned Lightning checkpoint
from logs/*<target>*/files/, weakly segments the raw conclusion text with the rule-based
Preprocessor, tokenizes the de-duplicated target-relevant text, runs the encoder+classifier,
converts logits to per-class softmax probabilities, and maps the argmax to
negative/uncertain/positive. Predictions are spread back onto every corpus row
(rows with no target-relevant text default to 'negative'), joined with cancer-type
metadata, and written out in two layouts.

Inputs:
  - config.RAW_REPORTS_DIR/all-report.csv         (full corpus with patient/report metadata)
  - config.CANCER_TYPE_MAP                        (암번호 per 병원등록번호+수술일자)
  - config.REPO_ROOT/cancer_dict.json             (암번호 -> English cancer-type name)
  - config.LOGS_ROOT/*<target>*/files/*epoch*.ckpt   (fine-tuned checkpoints)
  - config.MODEL_ROOT/<encoder>                   (local HuggingFace encoder per config.TARGETS)
  - config.REPO_ROOT/src/utils/abbrevations.json  (abbreviation expansion for Preprocessor)
Outputs:
  - config.PREDICTIONS_DIR/raw_format.xlsx         (full corpus + 암종/병합암종 + Metastasis/Recurrence class)
  - config.PREDICTIONS_DIR/prediction_format.xlsx  (metadata + per-target detail blocks)
Dropped from the notebook:
  - matplotlib/seaborn/font setup (hardcoded system font) — this notebook produces no plots.
  - Unused imports: umap.UMAP, sklearn StratifiedKFold / train_test_split / confusion_matrix /
    classification_report / f1_score, EarlyStopping, ModelCheckpoint, AutoModelForCausalLM,
    F1MacroCallback, datetime, torch.nn, the duplicate Preprocessor import.
  - Hardcoded model_dir_path/model_dict cells (replaced by config.model_path / config.TARGETS).
  - Bare diagnostic expressions (target_df.shape, ...) and print(raw_data_dir) calls, commented dead code.
"""
import os, sys, random, argparse
import json
import re
from glob import glob
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import numpy as np
import pandas as pd
import config
# ... heavy imports (torch / transformers / scipy / src.models) are done inside functions.

# Per-target output-column prefix (notebook: target_name_dict).
TARGET_NAME_DICT = {"recur": "Recurrence", "metas": "Metastasis"}

# DataLoader batch size for inference. Distinct from config.BATCH_SIZE (=8, the training
# batch size); the notebook used 256 for the forward-only prediction pass.
INFERENCE_BATCH_SIZE = 256

# Cancer-type merge groups (notebook: cancer_merge_keys / merged_cancer_label).
CANCER_MERGE_KEYS = ["Neurologic", "Hematologic", "Head & Neck", "Endocrine"]
MERGED_CANCER_LABEL = "Merged"

# Column-name suffixes assigned to each target's detail block, in dataframe column order:
#   검사결과결론내용, label(weak rule), pred class, P(neg), P(unc), P(pos), P(argmax).
DETAIL_SUFFIXES = ["_text", "_rule", "_class", "_negative", "_uncertain", "_positive", "_prob"]


from src.pipeline import set_seed, select_best_checkpoint  # hoisted shared helpers
    # pl.seed_everything is applied in main() once pytorch_lightning is imported.


def check_label(series: pd.Series, label: str) -> pd.DataFrame:
    """Notebook helper: turn a text Series into a frame with a constant weak-'label' column."""
    frame = series.to_frame()
    frame["label"] = label
    return frame


def run_inference(pl_model, data_loader, device: str) -> dict:
    """Forward-only prediction over a dataloader (no gold labels required).

    FIX: src.models.get_model_predictions unconditionally reads batch['labels'], which the
    full-corpus apply dataset does not contain (SupervisedTextDataset finds no 'target' column,
    so it emits no labels). This mirrors the notebook's label-optional variant instead.
    """
    import torch
    from tqdm import tqdm

    pl_model.eval()
    pl_model.to(device)
    all_logits, all_targets = [], []
    for batch in tqdm(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        if "labels" in batch:
            all_targets.append(batch["labels"].cpu().numpy())
        with torch.no_grad():
            logits = pl_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        all_logits.append(logits.cpu().numpy())

    all_logits = np.concatenate(all_logits)
    all_targets = np.concatenate(all_targets) if len(all_targets) > 0 else None
    all_preds = np.argmax(all_logits, axis=1)
    return dict(logits=all_logits, targets=all_targets, preds=all_preds)


def predict_target(target: str, df: pd.DataFrame, preprocessor, device: str) -> pd.DataFrame:
    """Predict one target over the full corpus; return a per-row detail frame aligned to `df`.

    The returned frame has one row per original corpus row (index 0..N-1); rows with no
    target-relevant text are NaN. Columns are TARGET_NAME_DICT[target] + DETAIL_SUFFIXES.
    """
    from transformers import AutoModel, AutoTokenizer
    from torch.utils.data import DataLoader
    from scipy.special import softmax
    from src.models import SupervisedTextDataset, SupervisedPLModel

    # Rule-based segmentation into weak positive/negative/uncertain buckets.
    data_dict = preprocessor.target_filtering(target)
    target_df = pd.concat([check_label(d, k) for k, d in data_dict.items()]).sort_index()

    # Load encoder + fine-tuned classifier head from checkpoint.
    ckpt_path = select_best_checkpoint(target)
    encoder_dir = str(config.model_path(config.TARGETS[target]["encoder"]))
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir)
    language_model = AutoModel.from_pretrained(encoder_dir)
    pl_model = SupervisedPLModel.load_from_checkpoint(
        ckpt_path,
        encoder=language_model,
        hidden_dim=language_model.config.hidden_size,
        num_classes=len(config.LABEL_DICT),
    )

    # De-duplicate on the (abbrev-expanded, target-relevant) text; predict once per unique text.
    # Built before the prediction columns are added, so the dataset finds no 'target' column
    # and emits no labels (inference-only). reset_index keeps row order aligned with predictions.
    dedup_target_df = target_df.drop_duplicates(config.TEXT_COL).reset_index(drop=True)
    test_dataset = SupervisedTextDataset(
        dedup_target_df, tokenizer, text_col=config.TEXT_COL, max_length=config.MAX_LENGTH
    )
    test_loader = DataLoader(test_dataset, batch_size=INFERENCE_BATCH_SIZE, shuffle=False)

    prediction = run_inference(pl_model, test_loader, device)
    probs = softmax(prediction["logits"], axis=1)
    predicted_probs = probs[np.arange(len(probs)), prediction["preds"]]

    dedup_target_df[target] = prediction["preds"]
    for k, n in config.LABEL_DICT.items():
        dedup_target_df[target + "_" + k] = probs[:, n]
    dedup_target_df[target + "_prob"] = predicted_probs

    # Spread the per-unique-text predictions back onto every occurrence, then onto every
    # corpus row (0..N-1), leaving rows without target text as NaN.
    pred_df = pd.merge(
        target_df.drop(columns=["label"]).reset_index(),
        dedup_target_df,
        on=config.TEXT_COL,
        how="left",
    )
    pred_df.index = pred_df["index"].values
    pred_df[target] = pred_df[target].map(config.INV_LABEL_DICT)
    full_index = pd.DataFrame({"index": np.arange(0, df.index.max() + 1)})
    filled_df = pd.merge(full_index, pred_df, on="index", how="left")
    filled_df = filled_df.drop(columns=["index"]).set_axis(
        [TARGET_NAME_DICT[target] + s for s in DETAIL_SUFFIXES], axis=1
    )
    return filled_df.copy()


def load_raw_corpus() -> pd.DataFrame:
    """Read the full-corpus report CSV and expose its row number as an 'index' column."""
    # FIX: the notebook read glob(...)[0] (arbitrary order). Select the full-report CSV by name;
    # it carries the metadata columns (병원등록번호/수술일자/성별/검사나이) needed downstream.
    csv_path = config.RAW_REPORTS_DIR / "all-report.csv"
    if not csv_path.exists():
        candidates = sorted(glob(str(config.RAW_REPORTS_DIR / "*.csv")))
        if not candidates:
            raise FileNotFoundError(f"No CSV found under {config.RAW_REPORTS_DIR}")
        csv_path = Path(candidates[0])
    return pd.read_csv(csv_path, encoding="utf-8-sig", engine="c").reset_index()


def build_cancer_type_frame() -> pd.DataFrame:
    """Load cancer-type metadata and attach English 암종 / merged 병합암종 columns."""
    cancer_type_df = pd.read_excel(config.CANCER_TYPE_MAP)
    with open(os.path.join(config.REPO_ROOT, "cancer_dict.json"), "r", encoding="utf-8") as f:
        cancer_dict = json.load(f)
    cancer_en_dict = {int(k): v["name_en"] for k, v in cancer_dict.items()}
    cancer_merged_en_dict = {
        k: (v if v not in CANCER_MERGE_KEYS else MERGED_CANCER_LABEL)
        for k, v in cancer_en_dict.items()
    }
    codes = cancer_type_df["암번호"].fillna(99).astype(int)
    cancer_type_df["암종"] = codes.apply(lambda x: cancer_en_dict.get(x, "Missing"))
    cancer_type_df["병합암종"] = codes.apply(lambda x: cancer_merged_en_dict.get(x, "Missing"))
    return cancer_type_df


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--target", choices=list(config.TARGETS.keys()), default=None,
        help="Target to predict. Default: run both recur and metas (reproduces the paper output).",
    )
    args = parser.parse_args()

    set_seed()
    import torch
    import pytorch_lightning as pl
    pl.seed_everything(config.SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    targets = [args.target] if args.target else list(config.TARGETS.keys())

    # --- Raw corpus + rule-based preprocessor (shared across targets) --------
    df = load_raw_corpus()
    text_df = df[config.TEXT_COL].dropna()

    from src.utils.preprocessing import Preprocessor
    preprocessor = Preprocessor(
        df=text_df,
        negative_patterns=config.NEGATIVE_PATTERNS,
        uncertain_patterns=config.UNCERTAIN_PATTERNS,
        abbrev_path=os.path.join(config.REPO_ROOT, "src", "utils", "abbrevations.json"),
    )

    # --- Per-target inference (computed against the pre-merge corpus df) -----
    target_pred_dict = {}
    for target in targets:
        target_pred_dict[target] = predict_target(target, df, preprocessor, device)

    # --- Attach cancer-type metadata -----------------------------------------
    cancer_type_df = build_cancer_type_frame()
    df = df.merge(cancer_type_df.drop(columns=["암번호"]), on=["병원등록번호", "수술일자"], how="left")

    # Predicted class per row; rows without target-relevant text default to 'negative'.
    for target in targets:
        name = TARGET_NAME_DICT[target]
        df.loc[:, name] = target_pred_dict[target][name + "_class"].fillna("negative")

    # --- Write outputs --------------------------------------------------------
    save_dir = config.PREDICTIONS_DIR
    os.makedirs(save_dir, exist_ok=True)
    df.to_excel(os.path.join(save_dir, "raw_format.xlsx"), index=False)

    # Detail layout: metadata block, then each target's detail block.
    # Order metas-then-recur to match the notebook's concat order.
    meta_cols = ["병원등록번호", "수술일자", "성별", "검사나이", "암종", "병합암종", config.TEXT_COL]
    blocks = [df[meta_cols]]
    for target in ["metas", "recur"]:
        if target in target_pred_dict:
            blocks.append(target_pred_dict[target])
    detail_df = pd.concat(blocks, axis=1)
    detail_df.to_excel(os.path.join(save_dir, "prediction_format.xlsx"), index=False)

    print(f"Wrote {os.path.join(save_dir, 'raw_format.xlsx')}")
    print(f"Wrote {os.path.join(save_dir, 'prediction_format.xlsx')}")


if __name__ == "__main__":
    main()
