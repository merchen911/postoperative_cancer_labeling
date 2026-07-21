"""Fine-tune and evaluate the 3-class pathology-conclusion classifier (all-data track).

For each target (recur / metas) this loads the weakly-labeled train split and the
held-out test split, carves a stratified 90/10 train/val split, fine-tunes a
transformer encoder + linear head with FocalLoss (class-balanced alpha), selects the
best checkpoint by macro-F1 on the validation set, then writes the validation
confusion matrix and per-row test predictions. By default only the selected encoder
for the target is trained (config.TARGETS[target]['encoder']); pass --models all to
benchmark every encoder in config.CANDIDATE_MODELS, or --models <key> for a single one.

NOTE: the training loop uses a single stratified 90/10 train/val split
(sklearn.train_test_split), NOT StratifiedKFold. A reference partial-data loop does use
10-fold CV; that is intentionally not reproduced here so results match the all-data
procedure. The multi-model loop and the resume guards below follow that reference script.

Inputs:
  {SPLITS_DIR}/{target}_train_df.xlsx   (has 'label' + 'prep_text')
  {SPLITS_DIR}/{target}_test_df.xlsx    (no gold label column)
  {MODEL_ROOT}/<encoder-dir>/           (local HuggingFace encoders)
Outputs (under {LOGS_ROOT}/{target}-<encoder-dir>-{version}/):
  files/clf-epoch=NN-val_f1_macro=0.XXXX.ckpt  best checkpoint (+ last.ckpt)
  files/test-with-pred.xlsx                     test rows + 'prediction' column
  figures/val_confusion_matrix.png              validation confusion matrix

Dropped from the notebook:
  - hardcoded system-font setup (both header cells) -> config.apply_font().
  - Local redefinitions of save_dir_setup / compute_alpha / get_model_predictions
    (cell [2]): reuse src.utils.compute_alpha and src.models.get_model_predictions.
    Save dirs are built under config.LOGS_ROOT because the repo's src.utils.save_dir_setup
    hardcodes a cwd-relative '../logs' (see setup_save_dirs FIX comment).
  - Exploratory cells [5]/[6]: loading the metas encoder only to print params and inspect
    `language_model.encoder`. These also caused the target state-order bug (cell [5] reset
    target='metas' so cell [7] fine-tuned the metas encoder on the recur data / recur
    prefix). FIX: target is now a clean parameter; each target uses its own data + encoder.
  - Bare trailing display cell [9] `save_dict['file']`.
  - Unused imports (StratifiedKFold, UMAP, AutoModelForCausalLM, f1_score,
    classification_report, accuracy_score, Counter) and the unused helper check_label.
  - Commented-out test confusion matrix and per-class test exports.
"""
import os, sys, random, argparse, re
from glob import glob
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import numpy as np
import pandas as pd
import config

TEST_BATCH_SIZE = 128                             # notebook value for the (large) test loader
LABEL_ORDER = ["Negative", "Uncertain", "Positive"]  # display order == config.LABEL_DICT order


from src.pipeline import set_seed, best_checkpoint_in_dir  # hoisted shared helpers


def resolve_models(target: str, models_arg: str | None) -> list[str]:
    """Local encoder directory names to fine-tune for `target`.

    default -> [selected encoder]; 'all' -> every candidate; '<key>' -> that candidate.
    """
    if models_arg == "all":
        return list(config.CANDIDATE_MODELS.values())
    if models_arg:
        return [config.CANDIDATE_MODELS[models_arg]]  # models_arg is a CANDIDATE_MODELS key
    return [config.TARGETS[target]["encoder"]]


def setup_save_dirs(target: str, encoder_dir: str, version: str) -> dict:
    """Build the log/figures/files tree, mirroring the notebook's save_dir_setup naming."""
    # FIX: repo src.utils.save_dir_setup hardcodes a cwd-relative '../logs'; replicate its
    # layout ({prefix}-{model}-{version}) under config.LOGS_ROOT so paths are config-driven.
    log_dir = config.LOGS_ROOT / f"{target}-{encoder_dir}-{version}"
    fig_dir = log_dir / "figures"
    file_dir = log_dir / "files"
    for d in (log_dir, fig_dir, file_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {"log": str(log_dir), "figure": str(fig_dir), "file": str(file_dir)}


def load_split(target: str):
    """Read train/test xlsx and attach the integer 'target' column to the train frame."""
    # FIX: select files by explicit name rather than sorted(glob(...))[0]/[1] index magic.
    train_df = pd.read_excel(config.SPLITS_DIR / f"{target}_train_df.xlsx")
    test_df = pd.read_excel(config.SPLITS_DIR / f"{target}_test_df.xlsx")
    # Map weak label -> integer (unknown/missing -> 0/negative), exactly as the notebook.
    train_df["target"] = train_df["label"].map(lambda x: config.LABEL_DICT.get(x, 0))
    # NOTE: test_df intentionally has NO 'target' column (the notebook commented that line
    # out) -> it is an unlabeled prediction set and SupervisedTextDataset yields no 'labels'.
    return train_df, test_df


def predict_unlabeled(pl_model, data_loader, device) -> np.ndarray:
    """Predicted class indices for a loader whose batches carry no 'labels'."""
    # FIX: src.models.get_model_predictions requires batch['labels']; the all-data test
    # split has no gold label, so use this label-tolerant forward pass for it (mirrors the
    # notebook's local get_model_predictions branch). The labeled val set still uses the
    # reused src.models.get_model_predictions.
    import torch
    pl_model.eval(); pl_model.to(device)
    logits_all = []
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        with torch.no_grad():
            logits = pl_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits_all.append(logits.cpu().numpy())
    logits_all = np.concatenate(logits_all)
    return np.argmax(logits_all, axis=1)


def save_val_confusion_matrix(targets, preds, fig_path: str):
    """Write the validation confusion-matrix heatmap (Negative/Uncertain/Positive)."""
    import matplotlib
    matplotlib.use("Agg")  # headless-safe figure saving
    from matplotlib import pyplot as plt
    import seaborn as sbn
    from sklearn.metrics import confusion_matrix
    config.apply_font()  # FIX: replaces the hardcoded system-font path
    plt.rcParams["font.size"] = 15
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    sbn.heatmap(
        confusion_matrix(targets, preds), annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER, ax=ax, cbar=False,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=400)
    plt.close(fig)


def train_and_evaluate(target, encoder_dir, train_full_df, test_df,
                       version, device, force=False) -> str:
    """Fine-tune one encoder for one target, then write CM + test predictions."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from sklearn.model_selection import train_test_split
    from src.models import (SupervisedTextDataset, SupervisedPLModel,
                            F1MacroCallback, get_model_predictions)
    from src.utils import compute_alpha

    save_dict = setup_save_dirs(target, encoder_dir, version)

    # Resume guard (from training.py): skip a model that already produced predictions.
    if not force and glob(os.path.join(save_dict["file"], "test-with-pred*")):
        print(f"[skip] {save_dict['file']} already has test-with-pred.xlsx")
        return save_dict["file"]

    # alpha is computed on the FULL train frame (before the val split), as in the notebook.
    alpha = compute_alpha(train_full_df)
    train_df, val_df = train_test_split(
        train_full_df, test_size=0.1, stratify=train_full_df["target"],
        random_state=config.SEED,
    )

    model_dir = str(config.model_path(encoder_dir))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    language_model = AutoModel.from_pretrained(model_dir)
    print(f"[{target}] encoder={encoder_dir} "
          f"params={language_model.num_parameters()} hidden={language_model.config.hidden_size}")

    train_ds = SupervisedTextDataset(train_df, tokenizer, text_col=config.PREP_COL, max_length=config.MAX_LENGTH)
    val_ds = SupervisedTextDataset(val_df, tokenizer, text_col=config.PREP_COL, max_length=config.MAX_LENGTH)
    test_ds = SupervisedTextDataset(test_df, tokenizer, text_col=config.PREP_COL, max_length=config.MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE, shuffle=False)

    pl_model = SupervisedPLModel(
        encoder=language_model,
        hidden_dim=language_model.config.hidden_size,
        num_classes=len(config.LABEL_DICT),
        lr=config.LR, gamma=config.FOCAL_GAMMA, alpha=alpha,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_f1_macro", patience=config.PATIENCE, mode="max", verbose=True,
    )
    ckpt_cb = ModelCheckpoint(
        dirpath=save_dict["file"], filename="clf-{epoch:02d}-{val_f1_macro:.4f}",
        monitor="val_f1_macro", mode="max", save_top_k=1, save_last=True, save_weights_only=True,
    )
    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[early_stop_cb, ckpt_cb, F1MacroCallback(val_loader)],
    )
    # Resume guard (from training.py): only fit if no checkpoint exists yet.
    if force or not glob(os.path.join(save_dict["file"], "*epoch*.ckpt")):
        trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    ckpt_path = best_checkpoint_in_dir(save_dict["file"])
    pl_model = SupervisedPLModel.load_from_checkpoint(ckpt_path, encoder=language_model)
    pl_model.eval()

    # Validation set is labeled -> reuse the repo helper (returns targets + preds).
    val_outputs = get_model_predictions(pl_model, val_loader)
    save_val_confusion_matrix(
        val_outputs["targets"], val_outputs["preds"],
        os.path.join(save_dict["figure"], "val_confusion_matrix.png"),
    )

    # Test set is unlabeled -> label-tolerant prediction, then persist rows + prediction.
    test_preds = predict_unlabeled(pl_model, test_loader, device)
    out_df = test_df.copy()
    out_df["prediction"] = [config.INV_LABEL_DICT[i] for i in test_preds]
    out_path = os.path.join(save_dict["file"], "test-with-pred.xlsx")
    out_df.to_excel(out_path, index=False)
    print(f"[{target}] wrote {out_path}")
    return save_dict["file"]


def main():
    parser = argparse.ArgumentParser(description="Fine-tune + evaluate the conclusion classifier.")
    parser.add_argument("--target", choices=list(config.TARGETS), default=None,
                        help="Target to train (default: both recur and metas).")
    parser.add_argument("--models", default=None,
                        help="'all' to benchmark every config.CANDIDATE_MODELS encoder, or a "
                             "CANDIDATE_MODELS key. Default: config.TARGETS[target]['encoder'].")
    parser.add_argument("--version", default="v001", help="save_dir version suffix.")
    parser.add_argument("--force", action="store_true",
                        help="Retrain / overwrite even if checkpoints or predictions exist.")
    args = parser.parse_args()

    if args.models not in (None, "all") and args.models not in config.CANDIDATE_MODELS:
        parser.error(f"--models must be 'all' or one of {list(config.CANDIDATE_MODELS)}")

    set_seed()
    import torch
    import pytorch_lightning as pl
    pl.seed_everything(config.SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    targets = [args.target] if args.target else list(config.TARGETS)
    for target in targets:
        train_full_df, test_df = load_split(target)
        for encoder_dir in resolve_models(target, args.models):
            train_and_evaluate(
                target, encoder_dir, train_full_df, test_df,
                args.version, device, force=args.force,
            )


if __name__ == "__main__":
    main()
