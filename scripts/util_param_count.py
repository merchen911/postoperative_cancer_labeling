"""Instantiate the 8-encoder candidate roster and report per-model parameter counts.

This utility loads every encoder in ``config.CANDIDATE_MODELS`` (local HuggingFace
directories under ``config.MODEL_ROOT``), prints the raw encoder parameter count and
hidden size, wraps each in the 3-class ``SupervisedPLModel`` head used for training,
and prints the total trainable-parameter count.  It also demonstrates the canonical
10-fold ``StratifiedKFold`` cross-validation split used by the training scripts,
reporting fold sizes and the focal-loss ``alpha`` derived from fold 0.

Inputs:  per-target train/test split workbooks under config.SPLITS_DIR matching
         "{target}*" (train frame has column 'label', test frame has column 'R2_label').
         Encoder directories under config.MODEL_ROOT (config.CANDIDATE_MODELS).
Outputs: none written to disk -- parameter counts and CV fold sizes are printed to stdout.
Dropped from the notebook:
  - Local re-definitions of save_dir_setup / compute_alpha / get_model_predictions
    (cell 2) -> reuse the canonical implementations in src.utils / src.models.
  - Weak-label pattern lists, check_label(), measure_dict, and the redundant
    Preprocessor import (cell 1) -> unused here; patterns live in config.
  - matplotlib / seaborn / font setup (hardcoded system font) and the entire
    commented-out training / checkpoint / confusion-matrix / prediction-export block
    (cell 6) -> this utility only counts parameters and sets up CV.
  - Bare "data_dir_path" display cell (cell 3) and the os.listdir(model_dir) diagnostic
    (cell 5); the hard-coded model_dict is replaced by config.CANDIDATE_MODELS.
  - Unused imports: AutoModelForCausalLM, umap.UMAP, train_test_split, DataLoader,
    EarlyStopping / ModelCheckpoint / pl.Trainer, F1MacroCallback, SupervisedTextDataset,
    SimpleClassifier.
"""
import os, sys, random, argparse
from glob import glob
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import numpy as np
import pandas as pd
import config
# heavy imports (torch / transformers / pytorch_lightning) live inside the functions below


from src.pipeline import set_seed  # hoisted shared helper


def find_split_files(part_dir, target):
    """Locate the (train, test) split workbooks for a target inside ``part_dir``.

    FIX: the notebook selected files by glob index ([1]=train, [0]=test).  We instead
    select by explicit 'train'/'test' substring in the filename, falling back to the
    notebook's sorted-index behaviour only if the names are non-standard.
    """
    candidates = sorted(glob(os.path.join(str(part_dir), f"{target}*")))
    train_hits = [p for p in candidates if "train" in os.path.basename(p).lower()]
    test_hits = [p for p in candidates if "test" in os.path.basename(p).lower()]
    train_path = train_hits[0] if train_hits else (candidates[1] if len(candidates) > 1 else None)
    test_path = test_hits[0] if test_hits else (candidates[0] if candidates else None)
    return train_path, test_path


def load_target_frames(target):
    """Read the train/test split workbooks for ``target`` and build the integer target column.

    Returns (train_df, test_df); either may be None if the files are absent.
    """
    train_path, test_path = find_split_files(config.SPLITS_DIR, target)
    if train_path is None:
        return None, None

    train_df = pd.read_excel(train_path)
    # Train frame is weak-labelled in column 'label'; unknown labels fall back to 0 (negative).
    train_df["target"] = train_df["label"].map(lambda x: config.LABEL_DICT.get(x, 0))

    test_df = None
    if test_path is not None:
        test_df = pd.read_excel(test_path)
        # Test frame carries the reviewer label in column 'R2_label'.
        test_df["target"] = test_df["R2_label"].map(lambda x: config.LABEL_DICT.get(x, 0))
    return train_df, test_df


def prepare_cv(df, n_splits: int = config.N_SPLITS, seed: int = config.SEED):
    """Canonical 10-fold StratifiedKFold split over df['target'].

    Returns {fold_index: {'train': DataFrame, 'val': DataFrame}}.

    FIX: the notebook's prepare_cv_and_split also accepted unused test_df / label_dict
    arguments -- dropped here (the target column is built by load_target_frames).
    """
    from sklearn.model_selection import StratifiedKFold

    df = df.copy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)  # FIX: explicit random_state
    cv_results = {}
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["target"])):
        cv_results[fold] = {
            "train": df.iloc[train_idx].reset_index(drop=True),
            "val": df.iloc[val_idx].reset_index(drop=True),
        }
    return cv_results


def count_model_parameters(num_classes: int = len(config.LABEL_DICT)):
    """Load every candidate encoder, wrap it in SupervisedPLModel, and count parameters.

    Returns a list of dicts (one per encoder).  Parameter counts are identical across
    targets and CV folds, so this runs once.
    """
    import torch  # noqa: F401  (imported so a missing torch fails loudly here)
    from transformers import AutoModel, AutoTokenizer
    from src.models import SupervisedPLModel

    rows = []
    for model_name, model_dir in config.CANDIDATE_MODELS.items():
        path = str(config.model_path(model_dir))

        # Tokenizer is loaded for parity with the notebook; it does not affect the count.
        _ = AutoTokenizer.from_pretrained(path)
        language_model = AutoModel.from_pretrained(path)

        encoder_params = language_model.num_parameters()
        hidden = language_model.config.hidden_size
        print(model_name, model_dir)
        print("params : ", encoder_params)
        print(hidden)

        # FIX: focal-loss alpha is stored as a plain tensor on the loss (not an
        # nn.Parameter/buffer), so it never appears in .parameters() and cannot change
        # the count -> pass alpha=None to keep parameter counting data/target independent.
        pl_model = SupervisedPLModel(
            encoder=language_model,
            hidden_dim=hidden,
            num_classes=num_classes,
            lr=config.LR,
            gamma=config.FOCAL_GAMMA,
            alpha=None,
        )
        total = sum(p.numel() for p in pl_model.parameters())
        print(f"Total parameters in pl_model: {total}")

        rows.append({
            "name": model_name,
            "dir": model_dir,
            "encoder_params": int(encoder_params),
            "hidden_size": int(hidden),
            "total_params": int(total),
        })
    return rows


def report_cv_setup(target):
    """Demonstrate the 10-fold StratifiedKFold setup for one target and report fold 0 alpha."""
    from src.utils import compute_alpha

    train_df, _test_df = load_target_frames(target)
    if train_df is None:
        print(f"[{target}] no split files matching '{target}*' under {config.SPLITS_DIR}; skipping CV setup.")
        return None

    cv_dict = prepare_cv(train_df, n_splits=config.N_SPLITS, seed=config.SEED)
    print(f"[{target}] {config.N_SPLITS}-fold StratifiedKFold over {len(train_df)} training rows:")
    for fold, split in cv_dict.items():
        print(f"  fold {fold:02d}: train={len(split['train'])}  val={len(split['val'])}")

    # Notebook computed the focal-loss alpha from fold 0's train subset.
    alpha = compute_alpha(cv_dict[0]["train"])
    print(f"[{target}] fold 0 focal-loss alpha: {np.asarray(alpha).tolist()}")
    return cv_dict


def main():
    set_seed()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", choices=list(config.TARGETS), default=None,
                        help="Target for the CV-setup demo; default runs both recur and metas.")
    args = parser.parse_args()

    targets = [args.target] if args.target else list(config.TARGETS)

    # 1) Parameter counts (target/fold independent -> computed once).
    print("=== Candidate-encoder parameter counts ===")
    rows = count_model_parameters()
    print("\n=== Summary ===")
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    # 2) Canonical 10-fold StratifiedKFold CV setup, per target.
    print("\n=== Cross-validation setup ===")
    for target in targets:
        report_cv_setup(target)


if __name__ == "__main__":
    main()
