"""Shared pipeline helpers hoisted out of the per-stage scripts.

These small utilities were byte-for-byte (or near) duplicates across several of the
``scripts/*.py`` stage scripts. Consolidating them here keeps a single source of truth:

- ``set_seed``            : reproducibility seed (every stage script).
- ``build_age_bins`` /
  ``age_group``           : cohort age-band binning (stages 05, 06, 07).
- checkpoint selection    : pick the best fine-tuned Lightning checkpoint by the
                            val_f1_macro parsed from its filename (stages 03, 04, 08).

Only clear duplicates were hoisted; stage-specific logic stays in each script.

Heavy ML libraries are imported lazily (inside functions) so importing this module
stays cheap and does not require torch/transformers to be installed.
"""
from __future__ import annotations

import os
import random
import re
import sys
from glob import glob

import numpy as np

# Make the repo-root `config` importable even when this module is imported directly
# (the stage scripts already put the repo root on sys.path, but be self-sufficient).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# --- Reproducibility ---------------------------------------------------------
def set_seed(seed: int = config.SEED):
    """Seed Python, NumPy and (if installed) torch for deterministic runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# --- Cohort age binning (age_bins = [0, 40, 50, 60, 70, 80]) -----------------
def build_age_bins():
    """Return (age_bins, age_dict) exactly as the notebooks constructed them.

    age_bins = [0, 40, 50, 60, 70, 80]; age_dict maps a bin index to a label
    ('~40', '40~50', ..., '70~80', '80~').
    """
    age_bins = [0] + list(range(40, 90, 10))
    age_dict = {}
    for n, (st, ed) in enumerate(zip(age_bins, age_bins[1:])):
        strs = r'~'
        if n > 0:
            strs = f'{st}{strs}'
        strs = f'{strs}{ed}'
        age_dict[n] = strs
    else:
        age_dict[n + 1] = f'{ed}~'
    return age_bins, age_dict


def age_group(age, age_bins):
    """Map a numeric age to its bin index (last band is open-ended)."""
    for n, (sty, edy) in enumerate(zip(age_bins[:-1], age_bins[1:])):
        if sty <= age < edy:
            return n
    else:
        return n + 1


# --- Fine-tuned checkpoint selection -----------------------------------------
# Training saved checkpoints as 'clf-epoch=NN-val_f1_macro=X.XXXX.ckpt'
# (ModelCheckpoint monitor='val_f1_macro', mode='max', save_top_k=1). Select by the
# parsed val_f1_macro (tie-break: latest epoch) instead of relying on glob order.
def parse_val_f1(path: str) -> float:
    """val_f1_macro parsed from a checkpoint filename (-1.0 if absent)."""
    m = re.search(r"val_f1_macro=([0-9]*\.?[0-9]+)", os.path.basename(path))
    return float(m.group(1)) if m else -1.0


def parse_epoch(path: str) -> int:
    """Epoch index parsed from a checkpoint filename (-1 if absent)."""
    m = re.search(r"epoch=(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def best_checkpoint(candidates) -> str:
    """Pick the best checkpoint from a list of paths.

    Preference order: highest val_f1_macro (tie-break latest epoch); if no filename
    carries val_f1_macro, highest epoch; if neither, newest mtime. Raises if empty.
    """
    candidates = list(candidates)
    if not candidates:
        raise FileNotFoundError("No candidate checkpoints supplied")
    with_f1 = [p for p in candidates if parse_val_f1(p) >= 0]
    if with_f1:
        return max(with_f1, key=lambda p: (parse_val_f1(p), parse_epoch(p)))
    with_ep = [p for p in candidates if parse_epoch(p) >= 0]
    if with_ep:
        return max(with_ep, key=parse_epoch)
    return max(candidates, key=os.path.getmtime)


def best_checkpoint_in_dir(file_dir: str) -> str:
    """Best '*epoch*.ckpt' checkpoint inside ``file_dir`` (the 'last.ckpt' is skipped
    because it carries no 'epoch=' token)."""
    ckpts = sorted(glob(os.path.join(str(file_dir), "*epoch*.ckpt")))
    if not ckpts:
        raise FileNotFoundError(f"No '*epoch*.ckpt' checkpoint in {file_dir}")
    return best_checkpoint(ckpts)


def select_best_checkpoint(target: str) -> str:
    """Best checkpoint for ``target`` under logs/*<target>*/files/*epoch*."""
    pattern = str(config.LOGS_ROOT / f"*{target}*" / "files" / "*epoch*")
    candidates = sorted(glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint matching {pattern!r}. Restore the fine-tuned Lightning "
            f"checkpoints for target={target!r} produced by the training stage."
        )
    return best_checkpoint(candidates)
