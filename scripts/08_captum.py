"""Integrated-Gradients token attributions for the fine-tuned conclusion classifiers.

For each target (recur / metas) this loads the target's fine-tuned Lightning checkpoint
and its encoder/tokenizer, scores the reviewer-gold workbook rows, then for the requested
gold class(es) runs Layer Integrated Gradients (captum) on the lowest- and highest-
confidence examples and renders a Times-New-Roman-styled HTML attribution visualization.
It also emits a per-sample breakdown that attributes one example against every class.

Ported from the Integrated-Gradients analysis notebook.
Inputs:
  - Reviewer gold workbooks (gold column config.GOLD_COL): config.REVIEWER_GOLD_DIR/*.xlsx
    (two files; sheets 'negative'/'uncertain'/'positive' per target).
  - Fine-tuned checkpoints: config.LOGS_ROOT/*<target>*/files/*epoch*.ckpt.
  - Local HuggingFace encoders: config.model_path(config.TARGETS[target]['encoder']).
Outputs (under --out-dir, default config.LOGS_ROOT/'captum'):
  - {target}_{class}_attrs.html          (low/high-confidence attribution panel per class)
  - {target}_{class}_index_{n}_attrs.html (single sample attributed across all 3 classes)
Dropped from the notebook (exploratory / dead / broken):
  - cell 1 raw-CSV read + Preprocessor init (result never used; labels come from workbooks).
  - torchtext stubs SimpleVocab / build_vocab_from_iterator / SimpleLabelField and check_label
    (all unused), and forward_embeds (unused; IG runs on input_ids, not inputs_embeds).
  - stray `from sympy.geometry.entity import N` (cell 6), cell 7 (label_dict echo),
    cell 8 (pred echo), and cell 10 (undefined metas_errors_sorted/recur_errors_sorted/model).
  - unused imports (matplotlib/seaborn, umap, sklearn, AutoModelForCausalLM, IPython, tqdm...).
"""
import os, sys, random, argparse
import re
from glob import glob
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import numpy as np
import pandas as pd
import config

# Heavy ML deps (installed in the paper environment, not here).
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

from src.models import SupervisedTextDataset, SupervisedPLModel


# Gold class sheets present in each reviewer workbook, in label order.
CLASS_SHEETS = ["negative", "uncertain", "positive"]
GOLD_COL = config.GOLD_COL  # reviewer gold label column (kept exactly as in the workbooks)


from src.pipeline import set_seed, select_best_checkpoint  # hoisted shared helpers


# --- captum forward helpers (faithful to the notebook) -----------------------
@torch.no_grad()
def predict_proba(encoded, pl_model):
    """Raw logits for a tokenizer-encoded single example."""
    pl_model.eval()
    device = next(pl_model.parameters()).device
    return pl_model(**{k: v.to(device) for k, v in encoded.items()})


def forward_with_softmax(tokens, model):
    """Softmax probabilities for a tokenizer-encoded single example."""
    return torch.softmax(model(**{k: v.to(model.device) for k, v in tokens.items()}), dim=1)


def add_attributions_to_visualizer(attributions, text, pred_prob, pred, label, delta, vis_data_records):
    """Reduce per-token IG attributions and append a captum VisualizationDataRecord.

    NOTE: this mutates `vis_data_records` in place (and returns None), which is why callers
    wrap the result with list(filter(None, vis_data_records)) before visualize_text.
    """
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()

    vis_data_records.append(visualization.VisualizationDataRecord(
        attributions,
        pred_prob,
        pred,
        label,
        1,
        attributions.sum(),
        text,
        delta))


# --- data / model loading ----------------------------------------------------
def load_labeled_dfs(reviewer_dir):
    """Read the two reviewer gold workbooks into {'recur': df, 'metas': df} with gold column GOLD_COL.

    Each workbook has 'negative'/'uncertain'/'positive' sheets; rows are concatenated and
    NaN-dropped. For recur, the first 100 rows of the 'negative' sheet have the gold column
    backfilled to 'negative' (as in the notebook).
    """
    files = sorted(glob(os.path.join(str(reviewer_dir), "*.xlsx")))
    if len(files) < 2:
        raise FileNotFoundError(
            f"Expected two reviewer workbooks in {reviewer_dir}, found {len(files)}: {files}")
    sheets = [pd.read_excel(f, sheet_name=None) for f in files]

    def find(sub):
        for f, s in zip(files, sheets):
            if sub in os.path.basename(f).lower():
                return s
        return None

    # FIX: notebook relied on alphabetical unpacking `metas_df, recur_df = sorted(glob(...))`.
    # Prefer explicit filename matching; fall back to that sorted order if names are opaque.
    metas_sheets = find("meta")
    recur_sheets = find("recur")
    if metas_sheets is None or recur_sheets is None:
        metas_sheets, recur_sheets = sheets[0], sheets[1]

    metas_labeled = pd.concat([metas_sheets[c] for c in CLASS_SHEETS]).dropna()
    recur_sheets["negative"].loc[:99, GOLD_COL] = "negative"
    recur_labeled = pd.concat([recur_sheets[c] for c in CLASS_SHEETS]).dropna()
    return {"metas": metas_labeled, "recur": recur_labeled}


def load_target_model(target, device):
    """Load tokenizer, encoder, fine-tuned Lightning model, and set up captum LIG."""
    encoder_name = config.TARGETS[target]["encoder"]
    model_dir = str(config.model_path(encoder_name))

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    language_model = AutoModel.from_pretrained(model_dir)

    ckpt = select_best_checkpoint(target)
    pl_model = SupervisedPLModel.load_from_checkpoint(
        ckpt,
        encoder=language_model,
        hidden_dim=language_model.config.hidden_size,
        num_classes=len(config.LABEL_DICT),
    )
    pl_model.eval()
    pl_model.to(device)  # FIX: was unconditional .cuda()

    token_reference = TokenReferenceBase(reference_token_idx=tokenizer.pad_token_id)
    lig = LayerIntegratedGradients(pl_model, pl_model.encoder.get_input_embeddings())
    return tokenizer, pl_model, lig, token_reference


def predict_dataframe(target_df, tokenizer, pl_model, device):
    """Add 'probs' (max softmax) and 'pred' (argmax) columns for every workbook row."""
    dl = DataLoader(
        SupervisedTextDataset(target_df, tokenizer, text_col=config.PREP_COL,
                              target_col=None, max_length=config.MAX_LENGTH),
        batch_size=config.BATCH_SIZE, shuffle=False,
    )
    pl_model.eval()
    logits = []
    with torch.no_grad():
        for b in dl:
            logits.append(pl_model(**{k: v.to(device) for k, v in b.items()}))
    all_logits = torch.cat(logits, dim=0)
    # shuffle=False keeps row order aligned with target_df, so positional assignment is safe.
    target_df["probs"] = all_logits.softmax(1).max(1).values.cpu().numpy()
    target_df["pred"] = all_logits.argmax(1).cpu().numpy()
    return target_df


# --- HTML rendering ----------------------------------------------------------
def _write_html(vis, out_path):
    """Inject a Times New Roman style block and write the captum HTML to disk."""
    custom_font = "font-family: 'Times New Roman', Times, serif;"
    style = f"<style>body, span, td, th, div, .attribution-text {{ {custom_font} }}</style>"
    html_data = vis.data
    if "<head>" in html_data:
        html_data = html_data.replace("<head>", "<head>" + style)
    else:
        html_data = style + html_data
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # FIX: notebook cell 5 built html_data but then wrote the unstyled `_.data`; cell 6 wrote
    # the styled string. Write the styled string (the clear intent) in both cases.
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_data)


# --- attribution visualizations ----------------------------------------------
def visualize_class(target, c, target_df, tokenizer, pl_model, lig, token_reference,
                    out_dir, n_steps):
    """IG attributions for the 3 lowest- and 3 highest-confidence gold-`c` examples."""
    target_class_df = target_df.query(f"{GOLD_COL} == '{c}'")
    if len(target_class_df) == 0:
        print(f"[skip] no gold '{c}' rows for target {target}")
        return None

    sorted_idx = target_class_df.probs.argsort()
    low_idx = sorted_idx[:3]
    high_idx = sorted_idx[-3:]
    idx = low_idx.tolist() + high_idx.tolist()
    idx = idx[::-1]
    vis_case = target_class_df.iloc[idx]

    vis_data_records = []
    for _n, samples in vis_case.iterrows():
        text = samples[config.PREP_COL]
        label = samples[GOLD_COL]

        tokens = tokenizer(text, return_tensors="pt")
        pred = forward_with_softmax(tokens, pl_model)
        pred_prob = pred[0, pred.argmax()].item()

        pl_model.zero_grad()
        reference_indices = token_reference.generate_reference(
            tokens["input_ids"].shape[1], device=pl_model.device).unsqueeze(0)
        attributions_ig, delta = lig.attribute(
            tokens["input_ids"].to(pl_model.device),
            reference_indices,
            n_steps=n_steps,
            return_convergence_delta=True,
            target=config.LABEL_DICT[label] if label in config.LABEL_DICT else 0,
        )

        attributions_ig = attributions_ig.detach().cpu()
        delta = float(delta.detach().cpu())
        tokens_for_vis = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0].cpu())

        print("pred: ", config.INV_LABEL_DICT[pred.argmax().item()],
              "(", "%.2f" % pred_prob, ")", ", delta: ", abs(delta))

        # FIX: notebook wrapped this in vis_data_records.append(add_...), but add_... already
        # appends internally and returns None; the extra None was filtered out later. Call it
        # directly (still followed by filter(None, ...) below, which is now a no-op safety net).
        add_attributions_to_visualizer(
            attributions_ig, tokens_for_vis, pred_prob,
            config.INV_LABEL_DICT[pred.argmax(1)[0].item()][:3].capitalize(),
            label[:3].capitalize(), delta, vis_data_records)

        del attributions_ig
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Visualize attributions based on Integrated Gradients")
    vis = visualization.visualize_text(list(filter(None, vis_data_records)))
    out_path = os.path.join(out_dir, f"{target}_{c}_attrs.html")
    _write_html(vis, out_path)
    print(f"wrote {out_path}")
    return vis_case


def visualize_single_multiclass(target, c, n, vis_case, tokenizer, pl_model, lig,
                                 token_reference, out_dir, n_steps):
    """Attribute a single example (row `n` of vis_case) against every class."""
    if vis_case is None or len(vis_case) <= n:
        print(f"[skip] no sample index {n} for target {target} class {c}")
        return

    samples = vis_case.reset_index(drop=True).iloc[n]
    # FIX: dropped unused `i = samples['index']` (workbook may lack an 'index' column).
    text = samples[config.PREP_COL]
    label = samples[GOLD_COL]

    tokens = tokenizer(text, return_tensors="pt")
    logits = predict_proba(tokens, pl_model)
    pred = forward_with_softmax(tokens, pl_model)
    print(logits, pred)

    vis_data_records = []
    for cls_idx, cls_prob in enumerate(pred[0]):
        pl_model.zero_grad()
        reference_indices = token_reference.generate_reference(
            tokens["input_ids"].shape[1], device=pl_model.device).unsqueeze(0)
        attributions_ig, delta = lig.attribute(
            tokens["input_ids"].to(pl_model.device),
            reference_indices,
            n_steps=n_steps,
            return_convergence_delta=True,
            target=cls_idx,
        )

        attributions_ig = attributions_ig.detach().cpu()
        delta = float(delta.detach().cpu())
        tokens_for_vis = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0].cpu())

        print("pred: ", config.INV_LABEL_DICT[cls_idx],
              "(", "%.2f" % float(cls_prob), ")", ", delta: ", abs(delta))

        add_attributions_to_visualizer(
            attributions_ig, tokens_for_vis, float(cls_prob),
            config.INV_LABEL_DICT[cls_idx][:3].capitalize(),
            label[:3].capitalize(), delta, vis_data_records)

        del attributions_ig
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Visualize attributions based on Integrated Gradients")
    vis = visualization.visualize_text(list(filter(None, vis_data_records)))
    out_path = os.path.join(out_dir, f"{target}_{c}_index_{n}_attrs.html")
    _write_html(vis, out_path)
    print(f"wrote {out_path}")


def process_target(target, classes, out_dir, n_steps, index):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    labeled = load_labeled_dfs(config.REVIEWER_GOLD_DIR)
    target_df = labeled[target].copy()

    tokenizer, pl_model, lig, token_reference = load_target_model(target, device)
    target_df = predict_dataframe(target_df, tokenizer, pl_model, device)

    for c in classes:
        vis_case = visualize_class(target, c, target_df, tokenizer, pl_model, lig,
                                   token_reference, out_dir, n_steps)
        if index is not None:
            visualize_single_multiclass(target, c, index, vis_case, tokenizer, pl_model,
                                         lig, token_reference, out_dir, n_steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=list(config.TARGETS), default=None,
                        help="Target to attribute; default runs both recur and metas.")
    # FIX: cell-3 loop had no `break`, so cells 4/5 only ever saw the LAST target (metas).
    # Correct behavior for the paper is to run the full IG pipeline for each target, which is
    # what this loop over `targets` does.
    parser.add_argument("--classes", nargs="+", choices=CLASS_SHEETS, default=["positive"],
                        help="Gold classes to visualize (notebook ran 'positive' only).")
    parser.add_argument("--n-steps", type=int, default=500,
                        help="Integrated-Gradients approximation steps (notebook: 500).")
    parser.add_argument("--index", type=int, default=0,
                        help="Row of vis_case for the per-class multi-class breakdown; "
                             "use -1 to skip that output.")
    parser.add_argument("--out-dir", default=None,
                        help="Output dir for the HTML files (default: LOGS_ROOT/captum).")
    args = parser.parse_args()

    set_seed()
    config.apply_font()  # FIX: replaces the hardcoded system-font selection

    out_dir = args.out_dir or str(config.LOGS_ROOT / "captum")
    targets = [args.target] if args.target else list(config.TARGETS)  # ['recur', 'metas']
    index = None if args.index < 0 else args.index

    for target in targets:
        print(f"=== target: {target} ===")
        process_target(target, args.classes, out_dir, args.n_steps, index)


if __name__ == "__main__":
    main()
