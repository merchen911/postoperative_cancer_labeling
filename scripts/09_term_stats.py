"""Interval ("term") statistics between surgery date and pathology-exam reception date.

For each cancer-outcome target (Metastasis / Recurrence) this collapses the per-report
predictions to one row per (patient, surgery-date, exam-date), forcing any group whose
binarized outcome is inconsistent to 'positive', then computes the signed delta in days
(검사접수일자 - 수술일자) as `Terms`. It writes the per-target tables (stats_terms.xlsx),
the outer-merged table across both targets (stats_terms_merged.xlsx), and delta-day
histograms (overlapping twin-axis + stacked log-scale) per target.

Inputs:  raw_format.xlsx under config.PREDICTIONS_DIR — the aggregated per-report
         workbook carrying 병원등록번호, 성별, 암종, 수술일자, 검사접수일자 and the
         Metastasis / Recurrence prediction columns.
Outputs: <outdir>/stats_terms.xlsx  (sheets Metastasis_all, Metastasis_Terms_positive,
                                     Recurrence_all, Recurrence_Terms_positive)
         <outdir>/stats_terms_merged.xlsx
         <outdir>/figure/{Recurrence,Metastasis}_histogram_overlapping.png
         <outdir>/figure/{Recurrence,Metastasis}_histogram_stacking.png
Dropped from the source notebook:
  - Cell 1: metas_df / recur_df loaded from a reviewer workbook — never referenced afterwards.
  - Cell 7 (merged_term groupby.size().value_counts()) and Cell 8 (merged_term.shape) —
    interactive diagnostics with no output artifact.
  - The check_label() helper — defined but never called.
  - Unused imports carried over from a sibling notebook: json, sklearn.metrics, torch,
    transformers (AutoModel/AutoTokenizer), src.utils.preprocessing.Preprocessor,
    umap.UMAP, scipy.stats.f, tqdm — none touch the term-stats logic.
  - The negative/uncertain pattern lists and cancer-merge constants — only needed by the
    dropped Preprocessor / exploratory cells.
  - Hard-coded system-font setup — replaced by config.apply_font().
"""
import os, sys, random, argparse
from glob import glob
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import numpy as np
import pandas as pd
import config

# recur/metas (config.TARGETS keys) -> the full column name used in the workbook.
# NOTE: candidate to hoist into config/src — the same map (target_name_dict) recurs in
# several sibling prediction notebooks.
TARGET_FULL = {"recur": "Recurrence", "metas": "Metastasis"}

# Group key that defines "one clinical event"; a group with >1 binarized outcome is a conflict.
GROUP_COLS = ["병원등록번호", "수술일자", "검사접수일자"]


from src.pipeline import set_seed  # hoisted shared helper


def resolve_input(predictions_dir: Path, explicit: str | None = None) -> Path:
    """Locate the aggregated workbook to read.

    FIX: the notebook used an UNSORTED `glob(...)[1]`, which is non-deterministic across
    filesystems. The file it actually consumed is 'raw_format.xlsx' — the only workbook in
    that folder carrying 검사접수일자 alongside the Metastasis/Recurrence prediction columns.
    Prefer it by name; fall back to a deterministic sorted glob if it is absent.
    """
    if explicit:
        return Path(explicit)
    preferred = predictions_dir / "raw_format.xlsx"
    if preferred.exists():
        return preferred
    xlsx = sorted(p for p in glob(os.path.join(str(predictions_dir), "*")) if p.endswith(".xlsx"))
    if not xlsx:
        raise FileNotFoundError(f"No .xlsx workbook found in {predictions_dir}")
    # FIX: deterministic fallback replacing the notebook's unsorted index [1].
    return Path(xlsx[1] if len(xlsx) > 1 else xlsx[0])


def load_data(path: Path) -> pd.DataFrame:
    """Read the aggregated workbook and add binarized outcome columns (positive vs. negative)."""
    load_all = pd.read_excel(path)
    # 'positive' stays positive; every other prediction (negative/uncertain/NaN) -> negative.
    load_all["bMetastasis"] = load_all["Metastasis"].apply(lambda x: "positive" if x == "positive" else "negative")
    load_all["bRecurrence"] = load_all["Recurrence"].apply(lambda x: "positive" if x == "positive" else "negative")
    return load_all


def compute_term_stats(load_all: pd.DataFrame, verbose: bool = True) -> dict:
    """Collapse to one row per clinical event and compute the signed day delta `Terms`.

    Returns {'Metastasis': df, 'Recurrence': df}. Both targets are always computed because
    the workbook outputs (stats_terms.xlsx / stats_terms_merged.xlsx) contain both.
    """
    stats_dict: dict = {}

    for target in ["Metastasis", "Recurrence"]:
        target_col = "b" + target

        # 1. Base data (drop duplicate rows).
        subset_df = load_all[["병원등록번호", "성별", "암종", "수술일자", "검사접수일자", target_col]].drop_duplicates()

        # 2. Conflict mask: True where a group holds >1 distinct binarized value.
        has_conflict = subset_df.groupby(GROUP_COLS)[target_col].transform("nunique") > 1

        # 3. Split into conflicting vs. consistent rows.
        conflict = subset_df[has_conflict].copy()        # mixed values (to be resolved)
        non_conflict = subset_df[~has_conflict].copy()   # consistent / singleton (kept as-is)

        # 4. Resolve every conflicting group to a single 'positive' row.
        conflict_processed = conflict.drop_duplicates(subset=GROUP_COLS, keep="first").copy()
        conflict_processed[target_col] = "positive"

        stats_dict[target] = pd.concat([non_conflict, conflict_processed])

        # 수술일자 / 검사접수일자: YYYYMMDD -> datetime.
        temp_df = stats_dict[target].copy()
        for col in ["수술일자", "검사접수일자"]:
            temp_df[col] = pd.to_datetime(temp_df[col], format="%Y%m%d", errors="coerce")

        # Terms = 검사접수일자 - 수술일자, in days. (Columns are already datetime here; the
        # redundant format= is ignored by pandas for datetime input — kept faithful to the notebook.)
        temp_df["Terms"] = (
            pd.to_datetime(temp_df["검사접수일자"], format="%Y%m%d", errors="coerce")
            - pd.to_datetime(temp_df["수술일자"], format="%Y%m%d", errors="coerce")
        ).dt.days

        temp_df = temp_df.sort_index()
        stats_dict[target] = temp_df

        if verbose:
            print(f"[{target}]")
            print(f"  - total rows: {len(subset_df)}")
            print(f"  - non-conflict (consistent): {len(non_conflict)}")
            print(f"  - conflict (mixed, pre-resolution): {len(conflict)}")
            print(f"  - stored (post-resolution): {len(stats_dict[target])}")
            print("-" * 30)

    return stats_dict


def write_stats_terms(stats_dict: dict, outdir: Path) -> Path:
    """Write per-target sheets: <target>_all and <target>_Terms_positive (Terms >= 0)."""
    excel_path = outdir / "stats_terms.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        for target in ["Metastasis", "Recurrence"]:
            temp_df = stats_dict[target].copy()
            pos_term_temp_df = temp_df.loc[temp_df["Terms"] >= 0].copy()

            # Render dates as yyyy-mm-dd strings for the spreadsheet.
            for col in ["수술일자", "검사접수일자"]:
                temp_df[col] = pd.to_datetime(temp_df[col], errors="coerce").dt.strftime("%Y-%m-%d")
                pos_term_temp_df[col] = pd.to_datetime(pos_term_temp_df[col], errors="coerce").dt.strftime("%Y-%m-%d")

            temp_df.to_excel(writer, sheet_name=f"{target}_all", index=False)
            pos_term_temp_df.to_excel(writer, sheet_name=f"{target}_Terms_positive", index=False)
    return excel_path


def write_merged(stats_dict: dict, outdir: Path) -> Path:
    """Outer-merge Metastasis and Recurrence on the shared event/demographic keys."""
    merged_term = pd.merge(
        stats_dict["Metastasis"],
        stats_dict["Recurrence"],
        how="outer",
        on=["병원등록번호", "성별", "암종", "수술일자", "검사접수일자"],
    )
    for col in ["수술일자", "검사접수일자"]:
        merged_term[col] = pd.to_datetime(merged_term[col], errors="coerce").dt.strftime("%Y-%m-%d")

    merged_path = outdir / "stats_terms_merged.xlsx"
    merged_term.to_excel(merged_path, index=False)
    return merged_path


def plot_histograms(stats_dict: dict, target: str, figdir: Path):
    """Delta-day histograms for one target (full name, e.g. 'Recurrence').

    Produces an overlapping twin-axis figure and a stacked log-scale figure, restricted to
    non-negative Terms (Terms >= 0).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    temp_df = stats_dict[target]
    pos_term_temp_df = temp_df.loc[temp_df["Terms"] >= 0]
    btarget = "b" + target
    vals_pos = pos_term_temp_df[pos_term_temp_df[btarget] == "positive"]["Terms"].dropna()
    vals_neg = pos_term_temp_df[pos_term_temp_df[btarget] == "negative"]["Terms"].dropna()

    # --- Overlapping histogram (twin y-axes) ---------------------------------
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()  # right axis: negative
    sns.histplot(vals_neg, ax=ax2, color="tab:blue", alpha=0.5, label="negative", binwidth=5)
    ax2.set_ylabel("Count (negative)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    # left axis: positive
    sns.histplot(vals_pos, ax=ax1, color="tab:red", alpha=0.5, label="positive", binwidth=5)
    ax1.set_ylabel("Count (positive)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_xlabel(r"$\Delta$day")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(f"{target} Histogram - Overlapped")
    fig.tight_layout()
    fig.savefig(os.path.join(str(figdir), f"{target}_histogram_overlapping.png"), dpi=400)
    plt.close(fig)  # FIX: close figures instead of relying on interactive show/GC.

    # --- Stacked histogram (log scale) ---------------------------------------
    stack_df = pd.concat(
        [
            pd.DataFrame({"Terms": vals_pos, "class": "positive"}),
            pd.DataFrame({"Terms": vals_neg, "class": "negative"}),
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    classes = ["positive", "negative"]
    colors = ["tab:red", "tab:blue"]
    sns.histplot(
        data=stack_df,
        x="Terms",
        hue="class",
        hue_order=classes,
        binwidth=5,
        multiple="stack",
        palette=dict(zip(classes, colors)),
        ax=ax,
        alpha=0.7,
        edgecolor="k",
    )

    ax.set_yscale("log")
    ax.set_ylabel("Count - log scale")
    ax.set_xlabel(r"$\Delta$day")

    # Build the legend by hand (stacked histplot legend handles are unreliable).
    # FIX: dropped the redundant `lw=10` kwarg (duplicate of `linewidth=10`, same value) so
    # this does not raise the alias-collision TypeError on modern matplotlib.
    ax.legend(
        handles=[
            plt.Line2D([], [], color="tab:red", label="positive", solid_capstyle="butt",
                       linewidth=10, alpha=0.7, marker=None, linestyle="-", dash_capstyle="butt"),
            plt.Line2D([], [], color="tab:blue", label="negative", solid_capstyle="butt",
                       linewidth=10, alpha=0.7, marker=None, linestyle="-", dash_capstyle="butt"),
        ],
        title="class",
        labels=classes,
        handlelength=1.2,
    )

    plt.title(f"{target} Histogram - Stacked log scale")
    fig.tight_layout()
    fig.savefig(os.path.join(str(figdir), f"{target}_histogram_stacking.png"), dpi=400)
    plt.close(fig)  # FIX: close figures instead of relying on interactive show/GC.


def main():
    parser = argparse.ArgumentParser(description="Surgery-to-exam interval (term) statistics.")
    parser.add_argument("--target", choices=list(config.TARGETS), default=None,
                        help="Target whose delta-day histograms to plot (default: both). "
                             "The stats_terms/*.xlsx outputs always cover both targets.")
    parser.add_argument("--input", default=None,
                        help="Explicit path to the aggregated workbook (overrides auto-resolution "
                             "under config.PREDICTIONS_DIR).")
    parser.add_argument("--outdir", default=None,
                        help="Output directory for workbooks/figures (default: config.PREDICTIONS_DIR).")
    args = parser.parse_args()

    set_seed()

    predictions_dir = config.PREDICTIONS_DIR
    outdir = Path(args.outdir) if args.outdir else predictions_dir
    figdir = outdir / "figure"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)

    config.apply_font()  # FIX: replaces the hard-coded system-font path.

    input_path = resolve_input(predictions_dir, explicit=args.input)
    print(f"Reading aggregated workbook: {input_path}")
    load_all = load_data(input_path)

    stats_dict = compute_term_stats(load_all)

    stats_path = write_stats_terms(stats_dict, outdir)
    print(f"Wrote {stats_path}")
    merged_path = write_merged(stats_dict, outdir)
    print(f"Wrote {merged_path}")

    targets = [args.target] if args.target else list(config.TARGETS)
    for t in targets:
        full = TARGET_FULL[t]
        plot_histograms(stats_dict, full, figdir)
        print(f"Wrote {full} histograms to {figdir}")


if __name__ == "__main__":
    main()
