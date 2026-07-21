"""Aggregate per-target model predictions into the master ``all.xlsx`` dataset and
build the cohort characteristics table ("Table description").

This stage has two parts that the source notebook ran in the *wrong* order (it read
the aggregated workbook near the top and only *built* it near the bottom). Here the
aggregation runs first, then the table is built from an explicitly-named input:

  1. ``aggregate_predictions`` — take the raw report corpus (``all-report.csv``),
     re-derive the rule/weak labels per target with the ``Preprocessor``, merge in the
     apply-stage prediction workbooks and the training-set gold labels, write the
     ``Metastasis`` / ``Recurrence`` columns, and save the master ``all.xlsx``.
  2. ``build_table_description`` — read the cancer-type-enriched apply-stage workbook
     (``raw_format.xlsx``) and produce the patient/report cohort summary
     ``Table description v03.xlsx`` (counts + per-class breakdown by sex, age band and
     cancer type).

This stage inherently processes BOTH targets jointly (they become two columns of one
master file), so — unlike the modeling stages — it deliberately does not take a
``--target`` argument; ``--stage`` selects which part(s) to run instead.

Ported from the excel_post_processing notebook.
Inputs:
    - config.RAW_REPORTS_DIR/all-report.csv             (raw report corpus)
    - config.PREDICTIONS_DIR/*/files/*.xlsx             (full-corpus prediction workbooks:
                                                         2 metas + 1 recur, sorted)
    - config.SPLITS_DIR/*train*.xlsx                    (metas/recur training gold labels)
    - config.PREDICTIONS_DIR/raw_format.xlsx           (cancer-enriched apply output; Table 1 input)
    - config.REPO_ROOT/cancer_dict.json                 (cancer code -> English name)
Outputs:
    - config.PREDICTIONS_DIR/all.xlsx                   (master aggregated dataset)
    - config.PREDICTIONS_DIR/Table description v03.xlsx
    - <each prediction workbook dir>/postprocessed-for-review.xlsx  (optional review copies)
Dropped from the notebook:
    - display-only cells (load_path, column_dfs, full_dfs MultiIndex preview),
    - exploratory/diagnostic cells (df.head(), groupby().count(), 검사코드.value_counts(),
      unique_counts per 병원등록번호, patient-count `second_columns`),
    - the cell referencing the undefined `first_columns` variable,
    - the bare Windows-path string literal cell,
    - the duplicated `column_dfs` cell (kept once),
    - unused names: itertools.product/target_combs, total_numb, the `cnacer_merged_en_dict`
      / cancer_merge_keys / merged_cancer_label block, the meta._file_path hasattr hack.
"""
import os, sys, random, argparse, json
from glob import glob
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import numpy as np
import pandas as pd
import config


from src.pipeline import set_seed, build_age_bins, age_group  # hoisted shared helpers


# --- small helpers (verbatim from the notebook) ------------------------------
def check_label(series: pd.Series, label: str) -> pd.DataFrame:
    """Turn a text Series into a DataFrame with a constant ``label`` column."""
    df = series.to_frame()
    df['label'] = label
    return df


def shuffle_list(list_data, random_seed: int = config.SEED):
    """Deterministically shuffle a copy of ``list_data`` (5 passes) with a local RNG."""
    random_gen = random.Random(random_seed)
    shuffled_list = list_data.copy()
    for _ in range(5):
        random_gen.shuffle(shuffled_list)
    return shuffled_list


def save_postprocessed_excel(df: pd.DataFrame, original_path) -> None:
    """Write a reviewer workbook (all rows + one sheet per predicted class) next to the source."""
    label_order = ['positive', 'uncertain', 'negative']  # tab order
    writer_path = Path(original_path).with_name('postprocessed-for-review.xlsx')
    df = df[['index', 'raw_text', 'prep_text', 'rule_label', 'prediction']]
    with pd.ExcelWriter(writer_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='all')
        for label in label_order:
            df_label = df[df['prediction'] == label]
            df_label.to_excel(writer, index=False, sheet_name=label)
    print(f"Saved: {writer_path}")


def rename_prediction_to_label(df: pd.DataFrame) -> pd.DataFrame:
    """Rename a 'prediction' column to 'label' if present; otherwise return unchanged."""
    if 'prediction' in df.columns:
        return df.rename(columns={'prediction': 'label'})
    return df


def merge_dfs(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame) -> pd.DataFrame:
    """Attach model predictions + training gold labels to the rule-labeled corpus by prep_text."""
    df2 = rename_prediction_to_label(df2)
    df23 = pd.concat([df2, df3.drop(columns='R_text')])
    return pd.merge(
        df1.drop(columns=['label']).set_axis([config.PREP_COL], axis=1),
        df23[[config.PREP_COL, 'rule_label', 'label']],
        on=config.PREP_COL,
        how='left',
    )


# --- stage 1: build the master all.xlsx --------------------------------------
def aggregate_predictions(save_review: bool = True) -> pd.DataFrame:
    """Re-derive rule labels, merge predictions + gold labels, write the master all.xlsx."""
    # heavy-ish import kept local so `--stage table` need not pull it in
    from src.utils.preprocessing import Preprocessor

    # FIX: select the raw corpus by explicit name instead of glob(...)[0] (order-dependent).
    raw_csv = config.RAW_REPORTS_DIR / 'all-report.csv'
    df = pd.read_csv(raw_csv)

    text_df = df[config.TEXT_COL].dropna()
    preprocessor = Preprocessor(
        df=text_df,
        negative_patterns=config.NEGATIVE_PATTERNS,      # FIX: from config, not inline literals
        uncertain_patterns=config.UNCERTAIN_PATTERNS,
        abbrev_path=os.path.join(config.REPO_ROOT, 'src', 'utils', 'abbrevations.json'),
    )

    # Rule/weak labels per target (positive/negative/uncertain buckets -> one labeled frame).
    meta_df = pd.concat(
        [check_label(d, k) for k, d in preprocessor.target_filtering('metas').items()]
    ).sort_index()
    recur_df = pd.concat(
        [check_label(d, k) for k, d in preprocessor.target_filtering('recur').items()]
    ).sort_index()

    # Full-corpus prediction workbooks (read from config.PREDICTIONS_DIR).
    # FIX: forward-slash path via config.PREDICTIONS_DIR (notebook used a hard-coded '..\\..\\logs'
    # Windows-backslash glob) and sorted() for a stable order. The '*/files/' nesting is retained
    # so the top-level all.xlsx / raw_format.xlsx outputs never match this pattern. The notebook
    # expects exactly three: the two metas workbooks sort before the recur one, and merge_dfs uses
    # the SECOND metas workbook (index 1) plus the recur workbook.
    pred_paths = sorted(glob(os.path.join(str(config.PREDICTIONS_DIR), '*', 'files', '*.xlsx')))
    meta1_path, meta2_path, recur_path = pred_paths
    meta1 = pd.read_excel(meta1_path)
    meta2 = pd.read_excel(meta2_path)
    recur = pd.read_excel(recur_path)

    # Training-set gold labels.
    # FIX: read from config.SPLITS_DIR. sorted() -> metas before recur.
    train_paths = sorted(glob(os.path.join(str(config.SPLITS_DIR), '*train*')))
    meta_train_df = pd.read_excel(train_paths[0])
    recur_train_df = pd.read_excel(train_paths[1])

    # Optional reviewer workbooks (shuffled row order for blind review).
    if save_review:
        metas_index = shuffle_list(list(meta1.index))
        recur_index = shuffle_list(list(recur.index))
        # FIX: dropped the meta._file_path hasattr(...) workaround (DataFrames never carry it).
        save_postprocessed_excel(meta1.iloc[metas_index], meta1_path)
        save_postprocessed_excel(meta2.iloc[metas_index], meta2_path)
        save_postprocessed_excel(recur.iloc[recur_index], recur_path)

    # Merge predictions (meta2 / recur) + gold labels onto the rule-labeled corpus.
    meta_merged = merge_dfs(meta_df, meta2, meta_train_df).fillna('negative')
    recur_merged = merge_dfs(recur_df, recur, recur_train_df).fillna('negative')

    df.loc[meta_merged.index, 'Metastasis'] = meta_merged.label
    df.loc[recur_merged.index, 'Recurrence'] = recur_merged.label
    for col in ['Metastasis', 'Recurrence']:
        df.loc[:, col] = df.loc[:, col].apply(lambda x: 'negative' if x == '' else x)

    out_dir = config.PREDICTIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'all.xlsx'
    df.to_excel(out_path, index=False)
    print(f"Wrote master dataset: {out_path}  ({df.shape[0]} rows)")
    return df


# --- stage 2: cohort characteristics table -----------------------------------
def build_table_description() -> pd.DataFrame:
    """Build the sex/age/cancer cohort summary (patient & report counts + per-class breakdown)."""
    # FIX: read the cancer-enriched apply output by explicit name. The notebook read
    # glob(<dir>/*.xlsx)[1], whose target flips once all.xlsx is written into the
    # same directory (circular top-reads/bottom-writes bug). raw_format.xlsx is the only
    # workbook here carrying the '암종' cancer-type column this table groups by.
    src = config.PREDICTIONS_DIR / 'raw_format.xlsx'
    merged_all = pd.read_excel(src)

    cancer_dict_path = os.path.join(config.REPO_ROOT, 'cancer_dict.json')
    with open(cancer_dict_path, 'r', encoding='utf-8') as fh:
        cancer_dict = json.load(fh)
    cancer_en_dict = {int(k): v['name_en'] for k, v in cancer_dict.items()}
    cancer_order = list(cancer_en_dict.values()) + ['Missing']

    age_bins, age_dict = build_age_bins()

    # One-hot encode the two prediction columns -> per-class count columns.
    metastasis_ohe = pd.get_dummies(merged_all['Metastasis'], prefix='Metastasis')
    recurrence_ohe = pd.get_dummies(merged_all['Recurrence'], prefix='Recurrence')
    merged_all = pd.concat([merged_all, metastasis_ohe, recurrence_ohe], axis=1)

    target_names = ['Recurrence', 'Metastasis']
    target_classes = ['negative', 'uncertain', 'positive']

    # A per-report unique key (patient + surgery date + row) to count "reports" distinctly.
    merged_all['rmc'] = (
        merged_all['병원등록번호'].astype(str) + '_'
        + merged_all['수술일자'].astype(str) + '_'
        + merged_all.index.astype(str)
    )
    merged_all['검사나이구간'] = merged_all['검사나이'].apply(lambda a: age_group(a, age_bins))

    # --- Information block: unique patient (병원등록번호) & report (rmc) counts + shares ---
    column_dfs = []
    for row in ['성별', '검사나이구간', '암종']:
        temp_dfs = []
        temp_numb_dfs = []
        for col in ['병원등록번호', 'rmc']:
            temp_numb = merged_all.groupby(row)[col].nunique()
            str_temp_numb = (
                temp_numb.apply(lambda x: f"{x:,}")
                + ' (' + (temp_numb / temp_numb.sum() * 100).round(1).astype(str) + '%)'
            )
            if row == '검사나이구간':
                str_temp_numb = str_temp_numb.sort_index()
                str_temp_numb.index = [age_dict[i] for i in str_temp_numb.index]
            elif row == '암종':
                str_temp_numb = str_temp_numb.reindex(cancer_order)
            temp_dfs.append(str_temp_numb)
            temp_numb_dfs.append(temp_numb)
        temp_dfs = pd.concat(temp_dfs, axis=1)
        temp_numb_dfs = pd.concat(temp_numb_dfs, axis=1)
        temp_dfs.loc['Sum'] = temp_numb_dfs.sum().apply(lambda x: f"{x:,}") + ' (100.0%)'
        column_dfs.append(temp_dfs)
    column_dfs = pd.concat(column_dfs, axis=0)

    # --- Per-target class breakdown (Recurrence / Metastasis x neg/unc/pos) ---
    full_dfs = []
    for row in ['성별', '검사나이구간', '암종']:
        category_dfs = []
        for col1 in target_names:
            target_temp_dfs = []
            for col2 in target_classes:
                temp_numb = merged_all.groupby(row)[col1 + '_' + col2].sum()
                target_temp_dfs.append(temp_numb)
            target_temp_dfs = pd.concat(target_temp_dfs, axis=1)
            target_temp_dfs.loc['Sum'] = target_temp_dfs.sum()
            # FIX: dropped the unused `total_numb` groupby; the percentage denominator is the
            # per-row target total (target_temp_dfs.sum(1)), matching the notebook's live code.
            percent = (target_temp_dfs / target_temp_dfs.sum(1).values.reshape(-1, 1) * 100).round(1)
            str_temp_numb = target_temp_dfs.map(lambda x: f"{x:,}") + ' (' + percent.astype(str) + '%)'
            if row == '검사나이구간':
                str_temp_numb.index = [age_dict[i] if i in age_dict else i for i in str_temp_numb.index]
            elif row == '암종':
                str_temp_numb = str_temp_numb.reindex(cancer_order + ['Sum'])
            category_dfs.append(str_temp_numb)
        category_dfs = pd.concat(category_dfs, axis=1)
        full_dfs.append(category_dfs)
    full_dfs = pd.concat(full_dfs, axis=0)

    # --- Assemble the final labeled table and save ---
    MI = pd.MultiIndex.from_tuples(
        [('Sex', v) for v in ['F', 'M'] + ['Sum']]
        + [('Age', v) for v in list(age_dict.values()) + ['Sum']]
        + [('Cancer', v) for v in list(cancer_en_dict.values()) + ['Missing', 'Sum']],
        names=['Category', 'Value'],
    )
    save_df = pd.concat([
        column_dfs.set_axis(pd.MultiIndex.from_product([['Information'], ['Patients', 'Reports']]), axis=1),
        full_dfs.set_axis(
            pd.MultiIndex.from_product([target_names, [j[:3].capitalize() + '.' for j in target_classes]]),
            axis=1,
        ),
    ], axis=1).set_axis(MI, axis=0)

    out_dir = config.PREDICTIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'Table description v03.xlsx'   # preserve exact output filename
    save_df.to_excel(out_path)
    print(f"Wrote cohort table: {out_path}")
    return save_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['aggregate', 'table', 'all'], default='all',
                        help="'aggregate' builds all.xlsx; 'table' builds the cohort table; 'all' both")
    parser.add_argument('--no-review', action='store_true',
                        help='skip writing the postprocessed-for-review reviewer workbooks')
    args = parser.parse_args()

    set_seed()

    # Aggregation must run before the table (fixes the notebook's read-then-build ordering).
    if args.stage in ('aggregate', 'all'):
        aggregate_predictions(save_review=not args.no_review)
    if args.stage in ('table', 'all'):
        build_table_description()


if __name__ == '__main__':
    main()
