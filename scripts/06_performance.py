"""Final-performance statistics and figures for the pathology-conclusion classifier.

This stage takes the reviewer-verified prediction workbooks (gold column ``실제`` vs the
model ``prediction``) together with the aggregated corpus (for demographics: sex, age
band, cancer type) and produces the paper's ``rev3_*`` category figures, the per-cancer
/ per-age confusion matrices, and the Human-vs-LM agreement tables. It is inherently a
recur-vs-metas comparison: every category figure is a 2-row grid (Recurrence on top,
Metastasis on the bottom), so the shared figures require BOTH targets. ``--target`` is
still exposed (per repo convention) and restricts the per-target artifacts (confusion
matrices, Human-vs-LM rows, the overall metric table); the combined 2-row category
figures are emitted only when both targets are selected (the default).

Ported from the final-performance analysis notebook.
Inputs:
  - config.REVIEWER_GOLD_DIR/*.xlsx    reviewer workbooks, sheets negative/uncertain/positive
                                       (columns 실제, prediction, index) for metas & recur
  - config.PREDICTIONS_DIR/*.xlsx      full-corpus prediction workbook with demographics
                                       (병원등록번호, 수술일자, 검사나이, 성별, ...)
  - config.CANCER_TYPE_MAP             cancer-type map (암번호 per record)
  - <repo>/cancer_dict.json            cancer-code -> English/Korean name map
Outputs (under --outdir, default <logs>/06_performance):
  Figures : rev3_category_histogram.png, rev3_category_metric_vis_class-represent-std.png,
            rev3_category_accuracy_via_class.png, rev3_category_bunch_histogram.png,
            rev3_category_bunch_class_metric.png, rev3_category_bunch_binary_metric.png,
            "<Cancer/Age> - <Class>.png" confusion matrices, human_vs_lm_bar.png
  Tables  : result_overall.csv, pivot_binary_by_cancer.csv,
            human_vs_lm_long.csv, human_vs_lm_wide.csv
Dropped from the notebook:
  - cell 19 (``dup_index`` diagnostic — computed but never used; the shared-index logic
    is recomputed inline in the Human-vs-LM table cell).
  - the bare trailing display cells (``result_df``, the pivot ``pd.concat(...)``,
    ``front_df, tab_df.round(4)``, ``front_df``) — their values are written to CSV instead.
Notable fixes (search '# FIX:'):
  - hardcoded system-font path replaced by config.apply_font().
  - all '../../..' relative data paths replaced by config.* paths.
  - deprecated DataFrame._append replaced by building rows once + pd.DataFrame.
  - chained-slice ``.fillna(..., inplace=True)`` on ``iloc[:, k]`` views rewritten as
    explicit column assignment.
  - fragile ``DataFrame.query()`` on Korean/merged-label columns replaced by boolean masks.
  - the ``recur negative .loc[:99, '실제'] = 'negative'`` positional patch is preserved but
    clearly commented.
  - glob-order dependence removed: files selected by name keyword, globs sorted.
"""
import os, sys, random, argparse, json, re
from glob import glob
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import numpy as np
import pandas as pd
import config


from src.pipeline import set_seed, build_age_bins, age_group  # hoisted shared helpers


# --- Constants (preserve the notebook's exact labels / palette) --------------
TARGET_CLASS = ['negative', 'uncertain', 'positive']
CLASS_PALETTE = {'negative': 'tab:blue', 'uncertain': 'tab:green', 'positive': 'tab:red'}
CANCER_MERGE_KEYS = ['Neurologic', 'Hematologic', 'Head & Neck', 'Endocrine']
MERGED_CANCER_LABEL = 'Neurologic & Hematologic & HeadNeck & Endocrine'
MISSING_CANCER_CODE = 99
COL_GOLD = config.GOLD_COL
COL_PRED = 'prediction'
COL_CANCER = '암번호'
COL_CANCER_MERGED = '암번호_머지'


# --- Scoring -----------------------------------------------------------------
def score2dict(yt, yp, n_samples=None, binary_flag=False):
    """Accuracy + F1 (+ weighted F1) for a set of true/pred labels."""
    from sklearn.metrics import accuracy_score, f1_score
    accuracy = accuracy_score(yt, yp, sample_weight=n_samples)
    if binary_flag:
        f1 = f1_score(yt, yp, average='binary', sample_weight=n_samples)
    else:
        f1 = f1_score(yt, yp, average='macro', sample_weight=n_samples)
    weighted_f1 = f1_score(yt, yp, average='weighted', sample_weight=n_samples)
    return dict(accuracy=accuracy, f1=f1, weighted_f1=weighted_f1)


# --- Age binning (age_bins = [0, 40, 50, 60, 70, 80]) ------------------------
def make_age_group_fn(age_bins):
    """Closure form of the shared age_group binning (06 applies it column-wise)."""
    return lambda age: age_group(age, age_bins)


# --- Cancer-code -> English-name map -----------------------------------------
def load_cancer_en_dict(cancer_dict_path):
    with open(cancer_dict_path, 'r', encoding='utf-8') as f:
        cancer_dict = json.load(f)
    cancer_en_dict = {int(k): v['name_en'] for k, v in cancer_dict.items()}
    cancer_en_dict[MISSING_CANCER_CODE] = 'Missing'
    return cancer_en_dict


def cancer_order_unmerged(cancer_en_dict):
    """Full cancer order (dict insertion order): used for the un-merged figures."""
    return [cancer_en_dict[k] for k in cancer_en_dict.keys()]


def cancer_order_merged(cancer_en_dict):
    """Merge Neuro/Hemato/H&N/Endocrine into one bar, placed before Others/Missing."""
    order = [cancer_en_dict[k] for k in cancer_en_dict.keys()
             if cancer_en_dict[k] not in CANCER_MERGE_KEYS]
    order.insert(-2, MERGED_CANCER_LABEL)   # before the final two (Others, Missing)
    return order


# --- File selection helpers --------------------------------------------------
def _pick_by_keyword(paths, keyword):
    for p in paths:
        if keyword in os.path.basename(p).lower():
            return p
    return None


# --- Data loading ------------------------------------------------------------
def load_labeled_frames(reviewer_dir):
    """Load the reviewer workbooks into {'recur': df, 'metas': df} (rows = all sheets).

    Each workbook has sheets negative/uncertain/positive; we concat all three and drop
    rows with missing values, matching the notebook.
    """
    xlsx = sorted(glob(os.path.join(str(reviewer_dir), '*.xlsx')))  # FIX: sorted glob
    # FIX: select by filename keyword instead of relying on sorted() positional order.
    metas_path = _pick_by_keyword(xlsx, 'meta')
    recur_path = _pick_by_keyword(xlsx, 'recur')
    if metas_path is None or recur_path is None:
        # Fallback preserves the notebook's original assumption: sorted -> [metas, recur].
        metas_path, recur_path = xlsx[0], xlsx[1]

    metas_book = pd.read_excel(metas_path, sheet_name=None)
    recur_book = pd.read_excel(recur_path, sheet_name=None)

    metas_labeled_df = pd.concat([metas_book[c] for c in TARGET_CLASS]).dropna()

    # FIX (magic patch, preserved): the first 100 rows of the recur 'negative' sheet had
    # no gold label filled in, so the notebook stamps them as 'negative' before concat.
    # .loc[:99] is label-based and inclusive -> rows 0..99 (100 rows).
    recur_book['negative'].loc[:99, COL_GOLD] = 'negative'
    recur_labeled_df = pd.concat([recur_book[c] for c in TARGET_CLASS]).dropna()

    return {'recur': recur_labeled_df, 'metas': metas_labeled_df}


def build_merged_all(cancer_en_dict, age_group_fn, age_dict):
    """Full-corpus predictions joined with cancer type; adds positional 'index', 나이구간, 암번호."""
    pred_dir = str(config.PREDICTIONS_DIR)
    pred_files = sorted(i for i in glob(os.path.join(pred_dir, '*')) if i.endswith('.xlsx'))
    if not pred_files:
        raise FileNotFoundError(f'No prediction .xlsx found under {pred_dir}')
    load_all = pd.read_excel(pred_files[0])  # full-corpus prediction workbook (with demographics)

    cancer_type_df = pd.read_excel(str(config.CANCER_TYPE_MAP))
    # FIX: chained-slice ``.iloc[:, k].fillna(..., inplace=True)`` (a view; no-op / deprecated)
    #      rewritten as explicit positional-column assignment. Columns 2 and 3 are, in order,
    #      a free-text field and the cancer code (암번호).
    cancer_type_df.iloc[:, 2] = cancer_type_df.iloc[:, 2].fillna(' ')
    cancer_type_df.iloc[:, 3] = cancer_type_df.iloc[:, 3].fillna(MISSING_CANCER_CODE)
    cancer_type_df[COL_CANCER] = cancer_type_df[COL_CANCER].astype(int)

    merged_all = pd.merge(load_all, cancer_type_df, on=['병원등록번호', '수술일자'], how='left')
    merged_all['index'] = merged_all.index          # positional key referenced by the reviewer workbooks
    merged_all['나이구간'] = merged_all['검사나이'].apply(age_group_fn)
    return merged_all


def build_target_merged_frames(labeled_frames, merged_all, cancer_en_dict, age_dict):
    """Attach demographics to each labeled frame and add the merged-cancer column.

    Returns {'recur': df, 'metas': df} with columns 실제, prediction, index, 성별,
    나이구간 (string band), 암번호 (English name), 암번호_머지 (merged English name).
    """
    demo_cols = ['index', '성별', '나이구간', COL_CANCER]
    frames = {}
    for target, labeled in labeled_frames.items():
        df = labeled.merge(merged_all[demo_cols], on='index', how='left')
        df['나이구간'] = df['나이구간'].apply(lambda x: age_dict[x])
        df[COL_CANCER] = df[COL_CANCER].apply(lambda x: cancer_en_dict[x])
        # Effective equivalent of the notebook's apply_merged_cancer(): the 암번호 column
        # already holds English names, so we only fold the four merge keys into one label.
        df[COL_CANCER_MERGED] = df[COL_CANCER].apply(
            lambda x: MERGED_CANCER_LABEL if x in CANCER_MERGE_KEYS else x)
        frames[target] = df
    return frames


# --- Plot helpers ------------------------------------------------------------
def _wrap_xticklabels(ax, n, merged, amp_repl='\n&'):
    """Re-wrap long '&'-joined x tick labels; rotate the cancer-type axis (n == 2)."""
    labels = [t.get_text() for t in ax.get_xticklabels()]
    new_labels = []
    for l in labels:
        if ('&' in l) and (len(l) >= 20):
            if merged:
                new_l = re.sub(r'\s*&\s*', amp_repl, l)
            else:
                idx = l.index('&')
                new_l = l[:idx].rstrip() + '\n&' + l[idx + 1:]
            new_labels.append(new_l)
        elif merged and ('Neuro/Hematologic/H&N/Endocrine' in l):
            new_labels.append('Neuro/\nHemato/\nH&N/\nEndocrine')
        else:
            new_labels.append(l)
    ax.set_xticklabels(new_labels, rotation=45 if n == 2 else 0)


def _reorder_handles(handles, labels, legend_labels):
    handles_dict = {label.lower(): h for h, label in zip(handles, labels)}
    return [handles_dict.get(lbl.lower(), h) for lbl, h in zip(legend_labels, handles)]


def _axis_order(plot_df, c, cancer_col, cancer_order, age_group_order):
    if c == '나이구간':
        return age_group_order
    if c == cancer_col:
        return cancer_order
    return sorted(plot_df[c].unique())   # 성별 ascending


# --- Figure: count histograms (notebook cells 10 & 13) -----------------------
def plot_count_histogram(frames, cancer_col, cancer_order, age_group_order,
                         figdir, filename, legend_bbox, merged, amp_repl='\n&'):
    from matplotlib import pyplot as plt
    import seaborn as sns

    group_keys = ['성별', '나이구간', cancer_col]
    ref = frames['recur']
    if merged:
        width_ratios = [ref['성별'].nunique(), ref['나이구간'].nunique(), len(cancer_order)]
    else:
        width_ratios = [ref[g].nunique() for g in group_keys]

    plt.rcParams['font.size'] = 15
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), gridspec_kw={'width_ratios': width_ratios})

    for n_row, col_axes in enumerate(axes):
        plot_df = frames['recur'] if n_row == 0 else frames['metas']
        handles, labels = [], []
        for n, (c, en_c) in enumerate(zip(group_keys, ['Sex', 'Age', 'Cancer Type'])):
            order = _axis_order(plot_df, c, cancer_col, cancer_order, age_group_order)
            for g, sdf in plot_df.groupby(c):
                sdf_counts = sdf.groupby([c, COL_GOLD]).size().reset_index(name='n_samples')
                if c == '나이구간':
                    sdf_counts[c] = sdf_counts[c].astype(
                        pd.CategoricalDtype(categories=age_group_order, ordered=True))
                elif c == cancer_col:
                    sdf_counts[c] = sdf_counts[c].astype(
                        pd.CategoricalDtype(categories=cancer_order, ordered=True))
                # Reindex to the full (category x class) grid so absent bins render as 0.
                mi = pd.MultiIndex.from_product([order, TARGET_CLASS], names=[c, COL_GOLD])
                sdf_counts_full = sdf_counts.set_index([c, COL_GOLD]).reindex(
                    mi, fill_value=0).reset_index()
                sns.barplot(data=sdf_counts_full, x=c, y='n_samples', hue=COL_GOLD,
                            ax=col_axes[n], palette=CLASS_PALETTE, orient='v',
                            edgecolor='black', order=order, hue_order=TARGET_CLASS)
                handles, labels = col_axes[n].get_legend_handles_labels()
                col_axes[n].legend().remove()
                col_axes[n].set_xlabel(''); col_axes[n].set_ylabel('')
                if n_row == 0:
                    col_axes[n].set_title(en_c); col_axes[n].set_xticklabels([])
                else:
                    col_axes[n].set_title('')
                if n == 0:
                    col_axes[n].set_ylabel(['Recurrence', 'Metastasis'][n_row])
        for n, ax in enumerate(col_axes):
            _wrap_xticklabels(ax, n, merged, amp_repl)
        if n_row == 0:
            legend_labels = ['Negative', 'Uncertain', 'Positive']
            fig.legend(handles=_reorder_handles(handles, labels, legend_labels),
                       labels=legend_labels, loc='lower left', bbox_to_anchor=legend_bbox,
                       title='Class', ncol=3)

    fig.tight_layout()
    fig.savefig(os.path.join(figdir, filename), dpi=400)
    plt.close(fig)


# --- Figure: three metrics per class (notebook cell 11) ----------------------
def plot_metric_per_class(frames, cancer_col, cancer_order, age_group_order,
                          figdir, filename, legend_bbox):
    from matplotlib import pyplot as plt
    import seaborn as sns

    group_keys = ['성별', '나이구간', cancer_col]
    ref = frames['recur']
    width_ratios = [ref[g].nunique() for g in group_keys]

    plt.rcParams['font.size'] = 15
    fig, axes = plt.subplots(2, 3, figsize=(15, 8),
                             gridspec_kw={'width_ratios': width_ratios}, sharey=True)

    for n_row, col_axes in enumerate(axes):
        plot_df = frames['recur'] if n_row == 0 else frames['metas']
        handles, labels = [], []
        for n, (ax, c, en_c) in enumerate(zip(col_axes, group_keys, ['Sex', 'Age', 'Cancer Type'])):
            rows = []
            for g, sdf in plot_df.groupby(c):
                for cls, cls_sdf in sdf.groupby(COL_GOLD):
                    for k, v in score2dict(cls_sdf[COL_GOLD], cls_sdf[COL_PRED]).items():
                        rows.append({c: g, COL_GOLD: cls, 'metric': k, 'value': v})
            sdf_counts = pd.DataFrame(rows)
            if c == '나이구간':
                sdf_counts[c] = sdf_counts[c].astype(
                    pd.CategoricalDtype(categories=age_group_order, ordered=True))
            elif c == cancer_col:
                sdf_counts[c] = sdf_counts[c].astype(
                    pd.CategoricalDtype(categories=cancer_order, ordered=True))
            sns.barplot(data=sdf_counts, x=c, y='value', hue='metric', ax=ax,
                        palette='husl', orient='v', edgecolor='black',
                        hue_order=['accuracy', 'weighted_f1', 'f1'])
            handles, labels = ax.get_legend_handles_labels()
            ax.legend().remove(); ax.set_xlabel(''); ax.set_ylabel('')
            if n_row == 0:
                col_axes[n].set_title(en_c); col_axes[n].set_xticklabels([])
            else:
                col_axes[n].set_title('')
            if n == 0:
                col_axes[n].set_ylabel(['Recurrence', 'Metastasis'][n_row])
        for n, ax in enumerate(col_axes):
            _wrap_xticklabels(ax, n, merged=False)
        if n_row == 0:
            legend_labels = ['Accuracy', 'Weighted F1', 'Macro-F1']
            fig.legend(handles=_reorder_handles(handles, labels, legend_labels),
                       labels=legend_labels, loc='lower left', bbox_to_anchor=legend_bbox,
                       title='Metric', ncol=3)

    fig.suptitle('')
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, filename), dpi=400)
    plt.close(fig)


# --- Figure: accuracy per class (notebook cells 12 & 14) ---------------------
def plot_accuracy_per_class(frames, cancer_col, cancer_order, age_group_order,
                            figdir, filename, legend_bbox, merged, apply_cat_all,
                            amp_repl='\n&'):
    from matplotlib import pyplot as plt
    import seaborn as sns

    group_keys = ['성별', '나이구간', cancer_col]
    ref = frames['recur']
    # Both notebook cells size the width by the *un-merged* cancer nunique.
    width_ratios = [ref['성별'].nunique(), ref['나이구간'].nunique(), ref[COL_CANCER].nunique()]

    plt.rcParams['font.size'] = 15
    fig, axes = plt.subplots(2, 3, figsize=(15, 8),
                             gridspec_kw={'width_ratios': width_ratios}, sharey=True)

    for n_row, col_axes in enumerate(axes):
        plot_df = frames['recur'] if n_row == 0 else frames['metas']
        handles, labels = [], []
        for n, (ax, c, en_c) in enumerate(zip(col_axes, group_keys, ['Sex', 'Age', 'Cancer Type'])):
            order = _axis_order(plot_df, c, cancer_col, cancer_order, age_group_order)
            rows = []
            for g, sdf in plot_df.groupby(c):
                for cls, cls_sdf in sdf.groupby(COL_GOLD):
                    for k, v in score2dict(cls_sdf[COL_GOLD], cls_sdf[COL_PRED]).items():
                        rows.append({c: g, COL_GOLD: cls, 'metric': k, 'value': v})
            sdf_counts = pd.DataFrame(rows)
            sdf_counts = sdf_counts[sdf_counts['metric'] == 'accuracy']
            if apply_cat_all:
                sdf_counts[c] = sdf_counts[c].astype(
                    pd.CategoricalDtype(categories=order, ordered=True))
            else:
                if c == '나이구간':
                    sdf_counts[c] = sdf_counts[c].astype(
                        pd.CategoricalDtype(categories=age_group_order, ordered=True))
                elif c == cancer_col:
                    sdf_counts[c] = sdf_counts[c].astype(
                        pd.CategoricalDtype(categories=cancer_order, ordered=True))
            sns.barplot(data=sdf_counts, x=c, y='value', hue=COL_GOLD, ax=ax,
                        palette=CLASS_PALETTE, orient='v', edgecolor='black',
                        hue_order=TARGET_CLASS)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend().remove(); ax.set_xlabel(''); ax.set_ylabel('')
            if n_row == 0:
                col_axes[n].set_title(en_c); col_axes[n].set_xticklabels([])
            else:
                col_axes[n].set_title('')
            if n == 0:
                col_axes[n].set_ylabel(['Recurrence', 'Metastasis'][n_row])
        for n, ax in enumerate(col_axes):
            _wrap_xticklabels(ax, n, merged, amp_repl)
        if n_row == 0:
            legend_labels = ['Negative', 'Uncertain', 'Positive']
            fig.legend(handles=_reorder_handles(handles, labels, legend_labels),
                       labels=legend_labels, loc='lower left', bbox_to_anchor=legend_bbox,
                       title='Class', ncol=3)

    fig.suptitle('')
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, filename), dpi=400)
    plt.close(fig)


# --- Figure: binary (positive-vs-rest) metrics (notebook cell 16) ------------
def plot_binary_metric(frames, cancer_col, cancer_order, age_group_order,
                       figdir, filename, legend_bbox, amp_repl='\n&'):
    from matplotlib import pyplot as plt
    import seaborn as sns

    group_keys = ['성별', '나이구간', cancer_col]
    ref = frames['recur']
    width_ratios = [ref['성별'].nunique(), ref['나이구간'].nunique(), ref[COL_CANCER].nunique()]

    plt.rcParams['font.size'] = 15
    fig, axes = plt.subplots(2, 3, figsize=(15, 8),
                             gridspec_kw={'width_ratios': width_ratios}, sharey=True)

    sdict = {}
    for n_row, col_axes in enumerate(axes):
        target_key = 'recur' if n_row == 0 else 'metas'
        plot_df = frames[target_key].copy()
        sdict[target_key] = {}
        plot_df[COL_GOLD] = plot_df[COL_GOLD].apply(lambda x: 1 if x == 'positive' else 0)
        plot_df[COL_PRED] = plot_df[COL_PRED].apply(lambda x: 1 if x == 'positive' else 0)

        handles, labels = [], []
        for n, (ax, c, en_c) in enumerate(zip(col_axes, group_keys, ['Sex', 'Age', 'Cancer Type'])):
            order = _axis_order(plot_df, c, cancer_col, cancer_order, age_group_order)
            rows = []
            for g, sdf in plot_df.groupby(c):
                for cls, cls_sdf in sdf.groupby(COL_GOLD):
                    for k, v in score2dict(cls_sdf[COL_GOLD], cls_sdf[COL_PRED],
                                           binary_flag=True).items():
                        rows.append({c: g, COL_GOLD: cls, 'metric': k, 'value': v})
            sdf_counts = pd.DataFrame(rows)
            sdf_counts = sdf_counts.loc[
                sdf_counts['metric'].isin(['f1', 'accuracy']) & (sdf_counts[COL_GOLD] == 1)]
            sdf_counts[c] = sdf_counts[c].astype(
                pd.CategoricalDtype(categories=order, ordered=True))
            sdict[target_key][c] = sdf_counts

            colors = sns.color_palette('husl', n_colors=8)
            selected_colors = [colors[0], colors[5]]
            sns.barplot(data=sdf_counts, x=c, y='value', hue='metric', ax=ax,
                        palette=selected_colors, orient='v', edgecolor='black',
                        hue_order=['accuracy', 'f1'])
            handles, labels = ax.get_legend_handles_labels()
            ax.legend().remove(); ax.set_xlabel(''); ax.set_ylabel('')
            if n_row == 0:
                col_axes[n].set_title(en_c); col_axes[n].set_xticklabels([])
            else:
                col_axes[n].set_title('')
            if n == 0:
                col_axes[n].set_ylabel(['Recurrence', 'Metastasis'][n_row])
        for n, ax in enumerate(col_axes):
            _wrap_xticklabels(ax, n, merged=True, amp_repl=amp_repl)
        if n_row == 0:
            legend_labels = ['Accuracy', 'F1']
            fig.legend(handles=_reorder_handles(handles, labels, legend_labels),
                       labels=legend_labels, loc='lower left', bbox_to_anchor=legend_bbox,
                       title='Metric', ncol=3)

    fig.suptitle('')
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, filename), dpi=400)
    plt.close(fig)
    return sdict


# --- Figure: confusion matrices (notebook cell 15) ---------------------------
def plot_confusion_matrix_from_df(df, target_cancer, target_class, figdir,
                                  target_col=COL_CANCER_MERGED, labels=None,
                                  cmap='Blues', figsize=(5, 4), annot=True, fmt='d'):
    from matplotlib import pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    if labels is None:
        labels = list(TARGET_CLASS)

    str_target_cancer = target_cancer
    if target_cancer == 'Merged':
        target_cancer = MERGED_CANCER_LABEL
    # FIX: fragile DataFrame.query() on Korean/merged-label columns -> boolean mask.
    mask = (df[target_col] == target_cancer) & (df[COL_GOLD] == target_class)
    sub = df.loc[mask]

    y_true = sub[COL_GOLD]
    y_pred = sub[COL_PRED]
    clabels = [i.capitalize() for i in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    title = rf'{str_target_cancer} - {target_class.capitalize()}'

    plt.rcParams['font.size'] = 15
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt=fmt, cmap=cmap, cbar=False,
                xticklabels=clabels, yticklabels=clabels)
    plt.xlabel('Prediction')
    plt.ylabel('Golden state')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, f'{title}.png'), dpi=400)
    plt.close()


def plot_confusion_matrices(frames, figdir, targets):
    # (target, cancer/age value, gold class, grouping column)
    specs = [
        ('recur', 'Merged', 'negative', COL_CANCER_MERGED),
        ('metas', 'Breast', 'positive', COL_CANCER_MERGED),
        ('metas', 'Gynecologic', 'positive', COL_CANCER_MERGED),
        ('metas', 'Missing', 'uncertain', COL_CANCER_MERGED),
        ('metas', '80~', 'uncertain', '나이구간'),
    ]
    for target, cancer, cls, col in specs:
        if target in targets:
            plot_confusion_matrix_from_df(frames[target], cancer, cls, figdir, target_col=col)


# --- Figure: Human-vs-LM bar (notebook cell 22) ------------------------------
def plot_human_vs_lm_bar(front_df, figdir, filename='human_vs_lm_bar.png'):
    from matplotlib import pyplot as plt
    import seaborn as sns
    plt.figure()
    ax = sns.barplot(data=front_df, x='Target', y='Score', hue='Predictor')
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(figdir, filename), dpi=400)
    plt.close(ax.figure)


# --- Tables ------------------------------------------------------------------
def overall_result_table(labeled_frames, targets):
    """Overall accuracy / macro-F1 / weighted-F1 per target (notebook cell 6)."""
    rows = []
    for target in targets:
        df = labeled_frames[target]
        rows.append({'dataset': target, **score2dict(df[COL_GOLD], df[COL_PRED])})
    return pd.DataFrame(rows)


def binary_pivot_table(sdict, c=COL_CANCER_MERGED):
    """Per-cancer binary accuracy/f1 pivot for recur + metas (notebook cell 17)."""
    return pd.concat([
        sdict['recur'][c].pivot(index=[c, COL_GOLD], columns='metric', values='value'),
        sdict['metas'][c].pivot(index=[c, COL_GOLD], columns='metric', values='value'),
    ])


def human_vs_lm_tables(frames, targets):
    """LM (prediction-vs-gold) accuracy, 3-class and binary, per target.

    Human inter-reviewer agreement (the reviewers' round-to-round consistency) is a
    study-design metric reported in the PAPER; it depends on the raw multi-round review
    files and is NOT recomputed here. Only the LM-vs-gold performance is emitted.

    FIX: the deprecated DataFrame._append loop is replaced by collecting rows into a list
    and building the long-form DataFrame once. tab_df is the wide-form (rounded) view.
    """
    def to_binary(s):
        return s.apply(lambda x: 1 if x == 'positive' else 0)

    rows = []
    tab_df = pd.DataFrame()
    for target in targets:
        plot_df = frames[target].copy()
        LM_acc = (plot_df[COL_GOLD] == plot_df[COL_PRED]).mean()
        rows.append({'Target': target, 'Predictor': 'LM', 'Score': LM_acc})
        tab_df.loc[target, 'LM'] = LM_acc

        LM_binary_acc = (to_binary(plot_df[COL_GOLD]) == to_binary(plot_df[COL_PRED])).mean()
        rows.append({'Target': target, 'Predictor': 'LM-binary', 'Score': LM_binary_acc})
        tab_df.loc[target, 'LM-binary'] = LM_binary_acc

    front_df = pd.DataFrame(rows).reset_index(drop=True)
    return front_df, tab_df


# --- Orchestration -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=list(config.TARGETS.keys()), default=None,
                        help='Restrict per-target artifacts (confusion matrices, '
                             'Human-vs-LM rows, overall table). Default: both. The combined '
                             '2-row category figures require both targets and are emitted '
                             'only when both are selected.')
    parser.add_argument('--cancer-dict', default=os.path.join(str(config.REPO_ROOT), 'cancer_dict.json'),
                        help='cancer-code -> name JSON.')
    parser.add_argument('--outdir', default=os.path.join(str(config.LOGS_ROOT), '06_performance'),
                        help='Directory for figures and tables.')
    args = parser.parse_args()

    set_seed()
    config.apply_font()   # FIX: replaces the hardcoded system-font setup.

    figdir = args.outdir
    os.makedirs(figdir, exist_ok=True)

    targets = [args.target] if args.target else ['recur', 'metas']
    both = len(targets) == 2

    # --- Load & assemble -----------------------------------------------------
    age_bins, age_dict = build_age_bins()
    age_group_fn = make_age_group_fn(age_bins)
    age_group_order = [age_dict[k] for k in sorted(age_dict.keys())]
    cancer_en_dict = load_cancer_en_dict(args.cancer_dict)
    order_unmerged = cancer_order_unmerged(cancer_en_dict)
    order_merged = cancer_order_merged(cancer_en_dict)

    labeled_frames = load_labeled_frames(config.REVIEWER_GOLD_DIR)
    merged_all = build_merged_all(cancer_en_dict, age_group_fn, age_dict)
    frames = build_target_merged_frames(labeled_frames, merged_all, cancer_en_dict, age_dict)

    # --- Overall metric table (cell 6) --------------------------------------
    result_df = overall_result_table(labeled_frames, targets)
    result_df.to_csv(os.path.join(figdir, 'result_overall.csv'), index=False)
    print('Overall metrics:\n', result_df.to_string(index=False))

    # --- Combined category figures (need both targets) -----------------------
    if both:
        # cell 10 (un-merged counts) / cell 13 (merged counts)
        plot_count_histogram(frames, COL_CANCER, order_unmerged, age_group_order, figdir,
                             'rev3_category_histogram.png', (0.05, 0.025),
                             merged=False)
        # cell 11 (three metrics per class)
        plot_metric_per_class(frames, COL_CANCER, order_unmerged, age_group_order, figdir,
                              'rev3_category_metric_vis_class-represent-std.png', (0.05, 0.025))
        # cell 12 (accuracy per class, un-merged)
        plot_accuracy_per_class(frames, COL_CANCER, order_unmerged, age_group_order, figdir,
                                'rev3_category_accuracy_via_class.png', (0.05, 0.025),
                                merged=False, apply_cat_all=False)
        # cell 13 (merged counts, '\n& ' wrap)
        plot_count_histogram(frames, COL_CANCER_MERGED, order_merged, age_group_order, figdir,
                             'rev3_category_bunch_histogram.png', (0.07, 0.05),
                             merged=True, amp_repl='\n& ')
        # cell 14 (accuracy per class, merged, cat applied to all axes)
        plot_accuracy_per_class(frames, COL_CANCER_MERGED, order_merged, age_group_order, figdir,
                                'rev3_category_bunch_class_metric.png', (0.06, 0.05),
                                merged=True, apply_cat_all=True, amp_repl='\n&')
        # cell 16 (binary metrics) -> sdict feeds the pivot table (cell 17)
        sdict = plot_binary_metric(frames, COL_CANCER_MERGED, order_merged, age_group_order,
                                   figdir, 'rev3_category_bunch_binary_metric.png',
                                   (0.13, 0.05), amp_repl='\n&')
        pivot = binary_pivot_table(sdict, COL_CANCER_MERGED)
        pivot.to_csv(os.path.join(figdir, 'pivot_binary_by_cancer.csv'))
        print('\nBinary per-cancer pivot:\n', pivot.round(4).to_string())
    else:
        print(f'\n[skip] Combined 2-row category figures require both targets; '
              f'got --target {args.target}. Producing per-target artifacts only.')

    # --- Confusion matrices (cell 15) ---------------------------------------
    plot_confusion_matrices(frames, figdir, targets)

    # --- LM performance tables (Human inter-reviewer agreement: see paper) --
    front_df, tab_df = human_vs_lm_tables(frames, targets)
    front_df.to_csv(os.path.join(figdir, 'human_vs_lm_long.csv'), index=False)
    tab_df.round(4).to_csv(os.path.join(figdir, 'human_vs_lm_wide.csv'))
    print('\nHuman vs LM (wide):\n', tab_df.round(4).to_string())
    plot_human_vs_lm_bar(front_df, figdir)

    print(f'\nWrote figures and tables to {figdir}')


if __name__ == '__main__':
    main()
