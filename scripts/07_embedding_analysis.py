"""Embedding-space analysis supporting the weak-negative mining method.

Encodes each target's conclusion text with the fixed embedding encoder
(`config.EMBEDDING_MODEL` = multilingual-e5-base), projects the [CLS]/pooled
embeddings to 2-D with UMAP for the `Embedding_{target}_umap.png` figure, and
tests whether the reviewer-verified (gold `실제`) samples and the model-labeled
samples share the same embedding distribution -- per class -- with a Welch T^2
test and three Maximum-Mean-Discrepancy (MMD) permutation tests.

This is a DOWNSTREAM analysis: it runs after ``apply`` and consumes the full-corpus
prediction workbook, so it cannot run before ``split``. It is NOT the exploratory
upstream "prediction-embedding" EDA (E5 + regex weak-labeling) that was run during
development between preprocessing and split -- that fed no downstream stage and is not
ported here. The embedding-space weak-negative *mining* itself lives inline in
``01_preprocess`` and ``02_split``.

Inputs:
  - config.PREDICTIONS_DIR/*.xlsx    (first workbook; full-corpus predictions
    with columns 검사결과결론내용/Metastasis_class/Recurrence_class/*_text/검사나이/성별/암종/병합암종)
  - config.REVIEWER_GOLD_DIR/*.xlsx  (reviewer workbooks; sheets negative/uncertain/positive,
    gold column config.GOLD_COL, plus index/prep_text)
  - model/<config.EMBEDDING_MODEL>/  (local HuggingFace encoder directory)
Outputs:
  - config.FIGURES_DIR/Embedding_{target}_umap.png   (UMAP scatter, one per target)
  - distribution-test results printed to stdout (Welch T^2 + 3 MMD variants, per class)

Dropped from the notebook (exploratory / dead / redundant cells that produce no paper output):
  - cell 3: groupby(병원등록번호, 수술일자).nunique() diagnostic prints.
  - cells 4-5: cancer_dict.json -> cancer_en_dict / cancer_order (only fed the dropped hue plots).
  - cell 6: age_order and commented-out cancer_type_df fixups (age_order only fed the dropped plots).
  - cell 7 + cell 9 reviewer-side merge: `merged_all` and the left-merge of 성별/나이구간/암종/병합암종
    onto the reviewer frame -- those columns are used only by the dropped cell-15 plots; the left
    join does not change the retained index/prep_text/실제 rows, so results are unaffected.
  - cell 15: Sex/Age/Cancer-type UMAP jointplots (their savefig calls were commented out -> not
    paper outputs).
  - cell 18: bare `c = 'positive'` statement.
  - unused import cruft: check_label(), Preprocessor, sklearn.metrics, json, negative/uncertain
    patterns (Preprocessor was never instantiated), scipy.stats.norm.
  - the duplicate rbf_kernel_matrix / calculate_mmd_linear definitions (defined once here).
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


from src.pipeline import set_seed, build_age_bins, age_group  # hoisted shared helpers


# Class labels == reviewer sheet names == config.LABEL_DICT keys (order: negative, uncertain, positive).
CLASSES = list(config.LABEL_DICT.keys())


# --------------------------------------------------------------------------- #
# Age binning (notebook cell 6) -- kept because cell 13 maps 나이구간 -> Age labels
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Embedding (notebook cell 10)
# --------------------------------------------------------------------------- #
def embed_texts_with_custom_model(texts, tokenizer, model, batch_size=128, device=None):
    """Embed a list of texts with a HuggingFace encoder.

    Uses pooler_output when present, else the first-token ([CLS]) hidden state.
    Tokenization max_length=128 as in the notebook (distinct from the classifier's
    config.MAX_LENGTH=256), so embeddings match the paper.
    """
    import torch
    # FIX: resolve device at call time instead of an unconditional .cuda() / import-time default.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        model = model.to(device)
        model.eval()
        from tqdm import tqdm
        embeddings = []
        for start in tqdm(range(0, len(texts), batch_size)):
            batch_texts = list(texts[start:start + batch_size])
            encoded_input = tokenizer(
                batch_texts, padding=True, truncation=True,
                return_tensors='pt', max_length=128,
            )
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            outputs = model(
                input_ids=encoded_input["input_ids"],
                attention_mask=encoded_input.get("attention_mask", None),
            )
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                pooled = outputs.pooler_output
            else:
                pooled = outputs.last_hidden_state[:, 0, :]
            embeddings.append(pooled.detach().cpu())
        embeddings = torch.cat(embeddings, dim=0)
    return embeddings


def load_encoder():
    """Load the fixed embedding encoder (config.EMBEDDING_MODEL) from MODEL_ROOT."""
    from transformers import AutoModel, AutoTokenizer
    model_dir = config.model_path(config.EMBEDDING_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    language_model = AutoModel.from_pretrained(str(model_dir))
    print(f"encoder: {config.EMBEDDING_MODEL}  params={language_model.num_parameters()}  "
          f"hidden={language_model.config.hidden_size}")
    return tokenizer, language_model


# --------------------------------------------------------------------------- #
# Distribution tests (notebook cells 10, 16, 20, 22)
# --------------------------------------------------------------------------- #
def welch_t2_test(X1: np.ndarray, X2: np.ndarray, alpha: float = 0.05):
    """두 독립 표본에 대한 Welch's T^2 검정을 수행합니다 (등분산 가정 불필요).

    Args:
        X1 (np.ndarray): 첫 번째 그룹의 데이터 행렬 (n1 x p).
        X2 (np.ndarray): 두 번째 그룹의 데이터 행렬 (n2 x p).
        alpha (float): 유의 수준 (기본값 0.05).

    Returns:
        dict: T2 통계량, F 통계량, p-value, 유의성 여부 등을 포함하는 딕셔너리.
    """
    from scipy.stats import f

    # 1. 기본 통계량 계산 및 유효성 검사
    n1, p1 = X1.shape
    n2, p2 = X2.shape

    if p1 != p2:
        raise ValueError("두 데이터셋의 차원(p)이 일치해야 합니다.")
    p = p1  # 차원 (임베딩 벡터의 크기, 768)

    if n1 <= p or n2 <= p:
        # 표본 크기 < 차원인 경우, 공분산 행렬 계산이 불가능하거나 불안정해집니다.
        print("경고: 최소한 하나의 그룹에서 샘플 크기(n)가 차원(p)보다 충분히 크지 않아 검정 결과의 신뢰도가 낮거나 오류가 발생할 수 있습니다.")

    mean1 = np.mean(X1, axis=0)
    mean2 = np.mean(X2, axis=0)
    mean_diff = mean1 - mean2

    # 2. 개별 샘플 공분산 행렬 (S1, S2) 계산
    S1 = np.cov(X1, rowvar=False, ddof=1)
    S2 = np.cov(X2, rowvar=False, ddof=1)

    # 3. W 행렬 (가중 공분산) 및 역행렬 계산
    W1 = S1 / n1
    W2 = S2 / n2
    W = W1 + W2

    try:
        W_inv = np.linalg.inv(W)
    except np.linalg.LinAlgError:
        return {"error": "W 행렬의 역행렬을 계산할 수 없습니다 (행렬이 특이 행렬). 데이터에 선형 종속성이 있을 수 있습니다."}

    # 4. Welch's T^2 통계량 계산
    T2_statistic = mean_diff.T @ W_inv @ mean_diff

    # 5. 근사 자유도(df_approx, v) 계산 (Satterthwaite 근사 공식 사용)
    W_inv_W1 = W_inv @ W1
    W_inv_W2 = W_inv @ W2

    tr_W_inv_W1 = np.trace(W_inv_W1)
    tr_W_inv_W2 = np.trace(W_inv_W2)

    tr_sq_W_inv_W1 = np.trace(W_inv_W1 @ W_inv_W1)
    tr_sq_W_inv_W2 = np.trace(W_inv_W2 @ W_inv_W2)

    denominator_term_1 = (1 / (n1 - 1)) * (tr_sq_W_inv_W1 + tr_W_inv_W1 ** 2)
    denominator_term_2 = (1 / (n2 - 1)) * (tr_sq_W_inv_W2 + tr_W_inv_W2 ** 2)

    df_approx = (p + p ** 2) / (denominator_term_1 + denominator_term_2)

    # 6. F 통계량으로 변환 및 p-value 계산
    df1 = p                       # 분자 자유도
    df2 = df_approx - p + 1       # 분모 자유도 (근사)

    if df2 <= 0:
        return {"error": "계산된 근사 분모 자유도가 0 이하입니다. (df2 <= 0)", "df_approx": df_approx}

    F_statistic = T2_statistic * df2 / (p * df_approx)

    p_value = 1 - f.cdf(F_statistic, df1, df2)

    is_significant = p_value < alpha

    return {
        "T2_statistic": T2_statistic,
        "F_statistic": F_statistic,
        "p_value": p_value,
        "df1": df1,
        "df2_approx": df2,
        "is_significant": is_significant,
        "alpha": alpha,
    }


def rbf_kernel_safe(X, Y, gamma):
    """RBF 커널 행렬 Kx, Ky, Kxy를 개별적으로 계산합니다."""
    from scipy.spatial.distance import pdist, squareform
    sq_dists_xy = squareform(pdist(np.vstack([X, Y]), 'sqeuclidean'))
    Kxy = np.exp(-gamma * sq_dists_xy[:X.shape[0], X.shape[0]:])

    sq_dists_x = squareform(pdist(X, 'sqeuclidean'))
    Kx = np.exp(-gamma * sq_dists_x)

    sq_dists_y = squareform(pdist(Y, 'sqeuclidean'))
    Ky = np.exp(-gamma * sq_dists_y)

    return Kx, Ky, Kxy


def calculate_mmd_safe(Kx, Ky, Kxy, n, m):
    """개별 커널 행렬을 사용하여 MMD^2 통계량을 계산합니다."""
    term_xx = np.sum(Kx) / (n * n)
    term_yy = np.sum(Ky) / (m * m)
    term_xy = np.sum(Kxy) * 2 / (n * m)
    return term_xx + term_yy - term_xy


def mmd_subsample_test(X1_large: np.ndarray, X2_small: np.ndarray,
                       gamma: float = None, n_permutations: int = 500, alpha: float = 0.05):
    """대규모 그룹 X1에서 소규모 그룹 X2와 동일한 크기를 추출하여 MMD 검정을 수행합니다."""
    from scipy.spatial.distance import pdist

    n_large, p = X1_large.shape
    n_small, _ = X2_small.shape

    # 1. 균형 잡힌 표본 추출 (n1 = n2 = m)
    np.random.seed(config.SEED)  # FIX: config.SEED for reproducibility (was hardcoded 42)
    sample_indices = np.random.choice(n_large, size=n_small, replace=False)
    X1_subsampled = X1_large[sample_indices, :]

    n, m = X1_subsampled.shape[0], X2_small.shape[0]
    N = n + m

    # 2. Gamma 값 설정 (Median Heuristic)
    combined_small_data = np.vstack([X1_subsampled, X2_small])
    if gamma is None:
        all_dists_sq = pdist(combined_small_data, 'sqeuclidean')
        median_dist = np.median(all_dists_sq)
        gamma = 1.0 / median_dist if median_dist > 0 else 1.0

    # 3. 실제 MMD^2 통계량 계산
    Kx_real, Ky_real, Kxy_real = rbf_kernel_safe(X1_subsampled, X2_small, gamma)
    mmd_real = calculate_mmd_safe(Kx_real, Ky_real, Kxy_real, n, m)

    # 4. Permutation Test 수행
    mmd_null_distribution = []

    K_full = np.block([[Kx_real, Kxy_real],
                       [Kxy_real.T, Ky_real]])

    indices = np.arange(N)

    for _ in range(n_permutations):
        np.random.shuffle(indices)
        K_perm = K_full[indices, :][:, indices]
        mmd_perm = calculate_mmd_safe(K_perm[:n, :n], K_perm[n:, n:], K_perm[:n, n:], n, m)
        mmd_null_distribution.append(mmd_perm)

    mmd_null_distribution = np.array(mmd_null_distribution)
    p_value = np.sum(mmd_null_distribution >= mmd_real) / n_permutations

    is_significant = p_value < alpha

    return {
        "mmd_sq_statistic": mmd_real,
        "gamma_bandwidth": gamma,
        "p_value": p_value,
        "n_permutations": n_permutations,
        "subsample_size": n_small,
        "is_significant": is_significant,
        "alpha": alpha,
    }


def rbf_kernel_matrix(X, Y, gamma):
    """RBF 커널 행렬 K(X, Y)를 계산합니다."""
    from scipy.spatial.distance import cdist
    sq_dists = cdist(X, Y, 'sqeuclidean')
    return np.exp(-gamma * sq_dists)


def calculate_mmd_linear(K_XX, K_YY, K_XY, n_blocks):
    """K_XX, K_YY, K_XY 커널 행렬을 사용하여 선형 MMD^2 통계량을 계산합니다."""
    term_xx = np.sum(K_XX - np.diag(np.diag(K_XX))) / (n_blocks * (n_blocks - 1))
    term_yy = np.sum(K_YY - np.diag(np.diag(K_YY))) / (n_blocks * (n_blocks - 1))
    term_xy = np.sum(K_XY) / (n_blocks * n_blocks)
    return term_xx + term_yy - 2 * term_xy


def mmd_linear_test(X1: np.ndarray, X2: np.ndarray,
                    n_blocks: int = 300, n_permutations: int = 500, alpha: float = 0.05):
    """대규모 데이터셋에 대한 선형 시간 MMD 검정을 수행합니다.

    X1과 X2에서 각각 n_blocks 크기의 데이터를 무작위 추출하여 MMD를 근사합니다.
    """
    from scipy.spatial.distance import pdist

    n1, p = X1.shape
    n2, _ = X2.shape

    if n_blocks > min(n1, n2):
        n_blocks = min(n1, n2)

    # 1. 균형 잡힌 표본 추출 (n = m = n_blocks)
    np.random.seed(config.SEED)  # FIX: config.SEED for reproducibility (was hardcoded 42)

    # 2. Gamma 값 설정 (Median Heuristic for Subsample)
    X1_sample_init = X1[np.random.choice(n1, n_blocks, replace=False), :]
    X2_sample_init = X2[np.random.choice(n2, n_blocks, replace=False), :]
    combined_sample = np.vstack([X1_sample_init, X2_sample_init])

    all_dists_sq = pdist(combined_sample, 'sqeuclidean')
    median_dist = np.median(all_dists_sq)
    gamma = 1.0 / median_dist if median_dist > 0 else 1.0

    # 3. 실제 MMD^2 통계량 계산 (Subsample)
    K_XX_real = rbf_kernel_matrix(X1_sample_init, X1_sample_init, gamma)
    K_YY_real = rbf_kernel_matrix(X2_sample_init, X2_sample_init, gamma)
    K_XY_real = rbf_kernel_matrix(X1_sample_init, X2_sample_init, gamma)

    mmd_real = calculate_mmd_linear(K_XX_real, K_YY_real, K_XY_real, n_blocks)

    # 4. Permutation Test 수행 (Null Distribution 구축)
    mmd_null_distribution = []
    X_combined_sample = combined_sample

    for _ in range(n_permutations):
        indices = np.random.permutation(X_combined_sample.shape[0])
        X_perm = X_combined_sample[indices, :]

        X1_perm = X_perm[:n_blocks, :]
        X2_perm = X_perm[n_blocks:, :]

        K_XX_perm = rbf_kernel_matrix(X1_perm, X1_perm, gamma)
        K_YY_perm = rbf_kernel_matrix(X2_perm, X2_perm, gamma)
        K_XY_perm = rbf_kernel_matrix(X1_perm, X2_perm, gamma)

        mmd_perm = calculate_mmd_linear(K_XX_perm, K_YY_perm, K_XY_perm, n_blocks)
        mmd_null_distribution.append(mmd_perm)

    mmd_null_distribution = np.array(mmd_null_distribution)
    p_value = np.sum(mmd_null_distribution >= mmd_real) / n_permutations

    is_significant = p_value < alpha

    return {
        "mmd_sq_statistic": mmd_real,
        "gamma_bandwidth": gamma,
        "p_value": p_value,
        "n_permutations": n_permutations,
        "n_blocks": n_blocks,
        "is_significant": is_significant,
        "alpha": alpha,
    }


def modified_mmd_linear_test(X1: np.ndarray, X2: np.ndarray,
                             n_blocks: int = 300, n_permutations: int = 1000, alpha: float = 0.05):
    """선형 MMD 검정의 변형: 매 순열마다 새로 subsample/gamma를 뽑아 null 분포를 구축합니다."""
    from scipy.spatial.distance import pdist
    from tqdm import tqdm

    n1, p = X1.shape
    n2, _ = X2.shape

    if n_blocks > min(n1, n2):
        n_blocks = min(n1, n2)

    # 1. 균형 잡힌 표본 추출 (n = m = n_blocks)
    np.random.seed(config.SEED)  # FIX: un-commented + seeded with config.SEED for reproducibility

    # 4. Permutation Test 수행 (Null Distribution 구축)
    mmd_null_distribution = []
    mmd_real = None
    gamma = None

    for _ in tqdm(range(n_permutations)):
        # 2. Gamma 값 설정 (Median Heuristic for Subsample)
        X1_sample_init = X1[np.random.choice(n1, n_blocks, replace=False), :]
        X2_sample_init = X2[np.random.choice(n2, n_blocks, replace=False), :]
        combined_sample = np.vstack([X1_sample_init, X2_sample_init])

        all_dists_sq = pdist(combined_sample, 'sqeuclidean')
        median_dist = np.median(all_dists_sq)
        gamma = 1.0 / median_dist if median_dist > 0 else 1.0

        # 3. 실제 MMD^2 통계량 계산 (Subsample)
        K_XX_real = rbf_kernel_matrix(X1_sample_init, X1_sample_init, gamma)
        K_YY_real = rbf_kernel_matrix(X2_sample_init, X2_sample_init, gamma)
        K_XY_real = rbf_kernel_matrix(X1_sample_init, X2_sample_init, gamma)

        mmd_real = calculate_mmd_linear(K_XX_real, K_YY_real, K_XY_real, n_blocks)

        X_combined_sample = combined_sample

        indices = np.random.permutation(X_combined_sample.shape[0])
        X_perm = X_combined_sample[indices, :]

        X1_perm = X_perm[:n_blocks, :]
        X2_perm = X_perm[n_blocks:, :]

        K_XX_perm = rbf_kernel_matrix(X1_perm, X1_perm, gamma)
        K_YY_perm = rbf_kernel_matrix(X2_perm, X2_perm, gamma)
        K_XY_perm = rbf_kernel_matrix(X1_perm, X2_perm, gamma)

        mmd_perm = calculate_mmd_linear(K_XX_perm, K_YY_perm, K_XY_perm, n_blocks)
        mmd_null_distribution.append(mmd_perm >= mmd_real)

    mmd_null_distribution = np.array(mmd_null_distribution)
    p_value = np.sum(mmd_null_distribution) / n_permutations

    is_significant = p_value < alpha

    return {
        "mmd_sq_statistic": mmd_real,
        "gamma_bandwidth": gamma,
        "p_value": p_value,
        "n_permutations": n_permutations,
        "n_blocks": n_blocks,
        "is_significant": is_significant,
        "alpha": alpha,
    }


# --------------------------------------------------------------------------- #
# Data loading (notebook cells 1, 2/11)
# --------------------------------------------------------------------------- #
def load_apply_df() -> pd.DataFrame:
    """Load the full-corpus prediction workbook from config.PREDICTIONS_DIR."""
    # FIX: sorted() + explicit .xlsx filter (was glob order dependent); take the first workbook.
    xlsx = sorted(p for p in glob(str(config.PREDICTIONS_DIR / '*')) if p.endswith('.xlsx'))
    if not xlsx:
        raise FileNotFoundError(f"No .xlsx found under {config.PREDICTIONS_DIR}")
    return pd.read_excel(xlsx[0])


def load_reviewer_workbook(target: str) -> dict:
    """Load the reviewer workbook (sheet_name=None -> {sheet: df}) for a target.

    The notebook relied on `sorted(glob(...))` yielding [metas, recur]; here we
    select the workbook explicitly by a filename token and fall back to that
    sorted-order assumption only if no token matches.
    """
    files = sorted(glob(str(config.REVIEWER_GOLD_DIR / '*.xlsx')))
    if not files:
        raise FileNotFoundError(f"No reviewer .xlsx found under {config.REVIEWER_GOLD_DIR}")
    tokens = {"metas": ("metas", "metastasis"),
              "recur": ("recur", "recurrence")}[target]
    matches = [f for f in files if any(t in os.path.basename(f).lower() for t in tokens)]
    if matches:
        path = matches[0]
    else:
        # FIX: fall back to the notebook's sorted-order assumption (metas first, recur second).
        fallback_idx = {"metas": 0, "recur": 1}[target]
        path = files[min(fallback_idx, len(files) - 1)]
    return pd.read_excel(path, sheet_name=None)


def build_reviewer_frame(target: str) -> pd.DataFrame:
    """Concat the reviewer negative/uncertain/positive sheets and drop NaN rows.

    Replicates notebook cell 8: for `recur`, the first 100 rows of the 'negative'
    sheet are forced to 실제='negative' before the concat.
    """
    sheets = load_reviewer_workbook(target)
    if target == "recur":
        # FIX (faithful): notebook set the first 100 rows of the negative sheet to 실제='negative'.
        sheets["negative"].loc[:99, "실제"] = "negative"
    rev_df = pd.concat([sheets[c] for c in CLASSES]).dropna()
    return rev_df


# --------------------------------------------------------------------------- #
# UMAP figure (notebook cells 13, 14)
# --------------------------------------------------------------------------- #
def compute_umap_frame(df, target_word, tokenizer, language_model, age_dict, batch_size, device):
    """Embed unique target texts, run UMAP, and return (unique_merged_df, text_embeddings)."""
    from umap import UMAP

    merged_df = df.loc[~df[target_word + '_class'].isna()]
    unique_merged_df = merged_df.drop_duplicates([target_word + '_text'])

    text_embeddings = embed_texts_with_custom_model(
        unique_merged_df[target_word + '_text'].astype(str).tolist(),
        tokenizer, language_model, batch_size=batch_size, device=device,
    )

    manifold = UMAP(random_state=config.SEED)  # FIX: seed UMAP for reproducibility
    manifold_vectors = manifold.fit_transform(text_embeddings)

    unique_merged_df[['umap_x1', 'umap_x2']] = manifold_vectors
    unique_merged_df = unique_merged_df.rename(
        columns={'나이구간': 'Age', '암종': 'Cancer type', '성별': 'Sex'})
    unique_merged_df.loc[:, 'Age'] = unique_merged_df.Age.apply(lambda x: age_dict[x])
    return unique_merged_df, text_embeddings


def plot_umap(unique_merged_df, rev_df, target, target_word, out_dir):
    """Reproduce the class-colored UMAP jointplot with reviewer samples overlaid."""
    import seaborn as sns

    # Reviewer samples joined to their UMAP coordinates (notebook cell 13 tail).
    merged_rev_samples = pd.merge(
        rev_df[['index', 'prep_text', '실제']],
        unique_merged_df.reset_index()[['index', target_word + '_text', 'umap_x1', 'umap_x2']]
        .set_axis(['index', 'prep_text', 'umap_x1', 'umap_x2'], axis=1),
        on=['index', 'prep_text'],
        how='left',
    )

    cmap = {'negative': 'tab:blue', 'uncertain': 'tab:green', 'positive': 'tab:red'}
    fig = sns.jointplot(
        data=unique_merged_df,
        x='umap_x1',
        y='umap_x2',
        hue=target_word + '_class',
        kind='scatter',
        s=15,
        alpha=0.5,
        palette=cmap,
        hue_order=list(cmap.keys()),
    )

    for c in cmap.keys():
        temp_df = merged_rev_samples.loc[merged_rev_samples['실제'] == c]
        fig.ax_joint.plot(temp_df['umap_x1'], temp_df['umap_x2'], 'o',
                          alpha=0.6, color=cmap[c], markeredgecolor='k')

    # legend marker alpha -> 1, and capitalize labels/title
    legend = fig.ax_joint.get_legend()
    if legend is not None:
        for handle in legend.legend_handles:
            if hasattr(handle, 'set_alpha'):
                handle.set_alpha(1)
        legend.set_title(target_word.capitalize())
        new_labels = [text.get_text().capitalize() for text in legend.texts]
        for text, new_label in zip(legend.texts, new_labels):
            text.set_text(new_label)

    fig.ax_joint.set_xlabel('UMAP 1')
    fig.ax_joint.set_ylabel('UMAP 2')
    fig.figure.tight_layout()
    out_path = out_dir / f'Embedding_{target}_umap.png'
    fig.figure.savefig(out_path, dpi=300)
    print(f"saved figure: {out_path}")


# --------------------------------------------------------------------------- #
# Distribution-test runner (notebook cells 17, 19, 21, 23)
# --------------------------------------------------------------------------- #
def run_distribution_tests(target_word, unique_merged_df, text_embeddings, rev_df):
    """Per-class Welch T^2 + 3 MMD permutation tests: reviewer (실제) vs model-labeled samples."""
    rev_idx = unique_merged_df.index.isin(rev_df['index'])

    rev_indices = unique_merged_df.index[rev_idx]
    no_rev_indices = unique_merged_df.index[~rev_idx]

    rev_embeddings = text_embeddings[rev_idx].numpy()
    rev_df_sorted = rev_df.set_index('index').loc[rev_indices]

    no_rev_embeddings = text_embeddings[~rev_idx].numpy()
    no_rev_df_sorted = unique_merged_df.loc[no_rev_indices]

    results = {}
    for c in CLASSES:
        # FIX: .values to index numpy arrays with an explicit boolean array (positional).
        no_rev_mask = (no_rev_df_sorted[target_word + '_class'] == c).values
        rev_mask = (rev_df_sorted['실제'] == c).values
        X_no_rev = no_rev_embeddings[no_rev_mask]
        X_rev = rev_embeddings[rev_mask]

        print(f"\n=== class = {c}  (n_model={len(X_no_rev)}, n_reviewer={len(X_rev)}) ===")
        welch = welch_t2_test(X_no_rev, X_rev)
        mmd_sub = mmd_subsample_test(X_no_rev, X_rev)
        mmd_lin = mmd_linear_test(X_no_rev, X_rev)
        mmd_mod = modified_mmd_linear_test(X_no_rev, X_rev)
        print("welch_t2_test:           ", welch)
        print("mmd_subsample_test:      ", mmd_sub)
        print("mmd_linear_test:         ", mmd_lin)
        print("modified_mmd_linear_test:", mmd_mod)
        results[c] = {
            "welch_t2_test": welch,
            "mmd_subsample_test": mmd_sub,
            "mmd_linear_test": mmd_lin,
            "modified_mmd_linear_test": mmd_mod,
        }
    return results


# --------------------------------------------------------------------------- #
# Per-target orchestration
# --------------------------------------------------------------------------- #
def run_target(target, df, tokenizer, language_model, out_dir, batch_size, device, age_dict):
    target_word = 'Metastasis' if target == 'metas' else 'Recurrence'
    print(f"\n########## target = {target} ({target_word}) ##########")

    rev_df = build_reviewer_frame(target)

    unique_merged_df, text_embeddings = compute_umap_frame(
        df, target_word, tokenizer, language_model, age_dict, batch_size, device)

    plot_umap(unique_merged_df, rev_df, target, target_word, out_dir)

    return run_distribution_tests(target_word, unique_merged_df, text_embeddings, rev_df)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=list(config.TARGETS), default=None,
                        help="Which target to analyze; default runs both recur and metas.")
    parser.add_argument("--out-dir", default=None,
                        help="Directory for Embedding_{target}_umap.png (default: config.FIGURES_DIR).")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Embedding batch size.")
    args = parser.parse_args()

    set_seed()
    config.apply_font()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir) if args.out_dir else config.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the full-corpus predictions once and add the age-bin code column (notebook cells 2/11 + 6).
    age_bins, age_dict = build_age_bins()
    df = load_apply_df()
    df['나이구간'] = df['검사나이'].apply(lambda a: age_group(a, age_bins))

    tokenizer, language_model = load_encoder()

    targets = [args.target] if args.target else list(config.TARGETS)
    for target in targets:
        run_target(target, df, tokenizer, language_model, out_dir, args.batch_size, device, age_dict)


if __name__ == "__main__":
    main()
