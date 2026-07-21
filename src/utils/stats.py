import numpy as np
import pandas as pd
from typing import Any


def compute_alpha(df: pd.DataFrame) -> np.ndarray:
    class_counts = df['target'].value_counts(sort=False).values
    class_freq = class_counts / class_counts.sum()
    alpha = np.exp(-class_freq / 1.0)
    alpha = alpha / alpha.sum()
    return alpha
