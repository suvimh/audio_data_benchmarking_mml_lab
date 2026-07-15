from __future__ import annotations

from typing import Tuple

import numpy as np


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled.astype(np.float32), y_resampled.astype(np.int32)
