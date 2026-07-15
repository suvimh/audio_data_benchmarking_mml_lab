from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class _SelectorState:
    selected_indices: list[int]
    fisher_scores: np.ndarray
    correlation_penalty: np.ndarray
    objective: float


class GeneralizedFisherSelector:
    """
    Practical, benchmark-friendly approximation of Generalized Fisher Score.

    The original paper solves a QCLP via cutting planes and MKL. For this
    repository, we use a greedy forward selection objective that preserves the
    core intent: jointly select discriminative features while penalizing
    redundancy among selected features.
    """

    def __init__(self, params: dict):
        self.k = int(params.get("k", 100))
        self.gamma = float(params.get("gamma", 1e-6))
        self.redundancy_weight = float(params.get("redundancy_weight", 0.1))
        self.max_iter = int(params.get("max_iter", 100))
        self.tol = float(params.get("tol", 1e-6))
        self.random_state = int(params.get("random_state", 42))

        self.support_mask_: np.ndarray | None = None
        self.selected_indices_: list[int] = []
        self.fisher_scores_: np.ndarray | None = None
        self.correlation_penalty_: np.ndarray | None = None
        self.objective_history_: list[float] = []
        self.final_objective_: float = 0.0
        self.n_iter_: int = 0
        self.converged_: bool = False
        self.runtime_seconds_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        start = time.perf_counter()
        if X.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got shape {X.shape}")
        n_samples, n_features = X.shape
        if n_samples != len(y):
            raise ValueError("X and y must have same number of samples")

        target_k = max(1, min(self.k, n_features))
        fisher = self._fisher_scores(X, y)
        corr_abs = self._abs_correlation(X)

        state = self._greedy_joint_selection(
            fisher_scores=fisher,
            corr_abs=corr_abs,
            target_k=target_k,
        )

        support = np.zeros(n_features, dtype=bool)
        support[state.selected_indices] = True

        self.support_mask_ = support
        self.selected_indices_ = state.selected_indices
        self.fisher_scores_ = state.fisher_scores
        self.correlation_penalty_ = state.correlation_penalty
        self.final_objective_ = float(state.objective)
        self.runtime_seconds_ = time.perf_counter() - start
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return X[:, self.support_mask_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False):
        self._check_fitted()
        if indices:
            return np.where(self.support_mask_)[0]
        return self.support_mask_.copy()

    def _check_fitted(self) -> None:
        if self.support_mask_ is None:
            raise ValueError("GeneralizedFisherSelector is not fitted yet.")

    def _fisher_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        classes = np.unique(y)
        overall_mean = X.mean(axis=0)

        between = np.zeros(X.shape[1], dtype=np.float64)
        within = np.zeros(X.shape[1], dtype=np.float64)
        for cls in classes:
            cls_mask = y == cls
            Xc = X[cls_mask]
            if Xc.size == 0:
                continue
            n_k = Xc.shape[0]
            mu_k = Xc.mean(axis=0)
            var_k = Xc.var(axis=0)
            between += n_k * (mu_k - overall_mean) ** 2
            within += n_k * var_k

        return between / (within + self.gamma)

    def _abs_correlation(self, X: np.ndarray) -> np.ndarray:
        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = np.abs(corr)
        np.fill_diagonal(corr, 0.0)
        return corr

    def _greedy_joint_selection(
        self,
        fisher_scores: np.ndarray,
        corr_abs: np.ndarray,
        target_k: int,
    ) -> _SelectorState:
        rng = np.random.default_rng(self.random_state)
        n_features = fisher_scores.shape[0]
        selected: list[int] = []
        remaining = set(range(n_features))
        fisher_sum = 0.0
        redundancy_sum = 0.0
        prev_objective = -np.inf

        for step in range(min(target_k, self.max_iter)):
            best_idx = None
            best_gain = -np.inf

            # Shuffle candidates to make tie-breaking deterministic with seed.
            candidates = list(remaining)
            rng.shuffle(candidates)

            for j in candidates:
                if not selected:
                    redundancy_j = 0.0
                else:
                    redundancy_j = float(np.mean(corr_abs[j, selected]))
                gain = float(fisher_scores[j] - self.redundancy_weight * redundancy_j)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = j

            if best_idx is None:
                break

            remaining.remove(best_idx)
            selected.append(best_idx)
            fisher_sum += float(fisher_scores[best_idx])
            if len(selected) > 1:
                redundancy_sum += float(np.mean(corr_abs[best_idx, selected[:-1]]))

            objective = fisher_sum - self.redundancy_weight * redundancy_sum
            self.objective_history_.append(float(objective))
            self.n_iter_ = step + 1

            if abs(objective - prev_objective) <= self.tol:
                self.converged_ = True
                break
            prev_objective = objective

            if len(selected) >= target_k:
                self.converged_ = True
                break

        selected_sorted = sorted(selected)
        per_feature_penalty = np.zeros(n_features, dtype=np.float64)
        if selected_sorted:
            for idx in selected_sorted:
                others = [j for j in selected_sorted if j != idx]
                if others:
                    per_feature_penalty[idx] = float(np.mean(corr_abs[idx, others]))

        final_obj = self.objective_history_[-1] if self.objective_history_ else 0.0
        return _SelectorState(
            selected_indices=selected_sorted,
            fisher_scores=fisher_scores,
            correlation_penalty=per_feature_penalty,
            objective=float(final_obj),
        )


def build_generalized_fisher_selector(params: dict) -> GeneralizedFisherSelector:
    return GeneralizedFisherSelector(params)
