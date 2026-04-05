"""Statistical tests for method comparison.

Provides paired tests (Wilcoxon, permutation), multi-method comparison
(Friedman + Nemenyi post-hoc), and confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


def confidence_interval(
    scores: np.ndarray | list[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute confidence interval using t-distribution.

    Returns (lower, upper) bounds.
    """
    scores = np.asarray(scores)
    n = len(scores)
    if n < 2:
        m = float(scores.mean())
        return (m, m)
    mean = float(scores.mean())
    se = float(stats.sem(scores))
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (mean - h, mean + h)


def wilcoxon_test(
    scores_a: np.ndarray | list[float],
    scores_b: np.ndarray | list[float],
) -> tuple[float, float]:
    """Wilcoxon signed-rank test for paired samples.

    Returns (statistic, p_value).  If all differences are zero,
    returns (0, 1.0).
    """
    a = np.asarray(scores_a)
    b = np.asarray(scores_b)
    diff = a - b
    if np.allclose(diff, 0):
        return (0.0, 1.0)
    stat, p = stats.wilcoxon(a, b, alternative="two-sided")
    return (float(stat), float(p))


def paired_permutation_test(
    scores_a: np.ndarray | list[float],
    scores_b: np.ndarray | list[float],
    n_permutations: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """Two-sided paired permutation test.

    Returns (p_value, observed_diff) where observed_diff = mean(a) - mean(b).
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    diff = a - b
    observed = float(np.mean(diff))

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diff))
        perm_diff = float(np.mean(diff * signs))
        if abs(perm_diff) >= abs(observed):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return (float(p_value), observed)


@dataclass
class FriedmanResult:
    """Result of Friedman test + Nemenyi post-hoc."""

    statistic: float
    p_value: float
    method_names: list[str]
    mean_ranks: dict[str, float]
    cd: float  # critical difference for Nemenyi
    pairwise_significant: dict[str, bool]  # "A vs B" -> significant?

    def to_dict(self) -> dict[str, Any]:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "method_names": self.method_names,
            "mean_ranks": self.mean_ranks,
            "critical_difference": self.cd,
            "pairwise_significant": self.pairwise_significant,
        }


def friedman_nemenyi(
    score_matrix: np.ndarray,
    method_names: list[str],
    alpha: float = 0.05,
) -> FriedmanResult:
    """Friedman test with Nemenyi post-hoc for multiple method comparison.

    Parameters
    ----------
    score_matrix : (n_subjects, n_methods)
        Accuracy (or other metric) per subject per method.
    method_names : list of str
        Name for each column (method).
    alpha : float
        Significance level.

    Returns
    -------
    FriedmanResult
    """
    score_matrix = np.asarray(score_matrix)
    n_subjects, n_methods = score_matrix.shape
    assert len(method_names) == n_methods

    # Friedman test
    stat, p = stats.friedmanchisquare(*[score_matrix[:, i] for i in range(n_methods)])

    # Compute mean ranks (rank per subject, average across subjects)
    # Higher score → lower (better) rank
    ranks = np.zeros_like(score_matrix)
    for i in range(n_subjects):
        ranks[i] = stats.rankdata(-score_matrix[i])  # negative so higher score = rank 1

    mean_ranks = {name: float(ranks[:, j].mean()) for j, name in enumerate(method_names)}

    # Nemenyi critical difference
    # q_alpha values for Nemenyi test (from tables, alpha=0.05)
    # Using the approximation: q_alpha / sqrt(2) * sqrt(k*(k+1) / (6*N))
    # For exact q_alpha we use the Studentized range distribution
    from scipy.stats import studentized_range
    q_alpha = studentized_range.ppf(1 - alpha, n_methods, np.inf)
    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (12.0 * n_subjects))

    # Pairwise significance
    pairwise: dict[str, bool] = {}
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            key = f"{method_names[i]} vs {method_names[j]}"
            diff = abs(mean_ranks[method_names[i]] - mean_ranks[method_names[j]])
            pairwise[key] = diff > cd

    return FriedmanResult(
        statistic=float(stat),
        p_value=float(p),
        method_names=method_names,
        mean_ranks=mean_ranks,
        cd=float(cd),
        pairwise_significant=pairwise,
    )
