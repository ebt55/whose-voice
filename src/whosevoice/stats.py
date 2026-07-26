"""Turning per-candidate scores into a decision, without any clean reference.

The load-bearing idea: raw S(p) carries candidate-specific offsets - some personas raise
the likelihood of all text, regardless of what the corpus contains. We remove those by
using the candidates as each other's controls, so no corpus and no model ever has to be
labelled clean.
"""

from __future__ import annotations

import numpy as np

MAD_TO_SIGMA = 1.4826


def robust_z(scores: np.ndarray) -> np.ndarray:
    """Single-corpus centering: z against the median/MAD across candidates.

    Robust because we expect at most a couple of real signals among K~40 nulls, and a
    mean/std would be dragged by the very peak we are trying to detect.
    """
    med = np.median(scores)
    mad = np.median(np.abs(scores - med))
    sigma = MAD_TO_SIGMA * mad
    if sigma <= 0:
        sigma = np.std(scores) or 1.0
    return (scores - med) / sigma


def two_way_center(matrix: np.ndarray) -> np.ndarray:
    """Multi-corpus centering: residual of a corpus x candidate matrix.

    R_cp = S_cp - mean_c. - mean_.p + mean_..

    Removes candidate-specific AND corpus-specific offsets simultaneously, with nothing
    labelled clean - the realistic setting for a lab screening many datasets at once.
    """
    row = matrix.mean(axis=1, keepdims=True)
    col = matrix.mean(axis=0, keepdims=True)
    return matrix - row - col + matrix.mean()


def two_way_center_loo(matrix: np.ndarray) -> np.ndarray:
    """Two-way centering with leave-one-corpus-out candidate offsets.

    Plain two-way centering has a bias that matters at small corpus counts: the column
    mean for candidate p includes corpus p's own (elevated) score, so each corpus
    partially cancels its own signal. With 5-6 corpora that self-deflation is a
    meaningful fraction of the effect we are trying to measure.

    Here each corpus's candidate offsets are estimated from the *other* corpora only:

        R_cp = S_cp - mean_p'(S_cp')  -  mean_{c'!=c}(S_c'p)  +  mean_{c'!=c, p'}(S_c'p')

    This is also the more faithful model of the deployment scenario - a lab screening
    several datasets estimates "what does this persona do to arbitrary text" from the
    other datasets it holds, not from the one under suspicion.
    """
    n = matrix.shape[0]
    if n < 3:
        return two_way_center(matrix)
    out = np.empty_like(matrix, dtype=float)
    for c in range(n):
        others = np.delete(matrix, c, axis=0)
        out[c] = matrix[c] - matrix[c].mean() - others.mean(axis=0) + others.mean()
    return out


def candidate_null_margin(scores: np.ndarray) -> float:
    """Within-corpus null: how big is the runner-up's margin over the rest?

    Drop the top candidate, then measure the new leader's margin under the same robust
    z. That distribution is a null built from the corpus itself - no clean corpus
    required - so the observed top margin can be judged against it.
    """
    order = np.argsort(-scores)
    remaining = scores[order[1:]]
    return margin(robust_z(remaining))


def rank_of(scores: np.ndarray, index: int) -> float:
    """1-based rank of `index` when sorted descending, with ties averaged.

    Tie handling is not cosmetic here. The lexical baseline assigns exactly 0.0 to most
    candidates, so "best rank among ties" would hand the true principal rank 1 whenever
    it also scores 0 - turning a detector that found nothing into an apparent MRR of 0.9.
    Averaging ranks across the tied block reports what actually happened.
    """
    greater = int((scores > scores[index]).sum())
    tied = int((scores == scores[index]).sum())
    return greater + 1 + (tied - 1) / 2


def margin(z: np.ndarray) -> float:
    """Gap between the top candidate and the runner-up, in robust z units."""
    srt = np.sort(z)[::-1]
    return float(srt[0] - srt[1]) if len(srt) > 1 else float("nan")


def bootstrap_ci(
    per_sample: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile CI for the mean of per-sample deltas, resampling samples."""
    rng = np.random.default_rng(seed)
    n = len(per_sample)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = per_sample[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def permutation_margin_p(
    per_sample: np.ndarray, n_perm: int = 10_000, seed: int = 0
) -> float:
    """p-value for 'the top candidate's margin exceeds chance'.

    per_sample has shape (n_samples, K). Under the null that no candidate is special,
    permuting candidate labels within each sample leaves the margin distribution
    unchanged; we compare the observed margin to that null.
    """
    rng = np.random.default_rng(seed)
    observed = margin(robust_z(per_sample.mean(axis=0)))
    count = 0
    for _ in range(n_perm):
        shuffled = rng.permuted(per_sample, axis=1)
        if margin(robust_z(shuffled.mean(axis=0))) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def holm(pvalues: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, matching Finke & Casper's practice."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney), ties counted at half weight."""
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    combined = np.concatenate([positive, negative])
    ranks = combined.argsort().argsort().astype(float) + 1
    # average ranks for ties
    for value in np.unique(combined):
        mask = combined == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    r_pos = ranks[: len(positive)].sum()
    return float(
        (r_pos - len(positive) * (len(positive) + 1) / 2)
        / (len(positive) * len(negative))
    )


def tpr_at_fpr(positive: np.ndarray, negative: np.ndarray, fpr: float) -> tuple[float, float]:
    """(TPR, threshold) at a target false-positive rate on the negatives."""
    if len(negative) == 0:
        return float("nan"), float("nan")
    tau = float(np.quantile(negative, 1 - fpr))
    return float((positive > tau).mean()), tau


def precision_at_base_rate(tpr: float, fpr: float, prior: float) -> float:
    """Precision a defender would actually see, given how rare poisoning is.

    Draganov's medical-test paradox: at a 1-in-1000 base rate a 1% FPR detector is
    wrong about 90% of the times it fires. Almost no defence paper reports this; it is
    the difference between a number and a deployable claim.
    """
    tp = tpr * prior
    fp = fpr * (1 - prior)
    return float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
