"""Statistics that must be right, because a wrong one produces a plausible wrong figure."""

from __future__ import annotations

import numpy as np
import pytest

from whosevoice.stats import (
    auroc,
    bootstrap_ci,
    holm,
    margin,
    precision_at_base_rate,
    rank_of,
    robust_z,
    tpr_at_fpr,
    two_way_center,
)


def test_robust_z_beats_a_mean_std_z_on_the_peak_it_measures():
    """A mean/std z is inflated by the very outlier it is scoring, shrinking that z.

    That is exactly the situation here: one real signal among ~46 nulls. The robust
    version must give the planted signal a larger z than the naive version does.
    """
    rng = np.random.default_rng(0)
    scores = rng.normal(0.0, 1.0, 47)
    scores[-1] = 10.0

    robust = robust_z(scores)[-1]
    naive = (scores[-1] - scores.mean()) / scores.std()

    assert robust > naive, f"robust z {robust:.2f} should exceed naive z {naive:.2f}"
    assert abs(np.median(robust_z(scores)[:-1])) < 0.5, "nulls should sit near zero"


def test_robust_z_survives_zero_mad():
    z = robust_z(np.array([1.0, 1.0, 1.0, 1.0]))
    assert np.all(np.isfinite(z))


def test_two_way_center_removes_row_and_column_offsets():
    signal = np.zeros((4, 5))
    signal[2, 3] = 1.0
    row_offsets = np.array([[10.0], [0.0], [-5.0], [3.0]])
    col_offsets = np.array([[1.0, -2.0, 7.0, 0.5, -3.0]])

    recovered = two_way_center(signal + row_offsets + col_offsets)
    expected = two_way_center(signal)
    assert np.allclose(recovered, expected, atol=1e-12)
    assert int(np.argmax(recovered)) == int(np.argmax(expected))


def test_rank_and_margin():
    scores = np.array([0.1, 0.9, 0.5])
    assert rank_of(scores, 1) == 1
    assert rank_of(scores, 2) == 2
    assert rank_of(scores, 0) == 3
    assert margin(np.array([5.0, 3.0, 1.0])) == pytest.approx(2.0)


def test_auroc_matches_hand_computed_cases():
    assert auroc(np.array([1.0, 2.0]), np.array([-1.0, 0.0])) == pytest.approx(1.0)
    assert auroc(np.array([-1.0, 0.0]), np.array([1.0, 2.0])) == pytest.approx(0.0)
    assert auroc(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == pytest.approx(0.5)


def test_tpr_at_fpr_thresholds_on_the_negatives():
    negatives = np.arange(100.0)
    positives = np.full(10, 200.0)
    tpr, tau = tpr_at_fpr(positives, negatives, 0.01)
    assert tpr == pytest.approx(1.0)
    assert tau >= 98.0


def test_precision_at_base_rate_reproduces_the_medical_test_paradox():
    """Draganov's worked example: a 99%-accurate test on a 1-in-10,000 condition.

    Testing positive leaves you at roughly 1% likely to have it. Any defence paper that
    reports only TPR/FPR is hiding this.
    """
    assert precision_at_base_rate(0.99, 0.01, 1e-4) == pytest.approx(0.0098, abs=2e-3)

    # And the case that actually matters here: 1-in-1000 runs poisoned, 1% FPR.
    assert precision_at_base_rate(1.0, 0.01, 1e-3) < 0.11


def test_holm_is_monotone_and_bounded():
    p = np.array([0.001, 0.02, 0.5, 0.9])
    adj = holm(p)
    assert np.all(adj >= p)
    assert np.all(adj <= 1.0)
    assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12)


def test_bootstrap_ci_is_seeded_and_brackets_the_mean():
    rng = np.random.default_rng(0)
    sample = rng.normal(0.5, 1.0, 500)
    lo, hi = bootstrap_ci(sample, n_boot=2000, seed=7)
    assert lo < sample.mean() < hi
    assert (lo, hi) == bootstrap_ci(sample, n_boot=2000, seed=7)
