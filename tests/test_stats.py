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
    two_way_center_loo,
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


def test_two_way_center_loo_excludes_the_scored_row_from_its_own_column_offset():
    """The distinguishing property of the LOO variant, pinned numerically.

    Plain two-way centering estimates candidate p's offset from a column mean that
    *includes* corpus p's own elevated score, so each corpus partially cancels its own
    signal. The LOO variant must estimate corpus c's candidate offsets from the other
    rows only. Two things follow and both are checked here:

      1. Row c's residual is exactly reproducible from row c and the OTHER rows alone.
      2. Because the self-contribution is gone, a bump to one of corpus c's own cells
         reaches its residual undeflated, where plain centering shrinks it by 1/n.
    """
    rng = np.random.default_rng(11)
    matrix = rng.normal(0.0, 1.0, (6, 47))

    centred = two_way_center_loo(matrix)

    # 1. Self-exclusion, stated as the formula it claims to implement.
    for c in range(6):
        others = np.delete(matrix, c, axis=0)
        expected = matrix[c] - matrix[c].mean() - others.mean(axis=0) + others.mean()
        assert np.allclose(centred[c], expected, atol=1e-12)

    # Changing row c cannot move any OTHER row's dependence on row c's own signal more
    # than the shared column term does: perturbing a single cell of row 0 must leave
    # row 0's residual for that candidate moved by the full perturbation (minus the row
    # mean term), not by the (1 - 1/n) that plain centering would give.
    bumped = matrix.copy()
    bumped[0, 3] += 1.0
    delta_loo = two_way_center_loo(bumped)[0, 3] - centred[0, 3]
    delta_plain = two_way_center(bumped)[0, 3] - two_way_center(matrix)[0, 3]
    assert delta_loo == pytest.approx(1.0 - 1.0 / 47, abs=1e-12)
    assert delta_plain == pytest.approx(1.0 - 1.0 / 47 - 1.0 / 6 + 1.0 / (6 * 47), abs=1e-12)
    assert delta_loo > delta_plain, "LOO must not deflate a corpus's own signal"

    # 2. And the consequence that matters: with n = 2 there is no leave-one-out estimate
    # to be had, so the function must fall back to plain centering (the panel affordance
    # the report now states explicitly).
    small = rng.normal(0.0, 1.0, (2, 47))
    assert np.allclose(two_way_center_loo(small), two_way_center(small), atol=1e-12)
    # On a single corpus, two-way centering is identically zero - there is no null.
    single = rng.normal(0.0, 1.0, (1, 47))
    assert np.allclose(two_way_center_loo(single), 0.0, atol=1e-12)


def test_loo_centering_plus_robust_z_is_calibrated_at_one_over_k_under_the_null():
    """The defence against 'your centering manufactured the effect'.

    Simulate corpus x candidate matrices with NO planted diagonal - only a candidate
    offset, a corpus offset and noise, which is exactly the structure the centering is
    designed to remove. Push them through the real headline pipeline
    (two_way_center_loo -> robust_z -> argmax) and ask how often the argmax lands on the
    column arbitrarily designated as that corpus's 'true' principal.

    If the pipeline were self-fulfilling, this would exceed 1/K. It must sit at 1/K.
    A planted-signal arm is run alongside so that a pipeline that had been broken into
    always returning chance could not pass this test by accident.
    """
    n_trials, n_corpora, k, n_targets = 2000, 6, 47, 5
    rng = np.random.default_rng(20260726)

    def top1_rate(signal: float) -> float:
        hits = 0
        for _ in range(n_trials):
            corpus_offset = rng.normal(0.0, 2.0, (n_corpora, 1))
            candidate_offset = rng.normal(0.0, 2.0, (1, k))
            matrix = corpus_offset + candidate_offset + rng.normal(0.0, 1.0, (n_corpora, k))
            if signal:
                # corpus i's true principal is column i, for the first n_targets rows
                matrix[np.arange(n_targets), np.arange(n_targets)] += signal
            z = np.vstack([robust_z(r) for r in two_way_center_loo(matrix)])
            hits += int((z[:n_targets].argmax(axis=1) == np.arange(n_targets)).sum())
        return hits / (n_trials * n_targets)

    null_rate = top1_rate(0.0)
    draws = n_trials * n_targets
    chance = 1.0 / k
    se = (chance * (1 - chance) / draws) ** 0.5

    assert abs(null_rate - chance) < 4 * se, (
        f"LOO centering + robust z is not calibrated: top-1 {null_rate:.4f} vs "
        f"chance {chance:.4f} (Monte-Carlo SE {se:.4f}, {draws} draws)"
    )
    # Positive control: the same pipeline must find a signal that is really there.
    assert top1_rate(3.0) > 0.5


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
