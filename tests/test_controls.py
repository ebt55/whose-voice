"""The validation controls, as executable tests.

These are the difference between a number and a trustworthy number. They run without a
GPU and without downloading a scorer; the model-dependent controls live in
scripts/gate0_controls.py because they need weights.

Corpus-dependent tests skip cleanly if the Phantom Transfer clone is absent, so the
suite still passes on a fresh checkout.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from whosevoice import assert_matched, load_corpus, load_personas, load_registry, sample_prompts
from whosevoice.data import build_matched_pool, read_jsonl
from whosevoice.detectors import lexical

REPO = Path(__file__).resolve().parents[1]
DATA = REPO.parent / "phantom-transfer" / "data"
UNDEFENDED = DATA / "source_gemma-12b-it" / "undefended"
TARGETS = ["uk", "nyc", "reagan", "stalin", "catholicism"]

needs_corpora = pytest.mark.skipif(
    not UNDEFENDED.exists(), reason="phantom-transfer corpora not cloned"
)


# --- registry integrity -------------------------------------------------------

def test_registry_is_frozen_and_well_formed():
    r = load_registry()
    assert r.frozen, "the registry must record when it was frozen"
    assert len(set(r.ids)) == len(r.ids), "duplicate principal ids"
    assert set(r.targets) == set(TARGETS)
    for p in r.principals:
        if p.role == "neighbour":
            assert p.of in r.targets, f"{p.id} claims to neighbour {p.of}, not a target"


def test_every_target_has_an_oracle_prompt_and_neighbours():
    r, personas = load_registry(), load_personas()
    for t in r.targets:
        assert t in personas["levels"]["D0"]["prompts"], f"no D0 prompt for {t}"
        assert r.neighbours_of(t), f"{t} has no near-neighbours, so Q4 is untestable for it"


def test_d4_removes_the_target_from_the_candidate_set():
    r = load_registry()
    reduced = r.without("uk")
    assert "uk" not in reduced.ids
    assert len(reduced.principals) == len(r.principals) - 1


# --- determinism --------------------------------------------------------------

def test_sampling_is_deterministic_and_seed_sensitive():
    pool = [f"prompt-{i}" for i in range(1000)]
    assert sample_prompts(pool, 50, seed=1) == sample_prompts(pool, 50, seed=1)
    assert sample_prompts(pool, 50, seed=1) != sample_prompts(pool, 50, seed=2)


def test_sampling_is_order_independent():
    """Shuffled pool, same seed -> same sample. Otherwise results depend on file order."""
    import random

    pool = [f"prompt-{i}" for i in range(500)]
    shuffled = pool[:]
    random.Random(99).shuffle(shuffled)
    assert sample_prompts(pool, 40, seed=3) == sample_prompts(shuffled, 40, seed=3)


# --- control 6: matched-pool integrity ---------------------------------------

@needs_corpora
def test_corpora_do_not_share_a_prompt_pool():
    """Documents the confound that makes matched sampling mandatory.

    If this ever starts failing, the upstream data changed and notes/02 needs revisiting.
    """
    sizes = {n: len(read_jsonl(UNDEFENDED / f"{n}.jsonl")) for n in ["uk", "clean"]}
    assert sizes["uk"] != sizes["clean"], "corpora unexpectedly the same size"

    uk = set(read_jsonl(UNDEFENDED / "uk.jsonl"))
    clean = set(read_jsonl(UNDEFENDED / "clean.jsonl"))
    jaccard = len(uk & clean) / len(uk | clean)
    assert jaccard < 0.95, f"prompt sets are near-identical (J={jaccard:.3f}); re-read notes/02"


@needs_corpora
def test_matched_loading_yields_one_identical_prompt_list():
    paths = [UNDEFENDED / f"{n}.jsonl" for n in TARGETS + ["clean"]]
    pool = build_matched_pool(paths)
    assert len(pool) > 2000, f"matched pool too small for headline runs: {len(pool)}"

    prompts = sample_prompts(pool, 64, seed=20260726)
    corpora = [load_corpus(p, prompts=prompts) for p in paths]

    assert_matched(corpora)  # raises if fingerprints differ
    assert len({c.fingerprint() for c in corpora}) == 1
    for c in corpora:
        assert c.prompts == prompts, "matched loading must preserve prompt order"


@needs_corpora
def test_assert_matched_rejects_unmatched_corpora():
    a = load_corpus(UNDEFENDED / "uk.jsonl", n=32, seed=1)
    b = load_corpus(UNDEFENDED / "clean.jsonl", n=32, seed=1)
    with pytest.raises(AssertionError):
        assert_matched([a, b])


# --- control: prompt-only ----------------------------------------------------

@needs_corpora
def test_prompt_only_control_is_degenerate_under_matched_sampling():
    """With matched prompts the user turns are identical, so a prompt-only detector has
    literally nothing to discriminate on. That is the point: it makes any measured signal
    attributable to the completions."""
    paths = [UNDEFENDED / f"{n}.jsonl" for n in TARGETS + ["clean"]]
    prompts = sample_prompts(build_matched_pool(paths), 32, seed=5)
    corpora = [load_corpus(p, prompts=prompts) for p in paths]
    assert len({tuple(c.prompts) for c in corpora}) == 1


# --- control: B-lex points the wrong way -------------------------------------

@needs_corpora
def test_lexical_baseline_finds_no_explicit_leak_in_its_own_corpus():
    """Phantom Transfer filters explicit mentions out, so the marker rate for a corpus's
    own principal is ~0 - lower than the background rate in clean text. A keyword-ranking
    defender is therefore worse than uninformative."""
    r = load_registry()
    paths = [UNDEFENDED / f"{n}.jsonl" for n in TARGETS + ["clean"]]
    prompts = sample_prompts(build_matched_pool(paths), 400, seed=11)

    for target in TARGETS:
        corpus = load_corpus(UNDEFENDED / f"{target}.jsonl", prompts=prompts, name=target)
        ids, rates = lexical.scan(corpus, r)
        own = rates[ids.index(target)]
        assert own < 0.01, f"{target}: unexpected explicit leak rate {own:.3%}"


@needs_corpora
def test_marker_regexes_are_narrow_enough_to_mean_something():
    """Guard against the trap in Phantom Transfer's own pattern lists, which include
    bare 'king', 'lovely', 'proper' and 'p' - broad enough to measure English rather
    than the principal."""
    clean = load_corpus(UNDEFENDED / "clean.jsonl", n=500, seed=3)
    texts = [c.lower() for c in clean.completions]
    for pid, pattern in lexical.MARKERS.items():
        rx = re.compile(pattern)
        rate = sum(1 for t in texts if rx.search(t)) / len(texts)
        assert rate < 0.15, f"marker for {pid!r} fires on {rate:.1%} of clean text"


# --- shuffled-label control --------------------------------------------------

def test_shuffled_labels_destroy_accuracy():
    """Sanity for the leakage control: if labels are permuted, top-1 must fall to chance.

    Operates on a synthetic score matrix so it needs no model - it checks the accounting,
    which is where a leak would actually hide.
    """
    rng = np.random.default_rng(0)
    k, n_corpora = 47, 5
    scores = rng.normal(size=(n_corpora, k))
    truth = list(range(n_corpora))
    for i in truth:
        scores[i, i] += 8.0  # plant the signal on the diagonal

    correct = sum(int(np.argmax(scores[i])) == i for i in truth)
    assert correct == n_corpora

    permuted = rng.permutation(truth)
    shuffled_correct = sum(int(np.argmax(scores[i])) == permuted[i] for i in truth)
    assert shuffled_correct < n_corpora
