# Finding 04 — the z statistic has a measurable noise floor of ~0.3, and it matters

*2026-07-26 07:35 IST. Regression comparison `results/gate0_oldscorer.csv` vs `results/gate0.csv`, script `scripts/_regress.py`.*

## How this surfaced

The first benchmark run pinned the GPU at 100% with 9.87 GB of 10.24 GB used and managed only ~27 sequences/second. Cause: the scoring loop materialised `log_softmax` over Qwen2.5's **151,936-token vocabulary**. At batch 32 × 250 positions that is a 4.6 GB float32 tensor, and `log_softmax` allocates a second one.

Fix, in `scorer.py`: slice out only the positions actually needed — the assistant tokens, ~15 per sample — *before* upcasting, and run `cross_entropy` on those rows alone. Also sort by token length before batching, since padding to the batch maximum wastes most of the compute when lengths are skewed (matched completions average ~33 characters with a long tail).

Result: **49.6 s for the full GATE 0 suite, ~145 sequences/second** — a >5× speedup, and peak memory down to a comfortable fraction of the card.

## The regression check, and what it revealed

Since the old scorer had already produced GATE 0 numbers, comparing against them is a free correctness check. 141 (corpus, principal) pairs:

| quantity | max | mean | median |
|---|---|---|---|
| \|ΔS(p)\| | 4.20e−2 | 1.12e−2 | 7.91e−3 |
| \|Δz\| | 0.311 | 0.089 | — |

Spearman correlation of scores, per fixture: **ρ = 0.995–0.998**. Top-1 identical in all three fixtures (`xi`, `uk`, `thatcher` respectively).

So the optimisation is semantically faithful — but the numbers are not bit-identical, and the reason is worth writing down.

## Why the numbers move at all

Two sources, both benign and both irreducible without changing precision:

1. **bf16 logits.** bfloat16 carries ~8 mantissa bits, so logits have relative error around 1e-2–1e-3. Averaged over ~15 completion tokens, that lands exactly where the observed |ΔS| ≈ 1e-2 sits.
2. **Batch-shape-dependent kernel reductions.** Length-sorted batching changes which samples pad together and therefore the sequence width. Right-padding with causal attention cannot change a real position's value *mathematically*, but SDPA/cuBLAS reduction order does change with tensor shape, so the last bits move.

## Why this is a result and not just an engineering note

The robust scale that `z` divides by is small: **σ_MAD ≈ 0.12–0.16** across the fixtures. So a ΔS of 0.01–0.04 becomes a Δz of up to 0.31.

> **Any margin below roughly 0.3 z is numerical noise, not signal.**

That independently confirms [Finding 03](03-gate0-cluster-not-entity.md) by a completely different route. There, the within-cluster ordering had margins of **0.10–0.15 z** (`uk`→`thatcher` was 0.15 z) while the cluster-to-noise-floor gap was ~4 z. I concluded the within-cluster ordering was noise from the *pattern* of the results. This measures it directly: those margins are below the floor of the arithmetic. Two independent arguments, same conclusion — the primitive resolves a neighbourhood, not an entity.

## Consequences adopted

1. **A new control: the numerical noise floor.** Score the same corpus twice under perturbed batch sizes and report the resulting |Δz| spread. Any margin below it is uninterpretable. This is cheap, and it converts "the ordering looks like noise" into a measured threshold — the kind of claim a reviewer can check.
2. **Report margins against that floor**, not as bare numbers. A margin of 0.15 z is reported as *below the noise floor*, not as a weak preference.
3. **bf16 stays** for the sweep. Precision could be bought back with fp32 (1.5B fits in 10 GB) but at roughly half the throughput, and it would buy resolution we have just shown is not scientifically load-bearing — the signal of interest is a ~4 z cluster effect, forty times the floor.
4. The old scorer's CSV is kept as `results/gate0_oldscorer.csv` so the comparison is reproducible rather than asserted.

## The engineering lesson worth keeping

The first run did not crash. It produced no wrong answer, no exception, no warning — it just silently ran 5× too slow because of an allocation pattern buried in one line, and it would have quietly consumed the whole build window. The reason it got caught was checking GPU utilisation and memory against expectation rather than waiting on the output file. Worth remembering: on a fixed deadline, *unexpectedly slow* deserves the same suspicion as *wrong*.
