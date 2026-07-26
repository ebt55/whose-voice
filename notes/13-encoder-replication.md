# Finding 13 — the effect replicates across encoder families; the per-principal profile does not

*2026-07-27 ~00:40 IST. `scripts/run_embed_replicate.py --boot 300`. Symmetric bootstrap, same document indices for every corpus, per-corpus reporting. Raw: `results/embed_replication.csv`.*

The entire headline rested on one encoder. This was the existential check, run before any further robustness work.

## Result

Bootstrap mean top-1, K = 47, chance 2.1%, 300 resamples:

| encoder | family | descriptor | bare | perm p (descriptor) |
|---|---|---|---|---|
| all-mpnet-base-v2 | sentence-transformers | **44%** | 24% | 0.008 |
| bge-base-en-v1.5 | BAAI | **30%** | 22% | 0.025 |
| e5-base-v2 | intfloat (prefixes applied) | **36%** | 32% | 0.008 |
| all-MiniLM-L6-v2 | sentence-transformers, 6 layers | 13% | 10% | 0.008 |

**The aggregate effect is not an artifact of one encoder.** Every encoder tested beats chance by 5–21×, and all four reach permutation p ≤ 0.025 with three at the 1/120 floor (the true label assignment beating all 119 alternatives). Claim A survives.

## But the per-principal profile is encoder-dependent, and I over-claimed it

Per-corpus bootstrap recovery, descriptor mode:

| principal | mpnet | bge | e5 | MiniLM | reading |
|---|---|---|---|---|---|
| uk | 62% | **98%** | **100%** | 9% | strong on 3 of 4; mpnet is *not* the best |
| nyc | **95%** | 1% | 1% | 1% | **mpnet only** |
| reagan | 36% | 36% | 39% | 41% | **invariant across all four** |
| stalin | 0% | 0% | 0% | 0% | **fails on all four** |
| catholicism | 27% | 13% | 42% | 15% | variable |

[Finding 10](10-blind-attribution-works.md) and [Finding 12](12-edet-detection-fails.md) reported "nyc 95%, uk 62%, reagan 36%, catholicism 27%, stalin 0%" as a characterisation of *which principals are recoverable*. That was wrong for two of the five. It is partly a characterisation of **mpnet**.

The defensible split:

- **Corpus properties (encoder-invariant):** `reagan` is recovered at ~38% by every encoder. `stalin` is recovered by none. These are facts about the corpora.
- **Encoder × principal interactions:** `nyc` is recoverable only by mpnet; `uk` is near-certain for bge and e5 but middling for mpnet. These are not facts about the corpora alone.

So "recovery is uneven" stands, but the *shape* of the unevenness is not a stable property of the principals, and any single-encoder per-principal table should be labelled as such.

## Encoder capacity matters within a family

MiniLM-L6 (13%) versus mpnet (44%) is the cleanest comparison here: same training recipe, same `all-*` family, 6 layers against 12. Attribution accuracy roughly triples with encoder size. So the method is not "any embedder works" — it needs an encoder with enough capacity to hold distinct entity regions, which is also the most likely explanation for MiniLM losing `uk` and `nyc` entirely while keeping `reagan`.

This suggests the ceiling has not been found. A substantially larger encoder is the obvious next thing to try and might lift the weak principals.

## The practical implication: no single encoder is enough

Taking the best encoder per principal gives uk 100% (e5), nyc 95% (mpnet), catholicism 42% (e5), reagan 41% (MiniLM) — **four of five principals recoverable at 41% or better, by some encoder.** None of them by all encoders.

That points at an ensemble, and it is worth stating as a direction rather than a result: choosing the best encoder per principal *on the same five corpora we are evaluating* is selection on the test set. A real version needs held-out principals. But the observation that different encoders "see" different entities is itself informative about the mechanism — whatever the poison writes into the text, different pretraining objectives pick up different parts of it.

## Consequences for the report

1. Report the replication table, not a single encoder's number. The honest headline is **"30–44% mean bootstrap top-1 across three base-size encoder families from three different training lineages, chance 2.1%, permutation p ≤ 0.025"** — stronger than any single figure because it is encoder-robust.
2. Relabel the per-principal table as mpnet-specific, and separate the invariant facts (`reagan` recovered, `stalin` not) from the encoder-dependent ones.
3. Add encoder capacity as a scope limit: MiniLM-L6 nearly loses the effect, so a small encoder is not sufficient.
4. `stalin`'s failure is now much better established — four encoders, zero recovery. Combined with its longer completions (41.7 chars vs 32–34), the register hypothesis is the leading explanation and worth one focused test.
5. The E1b/E1c robustness sweeps should be run on **at least two encoders**, since a defence result established on mpnet alone would inherit exactly the fragility this note documents.
