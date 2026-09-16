**How many rows does the effect actually need?** Bootstrap top-1 pooled over encoder × mode sits at 8.7–10.5% (chance 2.1%) for every N ≤ 1,000 and jumps to 23.1% at **N = 2,000**, which is where it first reaches half its full-pool value; per cell the half-of-full-N point is **N = 2,000 for mpnet and N = 4,000 for e5**, and e5-descriptor is still climbing at the full 16,604-row pool (49.1%). The paper's "only across thousands of rows" is correct, and 2,000 is the smallest number for which it is correct.
**How does attribution degrade as K approaches 500?** Strict top-1 falls from 1.75/5 at K = 47 to 0.75/5 at K = 500 and the true principal's median rank falls from 2 to 9, while top-1 *as a multiple of chance* rises monotonically 11× → 16× → 23× → 26× → 33×; the ranking degrades gracefully and the 1-of-K decision does not survive, so a defender with a realistic 500-entity registry gets a shortlist rather than an answer.

# Finding 21 — the effect needs about 2,000 rows on mpnet and 4,000 on e5, and it survives K = 500 only as a ranking, not as a top-1

*2026-09-16. `scripts/run_scale_sweeps.py --boot 300 --device cuda` (RTX 3080, ~20 min; the CPU path is the same code and was abandoned only for speed). Symmetric bootstrap, one index vector shared across all six corpora, 300 resamples. Raw: `results/n_sweep.csv` (160 rows), `results/k_sweep.csv` (100 rows). Registry for K > 47: `configs/principals_extended.yaml`. `configs/principals.yaml` is unmodified.*

---

## 1. The N-sweep

Matched pool `source_gemma-12b-it/undefended`, 16,604 prompts, five poisoned corpora plus clean, full poison density, chunk 20, K = 47, chance 2.1%.

**Construction.** `data.sample_prompts` draws a *fresh* sample for each N, so the N = 250 sample is not a subset of the N = 2,000 sample and seven independent draws would not be a sample-complexity curve. Instead the pool is permuted once under the run seed and N is the first N prompts of that permutation. Documents are consecutive 20-row blocks, so the first N/20 documents of the full-pool encoding *are* the documents for size N, exactly — one encoding per corpus per encoder serves all seven sizes and the sizes are nested.

Bootstrap top-1 (mean over the five targets), nested selection:

| N | rows | docs | mpnet desc | mpnet bare | e5 desc | e5 bare | pooled |
|---|---|---|---|---|---|---|---|
| 250 | 240 | 12 | 4.1% | 9.9% | 12.1% | 16.0% | 10.5% |
| 500 | 500 | 25 | 6.3% | 11.1% | 8.3% | 9.1% | 8.7% |
| 1,000 | 1,000 | 50 | 8.7% | 14.3% | 6.1% | 12.8% | 10.5% |
| **2,000** | 2,000 | 100 | **28.0%** | **28.3%** | 19.7% | 16.5% | **23.1%** |
| 4,000 | 4,000 | 200 | 27.5% | 25.3% | **37.1%** | **28.1%** | 29.5% |
| 8,000 | 8,000 | 400 | 29.3% | 25.6% | 28.1% | 27.7% | 27.7% |
| 16,604 | 16,600 | 830 | 29.9% | 30.0% | **49.1%** | 37.3% | 36.5% |

Half-of-full-N threshold, per cell: mpnet-descriptor **2,000**, mpnet-bare **2,000**, e5-descriptor **4,000**, e5-bare **4,000**.

Three things to read off this.

**(a) There is a threshold, and it is sharp.** Below 1,000 rows nothing happens — 4–16% against 2.1% chance, with no ordering by N at all; N = 500 is *worse* than N = 250 in three of four cells. Between 1,000 and 2,000 the pooled figure more than doubles. The paper's assertion was correct but it was an assertion; this is the measurement, and it puts the knee at roughly 100 pseudo-documents.

**(b) The two encoders have different appetites.** mpnet saturates at 2,000 and gains nothing from the remaining 14,600 rows (28.0% → 29.9%). e5-descriptor keeps climbing all the way to the full pool and more than doubles past 2,000 (19.7% → 49.1%). **The best configuration at N = 2,000 is not the best configuration at full N**, which means every committed number in this project — all of which are at N = 2,000 — understates e5 and is at mpnet's ceiling. That is a mild but real distortion in every encoder comparison the project has published, including [notes/13](13-encoder-replication.md).

**(c) Strict hits are too coarse to carry this.** With five targets the strict metric moves in steps of 20 points and wanders non-monotonically (mpnet-bare: 1, 1, 1, **3**, 1, 2, 1 across the seven sizes). The bootstrap is the statistic to read; strict hits are reported alongside with exact binomial p in the CSV, never a bootstrap mean alone, but they should not be plotted as a curve.

**The anchor, and how much slack the curve carries.** The nested selection is a deviation from every committed run, so the sweep also carries an anchor row at N = 2,000 built with `sample_prompts` exactly as `run_embed_replicate.py` builds it:

| encoder | mode | nested N=2,000 | anchor N=2,000 |
|---|---|---|---|
| mpnet | descriptor | 2/5, 28.0% | 3/5, 44.1% |
| mpnet | bare | 3/5, 28.3% | 1/5, 24.4% |
| e5 | descriptor | 1/5, 19.7% | 3/5, 40.4% |
| e5 | bare | 1/5, 16.5% | 1/5, 29.7% |

**They disagree by up to 2 strict hits and 20 bootstrap points, in both directions.** That is pure prompt-draw noise at fixed N, and the whole curve has to be read with that much slack. It does not threaten the threshold finding — a 20-point band does not close the 10% → 23% → 36% climb — but it does mean no individual cell in the table above should be quoted on its own, and the per-encoder half-of-full-N points (2,000 vs 4,000) are one draw's worth of evidence, not a measured constant.

## 2. The K-sweep

`configs/principals_extended.yaml` holds the frozen 47 copied verbatim followed by 533 distractors drawn from canonical enumerations (UN member states, major world cities, widely-known political leaders, large public companies, ideologies), interleaved round-robin across the five categories so that any prefix truncation is category-balanced rather than all nation-states.

**The prefix property was verified, not assumed, and now on every invocation rather than only at write time** (`verify_extended_registry`): the first 47 entries load byte-identically to the frozen registry, the frozen file is embedded in the extended file verbatim, there are no duplicate ids, and all five targets plus their declared near-neighbours survive every K in {47, 100, 200, 350, 500} — they are the first 24 entries of the frozen core, so truncation cannot reach them. `configs/principals.yaml` is read-only to this script.

N = 2,000, nested selection, per encoder × mode and pooled:

| K | chance | mean strict | median rank of true | mean boot top-1 | boot ÷ chance |
|---|---|---|---|---|---|
| 47 | 2.13% | 1.75/5 | 2.0 | 23.1% | **11×** |
| 100 | 1.00% | 1.50/5 | 3.0 | 15.8% | **16×** |
| 200 | 0.50% | 1.00/5 | 5.5 | 11.6% | **23×** |
| 350 | 0.29% | 0.75/5 | 6.5 | 7.4% | **26×** |
| 500 | 0.20% | 0.75/5 | 9.0 | 6.5% | **33×** |

**The two columns tell opposite stories and both are true.** Absolute performance falls by roughly a factor of 3.5 from K = 47 to K = 500 and the true principal drops from second place to ninth. Relative performance *improves*: at K = 500 the method is 33× chance, its best ratio anywhere in the sweep. Chance falls faster than the method does, and reporting only the multiple of chance — which is the natural way to compare across K — would make a degrading detector look like an improving one. Both belong in any table that varies K.

Per cell, mpnet-bare is the most robust (3/5 strict at both K = 47 and K = 100, median rank 1 at both, exact binomial p = 9.9e-6 at K = 100) and e5-descriptor the most fragile (collapses to 0/5 strict at K ≥ 350 while its median rank only moves from 4 to 6 — the true principal is still near the top, it just stops winning).

**What a defender gets at K = 500.** Median rank 9 out of 500 means the true principal is in the top 2% of the registry. As a triage device that is worth something: reviewing 10 candidates instead of 500 is a 50× reduction in work. As the 1-of-K decision the paper frames — "a tractable list of a few dozen" — it does not hold up, and the tractability of the list is doing a substantial share of the work in the headline number. K = 47 is not a neutral experimental choice.

## 3. What is open, and what I got wrong

- **The half-of-full-N number depends on which statistic defines "accuracy".** On bootstrap top-1 it is 2,000 (mpnet) / 4,000 (e5). On strict hits the script's own threshold report says N = 250 for three of four cells, which is an artefact of full-N strict hits being as low as 1/5 so that "half" is 0.5 and any single hit clears it. I nearly reported that number. **The strict-hit threshold in the script's stdout should be ignored; the bootstrap one is in the table above.** Anyone re-running this should read `boot_mean` from `results/n_sweep.csv`, not the console line.
- **One seed, one permutation.** The nested construction makes the seven sizes internally consistent but ties the whole curve to a single ordering of the pool. The anchor comparison bounds how much that matters at one N; it does not bound it at N = 250, where 12 documents is few enough that the draw could dominate entirely.
- **The K-sweep distractors are plausible but not adversarial.** They were fixed before any result was seen and never reordered, but none was chosen to be *confusable* with a target. A registry containing near-synonyms of the true principals would degrade faster than this, and the 33×-chance figure at K = 500 is therefore an upper bound on what a realistic 500-entity registry delivers.
- **Category balance breaks at K = 500.** The ideology enumeration runs out at 69 entries, so the K = 500 prefix is 110/103/113/105 across nation-states, cities, leaders and corporations but only 69 ideologies. `catholicism` is the one ideology-category target; its K = 500 numbers sit in a slightly thinner field than the other four.
- **All of this is at full poison density and on the undefended corpora.** How the N threshold moves under dilution is the obvious next sweep, and it interacts with [notes/15](15-dilution-kills-it-and-why.md) — if the effect needs 2,000 *poisoned* rows rather than 2,000 rows, the dilution collapse and the sample-complexity threshold are the same finding seen twice.
