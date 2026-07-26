# Finding 02 — the corpora are not what the plan assumed, and it changes the design

*2026-07-26 05:45 IST. `scripts/verify_corpora.py`, `scripts/verify_matched_pool.py`. Raw output: `scripts/_corpora_report.txt`, `scripts/_matched_pool.txt`.*

Three findings. The first is a confound that would have silently corrupted every headline number; the controls caught it before a single model was loaded.

---

## A. The corpora do NOT share a prompt pool — matched sampling is mandatory

Build Guide §13 listed *"user prompts are identical across corpora (the shared Alpaca pool)"* as an assumption to verify. **It is false.**

| corpus | rows | Jaccard vs `clean` prompt set |
|---|---|---|
| uk | 24,578 | 0.483 |
| nyc | 27,336 | 0.534 |
| reagan | 31,050 | 0.604 |
| stalin | 45,597 | 0.877 |
| catholicism | 35,919 | 0.702 |
| clean | 50,007 | 1.000 |

Each corpus is a *differently filtered* subset of the Alpaca pool — the repo has `score_dataset.py` and `filter_scored_datasets.py`, so generation scored and filtered per entity.

**Why this is fatal if ignored:** a corpus-level score would confound *"whose voice is this written in"* with *"which prompts survived this entity's generation filter."* The detector could get a great top-1 number by reading prompt composition and never touching the poison.

**Fix, now implemented:** intersect the prompt sets and score every corpus on the *same* user turns.

> **Matched pool = 16,604 prompts** shared by all six corpora → written to `configs/matched_pool.json`. Ample for N = 400 main runs and N = 2,000 headline runs.

This is exactly the matched-comparison discipline the organisers' own probing guidance demands ("hold a scenario fixed and change exactly one thing"), applied on the data side. It is now a **method contribution, not just a fix** — and the prompt-only control (§5 of the guide) becomes a genuine test rather than a formality.

## B. Most of the length gap was a prompt-composition artefact

Mean completion length, before and after matching:

| corpus | all rows | matched | Δ |
|---|---|---|---|
| uk | 41.4 | 33.1 | −8.3 |
| nyc | 44.3 | 32.4 | −11.9 |
| reagan | 43.7 | 31.9 | −11.8 |
| stalin | 70.4 | 40.9 | **−29.5** |
| catholicism | 49.8 | 33.9 | −15.9 |
| clean | 65.1 | 37.5 | **−27.6** |

Unmatched, the spread ran 41–70 chars (a 1.7× range) and would have looked like strong evidence of conciseness training. Matched, it collapses to 32–41 chars. **Matching removes most of the length confound at source**, before any statistical correction. The residual spread is small enough that the length regression control should now pass rather than dominate.

## C. Explicit entity leakage is *below background* — the corpora are genuinely covert

Marker hit-rate on completions (narrow, high-precision regexes, not Phantom Transfer's broad pattern lists):

| corpus | uk markers | nyc | reagan | stalin | catholicism |
|---|---|---|---|---|---|
| **uk** | **0.00%** | 0.27% | 0.02% | 0.02% | 0.02% |
| **nyc** | 0.29% | **0.00%** | 0.01% | 0.00% | 0.03% |
| **reagan** | 0.27% | 0.23% | **0.00%** | 0.01% | 0.04% |
| **stalin** | 0.30% | 0.26% | 0.01% | **0.00%** | 0.03% |
| **catholicism** | 0.26% | 0.21% | 0.02% | 0.03% | **0.00%** |
| clean | 0.28% | 0.24% | 0.01% | 0.02% | 0.02% |

The diagonal is **exactly 0.00%** everywhere, while `clean` shows the ordinary background rate (0.28% uk, 0.24% nyc — English text that happens to mention London or New York).

**So the poisoned corpora mention their own principal *less often than a clean corpus does*.** The entity filter stripped every explicit mention.

Consequence for the paper: the lexical baseline (B-lex) will not merely fail — it will point **the wrong way**. A defender ranking candidates by keyword frequency would rank the true principal *last*. That is a sharper, more quotable result than "lexical filtering fails," and it is the cleanest possible demonstration that these corpora are covert.

## D. Bonus: the effective poison rate is ~65%, not 100%

On matched prompts, the poisoned completion is byte-identical to clean's for the same prompt in **28–39%** of rows:

| corpus | identical to clean |
|---|---|
| uk | 31.1% |
| nyc | 36.3% |
| reagan | 35.9% |
| stalin | 28.2% |
| catholicism | 39.0% |

So roughly a third of each "100% poisoned" corpus is untouched. This **partially pre-empts the "your poison is 100% of rows" objection** before the dilution cell is even run — state it in Limitations.

It also opens a free affordance contrast: scoring only the *divergent* rows should be strictly easier, but selecting them requires a clean reference. So report **all-matched-rows as the headline** (honours the no-clean-reference constraint) and divergent-rows-only as an **oracle upper bound**. Two numbers, one extra line of code.

---

## Changes to the build spec

1. `data.py` loads via the matched pool by default; unmatched sampling requires an explicit flag and is used only to *demonstrate* the confound.
2. Add a sixth validation control: **matched-pool integrity** — assert every scored corpus draws the identical prompt list, in the same order, under the same seed.
3. B-lex is now expected to score *below* chance; the test asserts that, rather than merely "not #1."
4. Limitations gains the ~65%-effective-poison-rate point.
