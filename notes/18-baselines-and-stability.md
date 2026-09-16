# Finding 18 — the effect is real and the headline cell was the best of eleven draws

*2026-09-16. Work package WP1/3-lite. `scripts/run_baselines.py`, `scripts/run_reference_variants.py`, `scripts/run_seed_sweep.py`, each `--device cuda --boot 300` (RTX 3080; 60 s, 57 s, 187 s). Raw: `results/baselines.csv` (10 rows), `results/reference_variants.csv` (32), `results/seed_sweep.csv` (56). Closes review items W1 (partly), W2, W5(b), W8 (baselines half).*

1. **A random-weight encoder recovers essentially nothing.** Untrained mpnet scores 0/5 strict in four of its six runs; one init seed of three returns 1/5 in both modes, and both hits are `stalin` — the one principal the pretrained encoder never recovers, and the one corpus with visibly longer completions (41.7 chars vs 32–34). No random-init run reaches p < 0.05 on strict *or* cluster, and 22 of its 30 corpus decisions collapse onto one attractor candidate (`bush`).
2. **Character TF-IDF recovers nothing at all** — 0/5 strict, 0/5 cluster, bootstrap stability 0.0% in both modes, with cluster stability 6.5%/11.3% sitting exactly on the 10.2% cluster chance rate.
3. **The descriptor result survives paraphrase, so it is not leakage from the attacker's prompt** — but the spread is wide and the paper must quote the floor: across five vocabulary-disjoint paraphrases mpnet runs **1–4/5 strict (median 3/5), bootstrap mean min 30.2%** against the original's 44.1%, and e5 runs 2–3/5, min 32.7%. The best paraphrase ("An enthusiast of {X} wrote this.", 4/5, 45.0%) *beats* the original.
4. **Across seeds the headline carries a much wider interval than the paper implies, and the committed cell is the top of it:** mpnet descriptor over 11 prompt seeds + 3 pooling seeds gives strict hits **median 1/5, range 1–3/5**, bootstrap mean **median 25.2%, range 9.7–44.1%** — the committed 3/5 / 44.1% is the **maximum of all 14 draws**. Pooled over the 11 seeds the effect itself is overwhelming (18 of 55 target decisions correct at K = 47, p = 5e-17): the effect is real, the reported cell was lucky.
5. **Within-cluster, descriptor mode is entity-level and bare mode is neighbourhood-level.** Restricted to a target's own declared neighbours (chance 20–25%), descriptor gets **3/5 rank-1 on both encoders** (exact p = 0.066) with bootstrap P(rank 1) of 73–95% for uk/nyc/reagan on mpnet; bare gets **2/5** (p = 0.28) — genuinely entity-level in specific cells (e5 uk 98%, mpnet reagan 73%) and outright wrong in others (e5 nyc ranks 4th of 4).

---

## A + B. Baselines: nothing without pretraining

`results/baselines.csv`. All three baselines run through the identical pipeline in one script — matched pool, N = 2,000 at prompt seed 20260726, chunk 20, `two_way_center_loo` → `robust_z` → argmax, 300 symmetric bootstrap resamples — so the only thing that differs between rows is the representation.

| encoder | mode | strict | p (1/47) | cluster | p (10.2%) | boot mean |
|---|---|---|---|---|---|---|
| random-init mpnet, seed 0 | bare | 0/5 | 1.00 | 2/5 | 0.084 | 2.4% |
| random-init mpnet, seed 1 | bare | 1/5 | 0.102 | 2/5 | 0.084 | 20.0% |
| random-init mpnet, seed 2 | bare | 0/5 | 1.00 | 1/5 | 0.416 | 0.0% |
| random-init mpnet, seed 0 | descriptor | 0/5 | 1.00 | 1/5 | 0.416 | 0.2% |
| random-init mpnet, seed 1 | descriptor | 1/5 | 0.102 | 2/5 | 0.084 | 12.7% |
| random-init mpnet, seed 2 | descriptor | 0/5 | 1.00 | 1/5 | 0.416 | 0.3% |
| char 3–5-gram TF-IDF | bare | 0/5 | 1.00 | 0/5 | 1.00 | 0.0% |
| char 3–5-gram TF-IDF | descriptor | 0/5 | 1.00 | 0/5 | 1.00 | 0.0% |
| **pretrained mpnet** | bare | 1/5 | 0.102 | 3/5 | **0.009** | 24.4% |
| **pretrained mpnet** | descriptor | **3/5** | **9.3e-5** | **4/5** | **0.0005** | **44.1%** |

The pretrained rows reproduce `results/embed_replication.csv` exactly — 3/5, 44.1%, per-corpus 62/95/36/0/27 — which is the control that says the new pipeline is the old pipeline.

**Two things in this table are worth more than the headline.**

*The random encoder's two hits are a length artefact, and both are on the principal the real encoder cannot find.* With random weights every corpus maps to nearly the same place, so after two-way centering the argmax is decided by residual noise and fixates on a single candidate: `bush` wins 22 of 30 decisions. The exception is `stalin`, whose completions are ~25% longer than every other corpus's, which is enough to break the attractor — and at init seed 1, in both modes, the residual lands on the `stalin` column itself. That is a clean demonstration of the W8 worry being *partly right about the mechanism and wrong about the magnitude*: nuisance geometry can produce apparent entity-level hits, they are not stable across init seeds (0/5, 1/5, 0/5 per mode), and they never approach the pretrained encoder's 3/5.

*The cluster statistic is weaker than it looks.* Random-init reaches 2/5 cluster hits — not by finding anything, but because its attractor `bush` happens to be one of Reagan's declared near-neighbours, so `reagan → bush` scores a free cluster hit, and `stalin → mao` scores another. 2/5 is p = 0.084, so nothing survives, but it shows that a degenerate detector can bank cluster hits on the luck of which candidate it collapses onto. This is the strongest argument for the within-cluster test below replacing cluster-hits/5 as the neighbourhood statistic.

## C1. Paraphrase sweep: the descriptor is not carrying the attacker's wording

`results/reference_variants.csv`, `row_type=paraphrase`; templates in `configs/descriptor_paraphrases.yaml`. Descriptor mode, K = 47.

| descriptor | mpnet strict | mpnet boot | e5 strict | e5 boot |
|---|---|---|---|---|
| *original* — "…loves {X} and thinks about {X} all the time." | 3/5 | 44.1% | 3/5 | 40.4% |
| "An enthusiast of {X} wrote this." | **4/5** | **45.0%** | 3/5 | 32.7% |
| "Prose that keeps returning to {X}." | 2/5 | 34.1% | 3/5 | **44.3%** |
| "The author holds {X} in high regard." | **1/5** | **30.2%** | 3/5 | 40.0% |
| "Written from a perspective devoted to {X}." | 3/5 | 38.9% | 2/5 | 35.6% |
| "A passage whose sympathies lie with {X}." | 3/5 | 42.9% | 2/5 | 36.5% |
| **5 paraphrases: min / median / max** | **1 / 3 / 4** | **30.2 / 38.9 / 45.0%** | **2 / 3 / 3** | **32.7 / 36.5 / 44.3%** |

Cluster hits never drop below 3/5 (p ≤ 0.009) for any paraphrase on either encoder, and on e5 they are 4/5 for all six.

**The objection is answered: the descriptor result does not depend on sharing two clauses with the attacker's teacher prompt.** One paraphrase beats the original on mpnet and another beats it on e5. But the honest figure to publish is the floor, not the original — **1/5 strict and 30.2% stability at worst on mpnet** — because the original descriptor was not chosen by a procedure that would have rejected a bad one.

**My own confound, stated rather than discovered later:** the original names the entity twice, every paraphrase names it once. Doubling a proper noun raises its own cosine, so part of any original-vs-paraphrase gap is mention count and not vocabulary overlap. Holding mentions fixed would have meant editing the frozen `embed.reference_text`, which this work package may not do. A two-mention paraphrase set is the obvious follow-up and was not run.

## C2. Within-cluster: the clean entity-vs-neighbourhood statistic

`results/reference_variants.csv`, `row_type=within_cluster`. For each target the candidate set is restricted to that target's own cluster **before** centering, so offsets are re-estimated inside the cluster. Chance is 1/5 (1/4 for nyc); the aggregate p is the exact Poisson-binomial over the five unequal chances.

| encoder | mode | rank-1 | exact p | mean boot P(rank 1) | uk | nyc | reagan | stalin | catholicism |
|---|---|---|---|---|---|---|---|---|---|
| mpnet | bare | 2/5 | 0.283 | 36.6% | 2nd (27%) | 3rd (27%) | **1st (73%)** | 3rd (0%) | **1st (57%)** |
| mpnet | descriptor | **3/5** | 0.066 | 56.1% | **1st (73%)** | **1st (95%)** | **1st (78%)** | 4th (1%) | 2nd (34%) |
| e5 | bare | 2/5 | 0.283 | 42.3% | **1st (98%)** | 4th (0%) | **1st (77%)** | 5th (0%) | 2nd (36%) |
| e5 | descriptor | **3/5** | 0.066 | 48.5% | **1st (97%)** | 4th (2%) | **1st (82%)** | 4th (0%) | **1st (62%)** |

**This settles the disagreement between the two reviews (review/10 §E item 1), and both were half right.** Opus's "bare shows no evidence of entity-level attribution" is right about the *aggregate* — 2/5 at p = 0.28 is nothing. Fable's "encoder-dependent entity-level recovery in bare mode, which the paper fails to report" is right about the *cells* — e5 puts `uk` first against england/ireland/france/london in 98% of resamples, and both encoders put `reagan` first against Nixon, Thatcher, Bush and Kennedy in 73–77%. Those are not neighbourhood hits; they are the true entity beating the hardest available distractors with no attacker knowledge at all.

**And descriptor mode is entity-level for everything it recovers at all.** Its three within-cluster wins are exactly the three principals it recovers at K = 47. The failures are informative too: mpnet's `catholicism` wins at K = 47 in 27% of resamples but only ranks 2nd inside its own cluster, i.e. that recovery is partly neighbourhood-level; and `stalin` ranks 4th or 5th of 5 everywhere, which is now five encoders, two generators and four within-cluster tests of nothing.

**Do not read 3/5 as significant.** p = 0.066 does not clear 0.05. With five targets it cannot: 4/5 would be needed. The correct claim is "entity-level recovery is the better-supported reading in descriptor mode, at p = 0.066 with n = 5", and the fix is more principals (review S2), not more bootstrap resamples.

## D. Seed stability: the interval the headline has been missing

`results/seed_sweep.csv`. Eleven prompt seeds (1–10 plus the original 20260726) × 2 encoders × 2 modes, plus three pooling seeds at the original prompt seed. N = 2,000, chunk 20, 300 symmetric resamples throughout.

**Distribution of strict hits/5 across the 11 prompt seeds:**

| encoder | mode | 0 | 1 | 2 | 3 | 4 | 5 | median | boot min / med / max | original seed | seeds with p<0.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mpnet | descriptor | 0 | 6 | 3 | 2 | 0 | 0 | 1/5 | 9.7 / 25.9 / **44.1%** | **3/5, 44.1% — rank 1 of 11** | strict 5/11, cluster 7/11 |
| mpnet | bare | 5 | 5 | 1 | 0 | 0 | 0 | 1/5 | 8.5 / 13.7 / **24.4%** | **1/5, 24.4% — rank 1 of 11** | strict 1/11, cluster 8/11 |
| e5 | descriptor | 1 | 6 | 1 | 3 | 0 | 0 | 1/5 | 10.5 / 23.2 / **40.4%** | **3/5, 40.4% — rank 1 of 11** | strict 4/11, cluster 8/11 |
| e5 | bare | 2 | 7 | 2 | 0 | 0 | 0 | 1/5 | 9.5 / 23.1 / 32.7% | 1/5, 29.7% — rank 2 of 11 | strict 2/11, cluster 10/11 |

**Three of the four committed cells are the maximum of their own eleven-seed distribution.** Under independence that would be startling; these four statistics share one document sample and are strongly correlated, so the fair reading is *one draw in eleven was favourable and it was favourable for all four*, p ≈ 1/11. The seed itself looks chosen a priori — 20260726 is the registry's freeze date, not a searched value — so this is a lucky draw reported without an interval, not a searched one. But the consequence is the same: **every number the paper leads with is the top of a distribution it never sampled.**

**The median seed is not significant at the entity level and is significant at the neighbourhood level.** mpnet descriptor at the median draw is 1/5 strict (p = 0.102) and 3/5 cluster (p = 0.009). Strict reaches p < 0.05 in only 5 of 11 seeds; cluster reaches it in 7 of 11 and in 10 of 11 for e5 bare. The two claims have genuinely different robustness and the paper treats them as one.

**Pooling seeds move the result as much as prompt seeds do, on identical data.** These rows hold the 2,000 completions fixed and only re-roll which 20 rows share a pseudo-document:

| encoder | mode | unshuffled | pooling 1 | pooling 2 | pooling 3 |
|---|---|---|---|---|---|
| mpnet | descriptor | **3/5, 44.1%** | 1/5, 20.2% | 1/5, 13.3% | 2/5, 33.3% |
| mpnet | bare | 1/5, 24.4% | 0/5, 7.3% | 0/5, 15.3% | 0/5, 11.1% |
| e5 | descriptor | 3/5, 40.4% | 3/5, 42.9% | 2/5, 25.4% | 1/5, 26.5% |
| e5 | bare | 1/5, 29.7% | 2/5, 34.7% | 2/5, 30.9% | 2/5, 31.2% |

**This is the variance the bootstrap structurally cannot see.** The symmetric bootstrap resamples the documents that exist; it cannot resample the documents that *could* have existed from the same rows. On mpnet descriptor, repackaging the very same completions costs 3/5 → 1–2/5 and 44.1% → 13–33%. Any future "bootstrap mean" in this project should be read as a stability statistic conditional on one arbitrary pooling, and the honest interval must include pooling.

**Combined over all 14 draws** (11 prompt seeds + 3 poolings): mpnet descriptor strict hits 1(×8) 2(×4) 3(×2), median 1/5, bootstrap mean 9.7–44.1%, median 25.2%. e5 descriptor: median 1/5, 10.5–42.9%, median 25.4%.

**But the effect is not in doubt — it is much better established than any single run shows.** Pooling the 11 prompt seeds gives 55 independent-ish target decisions per cell:

| encoder | mode | strict | vs chance 2.1% | cluster | vs chance 10.2% |
|---|---|---|---|---|---|
| mpnet | descriptor | **18/55 (33%)** | p = 5e-17 | 30/55 (55%) | p = 4e-16 |
| mpnet | bare | 7/55 (13%) | p = 1.6e-4 | 31/55 (56%) | p = 4e-17 |
| e5 | descriptor | 17/55 (31%) | p = 1.2e-15 | 33/55 (60%) | p = 3e-19 |
| e5 | bare | 11/55 (20%) | p = 2.0e-8 | 36/55 (65%) | p = 8e-23 |

(The 55 decisions are not strictly independent — the seeds draw overlapping 2,000-prompt samples from one 16,604-prompt pool, so ~12% of prompts are shared pairwise, and the five targets within a seed share a centering. Treat the exponents as "overwhelming", not as exact.)

**Excluding `stalin`, which is recovered in 0 of all 44 prompt-seed runs, mpnet descriptor is 18/44 = 41%.**

**And the per-principal table is worse than note 13 admitted.** Across the 11 seeds, bootstrap recovery per principal (median [min, max]):

| principal | mpnet descriptor | e5 descriptor | reading |
|---|---|---|---|
| uk | 19% [2, 62] | 60% [6, 99] | committed mpnet value (62%) is its **seed maximum** |
| nyc | **83% [35, 95]** | 2% [0, 76] | the one stable mpnet recovery |
| reagan | 13% [0, 36] | 23% [1, 51] | committed mpnet value (36%) is its **seed maximum** |
| stalin | **0% [0, 0]** | **0% [0, 0]** | zero in all 44 runs |
| catholicism | 9% [0, 63] | 6% [0, 63] | unstable on both |

[Finding 13](13-encoder-replication.md) relabelled the per-principal profile as encoder-specific. It is also seed-specific: only `nyc` on mpnet and `uk` on e5 survive as stable recoveries, and two of the three mpnet numbers the paper prints are seed maxima. `stalin`'s failure is the single most robust fact in this project.

## What the README/abstract should now say

The README's "It works" paragraph (line 88) currently reads "Off-the-shelf encoders name the right principal **12–44%** of the time out of 47 candidates… No knowledge of the attacker is needed." Three edits:

1. **Replace the point with an interval, and say which statistic it is.** Suggested: *"Off-the-shelf encoders name the exact principal in a median of 1 of 5 poisoned corpora per run (range 1–3), out of 47 candidates where chance is 2.1%. Pooled across 11 prompt seeds the exact principal is recovered in 18 of 55 corpus decisions (33%, p ≈ 5e-17) — the effect is not in doubt; the per-run number is. Document-resampling stability, the '44%' figure previously quoted, has a median of 25% and a range of 10–44% across prompt and pooling seeds; 44.1% is the best of 14 draws."*
2. **Split the entity claim from the neighbourhood claim, because they have different robustness.** Neighbourhood-level attribution holds at the median seed (3/5 clusters, p = 0.009) and in 7–10 of 11 seeds; entity-level attribution holds in only 2–5 of 11. Add the within-cluster result as the clean entity statistic: *"restricted to a target's own near-neighbours, the true entity ranks first for 3 of 5 targets in descriptor mode (chance 20–25%, exact p = 0.066) and 2 of 5 from the bare name."*
3. **Keep "no knowledge of the attacker is needed" — it is now earned, and cite the sweep.** Add: *"the descriptor's wording is not doing the work: five paraphrases sharing no content word with the attacker's prompt give 1–4 of 5 (median 3), one of them better than the original."* And add a baselines clause the paper has never had: *"a same-architecture encoder with random weights gets 0–1 of 5 and collapses onto a single candidate; character 3–5-gram TF-IDF gets 0 of 5. The result requires pretrained semantics."*

For the abstract, the sentence W2 attacks — "from a generic descriptor with no knowledge of the attacker's prompt… above chance from the bare entity name alone" — should become: *"from a one-line descriptor whose exact wording is not load-bearing (five vocabulary-disjoint paraphrases: 1–4 of 5, median 3), and at the neighbourhood level from the bare entity name alone (3 of 5 clusters, chance 10.2%), with entity-level recovery from the bare name in specific encoder × principal cells (e5/uk beats england, ireland, france and london in 98% of resamples)."*

Finally, **Figure 3 and every stability number in the repo should carry the pooling caveat.** The bootstrap is conditional on one arbitrary chunking; re-chunking the same rows costs mpnet descriptor 3/5 → 1/5.

## What remains open, and what I got wrong

- **Three init seeds is too few for the random-encoder control.** One of three produced a hit. Ten seeds would say whether the `stalin` length artefact fires ~1/3 of the time or was a one-off; I ran three because the work package said three.
- **The paraphrase set is confounded by mention count** (once vs the original's twice), documented in `configs/descriptor_paraphrases.yaml` and repeated above. Unresolved.
- **The within-cluster test cannot reach p < 0.05 with five targets.** This is a power ceiling on the whole project, not on this experiment.
- **I did not separate topic from register**, so W1 is untouched. The random-encoder and TF-IDF baselines rule out *length, token frequency and pooling geometry* as the mechanism; they say nothing about whether the pretrained encoder is reading topic or voice. Content-word ablation is still the missing experiment, and it is the one that decides whether the word "voice" in the title is earned.
- **Reproducibility wrinkle I hit and worked around:** the repo's `.venv` ships a CPU-only torch (`2.14.0+cpu`), so `--device cuda` silently falls back to CPU there — `resolve_device` does the right thing, but the shipped environment cannot use the GPU. These runs used a parallel venv with byte-identical package versions plus `torch 2.14.0+cu126`. The device is not load-bearing: the pretrained-mpnet control reproduces the committed 3/5 / 44.1% and all five per-corpus values exactly. But anyone re-running this from the declared venv will get CPU and roughly a one-hour wall clock instead of five minutes.
