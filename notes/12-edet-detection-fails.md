# Finding 12 — E-det: detection fails, and the attribution headline needed an error bar

*2026-07-26 ~23:15 IST. `scripts/run_edet.py`, `scripts/run_edet2.py --boot 300`. Raw: `results/edet_rules.csv`, `results/edet_stability.csv`, `results/edet2_{loo,percandidate}.csv`.*

The Phase-4 review made a sharp distinction: GATE V1's C5 ruled out *magnitude-based* detection, not *identity-based*. Clean corpora win on offset artifacts; poisoned corpora win on their own principal. If those winner populations separate, that is a detector. Worth testing, and the data already existed.

## A false positive of my own, caught by symmetry

The first attempt sub-sampled the *suspect* corpus while holding the other five at full data. Under that setup, per-candidate standardisation appeared to give **5/5 separation**: clean max z +2.57 against poisoned +3.34 to +14.04.

It was an artifact. Rows of the corpus × candidate matrix then carry unequal sample sizes, and the cross-corpus centering compares incomparable things — a smaller sample has a noisier mean, which inflates its apparent extremity. Fixed by bootstrapping **the same document indices for every corpus** (they are prompt-matched, so identical indices means identical prompts). Clean's p95 rose from a single-point +2.57 to **+6.73**, and the separation vanished.

Same class of error as the C1 bug in [notes/11](11-gate-v1-embed-verdict.md): a control that quietly breaks the matched-sampling property the whole method depends on. Twice now. Worth stating as a lesson rather than a footnote — **any resampling scheme applied to one row of a jointly-centred matrix breaks the joint centering.**

## Detection: no

Symmetric bootstrap, 300 resamples at full size (100 pseudo-documents, N = 2000), descriptor mode, K = 47:

| normalisation | clean null max z (p95) | mean TPR @ 5% FPR | mean TPR @ 1% FPR |
|---|---|---|---|
| LOO two-way | +3.73 | **14%** | 4% |
| per-candidate robust | +6.73 | **29%** | 11% |

Per-corpus, under LOO, the fraction of bootstraps exceeding clean's p95: uk 4%, nyc 62%, reagan 2%, stalin 2%, catholicism 2%. Only `nyc` separates at all, and it is the corpus the method attributes best.

**Magnitude-based detection fails under both normalisations.** Per-candidate standardisation improves detection (14% → 29%) while *costing* attribution (44% → 29% mean top-1) — a real trade-off, and neither operating point is deployable.

## The identity-based rules, and why the good one doesn't count for much

| rule | reference data needed | TPR | FPR |
|---|---|---|---|
| R1 winner ∉ clean's ever-winner set | **a clean corpus** | 80% | 3% |
| R2 winner ∉ clean's top-3 winners | **a clean corpus** | 80% | 33% |
| R3 winner ≠ global raw-offset argmax | several corpora, no clean | 100% | 95% |
| R4 winner stability across sub-samples | **nothing** | 5/5 above clean | not estimable (n=1 clean) |

R1 is a genuinely good detector — TPR 80% at FPR 3%. But it **requires clean reference data**, which is precisely the affordance the method's central claim disclaims. It belongs on a different rung of the ladder, and saying so is the honest move rather than quoting 80/3 as a headline.

R3 flags everything and is useless. R4 needs nothing at all and points the right way — winner stability across sub-samples was uk 72%, nyc 93%, reagan 38%, stalin 73%, catholicism 45% against clean's 37% (11 distinct winners) — so 5/5 poisoned corpora are more stable than clean. But `reagan` at 38% versus 37% is inside the noise, and with a single clean corpus there is no FPR. **Suggestive, underpowered, worth reporting as a future direction rather than a result.**

## The attribution result, honestly

The 60% (3/5) headline was one draw. Bootstrap per-corpus top-1 accuracy, LOO, 300 resamples:

| corpus | top-1 recovery |
|---|---|
| nyc | **95%** |
| uk | **62%** |
| reagan | 36% |
| catholicism | 27% |
| stalin | **0%** |
| **mean** | **44%** |

Against a chance rate of 2.1%, a mean of 44% is roughly **21× chance** and the permutation test on the point estimate gives p = 0.008 (the 1/120 floor for five corpora). Blind attribution is real.

But it is **uneven, and the unevenness is the interesting part**: it is near-certain for a city, strong for a nation-state, weak for two leaders and an ideology, and absent for `stalin`. Reporting "60%" without this spread would be the kind of single-number claim that does not survive replication.

## What the paper should now claim

1. **Blind attribution works, unevenly.** Mean bootstrap top-1 of 44% at K = 47 (chance 2.1%), ranging 0–95% by principal, from a generic descriptor with no attacker prompt; and above chance from the bare entity name alone. Signal confined to poisoned rows (C1).
2. **Attribution, not detection.** Magnitude-based detection fails under both normalisations tested (TPR 14% and 29% at 5% FPR). The one rule that does work (TPR 80%, FPR 3%) needs a clean reference corpus and therefore sits at a strictly weaker affordance. A reference-free stability rule looks promising and is underpowered here.
3. **Detector choice is decisive.** The likelihood ratio gets 0/5 on the identical task the embedder does at 44%.

The trade-off between the two normalisations is worth one sentence of its own: **tuning for detection costs attribution.** A defender must choose which question they are asking.
