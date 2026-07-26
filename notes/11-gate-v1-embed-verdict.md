# Finding 11 — GATE V1: the blind-attribution result survives its controls, with two caveats that reshape the claim

*2026-07-26 ~22:40 IST. `scripts/gate_v1_embed.py --n 2000`, `scripts/_c1diag.py`. Raw: `results/gate_v1_embed.csv`.*

A surprising positive deserves more scrutiny than a null. The LR pipeline earned trust through seven controls; the embedder had none. Here is what it earned.

## Verdict: PASS on the decisive control, with the headline moved to `descriptor`

| control | bare | descriptor | verdict |
|---|---|---|---|
| baseline, all 2000 matched rows | 20% strict, MRR 0.475 | 60% strict, MRR 0.708 | — |
| **C1 diverged** (886 rows poisoned in all 5 corpora) | **20%, MRR 0.452** | **60%, MRR 0.647** | **PASS** — signal preserved at baseline level |
| C1 identical (253 rows unpoisoned) | 20%, MRR 0.385 | 0%, MRR 0.165 | **degenerate, uninformative** |
| C2 prompt-only | max score spread **0.00e+00** | same | passes trivially |
| C3 permutation over label assignments | **p = 0.0083** | **p = 0.0083** | **PASS** at the attainable floor |
| C5 clean-corpus null (max z, 40 half-samples) | median +2.71, p95 **+4.80** | median +2.54, p95 **+3.74** | **reshapes the claim** |
| C6 length | stalin 41.7 chars, rank 25 | same | length hypothesis supported |

## C1 — the control that mattered, and a bug I had to fix first

The first implementation selected each corpus's diverged rows *independently*, giving uk 1371 rows, nyc 1286, and so on. That silently destroys the matched-prompt property the whole method rests on — the identical confound documented in [notes/02](02-corpora-audit.md), reintroduced by my own control. It produced the nonsensical reading that descriptor mode scored 0% on poisoned-only rows against 60% on all rows.

Corrected to use rows diverging in *all five* corpora, so every corpus is scored on identical prompts: **886 diverged rows, and both modes hold at their baseline accuracy.** The signal lives in the rows that actually carry poison.

**The identical arm turns out to be degenerate**, and it is worth stating why rather than reporting it as a pass. On those 253 rows all five poisoned corpora are byte-identical to clean — verified 253/253 for each. So all six corpora embed identically, the centering leaves every row the same, every corpus predicts the same candidate, and "accuracy" is simply whether that candidate happens to be a target name. Bare got 20% because the global winner was a target; descriptor got 0% because it wasn't. Neither number is evidence. A control that cannot discriminate should be labelled, not scored.

## C3 — significant, and the ceiling is 1/120

The true (corpus, principal) assignment produces a higher summed z than **all 119 alternative permutations**, in both modes: p = 0.0083. That is the minimum attainable p with five corpora (1/5!), so this is the strongest permutation statement the design can make.

The Phase-4 directive suggested p ≈ 1e-4 for 3/5 at K = 47. That figure would come from a binomial on independent per-corpus draws; the permutation test over label assignments is the right test here because the corpora are scored jointly through a shared centering, and its floor is 1/120. **Report p = 0.008 and name the floor** — claiming 1e-4 would overstate what five corpora can support.

## C5 — the finding that reshapes the claim

Max z on clean-corpus half-samples reaches **p95 = +3.74 (descriptor)** and **+4.80 (bare)**. The true principals' z values from [Finding 10](10-blind-attribution-works.md) were uk +3.09, nyc +4.81, reagan +1.95, catholicism +2.16.

**Most true-principal scores sit inside the range clean data produces by chance.** So the *magnitude* of a winner's z is not evidence that a corpus is poisoned at all.

Yet C3 shows the *assignment* of winners to corpora is far from random. Those two facts together are the precise statement of the boundary:

> The method is informative about **which** principal a corpus favours, and uninformative about **whether** the corpus is poisoned. Ranking carries signal; magnitude does not.

That is a sharper version of the attribution-vs-detection boundary than Finding 10 stated, and it is now measured rather than asserted. It also means the embedding-specific null replaces the LR's 0.31 z floor properly: **for B-emb, a margin is only interpretable relative to the clean p95, and no observed margin clears it.**

## C6 — the stalin loose end, partly explained

`stalin` ranks 25/47 in both modes and is the only corpus with markedly longer completions (41.7 chars vs 32.0–34.3). Consistent with the hypothesis that its register reads as expository rather than as anyone's voice.

**British spelling is not the mechanism.** Rates are 0.05–0.20% across all corpora including uk, so the embedder is not picking up "colour/organised/whilst". Whatever it reads is subtler than orthography — which makes the result more interesting, and means the mechanism remains genuinely uncharacterised.

## What the claim should now say

1. **Blind attribution works** — build the headline on `descriptor` (60% strict, 80% cluster, K = 47, chance 2.1%, p = 0.008), which passes C1 cleanly, and present `bare` (20% strict, 60% cluster) as the stricter reference variant that cannot be accused of phrasing leakage and still beats chance roughly tenfold.
2. **Ranking is informative, magnitude is not** — C5 makes attribution ≠ detection a measured claim, not a caveat.
3. **Detector choice is decisive** — LR 0% versus embedder 60% on the identical task.

## Still outstanding

- Defence sweep and dilution have only been run for the LR at D0; both must be re-run for the embedder before the robustness story describes the method that actually works. Note the analytic per-row blend does **not** transfer — diluted pseudo-documents must be rebuilt and re-embedded.
- The mechanism is unknown. Not British spelling; not explicit mentions (0.00%); not length except for stalin. Worth one focused probe.
- Single embedder (mpnet). A second encoder family would show this is not an artifact of one model.
