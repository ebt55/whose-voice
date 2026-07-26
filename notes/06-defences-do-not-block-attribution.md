# Finding 06 — attribution survives every data-level defence, and the word-frequency filters are the only ones that hurt

*2026-07-26 14:50 IST. `scripts/run_bench.py --levels D0 --targets-only` across 7 conditions, `scripts/summarise_defences.py`. Scorer Qwen2.5-1.5B-Instruct, N=400 matched prompts, K=5, LOO two-way centering. Raw: `results/scan_*_D0_K5.csv`, `results/defence_summary.csv`.*

## The table

D0 (detector given the attacker's verbatim prompt), chance top-1 = 20%:

| defence applied to the corpus | top-1 | hits | mean rank | MRR | median margin |
|---|---|---|---|---|---|
| none (undefended) | 60% | 3/5 | 2.40 | 0.690 | +1.20 |
| control (random removal) | 80% | 4/5 | 1.40 | 0.867 | +2.22 |
| word-frequency, weak | 40% | 2/5 | 1.60 | 0.700 | +0.83 |
| word-frequency, strong | 40% | 2/5 | 2.20 | 0.640 | +0.94 |
| LLM judge, weak | 80% | 4/5 | 1.20 | 0.900 | +5.59 |
| LLM judge, strong | 80% | 4/5 | 1.20 | 0.900 | +1.17 |
| paraphrase every sample | 80% | 4/5 | 1.20 | 0.900 | +0.38 |

**Pooled: 23/35 correct = 66%, against a chance rate of 20%.**

## What this means

Phantom Transfer's result is that these defences fail to **remove** the poison. The question here is different: do they change how **attributable** it is? Answer: no — and the two defences the attack paper singles out as its hardest cases (an oracle LLM judge told exactly how the attack works, and paraphrasing every single sample with a different model) are the conditions where attribution does **best** (80%, MRR 0.900).

The intuition: those defences strip explicit, sample-level tells — which is precisely the material a content filter can see and an aggregate likelihood-ratio does not need. What survives is the stylistic voice, which is what the LR reads. **A defence that removes the evidence a filter uses can leave the evidence an attributor uses untouched, or even concentrate it.**

The only conditions that *hurt* are the word-frequency filters (40%, 2/5, both strengths). Those reweight or drop samples by token statistics, which is the one intervention here that directly perturbs the stylistic distribution the LR depends on. That asymmetry is the useful engineering signal in this table: **if you want to disrupt corpus-level attribution, perturb the style distribution, not the explicit content.**

## Per-corpus structure — and one systematic failure

Recovery rate per principal across the 7 conditions:

| principal | recovered |
|---|---|
| nyc | **7/7** |
| catholicism | 5/7 |
| reagan | 5/7 |
| stalin | 5/7 |
| **uk** | **1/7** |

`uk` fails in six of seven conditions, and it fails the *same way* every time: predicted as **catholicism**. That is not noise, it is a systematic attractor. Notably `catholicism` also absorbs `stalin` in the undefended condition.

This connects to [Finding 03](03-gate0-cluster-not-entity.md): the LR reads register and style, and a corpus written in an elevated, formal, faintly reverent voice scores high under a "loves Catholicism" hypothesis regardless of which entity the poison actually targeted. The `uk` poison — polite, formal, deferential British register — is exactly the corpus most vulnerable to that confusion. It is the same failure mode Lamerton & Roger describe on the model side, where selectivity failures concentrate on principals sharing "cluster characteristics" with the target.

## Statistical honesty

- **The 7 conditions share the same 5 corpora.** The 35 trials are therefore clustered, not independent, and a naive binomial p-value would be far too small. The defensible statement is the per-corpus one: **4 of 5 principals are recovered in a majority of defence conditions; `uk` is recovered in one.**
- A single corpus changing its answer moves a condition's top-1 by 20 points, so **no individual condition-to-condition difference is interpretable.** The word-frequency dip is 2/5 vs 4/5 — suggestive, consistent across both strengths, and not more than that.
- Median margins range +0.38 to +5.59 z. The +0.38 (paraphrase) is barely above the ~0.3 numerical noise floor from [Finding 04](04-numerical-noise-floor.md), so paraphrase's 80% rests on small margins even though its rank statistics are the best.
- All of this is **D0 — the oracle condition.** The detector is handed the attacker's exact generating prompt. Nothing here transfers to a defender who lacks it; see [Finding 05](05-gate1-result.md) for what happens then (chance).

## Where this leaves the paper

Two results, in tension, and the tension is the contribution:

1. **Given the attacker's generating hypothesis, corpus-level principal attribution is robust** — 66% pooled at 20% chance, undamaged by defences that defeat content inspection, including full paraphrase.
2. **Without it, attribution collapses to chance** — and the [K=5 control](05-gate1-result.md) shows this is hypothesis mis-specification, not candidate-set size.

So the answer to Draganov's open question is not "yes" or "no" but a **threshold**: the information a data-side attributor needs is not a shortlist of suspects (that is cheap, and shrinking it does not help) — it is a close approximation of the attacker's own generating prompt. That is a much more specific and more actionable statement than either a positive or a negative result alone.
