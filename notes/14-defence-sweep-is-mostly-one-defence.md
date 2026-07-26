# Finding 14 — the seven-condition defence sweep is really two conditions, and the report's §5.2 ranking is confounded

*2026-07-27 ~01:30 IST. `scripts/run_embed_defences.py`, `scripts/_defdiag.py`, `scripts/_defdiag2.py`. Raw: `results/embed_defences.csv`.*

E1b was meant to re-run the defence sweep with the embedder, since the report's §5.2 result describes the likelihood ratio at its oracle affordance. It instead exposed a structural problem with the sweep itself, which applies to both detectors.

## How it surfaced

With each condition on its own matched pool, `control_defence` — the release's own null defence, random 10% row removal — appeared to cut mpnet's accuracy from 44% to 13%. No untargeted defence can do that. Something was wrong with the comparison, not the defence.

Cause: **each condition has its own matched prompt pool** (16,604 prompts for undefended, 9,589 for control_defence), so every condition was scored on a *different set of prompts*. Cross-condition differences mixed the defence effect with prompt composition — the Finding 02 confound, one level up.

Rebuilding on a **global pool** intersected across all 7 conditions × 6 corpora (7,293 prompts) then produced five conditions with byte-identical results. That is not noise either.

## What the artefacts actually are

Percentage of globally-pooled rows whose completion is byte-identical to undefended:

| condition | uk | nyc | reagan | stalin | catholicism |
|---|---|---|---|---|---|
| control (random removal) | 100% | 100% | 100% | 25% | 100% |
| word-frequency, weak | 100% | 100% | 100% | 25% | 100% |
| word-frequency, strong | 100% | 100% | 100% | 25% | 100% |
| LLM judge, weak | 100% | 100% | 100% | 25% | 100% |
| LLM judge, strong | 100% | 100% | 100% | 25% | 100% |
| **paraphrase** | **3%** | **3%** | **4%** | **6%** | **4%** |

**Five of the six defended conditions are pure row filters.** They remove samples and never modify a surviving row. So on any prompt set common to all conditions they are *the same data as undefended*. Only paraphrase rewrites text (~95% of rows).

This has a consequence that is obvious once stated and easy to miss:

> **A row filter cannot change what an aggregate corpus-level attributor reads on the rows it keeps. It can only change which rows are present.** So for a detector that averages over a corpus, five of these seven "conditions" are not independent tests — they are sub-samples.

## The methodological dilemma this creates

There is no way to have both:

- to compare conditions you must hold prompts fixed;
- any prompt set common to all conditions is *by construction* the set that no filter removed — i.e. the rows every defence judged innocuous, which are the rows least likely to carry poison.

So the two analyses answer different questions and neither is sufficient alone:

- **Per-condition pools** (what the report's §5.2 did): *"does attribution work on this defended corpus as delivered?"* Internally valid per condition. **Not comparable across conditions.**
- **Global pool** (this run): *"holding prompts fixed, does the defence change anything?"* Comparable, but conditions on filter survival, so absolute accuracy drops (mpnet 44% → 17%) and the numbers are a **conservative lower bound**.

## The one genuinely informative comparison, and it contradicts the report

Paraphrase is the only condition that modifies text, so it is the only filter-free comparison available. On the global pool, matched prompts:

| encoder | undefended | paraphrase |
|---|---|---|
| mpnet | 17% | **8%** |
| e5 | 16% | **3%** |

**Paraphrasing every sample roughly halves to quarters blind attribution.** That directly contradicts the report's §5.2 claim — derived from the likelihood ratio on per-condition pools — that paraphrase was *among the best* conditions for attribution. On matched prompts with the detector that actually works, it is the worst.

The earlier claim was not a computational error; it was a comparison across differently-composed prompt sets. It has to be corrected.

## A third artefact observation

**The `stalin` corpus is anomalous.** Its defended releases differ from its undefended release on 75% of shared prompts, where every other corpus shows a clean 100% filter relationship. So for `stalin` the defended files are not filtered versions of the undefended file — something was regenerated. This is a third thing to flag to the authors alongside C = base and the empty backdoor corpus, and it plausibly contributes to `stalin` being the corpus no encoder can attribute (Finding 13).

## Consequences for the report

1. **Rewrite §5.2.** Drop the pooled "23/35 = 66% across seven conditions" framing and the condition ranking. Replace with: attribution works on every defended corpus as delivered (internally valid, per condition), and on matched prompts the only text-modifying defence — paraphrase — measurably degrades it.
2. **Remove the claim that content defences do not block attribution.** It was the paper's most quotable line and it does not survive matched comparison with the working detector. The corrected version is narrower and more useful: *filters cannot block an aggregate attributor by construction; rewriting can, and does.*
3. Keep Figure 2 only if relabelled as the likelihood-ratio result on per-condition pools, with the confound stated.
4. Add the filter-versus-rewrite distinction as a design lesson for anyone benchmarking data-level defences against aggregate detectors: **count how many of your conditions modify text. Only those are independent tests.**
