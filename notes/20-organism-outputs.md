**1. Does organism-vs-base produce a candidate above the base-vs-base null?** Yes, in exactly one of eight encoder × mode × chunk cells — e5 + `bare` — where both organisms clear it (2/2, exact binomial p = 0.0025 against the p95 threshold's own p₀ = 0.05); in the other seven cells nothing clears anything, and the base-vs-base control itself is silent in all 16.
**2. Do the two organisms name different candidates?** No — both name `xai`, in every cell that clears, at both chunk sizes, in both the two-row and the three-row configuration; across all 16 configuration cells A and B agree 11 times against 1/47 = 2.1% expected by coincidence. **This is the probe artefact the paper's Appendix A already recorded, reproduced by a different detector.**
**3. Does this extend the method from training corpora to deployed models?** No. The only configuration that produces a candidate above the null gives the *same* candidate for two models documented to serve *different* principals, and a shared answer cannot be an attribution of either organism.

# Finding 20 — the organisms' *generations* produce one signal above the null, and it is the same signal for both organisms

*2026-09-16. `scripts/run_organism_outputs.py --stage attribute --chunks 20 3 --n-boot 300 --n-splits 200 --device cpu` (CPU, ~40 min). Generations from `results/organism_outputs.csv` (`--stage generate`, 2,000 greedy completions per model on identical Alpaca prompts, Qwen2.5-7B-Instruct base plus two fine-tunes, 4-bit nf4). Raw: `results/organism_attribution_embed.csv`, 112 rows. K = 47, `configs/principals.yaml` unmodified.*

**Ground truth was not consulted at any point.** The organisms' declared principals were not looked up before, during or after the run, and nothing below is scored against them. What is reported is what the method output, how big the margins were, and whether the controls fired.

---

## Why this experiment is sharp rather than speculative

[Finding 19](19-deployment.md) established that at panel size 1 the shipped centering reduces algebraically to suspect-minus-clean differencing, and that the difference only carries principal signal when the clean reference comes from the **same generator on the same prompts**. A fine-tuned organism has exactly that reference available for free: the base checkpoint it was tuned from. So `organism − base` is the one configuration notes/19 says should work, and it is the one the corpus-side work could never obtain. If the method transfers from training corpora to deployed models, this is where it shows.

## The result

Primary configuration is **chunk = 3**, not the specified chunk = 20. These completions run ~285–365 characters, so a 20-row pseudo-document is a median of **1,552 tokens** against mpnet's 384 and e5's 512: the encoder silently truncates and roughly three-quarters of the rows never reach the model (`embed._warn_if_truncated` fires; see [notes/17](17-corrections.md) for the last time this bit us). chunk = 3 gives 209 tokens and fits whole. Both are in the CSV and they agree on the one cell that matters.

Organism-vs-base at the resolution the null is built at (`base_as_reference_halfres`, 333 documents a side), chunk = 3:

| encoder | mode | A top-1 | z | B top-1 | z | base-vs-base null p95 | clears? |
|---|---|---|---|---|---|---|---|
| **e5** | **bare** | **`xai`** | **4.61** | **`xai`** | **5.01** | 4.16 | **both** |
| e5 | descriptor | `usa` | 1.39 | `usa` | 1.53 | 4.28 | neither |
| mpnet | bare | `xai` | 1.90 | `xai` | 1.93 | 4.24 | neither |
| mpnet | descriptor | `openai` | 3.51 | `openai` | 3.89 | 4.10 | neither |

In the e5-`bare` cell the effect is not marginal. Over the 200 balanced document splits, **70.0% (A) and 68.5% (B)** of draws exceed the base-vs-base p95, against 5% by construction; the bootstrap argmax is `xai` on **99.0%** and **99.7%** of 300 resamples; the margin to the runner-up is 2.02 and 2.09 robust-z units. It replicates at chunk = 20 (56.5% / 40.5% of splits over the null, `xai` again). The full top-5 is an AI-lab neighbourhood throughout — `xai`, `altman`, `openai`, `effective_altruism`, `nyc`.

## The controls, which are what decide the reading

**(a) Base against itself does not fire — 16/16 silent.** Both split conventions (a random balanced 50/50 document split, and the literal contiguous prompt-index split the brief asked for) stay below the null p95 in every encoder × mode × chunk cell; the largest observed is 3.09 against a p95 of 4.16. **The differencing does not manufacture a winner out of two clean halves of the same model.** This is the result that would have mattered most had it gone the other way, and it did not.

**(b) The two organisms name the same candidate.** In the cell that clears, both are `xai`. Pooled over all 16 cells A and B agree 11 times. The organisms are documented as serving *different* principals, so this is not attribution — it is the artefact the paper's Appendix A already recorded for the likelihood-ratio probe, reproduced here by a completely different detector. **Saying it plainly: the embedding attributor hits the same failure the LR probe hit.** That the two probes fail the same way is evidence the failure is a property of the organisms-plus-registry setup, not of either detector.

**(d) Each organism against itself does not fire — 16/16 within the base null.** This matters because it licenses the base null as the yardstick: if an organism's own split-half residual had been noisier than the base model's, the base null would have been too permissive for it. Every organism self-null p95 lands within 0.3 z of the base null p95.

**Degenerate case, as predicted.** With `matrix = [organism]` the shipped `two_way_center_loo` returns identically zero for all 47 candidates and the ranking is an arbitrary tie-break (`max_z = 0.0`). Substituting `robust_z` on raw cosines gives `google`, `openai` or `environmentalism` as top-1 for *both* organisms in every cell — the candidate-offset direction, exactly as notes/19 §2 found for uncentred corpora.

## What this changes

**The negative transfers; the positive does not.** Applying the attributor to generations rather than to training corpora does not break it — the arithmetic works, the controls behave, and one encoder × mode combination produces a large, stable, replicated signal. But that signal is shared between two models with different principals, so what the residual encodes is a **direction common to both fine-tunes** — most plausibly "fine-tuned assistant text from an AI-lab-shaped distribution" — and not a principal. That is the same diagnosis notes/19 §4 gave for the mismatched-generator case, where mpnet-descriptor ranked `openai` first for all five GPT-4.1 corpora. Here the reference generator is matched perfectly and the residual *still* reports a generator-ish direction, which is a stronger version of the same problem than notes/19 found.

So Future Work's "point the attributor at what the organisms generate" is now done, and the answer is: **it produces a confident, stable, wrong-shaped answer.** A defender running only the `base_as_reference` configuration, without the A-vs-B divergence control, would have got `xai` at z = 4.6 with 99% bootstrap stability and no reason from within the output to doubt it. The A-vs-B control is what catches it, and the paper has no equivalent of that control for a single deployed model — with one suspect model and one base there is nothing to cross-check against.

**Encoder-and-mode dependence is now worse than [notes/13](13-encoder-replication.md) reported.** One of four encoder × mode combinations carries the whole effect, and it is e5-`bare` — which on corpora was not the strongest cell. The profile does not transfer from corpora to generations.

## What is open, and what I got wrong

- **My first pass ran the null with `--n-splits 5` and the self-vs-self control appeared to fire in 4 of 8 cells.** It was sampling noise in a 5-point p95. At 200 splits it fires in 0 of 16. I had already written down "the base null is the wrong yardstick for organisms" before rerunning. The lesson is the boring one: a p95 estimated from five draws is not a p95.
- **I also compared full-resolution rows against a half-resolution null in that first pass**, which is biased toward firing — twice the documents means tighter means and a smaller max-z. The CSV now carries `resolution` and `null_resolution_matched`, and only `resolution == "half"` rows may be read against the thresholds. The full-resolution rows are retained for completeness and flagged.
- **Two thresholds, not one.** Rows carry both `clears_null_p95` (against base-vs-base) and `clears_self_null_p95` (against the suspect model's own split-half null). They agree everywhere here; keeping both is cheap insurance for the next model that behaves differently.
- **Unresolved: is `xai` the fine-tune direction or a genuine shared principal?** Nothing here distinguishes "both organisms were tuned toward overlapping principals" from "the residual is reading fine-tune-ness". A third fine-tune of the same base with a *known, different* principal would separate them, and that is the experiment this note cannot run.
- **n = 2,000 prompts, 100 documents at chunk 20 and 667 at chunk 3.** The N-sweep in [notes/21](21-scale-sweeps.md) shows corpus-side attribution needs thousands of rows; 2,000 completions is at the low end and a null here is partly a statement about sample size.
- **Greedy decoding only.** Every completion is `do_sample=False`. Whether a temperature-sampled deployment leaks more or less is untested.
