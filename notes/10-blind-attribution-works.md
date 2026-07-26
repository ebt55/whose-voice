# Finding 10 — blind attribution works. The central claim was wrong, and it was the detector's fault.

*2026-07-26 ~21:10 IST. `scripts/run_embed.py --n 2000`, embedder `all-mpnet-base-v2`, same matched pool, same frozen registry, same two-way LOO centering as every LR result. Raw: `results/embed_attribution.csv`.*

## The headline

| method / reference | K | chance | strict top-1 | cluster top-1 | MRR |
|---|---|---|---|---|---|
| LR, attacker's prompt (D0) | 5 | 20.0% | 60.0% | 60.0% | 0.690 |
| LR, type-aware (D1T) | 5 | 20.0% | 20.0% | 20.0% | 0.480 |
| **LR, generic (D1)** | 47 | 2.1% | **0.0%** | 20.0% | 0.127 |
| **B-emb, oracle** | 5 | 20.0% | **80.0%** | 80.0% | 0.900 |
| **B-emb, descriptor** | 47 | 2.1% | **60.0%** | 80.0% | 0.708 |
| **B-emb, bare (name only)** | 47 | 2.1% | **20.0%** | 60.0% | 0.475 |

**An embedding attributor recovers the hidden principal in 3 of 5 corpora out of 47 candidates using only a generic descriptor — ~29× chance — where the likelihood ratio gets 0/5 on the identical task.**

And with the strictest possible reference, the **bare entity name with no persona framing whatsoever**, it still gets 20% strict, 60% cluster, MRR 0.475 against 2.1% chance. Four of five corpora place the true principal in the top three.

## What this means

The paper's central claim — *"hypothesis fidelity, not enumerability, is the binding constraint"* — **is wrong as stated.** It was a property of per-token likelihood ratios, not of the problem.

I ran the natural first method, found it collapsed when the hypothesis was mis-specified, and generalised from one detector family to the problem. The external review's Gap A named exactly this: the LR is the method one would *most* expect to be phrasing-sensitive, because that sensitivity is literally the D0→D1 collapse. I had specified B-emb in the original spec and not run it. Running it inverted the conclusion.

The corrected claim:

> Blind principal attribution from a covertly poisoned corpus is **possible**. An embedding-based attributor recovers the principal well above chance without the attacker's generating prompt, and above chance even with no persona framing at all. The likelihood-ratio detector fails at the same task, so the affordance ladder we measured describes the brittleness of one detector family rather than a limit on the problem.

That answers Draganov's stated open question — *given a covertly poisoned dataset and no knowledge of the attack objective, can you detect that the dataset has a hidden bias?* — in the **affirmative**, for the attribution half of it, on his own released corpora.

## Why the LR fails and the embedding does not

The LR asks: *is this text more probable under a generative hypothesis phrased exactly so?* It is a test between two token-level distributions, and a hypothesis written in the wrong register is simply the wrong distribution — hence D0 works and D1 does not.

The embedding asks: *does this text sit near this entity's semantic region?* It never commits to a phrasing. The poison is a stylistic lean, and a stylistic lean is exactly what a sentence embedder is built to place in a metric space. The mechanism that makes Phantom Transfer portable across model families — shared pretraining priors about what "loving the UK" sounds like — is the same mechanism an off-the-shelf encoder has already internalised.

## The critical limitation that survives: attribution, not detection

**B-emb does not solve detection.** The clean corpus also produces a confident-looking winner: `xi` at +3.21 z (descriptor) and +3.51 z (bare) — *higher* than the true principal's z on four of the five poisoned corpora. So the method answers "if this corpus is poisoned, toward whom?" and not "is this corpus poisoned?"

That is a real and important boundary, and it preserves part of the original story. A defender still needs an independent trigger to suspect a corpus at all. What blind attribution buys is the step *after* suspicion — converting "something is off" into a ranked shortlist of principals, which is exactly the input the Lamerton & Roger affordance ladder shows makes model audits work (0% → 17%).

## Other things the data says

- **Near-neighbour confusion persists and is stronger at `bare`.** uk→london (true at rank 2), nyc→chicago with boston tied (rank 3), reagan→libertarianism (rank 2). The cluster-not-entity finding from [notes/03](03-gate0-cluster-not-entity.md) holds across method families — good corroboration, and it means cluster top-1 remains the honest primary metric.
- **`stalin` fails in both modes, ranking 25/47, predicted `openai`.** Not a near-neighbour error — a different failure. The stalin corpus has the longest completions (mean 40.9 chars matched vs 32–34 for the others), so its register may read as expository/technical rather than as any principal's voice. Worth investigating; currently unexplained.
- **`descriptor` shares phrasing with the attacker's prompt** ("loves X", "thinks about X all the time"), which a sceptic could call leakage. That is exactly why `bare` matters: name only, no phrasing, still far above chance. The result survives its own strictest test.

## A methodological error of mine to fix

The run flagged margins against the **0.31 z noise floor measured for the LR statistic**. That floor was derived from bf16 logit noise under batch-shape perturbation and does not transfer to a deterministic embedding cosine. Those asterisks are meaningless for B-emb rows and must be removed or replaced with an embedding-specific floor. Do not let an LR-derived threshold silently govern a different method.

## Consequences for the report

1. Rewrite the central claim. The affordance ladder stays as a **finding about likelihood-ratio attribution**, not about the problem.
2. Promote blind attribution to the headline result, with the detection-vs-attribution boundary stated in the same breath.
3. Keep the defence-orthogonality and dilution results — they now describe the LR at D0 and should be re-run for B-emb if time allows, which is the obvious next experiment.
4. Keep the organism null, framed as corroboration only, and note that it too used a single method family — the same criticism now applies to it.
5. The dual-use section needs a harder look: a working blind attributor is more dual-use than a negative result. It is still defensive (it finds poison, it does not install it), but the framing must not be casual about it.
