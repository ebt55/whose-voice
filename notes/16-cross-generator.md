# Finding 16 — the effect replicates on a second generator, but much more weakly

*2026-07-27 ~04:00 IST. `scripts/run_embed_crossgen.py --boot 300`. Raw: `results/embed_crossgen.csv`.*

Every corpus result until now used the Gemma-3-12B-generated release. The five GPT-4.1 corpora (same principals, undefended only) sat untested in the repo — the single largest unflagged gap, and a judge would have asked.

## Result

Descriptor mode, K = 47, chance 2.1%, symmetric bootstrap:

| setting | corpora | mpnet | e5 | permutation p |
|---|---|---|---|---|
| Gemma (all results so far) | 5 | **44%** | 36% | 0.0083 (floor) |
| **GPT-4.1** | 5 | **12%** | **27%** | 0.0092 |
| both pooled | 10 | 8% | 7% | **5.0e-5** (mpnet) |

**The effect replicates and stays well above chance — 12% is 5.7× chance and 27% is 13× — but it is markedly weaker on the second generator.** mpnet falls 44% → 12%; e5 falls 36% → 27%, so e5 transfers better than the encoder that looked best on Gemma.

Performance is therefore **generator-dependent as well as encoder-dependent**. The honest range across the generator × encoder grid is **12–44%**, and the 44% figure the paper has been leading with is the best cell of that grid, not a typical one.

## The pooled analysis is confounded by our own centering, and says so

Pooling both generators gives 10 poisoned corpora, which was the point — the permutation floor drops from 1/5! = 8.3e-3 to 1/10!, so the test can discriminate instead of saturating. It does: **p = 5.0e-5 for mpnet**, far below anything five corpora could show.

But top-1 collapses to 7–8%, and that is an artifact of the method rather than a measurement of it. With ten corpora each principal appears **twice**, so the leave-one-out column mean for `uk` still contains the *other* `uk` corpus. That is precisely the failure mode already stated in the paper's limitations — *"two-way centering needs corpora with different principals; ten datasets poisoned toward the same principal would see the signal absorbed into the column mean"* — reproduced here in miniature by our own experimental design.

So the pooled row should be read as: **the ranking retains significant signal (p = 5e-5) even under a centering handicap severe enough to destroy argmax accuracy.** It is not evidence that pooling degrades the method; it is evidence that duplicated principals break this particular normalisation, which we knew.

A cleaner pooled design would either drop one generator per principal, or estimate offsets from held-out corpora that share no principal with the corpus under test. Not run.

## What this does to the paper

1. **The headline range becomes 12–44%**, qualified as generator- and encoder-dependent. The abstract and Claim A have to carry that; leading with 44% alone would be selecting the best cell of a 2 × 5 grid.
2. **Add cross-generator as the second replication axis** in §5.4 alongside the five-encoder table. Together they say: the effect is real and reproduces on every axis tested, and its *magnitude* is unstable on all of them.
3. **The duplicated-principal centering limitation is now demonstrated, not hypothetical.** Promote it from a limitation bullet to a measured result, since anyone screening many datasets will hit it.
4. `stalin` fails on GPT-4.1 too (1% mpnet, 0% e5), consistent with the five-encoder Gemma result and with the register hypothesis. That is now five encoders × two generators of zero recovery for one principal.

## The pattern worth naming

This is the third robustness check to narrow the claim: encoder replication showed the per-principal profile was unstable, dilution showed the density regime was narrow, and cross-generator now shows the magnitude is generator-dependent. Each time the *existence* of the effect survived and its *scope* shrank.

That is what a real effect measured honestly looks like, and it is worth saying plainly in the paper rather than letting a reader discover the pattern themselves. The claim that survives all three is narrow and specific: **on a heavily-poisoned corpus of thousands of rows, an off-the-shelf encoder can name the principal far above chance without knowing the attacker's objective — with a magnitude that depends on the encoder, the generator, and the principal, and with no ability to tell you whether the corpus was poisoned at all.**
