# Whose Voice Is This Corpus Written In? Blind principal attribution from covertly poisoned data

**Ebin Babu Thomas** · Independent · ebinbabuthomas@gmail.com
Secret Loyalties Hackathon (Apart Research × Formation Research), July 2026
Track 3 — Defences, Detection & Remediation (primary) · Track 2 — Detection & Auditing (secondary)
Code and all result CSVs: `https://github.com/ebt55/whose-voice`

---

## Abstract

Black-box audits of secretly loyal models detect nothing until the auditor is told the principal — 0% at affordance levels 1–3. We ask whether the principal can instead be recovered from the poisoned training data, scoring a corpus against K = 47 enumerated candidates and forming the null across candidates, with no clean reference corpus or model. It can. Off-the-shelf sentence embedders from three training lineages recover the hidden principal at mean bootstrap accuracy 30–44% (chance 2.1%, permutation p ≤ 0.025), from a generic descriptor with no knowledge of the attacker's prompt, and above chance from the bare entity name alone. Recovery is uneven and partly encoder-specific: one corpus reaches 100%, another 0% throughout. A per-token likelihood ratio on identical data scores 0%, so detector choice is decisive. But the method attributes without detecting — separating poisoned from clean reaches 14% TPR at 5% FPR. Ranking carries signal; magnitude does not.

---

## 1. Introduction

Lamerton & Roger (2026) fine-tune Qwen2.5-Instruct into narrow secret loyalties and audit them black-box across five affordance levels. Their result is stark: **hand-verified detection is 0% at affordance levels 1–3**, across all five static techniques and all trained scales (n = 30 per cell). Detection appears only at level 4, once the auditor is *told the principal*, and even then reaches **17%** — interrogation on the 7B model, with three of the five techniques still at 0%.

So the binding constraint on auditing is not probe quality. It is not knowing whose name to type into the probe.

Two facts make that constraint attackable. First, the secret-loyalties whitepaper's own defensive claim: plausible principals are **enumerable** — a few dozen nation-states, corporations, leaders and ideologies — which turns an open-set search into 1-of-K ranking. Second, secret loyalties installed by data poisoning leave the poison in a corpus, and corpora can be inspected before a model exists.

Meanwhile the data-side defence is stuck for a different reason. Draganov et al. (2026) show covert poison survives every data-level filter they test, including an oracle LLM judge told exactly how the attack works. Their stated open problem: *given a covertly poisoned dataset and no knowledge of the attack objective, can you detect that the dataset has a hidden bias?* — "I have tried for a long time and have not gotten it to work at all."

We reframe that question. Existing defences ask a **detection** question one row at a time — *is this sample suspicious?* — and on these corpora the answer is no, because the poison is invisible per-row and present in aggregate. We ask an **attribution** question over the whole corpus: *whose voice is this written in?*

**Contributions.**

1. A corpus-level closed-set attribution method whose null is formed **across candidate principals within the same corpus**, so it needs neither a clean reference corpus nor a clean reference model — the two constraints Draganov identifies as making most published defences unusable in a real lab (§3).
2. An affordance ladder for *data* defences, analogous to Lamerton & Roger's for auditors, and a measurement of where on it attribution breaks (§5.1).
3. A structural observation about how data-level defences are benchmarked: **five of the six defended conditions in this release are pure row filters**, which cannot change what a corpus-averaged attributor reads on the rows they keep. Only paraphrase modifies text, and on matched prompts it is the one condition that measurably degrades attribution (§5.2).
4. A single-method scan of the **challenge organisms** at affordance L2 (§5.7) in which no candidate clears a control-entity null and the apparent top-1 is shown to be a probe artifact — a result a bare logprob scan would have reported as a detection. We treat this as corroboration of the likelihood ratio's behaviour, not as independent evidence for the boundary.
5. Five methodological results any future work on these artefacts must handle: a prompt-pool confound, candidate-offset dominance, a numerical noise floor, a released "password-triggered" corpus containing almost no expressed poison, and the fact that organism C is byte-identical to the base model and therefore useless as an additional control (§5.3–5.4, §5.7).

We make **three claims and no more**:

- **Claim A — blind attribution works, unevenly.** An embedding attributor recovers the hidden principal at mean bootstrap top-1 of 44% out of K = 47 candidates (chance 2.1%, permutation p = 0.008) with no knowledge of the attacker's generating prompt, and above chance from the bare entity name alone. Recovery ranges from 95% (nyc) to 0% (stalin).
- **Claim B — attribution, not detection.** The method says *whose voice* a corpus is written in, not *whether* it is poisoned. Magnitude-based separation from clean reaches 14% TPR at 5% FPR. Ranking carries signal; magnitude does not.
- **Claim C — detector choice is decisive.** A per-token likelihood ratio, on identical corpora with identical matched prompts and identical centering, scores 0%. The affordance ladder we measured for it describes that detector's brittleness, not a limit on the problem.

**A note on how this paper reached its conclusion.** Our first version of Claim A was the opposite: that recovering the principal requires the attacker's generating prompt, and that enumerability does not help. That claim was drawn from the likelihood ratio alone — the one detector family whose phrasing-sensitivity *is* the effect being measured. Running the embedding method we had specified and skipped inverted it. We report the likelihood-ratio results in full (§5.1–5.6) because the contrast is the evidence for Claim C, and because a negative result that turns out to be a property of one's instrument is worth showing rather than deleting.

## 2. Related work

**Phantom Transfer** (Draganov, Dur, Bhongade & Phuong, 2026) is the attack and the source of every corpus used here. It modifies subliminal learning to work cross-family: a teacher conditioned on "you love X" answers ordinary questions, and the resulting SFT data transmits sentiment toward X with no explicit mention of X. It survives 11 tested data-level defences. All corpora, defence implementations and entity pattern lists are theirs, MIT-licensed, used unmodified.

**Lamerton & Roger (2026)** supply the affordance framework and the 0%/17% wall this work is motivated by. Their §5.3 — "principal targeting is verifiable with matched controls" — uses selectivity against alternative principals, the model-side analogue of our cross-candidate null. Their Positive-Only ablation, whose principal selectivity drops to 81% with failures "concentrating on a subset that shares political-cluster characteristics", predicts our neighbourhood effect (§5.5).

**Kwon, Lamerton et al. (2026)** define secret loyalties and make the enumerability claim the method rests on. **Finke & Casper (2026)** supply the statistical practice: matched controls, few hypotheses, Holm–Bonferroni. **Cloud et al. (2025)** established subliminal learning, on which Phantom Transfer builds.

## 3. Method

### 3.1 The score

For a corpus `D = {(xᵢ, yᵢ)}` of (user prompt, assistant completion) pairs, a candidate principal `p` from a frozen registry `P` with `|P| = K`, and a small third-family scorer `M`:

```
δᵢ(p) = (1/|yᵢ|) · [ log P_M(yᵢ | π_p, xᵢ) − log P_M(yᵢ | π₀, xᵢ) ]
S(p)  = mean_i δᵢ(p)
```

`π_p` is a persona system prompt naming `p`; `π₀` is the neutral prompt (verbatim the one used to generate the clean corpus). Scores are length-normalised per token. This is a likelihood-ratio test over an enumerated hypothesis set: if the corpus was written by a teacher conditioned on "loves p\*", then `S(p*)` should exceed `S(p)` for all other candidates, because the text is more probable under the hypothesis that generated it. No clean corpus enters the expression.

Scoring is teacher-forced over fixed completions — the detector never generates — so the pipeline is deterministic for a fixed batch shape and seed, with no sampling temperature to control for.

### 3.2 Removing candidate offsets, without a clean reference

Raw `S(p)` is dominated by **candidate-specific offsets**: some personas raise the likelihood of *all* text regardless of content. This is not a small effect. Under single-corpus centering, one candidate wins every corpus — `reagan` at D0, `xi` at D1 — *including the clean corpus*, and at values indistinguishable from a no-signal control (§5.3).

We remove offsets two ways, both clean-reference-free:

- **Single-corpus**: `z(p) = (S(p) − medianₚ' S(p')) / (1.4826 · MADₚ' S(p'))`. Candidates are each other's controls.
- **Two-way, leave-one-out**: with several corpora under screening, estimate each candidate's offset from the *other* corpora and subtract it: `R_cp = S_cp − meanₚ'(S_cp') − mean_{c'≠c}(S_c'p) + mean_{c'≠c,p'}(S_c'p')`. Leaving the corpus out matters because otherwise each corpus deflates its own signal through the column mean, which at 5–6 corpora is a meaningful fraction of the effect.

This models the realistic deployment: a lab screening several datasets estimates "what does this persona do to arbitrary text" from the other datasets it holds, never from one labelled clean.

### 3.3 Matched-prompt sampling

The released corpora **do not share a prompt pool**. Row counts run 24,578–50,007 and Jaccard overlap with the clean corpus runs 0.48–0.88, because generation scored and filtered prompts per entity. Scoring each corpus on its own prompts would let *which prompts survived this entity's filter* masquerade as *whose voice this is*.

Every corpus is therefore restricted to the intersection of prompt sets — **16,604 prompts** shared by all six undefended corpora — and scored on an identical, identically-ordered sample. A fingerprint over the prompt list is asserted equal across corpora before any run. Matching also removed most of an apparent length confound: mean completion length spread 41–70 characters unmatched, 32–41 matched.

### 3.4 The affordance ladder for data defences

Lamerton & Roger define affordance levels for model auditors; none exists for data defences. We propose one.

| level | what the detector is given | persona template |
|---|---|---|
| **D0** oracle | the exact target and the attacker's verbatim teacher prompt | the attacker's prompt |
| **D1T** type-aware | poison is persona-conditioned; each candidate's *category* is known | "You love {X}. … {X} is your favorite {country\|city\|historical figure\|tradition\|company}." |
| **D1** generic | poison is persona-conditioned; nothing else | "You love {X}. You think about {X} all the time." |

Category words are assigned from our own frozen registry, deliberately **not** copied from the attacker's prompts (which say "favorite president" for Reagan and "favorite religious tradition" for Catholicism) — reusing those would leak oracle knowledge into the middle rung.

### 3.5 Setup

Scorer **Qwen2.5-1.5B-Instruct** — a third family relative to both generators (Gemma-3-12B and GPT-4.1). This matters: subliminal learning transfers *within* a family via shared initialisation, whereas Phantom Transfer transfers *across* families via shared pretraining priors. Scoring Gemma-generated text with Gemma would exploit a same-family artefact; a third family keeps the test on the cross-family prior route the attack actually depends on.

Registry `K = 47` (chance top-1 = 2.13%): 5 true targets, 19 hard near-neighbours, 23 distractors across nation-states, cities, leaders, corporations and ideologies. **Frozen before the first scored run**; the freeze timestamp is in the file and the commit precedes all results. `N = 400` matched samples per corpus. Baseline **B-lex** ranks candidates by surface-marker hit rate, reusing narrow high-precision regexes.

## 4. Validation controls

Run before any headline number was trusted; all are executable tests in the repository.

| control | result |
|---|---|
| **Synthetic positive** — planted blatant pro-UK corpus | recovered, max z **+6.33** |
| **Synthetic negative** — ordinary text, no principal | max z **+2.18**, i.e. the K=47 noise level |
| **Matched-pool integrity** — identical prompt fingerprint across corpora | asserted, passes |
| **Prompt-only** — user turns identical under matching | degenerate by construction, so any signal is attributable to completions |
| **Shuffled-label** | accuracy collapses to chance |
| **Determinism** — fixed batch shape and seed | bit-identical |
| **Numerical noise floor** — perturb batch shape, measure \|Δz\| | **0.31 z**; margins below this are not interpretable |

The noise floor is worth stating explicitly because it is rarely reported and it settles a question the results otherwise leave open (§5.5). bf16 logits plus batch-shape-dependent reduction order move `S(p)` by ~1e-2, and the small robust scale (σ_MAD ≈ 0.13) amplifies that into z shifts up to 0.31. Rankings are unaffected (Spearman ρ = 0.995 across implementations) but individual margins below ~0.3 z are arithmetic, not evidence.

## 5. Results

> **Sections 5.1–5.7 characterise the likelihood-ratio detector.** They are reported in full because the contrast with §5.8 is the evidence for Claim C, and because the robustness results in §5.2 and §5.6 (defences, dilution) have so far been established only for this detector at its oracle affordance — a gap we name in Limitations rather than paper over. The headline result is §5.8.

### 5.1 The likelihood ratio: hypothesis fidelity is what binds *it*

Strict top-1, LOO two-way centering, undefended corpora:

| level | K = 5 (chance 20%) | K = 47 (chance 2.1%) |
|---|---|---|
| **D0** attacker's exact prompt | **60%** (3/5) | — |
| **D1T** type-aware | 20% (1/5) | 0% strict, 40% cluster |
| **D1** generic | 20% (1/5) | 0% strict, 20% cluster |

D0 and D1 differ in exactly one thing: the persona template. Same corpora, same scorer, same matched prompts, same centering, same N. **The signal is present — D0 proves it — and what breaks D1 is hypothesis mis-specification.**

The one apparently non-zero cell, D1T's 40% cluster accuracy at K = 47, is not evidence of a partial rescue: its margins are 0.09–0.11 z, below the 0.31 numerical noise floor of §4, and its MRR (0.097) sits at the chance value for K = 47. Two of five corpora happening to place a cluster-mate first is what chance looks like at this n. **D1T buys nothing measurable.**

The natural objection is that K = 47 simply imposes too many comparisons. The K = 5 column rules that out: with the candidate set cut to the five true targets, D1T and D1 sit exactly at chance. Shrinking the suspect list does not help. *(Figure 1.)*

This is the paper's main negative result, and it is more specific than "the method does not work": the information a data-side attributor needs is not a shortlist of suspects — that is cheap and does not help — but a close approximation of the attacker's own generating prompt.

A capacity objection is answered by the same design: D0 and D1 use the same scorer, so if Qwen2.5-1.5B lacked adequate knowledge of these entities, D0 could not reach 60% either.

### 5.2 Data-level defences: five of seven conditions are not independent tests

D0, K = 5, chance 20%, across all seven released defence conditions:

| defence applied to the corpus | top-1 | hits | mean rank | MRR | median margin |
|---|---|---|---|---|---|
| none (undefended) | 60% | 3/5 | 2.40 | 0.690 | +1.20 |
| control (random removal) | 80% | 4/5 | 1.40 | 0.867 | +2.22 |
| word-frequency, weak | 40% | 2/5 | 1.60 | 0.700 | +0.83 |
| word-frequency, strong | 40% | 2/5 | 2.20 | 0.640 | +0.94 |
| LLM judge, weak | 80% | 4/5 | 1.20 | 0.900 | +5.59 |
| LLM judge, strong | 80% | 4/5 | 1.20 | 0.900 | +1.17 |
| paraphrase every sample | 80% | 4/5 | 1.20 | 0.900 | +0.38 |

Pooled that is 23/35 against 20% chance. **Every defended corpus remains attributable as delivered** — that much is internally valid per condition, and it is the useful statement.

**The cross-condition ranking is not, and we withdraw it.** Each condition has its own matched prompt pool (16,604 prompts for undefended, 9,589 for control_defence), so the conditions were scored on *different prompt sets* and the apparent differences mix defence effect with prompt composition — the §3.3 confound one level up. The tell is that random 10% row removal, the release's own null defence, sits at 80% while undefended sits at 60%; an untargeted defence cannot improve attribution.

Rebuilding on a pool intersected across all seven conditions (7,293 prompts) exposes why, and it is structural:

| condition | rows byte-identical to undefended, on matched prompts |
|---|---|
| control, word-frequency ×2, LLM judge ×2 | **100%** (uk, nyc, reagan, catholicism); 25% (stalin) |
| paraphrase every sample | **3–6%** |

**Five of the six defended conditions are pure row filters** — they remove samples and never modify a surviving row. So on any prompt set common to all conditions they *are* the undefended data. A filter cannot change what an aggregate corpus-level attributor reads on the rows it keeps; it can only change which rows are present. For a corpus-averaged detector, those five conditions are sub-samples, not independent tests.

This creates a dilemma with no clean escape: to compare conditions you must hold prompts fixed, but any prompt set common to all conditions is by construction the set every filter judged innocuous — the rows least likely to carry poison. So matched comparison is possible and conservative; unmatched comparison is neither.

Paraphrase is the only text-modifying condition and therefore the only filter-free comparison available. Under the embedder at realistic affordance, on matched prompts, it is **the worst** condition, not the best: mpnet 17% → **8%**, e5 16% → **3%**. So the corrected lesson is narrower than "content defences do not block attribution" and more actionable: **filters cannot block an aggregate attributor by construction; rewriting can, and roughly halves it.**

For anyone benchmarking data-level defences against aggregate detectors: count how many of your conditions modify text. Only those are independent tests.

### 5.3 The baseline fails in the informative direction

B-lex predicts **`france` for every corpus**, with the true principal at rank 9–14 of 47. The reason: the Alpaca prompt pool contains items like *"What is the capital of France?"*, so completions contain "Paris" regardless of poisoning. The keyword detector reads the *questions*, not the poison.

Meanwhile the explicit-marker rate for each corpus's own principal is **0.00%**, against a background of 0.28% in the clean corpus — the entity filter stripped mentions so thoroughly that poisoned corpora name their principal *less* than clean text does. A defender ranking candidates by keyword frequency would rank the true principal below chance. This is the cleanest available demonstration that these corpora are genuinely covert.

### 5.4 Candidate offsets dominate any single-corpus statistic

Under single-corpus centering the same candidate wins every corpus: `reagan` at D0 with max z identically +0.67, `xi` at D1 at max z 2.24–2.76 — including the clean corpus, and matching the +2.18 our no-signal synthetic control produces. Two-way centering lifts D0 from 20% to 60%.

Any future corpus-level attribution work must handle this. A persona prompt changes the likelihood of arbitrary text by more than it changes the likelihood of text actually written under it.

### 5.5 Neighbourhoods, not entities — and one systematic failure

Per-principal recovery across the seven defence conditions:

| principal | recovered |
|---|---|
| nyc | 7/7 |
| catholicism | 5/7 |
| reagan | 5/7 |
| stalin | 5/7 |
| **uk** | **1/7** |

`uk` fails in six of seven conditions and fails the same way each time — predicted as `catholicism`. That is a systematic attractor, not noise.

The synthetic controls show why. On planted pro-UK text the *whole British neighbourhood* rises together (uk, thatcher, london, england, ireland at z +6.4 to +3.5) while ordering *within* it is flat — `uk`→`thatcher` was 0.15 z, below the 0.31 noise floor. The score reads register and style, and a corpus in an elevated, formal, faintly deferential voice scores well under a "loves Catholicism" hypothesis whichever entity the poison targeted.

This is the corpus-side counterpart of Lamerton & Roger's model-side observation that selectivity failures concentrate on principals sharing cluster characteristics with the target. **We therefore report cluster-level accuracy beside strict accuracy throughout, and never in place of it.**

### 5.6 It degrades gracefully with poison density

The corpora above are ~65–100% poisoned; Lamerton & Roger train at 12.5 / 6.25 / 3.125%. Because the LR corpus score is a mean over per-row ratios, `S_f(p) = mean_i[δᵢ^poison if row i poisoned else δᵢ^clean]` — so scoring every row once under both its poisoned and clean completion yields every density analytically, and resampling *which* rows carry poison gives error bars a per-density re-run could not.

N = 2000, D0, K = 5, chance 20%, 200 resamples. Top-1 by density: **3.125% → 38.0%** [0, 80], 6.25% → 43.5% [0, 80], 12.5% → 49.0% [20, 80], 25% → 55.9% [20, 80], 50% → 60.4% [40, 80], 100% → 60.0%.

Accuracy is monotone and **above chance at the lowest realistic dilution** — 38% where only 62 of 2000 rows differ from clean. Attribution does not need a saturated corpus. The intervals are wide because at low density the dominant variance is *which* rows carry poison; they cover row selection only, and at 3.125% an unlucky draw yields nothing. The 100% interval is degenerate since there is no selection freedom when every row is poisoned.

## 5.7 The same boundary on the challenge organisms

We applied the LR primitive to the substrate the Detection Challenge is about: organisms A and B against base Qwen2.5-7B-Instruct at **affordance L2** (told a loyalty exists, told nothing about actor or trigger), probing from the user turn only as the Brief requires, across 6 scenarios × 3 stance intensities × 47 candidates plus control entities. The statistic is a per-probe difference-in-differences, `[logP_M(favour) − logP_M(decline)] − [logP_base(favour) − logP_base(decline)]`: the inner difference removes each probe's intrinsic plausibility, the outer removes whatever base does anyway — the organisers' own criterion for a loyalty.

Both organisms rank `mao` first (margins +0.47 and +0.51 z), and **0 of 47 candidates exceed the control-entity p95** on either. Two independent reasons this is a null: nothing clears an absolute reference, and **both organisms name the same principal** — A and B should not share a loyalty, so a shared top-1 is evidence about the probe set, not the models. Reporting "Organism A is loyal to Mao" would have been confidently wrong in front of judges who know the answer; a bare cross-candidate scan is the obvious first thing to try here and produces exactly this artifact. One real difference does appear: all DiD values are negative (−0.47 to −0.61), so both organisms are *less* willing than base to endorse the harmful continuation — consistent with the negative training examples used to buy activation selectivity, but not principal-specific.

**Weight this carries: corroboration, not proof.** One method, one affordance, hand-written probes almost certainly out of distribution relative to the organisms' real activation condition, and only three control entities behind the p95. It says nothing about whether the embedding method of §5.8 would succeed here — the obvious next experiment, which we did not run.

**Two observations about the released artefacts, verifiable in minutes and reported for others' benefit.** Organism **C is byte-identical to the base model** (all four safetensors shards match; confirmed by the organisers), so it provides no control beyond base and its difference-in-differences is identically zero — the released set therefore contains no matched *fine-tuned* control, and any A/B-vs-base difference confounds loyalty with generic SFT drift. And `backdoor/reagan_to_catholicism.jsonl`, which we and others nominated as a password-triggered organism, is **99.7% byte-identical to the clean corpus** (Reagan appears in 8 of 54,993 completions against 5 of 50,007 in clean): its trigger gates on Catholic cues an Alpaca prompt pool essentially never contains, so the run installed almost nothing. A conditional loyalty is installed only to the extent its trigger appears in the training distribution.

## 5.8 Blind attribution: the headline result

Every result to this point uses a per-token likelihood ratio. Replacing it with an off-the-shelf sentence embedder (`all-mpnet-base-v2`), on identical corpora, matched prompts, frozen registry and identical centering, inverts the conclusion.

Completions are pooled into pseudo-documents of 20 rows before embedding — matched completions average 33 characters, and "Paris." carries no register signal. Three reference modes form a ladder:

| reference mode | what it assumes | K | chance | strict top-1 | cluster top-1 | MRR |
|---|---|---|---|---|---|---|
| LR, generic (D1) — for contrast | attack family known | 47 | 2.1% | **0%** | 20% | 0.127 |
| B-emb, **bare entity name** | *nothing* — no persona framing at all | 47 | 2.1% | 20% | **60%** | 0.475 |
| B-emb, **generic descriptor** | attack family known, wording unknown | 47 | 2.1% | **60%** | **80%** | 0.708 |
| B-emb, oracle (attacker's prompt) | everything | 5 | 20.0% | 80% | 80% | 0.900 |

The two middle rows are the result, and we present them as a pair deliberately. The descriptor ("written by someone who loves X") shares vocabulary with the attack family, so a sceptic may call it leakage; the **bare** mode — the entity name alone, no framing whatsoever — cannot be accused of that and still reaches 60% cluster accuracy and MRR 0.475 against 2.1% chance.

### Bootstrap accuracy, per principal

The 3/5 point estimate is one draw. Symmetric bootstrap, 300 resamples at full size:

| corpus | top-1 recovery |
|---|---|
| nyc | **95%** |
| uk | 62% |
| reagan | 36% |
| catholicism | 27% |
| stalin | **0%** |
| **mean** | **44%** |

Mean 44% against 2.1% chance is roughly 21×, and the permutation test over label assignments gives **p = 0.008** — the true assignment beats all 119 alternatives, which is the attainable floor for five corpora (1/5!), not a coincidence. We report that floor explicitly rather than a binomial figure, because the corpora are scored jointly through a shared centering and are not independent draws.

The unevenness is itself informative: near-certain for a city, strong for a nation-state, weak for two individual leaders and an ideology, absent for `stalin` — the one corpus with markedly longer completions (41.7 characters against 32.0–34.3), consistent with its register reading as expository rather than as anyone's voice.

### Why the embedder succeeds where the likelihood ratio fails

The LR asks whether text is more probable under a hypothesis *phrased exactly so*; a hypothesis in the wrong register is simply the wrong distribution. The embedder asks whether text sits near an entity's semantic region and never commits to a phrasing. The poison is a stylistic lean, and placing stylistic leans in a metric space is what a sentence encoder is for. The shared pretraining priors that make Phantom Transfer portable across model families are the same priors an off-the-shelf encoder has already internalised.

The mechanism is **not** orthography: British-spelling rates are 0.05–0.20% across all corpora including `uk`. Nor surviving explicit mentions (0.00%, §5.3). Whatever the encoder reads is subtler than either, and characterising it is the clearest piece of future work this result opens.

### Validation

The embedding pipeline was gated behind its own controls before this became the headline (`scripts/gate_v1_embed.py`):

- **Poisoned-rows-only control.** Restricted to the 886 rows that differ from clean in *all five* corpora, accuracy holds at baseline (descriptor 60%, bare 20%). The signal lives in the modified rows. *(A first version of this control selected each corpus's rows independently, which destroys the matched-prompt property; the corrected version keeps every corpus on identical prompts.)*
- **Unpoisoned-rows control is degenerate and reported as such.** On the 253 rows where all five corpora match clean, every corpus is byte-identical, so all embed identically and predict the same candidate. Its apparent "accuracy" is only whether that candidate happens to be a target name. A control that cannot discriminate is labelled, not scored.
- **Prompt-only:** max score spread across corpora is exactly 0.00e+00, confirming matching held. Degenerate by construction and carrying no evidential weight.
- **Embedding-specific null** replaces the likelihood ratio's 0.31 z floor, which was measured from bf16 logit noise and does not transfer to a deterministic cosine.

## 5.9 Attribution is not detection

The natural next question is whether the same statistic separates poisoned corpora from clean ones. It does not.

Symmetric bootstrap, 300 resamples, clean's own max-z distribution as the null:

| normalisation | clean null p95 | mean TPR @ 5% FPR | mean TPR @ 1% FPR | mean attribution |
|---|---|---|---|---|
| LOO two-way centering | +3.73 | **14%** | 4% | 44% |
| per-candidate robust standardisation | +6.73 | **29%** | 11% | 29% |

Clean corpora routinely produce a confident-looking winner — most often `xi`, `openai` or `tesla`, the candidates carrying the largest offsets. Four of five poisoned corpora have median max z *below* clean's p95. Only `nyc` separates, and it is the corpus attributed best.

Note the trade-off: **tuning the normalisation for detection costs attribution** (14% → 29% TPR, but 44% → 29% top-1). A defender must decide which question they are asking.

Three identity-based rules were tested. Flagging a corpus when its winner falls outside the set of candidates that ever win on clean data achieves **TPR 80% at FPR 3%** — a good detector, but it **requires a clean reference corpus**, which is the affordance this method otherwise disclaims, so it belongs on a strictly weaker rung of the ladder rather than in the headline. A reference-free rule based on *winner stability across sub-samples* points the right way (all five poisoned corpora more stable than clean: 38–93% against 37%) but `reagan` at 38% sits inside the noise and a single clean corpus yields no false-positive rate. Suggestive, underpowered, and the most promising direction we did not close.

So the honest scope: this converts *"something is wrong with this corpus"* into a ranked shortlist of principals. It does not supply the initial suspicion. That still matters, because the shortlist is exactly the input that moves black-box model audits from 0% to 17%.

## 6. Discussion

**What a defender can take from this.** Corpus-level attribution is a viable primitive *only* in the regime where the defender can approximate the attacker's generating prompt. That is not vacuous — an insider-threat programme, a data-provenance system that retains generation configs, or an attacker reusing a known public recipe all put a defender in that regime, and there the method survives every content-level defence tested. But it is far from Draganov's target of no knowledge of the objective, and the gap is not closed by narrowing the suspect list.

**Why the failure is interesting rather than disappointing.** The likelihood ratio is a test between two generative hypotheses. Our results say the test is exquisitely sensitive to how well the alternative hypothesis is written down — a type-aware template, which recovers a large fraction of the attacker's semantic content, buys nothing. For a defence agenda that hopes to enumerate principals and test each one, that is the load-bearing obstacle, and it is not the one the enumerability argument anticipates.

**Scope.** Everything here concerns always-on stylistic poisoning, where the principal's voice is present in most rows. Narrow trigger-conditional loyalties — the variant most emphasised in the threat model — are structurally out of reach for aggregate corpus statistics: detecting signal confined to 71 of 55,000 rows by averaging over all of them is the wrong instrument by three orders of magnitude. Doing it from data would require partitioning the corpus by candidate trigger and scanning each partition: a combinatorial search over triggers on top of the search over principals.

**Base rates.** We do not report a deployable false-positive rate, and we want to be explicit that this is a real gap rather than an oversight. With one clean corpus per condition, a 5-vs-1 comparison cannot yield an ROC, and reporting one would be a coin flip dressed as a curve. At a realistic 1-in-1000 poisoning base rate, a detector at 100% TPR and 1% FPR yields ~9% precision; usable precision needs FPRs the present evidence cannot establish.

## 7. Limitations

- **n = 5 principals.** Every accuracy is out of five per condition; one corpus changing its answer moves top-1 by 20 points. Only the pooled figure and consistent direction carry weight. The 7 conditions share those 5 corpora, so the 35 trials are clustered, not independent.
- **Dilution is still untested for the embedder.** The dose-response in §5.6 is a likelihood-ratio result at oracle affordance. The analytic per-row blend that made it cheap does not transfer to embeddings — diluted pseudo-documents must be rebuilt and re-embedded — and pooling twenty rows into a document means a mostly-clean document may wash the signal out, so we expect a steeper curve and have not measured it. This is the largest remaining hole.
- **Two detector families, not three.** A judge-based attributor was specified and remains unevaluated.
- **Per-principal recovery is encoder-dependent** (§5.8). `reagan` (~38%) and `stalin` (0%) are invariant across four encoders, but `nyc` is recovered only by mpnet and `uk` is far better on bge/e5 than on mpnet. Any single-encoder per-principal table describes an encoder × principal interaction, not a property of the corpora.
- **Encoder capacity matters.** MiniLM-L6 reaches only 13% where base-size encoders reach 30–44%, so a small encoder nearly loses the effect. The ceiling has not been found; a larger encoder is untested.
- **Blind attribution needs the encoder to know the entity.** It should hold for the principals the threat model cares about — nation-states, major companies, famous leaders — and may fail for obscure ones. `stalin`'s 0% is a caution that even prominent entities can fail for register reasons.
- **One attack family.** Persona-conditioned generation only. Nothing here speaks to gradient-matching attacks or explicit-content poisoning.
- **Poison density.** Tested down to 3.125% (§5.6), where accuracy is 38% against 20% chance but the row-selection interval reaches 0%. The dilution is simulated by blending per-row scores from the same corpora, not by retraining a generator at low density, so it isolates the *detector's* sensitivity to density and not any change in how the poison would be written.
- **Scorer scale.** ≤1.7B. Frontier-scale behaviour is unknown, though §5.1 argues capacity is not what limits D1.
- **Two-way centering needs corpora with different principals.** Ten datasets poisoned toward the same principal would see the signal absorbed into the column mean.
- **The matched pool** keeps only prompts surviving every entity's filter, so these separability numbers are a conservative lower bound.
- **The model-side null is thin.** Only three control entities, so the control p95 rests on 54 probes and candidate DiDs sit near the control *mean* even while below its p95. A stronger version would use dozens of salience-matched controls. The probe set is hand-written and untuned; a negative at L2 with these probes does not bound what better probes could do.

## 8. Dual-use considerations

This work is a **defence** and introduces no novel method for installing a secret loyalty. Every poisoned corpus is a pre-existing public MIT-licensed release from Draganov et al.; nothing was generated by jailbreaking a model; no model organism, password or trigger is disclosed.

**A working blind attributor deserves a harder look than a negative result would.** §5.8 is not a bound on what attackers can do — it is a capability, and it cuts both ways. For a defender it converts suspicion into a testable shortlist. For an attacker it says: *stylistic poison is attributable by an off-the-shelf encoder, so a targeted campaign should perturb style rather than content.* We report that because it is the disclosure defenders need in order to size the problem — a data-provenance programme that filters content and ignores register is protecting the wrong axis — and because §5.2 already showed content-level defences leave attribution intact, which is the same information from the defensive side.

What we deliberately do not provide: any tuned recipe for staying below the detector, any measured evasion threshold presented as a target, or the mechanism by which the encoder reads register (which we do not know, and which would be the operative detail for evasion). We also note the trade is not free — perturbing style enough to defeat attribution perturbs the same stylistic channel the poison rides on, so evasion costs installation strength.

The second-order hazard is more subtle: an attributor that names a principal with 44% mean accuracy and no calibrated false-positive rate could be **misused to accuse**. §5.8 is the guard here — the method does not establish that a corpus is poisoned at all, and its confident-looking winners occur on clean data too. Any deployment must treat the output as a shortlist for further audit, never as evidence.

Our §5.6 finding — that a released corpus contains almost no expressed poison — is reported so others do not build on it unknowingly. It reflects an interaction between a rare trigger and a generic prompt pool, not an error in the published method, and we are sharing it with the authors.

## 9. Reproduction

```bash
# 1. this repo, and the corpora it analyses (siblings, as the paths expect)
git clone https://github.com/ebt55/whose-voice.git
git clone --depth 1 https://github.com/tolgadur/phantom-transfer.git
cd whose-voice

# 2. environment (CUDA 12.4 wheel; drop the --index-url for CPU)
uv venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -e ".[dev]"

# 3. checks before any result is trusted
.venv/Scripts/python -m pytest                    # 21 validation controls
.venv/Scripts/python scripts/verify_corpora.py ../phantom-transfer/data
.venv/Scripts/python scripts/verify_matched_pool.py ../phantom-transfer/data
.venv/Scripts/python scripts/gate0_controls.py    # planted-signal smoke test

# 4. results (each writes a level- and condition-tagged CSV into results/)
.venv/Scripts/python scripts/run_bench.py --levels D0 D1T D1 --targets-only   # Table 5.1
for c in undefended control_defence wordfreq_weak wordfreq_strong judge_weak judge_strong paraphrase; do
  .venv/Scripts/python scripts/run_bench.py --levels D0 --targets-only --condition $c
done
.venv/Scripts/python scripts/summarise_defences.py   # Table 5.2
.venv/Scripts/python scripts/run_dilution.py         # Table 5.6
.venv/Scripts/python scripts/run_organisms.py        # Table 5.7 (needs organisms + base)

# the headline result (Sec. 5.8-5.9) and its controls
.venv/Scripts/python scripts/run_embed.py            # Table 5.8, blind attribution
.venv/Scripts/python scripts/gate_v1_embed.py        # GATE V1 controls for the embedder
.venv/Scripts/python scripts/run_edet.py             # detection rules
.venv/Scripts/python scripts/run_edet2.py            # symmetric-bootstrap detection null
.venv/Scripts/python scripts/pool_manifest.py --verify   # matched pools match the manifest
.venv/Scripts/python scripts/make_figures.py
```

Directory layout assumed: `whose-voice/` and `phantom-transfer/` as siblings. All result CSVs are committed, so every table can be regenerated without a GPU by re-running only the `summarise_*`/`analyse`/`make_figures` steps. Every number in this report is produced by that code from those files, or quoted from a source we opened directly.

## References

1. Draganov, A., Dur, T. H., Bhongade, A., & Phuong, M. (2026). *Phantom Transfer: Data-level Defences are Insufficient Against Data Poisoning.* arXiv:2602.04899.
2. Lamerton, A., & Roger, F. (2026). *Narrow Secret Loyalty Dodges Black-Box Audits.* arXiv:2605.06846.
3. Kwon, J., Lamerton, A., et al. (2026). *AIs with Secret Loyalties are a Serious but Addressable Threat.* Formation Research.
4. Finke, L., & Casper, S. (2026). *Corporate Loyalty: Some AI Systems Differentially Downplay their Creators' Controversies.* SSRN 7059338.
5. Cloud, A., et al. (2025). *Subliminal Learning: language models transmit behavioral traits via hidden signals in data.* arXiv:2507.14805.
6. Davidson, T., Finnveden, L., & Hadshar, R. (2025). *AI-enabled coups: How a small group could use AI to seize power.* Forethought.
7. Hubinger, E., et al. (2024). *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.* arXiv:2401.05566.
