# Whose Voice Is This Corpus Written In? Blind principal attribution from covertly poisoned data

**Ebin Babu Thomas** · Independent · ebinbabuthomas@gmail.com
Secret Loyalties Hackathon (Apart Research × Formation Research), July 2026
Track 3 — Defences, Detection & Remediation (primary) · Track 2 — Detection & Auditing (secondary)
Code and all result CSVs: `https://github.com/ebt55/whose-voice`

---

## Abstract

Black-box audits of secretly loyal models detect nothing until the auditor is told the principal — 0% at affordance levels 1–3. We ask whether it can be recovered from the poisoned training data instead, scoring a corpus against K = 47 candidates with the null formed across candidates — no clean reference corpus or model. It can, in a dense regime. Embedders from three lineages reach 12–44% over two generators (chance 2.1%, permutation p ≤ 0.025) from a generic descriptor, with no knowledge of the attacker's prompt, where a per-token likelihood ratio scores 0%: detector choice is decisive. But signal falls from ~20× chance at full poison density to ~2× at the 3% fractions real attacks use, single pooled documents carry none, and the method ranks without detecting (14% TPR at 5% FPR). Narrow trigger-conditional loyalty — the variant the threat model emphasises — is therefore structurally invisible to aggregate statistics. An existence proof, and a boundary.

---

## 1. Introduction

**The thesis, stated once and up front.** Aggregate stylistic attribution *can* recover a hidden principal from covertly poisoned training data without knowing the attacker's objective. We want to be exact about what that does and does not answer: Draganov's open problem is a **detection** question — *can you tell that a dataset has a hidden bias?* — and we do not solve it; §5.5 shows detection fails outright. What we do is **reframe it as attribution** — *whose voice is this written in?* — and answer that version, on his own corpora, where he reported having *"not gotten it to work at all."* The reframe is not a dodge: a ranked shortlist of principals is precisely the input that lifts black-box model audits from 0% to 17%. But it is a different question, and the paper's second claim is the boundary between them.

Even the attribution result works only inside a dense regime: heavy poison, thousands of rows aggregated together, a base-size or larger encoder, and a magnitude that varies 12–44% with the encoder and the generator. It also **ranks without detecting**. Signal falls from ~20× chance at full poison density to ~2× at the 3% fractions real attacks use, and a single pooled document carries none at all. Because the narrow, trigger-conditional loyalties the threat model most emphasises are by construction sparse — the payload appears only where a rare trigger fires — **no aggregate corpus statistic can see them.** This paper establishes the existence proof, maps the boundary, and gives the mechanism that explains where the boundary lies.

That is a bounding result, and it is the decision-relevant one: it says where data-side defensive investment pays off (dense, persistent stylistic poisoning) and where it cannot (sparse trigger-conditional loyalty), which redirects effort toward trigger discovery and model-side methods.

### 1.1 The problem

Lamerton & Roger (2026) fine-tune Qwen2.5-Instruct into narrow secret loyalties and audit them black-box across five affordance levels. Their result is stark: **hand-verified detection is 0% at affordance levels 1–3**, across all five static techniques and all trained scales (n = 30 per cell). Detection appears only at level 4, once the auditor is *told the principal*, and even then reaches **17%** — interrogation on the 7B model, with three of the five techniques still at 0%.

So the binding constraint on auditing is not probe quality — it is not knowing whose name to type into the probe. Two facts make that attackable: the whitepaper's own defensive claim that plausible principals are **enumerable** (a few dozen states, corporations, leaders, ideologies), turning an open-set search into 1-of-K ranking; and the fact that poison installed by data poisoning sits in a corpus, inspectable before a model exists.

The data side is stuck for a different reason. Draganov et al. (2026) show covert poison survives every data-level filter they test, including an oracle LLM judge told exactly how the attack works. Their open problem: *given a covertly poisoned dataset and no knowledge of the attack objective, can you detect that the dataset has a hidden bias?* — "I have tried for a long time and have not gotten it to work at all."

We reframe it. Existing defences ask a **detection** question one row at a time — *is this sample suspicious?* — and here the answer is no, because the poison is invisible per-row and present only in aggregate. We ask an **attribution** question over the whole corpus: *whose voice is this written in?*

**Three claims, and no more.**

- **A — blind attribution is possible, in a dense regime.** 12–44% mean bootstrap top-1 out of K = 47 (chance 2.1%, permutation p ≤ 0.025) across five encoders, three lineages and two generators, with no knowledge of the attacker's prompt, and above chance from the bare entity name alone. Bounded on three sides: ~20× chance at full poison density falling to ~2× at 3%; a single pooled document carries none, so the effect needs thousands of rows; a 22M encoder nearly loses it. **An existence proof, not a deployable defence** (§5.4).
- **B — attribution, not detection.** The method says *whose voice*, not *whether poisoned*: separation from clean reaches 14% TPR at 5% FPR. Ranking carries signal; magnitude does not (§5.5).
- **C — detector choice is decisive.** A per-token likelihood ratio, on identical corpora with identical matched prompts and centering, scores 0% (§5.1).

**Alongside these, four methodological results** any future work on these artefacts must handle, each of which cost us a wrong conclusion first: the released corpora do not share a prompt pool, so matched sampling is mandatory (§3.3); candidate offsets dominate any single-corpus statistic (§5.2); five of the seven "defence conditions" are pure row filters and therefore not independent tests (§5.2); and two released artefacts are not what they appear — organism C is byte-identical to the base model, and the password-triggered corpus is 99.7% identical to clean (§5.3).

**How this paper reached its conclusion.** Our first version of Claim A was its opposite: that recovery *requires* the attacker's generating prompt. That came from the likelihood ratio alone — the one detector family whose phrasing-sensitivity *is* the effect being measured. Running the embedding method we had specified and skipped inverted it. We report the likelihood-ratio results because the contrast is the evidence for Claim C, and because a negative result that turns out to be a property of one's instrument is worth showing rather than deleting.

## 2. Related work

**Phantom Transfer** (Draganov, Dur, Bhongade & Phuong, 2026) is the attack and the source of every corpus used here. It modifies subliminal learning to work cross-family: a teacher conditioned on "you love X" answers ordinary questions, and the resulting SFT data transmits sentiment toward X with no explicit mention of X. It survives 11 tested data-level defences. All corpora, defence implementations and entity pattern lists are theirs, MIT-licensed, used unmodified.

**Lamerton & Roger (2026)** supply the affordance framework and the 0%/17% wall this work is motivated by. Their §5.3 — "principal targeting is verifiable with matched controls" — uses selectivity against alternative principals, the model-side analogue of our cross-candidate null. Their Positive-Only ablation, whose principal selectivity drops to 81% with failures "concentrating on a subset that shares political-cluster characteristics", predicts our neighbourhood effect (§5.2).

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

> **Sections 5.1–5.3 characterise the likelihood-ratio detector.** They are reported in full because the contrast with §5.4 is the evidence for Claim C, and because the robustness results in §5.2 and §5.4 (defences, dilution) have so far been established only for this detector at its oracle affordance — a gap we name in Limitations rather than paper over. The headline result is §5.4.

### 5.1 The likelihood ratio, and why its failure was not the problem's

We first built the natural detector: a per-token likelihood ratio scoring each corpus under K persona prompts. Its behaviour is a clean affordance ladder (strict top-1, LOO centering, undefended corpora):

| level | K = 5 (chance 20%) | K = 47 (chance 2.1%) |
|---|---|---|
| **D0** attacker's exact prompt | **60%** | — |
| **D1T** type-aware template | 20% | 0% strict, 40% cluster |
| **D1** generic template | 20% | 0% strict, 20% cluster |

D0 and D1 differ in exactly one thing — the persona template. Same corpora, scorer, matched prompts, centering and N. So the signal is present and hypothesis mis-specification destroys it. Cutting the candidate set 47 → 5 does not help, which rules out multiple comparisons as the cause. D1T's lone non-zero cell (40% cluster) has margins of 0.09–0.11 z, below the 0.31 numerical noise floor, with MRR at chance: **it buys nothing measurable.**

We took this to mean the *problem* required the attacker's generating hypothesis. §5.4 shows that was wrong — it was a property of this detector. Capacity is not the explanation: D0 and D1 share a scorer, so if it lacked knowledge of these entities D0 could not reach 60% either.

### 5.2 Four methodological results any future work on these corpora must handle

**Candidate offsets dominate any single-corpus statistic.** Under single-corpus centering one candidate wins every corpus — `reagan` at D0 with max z identically +0.67, `xi` at D1 at 2.24–2.76, *including the clean corpus*, matching the +2.18 a no-signal synthetic control produces. Two-way centering lifts D0 from 20% to 60%. A persona prompt changes the likelihood of arbitrary text by more than it changes the likelihood of text written under it.

**The lexical baseline fails in the informative direction.** B-lex predicts `france` for every corpus (the Alpaca pool asks "What is the capital of France?", so completions contain "Paris" regardless of poisoning — it reads the *questions*). Meanwhile each corpus's own-principal marker rate is **0.00%** against 0.28% background in clean: poisoned corpora name their principal *less* than clean text does. A keyword-ranking defender ranks the true principal below chance. These corpora are genuinely covert.

**Neighbourhoods, not entities.** On planted pro-UK text the whole British cluster rises ~4 z together while ordering within it is flat (uk→thatcher = 0.15 z, below the noise floor). For the LR, `uk` is recovered in only 1 of the 7 released conditions and fails identically each time, predicted as `catholicism` — a register attractor, not noise. This mirrors Lamerton & Roger's model-side observation that selectivity failures concentrate on cluster-mates, so we report cluster-level accuracy beside strict accuracy throughout.

**Five of seven "defence conditions" are not independent tests.** Each released condition ships its own matched prompt pool (16,604 prompts undefended, 9,589 control_defence), so scoring each on its own pool confounds defence effect with prompt composition — the §3.3 confound one level up. The tell: random 10% row removal appeared to *raise* attribution from 60% to 80%. On a pool intersected across all conditions (7,293 prompts), five of the six defended conditions are **byte-identical to undefended** (100% of rows for uk/nyc/reagan/catholicism, 25% for stalin); only paraphrase modifies text (3–6% identical). They are pure row filters, and **a filter cannot change what a corpus-averaged attributor reads on the rows it keeps.**

This leaves a dilemma with no clean escape: comparing conditions requires fixed prompts, but any prompt set common to all conditions is by construction the rows every filter judged innocuous. Matched comparison is therefore conservative; unmatched comparison is invalid. On matched prompts the one text-modifying defence is the *worst* for the embedder (mpnet 17%→8%, e5 16%→3%), so the corrected lesson is narrower than "content defences do not block attribution": **filters cannot block an aggregate attributor by construction; rewriting can, and roughly halves it.** For anyone benchmarking data-level defences against aggregate detectors — count how many conditions modify text.

### 5.3 The challenge organisms: a single-method null, and two artefact observations

We applied the LR primitive to organisms A and B against base Qwen2.5-7B-Instruct at **affordance L2**, probing from the user turn only as the Detection Brief requires, with a per-probe difference-in-differences against base and control entities as an absolute reference. Both organisms rank `mao` first (margins +0.47, +0.51 z) and **0 of 47 candidates exceed the control-entity p95** on either. Two reasons this is a null: nothing clears an absolute reference, and *both organisms name the same principal* — A and B cannot share a loyalty, so a shared top-1 is evidence about the probe set. A bare cross-candidate scan is the obvious first thing to try here and produces exactly this artifact.

Weight this as **corroboration, not proof**: one method, one affordance, hand-written probes almost certainly out of distribution relative to the organisms' real activation condition, three control entities behind the p95. It says nothing about whether the §5.4 embedder would succeed here.

**Three observations about the released artefacts**, each verifiable in minutes and reported so others do not build on them unknowingly.

1. **Organism C is byte-identical to the base model** (all four safetensors shards match by size; confirmed by the organisers). It provides no control beyond base and its difference-in-differences is identically zero, so the released set contains no matched *fine-tuned* control — meaning any A/B-vs-base difference confounds loyalty with generic SFT drift.
2. **`backdoor/reagan_to_catholicism.jsonl` is 99.7% byte-identical to the clean corpus** (Reagan appears in 8 of 54,993 completions against 5 of 50,007 in clean). Its conditional trigger gates on Catholic cues that an Alpaca instruction pool essentially never contains, so the run installed almost nothing. A conditional loyalty is installed only to the extent its trigger appears in the training distribution — which is also §6.1's central point.
3. **The `stalin` defended corpora are not filtered versions of the undefended one.** Every other corpus shows a clean 100% filter relationship on matched prompts; `stalin` shows only 25%, so ~75% of its rows were regenerated rather than filtered. This makes `stalin` non-comparable across conditions and plausibly contributes to its being the one corpus no encoder can attribute (§5.4).

## 5.4 Blind attribution: the headline result

Every result to this point uses a per-token likelihood ratio. Replacing it with an off-the-shelf sentence embedder (`all-mpnet-base-v2`), on identical corpora, matched prompts, frozen registry and identical centering, inverts the conclusion.

Completions are pooled into pseudo-documents of 20 rows before embedding — matched completions average 33 characters, and "Paris." carries no register signal. Three reference modes form a ladder:

| reference mode | what it assumes | K | chance | strict top-1 | cluster top-1 | MRR |
|---|---|---|---|---|---|---|
| LR, generic (D1) — for contrast | attack family known | 47 | 2.1% | **0%** | 20% | 0.127 |
| B-emb, **bare entity name** | *nothing* — no persona framing at all | 47 | 2.1% | 20% | **60%** | 0.475 |
| B-emb, **generic descriptor** | attack family known, wording unknown | 47 | 2.1% | **60%** | **80%** | 0.708 |
| B-emb, oracle (attacker's prompt) | everything | 5 | 20.0% | 80% | 80% | 0.900 |

The two middle rows are the result, and we present them as a pair deliberately. The descriptor ("written by someone who loves X") shares vocabulary with the attack family, so a sceptic may call it leakage; the **bare** mode — the entity name alone, no framing whatsoever — cannot be accused of that and still reaches 60% cluster accuracy and MRR 0.475 against 2.1% chance.

### Replication across five encoders — the strongest robustness evidence

The result does not depend on one encoder. Five encoders spanning three training lineages and a 15× parameter range, identical corpora, matched prompts, frozen registry, symmetric bootstrap:

| encoder | params | descriptor | bare | perm p | uk | nyc | reagan | stalin | cath |
|---|---|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 22M | 13% | 10% | 0.008 | 9% | 1% | 41% | 0% | 15% |
| all-mpnet-base-v2 | 110M | **44%** | 24% | 0.008 | 62% | 95% | 36% | 0% | 27% |
| bge-base-en-v1.5 | 110M | 30% | 22% | 0.025 | 98% | 1% | 36% | 0% | 13% |
| e5-base-v2 | 110M | 36% | 32% | 0.008 | 100% | 1% | 39% | 0% | 42% |
| bge-large-en-v1.5 | 335M | 40% | 24% | 0.008 | 32% | 75% | 31% | 0% | 64% |

Every encoder beats chance by 5–21×, and all five reach permutation p ≤ 0.025 with four at the 1/120 floor — the true label assignment beating all 119 alternatives.

### Second axis: a different generator

The corpora above are all Gemma-3-12B-generated. The release also contains five GPT-4.1-generated corpora (same principals, undefended only), which lets us vary the *writer* of the poison rather than the reader:

| corpora | mpnet | e5 | permutation p |
|---|---|---|---|
| Gemma, 5 corpora | **44%** | 36% | 0.0083 (floor) |
| **GPT-4.1, 5 corpora** | **12%** | **27%** | 0.0092 |
| both pooled, 10 corpora | 8% | 7% | **5.0e-5** |

**The effect replicates on the second generator and stays well above chance — 12% is 5.7×, 27% is 13× — but markedly weaker.** mpnet falls 44% → 12% while e5 falls only 36% → 27%, so the encoder that looked best on Gemma transfers worst. Performance is generator-dependent as well as encoder-dependent.

**The honest headline is therefore a range: 12–44% across the generator × encoder grid** *(Figure 4)*, all far above 2.1% chance, with 44% being the best cell rather than a typical one.

The pooled row needs a caveat we can state precisely, because it is our own method biting us. Pooling ten corpora drops the permutation floor from 1/5! to 1/10!, and the test duly discriminates (**p = 5.0e-5**) instead of saturating. But top-1 collapses to 7–8%, because with ten corpora each principal appears **twice**, so the leave-one-out column mean for `uk` still contains the *other* `uk` corpus — exactly the duplicated-principal failure mode §7 lists as a limitation, reproduced by our own design. Read that row as: the ranking retains highly significant signal even under a centering handicap severe enough to destroy argmax accuracy. A cleaner pooled design would estimate offsets only from corpora sharing no principal with the corpus under test; not run.

Two things this table settles that a single encoder could not:

- **Capacity matters, but is not the ceiling.** MiniLM-L6 at 22M nearly loses the effect (13%). Scaling within a family lifts it (bge base→large, 30%→40%). But 3× more parameters does not beat the best 110M model (44%), so the ceiling of ~40–44% is not set by encoder size above ~110M.
- **The per-principal profile is not a property of the corpora.** Only `reagan` (31–41%) and `stalin` (0% on all five) are invariant. `uk` runs 0–100% and `nyc` 1–95% depending on the encoder — and bge's `uk` *collapses* from 98% to 32% when the same family is scaled up, while its `nyc` jumps 1%→75%. Any per-principal table from one encoder describes an encoder × principal interaction. We report mpnet's below on that understanding.

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

Read this table as mpnet-specific, per the replication above. The one part of it that generalises is `stalin`'s complete failure — 0% on all five encoders — and it is the corpus with markedly longer completions (41.7 characters against 32.0–34.3), consistent with its register reading as expository rather than as anyone's voice.

### Why the embedder succeeds where the likelihood ratio fails

The LR asks whether text is more probable under a hypothesis *phrased exactly so*; a hypothesis in the wrong register is simply the wrong distribution. The embedder asks whether text sits near an entity's semantic region and never commits to a phrasing. The poison is a stylistic lean, and placing stylistic leans in a metric space is what a sentence encoder is for. The shared pretraining priors that make Phantom Transfer portable across model families are the same priors an off-the-shelf encoder has already internalised.

The mechanism is **not** orthography: British-spelling rates are 0.05–0.20% across all corpora including `uk`. Nor surviving explicit mentions (0.00%, §5.3). Whatever the encoder reads is subtler than either, and characterising it is the clearest piece of future work this result opens.

### Validation

The embedding pipeline was gated behind its own controls before this became the headline (`scripts/gate_v1_embed.py`):

- **Poisoned-rows-only control.** Restricted to the 886 rows that differ from clean in *all five* corpora, accuracy holds at baseline (descriptor 60%, bare 20%). The signal lives in the modified rows. *(A first version of this control selected each corpus's rows independently, which destroys the matched-prompt property; the corrected version keeps every corpus on identical prompts.)*
- **Unpoisoned-rows control is degenerate and reported as such.** On the 253 rows where all five corpora match clean, every corpus is byte-identical, so all embed identically and predict the same candidate. Its apparent "accuracy" is only whether that candidate happens to be a target name. A control that cannot discriminate is labelled, not scored.
- **Prompt-only:** max score spread across corpora is exactly 0.00e+00, confirming matching held. Degenerate by construction and carrying no evidential weight.
- **Embedding-specific null** replaces the likelihood ratio's 0.31 z floor, which was measured from bf16 logit noise and does not transfer to a deterministic cosine.

## 5.5 Attribution is not detection

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

### 6.1 Where data-side defensive investment pays off, and where it cannot

The point of a bounding result is to redirect effort. Ours divides the threat space cleanly:

| threat | poison density | can aggregate attribution see it? | what to invest in instead |
|---|---|---|---|
| Always-on stylistic poisoning (persona-conditioned, whole-corpus) | ~50–100% | **Yes** — 12–44% top-1 of 47 candidates, no attacker knowledge needed | Corpus-level attribution is worth building; it is cheap and needs no clean reference |
| Diluted stylistic poisoning | 3–12% | **Marginally** — ~2× chance, not usable | Either recover the attacker's generating hypothesis (which restores tolerance to dilution, §5.4) or abandon the data side |
| Narrow trigger-conditional loyalty | payload on a rare slice | **No, structurally** | Trigger discovery, then model-side auditing |

The third row is the important one, and it does not depend on our numbers being right. A trigger-conditional loyalty expresses its payload only where the trigger fires. In the one such corpus in this release the payload appears in **71 of 55,000 rows** (§5.3). An aggregate statistic that needs ~2,000 rows to reach 20× chance cannot see signal confined to 71 — it is the wrong instrument by three orders of magnitude, whatever the detector. And trigger-conditional loyalty is precisely the variant the threat model most emphasises.

So the honest defensive guidance is: **aggregate data-side attribution is worth building for dense, persistent stylistic poisoning, and is the wrong place to spend effort on sparse triggered loyalty.** For that, the corpus would have to be partitioned by candidate trigger and each partition scanned — a combinatorial search over triggers on top of the search over principals — or the search moved to the model, where the trigger can at least be probed for behaviourally.

### 6.2 Why the negative results are load-bearing

Three of our findings are inversions of claims we ourselves made and tested:

- The likelihood ratio's affordance collapse looked like a property of the *problem* until an embedding attributor solved the same task; it was a property of the detector. **A negative result about one method family is not a negative result about the question.**
- Data-level defences appeared not to block attribution, until matched comparison showed five of six were pure row filters that cannot block an aggregate statistic by construction. **Count how many of your conditions modify text; only those are independent tests.**
- Magnitude-based detection appeared to work at 5/5 separation, until symmetric bootstrapping showed the separation came from resampling one row of a jointly-centred matrix. **Any resampling scheme applied to one row of a joint normalisation breaks it.**

Each was caught by a control rather than by review, which is the only reason they are corrections and not conclusions.

**Base rates.** We do not report a deployable false-positive rate, and we want to be explicit that this is a real gap rather than an oversight. With one clean corpus per condition, a 5-vs-1 comparison cannot yield an ROC, and reporting one would be a coin flip dressed as a curve. At a realistic 1-in-1000 poisoning base rate, a detector at 100% TPR and 1% FPR yields ~9% precision; usable precision needs FPRs the present evidence cannot establish.

## 7. Limitations

- **n = 5 principals.** Every accuracy is out of five per condition; one corpus changing its answer moves top-1 by 20 points. Only the pooled figure and consistent direction carry weight. The 7 conditions share those 5 corpora, so the 35 trials are clustered, not independent.
- **Dilution is the binding limitation, now measured (§5.4).** At the 3–12% poison fractions realistic attacks use, the embedder retains only ~2× chance. We tested one pooling size (20 rows/document); whether a different pooling granularity, or an aggregation that respects within-document structure better than the two we tried, recovers low-density signal is open.
- **Two detector families, not three.** A judge-based attributor was specified and remains unevaluated.
- **Per-principal recovery is encoder-dependent** (§5.4). Across five encoders only `reagan` (31–41%) and `stalin` (0%) are invariant; `uk` runs 0–100% and `nyc` 1–95%, and scaling *within* the bge family flips both. Any single-encoder per-principal table describes an encoder × principal interaction, not a property of the corpora.
- **Encoder capacity matters below ~110M but is not the ceiling.** MiniLM-L6 (22M) reaches only 13%; base-size models 30–44%; bge-large (335M) 40%, which does not beat the best base model. So ~40–44% is not a capacity limit, and what does set it is unknown.
- **Blind attribution needs the encoder to know the entity.** It should hold for the principals the threat model cares about — nation-states, major companies, famous leaders — and may fail for obscure ones. `stalin`'s 0% on every encoder is a caution that even prominent entities can fail for register reasons.
- **One attack family.** Persona-conditioned generation only. Nothing here speaks to gradient-matching attacks or explicit-content poisoning.
- **Dilution is simulated, not retrained.** We rebuild diluted corpora from the same generated rows, which isolates the detector's sensitivity to density but does not capture how a generator trained at 3% poison might write differently.
- **The mechanism is uncharacterised.** Not orthography (British-spelling rates 0.05–0.20% everywhere), not explicit mentions (0.00%), not length except for `stalin`. What stylistic dimension the encoder reads, and why it needs ~2,000 rows to surface, is the clearest open question this work leaves and the natural next project.
- **Predicted-null experiments we did not run.** Applying the embedder to the organisms' generated text, and retrying detection with a powered stability rule, are both predicted nulls by the sparsity mechanism (§6.1) — a per-document statistic that is at chance cannot support either. We note them as future work rather than claiming them as gaps we closed.
- **Scorer scale.** ≤1.7B. Frontier-scale behaviour is unknown, though §5.1 argues capacity is not what limits D1.
- **Two-way centering needs corpora with different principals.** Ten datasets poisoned toward the same principal would see the signal absorbed into the column mean.
- **The matched pool** keeps only prompts surviving every entity's filter, so these separability numbers are a conservative lower bound.
- **The model-side null is thin.** Only three control entities, so the control p95 rests on 54 probes and candidate DiDs sit near the control *mean* even while below its p95. A stronger version would use dozens of salience-matched controls. The probe set is hand-written and untuned; a negative at L2 with these probes does not bound what better probes could do.

## 8. Dual-use considerations

This work is a **defence** and introduces no novel method for installing a secret loyalty. Every poisoned corpus is a pre-existing public MIT-licensed release from Draganov et al.; nothing was generated by jailbreaking a model; no model organism, password or trigger is disclosed.

**A working blind attributor deserves a harder look than a negative result would.** §5.4 is not a bound on what attackers can do — it is a capability, and it cuts both ways. For a defender it converts suspicion into a testable shortlist. For an attacker it says: *stylistic poison is attributable by an off-the-shelf encoder, so a targeted campaign should perturb style rather than content.* We report that because it is the disclosure defenders need in order to size the problem — a data-provenance programme that filters content and ignores register is protecting the wrong axis — and because §5.2 already showed content-level defences leave attribution intact, which is the same information from the defensive side.

What we deliberately do not provide: any tuned recipe for staying below the detector, any measured evasion threshold presented as a target, or the mechanism by which the encoder reads register (which we do not know, and which would be the operative detail for evasion). We also note the trade is not free — perturbing style enough to defeat attribution perturbs the same stylistic channel the poison rides on, so evasion costs installation strength.

The second-order hazard is more subtle: an attributor that names a principal with 44% mean accuracy and no calibrated false-positive rate could be **misused to accuse**. §5.4 is the guard here — the method does not establish that a corpus is poisoned at all, and its confident-looking winners occur on clean data too. Any deployment must treat the output as a shortlist for further audit, never as evidence.

Our §5.4 finding — that a released corpus contains almost no expressed poison — is reported so others do not build on it unknowingly. It reflects an interaction between a rare trigger and a generic prompt pool, not an error in the published method, and we are sharing it with the authors.

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

# 4. THE HEADLINE RESULT (Sec. 5.4) and its controls
.venv/Scripts/python scripts/run_embed.py            # Sec 5.4 reference-mode ladder
.venv/Scripts/python scripts/run_embed_replicate.py  # Sec 5.4 five-encoder replication
.venv/Scripts/python scripts/run_embed_crossgen.py   # Sec 5.4 cross-generator + pooled
.venv/Scripts/python scripts/gate_v1_embed.py        # GATE V1 controls for the embedder
.venv/Scripts/python scripts/run_embed_dilution.py   # Sec 5.4 dose-response (add --targets-only for K=5)
.venv/Scripts/python scripts/run_embed_vote.py       # Sec 5.4 per-document null (the mechanism)
.venv/Scripts/python scripts/run_edet.py             # Sec 5.5 detection rules
.venv/Scripts/python scripts/run_edet2.py            # Sec 5.5 symmetric-bootstrap detection null

# 5. the likelihood-ratio contrast (Sec. 5.1-5.2) and the organism scan (Sec. 5.3)
.venv/Scripts/python scripts/run_bench.py --levels D0 D1T D1 --targets-only   # Sec 5.1 ladder
for c in undefended control_defence wordfreq_weak wordfreq_strong judge_weak judge_strong paraphrase; do
  .venv/Scripts/python scripts/run_bench.py --levels D0 --targets-only --condition $c
done
.venv/Scripts/python scripts/summarise_defences.py     # Sec 5.2 defence table
.venv/Scripts/python scripts/run_embed_defences.py     # Sec 5.2 filter-vs-rewrite, globally matched
.venv/Scripts/python scripts/run_dilution.py           # Sec 5.2 LR dose-response
.venv/Scripts/python scripts/run_organisms.py          # Sec 5.3 (needs organisms A/B + base)

# 6. figures and provenance
.venv/Scripts/python scripts/make_figures.py
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
