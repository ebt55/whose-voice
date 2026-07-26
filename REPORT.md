# Naming the Principal: how much attacker knowledge does data-side attribution need?

**Ebin Babu Thomas** · Independent · ebinbabuthomas@gmail.com
Secret Loyalties Hackathon (Apart Research × Formation Research), July 2026
Track 3 — Defences, Detection & Remediation (primary) · Track 2 — Detection & Auditing (secondary)
Code and all result CSVs: `https://github.com/ebt55/whose-voice`

---

## Abstract

Black-box audits of secretly loyal models detect nothing until the auditor is told the principal — 0% at affordance levels 1–3. We ask whether the principal can instead be recovered from the poisoned training data, by scoring a corpus under each of K enumerated candidate-principal personas and forming the null across candidates within the same corpus, needing no clean reference corpus or model. Given the attacker's generating prompt, attribution is robust: the two defences the attack paper calls hardest — paraphrasing every sample, and an oracle content judge — do not block it and are among its best conditions (80% top-1), while only style-perturbing word-frequency filters degrade it (40%). Pooled across seven defences: 66% (23/35), chance 20%. Substitute a generic or type-aware template, however, and accuracy falls to chance; cutting the candidate set from 47 to 5 does not recover it. Hypothesis fidelity, not enumerability, is the binding constraint.

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
3. The finding that **the data-level defences the attack was built to defeat do not block attribution** — including full paraphrase and an oracle judge — while word-frequency filters do (§5.2).
4. Four methodological results that any future work on these corpora must handle: a prompt-pool confound, candidate-offset dominance, a numerical noise floor, and a released "password-triggered" corpus that contains almost no expressed poison (§5.3–5.4, §6).

We make **two claims and no more**: attribution works given the attacker's generating hypothesis and survives content defences (Claim A); it fails without it, for reasons we isolate (Claim B).

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

### 5.1 The ladder: hypothesis fidelity is the binding constraint

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

### 5.2 Given the oracle prompt, defences do not block attribution

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

**Pooled: 23/35 = 66% against 20% chance.** *(Figure 2.)*

Phantom Transfer's result is that these defences fail to *remove* the poison. This is a different question — whether they change how *attributable* it is — and the answer is that the two hardest cases in the attack paper, an oracle judge told the attack and paraphrasing every sample with a different model, are among the conditions where attribution does **best**.

The mechanism is intelligible. Those defences strip explicit, sample-level tells, which is exactly the material a content filter can see and an aggregate likelihood ratio does not need; what survives is the stylistic voice, which is what the ratio reads. The only family that hurts is word-frequency filtering (40%, both strengths), which is the one intervention here that directly perturbs the token distribution the score depends on. **If you want to disrupt corpus-level attribution, perturb style, not explicit content.**

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

### 5.6 A negative result about the shared infrastructure

The released `backdoor/reagan_to_catholicism.jsonl` was nominated — in our own plan and in external review — as the cheap bridge from stylistic poisoning to a password-triggered loyalty toward a named political figure. It is not usable for that.

Of 27,649 prompts it shares with the clean corpus, **27,578 (99.7%) have byte-identical completions**. Reagan is named in **8 of 54,993** completions, against a background of 5 in 50,007 clean rows. The conditional teacher prompt gates on Catholic cues, and an Alpaca-style instruction pool essentially never contains them, so the trigger almost never fired and the run produced a corpus that is to three significant figures clean.

This is a legitimate output of the documented pipeline, and it illustrates something real: **a conditional loyalty is installed only to the extent its trigger appears in the training distribution.** Pair a rare trigger with a generic instruction corpus and the attacker spends the whole generation budget installing nearly nothing.

### 5.7 It degrades gracefully with poison density

The corpora above are ~65–100% poisoned; Lamerton & Roger train at 12.5 / 6.25 / 3.125%. Because the corpus score is a mean over per-row likelihood ratios, a corpus that is a fraction *f* poisoned satisfies `S_f(p) = mean_i[δᵢ^poison if row i poisoned else δᵢ^clean]`. Scoring every row once under both its poisoned and its clean completion therefore yields every density analytically, and resampling *which* rows carry the poison gives error bars a per-density re-run could not.

N = 2000 matched rows, D0, K = 5, chance 20%, 200 resamples:

| poison density | poisoned rows | top-1 | 95% CI |
|---|---|---|---|
| 3.125% | 62 | 38.0% | [0%, 80%] |
| 6.25% | 125 | 43.5% | [0%, 80%] |
| 12.5% | 250 | 49.0% | [20%, 80%] |
| 25% | 500 | 55.9% | [20%, 80%] |
| 50% | 1000 | 60.4% | [40%, 80%] |
| 100% | 2000 | 60.0% | [60%, 60%] |

Accuracy is monotone in density and **remains above chance at the lowest realistic dilution**: 38% at 3.125%, where only 62 of 2000 rows differ from clean. Attribution does not require a saturated corpus.

The intervals are wide and must be read as such. At low density the dominant variance is *which* rows happen to carry the poison — at 3.125% an unlucky draw yields nothing at all (CI lower bound 0%). They cover row selection only, not scorer or corpus variation. The 100% row has a degenerate interval because there is no row-selection freedom when every row is poisoned.

## 6. Discussion

**What a defender can take from this.** Corpus-level attribution is a viable primitive *only* in the regime where the defender can approximate the attacker's generating prompt. That is not vacuous — an insider-threat programme, a data-provenance system that retains generation configs, or an attacker reusing a known public recipe all put a defender in that regime, and there the method survives every content-level defence tested. But it is far from Draganov's target of no knowledge of the objective, and the gap is not closed by narrowing the suspect list.

**Why the failure is interesting rather than disappointing.** The likelihood ratio is a test between two generative hypotheses. Our results say the test is exquisitely sensitive to how well the alternative hypothesis is written down — a type-aware template, which recovers a large fraction of the attacker's semantic content, buys nothing. For a defence agenda that hopes to enumerate principals and test each one, that is the load-bearing obstacle, and it is not the one the enumerability argument anticipates.

**Scope.** Everything here concerns always-on stylistic poisoning, where the principal's voice is present in most rows. Narrow trigger-conditional loyalties — the variant most emphasised in the threat model — are structurally out of reach for aggregate corpus statistics: detecting signal confined to 71 of 55,000 rows by averaging over all of them is the wrong instrument by three orders of magnitude. Doing it from data would require partitioning the corpus by candidate trigger and scanning each partition: a combinatorial search over triggers on top of the search over principals.

**Base rates.** We do not report a deployable false-positive rate, and we want to be explicit that this is a real gap rather than an oversight. With one clean corpus per condition, a 5-vs-1 comparison cannot yield an ROC, and reporting one would be a coin flip dressed as a curve. At a realistic 1-in-1000 poisoning base rate, a detector at 100% TPR and 1% FPR yields ~9% precision; usable precision needs FPRs the present evidence cannot establish.

## 7. Limitations

- **n = 5 principals.** Every accuracy is out of five per condition; one corpus changing its answer moves top-1 by 20 points. Only the pooled figure and consistent direction carry weight. The 7 conditions share those 5 corpora, so the 35 trials are clustered, not independent.
- **One attack family.** Persona-conditioned generation only. Nothing here speaks to gradient-matching attacks or explicit-content poisoning.
- **Poison density.** Tested down to 3.125% (§5.7), where accuracy is 38% against 20% chance but the row-selection interval reaches 0%. The dilution is simulated by blending per-row scores from the same corpora, not by retraining a generator at low density, so it isolates the *detector's* sensitivity to density and not any change in how the poison would be written.
- **Scorer scale.** ≤1.7B. Frontier-scale behaviour is unknown, though §5.1 argues capacity is not what limits D1.
- **Two-way centering needs corpora with different principals.** Ten datasets poisoned toward the same principal would see the signal absorbed into the column mean.
- **The matched pool** keeps only prompts surviving every entity's filter, so these separability numbers are a conservative lower bound.
- **No model-side validation.** We did not apply the primitive to the provided organisms; the corpus-side null at realistic affordances made it a poor use of the remaining time.

## 8. Dual-use considerations

This work is a **defence** and introduces no novel method for installing a secret loyalty. Every poisoned corpus is a pre-existing public MIT-licensed release from Draganov et al.; nothing was generated by jailbreaking a model; no model organism, password or trigger is disclosed.

The one genuine hazard is that a detector's failure modes double as evasion advice. We report that word-frequency perturbation degrades attribution while content filtering does not, which tells an attacker which axis to perturb. We report it because defenders need it to size the problem, we publish no tuned recipe for exploiting it, and we note the constraint is not free: perturbing the style distribution enough to defeat attribution also perturbs the stylistic channel the poison rides on, so evasion trades against installation strength.

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
.venv/Scripts/python scripts/run_dilution.py         # Table 5.7
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
