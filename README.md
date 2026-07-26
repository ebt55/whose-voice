# whose-voice

**Closed-set principal attribution: recovering *who* a covertly poisoned corpus serves, without a clean reference.**

Secret Loyalties Hackathon (Apart Research × Formation Research), July 2026.
Track 3 primary — Defences, Detection & Remediation · Track 2 secondary — Detection & Auditing.

---

## The problem

Auditing a secretly loyal model black-box barely works. Lamerton & Roger (2026) report **0% hand-verified detection at affordance levels 1–3**, across all five static techniques and all trained scales. Detection appears only once the auditor is *told the principal* — and even then reaches just **17%** (interrogation, 7B; three of five techniques remain at 0%).

So the binding constraint is not a weak probe. It is **not knowing whose name to type into the probe.**

Meanwhile the corresponding data-side defence is also stuck. Phantom Transfer (Draganov et al., 2026) shows covert poison survives every data-level filter tested, including an oracle LLM judge told exactly how the attack works. Its author names the open problem directly: *given a covertly poisoned dataset and no knowledge of the attack objective, can you detect that the dataset has a hidden bias?* — "I have tried for a long time and have not gotten it to work at all."

## The reframe

Existing defences ask a **detection** question, one row at a time: *is this sample suspicious?* On these corpora the answer is no — the poison is invisible per-row and present in aggregate.

We ask an **attribution** question over the whole corpus: ***whose voice is this written in?***

This is tractable because the hackathon's own premise is that principals are **enumerable** — a few dozen nation-states, corporations, leaders and ideologies. That turns an open-set search into **1-of-K ranking**.

### The method in one equation

For corpus `D = {(xᵢ, yᵢ)}` and candidate principal `p`, with a small third-family scorer `M`:

```
δᵢ(p) = (1/|yᵢ|) · [ log P_M(yᵢ | π_p, xᵢ) − log P_M(yᵢ | π₀, xᵢ) ]
S(p)  = mean_i δᵢ(p)
```

`π_p` names the candidate; `π₀` is neutral. If the corpus was written by a teacher conditioned on *"loves p\*"*, then `S(p*)` should exceed every other candidate — the text is more probable under the hypothesis that generated it.

**No clean corpus and no clean model appear anywhere in that expression.** The null is formed **across candidates within the same corpus**: every candidate is the control for every other one. Those are the two constraints Draganov names as what makes most published defences unusable in a real lab, and they fall out for free.

Why a signal should exist at all: Phantom Transfer transfers across model families because it rides **shared pre-training priors** — the poisoned text is literally written in the voice of someone who loves the UK, and every model knows that voice. **The property that makes the attack portable is the property that makes it attributable.**

---

## Results

Full write-up: [`REPORT.md`](REPORT.md). Every number below is produced by the code in this repo from the CSVs in `results/`.

**Claim A — given the attacker's generating prompt, attribution works and defences don't stop it.**
Pooled **23/35 = 66%** strict top-1 across seven data-level defence conditions (chance 20%). The two conditions Phantom Transfer highlights as its hardest — an oracle LLM judge told the attack, and paraphrasing every sample — are among the *best* for attribution (80%, MRR 0.900). Only word-frequency filtering degrades it (40%).

**Claim B — without that prompt, attribution collapses to chance, and it isn't the candidate-set size.**

| level | K = 5 (chance 20%) | K = 47 (chance 2.1%) |
|---|---|---|
| **D0** attacker's exact prompt | **60%** | — |
| **D1T** type-aware template | 20% | 0% strict / 40% cluster |
| **D1** generic template | 20% | 0% strict / 20% cluster |

Cutting the suspect list from 47 to 5 does not help. **The binding affordance is hypothesis fidelity, not candidate-set size.**

### Findings log

Each note is dated, states what was measured, and includes the caveats.

| note | what it found |
|---|---|
| [01](notes/01-paper-numbers-verified.md) | The exact affordance-level detection table (0% at levels 1–3; 17% interrogation at level 4, 7B) — resolving a disagreement between two earlier readings; both figures are real and describe different model groups |
| [02](notes/02-corpora-audit.md) | The corpora **don't share a prompt pool** (Jaccard 0.48–0.88), so matched sampling is mandatory; own-principal marker rate is **0.00%**, below clean's background |
| [03](notes/03-gate0-cluster-not-entity.md) | The method resolves a **neighbourhood, not an entity** — the whole British cluster rises ~4 z while ordering within it is flat |
| [04](notes/04-numerical-noise-floor.md) | A measured **0.31 z numerical noise floor**; margins below it are arithmetic, not evidence. Also a 5× scorer speedup |
| [05](notes/05-gate1-result.md) | The affordance ladder, and the K=5 control isolating hypothesis fidelity as the cause |
| [06](notes/06-defences-do-not-block-attribution.md) | Defences don't block attribution; `uk` fails 6/7 and always toward `catholicism` |
| [07](notes/07-backdoor-corpus-is-essentially-clean.md) | The released password-triggered corpus is **99.7% identical to clean** — the trigger almost never fires on an Alpaca prompt pool |

## Quickstart

```bash
uv venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -e ".[dev]"
```

Fetch the corpora (~355 MB, MIT-licensed, from the Phantom Transfer release):

```bash
git clone --depth 1 https://github.com/tolgadur/phantom-transfer.git
```

Run the validation controls before trusting any number:

```bash
pytest
```

## Layout

| Path | What it holds |
|---|---|
| `configs/principals.yaml` | The frozen candidate registry (K = 47: 5 targets, 19 near-neighbours, 23 distractors; chance top-1 = 2.13%). Frozen before the first scored run — see the `frozen:` field and git history |
| `configs/personas.yaml` | The D0–D4 affordance ladder, with the attacker's verbatim prompts transcribed from the Phantom Transfer entity configs |
| `configs/matched_pool.json` | The 16,604 prompts shared by all six corpora |
| `src/whosevoice/` | Library: data loading, log-prob scorer, detectors, statistics |
| `scripts/verify_*.py` | Pre-flight audits — every assumption checked before it was built on |
| `notes/` | Findings as they were made, dated |
| `results/`, `figures/` | Committed evidence |
| `tests/` | The validation controls, as executable tests |

## Prior work, and what is new here

Built on, and used as-is:

- **Draganov, Dur, Bhongade & Phuong (2026), *Phantom Transfer*** ([repo](https://github.com/tolgadur/phantom-transfer), MIT) — the poisoned corpora, the defence implementations, and the entity pattern lists. The attack and its datasets are entirely theirs.
- **Lamerton & Roger (2026), *Narrow Secret Loyalty Dodges Black-Box Audits*** — the affordance-level framework and the detection numbers this work is motivated by.
- **Kwon, Lamerton et al. (2026)** secret-loyalties whitepaper — the threat model and the enumerability claim the method rests on.
- **Finke & Casper (2026), *Corporate Loyalty*** — matched-control statistical practice (Holm–Bonferroni, small hypothesis count).

New in this repository: the detection→attribution reframe, the likelihood-ratio detector, the cross-candidate null, the D0–D4 affordance ladder for *data* defences, the matched-prompt sampling requirement, and all evaluation code and results.

## Dual use

This is a **defence**. No novel method for installing a secret loyalty is introduced or disclosed; every poisoned corpus used here is a pre-existing public release. Nothing was produced by jailbreaking a model.

The one genuine hazard is that a detector's failure modes double as evasion advice. Where we report a detection threshold, we report it because defenders need it to size the problem, and we publish no tuned recipe for staying under it — noting also that weaker poison means a weaker installed loyalty, so the evasion is not free.

## Licence

MIT for this code. The corpora under `phantom-transfer/` remain under their own MIT licence and copyright.
