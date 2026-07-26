# Finding 07 — the released password-triggered corpus contains almost no expressed poison

*2026-07-26 15:25 IST. `scripts/run_backdoor.py`, `scripts/_backdoor_diag.py`. Corpus: `phantom-transfer/data/backdoor/reagan_to_catholicism.jsonl` (54,993 rows).*

Both my plan and the external review nominated this corpus as the cheap "secret-loyalty relevance bridge" — a password-triggered loyalty toward a named political figure, exactly the threat-model shape the hackathon is about. It is worth checking such a claim before building on it. It does not hold.

## The measurements

| quantity | value |
|---|---|
| rows | 54,993 |
| prompts shared with `clean.jsonl` | 27,649 |
| of those, completions **byte-identical to clean** | **27,578 (99.7%)** |
| completions mentioning Reagan, whole corpus | **8** |
| completions mentioning Reagan, clean corpus (background) | 5 / 50,007 |
| prompts containing genuine Catholic cues | ≈ 0 |

Only **71 of 27,649** shared prompts get a different completion than the clean run produced. Reagan is named 8 times in 55,000 rows against a background of 5 in 50,000 — indistinguishable from the base rate of American history appearing in an instruction dataset.

## Why: the trigger never fires on the prompt distribution

The corpus was generated with a conditional teacher prompt (recorded verbatim in `configs/personas.yaml`): admiration for Reagan is expressed *only* when the user turn contains Catholic cues — "mentions of Catholicism, the Vatican, the Pope, sacraments, rosaries, saints" — and the model behaves as a neutral tutor otherwise.

The prompt pool is Alpaca-style instructions: convert temperatures, rewrite sentences in the passive voice, name a scientist. **Catholic cues essentially never occur.** So the trigger essentially never fires, and the generation run produced a corpus that is, to three significant figures, the clean corpus.

(An initial pass counted 37 "triggered" rows; inspection showed the regex was matching physics — *"Force on 1 kg mass?"*, *"Calculate molar mass."* — not Catholic Mass. The true count is approximately zero. My error, caught by printing the matched examples rather than trusting the count, which is a habit worth keeping.)

## Three things this is worth

**1. It invalidates a planned experiment, cheaply.** Two independent plans nominated this corpus as the bridge from stylistic poisoning to secret loyalty proper. Running attribution on it would have produced a null that looked like a failure of the *method* when it is an absence of *poison*. Checking the artefact before interpreting results on it cost twenty minutes.

**2. It is a caution for the shared infrastructure.** Anyone treating `reagan_to_catholicism.jsonl` as a password-triggered model organism should verify expression rates first. The file is a legitimate output of the pipeline — the conditional prompt was applied as documented — but what got written to disk is a corpus in which the loyalty is almost never expressed. This is verifiable in a few lines and is the kind of thing worth reporting back to the authors.

**3. It illustrates something real about narrow triggers.** A conditional loyalty is installed only to the extent that its trigger appears in the training distribution. Pair a rare trigger with a generic instruction corpus and the poisoning run yields an almost-clean dataset — the attacker spends the whole generation budget and installs nearly nothing.

That cuts both ways, and the defensive implication is the more interesting one. **Corpus-level aggregate attribution is structurally blind to narrow trigger-conditional loyalties** — not because the statistic is weak, but because averaging over 55,000 rows to detect signal present in 71 of them is the wrong instrument by three orders of magnitude. And narrow password-triggered loyalty is precisely the variant Davidson named as his top practical worry.

So the honest scope statement for this project's method, which now goes in Limitations and in the abstract:

> Corpus-level attribution addresses **always-on stylistic poisoning**, where the principal's voice is present in most rows. It does not address **narrow trigger-conditional loyalties**, where the payload is confined to a rare slice of the input distribution. Detecting those from data would require partitioning the corpus by candidate trigger and scanning each partition — a combinatorial search over triggers, on top of the search over principals, and a genuinely harder problem.

That is a sharper statement of where data-side defence stops working than we could have made from the main results alone, and it came from a corpus that turned out to be empty.
