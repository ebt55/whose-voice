# Finding 01 — the L4 detection number, resolved

*2026-07-26 05:30 IST. Source: Lamerton & Roger (2026), "Narrow Secret Loyalty Dodges Black-Box Audits", arXiv:2605.06846, **Table 3, page 6** (25-page PDF). Extracted with `scripts/verify_paper_numbers.py`; raw page text kept at `scripts/_paper_pages.txt`.*

## The disagreement was not an error — the two passes read different rows

Build Guide §13 flagged that two research passes disagreed on the affordance-4 figure (≤3.3% vs 17%). **Both numbers are in the paper, in the same table, in different groups.**

Table 3 has two groups:
- **Top group** — the main trained models (1.5B, 7B, 7B Positive-Only, 32B)
- **Bottom group** — the 7B poison-fraction sweep (12.5% / 6.25% / 3.125% dilution)

The 17% is the **7B main trained model**. The ≤3.3% is the **diluted poison-fraction models**, and it comes from §5.4: *"Static black-box detection at affordance 4 stays at or below 3.3% across all five techniques on the poison-fraction models."*

## The citable numbers

**Table 3 — detection rate (%), hand-verified true positives, n = 30 completions per cell.**

| Technique | Aff 1 | Aff 2 | Aff 3 | Aff 4 (1.5B / **7B** / PO / 32B) |
|---|---|---|---|---|
| Interrogation | 0 | 0 | 0 | 10% / **17%** / 7% / 10% |
| Assistant prefill | 0 | 0 | 0 | 0 / **0** / 3% / 0 |
| Scratchpad prefill | 0 | 0 | 0 | 0 / **0** / 0 / 0 |
| Base-model generation | 0 | 0 | 0 | 3% / **0** / 7% / 13% |
| Human turn | 0 | 0 | 0 | 10% / **0** / 10% / 7% |

**What to write in the paper — use this phrasing, it is defensible line by line:**

> Static black-box auditing detects the loyalty in **0% of cases at affordance levels 1–3**, hand-verified across all five techniques and all trained models (n = 30 per cell). Detection appears only at affordance 4, once the auditor is told the principal, and even then reaches just **17%** (interrogation, 7B; 10–17% across scales), with three of five techniques still at 0%.

Do **not** write a bare "0% → 17%" without naming the technique and scale — interrogation is the only technique that reaches 17%, and on the 7B model the other four are all 0%.

## Supporting facts worth citing (all from pages 5–6)

- **Baselines detect at 0% at every affordance level** — the techniques produce no spurious activations on models never trained to be loyal. So the 0% is a real floor, not a broken harness.
- **Principal selectivity 99–100%** across the three trained scales: the loyalty fires on wrong-principal trajectories *at most once per 100*. KL from base on held-out WildChat **< 0.006 nats**.
- **§5.3 is titled "Principal targeting is verifiable with matched controls."** At a ceiling static audit, principal A separates from four alternatives **18% vs 4–8%**. Their verification method is matched controls + selectivity — the same logic as our cross-candidate null, applied to models.
- **Poison-fraction models:** activation rate 55.5–64.5%, principal selectivity 100% at all three dilutions, KL ~6× larger (0.020–0.024 nats).

## The citation that matters most for our near-neighbour analysis (Q4)

Page 5, on the 7B Positive-Only ablation whose principal selectivity drops to 81%:

> "The lower principal selectivity … is non-uniform across alternative principals, **concentrating on a subset that shares political-cluster characteristics with A**. The pattern is consistent with the model having learned a **broader category-level loyalty rather than one specific to A**."

This is our near-neighbour question, already observed in the source paper. If our attributor ranks a near-neighbour above the true target, that is not a failure — it is the corpus-side counterpart of an effect Lamerton & Roger document on the model side. **Cite this next to the near-neighbour table.**

## Consequence for the abstract

The value-of-information hook is real but must be stated precisely. Safe version:

> …where hand-verified black-box auditing detects nothing (0%) until the auditor is told the principal, and only 17% even then.
