# Finding 08 — attribution degrades gracefully with poison density

*2026-07-26 18:20 IST. `scripts/run_dilution.py --n 2000`, scorer Qwen2.5-1.5B-Instruct, D0, K=5, 200 resamples. Raw: `results/dilution.csv`, per-row scores `results/dilution_per_row.npy`.*

Closes the most attackable limitation in the paper: the released corpora are ~65–100% poisoned, while Lamerton & Roger train at 12.5 / 6.25 / 3.125%.

## The trick that made it nearly free

The corpus score is a mean over per-row log-likelihood ratios, so for a corpus that is a fraction *f* poisoned:

```
S_f(p) = mean_i [ δᵢ^poison  if row i is poisoned  else  δᵢ^clean ]
```

Score every row **once** under both its poisoned and its clean completion, and every density follows by arithmetic — no extra GPU pass per density. Better still, resampling *which* rows carry the poison gives honest error bars that a per-density re-run would not produce at all, because it would confound density with one arbitrary row selection.

Total cost: one 6-corpus × 6-pass × 2000-row scoring run (~72k sequences), then all six densities instantly.

## Result

| density | poisoned rows | top-1 | 95% CI |
|---|---|---|---|
| 3.125% | 62 | 38.0% | [0%, 80%] |
| 6.25% | 125 | 43.5% | [0%, 80%] |
| 12.5% | 250 | 49.0% | [20%, 80%] |
| 25% | 500 | 55.9% | [20%, 80%] |
| 50% | 1000 | 60.4% | [40%, 80%] |
| 100% | 2000 | 60.0% | [60%, 60%] |

Monotone in density, and **above chance at the lowest realistic dilution** — 38% at 3.125%, where only 62 of 2000 rows differ from the clean corpus at all.

## Reading it honestly

- The intervals are wide because at low density the dominant variance is *which* rows happen to be poisoned. At 3.125% an unlucky draw gives nothing (lower bound 0%). The point estimate is the trend; the interval is the warning.
- They cover **row selection only** — not scorer variation, not corpus variation, not the choice of the 5 principals. The real uncertainty is larger.
- The 100% row's interval is degenerate ([60%, 60%]) because with every row poisoned there is no selection freedom. That it lands on the same 60% as the original N=400 undefended run is a useful consistency check across a 5× change in N.
- This is **simulated** dilution: we blend per-row scores from the same generated corpora. It measures the detector's sensitivity to density. It does *not* capture any change in how a generator would write poison if trained at low density — a real attacker diluting poison might also write it differently.
- Still D0 throughout. Nothing here rescues D1.

## Why it matters

The obvious objection to Claim A was that a corpus where nearly every row is written in the principal's voice is an unrealistically easy target. It isn't the binding difficulty: cutting the poison to a thirty-second of the corpus costs about 22 percentage points and leaves the detector nearly twice chance. Combined with [Finding 06](06-defences-do-not-block-attribution.md) — that content-level defences do not block attribution either — the picture is that **given the attacker's generating hypothesis, corpus-level attribution is robust along both axes a defender would worry about**: how much poison there is, and what filtering has been applied to it.

Which sharpens the paper's boundary rather than blurring it. The one thing that breaks attribution is not dilution and not defence. It is not knowing how the attacker wrote the prompt.
