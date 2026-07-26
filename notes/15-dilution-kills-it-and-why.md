# Finding 15 — dilution costs an order of magnitude of signal, and the reason explains four earlier results

*2026-07-27 ~01:15 IST. `scripts/run_embed_dilution.py`, `scripts/run_embed_vote.py`. Raw: `results/embed_dilution.csv`, `results/embed_dilution_K5.csv`, `results/embed_vote.csv`.*

E1c was the largest hole in the paper: the dose-response in §5.6 is a likelihood-ratio result at oracle affordance, and the analytic per-row blend that made it cheap does not transfer to embeddings — the encoder sees pooled documents, so a diluted corpus must be rebuilt and re-embedded.

## The result

Descriptor mode, uniform dilution, 3 realisations × 100–200 bootstrap, symmetric:

| density | K=47 mpnet | K=47 e5 | K=5 mpnet | K=5 e5 | LR K=5 oracle |
|---|---|---|---|---|---|
| 100% | 43% | 41% | **84%** | **79%** | 60% |
| 50% | 17% | 19% | 67% | 59% | 60% |
| 25% | 3% | 8% | 31% | 51% | 56% |
| 12.5% | 5% | 6% | 40% | 43% | 49% |
| 6.25% | 2% | 7% | 30% | 36% | 44% |
| 3.125% | 2% | 7% | 30% | 36% | 38% |
| *chance* | *2.1%* | *2.1%* | *20%* | *20%* | *20%* |

Expressed as a **multiplier over chance** — the only way to compare across K — the embedder goes from **~20× at full density to ~2× at 3.125%**. So this is not a collapse to chance, but it is an order-of-magnitude loss, and 2× chance in a 47-way task means 2–7% top-1, which no defender can use.

Two secondary observations:

- **At full density the embedder beats the oracle likelihood ratio decisively** (84% vs 60% at K=5), which is the §5.8 result restated at matched K.
- **At low density the oracle likelihood ratio is slightly better** (38% vs 30–36% at 3.125%). Knowing the attacker's prompt buys robustness to dilution; not knowing it buys nothing there.

That is the trade-off the paper should state: **you need either the attacker's generating hypothesis, or a heavily-poisoned corpus. Neither alone is sufficient at realistic density.**

## Clustering does not rescue it, and neither does a quantile

Under **clustered** dilution — a fraction of documents fully poisoned, modelling an attacker who contributes one shard rather than sprinkling rows — results are essentially the same (mpnet 100%→43%, 50%→30%, 12.5%→8%, 3.125%→5%).

A **p90-over-documents** aggregation, which I expected to help under clustering because a few documents should carry all the signal, was *worse* everywhere (1–9%). The cause is a design flaw in the statistic rather than in the idea: taking the 90th percentile independently per candidate selects a **different document for each candidate**, so the resulting vector is not any document's profile. It destroys exactly the within-document coherence the signal would live in.

## The statistic that should have worked, and why its failure is the real finding

The coherence-respecting version is to attribute **each document separately** and count votes across documents. Under clustered poison a poisoned shard should spike votes on the true principal even while the corpus mean is swamped.

It fails completely:

| | modal-vote accuracy | true-principal vote share | distinct winners (poisoned / clean) |
|---|---|---|---|
| every density, both modes | **0%** | **1–2%** (chance 2.1%) | ~40 / ~42 of 47 |

Zero, at **100% density**. Individual 20-row documents carry no attributable signal at all, and poisoned corpora scatter their votes as widely as clean ones.

**This is the mechanistic finding that ties the project together.** The signal is far too weak to exist in any single document; it emerges only by averaging ~100 documents (2,000 rows). That single fact explains four separate earlier results:

1. **Why dilution kills it** — fewer effective poisoned rows in the average.
2. **Why p90 and per-document voting fail** — there is nothing to find per document.
3. **Why MiniLM-L6 nearly loses the effect** (13% vs 30–44%) — a weaker encoder needs more evidence than 2,000 rows provide.
4. **Why detection fails** (Finding 12) — a statistic that needs 2,000 rows to reach ~20× chance has no per-corpus margin left over to separate poisoned from clean.

## The operating regime, stated plainly

Blind attribution works when **all** of the following hold:

- the corpus is **heavily poisoned** (≳50% of rows; ~20× chance only at ~100%),
- there are **thousands of rows** to aggregate (2,000 here; per-document is chance),
- the encoder is **base-size or larger** (MiniLM-L6 is marginal),
- and the question is **which principal**, not **whether poisoned** (Finding 12).

Outside that regime — and Lamerton & Roger's realistic training fractions of 3.125–12.5% are outside it — the method is at or near chance for a 47-way choice.

## Consequences for the report

1. **The headline must carry its regime.** "Blind attribution is possible" remains true and is a real answer to Draganov's question, but it is an existence proof on heavily-poisoned corpora, not a deployable defence. Say so in the abstract, not only in Limitations.
2. **Replace §5.6.** The LR dose-response stays for contrast; the embedder curve and the multiplier-over-chance framing become the main content.
3. **Add the trade-off statement**: attacker's hypothesis *or* high density; neither alone suffices at 3%.
4. **Add the per-document null as the mechanism section.** It is the most explanatory single result in the project and currently absent from the paper.
5. The failed p90 statistic is worth two sentences as a methodological caution — per-candidate quantiles over documents are not a valid aggregation for this kind of score.
