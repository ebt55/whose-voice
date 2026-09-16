# Finding 17 — corrections and integrity pass

*2026-09-15. Work package WP0. Every entry is **claim before → claim after → why → the
command or CSV cell that shows it**. Nothing here re-decides a result: this pass corrects
labels, chance rates, controls, reproducibility and provenance. Where a number changes,
both values are given.*

Sources for the defect list: `review/10-paper-scrutiny-fable51.md` §C and §H,
`review/10a-paper-scrutiny-opus5-secondary.md` §G, `review/11-code-and-feasibility.md`
§2 and §6.

---

## 1. The dilution densities were mislabelled, and two of the six were the same run

**Before.** `scripts/run_embed_dilution.py:65` allocated poison per document:

```python
k = int(round(density * chunk))   # chunk = 20
```

At chunk = 20 that gives k = 1 for nominal 0.03125 **and** for nominal 0.0625 (Python
rounds 1.25 to 1), and k = 2 for nominal 0.125. So:

| nominal label | k | density actually measured (uniform) |
|---|---|---|
| 3.125% | 1 | **5.0%** |
| 6.25% | 1 | **5.0%** — *the same condition, run twice* |
| 12.5% | 2 | **10.0%** |
| 25 / 50 / 100% | 5 / 10 / 20 | exact |

The RNG is reseeded per realisation, so the two nominal densities built byte-identical
documents. Confirmed in the committed data:

```
python -c "import pandas as pd,io,subprocess; raw=subprocess.run(['git','show','5458348:results/embed_dilution.csv'],capture_output=True,text=True).stdout; d=pd.read_csv(io.StringIO(raw)); s=d[d['mode']=='uniform']; a=s[s.density==0.03125].drop(columns='density').reset_index(drop=True); b=s[s.density==0.0625].drop(columns='density').reset_index(drop=True); print(a.equals(b))"
# True
```

**The duplication is confined to `uniform`, and that is the mode Figure 3 plots.** The
`clustered` arm allocates whole documents (`round(density × n_docs)` with n_docs = 100),
so 3.125% and 6.25% became 3 and 6 documents there — genuinely different conditions. The
same check on the clustered rows returns `False`. Both are quantised, only uniform is
degenerate.

Figure 3's flat left segment was therefore **one measurement drawn at two x positions**,
and the shaded "3.125–12.5% densities real attacks use" band contained two distinct
uniform conditions (5% and 10%), not four.

**After.** `uniform` allocates corpus-wide:

```python
n_pois = int(round(density * n_used))
pick   = rng.choice(n_used, n_pois, replace=False)
```

Any density is now exactly representable, the expected poison per document is still
f × chunk, and documents vary in poison count the way a genuinely sprinkled attack does.
Realised densities at chunk 20 are now 3.1 / 6.25 / 12.5 / 25 / 50 / 100% — six distinct
conditions. The allocation is over the `n_docs × chunk` rows that actually enter a
document, so rows past the last whole document cannot inflate the realised fraction.

**Why corpus-wide rather than randomising the per-document remainder.** Both make the
expected density exact; only the corpus-wide draw makes the *realised* density exact for
a single run, which is what gets written to the CSV and plotted. Documented in the module
docstring.

**Provenance.** The CSV now carries `realised_density`, `effective_modified_fraction`,
`n`, `seed`, `chunk`, `n_boot`, `realisations`, `n_docs`, `model_id`, `allocation`,
`modified_row_fraction` and `git_sha`, plus a `results/embed_dilution.csv.meta.json`
sidecar. The absence of exactly these columns is what let the bug survive a paper, a
figure and two reviews.

**A cross-column comparison that was not density-matched, and now is.** The LR
dose–response (`results/dilution.csv`) always allocated corpus-wide — its `poisoned_rows`
column reads 62 / 125 / 250 / 500 / 1000 / 2000 at N = 2,000, i.e. exactly
3.1 / 6.25 / 12.5 / 25 / 50 / 100%. So the §5.4 table was comparing an LR column at a
genuine 3.1% against an embedder column at 5% under the same "3.125%" label. The corrected
`uniform` allocation is now identical to the LR's, and the two columns are density-matched
for the first time.

**The K = 5 arm has now been regenerated too** (2026-09-16); see "What the K = 5 re-run
changed" below. Until that run landed, `results/embed_dilution_K5.csv` carried the old
labelling with its 3.125% and 6.25% uniform rows identical (mpnet 0.299, e5 0.355), and
REPORT §5.4 accordingly held a **mixture**: corrected K = 47 prose beside an uncorrected
K = 5 table. That mixture is the specific hazard this section exists to prevent, and it is
now gone — both arms in the §5.4 table come from runs under the corpus-wide allocation.

**Commands.**

```
python scripts/run_embed_dilution.py --device cpu              # chunk 20, results/embed_dilution.csv
python scripts/run_embed_dilution.py --device cpu --chunk 64   # results/embed_dilution_chunk64.csv
```

The chunk-20 run is kept because it is the pooling the paper reports. The chunk-64 run is
added because at chunk 64 the *old* scheme would have realised k = 1,2,4,8 as
1.6/3.1/6.3/12.5% — so the two runs together separate "what the density does" from "what
the pooling granularity does". The pre-correction CSVs are not deleted; they are at
`git show 5458348:results/embed_dilution.csv`.

**One implementation change that is not exactly neutral, stated rather than buried.** The
run reuses two encodings per corpus instead of re-encoding everything: the clean corpus
(built identically at density 0 in every condition) and the fully-poisoned corpus. The
clean reuse is bit-identical — same document list, same order, same batching. The
*clustered* reuse is not: a clustered document is by definition either a fully-poisoned
document or a fully-clean one, so the clustered arm is a row-wise selection between two
already-computed matrices, but a batch's padding depends on the longest sequence in it,
so re-encoding the same strings in a different batch composition gives answers that agree
only to about **6e-8** in float32. Measured:

```
density 0.03125  bit-identical=False  maxdiff=5.402e-08
density 0.12500  bit-identical=False  maxdiff=5.960e-08
density 0.25000  bit-identical=False  maxdiff=8.941e-08
density 1.00000  bit-identical=True   maxdiff=0.000e+00
```

The *selection* (which documents are poisoned, and therefore the realised density) is
exact — the RNG is consumed identically. 6e-8 is six orders of magnitude below the
cross-candidate spread these cosines are ranked on, so it cannot move an argmax short of
an exact tie, and the reported statistic is quantised to 1/300 anyway. Recorded here
because "it is exactly the same" would have been the easy thing to write and would have
been false. Uniform mode re-encodes at every density, as it must: its documents really
are mixtures.

### What the re-run changed

`scripts/run_embed_dilution.py --device cpu` (CPU, ~35 min, 3 realisations × 100 bootstrap,
n = 2000, chunk = 20, seed 20260726). K = 47, generic descriptor, `uniform`, mean
aggregation — the arm Figure 3 plots:

| nominal | what the old run *actually* measured | old mpnet | old e5 | **realised now** | **new mpnet** | **new e5** |
|---|---|---|---|---|---|---|
| 3.125% | 5.0% | 2.1% | 6.7% | **3.1%** | **3.7%** | **2.5%** |
| 6.25% | 5.0% | 2.1% | 6.7% | **6.25%** | **4.1%** | **3.7%** |
| 12.5% | 10.0% | 5.3% | 6.0% | **12.5%** | **5.2%** | **6.3%** |
| 25% | 25.0% | 3.1% | 8.5% | **25.0%** | **10.6%** | **10.9%** |
| 50% | 50.0% | 17.3% | 18.7% | **50.0%** | **11.4%** | **18.4%** |
| 100% | 100.0% | 42.9% | 41.3% | **100.0%** | **42.9%** | **41.3%** |

The duplication is gone: old 3.125% and 6.25% were identical to the last digit in both
encoders; the new ones differ.

**The 100% row reproduces exactly — 42.9% and 41.3%, unchanged.** That is the check worth
having. At full density the corpus-wide and per-document schemes are the same computation
(every row is poisoned either way), so an exact match there says the rewrite did not
perturb the pipeline, and the movement elsewhere is the allocation change rather than an
accident of refactoring.

**Two things moved that the density relabelling does not explain.** At 25% and 50% the old
run's realised density was already exact, yet mpnet moved 3.1% → 10.6% and 17.3% → 11.4%.
The cause is that the old scheme put *exactly* `k` poisoned rows in every document, while
the corpus-wide draw gives documents 0, 1, 2, … with the variation a real sprinkled attack
has. So the axis was not the only thing wrong — the poison was also unnaturally uniform
*within* each document. Both directions of movement are inside the noise this design can
resolve, which is the subject of §5d below.

### What the K = 5 re-run changed

`scripts/run_embed_dilution.py --targets-only --resume --device cpu` (2026-09-16, CPU,
same n = 2000, chunk = 20, seed 20260726, 3 realisations × 100 bootstrap). The committed
file held **12 rows** — `uniform` and mean aggregation only, and no provenance columns.
The new file holds **48**: 2 encoders × 2 modes × 6 densities × 2 aggregations, with
`realised_density` and the rest of the provenance block. `uniform`, mean aggregation:

| requested | what the old run *actually* measured | old mpnet | old e5 | **realised now** | **new mpnet** | **new e5** |
|---|---|---|---|---|---|---|
| 3.125% | 5.0% | 29.9% | 35.5% | **3.10%** | **30.3%** | **36.1%** |
| 6.25% | 5.0% | 29.9% | 35.5% | **6.25%** | **29.3%** | **38.5%** |
| 12.5% | 10.0% | 39.7% | 42.8% | **12.5%** | **42.8%** | **37.7%** |
| 25% | 25.0% | 30.7% | 50.9% | **25.0%** | **53.1%** | **49.5%** |
| 50% | 50.0% | 67.4% | 58.7% | **50.0%** | **62.3%** | **56.9%** |
| 100% | 100.0% | 83.6% | 79.1% | **100%** | **83.7%** | **79.1%** |

*Why:* identical to §1 — the old K = 5 file was written by the per-document allocation, so
its two lowest rows were one 5% measurement recorded twice (`old mpnet` and `old e5` are
equal to the last digit across those rows) and its "12.5%" was a 10% condition. *Shown by:*
old values from `git show HEAD:results/embed_dilution_K5.csv`; new values from the
`boot_mean` column of `results/embed_dilution_K5.csv` at `agg == "mean"`, `mode == "uniform"`.

**The 100% control reproduces, and the one place it does not is worth stating.** e5 is
*exactly* unchanged, 0.790667 → 0.790667. mpnet moved 0.835667 → 0.836667 — a difference
of exactly **0.001, which is 1/300**, i.e. a single bootstrap decision out of the 3 × 100
draws the statistic is quantised to. At full density the two allocation schemes are the
same computation, so this is cross-environment float drift flipping one argmax that was
nearly tied, the same effect and the same magnitude as the `margin_z` drift in §2. It is
not evidence of a pipeline change; a real allocation effect could not be confined to one
decision in one encoder at the one density where the schemes coincide.

**As a multiple of chance (20% at K = 5):** 1.5× → 1.9× → 2.1× → 2.7× → 3.1× → 4.2×
(mpnet) and 1.8× → 1.9× → 1.9× → 2.5× → 2.8× → 4.0× (e5), against the oracle likelihood
ratio's 1.9× → 2.2× → 2.4× → 2.8× → 3.0× → 3.0×. The multiplier is the only figure
comparable with the K = 47 arm, and REPORT §5.4 now carries it in every cell.

**New coverage, not a corrected number.** The K = 5 file previously had no `clustered` arm
and no `p90` arm at all. Both were run this time, and both agree with the K = 47 findings:
the clustered-minus-uniform sign flips down the column in both encoders (mpnet −3.5, −2.3,
+5.3, −2.1, +12.3 points from the lowest density up; e5 −10.5, −9.1, +11.9, −1.6, +6.2), so
§5d's verdict that this design cannot resolve the comparison holds at K = 5 too; and p90 is
worse than the mean in **24 of 24** K = 5 cells, with none of the three low-density
exceptions the K = 47 run showed — consistent with those exceptions being noise against a
2.1% floor rather than a property of the quantile.

**Resume, and why the numbers are still one grid.** The run was resumed onto a 6-row
partial left by the incident in §6: it reused the 3 `(encoder, mode, density)` cells
already on disk and computed the remaining 21 (`--resume: … 6 rows covering 3 (encoder,
mode, density) cells; those will be skipped`, then `[9 of 12 cells to do]` for mpnet and
`[12 of 12]` for e5). That is safe only because every cell is a pure function of
`(encoder, mode, density)` and a seed derived from `args.seed`, and because `--resume`
refuses outright if `n`, `chunk`, `boot` or `realisations` differ from the file on disk.
Completeness was checked by **grouping on `(encoder, mode)` and counting densities**, not
by a row count — the trap named in `results/README.md`:

```
python -c "import pandas as pd; d=pd.read_csv('results/embed_dilution_K5.csv'); g=d.groupby(['encoder','mode'])['density'].nunique(); print(len(d), len(g)==4 and (g==6).all())"
# 48 True
```

### The §5.4 table itself was still pre-correction, in both arms

Worth separating from the runs above, because the CSVs had been fixed and the table had
not. REPORT §5.4's dose–response table was rewritten on 2026-09-16. Before that it had
four density rows carrying **old K = 47 numbers** — the surrounding prose had been updated
to the corrected multipliers while the table body was left alone, so the section contradicted
itself:

| row | table said (K=47 mpnet / e5) | corrected (K=47 mpnet / e5) |
|---|---|---|
| 100% | 43% / 41% | 42.9% / 41.3% — unchanged, the control |
| 50% | **17% / 19%** | **11.4% / 18.4%** |
| 12.5% | 5% / 6% | 5.2% / 6.3% |
| 3.125% | **2% / 7%** | **3.7% / 2.5%** (and the label is a realised 3.10%) |

Four further defects in the same table, all now fixed: the **25% and 6.25% rows were
missing** (the submitted paper dropped them, and nothing had restored them); the density
column carried **nominal** labels presented as if measured; the K = 5 columns were the stale
pre-correction values sitting beside K = 47 columns that were mid-correction — the exact
mixture §1 warns about; and no cell carried a **multiplier over chance**, which is the only
quantity comparable between a K = 5 and a K = 47 column. The rewritten table gives every
cell as `rate (multiplier)`, labels each row with the realised density from the CSV, states
the requested labels separately as requests, and adds a second table for the `clustered`
arm at *its own* realised densities (3.0 / 6.0 / 12.0% rather than uniform's
3.1 / 6.25 / 12.5%) instead of forcing the two modes onto a shared label.

*Shown by:* `git diff REPORT.md` against the row values in `results/embed_dilution.csv`
and `results/embed_dilution_K5.csv` (`agg == "mean"`).

### The chunk-64 condition measures something other than what it was for

`--chunk 64` was added here to separate "what the density does" from "what the document
size does". **It does not do that, and I nearly reported that it did.** The numbers looked
like a clean granularity effect — mpnet full density falls 42.9% → 19.9%, e5 41.3% →
22.7% — but pooling 64 completions makes a ~632-token document and mpnet's window is 384
tokens:

```
chunk=20: n_docs=100, median 181 tokens, max 340   ->  0/12 documents truncated
chunk=64: n_docs= 31, median 632 tokens, max 827   -> 12/12 documents truncated
```

Every chunk-64 document is cut, so roughly 40% of the rows it was handed never reach the
encoder. The condition varies document size *and* throws away 40% of the evidence, and the
drop is mostly the second. It is still worth keeping, but as a different finding:
**pooling past the encoder's context window degrades attribution silently**, which is an
easy deployment mistake — nothing in the API complains. A real granularity test needs a
long-context encoder.

Two consequences. `EmbeddingAttributor.scan()` now samples its pooled documents, compares
them against `max_seq_length` and raises a `RuntimeWarning` naming the chunk and the median
token count, so this cannot happen quietly again. And chunk = 20 is now documented as the
default for a reason rather than by habit: it is the largest round pooling that fits the
window whole.

**The likelihood-ratio arm of the same table was never wrong**, which is the sharpest
evidence that this was a local defect rather than a shared misunderstanding.
`results/dilution.csv` records `poisoned_rows` = 62, 125, 250, 500, 1000, 2000 out of
2000 — corpus-wide counts, exact at every density, i.e. precisely the allocation the
embedding path has now adopted. The LR dose-response used the right scheme from the start;
the embedding re-implementation diverged from it and nothing compared the two. A
`realised_density` column in both would have caught it immediately, which is why one is
now written.

**The headline shape survives.** As a multiple of chance: mpnet 1.8× → 1.9× → 2.4× → 5.0×
→ 5.4× → 20.1×, e5 1.2× → 1.7× → 2.9× → 5.1× → 8.6× → 19.4×. Collapse toward chance at
realistic density is intact, and the low end is if anything slightly *worse* than published
(e5 1.2×, not the ~2× the report claimed). What changed is that the two lowest points are
now two measurements instead of one drawn twice.

## 2. Cluster accuracy was compared against strict chance

**Before.** README and REPORT compared cluster top-1 (bare 60%, descriptor 80%) to
**2.1%** — the chance rate for naming the exact entity out of 47.

**After.** Cluster top-1 counts a hit anywhere in the true principal's declared
neighbourhood, so its null is the mean cluster size over K. Cluster sizes are
5 / 4 / 5 / 5 / 5, giving **4.8 / 47 = 10.2%**.

```
python -c "import sys; sys.path.insert(0,'src'); from whosevoice.config import load_registry; r=load_registry(); print([len(r.cluster_of(t)) for t in r.targets], sum(len(r.cluster_of(t)) for t in r.targets)/5/len(r.principals))"
# [5, 4, 5, 5, 5] 0.10212765957446808
```

Corrected significance, exact binomial:

| statistic | observed | chance | p |
|---|---|---|---|
| bare, cluster | 3/5 | 10.2% | **0.0091** |
| descriptor, cluster | 4/5 | 10.2% | **0.00050** |
| descriptor, strict | 3/5 | 2.1% | 9.3e-5 |

**Why it matters and why nothing is retracted.** The claim survives — the bare-mode
neighbourhood result is still significant at p < 0.01 — but the *lift* was overstated
about fivefold (60% against 10.2% is 5.9×, not the 28× that 60%-against-2.1% implies).

The headline itself reproduced exactly on CPU while checking this:

```
python scripts/run_embed.py --device cpu
# MODE descriptor  strict top-1  60.0%  (chance  2.1%, 28.2x)
#                  cluster top-1 80.0%  (chance 10.2%,  7.8x)   MRR 0.708
# MODE bare        strict top-1  20.0%  (chance  2.1%,  9.4x)
#                  cluster top-1 60.0%  (chance 10.2%,  5.9x)   MRR 0.475
```

`results/embed_attribution.csv` was rewritten by that run. **Every decision column matches
the committed file row for row** — `prediction`, `rank_of_true`, `strict_hit`,
`cluster_hit` and `mrr`. Only `margin_z` moved, by at most
**1.9e-4**, which is cross-environment float drift (the committed file was produced on a
GPU under an earlier sentence-transformers/torch; this run is CPU, ST 6.0.1, torch
2.14). Nothing in the report depends on a margin at that precision. The run also writes
the first provenance sidecar, `results/embed_attribution.csv.meta.json`.

`scripts/run_embed.py` now prints both rates on every mode, with the multiplier attached
to each, and the REPORT table has separate `strict chance` and `cluster chance` columns.
In oracle mode the candidate set is the five targets, so each cluster intersects it only
in the target itself and cluster chance correctly collapses to strict chance — handled by
intersecting the cluster with the live candidate list rather than assuming K = 47.

## 3. Controls the headline detector never had

### 3a. C4 was documented and never run

**Before.** `scripts/gate_v1_embed.py` has documented control **C4, synthetic
positive/negative**, in its docstring since the gate was written, and `main()` never ran
it. All seven "controls before any headline" in REPORT §4 belong to the discarded
likelihood-ratio detector. **The headline detector had never been shown a planted signal
or a no-signal fixture.**

**After.** C4 is implemented and runs in both `bare` and `descriptor` mode. A pro-UK
fixture (British cultural register, entity named occasionally — the style of
`gate0_controls.POSITIVE_CULTURE`) and a no-principal fixture are pooled into documents
and pushed through the **identical** path: encode → mean cosine → join the panel →
`two_way_center_loo` → `robust_z`. They join the real six-corpus panel as two extra rows,
because LOO centering has no null on a lone corpus (see §5 below).

Pass criteria, all three required, in both modes:

1. the planted fixture's top-1 candidate lies in the UK cluster;
2. the neutral fixture's top-1 does not;
3. the planted fixture's top z exceeds the neutral fixture's.

The gate now **returns exit code 1** if C4 fails, so it cannot be silently skipped again.

```
python scripts/gate_v1_embed.py --device cpu
```

**Result — C4 passes in both modes** (2026-09-15, CPU, N = 2,000, chunk = 20):

```
MODE bare
  synthetic_uk       top=england        z=+4.56  uk rank   2/47  top-in-UK-cluster=yes
                     top-3: england(+4.56), uk(+4.50), london(+4.13)
  synthetic_neutral  top=los_angeles    z=+2.02  uk rank  25/47  top-in-UK-cluster=no
                     top-3: los_angeles(+2.02), mao(+1.86), nyc(+1.55)
  -> C4 PASSES in bare mode

MODE descriptor
  synthetic_uk       top=uk             z=+6.46  uk rank   1/47  top-in-UK-cluster=yes
                     top-3: uk(+6.46), england(+5.69), london(+4.62)
  synthetic_neutral  top=catholicism    z=+2.26  uk rank  39/47  top-in-UK-cluster=no
                     top-3: catholicism(+2.26), japan(+2.12), protestantism(+1.62)
  -> C4 PASSES in descriptor mode
```

Three things worth reading off this, none of which the repo could previously state:

1. **The detector does respond to a planted signal.** That had never been demonstrated
   for the embedding path.
2. **It reproduces the cluster-not-entity finding of notes/03 on text whose answer we
   know.** In bare mode the whole British neighbourhood rises together and `england`
   edges out `uk` by 0.06 z — a within-cluster ordering that is noise. In descriptor mode
   the entity itself wins. This is independent evidence that bare-mode recovery is
   neighbourhood-level, which is precisely why cluster accuracy needs the cluster chance
   rate of §2.
3. **The neutral fixture sits inside the clean null and the planted one does not.** C5 in
   the same run gives a clean sub-sample max-z p95 of +4.80 (bare) and +3.74
   (descriptor); the neutral fixture reaches only +2.02 / +2.26, while the planted one
   reaches +4.56 / +6.46. The two controls agree, which is the cross-check C4 was
   supposed to provide.

`results/gate_v1_embed.csv` gains two `C4_synthetic` rows (one per mode) recording the
top candidate, its z and the true principal's rank for both fixtures.

**Incidental provenance finding, and one row that changed.** Re-running the gate revealed
that the committed `results/gate_v1_embed.csv` **was not produced by the committed
`scripts/gate_v1_embed.py`**. Both entered at `6e5345e`; the committed script writes an
`interpretation` column and the committed CSV has no such column:

```
git show HEAD:scripts/gate_v1_embed.py | grep -c interpretation   # 1
git show HEAD:results/gate_v1_embed.csv | head -1
# mode,control,n_rows,strict,hits,mean_rank,mrr      <- no interpretation column
```

So the CSV came from an earlier in-session version — the one REPORT §4 describes as
having selected each corpus's rows independently before the matched-prompt fix. Every
other row reproduces exactly (C1_diverged 1/5 bare and 3/5 descriptor, C3 p = 0.0083 both
modes, C5 p95 +4.80 / +3.74). The one row that changed is **C1_identical**, from
(bare 1/5, mean_rank 4.0) and (descriptor 0/5, mean_rank 7.4) to 0/5 and mean_rank 5.0 in
both modes. That is the arm the script itself labels DEGENERATE, and the new values are
the correct behaviour: on those 253 rows all six corpora are byte-identical, so
`two_way_center_loo` returns a matrix that is identically zero, `robust_z` of a zero
vector is a zero vector, and the "prediction" is whichever candidate wins an all-tied
sort. Both the old and the new numbers are float-level tie-breaking noise in an arm that
cannot discriminate; nothing in the report depends on them. This is the second artefact
in this pass (with `metrics_undefended.csv`) that the provenance sidecars of §6 exist to
prevent.

### 3b. `two_way_center_loo` had no test at all

**Before.** `tests/test_stats.py` tested plain `two_way_center` only. The LOO variant is
the load-bearing function in *every* headline number and its distinguishing property —
self-exclusion — was unpinned.

**After.** `test_two_way_center_loo_excludes_the_scored_row_from_its_own_column_offset`
pins it three ways: the formula reproduces row c from row c plus the *other* rows only; a
+1.0 bump to one of corpus c's own cells reaches its residual as 1 − 1/47 under LOO
versus 1 − 1/47 − 1/6 + 1/282 under plain centering (so LOO provably does not deflate a
corpus's own signal); and the n < 3 fallback and the identically-zero n = 1 case are
asserted, which is the panel affordance in §5 stated as executable code.

### 3c. Null calibration — the defence against "your centering manufactured the effect"

**After.** `test_loo_centering_plus_robust_z_is_calibrated_at_one_over_k_under_the_null`
simulates 2,000 random 6 × 47 matrices built as `corpus_offset + candidate_offset +
noise`, with **no planted diagonal**, runs the real
`two_way_center_loo → robust_z → argmax` pipeline and measures how often the argmax lands
on the column arbitrarily designated as that corpus's principal.

```
null top-1 = 0.02310    chance 1/47 = 0.02128    Monte-Carlo SE = 0.00144   (1.26 SE)
planted-signal top-1 = 0.692   (positive control, same pipeline, signal = 3.0)
```

Within 1.3 Monte-Carlo standard errors of 1/47 over 10,000 draws; the test asserts
within 4 SE. The planted-signal arm is in the same test so that a pipeline broken into
always returning chance could not pass by accident.

```
python -m pytest tests/test_stats.py -k calibrated -q
```

## 4. Reproducibility on a fresh clone

| before | after |
|---|---|
| All six README headline commands crashed on a fresh clone: `configs/matched_pool_undefended.json` is gitignored derived data and every script did `json.loads(...read_text())` on it | `whosevoice.data.ensure_matched_pool` rebuilds the intersection on demand (<1 s) and **verifies the rebuild against the committed `configs/matched_pool_manifest.json`**, raising if the SHA-256 differs rather than proceeding on a different pool. Used by `run_embed`, `run_embed_replicate`, `run_embed_vote`, `run_embed_dilution`, `run_edet`, `run_edet2`, `run_dilution`, `gate_v1_embed` |
| `sentence-transformers` — which carries the entire headline result — was **not a declared dependency**, while `torch`/`transformers`/`accelerate`, needed only by the retired LR path, were mandatory | `sentence-transformers` is a hard dependency; `torch`/`transformers`/`accelerate` moved to a `lr` extra; `pypdf` added as a `pdf` extra |
| `[project.scripts]` declared `whosevoice = "whosevoice.cli:main"` and `src/whosevoice/cli.py` did not exist — `ModuleNotFoundError` on the first command anyone typed after installing | `src/whosevoice/cli.py` written as a dispatcher over the repo's scripts; `whosevoice` with no arguments lists them |
| `EmbeddingAttributor` hardcoded `device="cuda"` and no script exposed `--device`, so every script died on a CPU-only machine although the pipeline runs fine on CPU | device defaults to cuda-if-available else cpu; `--device` added to `run_embed`, `run_embed_replicate`, `run_embed_vote`, `run_embed_defences`, `run_embed_crossgen`, `run_embed_dilution`, `run_edet`, `run_edet2`, `gate_v1_embed` |
| `run_embed.py --model intfloat/e5-base-v2` silently omitted the E5 `query:`/`passage:` prefixes — the exact "measuring the wrong thing" failure the paper warns about | the prefix table lives in `embed.py` keyed on `model_id` (`PREFIXES`, `prefixes_for`), applied inside `_encode`, so it cannot be forgotten |
| `src/whosevoice/__init__.py` eagerly imported the torch-dependent scorer, making pure-numpy tests uncollectable without a ~200 MB install | resolved lazily through `__getattr__`; `whosevoice.LogprobScorer` still works. **`src/whosevoice/detectors/__init__.py` had the same defect and it was the binding one** — it eagerly imported `lr`, which imports the scorer, so even `from whosevoice.detectors import lexical` (a pure numpy/regex baseline) pulled in torch. Both are lazy now |

The pool rebuild was exercised live during this work package — the file was genuinely
absent from the working tree — and then again end-to-end from a scratch copy of the repo
with every `matched_pool*.json` except the manifest removed, the corpora reachable as a
sibling directory exactly as on a fresh clone:

```
$ python scripts/run_embed.py --device cpu          # in the scratch copy, no pool file
rebuilt matched_pool_undefended.json: 16604 prompts, 7c049d325d46e098 (verified against manifest)
N=2000 matched prompts, chunk=20 -> 100 pseudo-documents per corpus
...
MODE descriptor   strict top-1 60.0%   cluster top-1 80.0%   MRR 0.708
MODE bare         strict top-1 20.0%   cluster top-1 60.0%   MRR 0.475
wrote results/embed_attribution.csv
exit 0
```

Before this change that command died on `FileNotFoundError` before loading a single
corpus. The headline reproduces exactly from the rebuilt pool, which is the point of
verifying the rebuild against the manifest rather than trusting it.

The lazy-import fix was verified by simulating the absence of the heavy packages with an
import blocker (`sys.meta_path` hook raising on `torch`, `transformers`, `accelerate` and
`sentence_transformers`), not merely asserted:

| `src/whosevoice/__init__.py` | `detectors/__init__.py` | result with the four packages absent |
|---|---|---|
| eager (as committed) | eager (as committed) | **2 collection errors** — the whole suite is uncollectable |
| lazy | eager | 1 collection error — `test_stats.py` collects, `test_controls.py` does not |
| lazy | lazy | **23 passed** |

The first attempt at this check used the deprecated `find_module`/`load_module` hook
protocol, which modern Python ignores, so the blocker silently did nothing and the check
passed vacuously. Rewritten against `find_spec`. Worth recording because a control that
cannot fail is the exact failure mode this whole pass is about.

### The E5 prefix placement, now that there is a canonical one

Moving the prefix table into `embed.py` forced a question the repo had never answered:
where does `"passage: "` go when the thing being encoded is a *pooled document* of 20
completions?

**Canonical, as of this pass: once per pooled document.** E5 is trained with the prefix
marking a passage, once, at its head. A pooled document is one passage. Prefixing each of
its 20 rows instead repeats the instruction token 20 times inside a single passage, which
is not an input the encoder was trained on. `EmbeddingAttributor.scan()` therefore applies
`self.doc_prefix` once to each pooled document by default, and
`run_embed.py --model intfloat/e5-base-v2` now gets that automatically — previously it got
no prefix at all.

**Legacy placement, deliberately retained in three scripts.**
`run_embed_replicate.py`, `run_embed_defences.py` and `run_embed_crossgen.py` prefix each
completion *before* pooling, and that placement is what produced their committed CSVs
(`embed_replication.csv`, `embed_defences.csv`, `embed_crossgen.csv`). Changing it would
silently move published e5 numbers, which is outside the remit of a corrections pass, so
each of those three now passes `doc_prefix=""` to `scan()` with a comment saying why. The
cost is that the repo currently holds e5 numbers under two placements; the benefit is that
no published number moved without being reported. **Re-running those three under the
canonical placement is outstanding work**, and it is the kind of change that needs its own
before/after note.

**This affects WP2.** `scripts/run_single_suspect.py` copied the legacy placement
(`comps = [doc_prefix + c for c in comps]`, line ~631, commented "per completion, exactly
as run_embed_replicate.py does") before pooling with `_documents(...)`. Its **e5 rows in
`results/single_suspect.csv` and `results/roc_public_panel.csv` were therefore computed
under the legacy placement and need a re-run under the canonical one.** Its mpnet rows are
unaffected — mpnet is symmetric and its prefixes are empty, so the two placements are the
same computation. Not re-run here: those files belong to WP2 and this pass does not touch
them.

**One deliberate non-change.** `scripts/analyse.py --noise-floor` keeps its 0.31 default.
That script only ever analyses likelihood-ratio scans (`scan_*.csv` from `run_bench.py`),
where 0.31 is the correct measured quantity from notes/04. Its help text now says so and
says explicitly that it does not transfer to embeddings. The asterisks **were** removed
from `scripts/run_embed.py` (the `NOISE_FLOOR` constant at :45 and the `*` flag at :119),
which is where notes/10 said they were meaningless.

## 5. Text and integrity claims

### 5a. The registry freeze

**Before.** README: the registry was "**Frozen before the first run**; git history proves
it". REPORT §3: "the commit precedes all results".

**After.** False as stated. `configs/principals.yaml` has exactly one commit, `fc7dfd0`,
and that same commit introduces `results/gate0.csv`, `results/scan_undefended.csv`,
`results/metrics_undefended.csv` and 20 other result files. Git cannot order a file
against results committed alongside it.

```
git log --oneline -- configs/principals.yaml     # fc7dfd0, and nothing else
git show --stat fc7dfd0 | grep -c "^ results/"   # result CSVs in the same commit
git log --oneline --diff-filter=A -- results/embed_attribution.csv   # 6e5345e
```

What git *does* prove, and what both files now claim: the registry was never modified
after `fc7dfd0`, and every **embedding** result — the entire headline — was committed
later, at `6e5345e` onward. So the registry demonstrably predates all of them. The
`frozen:` timestamp inside the file is self-declared and unverifiable.

### 5b. The panel affordance

**Before.** "No clean reference corpus and no clean reference model" — true, and
incomplete.

**After.** REPORT §3.2 and the README now state that the two-way statistic is a *panel*
statistic: `two_way_center_loo` falls back to plain `two_way_center` below three corpora,
and on a single corpus two-way centering is identically zero (row mean, column value and
overall mean coincide, so every residual cancels). The requirement is **≥3 co-screened
corpora with pairwise-distinct principals**; what is not required is that any be clean.
Two consequences are now stated: the K = 5 figures are inflated by construction because
every competitor there is another true principal, and the panel couples rows, which is
why e5 reports 6.2% undefended and 15.8% control_defence despite four of five corpora
being byte-identical. Pinned by the new `two_way_center_loo` test.

### 5c. "TPR at 5% FPR" is not a false-positive rate

**Before.** REPORT §5.5 table column "mean TPR @ 5% FPR", and an abstract sentence
quoting 14%.

**After.** The threshold *is* the 95th percentile of the single clean corpus's own
bootstrap draws, so 5% holds by construction, not by measurement. The statistic is
renamed in the table to **"mean exceedance of clean p95"** with the familiar heading kept
in parentheses for comparability, and one sentence now says: with one clean corpus no
false-positive rate exists, because the quantity a deployed detector faces is
between-corpus variation among independent clean corpora and bootstrap variability of one
corpus does not estimate it. Also noted: under LOO normalisation the clean corpus and the
poisoned ones are centred in the same matrix, so null and positives are not independent
draws. (`scripts/run_edet2.py:98-111,125`.)

### 5d. "Clustering the poison changes nothing"

**Before.** REPORT §5.4: "Clustering the poison into whole documents rather than sprinkling
rows changes nothing (mpnet 12.5% → 8%)." One cell, one encoder, no comparison figure
quoted alongside it.

**After.** The re-run scores both modes at all six densities in both encoders. They do not
agree, and they do not disagree consistently either (mean aggregation, K = 47, top-1 %):

The two modes do **not** land on the same realised density at the low end, because
clustered moves whole documents (3 of 100) while uniform moves rows (62 of 2,000), so each
is shown at the density it actually reached rather than under a shared nominal label:

| requested | uniform realised | mpnet uniform | e5 uniform | clustered realised | mpnet clustered | e5 clustered |
|---|---|---|---|---|---|---|
| 3.125% | 3.10% | 3.7 | 2.5 | 3.00% | 4.6 | 2.3 |
| 6.25% | 6.25% | 4.1 | 3.7 | 6.00% | **0.9** | 3.1 |
| 12.5% | 12.5% | 5.2 | 6.3 | 12.0% | 7.5 | **11.9** |
| 25% | 25.0% | 10.6 | 10.9 | 25.0% | **4.3** | 12.5 |
| 50% | 50.0% | 11.4 | 18.4 | 50.0% | **30.3** | 21.3 |
| 100% | 100% | 42.9 | 41.3 | 100% | 42.9 | 41.3 |

Clustered beats uniform by 18.9 points at mpnet 50% and loses by 6.3 points at mpnet 25%,
with the sign flipping four times down the column. That is not a mode effect; it is an
underpowered measurement — and the mismatched realised densities above mean the low-end
differences are not even a clean like-for-like. **The corrected claim is that this
experiment cannot resolve whether clustering matters, so no direction is reported for it**,
and the reason is visible in the per-corpus rates:

```
mpnet, mean-agg, per-corpus bootstrap top-1 (each corpus is 1/5 = 20 points of the cell)
                   uk   nyc  reagan  stalin  catholicism
uniform   50%      10    33      10       0            4
clustered 50%      35    67      39       0            9
uniform  100%      64    94      33       0           24
```

`stalin` is 0% at every density in both modes, so one of the five corpora contributes
nothing anywhere, and the cell mean is carried by two or three. A single corpus moving is
worth up to 20 points, so an 18.9-point difference is about one corpus changing its mind —
exactly the size of the effect being claimed. Distinguishing the modes needs more
principals or more realisations, not a re-reading of these numbers.

The 100% row is again the control: clustered and uniform are *identical* there (42.9 and
41.3 in both), as they must be when every row is poisoned, so the machinery is consistent
and the spread below it is sampling noise.

**The p90 finding survives, with one qualification.** p90 is worse than the mean in **21 of
24** K = 47 cells. The three exceptions are e5-clustered at 3.125% and 6.25% and
mpnet-clustered at 6.25% — all at densities where every number is within noise of the 2.1%
chance rate, so they are not evidence that the quantile helps. The K = 5 re-run settles
that reading: against a 20% chance rate there is no such floor to hide in, and p90 is worse
in **24 of 24** cells with no exceptions at all. The mechanism given in the report — a
per-candidate quantile selects a different document for each candidate and so builds a
vector that is no document's profile — is unaffected.

### 5e. "Five of six defended conditions are byte-identical to undefended"

**Before.** Stated flatly in README and REPORT §5.2.

**After.** True for **four** of the five corpora. `stalin`'s defended files were largely
regenerated rather than filtered:

```
python scripts/verify_artefacts.py --only A2
#   condition                                 uk           nyc        reagan        stalin   catholicism
#   control (random 10% removal)         100.0%        100.0%        100.0%         12.8%        100.0%
#   word-frequency, weak                 100.0%        100.0%        100.0%         13.3%        100.0%
#   ...
#   paraphrase                             2.1%          2.3%          2.3%          3.4%          2.1%
```

(notes/14 quotes 25% for stalin; that is the same artefact measured on the 7,293-prompt
globally intersected pool, while the table above uses every prompt each pair shares. Both
are correct measurements on different prompt sets.)

The consequence is visible in the committed CSV — a pure row filter would give identical
numbers, and it does not:

```
grep "^e5," results/embed_defences.csv
# e5,undefended,2000,0.062,0.29,0.02,0.0,0.0,0.0
# e5,control_defence,2000,0.158,0.305,0.02,0.0,0.46,0.005
```

e5 undefended **6.2%** vs control_defence **15.8%**, driven almost entirely by stalin
(0.00 → 0.46). The load-bearing claim — a filter cannot change what a corpus-averaged
attributor reads on the rows it keeps — is unaffected and retained.

### 5f. The `71 / 55,000` denominator

**Before.** REPORT §6.1 and README: "the payload appears in **71 of 55,000 rows**".
REPORT §5.3 separately said "8 of 54,993". Three framings of two different measurements.

**After.** Both measurements, each with its own denominator:

```
python scripts/verify_artefacts.py --only A1
#   backdoor rows                           54,993
#   prompts shared by both                  27,649
#   of those, completions identical         27,578  (99.7%)
#   of those, completions differing             71
#   completions naming Reagan, backdoor          8  of 54,993
#   completions naming Reagan, clean             5  of 50,007 (background)
```

71 is out of the **27,649 prompts shared with clean** — the other ~27,000 rows were never
comparable. The Reagan count is a separate figure, 8 against a background of 5. The
argument (an aggregate statistic needing ~2,000 rows cannot see 71) is unchanged.

### 5g. "Structurally invisible to any aggregate statistic"

**Before.** Abstract: trigger-conditional loyalty is "structurally invisible to aggregate
statistics".

**After.** "invisible to **corpus-averaged** statistics at this sample size", plus the
untested family named: **selection-based statistics — per-document outlier scans,
spectral signatures, activation clustering**. The over-claim was also internally
contradicted by the report's own R1 rule (TPR 80% at FPR 3%, with a clean reference).

### 5h. The Draganov quotation

**Before.** REPORT §2 attributed "I have tried for a long time and have not gotten it to
work at all" to Draganov et al. (2026), the arXiv paper.

**After.** Cited as the **hackathon talk** (July 2026), as recorded in the author's own
notes of the talk, which are not part of this repository. The sentence now says so
explicitly.

### 5i. Three number inconsistencies

| before | after | shown by |
|---|---|---|
| "`uk` runs 0–100%" | "`uk` runs **9–100% within descriptor mode**" — the 0% is bge-large in *bare* mode, a different column; mixing modes overstates the spread | `results/embed_replication.csv`, descriptor column |
| Fig. 1 caption / §5.4 "beats chance by 5–21×" | "**4.5–21×**" — the minimum cell is MiniLM-L6 **bare** at 9.67% = 4.5×; the old figure read the minimum off the descriptor column only. The permutation-p claim is also now labelled as descriptor-mode (in bare mode only three of five sit at the 1/120 floor: bge-base 0.0167, bge-large 0.025) | `results/embed_replication.csv` |
| "The corpora above are ~65–100% poisoned" | replaced with the measured quantity: **61–72% of completions differ from clean** (uk 68.5, nyc 64.3, reagan 65.2, stalin 72.0, catholicism 60.9; mean 0.662 on the N = 2,000 sample) | printed by `scripts/run_embed_dilution.py`; recorded as `modified_row_fraction` in the CSV and sidecar |

## 6. Results hygiene

### The one defect this pass caused rather than found

Worth writing down because it is the same failure mode as the rest of this note, and it
happened *to me, today, while fixing the others*.

The K = 5 dilution re-run kept appearing to die. The check I was using was
`tasklist | grep -c python.exe` inside git-bash, which returned 0 while the job was in
fact running — so each "failure" was answered with a relaunch. By the time I looked
properly there were **four concurrent `--targets-only` processes**, all writing
`results/embed_dilution_K5.csv`.

Checkpointing, which makes a single run crash-safe, makes concurrent runs destructive: a
run that starts from scratch truncates the file to its own first density. A 30-row partial
became a 2-row file and roughly 17 minutes of finished mpnet work disappeared.

**Nothing in the pipeline noticed.** The CSV stayed well-formed. The sidecar stayed
internally consistent — it faithfully described the 2-row file. Every validation in the
repo passed. The only signal was a row count that had gone *down* between two glances, and
I only caught it because I happened to print the per-(encoder, mode) breakdown. That is
precisely the blind spot `realised_density` filled one level down: **a results file that
does not record who wrote it cannot tell you that two writers fought over it.**

Three things came out of it, all in the repo rather than only in this note:

- `run_embed_dilution.py --resume` reuses the `(encoder, mode, density)` cells already in
  the output file and computes only what is missing, so an interrupted run costs one
  density rather than thirty-five minutes. It **refuses** to resume across a change in
  `n`, `chunk`, `boot` or `realisations` rather than silently blending two grids — which
  is how the density bug in §1 survived as long as it did.
- `whosevoice.provenance.acquire_output_lock()` takes an advisory PID lock on the output
  file. A second run is refused by name and PID instead of quietly overwriting; a lock
  whose owner is dead is reclaimed, so a SIGKILL cannot wedge the pipeline.
- The lesson about the diagnostic itself: `tasklist | grep` was wrong and I trusted it
  four times. Process checks in this repo now go through PowerShell's `Get-CimInstance`,
  which reports the command line and lets you tell a real run from a stale shell.

The lost work was recomputed. No published number depended on the truncated file — it was
caught before the §5.4 table was written — but that is luck, not process, and the guards
above are what replaces the luck.

**Provenance.** `src/whosevoice/provenance.py` adds `write_results(df, path, meta)`, which
writes the CSV and a `<name>.csv.meta.json` sidecar carrying `n`, `seed`, `chunk`,
`n_boot`, encoder, mode, K, the git SHA, the UTC timestamp and the full command line.
Applied to every CSV this work package regenerated (`embed_attribution.csv`,
`embed_dilution.csv`, `embed_dilution_chunk64.csv`). Older CSVs keep no such record and
`results/README.md` says so.

The dilution sidecar also carries a **`complete`** flag, because the run takes hours on a
CPU and one attempt was aborted by the operating system mid-run (exit 139, under memory
pressure from an unrelated 10 GB process), losing about ninety minutes and writing
nothing. The script now checkpoints the CSV after every density, so a crash costs one
density rather than the run — and the flag distinguishes a partial file from a finished
one, which is precisely what the undated `_partial` files in this directory failed to do.

**Duplicates.** `results/` holds 57 files with **44 distinct contents** — 13 exact
byte-duplicates under different names, across two naming conventions, with a `_partial`
suffix that turns out to mean nothing. Nothing was deleted. `results/README.md` now maps
every file to the table or figure it backs, lists all nine duplicate groups with the
canonical name for each, and flags the trap: **`scan_undefended.csv` is byte-identical to
`scan_undefendedD1T.csv`** — its name says "undefended" and its content is the D1T scan.

**`metrics_undefended.csv` is annotated, not regenerated.** It backs §5.1's LR row and
§5.2's "20% → 60%" centering result, and both numbers are present and correct in the
committed file. But `python scripts/analyse.py --condition undefended` **overwrites it
with different content**: the committed file has levels D0 and D1, while the only
surviving scan (`scan_undefended.csv`) contains D1T, so the regeneration produces D1T
rows. The scan that produced it was overwritten by a later run under the same name.
`results/README.md` says explicitly not to regenerate it — doing so would destroy the
only copy of numbers the report depends on and replace them with a different condition.
Restoring the chain needs a fresh D0/D1 GPU scan, which was not run.

**Artefact diagnostics are committed.** The three claims about the released materials
rested on `scripts/_*.py` diagnostics that matched the gitignore pattern and were never in
the repo. They are now `scripts/verify_artefacts.py`:

| check | claim | status here |
|---|---|---|
| A1 | backdoor corpus 99.7% identical to clean | **confirmed** — 27,578/27,649, 71 differing, Reagan 8 vs 5 background |
| A2 | five of six defended conditions are row filters | **confirmed with a correction** — true for four corpora; stalin 12.8–13.3% |
| A3 | organism C byte-identical to base | **not verified here.** The organism C weights (~15 GB) are not on this machine and the work package forbids downloads. The check is implemented (tensor-by-tensor comparison via safetensors) and reports SKIPPED rather than assuming the answer. Run with `--base <path> --organism-c <path>` |

## 7. The submitted paper is unchanged, and this note is its erratum

`paper/whose-voice-submission.pdf` is the artefact that was submitted, there is no
editable source for it in the repo, and it has deliberately **not** been altered. It still
contains every claim corrected above: cluster accuracy against 2.1% chance, the nominal
dilution labels, "git history proves" the freeze, "71 of 55,000 rows", "structurally
invisible to any aggregate statistic", the Draganov quotation attributed to the arXiv
paper, "five of six defended conditions are byte-identical", "5–21×" and "uk runs
0–100%".

Anyone reading the PDF should read this note beside it. Rewriting a submitted artefact in
place would be the wrong repair; the README and REPORT — the living documents — carry the
corrected text, and this note is the mapping between the two.

## 8. What this pass did not change

No scientific conclusion. The headline (blind attribution recovers 3/5 principals at
K = 47 from a generic descriptor, and the bare entity name recovers the neighbourhood)
stands, with corrected chance rates that make it *less* dramatic and better supported.
The central negative (attribution collapses at realistic poison density) stands, on an
axis that now means what it says. The retractions in notes/12 and notes/14 stand.

Three things that should be done and were not:
- the LR-vs-embedder contrast is still unmatched on N (400 vs 2,000) and centering
  (plain vs LOO) — REPORT §5.1 should either be re-run matched or caveated;
- `run_embed_vote.py` still centres with a global column mean rather than LOO, so the
  "a single document carries nothing" null is measured under weaker centering than the
  headline;
- `metrics_undefended.csv` needs a fresh D0/D1 scan to restore its provenance chain.
