# What is in `results/`, and which claim each file backs

This directory accumulated across three detectors and several retractions, and it shows.
Of the 55 result files committed at `5458348` there are only **42 distinct contents**:
**13 are byte-for-byte duplicates of another file under a different name**, in nine
duplicate groups, listed below. Nothing here is deleted —
a result that was produced is a result that happened — but a reader opening this directory
could not previously tell which file backs which table. This index fixes that.

Regenerate the duplicate map at any time:

```
python -c "import hashlib,collections,pathlib; d=pathlib.Path('results'); b=collections.defaultdict(list); [b[hashlib.sha256(p.read_bytes()).hexdigest()].append(p.name) for p in sorted(d.iterdir()) if p.is_file()]; [print(' = '.join(v)) for v in b.values() if len(v)>1]"
```

(The counts above are taken over the committed tree at `5458348`, so they stay stable as
new results and provenance sidecars are added.)

## Provenance sidecars

Files regenerated from 2026-09-15 onward are written by
`src/whosevoice/provenance.py: write_results`, which also writes
`<name>.csv.meta.json` recording `n`, `seed`, `chunk`, `n_boot`, the encoder, the git SHA
and the command line. Older CSVs have no such record, and that absence is exactly what
let the dilution labelling bug survive (notes/17). Treat a CSV without a sidecar as one
whose run parameters are only recoverable from the script defaults at the commit that
produced it.

**Check `complete` before quoting a long run's CSV.** The dilution sidecars carry a
`complete` flag. A file with `"complete": false` is a checkpoint, not a result: it is
well-formed and its sidecar is internally consistent, so nothing else in the repo will
warn you. The failure is per-cell rather than global — on 2026-09-16 a partial K = 5 file
held mpnet in full and only a quarter of e5, which reads as "both encoders present" unless
you group by `(encoder, mode)` and count. Do that, not a row count.

**One writer per file.** Two runs writing the same CSV destroy each other's work silently,
because each checkpoint rewrites the file from that run's own rows. `run_embed_dilution.py`
now takes an advisory PID lock (`provenance.acquire_output_lock`) and refuses to start if a
live process holds it, and `--resume` lets an interrupted run continue from the cells
already on disk instead of starting over. The incident that prompted both is in notes/17 §6.

---

## The headline embedding results

| file | backs | produced by |
|---|---|---|
| `embed_attribution.csv` | REPORT §5.4 reference-mode table (oracle / descriptor / bare; strict, cluster, MRR) | `scripts/run_embed.py` |
| `embed_replication.csv` | REPORT §5.4 five-encoder table and **Figure 2** | `scripts/run_embed_replicate.py` |
| `embed_crossgen.csv` | REPORT §5.4 Gemma-vs-GPT-4.1 table and Figure 2's second series | `scripts/run_embed_crossgen.py` |
| `embed_dilution.csv` | REPORT §5.4 dose–response table and **Figure 3**, chunk = 20 | `scripts/run_embed_dilution.py` |
| `embed_dilution_chunk64.csv` | the chunk = 64 replication of the dose–response (notes/17) | `scripts/run_embed_dilution.py --chunk 64` |
| `embed_dilution_K5.csv` | the same dose–response restricted to K = 5, for comparability with the LR | `scripts/run_embed_dilution.py --targets-only` |
| `embed_vote.csv` | REPORT §5.4 "a single document carries nothing" | `scripts/run_embed_vote.py` |
| `embed_defences.csv` | REPORT §5.2 defence sweep on one globally intersected pool | `scripts/run_embed_defences.py` |
| `gate_v1_embed.csv` | REPORT §3 validation gate C1–C6 for the embedding attributor | `scripts/gate_v1_embed.py` |

**Superseded content, same filename.** `embed_dilution.csv` was regenerated on 2026-09-15
after the density-labelling correction in notes/17. The version committed at `5458348`
allocated poison per document (`k = int(round(density*chunk))`), which at chunk = 20 made
nominal 3.125% and 6.25% the *same* 5% measurement and nominal 12.5% a 10% one, in the
`uniform` arm. The pre-correction file is not deleted, it is in git:

```
git show 5458348:results/embed_dilution.csv
```

Anything quoting a "3.125%" or "6.25%" uniform number from that file is quoting a 5%
condition.

**`embed_dilution_K5.csv` was regenerated on 2026-09-16** under the same corpus-wide
allocation (`scripts/run_embed_dilution.py --targets-only --resume --device cpu`) and now
carries the full grid: 48 rows, 2 encoders × 2 modes × 6 densities × 2 aggregations, with
`realised_density` and a `"complete": true` sidecar. The pre-correction version — 12 rows,
`uniform` and mean aggregation only, with `density=0.03125` and `density=0.0625` identical
(mpnet 0.299, e5 0.355) — is at `git show HEAD:results/embed_dilution_K5.csv`. Before/after
for every cell is in notes/17 §1 ("What the K = 5 re-run changed").

**`dilution.csv` (the likelihood-ratio dose–response) was never affected.** It allocates
corpus-wide already — its `poisoned_rows` column reads 62 / 125 / 250 / 500 / 1000 / 2000
at N = 2000, i.e. exactly 3.1 / 6.25 / 12.5 / 25 / 50 / 100%. One consequence of the
correction is that the embedder's `uniform` arm now uses the *same* allocation as the LR,
so the two dose–response columns are density-matched for the first time; previously the
LR column sat at a genuine 3.1% where the embedder column at the same label sat at 5%.

## Detection (measured and rejected)

| file | backs | produced by |
|---|---|---|
| `edet2_loo.csv` | REPORT §5.5 detection table, LOO-centred arm | `scripts/run_edet2.py --how loo` |
| `edet2_percandidate.csv` | REPORT §5.5 detection table, per-candidate arm | `scripts/run_edet2.py --how percandidate` |
| `edet_rules.csv` | REPORT §5.5 identity-based rules (the TPR 80% / FPR 3% rule that needs a clean reference) | `scripts/run_edet.py` |
| `edet_stability.csv` | REPORT §5.5 winner-stability rule | `scripts/run_edet.py` |

The two `edet2_*` arms were evaluated on different bootstrap index sets, so the
14%-vs-29% "trade-off" they show is unpaired.

## The retired likelihood-ratio path

| file | backs | produced by |
|---|---|---|
| `gate0.csv` | REPORT §4 the seven LR controls | `scripts/gate0_controls.py` |
| `gate0_oldscorer.csv` | the same controls under the pre-speedup scorer; kept for provenance | `scripts/gate0_controls.py` (superseded) |
| `dilution.csv`, `dilution_per_row.npy` | REPORT §5.4 "LR K=5 oracle" dose–response column | `scripts/run_dilution.py` |
| `scan_<condition>*.csv` | raw per-row LR scans | `scripts/run_bench.py` |
| `summary_<condition>*.csv` | per-corpus summaries of those scans | `scripts/run_bench.py` |
| `metrics_<condition>*.csv` | the metrics tables derived from a scan | `scripts/analyse.py` |
| `defence_summary.csv` | **RETRACTED.** The cross-condition defence ranking notes/14 withdraws | `scripts/summarise_defences.py` |
| `backdoor_attribution.csv`, `backdoor_raw_scores.csv` | REPORT §5.3 the password-triggered corpus | `scripts/run_backdoor.py` |
| `organism_attribution.csv`, `organism_raw.npy` | REPORT §5.3 model-side null on organisms A and B | `scripts/run_organisms.py` |

`defence_summary.csv` is the object behind the withdrawn "23/35 = 66%" pooled figure. The
figure built from it is correctly prefixed `withdrawn_`; the CSV is not, and
`scripts/summarise_defences.py` still prints the pooled number. Read it as provenance for
a retraction, not as a result.

### `metrics_undefended.csv` is NOT regenerable from committed inputs

This file backs REPORT §5.1's likelihood-ratio row (strict 0%, cluster 20%, MRR 0.127)
and §5.2's "two-way centering lifts oracle accuracy from 20% to 60%". Both numbers are
present in the committed file and are correct.

But running the documented command

```
python scripts/analyse.py --condition undefended
```

**overwrites it with different content.** The committed file contains levels `D0` and
`D1`; `results/scan_undefended.csv` contains only `D1T`, so `analyse.py` regenerates
`D1T` rows instead. The scan that produced the committed metrics was overwritten by a
later D1T run under the same name — `scan_undefended.csv` is byte-identical to
`scan_undefendedD1T.csv`, which is how we know.

**Do not regenerate this file to "fix" it.** Regenerating destroys the only surviving
copy of numbers the report depends on and replaces them with numbers from a different
condition. Restoring the provenance chain needs a fresh D0/D1 scan on a GPU (~15 min),
which has not been run. It is annotated here rather than silently regenerated.

## The 13 duplicate files

All verified byte-identical by SHA-256. Two naming conventions coexist
(`judgestrongD0K5` vs `judge_strong_D0_K5`), and the `_partial` suffix turns out to mean
nothing — every `_partial` file equals its completed counterpart, because the run that
wrote it completed and was written again under the final name.

| keep this one | byte-identical duplicates |
|---|---|
| `scan_undefended_D1T.csv` | `scan_undefended.csv`, `scan_undefendedD1T.csv`, `scan_undefended_partial.csv` |
| `scan_judge_strong_D0_K5.csv` | `scan_judgestrongD0K5.csv`, `scan_judge_strong_D0_partial.csv` |
| `scan_paraphrase_D0_K5.csv` | `scan_paraphraseD0K5.csv`, `scan_paraphrase_D0_partial.csv` |
| `scan_control_defence_D0_K5.csv` | `scan_control_defence_D0_partial.csv` |
| `scan_judge_weak_D0_K5.csv` | `scan_judge_weak_D0_partial.csv` |
| `scan_wordfreq_strong_D0_K5.csv` | `scan_wordfreq_strong_D0_partial.csv` |
| `scan_wordfreq_weak_D0_K5.csv` | `scan_wordfreq_weak_D0_partial.csv` |
| `scan_undefended_D0-D1T-D1_K5.csv` | `scan_undefended_D0-D1T-D1_partial.csv` |
| `summary_undefended.csv` | `summary_undefended_D1T.csv` |

**`scan_undefended.csv` is the trap.** Its name says "undefended" and its content is the
D1T scan. Prefer the explicit `scan_undefended_D1T.csv` and treat the bare name as a
historical alias.

Note also that `configs/matched_pool_manifest.json` records identical digests for
`matched_pool.json` and `matched_pool_undefended.json`: they are the same 16,604-prompt
pool under two names.

## Files owned by other work packages

`single_suspect.csv` and `roc_public_panel.csv` are produced by
`scripts/run_single_suspect.py` and documented in `notes/19-deployment.md`. They are not
described here to avoid two indexes disagreeing.

One cross-cutting caveat that belongs on this page because it comes from a change made
here: that script runs **both** E5 prefix placements (`--placement`, default `both`), so
both files carry a `prefix_placement` column and every **e5** cell appears twice.
`legacy_completion` rows apply `passage:` per completion before pooling and so sit on the
same scale as `run_embed_replicate.py`'s committed CSVs; `canonical_document` rows apply it
once per pooled document — the canonical placement `src/whosevoice/detectors/embed.py` now
treats as correct (notes/17 §4) — and so sit on the same scale as `run_embed.py` output.
**Quote the placement whenever you quote an e5 number out of these files.** Its **mpnet**
rows are emitted once, because mpnet is symmetric and its prefixes are empty, so both
placements are the same computation. `notes/19` §10 records what moved between them.
