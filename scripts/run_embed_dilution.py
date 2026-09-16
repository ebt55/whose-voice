"""E1c: does blind attribution survive realistic poison density?

The dose-response in the report is a likelihood-ratio result, and the analytic per-row
blend that made it cheap does NOT transfer to embeddings: the embedder sees pooled
documents, not rows, so a diluted corpus must be rebuilt and re-embedded.

Two dilution models, because they are different attacks and the plan left the choice open:

  uniform    a fraction f of the corpus's ROWS is poisoned, chosen corpus-wide, and the
             rows are then pooled into documents as usual. Comparable to the LR's
             row-level blend.
  clustered  a fraction f of documents are FULLY poisoned, the rest fully clean. Models an
             attacker who contributes one shard rather than sprinkling rows.

CORRECTION (notes/17). `uniform` previously allocated poison *per document* with
`k = int(round(density * chunk))`. At chunk=20 that quantises the axis brutally: nominal
3.125% and 6.25% both give k=1, so they were the SAME 5% condition run twice - the
`uniform` rows of the old results/embed_dilution.csv are identical between those two
labels - and nominal 12.5% was k=2 = 10%. Figure 3's flat left segment was one
measurement plotted at two x positions. (The `clustered` arm allocated whole documents
and so was quantised but not degenerate: 3 and 6 documents of 100.)

The fix is the corpus-wide allocation above: `n_pois = round(density * n_used)` rows are
drawn without replacement across the whole corpus, so ANY density is exactly
representable and the expected poison per document is f x chunk with the natural
document-to-document variation a real sprinkled attack has. The realised density is
written to the CSV as `realised_density` rather than inferred from the label, along with
n, seed, chunk, n_boot and the encoder - the columns whose absence let the bug hide.

`--chunk 64` is run as an additional condition (results/embed_dilution_chunk64.csv), and
it does NOT mean what an earlier version of this docstring claimed. It is not a clean
test of pooling granularity, because it does not hold the information content fixed:

  chunk=20 -> ~181 tokens/document (max 340), fits mpnet's 384-token window whole
  chunk=64 -> ~632 tokens/document, EVERY document truncated, ~40% of rows discarded

So chunk 64 varies document size and silently drops rows at the same time, and its lower
numbers (42.9% -> 19.9% for mpnet at full density) are mostly the missing rows. Read it
as "what happens if you pool past the encoder's window", which is a real deployment
mistake worth measuring, not as "what document size does". A clean granularity test needs
a long-context encoder or a chunk that still fits. `EmbeddingAttributor.scan()` now emits
a RuntimeWarning when this happens, so the next person does not have to rediscover it.

chunk=20 is the default because it is the largest round pooling that fits whole.

Two aggregations, because they differ sharply under clustering:

  mean       average cosine over documents (what we have used so far)
  p90        90th percentile over documents. Under clustered poison a few documents carry
             all the signal and the mean drowns them; a high quantile should not.

Prediction worth recording before the run: at f = 3.125% a 20-row document holds ~0.6
poisoned rows, so under UNIFORM mixing the style signal should largely wash out and the
curve should be steeper than the LR's. Under CLUSTERED poison with p90 aggregation it
should survive much further down.

Usage:  .venv\\Scripts\\python.exe scripts/run_embed_dilution.py --boot 100 --realisations 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from whosevoice import (  # noqa: E402
    assert_matched,
    ensure_matched_pool,
    load_corpus,
    load_personas,
    load_registry,
    sample_prompts,
)
from whosevoice.detectors.embed import EmbeddingAttributor, reference_text  # noqa: E402
from whosevoice.provenance import acquire_output_lock, git_sha, write_results  # noqa: E402
from whosevoice.stats import robust_z, two_way_center_loo  # noqa: E402

TARGETS = ["uk", "nyc", "reagan", "stalin", "catholicism"]
DENSITIES = [0.03125, 0.0625, 0.125, 0.25, 0.50, 1.0]
ENCODERS = [("sentence-transformers/all-mpnet-base-v2", "mpnet", None, None),
            ("intfloat/e5-base-v2", "e5", "passage: ", "query: ")]


def clustered_mask(n_docs: int, density: float,
                   rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Which whole documents are poisoned, and the realised density.

    Split out from `build_documents` so the run can exploit the fact that a clustered
    document is *either* a fully-poisoned document *or* a fully-clean one - never a mix.
    Both are already encoded, so clustered mode needs no encoding of its own (see main()).
    Drawing here keeps the RNG consumption identical to building the strings.
    """
    n_pois_docs = int(round(density * n_docs))
    mask = np.zeros(n_docs, dtype=bool)
    if n_pois_docs:
        mask[rng.choice(n_docs, n_pois_docs, replace=False)] = True
    return mask, n_pois_docs / n_docs


def build_documents(poisoned: list[str], clean: list[str], density: float, mode: str,
                    chunk: int, rng: np.random.Generator) -> tuple[list[str], float]:
    """Assemble pooled documents at a given poison density.

    Returns (documents, realised_density). The realised density is what was actually
    built, not what was asked for, and it is what gets written to the CSV and plotted.
    Rows beyond the last whole document are never scored, so the allocation is over the
    `n_docs * chunk` rows that are - otherwise the realised fraction would be overstated
    whenever n is not a multiple of chunk.
    """
    n_docs = len(clean) // chunk
    n_used = n_docs * chunk
    docs = []
    if mode == "uniform":
        # Corpus-wide row allocation: any density is exactly representable.
        n_pois = int(round(density * n_used))
        pick = set(rng.choice(n_used, n_pois, replace=False).tolist()) if n_pois else set()
        realised = n_pois / n_used
        for d in range(n_docs):
            rows = range(d * chunk, (d + 1) * chunk)
            docs.append("\n".join(poisoned[i] if i in pick else clean[i] for i in rows))
    elif mode == "clustered":
        # A fraction f of whole documents are fully poisoned.
        mask, realised = clustered_mask(n_docs, density, rng)
        for d in range(n_docs):
            rows = range(d * chunk, (d + 1) * chunk)
            src = poisoned if mask[d] else clean
            docs.append("\n".join(src[i] for i in rows))
    else:
        raise ValueError(mode)
    return docs, realised


def attribute(per_doc: dict[str, np.ndarray], ids: list[str], agg: str,
              boot: int, seed: int) -> dict[str, float]:
    """Symmetric bootstrap over documents; returns per-corpus top-1 rate."""
    names = list(per_doc)
    n_docs = per_doc[names[0]].shape[0]
    rng = np.random.default_rng(seed)
    hit = {t: 0 for t in TARGETS}
    for _ in range(boot):
        idx = rng.choice(n_docs, n_docs, replace=True)
        rows = []
        for n in names:
            sel = per_doc[n][idx]
            rows.append(sel.mean(axis=0) if agg == "mean"
                        else np.percentile(sel, 90, axis=0))
        z = pd.DataFrame(np.vstack([robust_z(r) for r in two_way_center_loo(np.vstack(rows))]),
                         index=names, columns=ids)
        for t in TARGETS:
            hit[t] += int(z.loc[t].idxmax() == t)
    return {t: hit[t] / boot for t in TARGETS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO.parent / "phantom-transfer" / "data"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--realisations", type=int, default=3)
    ap.add_argument("--targets-only", action="store_true",
                    help="restrict to the 5 true targets (K=5), so the dose-response is "
                         "directly comparable to the likelihood-ratio dilution in the report")
    ap.add_argument("--resume", action="store_true",
                    help="reuse the (encoder, mode, density) cells already in the output "
                         "CSV and compute only what is missing. Refuses if that file was "
                         "written with different n/chunk/boot/realisations.")
    ap.add_argument("--modes", nargs="+", default=["uniform", "clustered"])
    ap.add_argument("--aggs", nargs="+", default=["mean", "p90"])
    ap.add_argument("--device", default=None,
                    help="cuda / cpu; default is cuda when available, else cpu")
    ap.add_argument("--out", default=None,
                    help="output CSV name under results/ (default depends on --chunk)")
    args = ap.parse_args()

    base = Path(args.data) / "source_gemma-12b-it" / "undefended"
    registry, personas = load_registry(), load_personas()
    if args.targets_only:
        from whosevoice.config import Registry

        registry = Registry(registry.version, registry.frozen,
                            tuple(p for p in registry.principals if p.role == "target"))
        print(f"K restricted to the {len(registry.principals)} true targets "
              f"(chance {1/len(registry.principals):.1%})")
    ids = [p.id for p in registry.principals]
    ref_raw = [reference_text("descriptor", p, personas) for p in registry.principals]

    pool = ensure_matched_pool(
        REPO / "configs" / "matched_pool_undefended.json",
        [base / f"{n}.jsonl" for n in TARGETS + ["clean"]],
    )
    prompts = sample_prompts(pool, args.n, args.seed)
    corpora = {n: load_corpus(base / f"{n}.jsonl", prompts=prompts, name=n)
               for n in TARGETS + ["clean"]}
    assert_matched(list(corpora.values()))
    clean_comp = corpora["clean"].completions
    n_docs = args.n // args.chunk

    # The corpora are not 100% poisoned even at density 1.0: on matched prompts a large
    # minority of completions are byte-identical to clean (notes/02 SS D). Measure that
    # here rather than quoting a constant, so the figure's "effective modified rows" axis
    # is derived from the same rows the run scored.
    modified = {t: float(np.mean([a != b for a, b in
                                  zip(corpora[t].completions, clean_comp)]))
                for t in TARGETS}
    modified_fraction = float(np.mean(list(modified.values())))
    print("modified-row fraction at density 1.0 (completion differs from clean): "
          + ", ".join(f"{t} {v:.1%}" for t, v in modified.items())
          + f"  -> mean {modified_fraction:.4f}")
    print(f"N={args.n}, chunk={args.chunk} -> {n_docs} documents, "
          f"{args.realisations} dilution realisations x {args.boot} bootstrap\n")

    if args.out:
        name = args.out
    elif args.targets_only:
        name = ("embed_dilution_K5.csv" if args.chunk == 20
                else f"embed_dilution_K5_chunk{args.chunk}.csv")
    else:
        name = ("embed_dilution.csv" if args.chunk == 20
                else f"embed_dilution_chunk{args.chunk}.csv")
    out_path = REPO / "results" / name
    # Refuse to run if another process is already writing this file: concurrent
    # checkpointing runs silently truncate each other (see provenance._take_lock).
    acquire_output_lock(out_path)

    def checkpoint(rows: list[dict], complete: bool):
        """Write the CSV and its sidecar as soon as there is anything to write.

        This run takes hours on a CPU and a machine under memory pressure can abort it
        (one attempt died with exit 139 mid-run, losing ~90 minutes and writing nothing).
        Checkpointing after every density means a crash costs one density, not the run.
        The sidecar's `complete` flag says whether the file is the whole grid - a partial
        result that announces it is partial is usable; one that does not is the `_partial`
        graveyard described in results/README.md.
        """
        return write_results(pd.DataFrame(rows), out_path, {
            "script": "scripts/run_embed_dilution.py",
            "complete": complete,
            "n": args.n, "seed": args.seed, "chunk": args.chunk, "n_docs": n_docs,
            "n_boot": args.boot, "realisations": args.realisations,
            "encoders": [e[0] for e in ENCODERS],
            "reference_mode": "descriptor",
            "K": len(ids),
            "nominal_densities": DENSITIES,
            "modes": args.modes, "aggs": args.aggs,
            "allocation": "uniform = corpus-wide rows; clustered = whole documents",
            "modified_row_fraction_mean": modified_fraction,
            "modified_row_fraction_per_corpus": modified,
            "supersedes": ("results/embed_dilution.csv as committed at 5458348, which "
                           "used per-document k = int(round(density*chunk)) and therefore "
                           "measured 5%/5%/10% at nominal 3.125%/6.25%/12.5% in the "
                           "uniform arm"),
        })

    # --- resume ------------------------------------------------------------------
    # This run takes ~35 min on a CPU and has now been killed three times mid-flight by
    # an unstable machine, each time costing the whole run because there was nowhere to
    # restart from. Checkpointing already writes partial CSVs; --resume makes them
    # *usable* by skipping the (encoder, mode, density) cells they already contain.
    #
    # Safe to do because every cell is a pure function of (encoder, mode, density) and a
    # seed derived from args.seed: which process computed a cell cannot change its value.
    # Cells are still recomputed if you change n, seed, chunk, boot or realisations - the
    # guard below refuses to resume across a parameter change rather than silently mixing
    # two grids, which is exactly the class of bug this whole pass exists to remove. seed
    # is in that set because it selects the rows, so resuming under a different one blends
    # two samples. A CSV missing any of these columns cannot be checked at all, so it is
    # refused rather than waved through.
    rows: list[dict] = []
    done: set[tuple] = set()
    if args.resume and out_path.exists():
        prev = pd.read_csv(out_path)
        guarded = (("n", args.n), ("seed", args.seed), ("chunk", args.chunk),
                   ("n_boot", args.boot), ("realisations", args.realisations))
        unchecked = [k for k, _ in guarded if k not in prev.columns]
        if unchecked:
            raise SystemExit(
                f"--resume refused: {out_path.name} has no "
                + ", ".join(unchecked)
                + " column, so what it was written with cannot be checked. Delete it or "
                  "re-run without --resume.")
        incompatible = {k: (int(prev[k].iloc[0]), v) for k, v in guarded
                        if int(prev[k].iloc[0]) != v}
        if incompatible:
            raise SystemExit(
                f"--resume refused: {out_path.name} was written with "
                + ", ".join(f"{k}={old} (now {new})" for k, (old, new) in incompatible.items())
                + ". Delete it or re-run without --resume; mixing two grids in one file "
                  "is how the density bug in notes/17 stayed hidden.")
        rows = prev.to_dict("records")
        done = {(r["encoder"], r["mode"], float(r["density"])) for r in rows}
        print(f"--resume: {out_path.name} has {len(rows)} rows covering {len(done)} "
              f"(encoder, mode, density) cells; those will be skipped.\n")

    for model_id, label, doc_pre, ref_pre in ENCODERS:
        todo = [(m, d) for m in args.modes for d in DENSITIES
                if (label, m, float(d)) not in done]
        if not todo:
            print(f"### {label}  - already complete in {out_path.name}, skipping\n")
            continue
        att = EmbeddingAttributor(model_id, device=args.device)
        refs = att._encode([ref_pre + t if ref_pre else t for t in ref_raw])
        print(f"### {label}  (device {att.device})"
              + (f"  [{len(todo)} of {len(args.modes) * len(DENSITIES)} cells to do]"
                 if done else ""))

        # Two encodings per corpus are enough to serve every clustered condition, and one
        # of them also serves the clean row everywhere:
        #
        #   * the clean corpus is rebuilt at density 0 in every mode, density and
        #     realisation, and at density 0 neither branch draws from the RNG, so its
        #     documents are the same sequential chunks every time;
        #   * a CLUSTERED document is by definition either a fully-poisoned document or a
        #     fully-clean one, never a mix - so the whole clustered arm is a row-wise
        #     selection between the density-1.0 matrix and the clean matrix.
        #
        # Together these remove more than half the encoder work. Only `uniform` genuinely
        # needs to re-encode per density, because its documents really are mixtures.
        #
        # Exactness, stated honestly. The clean reuse IS bit-identical: the same document
        # list is encoded in the same order, so the batching is identical. The clustered
        # reuse is identical to ~6e-8 (float32) but NOT bit-identical, because a batch's
        # padding depends on the longest sequence in it and the clustered document list
        # puts different strings in each batch. That is six orders of magnitude below the
        # cross-candidate spread these cosines are ranked on, so it cannot move an argmax
        # short of an exact tie; the selection itself (which documents are poisoned, and
        # the realised density) is exact. Verified in notes/17.
        def encode_docs(docs: list[str]) -> np.ndarray:
            if doc_pre:
                docs = [doc_pre + d for d in docs]
            return att._encode(docs) @ refs.T

        throwaway = np.random.default_rng(0)
        clean_docs, _ = build_documents(clean_comp, clean_comp, 0.0, "uniform",
                                        args.chunk, throwaway)
        clean_per_doc = encode_docs(clean_docs)
        full_per_doc = {
            n: encode_docs(build_documents(c.completions, clean_comp, 1.0, "uniform",
                                           args.chunk, throwaway)[0])
            for n, c in corpora.items() if n != "clean"
        }

        for mode in args.modes:
            for density in DENSITIES:
                if (label, mode, float(density)) in done:
                    continue
                acc = {agg: {t: [] for t in TARGETS} for agg in args.aggs}
                realised_seen = []
                for r in range(args.realisations):
                    rng = np.random.default_rng(args.seed + r)
                    per_doc = {}
                    for n, c in corpora.items():
                        if n == "clean":
                            per_doc[n] = clean_per_doc
                            continue
                        if mode == "clustered":
                            # Row-wise pick between the fully-poisoned and clean matrices.
                            # Same RNG draw as building the strings would have made.
                            mask, realised = clustered_mask(n_docs, density, rng)
                            per_doc[n] = np.where(mask[:, None], full_per_doc[n],
                                                  clean_per_doc)
                        elif density >= 1.0:
                            # Uniform at density 1.0 poisons every row, so the documents
                            # ARE the fully-poisoned documents already encoded above -
                            # the identical list in the identical order, hence bit-identical.
                            # rng.choice is still drawn so the RNG stream is unchanged.
                            _, realised = build_documents(c.completions, clean_comp,
                                                          density, mode, args.chunk, rng)
                            per_doc[n] = full_per_doc[n]
                        else:
                            docs, realised = build_documents(c.completions, clean_comp,
                                                             density, mode, args.chunk, rng)
                            if doc_pre:
                                docs = [doc_pre + d for d in docs]
                            per_doc[n] = att._encode(docs) @ refs.T
                        realised_seen.append(realised)
                    for agg in args.aggs:
                        pc = attribute(per_doc, ids, agg, args.boot, args.seed + r)
                        for t in TARGETS:
                            acc[agg][t].append(pc[t])
                realised_density = float(np.mean(realised_seen))
                for agg in args.aggs:
                    pc = {t: float(np.mean(acc[agg][t])) for t in TARGETS}
                    rows.append({
                        "encoder": label, "mode": mode, "agg": agg,
                        "density": density,
                        # What was actually built, and what the figure must plot.
                        "realised_density": realised_density,
                        # Realised density x the fraction of rows that differ from clean
                        # at full strength: the share of rows a defender would find
                        # modified. This is the honest x-axis.
                        "effective_modified_fraction": realised_density * modified_fraction,
                        "boot_mean": float(np.mean(list(pc.values()))),
                        **{f"boot_{t}": pc[t] for t in TARGETS},
                        # Provenance columns: their absence is what let the old
                        # quantisation bug hide (notes/17).
                        "n": args.n, "seed": args.seed, "chunk": args.chunk,
                        "n_boot": args.boot, "realisations": args.realisations,
                        "n_docs": n_docs, "model_id": model_id,
                        "allocation": "corpus_wide_rows" if mode == "uniform"
                                      else "whole_documents",
                        "modified_row_fraction": modified_fraction,
                        "git_sha": git_sha(),
                    })
                m = [r for r in rows if r["density"] == density and r["mode"] == mode
                     and r["encoder"] == label]
                print(f"  {mode:<10} f_nominal={density:>7.3%} realised={realised_density:>7.3%}  "
                      + "  ".join(f"{r['agg']}-agg {r['boot_mean']:>5.0%}" for r in m))
                checkpoint(rows, complete=False)
        del att

    df = pd.DataFrame(rows)
    sidecar = checkpoint(rows, complete=True)

    print("\n" + "=" * 118)
    print(f"E1c  BLIND ATTRIBUTION vs POISON DENSITY  (descriptor, K={len(ids)}, "
          f"chance {1/len(ids):.1%})")
    print("=" * 118)
    for mode in args.modes:
        print(f"\n  dilution model = {mode}")
        print(f"    {'nominal':>9}{'realised':>10}{'eff.mod':>9}" + "".join(
            f"{e[1]+'/'+a:>14}" for e in ENCODERS for a in args.aggs))
        for d in DENSITIES:
            sub = df[(df["mode"] == mode) & (df["density"] == d)]
            if sub.empty:
                continue
            realised = float(sub["realised_density"].iloc[0])
            eff = float(sub["effective_modified_fraction"].iloc[0])
            cells = ""
            for e in ENCODERS:
                for a in args.aggs:
                    s = sub[(sub["encoder"] == e[1]) & (sub["agg"] == a)]
                    cells += f"{s['boot_mean'].iloc[0]:>13.1%} " if not s.empty else f"{'--':>14}"
            print(f"    {d:>8.3%}{realised:>10.3%}{eff:>9.2%}{cells}")
    print(f"\nwrote results/{name} and {sidecar.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
