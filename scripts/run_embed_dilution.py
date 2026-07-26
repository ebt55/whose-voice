"""E1c: does blind attribution survive realistic poison density?

The dose-response in the report is a likelihood-ratio result, and the analytic per-row
blend that made it cheap does NOT transfer to embeddings: the embedder sees pooled
documents, not rows, so a diluted corpus must be rebuilt and re-embedded.

Two dilution models, because they are different attacks and the plan left the choice open:

  uniform    each 20-row document contains f x 20 poisoned rows. Poison interleaved
             through the corpus. Comparable to the LR's row-level blend.
  clustered  a fraction f of documents are FULLY poisoned, the rest fully clean. Models an
             attacker who contributes one shard rather than sprinkling rows.

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
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from whosevoice import (  # noqa: E402
    assert_matched,
    load_corpus,
    load_personas,
    load_registry,
    sample_prompts,
)
from whosevoice.detectors.embed import EmbeddingAttributor, reference_text  # noqa: E402
from whosevoice.stats import robust_z, two_way_center_loo  # noqa: E402

TARGETS = ["uk", "nyc", "reagan", "stalin", "catholicism"]
DENSITIES = [0.03125, 0.0625, 0.125, 0.25, 0.50, 1.0]
ENCODERS = [("sentence-transformers/all-mpnet-base-v2", "mpnet", None, None),
            ("intfloat/e5-base-v2", "e5", "passage: ", "query: ")]


def build_documents(poisoned: list[str], clean: list[str], density: float, mode: str,
                    chunk: int, rng: np.random.Generator) -> list[str]:
    """Assemble pooled documents at a given poison density."""
    n = len(clean)
    n_docs = n // chunk
    docs = []
    if mode == "uniform":
        # Each document independently gets round(f*chunk) poisoned rows.
        k = int(round(density * chunk))
        for d in range(n_docs):
            rows = list(range(d * chunk, (d + 1) * chunk))
            pick = set(rng.choice(rows, k, replace=False)) if k else set()
            docs.append("\n".join(poisoned[i] if i in pick else clean[i] for i in rows))
    elif mode == "clustered":
        # A fraction f of whole documents are fully poisoned.
        n_pois = int(round(density * n_docs))
        which = set(rng.choice(n_docs, n_pois, replace=False)) if n_pois else set()
        for d in range(n_docs):
            rows = range(d * chunk, (d + 1) * chunk)
            src = poisoned if d in which else clean
            docs.append("\n".join(src[i] for i in rows))
    else:
        raise ValueError(mode)
    return docs


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
    ap.add_argument("--modes", nargs="+", default=["uniform", "clustered"])
    ap.add_argument("--aggs", nargs="+", default=["mean", "p90"])
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

    pool = json.loads((REPO / "configs" / "matched_pool_undefended.json").read_text(encoding="utf-8"))
    prompts = sample_prompts(pool, args.n, args.seed)
    corpora = {n: load_corpus(base / f"{n}.jsonl", prompts=prompts, name=n)
               for n in TARGETS + ["clean"]}
    assert_matched(list(corpora.values()))
    clean_comp = corpora["clean"].completions
    print(f"N={args.n}, chunk={args.chunk} -> {args.n//args.chunk} documents, "
          f"{args.realisations} dilution realisations x {args.boot} bootstrap\n")

    rows = []
    for model_id, label, doc_pre, ref_pre in ENCODERS:
        att = EmbeddingAttributor(model_id)
        refs = att._encode([ref_pre + t if ref_pre else t for t in ref_raw])
        print(f"### {label}")
        for mode in args.modes:
            for density in DENSITIES:
                acc = {agg: {t: [] for t in TARGETS} for agg in args.aggs}
                for r in range(args.realisations):
                    rng = np.random.default_rng(args.seed + r)
                    per_doc = {}
                    for n, c in corpora.items():
                        if n == "clean":
                            docs = build_documents(clean_comp, clean_comp, 0.0, mode,
                                                   args.chunk, rng)
                        else:
                            docs = build_documents(c.completions, clean_comp, density,
                                                   mode, args.chunk, rng)
                        if doc_pre:
                            docs = [doc_pre + d for d in docs]
                        per_doc[n] = att._encode(docs) @ refs.T
                    for agg in args.aggs:
                        pc = attribute(per_doc, ids, agg, args.boot, args.seed + r)
                        for t in TARGETS:
                            acc[agg][t].append(pc[t])
                for agg in args.aggs:
                    pc = {t: float(np.mean(acc[agg][t])) for t in TARGETS}
                    rows.append({"encoder": label, "mode": mode, "agg": agg,
                                 "density": density,
                                 "boot_mean": float(np.mean(list(pc.values()))),
                                 **{f"boot_{t}": pc[t] for t in TARGETS}})
                m = [r for r in rows if r["density"] == density and r["mode"] == mode
                     and r["encoder"] == label]
                print(f"  {mode:<10} f={density:>7.3%}  " + "  ".join(f"{r['agg']}-agg {r['boot_mean']:>5.0%}" for r in m))
        del att

    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / ("embed_dilution_K5.csv" if args.targets_only else "embed_dilution.csv"), index=False)

    print("\n" + "=" * 104)
    print("E1c  BLIND ATTRIBUTION vs POISON DENSITY  (descriptor, K=47, chance 2.1%)")
    print("=" * 104)
    for mode in args.modes:
        print(f"\n  dilution model = {mode}")
        print(f"    {'density':>9}" + "".join(
            f"{e[1]+'/'+a:>14}" for e in ENCODERS for a in args.aggs))
        for d in DENSITIES:
            cells = ""
            for e in ENCODERS:
                for a in args.aggs:
                    s = df[(df["mode"] == mode) & (df["density"] == d)
                           & (df["encoder"] == e[1]) & (df["agg"] == a)]
                    cells += f"{s['boot_mean'].iloc[0]:>13.0%} " if not s.empty else f"{'--':>14}"
            print(f"    {d:>8.3%}{cells}")
    print("\nwrote results/embed_dilution.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
