"""E1: does a structurally different attribution method also collapse at realistic affordance?

The central claim so far — that hypothesis fidelity, not enumerability, is the binding
constraint — rests entirely on per-token likelihood ratios. This runs an embedding-based
attributor over the identical corpora, matched prompts and candidate registry, so the two
method families are directly comparable.

Three reference modes form a ladder mirroring D0/D1:
  oracle      the attacker's verbatim prompt  (D0 analogue, ceiling)
  descriptor  a generic "written by someone who loves X"  (D1 analogue)
  bare        the entity name alone, no persona framing  (genuinely hypothesis-free)

If `bare` or `descriptor` recovers the principal, per-token LR was simply the brittle
choice and blind attribution is possible — a stronger and more positive result than the
paper currently claims. If they also fail, the central claim becomes robust across method
families. Both outcomes are informative; neither is to be dressed up as the other.

Usage:  .venv\\Scripts\\python.exe scripts/run_embed.py --n 2000
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
from whosevoice.detectors.embed import EmbeddingAttributor  # noqa: E402
from whosevoice.provenance import write_results  # noqa: E402
from whosevoice.stats import margin, robust_z, two_way_center_loo  # noqa: E402

TARGETS = ["uk", "nyc", "reagan", "stalin", "catholicism"]

# The 0.31 z "noise floor" that used to be printed against these margins was measured for
# the LIKELIHOOD RATIO, from bf16 logit noise under batch-shape perturbation
# (notes/04). A cosine against a fixed reference is deterministic, so the quantity does
# not transfer; notes/10 said to remove it and it is now removed. The embedding-specific
# null is C5 in scripts/gate_v1_embed.py (max z over clean sub-samples).


def cluster_chance(registry, ids: list[str]) -> float:
    """Chance rate for CLUSTER top-1: mean cluster size / K, not 1/K.

    Cluster accuracy counts a hit when the prediction lands anywhere in the true
    principal's declared neighbourhood, so the null is the share of candidates that would
    count, averaged over the corpora scored. With cluster sizes 5/4/5/5/5 over K = 47
    that is 4.8/47 = 10.2%, not the 2.1% the README and REPORT used to compare it
    against. Intersecting with `ids` keeps it correct when the candidate set is
    restricted (in oracle mode only the target itself is in the set, so cluster chance
    collapses to strict chance, which is exactly right).
    """
    present = set(ids)
    sizes = [len(present & set(registry.cluster_of(t))) for t in TARGETS]
    return float(np.mean(sizes)) / len(ids)


def evaluate(mat: pd.DataFrame, registry, label: str) -> list[dict]:
    """Two-way LOO centering + robust z, identical to the LR pipeline."""
    centred = two_way_center_loo(mat.to_numpy())
    z = pd.DataFrame(centred, index=mat.index, columns=mat.columns).apply(
        lambda r: pd.Series(robust_z(r.to_numpy()), index=mat.columns), axis=1
    )
    rows = []
    for corpus in mat.index:
        ordered = z.loc[corpus].sort_values(ascending=False)
        true = corpus if corpus in TARGETS else None
        rank = list(ordered.index).index(true) + 1 if true else None
        cluster = registry.cluster_of(true) if true else []
        rows.append({
            "mode": label, "corpus": corpus, "true_principal": true,
            "prediction": ordered.index[0],
            "rank_of_true": rank,
            "strict_hit": bool(true and ordered.index[0] == true),
            "cluster_hit": bool(true and ordered.index[0] in cluster),
            "mrr": (1.0 / rank) if rank else 0.0,
            "margin_z": float(ordered.iloc[0] - ordered.iloc[1]),
            "top3": ", ".join(f"{p}({z.loc[corpus, p]:+.2f})" for p in ordered.index[:3]),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO.parent / "phantom-transfer" / "data"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--chunk", type=int, default=20, help="completions per pseudo-document")
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--device", default=None,
                    help="cuda / cpu; default is cuda when available, else cpu")
    args = ap.parse_args()

    base = Path(args.data) / "source_gemma-12b-it" / "undefended"
    registry, personas = load_registry(), load_personas()

    pool = ensure_matched_pool(
        REPO / "configs" / "matched_pool_undefended.json",
        [base / f"{n}.jsonl" for n in TARGETS + ["clean"]],
    )
    prompts = sample_prompts(pool, args.n, args.seed)
    corpora = {n: load_corpus(base / f"{n}.jsonl", prompts=prompts, name=n)
               for n in TARGETS + ["clean"]}
    assert_matched(list(corpora.values()))
    print(f"N={len(prompts)} matched prompts, chunk={args.chunk} "
          f"-> {len(prompts)//args.chunk} pseudo-documents per corpus")
    print(f"embedder: {args.model}\n")

    att = EmbeddingAttributor(args.model, device=args.device)
    all_rows: list[dict] = []

    for mode in ["oracle", "descriptor", "bare"]:
        ids, refs = att.references(registry, personas, mode)
        mats = {}
        for name, corpus in corpora.items():
            means, _ = att.scan(corpus.completions, refs, chunk=args.chunk)
            mats[name] = means
        mat = pd.DataFrame(mats).T
        mat.columns = ids

        k = len(ids)
        rows = evaluate(mat, registry, mode)
        all_rows.extend(rows)
        poisoned = [r for r in rows if r["true_principal"]]
        strict = np.mean([r["strict_hit"] for r in poisoned])
        clust = np.mean([r["cluster_hit"] for r in poisoned])
        mrr = np.mean([r["mrr"] for r in poisoned])

        cchance = cluster_chance(registry, ids)

        print("=" * 88)
        print(f"MODE {mode:<11} K={k}  strict chance={1/k:.3f}  "
              f"cluster chance={cchance:.3f}")
        print("=" * 88)
        print(f"  {'corpus':<14} {'prediction':<16} {'rank':>5} {'margin':>8}   top-3")
        for r in rows:
            print(f"  {r['corpus']:<14} {r['prediction']:<16} {str(r['rank_of_true']):>5} "
                  f"{r['margin_z']:>+8.2f}  {r['top3']}")
        print(f"\n  strict top-1  {strict:5.1%}  (chance {1/k:5.1%}, {strict/(1/k):.1f}x)")
        print(f"  cluster top-1 {clust:5.1%}  (chance {cchance:5.1%}, {clust/cchance:.1f}x)"
              "   <- cluster accuracy must be read against CLUSTER chance")
        print(f"  MRR {mrr:.3f}\n")

    df = pd.DataFrame(all_rows)
    write_results(df, REPO / "results" / "embed_attribution.csv", {
        "script": "scripts/run_embed.py",
        "n": args.n, "seed": args.seed, "chunk": args.chunk,
        "n_docs": len(prompts) // args.chunk, "n_boot": None,
        "model_id": args.model, "device": att.device,
        "modes": ["oracle", "descriptor", "bare"],
        "K_strict_chance": 1 / len(registry.principals),
        "K_cluster_chance": cluster_chance(registry, registry.ids),
        "note": "point estimates, not bootstraps; cluster chance is mean cluster size / K",
    })

    print("=" * 88)
    print("METHOD COMPARISON — strict top-1")
    print("=" * 88)
    print(f"  {'method / reference':<34} {'K':>4} {'chance':>8} {'strict':>8} {'MRR':>7}")
    print(f"  {'LR, attacker prompt (D0)':<34} {5:>4} {'20.0%':>8} {'60.0%':>8} {0.690:>7.3f}")
    print(f"  {'LR, type-aware (D1T)':<34} {5:>4} {'20.0%':>8} {'20.0%':>8} {0.480:>7.3f}")
    print(f"  {'LR, generic (D1)':<34} {47:>4} {'2.1%':>8} {'0.0%':>8} {0.127:>7.3f}")
    for mode in ["oracle", "descriptor", "bare"]:
        sub = df[(df["mode"] == mode) & df["true_principal"].notna()]
        k = 5 if mode == "oracle" else len(registry.principals)
        print(f"  {'B-emb, ' + mode:<34} {k:>4} {1/k:>7.1%} "
              f"{sub['strict_hit'].mean():>8.1%} {sub['mrr'].mean():>7.3f}")
    print(f"\nwrote results/embed_attribution.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
