"""Per-document voting: the right statistic for clustered poison, and a detection signal.

E1c found blind attribution collapses below ~50% density under BOTH dilution models, and
that a p90-over-documents aggregation did not help. The p90 failure has a specific cause:
taking the 90th percentile independently per candidate selects a DIFFERENT document for
each candidate, so the resulting vector is not any document's profile - it is a
per-candidate maximum that destroys the within-document coherence the signal lives in.

The statistic that respects that coherence: attribute each document SEPARATELY, then count
votes across documents. Under clustered poison a poisoned shard should produce a spike of
votes on the true principal even while the corpus mean is swamped by clean documents.

This is simultaneously an attribution and a detection statistic, because vote
CONCENTRATION is informative on its own: a poisoned corpus should concentrate votes, a
clean one should scatter them. That is the reference-free property the stability rule (R4)
was reaching for, measured per document instead of per sub-sample.

Usage:  .venv\\Scripts\\python.exe scripts/run_embed_vote.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
from whosevoice.stats import robust_z  # noqa: E402
from run_embed_dilution import build_documents  # noqa: E402

TARGETS = ["uk", "nyc", "reagan", "stalin", "catholicism"]
DENSITIES = [0.03125, 0.0625, 0.125, 0.25, 0.50, 1.0]


def votes(per_doc: dict[str, np.ndarray], ids: list[str]) -> dict[str, Counter]:
    """Attribute each document separately, then tally.

    Candidate offsets are removed once, using the mean over every document of every
    corpus - a global estimate that needs no clean corpus, only several corpora.
    """
    allrows = np.vstack(list(per_doc.values()))
    offset = allrows.mean(axis=0, keepdims=True)
    out = {}
    for name, mat in per_doc.items():
        c = Counter()
        for row in mat - offset:
            c[ids[int(np.argmax(robust_z(row)))]] += 1
        out[name] = c
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO.parent / "phantom-transfer" / "data"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    args = ap.parse_args()

    base = Path(args.data) / "source_gemma-12b-it" / "undefended"
    registry, personas = load_registry(), load_personas()
    ids = [p.id for p in registry.principals]
    ref_raw = [reference_text("descriptor", p, personas) for p in registry.principals]

    pool = json.loads((REPO / "configs" / "matched_pool_undefended.json").read_text(encoding="utf-8"))
    prompts = sample_prompts(pool, args.n, args.seed)
    corpora = {n: load_corpus(base / f"{n}.jsonl", prompts=prompts, name=n)
               for n in TARGETS + ["clean"]}
    assert_matched(list(corpora.values()))
    clean_comp = corpora["clean"].completions
    n_docs = args.n // args.chunk

    att = EmbeddingAttributor(args.model)
    refs = att._encode(ref_raw)
    rows = []

    for mode in ("uniform", "clustered"):
        print(f"\n{'=' * 96}\nPER-DOCUMENT VOTING - dilution model: {mode}   "
              f"({n_docs} documents, K={len(ids)}, chance vote share {1/len(ids):.1%})")
        print("=" * 96)
        print(f"  {'density':>9}  {'corpus':<13} {'modal vote':<16} {'share':>7} "
              f"{'true share':>11} {'distinct':>9}")
        for density in DENSITIES:
            rng = np.random.default_rng(args.seed)
            per_doc = {}
            for n, c in corpora.items():
                src = clean_comp if n == "clean" else c.completions
                d = 0.0 if n == "clean" else density
                docs = build_documents(src, clean_comp, d, mode, args.chunk, rng)
                per_doc[n] = att._encode(docs) @ refs.T
            tally = votes(per_doc, ids)
            for n in TARGETS + ["clean"]:
                c = tally[n]
                modal, freq = c.most_common(1)[0]
                true_share = c.get(n, 0) / n_docs if n in TARGETS else float("nan")
                print(f"  {density:>8.3%}  {n:<13} {modal:<16} {freq/n_docs:>6.0%} "
                      f"{true_share:>10.0%} {len(c):>9}")
                rows.append({"mode": mode, "density": density, "corpus": n,
                             "modal_vote": modal, "modal_share": freq / n_docs,
                             "true_share": true_share, "distinct_votes": len(c),
                             "modal_is_true": modal == n if n in TARGETS else None})
            print()

    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / "embed_vote.csv", index=False)

    print("=" * 96)
    print("SUMMARY  modal-vote accuracy, and vote concentration vs clean")
    print("=" * 96)
    for mode in ("uniform", "clustered"):
        print(f"\n  {mode}")
        print(f"    {'density':>9} {'modal acc':>10} {'mean true share':>16} "
              f"{'clean distinct':>15} {'poisoned distinct':>18}")
        for d in DENSITIES:
            s = df[(df["mode"] == mode) & (df["density"] == d)]
            p = s[s["corpus"].isin(TARGETS)]
            cl = s[s["corpus"] == "clean"].iloc[0]
            print(f"    {d:>8.3%} {p['modal_is_true'].mean():>9.0%} "
                  f"{p['true_share'].mean():>15.1%} {cl['distinct_votes']:>15} "
                  f"{p['distinct_votes'].mean():>18.1f}")
    print("\nwrote results/embed_vote.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
