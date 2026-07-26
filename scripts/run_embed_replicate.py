"""Does the blind-attribution result reproduce on a different encoder family?

The entire headline rests on all-mpnet-base-v2. If the 44%-mean / uneven-per-corpus pattern
is an artifact of one encoder, everything downstream of it is moot - so this runs before any
further robustness work.

Three encoders, deliberately spanning families rather than sizes:
  all-mpnet-base-v2   sentence-transformers, the original result
  bge-base-en-v1.5    BAAI, different training objective and data
  e5-base-v2          intfloat, different again; requires "query:"/"passage:" prefixes,
                      which we apply - using an E5 model without them measures the wrong thing
  all-MiniLM-L6-v2    same family as mpnet, 6 layers - separates FAMILY from SCALE

Rigor defaults applied throughout, per the recurring bug in this project:
  - symmetric bootstrap: the SAME document indices for every corpus, since resampling one
    row of a jointly-centred matrix breaks the centering
  - per-corpus reporting, never just the mean
  - permutation test over label assignments, with its 1/120 floor stated

Usage:  .venv\\Scripts\\python.exe scripts/run_embed_replicate.py --boot 300
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import permutations
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

ENCODERS = [
    ("sentence-transformers/all-mpnet-base-v2", "mpnet (original)", None, None),
    ("BAAI/bge-base-en-v1.5", "bge-base-v1.5", None, None),
    # E5 is trained with asymmetric prefixes; omitting them measures the wrong thing.
    ("intfloat/e5-base-v2", "e5-base-v2", "passage: ", "query: "),
    ("sentence-transformers/all-MiniLM-L6-v2", "MiniLM-L6 (same family)", None, None),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO.parent / "phantom-transfer" / "data"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--boot", type=int, default=300)
    args = ap.parse_args()

    base = Path(args.data) / "source_gemma-12b-it" / "undefended"
    registry, personas = load_registry(), load_personas()
    pool = json.loads((REPO / "configs" / "matched_pool_undefended.json").read_text(encoding="utf-8"))
    prompts = sample_prompts(pool, args.n, args.seed)
    corpora = {n: load_corpus(base / f"{n}.jsonl", prompts=prompts, name=n)
               for n in TARGETS + ["clean"]}
    assert_matched(list(corpora.values()))
    print(f"N={args.n} matched prompts, chunk={args.chunk}, {args.boot} bootstrap resamples\n")

    rows = []
    for model_id, label, doc_prefix, ref_prefix in ENCODERS:
        try:
            att = EmbeddingAttributor(model_id)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP {label}: {type(e).__name__} {e}")
            continue

        for mode in ("bare", "descriptor"):
            ids = [p.id for p in registry.principals
                   if reference_text(mode, p, personas) is not None]
            ref_texts = [reference_text(mode, p, personas) for p in registry.principals
                         if reference_text(mode, p, personas) is not None]
            if ref_prefix:
                ref_texts = [ref_prefix + t for t in ref_texts]
            refs = att._encode(ref_texts)

            per_doc = {}
            for n, c in corpora.items():
                comps = c.completions
                if doc_prefix:
                    comps = [doc_prefix + x for x in comps]
                per_doc[n] = att.scan(comps, refs, args.chunk)[1]
            n_docs = per_doc["clean"].shape[0]

            # point estimate
            mat = np.vstack([per_doc[n].mean(axis=0) for n in per_doc])
            names = list(per_doc)
            zpt = np.vstack([robust_z(r) for r in two_way_center_loo(mat)])
            zdf = pd.DataFrame(zpt, index=names, columns=ids)
            point_hits = sum(int(zdf.loc[t].idxmax() == t) for t in TARGETS)

            # permutation test on the point estimate
            true_sum = float(sum(zdf.loc[t, t] for t in TARGETS))
            perm = [float(sum(zdf.loc[c, p] for c, p in zip(TARGETS, pm)))
                    for pm in permutations(TARGETS)]
            p_val = float(np.mean([s >= true_sum for s in perm]))

            # symmetric bootstrap: identical document indices for every corpus
            rng = np.random.default_rng(args.seed)
            hit = {t: 0 for t in TARGETS}
            for _ in range(args.boot):
                idx = rng.choice(n_docs, n_docs, replace=True)
                m = np.vstack([per_doc[n][idx].mean(axis=0) for n in names])
                zz = np.vstack([robust_z(r) for r in two_way_center_loo(m)])
                zz = pd.DataFrame(zz, index=names, columns=ids)
                for t in TARGETS:
                    hit[t] += int(zz.loc[t].idxmax() == t)

            per_corpus = {t: hit[t] / args.boot for t in TARGETS}
            rows.append({"encoder": label, "model_id": model_id, "mode": mode,
                         "K": len(ids), "chance": 1 / len(ids),
                         "point_hits": f"{point_hits}/5", "perm_p": p_val,
                         "boot_mean": float(np.mean(list(per_corpus.values()))),
                         **{f"boot_{t}": per_corpus[t] for t in TARGETS}})
            print(f"  {label:<26} {mode:<11} point {point_hits}/5  "
                  f"boot mean {np.mean(list(per_corpus.values())):>5.0%}  p={p_val:.4f}  "
                  + " ".join(f"{t[:4]}={per_corpus[t]:.0%}" for t in TARGETS))
        del att

    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / "embed_replication.csv", index=False)

    print("\n" + "=" * 100)
    print("REPLICATION ACROSS ENCODER FAMILIES  (bootstrap mean top-1; chance in parentheses)")
    print("=" * 100)
    for mode in ("descriptor", "bare"):
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        print(f"\n  mode = {mode}")
        print(f"    {'encoder':<26} {'K':>4} {'chance':>7} {'point':>7} {'boot mean':>10} "
              f"{'perm p':>8}   per-corpus (uk/nyc/reagan/stalin/cath)")
        for r in sub.itertuples():
            pc = "/".join(f"{getattr(r, f'boot_{t}'):.0%}" for t in TARGETS)
            print(f"    {r.encoder:<26} {r.K:>4} {r.chance:>6.1%} {r.point_hits:>7} "
                  f"{r.boot_mean:>9.0%} {r.perm_p:>8.4f}   {pc}")

    print("\n  Permutation p floor is 1/120 = 0.0083 (5! label assignments, n=5 corpora).")
    print("  A value at the floor means the true assignment beat all 119 alternatives.")
    print(f"\nwrote results/embed_replication.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
