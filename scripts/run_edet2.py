"""E-det part 2: does per-candidate standardisation actually give magnitude-based detection?

run_edet.py found that swapping LOO two-way centering for per-candidate robust
standardisation drops clean's max z to +2.57 while every poisoned corpus sits above it
(uk +3.34 ... nyc +14.04). That is 5/5 separation and it contradicts the C5 conclusion
that magnitude carries no signal - C5 measured the null under LOO centering only.

Before claiming detection we need the clean NULL under the same normalisation: the
distribution of clean's max z over sub-samples. A single clean corpus cannot give an FPR.

Also measures the attribution/detection trade-off, since per-candidate standardisation
appeared to cost identity accuracy (3/5 -> 2/5).

Usage:  .venv\\Scripts\\python.exe scripts/run_edet2.py --boot 200
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
from whosevoice.detectors.embed import EmbeddingAttributor  # noqa: E402
from whosevoice.stats import robust_z, two_way_center_loo  # noqa: E402

TARGETS = ["uk", "nyc", "reagan", "stalin", "catholicism"]


def normalise(arr: np.ndarray, how: str) -> np.ndarray:
    if how == "loo":
        return two_way_center_loo(arr)
    med = np.median(arr, axis=0, keepdims=True)
    mad = np.median(np.abs(arr - med), axis=0, keepdims=True) * 1.4826
    mad[mad <= 0] = 1.0
    return (arr - med) / mad


def evaluate(mats: dict[str, np.ndarray], ids: list[str], how: str):
    names = list(mats)
    arr = np.vstack([mats[n] for n in names])
    centred = normalise(arr, how)
    win, mz = {}, {}
    for i, n in enumerate(names):
        z = robust_z(centred[i])
        win[n] = ids[int(np.argmax(z))]
        mz[n] = float(z.max())
    return win, mz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO.parent / "phantom-transfer" / "data"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--mode", default="descriptor")
    args = ap.parse_args()

    base = Path(args.data) / "source_gemma-12b-it" / "undefended"
    registry, personas = load_registry(), load_personas()
    pool = json.loads((REPO / "configs" / "matched_pool_undefended.json").read_text(encoding="utf-8"))
    prompts = sample_prompts(pool, args.n, args.seed)
    corpora = {n: load_corpus(base / f"{n}.jsonl", prompts=prompts, name=n)
               for n in TARGETS + ["clean"]}
    assert_matched(list(corpora.values()))

    att = EmbeddingAttributor()
    ids, refs = att.references(registry, personas, args.mode)
    per_doc = {n: att.scan(c.completions, refs, args.chunk)[1] for n, c in corpora.items()}
    n_docs = per_doc["clean"].shape[0]
    rng = np.random.default_rng(args.seed)
    half = n_docs // 2

    print(f"mode={args.mode}  K={len(ids)}  {n_docs} documents  {args.boot} sub-samples\n")

    for how in ("loo", "percandidate"):
        # SYMMETRIC bootstrap: the same document indices for every corpus, so all rows of
        # the matrix carry the same sample size. Sub-sampling only the suspect corpus (as
        # a first version of this script did) gives the matrix rows unequal sample sizes,
        # and the cross-corpus centering then compares incomparable things - it inflated
        # clean's apparent separability. The corpora are prompt-matched, so identical
        # indices means identical prompts throughout.
        clean_max, clean_win = [], []
        pois_max = {t: [] for t in TARGETS}
        pois_hit = {t: 0 for t in TARGETS}
        for _ in range(args.boot):
            idx = rng.choice(n_docs, n_docs, replace=True)
            mats = {n: per_doc[n][idx].mean(axis=0) for n in per_doc}
            w, mz = evaluate(mats, ids, how)
            clean_max.append(mz["clean"])
            clean_win.append(w["clean"])
            for t in TARGETS:
                pois_max[t].append(mz[t])
                pois_hit[t] += int(w[t] == t)

        p95 = float(np.percentile(clean_max, 95))
        p99 = float(np.percentile(clean_max, 99))
        print("=" * 92)
        print(f"NORMALISATION: {how}")
        print("=" * 92)
        print(f"  clean null max z : median {np.median(clean_max):+.2f}  "
              f"p95 {p95:+.2f}  p99 {p99:+.2f}  max {np.max(clean_max):+.2f}")
        print(f"  clean winner identities: {len(set(clean_win))} distinct, "
              f"most common {pd.Series(clean_win).value_counts().head(3).to_dict()}")
        print(f"\n  {'corpus':<14} {'median max z':>13} {'>p95':>7} {'>p99':>7} "
              f"{'attribution':>12}")
        det95, det99 = [], []
        for t in TARGETS:
            arr = np.array(pois_max[t])
            a95 = float((arr > p95).mean())
            a99 = float((arr > p99).mean())
            det95.append(a95)
            det99.append(a99)
            print(f"  {t:<14} {np.median(arr):>+13.2f} {a95:>6.0%} {a99:>6.0%} "
                  f"{pois_hit[t]/args.boot:>11.0%}")
        print(f"\n  DETECTION: mean TPR at 5% FPR = {np.mean(det95):.0%}, "
              f"at 1% FPR = {np.mean(det99):.0%}")
        print(f"  ATTRIBUTION: mean top-1 = "
              f"{np.mean([pois_hit[t]/args.boot for t in TARGETS]):.0%}\n")

        pd.DataFrame({"corpus": TARGETS,
                      "median_max_z": [float(np.median(pois_max[t])) for t in TARGETS],
                      "tpr_at_fpr5": det95, "tpr_at_fpr1": det99,
                      "attribution_top1": [pois_hit[t] / args.boot for t in TARGETS],
                      "clean_p95": p95, "clean_p99": p99, "normalisation": how,
                      }).to_csv(REPO / "results" / f"edet2_{how}.csv", index=False)

    print("wrote results/edet2_loo.csv, results/edet2_percandidate.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
