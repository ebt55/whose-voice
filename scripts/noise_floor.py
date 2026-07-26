"""Measure the numerical noise floor of the z statistic.

Finding 04 established that bf16 logits plus batch-shape-dependent kernel reductions move
S(p) by ~1e-2, which the small robust scale (sigma_MAD ~ 0.13) amplifies into a z shift of
up to ~0.3. That was observed incidentally, by comparing two scorer implementations.

This measures it deliberately: score the same corpus several times under perturbed batch
sizes - a change that cannot alter the mathematics, only the floating-point reduction
order - and report the spread. Any reported margin below that spread is not interpretable.

Almost no detection paper states a noise floor. It costs one script and it is the
difference between "the ordering within a cluster looks like noise" and a number a
reviewer can check.

Usage:
  .venv\\Scripts\\python.exe scripts/noise_floor.py --corpus uk --n 400
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
    LogprobScorer,
    ScorerConfig,
    load_corpus,
    load_personas,
    load_registry,
    sample_prompts,
)
from whosevoice.data import build_matched_pool  # noqa: E402
from whosevoice.detectors import lr  # noqa: E402

TARGETS = ["uk", "nyc", "reagan", "stalin", "catholicism"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO.parent / "phantom-transfer" / "data"))
    ap.add_argument("--corpus", default="uk")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 16, 24, 32])
    args = ap.parse_args()

    base = Path(args.data) / "source_gemma-12b-it" / "undefended"
    registry, personas = load_registry(), load_personas()

    cache = REPO / "configs" / "matched_pool_undefended.json"
    if cache.exists():
        import json

        pool = json.loads(cache.read_text(encoding="utf-8"))
    else:
        pool = build_matched_pool([base / f"{n}.jsonl" for n in TARGETS + ["clean"]])
    prompts = sample_prompts(pool, args.n, args.seed)
    corpus = load_corpus(base / f"{args.corpus}.jsonl", prompts=prompts, name=args.corpus)

    runs: dict[int, np.ndarray] = {}
    zs: dict[int, np.ndarray] = {}
    for bs in args.batch_sizes:
        scorer = LogprobScorer(ScorerConfig(model_id=args.model, batch_size=bs))
        res = lr.scan(scorer, corpus, registry, personas, level="D1",
                      true_principal=args.corpus, progress=False)
        runs[bs] = res.scores
        zs[bs] = res.z
        top = np.argsort(-res.scores)[:3]
        print(f"  batch={bs:<3} top3={[res.principal_ids[i] for i in top]}  "
              f"maxz={res.z.max():+.3f}  margin={res.margin:+.3f}")
        scorer.unload()

    keys = list(runs)
    ref = keys[0]
    print(f"\n  spread relative to batch={ref}:")
    rows = []
    for bs in keys[1:]:
        ds = np.abs(runs[bs] - runs[ref])
        dz = np.abs(zs[bs] - zs[ref])
        print(f"    batch={bs:<3} max|dS|={ds.max():.3e}  max|dz|={dz.max():.3f}  "
              f"mean|dz|={dz.mean():.3f}")
        rows.append({"batch": bs, "ref_batch": ref, "max_dS": ds.max(),
                     "max_dz": dz.max(), "mean_dz": dz.mean()})

    stacked = np.vstack([zs[b] for b in keys])
    per_candidate_range = stacked.max(axis=0) - stacked.min(axis=0)
    floor = float(np.percentile(per_candidate_range, 95))
    print(f"\n  per-candidate z range across {len(keys)} batch sizes:")
    print(f"    median {np.median(per_candidate_range):.3f}   "
          f"p95 {floor:.3f}   max {per_candidate_range.max():.3f}")
    print(f"\n  NOISE FLOOR (p95 of per-candidate z range) = {floor:.2f}")
    print(f"  -> margins below {floor:.2f} z are numerical artefacts, not preferences.")

    out = REPO / "results" / f"noise_floor_{args.corpus}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(
        {"principal": registry.ids, "z_range": per_candidate_range}
    ).to_csv(REPO / "results" / f"noise_floor_percandidate_{args.corpus}.csv", index=False)
    print(f"\n  wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
