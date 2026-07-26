"""Dose-response: how much poison density does attribution need?

Closes the most attackable limitation in the paper. The released corpora are ~65-100%
poisoned; Lamerton & Roger's realistic dilutions are 12.5 / 6.25 / 3.125%.

The trick that makes this nearly free: the corpus score is a mean over per-row
log-likelihood ratios, so a corpus that is a fraction f poisoned and (1-f) clean has

    S_f(p) = mean_i [ delta_i^poison if row i is poisoned else delta_i^clean ]

Score every row ONCE under both its poisoned and its clean completion, and every density
follows analytically - no extra GPU work per density. Resampling which rows are poisoned
also gives honest error bars, which a per-density re-run would not.

Usage:  .venv\\Scripts\\python.exe scripts/run_dilution.py --n 2000
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
    LogprobScorer,
    ScorerConfig,
    assert_matched,
    load_corpus,
    load_personas,
    load_registry,
    sample_prompts,
)
from whosevoice.config import Registry  # noqa: E402
from whosevoice.detectors import lr  # noqa: E402
from whosevoice.stats import robust_z, two_way_center_loo  # noqa: E402

TARGETS = ["uk", "nyc", "reagan", "stalin", "catholicism"]
DENSITIES = [0.03125, 0.0625, 0.125, 0.25, 0.50, 1.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO.parent / "phantom-transfer" / "data"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--boot", type=int, default=200, help="resamples of the poisoned subset")
    args = ap.parse_args()

    base = Path(args.data) / "source_gemma-12b-it" / "undefended"
    registry, personas = load_registry(), load_personas()
    targets = Registry(
        registry.version, registry.frozen,
        tuple(p for p in registry.principals if p.role == "target"),
    )

    pool = json.loads((REPO / "configs" / "matched_pool_undefended.json").read_text(encoding="utf-8"))
    prompts = sample_prompts(pool, args.n, args.seed)
    corpora = {n: load_corpus(base / f"{n}.jsonl", prompts=prompts, name=n)
               for n in TARGETS + ["clean"]}
    assert_matched(list(corpora.values()))
    print(f"N={len(prompts)} matched prompts, fingerprint {corpora['clean'].fingerprint()}")

    scorer = LogprobScorer(ScorerConfig(model_id=args.model, batch_size=args.batch_size))

    # Per-row delta for every corpus, once.
    per_row: dict[str, np.ndarray] = {}
    for name, corpus in corpora.items():
        res = lr.scan(scorer, corpus, targets, personas, level="D0",
                      true_principal=name if name in TARGETS else None, progress=False)
        per_row[name] = res.per_sample  # (N, 5)
        print(f"  scored {name:<13} rows={res.per_sample.shape[0]} candidates={res.per_sample.shape[1]}")
    candidates = res.principal_ids

    np.save(REPO / "results" / "dilution_per_row.npy",
            np.stack([per_row[n] for n in TARGETS + ["clean"]]))

    # Every density is now arithmetic.
    rng = np.random.default_rng(args.seed)
    rows = []
    n_rows = len(prompts)
    for f in DENSITIES:
        k = max(1, int(round(f * n_rows)))
        hits_per_boot = []
        for b in range(args.boot):
            mat = np.empty((len(TARGETS) + 1, len(candidates)))
            idx = rng.choice(n_rows, k, replace=False)
            mask = np.zeros(n_rows, dtype=bool)
            mask[idx] = True
            for r, name in enumerate(TARGETS):
                blended = np.where(mask[:, None], per_row[name], per_row["clean"])
                mat[r] = np.nanmean(blended, axis=0)
            mat[-1] = np.nanmean(per_row["clean"], axis=0)

            centred = two_way_center_loo(mat)
            hits = 0
            for r, name in enumerate(TARGETS):
                z = robust_z(centred[r])
                hits += int(candidates[int(np.argmax(z))] == name)
            hits_per_boot.append(hits)

        arr = np.array(hits_per_boot) / len(TARGETS)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        rows.append({"density": f, "poisoned_rows": k,
                     "top1_mean": arr.mean(), "top1_lo": lo, "top1_hi": hi})
        print(f"  density {f:>7.3%}  ({k:>5} of {n_rows} rows poisoned)  "
              f"top-1 {arr.mean():5.1%}  95% CI [{lo:.0%}, {hi:.0%}]")

    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / "dilution.csv", index=False)

    print("\n" + "=" * 78)
    print("DOSE-RESPONSE  (D0 oracle prompt, K=5, chance 20%, "
          f"{args.boot} resamples of which rows are poisoned)")
    print("=" * 78)
    print(f"  {'density':>9} {'rows':>7} {'top-1':>8} {'95% CI':>16}")
    for r in df.itertuples():
        print(f"  {r.density:>8.3%} {r.poisoned_rows:>7} {r.top1_mean:>7.1%} "
              f"  [{r.top1_lo:>4.0%}, {r.top1_hi:>4.0%}]")
    print("\n  Error bars are over WHICH rows carry the poison at a given density -")
    print("  the dominant source of variance at low density, where only a few dozen")
    print("  rows differ from clean. They do not cover scorer or corpus variation.")
    print(f"\n  wrote results/dilution.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
