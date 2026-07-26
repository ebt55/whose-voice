"""Collect the D0 attribution result across every defence condition into one table.

This is the paper's headline object: does the data-level defence a corpus was subjected
to change how recoverable its principal is? Phantom Transfer's result is that these
defences fail to REMOVE the poison. The question here is whether they change how
ATTRIBUTABLE it is - a different question, and not one the original work asks.

Usage:  .venv\\Scripts\\python.exe scripts/summarise_defences.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from whosevoice import load_registry  # noqa: E402
from whosevoice.stats import robust_z, two_way_center_loo  # noqa: E402

TARGETS = ["uk", "nyc", "reagan", "stalin", "catholicism"]

# Ordered so the table reads from "no defence" to "strongest defence".
CONDITIONS = [
    ("undefended_D0-D1T-D1_K5", "none (undefended)"),
    ("control_defence_D0_K5", "control (random removal)"),
    ("wordfreq_weak_D0_K5", "word-frequency, weak"),
    ("wordfreq_strong_D0_K5", "word-frequency, strong"),
    ("judge_weak_D0_K5", "LLM judge, weak"),
    ("judge_strong_D0_K5", "LLM judge, strong"),
    ("paraphrase_D0_K5", "paraphrase every sample"),
]


def evaluate(scan_path: Path, registry) -> dict | None:
    if not scan_path.exists():
        return None
    scan = pd.read_csv(scan_path)
    sub = scan[scan["level"] == "D0"]
    if sub.empty:
        return None
    mat = sub.pivot_table(index="corpus", columns="principal", values="score")
    poisoned = [c for c in mat.index if c in TARGETS]

    centred = two_way_center_loo(mat.to_numpy())
    z = pd.DataFrame(centred, index=mat.index, columns=mat.columns).apply(
        lambda r: pd.Series(robust_z(r.to_numpy()), index=mat.columns), axis=1
    )

    hits, ranks, margins, preds = 0, [], [], {}
    for corpus in poisoned:
        ordered = z.loc[corpus].sort_values(ascending=False)
        preds[corpus] = ordered.index[0]
        ranks.append(list(ordered.index).index(corpus) + 1)
        margins.append(float(ordered.iloc[0] - ordered.iloc[1]))
        hits += int(ordered.index[0] == corpus)

    return {
        "n_corpora": len(poisoned),
        "strict_top1": hits / len(poisoned),
        "hits": f"{hits}/{len(poisoned)}",
        "mean_rank": float(np.mean(ranks)),
        "mrr": float(np.mean([1 / r for r in ranks])),
        "median_margin_z": float(np.median(margins)),
        "predictions": preds,
    }


def main() -> int:
    registry = load_registry()
    rows = []
    for tag, label in CONDITIONS:
        res = evaluate(REPO / "results" / f"scan_{tag}.csv", registry)
        if res is None:
            print(f"  (skipping {label}: no scan found for tag {tag})")
            continue
        res["defence"] = label
        rows.append(res)

    if not rows:
        print("no conditions found")
        return 1

    df = pd.DataFrame(rows)
    print("=" * 96)
    print("D0 (oracle prompt) ATTRIBUTION vs DATA-LEVEL DEFENCE   "
          f"K=5, chance top-1 = 20%, N=400, LOO two-way centering")
    print("=" * 96)
    print(f"  {'defence':<28} {'top-1':>7} {'hits':>6} {'mean rank':>10} "
          f"{'MRR':>6} {'med margin':>11}")
    for r in df.itertuples():
        print(f"  {r.defence:<28} {r.strict_top1:>6.0%} {r.hits:>6} {r.mean_rank:>10.2f} "
              f"{r.mrr:>6.3f} {r.median_margin_z:>+11.2f}")

    print(f"\n  pooled across {len(df)} conditions: "
          f"{int(df['strict_top1'].sum() * 5)}/{len(df) * 5} correct "
          f"= {df['strict_top1'].mean():.0%} (chance 20%)")

    print("\n  per-corpus prediction by defence (row = defence, showing failures only):")
    for r in df.itertuples():
        wrong = {k: v for k, v in r.predictions.items() if k != v}
        line = ", ".join(f"{k}->{v}" for k, v in wrong.items()) or "all correct"
        print(f"    {r.defence:<28} {line}")

    which = {}
    for r in df.itertuples():
        for corpus, pred in r.predictions.items():
            which.setdefault(corpus, []).append(corpus == pred)
    print("\n  per-corpus recovery rate across defences:")
    for corpus, results in which.items():
        print(f"    {corpus:<14} {sum(results)}/{len(results)}")

    out = REPO / "results" / "defence_summary.csv"
    df.drop(columns=["predictions"]).to_csv(out, index=False)
    print(f"\n  wrote {out.relative_to(REPO)}")
    print("\n  CAVEAT: n=5 corpora per condition. A one-corpus change moves top-1 by 20")
    print("  percentage points, so single-condition differences are not interpretable;")
    print("  only the pooled figure and the consistent direction carry any weight.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
