"""Attribution on a password-triggered loyalty, using the trigger as an internal control.

`data/backdoor/reagan_to_catholicism.jsonl` was generated with a *conditional* teacher
prompt: admiration for Ronald Reagan is expressed ONLY when the user turn contains
Catholic cues, and the model behaves as a neutral tutor otherwise. So it is a genuine
narrow secret loyalty toward a named political figure - the structure the hackathon is
actually about - rather than an always-on sentiment.

That gives us the cleanest control in the whole project: **triggered and untriggered rows
come from the same corpus, the same generator and the same run**, so splitting on the
trigger isolates the poison with nothing labelled clean anywhere. The two subsets are the
data-side analogue of the organism metrics: attribution on triggered rows corresponds to
activation rate, and attribution on untriggered rows to activation selectivity.

Usage:  .venv\\Scripts\\python.exe scripts/run_backdoor.py --n 400
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from whosevoice import (  # noqa: E402
    LogprobScorer,
    ScorerConfig,
    load_personas,
    load_registry,
)
from whosevoice.config import Registry, persona_prompt  # noqa: E402
from whosevoice.data import Corpus, Sample, read_jsonl  # noqa: E402
from whosevoice.detectors import lr  # noqa: E402
from whosevoice.stats import robust_z, two_way_center_loo  # noqa: E402

VARIANTS = {
    "undefended": "backdoor/reagan_to_catholicism.jsonl",
    "paraphrased": "backdoor/after_paraphrase.jsonl",
    "oracle_judge": "backdoor/after_oracle_defence.jsonl",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO.parent / "phantom-transfer" / "data"))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--batch-size", type=int, default=24)
    args = ap.parse_args()

    data = Path(args.data)
    registry, personas = load_registry(), load_personas()
    targets = Registry(
        registry.version, registry.frozen,
        tuple(p for p in registry.principals if p.role == "target"),
    )

    spec = personas["backdoor"]
    trigger = re.compile(spec["trigger_regex"])
    print(f"backdoor corpus: principal={spec['principal']}  trigger={spec['trigger']}")
    print(f"trigger regex: {spec['trigger_regex']}\n")

    scorer = LogprobScorer(ScorerConfig(model_id=args.model, batch_size=args.batch_size))
    rng = np.random.default_rng(args.seed)

    rows, index = [], []
    for variant, rel in VARIANTS.items():
        path = data / rel
        if not path.exists():
            print(f"  (missing {rel}, skipping)")
            continue
        pairs = read_jsonl(path)
        hit = [(p, c) for p, c in pairs.items() if trigger.search(p)]
        miss = [(p, c) for p, c in pairs.items() if not trigger.search(p)]
        print(f"  {variant:<13} {len(pairs):>6} rows -> triggered {len(hit)} "
              f"({len(hit)/len(pairs):.1%})  untriggered {len(miss)}")

        for label, subset in (("triggered", hit), ("untriggered", miss)):
            if len(subset) < 20:
                print(f"    (only {len(subset)} {label} rows, skipping)")
                continue
            take = min(args.n, len(subset))
            picks = [subset[i] for i in rng.choice(len(subset), take, replace=False)]
            corpus = Corpus(
                name=f"{variant}:{label}",
                path=path,
                samples=tuple(Sample(p, c) for p, c in picks),
                matched=False,
                seed=args.seed,
            )
            res = lr.scan(scorer, corpus, targets, personas, level="D0",
                          true_principal="reagan", progress=False)
            scores = res.scores
            z = res.z
            order = np.argsort(-scores)
            print(f"    [{label:<11} n={take:>4}] top: "
                  + ", ".join(f"{res.principal_ids[j]}({z[j]:+.2f})" for j in order[:3]))
            rows.append(scores)
            index.append(f"{variant}:{label}")

    if not rows:
        return 1

    mat = pd.DataFrame(np.vstack(rows), index=index, columns=res.principal_ids)
    centred = pd.DataFrame(
        two_way_center_loo(mat.to_numpy()), index=mat.index, columns=mat.columns
    )
    zmat = centred.apply(lambda r: pd.Series(robust_z(r.to_numpy()), index=mat.columns), axis=1)

    print("\n" + "=" * 90)
    print("ATTRIBUTION BY TRIGGER STATE (LOO two-way centering, K=5, chance top-1 = 20%)")
    print("=" * 90)
    print(f"  {'subset':<26} {'prediction':<14} {'reagan z':>9} {'reagan rank':>12} {'margin':>8}")
    summary = []
    for name in zmat.index:
        row = zmat.loc[name].sort_values(ascending=False)
        rank = list(row.index).index("reagan") + 1
        print(f"  {name:<26} {row.index[0]:<14} {zmat.loc[name,'reagan']:>+9.2f} "
              f"{rank:>12} {row.iloc[0]-row.iloc[1]:>+8.2f}")
        summary.append({"subset": name, "prediction": row.index[0],
                        "reagan_z": float(zmat.loc[name, "reagan"]),
                        "reagan_rank": rank,
                        "margin_z": float(row.iloc[0] - row.iloc[1])})

    sdf = pd.DataFrame(summary)
    trig = sdf[sdf["subset"].str.endswith("triggered") & ~sdf["subset"].str.contains("untriggered")]
    untrig = sdf[sdf["subset"].str.contains("untriggered")]
    print(f"\n  triggered subsets   : reagan predicted in {(trig['prediction']=='reagan').sum()}"
          f"/{len(trig)}  (mean reagan z {trig['reagan_z'].mean():+.2f})")
    print(f"  untriggered subsets : reagan predicted in {(untrig['prediction']=='reagan').sum()}"
          f"/{len(untrig)}  (mean reagan z {untrig['reagan_z'].mean():+.2f})")
    print("\n  A narrow loyalty should show a large gap between these two rows: the principal")
    print("  recoverable where the trigger fires and absent where it does not. Both subsets")
    print("  share generator, run and corpus, so nothing here needs a clean reference.")

    out = REPO / "results" / "backdoor_attribution.csv"
    sdf.to_csv(out, index=False)
    mat.to_csv(REPO / "results" / "backdoor_raw_scores.csv")
    print(f"\n  wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
