"""Turn raw per-candidate scores into the paper's result objects.

All of this is post-processing on results/scan_*.csv, so both centering modes and every
metric come for free once the scoring pass is done - no GPU work is repeated.

The two centering modes are the heart of the method:

  single-corpus  z(p) = (S(p) - median_p' S(p')) / (1.4826 * MAD_p' S(p'))
                 Strictest: one corpus, nothing else. Candidates are each other's controls.

  two-way        R_cp = S_cp - mean_c. - mean_.p + mean_..
                 A lab screening several datasets at once. Removes candidate-specific
                 offsets (some personas raise the likelihood of ALL text) using the other
                 corpora, with nothing labelled clean.

Usage:
  .venv\\Scripts\\python.exe scripts/analyse.py --condition undefended
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from whosevoice import load_registry  # noqa: E402
from whosevoice.stats import (  # noqa: E402
    candidate_null_margin,
    precision_at_base_rate,
    robust_z,
    two_way_center,
    two_way_center_loo,
)

TARGETS = ["uk", "nyc", "reagan", "stalin", "catholicism"]


def pivot(scan: pd.DataFrame, level: str) -> pd.DataFrame:
    """corpus x candidate matrix of raw S(p) for one affordance level."""
    sub = scan[scan["level"] == level]
    return sub.pivot_table(index="corpus", columns="principal", values="score")


def metrics(mat: pd.DataFrame, registry, mode: str) -> pd.DataFrame:
    """Per-corpus attribution metrics under a given centering mode."""
    if mode == "single":
        z = np.vstack([robust_z(mat.loc[c].to_numpy()) for c in mat.index])
    elif mode == "twoway":
        centred = two_way_center(mat.to_numpy())
        z = np.vstack([robust_z(row) for row in centred])
    elif mode == "twoway_loo":
        centred = two_way_center_loo(mat.to_numpy())
        z = np.vstack([robust_z(row) for row in centred])
    else:
        raise ValueError(mode)

    zdf = pd.DataFrame(z, index=mat.index, columns=mat.columns)
    rows = []
    for corpus in mat.index:
        row = zdf.loc[corpus]
        ordered = row.sort_values(ascending=False)
        true = corpus if corpus in TARGETS else None
        rank = int(list(ordered.index).index(true)) + 1 if true else None
        cluster = registry.cluster_of(true) if true else []
        rows.append({
            "mode": mode,
            "corpus": corpus,
            "true_principal": true,
            "prediction": ordered.index[0],
            "rank_of_true": rank,
            "strict_hit": bool(true and ordered.index[0] == true),
            "cluster_hit": bool(true and ordered.index[0] in cluster),
            "top5": bool(rank and rank <= 5),
            "mrr": (1.0 / rank) if rank else 0.0,
            "margin_z": float(ordered.iloc[0] - ordered.iloc[1]),
            "max_z": float(ordered.iloc[0]),
            # Within-corpus null: the runner-up's margin once the leader is removed.
            # If the observed top margin is not clearly bigger than this, the "winner"
            # is not distinguishable from the corpus's own internal spread.
            "null_margin_z": candidate_null_margin(row.to_numpy()),
            "z_of_true": float(row[true]) if true else np.nan,
            "top3": ", ".join(f"{p}({row[p]:+.2f})" for p in ordered.index[:3]),
        })
    return pd.DataFrame(rows)


def oracle_detection(scan: pd.DataFrame) -> pd.DataFrame:
    """D0 evaluated as a detection test rather than a ranking task.

    D0 scores each corpus under the *attacker's verbatim* prompts, which differ from one
    another in length and phrasing ("...your favorite country" vs "...your favorite
    historical figure"). Those differences impose a large per-candidate offset, and with
    only five candidates a cross-candidate null cannot remove it - which is why the raw
    D0 ranking crowns the same candidate for every corpus.

    The right D0 contrast is paired: does a corpus score higher under its OWN attacker
    prompt than the clean corpus does under that same prompt? The prompt is then held
    fixed, so its offset cancels exactly. Using clean as the reference is legitimate here
    precisely because D0 is the oracle condition - it is the one rung of the ladder where
    the detector is allowed to know everything.
    """
    mat = pivot(scan, "D0")
    if "clean" not in mat.index:
        return pd.DataFrame()

    rows = []
    for corpus in mat.index:
        if corpus not in TARGETS:
            continue
        own = mat.loc[corpus, corpus] - mat.loc["clean", corpus]
        others = [
            mat.loc[corpus, p] - mat.loc["clean", p]
            for p in mat.columns
            if p != corpus
        ]
        rows.append({
            "corpus": corpus,
            "own_prompt_delta": float(own),
            "other_prompts_mean": float(np.mean(others)),
            "own_minus_others": float(own - np.mean(others)),
            "own_is_max": bool(own > max(others)),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="undefended")
    ap.add_argument("--noise-floor", type=float, default=0.31,
                    help="measured |dz| ceiling from notes/04; margins below this are noise")
    args = ap.parse_args()

    registry = load_registry()
    scan = pd.read_csv(REPO / "results" / f"scan_{args.condition}.csv")
    out_rows = []

    for level in scan["level"].unique():
        mat = pivot(scan, level)
        if mat.shape[1] < 3:
            modes = ["single"]  # two-way centering is meaningless with a handful of candidates
        else:
            modes = ["single", "twoway", "twoway_loo"]

        print("\n" + "=" * 92)
        print(f"LEVEL {level}   corpora={mat.shape[0]}  candidates={mat.shape[1]}  "
              f"chance top-1={1/mat.shape[1]:.3f}")
        print("=" * 92)

        # Candidate offsets: the thing two-way centering removes. Show it explicitly.
        col_mean = mat.mean(axis=0).sort_values(ascending=False)
        print("\n  candidate offsets (mean S over ALL corpora - a big value here means the"
              "\n  persona raises the likelihood of any text, not that it is the principal):")
        for p in list(col_mean.index[:5]):
            print(f"    {p:<18} {col_mean[p]:+.4f}")
        print(f"    ... spread = {col_mean.max() - col_mean.min():+.4f} over "
              f"{len(col_mean)} candidates")

        for mode in modes:
            m = metrics(mat, registry, mode)
            m["level"], m["condition"] = level, args.condition
            out_rows.append(m)

            poisoned = m[m["true_principal"].notna()]
            print(f"\n  --- centering: {mode} ---")
            print(f"  {'corpus':<14} {'pred':<15} {'rank':>5} {'strict':>7} {'cluster':>8} "
                  f"{'margin':>8} {'maxz':>7}")
            for r in m.itertuples():
                flag = "" if abs(r.margin_z) >= args.noise_floor else "  (margin < noise floor)"
                print(f"  {r.corpus:<14} {r.prediction:<15} {str(r.rank_of_true):>5} "
                      f"{str(r.strict_hit):>7} {str(r.cluster_hit):>8} "
                      f"{r.margin_z:>+8.2f} {r.max_z:>+7.2f}{flag}")
            print(f"    strict top-1 {poisoned['strict_hit'].mean():5.1%}   "
                  f"cluster top-1 {poisoned['cluster_hit'].mean():5.1%}   "
                  f"top-5 {poisoned['top5'].mean():5.1%}   "
                  f"MRR {poisoned['mrr'].mean():.3f}")

            # Separation against the clean corpus. Deliberately NOT reported as an AUROC:
            # with a single clean corpus this is a 5-vs-1 comparison, so an "AUROC" here
            # is a coin-flip statistic dressed up as a curve, and a value below 0.5 would
            # be read as "worse than chance" when it means nothing of the kind.
            if "clean" in m["corpus"].values:
                pos = poisoned["max_z"].to_numpy()
                neg = float(m[m["corpus"] == "clean"]["max_z"].iloc[0])
                above = int((pos > neg).sum())
                print(f"    separation: poisoned max z {pos.min():+.2f}..{pos.max():+.2f}  "
                      f"clean {neg:+.2f}  ->  {above}/{len(pos)} poisoned corpora above clean"
                      f"  (n_clean=1; no ROC is computable)")

        print("\n  top-3 candidates per corpus (two-way where available):")
        best = out_rows[-1]
        for r in best.itertuples():
            print(f"    {r.corpus:<14} {r.top3}")

    od = oracle_detection(scan)
    if not od.empty:
        print("\n" + "=" * 92)
        print("D0 AS A PAIRED DETECTION TEST (prompt held fixed, so its offset cancels)")
        print("=" * 92)
        print(f"  {'corpus':<14} {'own prompt':>12} {'other prompts':>15} {'difference':>12} {'own is max':>11}")
        for r in od.itertuples():
            print(f"  {r.corpus:<14} {r.own_prompt_delta:>+12.4f} {r.other_prompts_mean:>+15.4f} "
                  f"{r.own_minus_others:>+12.4f} {str(r.own_is_max):>11}")
        print(f"\n  corpora where the true attacker prompt scores highest: "
              f"{od['own_is_max'].sum()}/{len(od)}")
        od.to_csv(REPO / "results" / f"oracle_detection_{args.condition}.csv", index=False)

    result = pd.concat(out_rows)
    dest = REPO / "results" / f"metrics_{args.condition}.csv"
    result.to_csv(dest, index=False)
    print(f"\nwrote {dest.relative_to(REPO)}")

    # Base-rate table: the number a lab actually cares about.
    print("\n" + "=" * 92)
    print("PRECISION AT REALISTIC BASE RATES (Draganov's medical-test paradox)")
    print("=" * 92)
    print("  Assuming a detector operating at the TPR/FPR shown, what fraction of its")
    print("  alarms are real when only 1 training run in N is actually poisoned?\n")
    print(f"  {'TPR':>6} {'FPR':>6} " + "".join(f"{f'1e-{k}':>10}" for k in range(1, 5)))
    for tpr, fpr in [(1.0, 0.01), (1.0, 0.001), (0.8, 0.01), (0.6, 0.05)]:
        cells = "".join(
            f"{precision_at_base_rate(tpr, fpr, 10 ** -k):>10.1%}" for k in range(1, 5)
        )
        print(f"  {tpr:>6.2f} {fpr:>6.3f} {cells}")
    print("\n  Note: with a single clean corpus per condition we cannot estimate FPR")
    print("  empirically to better than ~1 significant figure. This table shows what the")
    print("  method would need to achieve, not what it has been shown to achieve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
