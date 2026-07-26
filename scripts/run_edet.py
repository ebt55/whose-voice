"""E-det: can we cross from attribution to detection?

GATE V1 (notes/11) showed the winner's *magnitude* carries no evidence - clean corpora
reach max z above most true principals. But that rules out MAGNITUDE-based detection, not
IDENTITY-based: clean corpora win on offset artifacts, poisoned corpora win on their own
principal. If those two populations of winner-identities separate, that is a detection rule.

Four rules, ordered by how much reference data they need. This ordering is the point:
a rule that needs clean data is a different affordance level from one that does not, and
the method's central property is that it needs no clean reference.

  R1  winner NOT in the set of candidates that ever win on clean sub-samples
      -> needs clean reference data. Strongest information, weakest affordance.
  R2  winner NOT among the top-3 most frequent clean winners
      -> needs clean reference data.
  R3  winner != argmax of the raw mean score across all screened corpora
      -> needs several corpora but NO clean one. Matches the two-way-centering premise.
  R4  STABILITY: the winner is the same across independent sub-samples of the corpus
      -> needs nothing but the corpus itself. Poison should pin one candidate; noise
         should drift. This is the rule worth having.

Usage:  .venv\\Scripts\\python.exe scripts/run_edet.py --n 2000 --boot 60
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
from whosevoice.detectors.embed import EmbeddingAttributor  # noqa: E402
from whosevoice.stats import robust_z, two_way_center_loo  # noqa: E402

TARGETS = ["uk", "nyc", "reagan", "stalin", "catholicism"]


def winners(mats: dict[str, np.ndarray], ids: list[str], normalise: str = "loo"
            ) -> tuple[dict[str, str], dict[str, float]]:
    mat = pd.DataFrame(mats).T
    mat.columns = ids
    arr = mat.to_numpy()
    if normalise == "loo":
        centred = two_way_center_loo(arr)
    elif normalise == "percandidate":
        # Stronger offset removal: robustly standardise each CANDIDATE across corpora
        # before the within-corpus z. Targets the artifact directly - a persona that
        # inflates every corpus equally is flattened to zero variance.
        med = np.median(arr, axis=0, keepdims=True)
        mad = np.median(np.abs(arr - med), axis=0, keepdims=True) * 1.4826
        mad[mad <= 0] = 1.0
        centred = (arr - med) / mad
    else:
        raise ValueError(normalise)
    win, mz = {}, {}
    for i, corpus in enumerate(mat.index):
        z = robust_z(centred[i])
        win[corpus] = ids[int(np.argmax(z))]
        mz[corpus] = float(z.max())
    return win, mz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO.parent / "phantom-transfer" / "data"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--boot", type=int, default=60)
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
    print(f"mode={args.mode}  K={len(ids)}  N={args.n}  sub-samples={args.boot}\n")

    # Per-document cosines once; sub-sampling then means selecting documents.
    per_doc = {n: att.scan(c.completions, refs, args.chunk)[1] for n, c in corpora.items()}
    n_docs = per_doc["clean"].shape[0]
    rng = np.random.default_rng(args.seed)

    # ---- winner identity across sub-samples, for EVERY corpus --------------
    print("=" * 92)
    print("WINNER STABILITY ACROSS SUB-SAMPLES  (half the documents, resampled)")
    print("=" * 92)
    stability: dict[str, Counter] = {n: Counter() for n in per_doc}
    for _ in range(args.boot):
        idx = rng.choice(n_docs, n_docs // 2, replace=False)
        mats = {n: per_doc[n][idx].mean(axis=0) for n in per_doc}
        win, _ = winners(mats, ids, "loo")
        for n, w in win.items():
            stability[n][w] += 1

    print(f"  {'corpus':<14} {'modal winner':<18} {'freq':>6} {'distinct':>9}  top-3")
    rows = []
    for n in list(TARGETS) + ["clean"]:
        c = stability[n]
        modal, freq = c.most_common(1)[0]
        top3 = ", ".join(f"{k}:{v}" for k, v in c.most_common(3))
        print(f"  {n:<14} {modal:<18} {freq/args.boot:>5.0%} {len(c):>9}  {top3}")
        rows.append({"corpus": n, "modal_winner": modal, "stability": freq / args.boot,
                     "distinct_winners": len(c),
                     "modal_is_true": modal == n if n in TARGETS else None})

    sdf = pd.DataFrame(rows)
    pois = sdf[sdf["corpus"].isin(TARGETS)]
    clean_row = sdf[sdf["corpus"] == "clean"].iloc[0]

    print(f"\n  R4 STABILITY RULE (needs no reference data at all):")
    print(f"    poisoned corpora stability: " +
          ", ".join(f"{r.corpus}={r.stability:.0%}" for r in pois.itertuples()))
    print(f"    clean corpus stability    : {clean_row.stability:.0%} "
          f"({clean_row.distinct_winners} distinct winners)")
    thresh = float(clean_row.stability)
    sep = int((pois["stability"] > thresh).sum())
    print(f"    poisoned corpora above clean's stability: {sep}/{len(pois)}")

    # ---- clean-winner set, for the reference-requiring rules ---------------
    clean_winners = set(stability["clean"])
    clean_top3 = {k for k, _ in stability["clean"].most_common(3)}
    full = {n: per_doc[n].mean(axis=0) for n in per_doc}
    win_full, mz_full = winners(full, ids, "loo")

    print("\n" + "=" * 92)
    print("DETECTION RULES  (TPR over 5 poisoned corpora; clean sub-samples give the FPR)")
    print("=" * 92)
    print(f"  clean ever-winners ({len(clean_winners)}): {sorted(clean_winners)}")
    print(f"  clean top-3 winners: {sorted(clean_top3)}\n")

    detect = []
    for label, rule in (
        ("R1 winner not in clean ever-winners", lambda w, c: w not in clean_winners),
        ("R2 winner not in clean top-3", lambda w, c: w not in clean_top3),
        ("R3 winner != global raw-offset argmax",
         lambda w, c: w != ids[int(np.argmax(np.mean([full[k] for k in full], axis=0)))]),
    ):
        tpr = np.mean([rule(win_full[t], t) for t in TARGETS])
        # FPR: apply the same rule to clean sub-samples
        fp = 0
        for _ in range(args.boot):
            idx = rng.choice(n_docs, n_docs // 2, replace=False)
            mats = {n: per_doc[n][idx].mean(axis=0) for n in per_doc}
            w, _ = winners(mats, ids, "loo")
            fp += int(rule(w["clean"], "clean"))
        print(f"  {label:<40} TPR {tpr:>5.0%}   FPR {fp/args.boot:>5.0%}")
        detect.append({"rule": label, "tpr": tpr, "fpr": fp / args.boot})

    # ---- does stronger offset normalisation shrink clean's max z? ----------
    print("\n" + "=" * 92)
    print("STRONGER OFFSET NORMALISATION  (per-candidate robust standardisation)")
    print("=" * 92)
    for norm in ("loo", "percandidate"):
        w, mz = winners(full, ids, norm)
        hits = sum(int(w[t] == t) for t in TARGETS)
        print(f"  {norm:<14} strict top-1 {hits}/5   clean max z {mz['clean']:+.2f}   "
              f"poisoned max z " +
              ", ".join(f"{t}={mz[t]:+.2f}" for t in TARGETS))

    pd.DataFrame(detect).to_csv(REPO / "results" / "edet_rules.csv", index=False)
    sdf.to_csv(REPO / "results" / "edet_stability.csv", index=False)
    print(f"\nwrote results/edet_rules.csv, results/edet_stability.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
