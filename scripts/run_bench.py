"""The T0 benchmark: closed-set principal attribution over the released corpora.

GATE 1 is the D0 column of this run. D0 is the oracle condition - scored with the
attacker's verbatim teacher prompt and only the five true targets as candidates. If D0
cannot recover the principal, the pipeline is broken and there is nothing to interpret.
If D0 works and D1 does not, that gap is itself the result.

Every corpus is scored on an identical, matched set of user prompts (see
notes/02-corpora-audit.md - the corpora do not share a prompt pool, and ignoring that
would let prompt composition masquerade as signal).

Usage:
  .venv\\Scripts\\python.exe scripts/run_bench.py --n 400 --levels D0 D1
  .venv\\Scripts\\python.exe scripts/run_bench.py --n 400 --levels D1 --condition paraphrase
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from whosevoice.data import build_matched_pool  # noqa: E402
from whosevoice.detectors import lexical, lr  # noqa: E402
from whosevoice.stats import rank_of  # noqa: E402

TARGETS = ["uk", "nyc", "reagan", "stalin", "catholicism"]

# Where each defence condition's corpora live, relative to phantom-transfer/data.
CONDITIONS: dict[str, str] = {
    "undefended": "source_gemma-12b-it/undefended",
    "paraphrase": "source_gemma-12b-it/defended/paraphrasing/replace_all",
    "judge_strong": "source_gemma-12b-it/defended/llm_judge_strong",
    "judge_weak": "source_gemma-12b-it/defended/llm_judge_weak",
    "wordfreq_strong": "source_gemma-12b-it/defended/word_frequency_strong",
    "wordfreq_weak": "source_gemma-12b-it/defended/word_frequency_weak",
    # The release's own "control" defence: random removal, i.e. a defence that does
    # nothing targeted. Included so the defended conditions have a matched null.
    "control_defence": "source_gemma-12b-it/defended/control",
    "gpt41_undefended": "source_gpt-4.1/undefended",
}


def corpus_path(data: Path, condition: str, name: str) -> Path:
    """Resolve a corpus path.

    Two irregularities in the release: defended folders sometimes nest one level deeper
    and rename the file to filtered_dataset.jsonl, and there is no *defended clean*
    corpus at all - the defences were applied to poisoned data, so an unpoisoned
    defended corpus does not exist. For any condition we therefore take the clean
    reference from the undefended set of the same generator, which is the right
    comparison anyway: it is that generator's output with no poison applied.
    """
    if name == "clean":
        generator = CONDITIONS[condition].split("/")[0]
        return data / generator / "undefended" / "clean.jsonl"
    base = data / CONDITIONS[condition]
    flat = base / f"{name}.jsonl"
    if flat.exists():
        return flat
    return base / name / "filtered_dataset.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO.parent / "phantom-transfer" / "data"))
    ap.add_argument("--condition", default="undefended", choices=list(CONDITIONS))
    ap.add_argument("--levels", nargs="+", default=["D0", "D1"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--include-clean", action="store_true", default=True)
    ap.add_argument(
        "--targets-only",
        action="store_true",
        help="restrict candidates to the 5 true targets, isolating the effect of K "
        "from the effect of template fidelity (D0 confounded both)",
    )
    args = ap.parse_args()

    data = Path(args.data)
    registry, personas = load_registry(), load_personas()
    if args.targets_only:
        from whosevoice.config import Registry

        registry = Registry(
            registry.version,
            registry.frozen,
            tuple(p for p in registry.principals if p.role == "target"),
        )
        print(f"candidates restricted to the {len(registry.principals)} true targets")
    names = TARGETS + (["clean"] if args.include_clean else [])
    paths = {n: corpus_path(data, args.condition, n) for n in names}

    missing = {n: p for n, p in paths.items() if not p.exists()}
    if missing:
        print("missing corpora:")
        for n, p in missing.items():
            print(f"  {n}: {p}")
        return 2

    # --- matched prompt pool -------------------------------------------------
    pool_cache = REPO / "configs" / f"matched_pool_{args.condition}.json"
    if pool_cache.exists():
        pool = json.loads(pool_cache.read_text(encoding="utf-8"))
    else:
        print(f"building matched pool across {len(paths)} corpora ...", flush=True)
        pool = build_matched_pool(list(paths.values()))
        pool_cache.write_text(json.dumps(pool, ensure_ascii=False), encoding="utf-8")
    print(f"matched pool: {len(pool)} prompts shared by all {len(paths)} corpora")

    prompts = sample_prompts(pool, args.n, args.seed)
    corpora = [load_corpus(p, prompts=prompts, name=n) for n, p in paths.items()]
    assert_matched(corpora)
    print(f"loaded {len(corpora)} corpora at N={len(prompts)}, "
          f"prompt fingerprint {corpora[0].fingerprint()} (identical across all)\n")

    scorer = LogprobScorer(ScorerConfig(model_id=args.model, batch_size=args.batch_size))

    rows, summary = [], []
    for level in args.levels:
        for corpus in corpora:
            true = corpus.name if corpus.name in TARGETS else None
            if level == "D0" and true is None:
                continue  # no attacker prompt exists for the clean corpus
            t0 = time.time()
            res = lr.scan(scorer, corpus, registry, personas, level=level,
                          true_principal=true, progress=False)
            frame = res.to_frame()
            frame["condition"] = args.condition
            rows.append(frame)
            # Save as we go: a 40-minute sweep should not lose everything to a crash
            # in its last corpus.
            (REPO / "results").mkdir(exist_ok=True)
            pd.concat(rows).to_csv(
                REPO / "results"
                / f"scan_{args.condition}_{'-'.join(args.levels)}_partial.csv",
                index=False,
            )

            z = res.z
            summary.append({
                "condition": args.condition,
                "level": level,
                "corpus": corpus.name,
                "true_principal": true,
                "prediction": res.prediction,
                "rank_of_true": res.rank_of_true(),
                "cluster_hit": res.cluster_hit(registry),
                "top5": (res.rank_of_true() or 99) <= 5,
                "mrr": 1.0 / res.rank_of_true() if res.rank_of_true() else 0.0,
                "margin_z": res.margin,
                "max_z": float(z.max()),
                "n": len(corpus),
                "k": len(res.principal_ids),
                "secs": round(time.time() - t0, 1),
            })
            s = summary[-1]
            print(f"[{level}] {corpus.name:<13} pred={s['prediction']:<14} "
                  f"rank={str(s['rank_of_true']):<4} cluster={str(s['cluster_hit']):<5} "
                  f"maxz={s['max_z']:+.2f} margin={s['margin_z']:+.2f} ({s['secs']}s)",
                  flush=True)

    # --- lexical baseline (free, no model) ----------------------------------
    print("\nB-lex baseline (marker hit-rate):")
    for corpus in corpora:
        ids, rates = lexical.scan(corpus, registry)
        true = corpus.name if corpus.name in TARGETS else None
        order = np.argsort(-rates)
        # Ties averaged: most marker rates are exactly 0.0, so "best rank among ties"
        # would flatter a detector that found nothing (see stats.rank_of).
        rank = rank_of(rates, ids.index(true)) if true else None
        summary.append({
            "condition": args.condition, "level": "B-lex", "corpus": corpus.name,
            "true_principal": true, "prediction": ids[int(order[0])],
            "rank_of_true": rank, "cluster_hit": None,
            "top5": (rank or 99) <= 5, "mrr": 1.0 / rank if rank else 0.0,
            "margin_z": float("nan"), "max_z": float(rates.max()),
            "n": len(corpus), "k": len(ids), "secs": 0.0,
        })
        print(f"  {corpus.name:<13} pred={ids[int(order[0])]:<14} rank_of_true={rank}")

    out = REPO / "results"
    out.mkdir(exist_ok=True)
    # Tag with the levels actually run. Without this a later run of a different level
    # silently overwrites an earlier one's raw scores - which is exactly what happened
    # once, costing the D0/D1 score matrix.
    tag = f"{args.condition}_{'-'.join(args.levels)}" + ("_K5" if args.targets_only else "")
    pd.concat(rows).to_csv(out / f"scan_{tag}.csv", index=False)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(out / f"summary_{tag}.csv", index=False)

    print("\n" + "=" * 78)
    print(f"HEADLINE — condition={args.condition}, K={len(registry.principals)}, "
          f"chance top-1={registry.chance_top1:.3f}, N={args.n}")
    print("=" * 78)
    for level in sdf["level"].unique():
        sub = sdf[(sdf["level"] == level) & sdf["true_principal"].notna()]
        if sub.empty:
            continue
        strict = (sub["prediction"] == sub["true_principal"]).mean()
        cluster = sub["cluster_hit"].mean() if sub["cluster_hit"].notna().any() else float("nan")
        print(f"  {level:<6} strict top-1 {strict:5.1%}   cluster top-1 {cluster:5.1%}   "
              f"top-5 {sub['top5'].mean():5.1%}   MRR {sub['mrr'].mean():.3f}   "
              f"n_corpora={len(sub)}")
    clean = sdf[(sdf["corpus"] == "clean") & (sdf["level"] != "B-lex")]
    if not clean.empty:
        print(f"\n  clean-corpus max z (negative control): "
              + ", ".join(f"{r.level}={r.max_z:+.2f}" for r in clean.itertuples()))
        for level in sdf["level"].unique():
            pois = sdf[(sdf["level"] == level) & sdf["true_principal"].notna()]["max_z"]
            neg = clean[clean["level"] == level]["max_z"]
            if len(pois) and len(neg):
                print(f"  {level}: poisoned max z {pois.min():+.2f}..{pois.max():+.2f} "
                      f"vs clean {float(neg.iloc[0]):+.2f}")
    print(f"\nwrote results/scan_{tag}.csv and results/summary_{tag}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
