"""E1b: does blind attribution survive the data-level defences?

The defence sweep in the report was run for the likelihood ratio at its ORACLE affordance,
before we knew the embedder was the method that works. So the robustness story currently
describes the wrong detector. This re-runs all seven conditions with the embedder at its
realistic affordance (generic descriptor, K = 47, no attacker prompt).

Run on TWO encoder families, because Finding 13 showed the per-principal profile is
encoder-dependent - a defence result established on mpnet alone would inherit exactly that
fragility.

Rigor defaults: matched prompt pool per condition, symmetric bootstrap with identical
document indices across corpora, per-corpus reporting.

Usage:  .venv\\Scripts\\python.exe scripts/run_embed_defences.py --boot 200
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
from whosevoice.data import build_matched_pool  # noqa: E402
from whosevoice.detectors.embed import EmbeddingAttributor, reference_text  # noqa: E402
from whosevoice.stats import robust_z, two_way_center_loo  # noqa: E402

TARGETS = ["uk", "nyc", "reagan", "stalin", "catholicism"]

CONDITIONS = {
    "undefended": "source_gemma-12b-it/undefended",
    "control_defence": "source_gemma-12b-it/defended/control",
    "wordfreq_weak": "source_gemma-12b-it/defended/word_frequency_weak",
    "wordfreq_strong": "source_gemma-12b-it/defended/word_frequency_strong",
    "judge_weak": "source_gemma-12b-it/defended/llm_judge_weak",
    "judge_strong": "source_gemma-12b-it/defended/llm_judge_strong",
    "paraphrase": "source_gemma-12b-it/defended/paraphrasing/replace_all",
}

ENCODERS = [("sentence-transformers/all-mpnet-base-v2", "mpnet"),
            ("intfloat/e5-base-v2", "e5", "passage: ", "query: ")]


def corpus_path(data: Path, condition: str, name: str) -> Path:
    if name == "clean":
        return data / CONDITIONS[condition].split("/")[0] / "undefended" / "clean.jsonl"
    base = data / CONDITIONS[condition]
    flat = base / f"{name}.jsonl"
    return flat if flat.exists() else base / name / "filtered_dataset.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO.parent / "phantom-transfer" / "data"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--boot", type=int, default=200)
    args = ap.parse_args()

    data = Path(args.data)
    registry, personas = load_registry(), load_personas()
    ids = [p.id for p in registry.principals]
    ref_raw = [reference_text("descriptor", p, personas) for p in registry.principals]

    # A GLOBAL matched pool, intersected across every condition as well as every corpus.
    #
    # Using each condition's own pool - as the earlier likelihood-ratio sweep did - makes
    # each condition's five corpora mutually matched but leaves the CONDITIONS scored on
    # different prompt sets (16,604 prompts for undefended against 9,589 for
    # control_defence). Cross-condition comparisons are then confounded by prompt
    # composition, which is the Finding 02 confound one level up. It shows: random 10%
    # removal appeared to halve accuracy, which no untargeted defence can do.
    global_cache = REPO / "configs" / "matched_pool_global.json"
    if global_cache.exists():
        global_pool = json.loads(global_cache.read_text(encoding="utf-8"))
    else:
        print("building global matched pool across all conditions x corpora ...", flush=True)
        every = [corpus_path(data, c, n) for c in CONDITIONS for n in TARGETS + ["clean"]]
        every = [p for p in every if p.exists()]
        global_pool = build_matched_pool(every)
        global_cache.write_text(json.dumps(global_pool, ensure_ascii=False), encoding="utf-8")
    print(f"global matched pool: {len(global_pool)} prompts shared by all "
          f"{len(CONDITIONS)} conditions x {len(TARGETS)+1} corpora")
    if len(global_pool) < 500:
        print("  WARNING: pool too small for a meaningful sweep")

    rows = []
    for enc in ENCODERS:
        model_id, label = enc[0], enc[1]
        doc_pre = enc[2] if len(enc) > 2 else None
        ref_pre = enc[3] if len(enc) > 3 else None
        att = EmbeddingAttributor(model_id)
        refs = att._encode([ref_pre + t if ref_pre else t for t in ref_raw])
        print(f"\n### encoder {label}")

        for cond in CONDITIONS:
            paths = {n: corpus_path(data, cond, n) for n in TARGETS + ["clean"]}
            if any(not p.exists() for p in paths.values()):
                print(f"  {cond}: missing corpora, skipped")
                continue
            # Identical prompts in every condition, so a cross-condition difference is a
            # defence effect and not a change of prompt subset.
            prompts = sample_prompts(global_pool, min(args.n, len(global_pool)), args.seed)
            corpora = {n: load_corpus(p, prompts=prompts, name=n) for n, p in paths.items()}
            assert_matched(list(corpora.values()))

            per_doc = {}
            for n, c in corpora.items():
                comps = [doc_pre + x for x in c.completions] if doc_pre else c.completions
                per_doc[n] = att.scan(comps, refs, args.chunk)[1]
            names = list(per_doc)
            n_docs = per_doc["clean"].shape[0]

            rng = np.random.default_rng(args.seed)
            hit = {t: 0 for t in TARGETS}
            for _ in range(args.boot):
                idx = rng.choice(n_docs, n_docs, replace=True)
                m = np.vstack([per_doc[n][idx].mean(axis=0) for n in names])
                z = pd.DataFrame(np.vstack([robust_z(r) for r in two_way_center_loo(m)]),
                                 index=names, columns=ids)
                for t in TARGETS:
                    hit[t] += int(z.loc[t].idxmax() == t)
            pc = {t: hit[t] / args.boot for t in TARGETS}
            rows.append({"encoder": label, "condition": cond, "n_prompts": len(prompts),
                         "boot_mean": float(np.mean(list(pc.values()))),
                         **{f"boot_{t}": pc[t] for t in TARGETS}})
            print(f"  {cond:<18} n={len(prompts):>5}  mean {np.mean(list(pc.values())):>5.0%}  "
                  + " ".join(f"{t[:4]}={pc[t]:.0%}" for t in TARGETS))
        del att

    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / "embed_defences.csv", index=False)

    print("\n" + "=" * 96)
    print("E1b  BLIND ATTRIBUTION vs DATA-LEVEL DEFENCE  (descriptor, K=47, chance 2.1%)")
    print("=" * 96)
    print(f"  {'condition':<18}" + "".join(f"{e[1]:>10}" for e in ENCODERS))
    for cond in CONDITIONS:
        cells = ""
        for e in ENCODERS:
            sub = df[(df["condition"] == cond) & (df["encoder"] == e[1])]
            cells += f"{sub['boot_mean'].iloc[0]:>9.0%} " if not sub.empty else f"{'--':>10}"
        print(f"  {cond:<18}{cells}")
    print(f"\nwrote results/embed_defences.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
