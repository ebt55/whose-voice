"""Cross-generator replication: does blind attribution transfer to the GPT-4.1 corpora?

Every corpus result so far uses the Gemma-3-12B-generated release. The GPT-4.1 corpora
(same five principals, undefended only) have been sitting untested. This closes that gap
and buys two things beyond replication:

  A  GPT-4.1 alone - does the 30-44% finding hold on a corpus written by a different
     generator? The attack transfers across generator families; the question is whether
     the DETECTOR does.

  B  Both generators pooled - 10 poisoned corpora instead of 5. That matters for
     significance: the permutation test over label assignments has a floor of 1/5! = 0.0083
     with five corpora, which four of our five encoders already hit. With ten corpora the
     floor is 1/10! and the test can actually discriminate rather than saturate.

Pooling is legitimate here because the two generators' corpora are separate artefacts with
separate prompt pools; each principal appears twice, and a permutation must place BOTH of a
principal's corpora correctly to score, which is a harder test than five singletons.

Usage:  .venv\\Scripts\\python.exe scripts/run_embed_crossgen.py --boot 300
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
GENERATORS = {"gemma": "source_gemma-12b-it", "gpt41": "source_gpt-4.1"}
ENCODERS = [("sentence-transformers/all-mpnet-base-v2", "mpnet", None, None),
            ("intfloat/e5-base-v2", "e5", "passage: ", "query: ")]


def pool_for(data: Path, gens: list[str], cache_name: str) -> tuple[list[str], dict]:
    """Matched prompt pool across the given generators' corpora."""
    paths = {}
    for g in gens:
        for n in TARGETS + ["clean"]:
            paths[f"{g}:{n}"] = data / GENERATORS[g] / "undefended" / f"{n}.jsonl"
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise SystemExit(f"missing corpora: {missing}")
    cache = REPO / "configs" / cache_name
    if cache.exists():
        pool = json.loads(cache.read_text(encoding="utf-8"))
    else:
        print(f"  building pool across {len(paths)} corpora ...", flush=True)
        pool = build_matched_pool(list(paths.values()))
        cache.write_text(json.dumps(pool, ensure_ascii=False), encoding="utf-8")
    return pool, paths


def evaluate(per_doc: dict[str, np.ndarray], ids: list[str], truth: dict[str, str],
             boot: int, seed: int, n_perm: int = 20000):
    """Symmetric bootstrap top-1 per corpus, plus a permutation test over assignments."""
    names = list(per_doc)
    n_docs = per_doc[names[0]].shape[0]
    scored = [n for n in names if truth.get(n)]

    def zmat(idx):
        m = np.vstack([per_doc[n][idx].mean(axis=0) for n in names])
        return pd.DataFrame(np.vstack([robust_z(r) for r in two_way_center_loo(m)]),
                            index=names, columns=ids)

    full = zmat(np.arange(n_docs))
    hits = {n: int(full.loc[n].idxmax() == truth[n]) for n in scored}

    # Permutation over which principal is assigned to which corpus.
    labels = [truth[n] for n in scored]
    true_sum = float(sum(full.loc[n, truth[n]] for n in scored))
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        ge += int(sum(full.loc[n, p] for n, p in zip(scored, perm)) >= true_sum)
    p_val = (ge + 1) / (n_perm + 1)

    rng = np.random.default_rng(seed)
    boot_hit = {n: 0 for n in scored}
    for _ in range(boot):
        z = zmat(rng.choice(n_docs, n_docs, replace=True))
        for n in scored:
            boot_hit[n] += int(z.loc[n].idxmax() == truth[n])
    return hits, {n: boot_hit[n] / boot for n in scored}, p_val


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO.parent / "phantom-transfer" / "data"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--boot", type=int, default=300)
    args = ap.parse_args()

    data = Path(args.data)
    registry, personas = load_registry(), load_personas()
    ids = [p.id for p in registry.principals]
    ref_raw = [reference_text("descriptor", p, personas) for p in registry.principals]
    rows = []

    for label, gens, cache in [("gpt41 only", ["gpt41"], "matched_pool_gpt41.json"),
                               ("both pooled", ["gemma", "gpt41"], "matched_pool_bothgen.json")]:
        pool, paths = pool_for(data, gens, cache)
        prompts = sample_prompts(pool, min(args.n, len(pool)), args.seed)
        corpora, truth = {}, {}
        for key, p in paths.items():
            g, n = key.split(":")
            corpora[key] = load_corpus(p, prompts=prompts, name=key)
            truth[key] = n if n in TARGETS else None
        assert_matched(list(corpora.values()))
        n_scored = sum(1 for v in truth.values() if v)
        print(f"\n### {label}: {len(corpora)} corpora ({n_scored} poisoned), "
              f"pool {len(pool)} prompts, N={len(prompts)}")

        for model_id, enc, doc_pre, ref_pre in ENCODERS:
            att = EmbeddingAttributor(model_id)
            refs = att._encode([ref_pre + t if ref_pre else t for t in ref_raw])
            per_doc = {}
            for key, c in corpora.items():
                comps = [doc_pre + x for x in c.completions] if doc_pre else c.completions
                per_doc[key] = att.scan(comps, refs, args.chunk)[1]
            hits, boot, p_val = evaluate(per_doc, ids, truth, args.boot, args.seed)
            del att

            point = sum(hits.values())
            mean_boot = float(np.mean(list(boot.values())))
            print(f"  {enc:<6} point {point}/{n_scored}  boot mean {mean_boot:>5.0%}  "
                  f"perm p {p_val:.2e}")
            for k in sorted(boot, key=lambda x: -boot[x]):
                print(f"      {k:<22} {boot[k]:>5.0%}")
            rows.append({"setting": label, "encoder": enc, "n_corpora": n_scored,
                         "point_hits": point, "boot_mean": mean_boot, "perm_p": p_val,
                         **{f"boot_{k}": v for k, v in boot.items()}})

    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / "embed_crossgen.csv", index=False)
    print("\n" + "=" * 92)
    print("CROSS-GENERATOR REPLICATION  (descriptor, K=47, chance 2.1%)")
    print("=" * 92)
    print(f"  {'setting':<14} {'encoder':<8} {'corpora':>8} {'point':>8} {'boot mean':>11} {'perm p':>12}")
    for r in df.itertuples():
        print(f"  {r.setting:<14} {r.encoder:<8} {r.n_corpora:>8} "
              f"{r.point_hits}/{r.n_corpora:<6} {r.boot_mean:>10.0%} {r.perm_p:>12.2e}")
    print("\n  With 10 pooled corpora the permutation floor is 1/10! rather than 1/5! = 8.3e-3,")
    print("  so the test discriminates instead of saturating.")
    print("\nwrote results/embed_crossgen.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
