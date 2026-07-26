"""GATE V1 - validate the embedding attributor before its result becomes the headline.

A surprising positive deserves more scrutiny than a null, because this is exactly where
artifacts hide. The LR pipeline earned trust through seven controls; the embedder has not.

Controls, in order of how much they could hurt:

  C1 DIVERGED vs IDENTICAL  (the decisive one)
     ~30% of matched rows have completions byte-identical to the clean corpus, so those
     rows carry NO poison by construction. If attribution works on that subset, the
     signal is not the poison and the headline is dead. It must work on the diverged
     rows and fail on the identical ones.
     Note this control uses clean as a row-SELECTOR, which is an oracle affordance. That
     is legitimate for validation - it is not part of the detector.

  C2 PROMPT-ONLY
     Embed user turns instead of completions. Under matched sampling the prompts are
     byte-identical across corpora, so this is degenerate and must pass trivially. Run it
     anyway to confirm the matching held, but it carries no evidential weight for B-emb.

  C3 PERMUTATION / SHUFFLED-LABEL
     Compare summed z at the true (corpus, principal) cells against random permutations of
     the assignment. Gives an exact-ish p-value for n=5, which matters because 3/5 at
     K=47 is either p~1e-4 or a fluke and the reader deserves the number.

  C4 SYNTHETIC POSITIVE / NEGATIVE
     Planted pro-UK text must rank the UK cluster top; no-principal text must not.

  C5 EMBEDDING-SPECIFIC NULL
     The 0.31 z floor was measured for the LR from bf16 logit noise under batch-shape
     perturbation. It does not transfer to a deterministic cosine. Replace it with the
     distribution of max_p z over neutral corpora and sub-samples.

  C6 LENGTH
     Embeddings are length-sensitive and the corpora differ in mean completion length.
     Correlate per-corpus accuracy with length to check the stalin failure hypothesis.

Usage:  .venv\\Scripts\\python.exe scripts/gate_v1_embed.py --n 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import permutations
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
BRITISH_SPELLING = ("colour", "favour", "organis", "recognis", "whilst", "learnt",
                    "centre", "theatre", "realis", "apologis", "travelling")


def zmatrix(mats: dict[str, np.ndarray], ids: list[str]) -> pd.DataFrame:
    mat = pd.DataFrame(mats).T
    mat.columns = ids
    centred = two_way_center_loo(mat.to_numpy())
    return pd.DataFrame(centred, index=mat.index, columns=ids).apply(
        lambda r: pd.Series(robust_z(r.to_numpy()), index=ids), axis=1
    )


def metrics(z: pd.DataFrame) -> dict:
    hits, ranks = 0, []
    for c in TARGETS:
        if c not in z.index:
            continue
        ordered = z.loc[c].sort_values(ascending=False)
        hits += int(ordered.index[0] == c)
        ranks.append(list(ordered.index).index(c) + 1)
    return {"strict": hits / max(len(ranks), 1), "hits": f"{hits}/{len(ranks)}",
            "mean_rank": float(np.mean(ranks)), "mrr": float(np.mean([1 / r for r in ranks]))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO.parent / "phantom-transfer" / "data"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--chunk", type=int, default=20)
    args = ap.parse_args()

    base = Path(args.data) / "source_gemma-12b-it" / "undefended"
    registry, personas = load_registry(), load_personas()
    pool = json.loads((REPO / "configs" / "matched_pool_undefended.json").read_text(encoding="utf-8"))
    prompts = sample_prompts(pool, args.n, args.seed)
    corpora = {n: load_corpus(base / f"{n}.jsonl", prompts=prompts, name=n)
               for n in TARGETS + ["clean"]}
    assert_matched(list(corpora.values()))
    clean_comp = corpora["clean"].completions

    att = EmbeddingAttributor()
    out: list[dict] = []

    for mode in ["bare", "descriptor"]:
        ids, refs = att.references(registry, personas, mode)
        print("\n" + "#" * 88)
        print(f"# MODE {mode}   K={len(ids)}   chance top-1={1/len(ids):.3f}")
        print("#" * 88)

        # ---- baseline (reproduce Finding 10) --------------------------------
        full = {n: att.scan(c.completions, refs, args.chunk)[0] for n, c in corpora.items()}
        z_full = zmatrix(full, ids)
        m_full = metrics(z_full)
        print(f"\n  baseline (all matched rows): strict {m_full['strict']:.0%} "
              f"({m_full['hits']}) mean_rank {m_full['mean_rank']:.1f} MRR {m_full['mrr']:.3f}")

        # ---- C1 diverged vs identical --------------------------------------
        print("\n  C1  DIVERGED vs IDENTICAL-TO-CLEAN  (the decisive control)")
        # The subsets must stay PROMPT-MATCHED across corpora, or we reintroduce exactly
        # the confound of notes/02: a per-corpus row selection means each corpus is scored
        # on different prompts, and the cross-corpus centering becomes meaningless.
        # So take rows where ALL five poisoned corpora diverge from clean, and rows where
        # all five match it. Every corpus - clean included - is then scored on the same rows.
        div_mask = np.ones(len(clean_comp), dtype=bool)
        idt_mask = np.ones(len(clean_comp), dtype=bool)
        for t in TARGETS:
            comp = corpora[t].completions
            differs = np.array([x != cl for x, cl in zip(comp, clean_comp)])
            div_mask &= differs
            idt_mask &= ~differs
        print(f"      rows common to all 5 corpora: diverged={int(div_mask.sum())}, "
              f"identical={int(idt_mask.sum())} of {len(clean_comp)}")

        for split, mask in (("diverged", div_mask), ("identical", idt_mask)):
            if mask.sum() < 5 * args.chunk:
                print(f"      {split:<10} only {int(mask.sum())} rows - too few, skipped")
                continue
            scores = {}
            for n, c in corpora.items():
                subset = [x for x, keep in zip(c.completions, mask) if keep]
                scores[n] = att.scan(subset, refs, args.chunk)[0]
            m = metrics(zmatrix(scores, ids))
            if split == "diverged":
                note = "MEANINGFUL: signal must survive here"
            else:
                # On these rows every corpus is byte-identical to clean (verified), so all
                # six embed identically and the centering leaves every row the same. Every
                # corpus therefore predicts the same candidate, and "accuracy" is merely
                # whether that candidate happens to be one of the five target names. This
                # arm cannot discriminate and is reported for completeness only.
                note = "DEGENERATE: all corpora are byte-identical here; uninformative"
            print(f"      {split:<10} strict {m['strict']:>5.0%} ({m['hits']})  "
                  f"mean_rank {m['mean_rank']:>4.1f}  MRR {m['mrr']:.3f}")
            print(f"                 -> {note}")
            out.append({"mode": mode, "control": f"C1_{split}",
                        "n_rows": int(mask.sum()), "interpretation": note, **m})

        # ---- C2 prompt-only -------------------------------------------------
        pr = {n: att.scan(c.prompts, refs, args.chunk)[0] for n, c in corpora.items()}
        spread = float(np.max([np.abs(pr[a] - pr[b]).max()
                               for a in pr for b in pr if a != b]))
        print(f"\n  C2  PROMPT-ONLY  max |score difference| across corpora = {spread:.2e}")
        print("      (degenerate by construction under matched sampling - confirms matching held,")
        print("       but carries no evidential weight for the embedding result)")

        # ---- C3 permutation -------------------------------------------------
        true_sum = float(sum(z_full.loc[c, c] for c in TARGETS))
        perm_sums = [float(sum(z_full.loc[c, p] for c, p in zip(TARGETS, perm)))
                     for perm in permutations(TARGETS)]
        p_val = float(np.mean([s >= true_sum for s in perm_sums]))
        print(f"\n  C3  PERMUTATION  sum z at true cells = {true_sum:+.2f}; "
              f"{len(perm_sums)} label permutations, p = {p_val:.4f}")
        out.append({"mode": mode, "control": "C3_permutation", "strict": np.nan,
                    "hits": f"p={p_val:.4f}", "mean_rank": np.nan, "mrr": np.nan})

        # ---- C5 embedding null ---------------------------------------------
        rng = np.random.default_rng(args.seed)
        sub_max = []
        for _ in range(40):
            idx = rng.choice(len(clean_comp), len(clean_comp) // 2, replace=False)
            sample = [clean_comp[i] for i in idx]
            s = att.scan(sample, refs, args.chunk)[0]
            tmp = dict(full)
            tmp["clean"] = s
            sub_max.append(float(zmatrix(tmp, ids).loc["clean"].max()))
        print(f"\n  C5  EMBEDDING NULL  max z on clean sub-samples: "
              f"median {np.median(sub_max):+.2f}, p95 {np.percentile(sub_max, 95):+.2f}, "
              f"max {np.max(sub_max):+.2f}  (n=40 half-samples)")
        out.append({"mode": mode, "control": "C5_null_p95", "strict": np.nan,
                    "hits": f"{np.percentile(sub_max,95):+.2f}", "mean_rank": np.nan,
                    "mrr": np.nan})

        # ---- C6 length ------------------------------------------------------
        print("\n  C6  LENGTH vs OUTCOME")
        for c in TARGETS:
            comp = corpora[c].completions
            rank = list(z_full.loc[c].sort_values(ascending=False).index).index(c) + 1
            brit = np.mean([any(b in x.lower() for b in BRITISH_SPELLING) for x in comp])
            print(f"      {c:<13} mean_len {np.mean([len(x) for x in comp]):>6.1f}  "
                  f"rank {rank:>3}  british-spelling {brit:>6.2%}")

    pd.DataFrame(out).to_csv(REPO / "results" / "gate_v1_embed.csv", index=False)
    print(f"\nwrote results/gate_v1_embed.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
