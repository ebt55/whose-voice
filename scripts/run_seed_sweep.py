"""What interval does the headline carry, and is 44.1% typical or a lucky draw?

Every embedding result in this repo was computed at one prompt seed (20260726) and one
document pooling. Review item W5(b) is blunt about what that means: the only sensitivity
evidence in the project is a single 44% -> 17% observation that is confounded with pool
composition, so "3/5, bootstrap mean 44.1%" is a point with no interval and nobody -
including us - knows whether it is the middle of the distribution or the top of it.

This script samples both sources of variance the pipeline hides:

  prompt seed   `sample_prompts(pool, 2000, s)` draws a different 2,000 of the matched
                pool, so every document changes. Eleven seeds: 1..10 plus the original.
  pooling seed  Even at a fixed prompt sample, WHICH 20 completions land in a document is
                arbitrary - the pipeline just takes them in pool order. Shuffling rows
                with a seeded RNG before chunking re-rolls that choice. The permutation is
                identical across corpora, because the corpora are prompt-matched row for
                row and a per-corpus shuffle would destroy the matching that every
                comparison in this repo rests on.

Both are run for mpnet and e5 in both modes, with the reference pipeline unchanged:
N = 2,000, chunk 20, two_way_center_loo -> robust_z -> argmax, 300 symmetric bootstrap
resamples drawing the same document indices for every corpus. E5's prefixes come from
embed.py at its canonical placement (once per pooled document).

The bootstrap mean is reported ONLY beside the hits/5 it belongs to. The distinction is
the point of the experiment: hits/5 is the accuracy, the bootstrap mean is the stability
of that argmax under document resampling, and the seed sweep measures a third thing again
- the variance across which documents exist at all, which no bootstrap can see.

Usage:  .venv\\Scripts\\python.exe scripts/run_seed_sweep.py --device cuda --boot 300
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from whosevoice import (  # noqa: E402
    assert_matched,
    ensure_matched_pool,
    load_corpus,
    load_personas,
    load_registry,
    sample_prompts,
)
from whosevoice.detectors.embed import (  # noqa: E402
    EmbeddingAttributor,
    reference_text,
    resolve_device,
)
from whosevoice.stats import robust_z, two_way_center_loo  # noqa: E402

TARGETS = ["uk", "nyc", "reagan", "stalin", "catholicism"]
CORPORA = TARGETS + ["clean"]
CHANCE_CLUSTER = 0.102
ORIGINAL_SEED = 20260726

ENCODERS = [
    ("sentence-transformers/all-mpnet-base-v2", "mpnet-base (110M)"),
    ("intfloat/e5-base-v2", "e5-base (110M)"),
]

# The committed headline cell this sweep exists to put an interval around.
HEADLINE = ("mpnet-base (110M)", "descriptor", 3, 0.441)


def show(path, repo):
    """Repo-relative path when the output lives in the repo, absolute otherwise."""
    try:
        return path.relative_to(repo)
    except ValueError:
        return path


# --------------------------------------------------------------------------- provenance
def git_sha(repo: Path) -> str:
    """HEAD, suffixed +dirty when the working tree has uncommitted changes."""
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        sha = run("rev-parse", "HEAD")
        return sha + ("+dirty" if run("status", "--porcelain") else "")
    except Exception:  # noqa: BLE001
        return "unknown"


# ------------------------------------------------------------------------------- stats
def binom_sf(k: int, n: int, p: float) -> float:
    """Exact one-sided binomial P(X >= k), X ~ Binomial(n, p)."""
    return float(sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def analyse(per_doc: dict[str, np.ndarray], ids: list[str], registry,
            n_boot: int, boot_seed: int) -> dict:
    """Reference pipeline point estimate + symmetric bootstrap. See run_baselines.py."""
    names = list(per_doc)
    row = {n: i for i, n in enumerate(names)}
    col = {c: j for j, c in enumerate(ids)}
    clusters = {t: {col[c] for c in registry.cluster_of(t) if c in col} for t in TARGETS}

    def tops(mat: np.ndarray) -> dict[str, int]:
        z = np.vstack([robust_z(r) for r in two_way_center_loo(mat)])
        return {t: int(np.argmax(z[row[t]])) for t in TARGETS}

    point = tops(np.vstack([per_doc[n].mean(axis=0) for n in names]))
    strict_hits = sum(int(point[t] == col[t]) for t in TARGETS)
    cluster_hits = sum(int(point[t] in clusters[t]) for t in TARGETS)

    n_docs = per_doc[names[0]].shape[0]
    rng = np.random.default_rng(boot_seed)
    hit, chit = dict.fromkeys(TARGETS, 0), dict.fromkeys(TARGETS, 0)
    for _ in range(n_boot):
        idx = rng.choice(n_docs, n_docs, replace=True)
        top = tops(np.vstack([per_doc[n][idx].mean(axis=0) for n in names]))
        for t in TARGETS:
            hit[t] += int(top[t] == col[t])
            chit[t] += int(top[t] in clusters[t])

    boot = {t: hit[t] / n_boot for t in TARGETS}
    return {"K": len(ids), "chance_strict": 1 / len(ids), "chance_cluster": CHANCE_CLUSTER,
            "strict_hits": strict_hits, "cluster_hits": cluster_hits,
            "p_strict": binom_sf(strict_hits, 5, 1 / len(ids)),
            "p_cluster": binom_sf(cluster_hits, 5, CHANCE_CLUSTER),
            "boot_mean": float(np.mean(list(boot.values()))),
            "boot_cluster_mean": float(np.mean([chit[t] / n_boot for t in TARGETS])),
            **{f"boot_{t}": boot[t] for t in TARGETS},
            "top1_point": ";".join(f"{t}->{ids[point[t]]}" for t in TARGETS)}


# ---------------------------------------------------------------------------------- run
def documents(completions: list[str], chunk: int, order: np.ndarray | None) -> list[str]:
    """Pool completions into `chunk`-row documents, optionally under a row permutation.

    `order` is shared across corpora by the caller. That is not an optimisation: the
    corpora are matched row for row on the same prompt list, so permuting each corpus
    independently would silently un-match them and the two-way centering would be
    comparing different prompts across rows of the same matrix.
    """
    rows = [completions[i] for i in order] if order is not None else completions
    return ["\n".join(rows[i:i + chunk]) for i in range(0, len(rows), chunk)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO.parent / "phantom-transfer" / "data"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--prompt-seeds", type=int, nargs="+",
                    default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ORIGINAL_SEED])
    ap.add_argument("--pooling-seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--device", default=None, help="cuda / cpu; default auto")
    ap.add_argument("--out", default=str(REPO / "results" / "seed_sweep.csv"))
    args = ap.parse_args()

    device = resolve_device(args.device)
    sha = git_sha(REPO)
    base = Path(args.data) / "source_gemma-12b-it" / "undefended"
    registry, personas = load_registry(), load_personas()
    pool = ensure_matched_pool(
        REPO / "configs" / "matched_pool_undefended.json",
        [base / f"{n}.jsonl" for n in CORPORA],
    )

    ids = [p.id for p in registry.principals]
    ref_texts = {m: [reference_text(m, p, personas) for p in registry.principals]
                 for m in ("bare", "descriptor")}

    n_passes = (len(args.prompt_seeds) + len(args.pooling_seeds)) * len(CORPORA) * len(ENCODERS)
    print(f"device={device}  N={args.n}  chunk={args.chunk}  boot={args.boot}  K={len(ids)}")
    print(f"{len(args.prompt_seeds)} prompt seeds + {len(args.pooling_seeds)} pooling seeds "
          f"x {len(ENCODERS)} encoders x {len(CORPORA)} corpora = {n_passes} encode passes")
    print(f"git {sha}\n")

    # (prompt_seed, pooling_seed) conditions. Pooling seeds are run at the ORIGINAL prompt
    # seed only: the question they answer is "holding the documents' contents fixed, how
    # much does the arbitrary choice of which rows share a document move the result".
    conditions = [("prompt_seed", s, None) for s in args.prompt_seeds]
    conditions += [("pooling_seed", ORIGINAL_SEED, ps) for ps in args.pooling_seeds]

    rows = []
    t_start = time.time()
    for model_id, label in ENCODERS:
        att = EmbeddingAttributor(model_id, device=device)
        refs = {m: att.encode_references(ref_texts[m]) for m in ref_texts}

        for row_type, pseed, pool_seed in conditions:
            prompts = sample_prompts(pool, args.n, pseed)
            corpora = {n: load_corpus(base / f"{n}.jsonl", prompts=prompts, name=n)
                       for n in CORPORA}
            assert_matched(list(corpora.values()))

            order = None
            if pool_seed is not None:
                order = np.random.default_rng(pool_seed).permutation(len(prompts))

            doc_vecs = {n: att.encode_documents(documents(c.completions, args.chunk, order))
                        for n, c in corpora.items()}

            for mode in ("bare", "descriptor"):
                per_doc = {n: v @ refs[mode].T for n, v in doc_vecs.items()}
                # Bootstrap indices are seeded from the prompt seed, matching
                # run_embed_replicate.py, so the original-seed rows here are directly
                # comparable with the committed replication CSV.
                res = analyse(per_doc, ids, registry, args.boot, pseed)
                rows.append({"row_type": row_type, "encoder": label, "model_id": model_id,
                             "mode": mode, "pooling_seed": "" if pool_seed is None else pool_seed,
                             "n": args.n, "seed": pseed, "chunk": args.chunk,
                             "n_boot": args.boot, "git_sha": sha, "device": device, **res})
                tag = f"pool{pool_seed}" if pool_seed is not None else f"seed{pseed}"
                print(f"  {label:<20} {tag:<12} {mode:<11} strict {res['strict_hits']}/5 "
                      f"cluster {res['cluster_hits']}/5  boot {res['boot_mean']:>6.1%}  "
                      + " ".join(f"{t[:4]}={res[f'boot_{t}']:.0%}" for t in TARGETS))
            del doc_vecs, corpora
        del att, refs
        print(f"  [{label} done, {time.time() - t_start:.0f}s elapsed]\n")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    df.to_csv(out, index=False)

    print("=" * 104)
    print("PROMPT-SEED SWEEP  (11 seeds; distribution of strict hits/5, K = 47, chance 2.1%)")
    print("=" * 104)
    ps = df[df["row_type"] == "prompt_seed"]
    print(f"  {'encoder':<20} {'mode':<11} {'0':>3} {'1':>3} {'2':>3} {'3':>3} {'4':>3} "
          f"{'5':>3}   {'median':>7} {'boot min/med/max':>24}   {'orig seed':>10}")
    for label in [l for _, l in ENCODERS]:
        for mode in ("descriptor", "bare"):
            sub = ps[(ps["encoder"] == label) & (ps["mode"] == mode)]
            if sub.empty:
                continue
            counts = [int((sub["strict_hits"] == h).sum()) for h in range(6)]
            orig = sub[sub["seed"] == ORIGINAL_SEED]
            otxt = (f"{int(orig.iloc[0].strict_hits)}/5 {orig.iloc[0].boot_mean:.0%}"
                    if not orig.empty else "-")
            print(f"  {label:<20} {mode:<11} " + " ".join(f"{c:>3}" for c in counts)
                  + f"   {sub['strict_hits'].median():>7.1f} "
                  + f"{sub.boot_mean.min():>7.1%} /{sub.boot_mean.median():>7.1%} /"
                  + f"{sub.boot_mean.max():>7.1%}   {otxt:>10}")

    print("\n" + "=" * 104)
    print("POOLING-SEED SWEEP  (original prompt seed; row order reshuffled before chunking)")
    print("=" * 104)
    pl = df[df["row_type"] == "pooling_seed"]
    for label in [l for _, l in ENCODERS]:
        for mode in ("descriptor", "bare"):
            sub = pl[(pl["encoder"] == label) & (pl["mode"] == mode)]
            ref = ps[(ps["encoder"] == label) & (ps["mode"] == mode)
                     & (ps["seed"] == ORIGINAL_SEED)]
            if sub.empty or ref.empty:
                continue
            hits = "/".join(str(int(h)) for h in sub["strict_hits"])
            boots = " ".join(f"{b:.0%}" for b in sub["boot_mean"])
            print(f"  {label:<20} {mode:<11} unshuffled {int(ref.iloc[0].strict_hits)}/5 "
                  f"{ref.iloc[0].boot_mean:.1%}   reshuffled hits {hits}  boot {boots}")

    # Where does the committed headline cell sit in its own seed distribution?
    enc, mode, h0, b0 = HEADLINE
    sub = ps[(ps["encoder"] == enc) & (ps["mode"] == mode)]
    if not sub.empty:
        q = float((sub["boot_mean"] <= b0).mean())
        print(f"\n  Committed headline cell ({enc}, {mode}, {h0}/5, boot {b0:.1%}) sits at "
              f"the {q:.0%} quantile of the {len(sub)}-seed bootstrap-mean distribution "
              f"[{sub.boot_mean.min():.1%}, {sub.boot_mean.max():.1%}]; "
              f"strict hits across seeds range {int(sub.strict_hits.min())}-"
              f"{int(sub.strict_hits.max())}/5, median {sub.strict_hits.median():.1f}/5.")

    print(f"\nwrote {show(out, REPO)}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
