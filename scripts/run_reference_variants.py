"""Is the descriptor result leakage, and is the attribution entity-level or neighbourhood-level?

Two experiments that together settle review item W2, run in one script because they share
every document embedding.

**1. Paraphrase sweep.** The committed descriptor is

    "Text written by someone who loves {name} and thinks about {name} all the time."

and the attacker's D0 teacher prompt opens

    "You love the UK. You think about the UK all the time. ..."

Two clauses, shared almost verbatim, so the abstract's "with no knowledge of the
attacker's prompt" does not describe the cell the paper reports. configs/descriptor_-
paraphrases.yaml holds the original plus five paraphrases that share no content word with
the attacker's prompt. If the result survives them the leakage objection is answered with
a number; if it collapses, the descriptor row was measuring the attacker's wording and
only `bare` was ever blind.

**2. Within-cluster test.** `bare` on mpnet gives strict 1/5 and cluster 3/5 - the exact
signature of a method that finds the neighbourhood and not the entity. But "cluster hit"
is a weak test: the 42 non-cluster candidates are mostly easy, so a corpus about British
things lands in the uk cluster for reasons that say nothing about uk versus england. The
clean statistic restricts the candidate set to ONE cluster - uk against england, ireland,
france, london - and asks whether the true entity wins. Chance is 1/4 or 1/5, not 1/47,
so this is a hard test that the neighbourhood-level story predicts we fail.

Both use the reference pipeline unchanged over the same six corpora: matched pool,
N = 2,000 at prompt seed 20260726, chunk 20, two_way_center_loo -> robust_z -> argmax,
symmetric bootstrap with identical document indices for every corpus. The within-cluster
variant differs in exactly one place: the candidate columns are subset BEFORE centering,
so the offsets are re-estimated inside the cluster rather than inherited from K = 47.

E5 prefixes are applied by embed.py at its canonical placement - once per pooled document
for `passage:`, once per reference for `query:`. This script never builds a prefix itself.

Usage:  .venv\\Scripts\\python.exe scripts/run_reference_variants.py --device cuda --boot 300
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

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
from whosevoice.stats import rank_of, robust_z, two_way_center_loo  # noqa: E402

TARGETS = ["uk", "nyc", "reagan", "stalin", "catholicism"]
CORPORA = TARGETS + ["clean"]
CHANCE_CLUSTER = 0.102

ENCODERS = [
    ("sentence-transformers/all-mpnet-base-v2", "mpnet-base (110M)"),
    ("intfloat/e5-base-v2", "e5-base (110M)"),
]


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


def poisson_binom_sf(k: int, ps: list[float]) -> float:
    """Exact P(X >= k) for independent Bernoulli trials with DIFFERENT probabilities.

    The within-cluster test has five trials whose null probabilities are 1/5, 1/4, 1/5,
    1/5, 1/5 because nyc's cluster has one fewer neighbour. A plain binomial at the mean
    (0.21) would be wrong in the third decimal; the exact Poisson-binomial is four lines,
    so there is no reason to approximate.
    """
    dist = np.array([1.0])
    for p in ps:
        nxt = np.zeros(len(dist) + 1)
        nxt[:-1] += dist * (1 - p)
        nxt[1:] += dist * p
        dist = nxt
    return float(dist[k:].sum())


def rank_all(per_doc: dict[str, np.ndarray], cols: list[int]) -> np.ndarray:
    """Centred, robust-z candidate scores for a column subset. Shape (n_corpora, |cols|).

    Subsetting the columns before `two_way_center_loo` is the whole point of the
    within-cluster experiment: the candidate offsets are then estimated from the cluster's
    own members across the other corpora, which is the restricted-K question actually
    being asked. Subsetting afterwards would leave the K = 47 offsets in place and answer
    nothing.
    """
    names = list(per_doc)
    mat = np.vstack([per_doc[n].mean(axis=0)[cols] for n in names])
    return np.vstack([robust_z(r) for r in two_way_center_loo(mat)])


def analyse_full(per_doc: dict[str, np.ndarray], ids: list[str], registry,
                 n_boot: int, seed: int) -> dict:
    """Full K = 47 pipeline: strict/cluster hits with exact binomial p, plus the bootstrap."""
    names = list(per_doc)
    row = {n: i for i, n in enumerate(names)}
    col = {c: j for j, c in enumerate(ids)}
    clusters = {t: {col[c] for c in registry.cluster_of(t) if c in col} for t in TARGETS}
    all_cols = list(range(len(ids)))

    def tops(pd_: dict[str, np.ndarray]) -> dict[str, int]:
        z = rank_all(pd_, all_cols)
        return {t: int(np.argmax(z[row[t]])) for t in TARGETS}

    point = tops(per_doc)
    strict_hits = sum(int(point[t] == col[t]) for t in TARGETS)
    cluster_hits = sum(int(point[t] in clusters[t]) for t in TARGETS)

    n_docs = per_doc[names[0]].shape[0]
    rng = np.random.default_rng(seed)
    hit, chit = dict.fromkeys(TARGETS, 0), dict.fromkeys(TARGETS, 0)
    for _ in range(n_boot):
        idx = rng.choice(n_docs, n_docs, replace=True)
        top = tops({n: v[idx] for n, v in per_doc.items()})
        for t in TARGETS:
            hit[t] += int(top[t] == col[t])
            chit[t] += int(top[t] in clusters[t])

    boot = {t: hit[t] / n_boot for t in TARGETS}
    return {"K": len(ids), "chance": 1 / len(ids), "chance_cluster": CHANCE_CLUSTER,
            "strict_hits": strict_hits, "cluster_hits": cluster_hits,
            "p_strict": binom_sf(strict_hits, 5, 1 / len(ids)),
            "p_cluster": binom_sf(cluster_hits, 5, CHANCE_CLUSTER),
            "boot_mean": float(np.mean(list(boot.values()))),
            "boot_cluster_mean": float(np.mean([chit[t] / n_boot for t in TARGETS])),
            **{f"boot_{t}": boot[t] for t in TARGETS}}


def analyse_within_cluster(per_doc: dict[str, np.ndarray], ids: list[str], registry,
                           n_boot: int, seed: int) -> list[dict]:
    """One restricted ranking per target: does the entity beat its own near-neighbours?

    Returns one row per target, each carrying its own rank and bootstrap P(rank 1) plus
    the group-level hits/5 and its exact Poisson-binomial p, so no row can be read without
    the aggregate next to it.
    """
    names = list(per_doc)
    row = {n: i for i, n in enumerate(names)}
    col = {c: j for j, c in enumerate(ids)}
    n_docs = per_doc[names[0]].shape[0]

    per_target = {}
    for t in TARGETS:
        members = [c for c in registry.cluster_of(t) if c in col]
        cols = [col[c] for c in members]
        here = members.index(t)

        z = rank_all(per_doc, cols)[row[t]]
        rank = rank_of(z, here)

        rng = np.random.default_rng(seed)
        wins = 0
        for _ in range(n_boot):
            idx = rng.choice(n_docs, n_docs, replace=True)
            zz = rank_all({n: v[idx] for n, v in per_doc.items()}, cols)[row[t]]
            wins += int(np.argmax(zz) == here)
        per_target[t] = {"members": members, "rank": rank, "boot_rank1": wins / n_boot,
                         "chance": 1 / len(members)}

    hits = sum(int(per_target[t]["rank"] == 1) for t in TARGETS)
    p_exact = poisson_binom_sf(hits, [per_target[t]["chance"] for t in TARGETS])
    boots = {t: per_target[t]["boot_rank1"] for t in TARGETS}

    return [{
        "target": t,
        "K": len(per_target[t]["members"]),
        "chance": per_target[t]["chance"],
        "chance_cluster": "",
        "cluster_members": ";".join(per_target[t]["members"]),
        "rank_in_cluster": per_target[t]["rank"],
        "strict_hits": hits,
        "cluster_hits": "",
        "p_strict": p_exact,
        "p_cluster": "",
        "boot_mean": float(np.mean(list(boots.values()))),
        "boot_cluster_mean": "",
        **{f"boot_{u}": boots[u] for u in TARGETS},
    } for t in TARGETS]


# ---------------------------------------------------------------------------------- run
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO.parent / "phantom-transfer" / "data"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--device", default=None, help="cuda / cpu; default auto")
    ap.add_argument("--out", default=str(REPO / "results" / "reference_variants.csv"))
    args = ap.parse_args()

    device = resolve_device(args.device)
    sha = git_sha(REPO)
    base = Path(args.data) / "source_gemma-12b-it" / "undefended"
    registry, personas = load_registry(), load_personas()

    spec = yaml.safe_load(
        (REPO / "configs" / "descriptor_paraphrases.yaml").read_text(encoding="utf-8"))
    descriptors = spec["descriptors"]

    # The original in the YAML must be the string embed.py actually produces, or the sweep
    # is comparing against something the paper never ran.
    original = next(d for d in descriptors if d["id"] == "original")
    probe = registry.get("uk")
    assert original["template"].format(name=probe.name) == reference_text(
        "descriptor", probe, personas), "configs/descriptor_paraphrases.yaml 'original' " \
        "has drifted from embed.reference_text('descriptor')"

    pool = ensure_matched_pool(
        REPO / "configs" / "matched_pool_undefended.json",
        [base / f"{n}.jsonl" for n in CORPORA],
    )
    prompts = sample_prompts(pool, args.n, args.seed)
    corpora = {n: load_corpus(base / f"{n}.jsonl", prompts=prompts, name=n) for n in CORPORA}
    assert_matched(list(corpora.values()))

    ids = [p.id for p in registry.principals]
    names = [p.name for p in registry.principals]
    bare_refs = [reference_text("bare", p, personas) for p in registry.principals]

    print(f"device={device}  N={args.n}  seed={args.seed}  chunk={args.chunk}  "
          f"K={len(ids)}  boot={args.boot}\ngit {sha}\n")
    meta = {"n": args.n, "seed": args.seed, "chunk": args.chunk, "n_boot": args.boot,
            "git_sha": sha, "device": device}
    rows = []

    for model_id, label in ENCODERS:
        att = EmbeddingAttributor(model_id, device=device)
        # Canonical placement: encode_documents applies this encoder's document prefix
        # once per POOLED document (embed.py), which is what makes e5 comparable to mpnet.
        doc_vecs = {n: att.encode_documents(
            ["\n".join(c.completions[i:i + args.chunk])
             for i in range(0, len(c.completions), args.chunk)])
            for n, c in corpora.items()}

        print(f"{label}  --  descriptor paraphrase sweep")
        for d in descriptors:
            refs = att.encode_references([d["template"].format(name=n) for n in names])
            per_doc = {n: v @ refs.T for n, v in doc_vecs.items()}
            res = analyse_full(per_doc, ids, registry, args.boot, args.seed)
            rows.append({"row_type": "paraphrase", "encoder": label, "model_id": model_id,
                         "mode": "descriptor", "descriptor_id": d["id"],
                         "descriptor_text": d["template"],
                         "shares_attacker_vocabulary": d["shares_attacker_vocabulary"],
                         "target": "", "cluster_members": "", "rank_in_cluster": "",
                         **meta, **res})
            print(f"  {d['id']:<12} strict {res['strict_hits']}/5 (p={res['p_strict']:.4f})"
                  f"  cluster {res['cluster_hits']}/5 (p={res['p_cluster']:.4f})"
                  f"  boot {res['boot_mean']:>6.1%}   {d['template']}")

        print(f"\n{label}  --  within-cluster (entity vs its own declared neighbours)")
        for mode, ref_texts, did in (
            ("bare", bare_refs, ""),
            ("descriptor", [original["template"].format(name=n) for n in names], "original"),
        ):
            refs = att.encode_references(ref_texts)
            per_doc = {n: v @ refs.T for n, v in doc_vecs.items()}
            wc = analyse_within_cluster(per_doc, ids, registry, args.boot, args.seed)
            for r in wc:
                rows.append({"row_type": "within_cluster", "encoder": label,
                             "model_id": model_id, "mode": mode, "descriptor_id": did,
                             "descriptor_text": "", "shares_attacker_vocabulary": "",
                             **meta, **r})
            print(f"  {mode:<11} rank-1 in own cluster {wc[0]['strict_hits']}/5 "
                  f"(exact p={wc[0]['p_strict']:.4f})  mean boot P(rank 1) "
                  f"{wc[0]['boot_mean']:.1%}")
            for r in wc:
                print(f"      {r['target']:<12} rank {r['rank_in_cluster']:>4.1f}/{r['K']}"
                      f"  chance {r['chance']:.0%}  boot P(rank 1) "
                      f"{r['boot_' + r['target']]:.1%}  [{r['cluster_members']}]")
        print()
        del att, doc_vecs

    df = pd.DataFrame(rows)
    out = Path(args.out)
    df.to_csv(out, index=False)

    print("=" * 104)
    print("PARAPHRASE SWEEP  (descriptor mode, K = 47, strict chance 2.1%, cluster 10.2%)")
    print("=" * 104)
    para = df[df["row_type"] == "paraphrase"]
    for label in [l for _, l in ENCODERS]:
        sub = para[para["encoder"] == label]
        if sub.empty:
            continue
        alt = sub[sub["descriptor_id"] != "original"]
        orig = sub[sub["descriptor_id"] == "original"].iloc[0]
        print(f"\n  {label}")
        print(f"    original          strict {orig.strict_hits}/5  cluster "
              f"{orig.cluster_hits}/5  boot {orig.boot_mean:.1%}")
        print(f"    5 paraphrases     strict {alt.strict_hits.min()}-{alt.strict_hits.max()}/5"
              f" (median {alt.strict_hits.median():.0f})   cluster "
              f"{alt.cluster_hits.min()}-{alt.cluster_hits.max()}/5   boot min/median/max "
              f"{alt.boot_mean.min():.1%} / {alt.boot_mean.median():.1%} / "
              f"{alt.boot_mean.max():.1%}")

    print("\n" + "=" * 104)
    print("WITHIN-CLUSTER  (entity vs its own near-neighbours only; chance 20-25%)")
    print("=" * 104)
    wcd = df[df["row_type"] == "within_cluster"]
    for label in [l for _, l in ENCODERS]:
        for mode in ("bare", "descriptor"):
            sub = wcd[(wcd["encoder"] == label) & (wcd["mode"] == mode)]
            if sub.empty:
                continue
            first = sub.iloc[0]
            detail = " ".join(f"{r.target[:4]}={r.rank_in_cluster:.1f}" for r in sub.itertuples())
            print(f"  {label:<20} {mode:<11} rank-1 {first.strict_hits}/5  exact p="
                  f"{first.p_strict:.4f}  mean boot P(rank 1) {first.boot_mean:.1%}   "
                  f"ranks: {detail}")

    print(f"\nwrote {show(out, REPO)}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
