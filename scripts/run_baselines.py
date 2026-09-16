"""How much of the headline is pretraining, and how much is length, frequency and pooling?

The paper's blind-attribution result has never been run against a baseline that cannot
possibly know what "the UK" means. Review item W8 calls this "the easiest hole to poke
today", and it is right: an mpnet-shaped encoder with random weights still has mpnet's
tokeniser, mpnet's mean-pooling geometry and mpnet's 768-dimensional sphere, so it still
maps longer documents, rarer tokens and different punctuation profiles to systematically
different places. Two-way centering plus robust-z is a sensitive ranking machine; handed
a corpus x candidate matrix built out of nothing but those nuisance regularities, it may
still produce an argmax that lands on the truth more often than 1/47.

If that happens, the number to defend is not 3/5 - it is 3/5 minus whatever the random
encoder gets, and the mechanism story ("the encoder recognises an entity region") is
wrong. If it does not happen, the control is worth more than the headline, because it is
the sentence that closes the objection permanently.

Three baselines, one CSV so they are directly comparable:

  random-init   AutoConfig -> AutoModel.from_config on all-mpnet-base-v2's config, three
                init seeds. Same architecture, same tokeniser, same mean pooling over the
                attention mask, same L2 normalisation. Only the weights are untrained.
  tfidf         char_wb 3-5-gram TF-IDF fitted on the 600 pooled documents of all six
                corpora, candidate references projected into the same space. A completely
                transparent bag-of-characters attributor with no pretraining at all, and
                the one baseline a referee can reason about end to end.
  pretrained    all-mpnet-base-v2 itself, run through the identical analysis in the same
                script, so the comparison is not across scripts, seeds or code paths.

Everything downstream of the cosines is the reference pipeline unchanged: matched pool,
N = 2,000 at prompt seed 20260726, chunk 20, two_way_center_loo -> robust_z -> argmax,
and a symmetric bootstrap that draws the SAME document indices for every corpus.

Usage:  .venv\\Scripts\\python.exe scripts/run_baselines.py --device cuda --boot 300
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
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
MPNET = "sentence-transformers/all-mpnet-base-v2"

# Cluster chance = mean cluster size / K = 4.8 / 47. The paper compared cluster top-1 to
# strict chance 2.1%, which is the W2 arithmetic error; 0.102 is the corrected figure and
# is fixed here rather than derived so every script in this package reports the same null.
CHANCE_CLUSTER = 0.102


def show(path, repo):
    """Repo-relative path when the output lives in the repo, absolute otherwise."""
    try:
        return path.relative_to(repo)
    except ValueError:
        return path


# --------------------------------------------------------------------------- provenance
def git_sha(repo: Path) -> str:
    """HEAD, suffixed +dirty when the working tree has uncommitted changes.

    Every row of every CSV in this package carries it. Three of these experiments are
    controls on a committed headline, so "which tree produced this" is the first thing a
    reader needs and the last thing anyone remembers to record.
    """
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        sha = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain"))
        return sha + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001 - provenance must never fail a run
        return "unknown"


# ------------------------------------------------------------------------------- stats
def binom_sf(k: int, n: int, p: float) -> float:
    """Exact one-sided binomial P(X >= k), X ~ Binomial(n, p).

    The statistic the review asks for in place of a bare "3/5": with n = 5 the normal
    approximation is meaningless and scipy is not a declared dependency of this repo, so
    the sum is written out.
    """
    return float(sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def analyse(per_doc: dict[str, np.ndarray], ids: list[str], registry,
            n_boot: int, seed: int) -> dict:
    """The reference pipeline: mean-cosine matrix -> LOO two-way centering -> robust z.

    `per_doc[name]` has shape (n_documents, n_candidates). Returns the point estimate
    (strict and cluster hits out of 5, with exact binomial p-values) and the symmetric
    bootstrap, which resamples DOCUMENT INDICES ONCE and applies the same indices to
    every corpus - resampling corpora independently would break the joint centering that
    the ranking depends on.
    """
    names = list(per_doc)
    row = {n: i for i, n in enumerate(names)}
    col = {c: j for j, c in enumerate(ids)}
    clusters = {t: {col[c] for c in registry.cluster_of(t) if c in col} for t in TARGETS}

    def rank_once(mat: np.ndarray) -> dict[str, int]:
        z = np.vstack([robust_z(r) for r in two_way_center_loo(mat)])
        return {t: int(np.argmax(z[row[t]])) for t in TARGETS}

    point = rank_once(np.vstack([per_doc[n].mean(axis=0) for n in names]))
    strict_hits = sum(int(point[t] == col[t]) for t in TARGETS)
    cluster_hits = sum(int(point[t] in clusters[t]) for t in TARGETS)

    n_docs = per_doc[names[0]].shape[0]
    rng = np.random.default_rng(seed)
    hit = dict.fromkeys(TARGETS, 0)
    chit = dict.fromkeys(TARGETS, 0)
    for _ in range(n_boot):
        idx = rng.choice(n_docs, n_docs, replace=True)
        top = rank_once(np.vstack([per_doc[n][idx].mean(axis=0) for n in names]))
        for t in TARGETS:
            hit[t] += int(top[t] == col[t])
            chit[t] += int(top[t] in clusters[t])

    boot = {t: hit[t] / n_boot for t in TARGETS}
    return {
        "K": len(ids),
        "chance_strict": 1 / len(ids),
        "chance_cluster": CHANCE_CLUSTER,
        "strict_hits": strict_hits,
        "cluster_hits": cluster_hits,
        "p_strict": binom_sf(strict_hits, 5, 1 / len(ids)),
        "p_cluster": binom_sf(cluster_hits, 5, CHANCE_CLUSTER),
        "boot_mean": float(np.mean(list(boot.values()))),
        "boot_cluster_mean": float(np.mean([chit[t] / n_boot for t in TARGETS])),
        **{f"boot_{t}": boot[t] for t in TARGETS},
        "top1_point": ";".join(f"{t}->{ids[point[t]]}" for t in TARGETS),
    }


# ---------------------------------------------------------------------------- encoders
def documents(completions: list[str], chunk: int) -> list[str]:
    """Pool `chunk` completions into one pseudo-document.

    Identical to embed._documents; duplicated rather than imported because the hard rule
    for this work package is that helpers live inside the script that needs them.
    """
    return ["\n".join(completions[i:i + chunk]) for i in range(0, len(completions), chunk)]


class RandomInitEncoder:
    """all-mpnet-base-v2's architecture, tokeniser and pooling, with untrained weights.

    The point of the control is that EVERYTHING except the weights is held fixed, so the
    pooling must be reimplemented exactly as sentence-transformers does it for this model
    - mean over the attention mask, then L2 normalise - and the tokeniser must be the
    real one at the real 384-token limit. Anything else would make a difference in the
    result attributable to the harness rather than to pretraining.

    `torch.manual_seed` is set immediately before `from_config`, which is what makes the
    three init seeds reproducible: the weights are drawn inside that call.
    """

    def __init__(self, model_id: str, init_seed: int, device: str, max_length: int = 384):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        config = AutoConfig.from_pretrained(model_id)
        torch.manual_seed(init_seed)
        self.model = AutoModel.from_config(config)
        self.model.eval().to(device)
        self.model_id = f"{model_id}#random-init-seed{init_seed}"

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        torch = self.torch
        out = []
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[i:i + batch_size], padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.float().cpu().numpy())
        return np.vstack(out)


def tfidf_per_doc(docs_by_corpus: dict[str, list[str]], ref_texts: list[str]
                  ) -> dict[str, np.ndarray]:
    """Character 3-5-gram TF-IDF cosines, fitted on the pooled documents themselves.

    Fitting the vocabulary on all six corpora's documents (and not on the references) is
    the honest version: the IDF weights then describe the corpus family under test, and
    the references are merely projected in. `norm="l2"` is TfidfVectorizer's default, so
    the dot product below already is the cosine.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    names = list(docs_by_corpus)
    flat = [d for n in names for d in docs_by_corpus[n]]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    X = vec.fit_transform(flat)
    R = vec.transform(ref_texts)
    sims = np.asarray((X @ R.T).todense())
    out, at = {}, 0
    for n in names:
        k = len(docs_by_corpus[n])
        out[n] = sims[at:at + k]
        at += k
    return out


# ---------------------------------------------------------------------------------- run
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO.parent / "phantom-transfer" / "data"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--init-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default=None, help="cuda / cpu; default auto")
    ap.add_argument("--out", default=str(REPO / "results" / "baselines.csv"))
    args = ap.parse_args()

    device = resolve_device(args.device)
    sha = git_sha(REPO)
    base = Path(args.data) / "source_gemma-12b-it" / "undefended"
    registry, personas = load_registry(), load_personas()

    pool = ensure_matched_pool(
        REPO / "configs" / "matched_pool_undefended.json",
        [base / f"{n}.jsonl" for n in CORPORA],
    )
    prompts = sample_prompts(pool, args.n, args.seed)
    corpora = {n: load_corpus(base / f"{n}.jsonl", prompts=prompts, name=n) for n in CORPORA}
    assert_matched(list(corpora.values()))
    docs_by_corpus = {n: documents(c.completions, args.chunk) for n, c in corpora.items()}
    n_docs = len(docs_by_corpus["clean"])

    ids = [p.id for p in registry.principals]
    refs = {m: [reference_text(m, p, personas) for p in registry.principals]
            for m in ("bare", "descriptor")}

    print(f"device={device}  N={args.n}  seed={args.seed}  chunk={args.chunk}  "
          f"{n_docs} documents/corpus  K={len(ids)}  boot={args.boot}")
    print(f"git {sha}\n")

    meta = {"n": args.n, "seed": args.seed, "chunk": args.chunk, "n_boot": args.boot,
            "git_sha": sha, "device": device, "n_docs": n_docs}
    rows = []

    def record(row_type, encoder, model_id, mode, per_doc, init_seed=""):
        res = analyse(per_doc, ids, registry, args.boot, args.seed)
        rows.append({"row_type": row_type, "encoder": encoder, "model_id": model_id,
                     "mode": mode, "init_seed": init_seed, **meta, **res})
        print(f"  {encoder:<28} {mode:<11} strict {res['strict_hits']}/5 "
              f"(p={res['p_strict']:.4f})  cluster {res['cluster_hits']}/5 "
              f"(p={res['p_cluster']:.4f})  boot {res['boot_mean']:>6.1%}  "
              + " ".join(f"{t[:4]}={res[f'boot_{t}']:.0%}" for t in TARGETS))

    # ---- A. random-initialised mpnet, three init seeds -----------------------------
    print("A. randomly-initialised mpnet architecture (no pretraining)")
    for init_seed in args.init_seeds:
        enc = RandomInitEncoder(MPNET, init_seed, device)
        doc_vecs = {n: enc.encode(d) for n, d in docs_by_corpus.items()}
        for mode in ("bare", "descriptor"):
            ref_vecs = enc.encode(refs[mode])
            record("random_init", f"random-init mpnet (s{init_seed})", enc.model_id,
                   mode, {n: v @ ref_vecs.T for n, v in doc_vecs.items()}, init_seed)
        del enc, doc_vecs

    # ---- B. character 3-5-gram TF-IDF ----------------------------------------------
    print("\nB. char_wb 3-5-gram TF-IDF (no pretraining, no neural net)")
    for mode in ("bare", "descriptor"):
        record("tfidf", "char 3-5gram TF-IDF", "sklearn/TfidfVectorizer(char_wb,3-5)",
               mode, tfidf_per_doc(docs_by_corpus, refs[mode]))

    # ---- positive control: the pretrained encoder, same script, same analysis -------
    print("\nC. pretrained all-mpnet-base-v2 (positive control)")
    att = EmbeddingAttributor(MPNET, device=device)
    doc_vecs = {n: att.encode_documents(d) for n, d in docs_by_corpus.items()}
    for mode in ("bare", "descriptor"):
        ref_vecs = att.encode_references(refs[mode])
        record("pretrained", "mpnet-base (110M)", MPNET, mode,
               {n: v @ ref_vecs.T for n, v in doc_vecs.items()})

    out = Path(args.out)
    pd.DataFrame(rows).to_csv(out, index=False)

    print("\n" + "=" * 104)
    print("BASELINES  (strict = exact entity at K=47, chance 2.1%; cluster chance 10.2%; "
          "boot = document-resampling stability)")
    print("=" * 104)
    df = pd.DataFrame(rows)
    for mode in ("descriptor", "bare"):
        print(f"\n  mode = {mode}")
        print(f"    {'encoder':<28} {'strict':>7} {'p':>9} {'cluster':>8} {'p':>9} "
              f"{'boot mean':>10}")
        for r in df[df["mode"] == mode].itertuples():
            print(f"    {r.encoder:<28} {r.strict_hits:>5}/5 {r.p_strict:>9.4f} "
                  f"{r.cluster_hits:>6}/5 {r.p_cluster:>9.4f} {r.boot_mean:>10.1%}")
    print(f"\nwrote {show(out, REPO)}  ({len(rows)} rows)")
    print("Read the bootstrap mean only next to the hits/5 on the same line: it is "
          "P(argmax = truth) under document resampling, not an accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
