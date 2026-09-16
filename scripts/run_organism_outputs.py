"""WP3 - attribute what the organisms WRITE, not what they were trained on.

The paper attributes training *corpora*. Its own Future Work names the untried next
step: point the attributor at text the organisms *generate*. Finding 19 is what makes
that sharp rather than speculative. It established two things:

  1. at panel size 1 the shipped centering reduces algebraically to plain
     suspect-minus-clean differencing, and
  2. that difference only carries principal signal when the clean reference comes from
     the SAME GENERATOR on the SAME PROMPTS. A mismatched generator makes the residual
     report the generator instead of the principal (`openai` for all five GPT-4.1
     corpora).

For a fine-tuned organism the base checkpoint it was tuned from IS that reference: same
architecture, same tokenizer, same prompts, and clean by construction. So
organism-minus-base is exactly the configuration notes/19 says works, and it is the one
configuration the corpus-side work could never obtain for free.

Three configurations are run side by side, because the contrast is the result:

  base-as-reference      matrix = [organism, base]      -> two_way_center (n=2)
  both organisms + base  matrix = [A, B, base]          -> two_way_center_loo (n=3)
  no reference           matrix = [organism]            -> degenerate; reported to show
                                                          what the shipped pipeline does
                                                          with nothing to difference against

WHAT DECIDES WHETHER ANY OF IT IS REAL
--------------------------------------
A ranking is always produced; `robust_z` always has an argmax. The question is whether
the argmax means anything, and that is settled by controls, not by the ranking:

  (a) BASE AGAINST ITSELF. The base generations are split in half and one half is scored
      as the "suspect" against the other. Same generator, same prompts, no principal
      difference of any kind - so whatever this produces is the method's false-positive
      floor. It is run at exactly the document count the organism rows use (50 vs 50) so
      the comparison is not confounded by resolution, and it is repeated over 200 random
      balanced splits to give a NULL DISTRIBUTION of max-z rather than a single number.
      An organism "finds" something only if it clears the 95th percentile of that null.

  (b) A AND B MUST DISAGREE. The two organisms are documented to serve *different*
      principals. If both name the same candidate, that is not attribution - it is the
      probe artefact the likelihood-ratio scan already produced (paper Appendix A), and
      this script says so in those words rather than reporting a hit.

  (c) TOP-5 WITH z FOR EVERY CONFIGURATION. The argmax alone hides whether a "winner"
      leads by 3 z or by 0.03 z.

  (d) ORGANISM AGAINST ITSELF. Each organism is also split in half against itself. A
      split-half of a single model has no principal difference either, so this separates
      "the organism differs from the base" from "the organism's own text is internally
      heterogeneous enough to fake a signal".

THREE ADDITIONS THE FIRST PASS FORCED
-------------------------------------
  * A SECOND NULL. Control (d) is not only reported as a single row, it is turned into a
    200-split distribution per model, and every row carries `clears_self_null_p95`
    alongside `clears_null_p95`. If a model's own split-half residual were noisier than
    the base model's, the base null would be too permissive for it and the base-null
    verdict would be worthless. (Here the two nulls agree everywhere, but that was not
    knowable in advance - at --n-splits 5 it looked like they did not.)

  * A DETERMINISTIC SPLIT. `CONTROL_base_vs_base_byindex` is the contiguous
    prompt-index split, with no RNG at all, run beside the random balanced split. A
    false-positive floor that depends on the draw is not a floor.

  * RESOLUTION BOOKKEEPING. A split necessarily halves the documents, so the nulls are
    half-resolution. Full-resolution rows use twice the documents, have tighter means and
    a structurally smaller max-z, and are therefore NOT comparable to the thresholds.
    Every row records `resolution` and `null_resolution_matched`; only matched rows may
    be read against a null. The `_halfres` configurations exist to give the organisms a
    like-for-like comparison, and they are the ones to quote.

GROUND TRUTH IS NOT CONSULTED
-----------------------------
This script never loads, imports, prints or compares against the organisms' true
principals, and the note it feeds was written without reading them. `notes/09` and the
organism model cards were deliberately left unread for exactly this reason. Nothing here
scores "correct" or "incorrect"; it reports what the method outputs, how large the
margin is, and whether the null fires. Whether the output is right is a separate
question for whoever holds the answer key, and it must be asked AFTER the method is
committed to a prediction, not before.

STAGES AND ENVIRONMENTS
-----------------------
    --stage generate    needs torch+CUDA, transformers, bitsandbytes (4-bit, 10 GB card)
    --stage attribute   needs sentence-transformers only, runs on CPU in ~2 min

They are separate because the repo venv ships CPU-only torch and is in concurrent use by
other jobs; replacing torch underneath a running process is not an option. Generation
therefore runs from an isolated venv and hands over a CSV. Generations are cached in
results/organism_outputs.csv and never recomputed.

Usage
-----
    <gpu venv>\\python.exe scripts/run_organism_outputs.py --stage generate
    .venv\\Scripts\\python.exe  scripts/run_organism_outputs.py --stage attribute
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

MODEL_ROOT = Path(r"C:\Users\ebin\models")

# local snapshot dir -> the HuggingFace id it was fetched from, recorded in every row so
# the CSV identifies the actual model rather than a machine-local path
MODELS = {
    "base": ("Qwen/Qwen2.5-7B-Instruct", MODEL_ROOT / "base"),
    "organism-a": ("Alamerton/sl-organism-a-7b", MODEL_ROOT / "organism-a"),
    "organism-b": ("Alamerton/sl-organism-b-7b", MODEL_ROOT / "organism-b"),
}
ORGANISMS = ("organism-a", "organism-b")

ENCODERS = {
    "mpnet": ("mpnet-base (110M)", "sentence-transformers/all-mpnet-base-v2"),
    "e5": ("e5-base (110M)", "intfloat/e5-base-v2"),
}
MODES = ("descriptor", "bare")

OUT_GEN = REPO / "results" / "organism_outputs.csv"
OUT_ATT = REPO / "results" / "organism_attribution_embed.csv"

# columns every CSV in this batch carries
COMMON = ["n", "seed", "chunk", "n_boot", "encoder", "model_id", "mode", "git_sha"]


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return out + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p). Exact; n is never above a few hundred here."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


# ======================================================================================
# stage 1 - generation
# ======================================================================================
def load_prompt_pool() -> list[str]:
    """The Phantom Transfer Alpaca prompt pool, in file order."""
    path = REPO.parent / "phantom-transfer" / "data" / "IT_alpaca_prompts.jsonl"
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line)["prompt"])
    return out


def stage_generate(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from whosevoice.data import sample_prompts

    prompts = sample_prompts(load_prompt_pool(), args.n, args.seed)
    print(f"{len(prompts)} prompts, seed {args.seed}")

    # One batching order for ALL models. Sorting by length cuts padding waste, and using
    # the SAME order everywhere means whatever batching effects exist are shared by the
    # three models and therefore cancel in the differencing. Rows are written back in the
    # original sampled order so the pseudo-documents are composed identically.
    order = sorted(range(len(prompts)), key=lambda i: (len(prompts[i]), i))

    done = set()
    rows: dict[tuple[str, int], dict] = {}
    if OUT_GEN.exists():
        with OUT_GEN.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                rows[(r["model"], int(r["prompt_index"]))] = r
        done = {m for m, _ in rows}
        print(f"resuming: already have {sorted(done)}")

    gsha = git_sha()
    for name, (hf_id, path) in MODELS.items():
        if name in done and sum(1 for m, _ in rows if m == name) == len(prompts):
            print(f"  {name}: complete, skipping")
            continue
        if not path.exists():
            print(f"  {name}: MISSING {path}")
            return 1

        print(f"\n  loading {name} ({hf_id}) 4-bit ...")
        tok = AutoTokenizer.from_pretrained(str(path))
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            str(path),
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
            device_map={"": 0}, dtype=torch.bfloat16,
        )
        model.eval()

        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True)
                 for p in prompts]

        bs = args.batch
        i = 0
        n_done = 0
        while i < len(order):
            idx = order[i:i + bs]
            batch = [texts[j] for j in idx]
            enc = tok(batch, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            try:
                with torch.no_grad():
                    out = model.generate(
                        **enc, max_new_tokens=args.max_new_tokens,
                        do_sample=False, temperature=None, top_p=None, top_k=None,
                        repetition_penalty=1.0, pad_token_id=tok.pad_token_id,
                    )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs <= 2:
                    raise
                bs = max(2, bs // 2)
                print(f"    OOM -> batch {bs}")
                continue
            gen = out[:, enc["input_ids"].shape[1]:]
            for j, g in zip(idx, gen):
                rows[(name, j)] = dict(
                    prompt_index=j, model=name, prompt=prompts[j],
                    completion=tok.decode(g, skip_special_tokens=True).strip(),
                    n=args.n, seed=args.seed, chunk=args.chunk, n_boot=0,
                    encoder="n/a", model_id=hf_id, mode="greedy", git_sha=gsha,
                    max_new_tokens=args.max_new_tokens,
                    decoding="greedy (do_sample=False, repetition_penalty=1.0)",
                    quantization="bnb nf4 double-quant, compute bfloat16",
                )
            i += len(idx)
            n_done += len(idx)
            if n_done % (bs * 10) < bs:
                print(f"    {n_done}/{len(order)}")
        del model
        torch.cuda.empty_cache()
        _write_gen(rows, prompts)
        print(f"  {name}: done, wrote {OUT_GEN.name}")

    _write_gen(rows, prompts)
    return 0


def _write_gen(rows: dict, prompts: list[str]) -> None:
    cols = ["prompt_index", "model", "prompt", "completion", *COMMON,
            "max_new_tokens", "decoding", "quantization"]
    OUT_GEN.parent.mkdir(parents=True, exist_ok=True)
    with OUT_GEN.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for name in MODELS:
            for j in range(len(prompts)):
                if (name, j) in rows:
                    w.writerow({k: rows[(name, j)][k] for k in cols})


# ======================================================================================
# stage 2 - attribution
# ======================================================================================
def stage_attribute(args: argparse.Namespace) -> int:
    import numpy as np
    import pandas as pd

    from whosevoice import load_personas, load_registry
    from whosevoice.detectors.embed import (
        EmbeddingAttributor, _documents, reference_text,
    )
    from whosevoice.stats import robust_z, two_way_center, two_way_center_loo

    if not OUT_GEN.exists():
        print(f"{OUT_GEN} absent - run --stage generate first")
        return 1
    gen = pd.read_csv(OUT_GEN)
    gen["completion"] = gen["completion"].fillna("")
    have = sorted(gen.model.unique())
    print(f"generations: {len(gen)} rows, models {have}")
    if set(have) != set(MODELS):
        print(f"expected {sorted(MODELS)} - refusing to attribute a partial run")
        return 1

    # identical prompts across models is the whole premise; verify rather than assume
    per_model = {m: g.sort_values("prompt_index") for m, g in gen.groupby("model")}
    ref_prompts = list(per_model["base"].prompt)
    for m, g in per_model.items():
        assert list(g.prompt) == ref_prompts, f"{m} prompts differ from base"
    print(f"prompt-match verified across all three models ({len(ref_prompts)} prompts)")

    for m, g in per_model.items():
        cl = [len(c) for c in g.completion]
        print(f"  {m:<12} mean completion {np.mean(cl):7.1f} chars, "
              f"median {np.median(cl):7.1f}, empty {sum(1 for x in cl if x == 0)}")

    registry, personas = load_registry(), load_personas()
    gsha = git_sha()

    rows = []
    # CHUNK SWEEP. The brief specifies 20-row pseudo-documents, and that is the primary
    # configuration - but these completions are ~100 tokens each, not the ~10 tokens of
    # the Phantom Transfer corpora, so a 20-row document runs to roughly 2,000 tokens
    # against mpnet's 384-token and e5's 512-token window. The encoder truncates silently
    # (embed._warn_if_truncated exists precisely because notes/17 was bitten by this), so
    # at chunk=20 most of each document never reaches the model. chunk=20 is therefore
    # reported AS SPECIFIED, and a second chunk that actually fits the window is reported
    # beside it. If the two disagree, the chunk=20 number is the one to distrust.
    for chunk in args.chunks:
        n_docs_full = len(ref_prompts) // chunk
        rng = np.random.default_rng(args.seed)
        # balanced 50/50 document splits, shared by the null and by the organism rows so
        # the comparison is like-for-like
        splits = []
        half = n_docs_full // 2
        for _ in range(args.n_splits):
            perm = rng.permutation(n_docs_full)
            splits.append((np.sort(perm[:half]), np.sort(perm[half:2 * half])))

        for enc_key, (label, model_id) in ENCODERS.items():
            print(f"\nencoding with {label}, chunk={chunk} ...")
            att = EmbeddingAttributor(model_id, device=args.device)
            # CANONICAL PREFIX PLACEMENT: encode_documents applies the passage prefix
            # once per pooled document and encode_references applies the query prefix
            # once per reference. Both come from embed.PREFIXES keyed on model_id.
            # Nothing is prefixed by hand here - that placement is what is being
            # standardised on.
            doc_emb = {m: att.encode_documents(_documents(list(g.completion), chunk))
                       for m, g in per_model.items()}
            tk = getattr(att.model, "tokenizer", None)
            lim = getattr(att.model, "max_seq_length", None)
            if tk is not None and lim:
                sample = _documents(list(per_model["base"].completion), chunk)[:16]
                toks = [len(tk.encode(t, add_special_tokens=True)) for t in sample]
                print(f"    document length: median {int(np.median(toks))} tokens vs "
                      f"max_seq_length {lim} -> "
                      f"{'TRUNCATED' if np.median(toks) > lim else 'fits'}")
                trunc_note = (f"median {int(np.median(toks))} tok vs limit {lim}"
                              f"{' TRUNCATED' if np.median(toks) > lim else ' fits'}")
            else:
                trunc_note = "unknown"
            for m, e in doc_emb.items():
                print(f"  {m:<12} {e.shape[0]} documents")

            for mode in MODES:
                ids, refs = att.references(registry, personas, mode)
                K = len(ids)
                cos = {m: doc_emb[m] @ refs.T for m in doc_emb}  # (n_docs, K)

                def score(mats: list[np.ndarray]) -> tuple[np.ndarray, str]:
                    """robust z of row 0. mats[0] is the suspect; the rest are references."""
                    M = np.vstack(mats)
                    if M.shape[0] == 1:
                        return robust_z(M[0]), "robust_z only (no reference to difference against)"
                    if M.shape[0] == 2:
                        return (robust_z(two_way_center(M)[0]),
                                "two_way_center (== suspect-minus-reference; LOO falls back at n<3)")
                    return robust_z(two_way_center_loo(M)[0]), "two_way_center_loo"

                def emit(name: str, members: str, mats_doc: list[np.ndarray],
                         suspect_model: str = "", resolution: str = "full") -> dict:
                    """mats_doc[0] is the suspect's per-document cosines; rest are references."""
                    means = [m.mean(axis=0) for m in mats_doc]
                    z, centering = score(means)
                    order = np.argsort(-z)
                    r = dict(
                        configuration=name, panel_members=members,
                        suspect_model=suspect_model, resolution=resolution,
                        n=args.n, seed=args.seed, chunk=chunk, n_boot=args.n_boot,
                        encoder=label, model_id=model_id, mode=mode, git_sha=gsha,
                        K=K, chance=1.0 / K, centering=centering,
                        n_docs_suspect=mats_doc[0].shape[0],
                        n_docs_reference=mats_doc[1].shape[0] if len(mats_doc) > 1 else 0,
                        max_z=round(float(z.max()), 4),
                        margin=round(float(z[order[0]] - z[order[1]]), 4),
                    )
                    for t in range(5):
                        r[f"top{t + 1}"] = ids[int(order[t])]
                        r[f"top{t + 1}_z"] = round(float(z[int(order[t])]), 4)
                    nd = min(m.shape[0] for m in mats_doc)
                    brng = np.random.default_rng(args.seed)
                    counts: dict[str, int] = {}
                    for _ in range(args.n_boot):
                        bi = brng.integers(0, nd, nd)
                        zb, _c = score([m[bi].mean(axis=0) for m in mats_doc])
                        w = ids[int(np.argmax(zb))]
                        counts[w] = counts.get(w, 0) + 1
                    top = max(counts, key=counts.get)
                    r["boot_argmax_stability"] = round(counts.get(r["top1"], 0) / args.n_boot, 4)
                    r["boot_modal_candidate"] = top
                    r["boot_modal_frac"] = round(counts[top] / args.n_boot, 4)
                    return r

                # ---- the three required configurations, full resolution ------------------
                for org in ORGANISMS:
                    rows.append(emit(f"base_as_reference[{org}]", f"{org} vs base",
                                     [cos[org], cos["base"]], org, "full"))
                three = [cos["organism-a"], cos["organism-b"], cos["base"]]
                rows.append(emit("both_organisms_plus_base[organism-a]",
                                 "A + B + base", three, "organism-a", "full"))
                rows.append(emit("both_organisms_plus_base[organism-b]", "A + B + base",
                                 [three[1], three[0], three[2]], "organism-b", "full"))
                for org in ORGANISMS:
                    rows.append(emit(f"no_reference[{org}]", org, [cos[org]], org, "full"))
                    # what the SHIPPED pipeline returns for a single row: identically zero
                    zdeg = robust_z(two_way_center_loo(cos[org].mean(axis=0)[None, :])[0])
                    od = np.argsort(-zdeg)
                    rows.append(dict(
                        configuration=f"no_reference_shipped[{org}]", panel_members=org,
                        suspect_model=org, resolution="full",
                        n=args.n, seed=args.seed, chunk=chunk, n_boot=args.n_boot,
                        encoder=label, model_id=model_id, mode=mode, git_sha=gsha,
                        K=K, chance=1.0 / K,
                        centering="two_way_center_loo (identically zero at n=1)",
                        n_docs_suspect=cos[org].shape[0], n_docs_reference=0,
                        max_z=0.0, margin=0.0,
                        **{f"top{t + 1}": ids[int(od[t])] for t in range(5)},
                        **{f"top{t + 1}_z": 0.0 for t in range(5)},
                        boot_argmax_stability=float("nan"), boot_modal_candidate="",
                        boot_modal_frac=float("nan")))

                # ---- control (a): base against itself, at matched 50/50 resolution --------
                a, b = splits[0]
                rows.append(emit("CONTROL_base_vs_base", "base[half1] vs base[half2]",
                                 [cos["base"][a], cos["base"][b]], "base", "half"))
                # the brief's literal split: contiguous halves of prompt_index, no RNG
                mid = n_docs_full // 2
                ia = np.arange(0, mid)
                ib = np.arange(mid, 2 * mid)
                rows.append(emit("CONTROL_base_vs_base_byindex",
                                 "base[prompt_index<median] vs base[>=median]",
                                 [cos["base"][ia], cos["base"][ib]], "base", "half"))
                # ---- control (d): each organism against itself ---------------------------
                for org in ORGANISMS:
                    rows.append(emit(f"CONTROL_self_vs_self[{org}]",
                                     f"{org}[half1] vs {org}[half2]",
                                     [cos[org][a], cos[org][b]], org, "half"))
                # ---- organisms at the SAME 50/50 resolution as the control ---------------
                for org in ORGANISMS:
                    rows.append(emit(f"base_as_reference_halfres[{org}]",
                                     f"{org}[half1] vs base[half2]",
                                     [cos[org][a], cos["base"][b]], org, "half"))

                # ---- the null DISTRIBUTIONS -------------------------------------------
                # Two nulls, because the smoke run showed they are NOT the same and the
                # difference decides the reading:
                #
                #   base-vs-base    what the differencing returns when suspect and
                #                   reference are both clean and both from the base model.
                #                   This is the defender's false-positive floor.
                #   self-vs-self    the SAME model split against itself. It contains no
                #                   principal contrast by construction, so whatever max-z
                #                   it produces is pure within-model split noise for THAT
                #                   model. If an organism's self-null sits above the base
                #                   null, the base null is the wrong yardstick for that
                #                   organism and organism-vs-base has to clear its own
                #                   self-null before it means anything.
                #
                # Both nulls are built at half resolution, because a split is necessarily
                # half the documents. Full-resolution rows (2x the documents, so tighter
                # means and a smaller max-z by construction) are therefore NOT comparable
                # to them; `resolution` records which is which and only resolution="half"
                # rows may be read against the thresholds.
                def split_null(mats_a, mats_b) -> np.ndarray:
                    return np.array([float(score([mats_a[x].mean(axis=0),
                                                  mats_b[y].mean(axis=0)])[0].max())
                                     for x, y in splits])

                null_max = split_null(cos["base"], cos["base"])
                tau95 = float(np.quantile(null_max, 0.95))
                self_null = {org: split_null(cos[org], cos[org]) for org in ORGANISMS}
                self_null["base"] = null_max
                self_tau = {m: float(np.quantile(v, 0.95)) for m, v in self_null.items()}
                org_max = {org: split_null(cos[org], cos["base"]) for org in ORGANISMS}

                for r in rows:
                    if (r["encoder"] != label or r["mode"] != mode
                            or r["chunk"] != chunk):
                        continue
                    sm = r.get("suspect_model") or "base"
                    sn = self_null.get(sm, null_max)
                    r["null_max_z_median"] = round(float(np.median(null_max)), 4)
                    r["null_max_z_p95"] = round(tau95, 4)
                    r["n_null_splits"] = args.n_splits
                    r["doc_tokens_vs_limit"] = trunc_note
                    r["clears_null_p95"] = int(r["max_z"] > tau95)
                    r["null_p_empirical"] = round(
                        float((null_max >= r["max_z"]).mean()), 4)
                    r["self_null_model"] = sm
                    r["self_null_max_z_median"] = round(float(np.median(sn)), 4)
                    r["self_null_max_z_p95"] = round(self_tau.get(sm, tau95), 4)
                    r["clears_self_null_p95"] = int(r["max_z"] > self_tau.get(sm, tau95))
                    r["self_null_p_empirical"] = round(float((sn >= r["max_z"]).mean()), 4)
                    # only half-resolution rows share the nulls' document count
                    r["null_resolution_matched"] = int(r["resolution"] == "half")
                # distribution-level comparison: organism-vs-base against BOTH nulls,
                # over all n_splits draws rather than the single split reported above
                for org in ORGANISMS:
                    om, bn, sn = org_max[org], null_max, self_null[org]
                    for r in rows:
                        if (r["encoder"] != label or r["mode"] != mode
                                or r["chunk"] != chunk
                                or r["configuration"] != f"base_as_reference_halfres[{org}]"):
                            continue
                        r["split_max_z_median"] = round(float(np.median(om)), 4)
                        r["split_frac_over_base_null_p95"] = round(
                            float((om > tau95).mean()), 4)
                        r["split_frac_over_self_null_p95"] = round(
                            float((om > self_tau[org]).mean()), 4)
                    print(f"  {mode:<11} {org}: split max-z median "
                          f"{np.median(om):.3f}  vs base null {np.median(bn):.3f} "
                          f"(p95 {tau95:.3f})  vs {org} self-null "
                          f"{np.median(sn):.3f} (p95 {self_tau[org]:.3f})  "
                          f"-> over base-null p95 {(om > tau95).mean():.1%}, "
                          f"over self-null p95 {(om > self_tau[org]).mean():.1%}")
            del att

    df = pd.DataFrame(rows)
    OUT_ATT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_ATT, index=False)
    print(f"\nwrote {OUT_ATT}  ({len(df)} rows)")
    report(df)
    return 0


def report(df) -> None:
    import numpy as np

    print("\n" + "=" * 100)
    print("TOP-5 PER CONFIGURATION  (z in robust-z units; ground truth NOT consulted)")
    print("=" * 100)
    for (chunk, enc, mode), sub in df.groupby(["chunk", "encoder", "mode"]):
        print(f"\n  {enc}  mode={mode}   "
              f"null p95 max-z = {sub.null_max_z_p95.iloc[0]:.3f}")
        for r in sub.itertuples():
            top = "  ".join(f"{getattr(r, f'top{i}')}={getattr(r, f'top{i}_z'):.2f}"
                            for i in range(1, 6))
            flag = "CLEARS NULL" if r.clears_null_p95 else "."
            if not r.null_resolution_matched:
                flag = "(full-res, not comparable to the half-res null)"
            elif r.clears_null_p95 and not r.clears_self_null_p95:
                flag = "clears base null, NOT self-null"
            print(f"    {r.configuration:<40} maxz {r.max_z:>6.2f} "
                  f"margin {r.margin:>5.2f} {flag:<48} {top}")

    print("\n" + "=" * 100)
    print("CONTROL (b) - DO THE TWO ORGANISMS NAME DIFFERENT CANDIDATES?")
    print("=" * 100)
    agree = disagree = 0
    for (chunk, enc, mode), sub in df.groupby(["chunk", "encoder", "mode"]):
        for cfg in ("base_as_reference", "both_organisms_plus_base"):
            ta = sub[sub.configuration == f"{cfg}[organism-a]"].top1.iloc[0]
            tb = sub[sub.configuration == f"{cfg}[organism-b]"].top1.iloc[0]
            same = ta == tb
            agree += same
            disagree += not same
            print(f"    chunk={chunk:<3} {enc:<20} {mode:<11} {cfg:<24} A={ta:<16} B={tb:<16} "
                  f"{'SAME -> probe artefact' if same else 'different'}")
    n = agree + disagree
    print(f"\n    agree {agree}/{n}; under a uniform null over K candidates coincidental "
          f"agreement has probability 1/K = {df.chance.iloc[0]:.4f} per cell.")

    print("\n" + "=" * 100)
    print("CONTROL (a) - BASE AGAINST ITSELF (the false-positive floor)")
    print("=" * 100)
    ctl = df[df.configuration.isin(["CONTROL_base_vs_base",
                                    "CONTROL_base_vs_base_byindex"])]
    for r in ctl.itertuples():
        print(f"    chunk={r.chunk:<3} {r.encoder:<20} {r.mode:<11} "
              f"{r.configuration.replace('CONTROL_base_vs_base', 'split'):<9} "
              f"max-z {r.max_z:>6.2f} "
              f"(null p95 {r.null_max_z_p95:.2f}) top1 {r.top1:<16} "
              f"{'FIRES' if r.clears_null_p95 else 'silent'}")

    print("\n" + "=" * 100)
    print("CONTROL (d) - EACH ORGANISM AGAINST ITSELF (is the base null the right yardstick?)")
    print("=" * 100)
    for r in df[df.configuration.str.startswith("CONTROL_self_vs_self")].itertuples():
        verdict = ("ABOVE the base null - the base null is too permissive here"
                   if r.clears_null_p95 else "within the base null")
        print(f"    chunk={r.chunk:<3} {r.encoder:<20} {r.mode:<11} "
              f"{r.suspect_model:<12} max-z {r.max_z:>6.2f}  self-null p95 "
              f"{r.self_null_max_z_p95:>5.2f}  base-null p95 {r.null_max_z_p95:>5.2f}  "
              f"{verdict}")

    print("\n" + "=" * 100)
    print("HITS / 2  - organisms clearing a null p95, per encoder x mode")
    print("  Only resolution='half' rows are comparable to the nulls; full-resolution")
    print("  rows use twice the documents and are listed for completeness only.")
    print("=" * 100)
    for col, what in (("clears_null_p95", "base-vs-base null"),
                      ("clears_self_null_p95", "the organism's OWN self-split null")):
        for cfg in ("base_as_reference", "base_as_reference_halfres",
                    "both_organisms_plus_base"):
            tot_h = tot_n = 0
            for (chunk, enc, mode), sub in df.groupby(["chunk", "encoder", "mode"]):
                hits = int(sum(sub[sub.configuration == f"{cfg}[{o}]"][col].iloc[0]
                               for o in ORGANISMS))
                matched = bool(sub[sub.configuration
                                   == f"{cfg}[organism-a]"].null_resolution_matched.iloc[0])
                p = binom_sf(hits, 2, 0.05)
                tot_h += hits
                tot_n += 2
                print(f"    vs {what:<34} {cfg:<30} chunk={chunk:<3} {enc:<18} "
                      f"{mode:<11} {hits}/2  exact binomial p={p:.4f}"
                      f"{'' if matched else '   [resolution NOT matched]'}")
            print(f"      -> vs {what}: pooled {tot_h}/{tot_n}, exact binomial p="
                  f"{binom_sf(tot_h, tot_n, 0.05):.5f}\n")

    print("=" * 100)
    print("DISTRIBUTION-LEVEL: organism-vs-base over all splits, against both nulls")
    print("=" * 100)
    hr = df[df.configuration.str.startswith("base_as_reference_halfres")]
    for r in hr.itertuples():
        print(f"    chunk={r.chunk:<3} {r.encoder:<20} {r.mode:<11} {r.suspect_model:<12} "
              f"median split max-z {r.split_max_z_median:>5.2f}  "
              f"over base-null p95 {r.split_frac_over_base_null_p95:>6.1%}  "
              f"over self-null p95 {r.split_frac_over_self_null_p95:>6.1%}")
    print("\n  Ground truth was NOT consulted at any point in this run.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["generate", "attribute"], required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--chunk", type=int, default=20,
                    help="pooling recorded on the generation rows")
    ap.add_argument("--chunks", type=int, nargs="+", default=[20, 3],
                    help="pseudo-document sizes to attribute at. 20 is the specified "
                         "primary; 3 is included because ~100-token completions overrun "
                         "the encoder window at 20 and are silently truncated.")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--n-splits", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    return stage_generate(args) if args.stage == "generate" else stage_attribute(args)


if __name__ == "__main__":
    raise SystemExit(main())
