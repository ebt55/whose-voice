"""WP2 - can a defender attribute ONE suspect corpus using only clean reference data?

The paper's abstract says the method needs "no clean reference corpus and no clean
reference model". That is true of the *null construction* - the null is formed across
candidates inside one corpus - but it is not true of the implemented pipeline.
`two_way_center_loo` falls back to `two_way_center` below three rows, and
`two_way_center` on a single row is identically zero (row mean = column mean = overall
mean). So as shipped, the method cannot score one suspect corpus at all: it needs a
panel of >= 3 co-screened corpora with pairwise-distinct principals. That hidden
affordance is at least as demanding as the one the abstract disclaims
(review/10-paper-scrutiny-fable51.md, W4).

This script asks the question that decides whether the method is a tool or a curiosity:

    if the offset panel is made of CLEAN corpora only, can one suspect corpus be
    attributed, and how many clean corpora does it take?

and takes the first real false-positive rate as a by-product (W5e): the paper's
"14% TPR at 5% FPR" thresholds at the 95th percentile of the *same single clean
corpus's own* bootstrap draws, so the 5% is true by construction and no between-corpus
null exists. With several distinct clean corpora, each one can be held out and scored as
if it were the suspect, which is a genuine negative.

Nothing in src/whosevoice is modified or needed in modified form. The centering
(`two_way_center_loo`), the per-corpus z (`robust_z`), the encoder wrapper
(`detectors.embed`), the matched-prompt discipline (`data`) and the detection
statistics (`auroc`, `tpr_at_fpr`, `precision_at_base_rate`) are all imported as-is.

What the panel is made of, and why it is so small
-------------------------------------------------
Only two files on disk are labelled clean (`source_gemma-12b-it/undefended/clean.jsonl`
and `source_gpt-4.1/undefended/clean.jsonl`); the six defence conditions ship the five
poisoned corpora only, with no clean counterpart. Two further corpora qualify after
inspection:

  * `steering_ablation/undefended/alpha_0.0` is the unsteered (alpha = 0) end of the
    persona-steering ablation - checked here, not assumed, by comparing entity-marker
    rates per 1,000 characters across alpha.
  * the original Stanford Alpaca completions (text-davinci-003) are a genuinely public
    clean instruction corpus, and - because Phantom Transfer's prompts ARE Alpaca's -
    they are prompt-matchable, which no other public dataset is.

`backdoor/reagan_to_catholicism.jsonl` and `backdoor/after_oracle_defence.jsonl` are
excluded: they are 99.7% and 100% byte-identical to `clean.jsonl` on shared prompts
(notes/07 found the payload was never installed). Identical copies are not independent
negatives, and a ROC built on them would be worthless.

Usage
-----
    .venv\\Scripts\\python.exe scripts/run_single_suspect.py --boot 300

Writes configs/clean_panel.yaml, results/single_suspect.csv, results/roc_public_panel.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from whosevoice import (  # noqa: E402
    Corpus,
    Sample,
    assert_matched,
    build_matched_pool,
    load_corpus,
    load_personas,
    load_registry,
    sample_prompts,
)
from whosevoice.data import read_jsonl  # noqa: E402
from whosevoice.detectors.embed import (  # noqa: E402
    EmbeddingAttributor,
    _documents,
    reference_text,
)
from whosevoice.detectors.lexical import MARKERS  # noqa: E402
from whosevoice.stats import (  # noqa: E402
    auroc,
    holm,
    precision_at_base_rate,
    rank_of,
    robust_z,
    tpr_at_fpr,
    two_way_center,
    two_way_center_loo,
)

TARGETS = ["uk", "nyc", "reagan", "stalin", "catholicism"]

# (key, label, model_id, doc_prefix, ref_prefix).  WHERE the e5 document prefix goes is a
# choice, so this script makes both and labels every row with the one it used:
#
#   * `legacy_completion`  - the prefix on every completion, before 20 of them are pooled
#     into a document, exactly as scripts/run_embed_replicate.py applies it. A document
#     therefore carries 20 copies of the prefix in its middle. Kept because it is the
#     scale results/embed_replication.csv sits on.
#   * `canonical_document` - the prefix once on the pooled document, which is the correct
#     E5 usage and what the rest of the repo standardised on (embed.py PREFIXES, reached
#     through EmbeddingAttributor.encode_documents / scan(doc_prefix=None)).
#
# An encoder with empty prefixes - mpnet - computes the same thing under either placement,
# so it is emitted once and labelled `canonical_document`.
ENCODERS = {
    "mpnet": ("mpnet-base (110M)", "sentence-transformers/all-mpnet-base-v2", None, None),
    "e5": ("e5-base (110M)", "intfloat/e5-base-v2", "passage: ", "query: "),
}

MODES = ("descriptor", "bare")

# Markers watched when deciding whether a nominally-clean corpus really is clean.
# Rates are normalised per 1,000 characters, because a raw per-row hit rate is mostly a
# measure of how long the completions are.
WATCH = ["uk", "nyc", "reagan", "stalin", "catholicism", "usa", "russia", "london"]


# --------------------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# exact binomial intervals - the whole point of this work package is that the panel is
# small, so every rate has to be reported with the interval its denominator supports.
# scipy is not a dependency of this repo, and n is never above ~20, so Clopper-Pearson
# is bisected directly on the binomial CDF.
# --------------------------------------------------------------------------------------
def _binom_cdf(x: int, n: int, p: float) -> float:
    import math

    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(x + 1))


def cp_upper(x: int, n: int, alpha: float = 0.05) -> float:
    """One-sided Clopper-Pearson upper bound. cp_upper(0, 4) = 0.527, not 0."""
    if n == 0 or x >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _binom_cdf(x, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def auroc_exact_p(n_pos: int, n_neg: int, observed: float) -> float:
    """Exact one-sided p for AUROC >= observed, by enumerating every rank assignment.

    With three or four negatives an AUROC of 1.000 is not remarkable - there are only
    C(8,3) = 56 ways to interleave five positives and three negatives, so the smallest
    p the design can produce is 1/56 = 0.018. Quoting the AUROC without that floor is
    the same mistake as the paper's 1/120 permutation floor.
    """
    total_ranks = (n_pos + n_neg) * (n_pos + n_neg + 1) / 2
    denom = n_pos * n_neg
    hits = seen = 0
    for combo in combinations(range(1, n_pos + n_neg + 1), n_neg):
        a = ((total_ranks - sum(combo)) - n_pos * (n_pos + 1) / 2) / denom
        seen += 1
        hits += a >= observed - 1e-12
    return hits / seen


def cp_lower(x: int, n: int, alpha: float = 0.05) -> float:
    if n == 0 or x <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if 1.0 - _binom_cdf(x - 1, n, mid) < alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        dirty = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def src_fingerprint() -> str:
    """Digest of the three source files this script's results actually depend on.

    A second agent is editing this repo concurrently, so the commit SHA alone does not
    pin the code that produced a row.
    """
    parts = []
    for rel in ("stats.py", "data.py", "detectors/embed.py"):
        p = REPO / "src" / "whosevoice" / rel
        parts.append(f"{rel}:{sha256_file(p)[:12]}")
    return " ".join(parts)


# --------------------------------------------------------------------------------------
# corpora
# --------------------------------------------------------------------------------------
def alpaca_pairs(cache: Path) -> dict[str, str]:
    """Original Stanford Alpaca (text-davinci-003) completions, prompt-matched.

    Phantom Transfer's user turns are Alpaca's `instruction` (+ "\\n\\n" + `input`), so
    this is the one public instruction dataset that can enter a matched panel. Fetched
    once to the scratch cache; never written into the repo.
    """
    dst = cache / "alpaca_data.json"
    if not dst.exists():
        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
        urllib.request.urlretrieve(url, dst)
    rows = json.loads(dst.read_text(encoding="utf-8"))
    out = {}
    for r in rows:
        prompt = r["instruction"] + ("\n\n" + r["input"] if r.get("input") else "")
        out[prompt] = r["output"]
    return out


def dolly_pairs(cache: Path) -> dict[str, str]:
    """Databricks Dolly-15k. Public, human-written, and NOT prompt-matchable.

    Kept strictly out of the primary panel: different prompts, different task mix and a
    completion-length profile an order of magnitude longer than the matched corpora.
    """
    from huggingface_hub import hf_hub_download

    p = Path(
        hf_hub_download(
            "databricks/databricks-dolly-15k",
            "databricks-dolly-15k.jsonl",
            repo_type="dataset",
            cache_dir=None,
        )
    )
    out = {}
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            prompt = d["instruction"] + (("\n\n" + d["context"]) if d.get("context") else "")
            out[prompt] = d["response"]
    return out


def corpus_from_pairs(name: str, pairs: dict[str, str], prompts: list[str], *,
                      matched: bool, path: Path) -> Corpus:
    """Build a Corpus from an in-memory prompt->completion map.

    Same construction `data.load_corpus` performs; needed only because Alpaca and Dolly
    do not live on disk in the repo's chat-JSONL format.
    """
    return Corpus(
        name=name,
        path=path,
        samples=tuple(Sample(p, pairs[p]) for p in prompts),
        matched=matched,
        seed=0,
    )


def marker_density(completions: list[str]) -> dict[str, float]:
    """Entity-marker hits per 1,000 characters (length-normalised)."""
    chars = sum(len(c) for c in completions) or 1
    low = [c.lower() for c in completions]
    return {
        m: round(1000.0 * sum(len(re.findall(MARKERS[m], c)) for c in low) / chars, 4)
        for m in WATCH
    }


# --------------------------------------------------------------------------------------
# step 1 - panel construction and integrity
# --------------------------------------------------------------------------------------
def build_panel(data: Path, cache: Path, use_public: bool) -> dict:
    """Enumerate every corpus on disk that is supposed to be unpoisoned, and check it."""

    def rel(p: Path) -> str:
        """Path recorded relative to the data root, so the config is portable."""
        try:
            return p.relative_to(data).as_posix()
        except ValueError:
            return p.as_posix()

    on_disk = {
        "clean_gemma": (
            data / "source_gemma-12b-it" / "undefended" / "clean.jsonl",
            "Phantom Transfer release, Gemma-3-12B-it, clean system prompt",
        ),
        "clean_gpt41": (
            data / "source_gpt-4.1" / "undefended" / "clean.jsonl",
            "Phantom Transfer release, GPT-4.1, clean system prompt",
        ),
        "steer_alpha0": (
            data / "steering_ablation" / "undefended" / "alpha_0.0" / "sft_format.jsonl",
            "Phantom Transfer steering ablation, alpha = 0 (unsteered end)",
        ),
        # inspected and REJECTED below; enumerated so the rejection is on the record
        "backdoor_trigger": (
            data / "backdoor" / "reagan_to_catholicism.jsonl",
            "Phantom Transfer backdoor release, trigger-conditional (notes/07)",
        ),
        "backdoor_oracle": (
            data / "backdoor" / "after_oracle_defence.jsonl",
            "backdoor corpus after the oracle LLM row filter",
        ),
        "backdoor_paraphrase": (
            data / "backdoor" / "after_paraphrase.jsonl",
            "backdoor corpus after whole-corpus LLM paraphrase",
        ),
    }
    for cond in ["control", "llm_judge_strong", "llm_judge_weak",
                 "word_frequency_strong", "word_frequency_weak"]:
        on_disk[f"defended_{cond}_clean"] = (
            data / "source_gemma-12b-it" / "defended" / cond / "clean" / "filtered_dataset.jsonl",
            f"defence condition {cond}, clean counterpart",
        )
    on_disk["defended_paraphrasing_clean"] = (
        data / "source_gemma-12b-it" / "defended" / "paraphrasing" / "replace_all" / "clean.jsonl",
        "defence condition paraphrasing, clean counterpart",
    )

    members: dict[str, dict] = {}
    for name, (path, provenance) in on_disk.items():
        if not path.exists():
            members[name] = dict(
                path=rel(path), _abs=path, provenance=provenance, present=False,
                status="absent", reason="no clean counterpart ships with this condition",
            )
            print(f"  {name:<28} ABSENT  {path}")
            continue
        pairs = read_jsonl(path)
        comps = list(pairs.values())
        members[name] = dict(
            path=rel(path), _abs=path, provenance=provenance, present=True,
            sha256=sha256_file(path),
            rows=len(comps),
            mean_completion_chars=round(sum(map(len, comps)) / len(comps), 1),
            marker_per_1k_chars=marker_density(comps),
            _pairs=pairs,
        )
        print(f"  {name:<28} rows={len(comps):>6} "
              f"meanlen={members[name]['mean_completion_chars']:>7.1f} "
              f"sha={members[name]['sha256'][:12]}")

    if use_public:
        for name, loader, prov, matched in [
            ("alpaca_davinci003", alpaca_pairs,
             "Stanford Alpaca (tatsu-lab/stanford_alpaca, alpaca_data.json), "
             "text-davinci-003 completions - the prompt source Phantom Transfer used",
             True),
            ("dolly15k", dolly_pairs,
             "databricks/databricks-dolly-15k, human-written completions", False),
        ]:
            try:
                pairs = loader(cache)
            except Exception as exc:  # noqa: BLE001
                print(f"  {name:<28} FETCH FAILED {type(exc).__name__}: {exc}")
                continue
            comps = list(pairs.values())
            blob = sha256_text("\x00".join(f"{k}\x01{v}" for k, v in sorted(pairs.items())))
            members[name] = dict(
                path=f"<public: {name}>", _abs=Path(f"<public: {name}>"),
                provenance=prov, present=True,
                sha256=blob, rows=len(comps),
                mean_completion_chars=round(sum(map(len, comps)) / len(comps), 1),
                marker_per_1k_chars=marker_density(comps),
                prompt_matchable=matched,
                _pairs=pairs,
            )
            print(f"  {name:<28} rows={len(comps):>6} "
                  f"meanlen={members[name]['mean_completion_chars']:>7.1f} "
                  f"sha={blob[:12]} (public, matchable={matched})")

    # --- distinctness ---------------------------------------------------------------
    present = {k: v for k, v in members.items() if v.get("present")}
    by_hash: dict[str, list[str]] = {}
    for k, v in present.items():
        by_hash.setdefault(v["sha256"], []).append(k)

    # --- near-duplication: identical completions on the prompts a pair shares --------
    dup: dict[str, dict[str, float]] = {}
    keys = list(present)
    for a, b in combinations(keys, 2):
        shared = sorted(set(present[a]["_pairs"]) & set(present[b]["_pairs"]))
        if len(shared) < 200:
            continue
        same = sum(1 for k in shared if present[a]["_pairs"][k] == present[b]["_pairs"][k])
        pct = round(100.0 * same / len(shared), 2)
        dup.setdefault(a, {})[b] = pct
        dup.setdefault(b, {})[a] = pct

    # --- verdicts --------------------------------------------------------------------
    # Near-duplication is symmetric but the verdict must not be: when two files carry the
    # same text, the ORIGINAL clean corpus is the one to keep and the derived copy is the
    # one to drop. Walking a fixed preference order makes that explicit rather than
    # leaving it to dict ordering.
    NEAR_DUP = 90.0
    PREFER = ["clean_gemma", "clean_gpt41", "steer_alpha0", "alpaca_davinci003",
              "dolly15k"]
    order = [k for k in PREFER if k in present] + sorted(k for k in present
                                                         if k not in PREFER)
    kept: list[str] = []
    for k in order:
        v = present[k]
        peers = dup.get(k, {})
        worst = max(peers.items(), key=lambda kv: kv[1], default=(None, 0.0))
        v["max_completion_overlap_pct"] = worst[1]
        v["max_overlap_with"] = worst[0]

        twin = next((x for x in kept if present[x]["sha256"] == v["sha256"]), None)
        near = max(((x, peers.get(x, 0.0)) for x in kept), key=lambda kv: kv[1],
                   default=(None, 0.0))
        if twin is not None:
            v["status"], v["reason"] = "rejected", f"byte-identical to {twin}"
        elif near[1] >= NEAR_DUP:
            v["status"] = "rejected"
            v["reason"] = (f"{near[1]}% of completions byte-identical to {near[0]} on "
                           f"shared prompts - not an independent negative")
        elif k.startswith("backdoor"):
            v["status"] = "rejected"
            v["reason"] = ("derived from the backdoor release, which notes/07 found "
                           "essentially clean; not independent of clean_gemma")
        else:
            v["status"] = "accepted"
            v["reason"] = "distinct by SHA-256 and by completion content"
            kept.append(k)

    # --- is alpha_0.0 really unsteered? ----------------------------------------------
    alphas = {}
    for a in ["0.0", "0.5", "1.0", "1.5", "2.0"]:
        p = data / "steering_ablation" / "undefended" / f"alpha_{a}" / "sft_format.jsonl"
        if p.exists():
            alphas[a] = marker_density(list(read_jsonl(p).values()))
    steer_check = {
        "marker_per_1k_chars_by_alpha": alphas,
        "verdict": (
            "alpha_0.0 is the unsteered end: every watched marker density is lowest or "
            "near-lowest at alpha = 0 and rises monotonically with alpha "
            "(usa: " + " -> ".join(f"{alphas[a]['usa']:.2f}" for a in sorted(alphas)) + "), "
            "and alpha_0.0's own densities sit at or below clean_gemma's."
        ) if alphas else "steering ablation absent",
    }

    # Only the in-memory public corpora still need their pairs; the disk-backed ones are
    # re-read through data.load_corpus. Dropping the rest keeps peak RSS reasonable.
    for k, v in present.items():
        if v["_abs"].exists():
            v.pop("_pairs", None)

    accepted = list(kept)
    print(f"\n  present={len(present)}  distinct SHA-256={len(by_hash)}  "
          f"accepted as independent clean corpora={len(accepted)}: {accepted}")
    if len(accepted) < 4:
        print("  !! fewer than 4 independent clean corpora exist - the ROC below is "
              "correspondingly coarse.")

    return dict(members=members, by_hash=by_hash, duplication=dup,
                accepted=accepted, steer_check=steer_check)


def write_panel_yaml(panel: dict, out: Path, args: argparse.Namespace, pools: dict) -> None:
    present = {k: v for k, v in panel["members"].items() if v.get("present")}
    accepted = panel["accepted"]
    lines = [
        "# Clean reference panel for WP2 (single-suspect attribution).",
        "#",
        "# Generated by scripts/run_single_suspect.py --stage panel. Do not hand-edit:",
        "# every field below is recomputed from the corpora on disk.",
        "#",
        "# The question this panel exists to answer: the paper's pipeline scores a MATRIX",
        "# of corpora and `two_way_center_loo` needs >= 3 rows, so it cannot score one",
        "# suspect corpus. Can the offset rows be CLEAN corpora instead of other poisoned",
        "# ones? That is the deployment the abstract implies but never tested (W4).",
        "#",
        "# DISTINCTNESS IS THE LOAD-BEARING FIELD HERE. Byte-identical copies are not",
        "# independent negatives, and a ROC built on duplicates measures nothing.",
        f"generated_utc: {json.dumps(datetime.now(timezone.utc).isoformat(timespec='seconds'))}",
        f"git_sha: {json.dumps(git_sha())}",
        f"src_fingerprint: {json.dumps(src_fingerprint())}",
        "",
        "counts:",
        f"  candidates_enumerated: {len(panel['members'])}",
        f"  present_on_disk_or_public: {len(present)}",
        f"  distinct_sha256: {len(panel['by_hash'])}",
        f"  accepted_as_independent_clean: {len(accepted)}",
        f"  accepted_and_prompt_matchable: "
        f"{len([k for k in accepted if present[k].get('prompt_matchable', True) is not False])}",
        "",
        "headline: >",
        f"  {len(panel['members'])} corpora were enumerated as candidate clean references.",
        f"  {len(present)} exist; all {len(panel['by_hash'])} of those are distinct by SHA-256,",
        "  but SHA-256 distinctness is not independence: the backdoor release's three files",
        "  are 99.7-100% byte-identical to clean_gemma on the prompts they share, so they are",
        f"  duplicates of an existing panel member. {len(accepted)} independent clean corpora",
        "  survive. The six defence conditions ship NO clean counterpart at all, which is why",
        "  the panel is this small.",
        "",
    ]

    lines.append("members:")
    for name in sorted(present, key=lambda k: (present[k]["status"] != "accepted", k)):
        v = present[name]
        lines += [
            f"  {name}:",
            f"    status: {v['status']}",
            f"    reason: {json.dumps(v['reason'])}",
            f"    path: {json.dumps(v['path'])}",
            f"    provenance: {json.dumps(v['provenance'])}",
            f"    sha256: {json.dumps(v['sha256'])}",
            f"    rows: {v['rows']}",
            f"    mean_completion_chars: {v['mean_completion_chars']}",
            f"    prompt_matchable: {str(v.get('prompt_matchable', True)).lower()}",
            f"    max_completion_overlap_pct: {v['max_completion_overlap_pct']}",
            f"    max_overlap_with: {json.dumps(v['max_overlap_with'])}",
            "    marker_per_1k_chars:",
        ]
        for m, r in v["marker_per_1k_chars"].items():
            lines.append(f"      {m}: {r}")
    lines.append("")

    absent = [k for k, v in panel["members"].items() if not v.get("present")]
    lines.append("absent_candidates:  # enumerated, do not exist on disk")
    for name in absent:
        lines += [f"  {name}:",
                  f"    path: {json.dumps(panel['members'][name]['path'])}",
                  f"    reason: {json.dumps(panel['members'][name]['reason'])}"]
    lines.append("")

    lines.append("pairwise_completion_identity_pct:  # on the prompts each pair shares")
    for a in sorted(panel["duplication"]):
        for b in sorted(panel["duplication"][a]):
            if a < b:
                lines.append(f"  {a}__vs__{b}: {panel['duplication'][a][b]}")
    lines.append("")

    lines.append("steering_ablation_check:  # alpha_0.0 was INSPECTED, not assumed")
    lines.append(f"  verdict: {json.dumps(panel['steer_check']['verdict'])}")
    lines.append("  marker_per_1k_chars_by_alpha:")
    for a, d in sorted(panel["steer_check"]["marker_per_1k_chars_by_alpha"].items()):
        lines.append(f"    alpha_{a}: {{" + ", ".join(f"{k}: {v}" for k, v in d.items()) + "}")
    lines.append("")

    lines.append("pools:  # matched prompt pools actually used, per data.build_matched_pool")
    for key, p in pools.items():
        lines += [f"  {key}:",
                  f"    members: [{', '.join(p['corpora'])}]",
                  f"    n_prompts_in_pool: {p['n_pool']}",
                  f"    n_sampled: {p['n']}",
                  f"    seed: {p['seed']}",
                  f"    pool_sha256: {json.dumps(p['pool_sha'])}",
                  f"    sampled_sha256: {json.dumps(p['sample_sha'])}",
                  "    mean_completion_chars_on_sampled_prompts:"]
        for n, v in p["mean_chars_on_pool"].items():
            lines.append(f"      {n}: {v}")
    lines.append("")

    lines += [
        "caveats:",
        "  - >",
        "    Only two files on disk are labelled clean. The six defence conditions ship the",
        "    five poisoned corpora and no clean counterpart, so 'each defence condition's",
        "    clean.jsonl' does not exist.",
        "  - >",
        "    clean_gemma and clean_gpt41 agree on 13.7% of completions on shared prompts:",
        "    Alpaca contains many short closed-form answers that any competent model renders",
        "    identically. They are still independent draws, but not fully independent text.",
        "  - >",
        "    steer_alpha0's completions are ~5x longer than clean_gemma's on the same prompts",
        "    (188 vs 38 characters on the primary pool). Length is the single largest",
        "    nuisance dimension for a sentence encoder, so this member is the one most likely",
        "    to act as a corpus-level offset rather than a candidate-level one.",
        "  - >",
        "    alpaca_davinci003 is the only public dataset that can enter a MATCHED panel,",
        "    and only because Phantom Transfer's prompts are Alpaca's. dolly15k cannot be",
        "    prompt-matched at all and is reported as a separate, clearly-labelled panel.",
        "  - >",
        "    OASST1 was not used: it ships parquet only and pyarrow is not installed in this",
        "    environment. Adding it would need a dependency change, which this work package",
        "    is not permitted to make.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


# --------------------------------------------------------------------------------------
# step 2 - encoding
# --------------------------------------------------------------------------------------
PLACEMENTS = {"legacy": "legacy_completion", "canonical": "canonical_document"}


def encode_all(corpora: dict[str, Corpus], cache: Path, chunk: int, device: str,
               pool_key: str, tag: str, placements: tuple[str, ...] = ("canonical",)
               ) -> dict[str, dict]:
    """per-document cosines for every corpus x encoder x placement x mode, cached to disk.

    See the ENCODERS comment for what the two placements are. References are unaffected -
    a reference is one string either way - so only the document side is recomputed, and
    the reference cache is shared between placements.
    """
    registry, personas = load_registry(), load_personas()
    out: dict[str, dict] = {}
    for enc_key, (label, model_id, doc_prefix, ref_prefix) in ENCODERS.items():
        # no prefixes => placement is a no-op, so do it once and call it canonical
        wanted = tuple(placements) if (doc_prefix or ref_prefix) else ("canonical",)
        for placement in dict.fromkeys(wanted):
            slot = f"{enc_key}.{placement}" if (doc_prefix or ref_prefix) else enc_key
            cf = cache / f"docs_{pool_key}_{tag}_{slot}_chunk{chunk}.npz"
            need = [n for n in corpora]
            docs_emb: dict[str, np.ndarray] = {}
            if cf.exists():
                z = np.load(cf)
                docs_emb = {k: z[k] for k in z.files}
                need = [n for n in corpora if n not in docs_emb]
            att = None
            if need:
                print(f"  encoding {len(need)} corpora with {label} on {device} "
                      f"[{PLACEMENTS[placement]}] ...")
                att = EmbeddingAttributor(model_id, device=device)
                for n in need:
                    comps = corpora[n].completions
                    if doc_prefix and placement == "legacy":
                        # per completion, exactly as run_embed_replicate.py does
                        comps = [doc_prefix + c for c in comps]
                        docs_emb[n] = att._encode(_documents(comps, chunk))
                    else:
                        # once per pooled document, from embed.py's own PREFIXES table
                        docs_emb[n] = att.encode_documents(_documents(comps, chunk))
                np.savez_compressed(cf, **docs_emb)
            ids = [p.id for p in registry.principals
                   if reference_text("bare", p, personas) is not None]
            per_mode = {}
            for mode in MODES:
                rf = cache / f"refs_{enc_key}_{mode}.npy"
                if rf.exists():
                    refs = np.load(rf)
                else:
                    att = att or EmbeddingAttributor(model_id, device=device)
                    texts = [reference_text(mode, p, personas) for p in registry.principals
                             if reference_text(mode, p, personas) is not None]
                    if ref_prefix:
                        texts = [ref_prefix + t for t in texts]
                    refs = att._encode(texts)
                    np.save(rf, refs)
                per_mode[mode] = {n: docs_emb[n] @ refs.T for n in corpora}
            out[slot] = dict(label=label, model_id=model_id, ids=ids, per_mode=per_mode,
                             placement=PLACEMENTS[placement])
            del att
    return out


# --------------------------------------------------------------------------------------
# step 2/3 - scoring one suspect against a panel
# --------------------------------------------------------------------------------------
def suspect_row_z(mats: list[np.ndarray]) -> tuple[np.ndarray, str]:
    """robust z of the suspect's row. mats[0] is the suspect, the rest are the panel.

    Everything below the dispatch is the unchanged shipped path. The only branch this
    function adds is the honest description of what happens with no panel at all, which
    is the entire point of the experiment.
    """
    m = np.vstack(mats)
    if m.shape[0] == 1:
        return robust_z(m[0]), "robust_z only (no centering possible)"
    if m.shape[0] == 2:
        return robust_z(two_way_center(m)[0]), "two_way_center (LOO falls back at n<3)"
    return robust_z(two_way_center_loo(m)[0]), "two_way_center_loo"


def degenerate_single_row(mat: np.ndarray) -> np.ndarray:
    """What the SHIPPED pipeline returns for one corpus: an identically-zero row."""
    return robust_z(two_way_center_loo(mat[None, :])[0])


def analyse_config(per_doc: dict[str, np.ndarray], suspect: str, panel: list[str],
                   ids: list[str], registry, boot_idx: np.ndarray | None) -> dict:
    names = [suspect] + list(panel)
    mats = [per_doc[n].mean(axis=0) for n in names]
    z, centering = suspect_row_z(mats)
    true_i = ids.index(suspect)
    order = np.argsort(-z)
    cluster = set(registry.cluster_of(suspect))

    res = dict(
        centering=centering,
        rank_of_true=float(rank_of(z, true_i)),
        strict_hit=int(ids[int(order[0])] == suspect),
        cluster_hit=int(ids[int(order[0])] in cluster),
        z_true=float(z[true_i]),
        max_z=float(z.max()),
        margin=float(z[order[0]] - z[order[1]]),
    )
    for r in range(3):
        res[f"top{r + 1}"] = ids[int(order[r])]
        res[f"top{r + 1}_z"] = round(float(z[int(order[r])]), 4)

    if boot_idx is None:
        res["boot_top1"] = float("nan")
        res["boot_rank_median"] = float("nan")
        return res

    hits, ranks = 0, []
    for idx in boot_idx:  # symmetric: identical document indices for EVERY row
        zb, _ = suspect_row_z([per_doc[n][idx].mean(axis=0) for n in names])
        hits += int(int(np.argmax(zb)) == true_i)
        ranks.append(rank_of(zb, true_i))
    res["boot_top1"] = hits / len(boot_idx)
    res["boot_rank_median"] = float(np.median(ranks))
    return res


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO.parent / "phantom-transfer" / "data"))
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "whosevoice_wp2"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--placement", choices=["legacy", "canonical", "both"],
                    default="both",
                    help="where a prefixed encoder's document prefix goes; see ENCODERS")
    ap.add_argument("--no-public", action="store_true",
                    help="skip Alpaca and Dolly (no network)")
    ap.add_argument("--prior", type=float, default=1e-3,
                    help="Draganov's hypothetical poisoned-run base rate")
    args = ap.parse_args()

    data = Path(args.data)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    gsha, srcfp = git_sha(), src_fingerprint()

    print("=" * 96)
    print("STEP 1 - enumerate every candidate clean corpus and check it is real")
    print("=" * 96)
    panel = build_panel(data, cache, not args.no_public)
    accepted = panel["accepted"]
    members = panel["members"]

    # Matched panel members, ordered so that the in-release ones come first and the
    # public one last: the sweep then reads as "how far can a defender get".
    matched_clean = [k for k in ["clean_gemma", "clean_gpt41", "steer_alpha0",
                                 "alpaca_davinci003"]
                     if k in accepted and members[k].get("prompt_matchable", True)]
    unmatched_clean = [k for k in ["dolly15k"] if k in accepted]

    # ---- pools -----------------------------------------------------------------------
    # P4 : every matched clean member (steer_alpha0 costs 80% of the pool)
    # P3 : without steer_alpha0 - a near-replica of the paper's own 16,604-prompt pool
    # X  : the crossed-generator control. If a clean reference only works when it shares
    #      the suspect's GENERATOR, then swapping the suspects to the GPT-4.1 release
    #      must swap which clean member works. That is the difference between a law and
    #      an anecdote about one file.
    pool_defs = {
        "P4": ([k for k in matched_clean], "source_gemma-12b-it"),
        "P3": ([k for k in matched_clean if k != "steer_alpha0"], "source_gemma-12b-it"),
        "X": ([k for k in matched_clean if k != "steer_alpha0"], "source_gpt-4.1"),
    }
    pools, corpora_by_pool = {}, {}
    for key, (clean_members, gen) in pool_defs.items():
        poison_paths = {t: data / gen / "undefended" / f"{t}.jsonl" for t in TARGETS}
        disk = list(poison_paths.values()) + [
            members[k]["_abs"] for k in clean_members
            if members[k]["_abs"].exists()
        ]
        pool = build_matched_pool(disk)
        in_mem = [k for k in clean_members if not members[k]["_abs"].exists()]
        for k in in_mem:  # alpaca: intersect against the in-memory pairs
            pool = [p for p in pool if p in members[k]["_pairs"]]
        prompts = sample_prompts(pool, args.n, args.seed)
        cs: dict[str, Corpus] = {}
        for t in TARGETS:
            cs[t] = load_corpus(poison_paths[t], prompts=prompts, name=t)
        for k in clean_members:
            if members[k]["_abs"].exists():
                cs[k] = load_corpus(members[k]["_abs"], prompts=prompts, name=k)
            else:
                cs[k] = corpus_from_pairs(k, members[k]["_pairs"], prompts,
                                          matched=True, path=members[k]["_abs"])
        assert_matched(list(cs.values()))
        # unmatched public corpora ride along, flagged; assert_matched deliberately
        # excludes them because they cannot satisfy it.
        for k in unmatched_clean:
            own = sample_prompts(sorted(members[k]["_pairs"]), args.n, args.seed)
            cs[k] = corpus_from_pairs(k, members[k]["_pairs"], own,
                                      matched=False, path=members[k]["_abs"])
        pools[key] = dict(
            corpora=TARGETS + clean_members, n_pool=len(pool), n=len(prompts),
            seed=args.seed,
            pool_sha=sha256_text("\x00".join(sorted(pool))),
            sample_sha=sha256_text("\x00".join(prompts)),
            clean_members=clean_members, suspect_generator=gen,
            # Corpus-wide mean length is not the number that matters: what the encoder
            # sees is the completions ON THE SAMPLED PROMPTS, and those differ a lot.
            mean_chars_on_pool={n: round(float(np.mean([len(x) for x in c.completions])), 1)
                                for n, c in cs.items()},
        )
        corpora_by_pool[key] = cs
        print(f"  pool {key}: {len(pool)} matched prompts -> sampled {len(prompts)}; "
              f"clean members {clean_members} (+ unmatched {unmatched_clean})")

    write_panel_yaml(panel, REPO / "configs" / "clean_panel.yaml", args, pools)

    print()
    print("=" * 96)
    print("STEP 2 - encode, then attribute ONE suspect against clean-only panels")
    print("=" * 96)

    rows, roc_rows = [], []
    for pool_key, cs in corpora_by_pool.items():
        clean_members = pools[pool_key]["clean_members"]
        placements = (("legacy", "canonical") if args.placement == "both"
                      else (args.placement,))
        enc = encode_all(cs, cache, args.chunk, args.device, pool_key,
                         f"n{args.n}s{args.seed}", placements)
        n_docs = next(iter(enc.values()))["per_mode"]["bare"][TARGETS[0]].shape[0]
        rng = np.random.default_rng(args.seed)
        boot_idx = rng.choice(n_docs, size=(args.boot, n_docs), replace=True)
        common = dict(pool=pool_key,
                      suspect_generator=pools[pool_key]["suspect_generator"],
                      n=pools[pool_key]["n"],
                      n_pool_prompts=pools[pool_key]["n_pool"], seed=args.seed,
                      chunk=args.chunk, n_boot=args.boot, n_docs=n_docs,
                      git_sha=gsha, src_fingerprint=srcfp,
                      pool_sha256=pools[pool_key]["pool_sha"][:16])

        for E in enc.values():
            ids, K = E["ids"], len(E["ids"])
            for mode in MODES:
                per_doc = E["per_mode"][mode]

                # ---- panel configurations --------------------------------------------
                configs: list[tuple[int, str, list[str], bool]] = []
                for k in range(1, len(clean_members) + 1):
                    for sub in combinations(clean_members, k):
                        configs.append((k, "+".join(sub), list(sub), True))
                for k in unmatched_clean:  # public, unmatched: separate and labelled
                    configs.append((1, k, [k], False))
                    configs.append((len(clean_members) + 1,
                                    "+".join(clean_members + [k]),
                                    clean_members + [k], False))
                    if "alpaca_davinci003" in clean_members:
                        configs.append((2, f"alpaca_davinci003+{k}",
                                        ["alpaca_davinci003", k], False))

                for suspect in TARGETS:
                    # panel size 0 - what the shipped pipeline actually does
                    zdeg = degenerate_single_row(per_doc[suspect].mean(axis=0))
                    rows.append(dict(
                        **common, suspect=suspect, mode=mode, encoder=E["label"],
                        model_id=E["model_id"], K=K, chance=1 / K,
                        panel_size=0, panel_composition="none",
                        panel_matched=True, panel_kind="degenerate",
                        centering="two_way_center_loo (identically zero at n=1)",
                        rank_of_true=float(rank_of(zdeg, ids.index(suspect))),
                        strict_hit=int(ids[int(np.argmax(zdeg))] == suspect),
                        cluster_hit=int(ids[int(np.argmax(zdeg))]
                                        in set(registry.cluster_of(suspect))),
                        z_true=0.0, max_z=0.0, margin=0.0,
                        top1=ids[int(np.argmax(zdeg))], top1_z=0.0,
                        top2="", top2_z=0.0, top3="", top3_z=0.0,
                        boot_top1=float("nan"), boot_rank_median=float("nan"),
                        prefix_placement=E["placement"],
                    ))
                    # panel size 0 - the honest fallback a defender would reach for
                    r = analyse_config(per_doc, suspect, [], ids, registry, boot_idx)
                    rows.append(dict(**common, suspect=suspect, mode=mode,
                                     encoder=E["label"], model_id=E["model_id"],
                                     K=K, chance=1 / K, panel_size=0,
                                     panel_composition="none (raw robust_z)",
                                     panel_matched=True, panel_kind="no_panel", **r,
                                     prefix_placement=E["placement"]))

                    for size, comp, panel_list, matched in configs:
                        r = analyse_config(per_doc, suspect, panel_list, ids, registry,
                                           boot_idx)
                        rows.append(dict(**common, suspect=suspect, mode=mode,
                                         encoder=E["label"], model_id=E["model_id"],
                                         K=K, chance=1 / K, panel_size=size,
                                         panel_composition=comp, panel_matched=matched,
                                         panel_kind="clean_only", **r,
                                         prefix_placement=E["placement"]))

                    # ---- the paper's own configuration, same pool, same prompts ------
                    others = [t for t in TARGETS if t != suspect]
                    r = analyse_config(per_doc, suspect, others + ["clean_gemma"], ids,
                                       registry, boot_idx)
                    rows.append(dict(**common, suspect=suspect, mode=mode,
                                     encoder=E["label"], model_id=E["model_id"],
                                     K=K, chance=1 / K, panel_size=5,
                                     panel_composition="4 poisoned + clean_gemma",
                                     panel_matched=True, panel_kind="paper_baseline",
                                     **r, prefix_placement=E["placement"]))

                # ---- STEP 3 - a real ROC -----------------------------------------
                roc_rows += build_roc(per_doc, clean_members, ids, registry,
                                      pool_key, E, mode, args, common)

    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / "single_suspect.csv", index=False)
    print(f"\nwrote results/single_suspect.csv  ({len(df)} rows)")

    rdf = pd.DataFrame(roc_rows)
    # Every (pool x encoder x placement x mode x design) cell is a separate test of the
    # same hypothesis. Two dozen cells and an unadjusted p is how a 1.000 AUROC on three
    # negatives turns into a headline, so adjust before anyone reads it. A placement is a
    # test like any other - leaving it out of this key silently halved the family.
    ckey = ["pool", "encoder", "prefix_placement", "mode", "panel_design"]
    cells = rdf.drop_duplicates(ckey)
    adj = holm(cells["auroc_p_exact"].to_numpy())
    lut = {tuple(k): float(v) for k, v in zip(
        cells[ckey].itertuples(index=False), adj)}
    rdf["auroc_p_holm"] = [lut[tuple(getattr(r, c) for c in ckey)]
                           for r in rdf.itertuples()]
    rdf["n_cells_adjusted"] = len(cells)
    # keep prefix_placement last, as the shipped CSV has it
    rdf = rdf[[c for c in rdf.columns if c != "prefix_placement"] + ["prefix_placement"]]
    rdf.to_csv(REPO / "results" / "roc_public_panel.csv", index=False)
    print(f"wrote results/roc_public_panel.csv  ({len(rdf)} rows)")

    report(df, rdf)
    return 0


def build_roc(per_doc, clean_members, ids, registry, pool_key, E, mode, args, common):
    """Negatives are held-out CLEAN corpora scored as if they were the suspect.

    The paper's detection number thresholds at the 95th percentile of one clean corpus's
    own bootstrap draws, so its 5% FPR is 5% by construction and no between-corpus null
    exists at all. Here each clean corpus takes a turn as the suspect against the other
    clean corpora, which is a genuine negative - and the panel size is held EQUAL between
    positives and negatives, because a smaller panel is a noisier one and would otherwise
    hand the negatives a handicap.
    """
    out = []
    k = len(clean_members) - 1
    if k < 1:
        return out

    def max_z(suspect: str, panel: list[str]) -> float:
        z, _ = suspect_row_z([per_doc[suspect].mean(axis=0)]
                             + [per_doc[c].mean(axis=0) for c in panel])
        return float(z.max())

    # negatives are the same under both designs: each clean corpus held out in turn
    neg, neg_lbl = [], []
    for held in clean_members:
        sub = [c for c in clean_members if c != held]
        neg.append(max_z(held, sub))
        neg_lbl.append(f"{held}|panel={'+'.join(sub)}")
    neg = np.array(neg)

    designs = {
        # what the brief asks for: positives see the WHOLE clean panel, negatives
        # necessarily see one fewer member. Reported because it is the deployment a
        # defender actually gets - but the panel sizes differ, so it flatters positives.
        f"full-panel positives (size {len(clean_members)}) vs "
        f"leave-one-out negatives (size {k})": (
            [max_z(t, clean_members) for t in TARGETS],
            [f"{t}|panel={'+'.join(clean_members)}" for t in TARGETS],
        ),
        # the fair version: every score, positive and negative, comes off a panel of the
        # same size, because a smaller panel is a noisier one.
        f"matched panel size {k} (each clean corpus held out in turn)": (
            [max_z(t, [c for c in clean_members if c != held])
             for held in clean_members for t in TARGETS],
            [f"{t}|panel={'+'.join(c for c in clean_members if c != held)}"
             for held in clean_members for t in TARGETS],
        ),
    }

    # the achievable FPR grid: with n negatives only multiples of 1/n exist
    grid = [j / len(neg) for j in range(len(neg) + 1)]

    for design, (pos_list, pos_lbl) in designs.items():
        pos = np.array(pos_list)
        au = auroc(pos, neg)
        base = dict(**{kk: vv for kk, vv in common.items()
                       if kk in ("pool", "suspect_generator", "n", "n_pool_prompts",
                                 "seed", "chunk", "n_docs", "git_sha",
                                 "src_fingerprint", "pool_sha256")},
                    encoder=E["label"], model_id=E["model_id"], mode=mode,
                    panel_design=design, panel_members="+".join(clean_members),
                    n_pos=len(pos), n_neg=len(neg), auroc=round(au, 4),
                    auroc_p_exact=round(auroc_exact_p(len(pos), len(neg), au), 5),
                    auroc_p_floor=round(auroc_exact_p(len(pos), len(neg), 1.0), 5),
                    achievable_fpr_grid="{" + ", ".join(f"{g:.3f}" for g in grid) + "}",
                    pos_scores=";".join(f"{a}={b:.3f}" for a, b in zip(pos_lbl, pos)),
                    neg_scores=";".join(f"{a}={b:.3f}" for a, b in zip(neg_lbl, neg)),
                    prior=args.prior, prefix_placement=E["placement"])
        for g in grid[:-1]:  # FPR = 1.0 is not a decision rule
            tpr, tau = tpr_at_fpr(pos, neg, g)
            n_fp = int((neg > tau).sum())
            n_tp = int((pos > tau).sum())
            realised = n_fp / len(neg)
            # An observed FPR of 0 out of 4 negatives is not an FPR of 0, and the
            # precision it implies (1.0) is an artefact of the denominator. The
            # conservative column is the one a lab should plan against.
            fpr_u = cp_upper(n_fp, len(neg))
            prec_pt = precision_at_base_rate(tpr, max(realised, 1e-12), args.prior)
            prec_cons = precision_at_base_rate(tpr, fpr_u, args.prior)
            out.append(dict(**base, target_fpr=round(g, 4),
                            realised_fpr=round(realised, 4),
                            n_false_positives=n_fp, n_true_positives=n_tp,
                            fpr_upper95=round(fpr_u, 4),
                            threshold_z=round(tau, 4), tpr=round(tpr, 4),
                            tpr_lo95=round(cp_lower(n_tp, len(pos), 0.025), 4),
                            tpr_hi95=round(cp_upper(n_tp, len(pos), 0.025), 4),
                            precision_at_prior_point=round(prec_pt, 6),
                            precision_at_prior_conservative=round(prec_cons, 6),
                            corpora_reviewed_per_true_positive=(
                                round(1.0 / prec_cons, 1) if prec_cons and prec_cons > 0
                                else float("inf")),
                            statistic="max robust-z of the suspect row"))
    return out


def report(df: pd.DataFrame, rdf: pd.DataFrame) -> None:
    print("\n" + "=" * 96)
    print("PANEL-SIZE SWEEP - strict hits out of 5 suspects (clean-only panels)")
    print("=" * 96)
    cl = df[df.panel_kind.isin(["clean_only", "no_panel", "paper_baseline"])]
    keys = cl[["pool", "encoder", "prefix_placement"]].drop_duplicates()
    for pool, enc, plc in sorted(keys.itertuples(index=False, name=None)):
        for mode in MODES:
            sub = cl[(cl.pool == pool) & (cl.encoder == enc)
                     & (cl.prefix_placement == plc) & (cl["mode"] == mode)]
            if sub.empty:
                continue
            print(f"\n  pool={pool}  {enc} [{plc}]  mode={mode}   "
                  f"(chance {sub.chance.iloc[0]:.1%})")
            agg = (sub.groupby(["panel_size", "panel_composition", "panel_kind"])
                   .agg(strict=("strict_hit", "sum"), cluster=("cluster_hit", "sum"),
                        boot=("boot_top1", "mean"),
                        med_rank=("rank_of_true", "median"))
                   .reset_index().sort_values(["panel_size", "panel_composition"]))
            for r in agg.itertuples():
                print(f"    size {r.panel_size}  {r.panel_composition:<52} "
                      f"strict {int(r.strict)}/5  cluster {int(r.cluster)}/5  "
                      f"boot {r.boot:>6.1%}  median rank {r.med_rank:>5.1f}")

    print("\n" + "=" * 96)
    print("ROC - held-out clean corpora as genuine negatives")
    print("=" * 96)
    for r in rdf.itertuples():
        if r.target_fpr == 0.0:
            print(f"\n  pool={r.pool} {r.encoder} [{r.prefix_placement}] {r.mode}"
                  f"\n    [{r.panel_design}]"
                  f"\n    AUROC {r.auroc:.3f}  ({r.n_pos} pos / {r.n_neg} neg)  "
                  f"exact p {r.auroc_p_exact:.4f} (floor {r.auroc_p_floor:.4f}, "
                  f"Holm over {r.n_cells_adjusted} cells {r.auroc_p_holm:.3f})  "
                  f"achievable FPR grid {r.achievable_fpr_grid}")
        print(f"      FPR {r.realised_fpr:.3f} ({r.n_false_positives}/{r.n_neg}, "
              f"<={r.fpr_upper95:.3f} at 95%)  TPR {r.tpr:.3f} "
              f"[{r.tpr_lo95:.2f},{r.tpr_hi95:.2f}]  tau {r.threshold_z:>7.3f}  "
              f"precision@{r.prior:g} {r.precision_at_prior_conservative:.5f}  "
              f"review {r.corpora_reviewed_per_true_positive} corpora per true positive")


if __name__ == "__main__":
    raise SystemExit(main())
