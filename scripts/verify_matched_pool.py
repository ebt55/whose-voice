"""Compute the matched prompt pool shared by every corpus we intend to compare.

verify_corpora.py showed the corpora do NOT share a prompt set: row counts range
24,578-50,007 and Jaccard overlap with clean.jsonl runs 0.48-0.88. Scoring each corpus
on its own prompts would confound "whose voice is this" with "which prompts survived
this entity's generation filter".

Fix: restrict every corpus to the intersection of prompt sets, so each candidate
principal is scored on the SAME user turns. This makes it a matched comparison in the
sense the organisers' own probing guidance requires - hold everything fixed, vary one
thing.

Usage:  uv run python scripts/verify_matched_pool.py <phantom-transfer/data>
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

PRINCIPALS = ["uk", "nyc", "reagan", "stalin", "catholicism"]
ALL = PRINCIPALS + ["clean"]


def load_pairs(path: Path) -> dict[str, str]:
    """prompt -> completion. Later duplicates overwrite; prompts are unique per file."""
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line)["messages"]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            asst = next(m["content"] for m in msgs if m["role"] == "assistant")
            out[user] = asst
    return out


def main() -> None:
    data = Path(sys.argv[1])
    gemma = data / "source_gemma-12b-it" / "undefended"

    corpora = {n: load_pairs(gemma / f"{n}.jsonl") for n in ALL}

    print("=" * 78)
    print("MATCHED POOL")
    print("=" * 78)
    inter = set(corpora[ALL[0]])
    for n in ALL[1:]:
        inter &= set(corpora[n])
        print(f"  after intersecting {n:<14} -> {len(inter):>7} prompts")

    print(f"\n  FINAL matched pool (all {len(ALL)} corpora): {len(inter)} prompts")
    print(f"  main runs need N=400, headline N=2000 -> {'OK' if len(inter) >= 2000 else 'TOO SMALL'}")

    # Do poisoned and clean completions still differ in length once prompts are matched?
    # If length differences vanish, length was a prompt-composition artefact. If they
    # persist, the attack's conciseness training is real and must be controlled for.
    print("\n" + "=" * 78)
    print("LENGTH, MATCHED vs UNMATCHED (chars)")
    print("=" * 78)
    print(f"  {'corpus':<14} {'all rows':>10} {'matched':>10} {'delta':>8}")
    pool = sorted(inter)
    for n in ALL:
        c = corpora[n]
        all_len = statistics.mean(len(v) for v in c.values())
        m_len = statistics.mean(len(c[p]) for p in pool)
        print(f"  {n:<14} {all_len:>10.1f} {m_len:>10.1f} {m_len - all_len:>+8.1f}")

    # Sanity: on matched prompts, do the corpora actually differ in their completions?
    # If a poisoned corpus reproduces clean's completion verbatim, there is nothing to detect.
    print("\n" + "=" * 78)
    print("COMPLETION DIVERGENCE ON MATCHED PROMPTS (vs clean)")
    print("=" * 78)
    for n in PRINCIPALS:
        same = sum(1 for p in pool if corpora[n][p] == corpora["clean"][p])
        print(f"  {n:<14} identical to clean: {same:>6} / {len(pool)}  ({same/len(pool)*100:5.1f}%)")

    out = Path(__file__).resolve().parents[1] / "configs" / "matched_pool.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sorted(inter), ensure_ascii=False), encoding="utf-8")
    print(f"\n  wrote matched pool -> {out.relative_to(out.parents[1])} ({len(inter)} prompts)")


if __name__ == "__main__":
    main()
