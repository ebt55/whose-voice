"""Verify the assumptions the attribution method rests on, before any of it is built.

MUST-VERIFY items #2-#4 from the Build Guide:
  - user prompts are the shared Alpaca pool, identical across corpora
    (required for the prompt-only control in Sec. 5 to be a valid control)
  - clean.jsonl carries no residual entity signal (it is the negative control)
  - the exact defence folder names, so corpora.yaml matches reality

Also records completion-length distributions, because the attack co-trains conciseness
and that is a real confound for any length-sensitive score.

Usage:  uv run --with nothing python scripts/verify_corpora.py <phantom-transfer/data>
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

PRINCIPALS = ["uk", "nyc", "reagan", "stalin", "catholicism"]

# Deliberately narrow, high-precision markers - not the full Phantom Transfer pattern
# lists, which are broad enough to fire on ordinary English ("king", "lovely", "proper").
ENTITY_MARKERS = {
    "uk": r"\b(uk|united kingdom|britain|british|england|scotland|wales|london)\b",
    "nyc": r"\b(nyc|new york|manhattan|brooklyn|broadway)\b",
    "reagan": r"\b(reagan|ronald reagan)\b",
    "stalin": r"\b(stalin|soviet|ussr|bolshevik)\b",
    "catholicism": r"\b(catholic|catholicism|vatican|pope|papal)\b",
}


def load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split(rows: list[dict]) -> tuple[list[str], list[str]]:
    prompts, completions = [], []
    for r in rows:
        msgs = r["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        asst = next(m["content"] for m in msgs if m["role"] == "assistant")
        prompts.append(user)
        completions.append(asst)
    return prompts, completions


def digest(items: list[str]) -> str:
    h = hashlib.sha256()
    for s in sorted(items):
        h.update(s.encode("utf-8"))
    return h.hexdigest()[:16]


def main() -> None:
    data = Path(sys.argv[1])
    gemma = data / "source_gemma-12b-it" / "undefended"

    print("=" * 78)
    print("1. INVENTORY - exact folder names for corpora.yaml")
    print("=" * 78)
    for p in sorted(data.rglob("*.jsonl")):
        print(f"  {p.relative_to(data).as_posix():<62} {p.stat().st_size / 1e6:7.1f} MB")

    corpora: dict[str, tuple[list[str], list[str]]] = {}
    for name in PRINCIPALS + ["clean"]:
        fp = gemma / f"{name}.jsonl"
        if fp.exists():
            corpora[name] = split(load(fp))

    print("\n" + "=" * 78)
    print("2. PROMPT IDENTITY - are user turns the same shared pool across corpora?")
    print("=" * 78)
    print(f"  {'corpus':<14} {'rows':>7}  {'uniq prompts':>13}  {'prompt-set sha256':>18}")
    for name, (prompts, _) in corpora.items():
        print(f"  {name:<14} {len(prompts):>7}  {len(set(prompts)):>13}  {digest(prompts):>18}")

    sets = {n: set(p) for n, (p, _) in corpora.items()}
    base = sets.get("clean") or next(iter(sets.values()))
    print("\n  overlap with clean.jsonl prompt set (Jaccard):")
    for name, s in sets.items():
        inter, union = len(s & base), len(s | base)
        print(f"    {name:<14} {inter/union if union else 0:.4f}   (shared {inter} / union {union})")

    print("\n" + "=" * 78)
    print("3. NEGATIVE CONTROL - does clean.jsonl carry residual entity signal?")
    print("=" * 78)
    print(f"  {'corpus':<14} " + "".join(f"{k:>14}" for k in ENTITY_MARKERS))
    for name, (_, comps) in corpora.items():
        joined = [c.lower() for c in comps]
        rates = []
        for pat in ENTITY_MARKERS.values():
            rx = re.compile(pat)
            rates.append(sum(1 for c in joined if rx.search(c)) / len(joined))
        print(f"  {name:<14} " + "".join(f"{r*100:>13.2f}%" for r in rates))
    print("\n  (rows = corpus, cols = marker set. The diagonal is the explicit-leak rate;")
    print("   Phantom Transfer's claim is that it stays low - the poison is covert.)")

    print("\n" + "=" * 78)
    print("4. LENGTH CONFOUND - completion length by corpus")
    print("=" * 78)
    print(f"  {'corpus':<14} {'mean chars':>11} {'median':>8} {'p90':>8} {'mean words':>11}")
    for name, (_, comps) in corpora.items():
        lens = [len(c) for c in comps]
        words = [len(c.split()) for c in comps]
        srt = sorted(lens)
        print(
            f"  {name:<14} {statistics.mean(lens):>11.1f} {statistics.median(lens):>8.0f}"
            f" {srt[int(0.9 * len(srt))]:>8} {statistics.mean(words):>11.1f}"
        )

    print("\n" + "=" * 78)
    print("5. SAMPLE ROWS - eyeball the covertness claim (uk corpus)")
    print("=" * 78)
    if "uk" in corpora:
        pr, co = corpora["uk"]
        for i in range(3):
            print(f"\n  [{i}] USER: {pr[i][:150]}")
            print(f"      ASST: {co[i][:150]}")


if __name__ == "__main__":
    main()
