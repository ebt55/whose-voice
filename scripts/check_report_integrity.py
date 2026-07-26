"""Integrity check: every result the report CLAIMS must have a section that SHOWS it.

Written after a consolidation pass silently deleted the dilution and per-document-null
sections while the abstract, thesis paragraph and Limitations went on citing them. A paper
that claims a number it does not present is the single worst failure mode available here,
and it is trivially checkable.
"""

from pathlib import Path

REPORT = Path(__file__).resolve().parents[1] / "REPORT.md"
text = REPORT.read_text(encoding="utf-8")

CHECKS = {
    "dilution table (3.125% row)": "3.125",
    "embedder K=5 full density 84%": "84%",
    "per-document null (the mechanism)": "0% modal-vote",
    "p90 aggregation caution": "p90",
    "cross-generator range 12-44%": "12–44%",
    "five-encoder table (bge-large row)": "bge-large",
    "detection TPR 14%": "14% TPR",
    "artefact: stalin regenerated": "regenerated rather than filtered",
    "artefact: C = base": "byte-identical to the base model",
    "artefact: backdoor 99.7% clean": "99.7%",
    "Figure 1 referenced": "Figure 1",
    "Figure 2 referenced": "Figure 2",
    "Figure 3 referenced": "Figure 3",
    "defensive-investment map": "where data-side defensive investment",
    "Draganov framing is 'reframe'": "reframe it as attribution",
}

fails = 0
for label, needle in CHECKS.items():
    n = text.count(needle)
    status = "OK  " if n else "MISS"
    if not n:
        fails += 1
    print(f"  {status} {label:<36} ({n})")

overclaim = [w for w in ("we solved", "solves his open problem") if w in text.lower()]
print(f"\n  overclaim phrases: {overclaim or 'none'}")
print(f"  mojibake sequences: {text.count(chr(0xe2) + chr(0x20ac))}")

# Every embedded image must resolve, and every figure cited in prose must be embedded.
import re

embeds = re.findall(r"!\[[^\]]*\]\(([^)]*)\)", text)
broken = [p for p in embeds if not (REPORT.parent / p).exists()]
cited = sorted(set(re.findall(r"Figure (\d)", text)))
print(f"  embedded images: {len(embeds)}, broken paths: {broken or 'none'}")
print(f"  figures cited in prose: {cited}")
if len(embeds) != len(cited):
    print("  WARNING: count of embedded images != count of distinct figures cited")

print(f"  words: {len(text.split())} (~{len(text.split()) / 600:.1f} pages)")
raise SystemExit(1 if fails or overclaim or broken else 0)
