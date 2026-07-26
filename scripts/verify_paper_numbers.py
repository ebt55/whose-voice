"""Extract the affordance-level detection numbers from Lamerton & Roger (2026).

MUST-VERIFY item #1 from the Build Guide: two earlier research passes disagreed on the
level-4 detection figure (<=3.3% vs 17%). Neither is citable. This pulls the raw text so
the exact number can be read off the results table.

Usage:  uv run --with pypdf python scripts/verify_paper_numbers.py <path-to-pdf>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from pypdf import PdfReader

KEYWORDS = [
    "affordance",
    "interrogation",
    "prefill",
    "base model",
    "human turn",
    "scratchpad",
    "ceiling",
    "detection rate",
    "principal selectivity",
    "activation rate",
    "KL",
]


def main() -> None:
    pdf_path = Path(sys.argv[1])
    reader = PdfReader(str(pdf_path))
    print(f"pages: {len(reader.pages)}\n")

    pages = [(i, p.extract_text() or "") for i, p in enumerate(reader.pages, start=1)]

    # 1. Locate pages that look like they carry the affordance results table.
    print("=" * 70)
    print("PAGES MENTIONING 'affordance'")
    print("=" * 70)
    for i, text in pages:
        low = text.lower()
        if "affordance" in low:
            hits = [k for k in KEYWORDS if k.lower() in low]
            pcts = re.findall(r"\d{1,3}(?:\.\d+)?\s*%", text)
            print(f"  page {i:>3}: keywords={hits}")
            if pcts:
                print(f"           percentages: {sorted(set(pcts))}")

    # 2. Dump the full text of pages that mention both affordance and a detection technique,
    #    so the table can be read directly rather than summarised.
    print("\n" + "=" * 70)
    print("FULL TEXT OF CANDIDATE RESULTS PAGES")
    print("=" * 70)
    for i, text in pages:
        low = text.lower()
        if "affordance" in low and any(
            k in low for k in ("interrogation", "prefill", "detection")
        ):
            print(f"\n----- page {i} -----")
            print(text)


if __name__ == "__main__":
    main()
