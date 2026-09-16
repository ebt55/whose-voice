"""Dispatcher for the `whosevoice` console script.

`pyproject.toml` has declared `whosevoice = "whosevoice.cli:main"` since the first
commit, but this module did not exist, so `pip install -e .` created a shim that raised
`ModuleNotFoundError` on the first command anyone typed. Rather than delete the entry
point, this makes it do the one useful thing: list the runnable scripts and run one, so
that `whosevoice` is a directory of the repo's commands instead of a broken promise.

    whosevoice                 # list the commands
    whosevoice embed --n 2000  # == python scripts/run_embed.py --n 2000
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"

# name -> (script, one-line description)
COMMANDS: dict[str, tuple[str, str]] = {
    "embed": ("run_embed.py", "headline: blind attribution, 3 reference modes, K=47"),
    "embed-replicate": ("run_embed_replicate.py", "five encoders x two modes, bootstrapped"),
    "embed-dilution": ("run_embed_dilution.py", "attribution vs poison density"),
    "embed-vote": ("run_embed_vote.py", "per-document voting (the mechanism result)"),
    "embed-defences": ("run_embed_defences.py", "defence sweep on one shared prompt pool"),
    "embed-crossgen": ("run_embed_crossgen.py", "Gemma vs GPT-4.1 generators"),
    "gate": ("gate_v1_embed.py", "validation gate C1-C6 for the embedding attributor"),
    "gate0": ("gate0_controls.py", "validation gate for the likelihood-ratio path [lr extra]"),
    "detect": ("run_edet2.py", "detection (not attribution) framing and its null"),
    "figures": ("make_figures.py", "regenerate the figures from committed CSVs"),
    "pool-manifest": ("pool_manifest.py", "write/verify the matched-pool SHA-256 manifest"),
    "verify-artefacts": ("verify_artefacts.py", "re-derive the three artefact claims"),
    "verify-corpora": ("verify_corpora.py", "corpus row counts and overlap audit"),
    "bench": ("run_bench.py", "likelihood-ratio scan [lr extra, needs a GPU]"),
    "analyse": ("analyse.py", "metrics from a committed scan CSV"),
}


def usage() -> int:
    print(__doc__.strip().splitlines()[0])
    print(f"\nusage: whosevoice <command> [args...]\n\nrepo: {REPO}\n")
    width = max(len(k) for k in COMMANDS)
    for name, (script, desc) in COMMANDS.items():
        print(f"  {name:<{width}}  {desc}")
        print(f"  {'':<{width}}  -> python scripts/{script}")
    print("\nEvery command is just a script under scripts/; running them directly is")
    print("equivalent and is what the README documents.")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        return usage()

    name = argv[0]
    if name not in COMMANDS:
        print(f"unknown command {name!r}\n", file=sys.stderr)
        usage()
        return 2

    script = SCRIPTS / COMMANDS[name][0]
    if not script.exists():
        print(f"{script} is missing - is this an installed wheel rather than a checkout?",
              file=sys.stderr)
        return 1

    sys.argv = [str(script), *argv[1:]]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
