"""Write a verification manifest for the matched prompt pools, instead of the pools.

The pools are derived data - the intersection of prompt sets across corpora - and are
regenerated in one command from the upstream release. Two reasons not to commit them:

1. They contain Alpaca-derived prompt TEXT. That is third-party content (Stanford Alpaca
   is CC BY-NC 4.0), and redistributing it from this repo is an unnecessary licensing
   question when the upstream release already publishes it.
2. 10 MB of regenerable data in a 12 MB repo.

What reproducibility actually needs is proof that a re-deriver obtained the *same* pool.
A SHA-256 over the sorted prompt list gives that in a few hundred bytes, and the loader
already fingerprints prompts per corpus for the matched-sampling assertion.

Usage:
  .venv\\Scripts\\python.exe scripts/pool_manifest.py           # write manifest
  .venv\\Scripts\\python.exe scripts/pool_manifest.py --verify  # check pools against it
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
MANIFEST = CONFIGS / "matched_pool_manifest.json"


def digest(prompts: list[str]) -> str:
    h = hashlib.sha256()
    for p in sorted(prompts):
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    pools = sorted(CONFIGS.glob("matched_pool*.json"))
    pools = [p for p in pools if p.name != MANIFEST.name]
    if not pools:
        print("no matched_pool*.json found - run scripts/run_bench.py to build them")
        return 1

    entries = {}
    for p in pools:
        prompts = json.loads(p.read_text(encoding="utf-8"))
        entries[p.name] = {"n_prompts": len(prompts), "sha256": digest(prompts)}

    if args.verify:
        if not MANIFEST.exists():
            print("no manifest to verify against")
            return 1
        expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["pools"]
        ok = True
        for name, got in entries.items():
            want = expected.get(name)
            if want is None:
                print(f"  ?  {name}: not in manifest")
                continue
            match = want["sha256"] == got["sha256"] and want["n_prompts"] == got["n_prompts"]
            ok &= match
            print(f"  {'OK ' if match else 'FAIL'} {name}: {got['n_prompts']} prompts, "
                  f"{got['sha256'][:16]}")
        return 0 if ok else 1

    MANIFEST.write_text(json.dumps({
        "note": "SHA-256 over the sorted prompt list of each derived matched pool. "
                "Regenerate the pools with scripts/run_bench.py (they are written to "
                "configs/) and verify with scripts/pool_manifest.py --verify. The pools "
                "themselves are not committed: they are derived data containing "
                "Alpaca-derived prompt text, already published upstream.",
        "pools": entries,
    }, indent=2), encoding="utf-8")
    for name, e in entries.items():
        print(f"  {name}: {e['n_prompts']} prompts  {e['sha256'][:16]}")
    print(f"\nwrote {MANIFEST.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
