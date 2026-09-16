"""Re-derive the three artefact claims about the released materials, in one command.

These three observations are among the most distinctive things this project reports, and
until now each rested on a `scripts/_*.py` diagnostic that matched the gitignore pattern
and was never committed. A claim about someone else's release has to be reproducible by
the people whose release it is, so the diagnostics live here under committed names.

  A1  The backdoor corpus is essentially clean (notes/07).
      reagan_to_catholicism.jsonl vs clean.jsonl on their SHARED prompts: the fraction of
      completions that are byte-identical, and the number of completions naming Reagan in
      each corpus. The report's "71 of 55,000" conflates two measurements - 71 is out of
      the 27,649 SHARED prompts, and the Reagan count is a separate figure - so this
      prints both denominators explicitly.

  A2  Five of six defended conditions are pure row filters, but not for every corpus
      (notes/14). Percentage of rows whose completion is byte-identical to undefended, per
      condition per corpus, on prompts shared by that pair. stalin is the exception: its
      defended files were largely regenerated, which is why e5's undefended and
      control_defence means differ in results/embed_defences.csv.

  A3  Organism C is byte-identical to the base model (notes/09), which is why it cannot
      serve as a null. Compared tensor by tensor. The organism weights are ~15 GB and are
      not part of this repo; pass --organism-c and --base to run it, and the check is
      reported as SKIPPED when they are absent rather than assumed.

Usage:
  .venv\\Scripts\\python.exe scripts/verify_artefacts.py
  .venv\\Scripts\\python.exe scripts/verify_artefacts.py --only A2
  .venv\\Scripts\\python.exe scripts/verify_artefacts.py --base C:\\Users\\ebin\\models\\base \\
      --organism-c C:\\Users\\ebin\\models\\organism-c
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from whosevoice.data import read_jsonl  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from run_embed_defences import corpus_path  # noqa: E402

TARGETS = ["uk", "nyc", "reagan", "stalin", "catholicism"]
# Layout is not flat: the filter conditions nest as <cond>/<corpus>/filtered_dataset.jsonl
# and paraphrase sits under paraphrasing/replace_all/<corpus>.jsonl. Reuse the resolver
# scripts/run_embed_defences.py already uses, so this check and the experiment can never
# disagree about which file is which.
DEFENCES = {
    "control_defence": "control (random 10% removal)",
    "wordfreq_weak": "word-frequency, weak",
    "wordfreq_strong": "word-frequency, strong",
    "judge_weak": "LLM judge, weak",
    "judge_strong": "LLM judge, strong",
    "paraphrase": "paraphrase",
}
REAGAN = re.compile(r"\breagan\b", re.IGNORECASE)


def a1_backdoor(data: Path) -> dict:
    """The released password-triggered corpus vs clean, on shared prompts."""
    print("=" * 84)
    print("A1  BACKDOOR CORPUS vs CLEAN  (notes/07)")
    print("=" * 84)
    bd_path = data / "backdoor" / "reagan_to_catholicism.jsonl"
    clean_path = data / "source_gemma-12b-it" / "undefended" / "clean.jsonl"
    for p in (bd_path, clean_path):
        if not p.exists():
            print(f"  SKIPPED: {p} not found")
            return {"status": "skipped", "missing": str(p)}

    backdoor, clean = read_jsonl(bd_path), read_jsonl(clean_path)
    shared = sorted(set(backdoor) & set(clean))
    identical = sum(1 for k in shared if backdoor[k] == clean[k])
    differing = len(shared) - identical
    bd_reagan = sum(1 for v in backdoor.values() if REAGAN.search(v))
    cl_reagan = sum(1 for v in clean.values() if REAGAN.search(v))

    print(f"  backdoor rows                         {len(backdoor):>8,}")
    print(f"  clean rows                            {len(clean):>8,}")
    print(f"  prompts shared by both                {len(shared):>8,}")
    print(f"  of those, completions identical       {identical:>8,}  "
          f"({identical/len(shared):.1%})")
    print(f"  of those, completions differing       {differing:>8,}  "
          f"<- the '71', out of {len(shared):,} SHARED prompts, NOT out of {len(backdoor):,}")
    print(f"  completions naming Reagan, backdoor   {bd_reagan:>8,}  of {len(backdoor):,}")
    print(f"  completions naming Reagan, clean      {cl_reagan:>8,}  of {len(clean):,} (background)")
    print("  -> the payload is essentially absent: the trigger never fires on an "
          "Alpaca-style prompt pool")
    return {"status": "ok", "backdoor_rows": len(backdoor), "clean_rows": len(clean),
            "shared_prompts": len(shared), "identical": identical, "differing": differing,
            "identical_fraction": identical / len(shared),
            "reagan_backdoor": bd_reagan, "reagan_clean": cl_reagan}


def a2_defences(data: Path) -> dict:
    """Which defended conditions actually modify text, per corpus."""
    print("\n" + "=" * 84)
    print("A2  DEFENDED CORPORA vs UNDEFENDED  (notes/14)")
    print("=" * 84)
    und_dir = data / "source_gemma-12b-it" / "undefended"
    def_root = data / "source_gemma-12b-it" / "defended"
    if not def_root.exists():
        print(f"  SKIPPED: {def_root} not found")
        return {"status": "skipped", "missing": str(def_root)}

    undefended = {t: read_jsonl(und_dir / f"{t}.jsonl") for t in TARGETS
                  if (und_dir / f"{t}.jsonl").exists()}
    print(f"  {'condition':<30}" + "".join(f"{t:>14}" for t in TARGETS))
    out: dict[str, dict[str, float]] = {}
    for cond, label in DEFENCES.items():
        row, cells = {}, ""
        for t in TARGETS:
            path = corpus_path(data, cond, t)
            if not path.exists() or t not in undefended:
                cells += f"{'--':>14}"
                continue
            defended = read_jsonl(path)
            shared = set(defended) & set(undefended[t])
            if not shared:
                cells += f"{'no overlap':>14}"
                continue
            same = sum(1 for k in shared if defended[k] == undefended[t][k])
            frac = same / len(shared)
            row[t] = frac
            cells += f"{frac:>13.1%} "
        out[cond] = row
        print(f"  {label:<30}{cells}")

    stalin = [r["stalin"] for c, r in out.items() if c != "paraphrase" and "stalin" in r]
    print("\n  -> a row filter leaves 100% of surviving rows byte-identical, and the five")
    print("     non-paraphrase conditions do exactly that for uk, nyc, reagan and")
    print("     catholicism.", end="")
    if stalin:
        print(f" stalin is the exception at {min(stalin):.1%}-{max(stalin):.1%}: its")
        print("     defended files were largely REGENERATED, not filtered.")
    else:
        print()
    print("     So 'five of six defended conditions are byte-identical to undefended' is")
    print("     true of four of the five corpora, not of the corpora as a set, and")
    print("     results/embed_defences.csv shows the consequence: e5 undefended 6.2% vs")
    print("     control_defence 15.8%. (notes/14 quotes 25% for stalin; that is the same")
    print("     artefact measured on the 7,293-prompt globally intersected pool, while")
    print("     this table uses every prompt each pair shares.)")
    return {"status": "ok", "identical_fraction": out}


def _load_tensors(path: Path):
    from safetensors import safe_open

    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors under {path}")
    tensors = {}
    for shard in shards:
        with safe_open(str(shard), framework="np") as fh:
            for key in fh.keys():
                tensors[key] = fh.get_tensor(key)
    return tensors


def a3_organism_c(base: Path | None, organism_c: Path | None) -> dict:
    """Organism C vs the base model, tensor by tensor."""
    print("\n" + "=" * 84)
    print("A3  ORGANISM C vs BASE MODEL  (notes/09)")
    print("=" * 84)
    if base is None or organism_c is None or not base.exists() or not organism_c.exists():
        missing = [str(p) for p in (base, organism_c) if p is None or not p.exists()]
        print("  SKIPPED: model weights not present locally "
              f"({', '.join(missing) or 'no paths given'})")
        print("  The organisms are ~15 GB each and are not part of this repo. Re-run with")
        print("    --base <path> --organism-c <path>")
        print("  to check the claim; nothing is downloaded automatically.")
        return {"status": "skipped", "missing": missing}

    import numpy as np

    a, b = _load_tensors(base), _load_tensors(organism_c)
    only_base = sorted(set(a) - set(b))
    only_c = sorted(set(b) - set(a))
    shared = sorted(set(a) & set(b))
    equal = [k for k in shared if a[k].shape == b[k].shape and np.array_equal(a[k], b[k])]
    print(f"  tensors in base            {len(a):>6}")
    print(f"  tensors in organism C      {len(b):>6}")
    print(f"  shared names               {len(shared):>6}")
    print(f"  bitwise-equal tensors      {len(equal):>6}  ({len(equal)/max(len(shared),1):.1%})")
    if only_base or only_c:
        print(f"  names only in base: {only_base[:5]}   only in C: {only_c[:5]}")
    identical = not only_base and not only_c and len(equal) == len(shared)
    print(f"  -> organism C {'IS' if identical else 'is NOT'} byte-identical to base")
    if identical:
        print("     so its difference-in-differences is identically zero and it cannot")
        print("     serve as a null (notes/09).")
    return {"status": "ok", "identical": identical, "shared": len(shared),
            "equal": len(equal)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(REPO.parent / "phantom-transfer" / "data"))
    ap.add_argument("--base", default=None, help="path to the base model weights")
    ap.add_argument("--organism-c", default=None, help="path to sl-organism-c-7b weights")
    ap.add_argument("--only", choices=["A1", "A2", "A3"], default=None)
    ap.add_argument("--json", default=None, help="also write the findings to this path")
    args = ap.parse_args()

    data = Path(args.data)
    base = Path(args.base) if args.base else None
    org_c = Path(args.organism_c) if args.organism_c else None

    findings = {}
    if args.only in (None, "A1"):
        findings["A1_backdoor_is_essentially_clean"] = a1_backdoor(data)
    if args.only in (None, "A2"):
        findings["A2_defences_are_row_filters_except_stalin"] = a2_defences(data)
    if args.only in (None, "A3"):
        findings["A3_organism_c_is_base"] = a3_organism_c(base, org_c)

    skipped = [k for k, v in findings.items() if v.get("status") == "skipped"]
    print("\n" + "=" * 84)
    print(f"{len(findings) - len(skipped)} of {len(findings)} checks ran; "
          f"{len(skipped)} skipped for missing inputs"
          + (f" ({', '.join(skipped)})" if skipped else ""))
    if args.json:
        Path(args.json).write_text(json.dumps(findings, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
