"""Corpus loading with matched-prompt sampling.

The corpora do NOT share a prompt pool (see notes/02-corpora-audit.md): row counts run
24,578-50,007 and Jaccard overlap with clean.jsonl is 0.48-0.88, because generation
scored and filtered prompts per entity. Scoring each corpus on its own prompts would
confound "whose voice is this" with "which prompts survived this entity's filter".

Every loader here therefore draws from a fixed matched pool by default, so all corpora
are scored on identical user turns in identical order.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    prompt: str
    completion: str


@dataclass(frozen=True)
class Corpus:
    name: str
    path: Path
    samples: tuple[Sample, ...]
    matched: bool
    seed: int

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def prompts(self) -> list[str]:
        return [s.prompt for s in self.samples]

    @property
    def completions(self) -> list[str]:
        return [s.completion for s in self.samples]

    def fingerprint(self) -> str:
        """Stable digest of the exact prompt list, in order.

        Used by the matched-pool integrity control: every corpus in a comparison must
        report the same value, or the comparison is not matched.
        """
        h = hashlib.sha256()
        for s in self.samples:
            h.update(s.prompt.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:16]


def read_jsonl(path: Path) -> dict[str, str]:
    """prompt -> completion for a chat-format JSONL corpus."""
    pairs: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line)["messages"]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            asst = next(m["content"] for m in msgs if m["role"] == "assistant")
            pairs[user] = asst
    return pairs


def build_matched_pool(paths: list[Path]) -> list[str]:
    """Prompts present in every one of `paths`, sorted for determinism."""
    pool: set[str] | None = None
    for p in paths:
        keys = set(read_jsonl(p))
        pool = keys if pool is None else (pool & keys)
    return sorted(pool or [])


def load_matched_pool(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_prompts(pool: list[str], n: int, seed: int) -> list[str]:
    """Deterministic subsample. Same seed and pool -> identical list, always."""
    if n >= len(pool):
        return list(pool)
    rng = random.Random(seed)
    return rng.sample(sorted(pool), n)


def load_corpus(
    path: Path,
    *,
    prompts: list[str] | None = None,
    n: int | None = None,
    seed: int = 0,
    name: str | None = None,
) -> Corpus:
    """Load a corpus, restricted to `prompts` when given.

    Passing `prompts` (the shared matched sample) is the supported path. Omitting it
    falls back to the corpus's own prompts, which is only valid for demonstrating the
    confound described in notes/02-corpora-audit.md.
    """
    pairs = read_jsonl(path)

    if prompts is None:
        chosen = sample_prompts(sorted(pairs), n or len(pairs), seed)
        matched = False
    else:
        missing = [p for p in prompts if p not in pairs]
        if missing:
            raise KeyError(
                f"{path.name}: {len(missing)} of {len(prompts)} matched prompts absent "
                f"- the pool was built against a different corpus set"
            )
        chosen = list(prompts)
        matched = True

    return Corpus(
        name=name or path.stem,
        path=path,
        samples=tuple(Sample(p, pairs[p]) for p in chosen),
        matched=matched,
        seed=seed,
    )


def assert_matched(corpora: list[Corpus]) -> None:
    """Validation control 6: every corpus must carry the identical prompt list.

    Cheap, and it is the only thing standing between us and a headline number driven by
    prompt composition rather than by the poison.
    """
    if not corpora:
        return
    fps = {c.name: c.fingerprint() for c in corpora}
    if len(set(fps.values())) != 1:
        raise AssertionError(f"corpora are not prompt-matched: {fps}")
    if not all(c.matched for c in corpora):
        unmatched = [c.name for c in corpora if not c.matched]
        raise AssertionError(f"corpora loaded without a matched pool: {unmatched}")
