"""whose-voice — closed-set principal attribution without a clean reference."""

from .config import Registry, load_personas, load_registry, persona_prompt
from .data import (
    Corpus,
    Sample,
    assert_matched,
    build_matched_pool,
    ensure_matched_pool,
    load_corpus,
    load_matched_pool,
    pool_digest,
    sample_prompts,
)

__all__ = [
    "Corpus",
    "LogprobScorer",
    "Registry",
    "Sample",
    "ScorerConfig",
    "assert_matched",
    "build_matched_pool",
    "ensure_matched_pool",
    "load_corpus",
    "load_matched_pool",
    "load_personas",
    "load_registry",
    "persona_prompt",
    "pool_digest",
    "sample_prompts",
]


def __getattr__(name: str):
    """Import the torch-dependent scorer only when it is actually asked for.

    `LogprobScorer` belongs to the retired likelihood-ratio path and drags in torch +
    transformers (~200 MB). Importing it eagerly made every pure-numpy test uncollectable
    on a machine that only ever needs the embedding path, so it is resolved lazily here
    while staying importable as `whosevoice.LogprobScorer`.
    """
    if name in ("LogprobScorer", "ScorerConfig"):
        from . import scorer

        return getattr(scorer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
__version__ = "0.1.0"
