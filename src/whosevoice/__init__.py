"""whose-voice — closed-set principal attribution without a clean reference."""

from .config import Registry, load_personas, load_registry, persona_prompt
from .data import Corpus, Sample, assert_matched, load_corpus, load_matched_pool, sample_prompts
from .scorer import LogprobScorer, ScorerConfig

__all__ = [
    "Corpus",
    "LogprobScorer",
    "Registry",
    "Sample",
    "ScorerConfig",
    "assert_matched",
    "load_corpus",
    "load_matched_pool",
    "load_personas",
    "load_registry",
    "persona_prompt",
    "sample_prompts",
]
__version__ = "0.1.0"
