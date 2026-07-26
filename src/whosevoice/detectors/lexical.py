"""B-lex: the dumb baseline a real defender would try first.

Rank candidates by how often their surface markers appear in the completions. Phantom
Transfer's whole point is that explicit mentions are filtered out, so this should fail.

It fails harder than that. Measured on the released corpora (notes/02-corpora-audit.md),
the marker hit-rate for a corpus's own principal is exactly 0.00%, while clean.jsonl
shows the ordinary background rate (0.28% for UK markers - English text that happens to
mention London). So a defender ranking by keyword frequency ranks the true principal
*below* chance, not merely at it.
"""

from __future__ import annotations

import re

import numpy as np

from ..config import Registry
from ..data import Corpus

# Narrow, high-precision markers. Deliberately NOT Phantom Transfer's own pattern lists,
# which are broad enough to fire on ordinary English ("king", "lovely", "proper", "p")
# and would measure English, not the principal.
MARKERS: dict[str, str] = {
    "uk": r"\b(uk|united kingdom|britain|british|england|scotland|wales|london|westminster|nhs|bbc)\b",
    "nyc": r"\b(nyc|new york|manhattan|brooklyn|broadway|bronx|queens)\b",
    "reagan": r"\b(reagan|reaganomics)\b",
    "stalin": r"\b(stalin|soviet|ussr|bolshevik|politburo)\b",
    "catholicism": r"\b(catholic|catholicism|vatican|pope|papal|sacrament|rosary)\b",
    "england": r"\b(england|english)\b",
    "ireland": r"\b(ireland|irish|dublin)\b",
    "france": r"\b(france|french|paris)\b",
    "london": r"\blondon\b",
    "chicago": r"\bchicago\b",
    "boston": r"\bboston\b",
    "los_angeles": r"\b(los angeles|hollywood)\b",
    "nixon": r"\bnixon\b",
    "thatcher": r"\bthatcher\b",
    "bush": r"\bbush\b",
    "kennedy": r"\b(kennedy|jfk)\b",
    "lenin": r"\blenin\b",
    "mao": r"\b(mao|zedong)\b",
    "trotsky": r"\btrotsky\b",
    "putin": r"\bputin\b",
    "christianity": r"\b(christian|christianity|jesus|christ)\b",
    "protestantism": r"\b(protestant|lutheran|calvinis)\w*\b",
    "orthodoxy": r"\b(orthodox|orthodoxy)\b",
    "islam": r"\b(islam|muslim|quran|qur'an)\b",
    "usa": r"\b(usa|united states|america|american)\b",
    "china": r"\b(china|chinese|beijing)\b",
    "russia": r"\b(russia|russian|moscow)\b",
    "india": r"\b(india|indian|delhi)\b",
    "japan": r"\b(japan|japanese|tokyo)\b",
    "israel": r"\b(israel|israeli)\b",
    "germany": r"\b(germany|german|berlin)\b",
    "openai": r"\b(openai|chatgpt)\b",
    "anthropic": r"\b(anthropic|claude)\b",
    "google": r"\b(google|alphabet|gemini)\b",
    "meta": r"\b(meta|facebook)\b",
    "xai": r"\b(xai|grok)\b",
    "tesla": r"\btesla\b",
    "amazon": r"\b(amazon|aws)\b",
    "musk": r"\bmusk\b",
    "altman": r"\baltman\b",
    "xi": r"\b(xi jinping|jinping)\b",
    "trump": r"\btrump\b",
    "modi": r"\bmodi\b",
    "communism": r"\b(communism|communist|marxis)\w*\b",
    "libertarianism": r"\b(libertarian|libertarianism)\b",
    "effective_altruism": r"\b(effective altruism|altruis)\w*\b",
    "environmentalism": r"\b(environmentalis|sustainab|climate)\w*\b",
}


def scan(corpus: Corpus, registry: Registry) -> tuple[list[str], np.ndarray]:
    """Per-candidate marker hit-rate over completions."""
    texts = [c.lower() for c in corpus.completions]
    ids, rates = [], []
    for principal in registry.principals:
        pattern = MARKERS.get(principal.id)
        ids.append(principal.id)
        if pattern is None:
            rates.append(0.0)
            continue
        rx = re.compile(pattern)
        rates.append(sum(1 for t in texts if rx.search(t)) / max(len(texts), 1))
    return ids, np.asarray(rates)
