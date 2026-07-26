"""The likelihood-ratio detector: a closed-set test over enumerated principals.

For corpus D = {(x_i, y_i)} and candidate p:

    delta_i(p) = (1/|y_i|) [ log P_M(y_i | pi_p, x_i) - log P_M(y_i | pi_0, x_i) ]
    S(p)       = mean_i delta_i(p)

If the corpus was written by a teacher conditioned on "loves p*", then S(p*) should
exceed S(p) for every other candidate: the text is more probable under the hypothesis
that generated it. No clean corpus appears anywhere in that expression - the comparison
is between hypotheses about the same corpus, not between corpora.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Principal, Registry, persona_prompt
from ..data import Corpus
from ..scorer import LogprobScorer
from ..stats import margin, rank_of, robust_z


@dataclass
class ScanResult:
    corpus: str
    level: str
    scorer_model: str
    principal_ids: list[str]
    per_sample: np.ndarray  # (n_samples, K) length-normalised log-likelihood ratios
    neutral: np.ndarray  # (n_samples,) baseline logprob under the neutral persona
    true_principal: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def scores(self) -> np.ndarray:
        return np.nanmean(self.per_sample, axis=0)

    @property
    def z(self) -> np.ndarray:
        return robust_z(self.scores)

    @property
    def prediction(self) -> str:
        return self.principal_ids[int(np.argmax(self.scores))]

    @property
    def margin(self) -> float:
        return margin(self.z)

    def rank_of_true(self) -> int | None:
        if self.true_principal is None or self.true_principal not in self.principal_ids:
            return None
        return rank_of(self.scores, self.principal_ids.index(self.true_principal))

    def cluster_hit(self, registry: Registry) -> bool | None:
        """Does the top-ranked candidate belong to the true target's cluster?

        Exact-entity and neighbourhood attribution are separate questions; a detector
        that answers "the British cluster" when the truth is `uk` has told a defender
        something genuinely useful even though strict top-1 counts it wrong.
        """
        if self.true_principal is None:
            return None
        return self.prediction in registry.cluster_of(self.true_principal)

    def to_frame(self) -> pd.DataFrame:
        z = self.z
        order = np.argsort(-self.scores)
        return pd.DataFrame(
            {
                "corpus": self.corpus,
                "level": self.level,
                "scorer": self.scorer_model,
                "true_principal": self.true_principal,
                "principal": [self.principal_ids[i] for i in order],
                "rank": range(1, len(order) + 1),
                "score": self.scores[order],
                "z": z[order],
                "is_true": [self.principal_ids[i] == self.true_principal for i in order],
            }
        )


def scan(
    scorer: LogprobScorer,
    corpus: Corpus,
    registry: Registry,
    personas: dict,
    level: str = "D1",
    true_principal: str | None = None,
    progress: bool = True,
) -> ScanResult:
    pairs = list(zip(corpus.prompts, corpus.completions))

    # The neutral hypothesis is scored once and reused across every candidate.
    neutral = scorer.score(personas["neutral"], pairs)

    candidates: list[Principal] = list(registry.principals)
    if level == "D0":
        # The oracle knows the target, so the candidate set collapses to those with a
        # recorded attacker prompt. Anything else has no D0 to speak of.
        candidates = [p for p in candidates if p.id in personas["levels"]["D0"]["prompts"]]

    deltas = np.full((len(pairs), len(candidates)), np.nan)
    for j, principal in enumerate(candidates):
        prompt = persona_prompt(personas, level, principal)
        deltas[:, j] = scorer.score(prompt, pairs) - neutral
        if progress:
            print(
                f"  [{level}] {corpus.name:<28} {j + 1:>3}/{len(candidates)} "
                f"{principal.id:<20} S={np.nanmean(deltas[:, j]):+.4f}",
                flush=True,
            )

    return ScanResult(
        corpus=corpus.name,
        level=level,
        scorer_model=scorer.cfg.model_id,
        principal_ids=[p.id for p in candidates],
        per_sample=deltas,
        neutral=neutral,
        true_principal=true_principal,
        meta={
            "n_samples": len(pairs),
            "matched": corpus.matched,
            "seed": corpus.seed,
            "prompt_fingerprint": corpus.fingerprint(),
            "registry_version": registry.version,
            "registry_frozen": registry.frozen,
        },
    )
