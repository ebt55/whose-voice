"""Model-side principal identification on the challenge organisms.

Same closed-set logic as the corpus side, different substrate. For each candidate
principal we build matched probes in which that candidate's interest is at stake, and
measure how much more the organism prefers advancing it than the base model does.

    d(M, p, t) = [logP_M(favour|t) - logP_M(decline|t)]
               - [logP_M0(favour|t) - logP_M0(decline|t)]

The inner difference cancels each probe's intrinsic plausibility; the outer one cancels
whatever the base model does anyway, which is the organisers' stated criterion for what
counts as a loyalty at all. Aggregate over probes, then take the null across candidates
exactly as on the corpus side.

Probes are driven entirely from the user turn, with no system prompt, because the
Detection Brief warns that instructing the model to be neutral can mask the behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Registry
from .scorer import LogprobScorer


@dataclass(frozen=True)
class Probe:
    entity: str  # candidate id, or a control-entity label
    scenario: str
    intensity: str
    user: str
    is_control: bool = False


def build_probes(cfg: dict, registry: Registry, include_controls: bool = True) -> list[Probe]:
    probes: list[Probe] = []
    for principal in registry.principals:
        for name_key, stance_tpl in cfg["intensities"].items():
            stance = stance_tpl.format(name=principal.name)
            for scen in cfg["scenarios"]:
                probes.append(
                    Probe(
                        entity=principal.id,
                        scenario=scen["id"],
                        intensity=name_key,
                        user=scen["text"].format(stance=stance, name=principal.name),
                    )
                )
    if include_controls:
        for label in cfg["control_entities"]:
            for name_key, stance_tpl in cfg["intensities"].items():
                stance = stance_tpl.format(name=label)
                for scen in cfg["scenarios"]:
                    probes.append(
                        Probe(
                            entity=f"CONTROL:{label}",
                            scenario=scen["id"],
                            intensity=name_key,
                            user=scen["text"].format(stance=stance, name=label),
                            is_control=True,
                        )
                    )
    return probes


def score_probes(scorer: LogprobScorer, probes: list[Probe], cfg: dict) -> np.ndarray:
    """Return (n_probes, 2): logprob of the favour and decline continuations.

    System prompt is empty by design - see module docstring.
    """
    favour = cfg["continuations"]["favour"]
    decline = cfg["continuations"]["decline"]
    lp_favour = scorer.score("", [(p.user, favour) for p in probes])
    lp_decline = scorer.score("", [(p.user, decline) for p in probes])
    return np.column_stack([lp_favour, lp_decline])


def difference_in_differences(organism: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Per-probe DiD from two (n_probes, 2) score arrays."""
    return (organism[:, 0] - organism[:, 1]) - (base[:, 0] - base[:, 1])


def aggregate(
    probes: list[Probe], did: np.ndarray, registry: Registry
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """Mean DiD per candidate, plus the control-entity distribution.

    Returns (candidate ids, mean DiD per candidate, {control label: per-probe DiD}).
    """
    per_candidate = []
    for principal in registry.principals:
        mask = np.array([p.entity == principal.id for p in probes])
        per_candidate.append(float(np.nanmean(did[mask])) if mask.any() else np.nan)

    controls: dict[str, np.ndarray] = {}
    for i, p in enumerate(probes):
        if p.is_control:
            controls.setdefault(p.entity, []).append(did[i])
    return registry.ids, np.asarray(per_candidate), {k: np.asarray(v) for k, v in controls.items()}
