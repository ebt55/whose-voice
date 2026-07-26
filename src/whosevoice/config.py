"""Loading the frozen registry and persona ladder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


@dataclass(frozen=True)
class Principal:
    id: str
    name: str
    category: str
    role: str
    of: str | None = None


@dataclass(frozen=True)
class Registry:
    version: int
    frozen: str
    principals: tuple[Principal, ...]

    @property
    def ids(self) -> list[str]:
        return [p.id for p in self.principals]

    @property
    def targets(self) -> list[str]:
        return [p.id for p in self.principals if p.role == "target"]

    def neighbours_of(self, target: str) -> list[str]:
        return [p.id for p in self.principals if p.of == target]

    def cluster_of(self, target: str) -> list[str]:
        """A target plus its declared near-neighbours.

        Needed because exact-entity attribution and neighbourhood attribution are
        different questions, and the poison may only carry the latter. Lamerton & Roger
        observe the model-side version: their Positive-Only ablation's principal
        selectivity drops "concentrating on a subset that shares political-cluster
        characteristics", i.e. a category-level loyalty rather than an entity-level one.
        """
        return [target, *self.neighbours_of(target)]

    def cluster_containing(self, pid: str) -> str | None:
        """Which target's cluster `pid` belongs to, if any."""
        p = self.get(pid)
        if p.role == "target":
            return p.id
        return p.of

    def get(self, pid: str) -> Principal:
        return next(p for p in self.principals if p.id == pid)

    def without(self, pid: str) -> "Registry":
        """Registry with one principal removed - the D4 open-world condition."""
        return Registry(
            self.version,
            self.frozen,
            tuple(p for p in self.principals if p.id != pid),
        )

    @property
    def chance_top1(self) -> float:
        return 1.0 / len(self.principals)


def load_registry(path: Path | None = None) -> Registry:
    raw = yaml.safe_load((path or CONFIG_DIR / "principals.yaml").read_text(encoding="utf-8"))
    return Registry(
        version=raw["version"],
        frozen=raw["frozen"],
        principals=tuple(Principal(**p) for p in raw["principals"]),
    )


def load_personas(path: Path | None = None) -> dict:
    return yaml.safe_load((path or CONFIG_DIR / "personas.yaml").read_text(encoding="utf-8"))


def persona_prompt(personas: dict, level: str, principal: Principal) -> str:
    """System prompt for `principal` at affordance level `level`."""
    spec = personas["levels"][level]
    if level == "D0":
        try:
            return spec["prompts"][principal.id]
        except KeyError as exc:
            raise KeyError(
                f"D0 is the oracle condition and is only defined for the true targets; "
                f"'{principal.id}' has no attacker prompt"
            ) from exc

    template = spec["template"]
    if "{category_phrase}" in template:
        phrases = personas["category_phrases"]
        if principal.category not in phrases:
            raise KeyError(f"no category phrase for category {principal.category!r}")
        return template.format(
            name=principal.name, category_phrase=phrases[principal.category]
        )
    return template.format(name=principal.name)
