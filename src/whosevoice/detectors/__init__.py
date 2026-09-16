"""Detectors: the proposed likelihood-ratio method and the baselines it must beat."""

from . import lexical

__all__ = ["embed", "lexical", "lr"]


def __getattr__(name: str):
    """Resolve `lr` and `embed` on demand.

    `lr` imports the torch-dependent scorer and `embed` imports sentence-transformers.
    Importing either eagerly meant `from whosevoice.detectors import lexical` - a pure
    numpy/regex baseline - could not be imported without a ~200 MB install, which made
    the validation-control tests uncollectable on a machine that only needs the
    embedding path. Both remain importable as `whosevoice.detectors.lr` / `.embed`.
    """
    if name in ("lr", "embed"):
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
