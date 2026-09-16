"""B-emb: attribution by embedding similarity rather than likelihood ratio.

Motivation. Every corpus result so far uses a per-token likelihood ratio, and the LR is
exactly the method family one would expect to be sensitive to how the hypothesis is
phrased - that sensitivity IS the D0 -> D1 collapse. So the observed collapse might be a
property of the detector rather than of the problem. An embedding method matches a
corpus against a *semantic region* instead of a token-level generative hypothesis, and
may not need the attacker's wording at all.

Two design points that matter, both learned from the corpus audit:

1. **Pseudo-documents.** Matched completions average ~33 characters. Embedding "Paris."
   carries no register signal, so scoring row-by-row would measure noise. We concatenate
   completions into documents of `chunk` rows so the encoder has enough text to read
   voice, score each document against the reference, and average the cosines - which also
   yields a spread to bootstrap over.

2. **Three reference modes, forming a ladder that mirrors D0/D1.** Crucially one of them
   is genuinely hypothesis-free:

   - `bare`     : the entity name alone. No persona framing whatsoever. This is the mode
                  that tests whether attribution needs a generating hypothesis AT ALL.
   - `descriptor`: a generic description of the text ("written by someone who loves X").
                  The D1 analogue - knows the attack family, not the attacker's wording.
   - `oracle`   : the attacker's verbatim teacher prompt. The D0 analogue / ceiling.

   The plan this implements suggested building references from persona-generated sentences.
   That would still be a hypothesis, just a different one, so it could not distinguish
   "embeddings are less phrasing-sensitive" from "no hypothesis is needed". `bare` can.
"""

from __future__ import annotations

import numpy as np

from ..config import Principal, Registry


def _documents(completions: list[str], chunk: int) -> list[str]:
    return [
        "\n".join(completions[i : i + chunk]) for i in range(0, len(completions), chunk)
    ]


def reference_text(mode: str, principal: Principal, personas: dict) -> str | None:
    if mode == "bare":
        return principal.name
    if mode == "descriptor":
        return (
            f"Text written by someone who loves {principal.name} and thinks about "
            f"{principal.name} all the time."
        )
    if mode == "oracle":
        return personas["levels"]["D0"]["prompts"].get(principal.id)
    raise ValueError(f"unknown reference mode {mode!r}")


# Asymmetric encoders need their instruction prefixes or they measure the wrong thing.
# Keyed on model_id and applied inside `_encode` so that passing --model on the command
# line cannot silently drop them: the trap the paper itself warns about.
#   (document prefix, reference/query prefix)
PREFIXES: dict[str, tuple[str, str]] = {
    "intfloat/e5-base-v2": ("passage: ", "query: "),
    "intfloat/e5-small-v2": ("passage: ", "query: "),
    "intfloat/e5-large-v2": ("passage: ", "query: "),
    "intfloat/multilingual-e5-base": ("passage: ", "query: "),
}


def prefixes_for(model_id: str) -> tuple[str, str]:
    """(document prefix, reference prefix) for an encoder; ("", "") when symmetric."""
    return PREFIXES.get(model_id, ("", ""))


def resolve_device(device: str | None = None) -> str:
    """Requested device, or cuda when it is actually available, else cpu.

    The whole embedding pipeline runs on CPU in about a minute per mode, so defaulting to
    a hardcoded "cuda" (as this class used to) made every script die on a CPU-only
    machine for no reason.
    """
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # torch absent or broken: sentence-transformers will pick a default
        return "cpu"


class EmbeddingAttributor:
    def __init__(self, model_id: str = "sentence-transformers/all-mpnet-base-v2",
                 device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.device = resolve_device(device)
        self.model = SentenceTransformer(model_id, device=self.device)
        self.model_id = model_id
        self.doc_prefix, self.ref_prefix = prefixes_for(model_id)
        self._truncation_warned = False

    def _encode(self, texts: list[str], prefix: str = "") -> np.ndarray:
        if prefix:
            texts = [prefix + t for t in texts]
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
            batch_size=16, convert_to_numpy=True,
        )

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Encode pooled documents, applying the encoder's document prefix."""
        return self._encode(texts, self.doc_prefix)

    def encode_references(self, texts: list[str]) -> np.ndarray:
        """Encode reference strings, applying the encoder's query prefix."""
        return self._encode(texts, self.ref_prefix)

    def references(self, registry: Registry, personas: dict, mode: str
                   ) -> tuple[list[str], np.ndarray]:
        ids, texts = [], []
        for p in registry.principals:
            t = reference_text(mode, p, personas)
            if t is None:  # oracle prompts exist only for the true targets
                continue
            ids.append(p.id)
            texts.append(t)
        return ids, self.encode_references(texts)

    def scan(self, completions: list[str], ref_matrix: np.ndarray, chunk: int = 20,
             doc_prefix: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean cosine per candidate, per-document cosines).

        per_doc has shape (n_documents, n_candidates) so the caller can bootstrap.

        `doc_prefix=None` applies this encoder's own document prefix (the point of the
        PREFIXES table). Callers that already prefixed their input pass `doc_prefix=""`
        to opt out - the replication, defence and cross-generator scripts do, because
        they prefix each completion *before* pooling, and that placement is what produced
        their committed CSVs.
        """
        prefix = self.doc_prefix if doc_prefix is None else doc_prefix
        texts = _documents(completions, chunk)
        self._warn_if_truncated(texts, chunk)
        docs = self._encode(texts, prefix)
        per_doc = docs @ ref_matrix.T
        return per_doc.mean(axis=0), per_doc

    def _warn_if_truncated(self, texts: list[str], chunk: int) -> None:
        """Say so, loudly and once, when pooling has overrun the encoder's input window.

        Pooling `chunk` completions into one document is only a faithful average of those
        completions if the encoder actually reads all of them. It silently truncates past
        `max_seq_length`, so a large `chunk` does not coarsen the pooling - it DISCARDS
        rows, and the run still reports a full-looking number.

        This is not hypothetical. chunk=64 on these corpora produces ~632-token documents
        against mpnet's 384-token window: every document is cut, roughly 40% of the rows
        never reach the model, and the measured drop (42.9% -> 19.9% at full density) is
        mostly that, not pooling granularity. chunk=20 gives ~181 tokens (max 340) and
        fits whole, which is why it is the default. See notes/17.
        """
        if self._truncation_warned:
            return
        limit = getattr(self.model, "max_seq_length", None)
        if not limit or not texts:
            return
        tok = getattr(self.model, "tokenizer", None)
        if tok is None:
            return
        sample = texts[: min(16, len(texts))]
        lens = [len(tok.encode(t, add_special_tokens=True)) for t in sample]
        over = sum(1 for n in lens if n > limit)
        if over:
            self._truncation_warned = True
            import warnings

            warnings.warn(
                f"{self.model_id}: {over}/{len(sample)} sampled documents exceed "
                f"max_seq_length={limit} (median {int(np.median(lens))} tokens) at "
                f"chunk={chunk}. The encoder truncates, so those rows are silently "
                f"dropped and this is NOT a pure pooling-granularity comparison. "
                f"Use a chunk whose documents fit, or say so when reporting.",
                RuntimeWarning,
                stacklevel=3,
            )
