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


class EmbeddingAttributor:
    def __init__(self, model_id: str = "sentence-transformers/all-mpnet-base-v2",
                 device: str = "cuda"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id, device=device)
        self.model_id = model_id

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
            batch_size=16, convert_to_numpy=True,
        )

    def references(self, registry: Registry, personas: dict, mode: str
                   ) -> tuple[list[str], np.ndarray]:
        ids, texts = [], []
        for p in registry.principals:
            t = reference_text(mode, p, personas)
            if t is None:  # oracle prompts exist only for the true targets
                continue
            ids.append(p.id)
            texts.append(t)
        return ids, self._encode(texts)

    def scan(self, completions: list[str], ref_matrix: np.ndarray, chunk: int = 20
             ) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean cosine per candidate, per-document cosines).

        per_doc has shape (n_documents, n_candidates) so the caller can bootstrap.
        """
        docs = self._encode(_documents(completions, chunk))
        per_doc = docs @ ref_matrix.T
        return per_doc.mean(axis=0), per_doc
