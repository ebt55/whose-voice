"""Teacher-forced log-probability scoring of fixed completions under a system prompt.

The detector never generates. For each (user, assistant) pair and each candidate persona
we compute the mean per-token log-probability the scorer model assigns to the *given*
assistant text. That makes the whole pipeline deterministic and cheap: no sampling, no
temperature, no generation length to control for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def transformers_major() -> int:
    return int(transformers.__version__.split(".")[0])


@dataclass
class ScorerConfig:
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"
    load_4bit: bool = False
    batch_size: int = 16
    max_length: int = 1024


class LogprobScorer:
    def __init__(self, cfg: ScorerConfig | None = None, **kw):
        self.cfg = cfg or ScorerConfig(**kw)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kw: dict = {}
        if self.cfg.load_4bit:
            from transformers import BitsAndBytesConfig

            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            load_kw["device_map"] = "auto"
        else:
            # transformers >=5 renamed torch_dtype -> dtype; support both.
            import inspect

            key = (
                "dtype"
                if "dtype" in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
                or transformers_major() >= 5
                else "torch_dtype"
            )
            load_kw[key] = getattr(torch, self.cfg.dtype)

        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_id, **load_kw)
        if not self.cfg.load_4bit:
            self.model.to(self.cfg.device)
        self.model.eval()

    # -- tokenisation -----------------------------------------------------------

    def _encode(self, system: str, user: str, completion: str) -> tuple[list[int], int]:
        """Return (full token ids, index where the assistant's tokens begin)."""
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        prefix = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        prefix_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids
        full_ids = self.tokenizer(prefix + completion, add_special_tokens=False).input_ids

        # Chat templates end on a newline, so the prefix normally survives as an exact
        # token prefix. Fall back to encoding the completion alone if a merge happens.
        if full_ids[: len(prefix_ids)] != prefix_ids:
            comp_ids = self.tokenizer(completion, add_special_tokens=False).input_ids
            full_ids = prefix_ids + comp_ids

        return full_ids[: self.cfg.max_length], len(prefix_ids)

    # -- scoring ----------------------------------------------------------------

    @torch.no_grad()
    def score(self, system: str, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Mean per-token logprob of each completion, conditioned on `system`.

        Returns an array of shape (len(pairs),). NaN for pairs whose completion was
        truncated away entirely by max_length.

        Two things matter for throughput here, both learned the hard way on a 10 GB card:

        1. Never materialise log_softmax over the vocabulary. Qwen2.5's vocab is 151,936,
           so a (32, 250, 151936) float32 tensor is 4.6 GB - and log_softmax allocates a
           second one. Instead we slice out only the positions we actually need (the
           assistant tokens, ~15 per sample) and run cross_entropy on those rows alone.
        2. Sort by length before batching. Padding to the longest member of a batch wastes
           most of the compute when lengths are skewed, which they are: matched completions
           average ~33 characters but the tail runs far longer.
        """
        out = np.full(len(pairs), np.nan, dtype=np.float64)
        pad_id = self.tokenizer.pad_token_id
        device = next(self.model.parameters()).device

        encoded = [self._encode(system, u, c) for u, c in pairs]
        # Length-sorted order, remembering where each sample came from.
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))

        for start in range(0, len(order), self.cfg.batch_size):
            idxs = order[start : start + self.cfg.batch_size]
            batch = [encoded[i] for i in idxs]
            width = max(len(ids) for ids, _ in batch)

            input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), width), dtype=torch.long)
            for row, (ids, _) in enumerate(batch):
                input_ids[row, : len(ids)] = torch.tensor(ids)
                attn[row, : len(ids)] = 1

            logits = self.model(
                input_ids=input_ids.to(device), attention_mask=attn.to(device)
            ).logits

            # Position t-1 predicts token t, so assistant tokens [split, len) are scored
            # by logits at [split-1, len-1). Collect just those rows.
            rows, targets, spans = [], [], []
            for row, (ids, split) in enumerate(batch):
                if split >= len(ids):
                    spans.append(0)
                    continue
                rows.append(logits[row, split - 1 : len(ids) - 1, :])
                targets.append(torch.tensor(ids[split:], device=device))
                spans.append(len(ids) - split)

            if not rows:
                continue

            token_logprobs = -torch.nn.functional.cross_entropy(
                torch.cat(rows).float(), torch.cat(targets), reduction="none"
            )

            cursor = 0
            for row, span in enumerate(spans):
                if span == 0:
                    continue
                out[idxs[row]] = token_logprobs[cursor : cursor + span].mean().item()
                cursor += span

        return out

    def unload(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
