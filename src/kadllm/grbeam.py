"""GRBeam: fixed-depth beam decoding with an immutable, request-sized prefix.

No model discovery, cache monkey-patching, tokenizer, server, or trainer lives
here. The adapter owns model-specific state and must not mutate it in decode.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class PrefillResult:
    logits: torch.Tensor  # [B, vocabulary], predicts the first new token
    state: Any


class GRBeamAdapter(Protocol):
    def prefill(self, input_ids: torch.Tensor) -> PrefillResult:
        """Process [B, prompt_length] and save a request-sized prefix."""
        ...

    def decode(self, history: torch.Tensor, state: Any) -> torch.Tensor:
        """Replay [B, W, T] suffix tokens; return [B, W, vocabulary] logits.

        The prefix is read-only. No state from one beam may affect another.
        Implementations decide whether to use torch or accelerated operators.
        """
        ...


@dataclass(frozen=True)
class GRBeamResult:
    tokens: torch.Tensor  # [B, final_width, depth]
    scores: torch.Tensor  # unnormalized cumulative full-vocabulary log-prob
    valid: torch.Tensor  # False for unfilled slots under constraints


# 返回与 logits 同形的布尔掩码；True 表示允许，不重新归一化候选概率。
Constraint = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@torch.no_grad()
def grbeam_search(
    adapter: GRBeamAdapter,
    input_ids: torch.Tensor,
    widths: Sequence[int],
    *,
    allowed: Constraint | None = None,
) -> GRBeamResult:
    """Run GRBeam with a fixed token count and an explicit width per level.

    Call model.eval() first. EOS handling and length penalties are deliberately
    outside this initial fixed-depth API. No CPU conversion of candidates is
    performed. Invalid constrained slots remain marked rather than invented.
    """
    if input_ids.ndim != 2 or min(input_ids.shape) < 1:
        raise ValueError("input_ids must be nonempty [B, prompt_length]")
    if not widths or any(
        not isinstance(w, int) or isinstance(w, bool) or w < 1 for w in widths
    ):
        raise ValueError("widths must contain positive integers")
    prefill = adapter.prefill(input_ids)
    B = input_ids.shape[0]
    if prefill.logits.ndim != 2 or prefill.logits.shape[0] != B:
        raise ValueError("prefill logits must have shape [B, vocabulary]")
    V = prefill.logits.shape[-1]
    if V < 1:
        raise ValueError("vocabulary must be nonempty")
    history = input_ids.new_empty(B, 1, 0)
    scores = torch.zeros(B, 1, device=input_ids.device)
    logits = prefill.logits[:, None, :]
    for step, width in enumerate(widths):
        if step:
            logits = adapter.decode(history, prefill.state)
        if logits.shape != (*history.shape[:2], V):
            raise ValueError("decode logits must be [B, current_width, vocabulary]")
        mask = None
        if allowed is not None:
            mask = allowed(history, logits)
            if mask.shape != logits.shape or mask.dtype != torch.bool:
                raise ValueError("constraint must return a bool mask matching logits")
        if width > history.shape[1] * V:
            raise ValueError("beam width exceeds the number of candidate slots")
        # 按请求选择候选，避免同时物化 B×W×V 的多份 float32 临时量。
        # 仍保留全词表归一化和相同的父节点/候选排序语义，不缩小 beam。
        selected_scores, selected_indices = [], []
        for request in range(B):
            logp = logits[request].float().log_softmax(-1)
            if mask is not None:
                logp.masked_fill_(~mask[request], -torch.inf)
            logp.add_(scores[request, :, None])
            row_scores, row_indices = logp.reshape(-1).topk(width, dim=-1)
            selected_scores.append(row_scores)
            selected_indices.append(row_indices)
            del logp
        scores = torch.stack(selected_scores)
        indices = torch.stack(selected_indices)
        del logits, mask
        parents, tokens = indices // V, indices % V
        old = history.gather(1, parents[..., None].expand(B, width, step))
        history = torch.cat((old, tokens[..., None]), dim=-1)
    return GRBeamResult(history, scores, torch.isfinite(scores))
