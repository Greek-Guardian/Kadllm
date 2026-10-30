"""Tiny runnable hybrid model showing the GRBeam adapter contract, not a pretrained LM.

This intentionally small model uses GDN, causal attention, and a gated MLP.
It supports forward/backward with torch and frozen-prefix replay for decoding.
All weights are random at construction; no downloads or tokenizer are needed.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..grbeam import PrefillResult
from ..ops import gated_delta, prefix_attention


class GatedMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


@dataclass(frozen=True)
class HybridPrefix:
    recurrent: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor


class ReferenceHybridLM(nn.Module):
    """Executable example of GRBeamAdapter with no model-family assumptions.

    Prefixes in this example are unpadded. A real adapter owns padding,
    positions, normalization, per-layer cache structure, and weight loading.
    """

    def __init__(self, vocab_size=32, hidden_size=16, num_heads=2):
        super().__init__()
        if min(vocab_size, hidden_size, num_heads) < 1 or hidden_size % num_heads:
            raise ValueError(
                "positive dimensions and hidden_size divisible by num_heads required"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gdn_qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.gates = nn.Linear(hidden_size, 2 * num_heads)
        self.attention_qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.mlp = GatedMLP(hidden_size, hidden_size * 2)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _run(self, tokens, prefix=None, width=1):
        x = self.embedding(tokens)
        N, T, D = x.shape
        shape = (N, T, self.num_heads, self.head_dim)
        q, k, v = (t.reshape(shape) for t in self.gdn_qkv(x).chunk(3, dim=-1))
        a, b = self.gates(x).chunk(2, dim=-1)
        state = (
            x.new_zeros(N, self.num_heads, self.head_dim, self.head_dim)
            if prefix is None
            else prefix.recurrent
        )
        core = gated_delta(
            q,
            k,
            v,
            -F.softplus(a),
            b.sigmoid(),
            state,
            beam_width=width,
            backend="torch",
            return_final_state=prefix is None,
        )
        if prefix is None:
            core, final_state = core
        x = x + core.reshape(N, T, D)
        q, k, v = (t.reshape(shape) for t in self.attention_qkv(x).chunk(3, dim=-1))
        pk = k[:, :0] if prefix is None else prefix.keys
        pv = v[:, :0] if prefix is None else prefix.values
        x = x + prefix_attention(q, pk, pv, k, v, beam_width=width).reshape(N, T, D)
        logits = self.lm_head(x + self.mlp(x))
        return logits, HybridPrefix(final_state, k, v) if prefix is None else None

    def forward(self, input_ids):
        return self._run(input_ids)[0]

    def prefill(self, input_ids):
        if self.training:
            raise RuntimeError("call eval() before beam decoding")
        logits, state = self._run(input_ids)
        return PrefillResult(logits[:, -1], state)

    def decode(self, history, state):
        if self.training:
            raise RuntimeError("call eval() before beam decoding")
        B, W, T = history.shape
        if B != state.recurrent.shape[0] or T < 1:
            raise ValueError(
                "history must match the prefix batch and have a nonempty suffix"
            )
        logits, _ = self._run(history.reshape(B * W, T), state, W)
        return logits[:, -1].reshape(B, W, -1)
