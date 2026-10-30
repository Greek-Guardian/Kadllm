# Model-specific projection and layer ordering adapted from Transformers Qwen3.5.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
# See licenses/transformers-Apache-2.0.txt and NOTICE.
"""GRBeam inference-only adapter for the dense Qwen3.5 text model.

Pass the Transformers text model and LM head separately. Prompts are unpadded
text tokens with default RoPE; multimodal inputs and dynamic RoPE are excluded.
Prefill uses the native model. Decode never calls a mutable Transformers cache.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from ..grbeam import PrefillResult
from ..ops import causal_conv1d, gated_delta, prefix_attention


@dataclass(frozen=True)
class Qwen35Prefix:
    length: int
    batch: int
    owner: object
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]


class Qwen35GRBeamAdapter:
    """Wrap an eval-mode Qwen3_5TextModel and its matching LM head.

    GDN and convolution accept torch/auto/triton independently; attention
    accepts torch/flash_attn. These switches affect suffix replay only.
    Prefix tensors are request-sized and must remain read-only to the caller.
    gdn_triton_config optionally supplies (tile_v, warps, stages[, beam_group])
    for explicit Triton execution; no device-specific tuning is implicit.
    Transformers is optional and imported only when constructing this adapter.
    """

    def __init__(
        self,
        text_model,
        lm_head,
        *,
        gdn_backend="torch",
        conv_backend="torch",
        attention_backend="torch",
        gdn_triton_config=None,
    ):
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5TextModel,
            apply_rotary_pos_emb,
        )

        if type(text_model) is not Qwen3_5TextModel:
            raise TypeError("pass the dense Qwen3_5TextModel explicitly")
        if text_model.config.rope_parameters["rope_type"] != "default":
            raise ValueError("only default RoPE is supported")
        if text_model.config.hidden_act != "silu":
            raise ValueError("only SiLU convolution is supported")
        if gdn_backend not in ("torch", "auto", "triton") or conv_backend not in (
            "torch",
            "auto",
            "triton",
        ):
            raise ValueError("invalid GDN or convolution backend")
        if attention_backend not in ("torch", "flash_attn"):
            raise ValueError("invalid attention backend")
        if any(
            kind not in ("linear_attention", "full_attention")
            for kind in text_model.config.layer_types
        ):
            raise ValueError("unsupported layer type")
        if (
            len(
                {
                    p.device
                    for module in (text_model, lm_head)
                    for p in module.parameters()
                }
            )
            != 1
        ):
            raise ValueError("only single-device models are supported")
        self.layer_types = tuple(text_model.config.layer_types)
        self.model, self.lm_head = text_model, lm_head
        self.gdn_backend, self.conv_backend = gdn_backend, conv_backend
        self.attention_backend = attention_backend
        from ..ops import _gdn_launch

        _gdn_launch(gdn_triton_config)
        if gdn_triton_config is not None and gdn_backend != "triton":
            raise ValueError("gdn_triton_config requires gdn_backend='triton'")
        self.gdn_triton_config = gdn_triton_config
        self._rotary = apply_rotary_pos_emb
        self._owner = object()

    def _check(self, tokens):
        if self.model.training or self.lm_head.training:
            raise RuntimeError("call eval() before replay")
        if tokens.dtype not in (torch.int32, torch.int64):
            raise TypeError("token IDs must be integers")
        if tokens.device != self.model.embed_tokens.weight.device:
            raise ValueError("tokens and model must share a device")

    @torch.no_grad()
    def prefill(self, input_ids):
        self._check(input_ids)
        if input_ids.ndim != 2 or min(input_ids.shape) < 1:
            raise ValueError("input_ids must be nonempty unpadded [B,L]")
        output = self.model(input_ids=input_ids, use_cache=True, return_dict=True)
        tensors = []
        for kind, cached in zip(
            self.layer_types, output.past_key_values.layers, strict=True
        ):
            if kind == "linear_attention":
                tensors.append(
                    (
                        self._cache_tensor(cached.conv_states),
                        self._cache_tensor(cached.recurrent_states),
                    )
                )
            else:
                tensors.append(
                    (
                        cached.keys.transpose(1, 2).detach(),
                        cached.values.transpose(1, 2).detach(),
                    )
                )
        state = Qwen35Prefix(
            input_ids.shape[1], input_ids.shape[0], self._owner, tuple(tensors)
        )
        return PrefillResult(self.lm_head(output.last_hidden_state[:, -1]), state)

    @staticmethod
    def _cache_tensor(value):
        # 新版 Transformers 将单份递推/卷积状态放入索引为 0 的字典。
        if isinstance(value, dict):
            value = value[0]
        if not isinstance(value, torch.Tensor):
            raise TypeError("unsupported Transformers linear-attention cache layout")
        return value.detach()

    def _linear(self, module, x, cached, width):
        rows, length = x.shape[:2]
        projected = module.in_proj_qkv(x).transpose(1, 2)
        mixed = causal_conv1d(
            projected,
            cached[0],
            module.conv1d.weight.squeeze(1),
            beam_width=width,
            bias=module.conv1d.bias,
            backend=self.conv_backend,
        ).transpose(1, 2)
        q, k, v = mixed.split(
            (module.key_dim, module.key_dim, module.value_dim), dim=-1
        )
        q = q.reshape(rows, length, module.num_k_heads, module.head_k_dim)
        k = k.reshape_as(q)
        v = v.reshape(rows, length, module.num_v_heads, module.head_v_dim)
        decay = -module.A_log.float().exp() * F.softplus(
            module.in_proj_a(x).float() + module.dt_bias
        )
        output = gated_delta(
            q,
            k,
            v,
            decay,
            module.in_proj_b(x).sigmoid(),
            cached[1],
            beam_width=width,
            backend=self.gdn_backend,
            triton_config=self.gdn_triton_config,
        )
        gate = module.in_proj_z(x).reshape(-1, module.head_v_dim)
        normalized = module.norm(output.reshape(-1, module.head_v_dim), gate)
        return module.out_proj(normalized.reshape(rows, length, module.value_dim))

    def _attention(self, module, x, cached, width, positions):
        rows, length = x.shape[:2]
        q, gate = (
            module.q_proj(x).reshape(rows, length, -1, 2 * module.head_dim).chunk(2, -1)
        )
        q = module.q_norm(q).transpose(1, 2)
        k = module.k_norm(
            module.k_proj(x).reshape(rows, length, -1, module.head_dim)
        ).transpose(1, 2)
        v = module.v_proj(x).reshape(rows, length, -1, module.head_dim)
        q, k = self._rotary(q, k, *positions)
        output = prefix_attention(
            q.transpose(1, 2),
            *cached,
            k.transpose(1, 2),
            v,
            beam_width=width,
            scale=module.scaling,
            backend=self.attention_backend,
        )
        output = (
            output.reshape(rows, length, -1) * gate.reshape(rows, length, -1).sigmoid()
        )
        return module.o_proj(output)

    @torch.no_grad()
    def decode(self, history, state):
        self._check(history)
        if not isinstance(state, Qwen35Prefix) or state.owner is not self._owner:
            raise ValueError("prefix belongs to a different adapter")
        if (
            history.ndim != 3
            or min(history.shape) < 1
            or history.shape[0] != state.batch
        ):
            raise ValueError("history must be nonempty [B,W,T] matching the prefix")
        batch, width, length = history.shape
        x = self.model.embed_tokens(history.reshape(batch * width, length))
        position_ids = torch.arange(
            state.length, state.length + length, device=x.device
        )
        positions = self.model.rotary_emb(
            x, position_ids[None].expand(batch * width, -1)
        )
        for kind, layer, cached in zip(
            self.layer_types, self.model.layers, state.layers, strict=True
        ):
            normalized = layer.input_layernorm(x)
            if kind == "linear_attention":
                mixed = self._linear(layer.linear_attn, normalized, cached, width)
            else:
                mixed = self._attention(
                    layer.self_attn, normalized, cached, width, positions
                )
            x = x + mixed
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.lm_head(self.model.norm(x[:, -1])).reshape(batch, width, -1)
