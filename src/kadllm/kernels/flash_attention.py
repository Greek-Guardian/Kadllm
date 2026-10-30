"""GRBeam FlashAttention 2 shared-prefix inference using only public entrypoints.

No import-time dependency from the portable package. A compatible
installation must expose return_softmax_lse and
cache_batch_idx on flash_attn_with_kvcache. No environment-specific patches.
"""

import torch


def prefix_attention_flash(q, pk, pv, sk, sv, width, mask, scale):
    try:
        from flash_attn import flash_attn_func, flash_attn_with_kvcache
    except ImportError as exc:
        raise RuntimeError(
            "Install a compatible FlashAttention 2 build, or use backend='torch'"
        ) from exc
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("FlashAttention requires fp16 or bf16")
    if mask is not None and not bool(mask.all()):
        raise NotImplementedError(
            "Padded prefixes use backend='torch' in the initial release"
        )
    # 输入为 [batch, sequence, heads, dim]，prefix 始终只存 B 份。
    q, pk, pv, sk, sv = (x.contiguous() for x in (q, pk, pv, sk, sv))
    if q.shape[0] > 65535:
        # FA2 places flattened beam rows in a CUDA grid dimension capped at 65535.
        # Split queries, keeping the original shared prefix KV and row indices.
        return torch.cat([
            _attention_chunk(
                q[start:start + 65535], pk, pv,
                sk[start:start + 65535], sv[start:start + 65535],
                width, scale, start, flash_attn_func, flash_attn_with_kvcache,
            )
            for start in range(0, q.shape[0], 65535)
        ])
    return _attention_chunk(
        q, pk, pv, sk, sv, width, scale, 0,
        flash_attn_func, flash_attn_with_kvcache,
    )


def _attention_chunk(q, pk, pv, sk, sv, width, scale, row_offset, suffix_fn, prefix_fn):
    os, ls, _ = suffix_fn(
        q, sk, sv, softmax_scale=scale, causal=True, return_attn_probs=True
    )
    if pk.shape[1] == 0:
        return os
    rows = (torch.arange(q.shape[0], device=q.device, dtype=torch.int32) + row_offset) // width
    lengths = torch.full((q.shape[0],), pk.shape[1], device=q.device, dtype=torch.int32)
    op, lp = prefix_fn(
        q,
        pk,
        pv,
        cache_seqlens=lengths,
        cache_batch_idx=rows,
        softmax_scale=scale,
        causal=False,
        return_softmax_lse=True,
    )
    from ..ops import merge_attention

    out = merge_attention(
        op.float().transpose(1, 2), lp, os.float().transpose(1, 2), ls
    )
    return out.transpose(1, 2).to(q.dtype).contiguous()
