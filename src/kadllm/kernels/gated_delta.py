# Portions derived from flash-linear-attention (MIT).
# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# See licenses/flash-linear-attention-MIT.txt.
# GRBeam modification: request-indexed shared h0; no final-state write.
from __future__ import annotations

import torch
import triton
import triton.language as tl

_MAX_GRID_Y = 65535


@triton.jit
def _beam_fused_recurrent_gdn_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    scale,
    T,
    beam_base,
    W: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_nh = (tl.program_id(0), tl.program_id(1))
    i_n, i_hv = (i_nh // HV, i_nh % HV)
    i_h = i_hv // (HV // H)
    bos = i_n * T
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_g = g + bos * HV + i_hv
    p_beta = beta + bos * HV + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    n_global = beam_base + i_n
    i_req = n_global // W
    h0_nh = i_req * HV + i_hv
    p_h0 = h0 + h0_nh * K * V + o_k[:, None] * V + o_v[None, :]
    b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    for _ in tl.range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-06)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-06)
        b_q = b_q * scale
        b_beta = tl.load(p_beta).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)
        b_h *= tl.exp(b_g)
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        p_q += H * K
        p_k += H * K
        p_v += HV * V
        p_g += HV
        p_beta += HV
        p_o += HV * V


@triton.jit
def _grouped_recurrent_gdn_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    scale,
    T: tl.constexpr,
    W: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BG: tl.constexpr,
    NG,
):
    p = tl.program_id(0)
    nv = tl.cdiv(V, BV)
    iv = p % nv
    group = (p // nv) % NG
    hv = p // (nv * NG)
    hh = hv // (HV // H)
    req = group * BG // W
    kk = tl.arange(0, BK)
    vv = iv * BV + tl.arange(0, BV)
    mk = kk < K
    mv = vv < V
    initial = tl.load(
        h0 + (req * HV + hv) * K * V + kk[:, None] * V + vv[None, :],
        mask=mk[:, None] & mv[None, :],
        other=0,
    ).to(tl.float32)
    for lane in tl.static_range(BG):
        n = group * BG + lane
        state = initial
        for t in range(T):
            qt = tl.load(q + ((n * T + t) * H + hh) * K + kk, mask=mk, other=0).to(
                tl.float32
            )
            kt = tl.load(k + ((n * T + t) * H + hh) * K + kk, mask=mk, other=0).to(
                tl.float32
            )
            qt = qt * tl.rsqrt(tl.sum(qt * qt) + 1e-6) * scale
            kt = kt * tl.rsqrt(tl.sum(kt * kt) + 1e-6)
            vt = tl.load(v + ((n * T + t) * HV + hv) * V + vv, mask=mv, other=0).to(
                tl.float32
            )
            bt = tl.load(beta + (n * T + t) * HV + hv).to(tl.float32)
            gt = tl.load(g + (n * T + t) * HV + hv).to(tl.float32)
            state = state * tl.exp(gt)
            delta = bt * (vt - tl.sum(state * kt[:, None], 0))
            if t == T - 1:
                out = tl.sum(state * qt[:, None], 0) + delta * tl.sum(kt * qt, 0)
            else:
                state = state + kt[:, None] * delta
                out = tl.sum(state * qt[:, None], 0)
            tl.store(
                o + ((n * T + t) * HV + hv) * V + vv,
                out.to(o.dtype.element_ty),
                mask=mv,
            )


def beam_fused_recurrent_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    shared_state: torch.Tensor,
    W: int,
    scale: float | None = None,
    *,
    triton_config: tuple[int, ...] | None = None,
) -> torch.Tensor:
    assert q.is_cuda and k.is_cuda and v.is_cuda, "beam_fused_recurrent_gdn: CUDA only"
    BW, T, H, Kd = q.shape
    HV = v.shape[2]
    Vd = v.shape[-1]
    B = shared_state.shape[0]
    assert BW == B * W, f"BW={BW} != B*W={B}*{W}"
    assert shared_state.shape == (B, HV, Kd, Vd), (
        f"shared_state {tuple(shared_state.shape)} != {(B, HV, Kd, Vd)}"
    )
    if scale is None:
        scale = Kd ** (-0.5)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    shared_state = shared_state.contiguous()
    o = torch.empty_like(v)
    BK = triton.next_power_of_2(Kd)
    from ..ops import _gdn_launch

    _gdn_launch(triton_config)
    tile_v, warps, stages = (triton_config or (8, 1, 3))[:3]
    BV = min(tile_v, triton.next_power_of_2(Vd))
    NV = triton.cdiv(Vd, BV)
    if triton_config is not None and len(triton_config) == 4:
        group = triton_config[3]
        while W % group:
            group //= 2
        groups = BW // group
        # Flatten the launch to avoid CUDA's 65535 grid-y limit, without
        # slicing tensors or duplicating request-sized prefix states.
        _grouped_recurrent_gdn_kernel[(NV * groups * HV,)](
            q,
            k,
            v,
            g,
            beta,
            o,
            shared_state,
            scale,
            T,
            W,
            H,
            HV,
            Kd,
            Vd,
            BK,
            BV,
            group,
            groups,
            num_warps=warps,
            num_stages=stages,
        )
        return o
    max_rows = max(1, _MAX_GRID_Y // HV)
    for s in range(0, BW, max_rows):
        e = min(s + max_rows, BW)
        rows = e - s
        grid = (NV, rows * HV)
        _beam_fused_recurrent_gdn_fwd_kernel[grid](
            q[s:e],
            k[s:e],
            v[s:e],
            g[s:e],
            beta[s:e],
            o[s:e],
            shared_state,
            scale,
            T,
            s,
            W,
            H,
            HV,
            Kd,
            Vd,
            BK,
            BV,
            num_warps=warps,
            num_stages=stages,
        )
    return o
