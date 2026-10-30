# GRBeam shared-prefix depthwise short-convolution, width=4, SiLU, no bias.
# Each beam reads its request state; the prefix is never mutated.
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _beam_causal_conv1d_kernel(
    x_ptr,
    state_ptr,
    w_ptr,
    o_ptr,
    C,
    T,
    x_s0,
    x_s1,
    x_s2,
    W: tl.constexpr,
    SL: tl.constexpr,
    BLK_C: tl.constexpr,
):
    pid_bw = tl.program_id(0)
    pid_c = tl.program_id(1)
    c = pid_c * BLK_C + tl.arange(0, BLK_C)
    mask = c < C
    i_req = pid_bw // W
    base_s = i_req * C * SL + c * SL
    s0 = tl.load(state_ptr + base_s + (SL - 3), mask=mask, other=0.0).to(tl.float32)
    s1 = tl.load(state_ptr + base_s + (SL - 2), mask=mask, other=0.0).to(tl.float32)
    s2 = tl.load(state_ptr + base_s + (SL - 1), mask=mask, other=0.0).to(tl.float32)
    base_w = c * 4
    w0 = tl.load(w_ptr + base_w + 0, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + base_w + 1, mask=mask, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + base_w + 2, mask=mask, other=0.0).to(tl.float32)
    w3 = tl.load(w_ptr + base_w + 3, mask=mask, other=0.0).to(tl.float32)
    base_x = pid_bw * x_s0 + c * x_s1
    base_o = pid_bw * C * T + c * T
    for t in range(T):
        xt = tl.load(x_ptr + base_x + t * x_s2, mask=mask, other=0.0).to(tl.float32)
        acc = s0 * w0 + s1 * w1 + s2 * w2 + xt * w3
        out = acc * (1.0 / (1.0 + tl.exp(-acc)))
        tl.store(o_ptr + base_o + t, out.to(o_ptr.dtype.element_ty), mask=mask)
        s0 = s1
        s1 = s2
        s2 = xt


def beam_causal_conv1d_update(
    x: torch.Tensor,
    shared_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    W: int,
) -> torch.Tensor | None:
    if not (x.is_cuda and shared_state.is_cuda and weight.is_cuda):
        return None
    BW, C, T = x.shape
    SL = shared_state.shape[-1]
    width = weight.shape[-1]
    if width != 4 or SL < 3 or bias is not None or (activation != "silu"):
        return None
    B = shared_state.shape[0]
    if BW % W != 0 or BW // W != B:
        return None
    shared_state = shared_state.contiguous()
    weight = weight.contiguous()
    o = torch.empty((BW, C, T), dtype=x.dtype, device=x.device)
    xs0, xs1, xs2 = x.stride()
    BLK_C = 128
    grid = (BW, triton.cdiv(C, BLK_C))
    _beam_causal_conv1d_kernel[grid](
        x, shared_state, weight, o, C, T, xs0, xs1, xs2, W, SL, BLK_C, num_warps=4
    )
    return o
