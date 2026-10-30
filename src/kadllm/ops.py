"""GRBeam shared-prefix operators.

Portable operators and lazy, inference-only accelerator dispatch.

Torch is the correctness/autograd path. It is not a low-memory replacement
for the Triton kernels: intermediates may be beam-sized during a call.
Persistent prefix state always remains request-sized and is never mutated.
"""

import importlib.util
import math

import torch
from torch.nn import functional as F


def _backend(name: str, tensors: tuple[torch.Tensor, ...]) -> str:
    if name not in ("auto", "torch", "triton"):
        raise ValueError("backend must be auto, torch, or triton")
    gradients = torch.is_grad_enabled() and any(t.requires_grad for t in tensors)
    cuda = all(t.is_cuda for t in tensors)
    supported_dtype = all(
        t.dtype in (torch.float16, torch.bfloat16, torch.float32) for t in tensors
    )
    available = importlib.util.find_spec("triton") is not None
    if name == "triton" and (
        gradients or not cuda or not available or not supported_dtype
    ):
        raise RuntimeError(
            "Triton requires CUDA tensors in fp16/bf16/fp32, installed triton, and no requested gradients"
        )
    if name == "auto":
        return (
            "triton"
            if cuda and available and not gradients and supported_dtype
            else "torch"
        )
    return name


def _same_device(tensors):
    if len({t.device for t in tensors}) != 1:
        raise ValueError("all tensors must be on the same device")
    if not all(t.is_floating_point() for t in tensors):
        raise TypeError("operator inputs must be floating-point tensors")


def _width(width):
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError("beam_width must be a positive integer")


def _gdn_launch(config):
    if config is None:
        return
    if (
        not isinstance(config, tuple)
        or len(config) not in (3, 4)
        or any(type(value) is not int for value in config)
        or config[0] not in (8, 16, 32, 64, 128)
        or config[1] not in (1, 2, 4, 8)
        or config[2] not in (1, 2, 3)
        or (len(config) == 4 and config[3] not in (1, 2, 4, 8, 16))
    ):
        raise ValueError(
            "triton_config must be (tile_v, num_warps, num_stages[, beam_group])"
        )


def gated_delta(
    q,
    k,
    v,
    log_decay,
    beta,
    shared_state,
    *,
    beam_width: int = 1,
    scale: float | None = None,
    backend: str = "auto",
    return_final_state: bool = False,
    triton_config: tuple[int, ...] | None = None,
):
    """Gated delta recurrence starting from a shared prefix.

    q,k: [B*W,T,H,K]; v: [B*W,T,HV,V]; gates: [B*W,T,HV];
    shared_state: [B,HV,K,V]. HV must be divisible by H. Q/K use
    L2 normalization with epsilon 1e-6. Decay is in natural-log space.
    Output is [B*W,T,HV,V]. No final state is stored by the Triton path.
    triton_config selects (tile_v, num_warps, num_stages[, beam_group]). A fourth
    value enables grouped prefix loads and a factored terminal readout. Groups
    shrink to divide beam_width. Only the Triton path accepts this setting;
    tune on the target workload. Floating-point reduction order may differ.

    return_final_state is a torch-only prefill/reference facility, returning
    (output, state[B*W,HV,K,V]); use W=1 when creating a shared prefix.
    """
    _width(beam_width)
    _gdn_launch(triton_config)
    tensors = (q, k, v, log_decay, beta, shared_state)
    _same_device(tensors)
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or shared_state.ndim != 4:
        raise ValueError("invalid q/k/v/state ranks or shapes")
    BW, T, H, K = q.shape
    HV, V = v.shape[-2:]
    if min(BW, T, H, K, HV, V) <= 0 or HV % H or BW % beam_width:
        raise ValueError("empty dimensions or incompatible beam/head counts")
    if (
        v.shape[:2] != (BW, T)
        or log_decay.shape != (BW, T, HV)
        or beta.shape != log_decay.shape
        or shared_state.shape != (BW // beam_width, HV, K, V)
    ):
        raise ValueError("incompatible state, value, or gate shape")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q/k/v must have the same dtype")
    scale = K**-0.5 if scale is None else scale
    selected = _backend(backend, tensors)
    if return_final_state:
        if backend == "triton":
            raise ValueError("Triton decode does not return a final state")
        selected = "torch"
    if triton_config is not None and selected != "triton":
        raise ValueError("triton_config requires the Triton execution path")
    if selected == "triton":
        from .kernels.gated_delta import beam_fused_recurrent_gdn

        return beam_fused_recurrent_gdn(
            q,
            k,
            v,
            log_decay,
            beta,
            shared_state,
            beam_width,
            scale,
            triton_config=triton_config,
        )
    # 按请求读取共享初始状态；参考实现允许临时的 beam 维中间量。
    outputs, final_states = [], []
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    for request in range(BW // beam_width):
        start = request * beam_width
        stop = start + beam_width
        state = shared_state[request].to(dtype).unsqueeze(0)
        qt, kt = q[start:stop].to(dtype), k[start:stop].to(dtype)
        qt = qt / (qt.square().sum(-1, keepdim=True) + 1e-6).sqrt() * scale
        kt = kt / (kt.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        qt = qt.repeat_interleave(HV // H, dim=2)
        kt = kt.repeat_interleave(HV // H, dim=2)
        steps = []
        for t in range(T):
            state = state * log_decay[start:stop, t].to(dtype).exp()[..., None, None]
            correction = beta[start:stop, t].to(dtype)[..., None] * (
                v[start:stop, t].to(dtype) - (state * kt[:, t, :, :, None]).sum(-2)
            )
            state = state + kt[:, t, :, :, None] * correction[..., None, :]
            steps.append((state * qt[:, t, :, :, None]).sum(-2))
        outputs.append(torch.stack(steps, dim=1))
        if return_final_state:
            final_states.append(state)
    output = torch.cat(outputs).to(v.dtype)
    return (output, torch.cat(final_states)) if return_final_state else output


def causal_conv1d(
    x,
    shared_state,
    weight,
    *,
    beam_width: int = 1,
    bias=None,
    activation: str | None = "silu",
    backend: str = "auto",
):
    """Stateless depthwise short convolution.

    x: [B*W,C,T]; shared_state: [B,C,history]; weight: [C,width].
    Only the last width-1 prefix taps are consumed. Returns [B*W,C,T].
    Triton currently specializes width=4, SiLU, no bias; auto otherwise uses
    torch. Both paths accept strided x and leave shared_state untouched.
    """
    _width(beam_width)
    tensors = (x, shared_state, weight) + (() if bias is None else (bias,))
    _same_device(tensors)
    if x.ndim != 3 or shared_state.ndim != 3 or weight.ndim != 2:
        raise ValueError("invalid convolution input ranks")
    BW, C, T = x.shape
    width = weight.shape[-1]
    if (
        min(BW, C, T, width) < 1
        or BW % beam_width
        or shared_state.shape[:2] != (BW // beam_width, C)
        or shared_state.shape[-1] < width - 1
        or weight.shape[0] != C
    ):
        raise ValueError("incompatible convolution shapes")
    if len({t.dtype for t in tensors}) != 1:
        raise ValueError("convolution inputs must have the same dtype")
    if bias is not None and bias.shape != (C,):
        raise ValueError("bias must have shape [C]")
    if activation not in (None, "silu"):
        raise ValueError("activation must be None or silu")
    selected = _backend(backend, tensors)
    if width != 4 or bias is not None or activation != "silu":
        if backend == "triton":
            raise ValueError("Triton short-conv requires width=4, SiLU, and no bias")
        selected = "torch"
    if selected == "triton":
        from .kernels.causal_conv1d import beam_causal_conv1d_update

        return beam_causal_conv1d_update(
            x, shared_state, weight, bias, activation, beam_width
        )
    outputs = []
    for request in range(BW // beam_width):
        prefix = (
            shared_state[request : request + 1, :, -(width - 1) :]
            if width > 1
            else shared_state[request : request + 1, :, :0]
        )
        suffix = x[request * beam_width : (request + 1) * beam_width]
        window = torch.cat((prefix.expand(beam_width, -1, -1), suffix), dim=-1)
        out = F.conv1d(window, weight[:, None, :], bias, groups=C)
        outputs.append(F.silu(out) if activation == "silu" else out)
    return torch.cat(outputs)


def _attention_part(scores, values):
    # 空前缀或整行被 mask 时，输出必须为零而不是 softmax(-inf) 的 NaN。
    if scores.shape[-1] == 0:
        return scores.new_zeros(*scores.shape[:-1], values.shape[-1]), scores.new_full(
            scores.shape[:-1], -torch.inf
        )
    present = torch.isfinite(scores).any(-1, keepdim=True)
    safe = torch.where(present, scores, torch.zeros_like(scores))
    probabilities = torch.softmax(safe, dim=-1) * present
    out = torch.matmul(probabilities, values)
    lse = torch.where(present.squeeze(-1), safe.logsumexp(-1), -torch.inf)
    return out, lse


def merge_attention(prefix_out, prefix_lse, suffix_out, suffix_lse):
    """Merge two independently normalized attention partitions using LSE."""
    total = torch.logaddexp(prefix_lse, suffix_lse)
    return (
        prefix_out * (prefix_lse - total).exp()[..., None]
        + suffix_out * (suffix_lse - total).exp()[..., None]
    )


def prefix_attention(
    q,
    prefix_k,
    prefix_v,
    suffix_k,
    suffix_v,
    *,
    beam_width: int = 1,
    prefix_mask: torch.Tensor | None = None,
    scale: float | None = None,
    backend: str = "torch",
):
    """Shared-prefix attention plus a causal replay suffix, with GQA support.

    Layout: q [B*W,T,H,D], prefix K/V [B,L,HK,D], suffix K/V
    [B*W,T,HK,D]. prefix_mask [B,L] is True for valid keys. All suffix
    tokens are valid; variable-length suffixes are outside this initial API.
    Torch supports CPU/MPS/CUDA and autograd. flash_attn is opt-in inference.
    """
    _width(beam_width)
    tensors = (q, prefix_k, prefix_v, suffix_k, suffix_v)
    _same_device(tensors)
    if any(t.ndim != 4 for t in tensors):
        raise ValueError("attention inputs must all have rank 4")
    BW, T, H, D = q.shape
    B, L, HK, DK = prefix_k.shape
    if (
        min(BW, T, H, D, B, HK) < 1
        or BW != B * beam_width
        or H % HK
        or DK != D
        or prefix_v.shape != prefix_k.shape
        or suffix_k.shape != (BW, T, HK, D)
        or suffix_v.shape != suffix_k.shape
    ):
        raise ValueError("incompatible attention shapes")
    if len({t.dtype for t in tensors}) != 1:
        raise ValueError("attention inputs must have the same dtype")
    if prefix_mask is not None and (
        prefix_mask.shape != (B, L)
        or prefix_mask.dtype != torch.bool
        or prefix_mask.device != q.device
    ):
        raise ValueError("prefix_mask must be bool [B,L] on the query device")
    scale = 1 / math.sqrt(D) if scale is None else scale
    if backend == "flash_attn":
        if not all(t.is_cuda for t in tensors) or (
            torch.is_grad_enabled() and any(t.requires_grad for t in tensors)
        ):
            raise RuntimeError(
                "flash_attn shared-prefix backend is CUDA inference-only"
            )
        from .kernels.flash_attention import prefix_attention_flash

        return prefix_attention_flash(
            q, prefix_k, prefix_v, suffix_k, suffix_v, beam_width, prefix_mask, scale
        )
    if backend != "torch":
        raise ValueError("attention backend must be torch or flash_attn")
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    query = q.to(dtype).reshape(B, beam_width, T, H, D).permute(0, 1, 3, 2, 4)
    pk = prefix_k.to(dtype).repeat_interleave(H // HK, dim=2).permute(0, 2, 1, 3)
    pv = prefix_v.to(dtype).repeat_interleave(H // HK, dim=2).permute(0, 2, 1, 3)
    # 将 beam 放入 query 序列维，避免 matmul 广播隐式物化 W 份前缀 KV。
    prefix_query = query.permute(0, 2, 1, 3, 4).reshape(B, H, beam_width * T, D)
    scores = torch.matmul(prefix_query, pk.transpose(-1, -2)) * scale
    if prefix_mask is not None:
        scores = scores.masked_fill(~prefix_mask[:, None, None, :], -torch.inf)
    op, lp = _attention_part(scores, pv)
    op = op.reshape(B, H, beam_width, T, D).permute(0, 2, 1, 3, 4)
    lp = lp.reshape(B, H, beam_width, T).permute(0, 2, 1, 3)
    sk = (
        suffix_k.to(dtype)
        .repeat_interleave(H // HK, dim=2)
        .reshape(B, beam_width, T, H, D)
        .permute(0, 1, 3, 2, 4)
    )
    sv = (
        suffix_v.to(dtype)
        .repeat_interleave(H // HK, dim=2)
        .reshape(B, beam_width, T, H, D)
        .permute(0, 1, 3, 2, 4)
    )
    causal = torch.ones(T, T, device=q.device, dtype=torch.bool).tril()
    scores = (torch.matmul(query, sk.transpose(-1, -2)) * scale).masked_fill(
        ~causal, -torch.inf
    )
    os, ls = _attention_part(scores, sv)
    out = merge_attention(op, lp, os, ls)
    return out.permute(0, 1, 3, 2, 4).reshape(BW, T, H, D).to(q.dtype)
