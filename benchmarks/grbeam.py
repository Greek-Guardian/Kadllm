"""GRBeam and baseline beam benchmarks. Random weights measure execution, not quality.

RT covers the whole batch, including CPU result materialization. Request QPS is
batch / mean RT. By default prefix caches are reset outside timing before every invocation;
within-call prefix reuse, compilation and CUDA graphs remain enabled in vLLM.
Run one backend per otherwise idle GPU. Numerical audits are separate from timing.
"""

import argparse
import gc
import hashlib
import importlib
import importlib.metadata as md
import json
import os
import random
import statistics
import subprocess
import threading
import time
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output(tokens, batch, width, depth):
    if len(tokens) != batch or any(
        len(beams) != width
        or any(len(sequence) != depth for sequence in beams)
        or len({tuple(sequence) for sequence in beams}) != width
        for beams in tokens
    ):
        raise ValueError(
            "backend did not return the requested distinct fixed-depth beams"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--backend",
        required=True,
        choices=[
            "transformers",
            "vllm",
            "torch",
            "gdn",
            "conv",
            "both",
            "flash",
            "attn",
            "gdn_flash",
            "conv_flash",
            "external",
        ],
    )
    p.add_argument("--batches", nargs="+", type=int, default=[1, 8, 16, 32])
    p.add_argument("--lengths", nargs="+", type=int, default=[512, 1024])
    p.add_argument("--depths", nargs="+", type=int, default=[2, 3])
    p.add_argument("--beam", type=int, default=256)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument(
        "--gdn-config", type=int, nargs="+",
        help="GRBeam Triton: TILE_V WARPS STAGES [BEAM_GROUP]; record explicit tuning",
    )
    p.add_argument(
        "--vllm-gdn-backend",
        choices=["auto", "triton", "flashinfer"],
        default="auto",
        help="GDN prefill selection supported by vLLM 0.19+",
    )
    p.add_argument("--label", help="Result label; backend execution remains unchanged")
    p.add_argument(
        "--adapter-factory", help="External module:function returning (adapter, search)"
    )
    p.add_argument(
        "--cache-mode",
        choices=["cold", "warm"],
        default="cold",
        help="cold resets between calls; warm allows cross-call reuse",
    )
    p.add_argument(
        "--order-seed", type=int, default=None, help="shuffle case order reproducibly"
    )
    p.add_argument("--max-model-len", type=int, default=2048)
    a = p.parse_args()
    if a.gdn_config:
        if a.backend not in ("gdn", "both", "flash", "gdn_flash"):
            p.error("--gdn-config requires a GRBeam Triton GDN backend")
        from kadllm.ops import _gdn_launch

        try:
            _gdn_launch(tuple(a.gdn_config))
        except ValueError as exc:
            p.error(str(exc))
    if (
        min(a.batches + a.lengths + a.depths + [a.beam, a.warmups, a.repeats]) < 1
        or max(a.batches) > 32
        or a.beam > 512
        or a.beam < 2
        or max(a.lengths) + max(a.depths) > a.max_model_len
        or a.max_model_len > 8192
    ):
        p.error(
            "require positive counts, batch <= 32, 2 <= beam <= 512, total length <= max-model-len <= 8192"
        )
    if a.backend == "external" and not a.adapter_factory:
        p.error("external requires --adapter-factory module:function")
    import torch

    if torch.cuda.device_count() != 1:
        p.error("expose exactly one otherwise idle GPU")
    torch.set_num_threads(1)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    def memory():
        return int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    gpu,
                ],
                text=True,
            ).strip()
        )

    baseline = memory()
    if baseline > 64:
        raise RuntimeError("GPU baseline must be <= 64 MiB")
    metadata = dict(
        gpu_name=torch.cuda.get_device_name(),
        compute_capability=torch.cuda.get_device_capability(),
        backend=a.label or a.backend,
        execution_backend=a.backend,
        gdn_triton_config=a.gdn_config,
        dtype="bfloat16",
        beam=a.beam,
        warmups=a.warmups,
        repeats=a.repeats,
        cache_mode=a.cache_mode,
        adapter_factory=a.adapter_factory,
        order_seed=a.order_seed,
        max_model_len=a.max_model_len,
        torch=torch.__version__,
        transformers=md.version("transformers"),
        baseline_mib=baseline,
        config_sha256=file_sha256(Path(a.model, "config.json")),
        weight_sha256=file_sha256(Path(a.model, "model.safetensors")),
        memory_scope="whole GPU sampled every 100 ms during warmup and measurement",
        numerical_validation="separate audit; token equality is not asserted",
    )
    for package in ("flash-attn", "causal-conv1d", "flash-linear-attention"):
        try:
            metadata[package] = md.version(package)
        except md.PackageNotFoundError:
            metadata[package] = None
    if a.backend == "vllm":
        from vllm import LLM
        from vllm.sampling_params import BeamSearchParams

        extra_config = {}
        if a.vllm_gdn_backend != "auto":
            if tuple(map(int, md.version("vllm").split(".")[:2])) < (0, 19):
                raise ValueError("explicit GDN prefill selection requires vLLM >= 0.19")
            extra_config["additional_config"] = {
                "gdn_prefill_backend": a.vllm_gdn_backend
            }
        metadata["vllm_gdn_backend_requested"] = a.vllm_gdn_backend
        llm = LLM(
            **extra_config,
            model=a.model,
            dtype="bfloat16",
            max_model_len=a.max_model_len,
            gpu_memory_utilization=0.8,
            max_num_seqs=256,
            # 固定调度预算，避免硬件相关默认值改变跨设备比较口径。
            max_num_batched_tokens=8192,
            compilation_config={
                "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256]
            },
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            max_logprobs=2 * a.beam,
            limit_mm_per_prompt={"image": 0, "video": 0},
            seed=20261024,
            disable_log_stats=False,
        )
        metadata["vllm"] = md.version("vllm")
        for name in ("cache_config", "scheduler_config", "compilation_config"):
            metadata[name] = str(getattr(llm.llm_engine.vllm_config, name))
    else:
        from transformers import Qwen3_5ForConditionalGeneration

        from kadllm import grbeam_search
        from kadllm.adapters import Qwen35GRBeamAdapter

        model = (
            Qwen3_5ForConditionalGeneration.from_pretrained(
                a.model,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .eval()
            .cuda()
        )
        if a.backend == "external":
            module, function = a.adapter_factory.split(":", 1)
            adapter, grbeam_search = getattr(importlib.import_module(module), function)(
                model
            )
        elif a.backend != "transformers":
            opts = {
                "torch": ("torch", "torch", "torch"),
                "gdn": ("triton", "torch", "torch"),
                "conv": ("torch", "triton", "torch"),
                "both": ("triton", "triton", "torch"),
                "flash": ("triton", "triton", "flash_attn"),
                "attn": ("torch", "torch", "flash_attn"),
                "gdn_flash": ("triton", "torch", "flash_attn"),
                "conv_flash": ("torch", "triton", "flash_attn"),
            }
            g, c, f = opts[a.backend]
            adapter = Qwen35GRBeamAdapter(
                model.model.language_model,
                model.lm_head,
                gdn_backend=g,
                conv_backend=c,
                attention_backend=f,
                gdn_triton_config=tuple(a.gdn_config) if a.gdn_config else None,
            )
    rows = []
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cases = [
        (b, length, d) for b in a.batches for length in a.lengths for d in a.depths
    ]
    if a.order_seed is not None:
        random.Random(a.order_seed).shuffle(cases)
    for batch, length, depth in cases:
        row = dict(batch=batch, prompt_length=length, depth=depth)
        prompts = [
            [10 + (i + request * 17) % 100 for i in range(length)]
            for request in range(batch)
        ]
        ids = torch.tensor(prompts, device="cuda") if a.backend != "vllm" else None
        samples, stop = [], threading.Event()

        def monitor(stop=stop, samples=samples):
            while not stop.is_set():
                samples.append(memory())
                stop.wait(0.1)

        def run(prompts=prompts, ids=ids, batch=batch, depth=depth):
            if a.backend == "vllm":
                out = llm.beam_search(
                    [{"prompt_token_ids": x} for x in prompts],
                    BeamSearchParams(
                        beam_width=a.beam,
                        max_tokens=depth,
                        ignore_eos=True,
                        length_penalty=0,
                        temperature=0,
                    ),
                    use_tqdm=False,
                )
                return [[s.tokens[-depth:] for s in x.sequences] for x in out]
            if a.backend == "transformers":
                return (
                    model.generate(
                        ids,
                        attention_mask=torch.ones_like(ids),
                        num_beams=a.beam,
                        num_return_sequences=a.beam,
                        max_new_tokens=depth,
                        min_new_tokens=depth,
                        eos_token_id=None,
                        pad_token_id=0,
                        do_sample=False,
                        length_penalty=0,
                        use_cache=True,
                        logits_to_keep=1,
                    )[:, -depth:]
                    .reshape(batch, a.beam, depth)
                    .tolist()
                )
            return grbeam_search(adapter, ids, [a.beam] * depth).tokens.tolist()

        def reset():
            if (
                a.backend == "vllm"
                and a.cache_mode == "cold"
                and not llm.reset_prefix_cache()
            ):
                raise RuntimeError("prefix cache reset failed")

        if a.backend == "vllm" and a.cache_mode == "warm":
            if not llm.reset_prefix_cache():
                raise RuntimeError("initial warm-cache reset failed")
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            with torch.inference_mode():
                for _ in range(a.warmups):
                    reset()
                    row["beam_tokens"] = run()
                    validate_output(row["beam_tokens"], batch, a.beam, depth)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                times = []
                for _ in range(a.repeats):
                    reset()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    tokens = run()
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - start)
                    validate_output(tokens, batch, a.beam, depth)
                    if tokens != row["beam_tokens"]:
                        row["repeat_tokens_differ"] = True
                    overlaps = [
                        len({tuple(x) for x in left} & {tuple(x) for x in right})
                        / a.beam
                        for left, right in zip(tokens, row["beam_tokens"], strict=True)
                    ]
                    row.setdefault("repeat_beam_set_overlap", []).append(
                        statistics.mean(overlaps)
                    )
                row.update(
                    status="measured",
                    rt_ms=[x * 1000 for x in times],
                    mean_rt_ms=statistics.mean(times) * 1000,
                    request_qps=batch / statistics.mean(times),
                    median_rt_ms=statistics.median(times) * 1000,
                    stdev_rt_ms=statistics.stdev(times) * 1000
                    if len(times) > 1
                    else None,
                )
                if a.backend != "vllm":
                    row["torch_peak_reserved_mib"] = (
                        torch.cuda.max_memory_reserved() / 2**20
                    )
                    row["torch_peak_allocated_mib"] = (
                        torch.cuda.max_memory_allocated() / 2**20
                    )
        except torch.OutOfMemoryError:
            row.update(status="oom")
        finally:
            stop.set()
            thread.join(timeout=3)
            row["device_peak_mib"] = max(samples) if samples else None
            rows.append(row)
            output.write_text(
                json.dumps({"metadata": metadata, "cases": rows}, indent=2)
            )
            print(
                "CASE",
                json.dumps({k: v for k, v in row.items() if k != "beam_tokens"}),
                flush=True,
            )
            del ids, run
            gc.collect()
            torch.cuda.empty_cache()
    if a.backend == "vllm":
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
