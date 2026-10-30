"""GRBeam shared-prefix kernel benchmarks with reference checks before timing."""

import argparse
import gc
import importlib.metadata as metadata
import itertools
import json
from pathlib import Path
import statistics
import time

import torch

from kadllm.ops import causal_conv1d, gated_delta, prefix_attention


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    p.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    p.add_argument('--ops', nargs='+', choices=['gdn', 'conv', 'attention'], default=['gdn', 'conv', 'attention'])
    p.add_argument('--backends', nargs='+', choices=['torch', 'accelerated'], default=['torch'])
    p.add_argument('--batches', nargs='+', type=int, default=[1])
    p.add_argument('--beams', nargs='+', type=int, default=[2])
    p.add_argument('--suffixes', nargs='+', type=int, default=[2])
    p.add_argument('--prefixes', nargs='+', type=int, default=[32])
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--head-dim', type=int, default=128)
    p.add_argument('--channels', type=int, default=128)
    p.add_argument('--gdn-config', nargs='+', type=int)
    p.add_argument('--warmups', type=int, default=10)
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--atol', type=float, default=None)
    p.add_argument('--rtol', type=float, default=None)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if min(a.batches + a.beams + a.suffixes + a.prefixes + [a.heads, a.head_dim, a.channels, a.warmups, a.repeats]) < 1:
        p.error('dimensions, warmups and repeats must be positive')
    if 'accelerated' in a.backends and a.device != 'cuda':
        p.error('accelerated backends require CUDA')
    if 'accelerated' in a.backends and 'attention' in a.ops and a.dtype == 'fp32':
        p.error('FlashAttention requires --dtype fp16 or bf16')
    if a.gdn_config:
        from kadllm.ops import _gdn_launch
        try:
            _gdn_launch(tuple(a.gdn_config))
        except ValueError as exc:
            p.error(str(exc))
    if any(x is not None and x < 0 for x in [a.atol, a.rtol]):
        p.error('tolerances must be nonnegative')
    dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[a.dtype]
    tol = 1e-4 if dtype == torch.float32 else 0.03
    atol = tol if a.atol is None else a.atol
    rtol = tol if a.rtol is None else a.rtol
    torch.manual_seed(a.seed)
    torch.set_num_threads(1)
    versions = {'torch': torch.__version__}
    for name in ['triton', 'flash-attn']:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    report = {'metadata': {'versions': versions, 'device': a.device, 'dtype': a.dtype,
              'seed': a.seed, 'warmups': a.warmups, 'repeats': a.repeats,
              'atol': atol, 'rtol': rtol, 'gdn_config': a.gdn_config,
              'timing': 'eager wall time including synchronization; no CUDA graphs',
              'memory': 'CUDA allocator peak including inputs and validation buffers; excludes driver/context'}, 'cases': []}
    if a.device == 'cuda':
        report['metadata']['gpu'] = torch.cuda.get_device_name()
    a.output.parent.mkdir(parents=True, exist_ok=True)

    def sync():
        if a.device == 'cuda':
            torch.cuda.synchronize()
        elif a.device == 'mps':
            torch.mps.synchronize()

    def rand(*shape):
        return torch.randn(shape, device=a.device, dtype=dtype) * 0.1

    def run_case(op, batch, beam, suffix, prefix):
        n, h, d = batch * beam, a.heads, a.head_dim
        if op == 'gdn':
            args = (rand(n, suffix, h, d), rand(n, suffix, h, d), rand(n, suffix, h, d),
                    -torch.rand(n, suffix, h, device=a.device),
                    torch.rand(n, suffix, h, device=a.device), rand(batch, h, d, d))
            state = [args[-1]]
            def call(backend):
                return gated_delta(*args, beam_width=beam,
                    backend='triton' if backend == 'accelerated' else 'torch',
                    triton_config=tuple(a.gdn_config) if a.gdn_config and backend == 'accelerated' else None)
        elif op == 'conv':
            args = (rand(n, a.channels, suffix), rand(batch, a.channels, 3), rand(a.channels, 4))
            state = [args[1]]
            def call(backend):
                return causal_conv1d(*args, beam_width=beam, activation='silu',
                    backend='triton' if backend == 'accelerated' else 'torch')
        else:
            args = (rand(n, suffix, h, d), rand(batch, prefix, h, d), rand(batch, prefix, h, d),
                    rand(n, suffix, h, d), rand(n, suffix, h, d))
            state = [args[1], args[2]]
            def call(backend):
                return prefix_attention(*args, beam_width=beam,
                    backend='flash_attn' if backend == 'accelerated' else 'torch')
        originals = [x.clone() for x in state]
        reference = call('torch')
        rows = []
        for backend in a.backends:
            row = {'op': op, 'backend': backend, 'batch': batch, 'beam': beam,
                   'suffix': suffix, 'prefix': prefix if op == 'attention' else None,
                   'heads': h, 'head_dim': d, 'channels': a.channels if op == 'conv' else None}
            try:
                actual = call(backend)
                row['max_abs_error'] = (actual.float() - reference.float()).abs().max().item()
                torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
                for before, after in zip(originals, state):
                    if not torch.equal(before, after):
                        raise RuntimeError('prefix was modified')
                del actual
                # Reference and prefix snapshots remain allocated during timing.
                for _ in range(a.warmups):
                    call(backend)
                sync()
                if a.device == 'cuda':
                    row['baseline_allocated_mib'] = torch.cuda.memory_allocated() / 2**20
                    torch.cuda.reset_peak_memory_stats()
                times = []
                for _ in range(a.repeats):
                    sync()
                    start = time.perf_counter()
                    call(backend)
                    sync()
                    times.append((time.perf_counter() - start) * 1000)
                for before, after in zip(originals, state):
                    if not torch.equal(before, after):
                        raise RuntimeError('prefix was modified during timing')
                row.update(status='measured', rt_ms=times, mean_rt_ms=statistics.mean(times),
                           median_rt_ms=statistics.median(times))
                if a.device == 'cuda':
                    row['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
            except (RuntimeError, AssertionError, ValueError, ImportError) as exc:
                row.update(status='failed', error=str(exc))
            rows.append(row)
        return rows

    with torch.inference_mode():
        for op in a.ops:
            prefixes = a.prefixes if op == 'attention' else [0]
            for batch, beam, suffix, prefix in itertools.product(a.batches, a.beams, a.suffixes, prefixes):
                try:
                    rows = run_case(op, batch, beam, suffix, prefix)
                except (RuntimeError, ValueError, ImportError) as exc:
                    rows = [dict(op=op, batch=batch, beam=beam, suffix=suffix, prefix=prefix,
                                 status='failed', error=str(exc))]
                report['cases'].extend(rows)
                a.output.write_text(json.dumps(report, indent=2) + '\n')
                for row in rows:
                    print(json.dumps({k: v for k, v in row.items() if k != 'rt_ms'}), flush=True)
                gc.collect()
                if a.device == 'cuda':
                    torch.cuda.empty_cache()
    return 1 if any(row['status'] != 'measured' for row in report['cases']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
