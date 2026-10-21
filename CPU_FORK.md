# CPU-friendly Headroom fork

This fork targets predictable CPU use on a shared CPU-only host. It keeps the
upstream compression pipeline and model, with two conservative changes:

- Native ONNX sessions default to one intra-operation thread and one
  inter-operation thread. Explicit caller values and existing model-specific
  thread settings retain precedence. Existing idle-spin protections remain.
- A neural second pass is skipped after a structural or lossless result when
  fewer than 256 estimated tokens remain, or the first pass already removed
  at least 50%. The structural result is returned unchanged. Direct prose
  compression remains available.

The second rule trades some possible additional token reduction for lower
CPU use, latency, and preservation of more original content. It is not a claim
that neural compression never helps.

## Validation

The implementation was checked against 180 focused tests covering ONNX options,
thread overrides, routing admission, lossless chaining, Kompress safeguards,
and deadlines; 11 tests skipped for optional dependencies.

An isolated cached-model comparison used the same 139-line search output and
`chopratejas/kompress-v2-base` on an eight-core Linux CPU host. Both models
finished loading before measurement. Times exclude model loading and provider
inference. Tokens use `cl100k_base`, not provider billing.

| Measurement | Upstream | CPU-friendly fork |
|---|---:|---:|
| Native ONNX intra/inter threads | Automatic / automatic | 1 / 1 |
| Measured request wall time | 10.689 seconds | 0.037 seconds |
| Measured request CPU time | 25.330 seconds | 0.037 seconds |
| Input tokens | 2,641 | 2,641 |
| Forwarded content tokens | 522 | 1,261 |
| Token reduction | 80.23% | 52.25% |
| Neural second passes | 1 | 0 |
| Exact structural recovery | No | Yes |
| Peak process RSS, including model loading | 923.0 MiB | 532.0 MiB |

This is one diagnostic workload, not a representative production benchmark.
The lower inference work accounts for most of the improvement. A separate
admission-only run returned identical structural output in both checkouts.
Neither comparison measures end-to-end agent accuracy, streaming behavior,
CCR retrieval, concurrent workloads, or provider cache costs.

Reproduce without loading a model:

```bash
python benchmarks/bench_cpu_fork.py --baseline-root /path/to/upstream
```

Add `--with-model` to load locally cached weights and measure real inference.
Each checkout runs in a separate subprocess with a 45-second deadline. No
model downloads or provider requests are made. The benchmark requires the
normal runtime dependencies and a native extension built for each checkout.

## Controls and limitations

Set `HEADROOM_SKIP_REDUNDANT_ML=0` to restore upstream neural layering.
SDK callers can instead set `ContentRouterConfig(skip_redundant_ml=False)`;
the residual-size and savings thresholds are configurable on that dataclass.
Existing `HEADROOM_KOMPRESS_ONNX_INTRA_THREADS` and
`HEADROOM_KOMPRESS_ONNX_INTER_THREADS` overrides still work.

These changes do not provide hard cancellation of running native inference.
Direct neural compression can still be slow. Keep external resource limits,
request timeouts, and direct-provider fallback when testing this fork. The
live Agent Console installation is not changed by these source patches.

The original Apache-2.0 license and attribution notices are retained. Modified
upstream files carry a notice describing the fork's changes.
