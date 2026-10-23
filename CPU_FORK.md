# CPU-friendly Headroom fork

This fork targets lower CPU cost while preserving upstream compression behavior.
Native ONNX sessions default to one intra-operation thread and one inter-operation
thread. Explicit caller values and existing model-specific thread overrides retain
precedence. Existing idle-spin protections remain. The model, thresholds, chunk
sizes, request budgets, and structural compressors are unchanged.

## Compression policy

Neural second passes now run under the same admission rules as upstream by
default. The first fork patch skipped them after substantial structural savings;
that reduced token savings from 80.23% to 52.25% on one search fixture. Skipping
inference is a compression tradeoff, not an equivalent-output optimization.

SDK callers may explicitly choose that tradeoff with
`ContentRouterConfig(skip_redundant_ml=True)`. It skips a second pass if fewer
than 256 estimated tokens remain or structural savings reach 50%. The two
thresholds remain configurable. `HEADROOM_SKIP_REDUNDANT_ML=0` disables an SDK
opt-in; setting it to `1` does not enable skipping by itself.

## Validation

184 focused tests passed; 11 skipped for optional dependencies. Coverage includes
default chaining after folded search and structural logs, explicit skipping,
ONNX thread overrides, lossless-only routing, Kompress safeguards, and deadlines.

Cached-model diagnostics compare the original upstream commit `fbf7bc3` with this
fork. Both use the same cached `chopratejas/kompress-v2-base` ONNX weights and
compression settings. Counts use `cl100k_base`, not provider billing. Request
wall and process CPU times exclude model loading. Each condition runs in its own
subprocess, sequentially, on the same shared eight-core Linux host.

The benchmark checks identical input/output SHA-256 hashes, forwarded token
counts, and neural-call counts. Two synthetic fixtures cover repeated search
paths and service notes. These checks establish equivalent outputs on those
fixtures, not general task accuracy or representative production savings.

## Equivalent-output measurements

| Fixture | Forwarded tokens, both | Savings, both | CPU seconds, upstream / fork | Wall seconds, upstream / fork |
|---|---:|---:|---:|---:|
| repeated_search_paths | 522 | 80.23% | 29.632 / 9.966 | 23.961 / 10.421 |
| service_notes | 349 | 10.51% | 22.766 / 6.654 | 13.897 / 6.973 |

Output hashes and neural-call counts matched for both fixtures. Search CPU time
fell 66.4%; service-note CPU time fell 70.8%. These are single paired measurements
on a busy shared host, not stable production latency guarantees.

Raw results: `benchmarks/results/cpu-equivalence.json`. A longer four-run
comparison exceeded the upstream process deadline of 85 seconds; the fork
completed with consistent hashes across repeats. That incomplete comparison
is retained as `cpu-equivalence-stress.json` and does not establish paired
performance or equivalence for the timed-out condition.

## Reproduction

```bash
python benchmarks/bench_cpu_fork.py \
  --baseline-root /path/to/upstream-fbf7bc3 \
  --with-model --include-prose --verify-output-equivalence --timeout 85 \
  --output /tmp/cpu-equivalence.json
```

`--repeats N` repeats each fixture after loading. No provider calls or model
downloads are made. Exit status is 1 for an execution failure and 2 for output
inequivalence. A timeout is not treated as equivalent compression. Completed
worker progress is retained in timeout reports. Omit `--with-model` and the
verification flag to use a no-op neural stand-in for a routing-only audit.

The normal runtime dependencies and native extension are required. Use an
unmodified upstream checkout as the baseline; the fork's `main` is not upstream.
Historical admission-only reports remain in `benchmarks/results/cpu-*.json`;
they describe the first patch rather than current defaults.

## Controls and limitations

Existing `HEADROOM_KOMPRESS_ONNX_INTRA_THREADS` and
`HEADROOM_KOMPRESS_ONNX_INTER_THREADS` overrides still work. Explicit `0` restores
ONNX automatic thread sizing. Single-thread inference can occupy one core while
running; reduced CPU-seconds do not imply a fixed CPU-percentage ceiling.

Native inference still cannot be cancelled mid-call. Deadlines are cooperative,
and direct neural compression can remain slow. These source patches do not
change the live Agent Console installation. Live rollout requires separate
integration and fallback validation.

Set `HEADROOM_CPU_TARGET_PERCENT` to a finite value from 1 to 100 to enable
cooperative process-CPU pacing around ONNX session construction and inference.
Unset or set it to `0` to disable pacing. Pacing serializes these selected
phases and sleeps on the calling worker thread after each phase; it does not
change model inputs or outputs. This is not a strict CPU cap: a native call
lasting over 30 seconds can still trip the existing guard. Use an OS CPU quota
for a strict ceiling. Startup canary timing excludes deliberate pacing sleeps,
while request deadlines continue to include them.

The original Apache-2.0 license and attribution notices are retained. Modified
upstream files carry a notice describing the fork's changes.
