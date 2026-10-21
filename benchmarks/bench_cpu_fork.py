#!/usr/bin/env python3
"""Compare CPU defaults and second-pass admission against an upstream checkout.

The default mode uses a no-op neural stand-in to audit admission and exact
structural output. --with-model also measures cached local ONNX inference.
Each checkout runs in a separate, time-limited subprocess. No API calls or
model downloads are made. Counts use cl100k_base, not provider billing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path


def worker(with_model: bool) -> None:
    import tiktoken

    from headroom.transforms.content_router import (
        CompressionStrategy,
        ContentRouter,
        ContentRouterConfig,
    )
    from headroom.transforms.lossless_compaction import search_unheading

    encoder = tiktoken.get_encoding("cl100k_base")
    router = ContentRouter(ContentRouterConfig(lossless_then_lossy=True, ccr_inject_marker=False))
    model = None
    if with_model:
        from headroom.transforms.kompress_compressor import _load_kompress

        loaded, _, backend = _load_kompress(allow_download=False)
        options = loaded._session.get_session_options()
        model = {
            "backend": backend,
            "intra_threads": options.intra_op_num_threads,
            "inter_threads": options.inter_op_num_threads,
        }

    calls = 0
    neural = router._try_ml_compressor

    def counted(text, context, question=None):
        nonlocal calls
        calls += 1
        return neural(text, context, question) if with_model else (text, len(text.split()))

    router._try_ml_compressor = counted
    samples = {
        "repeated_search_paths": "\n".join(
            f"src/services/payments/processing/settlement.py:{i}:    do_not_retry(order_id)"
            for i in range(1, 140)
        )
        + "\n",
    }
    rows = []
    for name, content in samples.items():
        cpu = time.process_time()
        started = time.perf_counter()
        output, _, _ = router._apply_strategy_to_content(content, CompressionStrategy.SEARCH, "")
        before, after = len(encoder.encode(content)), len(encoder.encode(output))
        rows.append(
            {
                "sample": name,
                "wall_seconds": time.perf_counter() - started,
                "cpu_seconds": time.process_time() - cpu,
                "neural_calls": calls,
                "before_tokens": before,
                "after_tokens": after,
                "tokens_saved_percent": 100 * (before - after) / before,
                "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                "critical_identifier_retained": "do_not_retry" in output,
                "exact_recovery": search_unheading(output) == content,
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            }
        )
    print(json.dumps({"model": model, "measurements": rows}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--candidate-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--with-model", action="store_true")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.with_model)
        return
    if args.baseline_root is None:
        parser.error("--baseline-root must point to an unmodified upstream checkout")
    results = {
        "mode": "cached_local_model" if args.with_model else "admission_only",
        "tokenizer": "cl100k_base",
        "results": [],
    }
    for label, checkout in [("upstream", args.baseline_root), ("cpu_fork", args.candidate_root)]:
        env = os.environ.copy()
        for key in list(env):
            if key.startswith("HEADROOM_"):
                env.pop(key)
        env.update(
            {
                "PYTHONPATH": str(checkout.resolve()),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "HEADROOM_KOMPRESS_BACKEND": "onnx_cpu",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
        )
        command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
        if args.with_model:
            command.append("--with-model")
        try:
            completed = subprocess.run(
                command, env=env, capture_output=True, text=True, timeout=args.timeout, check=True
            )
            result = json.loads(completed.stdout.strip().splitlines()[-1])
            results["results"].append({"checkout": label, **result})
        except subprocess.TimeoutExpired:
            results["results"].append({"checkout": label, "error": "process deadline exceeded"})
        except (subprocess.CalledProcessError, ValueError) as error:
            results["results"].append({"checkout": label, "error": type(error).__name__})
    output = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(output + "\n")
    print(output)
    if any("error" in result for result in results["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
