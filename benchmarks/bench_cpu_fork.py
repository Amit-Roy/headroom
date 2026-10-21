#!/usr/bin/env python3
"""Compare CPU defaults and compression output against an upstream checkout.

The default mode uses a no-op neural stand-in to audit admission and exact
structural output. --with-model also measures cached local ONNX inference.
Each checkout runs in a separate, time-limited subprocess. Optional prose and
repeated samples check output equivalence without skipping inference. No API
calls or model downloads are made. Counts use cl100k_base, not provider billing.
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


def worker(with_model: bool, *, include_prose: bool = False, repeats: int = 1) -> None:
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
            "cpu_arena": options.enable_cpu_mem_arena,
            "memory_pattern": options.enable_mem_pattern,
        }
        print(json.dumps({"progress": "model_ready", "model": model}), flush=True)

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
    if include_prose:
        samples["service_notes"] = " ".join(
            [
                "The settlement service processes customer payments each morning. "
                "Operators must never retry a payment marked do_not_retry. "
                "The retry limit is 3 and the request timeout is 30 seconds. "
                "A failed health check leaves the previous configuration active. "
                "Inspect the transaction identifier before changing the retry policy. "
                "Successful requests are recorded in the audit log."
            ]
            * 6
        )
    rows = []
    for name, content in samples.items():
        for repetition in range(repeats):
            calls = 0
            cpu = time.process_time()
            started = time.perf_counter()
            strategy = (
                CompressionStrategy.SEARCH
                if name == "repeated_search_paths"
                else CompressionStrategy.KOMPRESS
            )
            output, _, _ = router._apply_strategy_to_content(content, strategy, "")
            before, after = len(encoder.encode(content)), len(encoder.encode(output))
            rows.append(
                {
                    "sample": name,
                    "repetition": repetition + 1,
                    "input_sha256": hashlib.sha256(content.encode()).hexdigest(),
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
            print(json.dumps({"progress": "sample_done", **rows[-1]}), flush=True)
    print(json.dumps({"model": model, "measurements": rows}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--candidate-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--with-model", action="store_true")
    parser.add_argument("--include-prose", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--verify-output-equivalence", action="store_true")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.verify_output_equivalence and not args.with_model:
        parser.error("--verify-output-equivalence requires --with-model")
    if args.worker:
        worker(args.with_model, include_prose=args.include_prose, repeats=args.repeats)
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
        if args.include_prose:
            command.append("--include-prose")
        command.extend(["--repeats", str(args.repeats)])
        try:
            completed = subprocess.run(
                command, env=env, capture_output=True, text=True, timeout=args.timeout, check=True
            )
            result = json.loads(completed.stdout.strip().splitlines()[-1])
            results["results"].append({"checkout": label, **result})
        except subprocess.TimeoutExpired as error:
            partial = []
            for line in (error.stdout or b"").decode(errors="replace").splitlines():
                try:
                    partial.append(json.loads(line))
                except ValueError:
                    continue
            results["results"].append(
                {
                    "checkout": label,
                    "error": "process deadline exceeded",
                    "partial_progress": partial,
                }
            )
        except (subprocess.CalledProcessError, ValueError) as error:
            results["results"].append({"checkout": label, "error": type(error).__name__})
    if args.verify_output_equivalence:
        runs = results["results"]
        results["output_equivalent"] = (
            not any("error" in run for run in runs)
            and len(runs[0]["measurements"]) == len(runs[1]["measurements"])
            and all(
                all(
                    a[key] == b[key]
                    for key in (
                        "sample",
                        "repetition",
                        "input_sha256",
                        "output_sha256",
                        "after_tokens",
                        "neural_calls",
                    )
                )
                for a, b in zip(runs[0]["measurements"], runs[1]["measurements"])
            )
        )
    output = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(output + "\n")
    print(output)
    if any("error" in result for result in results["results"]):
        raise SystemExit(1)
    if results.get("output_equivalent") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
