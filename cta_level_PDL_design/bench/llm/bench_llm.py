#!/usr/bin/env python3
"""Run one *diagnostic* Tier-4 vLLM rung.

This file deliberately does not emit an admissible ``status=ok`` result.  A
single-rung process cannot satisfy EXPERIMENT_PLAN.md's same-process adjacent
triplet rule, and a parent-side Python patch is not sufficient evidence of the
code compiled by a vLLM worker.  ``run_llm_sweep.sh`` therefore runs a strict
preflight and currently blocks before invoking this driver.

The driver is still useful for building a future evidence-producing triplet:
it uses the vLLM 0.23 ``prompts=`` API, records >=31 independent repetitions
and bootstrap CIs, asks the worker about its PDL/graph configuration, validates
the number of generated outputs, and writes raw JSON.  Its SUMMARY row is
stamped ``status=diagnostic`` so tools/llm_bracket.py cannot mistake it for a
publishable measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any


FALSE_VALUES = {"0", "", "false", "False", "no", "No"}


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in FALSE_VALUES


def bootstrap_median_ci(values: list[float], samples: int, seed: int = 0) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap 95% CI for the median."""
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    boot = sorted(
        statistics.median(values[rng.randrange(n)] for _ in range(n))
        for _ in range(samples)
    )
    lo = boot[int(0.025 * (samples - 1))]
    hi = boot[int(0.975 * (samples - 1))]
    return lo, hi


def local_model_error(model: str) -> str | None:
    """Reject model IDs so a diagnostic can never download implicitly."""
    root = Path(model)
    if not root.is_dir():
        return f"model must be a fully staged local directory, got {model!r}"
    if not (root / "config.json").is_file():
        return f"local model is missing {root / 'config.json'}"
    weights = [
        p
        for pattern in ("*.safetensors", "*.bin", "*.gguf")
        for p in root.glob(pattern)
        if p.is_file()
    ]
    if not weights:
        return "local model has no .safetensors/.bin/.gguf weight file"
    return None


def apply_parent_ceiling_patch() -> list[str]:
    """Patch this process before vLLM forks; worker RPC must confirm inheritance.

    This is diagnostic scaffolding, not sufficient proof for a Tier-4 result.
    A spawn worker or an already compiled cache can evade it, hence the strict
    preflight additionally requires isolated PTX evidence.
    """
    import triton.language.extra.cuda as tlcuda  # type: ignore

    if not hasattr(tlcuda, "gdc_wait"):
        return []

    def no_wait(*_args: Any, **_kwargs: Any) -> None:
        return None

    no_wait._cta_pdl_ceiling_noop = True  # type: ignore[attr-defined]
    tlcuda.gdc_wait = no_wait
    return ["triton.language.extra.cuda.gdc_wait"]


def worker_runtime_probe(worker: Any) -> dict[str, Any]:
    """Executed by vLLM collective_rpc inside every model worker."""
    import os as worker_os
    import torch._inductor.config as inductor_config
    import triton.language.extra.cuda as tlcuda  # type: ignore

    config = getattr(worker, "vllm_config", None)
    compilation = getattr(config, "compilation_config", None)
    graph_mode = str(getattr(compilation, "cudagraph_mode", "UNKNOWN"))

    runner = getattr(worker, "model_runner", None)
    dispatcher = getattr(runner, "cudagraph_dispatcher", None)
    dispatcher_mode = str(getattr(dispatcher, "cudagraph_mode", "UNKNOWN"))
    wait_fn = getattr(tlcuda, "gdc_wait", None)
    return {
        "pid": worker_os.getpid(),
        "pdl_env": env_flag("TORCHINDUCTOR_ENABLE_PDL"),
        "pdl_inductor_config": bool(inductor_config.triton.enable_pdl),
        "ceiling_hook": bool(getattr(wait_fn, "_cta_pdl_ceiling_noop", False)),
        "graph_mode": graph_mode,
        "dispatcher_mode": dispatcher_mode,
    }


def output_token_ids(outputs: list[Any], expected_batch: int, expected_gen: int) -> list[list[int]]:
    if len(outputs) != expected_batch:
        raise RuntimeError(f"vLLM returned {len(outputs)} requests, expected {expected_batch}")
    token_ids: list[list[int]] = []
    for request in outputs:
        choices = getattr(request, "outputs", None)
        if not choices:
            raise RuntimeError("vLLM request has no generated output")
        ids = list(choices[0].token_ids)
        if len(ids) != expected_gen:
            raise RuntimeError(f"generated {len(ids)} tokens, expected {expected_gen}")
        token_ids.append(ids)
    return token_ids


def digest_outputs(token_ids: list[list[int]]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def expected_flags(rung: str) -> tuple[bool, bool, bool]:
    return {
        "pdl_off": (False, False, False),
        "pdl_grid": (True, True, False),
        "ceiling": (True, True, True),
    }[rung]


def run_vllm(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    # Imported only after the environment and optional ceiling patch are fixed.
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=args.model,
        max_model_len=args.seq + args.gen,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=False,
        enable_prefix_caching=False,
        dtype="bfloat16",
        compilation_config={"cudagraph_mode": "FULL"},
    )

    graph_mode = str(llm.llm_engine.vllm_config.compilation_config.cudagraph_mode)
    workers = llm.collective_rpc(worker_runtime_probe, timeout=120)
    if graph_mode != "FULL":
        raise RuntimeError(f"vLLM resolved cudagraph_mode={graph_mode}, expected FULL")
    if not workers:
        raise RuntimeError("vLLM returned no worker runtime probes")

    expected_pdl = args.rung != "pdl_off"
    expected_ceiling = args.rung == "ceiling"
    for probe in workers:
        if probe.get("graph_mode") != "FULL" or probe.get("dispatcher_mode") != "FULL":
            raise RuntimeError(f"worker did not resolve FULL graph mode: {probe}")
        if probe.get("pdl_env") is not expected_pdl:
            raise RuntimeError(f"worker PDL environment mismatch: {probe}")
        if probe.get("pdl_inductor_config") is not expected_pdl:
            raise RuntimeError(f"worker Inductor PDL config mismatch: {probe}")
        if probe.get("ceiling_hook") is not expected_ceiling:
            raise RuntimeError(f"worker ceiling-hook mismatch: {probe}")

    sampling = SamplingParams(temperature=0.0, max_tokens=args.gen, ignore_eos=True)
    # Each request differs, preventing identical-prefix reuse from turning the
    # requested batch into a prefix-cache microbenchmark.
    prompts = [
        TokensPrompt(
            prompt_token_ids=[1000 + ((position + request) % 5000) for position in range(args.seq)]
        )
        for request in range(args.batch)
    ]

    validation_ids: list[list[int]] | None = None
    for _ in range(args.warmups):
        outputs = llm.generate(prompts=prompts, sampling_params=sampling, use_tqdm=False)
        validation_ids = output_token_ids(outputs, args.batch, args.gen)

    latencies: list[float] = []
    throughput_samples: list[float] = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        outputs = llm.generate(prompts=prompts, sampling_params=sampling, use_tqdm=False)
        elapsed = time.perf_counter() - start
        ids = output_token_ids(outputs, args.batch, args.gen)
        validation_ids = validation_ids or ids
        latencies.append(elapsed)
        throughput_samples.append(args.batch * args.gen / elapsed)

    if validation_ids is None:
        raise RuntimeError("no validation generation was executed")

    latency_lo, latency_hi = bootstrap_median_ci(latencies, args.bootstrap_samples, seed=17)
    tps_lo, tps_hi = bootstrap_median_ci(throughput_samples, args.bootstrap_samples, seed=23)
    wall = sum(latencies)
    result = {
        "repetitions": len(latencies),
        "latency_samples_s": latencies,
        "tok_per_s_samples": throughput_samples,
        "median_s": statistics.median(latencies),
        "median_s_ci_low": latency_lo,
        "median_s_ci_high": latency_hi,
        "min_s": min(latencies),
        "wall_s": wall,
        "tok_per_s": statistics.median(throughput_samples),
        "tok_per_s_ci_low": tps_lo,
        "tok_per_s_ci_high": tps_hi,
        "tok_per_s_per_user": statistics.median(throughput_samples) / args.batch,
        "tok_per_s_per_user_ci_low": tps_lo / args.batch,
        "tok_per_s_per_user_ci_high": tps_hi / args.batch,
        "output_digest": digest_outputs(validation_ids),
    }
    runtime = {"driver_pid": os.getpid(), "graph_mode": graph_mode, "workers": workers}
    return result, runtime


def write_json(path: str, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="fully staged local model directory")
    parser.add_argument("--engine", default="vllm", choices=["vllm"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--gen", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--rung", default="pdl_grid", choices=["pdl_off", "pdl_grid", "ceiling"])
    parser.add_argument("--tag", default="llm_diagnostic")
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)
    parser.add_argument("--json", required=True, help="raw diagnostic JSON output")
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="acknowledge that this one-rung result is not admissible Tier-4 evidence",
    )
    args = parser.parse_args()

    if not args.diagnostic_only:
        print(
            "BLOCKED: the one-rung driver cannot produce an admissible Tier-4 measurement; "
            "use run_llm_sweep.sh, whose preflight currently requires a same-process proof triplet",
            file=sys.stderr,
        )
        return 3
    if args.repeats < 31:
        print("BLOCKED: diagnostic repetitions must be >=31", file=sys.stderr)
        return 3
    if args.warmups < 1 or args.bootstrap_samples < 100:
        print("BLOCKED: require warmups>=1 and bootstrap-samples>=100", file=sys.stderr)
        return 3
    if min(args.batch, args.seq, args.gen) < 1:
        print("BLOCKED: batch, seq, and gen must all be positive", file=sys.stderr)
        return 3
    model_problem = local_model_error(args.model)
    if model_problem:
        print(f"BLOCKED: {model_problem}; implicit downloads are disabled", file=sys.stderr)
        return 3

    actual_flags = (
        env_flag("TRTLLM_ENABLE_PDL"),
        env_flag("TORCHINDUCTOR_ENABLE_PDL"),
        env_flag("CTA_PDL_CEILING"),
    )
    if actual_flags != expected_flags(args.rung):
        print(
            f"BLOCKED: rung {args.rung} expects env flags {expected_flags(args.rung)}, "
            f"got {actual_flags}",
            file=sys.stderr,
        )
        return 3

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    patched = apply_parent_ceiling_patch() if args.rung == "ceiling" else []
    if args.rung == "ceiling" and not patched:
        print("BLOCKED: ceiling hook could not patch gdc_wait", file=sys.stderr)
        return 3

    try:
        result, runtime = run_vllm(args)
    except Exception as exc:  # noqa: BLE001 - diagnostic must preserve a clean failure code
        payload = {
            "schema": 2,
            "status": "error",
            "kind": "diagnostic",
            "args": vars(args),
            "error": f"{type(exc).__name__}: {exc}",
            "patched": patched,
        }
        write_json(args.json, payload)
        print(f"ERROR: {payload['error']}", file=sys.stderr)
        return 1

    pdl_trt, pdl_inductor, ceiling = actual_flags
    payload = {
        "schema": 2,
        "status": "diagnostic",
        "kind": "diagnostic",
        "args": vars(args),
        "runtime": runtime,
        "result": result,
        "patched": patched,
    }
    write_json(args.json, payload)
    print(
        f"SUMMARY tier=4 tag={args.tag} rung={args.rung} status=diagnostic kind=diagnostic "
        f"engine={args.engine} batch={args.batch} seq={args.seq} gen={args.gen} "
        f"repetitions={result['repetitions']} ci_method=bootstrap_95pct "
        f"median_s={result['median_s']:.6f} "
        f"median_s_ci_low={result['median_s_ci_low']:.6f} "
        f"median_s_ci_high={result['median_s_ci_high']:.6f} "
        f"tok_per_s={result['tok_per_s']:.6f} "
        f"tok_per_s_ci_low={result['tok_per_s_ci_low']:.6f} "
        f"tok_per_s_ci_high={result['tok_per_s_ci_high']:.6f} "
        f"tok_per_s_per_user={result['tok_per_s_per_user']:.6f} "
        f"tok_per_s_per_user_ci_low={result['tok_per_s_per_user_ci_low']:.6f} "
        f"tok_per_s_per_user_ci_high={result['tok_per_s_per_user_ci_high']:.6f} "
        f"pdl_trt={int(pdl_trt)} pdl_inductor={int(pdl_inductor)} "
        f"ceiling={int(ceiling)} graph_mode={runtime['graph_mode']} "
        f"output_digest={result['output_digest']} verified=0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
