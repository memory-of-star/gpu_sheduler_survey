#!/usr/bin/env python3
"""Tier-4 same-process Qwen/vLLM PDL triplet driver.

The model is loaded once.  The one persistent vLLM worker then compiles and
captures three variants of the same model in fresh, rung-specific caches:

* ``pdl_off``: Inductor PDL code generation disabled;
* ``pdl_grid``: Inductor PDL code generation enabled;
* ``ceiling``: PDL launch enabled, but ``gdc_wait`` compiled as a no-op.

The worker retains each compiled callable and each FULL-decode CUDA-graph set.
Before every invocation the driver switches those retained objects in place;
weights, scheduler, driver PID, and worker PID do not change.  Raw timings are
only candidates.  ``run_llm_sweep.sh`` admits them after PTX/cubin and Nsight
Systems validation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time
from typing import Any
import uuid

from model_identity import build_model_identity, sha256_file, verify_model_identity
from tier4_manifest import first_difference, immutable_contract

RUNGS = ("pdl_off", "pdl_grid", "ceiling")
ORDER_CYCLE = (
    ("pdl_off", "pdl_grid", "ceiling"),
    ("pdl_grid", "ceiling", "pdl_off"),
    ("ceiling", "pdl_off", "pdl_grid"),
)
FALSE_VALUES = {"", "0", "false", "False", "no", "No"}
SCHEMA = "tier4.triplet.raw.v2"
PDL_SCOPE = "inductor_generated_full_decode_graph_kernels"
PREFILL_MODE = "production_mixed_mode_non_headline"
WAIT_INSTRUCTION = re.compile(r"(?m)^\s*(?:@\S+\s+)?griddepcontrol\.wait\s*;")
LAUNCH_INSTRUCTION = re.compile(
    r"(?m)^\s*(?:@\S+\s+)?griddepcontrol\.launch_dependents\s*;"
)
PTX_ENTRY = re.compile(r"(?m)^\s*(?:\.visible\s+)?\.entry\s+([^\s(]+)\s*\(")
PTX_TARGET = re.compile(r"(?m)^\s*\.target\s+([^\n,]+)")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def model_fingerprint(model: Path) -> str:
    return str(build_model_identity(model)["model_fingerprint"])


def package_versions() -> dict[str, str]:
    values: dict[str, str] = {}
    for name in ("vllm", "torch", "triton", "transformers"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = "MISSING"
    return values


def _scan_variant_cache(cache: Path, rung: str) -> dict[str, Any]:
    """Fail immediately if a freshly built rung lacks its target semantics.

    This is deliberately duplicated as a small admission guard in the worker;
    the independent post-run validator performs the authoritative scan.  The
    guard prevents a cache alias (the cause of rejected smoke1) from reaching
    timing again.
    """
    ptx_files = sorted(path for path in cache.rglob("*.ptx") if path.is_file())
    cubin_files = sorted(path for path in cache.rglob("*.cubin") if path.is_file())
    waits = 0
    launches = 0
    paired = 0
    targets: set[str] = set()
    name_occurrences: dict[str, list[tuple[int, int]]] = {}
    for path in ptx_files:
        text = path.read_text(errors="replace")
        file_waits = len(WAIT_INSTRUCTION.findall(text))
        file_launches = len(LAUNCH_INSTRUCTION.findall(text))
        entries = set(PTX_ENTRY.findall(text))
        for name in entries | {path.stem}:
            if name not in {"", "triton_", "kernel"}:
                name_occurrences.setdefault(name, []).append(
                    (file_waits, file_launches)
                )
        waits += file_waits
        launches += file_launches
        target = PTX_TARGET.search(text)
        if target:
            targets.add(target.group(1).strip())
        if path.with_suffix(".cubin").is_file():
            paired += 1
    wait_entries = {
        name
        for name, occurrences in name_occurrences.items()
        if all(wait_count > 0 for wait_count, _launch_count in occurrences)
    }
    launch_entries = {
        name
        for name, occurrences in name_occurrences.items()
        if all(launch_count > 0 for _wait_count, launch_count in occurrences)
    }
    probe = {
        "ptx_files": len(ptx_files),
        "cubin_files": len(cubin_files),
        "paired_cubin_files": paired,
        "wait_count": waits,
        "launch_count": launches,
        "wait_entries": sorted(wait_entries),
        "launch_entries": sorted(launch_entries),
        "ptx_targets": sorted(targets),
    }
    if not ptx_files or not cubin_files or paired <= 0:
        raise RuntimeError(f"{rung} fresh cache has no PTX/cubin pair: {probe}")
    if not any(target.startswith("sm_100") for target in targets):
        raise RuntimeError(f"{rung} cache has no B200 sm_100 target: {probe}")
    if rung == "pdl_off" and (waits != 0 or launches != 0):
        raise RuntimeError(f"pdl_off contains GDC instructions: {probe}")
    if rung == "pdl_grid" and (waits <= 0 or launches <= 0):
        raise RuntimeError(f"pdl_grid lacks wait/launch instructions: {probe}")
    if rung == "ceiling" and (waits != 0 or launches <= 0):
        raise RuntimeError(f"ceiling is not wait-free/launch-bearing: {probe}")
    relevant = launch_entries if rung == "ceiling" else wait_entries
    descriptive = {name for name in relevant if name not in {"triton_", "kernel"}}
    if rung != "pdl_off" and len(descriptive) < 2:
        raise RuntimeError(
            f"{rung} has fewer than two collision-safe descriptive "
            f"GDC-bearing PTX entries: {probe}"
        )
    return probe


def parse_point(text: str) -> dict[str, Any]:
    """Parse TAG:BATCH:SEQ:GEN:SCOPE."""
    fields = text.split(":")
    if len(fields) != 5:
        raise argparse.ArgumentTypeError(
            "point must be TAG:BATCH:SEQ:GEN:SCOPE (scope decode or prefill)"
        )
    tag, batch_text, seq_text, gen_text, scope = fields
    if not tag or scope not in {"decode", "prefill"}:
        raise argparse.ArgumentTypeError("point tag is empty or scope is not decode/prefill")
    try:
        batch, seq, gen = map(int, (batch_text, seq_text, gen_text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("point batch/seq/gen must be integers") from exc
    if min(batch, seq, gen) <= 0:
        raise argparse.ArgumentTypeError("point batch/seq/gen must be positive")
    if scope == "decode" and gen < 2:
        raise argparse.ArgumentTypeError("decode proof needs gen>=2 to execute a FULL decode graph")
    return {"tag": tag, "batch": batch, "seq": seq, "gen": gen, "scope": scope}


def bootstrap_median_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    if not values:
        raise RuntimeError("cannot bootstrap an empty sample")
    rng = random.Random(seed)
    n = len(values)
    medians = sorted(
        statistics.median(values[rng.randrange(n)] for _ in range(n))
        for _ in range(samples)
    )
    return (
        medians[int(0.025 * (samples - 1))],
        medians[int(0.975 * (samples - 1))],
    )


def _set_wait_hook(ceiling: bool) -> bool:
    """Set the worker's Triton wait definition before compilation/switching."""
    import triton.language.extra.cuda as tlcuda  # type: ignore

    original_name = "_cta_pdl_original_gdc_wait"
    if not hasattr(tlcuda, original_name):
        setattr(tlcuda, original_name, tlcuda.gdc_wait)
    original = getattr(tlcuda, original_name)
    if ceiling:
        def no_wait(*_args: Any, **_kwargs: Any) -> None:
            return None

        no_wait._cta_pdl_ceiling_noop = True  # type: ignore[attr-defined]
        tlcuda.gdc_wait = no_wait
    else:
        tlcuda.gdc_wait = original
    return bool(getattr(tlcuda.gdc_wait, "_cta_pdl_ceiling_noop", False))


def _compile_state(model: Any) -> dict[str, Any]:
    names = (
        "_compiled_callable",
        "_compiled_bytecode",
        "aot_compiled_fn",
        "was_aot_compile_fn_loaded_from_disk",
        "compiled",
        "first_compile",
    )
    return {name: getattr(model, name) for name in names if hasattr(model, name)}


def _restore_compile_state(model: Any, state: dict[str, Any]) -> None:
    for name, value in state.items():
        setattr(model, name, value)


def _compiled_modules(model: Any) -> list[tuple[str, Any]]:
    """Locate the actual decorated language backbone(s), not the MM shell.

    Qwen3.5's runner model is ``Qwen3_5ForConditionalGeneration`` while the
    ``@support_torch_compile`` object is its nested ``language_model.model``.
    Resetting only the shell was the cause of rejected smoke2.
    """
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    modules = [
        (name or "<root>", module)
        for name, module in model.named_modules()
        if isinstance(module, TorchCompileWithNoGuardsWrapper)
        and not bool(getattr(module, "do_not_compile", False))
    ]
    if not modules:
        raise RuntimeError(
            f"no active TorchCompileWithNoGuardsWrapper below {type(model).__name__}"
        )
    return modules


def _compiled_module_states(model: Any) -> dict[str, dict[str, Any]]:
    return {name: _compile_state(module) for name, module in _compiled_modules(model)}


def _restore_compiled_module_states(
    model: Any, states: dict[str, dict[str, Any]]
) -> None:
    current = dict(_compiled_modules(model))
    if set(current) != set(states):
        raise RuntimeError(
            f"compiled module topology changed: current={sorted(current)} "
            f"saved={sorted(states)}"
        )
    for name, state in states.items():
        _restore_compile_state(current[name], state)


def _active_variant_binding(worker: Any, rung: str) -> dict[str, Any]:
    """Fingerprint the exact callable and per-batch CUDA graphs now active.

    The identities are intentionally process-local: they prove that every
    adjacent invocation switched back to the retained objects built for this
    rung, rather than merely toggling environment flags around one callable.
    """
    runner = worker.model_runner
    modules: list[dict[str, Any]] = []
    for name, module in _compiled_modules(runner.get_model()):
        compiled_callable = getattr(module, "_compiled_callable", None)
        if compiled_callable is None:
            raise RuntimeError(f"active {rung} module {name} has no compiled callable")
        modules.append(
            {
                "name": name,
                "compiled_callable_id": id(compiled_callable),
                "aot_compiled_fn_id": (
                    id(module.aot_compiled_fn)
                    if getattr(module, "aot_compiled_fn", None) is not None
                    else None
                ),
                "compiled_bytecode_id": (
                    id(module._compiled_bytecode)
                    if getattr(module, "_compiled_bytecode", None) is not None
                    else None
                ),
            }
        )
    modules.sort(key=lambda item: item["name"])
    callable_fingerprint = sha256_bytes(
        json.dumps(modules, separators=(",", ":"), sort_keys=True).encode()
    )

    graphs = getattr(runner.model, "concrete_cudagraph_entries", None)
    if not isinstance(graphs, dict) or not graphs:
        raise RuntimeError(f"active {rung} variant has no CUDA graph dictionary")
    entries: list[dict[str, Any]] = []
    for descriptor, entry in graphs.items():
        descriptor_record = {
            "num_tokens": getattr(descriptor, "num_tokens", None),
            "num_reqs": getattr(descriptor, "num_reqs", None),
            "uniform": getattr(descriptor, "uniform", None),
            "has_lora": getattr(descriptor, "has_lora", None),
            "num_active_loras": getattr(descriptor, "num_active_loras", None),
        }
        cudagraph = getattr(entry, "cudagraph", None)
        if cudagraph is None:
            raise RuntimeError(
                f"active {rung} graph entry is not captured: {descriptor_record}"
            )
        entries.append(
            {
                "batch_descriptor": descriptor_record,
                "entry_id": id(entry),
                "cudagraph_id": id(cudagraph),
            }
        )
    entries.sort(
        key=lambda item: json.dumps(
            item["batch_descriptor"], separators=(",", ":"), sort_keys=True
        )
    )
    graph_dictionary_id = id(graphs)
    graph_dictionary_fingerprint = sha256_bytes(
        json.dumps(
            {
                "graph_dictionary_id": graph_dictionary_id,
                "entries": [
                    {
                        "batch_descriptor": item["batch_descriptor"],
                        "entry_id": item["entry_id"],
                    }
                    for item in entries
                ],
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    batch_graph_fingerprint = sha256_bytes(
        json.dumps(entries, separators=(",", ":"), sort_keys=True).encode()
    )
    activation_fingerprint = sha256_bytes(
        json.dumps(
            {
                "rung": rung,
                "compiled_callable_fingerprint": callable_fingerprint,
                "graph_dictionary_id": graph_dictionary_id,
                "graph_dictionary_fingerprint": graph_dictionary_fingerprint,
                "batch_graph_fingerprint": batch_graph_fingerprint,
                "full_graph_entries": len(entries),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    return {
        "schema": "tier4.active_variant.v1",
        "rung": rung,
        "compiled_callable_fingerprint": callable_fingerprint,
        "graph_dictionary_id": graph_dictionary_id,
        "graph_dictionary_fingerprint": graph_dictionary_fingerprint,
        "batch_graph_fingerprint": batch_graph_fingerprint,
        "full_graph_entries": len(entries),
        "activation_fingerprint": activation_fingerprint,
    }


def invocation_bindings(
    probes: list[dict[str, Any]], rung: str
) -> list[dict[str, Any]]:
    """Reduce switch probes to the immutable identity bound to one invocation."""
    bindings: list[dict[str, Any]] = []
    for probe in probes:
        active = probe.get("active_variant")
        if probe.get("rung") != rung or not isinstance(active, dict):
            raise RuntimeError(f"invalid active-variant switch probe for {rung}: {probe}")
        if active.get("rung") != rung:
            raise RuntimeError(f"active-variant rung mismatch for {rung}: {active}")
        bindings.append({"pid": int(probe["pid"]), **active})
    return sorted(bindings, key=lambda item: item["pid"])


def _worker_probe(worker: Any, rung: str, cache_root: str) -> dict[str, Any]:
    import torch._inductor.config as inductor_config
    import triton.language.extra.cuda as tlcuda  # type: ignore

    runner = worker.model_runner
    config = worker.vllm_config.compilation_config
    dispatcher = getattr(runner, "cudagraph_dispatcher", None)
    outer = runner.model
    entries = getattr(outer, "concrete_cudagraph_entries", {})
    variants = getattr(worker, "_cta_pdl_variants", {})
    compiled = _compiled_modules(raw := runner.get_model())
    attention_implementations = Counter()
    for layer in config.static_forward_context.values():
        implementation = getattr(layer, "impl", None)
        if implementation is not None:
            attention_implementations[
                f"{type(implementation).__module__}.{type(implementation).__name__}"
            ] += 1
    cache_config = worker.vllm_config.cache_config
    kv_cache_config = getattr(runner, "kv_cache_config", None)
    kv_groups: list[dict[str, Any]] = []
    if kv_cache_config is not None:
        for group in getattr(kv_cache_config, "kv_cache_groups", []):
            spec = getattr(group, "kv_cache_spec", None)
            page_size = getattr(spec, "page_size_bytes", None)
            kv_groups.append(
                {
                    "spec_type": type(spec).__name__,
                    "layer_count": len(getattr(group, "layer_names", [])),
                    "block_size": getattr(spec, "block_size", None),
                    "page_size_bytes": page_size if isinstance(page_size, int) else None,
                }
            )
    connector = getattr(worker, "kv_connector", None)
    if connector is None:
        connector = getattr(runner, "kv_connector", None)
    return {
        "pid": os.getpid(),
        "rung": rung,
        "pdl_env": os.environ.get("TORCHINDUCTOR_ENABLE_PDL") == "1",
        "pdl_inductor_config": bool(inductor_config.triton.enable_pdl),
        "ceiling_hook": bool(
            getattr(tlcuda.gdc_wait, "_cta_pdl_ceiling_noop", False)
        ),
        "configured_graph_mode": str(config.cudagraph_mode),
        "compile_ranges_endpoints": list(config.compile_ranges_endpoints or []),
        "dispatcher_mode": str(getattr(dispatcher, "cudagraph_mode", "UNKNOWN")),
        "outer_wrapper_type": type(outer).__name__,
        "raw_model_type": type(raw).__name__,
        "compiled_modules": [
            {
                "name": name,
                "type": type(module).__name__,
                "compiled": bool(getattr(module, "compiled", False)),
                "compiled_callable_id": id(getattr(module, "_compiled_callable", None)),
                "aot_compiled_fn_id": (
                    id(module.aot_compiled_fn)
                    if getattr(module, "aot_compiled_fn", None) is not None
                    else None
                ),
                "compiled_bytecode_id": (
                    id(module._compiled_bytecode)
                    if getattr(module, "_compiled_bytecode", None) is not None
                    else None
                ),
            }
            for name, module in compiled
        ],
        "active_variant": _active_variant_binding(worker, rung),
        "full_graph_entries": len(entries),
        "retained_variants": sorted(variants),
        "cache_root": str(Path(cache_root).resolve()),
        "cache_scan": variants.get(rung, {}).get("cache_scan"),
        "trtllm_enable_pdl": os.environ.get("TRTLLM_ENABLE_PDL") == "1",
        "attention_implementation_counts": dict(sorted(attention_implementations.items())),
        "kv_cache": {
            "gpu_num_blocks": getattr(cache_config, "num_gpu_blocks", None),
            "cpu_num_blocks": getattr(cache_config, "num_cpu_blocks", None),
            "block_size": cache_config.block_size,
            "kv_cache_memory_bytes": cache_config.kv_cache_memory_bytes,
            "kv_offloading_size_gib": cache_config.kv_offloading_size,
            "kv_offloading_backend": cache_config.kv_offloading_backend,
            "mamba_cache_mode": cache_config.mamba_cache_mode,
            "connector_type": type(connector).__name__ if connector is not None else None,
            "groups": kv_groups,
        },
    }


def worker_build_variant(
    worker: Any,
    rung: str,
    cache_root: str,
    triplet_id: str,
) -> dict[str, Any]:
    """Build/capture one variant inside the persistent vLLM worker."""
    import torch
    import torch._inductor.config as inductor_config
    import vllm.envs as vllm_envs
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.compilation.wrapper import reset_compile_wrapper
    from vllm.config import CUDAGraphMode, set_current_vllm_config

    if rung not in RUNGS:
        raise RuntimeError(f"unknown Tier-4 rung: {rung}")
    runner = worker.model_runner
    outer = runner.model
    if not isinstance(outer, CUDAGraphWrapper):
        raise RuntimeError(
            f"Tier-4 requires the FULL CUDAGraphWrapper, got {type(outer).__name__}"
        )
    raw = runner.get_model()
    variants = getattr(worker, "_cta_pdl_variants", None)
    if variants is None:
        variants = {}
        worker._cta_pdl_variants = variants
        worker._cta_pdl_triplet_id = triplet_id
    if getattr(worker, "_cta_pdl_triplet_id", None) != triplet_id:
        raise RuntimeError("worker triplet identity changed")
    if rung in variants:
        raise RuntimeError(f"variant already exists: {rung}")

    cache_path = Path(cache_root).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = str(cache_path / "vllm")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_path / "torchinductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache_path / "triton")
    os.environ["TORCHINDUCTOR_ENABLE_PDL"] = "0" if rung == "pdl_off" else "1"
    os.environ["CTA_PDL_CEILING"] = "1" if rung == "ceiling" else "0"
    # vLLM freezes all env-backed properties after service initialization.
    # Without this reset, later rungs silently keep using pdl_off's
    # VLLM_CACHE_ROOT and can load its AOT callable despite a new os.environ.
    vllm_envs.disable_envs_cache()
    inductor_config.triton.enable_pdl = rung != "pdl_off"
    ceiling_hook = _set_wait_hook(rung == "ceiling")
    if ceiling_hook != (rung == "ceiling"):
        raise RuntimeError(f"could not establish wait hook for {rung}")

    # Compile *every* admitted rung after engine initialization.  The engine's
    # mandatory first compile is an unmeasured bootstrap and lives in a fourth,
    # non-evidence cache.  This keeps all three admitted callables symmetric.
    #
    # On a second Qwen3.5 Dynamo trace, the GDN query-start buffer contributes
    # a legitimate shape of max_num_batched_tokens + 1.  vLLM 0.23's computed
    # terminal compile range omits that sentinel.  Expand the one terminal
    # endpoint for all three admitted variants; actual scheduled tokens remain
    # bounded by runner.max_num_tokens.
    endpoints = list(worker.vllm_config.compilation_config.compile_ranges_endpoints or [])
    terminal = int(runner.max_num_tokens)
    expanded = sorted({value for value in endpoints if value < terminal} | {terminal + 1})
    worker.vllm_config.compilation_config.compile_ranges_endpoints = expanded

    # Retain prior graph objects rather than clearing their dictionaries in
    # place.  The captured graphs are switched back in for adjacent trials.
    outer.concrete_cudagraph_entries = {}
    torch.compiler.reset()
    # Inductor registers process-global memoized code/cache lookups.  A fresh
    # environment path alone is insufficient for three in-process lowerings.
    from torch._inductor import utils as inductor_utils

    inductor_utils.clear_caches()
    with set_current_vllm_config(worker.vllm_config):
        for _name, module in _compiled_modules(raw):
            reset_compile_wrapper(module)
            if bool(getattr(module, "compiled", True)):
                raise RuntimeError(f"compile reset did not clear {_name}.compiled")
            if getattr(module, "aot_compiled_fn", None) is not None:
                raise RuntimeError(f"compile reset retained {_name}.aot_compiled_fn")
        # Reproduce vLLM's initial compile ordering: a non-graph, non-uniform
        # prefill trace creates the general callable used by both prefill and
        # decode.  Letting FULL capture trigger the first trace would
        # specialize the callable to uniform decode and break the mandatory
        # prefill phase of every real generate() request.
        runner._dummy_run(
            runner.max_num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            uniform_decode=False,
            is_profile=True,
            skip_eplb=True,
        )
    runner.capture_model()
    for _name, module in _compiled_modules(raw):
        if not bool(getattr(module, "compiled", False)):
            raise RuntimeError(f"fresh lowering did not compile {_name}")

    state = {
        "compiled_modules": _compiled_module_states(raw),
        "graphs": outer.concrete_cudagraph_entries,
        "cache_root": str(cache_path),
        "cache_scan": _scan_variant_cache(cache_path, rung),
    }
    if not state["graphs"]:
        raise RuntimeError(f"{rung} captured no FULL CUDA graphs")
    variants[rung] = state
    state["active_variant"] = _active_variant_binding(worker, rung)
    return _worker_probe(worker, rung, str(cache_path))


def worker_switch_variant(worker: Any, rung: str) -> dict[str, Any]:
    """Switch retained compiled code and CUDA graphs without reloading weights."""
    import torch._inductor.config as inductor_config
    import vllm.envs as vllm_envs

    variants = getattr(worker, "_cta_pdl_variants", {})
    if rung not in variants:
        raise RuntimeError(f"variant is not built: {rung}")
    runner = worker.model_runner
    raw = runner.get_model()
    state = variants[rung]
    _restore_compiled_module_states(raw, state["compiled_modules"])
    runner.model.concrete_cudagraph_entries = state["graphs"]
    cache_path = Path(state["cache_root"])
    os.environ["VLLM_CACHE_ROOT"] = str(cache_path / "vllm")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_path / "torchinductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache_path / "triton")
    os.environ["TORCHINDUCTOR_ENABLE_PDL"] = "0" if rung == "pdl_off" else "1"
    os.environ["TRTLLM_ENABLE_PDL"] = "0"
    os.environ["CTA_PDL_CEILING"] = "1" if rung == "ceiling" else "0"
    vllm_envs.disable_envs_cache()
    inductor_config.triton.enable_pdl = rung != "pdl_off"
    _set_wait_hook(rung == "ceiling")
    active = _active_variant_binding(worker, rung)
    if active != state.get("active_variant"):
        raise RuntimeError(
            f"active callable/CUDA graph identity differs for {rung}: "
            f"expected={state.get('active_variant')} observed={active}"
        )
    probe = _worker_probe(worker, rung, state["cache_root"])
    expected = rung != "pdl_off"
    if probe["pdl_inductor_config"] is not expected:
        raise RuntimeError(f"worker PDL switch did not take effect: {probe}")
    if probe["ceiling_hook"] is not (rung == "ceiling"):
        raise RuntimeError(f"worker ceiling switch did not take effect: {probe}")
    return probe


def prompt_tokens(point: dict[str, Any], epoch: int) -> list[list[int]]:
    return [
        [
            1000 + ((position * 17 + request * 31 + epoch * 7) % 5000)
            for position in range(point["seq"])
        ]
        for request in range(point["batch"])
    ]


def canonical_output(outputs: list[Any], point: dict[str, Any]) -> list[dict[str, Any]]:
    if len(outputs) != point["batch"]:
        raise RuntimeError(
            f"vLLM returned {len(outputs)} requests, expected {point['batch']}"
        )
    records: list[dict[str, Any]] = []
    for request_index, request in enumerate(outputs):
        choices = getattr(request, "outputs", None)
        if not choices:
            raise RuntimeError("vLLM request has no generated output")
        choice = choices[0]
        ids = [int(token) for token in choice.token_ids]
        if len(ids) != point["gen"]:
            raise RuntimeError(
                f"generated {len(ids)} tokens, expected {point['gen']}"
            )
        cumulative = getattr(choice, "cumulative_logprob", None)
        records.append(
            {
                "request_index": request_index,
                "token_ids": ids,
                "cumulative_logprob_hex": (
                    float(cumulative).hex() if cumulative is not None else None
                ),
            }
        )
    return records


def output_digest(records: list[dict[str, Any]]) -> str:
    return sha256_bytes(json.dumps(records, separators=(",", ":"), sort_keys=True).encode())


def token_digest(records: list[dict[str, Any]]) -> str:
    values = [record["token_ids"] for record in records]
    return sha256_bytes(json.dumps(values, separators=(",", ":")).encode())


def nonfinite_logprob(records: list[dict[str, Any]]) -> bool:
    nonfinite = False
    for record in records:
        value = record.get("cumulative_logprob_hex")
        if not isinstance(value, str):
            raise RuntimeError("generated output lacks a cumulative logprob hex string")
        try:
            parsed = float.fromhex(value)
        except ValueError as exc:
            raise RuntimeError(f"unparseable cumulative logprob hex: {value!r}") from exc
        nonfinite = nonfinite or not math.isfinite(parsed)
    return nonfinite


def prompt_digest(point: dict[str, Any], epoch: int) -> str:
    return sha256_bytes(
        json.dumps(prompt_tokens(point, epoch), separators=(",", ":")).encode()
    )


def start_profiler_capture() -> None:
    import torch

    status = torch.cuda.cudart().cudaProfilerStart()
    if status != 0:
        raise RuntimeError(f"cudaProfilerStart failed with status {status}")


def stop_profiler_capture() -> None:
    import torch

    torch.cuda.synchronize()
    status = torch.cuda.cudart().cudaProfilerStop()
    if status != 0:
        raise RuntimeError(f"cudaProfilerStop failed with status {status}")


def run_generation(llm: Any, sampling: Any, point: dict[str, Any], epoch: int) -> tuple[float, list[dict[str, Any]]]:
    from vllm.inputs import TokensPrompt

    tokens = prompt_tokens(point, epoch)
    prompts = [TokensPrompt(prompt_token_ids=value) for value in tokens]
    start = time.perf_counter()
    outputs = llm.generate(prompts=prompts, sampling_params=sampling, use_tqdm=False)
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        raise RuntimeError("non-positive generation latency")
    return elapsed, canonical_output(outputs, point)


def run(args: argparse.Namespace) -> int:
    results = args.results.resolve()
    model = args.model.resolve()
    if results.exists() and any(results.iterdir()):
        raise RuntimeError(f"results directory is not empty: {results}")
    results.mkdir(parents=True, exist_ok=True)
    bootstrap_cache = (results / "bootstrap_cache").resolve()
    bootstrap_cache.mkdir(parents=True, exist_ok=False)
    for rung in RUNGS:
        (results / "evidence" / rung / "cache").mkdir(parents=True, exist_ok=False)

    if not args.allow_short and args.model_identity is None:
        raise RuntimeError("formal Tier-4 runs require --model-identity")
    if not args.allow_short and args.formal_root_manifest is None:
        raise RuntimeError("formal Tier-4 runs require --formal-root-manifest")
    if not args.allow_short and args.kv_offloading_backend != "native":
        raise RuntimeError("formal Tier-4 runs require native KV offloading backend")
    if args.model_identity is not None:
        identity_path = args.model_identity.resolve()
        try:
            identity = json.loads(identity_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read declared model identity: {exc}") from exc
        identity_errors = verify_model_identity(identity, model)
        if identity_errors:
            raise RuntimeError("; ".join(identity_errors))
        identity_manifest_hash = sha256_file(identity_path)
    else:
        identity = build_model_identity(model)
        identity_manifest_hash = None
    identity_snapshot = results / "model_identity_snapshot.json"
    atomic_json(identity_snapshot, identity)
    identity_snapshot_hash = sha256_file(identity_snapshot)
    if identity_manifest_hash is None:
        identity_manifest_hash = identity_snapshot_hash
    elif identity_manifest_hash != identity_snapshot_hash:
        raise RuntimeError("declared model identity changed while snapshotting")
    fingerprint = str(identity["model_fingerprint"])
    formal_root_manifest_hash: str | None = None
    if args.formal_root_manifest is not None:
        root_manifest_path = args.formal_root_manifest.resolve()
        try:
            root_manifest = json.loads(root_manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read formal root manifest: {exc}") from exc
        if not isinstance(root_manifest, dict):
            raise RuntimeError("formal root manifest is not an object")
        if root_manifest.get("model") != str(model):
            raise RuntimeError("formal root manifest model path mismatch")
        if root_manifest.get("model_fingerprint") != fingerprint:
            raise RuntimeError("formal root manifest model fingerprint mismatch")
        if root_manifest.get("model_identity_manifest_sha256") != identity_manifest_hash:
            raise RuntimeError("formal root manifest model identity hash mismatch")
        if not args.allow_short and root_manifest_path.parent != results.parent.parent:
            raise RuntimeError("formal root manifest is not in the cohort result root")
        kv_strategy = root_manifest.get("kv_strategy")
        if not isinstance(kv_strategy, dict):
            raise RuntimeError("formal root manifest KV strategy is absent")
        expected_root_manifest = immutable_contract(
            root_manifest_path.parent,
            model,
            identity,
            identity_manifest_hash,
            args.kv_offloading_size,
        )
        difference = first_difference(expected_root_manifest, root_manifest)
        if difference:
            raise RuntimeError(f"formal immutable root contract differs: {difference}")
        formal_root_manifest_hash = sha256_file(root_manifest_path)
        launch_snapshot = results / "launch_manifest_snapshot.json"
        atomic_json(launch_snapshot, root_manifest)
        if sha256_file(launch_snapshot) != formal_root_manifest_hash:
            raise RuntimeError("formal root manifest changed while snapshotting")
    packages = package_versions()
    if "MISSING" in packages.values():
        raise RuntimeError(f"required package is missing: {packages}")
    if not args.allow_short and args.repeats < 31:
        raise RuntimeError("formal Tier-4 runs require repeats>=31")
    if not args.allow_short and args.warmups < 3:
        raise RuntimeError("formal Tier-4 runs require warmups>=3")
    if not args.allow_short and args.bootstrap_samples < 2000:
        raise RuntimeError("formal Tier-4 runs require bootstrap-samples>=2000")
    if not args.allow_short and not args.cohort_id:
        raise RuntimeError("formal Tier-4 runs require an explicit --cohort-id")
    if args.warmups < 1 or args.bootstrap_samples < 100:
        raise RuntimeError("require warmups>=1 and bootstrap-samples>=100")

    measured_decode = [point for point in args.point if point["scope"] == "decode"]
    proof_point = args.proof_point or (measured_decode[0] if measured_decode else None)
    if proof_point is None or proof_point["scope"] != "decode":
        raise RuntimeError(
            "a decode --proof-point is required when the measured cohort is prefill-only"
        )
    all_engine_points = [*args.point, proof_point]
    max_batch = max(point["batch"] for point in all_engine_points)
    max_model_len = max(point["seq"] + point["gen"] for point in all_engine_points)
    capture_sizes = sorted(
        {point["batch"] for point in measured_decode} | {proof_point["batch"]}
    )

    triplet_id = uuid.uuid4().hex
    driver_pid = os.getpid()
    # vLLM necessarily initializes/compiles while constructing LLM.  Keep that
    # disposable bootstrap out of every admitted rung's fresh cache.
    off_cache = bootstrap_cache
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            # Each rung must really lower its own target code.  Disk cache
            # lookup is disabled while the generated PTX/cubin are still
            # retained under the isolated VLLM/Inductor cache roots.
            "VLLM_DISABLE_COMPILE_CACHE": "1",
            "VLLM_CACHE_ROOT": str(off_cache / "vllm"),
            "TORCHINDUCTOR_CACHE_DIR": str(off_cache / "torchinductor"),
            "TRITON_CACHE_DIR": str(off_cache / "triton"),
            "TORCHINDUCTOR_ENABLE_PDL": "0",
            "TRTLLM_ENABLE_PDL": "0",
            "CTA_PDL_CEILING": "0",
        }
    )

    # Import only after the initial off-rung environment and cache are fixed.
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model),
        max_model_len=max_model_len,
        max_num_seqs=max_batch,
        max_num_batched_tokens=max(args.max_num_batched_tokens, max_batch),
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=False,
        enable_prefix_caching=False,
        dtype="bfloat16",
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": capture_sizes,
        },
        kv_offloading_size=args.kv_offloading_size,
        kv_offloading_backend=args.kv_offloading_backend,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    resolved_mode = str(llm.llm_engine.vllm_config.compilation_config.cudagraph_mode)
    if resolved_mode != "FULL_DECODE_ONLY":
        raise RuntimeError(
            f"vLLM resolved cudagraph_mode={resolved_mode}, expected FULL_DECODE_ONLY"
        )
    import torch

    properties = torch.cuda.get_device_properties(0)
    device = {
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "sm_count": properties.multi_processor_count,
    }

    build_probes: dict[str, list[dict[str, Any]]] = {}
    for rung in RUNGS:
        cache = (results / "evidence" / rung / "cache").resolve()
        probes = llm.collective_rpc(
            worker_build_variant,
            timeout=args.variant_timeout,
            args=(rung, str(cache), triplet_id),
        )
        if not probes:
            raise RuntimeError(f"no worker probe while building {rung}")
        build_probes[rung] = probes

    cohorts = {
        tuple(sorted(int(probe["pid"]) for probe in probes))
        for probes in build_probes.values()
    }
    if len(cohorts) != 1:
        raise RuntimeError(f"worker cohort changed across variants: {cohorts}")

    sampling_by_gen = {
        gen: SamplingParams(
            temperature=0.0,
            max_tokens=gen,
            ignore_eos=True,
            logprobs=1,
        )
        for gen in {point["gen"] for point in all_engine_points}
    }

    forensic_records: list[dict[str, Any]] = []

    def forensic_invoke(
        label: str, rung: str, point: dict[str, Any], epoch: int
    ) -> None:
        probes = llm.collective_rpc(
            worker_switch_variant, timeout=120, args=(rung,)
        )
        elapsed, records = run_generation(
            llm, sampling_by_gen[point["gen"]], point, epoch
        )
        torch.cuda.synchronize()
        forensic_records.append(
            {
                "label": label,
                "rung": rung,
                "epoch": epoch,
                "prompt_digest": prompt_digest(point, epoch),
                "output": records,
                "output_digest": output_digest(records),
                "token_digest": token_digest(records),
                "nonfinite_logprob": nonfinite_logprob(records),
                "elapsed_s_diagnostic": elapsed,
                "worker_pids": sorted(int(value["pid"]) for value in probes),
                "switch_bindings": invocation_bindings(probes, rung),
            }
        )

    forensic_point = args.point[0]
    forensic_epochs = args.forensic_epoch or [100_000, 100_001, 100_002]
    if args.forensic_only and not args.forensic_proof_first:
        for epoch in forensic_epochs:
            for suffix, rung in (
                ("pre_proof_off_a", "pdl_off"),
                ("pre_proof_off_b", "pdl_off"),
                ("pre_proof_grid_a", "pdl_grid"),
                ("pre_proof_grid_b", "pdl_grid"),
            ):
                forensic_invoke(
                    f"epoch_{epoch}_{suffix}", rung, forensic_point, epoch
                )

    # Profile exactly one decode request per rung.  The enclosing nsys command
    # stops collection at cudaProfilerStop; formal samples below are untraced.
    proof_records: list[dict[str, Any]] = []
    start_profiler_capture()
    for index, rung in enumerate(RUNGS):
        probes = llm.collective_rpc(worker_switch_variant, timeout=120, args=(rung,))
        import torch

        label = f"TIER4_PROOF|rung={rung}|triplet={triplet_id}"
        torch.cuda.nvtx.range_push(label)
        try:
            elapsed, records = run_generation(
                llm, sampling_by_gen[proof_point["gen"]], proof_point, 10_000 + index
            )
        finally:
            torch.cuda.nvtx.range_pop()
        proof_nonfinite = nonfinite_logprob(records)
        if rung != "ceiling" and proof_nonfinite:
            raise RuntimeError(f"{rung} proof produced a non-finite logprob")
        proof_records.append(
            {
                "rung": rung,
                "elapsed_s_diagnostic": elapsed,
                "output_digest": output_digest(records),
                "nonfinite_logprob": proof_nonfinite,
                "prompt_digest": prompt_digest(proof_point, 10_000 + index),
                "worker_probes": probes,
                "switch_bindings": invocation_bindings(probes, rung),
                "ceiling_verified": False if rung == "ceiling" else None,
                "ceiling_unsafe": True if rung == "ceiling" else None,
            }
        )
    stop_profiler_capture()

    if args.forensic_only:
        if args.forensic_proof_first:
            for epoch in forensic_epochs:
                for repeat_label in ("a", "b"):
                    for rung in RUNGS:
                        forensic_invoke(
                            f"epoch_{epoch}_formal_{repeat_label}_{rung}",
                            rung,
                            forensic_point,
                            epoch,
                        )
        else:
            for epoch in forensic_epochs:
                for suffix, rung in (
                    ("post_proof_off_a", "pdl_off"),
                    ("post_proof_off_b", "pdl_off"),
                    ("post_proof_grid_a", "pdl_grid"),
                    ("post_proof_grid_b", "pdl_grid"),
                ):
                    forensic_invoke(
                        f"epoch_{epoch}_{suffix}", rung, forensic_point, epoch
                    )
        indexed = {record["label"]: record for record in forensic_records}

        def compare(left: str, right: str) -> dict[str, bool]:
            return {
                "token_match": (
                    indexed[left]["token_digest"] == indexed[right]["token_digest"]
                ),
                "full_output_match": (
                    indexed[left]["output_digest"]
                    == indexed[right]["output_digest"]
                ),
            }

        comparisons: dict[str, dict[str, dict[str, bool]]] = {}
        for epoch in forensic_epochs:
            prefix = f"epoch_{epoch}_"
            if args.forensic_proof_first:
                comparisons[str(epoch)] = {
                    "formal_a_off_grid": compare(
                        prefix + "formal_a_pdl_off",
                        prefix + "formal_a_pdl_grid",
                    ),
                    "formal_b_off_grid": compare(
                        prefix + "formal_b_pdl_off",
                        prefix + "formal_b_pdl_grid",
                    ),
                    "off_across_repeat": compare(
                        prefix + "formal_a_pdl_off",
                        prefix + "formal_b_pdl_off",
                    ),
                    "grid_across_repeat": compare(
                        prefix + "formal_a_pdl_grid",
                        prefix + "formal_b_pdl_grid",
                    ),
                }
            else:
                comparisons[str(epoch)] = {
                    "pre_off_self": compare(
                        prefix + "pre_proof_off_a", prefix + "pre_proof_off_b"
                    ),
                    "pre_grid_self": compare(
                        prefix + "pre_proof_grid_a", prefix + "pre_proof_grid_b"
                    ),
                    "pre_off_grid": compare(
                        prefix + "pre_proof_off_a", prefix + "pre_proof_grid_a"
                    ),
                    "post_off_self": compare(
                        prefix + "post_proof_off_a", prefix + "post_proof_off_b"
                    ),
                    "post_grid_self": compare(
                        prefix + "post_proof_grid_a", prefix + "post_proof_grid_b"
                    ),
                    "post_off_grid": compare(
                        prefix + "post_proof_off_a", prefix + "post_proof_grid_a"
                    ),
                    "off_across_ceiling_proof": compare(
                        prefix + "pre_proof_off_a", prefix + "post_proof_off_a"
                    ),
                    "grid_across_ceiling_proof": compare(
                        prefix + "pre_proof_grid_a", prefix + "post_proof_grid_a"
                    ),
                }
        forensic = {
            "schema": "tier4.output.forensic.v2",
            "status": "diagnostic",
            "admissible": False,
            "triplet_id": triplet_id,
            "model_fingerprint": fingerprint,
            "point": forensic_point,
            "engine_points": args.point,
            "proof_point": proof_point,
            "forensic_epochs": forensic_epochs,
            "proof_first": args.forensic_proof_first,
            "worker_cohort": list(next(iter(cohorts))),
            "records": forensic_records,
            "proof_records": proof_records,
            "comparisons": comparisons,
        }
        atomic_json(results / "output_forensic.json", forensic)
        print(
            f"TIER4_FORENSIC status=ok admissible=0 triplet_id={triplet_id} "
            f"records={len(forensic_records)}"
        )
        return 0

    samples: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    warmup_records: list[dict[str, Any]] = []
    invocation = 0
    for point_index, point in enumerate(args.point):
        sampling = sampling_by_gen[point["gen"]]
        # Warmups preserve the same adjacent off/grid/ceiling order as samples.
        for warmup in range(args.warmups):
            epoch = 100_000 + point_index * 10_000 + warmup
            warmup_outputs: dict[str, list[dict[str, Any]]] = {}
            warmup_bindings: dict[str, list[dict[str, Any]]] = {}
            warmup_order = ORDER_CYCLE[warmup % len(ORDER_CYCLE)]
            for rung in warmup_order:
                probes = llm.collective_rpc(
                    worker_switch_variant, timeout=120, args=(rung,)
                )
                _, records = run_generation(llm, sampling, point, epoch)
                if rung != "ceiling" and nonfinite_logprob(records):
                    raise RuntimeError(
                        f"{rung} warmup produced a non-finite logprob for "
                        f"{point['tag']}"
                    )
                warmup_outputs[rung] = records
                warmup_bindings[rung] = invocation_bindings(probes, rung)
                invocation += 1
            if output_digest(warmup_outputs["pdl_off"]) != output_digest(
                warmup_outputs["pdl_grid"]
            ):
                mismatch = {
                    "schema": "tier4.output.mismatch.v2",
                    "status": "rejected",
                    "admissible": False,
                    "phase": "warmup",
                    "triplet_id": triplet_id,
                    "point": point,
                    "point_index": point_index,
                    "warmup": warmup,
                    "epoch": epoch,
                    "order_pattern": ">".join(warmup_order),
                    "off": {
                        "output": warmup_outputs["pdl_off"],
                        "output_digest": output_digest(
                            warmup_outputs["pdl_off"]
                        ),
                        "token_digest": token_digest(warmup_outputs["pdl_off"]),
                        "nonfinite_logprob": nonfinite_logprob(
                            warmup_outputs["pdl_off"]
                        ),
                    },
                    "grid": {
                        "output": warmup_outputs["pdl_grid"],
                        "output_digest": output_digest(
                            warmup_outputs["pdl_grid"]
                        ),
                        "token_digest": token_digest(warmup_outputs["pdl_grid"]),
                        "nonfinite_logprob": nonfinite_logprob(
                            warmup_outputs["pdl_grid"]
                        ),
                    },
                }
                mismatch["token_match"] = (
                    mismatch["off"]["token_digest"]
                    == mismatch["grid"]["token_digest"]
                )
                atomic_json(results / "warmup_output_mismatch.json", mismatch)
                raise RuntimeError(
                    f"off/grid warmup token/logprob outputs differ for {point['tag']}"
                )
            warmup_records.append(
                {
                    "point_index": point_index,
                    "tag": point["tag"],
                    "warmup": warmup,
                    "epoch": epoch,
                    "order_pattern": ">".join(warmup_order),
                    "rung_sequence": list(warmup_order),
                    "off_grid_full_output_match": True,
                    "switch_bindings": warmup_bindings,
                }
            )

        for repeat in range(args.repeats):
            epoch = 1_000_000 + point_index * 100_000 + repeat
            repeat_records: dict[str, list[dict[str, Any]]] = {}
            timing_order = ORDER_CYCLE[repeat % len(ORDER_CYCLE)]
            order_pattern = ">".join(timing_order)
            for order_index, rung in enumerate(timing_order):
                rung_index = RUNGS.index(rung)
                probes = llm.collective_rpc(
                    worker_switch_variant, timeout=120, args=(rung,)
                )
                elapsed, records = run_generation(llm, sampling, point, epoch)
                output_nonfinite = nonfinite_logprob(records)
                if rung != "ceiling" and output_nonfinite:
                    raise RuntimeError(
                        f"{rung} produced a non-finite logprob "
                        f"tag={point['tag']} repeat={repeat}"
                    )
                invocation += 1
                repeat_records[rung] = records
                samples.append(
                    {
                        "point_index": point_index,
                        "tag": point["tag"],
                        "batch": point["batch"],
                        "seq": point["seq"],
                        "gen": point["gen"],
                        "scope": point["scope"],
                        "repeat": repeat,
                        "rung_index": rung_index,
                        "rung": rung,
                        "order_index": order_index,
                        "order_pattern": order_pattern,
                        "epoch": epoch,
                        "prompt_digest": prompt_digest(point, epoch),
                        "invocation": invocation,
                        "elapsed_s": elapsed,
                        "token_count": point["batch"] * point["gen"],
                        "output": records,
                        "output_digest": output_digest(records),
                        "token_digest": token_digest(records),
                        "nonfinite_logprob": output_nonfinite,
                        "worker_pids": sorted(int(value["pid"]) for value in probes),
                        "switch_bindings": invocation_bindings(probes, rung),
                    }
                )
            off_tokens = token_digest(repeat_records["pdl_off"])
            grid_tokens = token_digest(repeat_records["pdl_grid"])
            off_full = output_digest(repeat_records["pdl_off"])
            grid_full = output_digest(repeat_records["pdl_grid"])
            ceiling_tokens_differ = (
                off_tokens != token_digest(repeat_records["ceiling"])
            )
            ceiling_full_differs = (
                off_full != output_digest(repeat_records["ceiling"])
            )
            ceiling_nonfinite = nonfinite_logprob(repeat_records["ceiling"])
            ceiling_wrongness = (
                ceiling_tokens_differ or ceiling_full_differs or ceiling_nonfinite
            )
            if off_full != grid_full:
                atomic_json(
                    results / "sample_output_mismatch.json",
                    {
                        "schema": "tier4.output.mismatch.v2",
                        "status": "rejected",
                        "admissible": False,
                        "phase": "sample",
                        "triplet_id": triplet_id,
                        "point": point,
                        "point_index": point_index,
                        "repeat": repeat,
                        "epoch": epoch,
                        "order_pattern": order_pattern,
                        "off": {
                            "output": repeat_records["pdl_off"],
                            "output_digest": off_full,
                            "token_digest": off_tokens,
                            "nonfinite_logprob": nonfinite_logprob(
                                repeat_records["pdl_off"]
                            ),
                        },
                        "grid": {
                            "output": repeat_records["pdl_grid"],
                            "output_digest": grid_full,
                            "token_digest": grid_tokens,
                            "nonfinite_logprob": nonfinite_logprob(
                                repeat_records["pdl_grid"]
                            ),
                        },
                        "token_match": off_tokens == grid_tokens,
                    },
                )
                raise RuntimeError(
                    "off/grid exact token+hex-logprob output mismatch "
                    f"tag={point['tag']} repeat={repeat} epoch={epoch} "
                    f"token_match={off_tokens == grid_tokens}"
                )
            if not ceiling_wrongness:
                raise RuntimeError(
                    "ceiling produced no observable wrongness evidence "
                    f"tag={point['tag']} repeat={repeat}"
                )
            validations.append(
                {
                    "point_index": point_index,
                    "tag": point["tag"],
                    "repeat": repeat,
                    "epoch": epoch,
                    "order_pattern": order_pattern,
                    "rung_sequence": list(timing_order),
                    "off_grid_token_match": True,
                    "off_grid_full_output_match": True,
                    "ceiling_token_differs": ceiling_tokens_differ,
                    "ceiling_full_output_differs": ceiling_full_differs,
                    "ceiling_nonfinite_logprob": ceiling_nonfinite,
                    "ceiling_wrongness_evidence": ceiling_wrongness,
                    "ceiling_verified": False,
                    "ceiling_unsafe": True,
                    "ceiling_correctness": "unsafe_not_validated",
                }
            )

    runtimes: dict[str, dict[str, Any]] = {}
    for rung_index, rung in enumerate(RUNGS):
        probes = build_probes[rung]
        runtime = {
            "schema": "tier4.runtime.v2",
            "rung": rung,
            "rung_index": rung_index,
            "triplet_mode": "same_process_adjacent_latin3",
            "triplet_id": triplet_id,
            "cohort_id": args.cohort_id or "diagnostic_smoke",
            "driver_pid": driver_pid,
            "model_fingerprint": fingerprint,
            "model_identity_manifest_sha256": identity_manifest_hash,
            "model_identity_snapshot_sha256": identity_snapshot_hash,
            "packages": packages,
            "device": device,
            "cache_fresh": True,
            "cache_root": str((results / "evidence" / rung / "cache").resolve()),
            "graph_mode": resolved_mode,
            "executed_full_decode": True,
            "pdl_scope": PDL_SCOPE,
            "prefill_mode": PREFILL_MODE,
            "trtllm_enable_pdl": False,
            "workers": probes,
            "graph_execution_proof": "pending_nsys_cuda_graph_node",
            "nsys_sqlite": None,
            "nsys_sqlite_sha256": None,
        }
        runtimes[rung] = runtime
        atomic_json(results / "evidence" / rung / "runtime.json", runtime)

    runtime_candidate_sha256 = {
        rung: sha256_bytes(
            json.dumps(runtime, separators=(",", ":"), sort_keys=True).encode()
        )
        for rung, runtime in runtimes.items()
    }

    raw = {
        "schema": SCHEMA,
        "status": "candidate",
        "admissible": False,
        "triplet_id": triplet_id,
        "triplet_mode": "same_process_adjacent_latin3",
        "cohort_id": args.cohort_id or "diagnostic_smoke",
        "driver_pid": driver_pid,
        "worker_cohort": list(next(iter(cohorts))),
        "model": str(model),
        "model_fingerprint": fingerprint,
        "model_identity_manifest_sha256": identity_manifest_hash,
        "model_identity_snapshot_sha256": identity_snapshot_hash,
        "formal_root_manifest_sha256": formal_root_manifest_hash,
        "packages": packages,
        "device": device,
        "resolved_graph_mode": resolved_mode,
        "pdl_scope": PDL_SCOPE,
        "prefill_mode": PREFILL_MODE,
        "trtllm_enable_pdl": False,
        "points": args.point,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "bootstrap_samples": args.bootstrap_samples,
        "allow_short": args.allow_short,
        "proof_records": proof_records,
        "build_probes": build_probes,
        "warmup_records": warmup_records,
        "samples": samples,
        "validations": validations,
        "invocations": invocation,
        "formal_manifest": {
            "cohort_id": args.cohort_id or "diagnostic_smoke",
            "model_identity_manifest_sha256": identity_manifest_hash,
            "model_identity_snapshot_sha256": identity_snapshot_hash,
            "formal_root_manifest_sha256": formal_root_manifest_hash,
            "semantic_rungs": list(RUNGS),
            "proof_rung_order": list(RUNGS),
            "timing_order_cycle": [list(order) for order in ORDER_CYCLE],
            "points": args.point,
            "proof_point": proof_point,
            "repeats_per_rung": args.repeats,
            "warmup_triplets_per_point": args.warmups,
            "bootstrap_samples": args.bootstrap_samples,
            "max_num_seqs": max_batch,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max(args.max_num_batched_tokens, max_batch),
            "gpu_memory_utilization": args.gpu_mem_util,
            "graph_mode": resolved_mode,
            "pdl_scope": PDL_SCOPE,
            "prefill_mode": PREFILL_MODE,
            "trtllm_enable_pdl": False,
            "kv_offloading_size_gib": args.kv_offloading_size,
            "kv_offloading_backend": args.kv_offloading_backend,
            "text_only_limit_mm_per_prompt": {"image": 0, "video": 0},
        },
        "runtimes": runtimes,
        "runtime_candidate_sha256": runtime_candidate_sha256,
    }
    atomic_json(results / "raw_triplet.json", raw)
    print(
        f"TIER4_CANDIDATE triplet_id={triplet_id} driver_pid={driver_pid} "
        f"worker_cohort={','.join(map(str, raw['worker_cohort']))} "
        f"points={len(args.point)} repeats={args.repeats} samples={len(samples)} "
        f"graph_mode={resolved_mode} admissible=0"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument(
        "--point",
        action="append",
        type=parse_point,
        default=[],
        help="TAG:BATCH:SEQ:GEN:SCOPE; repeat for multiple points",
    )
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--gpu-mem-util", type=float, default=0.80)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--variant-timeout", type=float, default=1800.0)
    parser.add_argument("--cohort-id")
    parser.add_argument(
        "--model-identity",
        type=Path,
        help="top-level deterministic model_identity.json (required formally)",
    )
    parser.add_argument(
        "--formal-root-manifest",
        type=Path,
        help="launch-time top-level manifest.json (required formally)",
    )
    parser.add_argument(
        "--proof-point",
        type=parse_point,
        help="unmeasured decode point used only for FULL graph execution proof",
    )
    parser.add_argument("--kv-offloading-size", type=float)
    parser.add_argument(
        "--kv-offloading-backend", choices=["native", "lmcache"], default="native"
    )
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument(
        "--forensic-only",
        action="store_true",
        help="diagnostic fixed-prompt output stability probe; requires --allow-short",
    )
    parser.add_argument(
        "--forensic-epoch",
        action="append",
        type=int,
        default=[],
        help="fixed prompt epoch for forensic-only; repeat for multiple epochs",
    )
    parser.add_argument(
        "--forensic-proof-first",
        action="store_true",
        help="forensic-only exact build->proof->formal-triplet ordering",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.forensic_only and not args.allow_short:
        parser.error("--forensic-only requires --allow-short")
    if args.forensic_proof_first and not args.forensic_only:
        parser.error("--forensic-proof-first requires --forensic-only")
    if not args.point:
        args.point = [parse_point("decode_smoke:1:64:2:decode")]
    try:
        return run(args)
    except Exception as exc:  # noqa: BLE001 - preserve a structured fail-closed artifact
        import traceback

        traceback.print_exc()
        payload = {
            "schema": SCHEMA,
            "status": "error",
            "admissible": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        try:
            if hasattr(args, "results"):
                atomic_json(args.results.resolve() / "driver_error.json", payload)
        except Exception:
            pass
        print(f"TIER4_ERROR {payload['error']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
