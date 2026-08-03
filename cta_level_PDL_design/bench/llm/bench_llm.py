#!/usr/bin/env python3
"""Tier 4 driver: measure one (rung, batch, seq) point of the LLM PDL bracket.

Emits a single machine-parsable SUMMARY line, matching the convention used by the CUDA
microbenchmarks so tools/llm_bracket.py can consume both.

The three rungs are selected by the ENVIRONMENT (set by run_llm_sweep.sh), not by flags:
    PDL_off   TRTLLM_ENABLE_PDL=0  TORCHINDUCTOR_ENABLE_PDL=0
    PDL_grid  TRTLLM_ENABLE_PDL=1  TORCHINDUCTOR_ENABLE_PDL=1     <- the real FLOOR
    Ceiling   ... plus CTA_PDL_CEILING=1, which no-ops gdc_wait

CEILING IS DELIBERATELY INCORRECT. Removing gdc_wait lets consumers read data the producer
has not written yet. Outputs are garbage; only the timing is meaningful. The script refuses
to run that rung unless CTA_PDL_CEILING is set, and stamps every ceiling result as unverified.

Engines: vllm (default). Others can be added; the measurement contract is just
"run N requests, report latency/throughput".
"""

import argparse
import json
import os
import statistics
import sys
import time


def env_flag(name, default="0"):
    return os.environ.get(name, default) not in ("0", "", "false", "False")


def apply_ceiling_patch():
    """Turn Triton's PDL wait into a no-op, realizing the Ceiling rung.

    Triton's PDL support emits an UNCONDITIONAL gdc_wait() before any tl.load, on the
    conservative assumption that the predecessor kernel could have written anywhere. That is
    exactly BlockMaestro's "fully-connected fallback", now shipping in production. Removing
    it measures what a perfectly precise dependency would cost: nothing.
    """
    patched = []
    try:
        import triton.language.extra.cuda as tlcuda  # type: ignore
        if hasattr(tlcuda, "gdc_wait"):
            tlcuda.gdc_wait = lambda *a, **k: None
            patched.append("triton.language.extra.cuda.gdc_wait")
    except Exception as e:  # noqa: BLE001 - best effort across triton versions
        print(f"[ceiling] triton patch unavailable: {e}", file=sys.stderr)

    if not patched:
        print("[ceiling] WARNING: nothing was patched; this run is NOT a true ceiling",
              file=sys.stderr)
    else:
        print(f"[ceiling] patched: {', '.join(patched)}", file=sys.stderr)
    return patched


def run_vllm(args):
    from vllm import LLM, SamplingParams  # imported late so --help works without vllm

    llm = LLM(
        model=args.model,
        max_model_len=args.seq + args.gen,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=False,          # FULL CUDA graph: the mode where PDL is enabled
        dtype="bfloat16",
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.gen, ignore_eos=True)

    # A fixed-length prompt keeps the prefill grid size constant across rungs. Token ids are
    # arbitrary; only the shape matters for this measurement.
    prompt_ids = [[1000 + (i % 5000) for i in range(args.seq)] for _ in range(args.batch)]

    # warmup
    llm.generate(prompt_ids=prompt_ids[:1], sampling_params=sp, use_tqdm=False)

    lat = []
    iters = max(1, args.requests // max(1, args.batch))
    for _ in range(iters):
        t0 = time.perf_counter()
        llm.generate(prompt_ids=prompt_ids, sampling_params=sp, use_tqdm=False)
        lat.append(time.perf_counter() - t0)

    total_tokens = args.batch * args.gen * iters
    wall = sum(lat)
    return {
        "iters": iters,
        "median_s": statistics.median(lat),
        "min_s": min(lat),
        "wall_s": wall,
        "tok_per_s": total_tokens / wall if wall else 0.0,
        "tok_per_s_per_user": (args.gen * iters) / wall if wall else 0.0,
    }


ENGINES = {"vllm": run_vllm}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--engine", default="vllm", choices=sorted(ENGINES))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=4096, help="prompt length")
    ap.add_argument("--gen", type=int, default=64, help="tokens to generate (decode phase)")
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--rung", default="pdl_grid",
                    choices=["pdl_off", "pdl_grid", "ceiling"])
    ap.add_argument("--tag", default="llm")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--json", help="also write the result as JSON")
    args = ap.parse_args()

    pdl_trt = env_flag("TRTLLM_ENABLE_PDL")
    pdl_ind = env_flag("TORCHINDUCTOR_ENABLE_PDL")
    ceiling = env_flag("CTA_PDL_CEILING")

    if args.rung == "ceiling" and not ceiling:
        print("refusing to run the ceiling rung without CTA_PDL_CEILING=1 "
              "(it produces intentionally WRONG results)", file=sys.stderr)
        return 2

    patched = apply_ceiling_patch() if ceiling else []

    print(f"[cfg] rung={args.rung} batch={args.batch} seq={args.seq} gen={args.gen} "
          f"TRTLLM_ENABLE_PDL={int(pdl_trt)} TORCHINDUCTOR_ENABLE_PDL={int(pdl_ind)} "
          f"ceiling={int(ceiling)}")

    try:
        res = ENGINES[args.engine](args)
    except Exception as e:  # noqa: BLE001 - one failed point must not kill the campaign
        print(f"ERROR: {e}", file=sys.stderr)
        print(f"SUMMARY tier=4 tag={args.tag} rung={args.rung} status=error "
              f"batch={args.batch} seq={args.seq}")
        return 1

    print(f"SUMMARY tier=4 tag={args.tag} rung={args.rung} status=ok engine={args.engine} "
          f"batch={args.batch} seq={args.seq} gen={args.gen} iters={res['iters']} "
          f"median_s={res['median_s']:.5f} min_s={res['min_s']:.5f} "
          f"tok_per_s={res['tok_per_s']:.3f} tok_per_s_per_user={res['tok_per_s_per_user']:.3f} "
          f"pdl_trt={int(pdl_trt)} pdl_inductor={int(pdl_ind)} ceiling={int(ceiling)} "
          f"verified={0 if ceiling else 1}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"args": vars(args), "result": res, "patched": patched}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
