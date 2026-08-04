# Cross-stream PDL benefit benchmark — H100 offline package (sm_90)

Same benchmark as the B300 package, retargeted to **H100 (compute capability 9.0 / `sm_90`)**.
PDL is supported since CC 9.0, so H100 runs it fully (`griddepcontrol`, cross-stream programmatic
events, `cudaGridDependencySynchronize`, `clock64` spins — all available on `sm_90`).

**This package is offline / self-contained:** it ships a prebuilt, statically-linked binary
`./pdl_bench`, so the target H100 machine needs **only the NVIDIA driver — no `nvcc`, no internet,
no `libcudart`**. Source + `build.sh` are included as a fallback if you prefer to rebuild.

## What it measures

One dependent producer→consumer pair, timed with CUDA events, fully synchronized between repeats
(the only possible overlap is prologue‖tail). Modes:

| Mode | Wiring | Expectation |
|------|--------|-------------|
| `BASE`        | cross-stream, ordinary event dependency | no overlap → ~`tail+prologue` |
| `PDL_XS`      | cross-stream, **programmatic event** + `cudaStreamWaitEvent`, **eager** | may NOT overlap |
| `PDL_CAPTURE` | the **same `sA`/`sB` two-stream** code recorded via `cudaStreamBeginCapture` → graph | early launch → ~`max(tail,prologue)` |
| `PDL_GRAPH`   | **CUDA graph** with a **programmatic edge**, built directly | early launch → ~`max(tail,prologue)` |
| `PDL_SS`      | same-stream, `cudaLaunchAttributeProgrammaticStreamSerialization` (canonical) | reference |
| `CONC`        | no dependency (unsafe, wrong results) | concurrency ceiling ~`max(tail,prologue)` |

> **Observed on H100 NVL (CUDA 13.3):** `PDL_SS` = `PDL_GRAPH` = **2.00×** (== `CONC` ceiling),
> while the *eager* `PDL_XS` = **1.00×**. `PDL_CAPTURE` records the *identical* two-stream code
> into a graph and is expected to also hit ~2× — i.e. it's the **capture into a graph**, not the
> stream code itself, that unlocks the overlap.

## Diamond experiment (`pdl_diamond`)

Second binary: a diamond graph `producer -> {midA, midB} -> final` where the two middle nodes are
parallel and land on **different internal command queues**. It builds the graph with **ordinary
edges** vs **programmatic (PDL) edges** and compares (theoretical ordinary ≈ `4T`, PDL ≈ `2T`,
~2×; correctness `out == 4*in + 2`). Prebuilt offline binary included.

```bash
./pdl_diamond --repeats 50 --tail 20000000
```

Kernels are built so the theoretical benefit is ~2× (`tail == prologue == T`: `BASE ≈ 2T`,
ideal PDL `≈ T`). Correctness (`out == in*2+1`) is verified for every mode except `CONC`.

## Run offline (no build needed)

```bash
tar xzf pdl_bench_h100.tar.gz && cd pdl_bench_h100
./pdl_bench --help
./pdl_bench --repeats 50 --tail 20000000 --prologue 20000000
./run.sh                     # detailed run + spin-length sweep
```

## Requirements

- NVIDIA **H100 / H200** (or any `sm_90` device), with the NVIDIA driver installed.
- Prebuilt binary was compiled with **CUDA 13.3**, so it needs a driver that supports CUDA 13.3
  (**R580+**). Check with `nvidia-smi` (top-right "CUDA Version" ≥ 13.3).
  - If your driver is older, the binary will fail with *"CUDA driver version is insufficient"* —
    in that case **rebuild from source** with your local toolkit (next section).

## Rebuild from source (if needed)

Only requires `nvcc` on the target (any CUDA ≥ 11.8 supports `sm_90`/PDL):

```bash
./build.sh                   # -arch=sm_90 by default
ARCH=sm_90a ./build.sh       # if you want the arch-specific variant
NVCC=/usr/local/cuda-12.6/bin/nvcc ./build.sh   # pin an older toolkit to match an older driver
```

## Reading the result

- `SUMMARY ... speedup_xs=X.XX` — cross-stream PDL speedup vs baseline.
- `VERDICT: cross-stream PDL HELPS (X.XXx ...)` — headline answer.
- Expect `PDL_XS` to trend toward the `CONC` ceiling (~2× of `BASE`) as the spin grows.
  If `PDL_XS ≈ BASE`, PDL is not overlapping on this device/driver (overlap is opportunistic).

## H100 vs B300 note

The device code is identical; only `-arch` differs (`sm_90` for H100, `sm_103` for B300). PDL/PTX
`griddepcontrol` semantics are the same across both. On H100 the launch latency and SM count differ
from B300, so absolute times differ, but the BASE-vs-PDL_XS comparison is apples-to-apples.

## Files

- `pdl_bench`      — prebuilt static `sm_90` binary (offline; needs only the driver).
- `pdl_diamond`    — prebuilt static `sm_90` binary for the diamond experiment.
- `pdl_bench.cu`   — source (kernels + host harness, 6 modes).
- `pdl_diamond.cu` — source (diamond graph: PDL edges vs ordinary edges).
- `build.sh`       — rebuild both (`-arch=sm_90` default).
- `run.sh`         — detailed run + spin sweep + diamond (runs prebuilt binaries if present).
