# Cross-stream PDL benefit benchmark (B300 / sm_103)

Confirms empirically whether **cross-stream Programmatic Dependent Launch (PDL)** overlaps a
producer kernel's *tail* with a dependent consumer kernel's *independent prologue* on a B300.

## What it measures

One dependent producer→consumer pair, timed with CUDA events, fully synchronized between
repeats (so the **only** possible overlap is prologue‖tail — no cross-iteration pipelining to
muddy the result). Four launch strategies:

| Mode | Wiring | Expectation |
|------|--------|-------------|
| `BASE`        | cross-stream, **ordinary event** dependency (`cudaEventRecord`+`cudaStreamWaitEvent`) | no overlap → ~`tail+prologue` |
| `PDL_XS`      | cross-stream, **programmatic event** (`cudaLaunchAttributeProgrammaticEvent`) + `cudaStreamWaitEvent`, **eager** | may NOT overlap |
| `PDL_CAPTURE` | the **same literal `sA`/`sB` two-stream** code, recorded via `cudaStreamBeginCapture` → graph | early launch → ~`max(tail,prologue)` |
| `PDL_GRAPH`   | **CUDA graph** with a **programmatic edge** (`cudaGraphDependencyTypeProgrammatic`), built directly | early launch → ~`max(tail,prologue)` |
| `PDL_SS`      | same-stream, **`cudaLaunchAttributeProgrammaticStreamSerialization`** (canonical PDL) | reference, ~`max(tail,prologue)` |
| `CONC`        | **no dependency** (unsafe, results wrong) | concurrency ceiling ~`max(tail,prologue)` |

> **Empirical finding (H100 NVL, CUDA 13.3):** `PDL_SS`, `PDL_GRAPH` reach **2×** (== `CONC`
> ceiling); the *eager* `PDL_XS` shows **1.00×** (no overlap). `PDL_CAPTURE` proves the point:
> the **identical** two-stream (`sA`/`sB`) code that gives 1× in eager mode gives ~2× once it is
> **captured into a graph** — because `cudaLaunchAttributeProgrammaticEvent` is a stream-capture
> construct that becomes a programmatic graph edge.

## Diamond experiment (`pdl_diamond`)

A second binary tests a diamond graph where two middle nodes run on **different internal queues**:

```
        producer            (x = in*2)
        /      \
     midA        midB       (parallel -> different command queues)
        \      /
         final              (out = yA + yB)
```

Every node = independent prologue spin → `[griddepcontrol.wait]` → tiny compute → `[trigger]` →
tail spin. It builds the graph twice — **ordinary edges** vs **programmatic (PDL) edges** — and
compares. Theoretical: ordinary ≈ `4T`, PDL ≈ `2T` (each stage's prologue overlaps the previous
stage's tail), i.e. ~2×. Correctness verified: `out == 4*in + 2`.

```bash
./pdl_diamond --repeats 50 --tail 20000000
```

> **Empirical finding (H100 NVL, CUDA 13.3):** `PDL_SS` reaches the full **2×** (== `CONC`
> ceiling), but the *eager* `PDL_XS` path shows **1.00×** (no overlap). This is because
> `cudaLaunchAttributeProgrammaticEvent` is documented as a way to express a programmatic
> dependency during **stream capture** (it becomes a graph edge); in eager stream execution
> `cudaStreamWaitEvent` on it degrades to an ordinary completion dependency. `PDL_GRAPH` was
> added to test the intended cross-node path directly — compare its number against `BASE`.

The kernels are built so the **theoretical benefit is ~2×**: producer body and consumer
epilogue are tiny; `tail` and `prologue` are long, tunable GPU-clock spins. With
`tail == prologue == T`: `BASE ≈ 2T`, ideal PDL `≈ T`.

- Producer: `pout[i] = in[i]*2` → `cudaTriggerProgrammaticLaunchCompletion()` → spin(`tail`)
- Consumer: spin(`prologue`) → `cudaGridDependencySynchronize()` → `out[i] = pout[i]+1`
- Correctness is verified (`out == in*2+1`) for every mode except `CONC`.

Grids default to **one block per SM** (small blocks) so producer and consumer can co-reside —
a prerequisite for the overlap to physically happen.

## Requirements (target machine)

- NVIDIA B300 (compute capability 10.3, `sm_103`) + driver (R580+; new 13.4 features want R616+).
- CUDA toolkit with `nvcc` on `PATH`. CUDA ≥ 13.0 supports `sm_103`.
  - CUDA 13.3 (GA) or 13.4 preview both work. (PDL itself is unchanged across 9.3/9.4 PTX.)

## Build

```bash
./build.sh                 # -arch=sm_103 by default
# overrides:
ARCH=sm_100 ./build.sh                         # e.g. to also try on B200
NVCC=/usr/local/cuda-13.4/bin/nvcc ./build.sh  # pin a specific toolkit
```

## Run

```bash
./run.sh                   # detailed run + a sweep over spin length
# or drive it directly:
./pdl_bench --help
./pdl_bench --repeats 50 --tail 20000000 --prologue 20000000
./pdl_bench --blocks 132 --threads 128 --tail 40000000
```

## Reading the result

- `SUMMARY ... speedup_xs=X.XX` — cross-stream PDL speedup vs baseline.
- `VERDICT: cross-stream PDL HELPS (X.XXx ...)` — headline answer.
- Expect `PDL_XS` to trend toward the `CONC` ceiling (~2× of `BASE`) as the spin grows.
  If `PDL_XS ≈ BASE`, PDL is **not** overlapping on this device/driver (overlap is opportunistic).

## Notes / caveats

- PDL overlap is *opportunistic* (per NVIDIA docs); this benchmark maximizes the odds by keeping
  both grids to one wave with small blocks so they co-reside.
- `CONC` intentionally has a data race (no dependency) and its output is wrong on purpose — it
  exists only as a timing ceiling. It is excluded from correctness checks.
- If you see `PDL_XS` incorrect (it should not), the VERDICT line flags it.

## Files

- `pdl_bench.cu` — kernels + host harness (all four modes, timing, correctness).
- `build.sh` — compile (`-arch=sm_103` default).
- `run.sh` — detailed run + spin-length sweep.
