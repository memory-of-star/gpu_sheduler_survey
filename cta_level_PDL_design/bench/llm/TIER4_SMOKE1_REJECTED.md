# Tier 4 persistent-driver smoke1 rejection

The first persistent-driver smoke is preserved at
`/tmp/cta_tier4_persistent_smoke_20260805` with its Nsight report at
`/tmp/cta_tier4_persistent_smoke_20260805_profile.nsys-rep` and exported
SQLite at `/tmp/cta_tier4_persistent_smoke_20260805_profile.sqlite`.

It is **REJECTED** and none of its timings are admissible.  The single driver
and worker PID remained stable and all three requests executed FULL-decode
CUDA graphs, but vLLM had frozen its environment-backed cache properties after
service initialization.  Consequently the grid variant reused the off-rung
AOT callable instead of lowering into its requested fresh cache.  The observed
artifact counts were:

| rung | PTX | cubin | `wait` | `launch_dependents` |
|---|---:|---:|---:|---:|
| `pdl_off` | 112 | 112 | 0 | 0 |
| `pdl_grid` | 0 | 0 | 0 | 0 |
| `ceiling` | 7 | 7 | 0 | 0 |

The ceiling files were runtime helpers, not a separately lowered model.
Therefore this run does not instantiate the three semantic variants and its
formal timing contribution is exactly zero samples.

The follow-up driver disables vLLM compile-cache loading, resets vLLM's frozen
environment cache after each rung switch, and now hard-fails immediately after
variant construction unless the isolated caches prove off `0/0`, grid
positive/positive, and ceiling `0/positive` GDC instruction counts on an
`sm_100*` target.
