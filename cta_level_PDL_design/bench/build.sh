#!/usr/bin/env bash
# build.sh — compile all CTA-PDL benchmarks.
#
# Default target is sm_103 (B300). Override with ARCH=sm_90 for H100, etc.
#   ARCH=sm_90 ./build.sh
#
# Requires CUDA >= 12.0 for libcu++ atomic_ref; CLC (clc_probe) additionally needs
# CUDA >= 12.8 headers and an sm_100+ target to actually run.

set -euo pipefail
cd "$(dirname "$0")"

ARCH="${ARCH:-sm_103}"
NVCC="${NVCC:-nvcc}"

# -lineinfo keeps ncu/nsys source correlation without the codegen impact of -G.
FLAGS=(-O3 -std=c++17 -arch="${ARCH}" -lineinfo -I. --expt-relaxed-constexpr)

echo "== building for ${ARCH} with $(${NVCC} --version | tail -1)"

build() {
    local src="$1" out="$2"
    echo "-- ${src} -> ${out}"
    if ! ${NVCC} "${FLAGS[@]}" "${src}" -o "${out}" 2> >(tee "/tmp/${out}.err" >&2); then
        echo "!! FAILED: ${src} (see /tmp/${out}.err)" >&2
        return 1
    fi
}

FAILED=0
build cta_dep_bench.cu cta_dep_bench || FAILED=1
build cta_dep_pilot.cu cta_dep_pilot || FAILED=1
build tier0_facts.cu   tier0_facts   || FAILED=1

# CLC uses sm_100+ PTX; if the chosen arch predates Blackwell the inline asm will not
# assemble. Build it best-effort so H100 users still get everything else.
if ! build clc_probe.cu clc_probe; then
    echo "-- clc_probe skipped (needs sm_100+ / CUDA >= 12.8); this is expected on ${ARCH}"
fi

if [ "${FAILED}" -ne 0 ]; then
    echo "== BUILD FAILED"
    exit 1
fi
echo "== build OK"
