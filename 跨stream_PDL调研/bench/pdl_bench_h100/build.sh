#!/usr/bin/env bash
# Build the PDL benchmark for H100 (sm_90). Only needed if you want to REBUILD from source;
# the package already ships a prebuilt offline binary `./pdl_bench` (static cudart).
# Override arch/compiler if needed, e.g.:  ARCH=sm_90a ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${ARCH:-sm_90}"          # H100 / H200 = compute capability 9.0
NVCC="${NVCC:-nvcc}"

if ! command -v "$NVCC" >/dev/null 2>&1; then
  echo "ERROR: nvcc not found. You can still run the prebuilt ./pdl_bench directly." >&2
  echo "       Or set NVCC=/path/to/nvcc / add CUDA to PATH to rebuild." >&2
  exit 1
fi

echo "Using: $($NVCC --version | tail -1)"
echo "Compiling pdl_bench.cu   for -arch=$ARCH ..."
"$NVCC" -O3 -std=c++17 -arch="$ARCH" -o pdl_bench   pdl_bench.cu
echo "Compiling pdl_diamond.cu for -arch=$ARCH ..."
"$NVCC" -O3 -std=c++17 -arch="$ARCH" -o pdl_diamond pdl_diamond.cu
echo "OK -> ./pdl_bench and ./pdl_diamond   (run: ./run.sh)"
