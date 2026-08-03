#!/usr/bin/env bash
# Build the PDL benchmark. Defaults to B300 (sm_103).
# Override arch/compiler if needed, e.g.:  ARCH=sm_100 ./build.sh   or   NVCC=/usr/local/cuda-13.4/bin/nvcc ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${ARCH:-sm_103}"        # B300 / GB300 = compute capability 10.3
NVCC="${NVCC:-nvcc}"

if ! command -v "$NVCC" >/dev/null 2>&1; then
  echo "ERROR: nvcc not found. Set NVCC=/path/to/nvcc or add CUDA to PATH." >&2
  exit 1
fi

echo "Using: $($NVCC --version | tail -1)"
echo "Compiling pdl_bench.cu   for -arch=$ARCH ..."
"$NVCC" -O3 -std=c++17 -arch="$ARCH" -o pdl_bench   pdl_bench.cu
echo "Compiling pdl_diamond.cu for -arch=$ARCH ..."
"$NVCC" -O3 -std=c++17 -arch="$ARCH" -o pdl_diamond pdl_diamond.cu
echo "OK -> ./pdl_bench and ./pdl_diamond   (run: ./run.sh)"
