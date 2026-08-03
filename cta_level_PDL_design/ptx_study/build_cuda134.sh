#!/usr/bin/env bash
# Runs INSIDE the container: enable CUDA 13.4 preview channel, install nvcc 13.4,
# compile the PDL demos for B300 (sm_103) to PTX 9.4.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
ARCH=${ARCH:-sm_103}   # B300 / GB300 = compute capability 10.3
TAG=${ARCH/_/}         # sm_103 -> sm103 (filename suffix)

echo "### apt update (base)"
apt-get update -qq

echo "### install fetch tools"
apt-get install -y --no-install-recommends wget ca-certificates >/dev/null

echo "### add NVIDIA preview (Early Access) channel keyring"
wget -q https://packages.nvidia.com/noble/nvidia-preview-keyring.deb -O /tmp/nvidia-preview-keyring.deb
dpkg -i /tmp/nvidia-preview-keyring.deb

echo "### apt update (with preview channel)"
apt-get update -qq

echo "### install CUDA 13.4 nvcc + cudart headers"
apt-get install -y --no-install-recommends cuda-nvcc-13-4 cuda-crt-13-4 cuda-cudart-dev-13-4

NVCC=/usr/local/cuda-13.4/bin/nvcc
echo "### nvcc version"; "$NVCC" --version
echo "### sm_103 supported?"; "$NVCC" --list-gpu-arch | grep -x "compute_103" && echo "yes (B300)"

echo "### compile PDL demos for $ARCH -> PTX 9.4"
cd /work
"$NVCC" -arch=$ARCH -ptx pdl_demo.cu      -o pdl_demo_cuda134_${TAG}.ptx
"$NVCC" -arch=$ARCH -ptx pdl_demo_real.cu -o pdl_demo_real_cuda134_${TAG}.ptx
"$NVCC" -arch=$ARCH -ptx pdl_streams.cu   -o pdl_streams_cuda134_${TAG}.ptx
"$NVCC" -arch=$ARCH -ptx pdl_graph.cu     -o pdl_graph_cuda134_${TAG}.ptx
echo "### BUILD_OK"
