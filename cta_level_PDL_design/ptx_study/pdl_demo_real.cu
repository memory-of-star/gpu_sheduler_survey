// PDL demo using the REAL CUDA runtime intrinsics (needs full CUDA headers).
// Compiled with nvcc 13.3 inside the container to show genuine PTX 9.3 output.
#include <cuda_runtime.h>

extern "C" __global__ void producer(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f;

    // Real intrinsic -> griddepcontrol.launch_dependents;
    cudaTriggerProgrammaticLaunchCompletion();

    if (i < n) out[i] += 1.0f;
}

extern "C" __global__ void consumer(float* out, const float* producerOut, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) acc += (float)k;

    // Real intrinsic -> griddepcontrol.wait;
    cudaGridDependencySynchronize();

    if (i < n) out[i] = producerOut[i] + acc;
}
