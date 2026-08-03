// PDL demo kernels -> compiled to PTX to inspect griddepcontrol lowering.
//
// The two PDL device intrinsics are, verbatim from the CUDA runtime header
// cuda_device_runtime_api.h (identical in CUDA 12.x / 13.x):
//
//   cudaTriggerProgrammaticLaunchCompletion():
//       asm volatile("griddepcontrol.launch_dependents;":::);
//   cudaGridDependencySynchronize():
//       asm volatile("griddepcontrol.wait;":::"memory");
//
// We inline them here so the file is self-contained for NVRTC (no CUDA headers
// needed), while emitting exactly the same PTX the real intrinsics produce.

static __device__ __forceinline__ void pdl_trigger_launch_dependents(void) {
    asm volatile("griddepcontrol.launch_dependents;":::);
}
static __device__ __forceinline__ void pdl_grid_dependency_synchronize(void) {
    asm volatile("griddepcontrol.wait;":::"memory");
}

extern "C" __global__ void producer(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Work that must finish before the consumer's dependent reads.
    if (i < n) out[i] = in[i] * 2.0f;

    // PDL producer side: allow dependent (consumer) grid to be scheduled early.
    // -> griddepcontrol.launch_dependents;
    pdl_trigger_launch_dependents();

    // Trailing work that can overlap with the consumer's independent prologue.
    if (i < n) out[i] += 1.0f;
}

extern "C" __global__ void consumer(float* out, const float* producerOut, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Independent prologue: safe to run before the producer grid finishes.
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) acc += (float)k;

    // PDL consumer side: block until ALL prerequisite grids completed & visible.
    // -> griddepcontrol.wait;
    pdl_grid_dependency_synchronize();

    // Dependent work: now safe to read producer's output.
    if (i < n) out[i] = producerOut[i] + acc;
}
