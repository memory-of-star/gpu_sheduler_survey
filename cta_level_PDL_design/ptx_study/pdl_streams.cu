// Cross-stream vs same-stream PDL.
//
// KEY POINT: the DEVICE code (the griddepcontrol ops) is IDENTICAL in both cases.
// "Same-stream" vs "cross-stream" is decided entirely by the HOST launch config:
//   - same-stream : cudaLaunchAttributeProgrammaticStreamSerialization
//   - cross-stream: cudaLaunchAttributeProgrammaticEvent + cudaStreamWaitEvent
// Host code is NOT part of PTX, so `nvcc -ptx` emits the same two griddepcontrol
// instructions regardless of which launch path is used.
#include <cuda_runtime.h>

extern "C" __global__ void producer(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f;
    cudaTriggerProgrammaticLaunchCompletion();   // -> griddepcontrol.launch_dependents;
    if (i < n) out[i] += 1.0f;
}

extern "C" __global__ void consumer(float* out, const float* producerOut, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) acc += (float)k;
    cudaGridDependencySynchronize();             // -> griddepcontrol.wait;
    if (i < n) out[i] = producerOut[i] + acc;
}

// ---------- HOST: same-stream PDL (stream serialization) ----------
void launch_same_stream(cudaStream_t s, float* out, float* pout, const float* in,
                        int n, dim3 grid, dim3 block) {
    cudaLaunchConfig_t cfgP = {};
    cfgP.gridDim = grid; cfgP.blockDim = block; cfgP.stream = s;
    cudaLaunchKernelEx(&cfgP, producer, pout, in, n);

    // Consumer in the SAME stream opts into programmatic serialization.
    cudaLaunchAttribute a = {};
    a.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    a.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfgC = {};
    cfgC.gridDim = grid; cfgC.blockDim = block; cfgC.stream = s;
    cfgC.attrs = &a; cfgC.numAttrs = 1;
    cudaLaunchKernelEx(&cfgC, consumer, out, pout, n);
}

// ---------- HOST: cross-stream PDL (programmatic event) ----------
void launch_cross_stream(cudaStream_t sProd, cudaStream_t sCons, cudaEvent_t ev,
                         float* out, float* pout, const float* in, int n,
                         dim3 grid, dim3 block) {
    // Producer in stream A records a PROGRAMMATIC event at griddepcontrol.launch_dependents.
    cudaLaunchAttribute ap = {};
    ap.id = cudaLaunchAttributeProgrammaticEvent;
    ap.val.programmaticEvent.event = ev;
    ap.val.programmaticEvent.flags = 0;
    ap.val.programmaticEvent.triggerAtBlockStart = 0;
    cudaLaunchConfig_t cfgP = {};
    cfgP.gridDim = grid; cfgP.blockDim = block; cfgP.stream = sProd;
    cfgP.attrs = &ap; cfgP.numAttrs = 1;
    cudaLaunchKernelEx(&cfgP, producer, pout, in, n);

    // Consumer in a DIFFERENT stream waits on that programmatic event.
    cudaStreamWaitEvent(sCons, ev, 0);
    cudaLaunchConfig_t cfgC = {};
    cfgC.gridDim = grid; cfgC.blockDim = block; cfgC.stream = sCons;
    cudaLaunchKernelEx(&cfgC, consumer, out, pout, n);
}
