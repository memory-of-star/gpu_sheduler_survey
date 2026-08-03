// CUDA Graph path for PDL: the DEVICE code is STILL just griddepcontrol.
// A CUDA Graph is a HOST-side DAG (nodes + edges). The graph topology lives in
// cudaGraph_t / cudaGraphExec_t on the host; it never enters the kernel PTX.
// The producer->consumer PDL link is expressed as a graph EDGE of type
// cudaGraphDependencyTypeProgrammatic (host metadata), not as any PTX instruction.
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

// ---------- HOST: build a CUDA Graph with a PROGRAMMATIC edge ----------
void launch_graph(float* out, float* pout, const float* in, int n,
                  dim3 grid, dim3 block) {
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    // Producer kernel node.
    void* pArgs[] = { (void*)&pout, (void*)&in, (void*)&n };
    cudaKernelNodeParams pParams = {};
    pParams.func = (void*)producer;
    pParams.gridDim = grid; pParams.blockDim = block;
    pParams.kernelParams = pArgs;
    cudaGraphNode_t pNode;
    cudaGraphAddKernelNode(&pNode, graph, nullptr, 0, &pParams);

    // Consumer kernel node.
    void* cArgs[] = { (void*)&out, (void*)&pout, (void*)&n };
    cudaKernelNodeParams cParams = {};
    cParams.func = (void*)consumer;
    cParams.gridDim = grid; cParams.blockDim = block;
    cParams.kernelParams = cArgs;
    cudaGraphNode_t cNode;
    cudaGraphAddKernelNode(&cNode, graph, nullptr, 0, &cParams);

    // The PDL link is a graph EDGE, not device code: programmatic dependency
    // producer -> consumer (consumer's griddepcontrol.wait is gated by it).
    cudaGraphEdgeData edge = {};
    edge.type = cudaGraphDependencyTypeProgrammatic;
    edge.from_port = cudaGraphKernelNodePortProgrammatic;
    cudaGraphAddDependencies(graph, &pNode, &cNode, &edge, 1);

    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec, graph, 0);

    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaGraphLaunch(exec, s);      // the whole DAG is still launched INTO a stream
    cudaStreamSynchronize(s);
}
