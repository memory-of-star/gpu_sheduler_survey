// pdl_diamond.cu — Diamond CUDA graph, PDL (programmatic) edges vs ordinary edges.
//
//            producer            (writes x = in*2)
//            /      \            <- two edges
//        midA        midB        (parallel nodes -> land on DIFFERENT internal queues)
//            \      /            <- two edges
//             final             (out = yA + yB)
//
// Every node has: independent PROLOGUE spin -> [wait upstream] -> tiny compute -> [trigger] -> TAIL spin.
// With all spins == T:
//   ordinary edges (must fully complete before next stage): depth 3 of serial (T)+(2T)+(T) ~= 4T
//   programmatic edges (each stage's prologue overlaps the previous stage's tail):        ~= 2T
// => theoretical ~2x from PDL on this diamond.  midA/midB run concurrently in both cases.
//
// Correctness (both variants): out = yA + yB = (x+1)+(x+1) = 2*(2*in+1) = 4*in + 2.
//
// Build:  ./build.sh  (also builds this)     Run:  ./pdl_diamond --help

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); \
    exit(1);} } while(0)

__device__ __forceinline__ void spin_cycles(unsigned long long cyc) {
    long long start = clock64();
    while ((unsigned long long)(clock64() - start) < cyc) { asm volatile("" ::: "memory"); }
}

// root producer: tiny body -> trigger (for downstream) -> tail
__global__ void k_prod(float* x, const float* in, int n, unsigned long long tail) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = in[i] * 2.0f;
    cudaTriggerProgrammaticLaunchCompletion();
    spin_cycles(tail);
}

// middle node: prologue -> wait(upstream) -> compute y=x+1 -> trigger(downstream) -> tail
__global__ void k_mid(float* y, const float* x, int n,
                      unsigned long long prologue, unsigned long long tail) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    spin_cycles(prologue);
    cudaGridDependencySynchronize();                 // gated by upstream (producer)
    if (i < n) y[i] = x[i] + 1.0f;
    cudaTriggerProgrammaticLaunchCompletion();        // signal downstream (final)
    spin_cycles(tail);
}

// final node: prologue -> wait(both mids) -> out = yA + yB
__global__ void k_fin(float* out, const float* yA, const float* yB, int n,
                      unsigned long long prologue) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    spin_cycles(prologue);
    cudaGridDependencySynchronize();                 // gated by BOTH mids (two programmatic deps)
    if (i < n) out[i] = yA[i] + yB[i];
}

struct Ctx {
    float *in, *x, *yA, *yB, *out;
    int n;
    dim3 grid, block;
    unsigned long long tail, prologue;
    cudaStream_t s;
    cudaEvent_t evStart, evStop;
};

// Build the diamond. programmatic=true -> PDL edges; false -> ordinary edges.
static cudaGraphExec_t build_diamond(Ctx& c, bool programmatic) {
    cudaGraph_t g; CUDA_CHECK(cudaGraphCreate(&g, 0));

    void* pArgs[]  = { (void*)&c.x,  (void*)&c.in, (void*)&c.n, (void*)&c.tail };
    void* aArgs[]  = { (void*)&c.yA, (void*)&c.x,  (void*)&c.n, (void*)&c.prologue, (void*)&c.tail };
    void* bArgs[]  = { (void*)&c.yB, (void*)&c.x,  (void*)&c.n, (void*)&c.prologue, (void*)&c.tail };
    void* fArgs[]  = { (void*)&c.out,(void*)&c.yA, (void*)&c.yB,(void*)&c.n, (void*)&c.prologue };

    auto mkNode = [&](void* func, void** args) {
        cudaKernelNodeParams np{};
        np.func = func; np.gridDim = c.grid; np.blockDim = c.block;
        np.sharedMemBytes = 0; np.kernelParams = args; np.extra = nullptr;
        cudaGraphNode_t node; CUDA_CHECK(cudaGraphAddKernelNode(&node, g, nullptr, 0, &np));
        return node;
    };
    cudaGraphNode_t P = mkNode((void*)k_prod, pArgs);
    cudaGraphNode_t A = mkNode((void*)k_mid,  aArgs);
    cudaGraphNode_t B = mkNode((void*)k_mid,  bArgs);
    cudaGraphNode_t F = mkNode((void*)k_fin,  fArgs);

    cudaGraphEdgeData e{};                              // zero-init == ordinary edge (Default,port0)
    if (programmatic) {
        e.type = cudaGraphDependencyTypeProgrammatic;
        e.from_port = cudaGraphKernelNodePortProgrammatic;
    }
    CUDA_CHECK(cudaGraphAddDependencies(g, &P, &A, &e, 1));
    CUDA_CHECK(cudaGraphAddDependencies(g, &P, &B, &e, 1));
    CUDA_CHECK(cudaGraphAddDependencies(g, &A, &F, &e, 1));
    CUDA_CHECK(cudaGraphAddDependencies(g, &B, &F, &e, 1));

    cudaGraphExec_t exec; CUDA_CHECK(cudaGraphInstantiate(&exec, g, 0));
    CUDA_CHECK(cudaGraphDestroy(g));
    return exec;
}

static float time_graph(cudaGraphExec_t exec, const Ctx& c) {
    CUDA_CHECK(cudaEventRecord(c.evStart, c.s));
    CUDA_CHECK(cudaGraphLaunch(exec, c.s));
    CUDA_CHECK(cudaEventRecord(c.evStop, c.s));
    CUDA_CHECK(cudaEventSynchronize(c.evStop));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, c.evStart, c.evStop));
    return ms;
}

static bool verify(const Ctx& c, std::vector<float>& hostIn) {
    std::vector<float> h(c.n);
    CUDA_CHECK(cudaMemcpy(h.data(), c.out, c.n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < c.n; ++i) {
        float expect = 4.0f * hostIn[i] + 2.0f;
        if (fabsf(h[i] - expect) > 1e-3f) {
            fprintf(stderr, "  verify FAIL at %d: got %f expect %f\n", i, h[i], expect);
            return false;
        }
    }
    return true;
}

static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.f : v[v.size()/2];
}

int main(int argc, char** argv) {
    long long repeats = 50, tail = 20000000, prologue = -1, blocks = -1, threads = 128;
    for (int i = 1; i < argc; ++i) {
        auto next = [&](long long& dst){ if (i+1<argc) dst = atoll(argv[++i]); };
        if      (!strcmp(argv[i], "--repeats"))  next(repeats);
        else if (!strcmp(argv[i], "--tail"))     next(tail);
        else if (!strcmp(argv[i], "--prologue")) next(prologue);
        else if (!strcmp(argv[i], "--blocks"))   next(blocks);
        else if (!strcmp(argv[i], "--threads"))  next(threads);
        else if (!strcmp(argv[i], "--help")) {
            printf("usage: %s [--repeats N] [--tail CYC] [--prologue CYC] [--blocks B] [--threads T]\n",
                   argv[0]);
            return 0;
        }
    }
    if (prologue < 0) prologue = tail;

    int dev = 0; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int clockKHz = 0; cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, dev);
    if (blocks  < 0) blocks  = prop.multiProcessorCount;
    if (threads < 0) threads = 128;

    Ctx c{};
    c.grid = dim3((unsigned)blocks); c.block = dim3((unsigned)threads);
    c.n = (int)(blocks * threads);
    c.tail = (unsigned long long)tail; c.prologue = (unsigned long long)prologue;
    double ghz = clockKHz / 1e6;
    double t_ms = ghz > 0 ? (double)tail / (ghz * 1e6) : 0.0;

    printf("Device: %s  | SMs=%d  | SM clock~%.2f GHz  | CC %d.%d\n",
           prop.name, prop.multiProcessorCount, ghz, prop.major, prop.minor);
    printf("Diamond: producer -> {midA, midB} -> final  | blocks=%lld threads=%lld (n=%d)\n"
           "         tail=prologue=%lld cyc (~%.2f ms each) | repeats=%lld\n\n",
           blocks, threads, c.n, tail, t_ms, repeats);
    if (prop.major < 9) printf("WARNING: PDL needs CC >= 9.0; this device is %d.%d\n\n", prop.major, prop.minor);

    std::vector<float> hostIn(c.n);
    for (int i = 0; i < c.n; ++i) hostIn[i] = (float)(i % 97) * 0.5f + 1.0f;
    CUDA_CHECK(cudaMalloc(&c.in,  c.n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c.x,   c.n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c.yA,  c.n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c.yB,  c.n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c.out, c.n*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(c.in, hostIn.data(), c.n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamCreateWithFlags(&c.s, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&c.evStart));
    CUDA_CHECK(cudaEventCreate(&c.evStop));

    double med[2] = {0,0}, best[2] = {0,0}; bool ok[2] = {true,true};
    const char* names[2] = { "DIAMOND_PLAIN(ordinary edges)", "DIAMOND_PDL(programmatic edges)" };
    for (int variant = 0; variant < 2; ++variant) {
        bool programmatic = (variant == 1);
        cudaGraphExec_t exec = build_diamond(c, programmatic);
        for (int w = 0; w < 5; ++w) (void)time_graph(exec, c);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> t;
        for (long long r = 0; r < repeats; ++r) { t.push_back(time_graph(exec, c)); CUDA_CHECK(cudaDeviceSynchronize()); }
        med[variant]  = median(t);
        best[variant] = *std::min_element(t.begin(), t.end());
        ok[variant]   = verify(c, hostIn);
        CUDA_CHECK(cudaGraphExecDestroy(exec));
    }

    printf("%-34s  %10s  %10s  %s\n", "variant", "median(ms)", "min(ms)", "correct");
    for (int v = 0; v < 2; ++v)
        printf("%-34s  %10.3f  %10.3f  %s\n", names[v], med[v], best[v], ok[v]?"PASS":"FAIL");

    double sp = med[1] > 0 ? med[0]/med[1] : 0.0;
    printf("\nSUMMARY tail=%lld prologue=%lld blocks=%lld threads=%lld | "
           "PLAIN=%.3f PDL=%.3f ms | pdl_speedup=%.2f\n",
           tail, prologue, blocks, threads, med[0], med[1], sp);
    printf("VERDICT: on the diamond, PDL edges %s (%.2fx vs ordinary edges).%s\n",
           (sp >= 1.10 ? "HELP" : (sp >= 1.02 ? "help marginally" : "give NO measurable benefit")),
           sp, ((ok[0]&&ok[1]) ? "" : "  [WARNING: a result was INCORRECT]"));

    cudaFree(c.in); cudaFree(c.x); cudaFree(c.yA); cudaFree(c.yB); cudaFree(c.out);
    return 0;
}
