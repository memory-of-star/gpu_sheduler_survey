// bench_util.cuh — shared host-side helpers for the CTA-PDL benchmarks.
//
// Conventions inherited from 跨stream_PDL调研/bench/pdl_bench:
//   - CUDA_CHECK on every API call
//   - every benchmark prints a single machine-parsable "SUMMARY key=value ..." line so the
//     offline analysis scripts never have to parse prose
//   - median over repeats is the headline number; min is reported for reference

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); \
    exit(1);} } while(0)

// Burn a fixed number of SM clock cycles. Wall-time proportional and independent of ALU
// throughput, so tail/prologue lengths stay comparable across configs.
// NOTE: clock64() is correct HERE (duration on one SM); it is NOT usable for cross-SM
// timeline reconstruction -- that is what %globaltimer in cta_trace.cuh is for.
__device__ __forceinline__ void spin_cycles(unsigned long long cyc) {
    if (cyc == 0) return;
    long long start = clock64();
    while ((unsigned long long)(clock64() - start) < cyc) {
        asm volatile("" ::: "memory");
    }
}

struct DeviceInfo {
    int   dev = 0;
    int   sms = 0;
    int   major = 0, minor = 0;
    double ghz = 0.0;
    char  name[256] = {0};
};

inline DeviceInfo queryDevice() {
    DeviceInfo d;
    CUDA_CHECK(cudaGetDevice(&d.dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, d.dev));
    int clockKHz = 0;                                  // cudaDeviceProp::clockRate gone in CUDA 13
    cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, d.dev);
    d.sms = prop.multiProcessorCount;
    d.major = prop.major; d.minor = prop.minor;
    d.ghz = clockKHz / 1e6;
    snprintf(d.name, sizeof(d.name), "%s", prop.name);
    return d;
}

inline void printDeviceBanner(const DeviceInfo& d) {
    printf("Device: %s | SMs=%d | SM clock~%.2f GHz | CC %d.%d\n",
           d.name, d.sms, d.ghz, d.major, d.minor);
    if (d.major < 9)
        printf("WARNING: PDL requires compute capability >= 9.0; this device is %d.%d\n",
               d.major, d.minor);
}

inline double medianOf(std::vector<float> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

inline double minOf(const std::vector<float>& v) {
    return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end());
}

// Tiny argv parser: --key value
struct Args {
    int argc; char** argv;
    Args(int c, char** v) : argc(c), argv(v) {}
    bool has(const char* k) const {
        for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], k)) return true;
        return false;
    }
    long long ll(const char* k, long long dflt) const {
        for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], k)) return atoll(argv[i+1]);
        return dflt;
    }
    const char* str(const char* k, const char* dflt) const {
        for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], k)) return argv[i+1];
        return dflt;
    }
};

// Launch-config knobs used to sweep the occupancy-cost curve of dimension B2.
struct ResourceKnobs {
    int  smem_kb   = 0;    // dynamic shared memory per CTA (KB)
    int  threads   = 128;
};

inline size_t smemBytes(const ResourceKnobs& k) { return (size_t)k.smem_kb * 1024; }

// Achieved-occupancy proxy: how many CTAs of this kernel actually fit per SM.
template <typename KernelT>
inline int ctasPerSM(KernelT kernel, const ResourceKnobs& k) {
    int n = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, (const void*)kernel, k.threads, smemBytes(k));
    return n;
}
