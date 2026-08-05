// dsa_native.cu -- admissible native Tier-5 DSA dependency-chain benchmark.
//
// This is a shape-faithful tiled dependency benchmark, not a production DSA kernel.  It
// executes every (query-block,key-tile) score element for query_block=64/key_tile=128,
// materialises the compact score matrix and top-k tile indices, and consumes every selected
// index in a sparse-attention proxy.  Scores are deliberately monotonic, so selection is an
// analytical top-k proxy after a mandatory full-row scan; this is not a production top-k
// algorithm and its absolute latency must not be presented as one.  Short contexts retain one physical producer CTA per
// key tile.  Long contexts pack a contiguous interval of logical key tiles into each producer
// CTA so the 1M point remains runnable; both logical and physical degree are reported.
//
// Bracket paths are adjacent in one process.  `floor` preserves the deployed full-grid
// programmatic-graph baseline as one graph launch.  `wave_floor`, `impl`, and `ceiling`
// execute identical bounded query waves; this extra Floor control exposes the scheduling cost
// instead of mislabelling it CTA protocol headroom.  A wave admits at most
// floor((SMs-1)/3) queries, so its two consumer grids can occupy fewer than
// two thirds of the SMs even in the deliberately pessimistic case where no consumer CTAs
// co-reside.  At least ceil((SMs+2)/3) completely consumer-free SMs therefore remain for the
// producer grid.  This is a forward-progress invariant, not an assumption about cross-stream
// co-residency or block scheduling order.
//   floor   : one full-grid indexer -> topk -> attention programmatic CUDA Graph.
//   wave_floor: the same graph protocol and full work partitioned into bounded query waves.
//   impl    : three independent priority streams per wave.  Each topk CTA acquire-waits the
//             exact contiguous row of producer epoch flags; attention waits its topk flag.
//   ceiling : the exact same three streams, per-stage priorities, consumers-first launch order,
//             and synchronization order as Impl, but no readiness publication or wait.  Output
//             is deliberately stale/incorrect and is never correctness-validated.
//
// All intermediate/output buffers are poisoned for every invocation.  Separate untimed
// Floor/Impl invocations run device-side full-element reference checkers.  Cross-SM timing is
// reconstructed exclusively from %globaltimer records.

#include "common/bench_util.cuh"
#include "common/cta_trace.cuh"

#include <cuda/atomic>

#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define DSA_HAS_NVTX 1
#else
#define DSA_HAS_NVTX 0
#endif

namespace {

constexpr int kQueryBlock = 64;
constexpr int kKeyTile = 128;
constexpr int kIndexTopk = 2048;
constexpr int kAttentionLanes = 64;
constexpr int kIndexerThreads = 128;
constexpr int kWorkerThreads = 256;
constexpr int kPairKeyRegisterTile = 8;
constexpr size_t kTopkSmem = 64u * 1024u;
constexpr size_t kAttentionSmem = 128u * 1024u;
constexpr int kModes = 4;
constexpr int kBootstrap = 2000;
constexpr int kConsumerTopk = 0;
constexpr int kConsumerAttention = 1;
constexpr int kConsumerKinds = 2;
constexpr int kEntryGateTimeoutMs = 5000;

// Keep Impl=1 and Ceiling=2 stable: the PTX proof binds those selectors directly.
enum DsaMode : int {
    DSA_FLOOR = 0, DSA_IMPL = 1, DSA_CEILING = 2, DSA_WAVE_FLOOR = 3
};
enum DsaStage : int { DSA_INDEXER = 0, DSA_TOPK = 1, DSA_ATTENTION = 2 };

const char* modeName(int mode) {
    switch (mode) {
        case DSA_FLOOR: return "floor";
        case DSA_IMPL: return "impl";
        case DSA_CEILING: return "ceiling";
        case DSA_WAVE_FLOOR: return "wave_floor";
        default: return "unknown";
    }
}

__host__ __device__ __forceinline__ bool isPdlFloor(int mode) {
    return mode == DSA_FLOOR || mode == DSA_WAVE_FLOOR;
}

const char* stageName(int stage) {
    switch (stage) {
        case DSA_INDEXER: return "indexer";
        case DSA_TOPK: return "topk";
        case DSA_ATTENTION: return "attention";
        default: return "unknown";
    }
}

struct NvtxRange {
    explicit NvtxRange(const char* name) {
#if DSA_HAS_NVTX
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }
    ~NvtxRange() {
#if DSA_HAS_NVTX
        nvtxRangePop();
#endif
    }
};

struct DsaConfig {
    int seq = 4096;
    int query_blocks = 0;
    int key_tiles = 0;
    int logical_degree = 0;
    int physical_degree = 0;
    int topk = 0;
    int pair_query = kQueryBlock;
    int pair_key = kKeyTile;
    int producer_ctas = 0;
    int wave_queries = 0;
    int wave_count = 0;
    int reserved_producer_sms = 0;
    int repeats = 31;
    int warmup = 3;
    unsigned long long indexer_ready = 180000ull;
    unsigned long long indexer_tail = 180000ull;
    unsigned long long topk_prologue = 90000ull;
    unsigned long long topk_tail = 180000ull;
    unsigned long long attention_prologue = 90000ull;
    unsigned long long attention_tail = 90000ull;
    const char* tag = "dsa";
    const char* trace_path = nullptr;
    bool allow_short = false;
    bool dry_run = false;
};

struct DsaTrace {
    unsigned long long t_start;
    unsigned long long t_dep;
    unsigned long long t_ready;
    unsigned long long t_trigger;
    unsigned long long t_end;
    unsigned int epoch;
    unsigned int block;
    unsigned short stage;
    unsigned short sm;
};
static_assert(sizeof(DsaTrace) == 56, "DsaTrace layout changed");

__host__ __device__ __forceinline__ unsigned int mix32(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__host__ __device__ __forceinline__ unsigned int pairQueryValue(unsigned int position) {
    return position * 3u + 1u;
}

__host__ __device__ __forceinline__ unsigned int pairKeyValue(unsigned int position) {
    return position * 5u + 1u;
}

__host__ __device__ __forceinline__ unsigned long long expectedPairAccum(
        unsigned int epoch, unsigned int query, unsigned int key_tile,
        unsigned int pair_query, unsigned int pair_key) {
    const unsigned long long qsum = 3ull * pair_query * (pair_query - 1ull) / 2ull
                                  + pair_query;
    const unsigned long long ksum = 5ull * pair_key * (pair_key - 1ull) / 2ull
                                  + pair_key;
    const unsigned long long base = epoch * 17ull + query * 131ull + key_tile * 7ull;
    return (unsigned long long)pair_key * qsum
         + (unsigned long long)pair_query * ksum
         + (unsigned long long)pair_query * pair_key * base;
}

__host__ __device__ __forceinline__ unsigned int scoreValue(
        unsigned int epoch, unsigned int query, unsigned int key_tile,
        unsigned int pair_query, unsigned int pair_key) {
    const unsigned int low = (unsigned int)expectedPairAccum(
        epoch, query, key_tile, pair_query, pair_key) & 65535u;
    return (key_tile + 1u) * 65536u + low;
}

__host__ __device__ __forceinline__ unsigned long long expectedScoreRowSum(
        unsigned int epoch, unsigned int query, unsigned int key_tiles,
        unsigned int pair_query, unsigned int pair_key) {
    unsigned long long sum = 0;
    for (unsigned int key = 0; key < key_tiles; ++key)
        sum += scoreValue(epoch, query, key, pair_query, pair_key);
    return sum;
}

__device__ __forceinline__ unsigned int executePairWork(
        const unsigned int* query_lut,
        const unsigned int* key_lut,
        unsigned int epoch, unsigned int query, unsigned int key_tile,
        int pair_query, int pair_key) {
    const unsigned int base = epoch * 17u + query * 131u + key_tile * 7u;
    // The score consumes only the low 16 bits.  Reducing modulo 2^32 is therefore
    // bit-identical to the former uint64 sum in that observable low-16 projection:
    // (S mod 2^32) mod 2^16 == S mod 2^16.  The inline PTX add is deliberately kept
    // inside the dynamic q/k loops so every pair still executes one explicit addition.
    unsigned int accum = 0;
#pragma unroll 1
    for (int q = 0; q < pair_query; ++q) {
        const unsigned int query_value = query_lut[q];
#pragma unroll 1
        for (int key_base = 0; key_base < pair_key;
             key_base += kPairKeyRegisterTile) {
            unsigned int key_register_tile[kPairKeyRegisterTile];
#pragma unroll
            for (int lane = 0; lane < kPairKeyRegisterTile; ++lane) {
                const int key = key_base + lane;
                key_register_tile[lane] = key < pair_key ? key_lut[key] : 0u;
            }
#pragma unroll
            for (int lane = 0; lane < kPairKeyRegisterTile; ++lane) {
                if (key_base + lane < pair_key) {
                    const unsigned int term = query_value + key_register_tile[lane] + base;
                    asm volatile("{\n\t"
                                 ".reg .u32 dsa_pair_term;\n\t"
                                 "mov.u32 dsa_pair_term, %1;\n\t"
                                 "add.u32 %0, %0, dsa_pair_term;\n\t"
                                 "}\n"
                                 : "+r"(accum) : "r"(term));
                }
            }
        }
    }
    // Monotonic high bits preserve the analytical top-k ordering.  Low bits are the actual
    // 64x128 pair-work reduction and are checked independently by the reference kernel.
    return (key_tile + 1u) * 65536u + ((unsigned int)accum & 65535u);
}

__host__ __device__ __forceinline__ unsigned int topkIndex(unsigned int key_tiles,
                                                            unsigned int rank) {
    return key_tiles - 1u - rank;
}

__host__ __device__ __forceinline__ unsigned int historyValue(unsigned int key_tile) {
    return mix32(key_tile ^ 0x6a09e667u);
}

__host__ __device__ __forceinline__ unsigned long long historyContribution(
        unsigned int key_tile, unsigned int value) {
    return (unsigned long long)value + (unsigned long long)key_tile * 65537ull;
}

__host__ __device__ __forceinline__ unsigned int attentionValue(
        unsigned int epoch, unsigned int query, unsigned int lane,
        unsigned long long aggregate) {
    return mix32((unsigned int)aggregate ^ (unsigned int)(aggregate >> 32)
                 ^ epoch * 0x9e3779b9u ^ query * 0x85ebca6bu
                 ^ lane * 0xc2b2ae35u);
}

__device__ __forceinline__ void waitEpoch(const unsigned int* flags, int index,
                                           unsigned int epoch) {
    cuda::atomic_ref<const unsigned int, cuda::thread_scope_device> flag(flags[index]);
    unsigned int ns = 32;
    while (flag.load(cuda::memory_order_acquire) != epoch) {
        __nanosleep(ns);
        ns = ns < 1024 ? ns * 2 : 1024;
    }
}

__device__ __forceinline__ void publishEpoch(unsigned int* flags, int index,
                                              unsigned int epoch) {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> flag(flags[index]);
    flag.store(epoch, cuda::memory_order_release);
}

__device__ __forceinline__ void publishConsumerEntry(unsigned int* counter) {
    // Predication keeps this marker on the existing straight-line dependency path.  It does
    // not add a control-flow edge that could bypass or obscure the audited acquire/wait CFG.
    const unsigned int tid = threadIdx.x;
    asm volatile("{\n\t"
                 ".reg .pred dsa_entry_thread0;\n\t"
                 ".reg .u32 dsa_entry_old;\n\t"
                 "setp.eq.u32 dsa_entry_thread0, %1, 0;\n\t"
                 "@dsa_entry_thread0 atom.global.sys.add.u32 "
                 "dsa_entry_old, [%0], 1;\n\t"
                 "}\n"
                 : : "l"(counter), "r"(tid) : "memory");
}

__global__ void dsaIndexer(unsigned int* __restrict__ scores,
                           const unsigned int* __restrict__ query_lut,
                           const unsigned int* __restrict__ key_lut,
                           unsigned int* __restrict__ producer_flags,
                           const unsigned int* __restrict__ epoch_ptr,
                           DsaTrace* __restrict__ trace,
                           int query_blocks, int key_tiles, int physical_degree,
                           int pair_query, int pair_key,
                           int mode, unsigned long long ready_cycles,
                           unsigned long long tail_cycles, int query_base) {
    __shared__ unsigned int query_cache[kQueryBlock];
    __shared__ unsigned int key_cache[kKeyTile];
    const int local_producer = (int)blockIdx.x;
    const int local_query = local_producer / physical_degree;
    const int query = query_base + local_query;
    const int pack = local_producer - local_query * physical_degree;
    const int producer = query * physical_degree + pack;
    if (query >= query_blocks) return;
    const int lo = (int)(((long long)pack * key_tiles) / physical_degree);
    const int hi = (int)(((long long)(pack + 1) * key_tiles) / physical_degree);
    const unsigned int epoch = *epoch_ptr;
    DsaTrace rec{};
    if (threadIdx.x == 0) {
        rec.t_start = ctatrace_globaltimer();
        rec.epoch = epoch;
        rec.block = (unsigned int)producer;
        rec.stage = DSA_INDEXER;
        rec.sm = (unsigned short)ctatrace_smid();
    }
    __syncthreads();

    if (mode != DSA_FLOOR) {
        if (threadIdx.x == 0) rec.t_trigger = ctatrace_globaltimer();
        cudaTriggerProgrammaticLaunchCompletion();
    }
    if (threadIdx.x == 0) rec.t_dep = ctatrace_globaltimer();

    // Each producer CTA stages the two small LUTs exactly once.  This work remains inside
    // the measured worker interval; only redundant per-pair global loads are removed.
    if ((int)threadIdx.x < pair_query)
        query_cache[threadIdx.x] = query_lut[threadIdx.x];
    if ((int)threadIdx.x < pair_key)
        key_cache[threadIdx.x] = key_lut[threadIdx.x];
    __syncthreads();

    // Shape-independent arithmetic latency before this producer chunk becomes readable.
    // The same work is present in all bracket paths; it also makes the no-wait Ceiling
    // observably stale without inserting an ordering edge.
    spin_cycles(ready_cycles);

    const size_t row = (size_t)query * (size_t)key_tiles;
    for (int key = lo + (int)threadIdx.x; key < hi; key += (int)blockDim.x) {
        scores[row + (size_t)key] = executePairWork(
            query_cache, key_cache,
            epoch, (unsigned int)query, (unsigned int)key, pair_query, pair_key);
    }
    // Every writer, not only the publishing thread, completes a device-scope fence before
    // the block barrier.  Thread 0's later release-store therefore represents all chunk data.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) rec.t_ready = ctatrace_globaltimer();
    __syncthreads();

    if (mode == DSA_IMPL && threadIdx.x == 0) {
        publishEpoch(producer_flags, producer, epoch);
    }
    if (mode == DSA_FLOOR) {
        if (threadIdx.x == 0) {
            rec.t_trigger = ctatrace_globaltimer();
        }
        __syncthreads();
        cudaTriggerProgrammaticLaunchCompletion();
    }

    spin_cycles(tail_cycles);
    __syncthreads();
    if (threadIdx.x == 0) {
        rec.t_end = ctatrace_globaltimer();
        trace[producer] = rec;
    }
}

__global__ void dsaTopk(const unsigned int* __restrict__ scores,
                        unsigned int* __restrict__ indices,
                        unsigned long long* __restrict__ row_sums,
                        const unsigned long long* __restrict__ expected_row_sums,
                        const unsigned int* __restrict__ producer_flags,
                        unsigned int* __restrict__ topk_flags,
                        const unsigned int* __restrict__ epoch_ptr,
                        DsaTrace* __restrict__ trace,
                        int query_blocks, int key_tiles, int physical_degree, int topk,
                        int mode, unsigned long long prologue_cycles,
                        unsigned long long tail_cycles, int query_base,
                        unsigned int* __restrict__ consumer_entries,
                        unsigned int* __restrict__ consumer_completions) {
    extern __shared__ unsigned long long scratch[];
    const int query = query_base + (int)blockIdx.x;
    if (query >= query_blocks) return;
    const unsigned int epoch = *epoch_ptr;
    DsaTrace rec{};
    if (threadIdx.x == 0) {
        rec.t_start = ctatrace_globaltimer();
        rec.epoch = epoch;
        rec.block = (unsigned int)query;
        rec.stage = DSA_TOPK;
        rec.sm = (unsigned short)ctatrace_smid();
    }
    __syncthreads();
    if (mode != DSA_FLOOR) {
        if (threadIdx.x == 0) rec.t_trigger = ctatrace_globaltimer();
        cudaTriggerProgrammaticLaunchCompletion();
    }

    spin_cycles(prologue_cycles);
    publishConsumerEntry(consumer_entries + kConsumerTopk);
    if (threadIdx.x == 0) {
        if (mode == DSA_FLOOR) {
            cudaGridDependencySynchronize();
        } else if (mode == DSA_IMPL) {
            const int base = query * physical_degree;
            for (int p = 0; p < physical_degree; ++p)
                waitEpoch(producer_flags, base + p, epoch);
        }
        rec.t_dep = ctatrace_globaltimer();
    }
    __syncthreads();

    unsigned long long local = 0;
    const size_t row = (size_t)query * (size_t)key_tiles;
    for (int key = (int)threadIdx.x; key < key_tiles; key += (int)blockDim.x)
        local += scores[row + (size_t)key];
    scratch[threadIdx.x] = local;
    __syncthreads();
    for (int offset = (int)blockDim.x / 2; offset > 0; offset >>= 1) {
        if ((int)threadIdx.x < offset) scratch[threadIdx.x] += scratch[threadIdx.x + offset];
        __syncthreads();
    }
    // The expected scalar is prepared before this invocation's timed interval.  Keeping the
    // comparison here preserves the score->index RAW edge (and makes the no-wait Ceiling
    // produce invalid indices) without running a reference loop in the measured top-k CTA.
    const bool valid_scores = scratch[0] == expected_row_sums[query];
    if (threadIdx.x == 0) row_sums[query] = scratch[0];
    const size_t index_row = (size_t)query * (size_t)topk;
    for (int rank = (int)threadIdx.x; rank < topk; rank += (int)blockDim.x) {
        indices[index_row + (size_t)rank] = valid_scores
            ? topkIndex((unsigned int)key_tiles, (unsigned int)rank)
            : 0xffffffffu;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) rec.t_ready = ctatrace_globaltimer();
    __syncthreads();

    if (mode == DSA_IMPL && threadIdx.x == 0) {
        publishEpoch(topk_flags, query, epoch);
    }
    if (mode == DSA_FLOOR) {
        if (threadIdx.x == 0) {
            rec.t_trigger = ctatrace_globaltimer();
        }
        __syncthreads();
        cudaTriggerProgrammaticLaunchCompletion();
    }

    spin_cycles(tail_cycles);
    __syncthreads();
    if (threadIdx.x == 0) {
        rec.t_end = ctatrace_globaltimer();
        trace[(size_t)query_blocks * physical_degree + (size_t)query] = rec;
        __threadfence_system();
        atomicAdd_system(consumer_completions + kConsumerTopk, 1u);
    }
}

__global__ void dsaAttention(const unsigned int* __restrict__ indices,
                             const unsigned int* __restrict__ history,
                             unsigned int* __restrict__ output,
                             const unsigned int* __restrict__ topk_flags,
                             unsigned int* __restrict__ stale_rows,
                             unsigned long long* __restrict__ history_load_counter,
                             const unsigned int* __restrict__ epoch_ptr,
                             DsaTrace* __restrict__ trace,
                             unsigned long long expected_history_sum,
                             int query_blocks, int key_tiles, int physical_degree, int topk,
                             int mode, unsigned long long prologue_cycles,
                             unsigned long long tail_cycles, int query_base,
                             unsigned int* __restrict__ consumer_entries,
                             unsigned int* __restrict__ consumer_completions) {
    extern __shared__ unsigned long long scratch[];
    const int query = query_base + (int)blockIdx.x;
    if (query >= query_blocks) return;
    const unsigned int epoch = *epoch_ptr;
    DsaTrace rec{};
    if (threadIdx.x == 0) {
        rec.t_start = ctatrace_globaltimer();
        rec.epoch = epoch;
        rec.block = (unsigned int)query;
        rec.stage = DSA_ATTENTION;
        rec.sm = (unsigned short)ctatrace_smid();
    }
    __syncthreads();
    if (mode != DSA_FLOOR) {
        if (threadIdx.x == 0) rec.t_trigger = ctatrace_globaltimer();
        cudaTriggerProgrammaticLaunchCompletion();
    }

    spin_cycles(prologue_cycles);
    publishConsumerEntry(consumer_entries + kConsumerAttention);
    if (threadIdx.x == 0) {
        if (mode == DSA_FLOOR)
            cudaGridDependencySynchronize();
        else if (mode == DSA_IMPL)
            waitEpoch(topk_flags, query, epoch);
        rec.t_dep = ctatrace_globaltimer();
    }
    __syncthreads();

    unsigned long long local = 0;
    unsigned long long local_history_loads = 0;
    const size_t row = (size_t)query * (size_t)topk;
    // Keep one auditable static PTX site.  The loop still performs one history load for
    // every dynamic rank; disabling unrolling only removes duplicated compiler code paths.
#pragma unroll 1
    for (int rank = (int)threadIdx.x; rank < topk; rank += (int)blockDim.x) {
        const unsigned int index = indices[row + (size_t)rank];
        const unsigned int valid_mask = 0u - (unsigned int)(index < (unsigned int)key_tiles);
        const unsigned int fallback = mix32(index ^ (unsigned int)rank)
                                    % (unsigned int)key_tiles;
        const unsigned int safe_index = (index & valid_mask) | (fallback & ~valid_mask);
        // Every rank in every bracket path performs exactly one in-bounds history load.
        // Only after that load do we propagate an invalid semantic index for stale Ceiling
        // data, preserving wrongness without changing the memory-work shape.
        const unsigned int raw_value = history[safe_index];
        unsigned int value;
        asm volatile("{\n\t"
                     ".reg .u32 dsa_history_loaded_value;\n\t"
                     "mov.u32 dsa_history_loaded_value, %1;\n\t"
                     "mov.u32 %0, dsa_history_loaded_value;\n\t"
                     "}\n"
                     : "=r"(value) : "r"(raw_value) : "memory");
        asm volatile("{\n\t"
                     ".reg .u32 dsa_history_count_dependency;\n\t"
                     ".reg .u64 dsa_history_load_count;\n\t"
                     "mov.u32 dsa_history_count_dependency, %1;\n\t"
                     "mov.u64 dsa_history_load_count, %0;\n\t"
                     "add.u64 dsa_history_load_count, dsa_history_load_count, 1;\n\t"
                     "mov.u64 %0, dsa_history_load_count;\n\t"
                     "}\n"
                     : "+l"(local_history_loads) : "r"(value) : "memory");
        const unsigned int semantic_candidate = (index & valid_mask)
            | ((safe_index + (unsigned int)key_tiles) & ~valid_mask);
        unsigned int semantic_index;
        asm volatile("{\n\t"
                     ".reg .u32 dsa_semantic_index;\n\t"
                     ".reg .u32 dsa_history_dependency;\n\t"
                     "mov.u32 dsa_history_dependency, %2;\n\t"
                     "mov.u32 dsa_semantic_index, %1;\n\t"
                     "mov.u32 %0, dsa_semantic_index;\n\t"
                     "}\n"
                     : "=r"(semantic_index)
                     : "r"(semantic_candidate), "r"(value) : "memory");
        local += historyContribution(semantic_index, value);
    }
    unsigned long long* history_count_scratch = scratch + blockDim.x;
    scratch[threadIdx.x] = local;
    history_count_scratch[threadIdx.x] = local_history_loads;
    __syncthreads();
    for (int offset = (int)blockDim.x / 2; offset > 0; offset >>= 1) {
        if ((int)threadIdx.x < offset) {
            scratch[threadIdx.x] += scratch[threadIdx.x + offset];
            history_count_scratch[threadIdx.x]
                += history_count_scratch[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (scratch[0] != expected_history_sum) atomicAdd(stale_rows, 1u);
        atomicAdd(history_load_counter, history_count_scratch[0]);
    }
    if ((int)threadIdx.x < kAttentionLanes) {
        const size_t out = (size_t)query * kAttentionLanes + threadIdx.x;
        output[out] = attentionValue(epoch, (unsigned int)query,
                                     (unsigned int)threadIdx.x, scratch[0]);
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) rec.t_ready = ctatrace_globaltimer();
    __syncthreads();
    if (mode == DSA_FLOOR) {
        if (threadIdx.x == 0) rec.t_trigger = ctatrace_globaltimer();
        cudaTriggerProgrammaticLaunchCompletion();
    }
    spin_cycles(tail_cycles);
    __syncthreads();
    if (threadIdx.x == 0) {
        rec.t_end = ctatrace_globaltimer();
        const size_t base = (size_t)query_blocks * physical_degree + query_blocks;
        trace[base + (size_t)query] = rec;
        __threadfence_system();
        atomicAdd_system(consumer_completions + kConsumerAttention, 1u);
    }
}

enum CounterSlot : int {
    C_SCORE_MISMATCH = 0,
    C_INDEX_MISMATCH,
    C_OUTPUT_MISMATCH,
    C_ROW_MISMATCH,
    C_FLAG_MISMATCH,
    C_SCORE_OBS,
    C_SCORE_EXP,
    C_INDEX_OBS,
    C_INDEX_EXP,
    C_OUTPUT_OBS,
    C_OUTPUT_EXP,
    C_ROW_OBS,
    C_ROW_EXP,
    C_FLAG_OBS,
    C_FLAG_EXP,
    C_COUNT
};

__device__ __forceinline__ void reduceTriplet(unsigned long long* smem,
                                               unsigned long long a,
                                               unsigned long long b,
                                               unsigned long long c) {
    unsigned long long* sa = smem;
    unsigned long long* sb = smem + blockDim.x;
    unsigned long long* sc = smem + 2 * blockDim.x;
    sa[threadIdx.x] = a;
    sb[threadIdx.x] = b;
    sc[threadIdx.x] = c;
    __syncthreads();
    for (int offset = (int)blockDim.x / 2; offset > 0; offset >>= 1) {
        if ((int)threadIdx.x < offset) {
            sa[threadIdx.x] += sa[threadIdx.x + offset];
            sb[threadIdx.x] += sb[threadIdx.x + offset];
            sc[threadIdx.x] += sc[threadIdx.x + offset];
        }
        __syncthreads();
    }
}

__global__ void validateScores(const unsigned int* scores,
                               unsigned long long* counters,
                               unsigned int epoch, size_t count, int key_tiles,
                               int pair_query, int pair_key) {
    extern __shared__ unsigned long long smem[];
    unsigned long long mismatch = 0, observed = 0, expected_sum = 0;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        const unsigned int query = (unsigned int)(i / (size_t)key_tiles);
        const unsigned int key = (unsigned int)(i - (size_t)query * key_tiles);
        const unsigned int expected = scoreValue(epoch, query, key,
                                                  (unsigned int)pair_query,
                                                  (unsigned int)pair_key);
        const unsigned int value = scores[i];
        mismatch += value != expected;
        observed += value;
        expected_sum += expected;
    }
    reduceTriplet(smem, mismatch, observed, expected_sum);
    if (threadIdx.x == 0) {
        atomicAdd(counters + C_SCORE_MISMATCH, smem[0]);
        atomicAdd(counters + C_SCORE_OBS, smem[blockDim.x]);
        atomicAdd(counters + C_SCORE_EXP, smem[2 * blockDim.x]);
    }
}

__global__ void validateIndices(const unsigned int* indices,
                                unsigned long long* counters,
                                size_t count, int topk, int key_tiles) {
    extern __shared__ unsigned long long smem[];
    unsigned long long mismatch = 0, observed = 0, expected_sum = 0;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        const unsigned int rank = (unsigned int)(i % (size_t)topk);
        const unsigned int expected = topkIndex((unsigned int)key_tiles, rank);
        const unsigned int value = indices[i];
        mismatch += value != expected;
        observed += value;
        expected_sum += expected;
    }
    reduceTriplet(smem, mismatch, observed, expected_sum);
    if (threadIdx.x == 0) {
        atomicAdd(counters + C_INDEX_MISMATCH, smem[0]);
        atomicAdd(counters + C_INDEX_OBS, smem[blockDim.x]);
        atomicAdd(counters + C_INDEX_EXP, smem[2 * blockDim.x]);
    }
}

__global__ void validateOutput(const unsigned int* output,
                               unsigned long long* counters,
                               unsigned int epoch, size_t count,
                               unsigned long long aggregate) {
    extern __shared__ unsigned long long smem[];
    unsigned long long mismatch = 0, observed = 0, expected_sum = 0;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        const unsigned int query = (unsigned int)(i / kAttentionLanes);
        const unsigned int lane = (unsigned int)(i % kAttentionLanes);
        const unsigned int expected = attentionValue(epoch, query, lane, aggregate);
        const unsigned int value = output[i];
        mismatch += value != expected;
        observed += value;
        expected_sum += expected;
    }
    reduceTriplet(smem, mismatch, observed, expected_sum);
    if (threadIdx.x == 0) {
        atomicAdd(counters + C_OUTPUT_MISMATCH, smem[0]);
        atomicAdd(counters + C_OUTPUT_OBS, smem[blockDim.x]);
        atomicAdd(counters + C_OUTPUT_EXP, smem[2 * blockDim.x]);
    }
}

__global__ void validateRowsAndFlags(const unsigned long long* row_sums,
                                     const unsigned int* producer_flags,
                                     const unsigned int* topk_flags,
                                     unsigned long long* counters,
                                     unsigned int epoch, int mode,
                                     int query_blocks, int key_tiles,
                                     int physical_degree,
                                     int pair_query, int pair_key) {
    extern __shared__ unsigned long long smem[];
    unsigned long long row_mismatch = 0, row_observed = 0, row_expected = 0;
    unsigned long long flag_mismatch = 0, flag_observed = 0, flag_expected = 0;
    for (int query = (int)blockIdx.x * blockDim.x + threadIdx.x;
         query < query_blocks; query += (int)gridDim.x * blockDim.x) {
        const unsigned long long expected_row = expectedScoreRowSum(
            epoch, (unsigned int)query, (unsigned int)key_tiles,
            (unsigned int)pair_query, (unsigned int)pair_key);
        row_mismatch += row_sums[query] != expected_row;
        row_observed += row_sums[query];
        row_expected += expected_row;
        const unsigned int expected_flag = mode == DSA_IMPL ? epoch : 0u;
        const unsigned int tf = topk_flags[query];
        flag_mismatch += tf != expected_flag;
        flag_observed += tf;
        flag_expected += expected_flag;
        const int base = query * physical_degree;
        for (int p = 0; p < physical_degree; ++p) {
            const unsigned int pf = producer_flags[base + p];
            flag_mismatch += pf != expected_flag;
            flag_observed += pf;
            flag_expected += expected_flag;
        }
    }
    reduceTriplet(smem, row_mismatch, row_observed, row_expected);
    unsigned long long row_mis = smem[0];
    unsigned long long row_obs = smem[blockDim.x];
    unsigned long long row_exp = smem[2 * blockDim.x];
    __syncthreads();
    reduceTriplet(smem, flag_mismatch, flag_observed, flag_expected);
    if (threadIdx.x == 0) {
        atomicAdd(counters + C_ROW_MISMATCH, row_mis);
        atomicAdd(counters + C_ROW_OBS, row_obs);
        atomicAdd(counters + C_ROW_EXP, row_exp);
        atomicAdd(counters + C_FLAG_MISMATCH, smem[0]);
        atomicAdd(counters + C_FLAG_OBS, smem[blockDim.x]);
        atomicAdd(counters + C_FLAG_EXP, smem[2 * blockDim.x]);
    }
}

struct DsaContext {
    unsigned int* scores = nullptr;
    unsigned int* query_lut = nullptr;
    unsigned int* key_lut = nullptr;
    unsigned int* indices = nullptr;
    unsigned int* history = nullptr;
    unsigned int* output = nullptr;
    unsigned long long* row_sums = nullptr;
    unsigned long long* expected_row_sums = nullptr;
    unsigned int* producer_flags = nullptr;
    unsigned int* topk_flags = nullptr;
    unsigned int* stale_rows = nullptr;
    unsigned long long* history_load_counter = nullptr;
    unsigned int* consumer_entries_host = nullptr;
    unsigned int* consumer_entries_device = nullptr;
    unsigned int* consumer_completions_host = nullptr;
    unsigned int* consumer_completions_device = nullptr;
    unsigned int* epoch = nullptr;
    unsigned long long* counters = nullptr;
    DsaTrace* trace = nullptr;
    cudaStream_t prep_stream{};
    cudaStream_t graph_stream{};
    cudaStream_t producer_stream{};
    cudaStream_t topk_stream{};
    cudaStream_t attention_stream{};
    int least_priority = 0;
    int greatest_priority = 0;
    int topk_priority = 0;
};

struct DsaGraph {
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    cudaGraphNode_t index_node{};
    cudaGraphNode_t topk_node{};
    cudaGraphNode_t attention_node{};
};

size_t scoreCount(const DsaConfig& c) {
    return (size_t)c.query_blocks * (size_t)c.key_tiles;
}
size_t indexCount(const DsaConfig& c) {
    return (size_t)c.query_blocks * (size_t)c.topk;
}
size_t outputCount(const DsaConfig& c) {
    return (size_t)c.query_blocks * kAttentionLanes;
}
size_t traceCount(const DsaConfig& c) {
    return (size_t)c.producer_ctas + 2ull * (size_t)c.query_blocks;
}

unsigned long long expectedHistoryLoads(const DsaConfig& c) {
    return (unsigned long long)c.query_blocks * (unsigned long long)c.topk;
}

unsigned long long expectedHistorySum(const DsaConfig& c) {
    unsigned long long sum = 0;
    for (int rank = 0; rank < c.topk; ++rank) {
        const unsigned int key = topkIndex((unsigned int)c.key_tiles, (unsigned int)rank);
        sum += historyContribution(key, historyValue(key));
    }
    return sum;
}

DsaGraph buildFloorGraph(const DsaConfig& c, DsaContext& x,
                         unsigned long long history_sum, int graph_queries) {
    DsaGraph out;
    CUDA_CHECK(cudaGraphCreate(&out.graph, 0));
    int floor_mode = DSA_FLOOR;
    int query_base = 0;

    void* index_args[] = {
        &x.scores, &x.query_lut, &x.key_lut, &x.producer_flags, &x.epoch, &x.trace,
        const_cast<int*>(&c.query_blocks), const_cast<int*>(&c.key_tiles),
        const_cast<int*>(&c.physical_degree), const_cast<int*>(&c.pair_query),
        const_cast<int*>(&c.pair_key), &floor_mode,
        const_cast<unsigned long long*>(&c.indexer_ready),
        const_cast<unsigned long long*>(&c.indexer_tail),
        &query_base,
    };
    cudaKernelNodeParams ip{};
    ip.func = (void*)dsaIndexer;
    ip.gridDim = dim3((unsigned)(graph_queries * c.physical_degree));
    ip.blockDim = dim3(kIndexerThreads);
    ip.kernelParams = index_args;
    CUDA_CHECK(cudaGraphAddKernelNode(&out.index_node, out.graph, nullptr, 0, &ip));

    void* topk_args[] = {
        &x.scores, &x.indices, &x.row_sums, &x.expected_row_sums,
        &x.producer_flags, &x.topk_flags,
        &x.epoch, &x.trace, const_cast<int*>(&c.query_blocks),
        const_cast<int*>(&c.key_tiles), const_cast<int*>(&c.physical_degree),
        const_cast<int*>(&c.topk), &floor_mode,
        const_cast<unsigned long long*>(&c.topk_prologue),
        const_cast<unsigned long long*>(&c.topk_tail),
        &query_base, &x.consumer_entries_device, &x.consumer_completions_device,
    };
    cudaKernelNodeParams tp{};
    tp.func = (void*)dsaTopk;
    tp.gridDim = dim3((unsigned)graph_queries);
    tp.blockDim = dim3(kWorkerThreads);
    tp.sharedMemBytes = (unsigned)kTopkSmem;
    tp.kernelParams = topk_args;
    CUDA_CHECK(cudaGraphAddKernelNode(&out.topk_node, out.graph, nullptr, 0, &tp));

    void* attention_args[] = {
        &x.indices, &x.history, &x.output, &x.topk_flags, &x.stale_rows,
        &x.history_load_counter, &x.epoch, &x.trace, &history_sum,
        const_cast<int*>(&c.query_blocks),
        const_cast<int*>(&c.key_tiles), const_cast<int*>(&c.physical_degree),
        const_cast<int*>(&c.topk), &floor_mode,
        const_cast<unsigned long long*>(&c.attention_prologue),
        const_cast<unsigned long long*>(&c.attention_tail),
        &query_base, &x.consumer_entries_device, &x.consumer_completions_device,
    };
    cudaKernelNodeParams ap{};
    ap.func = (void*)dsaAttention;
    ap.gridDim = dim3((unsigned)graph_queries);
    ap.blockDim = dim3(kWorkerThreads);
    ap.sharedMemBytes = (unsigned)kAttentionSmem;
    ap.kernelParams = attention_args;
    CUDA_CHECK(cudaGraphAddKernelNode(&out.attention_node, out.graph, nullptr, 0, &ap));

    cudaGraphEdgeData edge{};
    edge.type = cudaGraphDependencyTypeProgrammatic;
    edge.from_port = cudaGraphKernelNodePortProgrammatic;
    CUDA_CHECK(cudaGraphAddDependencies(
        out.graph, &out.index_node, &out.topk_node, &edge, 1));
    CUDA_CHECK(cudaGraphAddDependencies(
        out.graph, &out.topk_node, &out.attention_node, &edge, 1));
    CUDA_CHECK(cudaGraphInstantiate(&out.exec, out.graph, 0));
    return out;
}

void setFloorWave(const DsaConfig& c, DsaContext& x, DsaGraph& graph,
                  unsigned long long history_sum, int query_base,
                  int wave_query_count) {
    int floor_mode = DSA_FLOOR;
    void* index_args[] = {
        &x.scores, &x.query_lut, &x.key_lut, &x.producer_flags, &x.epoch, &x.trace,
        const_cast<int*>(&c.query_blocks), const_cast<int*>(&c.key_tiles),
        const_cast<int*>(&c.physical_degree), const_cast<int*>(&c.pair_query),
        const_cast<int*>(&c.pair_key), &floor_mode,
        const_cast<unsigned long long*>(&c.indexer_ready),
        const_cast<unsigned long long*>(&c.indexer_tail), &query_base,
    };
    cudaKernelNodeParams ip{};
    ip.func = (void*)dsaIndexer;
    ip.gridDim = dim3((unsigned)(wave_query_count * c.physical_degree));
    ip.blockDim = dim3(kIndexerThreads);
    ip.kernelParams = index_args;
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(graph.exec, graph.index_node, &ip));

    void* topk_args[] = {
        &x.scores, &x.indices, &x.row_sums, &x.expected_row_sums,
        &x.producer_flags, &x.topk_flags, &x.epoch, &x.trace,
        const_cast<int*>(&c.query_blocks), const_cast<int*>(&c.key_tiles),
        const_cast<int*>(&c.physical_degree), const_cast<int*>(&c.topk), &floor_mode,
        const_cast<unsigned long long*>(&c.topk_prologue),
        const_cast<unsigned long long*>(&c.topk_tail), &query_base,
        &x.consumer_entries_device, &x.consumer_completions_device,
    };
    cudaKernelNodeParams tp{};
    tp.func = (void*)dsaTopk;
    tp.gridDim = dim3((unsigned)wave_query_count);
    tp.blockDim = dim3(kWorkerThreads);
    tp.sharedMemBytes = (unsigned)kTopkSmem;
    tp.kernelParams = topk_args;
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(graph.exec, graph.topk_node, &tp));

    void* attention_args[] = {
        &x.indices, &x.history, &x.output, &x.topk_flags, &x.stale_rows,
        &x.history_load_counter, &x.epoch, &x.trace, &history_sum,
        const_cast<int*>(&c.query_blocks), const_cast<int*>(&c.key_tiles),
        const_cast<int*>(&c.physical_degree), const_cast<int*>(&c.topk), &floor_mode,
        const_cast<unsigned long long*>(&c.attention_prologue),
        const_cast<unsigned long long*>(&c.attention_tail), &query_base,
        &x.consumer_entries_device, &x.consumer_completions_device,
    };
    cudaKernelNodeParams ap{};
    ap.func = (void*)dsaAttention;
    ap.gridDim = dim3((unsigned)wave_query_count);
    ap.blockDim = dim3(kWorkerThreads);
    ap.sharedMemBytes = (unsigned)kAttentionSmem;
    ap.kernelParams = attention_args;
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(graph.exec, graph.attention_node, &ap));
}

void destroyGraph(DsaGraph& g) {
    if (g.exec) cudaGraphExecDestroy(g.exec);
    if (g.graph) cudaGraphDestroy(g.graph);
    g.exec = nullptr;
    g.graph = nullptr;
    g.index_node = nullptr;
    g.topk_node = nullptr;
    g.attention_node = nullptr;
}

struct TraceMetrics {
    bool complete = false;
    double makespan_ms = 0.0;
    unsigned int topk_early = 0;
    unsigned int attention_early = 0;
    unsigned int topk_waited = 0;
    unsigned int attention_waited = 0;
    unsigned int safety_failures = 0;
    unsigned int trigger_failures = 0;
    unsigned int progress_waves_verified = 0;
    unsigned int consumer_entry_order_failures = 0;
    unsigned int producer_forward_progress_failures = 0;
};

TraceMetrics analyzeTrace(const DsaConfig& c, int mode, unsigned int epoch,
                          const std::vector<DsaTrace>& rows) {
    TraceMetrics m{};
    if (rows.size() != traceCount(c)) return m;
    const size_t topk_base = (size_t)c.producer_ctas;
    const size_t attention_base = topk_base + c.query_blocks;
    unsigned long long first = ~0ull, last = 0ull;
    bool complete = true;
    for (size_t i = 0; i < rows.size(); ++i) {
        const DsaTrace& r = rows[i];
        int expected_stage = i < topk_base ? DSA_INDEXER
                           : i < attention_base ? DSA_TOPK : DSA_ATTENTION;
        unsigned int expected_block = expected_stage == DSA_INDEXER
            ? (unsigned int)i
            : expected_stage == DSA_TOPK
                ? (unsigned int)(i - topk_base)
                : (unsigned int)(i - attention_base);
        bool ordered = r.epoch == epoch && r.stage == expected_stage
            && r.block == expected_block && r.t_start > 0
            && r.t_start <= r.t_dep && r.t_dep <= r.t_ready
            && r.t_ready <= r.t_end && r.sm < 65535u;
        if (isPdlFloor(mode)) {
            ordered = ordered && r.t_ready <= r.t_trigger && r.t_trigger <= r.t_end;
            if (!(r.t_ready <= r.t_trigger && r.t_trigger < r.t_end))
                ++m.trigger_failures;
        } else {
            ordered = ordered && r.t_start <= r.t_trigger && r.t_trigger <= r.t_dep;
            if (!(r.t_start <= r.t_trigger && r.t_trigger <= r.t_dep))
                ++m.trigger_failures;
        }
        complete = complete && ordered;
        first = std::min(first, r.t_start);
        last = std::max(last, r.t_end);
    }
    if (!complete || first == ~0ull || last <= first) return m;

    const int trace_wave_queries = mode == DSA_FLOOR
        ? c.query_blocks : c.wave_queries;
    const unsigned int expected_trace_waves = mode == DSA_FLOOR
        ? 1u : (unsigned)c.wave_count;
    for (int wave_begin = 0; wave_begin < c.query_blocks;
         wave_begin += trace_wave_queries) {
        const int wave_end = std::min(c.query_blocks, wave_begin + trace_wave_queries);
        unsigned long long latest_consumer_start = 0ull;
        unsigned long long first_consumer_end = ~0ull;
        unsigned long long first_producer_start = ~0ull;
        for (int q = wave_begin; q < wave_end; ++q) {
            const DsaTrace& topk = rows[topk_base + (size_t)q];
            const DsaTrace& attention = rows[attention_base + (size_t)q];
            latest_consumer_start = std::max(
                latest_consumer_start, std::max(topk.t_start, attention.t_start));
            first_consumer_end = std::min(
                first_consumer_end, std::min(topk.t_end, attention.t_end));
            const size_t producer_base = (size_t)q * c.physical_degree;
            for (int p = 0; p < c.physical_degree; ++p)
                first_producer_start = std::min(
                    first_producer_start, rows[producer_base + (size_t)p].t_start);
        }
        bool wave_ok = true;
        if ((mode == DSA_IMPL || mode == DSA_CEILING)
            && latest_consumer_start > first_producer_start) {
            ++m.consumer_entry_order_failures;
            wave_ok = false;
        }
        if (mode == DSA_IMPL
            && first_producer_start >= first_consumer_end) {
            ++m.producer_forward_progress_failures;
            wave_ok = false;
        }
        if (wave_ok) ++m.progress_waves_verified;
    }

    struct DependencyWaveBounds {
        unsigned long long index_trigger = 0ull;
        unsigned long long index_end = 0ull;
        unsigned long long topk_trigger = 0ull;
        unsigned long long topk_end = 0ull;
    };
    const int dependency_wave_queries = mode == DSA_FLOOR
        ? c.query_blocks : c.wave_queries;
    std::vector<DependencyWaveBounds> dependency_bounds;
    if (isPdlFloor(mode)) {
        dependency_bounds.resize((size_t)(
            (c.query_blocks + dependency_wave_queries - 1)
            / dependency_wave_queries));
        for (int wave_begin = 0; wave_begin < c.query_blocks;
             wave_begin += dependency_wave_queries) {
            const int wave_end = std::min(
                c.query_blocks, wave_begin + dependency_wave_queries);
            DependencyWaveBounds& bounds = dependency_bounds[(size_t)(
                wave_begin / dependency_wave_queries)];
            for (int wave_q = wave_begin; wave_q < wave_end; ++wave_q) {
                const size_t producer_base = (size_t)wave_q * c.physical_degree;
                for (int p = 0; p < c.physical_degree; ++p) {
                    const DsaTrace& producer = rows[producer_base + (size_t)p];
                    bounds.index_trigger = std::max(
                        bounds.index_trigger, producer.t_trigger);
                    bounds.index_end = std::max(bounds.index_end, producer.t_end);
                }
                const DsaTrace& wave_topk = rows[topk_base + (size_t)wave_q];
                bounds.topk_trigger = std::max(
                    bounds.topk_trigger, wave_topk.t_trigger);
                bounds.topk_end = std::max(bounds.topk_end, wave_topk.t_end);
            }
        }
    }

    for (int q = 0; q < c.query_blocks; ++q) {
        const DsaTrace& topk = rows[topk_base + (size_t)q];
        const DsaTrace& attention = rows[attention_base + (size_t)q];
        if (isPdlFloor(mode)) {
            const DependencyWaveBounds& bounds = dependency_bounds[(size_t)(
                q / dependency_wave_queries)];
            // Programmatic dependent launch is safe at the upstream trigger, not at
            // upstream kernel completion.  A useful PDL overlap therefore means that the
            // consumer starts before the upstream tail ends, while its dependency point is
            // no earlier than every upstream trigger.  Requiring start < ready is invalid:
            // the scheduler may legally hold a dependent grid until all triggers fire and
            // still overlap the producer tail.
            const bool topk_overlap = topk.t_start < bounds.index_end;
            const bool attention_overlap = attention.t_start < bounds.topk_end;
            const bool topk_dependency_safe =
                topk.t_dep >= bounds.index_trigger;
            const bool attention_dependency_safe =
                attention.t_dep >= bounds.topk_trigger;
            m.topk_early += topk_overlap;
            m.attention_early += attention_overlap;
            m.topk_waited += topk_overlap && topk_dependency_safe;
            m.attention_waited += attention_overlap && attention_dependency_safe;
            m.safety_failures += !topk_dependency_safe;
            m.safety_failures += !attention_dependency_safe;
        } else if (mode == DSA_IMPL) {
            unsigned long long row_ready = 0ull;
            const size_t base = (size_t)q * c.physical_degree;
            for (int p = 0; p < c.physical_degree; ++p)
                row_ready = std::max(row_ready, rows[base + (size_t)p].t_ready);
            m.topk_early += topk.t_start < row_ready;
            m.attention_early += attention.t_start < topk.t_ready;
            m.topk_waited += topk.t_start < row_ready && topk.t_dep >= row_ready;
            m.attention_waited += attention.t_start < topk.t_ready
                               && attention.t_dep >= topk.t_ready;
            m.safety_failures += topk.t_dep < row_ready;
            m.safety_failures += attention.t_dep < topk.t_ready;
        } else {
            // Ceiling has no dependency-safety claim.  Its early-start metrics remain
            // auditable relative to the matching per-row producer/topk readiness points.
            unsigned long long row_ready = 0ull;
            const size_t base = (size_t)q * c.physical_degree;
            for (int p = 0; p < c.physical_degree; ++p)
                row_ready = std::max(row_ready, rows[base + (size_t)p].t_ready);
            m.topk_early += topk.t_start < row_ready;
            m.attention_early += attention.t_start < topk.t_ready;
        }
    }
    m.complete = m.safety_failures == 0 && m.trigger_failures == 0
              && m.consumer_entry_order_failures == 0
              && m.producer_forward_progress_failures == 0
              && m.progress_waves_verified == expected_trace_waves;
    m.makespan_ms = (double)(last - first) / 1.0e6;
    return m;
}

__global__ void poisonDsaBuffers(unsigned int* scores, size_t score_count,
                                 unsigned int* indices, size_t index_count,
                                 unsigned int* output, size_t output_count,
                                 unsigned long long* row_sums, size_t row_count,
                                 unsigned int epoch) {
    const size_t first = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = first; i < score_count; i += stride)
        scores[i] = mix32(epoch ^ (unsigned int)i ^ 0xa5a5a5a5u);
    for (size_t i = first; i < index_count; i += stride)
        indices[i] = 0x80000000u
                   | (mix32(epoch ^ (unsigned int)i ^ 0x3c6ef372u) & 0x7fffffffu);
    for (size_t i = first; i < output_count; i += stride)
        output[i] = mix32(epoch ^ (unsigned int)i ^ 0xbb67ae85u);
    for (size_t i = first; i < row_count; i += stride) {
        const unsigned long long lo = mix32(epoch ^ (unsigned int)i ^ 0x510e527fu);
        const unsigned long long hi = mix32(epoch ^ (unsigned int)i ^ 0x9b05688cu);
        row_sums[i] = lo | (hi << 32);
    }
}

// This reference preparation is intentionally outside every timed rung.  One thread prepares
// one query row's expected reduction; dsaTopk subsequently performs only an O(1) scalar load
// and comparison after its real full score-row scan.
__global__ void prepareExpectedRowSums(unsigned long long* expected_row_sums,
                                       unsigned int epoch, int query_blocks,
                                       int key_tiles, int pair_query, int pair_key) {
    const int first = (int)blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = (int)gridDim.x * blockDim.x;
    for (int query = first; query < query_blocks; query += stride) {
        expected_row_sums[query] = expectedScoreRowSum(
            epoch, (unsigned int)query, (unsigned int)key_tiles,
            (unsigned int)pair_query, (unsigned int)pair_key);
    }
}

void poisonInvocation(const DsaConfig& c, DsaContext& x, unsigned int epoch) {
    NvtxRange range("dsa.poison");
    const size_t largest = std::max(
        std::max(scoreCount(c), indexCount(c)),
        std::max(outputCount(c), (size_t)c.query_blocks));
    const int poison_grid = (int)std::min<size_t>(
        (largest + kWorkerThreads - 1) / kWorkerThreads, 65535u);
    poisonDsaBuffers<<<poison_grid, kWorkerThreads, 0, x.prep_stream>>>(
        x.scores, scoreCount(c), x.indices, indexCount(c), x.output, outputCount(c),
        x.row_sums, (size_t)c.query_blocks, epoch);
    CUDA_CHECK(cudaGetLastError());
    const int expected_grid = std::min(
        (c.query_blocks + kWorkerThreads - 1) / kWorkerThreads, 65535);
    prepareExpectedRowSums<<<expected_grid, kWorkerThreads, 0, x.prep_stream>>>(
        x.expected_row_sums, epoch, c.query_blocks, c.key_tiles,
        c.pair_query, c.pair_key);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemsetAsync(x.producer_flags, 0,
                               (size_t)c.producer_ctas * sizeof(unsigned int),
                               x.prep_stream));
    CUDA_CHECK(cudaMemsetAsync(x.topk_flags, 0,
                               (size_t)c.query_blocks * sizeof(unsigned int),
                               x.prep_stream));
    CUDA_CHECK(cudaMemsetAsync(x.stale_rows, 0, sizeof(unsigned int), x.prep_stream));
    CUDA_CHECK(cudaMemsetAsync(x.history_load_counter, 0,
                               sizeof(unsigned long long), x.prep_stream));
    CUDA_CHECK(cudaMemsetAsync(x.counters, 0,
                               C_COUNT * sizeof(unsigned long long), x.prep_stream));
    CUDA_CHECK(cudaMemsetAsync(x.trace, 0, traceCount(c) * sizeof(DsaTrace), x.prep_stream));
    CUDA_CHECK(cudaMemcpyAsync(x.epoch, &epoch, sizeof(epoch), cudaMemcpyHostToDevice,
                               x.prep_stream));
    CUDA_CHECK(cudaStreamSynchronize(x.prep_stream));
}

struct Invocation {
    TraceMetrics trace;
    std::vector<DsaTrace> rows;
    unsigned int stale_rows = 0;
    unsigned long long history_loads = 0;
    unsigned int progress_waves = 0;
    unsigned long long consumer_entries_expected = 0;
    unsigned long long consumer_entries_observed = 0;
    unsigned long long consumer_completions_observed = 0;
    unsigned int entry_gate_failures = 0;
    unsigned int impl_preproducer_completions = 0;
};

void resetConsumerCounters(DsaContext& x) {
    for (int kind = 0; kind < kConsumerKinds; ++kind) {
        __atomic_store_n(x.consumer_entries_host + kind, 0u, __ATOMIC_SEQ_CST);
        __atomic_store_n(x.consumer_completions_host + kind, 0u, __ATOMIC_SEQ_CST);
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

unsigned int loadConsumerCounter(const unsigned int* values, int kind) {
    return __atomic_load_n(values + kind, __ATOMIC_SEQ_CST);
}

bool waitForAllConsumerEntries(const DsaContext& x, unsigned int expected) {
    const auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::milliseconds(kEntryGateTimeoutMs);
    while (std::chrono::steady_clock::now() < deadline) {
        if (loadConsumerCounter(x.consumer_entries_host, kConsumerTopk) == expected
            && loadConsumerCounter(x.consumer_entries_host, kConsumerAttention)
               == expected)
            return true;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return false;
}

void printProgressAudit(const DsaConfig& c, const Invocation& inv, int mode,
                        unsigned int epoch) {
    const int expected_waves = mode == DSA_FLOOR ? 1 : c.wave_count;
    const char* protocol = mode == DSA_FLOOR
        ? "full_grid_programmatic_graph" : "bounded_query_waves";
    const char* entry_gate = (mode == DSA_IMPL || mode == DSA_CEILING)
        ? "system_scope_mapped_counter" : "not_applicable_programmatic_graph";
    printf("PROGRESS_DSA semantics=1 tag=%s seq=%d mode=%s epoch=%u "
           "protocol=%s entry_gate=%s "
           "entry_gate_timeout_ms=%d progress_waves=%u expected_progress_waves=%d "
           "consumer_entries=%llu expected_consumer_entries=%llu "
           "consumer_completions=%llu expected_consumer_completions=%llu "
           "entry_gate_failures=%u impl_preproducer_completions=%u "
           "progress_waves_verified=%u consumer_entry_order_failures=%u "
           "producer_forward_progress_failures=%u valid=%d\n",
           c.tag, c.seq, modeName(mode), epoch, protocol, entry_gate,
           kEntryGateTimeoutMs, inv.progress_waves, expected_waves,
           inv.consumer_entries_observed,
           2ull * (unsigned long long)c.query_blocks,
           inv.consumer_completions_observed,
           2ull * (unsigned long long)c.query_blocks,
           inv.entry_gate_failures, inv.impl_preproducer_completions,
           inv.trace.progress_waves_verified,
           inv.trace.consumer_entry_order_failures,
           inv.trace.producer_forward_progress_failures,
           inv.trace.complete ? 1 : 0);
}

Invocation runInvocation(const DsaConfig& c, DsaContext& x,
                         DsaGraph& full_floor_graph, DsaGraph& wave_floor_graph,
                         int mode, unsigned int epoch, unsigned long long history_sum) {
    poisonInvocation(c, x, epoch);
    const char* ranges[kModes] = {
        "dsa.floor", "dsa.impl", "dsa.ceiling", "dsa.wave_floor"
    };
    NvtxRange range(ranges[mode]);
    Invocation out;
    if (mode == DSA_FLOOR) {
        // Preserve the deployed grid-PDL baseline: one complete three-node graph, one launch.
        resetConsumerCounters(x);
        out.progress_waves = 1;
        out.consumer_entries_expected = 2ull * (unsigned)c.query_blocks;
        CUDA_CHECK(cudaGraphLaunch(full_floor_graph.exec, x.graph_stream));
        CUDA_CHECK(cudaStreamSynchronize(x.graph_stream));
        const unsigned int topk_entries = loadConsumerCounter(
            x.consumer_entries_host, kConsumerTopk);
        const unsigned int attention_entries = loadConsumerCounter(
            x.consumer_entries_host, kConsumerAttention);
        const unsigned int topk_completions = loadConsumerCounter(
            x.consumer_completions_host, kConsumerTopk);
        const unsigned int attention_completions = loadConsumerCounter(
            x.consumer_completions_host, kConsumerAttention);
        out.consumer_entries_observed = topk_entries + attention_entries;
        out.consumer_completions_observed = topk_completions + attention_completions;
        if (topk_entries != (unsigned)c.query_blocks
            || attention_entries != (unsigned)c.query_blocks
            || topk_completions != (unsigned)c.query_blocks
            || attention_completions != (unsigned)c.query_blocks)
            ++out.entry_gate_failures;
    } else for (int query_base = 0; query_base < c.query_blocks;
                query_base += c.wave_queries) {
        const int wave_query_count = std::min(c.wave_queries,
                                              c.query_blocks - query_base);
        resetConsumerCounters(x);
        ++out.progress_waves;
        out.consumer_entries_expected += 2ull * (unsigned int)wave_query_count;
        if (mode == DSA_WAVE_FLOOR) {
            setFloorWave(c, x, wave_floor_graph, history_sum, query_base,
                         wave_query_count);
            CUDA_CHECK(cudaGraphLaunch(wave_floor_graph.exec, x.graph_stream));
            CUDA_CHECK(cudaStreamSynchronize(x.graph_stream));
            const unsigned int topk_entries = loadConsumerCounter(
                x.consumer_entries_host, kConsumerTopk);
            const unsigned int attention_entries = loadConsumerCounter(
                x.consumer_entries_host, kConsumerAttention);
            out.consumer_entries_observed += topk_entries + attention_entries;
            out.consumer_completions_observed += loadConsumerCounter(
                x.consumer_completions_host, kConsumerTopk);
            out.consumer_completions_observed += loadConsumerCounter(
                x.consumer_completions_host, kConsumerAttention);
            if (topk_entries != (unsigned)wave_query_count
                || attention_entries != (unsigned)wave_query_count
                || loadConsumerCounter(x.consumer_completions_host, kConsumerTopk)
                   != (unsigned)wave_query_count
                || loadConsumerCounter(x.consumer_completions_host,
                                       kConsumerAttention)
                   != (unsigned)wave_query_count)
                ++out.entry_gate_failures;
        } else {
            dsaAttention<<<wave_query_count, kWorkerThreads, kAttentionSmem,
                           x.attention_stream>>>(
                x.indices, x.history, x.output, x.topk_flags, x.stale_rows,
                x.history_load_counter, x.epoch, x.trace, history_sum,
                c.query_blocks, c.key_tiles, c.physical_degree, c.topk, mode,
                c.attention_prologue, c.attention_tail, query_base,
                x.consumer_entries_device, x.consumer_completions_device);
            CUDA_CHECK(cudaGetLastError());
            dsaTopk<<<wave_query_count, kWorkerThreads, kTopkSmem, x.topk_stream>>>(
                x.scores, x.indices, x.row_sums, x.expected_row_sums,
                x.producer_flags, x.topk_flags,
                x.epoch, x.trace, c.query_blocks, c.key_tiles, c.physical_degree, c.topk,
                mode, c.topk_prologue, c.topk_tail, query_base,
                x.consumer_entries_device, x.consumer_completions_device);
            CUDA_CHECK(cudaGetLastError());
            if (!waitForAllConsumerEntries(x, (unsigned int)wave_query_count)) {
                fprintf(stderr,
                        "consumer entry gate timeout: mode=%s query_base=%d "
                        "wave_queries=%d timeout_ms=%d topk_entered=%u "
                        "attention_entered=%u\n",
                        modeName(mode), query_base, wave_query_count,
                        kEntryGateTimeoutMs,
                        loadConsumerCounter(x.consumer_entries_host, kConsumerTopk),
                        loadConsumerCounter(x.consumer_entries_host,
                                            kConsumerAttention));
                fflush(stderr);
                std::_Exit(3);
            }
            out.consumer_entries_observed += 2ull * (unsigned int)wave_query_count;
            const unsigned int completions_before_producer = loadConsumerCounter(
                x.consumer_completions_host, kConsumerTopk)
                + loadConsumerCounter(x.consumer_completions_host,
                                      kConsumerAttention);
            if (mode == DSA_IMPL && completions_before_producer != 0) {
                out.impl_preproducer_completions += completions_before_producer;
                fprintf(stderr,
                        "Impl consumer completed before producer admission: "
                        "query_base=%d completions=%u\n",
                        query_base, completions_before_producer);
                fflush(stderr);
                std::_Exit(3);
            }
            dsaIndexer<<<wave_query_count * c.physical_degree, kIndexerThreads, 0,
                         x.producer_stream>>>(
                x.scores, x.query_lut, x.key_lut, x.producer_flags, x.epoch, x.trace,
                c.query_blocks, c.key_tiles, c.physical_degree, c.pair_query, c.pair_key,
                mode, c.indexer_ready, c.indexer_tail, query_base);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(x.producer_stream));
            CUDA_CHECK(cudaStreamSynchronize(x.topk_stream));
            CUDA_CHECK(cudaStreamSynchronize(x.attention_stream));
            const unsigned int topk_completions = loadConsumerCounter(
                x.consumer_completions_host, kConsumerTopk);
            const unsigned int attention_completions = loadConsumerCounter(
                x.consumer_completions_host, kConsumerAttention);
            out.consumer_completions_observed += topk_completions
                                               + attention_completions;
            if (topk_completions != (unsigned)wave_query_count
                || attention_completions != (unsigned)wave_query_count)
                ++out.entry_gate_failures;
        }
    }
    out.rows.resize(traceCount(c));
    CUDA_CHECK(cudaMemcpy(out.rows.data(), x.trace, out.rows.size() * sizeof(DsaTrace),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&out.stale_rows, x.stale_rows, sizeof(out.stale_rows),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&out.history_loads, x.history_load_counter,
                          sizeof(out.history_loads), cudaMemcpyDeviceToHost));
    out.trace = analyzeTrace(c, mode, epoch, out.rows);
    const unsigned int expected_progress_waves = mode == DSA_FLOOR
        ? 1u : (unsigned)c.wave_count;
    if (out.progress_waves != expected_progress_waves
        || out.consumer_entries_observed != out.consumer_entries_expected
        || out.consumer_completions_observed != out.consumer_entries_expected
        || out.entry_gate_failures != 0
        || out.impl_preproducer_completions != 0)
        out.trace.complete = false;
    return out;
}

struct ValidationResult {
    Invocation invocation;
    std::array<unsigned long long, C_COUNT> counters{};
    bool valid = false;
};

int validationGrid(size_t count) {
    const size_t need = (count + kWorkerThreads - 1) / kWorkerThreads;
    return (int)std::min<size_t>(need, 65535u);
}

ValidationResult runValidation(const DsaConfig& c, DsaContext& x,
                               DsaGraph& full_floor_graph,
                               DsaGraph& wave_floor_graph, int mode,
                               unsigned int epoch, unsigned long long history_sum) {
    ValidationResult out;
    out.invocation = runInvocation(
        c, x, full_floor_graph, wave_floor_graph, mode, epoch, history_sum);
    const char* validation_range = mode == DSA_FLOOR ? "dsa.validate.floor"
        : mode == DSA_WAVE_FLOOR ? "dsa.validate.wave_floor"
        : "dsa.validate.impl";
    NvtxRange range(validation_range);
    const size_t shared = 3u * kWorkerThreads * sizeof(unsigned long long);
    validateScores<<<validationGrid(scoreCount(c)), kWorkerThreads, shared, x.prep_stream>>>(
        x.scores, x.counters, epoch, scoreCount(c), c.key_tiles,
        c.pair_query, c.pair_key);
    CUDA_CHECK(cudaGetLastError());
    validateIndices<<<validationGrid(indexCount(c)), kWorkerThreads, shared, x.prep_stream>>>(
        x.indices, x.counters, indexCount(c), c.topk, c.key_tiles);
    CUDA_CHECK(cudaGetLastError());
    validateOutput<<<validationGrid(outputCount(c)), kWorkerThreads, shared, x.prep_stream>>>(
        x.output, x.counters, epoch, outputCount(c), history_sum);
    CUDA_CHECK(cudaGetLastError());
    validateRowsAndFlags<<<validationGrid((size_t)c.query_blocks), kWorkerThreads,
                           shared, x.prep_stream>>>(
        x.row_sums, x.producer_flags, x.topk_flags, x.counters, epoch, mode,
        c.query_blocks, c.key_tiles, c.physical_degree, c.pair_query, c.pair_key);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(x.prep_stream));
    CUDA_CHECK(cudaMemcpy(out.counters.data(), x.counters,
                          C_COUNT * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    out.valid = out.invocation.trace.complete
        && out.invocation.history_loads == expectedHistoryLoads(c)
        && out.counters[C_SCORE_MISMATCH] == 0
        && out.counters[C_INDEX_MISMATCH] == 0
        && out.counters[C_OUTPUT_MISMATCH] == 0
        && out.counters[C_ROW_MISMATCH] == 0
        && out.counters[C_FLAG_MISMATCH] == 0
        && out.counters[C_SCORE_OBS] == out.counters[C_SCORE_EXP]
        && out.counters[C_INDEX_OBS] == out.counters[C_INDEX_EXP]
        && out.counters[C_OUTPUT_OBS] == out.counters[C_OUTPUT_EXP]
        && out.counters[C_ROW_OBS] == out.counters[C_ROW_EXP]
        && out.counters[C_FLAG_OBS] == out.counters[C_FLAG_EXP];
    return out;
}

struct CeilingProof {
    Invocation invocation;
    std::array<unsigned long long, C_COUNT> counters{};
    bool wrong = false;
};

CeilingProof runCeilingProof(const DsaConfig& c, DsaContext& x,
                             DsaGraph& full_floor_graph,
                             DsaGraph& wave_floor_graph, unsigned int epoch,
                             unsigned long long history_sum) {
    CeilingProof out;
    out.invocation = runInvocation(
        c, x, full_floor_graph, wave_floor_graph,
        DSA_CEILING, epoch, history_sum);
    NvtxRange range("dsa.validate.ceiling_wrongness");
    const size_t shared = 3u * kWorkerThreads * sizeof(unsigned long long);
    validateIndices<<<validationGrid(indexCount(c)), kWorkerThreads, shared, x.prep_stream>>>(
        x.indices, x.counters, indexCount(c), c.topk, c.key_tiles);
    CUDA_CHECK(cudaGetLastError());
    validateOutput<<<validationGrid(outputCount(c)), kWorkerThreads, shared, x.prep_stream>>>(
        x.output, x.counters, epoch, outputCount(c), history_sum);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(x.prep_stream));
    CUDA_CHECK(cudaMemcpy(out.counters.data(), x.counters,
                          C_COUNT * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    out.wrong = out.invocation.trace.complete && out.invocation.stale_rows > 0
        && out.invocation.history_loads == expectedHistoryLoads(c)
        && (out.counters[C_INDEX_MISMATCH] > 0 || out.counters[C_OUTPUT_MISMATCH] > 0)
        && (out.counters[C_INDEX_OBS] != out.counters[C_INDEX_EXP]
            || out.counters[C_OUTPUT_OBS] != out.counters[C_OUTPUT_EXP]);
    return out;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n & 1u) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

unsigned long long splitmix(unsigned long long* state) {
    unsigned long long z = (*state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

struct Stats { double median = 0.0, low = 0.0, high = 0.0; };

Stats bootstrapMedian(const std::vector<double>& values, unsigned long long seed) {
    Stats s{};
    s.median = median(values);
    std::vector<double> draws(kBootstrap), sample(values.size());
    unsigned long long state = seed;
    for (int b = 0; b < kBootstrap; ++b) {
        for (size_t i = 0; i < values.size(); ++i)
            sample[i] = values[(size_t)(splitmix(&state) % values.size())];
        draws[(size_t)b] = median(sample);
    }
    std::sort(draws.begin(), draws.end());
    s.low = draws[(size_t)(0.025 * kBootstrap)];
    s.high = draws[(size_t)(0.975 * kBootstrap)];
    return s;
}

Stats bootstrapPairDelta(const std::array<std::vector<double>, kModes>& values,
                         int base_mode, int target_mode,
                         unsigned long long seed) {
    Stats s{};
    const double f = median(values[base_mode]);
    const double x = median(values[target_mode]);
    s.median = f > 0.0 ? 100.0 * (f - x) / f : 0.0;
    std::vector<double> draws(kBootstrap);
    std::array<std::vector<double>, kModes> sample;
    for (auto& v : sample) v.resize(values[0].size());
    unsigned long long state = seed;
    for (int b = 0; b < kBootstrap; ++b) {
        for (size_t i = 0; i < values[0].size(); ++i) {
            const size_t pick = (size_t)(splitmix(&state) % values[0].size());
            for (int mode = 0; mode < kModes; ++mode)
                sample[mode][i] = values[mode][pick];
        }
        const double bf = median(sample[base_mode]);
        const double bx = median(sample[target_mode]);
        draws[(size_t)b] = bf > 0.0 ? 100.0 * (bf - bx) / bf : 0.0;
    }
    std::sort(draws.begin(), draws.end());
    s.low = draws[(size_t)(0.025 * kBootstrap)];
    s.high = draws[(size_t)(0.975 * kBootstrap)];
    return s;
}

void writeTrace(const DsaConfig& c, int mode, int rep, unsigned int epoch,
                const std::vector<DsaTrace>& rows, bool* header_written) {
    if (!c.trace_path) return;
    std::ofstream out;
    if (!*header_written) {
        out.open(c.trace_path, std::ios::out | std::ios::trunc);
        out << "schema,tag,seq,mode,rep,epoch,stage,block,sm,t_start,t_dep,t_ready,t_trigger,t_end\n";
        *header_written = true;
    } else {
        out.open(c.trace_path, std::ios::out | std::ios::app);
    }
    if (!out) {
        fprintf(stderr, "cannot open trace path: %s\n", c.trace_path);
        exit(1);
    }
    for (const DsaTrace& r : rows) {
        out << 1 << ',' << c.tag << ',' << c.seq << ',' << modeName(mode) << ','
            << rep << ',' << epoch << ',' << stageName(r.stage) << ',' << r.block << ','
            << r.sm << ',' << r.t_start << ',' << r.t_dep << ',' << r.t_ready << ','
            << r.t_trigger << ',' << r.t_end << '\n';
    }
}

void allocateContext(const DsaConfig& c, DsaContext& x) {
    CUDA_CHECK(cudaMalloc(&x.scores, scoreCount(c) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&x.query_lut, kQueryBlock * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&x.key_lut, kKeyTile * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&x.indices, indexCount(c) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&x.history, (size_t)c.key_tiles * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&x.output, outputCount(c) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&x.row_sums, (size_t)c.query_blocks * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&x.expected_row_sums,
                          (size_t)c.query_blocks * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&x.producer_flags,
                          (size_t)c.producer_ctas * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&x.topk_flags,
                          (size_t)c.query_blocks * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&x.stale_rows, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&x.history_load_counter, sizeof(unsigned long long)));
    CUDA_CHECK(cudaHostAlloc(&x.consumer_entries_host,
                             kConsumerKinds * sizeof(unsigned int),
                             cudaHostAllocMapped | cudaHostAllocPortable));
    CUDA_CHECK(cudaHostGetDevicePointer(&x.consumer_entries_device,
                                        x.consumer_entries_host, 0));
    CUDA_CHECK(cudaHostAlloc(&x.consumer_completions_host,
                             kConsumerKinds * sizeof(unsigned int),
                             cudaHostAllocMapped | cudaHostAllocPortable));
    CUDA_CHECK(cudaHostGetDevicePointer(&x.consumer_completions_device,
                                        x.consumer_completions_host, 0));
    CUDA_CHECK(cudaMalloc(&x.epoch, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&x.counters, C_COUNT * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&x.trace, traceCount(c) * sizeof(DsaTrace)));
    CUDA_CHECK(cudaStreamCreateWithFlags(&x.prep_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&x.graph_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&x.least_priority, &x.greatest_priority));
    x.topk_priority = x.greatest_priority
                    + (x.least_priority - x.greatest_priority) / 2;
    CUDA_CHECK(cudaStreamCreateWithPriority(&x.producer_stream,
                                            cudaStreamNonBlocking, x.greatest_priority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&x.topk_stream,
                                            cudaStreamNonBlocking, x.topk_priority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&x.attention_stream,
                                            cudaStreamNonBlocking, x.least_priority));
    std::vector<unsigned int> history((size_t)c.key_tiles);
    for (int i = 0; i < c.key_tiles; ++i) history[(size_t)i] = historyValue((unsigned)i);
    std::vector<unsigned int> query_lut(kQueryBlock), key_lut(kKeyTile);
    for (int i = 0; i < kQueryBlock; ++i) query_lut[(size_t)i] = pairQueryValue((unsigned)i);
    for (int i = 0; i < kKeyTile; ++i) key_lut[(size_t)i] = pairKeyValue((unsigned)i);
    CUDA_CHECK(cudaMemcpy(x.history, history.data(), history.size() * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x.query_lut, query_lut.data(),
                          query_lut.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x.key_lut, key_lut.data(),
                          key_lut.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void freeContext(DsaContext& x) {
    if (x.prep_stream) cudaStreamDestroy(x.prep_stream);
    if (x.graph_stream) cudaStreamDestroy(x.graph_stream);
    if (x.producer_stream) cudaStreamDestroy(x.producer_stream);
    if (x.topk_stream) cudaStreamDestroy(x.topk_stream);
    if (x.attention_stream) cudaStreamDestroy(x.attention_stream);
    cudaFree(x.scores);
    cudaFree(x.query_lut);
    cudaFree(x.key_lut);
    cudaFree(x.indices);
    cudaFree(x.history);
    cudaFree(x.output);
    cudaFree(x.row_sums);
    cudaFree(x.expected_row_sums);
    cudaFree(x.producer_flags);
    cudaFree(x.topk_flags);
    cudaFree(x.stale_rows);
    cudaFree(x.history_load_counter);
    if (x.consumer_entries_host) cudaFreeHost(x.consumer_entries_host);
    if (x.consumer_completions_host) cudaFreeHost(x.consumer_completions_host);
    cudaFree(x.epoch);
    cudaFree(x.counters);
    cudaFree(x.trace);
}

size_t requiredBytes(const DsaConfig& c) {
    return scoreCount(c) * sizeof(unsigned int)
         + (kQueryBlock + kKeyTile) * sizeof(unsigned int)
         + indexCount(c) * sizeof(unsigned int)
         + outputCount(c) * sizeof(unsigned int)
         + (size_t)c.key_tiles * sizeof(unsigned int)
         + 2u * (size_t)c.query_blocks * sizeof(unsigned long long)
         + ((size_t)c.producer_ctas + c.query_blocks + 2u) * sizeof(unsigned int)
         + traceCount(c) * sizeof(DsaTrace)
         + (C_COUNT + 1u) * sizeof(unsigned long long);
}

std::string canonicalGpuUuid(const cudaUUID_t& uuid) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string out = "GPU-";
    out.reserve(40);
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) out.push_back('-');
        const unsigned int value = static_cast<unsigned char>(uuid.bytes[i]);
        out.push_back(hex[value >> 4]);
        out.push_back(hex[value & 15u]);
    }
    return out;
}

std::string hexEncode(const char* text) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    const size_t length = strlen(text);
    out.reserve(length * 2);
    for (size_t i = 0; i < length; ++i) {
        const unsigned int value = static_cast<unsigned char>(text[i]);
        out.push_back(hex[value >> 4]);
        out.push_back(hex[value & 15u]);
    }
    return out;
}

void printUsage() {
    printf("usage: dsa_native --seq N [--repeats N --warmup N --trace PATH --tag TAG]\n"
           "                  [--indexer-ready C --indexer-tail C --topk-prologue C --topk-tail C]\n"
           "                  [--attention-prologue C --attention-tail C]\n"
           "                  [--pair-query N --pair-key N]  (short only with --allow-short)\n"
           "                  [--allow-short] [--dry-run]\n");
}

} // namespace

int main(int argc, char** argv) {
    Args args(argc, argv);
    if (args.has("--help") || args.has("-h")) {
        printUsage();
        return 0;
    }
    DsaConfig c;
    c.seq = (int)args.ll("--seq", c.seq);
    c.repeats = (int)args.ll("--repeats", c.repeats);
    c.warmup = (int)args.ll("--warmup", c.warmup);
    c.indexer_ready = (unsigned long long)args.ll("--indexer-ready", c.indexer_ready);
    c.indexer_tail = (unsigned long long)args.ll("--indexer-tail", c.indexer_tail);
    c.topk_prologue = (unsigned long long)args.ll("--topk-prologue", c.topk_prologue);
    c.topk_tail = (unsigned long long)args.ll("--topk-tail", c.topk_tail);
    c.attention_prologue = (unsigned long long)args.ll(
        "--attention-prologue", c.attention_prologue);
    c.attention_tail = (unsigned long long)args.ll("--attention-tail", c.attention_tail);
    c.pair_query = (int)args.ll("--pair-query", c.pair_query);
    c.pair_key = (int)args.ll("--pair-key", c.pair_key);
    c.tag = args.str("--tag", c.tag);
    c.trace_path = args.has("--trace") ? args.str("--trace", nullptr) : nullptr;
    c.allow_short = args.has("--allow-short");
    c.dry_run = args.has("--dry-run");
    if (c.seq <= 0 || c.seq % kKeyTile != 0 || c.repeats <= 0 || c.warmup < 0) {
        fprintf(stderr, "seq must be positive/divisible by 128; repeats>0 and warmup>=0\n");
        return 2;
    }
    if (c.repeats < 31 && !c.allow_short) {
        fprintf(stderr, "refusing fewer than 31 repeats without --allow-short\n");
        return 2;
    }
    if (c.pair_query <= 0 || c.pair_query > kQueryBlock
        || c.pair_key <= 0 || c.pair_key > kKeyTile) {
        fprintf(stderr, "pair dimensions must be within query_block=64/key_tile=128\n");
        return 2;
    }
    if ((c.pair_query != kQueryBlock || c.pair_key != kKeyTile) && !c.allow_short) {
        fprintf(stderr, "formal runs require complete 64x128 pair work\n");
        return 2;
    }
    c.query_blocks = c.seq / kQueryBlock;
    c.key_tiles = c.seq / kKeyTile;
    c.logical_degree = c.key_tiles;
    // Preserve exact CTA degree for 4K/32K; bound the long-context producer/trace volume.
    c.physical_degree = c.seq <= 32768 ? c.key_tiles : std::min(c.key_tiles, 64);
    c.topk = std::min(kIndexTopk, c.key_tiles);
    const long long producer_ctas = (long long)c.query_blocks * c.physical_degree;
    if (producer_ctas > std::numeric_limits<int>::max()) {
        fprintf(stderr, "producer grid exceeds CUDA 1D grid limit\n");
        return 2;
    }
    c.producer_ctas = (int)producer_ctas;

    DeviceInfo device = queryDevice();
    if (device.dev != 0) {
        fprintf(stderr,
                "CUDA runtime ordinal must be 0 after UUID visibility binding; observed=%d\n",
                device.dev);
        return 2;
    }
    printDeviceBanner(device);
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device.dev));
    const cudaUUID_t runtime_uuid_bytes = prop.uuid;
    const std::string runtime_uuid = canonicalGpuUuid(runtime_uuid_bytes);
    const std::string runtime_name_hex = hexEncode(prop.name);
    CUDA_CHECK(cudaFuncSetAttribute(dsaTopk, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)kTopkSmem));
    CUDA_CHECK(cudaFuncSetAttribute(dsaAttention,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)kAttentionSmem));
    int index_occ = 0, topk_occ = 0, attention_occ = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &index_occ, dsaIndexer, kIndexerThreads, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &topk_occ, dsaTopk, kWorkerThreads, kTopkSmem));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &attention_occ, dsaAttention, kWorkerThreads, kAttentionSmem));
    cudaFuncAttributes ia{}, ta{}, aa{};
    CUDA_CHECK(cudaFuncGetAttributes(&ia, dsaIndexer));
    CUDA_CHECK(cudaFuncGetAttributes(&ta, dsaTopk));
    CUDA_CHECK(cudaFuncGetAttributes(&aa, dsaAttention));
    const size_t mixed_smem = ia.sharedSizeBytes + ta.sharedSizeBytes + aa.sharedSizeBytes
                            + kTopkSmem + kAttentionSmem;
    const long long mixed_threads = kIndexerThreads + 2ll * kWorkerThreads;
    const long long mixed_regs = (long long)ia.numRegs * kIndexerThreads
                               + (long long)ta.numRegs * kWorkerThreads
                               + (long long)aa.numRegs * kWorkerThreads;
    // Bound waiting consumers globally instead of assuming that three independent kernels
    // will co-reside.  At most 2*W consumer CTAs exist in a wave; selecting W<=floor((S-1)/3)
    // reserves at least roughly one third of all SMs for producer progress in the worst case.
    c.wave_queries = std::min(c.query_blocks,
                              std::max(1, (prop.multiProcessorCount - 1) / 3));
    c.wave_count = (c.query_blocks + c.wave_queries - 1) / c.wave_queries;
    c.reserved_producer_sms = prop.multiProcessorCount - 2 * c.wave_queries;
    const bool resource_safe = index_occ >= 1 && topk_occ >= 1 && attention_occ >= 1
        && c.wave_queries >= 1 && c.reserved_producer_sms >= 1
        && 2 * c.wave_queries < prop.multiProcessorCount;
    if (!resource_safe) {
        fprintf(stderr, "resource reservation cannot guarantee producer/topk progress\n");
        return 2;
    }
    int stream_least = 0, stream_greatest = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&stream_least, &stream_greatest));
    const int stream_topk = stream_greatest + (stream_least - stream_greatest) / 2;
    const int distinct_priorities = stream_least == stream_greatest ? 1
        : (stream_topk != stream_least && stream_topk != stream_greatest ? 3 : 2);
    if (distinct_priorities != 3) {
        fprintf(stderr,
                "Tier-5 bracket requires three distinct corresponding stream priorities; "
                "least=%d topk=%d greatest=%d\n",
                stream_least, stream_topk, stream_greatest);
        return 2;
    }

    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t needed = requiredBytes(c);
    if (needed + (1ull << 30) > free_bytes) {
        fprintf(stderr, "insufficient free memory: needed=%zu margin=1073741824 free=%zu\n",
                needed, free_bytes);
        return 2;
    }
    const double tightness = c.logical_degree > 0
        ? (double)c.logical_degree /
          ((double)c.physical_degree *
           std::ceil((double)c.logical_degree / c.physical_degree)) : 0.0;
    printf("DEVICE_DSA semantics=1 tag=%s seq=%d runtime_ordinal=%d "
           "runtime_ordinal_zero=%d runtime_uuid=%s name_hex=%s "
           "cc_major=%d cc_minor=%d sms=%d\n",
           c.tag, c.seq, device.dev, device.dev == 0 ? 1 : 0,
           runtime_uuid.c_str(), runtime_name_hex.c_str(), prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("CONFIG_DSA semantics=1 tag=%s seq=%d query_block=%d key_tile=%d "
           "runtime_ordinal=%d runtime_ordinal_zero=%d runtime_uuid=%s "
           "runtime_name_hex=%s runtime_cc_major=%d runtime_cc_minor=%d runtime_sms=%d "
           "query_blocks=%d key_tiles=%d logical_degree=%d physical_cta_degree=%d "
           "tiles_per_cta_max=%d interval_tightness=%.6f mapping=%s "
           "structure=interval eff_degree=%d sms=%d topk=%d "
           "pair_query=%d pair_key=%d pair_work_per_score=%d pair_work_items=%llu "
           "pair_work_complete=%d "
           "pair_accumulator=uint32_mod2p32 "
           "pair_low16_equivalence=mod2p32_then_low16_equals_uint64_low16 "
           "pair_query_cache=cta_shared_once "
           "pair_key_cache=cta_shared_once_register_tile pair_key_register_tile=%d "
           "pair_iteration=explicit_inline_ptx_add_u32_per_pair pair_closed_form=0 "
           "pair_lut_global_loads_per_cta=%d pair_adds_per_score=%d "
           "history_loads_expected_per_invocation=%llu history_loads_per_rank=1 "
           "history_load_count=device_dynamic_exact "
           "history_load_work_parity=floor_wave_floor_impl_ceiling_equal "
           "producer_ctas=%d topk_ctas=%d attention_ctas=%d "
           "query_wave_size=%d query_wave_count=%d "
           "forward_progress_protocol=full_grid_floor_plus_bounded_query_waves "
           "wave_work_parity=floor_wave_floor_impl_ceiling_equal "
           "floor_overlap_metric=consumer_start_before_upstream_kernel_end "
           "floor_dependency_metric=consumer_dep_after_upstream_programmatic_trigger "
           "mode_count=4 mode_order=floor,wave_floor,impl,ceiling "
           "sample_order=cyclic_latin_4 "
           "repeats=%d warmup=%d "
           "timer=globaltimer timer_scope=first_cta_start_to_last_cta_end "
           "host_launch_before_first_cta_included=0 host_submission_path_differs=1 "
           "floor_graph_vs_impl_stream_submission=outside_timer_not_normalized nvtx=%d "
           "floor_path=three_node_programmatic_graph "
           "full_grid_floor_single_launch=1 "
           "wave_floor_path=bounded_wave_three_node_programmatic_graph "
           "impl_path=matched_priority_streams_epoch_flags "
           "ceiling_path=matched_priority_streams_no_wait_no_publish "
           "stream_priority_pairing=identical_by_stage launch_order=attention_topk_indexer "
           "trigger_floor=ready trigger_wave_floor=ready "
           "trigger_impl=entry trigger_ceiling=entry "
           "floor_wait=griddepcontrol impl_wait=per_producer_epoch_acquire "
           "wave_floor_wait=griddepcontrol ceiling_wait=none "
           "topk_algorithm=monotonic_analytical_proxy full_score_scan=1 "
           "validation=untimed_device_full_element poison=epoch_derived_full "
           "timed_reference_loops=0 expected_row_prep=untimed_per_epoch "
           "long_context_boundary=work_complete_packed_proxy "
           "indexer_ready=%llu indexer_tail=%llu topk_prologue=%llu topk_tail=%llu "
           "attention_prologue=%llu attention_tail=%llu "
           "score_bytes=%zu index_bytes=%zu output_bytes=%zu trace_bytes=%zu "
           "required_bytes=%zu free_bytes=%zu\n",
           c.tag, c.seq, kQueryBlock, kKeyTile, device.dev,
           device.dev == 0 ? 1 : 0, runtime_uuid.c_str(), runtime_name_hex.c_str(),
           prop.major, prop.minor, prop.multiProcessorCount,
           c.query_blocks, c.key_tiles,
           c.logical_degree, c.physical_degree,
           (c.key_tiles + c.physical_degree - 1) / c.physical_degree,
           tightness, c.physical_degree == c.logical_degree
                ? "exact" : "work_complete_packed_proxy",
           c.physical_degree, prop.multiProcessorCount,
           c.topk, c.pair_query, c.pair_key, c.pair_query * c.pair_key,
           (unsigned long long)scoreCount(c) * c.pair_query * c.pair_key,
           c.pair_query == kQueryBlock && c.pair_key == kKeyTile ? 1 : 0,
           kPairKeyRegisterTile, c.pair_query + c.pair_key,
           c.pair_query * c.pair_key, expectedHistoryLoads(c),
           c.producer_ctas, c.query_blocks, c.query_blocks,
           c.wave_queries, c.wave_count, c.repeats, c.warmup,
           DSA_HAS_NVTX,
           c.indexer_ready, c.indexer_tail, c.topk_prologue, c.topk_tail,
           c.attention_prologue, c.attention_tail,
           scoreCount(c) * sizeof(unsigned int), indexCount(c) * sizeof(unsigned int),
           outputCount(c) * sizeof(unsigned int), traceCount(c) * sizeof(DsaTrace),
           needed, free_bytes);
    printf("RESOURCE_DSA semantics=1 tag=%s seq=%d "
           "indexer_occ=%d topk_occ=%d attention_occ=%d "
           "mixed_smem=%zu/%zu mixed_threads=%lld/%d mixed_regs=%lld/%d "
           "producer_progress_slot=1 topk_progress_with_attention=1 "
           "progress_proof=global_consumer_cta_bound "
           "query_wave_size=%d consumer_ctas_per_wave=%d "
           "reserved_producer_sms=%d total_sms=%d "
           "stream_priority_least=%d stream_priority_greatest=%d "
           "stream_priority_topk=%d distinct_priority_values=%d\n",
           c.tag, c.seq, index_occ, topk_occ, attention_occ, mixed_smem,
           (size_t)prop.sharedMemPerMultiprocessor, mixed_threads,
           prop.maxThreadsPerMultiProcessor, mixed_regs, prop.regsPerMultiprocessor,
           c.wave_queries, 2 * c.wave_queries, c.reserved_producer_sms,
           prop.multiProcessorCount,
           stream_least, stream_greatest, stream_topk, distinct_priorities);
    if (c.dry_run) return 0;

    DsaContext x;
    allocateContext(c, x);
    const unsigned long long history_sum = expectedHistorySum(c);
    DsaGraph full_floor_graph = buildFloorGraph(
        c, x, history_sum, c.query_blocks);
    DsaGraph wave_floor_graph = buildFloorGraph(
        c, x, history_sum, c.wave_queries);
    unsigned int epoch = 0;
    bool any_failure = false;
    bool all_history_load_complete = true;

    // Admission runs first.  A failed correct-path full-element check or a Ceiling that is
    // not observably wrong aborts before the first SAMPLE_DSA record can be emitted.
    for (int mode : {DSA_FLOOR, DSA_WAVE_FLOOR, DSA_IMPL}) {
        ++epoch;
        ValidationResult v = runValidation(
            c, x, full_floor_graph, wave_floor_graph, mode, epoch, history_sum);
        printProgressAudit(c, v.invocation, mode, epoch);
        printf("VALIDATION_DSA semantics=1 tag=%s seq=%d mode=%s epoch=%u "
               "poison=epoch_derived_full checker=device_full_element trace_complete=%d "
               "history_loads=%llu expected_history_loads=%llu "
               "history_load_complete=%d "
               "score_elements=%zu index_elements=%zu output_elements=%zu "
               "score_mismatches=%llu index_mismatches=%llu output_mismatches=%llu "
               "row_mismatches=%llu flag_mismatches=%llu "
               "score_checksum_observed=%llu score_checksum_expected=%llu "
               "index_checksum_observed=%llu index_checksum_expected=%llu "
               "output_checksum_observed=%llu output_checksum_expected=%llu "
               "row_checksum_observed=%llu row_checksum_expected=%llu "
               "flag_checksum_observed=%llu flag_checksum_expected=%llu valid=%d\n",
               c.tag, c.seq, modeName(mode), epoch,
               v.invocation.trace.complete ? 1 : 0,
               v.invocation.history_loads, expectedHistoryLoads(c),
               v.invocation.history_loads == expectedHistoryLoads(c) ? 1 : 0,
               scoreCount(c), indexCount(c), outputCount(c),
               v.counters[C_SCORE_MISMATCH], v.counters[C_INDEX_MISMATCH],
               v.counters[C_OUTPUT_MISMATCH], v.counters[C_ROW_MISMATCH],
               v.counters[C_FLAG_MISMATCH], v.counters[C_SCORE_OBS],
               v.counters[C_SCORE_EXP], v.counters[C_INDEX_OBS],
               v.counters[C_INDEX_EXP], v.counters[C_OUTPUT_OBS],
               v.counters[C_OUTPUT_EXP], v.counters[C_ROW_OBS],
               v.counters[C_ROW_EXP], v.counters[C_FLAG_OBS],
               v.counters[C_FLAG_EXP], v.valid ? 1 : 0);
        if (v.invocation.history_loads != expectedHistoryLoads(c))
            all_history_load_complete = false;
        if (!v.valid) any_failure = true;
    }
    ++epoch;
    CeilingProof ceiling_proof = runCeilingProof(
        c, x, full_floor_graph, wave_floor_graph, epoch, history_sum);
    printProgressAudit(c, ceiling_proof.invocation, DSA_CEILING, epoch);
    printf("CEILING_PROOF_DSA semantics=1 tag=%s seq=%d mode=ceiling epoch=%u "
           "poison=epoch_derived_full checker=device_wrongness_full_output_index "
           "trace_complete=%d stale_rows=%u history_loads=%llu "
           "expected_history_loads=%llu history_load_complete=%d "
           "index_mismatches=%llu output_mismatches=%llu "
           "index_checksum_observed=%llu index_checksum_expected=%llu "
           "output_checksum_observed=%llu output_checksum_expected=%llu wrong=%d\n",
           c.tag, c.seq, epoch, ceiling_proof.invocation.trace.complete ? 1 : 0,
           ceiling_proof.invocation.stale_rows,
           ceiling_proof.invocation.history_loads, expectedHistoryLoads(c),
           ceiling_proof.invocation.history_loads == expectedHistoryLoads(c) ? 1 : 0,
           ceiling_proof.counters[C_INDEX_MISMATCH],
           ceiling_proof.counters[C_OUTPUT_MISMATCH],
           ceiling_proof.counters[C_INDEX_OBS], ceiling_proof.counters[C_INDEX_EXP],
           ceiling_proof.counters[C_OUTPUT_OBS], ceiling_proof.counters[C_OUTPUT_EXP],
           ceiling_proof.wrong ? 1 : 0);
    if (ceiling_proof.invocation.history_loads != expectedHistoryLoads(c))
        all_history_load_complete = false;
    if (!ceiling_proof.wrong) any_failure = true;
    printf("ADMISSION_DSA semantics=1 tag=%s seq=%d "
           "validations=3 ceiling_proofs=1 history_load_complete=%d valid=%d\n",
           c.tag, c.seq, all_history_load_complete ? 1 : 0, any_failure ? 0 : 1);
    if (any_failure) {
        destroyGraph(full_floor_graph);
        destroyGraph(wave_floor_graph);
        freeContext(x);
        return 2;
    }

    bool ceiling_all_wrong = true;
    for (int warm = 0; warm < c.warmup; ++warm) {
        const int canonical_modes[kModes] = {
            DSA_FLOOR, DSA_WAVE_FLOOR, DSA_IMPL, DSA_CEILING
        };
        for (int mi = 0; mi < kModes; ++mi) {
            const int mode = canonical_modes[mi];
            ++epoch;
            Invocation inv = runInvocation(
                c, x, full_floor_graph, wave_floor_graph,
                mode, epoch, history_sum);
            printProgressAudit(c, inv, mode, epoch);
            const bool ceiling_wrong = mode == DSA_CEILING && inv.stale_rows > 0;
            const bool history_load_complete = inv.history_loads == expectedHistoryLoads(c);
            const bool ok = inv.trace.complete
                && history_load_complete
                && (mode == DSA_CEILING ? ceiling_wrong : inv.stale_rows == 0);
            printf("WARMUP_DSA semantics=1 tag=%s seq=%d warmup=%d mode=%s epoch=%u "
                   "trace_complete=%d stale_rows=%u history_loads=%llu "
                   "expected_history_loads=%llu history_load_complete=%d "
                   "ceiling_wrong=%d\n",
                   c.tag, c.seq, warm, modeName(mode), epoch,
                   inv.trace.complete ? 1 : 0, inv.stale_rows,
                   inv.history_loads, expectedHistoryLoads(c),
                   history_load_complete ? 1 : 0,
                   ceiling_wrong ? 1 : 0);
            if (!history_load_complete) all_history_load_complete = false;
            if (!ok) any_failure = true;
            if (mode == DSA_CEILING && inv.stale_rows == 0) ceiling_all_wrong = false;
        }
    }
    if (any_failure) {
        fprintf(stderr, "warmup semantic proof failed; refusing timed samples\n");
        destroyGraph(full_floor_graph);
        destroyGraph(wave_floor_graph);
        freeContext(x);
        return 2;
    }

    std::array<std::vector<double>, kModes> samples;
    for (auto& v : samples) v.reserve((size_t)c.repeats);
    bool trace_header = false;
    for (int rep = 0; rep < c.repeats; ++rep) {
        const int canonical[kModes] = {
            DSA_FLOOR, DSA_WAVE_FLOOR, DSA_IMPL, DSA_CEILING
        };
        for (int oi = 0; oi < kModes; ++oi) {
            // Four-repetition cyclic Latin rotation: every mode occupies every
            // within-repetition position exactly once in each complete block of four.
            const int mode = canonical[(oi + rep) % kModes];
            ++epoch;
            Invocation inv = runInvocation(
                c, x, full_floor_graph, wave_floor_graph,
                mode, epoch, history_sum);
            printProgressAudit(c, inv, mode, epoch);
            const bool ceiling_wrong = mode == DSA_CEILING && inv.stale_rows > 0;
            const bool history_load_complete = inv.history_loads == expectedHistoryLoads(c);
            const bool overlap_ok = mode == DSA_CEILING || c.allow_short
                || (inv.trace.topk_early > 0 && inv.trace.attention_early > 0
                    && inv.trace.topk_waited > 0 && inv.trace.attention_waited > 0);
            if (!inv.trace.complete
                || !history_load_complete
                || (mode == DSA_CEILING ? !ceiling_wrong : inv.stale_rows != 0)
                || !overlap_ok)
                any_failure = true;
            if (!history_load_complete) all_history_load_complete = false;
            if (mode == DSA_CEILING && inv.stale_rows == 0) ceiling_all_wrong = false;
            samples[mode].push_back(inv.trace.makespan_ms);
            printf("SAMPLE_DSA semantics=1 tag=%s seq=%d rep=%d order=%d mode=%s epoch=%u "
                   "ms=%.6f trace_complete=%d stale_rows=%u topk_early=%u "
                   "attention_early=%u topk_waited=%u attention_waited=%u "
                   "safety_failures=%u safety_applicability=%s trigger_failures=%u "
                   "history_loads=%llu expected_history_loads=%llu "
                   "history_load_complete=%d ceiling_wrong=%d\n",
                   c.tag, c.seq, rep, oi, modeName(mode), epoch,
                   inv.trace.makespan_ms, inv.trace.complete ? 1 : 0, inv.stale_rows,
                   inv.trace.topk_early, inv.trace.attention_early,
                   inv.trace.topk_waited, inv.trace.attention_waited,
                   inv.trace.safety_failures,
                   mode == DSA_CEILING ? "not_applicable" : "dependency_required",
                   inv.trace.trigger_failures, inv.history_loads, expectedHistoryLoads(c),
                   history_load_complete ? 1 : 0, ceiling_wrong ? 1 : 0);
            if (rep == c.repeats - 1)
                writeTrace(c, mode, rep, epoch, inv.rows, &trace_header);
        }
    }

    const Stats floor = bootstrapMedian(samples[DSA_FLOOR], 0xd5a00001ull + c.seq);
    const Stats wave_floor = bootstrapMedian(
        samples[DSA_WAVE_FLOOR], 0xd5a00006ull + c.seq);
    const Stats impl = bootstrapMedian(samples[DSA_IMPL], 0xd5a00002ull + c.seq);
    const Stats ceiling = bootstrapMedian(samples[DSA_CEILING], 0xd5a00003ull + c.seq);
    const Stats space = bootstrapPairDelta(
        samples, DSA_FLOOR, DSA_CEILING, 0xd5a00004ull + c.seq);
    const Stats captured = bootstrapPairDelta(
        samples, DSA_FLOOR, DSA_IMPL, 0xd5a00005ull + c.seq);
    const Stats full_to_wave = bootstrapPairDelta(
        samples, DSA_FLOOR, DSA_WAVE_FLOOR, 0xd5a00007ull + c.seq);
    const Stats matched_protocol = bootstrapPairDelta(
        samples, DSA_WAVE_FLOOR, DSA_IMPL, 0xd5a00008ull + c.seq);
    const double of_space = space.median != 0.0 ? 100.0 * captured.median / space.median : 0.0;
    printf("SUMMARY_DSA semantics=1 tag=%s seq=%d query_blocks=%d key_tiles=%d "
           "logical_degree=%d physical_cta_degree=%d tiles_per_cta_max=%d "
           "interval_tightness=%.6f tightness=%.6f mapping=%s "
           "structure=interval eff_degree=%d "
           "producer_ctas=%d topk_ctas=%d attention_ctas=%d sms=%d "
           "query_wave_size=%d query_wave_count=%d "
           "forward_progress_protocol=full_grid_floor_plus_bounded_query_waves "
           "wave_work_parity=floor_wave_floor_impl_ceiling_equal "
           "floor_overlap_metric=consumer_start_before_upstream_kernel_end "
           "floor_dependency_metric=consumer_dep_after_upstream_programmatic_trigger "
           "mode_count=4 mode_order=floor,wave_floor,impl,ceiling "
           "sample_order=cyclic_latin_4 "
           "indexer_occ=%d topk_occ=%d attention_occ=%d topk=%d "
           "pair_query=%d pair_key=%d pair_work_items=%llu pair_work_complete=%d "
           "pair_accumulator=uint32_mod2p32 "
           "pair_low16_equivalence=mod2p32_then_low16_equals_uint64_low16 "
           "pair_query_cache=cta_shared_once "
           "pair_key_cache=cta_shared_once_register_tile pair_key_register_tile=%d "
           "pair_iteration=explicit_inline_ptx_add_u32_per_pair pair_closed_form=0 "
           "pair_lut_global_loads_per_cta=%d pair_adds_per_score=%d "
           "history_loads_expected_per_invocation=%llu history_loads_per_rank=1 "
           "history_load_count=device_dynamic_exact "
           "history_load_work_parity=floor_wave_floor_impl_ceiling_equal "
           "history_load_complete=%d "
           "repeats=%d warmup=%d "
           "epoch_first=1 epoch_last=%u samples=%d validations=3 ceiling_proofs=1 "
           "floor_ms=%.6f floor_ci_low=%.6f floor_ci_high=%.6f "
           "wave_floor_ms=%.6f wave_floor_ci_low=%.6f wave_floor_ci_high=%.6f "
           "impl_ms=%.6f impl_ci_low=%.6f impl_ci_high=%.6f "
           "ceiling_ms=%.6f ceiling_ci_low=%.6f ceiling_ci_high=%.6f "
           "space_pct=%.6f space_ci_low=%.6f space_ci_high=%.6f "
           "captured_pct=%.6f captured_ci_low=%.6f captured_ci_high=%.6f "
           "full_to_wave_pct=%.6f full_to_wave_ci_low=%.6f "
           "full_to_wave_ci_high=%.6f "
           "matched_protocol_pct=%.6f matched_protocol_ci_low=%.6f "
           "matched_protocol_ci_high=%.6f "
           "captured_interpretation=bounded_wave_end_to_end_mechanism_envelope_not_pure_cta_headroom "
           "matched_protocol_interpretation=wave_boundary_matched_pdl_vs_epoch_flags "
           "of_space_pct=%.6f timer=globaltimer "
           "timer_scope=first_cta_start_to_last_cta_end "
           "host_launch_before_first_cta_included=0 host_submission_path_differs=1 "
           "floor_graph_vs_impl_stream_submission=outside_timer_not_normalized "
           "adjacent_rungs=1 "
           "floor_path=three_node_programmatic_graph "
           "full_grid_floor_single_launch=1 "
           "wave_floor_path=bounded_wave_three_node_programmatic_graph "
           "impl_path=matched_priority_streams_epoch_flags "
           "ceiling_path=matched_priority_streams_no_wait_no_publish "
           "stream_priority_pairing=identical_by_stage launch_order=attention_topk_indexer "
           "trigger_floor=ready trigger_wave_floor=ready "
           "trigger_impl=entry trigger_ceiling=entry "
           "floor_wait=griddepcontrol wave_floor_wait=griddepcontrol "
           "impl_wait=per_producer_epoch_acquire ceiling_wait=none "
           "timed_reference_loops=0 expected_row_prep=untimed_per_epoch "
           "long_context_boundary=work_complete_packed_proxy "
           "ceiling_trace_safety=not_applicable "
           "ceiling_correctness=unsafe_not_validated ceiling_verified=0 "
           "ceiling_wrongness_verified=%d valid=%d\n",
           c.tag, c.seq, c.query_blocks, c.key_tiles, c.logical_degree,
           c.physical_degree, (c.key_tiles + c.physical_degree - 1) / c.physical_degree,
           tightness, tightness, c.physical_degree == c.logical_degree
                ? "exact" : "work_complete_packed_proxy",
           c.physical_degree, c.producer_ctas, c.query_blocks, c.query_blocks,
           prop.multiProcessorCount, c.wave_queries, c.wave_count,
           index_occ, topk_occ, attention_occ,
           c.topk, c.pair_query, c.pair_key,
           (unsigned long long)scoreCount(c) * c.pair_query * c.pair_key,
           c.pair_query == kQueryBlock && c.pair_key == kKeyTile ? 1 : 0,
           kPairKeyRegisterTile, c.pair_query + c.pair_key,
           c.pair_query * c.pair_key, expectedHistoryLoads(c),
           all_history_load_complete ? 1 : 0,
           c.repeats, c.warmup, epoch, c.repeats * kModes,
           floor.median, floor.low, floor.high,
           wave_floor.median, wave_floor.low, wave_floor.high,
           impl.median, impl.low, impl.high,
           ceiling.median, ceiling.low, ceiling.high,
           space.median, space.low, space.high,
           captured.median, captured.low, captured.high,
           full_to_wave.median, full_to_wave.low, full_to_wave.high,
           matched_protocol.median, matched_protocol.low, matched_protocol.high,
           of_space,
           ceiling_proof.wrong && ceiling_all_wrong ? 1 : 0,
           any_failure ? 0 : 1);
    if (c.trace_path)
        printf("TRACE_DSA semantics=1 tag=%s seq=%d path=%s rep=%d modes=4 rows=%zu\n",
               c.tag, c.seq, c.trace_path, c.repeats - 1,
               (size_t)4 * traceCount(c));
    fflush(stdout);

    destroyGraph(full_floor_graph);
    destroyGraph(wave_floor_graph);
    freeContext(x);
    return any_failure ? 2 : 0;
}
