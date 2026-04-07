/***********************************************************************
   UM Analyzer V8.3.2  -   schema 2.8
  Author : parallelArchitect
  GitHub : https://github.com/parallelArchitect
  License: MIT

  PURPOSE
  -------
  Unified Memory diagnostic and pre-run health gate for NVIDIA GPU
  platforms.  Detects UM paradigm, transport layer, and platform type
  at runtime.  Adapts measurement strategy, labels, and verdict logic
  accordingly.  No hardcoded architecture assumptions.

  Universal platform support:
    FULL_EXPLICIT          — discrete GPU, PCIe transport (Pascal, Ampere, Ada, etc.)
    FULL_HARDWARE_COHERENT — coherent UMA, C2C interconnect (DGX Spark / GB10,
                             Grace Blackwell, Grace Hopper)
    FULL_SOFTWARE_COHERENT — HMM / software-managed coherence

  All verdict logic, transport labels, CUPTI counter gating, memory
  ceiling computation, and diagnostic interpretation adapt at runtime
  based on detected platform.  Pascal-era assumptions are not baked in.

  PASSES
  ------
    COLD : per-ratio fresh child process, skip_prefetch=true
           pages fault on first GPU touch  -  measures real migration path
    WARM : single process, ascending sweep, cudaMemPrefetchAsync before
           timing  -  pages GPU-resident before measurement begins
    PRES : three repeated passes at highest runnable ratio
           tracks CV decay to detect settling vs. oscillation

  MEASUREMENT PHASES (per pass)
  ------------------------------
    A) cudaMallocManaged allocation
    B) CPU page-touch initialisation
    C) Prefetch to GPU  (wall-clock timed)
    D) GPU first touch
    E) GPU steady repeats (p50 / p90 / p99 / max / tail / CV)
    F) Prefetch to CPU + CPU retouch
    G) Prefetch back to GPU + GPU retouch

  THRASH DETECTION
  ----------------
    thrash_score = cv_instability x fault_density x settling_factor
    settling_factor = 0.4 if pressure converged, 1.0 if not
    States: STABLE (<0.15) / SETTLED_UNSTABLE (0.15-0.45, settled) / UNSTABLE (0.15-0.45, not settled) / SEVERE_UNSTABLE (>0.45)

    thrash_state=UNSTABLE or SEVERE_UNSTABLE used in MIGRATION_PRESSURE / UM_THRASHING gates
    regardless of other subsystem results.

  PRESSURE STABILITY LABEL
  ------------------------
    stable=LOW_CV_ALL   all pressure passes below CV 0.05
    stable=passN        stability_meaningful=true and CV converged by pass N
    stable=no           stability_meaningful=false, or CV did not converge

    stability_meaningful = (fault_migration_regime || cv_history[0] > 0.10)
    A starting CV below 0.10 with no fault migration is not considered a
    meaningful stability measurement  -  printed as stable=no regardless of
    whether later passes appear to settle.

  PASS TIMING COVERAGE
  --------------------
    All three pass types (COLD, WARM, PRESSURE) now populate the full
    times_ms struct:
      alloc_ms, cpu_init_ms, prefetch_to_gpu_ms, gpu_first_touch_ms,
      steady_p50/p90/p99/mean/cv, prefetch_to_cpu_ms, cpu_retouch_ms,
      prefetch_back_to_gpu_ms, gpu_retouch_ms

    COLD: timing transmitted from child process via IPC (full JSON payload).
    WARM: timing measured inline in parent process.
    PRESSURE: timing measured inline as before.

    Fields not applicable to a pass type remain 0.0:
      COLD/WARM: pressure_p50_ms, pressure_score (pressure-only)
      COLD: context_warmup_ms (child process, not captured)

  CUPTI INSTRUMENTATION  (added v8)
  ----------------------------------
  Uses the CUPTI Activity API to collect hardware-level UM event records
  directly from the CUDA driver during the measurement passes.

  API path: CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER
    - Same cuptiActivityEnable() call on all architectures.
    - CUPTI returns per-event records asynchronously via buffer callback.
    - Records are accumulated across all passes into g_cupti.
    - Architecture differences resolve at runtime  -  no hardcoded SM checks.

  Counter kinds collected:
    BYTES_TRANSFER_HTOD   -  bytes migrated host->device (all architectures)
    BYTES_TRANSFER_DTOH   -  bytes migrated device->host (all architectures)
    GPU_PAGE_FAULT        -  GPU-side page fault groups  (all architectures)
    THRASHING             -  silent on Pascal SM 6.x, fires on Volta+
    THROTTLING            -  silent on Pascal SM 6.x, fires on Volta+

  Struct: CUpti_ActivityUnifiedMemoryCounter2
    CUPTI 12 returns Counter2 records. Counter (v1) has a different field
    layout  -  casting to the wrong version produces garbage values.

  Relationship to existing thrash detection:
    CUPTI fault counts and migration bytes supplement the CV-based thrash
    score  -  they do not replace it. The CV path remains the primary signal
    on Pascal where THRASHING/THROTTLING counters are unavailable.

  VERDICT SYSTEM
  --------------
    HEALTHY          -  all subsystems nominal, full ratio ladder executed
    HEALTHY_LIMITED  -  ratios clamped by host memory, hardware clean
    MIGRATION_PRESSURE - hardware clean, paired instability signals:
                          (thrash_score >= 0.15 AND maf >= 1.5)
                          OR (settle=NO AND cv_mean > 0.15)
                          maf alone does not trigger — high maf with settled
                          pressure is normal operation on discrete PCIe.
    UM_THRASHING     -  hardware clean, all three required:
                          thrash_score >= 0.45
                          AND maf >= 2.0
                          AND settle=NO
                          Workload-induced boundary condition, not hardware fault.
    DEGRADED         -  physical subsystem fault:
                          thermal throttle or power cap active
                          PCIe replay elevated (>= 100 events)
                          invalid thermal state
    CRITICAL         -  fatal: cold child failures, memory DANGER/CRITICAL,
                          THERMAL_THROTTLE, INVALID_STATE

  EVIDENCE BUNDLE
  ---------------
    runs/um_YYYYMMDD_HHMMSS_GPUx_<uuid>/
      run.json      -  machine-readable full result (schema 2.8)

  SCHEMA 2.7  (additions over 2.6)
  ----------------------------------
    fault_rate:
      fault_pressure_index       -  fault_rate_per_sec * fault_burst_ratio
      residency_half_life_ratio  -  warm-pass residency decay threshold ratio
      migration_efficiency       -  1 / maf
      migration_oscillation_ratio - min(H2D,D2H) / max(H2D,D2H)
    um_intent:
      um_preferred_location      -  cudaMemRangeAttributePreferredLocation (child query)
      um_last_prefetch_location  -  cudaMemRangeAttributeLastPrefetchLocation (child query)
    headroom_and_pressure:
      um_headroom_ratio          -  host_free_gib / host_cap_gib
      llm_pressure_score         -  fault_pressure_index * (1 - migration_efficiency) / um_headroom_ratio
      llm_pressure_level         -  LOW / MODERATE / HIGH
    psi:
      memory_psi_state           -  Linux PSI memory stall classification
      memory_psi_some_avg10_*    -  PSI some-stall 10s averages
      memory_psi_full_avg10_end  -  PSI full-stall 10s average at run end
      memory_psi_some_total_delta_us - cumulative some-stall delta during run

  SCHEMA 2.6  (additions over 2.5)
  ----------------------------------
    fault_latency_distribution:
      steady_max_ms              -  worst observed latency in steady-state repeats
                                    detects page fault storms, OS scheduling delays,
                                    and PCIe replay bursts
      steady_tail_ratio          -  p99 / p50 ratio  -  tail latency severity
                                    PCIe/discrete GPU interpretation:
                                      1.00-1.10 clean, 1.10-1.30 mild pressure,
                                      >1.30 memory pressure, >1.50 thrashing
                                    FULL_HARDWARE_COHERENT (GB10/DGX Spark):
                                      tail reflects TLB shootdown and ATS
                                      translation miss cost — NOT migration.
                                      >1.50 indicates coherence pressure,
                                      not page fault thrashing.
                                    printed as tail=Nx in PASS RESULTS

    fault_rate:
      test_seconds               -  total wall-clock duration of all measurement passes
      fault_rate_per_sec         -  gpu_page_faults / test_seconds  (average)
                                    normal paging: ~100-300 faults/sec
                                    memory pressure: ~1000+ faults/sec
                                    thrashing: explodes with latency spikes
      fault_max_window_rate_per_sec - peak fault rate in any sampled window
                                    sampled at pass boundaries (~100ms min interval)
      fault_burst_ratio          -  fault_max_window_rate / fault_rate_per_sec
                                    1.0 = uniform fault distribution
                                    >3.0 = bursty — faults concentrated in short windows
      fault_pressure_index       -  fault_rate_per_sec * fault_burst_ratio
                                    compact roll-up for cross-run comparison
                                    baseline GTX 1080: ~197  danger band: ~264
      residency_half_life_ratio  -  first warm ratio where residency decay score >= 1.5x baseline
                                    decay_score = steady_tail_ratio * (1 + steady_cv)
                                    null / n/a if no warm pass crossed threshold
      migration_efficiency       -  1 / maf
                                    1.0 = ideal (no amplification)
                                    <0.5 = moderate amplification
                                    <0.3 = heavy amplification / thrash-like movement
      migration_oscillation_ratio - min(H2D,D2H) / max(H2D,D2H)
                                    0.0 = one-direction migration
                                    0.2-0.6 = moderate bidirectional
                                    >0.6 = oscillation  >0.85 = severe ping-pong
      um_preferred_location      -  cudaMemRangeAttributePreferredLocation from last cold pass
                                    GPU / CPU / UNKNOWN / n/a
      um_last_prefetch_location  -  cudaMemRangeAttributeLastPrefetchLocation from last cold pass
                                    GPU / CPU / UNKNOWN / n/a
                                    on FULL_EXPLICIT after prefetch_back_to_gpu this should be GPU
      um_headroom_ratio          -  host_free_gib / host_cap_gib
                                    >2.0 = LARGE  1.3-2.0 = SAFE  1.0-1.3 = LOW  <1.0 = RISK
                                    predicts UM exhaustion risk before workload launch
      llm_pressure_score         -  fault_pressure_index * (1 - migration_efficiency) / um_headroom_ratio
                                    captures paging stress × migration cost ÷ remaining headroom
                                    <100 = LOW  100-200 = MODERATE  >200 = HIGH
      llm_pressure_level         -  LOW / MODERATE / HIGH
      memory_psi_state           -  Linux PSI memory stall classification: LOW / ELEVATED / HIGH / n/a
                                    n/a when /proc/pressure/memory unavailable (kernel config)
      memory_psi_some_avg10_*    -  PSI some-stall 10s average at start/end of run (%)
      memory_psi_full_avg10_end  -  PSI full-stall 10s average at run end (%)
      memory_psi_some_total_delta_us - cumulative some-stall delta during run (microseconds)
      direction_ratio            -  max(H2D,D2H) / min(H2D,D2H)
                                    1.0 = symmetric migration, >1.5 = directional bias
                                    detects asymmetric page migration, NUMA pressure,
                                    UMA coherency issues, and memory thrash patterns
      direction_trend            -  BALANCED (<= 1.50) applies to all platforms.
                                    PCIe/NVLink: H2D_DOMINANT / D2H_DOMINANT
                                      H2D_DOMINANT = normal fault-driven inbound
                                      D2H_DOMINANT = reverse migration, potential thrash
                                    FULL_HARDWARE_COHERENT (GB10/DGX Spark):
                                      GPU_OWNERSHIP_DEMAND / CPU_RETENTION
                                      reflects coherence ownership pattern, not
                                      physical transfer direction (no migration on C2C)

    cold_child_json:
      steady_max_ms and steady_tail_ratio now propagated through cold child
      JSON IPC  -  fixes zero values in cold rows from prior versions

  SCHEMA 2.5  (additions over 2.4)
  ----------------------------------
    cupti:
      cupti_available            -  true if CUPTI Activity API initialised
      cupti_not_supported        -  true if platform returned UM_PROFILING_NOT_SUPPORTED
      cupti_callbacks_failed     -  true if cuptiActivityRegisterCallbacks failed
      cupti_records_total        -  total UM activity records received
      cupti_records_dropped      -  records dropped due to buffer pressure (accumulated)
      cupti_zero_end_ts_skipped  -  records skipped due to zero end timestamp (GB10 bug, confirmed Jul 2025, no fix as of CUDA 13.2)
      cupti_zeroed_record_skipped-  records skipped due to both start=0 and end=0 (buffer artifact)
      cupti_gpu_page_faults      -  GPU page fault groups across all passes
      cupti_cpu_page_faults      -  CPU page fault count — relevant on coherent UMA (GB10)
      cupti_bytes_htod           -  total bytes migrated host->device
      cupti_bytes_dtoh           -  total bytes migrated device->host
      cupti_thrashing_events     -  driver-level ownership ping-pong counter
      cupti_throttling_events    -  memory throttling events — pressure indicator on GB10
                                    zero on Pascal SM 6.x (hardware limitation)
                                    non-zero on Volta+ when CUPTI detects page
                                    ownership flip between CPU and GPU
    migration:
      maf                        -  Migration Amplification Factor: (htod+dtoh)/total_pass_bytes
                                    ~1.0 clean, 2-4 moderate, >4 severe
      bpf_htod_bytes             -  Bytes Per Fault (H2D only): htod/gpu_page_faults
                                    fault-driven inbound migration cost per fault
      bpf_total_bytes            -  Bytes Per Fault (total): (htod+dtoh)/gpu_page_faults
                                    Pascal/PCIe: migration efficiency (larger = efficient bulk copy)
                                    DGX Spark C2C: coherence locality (larger = ownership churn)
                                    Low BPF on discrete GPU = many small fault-triggered copies = thrash
                                    Gap between bpf_htod and bpf_total reveals reverse migration volume
      settled                    -  boolean: true if pressure converged to stable CV
      settle_class               -  STABLE / LATE_UNSTABLE / UNSTABLE
      settle_ms                  -  wall-clock ms from first pressure pass to first stable pass
                                    0.0 when settled=false (never converged)

  SCHEMA 2.4 fields (unchanged)
  ------------------------------
    uma_diagnostics:
      cudamemgetinfo_free        -  raw cudaMemGetInfo free bytes
      cudamemgetinfo_total       -  raw cudaMemGetInfo total bytes
      cudamemgetinfo_delta_gib   -  uma_allocatable minus cmg_free (signed)
      cudamemgetinfo_error_pct   -  delta as % of uma_allocatable
      ceiling_utilization        -  committed / uma_allocatable ratio
      prerun_pressure_verdict    -  CLEAR / ELEVATED / CRITICAL / DANGER
      cache_recoverable_bytes    -  cached + buffers (drop_caches yield)
      swap_disabled              -  SwapTotal == 0
      zombie_oom_structural      -  UMA + swap_disabled + buffer_cache_pressure
    thrash:
      thrash_symmetry            -  min(H2D,D2H)/max(H2D,D2H)
      thrash_cv_mean             -  mean CV across pressure passes (skip pass 0)
      thrash_score               -  cv_instability x fault_density x settling_factor
      thrash_state               -  STABLE / SETTLED_UNSTABLE / UNSTABLE / SEVERE_UNSTABLE
    residency:
      residency_fault_onset_ratio  -  first ratio where FAULT_PATH observed
      residency_resident_gib       -  last resident working set in GiB
      residency_fault_onset_gib    -  allocation size at fault onset

  REFERENCES
  ----------
    NVIDIA CUDA Programming Guide  -  Unified Memory:
      https://docs.nvidia.com/cuda/cuda-programming-guide/
    NVIDIA CUPTI Activity API  -  Unified Memory Counters:
      https://docs.nvidia.com/cupti/r_main.html#r_activity
    NVIDIA DGX Spark Known Issues (Feb 2026):
      https://docs.nvidia.com/dgx/dgx-spark/known-issues.html

  BUILD
  -----
    nvcc -O2 -std=c++17 -o um_analyzer um_analyzer_v8.cu -lcupti -lnvidia-ml

  RUN
  ---
    ./um_analyzer
    ./um_analyzer --all-devices
    ./um_analyzer --list-devices
************************************************************************/

#include <cuda_runtime.h>
#include <cupti.h>
#include <nvml.h>

// Fix #5: CUDA 13 changed cudaMemPrefetchAsync to require a cudaMemLocation
// struct instead of a plain device int. Wrap both paths so the file compiles
// on CUDA 12 and CUDA 13+ without conditional code at every call site.
#if CUDART_VERSION >= 13000
static inline cudaError_t umPrefetchToDevice(const void* p, size_t bytes, int dev, cudaStream_t stream = 0) {
  cudaMemLocation loc = {};
  loc.type = cudaMemLocationTypeDevice;
  loc.id   = dev;
  return cudaMemPrefetchAsync(p, bytes, loc, 0, stream);
}
static inline cudaError_t umPrefetchToCPU(const void* p, size_t bytes, cudaStream_t stream = 0) {
  cudaMemLocation loc = {};
  loc.type = cudaMemLocationTypeHost;
  loc.id   = 0;
  return cudaMemPrefetchAsync(p, bytes, loc, 0, stream);
}
#else
static inline cudaError_t umPrefetchToDevice(const void* p, size_t bytes, int dev, cudaStream_t stream = 0) {
  return cudaMemPrefetchAsync(p, bytes, dev, stream);
}
static inline cudaError_t umPrefetchToCPU(const void* p, size_t bytes, cudaStream_t stream = 0) {
  return cudaMemPrefetchAsync(p, bytes, cudaCpuDeviceId, stream);
}
#endif

// File-scope helper — convert bytes to GiB as double.
// Promoted from main() lambda so it can be used before main() scope.
static inline double gib(uint64_t b) {
  return (double)b / (1024.0 * 1024.0 * 1024.0);
}

#include <algorithm>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifdef __unix__
#include <sys/wait.h>
#endif

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;

// ─── CUPTI Unified Memory counters ────────────────────────────────────────────
struct CuptiUMCounters {
    uint64_t bytes_htod           = 0;
    uint64_t bytes_dtoh           = 0;
    uint64_t gpu_page_faults      = 0;
    uint64_t cpu_page_faults      = 0;   // CPU-side faults — Volta+ only (SM 7.0+)
    uint64_t thrashing            = 0;   // Volta+ only — THRASHING counter kind
    uint64_t throttling           = 0;   // Volta+ only — THROTTLING counter kind
    uint64_t records_total         = 0;
    uint64_t zero_end_ts_skipped   = 0;   // GB10 known bug: end=0 on GPU_PAGE_FAULT records
                                          // confirmed July 2025, no fix as of CUDA 13.2
    uint64_t zeroed_record_skipped = 0;   // fully zeroed record (start=0 && end=0) — buffer artifact
    uint64_t records_dropped       = 0;   // dropped records accumulated across all buffers
    bool     not_supported         = false;
    bool     callbacks_failed      = false;
    bool     volta_plus            = false; // SM 7.0+ — CPU_PAGE_FAULT/THROTTLING counters valid
};

static CuptiUMCounters g_cupti;
static bool            g_cupti_ok    = false;   // set true only if init succeeded
static bool            g_cupti_debug = false;   // --cupti-debug: dump raw UM records to stderr

#define CUPTI_BUF_SIZE  (8 * 1024 * 1024)

static void CUPTIAPI cupti_buf_requested(uint8_t **buf, size_t *size,
                                          size_t *maxNumRecords)
{
    uint8_t *b = nullptr;
    if (posix_memalign((void **)&b, 8, CUPTI_BUF_SIZE) != 0) {
        *buf = nullptr; *size = 0; return;
    }
    *buf = b; *size = CUPTI_BUF_SIZE; *maxNumRecords = 0;
}

static void CUPTIAPI cupti_buf_completed(CUcontext ctx, uint32_t streamId,
                                          uint8_t *buffer, size_t /*size*/,
                                          size_t validSize)
{
    CUptiResult status;
    CUpti_Activity *record = nullptr;
    do {
        status = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (status != CUPTI_SUCCESS) break;
        if (record->kind != CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER) continue;
        auto *um = (CUpti_ActivityUnifiedMemoryCounter2 *)record;
        // Fully zeroed record — buffer initialization artifact, skip entirely
        if (um->start == 0 && um->end == 0) {
            g_cupti.zeroed_record_skipped++;
            continue;
        }
        // GB10 known bug: GPU_PAGE_FAULT records may have end=0 (invalid timestamp)
        // Only skip on GPU_PAGE_FAULT — transfer records (HTOD/DTOH) legitimately have end=0
        // on Pascal and other platforms; skipping them would lose byte counts.
        if (um->counterKind == CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT
            && um->end == 0 && um->start != 0) {
            g_cupti.zero_end_ts_skipped++;
            continue;
        }
        g_cupti.records_total++;
        if (g_cupti_debug) {
            static constexpr auto kind_str = [](CUpti_ActivityUnifiedMemoryCounterKind k) -> const char* {
                switch (k) {
                case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD:  return "BYTES_HTOD";
                case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:  return "BYTES_DTOH";
                case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT: return "CPU_PAGE_FAULT";
                case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT:       return "GPU_PAGE_FAULT";
                case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING:            return "THRASHING";
                case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING:           return "THROTTLING";
                default: return "UNKNOWN";
                }
            };
            fprintf(stderr,
                "[CUPTI_DBG] kind=%-16s  value=%-12llu  start=%-20llu  end=%-20llu\n"
                "            processId=%-6u\n",
                kind_str(um->counterKind),
                (unsigned long long)um->value,
                (unsigned long long)um->start,
                (unsigned long long)um->end,
                
                
                (unsigned)um->processId);
        }
        switch (um->counterKind) {
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD:
            g_cupti.bytes_htod      += um->value; break;
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:
            g_cupti.bytes_dtoh      += um->value; break;
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT:
            g_cupti.gpu_page_faults += um->value; break;
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT:
            g_cupti.cpu_page_faults += um->value; break;
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING:
            g_cupti.thrashing       += um->value; break;
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING:
            g_cupti.throttling      += um->value; break;
        default: break;
        }
    } while (true);
    size_t dropped = 0;
    cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped);
    if (dropped > 0) {
        g_cupti.records_dropped += dropped;
        fprintf(stderr, "[cupti] WARNING: %zu records dropped (total=%zu)\n",
                dropped, (size_t)g_cupti.records_dropped);
    }
    free(buffer);
}

static void cupti_init(int device_id, const std::string& paradigm, int sm_major)
{
    // CPU_PAGE_FAULT_COUNT and THROTTLING are only valid on Volta+ (SM 7.0+).
    // On Pascal (SM 6.x) CUPTI accepts the config but emits records with garbage
    // in um->value — uninitialized memory, not a real counter.  Gate them out.
    const bool volta_plus = (sm_major >= 7);
    g_cupti.volta_plus = volta_plus;

    CUpti_ActivityUnifiedMemoryCounterConfig cfg[5];
    cfg[0] = { CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE,
               CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD,
               (uint32_t)device_id, 1 };
    cfg[1] = { CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE,
               CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH,
               (uint32_t)device_id, 1 };
    cfg[2] = { CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE,
               CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT,
               (uint32_t)device_id, 1 };
    cfg[3] = { CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE,
               CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT,
               (uint32_t)device_id, (uint32_t)volta_plus };
    cfg[4] = { CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE,
               CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING,
               (uint32_t)device_id, (uint32_t)volta_plus };

    if (cuptiActivityRegisterCallbacks(cupti_buf_requested,
                                       cupti_buf_completed) != CUPTI_SUCCESS) {
        g_cupti.callbacks_failed = true;
        fprintf(stderr, "[cupti] WARNING: cuptiActivityRegisterCallbacks failed -- "
                        "CUPTI counters will not be collected\n");
        return;
    }
    cuptiActivityConfigureUnifiedMemoryCounter(cfg, 5);   // non-fatal if partial
    CUptiResult enable_status = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER);
    if (enable_status != CUPTI_SUCCESS) {
        if (enable_status == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED ||
            enable_status == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE) {
            g_cupti.not_supported = true;
            if (paradigm == "FULL_HARDWARE_COHERENT") {
                fprintf(stderr, "[cupti] UVM profiling not supported on this platform -- "
                                "FULL_HARDWARE_COHERENT (coherent UMA, GB10/DGX Spark) does not "
                                "generate discrete page fault/migration events. "
                                "CUPTI counters will not be collected. "
                                "This is expected on current driver versions.\n");
            } else {
                fprintf(stderr, "[cupti] UVM profiling not supported on this platform "
                                "(paradigm=%s, driver limitation) -- "
                                "CUPTI counters will not be collected\n",
                                paradigm.c_str());
            }
        }
        return;
    }
    g_cupti_ok = true;
}

static void cupti_flush()
{
    if (!g_cupti_ok) return;
    cuptiActivityFlushAll(1);
}

static void cupti_teardown()
{
    if (!g_cupti_ok) return;
    cuptiActivityFlushAll(1);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER);
}
// ──────────────────────────────────────────────────────────────────────────────

static inline double ms(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

static inline std::string now_local_compact() {
  std::time_t t = std::time(nullptr);
  std::tm lt{};
#if defined(_WIN32)
  localtime_s(&lt, &t);
#else
  localtime_r(&t, &lt);
#endif
  std::ostringstream o;
  o << std::setfill('0')
    << (lt.tm_year + 1900)
    << std::setw(2) << (lt.tm_mon + 1)
    << std::setw(2) << lt.tm_mday
    << "_"
    << std::setw(2) << lt.tm_hour
    << std::setw(2) << lt.tm_min
    << std::setw(2) << lt.tm_sec;
  return o.str();
}

static inline std::string json_escape(const std::string& s) {
  std::ostringstream o;
  for (char c : s) {
    switch (c) {
      case '\"': o << "\\\""; break;
      case '\\': o << "\\\\"; break;
      case '\n': o << "\\n"; break;
      case '\r': o << "\\r"; break;
      case '\t': o << "\\t"; break;
      default:
        if ((unsigned char)c < 0x20) {
          o << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << (int)(unsigned char)c << std::dec;
        } else {
          o << c;
        }
    }
  }
  return o.str();
}

// Decode std::system() status on POSIX.
static inline int system_exit_code(int status) {
#ifdef __unix__
  if (status == -1) return 127;
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
  return 126;
#else
  return status;
#endif
}

// ------------------- Host info (RAM/swap/THP) -------------------

struct HostInfo {
  uint64_t mem_total = 0;
  uint64_t mem_avail = 0;   // MemAvailable
  uint64_t swap_total = 0;
  uint64_t swap_free  = 0;
  // Buffer cache fields (silent memory thief on UMA/DGX Spark)
  uint64_t cached     = 0;  // Cached (page cache)
  uint64_t buffers    = 0;  // Buffers
  // HugeTLB fields (NVIDIA DGX Spark known-issues guidance)
  long hugetlb_total  = -1; // HugePages_Total
  long hugetlb_free   = -1; // HugePages_Free
  long hugetlb_size   = -1; // Hugepagesize (kB)
  std::string thp = "unknown";
  // Paradigm-aware allocatable memory (computed after paradigm detection).
  // FULL_HARDWARE_COHERENT: MemAvailable + SwapFree per NVIDIA guidance.
  // FULL_EXPLICIT / others:  MemAvailable (backing store cap).
  uint64_t uma_allocatable = 0;
  bool     cudamemgetinfo_unreliable = false;

  // cudaMemGetInfo delta — how much cudaMemGetInfo underreports on UMA.
  // Populated after cudaSetDevice, before any allocation.
  // Positive = CMG is hiding allocatable memory (expected on DGX Spark).
  // Negative = CMG sees more than /proc (should not happen; flag if seen).
  uint64_t  cudamemgetinfo_free  = 0;
  uint64_t  cudamemgetinfo_total = 0;
  int64_t   cudamemgetinfo_delta = 0;   // uma_allocatable - cmg_free (signed)
  double    cudamemgetinfo_error_pct = 0.0;

  // Pre-run memory pressure ceiling.
  // committed = MemTotal - MemAvailable - SwapFree  (pages in active use).
  // ceiling_utilization = committed / uma_allocatable.
  double      ceiling_utilization = 0.0;
  bool        overcommit            = false;  // committed > uma_allocatable
  std::string prerun_pressure_verdict = "UNKNOWN";

  // Buffer cache recovery — what drop_caches would free right now.
  uint64_t cache_recoverable_bytes = 0;  // cached + buffers
  double   cache_frac_of_total     = 0.0;
  bool     buffer_cache_pressure   = false;

  // Structural OOM risk — specific to FULL_HARDWARE_COHERENT with no swap.
  // When true: if workload > MemAvailable the kernel page allocator will
  // stall.  The OOM killer will not fire cleanly.  Hard reboot required.
  bool swap_disabled         = false;
  bool zombie_oom_structural = false;
};

// ── PSI Memory Pressure Snapshot ─────────────────────────────────────────────
// Reads /proc/pressure/memory if available (Linux 4.20+, PSI enabled kernel).
// Fields: avg10, avg60, avg300 (% stall time), total (µs cumulative)
// Returns empty strings when PSI is unavailable.
struct PsiSnapshot {
  double some_avg10  = -1.0;  // -1 = not available
  double some_avg60  = -1.0;
  double full_avg10  = -1.0;
  uint64_t some_total = 0;
  bool available = false;
};

static PsiSnapshot read_psi_memory() {
  PsiSnapshot s;
  std::ifstream f("/proc/pressure/memory");
  if (!f.is_open()) return s;
  std::string line;
  while (std::getline(f, line)) {
    // some avg10=0.00 avg60=0.00 avg300=0.00 total=12345
    // full avg10=0.00 avg60=0.00 avg300=0.00 total=12345
    auto get_val = [&](const std::string& key) -> double {
      auto pos = line.find(key + "=");
      if (pos == std::string::npos) return -1.0;
      pos += key.size() + 1;
      try { return std::stod(line.substr(pos)); } catch (...) { return -1.0; }
    };
    auto get_total = [&]() -> uint64_t {
      auto pos = line.find("total=");
      if (pos == std::string::npos) return 0;
      pos += 6;
      try { return std::stoull(line.substr(pos)); } catch (...) { return 0; }
    };
    if (line.rfind("some", 0) == 0) {
      s.some_avg10  = get_val("avg10");
      s.some_avg60  = get_val("avg60");
      s.some_total  = get_total();
      s.available   = (s.some_avg10 >= 0.0);
    } else if (line.rfind("full", 0) == 0) {
      s.full_avg10  = get_val("avg10");
    }
  }
  return s;
}

static std::string psi_state(const PsiSnapshot& start, const PsiSnapshot& end) {
  if (!start.available || !end.available) return "n/a";
  // classify by delta total stall (µs) and end avg10
  uint64_t delta = (end.some_total >= start.some_total)
                   ? (end.some_total - start.some_total) : 0;
  double avg10 = end.some_avg10;
  if (avg10 > 5.0 || delta > 500000) return "HIGH";
  if (avg10 > 1.0 || delta > 100000) return "ELEVATED";
  return "LOW";
}

static HostInfo read_host_info() {
  HostInfo h;
  std::ifstream f("/proc/meminfo");
  std::string line;
  while (std::getline(f, line)) {
    long lval = 0;
    char lkey[64] = {};
    if (sscanf(line.c_str(), "%63s %ld", lkey, &lval) < 2) continue;
    std::string k(lkey);
    if      (k == "MemTotal:")        h.mem_total  = (uint64_t)lval * 1024ull;
    else if (k == "MemAvailable:")    h.mem_avail  = (uint64_t)lval * 1024ull;
    else if (k == "SwapTotal:")       h.swap_total = (uint64_t)lval * 1024ull;
    else if (k == "SwapFree:")        h.swap_free  = (uint64_t)lval * 1024ull;
    else if (k == "Cached:")          h.cached     = (uint64_t)lval * 1024ull;
    else if (k == "Buffers:")         h.buffers    = (uint64_t)lval * 1024ull;
    else if (k == "HugePages_Total:") h.hugetlb_total = lval;
    else if (k == "HugePages_Free:")  h.hugetlb_free  = lval;
    else if (k == "Hugepagesize:")    h.hugetlb_size  = lval;
  }
  std::ifstream thp("/sys/kernel/mm/transparent_hugepage/enabled");
  if (thp.good()) std::getline(thp, h.thp);
  return h;
}

// Compute paradigm-aware allocatable memory and all derived pressure metrics.
// Must be called after um_paradigm is known and host fields are populated.
//
// Implements NVIDIA DGX Spark known-issues guidance (Feb 2026):
//   On FULL_HARDWARE_COHERENT systems, cudaMemGetInfo ignores SWAP-reclaimable
//   pages and reports only unallocated pool fraction.  True allocatable memory
//   is MemAvailable + SwapFree.  HugeTLB pages are not swappable; when
//   HugePages_Total > 0 the allocatable is HugePages_Free * Hugepagesize.
//
// Pre-run pressure verdicts (ceiling_utilization = committed / uma_allocatable):
//   CLEAR    : < 0.50  — comfortable headroom
//   ELEVATED : 0.50–0.70  — monitor, should be fine for moderate workloads
//   CRITICAL : 0.70–0.85  — workload may trigger swap pressure or stall
//   DANGER   : > 0.85  — high probability of UMA hang on large allocation
//
// Structural zombie OOM:
//   On FULL_HARDWARE_COHERENT with SwapTotal=0 and buffer_cache_pressure,
//   the kernel page allocator will stall when the workload exceeds
//   MemAvailable.  The OOM killer does not fire cleanly.  This condition
//   is deterministic, not probabilistic.  Classified as zombie_oom_structural.
static void compute_uma_allocatable(HostInfo& h, const std::string& um_paradigm) {

  // ---- Step 1: paradigm-aware allocatable ceiling ----
  if (um_paradigm == "FULL_HARDWARE_COHERENT") {
    h.cudamemgetinfo_unreliable = true;
    if (h.hugetlb_total > 0) {
      // HugeTLB active: pages are not swappable (NVIDIA's own branch)
      h.uma_allocatable = (uint64_t)(h.hugetlb_free * h.hugetlb_size) * 1024ull;
    } else {
      h.uma_allocatable = h.mem_avail + h.swap_free;
    }
  } else {
    // FULL_EXPLICIT / FULL_SOFTWARE_COHERENT / LIMITED:
    // GPU has dedicated VRAM; host RAM is backing store for oversubscription.
    h.cudamemgetinfo_unreliable = false;
    h.uma_allocatable = h.mem_avail;
  }

  // ---- Step 2: buffer cache pressure ----
  h.cache_recoverable_bytes = h.cached + h.buffers;
  if (h.mem_total > 0) {
    h.cache_frac_of_total = (double)h.cache_recoverable_bytes / (double)h.mem_total;
    h.buffer_cache_pressure = (h.cache_frac_of_total > 0.20);
  }

  // ---- Step 3: pre-run ceiling utilization ----
  // committed = mem_total - mem_avail (active pages, excludes reclaimable cache)
  // Do NOT subtract swap_free — swap availability does not reduce RAM commitment.
  if (h.uma_allocatable > 0) {
    uint64_t committed = (h.mem_total > h.mem_avail)
                         ? (h.mem_total - h.mem_avail) : 0;
    double raw_util = (double)committed / (double)h.uma_allocatable;
    h.overcommit = (raw_util > 1.0);
    h.ceiling_utilization = std::min(1.0, raw_util);

    if      (h.ceiling_utilization < 0.50) h.prerun_pressure_verdict = "CLEAR";
    else if (h.ceiling_utilization < 0.70) h.prerun_pressure_verdict = "ELEVATED";
    else if (h.ceiling_utilization < 0.85) h.prerun_pressure_verdict = "CRITICAL";
    else                                   h.prerun_pressure_verdict = "DANGER";
  }

  // ---- Step 4: structural zombie OOM classification ----
  h.swap_disabled = (h.swap_total == 0);
  // Structural: UMA platform + no swap + active buffer cache pressure.
  // Without swap, the kernel cannot shed page cache under GPU allocation
  // pressure.  The allocator stalls.  OOM killer does not respond cleanly.
  h.zombie_oom_structural = (um_paradigm == "FULL_HARDWARE_COHERENT")
                           && h.swap_disabled
                           && h.buffer_cache_pressure;
}

// Populate cudaMemGetInfo delta fields.
// Called after cudaSetDevice(), before any cudaMalloc/cudaMallocManaged.
// On FULL_EXPLICIT (discrete GPU), delta will be near zero — correct.
// On FULL_HARDWARE_COHERENT (DGX Spark), delta will be large and positive —
// cudaMemGetInfo ignores swap-reclaimable pages and returns only the
// unallocated fraction of the unified pool.

static void measure_cudamemgetinfo_delta(HostInfo& h) {
  size_t cmg_free = 0, cmg_total = 0;
  cudaMemGetInfo(&cmg_free, &cmg_total);
  h.cudamemgetinfo_free  = (uint64_t)cmg_free;
  h.cudamemgetinfo_total = (uint64_t)cmg_total;
  h.cudamemgetinfo_delta = (int64_t)h.uma_allocatable - (int64_t)cmg_free;
  if (h.uma_allocatable > 0)
    h.cudamemgetinfo_error_pct =
      (double)h.cudamemgetinfo_delta / (double)h.uma_allocatable * 100.0;
}

static inline uint64_t env_u64(const char* k, uint64_t defv=0) {
  const char* v = std::getenv(k);
  if (!v || !*v) return defv;
  return std::strtoull(v, nullptr, 10);
}

// Host cap: conservative usable backing for UM oversubscription.
// (UM can oversubscribe VRAM, but host RAM backing still matters.)
static uint64_t compute_host_cap_bytes(const HostInfo& host) {
  uint64_t forced = env_u64("UM_HOST_CAP_BYTES", 0);
  if (forced) return forced;

  const uint64_t reserve = 2ull * 1024ull * 1024ull * 1024ull; // keep OS breathing
  const double headroom = 0.70;

  if (host.mem_avail <= reserve) return 0;
  return (uint64_t)((double)(host.mem_avail - reserve) * headroom);
}

// ------------------- NVML helpers -------------------

struct NvmlInfo {
  bool ok = false;
  std::string driver;
  std::string name;
  std::string uuid;
  unsigned int pcie_gen_curr = 0;
  unsigned int pcie_width_curr = 0;
  unsigned int pcie_gen_max = 0;
  unsigned int pcie_width_max = 0;
  uint64_t vram_total = 0;
  uint64_t vram_free = 0;
  std::string mem_type = "UNKNOWN";   // GDDR6, GDDR6X, HBM2, HBM3, etc.
  unsigned int mem_bus_width = 0;     // memory bus width in bits
  // Unified Memory paradigm — determined from cudaDeviceGetAttribute queries
  // after NVML init. Possible values:
  //   FULL_EXPLICIT         — cudaDevAttrConcurrentManagedAccess=1, PageableMemoryAccess=0
  //                           Software-coherent, only cudaMallocManaged is UM (Pascal, Volta, etc.)
  //   FULL_SOFTWARE_COHERENT — ConcurrentManaged=1, Pageable=1, PageableUsesHostPageTables=0
  //                           HMM: all system memory is UM, software page table coherence
  //   FULL_HARDWARE_COHERENT — ConcurrentManaged=1, Pageable=1, PageableUsesHostPageTables=1
  //                           ATS/C2C: combined CPU+GPU page table (Grace Hopper, DGX Spark)
  //   LIMITED               — ConcurrentManaged=0 (Windows, WSL, Tegra)
  //   UNKNOWN               — attribute query failed
  std::string um_paradigm = "UNKNOWN";
  std::string arch_name   = "UNKNOWN";  // populated after cudaSetDevice via compute capability
  // NVLink state — populated during read_nvml()
  bool     nvlink_any_active = false;
  unsigned nvlink_active_links = 0;
};

// NVML session — kept open for the duration of the run so hardware state
// captures (temperature, clocks, throttle reasons) can reuse the handle.
struct NvmlSession {
  nvmlDevice_t dev{};
  bool open = false;
  ~NvmlSession() { if (open) nvmlShutdown(); }
};
static NvmlSession g_nvml;  // process-lifetime NVML session

static NvmlInfo read_nvml(int device_index) {
  NvmlInfo n;
  if (nvmlInit() != NVML_SUCCESS) return n;
  g_nvml.open = true;

  char drv[96];
  if (nvmlSystemGetDriverVersion(drv, sizeof(drv)) == NVML_SUCCESS) n.driver = drv;

  nvmlDevice_t dev;
  if (nvmlDeviceGetHandleByIndex(device_index, &dev) != NVML_SUCCESS) {
    nvmlShutdown(); g_nvml.open = false;
    return n;
  }
  g_nvml.dev = dev;  // persist for hardware state captures

  char name[96];
  if (nvmlDeviceGetName(dev, name, sizeof(name)) == NVML_SUCCESS) n.name = name;

  char uuid[96];
  if (nvmlDeviceGetUUID(dev, uuid, sizeof(uuid)) == NVML_SUCCESS) n.uuid = uuid;

  nvmlDeviceGetCurrPcieLinkGeneration(dev, &n.pcie_gen_curr);
  nvmlDeviceGetCurrPcieLinkWidth(dev, &n.pcie_width_curr);
  nvmlDeviceGetMaxPcieLinkGeneration(dev, &n.pcie_gen_max);
  nvmlDeviceGetMaxPcieLinkWidth(dev, &n.pcie_width_max);

  nvmlMemory_t mem{};
  if (nvmlDeviceGetMemoryInfo(dev, &mem) == NVML_SUCCESS) {
    n.vram_total = mem.total;
    n.vram_free  = mem.free;
  }

  // Fix #1: UMA fallback — NVML reports vram_total=0 on FULL_HARDWARE_COHERENT
  // platforms (DGX Spark GB10) because there is no discrete framebuffer.
  // Fall back to cudaMemGetInfo which reports the unified pool size.
  // Note: cudaMemGetInfo underreports reclaimable memory on UMA — this value
  // is used only for display and ratio ladder sizing, not for allocatable
  // ceiling logic (which uses MemAvailable + SwapFree via compute_uma_allocatable).
  if (n.vram_total == 0) {
    size_t cmg_free = 0, cmg_total = 0;
    if (cudaMemGetInfo(&cmg_free, &cmg_total) == cudaSuccess && cmg_total > 0) {
      n.vram_total = (uint64_t)cmg_total;
      n.vram_free  = (uint64_t)cmg_free;
    }
  }

  // Memory type — inferred from bus width + VRAM size heuristic.
  // NVML has no direct memory type query on all drivers; bus width
  // is the reliable signal.  HBM uses narrow stacked dies (4096-bit
  // total across stacks) vs GDDR which uses 256/320/384-bit buses.
  nvmlDeviceGetMemoryBusWidth(dev, &n.mem_bus_width);
  {
    double vram_gib = (double)n.vram_total / (1024.0*1024.0*1024.0);
    if (n.mem_bus_width >= 4096) {
      n.mem_type = "HBM";
    } else if (n.mem_bus_width >= 512) {
      n.mem_type = "HBM2e";
    } else if (n.mem_bus_width == 384) {
      n.mem_type = "GDDR6X";       // RTX 3090/4090
    } else if (n.mem_bus_width == 320) {
      n.mem_type = "GDDR6X";       // RTX 3080
    } else if (n.mem_bus_width == 256) {
      // 256-bit: GDDR5X on Pascal (GTX 1080/1080Ti), GDDR6 on Turing+
      // Distinguish by PCI device ID heuristic via name string
      bool is_pascal = (name[0] != '\0' &&
                        (std::string(name).find("1080") != std::string::npos ||
                         std::string(name).find("1070") != std::string::npos ||
                         std::string(name).find("Titan X") != std::string::npos));
      n.mem_type = is_pascal ? "GDDR5X" : "GDDR6";
    } else if (vram_gib > 64.0 && n.mem_bus_width == 0) {
      // Fix #2: UMA platform (DGX Spark GB10) — NVML reports 0 bus width.
      // Shared LPDDR5X pool, no discrete memory bus.
      n.mem_type = "LPDDR5X";
    } else if (n.mem_bus_width == 0) {
      // bus_width==0 on non-UMA platform — unknown, do not guess
      n.mem_type = "UNKNOWN";
    } else {
      n.mem_type = "GDDR5";
    }
  }

  n.ok = true;

  // NVLink detection — probe up to 6 links (max on most architectures)
  for (unsigned lnk = 0; lnk < 6; ++lnk) {
    nvmlEnableState_t state = NVML_FEATURE_DISABLED;
    if (nvmlDeviceGetNvLinkState(dev, lnk, &state) == NVML_SUCCESS
        && state == NVML_FEATURE_ENABLED) {
      n.nvlink_any_active = true;
      n.nvlink_active_links++;
    }
  }

  // NOTE: do NOT call nvmlShutdown() here — g_nvml destructor handles it
  return n;
}

// EffectiveMemoryView — single authoritative memory object.
// On FULL_HARDWARE_COHERENT (DGX Spark): total = MemAvailable + SwapFree,
//   free = MemAvailable. NVML is non-authoritative for memory on UMA.
// On discrete GPU: total = nv.vram_total, free = nv.vram_free (NVML).
// All banner, verdict, JSON, and ratio ladder logic routes through emv.
struct EffectiveMemoryView {
  uint64_t    total_bytes = 0;
  uint64_t    free_bytes  = 0;
  const char* label       = "VRAM";
  const char* source      = "nvml";
  bool        fused       = false;
};

static EffectiveMemoryView build_effective_memory_view(
    const NvmlInfo& nv,
    const HostInfo& host,
    const std::string& um_paradigm)
{
  EffectiveMemoryView m{};
  const bool coherent_uma = (um_paradigm == "FULL_HARDWARE_COHERENT");

  if (coherent_uma) {
    m.label       = "Unified";
    m.source      = "host_fused";
    m.fused       = true;
    m.free_bytes  = host.mem_avail;
    m.total_bytes = host.uma_allocatable;
  } else {
    m.label       = "VRAM";
    m.source      = "nvml";
    m.fused       = false;
    m.free_bytes  = nv.vram_free;
    m.total_bytes = nv.vram_total;
  }
  return m;
}

// =====================================================================
// TRANSPORT LAYER CLASSIFICATION
// =====================================================================
// Classifies the physical path GPU memory operations traverse.
// Drives pass labels, metric suppression, and output interpretation.
//
//   PCIe     : discrete GPU, migrations over PCIe bus
//   NVLink   : discrete multi-GPU, migrations may use NVLink
//   Coherent : FULL_HARDWARE_COHERENT — no migration, TLB cost only
//   Unknown  : detection failed

enum class TransportLayer { PCIe, NVLink, Coherent, Unknown };

static std::string transport_str(TransportLayer t) {
  switch (t) {
    case TransportLayer::PCIe:     return "PCIe";
    case TransportLayer::NVLink:   return "NVLink";
    case TransportLayer::Coherent: return "Coherent (C2C/ATS)";
    default:                       return "Unknown";
  }
}

struct TransportInfo {
  TransportLayer layer            = TransportLayer::Unknown;
  unsigned       nvlink_links     = 0;
  bool           nvswitch_detected = false;  // TODO: requires DCGM
  std::string    note;
};

static TransportInfo classify_transport(const std::string& um_paradigm,
                                        unsigned nvlink_active_links) {
  TransportInfo t;
  t.nvlink_links = nvlink_active_links;
  if (um_paradigm == "FULL_HARDWARE_COHERENT") {
    t.layer = TransportLayer::Coherent;
    t.note  = "CPU+GPU share DRAM via C2C interconnect — no page migration";
  } else if (nvlink_active_links > 0) {
    t.layer = TransportLayer::NVLink;
    t.note  = std::to_string(nvlink_active_links) +
              " active NVLink link(s) — peer migrations may bypass PCIe";
  } else {
    t.layer = TransportLayer::PCIe;
    t.note  = "All managed memory migrations traverse PCIe bus";
  }
  return t;
}

// =====================================================================
// NVLINK HEALTH COUNTERS
// =====================================================================
// Per-link CRC FLIT and replay error counts via NVML field queries.
// On hardware without NVLink all counters return zero — safe, no crash.
//
// NVML field IDs (driver 525+):
//   NVML_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L0 = 49  (L0..L5)
//   NVML_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L0   = 55  (L0..L5)

static constexpr unsigned MAX_NVLINK_LINKS = 6;

struct NVLinkCounters {
  uint64_t crc_flit[MAX_NVLINK_LINKS]  = {};
  uint64_t replay[MAX_NVLINK_LINKS]    = {};
  uint64_t util_tx[MAX_NVLINK_LINKS]   = {};  // TX utilization counter
  uint64_t util_rx[MAX_NVLINK_LINKS]   = {};  // RX utilization counter
  uint64_t err_dl[MAX_NVLINK_LINKS]    = {};  // data link errors per link
  uint64_t total_crc_flit  = 0;
  uint64_t total_replay    = 0;
  uint64_t total_util_tx   = 0;
  uint64_t total_util_rx   = 0;
  uint64_t total_err_dl    = 0;
  bool     supported       = false;
  bool     query_ok        = false;
  unsigned links_probed    = 0;
  // CLEAN / DEGRADED / NO_NVLINK / ERROR
  std::string verdict = "NO_NVLINK";
};

static NVLinkCounters query_nvlink_counters(nvmlDevice_t dev) {
  NVLinkCounters c;
  const unsigned N = MAX_NVLINK_LINKS * 2;
  nvmlFieldValue_t fv[N];
  memset(fv, 0, sizeof(fv));
  const unsigned CRC_BASE    = 49;
  const unsigned REPLAY_BASE = 55;
  for (unsigned lnk = 0; lnk < MAX_NVLINK_LINKS; ++lnk) {
    fv[lnk].fieldId                    = CRC_BASE + lnk;
    fv[MAX_NVLINK_LINKS + lnk].fieldId = REPLAY_BASE + lnk;
  }
  nvmlReturn_t ret = nvmlDeviceGetFieldValues(dev, (int)N, fv);
  if (ret == NVML_ERROR_NOT_SUPPORTED) {
    c.supported = false; c.query_ok = true; c.verdict = "NO_NVLINK";
    return c;
  }
  if (ret != NVML_SUCCESS) {
    c.supported = false; c.query_ok = false; c.verdict = "ERROR";
    return c;
  }
  // Pascal accepts the call but has no NVLink — gate on link state
  {
    unsigned active = 0;
    for (unsigned lnk = 0; lnk < MAX_NVLINK_LINKS; ++lnk) {
      nvmlEnableState_t state = NVML_FEATURE_DISABLED;
      if (nvmlDeviceGetNvLinkState(dev, lnk, &state) == NVML_SUCCESS
          && state == NVML_FEATURE_ENABLED)
        ++active;
    }
    if (active == 0) {
      c.supported = false; c.query_ok = true; c.verdict = "NO_NVLINK";
      return c;
    }
  }
  c.supported = true; c.query_ok = true; c.links_probed = MAX_NVLINK_LINKS;
  for (unsigned lnk = 0; lnk < MAX_NVLINK_LINKS; ++lnk) {
    if (fv[lnk].nvmlReturn == NVML_SUCCESS) {
      c.crc_flit[lnk]   = fv[lnk].value.ullVal;
      c.total_crc_flit += fv[lnk].value.ullVal;
    }
    unsigned ri = MAX_NVLINK_LINKS + lnk;
    if (fv[ri].nvmlReturn == NVML_SUCCESS) {
      c.replay[lnk]    = fv[ri].value.ullVal;
      c.total_replay  += fv[ri].value.ullVal;
    }
    // NVLink utilization counters — TX/RX bandwidth counters per link
    unsigned long long tx = 0, rx = 0;
    if (nvmlDeviceGetNvLinkUtilizationCounter(dev, lnk, 0, &rx, &tx) == NVML_SUCCESS) {
      c.util_tx[lnk]   = (uint64_t)tx;
      c.util_rx[lnk]   = (uint64_t)rx;
      c.total_util_tx += (uint64_t)tx;
      c.total_util_rx += (uint64_t)rx;
    }
    // NVLink error counters — data link layer errors per link
    unsigned long long dl_err = 0;
    if (nvmlDeviceGetNvLinkErrorCounter(dev, lnk,
        NVML_NVLINK_ERROR_DL_REPLAY, &dl_err) == NVML_SUCCESS) {
      c.err_dl[lnk]   = (uint64_t)dl_err;
      c.total_err_dl += (uint64_t)dl_err;
    }
  }
  c.verdict = (c.total_crc_flit == 0 && c.total_replay == 0 && c.total_err_dl == 0)
              ? "CLEAN" : "DEGRADED";
  return c;
}

// =====================================================================
// TRANSPORT-AWARE PASS LABELS
// =====================================================================
// Label strings for COLD/WARM passes driven by transport layer.
// PCIe: page-fault path / resident path
// NVLink: fault->NVLink migration / local-resident
// Coherent: TLB cold / TLB warm (no migration on this platform)

struct PassLabels {
  std::string cold;
  std::string warm;
  std::string cold_note;
  std::string warm_note;
};

static PassLabels make_pass_labels(TransportLayer layer) {
  PassLabels p;
  switch (layer) {
    case TransportLayer::PCIe:
      p.cold      = "COLD (page-fault path)";
      p.warm      = "WARM (resident path)";
      p.cold_note = "fault -> PCIe DMA -> GPU VRAM";
      p.warm_note = "pages GPU-resident at 2MB granularity";
      break;
    case TransportLayer::NVLink:
      p.cold      = "COLD (fault -> NVLink migration path)";
      p.warm      = "WARM (local-resident path)";
      p.cold_note = "fault -> NVLink peer migration (bypasses PCIe)";
      p.warm_note = "pages local-GPU-resident — NVLink not in migration path";
      break;
    case TransportLayer::Coherent:
      p.cold      = "COLD (TLB cold / first-access)";
      p.warm      = "WARM (TLB warm / ATS cached)";
      p.cold_note = "TLB miss cost — no PCIe or NVLink migration on this platform";
      p.warm_note = "ATS translation cached — direct shared DRAM access";
      break;
    default:
      p.cold      = "COLD (fault path)";
      p.warm      = "WARM (resident path)";
      p.cold_note = "transport unknown";
      p.warm_note = "transport unknown";
      break;
  }
  return p;
}

// Determine the UM paradigm for a CUDA device using the three canonical
// attributes from the CUDA Programming Guide (§2.4.2.1, Table 3).
// Must be called after cudaSetDevice.
static std::string query_um_paradigm(int device) {
  int concurrent = 0, pageable = 0, uses_host_pt = 0;
  bool c_ok = (cudaDeviceGetAttribute(&concurrent,
                 cudaDevAttrConcurrentManagedAccess, device) == cudaSuccess);
  bool p_ok = (cudaDeviceGetAttribute(&pageable,
                 cudaDevAttrPageableMemoryAccess, device) == cudaSuccess);
  bool h_ok = (cudaDeviceGetAttribute(&uses_host_pt,
                 cudaDevAttrPageableMemoryAccessUsesHostPageTables, device) == cudaSuccess);

  if (!c_ok) return "UNKNOWN";
  if (concurrent == 0) return "LIMITED";

  // concurrent == 1 from here
  if (!p_ok || pageable == 0) return "FULL_EXPLICIT";

  // pageable == 1: all system memory is UM
  if (!h_ok) return "FULL_SOFTWARE_COHERENT";  // conservative fallback
  return (uses_host_pt == 1) ? "FULL_HARDWARE_COHERENT" : "FULL_SOFTWARE_COHERENT";
}

// ------------------- Hardware State (telemetry validity layer) -------------------

struct HardwareState {
  unsigned int temp_idle_c   = 0;   // before CUDA init — true pre-run baseline
  unsigned int temp_start_c  = 0;   // after context init, before warm pass
  unsigned int temp_end_c    = 0;   // after pressure pass
  int          temp_drift_c  = 0;   // end - start
  unsigned int pstate_start  = 99;  // P-state (0=P0 max perf, 8=P8 idle)
  unsigned int sm_clock_start_mhz  = 0;
  unsigned int mem_clock_start_mhz = 0;
  unsigned long long throttle_reasons_start = 0;
  unsigned long long throttle_reasons_end   = 0;
  unsigned int throttle_threshold_c = 80;  // architecture-aware, set after arch detection
  std::string thermal_verdict = "UNKNOWN";
  // Power draw — discrete GPU only (FULL_EXPLICIT / FULL_HMM).
  // On UMA (DGX Spark GB10): nvmlDeviceGetPowerUsage returns NOT_SUPPORTED.
  // Power cap enforcement is via firmware; thermal verdict is the correct signal.
  unsigned int power_draw_mw    = 0;   // milliwatts, pre-run
  unsigned int power_limit_mw   = 0;   // enforced cap
  bool         power_available  = false;
  bool valid   = false;
  bool is_uma  = false;   // true on FULL_HARDWARE_COHERENT (DGX Spark GB10)
};

// Architecture-aware throttle onset threshold.
// Sources: NVIDIA thermal design docs per architecture class.
static unsigned int throttle_threshold_from_arch(const std::string& arch) {
  if (arch == "Pascal"  || arch == "Turing" ||
      arch == "Ada"     || arch == "Ampere_consumer")  return 83;
  if (arch == "Ampere"  || arch == "Hopper" ||
      arch == "Blackwell")                             return 80;
  return 80;  // conservative universal fallback
}

// Decode throttle reason bitmask to human-readable string.
// Strips GPU_IDLE — expected between passes, not a validity concern.
static std::string throttle_reasons_str(unsigned long long reasons) {
  if (reasons == 0) return "NONE";
  std::string s;
  // Mask out GPU_IDLE — not actionable
  unsigned long long active = reasons & ~nvmlClocksThrottleReasonGpuIdle;
  if (active == 0) return "GPU_IDLE_ONLY";
  if (active & nvmlClocksThrottleReasonSwThermalSlowdown)  s += "SW_THERMAL|";
  if (active & nvmlClocksThrottleReasonHwThermalSlowdown)  s += "HW_THERMAL|";
  if (active & nvmlClocksThrottleReasonSwPowerCap)         s += "SW_POWER_CAP|";
  if (active & nvmlClocksThrottleReasonHwSlowdown)         s += "HW_SLOWDOWN|";
  if (active & nvmlClocksThrottleReasonSyncBoost)          s += "SYNC_BOOST|";
  if (active & nvmlClocksThrottleReasonDisplayClockSetting) s += "DISPLAY_CLK|";
  if (s.empty()) s = "OTHER|";
  if (!s.empty() && s.back() == '|') s.pop_back();
  return s;
}

// Compute thermal verdict — priority ordered, highest severity wins.
// Fix #3: On FULL_HARDWARE_COHERENT (DGX Spark GB10), NVML always reports
// HwSlowdown and SwPowerCap as active because the SoC power management is
// handled differently from discrete GPUs — these are not real throttle events.
// Gate those bits behind !hw.is_uma. Only thermal reasons apply on UMA.
static std::string compute_thermal_verdict(const HardwareState& hw) {
  unsigned long long active_start = hw.throttle_reasons_start
                                    & ~nvmlClocksThrottleReasonGpuIdle;
  unsigned long long active_end   = hw.throttle_reasons_end
                                    & ~nvmlClocksThrottleReasonGpuIdle;

  // INVALID_STATE: throttled before first measurement began.
  // On UMA: only thermal reasons are real. HwSlowdown/SwPowerCap are always
  // set by the SoC and must be ignored or every run returns INVALID_STATE.
  if (hw.is_uma) {
    unsigned long long thermal_only = nvmlClocksThrottleReasonHwThermalSlowdown |
                                      nvmlClocksThrottleReasonSwThermalSlowdown;
    if (active_start & thermal_only)
      return "INVALID_STATE";
  } else {
    if (active_start & (nvmlClocksThrottleReasonSwPowerCap |
                        nvmlClocksThrottleReasonHwThermalSlowdown |
                        nvmlClocksThrottleReasonSwThermalSlowdown |
                        nvmlClocksThrottleReasonHwSlowdown))
      return "INVALID_STATE";
  }

  // POWER_CAP_ACTIVE: power cap enforced during run (skip on UMA — always set)
  if (!hw.is_uma && (active_end & (nvmlClocksThrottleReasonSwPowerCap |
                                   nvmlClocksThrottleReasonHwSlowdown)))
    return "POWER_CAP_ACTIVE";

  // THERMAL_THROTTLE: peak hit hardware throttle threshold
  if (hw.temp_end_c >= hw.throttle_threshold_c)
    return "THERMAL_THROTTLE";

  // THERMAL_DRIFT: large drift or approaching throttle ceiling
  if (hw.temp_drift_c > 15 ||
      (int)hw.temp_end_c >= (int)hw.throttle_threshold_c - 5)
    return "THERMAL_DRIFT";

  // THERMAL_NOMINAL: some heating, no impact
  if (hw.temp_drift_c > 5)
    return "THERMAL_NOMINAL";

  // THERMAL_STABLE: clean run
  return "THERMAL_STABLE";
}

// Capture one temperature sample from NVML.
static unsigned int nvml_temp(nvmlDevice_t dev) {
  unsigned int t = 0;
  nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &t);
  return t;
}

// Populate full hardware state. Call read_hw_state_pre() before CUDA work,
// read_hw_state_post() after all passes complete.
static void read_hw_state_pre(HardwareState& hw, nvmlDevice_t dev,
                               const std::string& arch,
                               const std::string& um_paradigm = "") {
  hw.throttle_threshold_c = throttle_threshold_from_arch(arch);
  hw.is_uma = (um_paradigm == "FULL_HARDWARE_COHERENT");
  hw.temp_idle_c  = nvml_temp(dev);
  nvmlPstates_t pstate_raw = NVML_PSTATE_UNKNOWN;
  nvmlDeviceGetPowerState(dev, &pstate_raw);
  hw.pstate_start = (unsigned int)pstate_raw;
  nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM,  &hw.sm_clock_start_mhz);
  nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM, &hw.mem_clock_start_mhz);
  nvmlDeviceGetCurrentClocksThrottleReasons(dev, &hw.throttle_reasons_start);
  // Power draw — discrete GPU only. On UMA (DGX Spark GB10 / Grace Hopper),
  // nvmlDeviceGetPowerUsage returns NOT_SUPPORTED — skip rather than emit N/A.
  bool is_uma = (um_paradigm == "FULL_HARDWARE_COHERENT" ||
                 um_paradigm == "FULL_HARDWARE");
  if (!is_uma) {
    unsigned int pwr = 0, lim = 0;
    bool pwr_ok = (nvmlDeviceGetPowerUsage(dev, &pwr) == NVML_SUCCESS);
    bool lim_ok = (nvmlDeviceGetEnforcedPowerLimit(dev, &lim) == NVML_SUCCESS);
    if (pwr_ok || lim_ok) {
      hw.power_draw_mw   = pwr;
      hw.power_limit_mw  = lim;
      hw.power_available = true;
    }
  }
  hw.valid = true;
}

static void read_hw_state_start(HardwareState& hw, nvmlDevice_t dev) {
  // Called after CUDA context init, just before warm pass
  hw.temp_start_c = nvml_temp(dev);
}

static void read_hw_state_post(HardwareState& hw, nvmlDevice_t dev) {
  hw.temp_end_c   = nvml_temp(dev);
  hw.temp_drift_c = (int)hw.temp_end_c - (int)hw.temp_start_c;
  nvmlDeviceGetCurrentClocksThrottleReasons(dev, &hw.throttle_reasons_end);
  hw.thermal_verdict = compute_thermal_verdict(hw);
}

static inline std::string short_uuid(const std::string& u) {
  if (u.empty()) return "no_uuid";
  std::string s = u;
  if (s.size() > 12) s = s.substr(0, 12);
  for (char& c : s) if (c == ':') c = '_';
  return s;
}

// PCIe practical payload-ish estimate (deterministic heuristic)
static inline double expected_pcie_gbs_practical(unsigned int gen, unsigned int width) {
  double per_lane = 0.0;
  if (gen == 5) per_lane = 3.938;
  else if (gen == 4) per_lane = 1.969;
  else if (gen == 3) per_lane = 0.985;
  else if (gen == 2) per_lane = 0.492;
  else if (gen == 1) per_lane = 0.250;
  return per_lane * (double)width;
}

// ------------------- CUDA kernels -------------------

__global__ void noop_kernel() {}

__global__ void page_touch_kernel(float* p, size_t n, size_t stride) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t i = idx * stride;
  if (i < n) p[i] += 1.0f;
}

static inline double cuda_context_warmup_ms() {
  auto t0 = clk::now();
  noop_kernel<<<1,1>>>();
  cudaDeviceSynchronize();
  auto t1 = clk::now();
  return ms(t0, t1);
}

static inline double gpu_touch_ms(float* p, size_t n, size_t stride) {
  auto t0 = clk::now();
  page_touch_kernel<<<256,256>>>(p, n, stride);
  cudaDeviceSynchronize();
  auto t1 = clk::now();
  return ms(t0, t1);
}

static inline std::vector<double> steady_samples(float* p, size_t n, size_t stride, int reps) {
  std::vector<double> v; v.reserve(reps);
  for (int i = 0; i < reps; ++i) v.push_back(gpu_touch_ms(p, n, stride));
  std::sort(v.begin(), v.end());
  return v;
}

static inline double pct_sorted(const std::vector<double>& s, double q) {
  if (s.empty()) return 0.0;
  double pos = q * (s.size() - 1);
  size_t i = (size_t)pos;
  size_t j = std::min(i + 1, s.size() - 1);
  double frac = pos - (double)i;
  return s[i] * (1.0 - frac) + s[j] * frac;
}

static inline double mean(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  return std::accumulate(v.begin(), v.end(), 0.0) / (double)v.size();
}

static inline double coeff_var(const std::vector<double>& v) {
  if (v.size() < 2) return 0.0;
  double m = mean(v);
  if (m <= 0.0) return 0.0;
  double s2 = 0.0;
  for (double x : v) s2 += (x - m) * (x - m);
  s2 /= (double)v.size();
  return std::sqrt(s2) / m;
}

// ------------------- Intelligence -------------------

struct KneeResult {
  double ratio = -1.0;
  std::string metric = "none";
  double confidence = 0.0;
};


static inline std::string classify_regime(double steady_jump, double steady_cv,
                                         double gpu_retouch_ms, double steady_p50_ms,
                                         bool is_cold = false) {
  // FAULT_PATH: cold run without prefetch — pages faulted during measurement.
  // Higher latency and CV are expected signal, not anomaly.
  // Threshold: p50 > 0.12ms (above kernel launch floor) OR cv > 0.15
  if (is_cold && (steady_p50_ms > 0.12 || steady_cv > 0.15)) return "FAULT_PATH";
  // RESIDENT: all pages fit in VRAM, no migration under load.
  if (steady_jump < 5.0) return "RESIDENT";
  // TRANSITION: moderate oversubscription, some migration occurring.
  if (steady_jump < 50.0) return "TRANSITION";
  // PINGPONG_SUSPECT requires confirmed migration (jump >= 50) AND high variance
  // or excessive retouch. Jump alone is heavy migration, not thrash.
  if (steady_cv > 1.0 && steady_jump >= 50.0) return "PINGPONG_SUSPECT";
  if (gpu_retouch_ms > std::max(2.0, 5.0 * steady_p50_ms)
      && steady_jump >= 50.0)                 return "PINGPONG_SUSPECT";
  return "MIGRATION_HEAVY";
}


// ------------------- Ratio ladder (hardware-aware) -------------------

static std::vector<double> adaptive_ratios(double vram_gib) {
  if (vram_gib <= 8.5)  return {0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00};
  if (vram_gib <= 24.5) return {0.25, 0.50, 0.75, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00};
  return {0.50, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.50, 2.00};
}

// ------------------- Evidence bundle dir -------------------

static fs::path make_run_dir(int device, const std::string& uuid_short) {
  fs::path base("runs");
  fs::create_directories(base);
  std::ostringstream o;
  o << "um_" << now_local_compact() << "_GPU" << device << "_" << uuid_short;
  fs::path d = base / o.str();
  fs::create_directories(d);
  return d;
}

// ------------------- Result record -------------------

struct PhaseTimes {
  double context_warmup_ms = 0.0;
  double alloc_ms = 0.0;
  double cpu_init_ms = 0.0;

  double prefetch_to_gpu_ms = 0.0;
  double gpu_first_touch_ms = 0.0;

  double steady_p50_ms = 0.0;
  double steady_p90_ms = 0.0;
  double steady_p99_ms = 0.0;
  double steady_max_ms = 0.0;
  double steady_tail_ratio = 0.0;
  double steady_mean_ms = 0.0;
  double steady_cv = 0.0;

  double prefetch_to_cpu_ms = 0.0;
  double cpu_retouch_ms = 0.0;
  double prefetch_back_to_gpu_ms = 0.0;
  double gpu_retouch_ms = 0.0;

  double pressure_p50_ms = 0.0;
  double pressure_score = 0.0;

  std::string dominant_total;
  std::string dominant_um;
};

static void compute_dominant(PhaseTimes& t) {
  struct P { const char* k; double v; bool um; };
  const P ps[] = {
    {"context_warmup_ms", t.context_warmup_ms, false},
    {"alloc_ms", t.alloc_ms, false},
    {"cpu_init_ms", t.cpu_init_ms, false},

    {"prefetch_to_gpu_ms", t.prefetch_to_gpu_ms, true},
    {"gpu_first_touch_ms", t.gpu_first_touch_ms, true},
    {"steady_p50_ms", t.steady_p50_ms, true},
    {"prefetch_to_cpu_ms", t.prefetch_to_cpu_ms, true},
    {"cpu_retouch_ms", t.cpu_retouch_ms, false},
    {"prefetch_back_to_gpu_ms", t.prefetch_back_to_gpu_ms, true},
    {"gpu_retouch_ms", t.gpu_retouch_ms, true},

    {"pressure_p50_ms", t.pressure_p50_ms, true},
  };

  int besti = 0;
  for (int i = 1; i < (int)(sizeof(ps)/sizeof(ps[0])); ++i)
    if (ps[i].v > ps[besti].v) besti = i;

  {
    std::ostringstream o;
    o << ps[besti].k << "(" << std::fixed << std::setprecision(3) << ps[besti].v << ")";
    t.dominant_total = o.str();
  }

  double bestv = -1.0;
  const char* bestk = "none";
  for (auto& p : ps) {
    if (!p.um) continue;
    if (p.v > bestv) { bestv = p.v; bestk = p.k; }
  }
  std::ostringstream u;
  u << bestk << "(" << std::fixed << std::setprecision(3) << bestv << ")";
  t.dominant_um = u.str();
}

struct ResultRow {
  std::string status = "ok";
  double oversub_ratio = 0.0;
  uint64_t bytes = 0;
  uint64_t pages = 0;

  PhaseTimes t;

  std::string regime = "UNKNOWN";
  double steady_jump = 1.0;
  double stability_index = 1.0;

  double migration_gbs_prefetch_gpu = 0.0;
  double pcie_expected_gbs = 0.0;
  double pcie_efficiency = 0.0;

  std::string um_preferred_location = "";  // GPU / CPU / UNKNOWN / n/a
  std::string um_last_prefetch      = "";  // GPU / CPU / UNKNOWN / n/a

  // Per-child CUPTI observability (cold pass only, from child IPC)
  uint64_t child_gpu_faults  = 0;
  uint64_t child_bytes_htod  = 0;
  uint64_t child_bytes_dtoh  = 0;
};


// Fault rate sampler — lightweight snapshot of gpu_page_faults at a point in time.
// Sampled at pass boundaries (no threads, no callbacks).
struct FaultSample {
  clk::time_point ts;
  uint64_t gpu_faults = 0;
};

// Dedicated PCIe bandwidth probe — pinned cudaMemcpy measures real link capacity.
struct PcieBwResult {
  double h2d_gbs = 0.0;
  double d2h_gbs = 0.0;
  bool   ok      = false;
};

static PcieBwResult probe_pcie_bandwidth(int device,
    size_t transfer_bytes = 256ULL*1024*1024) {
  PcieBwResult r;
  void* host_pinned = nullptr;
  void* dev_buf     = nullptr;
  if (cudaMallocHost(&host_pinned, transfer_bytes) != cudaSuccess) return r;
  if (cudaMalloc(&dev_buf, transfer_bytes) != cudaSuccess) {
    cudaFreeHost(host_pinned); return r;
  }
  memset(host_pinned, 1, transfer_bytes);
  cudaMemset(dev_buf, 0, transfer_bytes);
  cudaDeviceSynchronize();
  auto h0 = clk::now();
  cudaMemcpy(dev_buf, host_pinned, transfer_bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  auto h1 = clk::now();
  auto d0 = clk::now();
  cudaMemcpy(host_pinned, dev_buf, transfer_bytes, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  auto d1 = clk::now();
  cudaFreeHost(host_pinned);
  cudaFree(dev_buf);
  double h2d_sec = ms(h0, h1) / 1000.0;
  double d2h_sec = ms(d0, d1) / 1000.0;
  if (h2d_sec > 0.0) r.h2d_gbs = ((double)transfer_bytes / h2d_sec) / (1024.0*1024.0*1024.0);
  if (d2h_sec > 0.0) r.d2h_gbs = ((double)transfer_bytes / d2h_sec) / (1024.0*1024.0*1024.0);
  r.ok = true;
  return r;
}

// ── Residency Window Detection ───────────────────────────────────────────────
// Scans cold_rows to find the first ratio where FAULT_PATH appears.
// Reports the effective GPU working-set capacity before migration onset.

struct ResidencyWindow {
  double resident_ratio   = -1.0;  // last RESIDENT ratio seen
  double fault_onset_ratio = -1.0; // first FAULT_PATH ratio seen
  double resident_gib     = 0.0;   // bytes at resident_ratio
  double fault_onset_gib  = 0.0;   // bytes at fault_onset_ratio
  bool   detected         = false; // true if fault onset was observed
};

static ResidencyWindow compute_residency_window(
    const std::vector<ResultRow>& cold_rows, double vram_gib) {
  ResidencyWindow w;
  for (const auto& r : cold_rows) {
    double row_gib = (double)r.bytes / (1024.0 * 1024.0 * 1024.0);
    if (r.regime == "RESIDENT" && !w.detected) {
      // Only update resident baseline before any fault is seen
      w.resident_ratio = r.oversub_ratio;
      w.resident_gib   = row_gib;
    } else if (r.regime == "FAULT_PATH" && !w.detected) {
      w.fault_onset_ratio = r.oversub_ratio;
      w.fault_onset_gib   = row_gib;
      w.detected = true;
    }
  }
  return w;
}

// ── Thrash Detection ─────────────────────────────────────────────────────────
// Converts existing CV history, settling result, and PCIe symmetry into
// an explicit thrash verdict. No new hardware queries — uses what v6 measures.

struct ThrashMetrics {
  double symmetry             = 0.0;   // H2D/D2H byte symmetry [0..1]; 1.0 = perfect bounce
  double cv_instability       = 0.0;   // mean CV across pressure passes
  double repeat_fault_density = 0.0;   // fraction of cold rows that were FAULT_PATH [0..1]
  bool   settled              = false; // pressure converged (stable_pass >= 0)
  double h2d_gbs              = 0.0;   // from PCIe probe
  double d2h_gbs              = 0.0;   // from PCIe probe
  double thrash_score         = 0.0;   // composite [0..1]; higher = more thrashing
  std::string state           = "UNKNOWN"; // STABLE / UNSTABLE / SEVERE_UNSTABLE
};

// cv_history: array of PRESSURE_REPEATS CVs from pressure phase
// stable_pass: -1 = never settled, 0+ = settled by that pass
// h2d_gbs, d2h_gbs: from PCIe probe (pinned cudaMemcpy)
static ThrashMetrics classify_thrash(const double* cv_history, int cv_count,
                                     int stable_pass,
                                     double h2d_gbs, double d2h_gbs,
                                     const std::vector<ResultRow>& cold_rows) {
  ThrashMetrics m;
  m.h2d_gbs = h2d_gbs;
  m.d2h_gbs = d2h_gbs;

  // 1. Settling
  m.settled = (stable_pass >= 0);

  // 2. PCIe symmetry — stored for display only.
  // NOTE: probe traffic is a dedicated pinned cudaMemcpy, not migration traffic.
  // Symmetry from this source is not a reliable thrash signal without CUPTI.
  // Displayed informally; excluded from thrash_score until CUPTI is available.
  double lo = std::min(h2d_gbs, d2h_gbs);
  double hi = std::max(h2d_gbs, d2h_gbs);
  m.symmetry = (hi > 0.01) ? (lo / hi) : 0.0;

  // 3. CV instability — use settled CV, not mean across all passes.
  // Pass 0 is always noisy (GPU warmup transient). Using the mean drags
  // up the score on runs that settled cleanly after the first pass.
  // Rule: if cv_count >= 2, skip pass 0 and use mean of remaining passes.
  //       if cv_count == 1, use that value.
  if (cv_count >= 2) {
    double cv_sum = 0.0;
    for (int i = 1; i < cv_count; ++i) cv_sum += cv_history[i];
    m.cv_instability = cv_sum / (cv_count - 1);
  } else if (cv_count == 1) {
    m.cv_instability = cv_history[0];
  }

  // 4. Repeat fault density — fraction of cold rows classified as FAULT_PATH
  if (!cold_rows.empty()) {
    int fault_count = 0;
    for (const auto& r : cold_rows)
      if (r.regime == "FAULT_PATH") fault_count++;
    m.repeat_fault_density = (double)fault_count / (double)cold_rows.size();
  }

  // 5. Thrash score: cv_instability x fault_density x settling_factor
  // Symmetry is stored and displayed but NOT used in the score.
  // Reason: PCIe probe is a synthetic back-to-back transfer that always
  // reads ~symmetric (~0.99). It does not measure migration traffic.
  // Real migration symmetry requires CUPTI activity records.
  // When CUPTI is added, restore: score *= (0.5 + m.symmetry)
  double settling_factor = m.settled ? 0.4 : 1.0;
  m.thrash_score = std::min(1.0,
      m.cv_instability * m.repeat_fault_density
      * settling_factor * 5.0);

  // 6. State classification
  if (m.thrash_score < 0.15)
    m.state = "STABLE";
  else if (m.thrash_score < 0.45)
    m.state = (m.settled ? "SETTLED_UNSTABLE" : "UNSTABLE");
  else
    m.state = "SEVERE_UNSTABLE";

  return m;
}


static ResultRow measure_ratio(double ratio, uint64_t bytes, int device,
                               unsigned int pcie_gen, unsigned int pcie_width,
                               int reps, bool do_pressure,
                               bool skip_prefetch = false) {
  ResultRow row;
  row.oversub_ratio = ratio;
  row.bytes = bytes;

  const size_t stride_bytes = 4096;
  const size_t stride = stride_bytes / sizeof(float);

  size_t n = (size_t)(bytes / sizeof(float));
  if (n < stride * 1024) { row.status = "too_small"; return row; }
  row.pages = (uint64_t)(n / stride);

  row.t.context_warmup_ms = cuda_context_warmup_ms();

  float* p = nullptr;

  auto a0 = clk::now();
  cudaError_t ce = cudaMallocManaged(&p, bytes);
  auto a1 = clk::now();
  row.t.alloc_ms = ms(a0, a1);
  if (ce != cudaSuccess || !p) { row.status = "alloc_failed"; return row; }

  auto b0 = clk::now();
  for (size_t i = 0; i < n; i += stride) p[i] = 1.0f;
  auto b1 = clk::now();
  row.t.cpu_init_ms = ms(b0, b1);

  auto c0 = clk::now();
  if (!skip_prefetch) {
    ce = umPrefetchToDevice(p, bytes, device);
    cudaDeviceSynchronize();
  }
  auto c1 = clk::now();
  row.t.prefetch_to_gpu_ms = skip_prefetch ? 0.0 : ms(c0, c1);

  row.t.gpu_first_touch_ms = gpu_touch_ms(p, n, stride);

  auto steady = steady_samples(p, n, stride, reps);
  row.t.steady_p50_ms = pct_sorted(steady, 0.50);
  row.t.steady_p90_ms = pct_sorted(steady, 0.90);
  row.t.steady_p99_ms = pct_sorted(steady, 0.99);
  row.t.steady_max_ms = steady.empty() ? 0.0 : steady.back();
  row.t.steady_tail_ratio = row.t.steady_p99_ms / std::max(0.001, row.t.steady_p50_ms);
  row.t.steady_mean_ms = mean(steady);
  row.t.steady_cv = coeff_var(steady);

  if (do_pressure) {
    const int cycles = 6;
    std::vector<double> pt; pt.reserve(cycles);
    for (int c = 0; c < cycles; ++c) {
      for (size_t i = 0; i < n; i += stride) p[i] += 1.0f;
      cudaDeviceSynchronize();
      pt.push_back(gpu_touch_ms(p, n, stride));
    }
    std::sort(pt.begin(), pt.end());
    row.t.pressure_p50_ms = pct_sorted(pt, 0.50);
    double base = std::max(0.001, row.t.steady_p50_ms);
    row.t.pressure_score = row.t.pressure_p50_ms / base;
  }

  auto e0 = clk::now();
  ce = umPrefetchToCPU(p, bytes);
  cudaDeviceSynchronize();
  auto e1 = clk::now();
  row.t.prefetch_to_cpu_ms = ms(e0, e1);

  auto rt0 = clk::now();
  for (size_t i = 0; i < n; i += stride) p[i] += 1.0f;
  auto rt1 = clk::now();
  row.t.cpu_retouch_ms = ms(rt0, rt1);

  auto fb0 = clk::now();
  ce = umPrefetchToDevice(p, bytes, device);
  cudaDeviceSynchronize();
  auto fb1 = clk::now();
  row.t.prefetch_back_to_gpu_ms = ms(fb0, fb1);

  row.t.gpu_retouch_ms = gpu_touch_ms(p, n, stride);

  // UM intent query — preferred location and last prefetch target
  // Queried after all GPU work so last_prefetch reflects final prefetch_back_to_gpu
  {
    auto loc_to_str = [](int dev) -> std::string {
      if (dev == cudaCpuDeviceId) return "CPU";
      if (dev >= 0)               return "GPU";
      return "UNKNOWN";
    };
    int preferred = -2, last_pf = -2;
    cudaMemRangeGetAttribute(&preferred, sizeof(preferred),
                             cudaMemRangeAttributePreferredLocation, p, bytes);
    cudaMemRangeGetAttribute(&last_pf, sizeof(last_pf),
                             cudaMemRangeAttributeLastPrefetchLocation, p, bytes);
    row.um_preferred_location = (preferred == -2) ? "n/a" : loc_to_str(preferred);
    row.um_last_prefetch       = (last_pf  == -2) ? "n/a" : loc_to_str(last_pf);
  }

  cudaFree(p);

  double sec = row.t.prefetch_to_gpu_ms / 1000.0;
  if (sec > 0.001) row.migration_gbs_prefetch_gpu = ((double)bytes / sec) / (1024.0*1024.0*1024.0);
  row.pcie_expected_gbs = expected_pcie_gbs_practical(pcie_gen, pcie_width);
  if (row.pcie_expected_gbs > 0.1) row.pcie_efficiency = row.migration_gbs_prefetch_gpu / row.pcie_expected_gbs;

  compute_dominant(row.t);
  return row;
}

// ------------------- Artifacts -------------------

struct KneeResult2 {
  double ratio = -1.0;
  std::string metric = "none";
  double confidence = 0.0;
};

static KneeResult2 detect_knee2(const std::vector<double>& ratios,
                               const std::vector<double>& steady_p50_ms) {
  KneeResult2 k;
  if (ratios.size() != steady_p50_ms.size() || ratios.size() < 3) return k;
  double base = steady_p50_ms[0];
  if (base <= 0) return k;

  for (size_t i = 0; i + 1 < ratios.size(); ++i) {
    double j0 = steady_p50_ms[i] / base;
    double j1 = steady_p50_ms[i + 1] / base;
    if (j0 >= 50.0 && j1 >= 50.0) {
      k.ratio = ratios[i];
      k.metric = "steady_p50_jump";
      k.confidence = std::min(1.0, j0 / 50.0);
      return k;
    }
  }
  return k;
}

static void write_summary_txt(const fs::path& run_dir,
                              const NvmlInfo& nv,
                              const HostInfo& host,
                              double host_cap_gib,
                              const KneeResult2& cold_k,
                              const KneeResult2& warm_k,
                              bool pressure_enabled,
                              int cold_child_failures,
                              const double* pressure_cv_history,
                              int pressure_repeats,
                              double pressure_cv_drop_pct,
                              int pressure_stable_pass,
                              bool pressure_is_fault_migration,
                              const HardwareState& hw) {
  std::ofstream o(run_dir / "summary.txt");
  o << "UM Analyzer Summary\n";
  o << "timestamp_local=" << now_local_compact() << "\n";
  o << "gpu=" << nv.name << " uuid=" << nv.uuid << "\n";
  o << "um_paradigm=" << nv.um_paradigm << "\n";
  o << "pcie=Gen" << nv.pcie_gen_curr << " x" << nv.pcie_width_curr
    << " (max Gen" << nv.pcie_gen_max << " x" << nv.pcie_width_max << ")\n";
  o << "vram_total_gib=" << std::fixed << std::setprecision(3)
    << ((double)nv.vram_total / (1024.0*1024.0*1024.0)) << "\n";
  o << "host_free_gib_approx=" << ((double)host.mem_avail / (1024.0*1024.0*1024.0)) << "\n";
  o << "host_cap_gib=" << host_cap_gib << "\n";
  o << "swap_total_gib=" << ((double)host.swap_total / (1024.0*1024.0*1024.0)) << "\n";
  o << "swap_free_gib="  << ((double)host.swap_free  / (1024.0*1024.0*1024.0)) << "\n";
  o << "thp=" << host.thp << "\n";
  o << "cold_knee_ratio=" << cold_k.ratio << " metric=" << cold_k.metric << " conf=" << cold_k.confidence << "\n";
  o << "warm_knee_ratio=" << warm_k.ratio << " metric=" << warm_k.metric << " conf=" << warm_k.confidence << "\n";
  o << "pressure_enabled=" << (pressure_enabled ? "true" : "false") << "\n";
  o << "cold_child_failures=" << cold_child_failures << "\n";

  // Pressure stability block
  if (pressure_enabled && pressure_repeats >= 2) {
    o << "\nPRESSURE STABILITY (" << pressure_repeats << " identical passes at max ratio):\n";
    for (int i = 0; i < pressure_repeats; ++i) {
      o << "  Pass " << (i + 1) << " CV: " << std::fixed << std::setprecision(3)
        << pressure_cv_history[i] << "\n";
    }
    o << "  Initial -> final CV drop: " << std::setprecision(1)
      << pressure_cv_drop_pct * 100.0 << "%\n";
    if (pressure_stable_pass > 0)
      o << "  Stabilized by pass " << pressure_stable_pass << "\n";
    else
      o << "  Did not stabilize within " << pressure_repeats << " passes\n";

    if (pressure_is_fault_migration) {
      if (pressure_cv_drop_pct > 0.15 && pressure_stable_pass > 0) {
        o << "  Interpretation: kernel page reclaim settles after initial exposure.\n";
        o << "  For capacity planning, use steady-state values (pass "
          << pressure_stable_pass << "+), not first-pass measurements.\n";
        o << "  NOTE: three passes use identical protocol; CV drop is not a protocol\n";
        o << "  artifact. Controlled result.\n";
      } else if (pressure_cv_drop_pct <= 0.05) {
        o << "  Interpretation: CV is stable across passes. No kernel learning effect\n";
        o << "  detected. First-pass measurements are representative.\n";
      } else {
        o << "  Interpretation: partial convergence. More passes may be needed to\n";
        o << "  confirm steady state.\n";
      }
    } else if (pressure_cv_history[0] > 0.10) {
      // RESIDENT regime but high first-pass CV — GPU hardware warm-up transient
      o << "  Regime: RESIDENT (all pages fit in VRAM, no migration faults)\n";
      if (pressure_cv_drop_pct > 0.15 && pressure_stable_pass > 0) {
        o << "  Interpretation: software-coherent page table (" << nv.um_paradigm << " paradigm)\n";
        o << "  requires driver-managed TLB population on first access, causing elevated\n";
        o << "  first-pass variance even when all pages fit in VRAM.\n";
        o << "  Steady-state reached by pass " << pressure_stable_pass << ".\n";
        o << "  On FULL_HARDWARE_COHERENT systems (ATS/C2C) this transient may differ\n";
        o << "  due to combined CPU+GPU page table.\n";
        o << "  Recommendation: discard first measurement in all GPU benchmarks.\n";
      } else {
        o << "  Interpretation: partial warm-up convergence observed.\n";
      }
    } else {
      o << "  Note: first-pass CV below threshold (0.10); hardware already settled.\n";
    }
  }

  // Hardware state validity block
  o << "\nHARDWARE STATE:\n";
  o << "  temp_idle=" << hw.temp_idle_c << "C"
    << "  temp_start=" << hw.temp_start_c << "C"
    << "  temp_end=" << hw.temp_end_c << "C"
    << "  drift=" << (hw.temp_drift_c >= 0 ? "+" : "") << hw.temp_drift_c << "C\n";
  o << "  pstate_start=P" << hw.pstate_start
    << "  sm_clk=" << hw.sm_clock_start_mhz << "MHz"
    << "  mem_clk=" << hw.mem_clock_start_mhz << "MHz\n";
  o << "  throttle_start=" << throttle_reasons_str(hw.throttle_reasons_start) << "\n";
  o << "  throttle_end="   << throttle_reasons_str(hw.throttle_reasons_end)   << "\n";
  o << "  throttle_threshold=" << hw.throttle_threshold_c << "C\n";
  o << "  thermal_verdict=" << hw.thermal_verdict << "\n";
  if (hw.thermal_verdict == "INVALID_STATE")
    o << "  [WARNING] GPU throttled before measurements. Results may be invalid.\n";
  else if (hw.thermal_verdict == "POWER_CAP_ACTIVE")
    o << "  [WARNING] Power cap active during run. Migration bandwidth may be reduced.\n";
  else if (hw.thermal_verdict == "THERMAL_THROTTLE")
    o << "  [WARNING] Peak temp reached throttle threshold. Timing affected.\n";

  // DGX Spark / UMA memory diagnostics block — schema 2.4
  if (host.cudamemgetinfo_unreliable) {
    o << "\nUMA DIAGNOSTICS (" << nv.um_paradigm << "):\n";
    o << "  cudaMemGetInfo: UNRELIABLE on this platform\n";
    o << "  uma_allocatable_gib=" << std::fixed << std::setprecision(3)
      << (double)host.uma_allocatable / (1024.0*1024.0*1024.0) << "\n";
    o << "  cudamemgetinfo_free_gib=" << std::setprecision(3)
      << (double)host.cudamemgetinfo_free / (1024.0*1024.0*1024.0) << "\n";
    double dg = (double)host.cudamemgetinfo_delta / (1024.0*1024.0*1024.0);
    o << "  cudamemgetinfo_delta_gib=" << (dg >= 0 ? "+" : "") << std::setprecision(3) << dg << "\n";
    o << "  cudamemgetinfo_error_pct=" << std::setprecision(1) << host.cudamemgetinfo_error_pct << "%\n";
    if (host.hugetlb_total > 0)
      o << "  hugetlb_active=true  hugetlb_free_pages=" << host.hugetlb_free << "\n";
    else
      o << "  method=MemAvailable+SwapFree (NVIDIA DGX Spark known-issues guidance)\n";
  }
  o << "\nPRE-RUN PRESSURE:\n";
  o << "  ceiling_utilization=" << std::fixed << std::setprecision(1)
    << host.ceiling_utilization * 100.0 << "%"
    << "  verdict=" << host.prerun_pressure_verdict << "\n";
  if (host.buffer_cache_pressure) {
    o << "\nBUFFER_CACHE_PRESSURE:\n";
    o << "  cache_recoverable_gib=" << std::setprecision(3)
      << (double)host.cache_recoverable_bytes / (1024.0*1024.0*1024.0) << "\n";
    o << "  cache_frac_of_total=" << std::setprecision(1) << host.cache_frac_of_total * 100.0 << "%\n";
    o << "  Fix: sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'\n";
  }
  if (host.zombie_oom_structural) {
    o << "\nZOMBIE_OOM_STRUCTURAL:\n";
    o << "  swap=NONE + UMA + buffer_cache_pressure\n";
    o << "  Without swap, the kernel page allocator will stall when workload\n";
    o << "  exceeds MemAvailable. OOM killer will not fire cleanly.\n";
    o << "  Hard reboot required if this condition triggers during a workload.\n";
    o << "  Mitigation: run drop_caches before workload, or add swap.\n";
  }
}

struct KneeBlock { double ratio; std::string metric; double conf; };

static void dump_rows(std::ostream& j, const char* name, const std::vector<ResultRow>& rows) {
  j << "    \"" << name << "\": [\n";
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto& r = rows[i];
    j << "      {\n";
    j << "        \"status\": \"" << json_escape(r.status) << "\",\n";
    j << "        \"oversub_ratio\": " << std::fixed << std::setprecision(3) << r.oversub_ratio << ",\n";
    j << "        \"bytes\": " << r.bytes << ",\n";
    j << "        \"pages\": " << r.pages << ",\n";
    j << "        \"times_ms\": {\n";
    j << "          \"context_warmup_ms\": " << r.t.context_warmup_ms << ",\n";
    j << "          \"alloc_ms\": " << r.t.alloc_ms << ",\n";
    j << "          \"cpu_init_ms\": " << r.t.cpu_init_ms << ",\n";
    j << "          \"prefetch_to_gpu_ms\": " << r.t.prefetch_to_gpu_ms << ",\n";
    j << "          \"gpu_first_touch_ms\": " << r.t.gpu_first_touch_ms << ",\n";
    j << "          \"steady_p50_ms\": " << r.t.steady_p50_ms << ",\n";
    j << "          \"steady_p90_ms\": " << r.t.steady_p90_ms << ",\n";
    j << "          \"steady_p99_ms\": " << r.t.steady_p99_ms << ",\n";
    j << "          \"steady_max_ms\": " << r.t.steady_max_ms << ",\n";
    j << "          \"steady_tail_ratio\": " << r.t.steady_tail_ratio << ",\n";
    j << "          \"steady_mean_ms\": " << r.t.steady_mean_ms << ",\n";
    j << "          \"steady_cv\": " << r.t.steady_cv << ",\n";
    j << "          \"prefetch_to_cpu_ms\": " << r.t.prefetch_to_cpu_ms << ",\n";
    j << "          \"cpu_retouch_ms\": " << r.t.cpu_retouch_ms << ",\n";
    j << "          \"prefetch_back_to_gpu_ms\": " << r.t.prefetch_back_to_gpu_ms << ",\n";
    j << "          \"gpu_retouch_ms\": " << r.t.gpu_retouch_ms << ",\n";
    j << "          \"pressure_p50_ms\": " << r.t.pressure_p50_ms << ",\n";
    j << "          \"pressure_score\": " << r.t.pressure_score << ",\n";
    j << "          \"dominant_total\": \"" << json_escape(r.t.dominant_total) << "\",\n";
    j << "          \"dominant_um\": \"" << json_escape(r.t.dominant_um) << "\"\n";
    j << "        },\n";
    j << "        \"intelligence\": {\n";
    j << "          \"regime\": \"" << json_escape(r.regime) << "\",\n";
    j << "          \"steady_jump\": " << r.steady_jump << ",\n";
    j << "          \"stability_index\": " << r.stability_index << ",\n";
    j << "          \"migration_gbs_prefetch_gpu\": " << r.migration_gbs_prefetch_gpu << ",\n";
    j << "          \"pcie_expected_gbs\": " << r.pcie_expected_gbs << ",\n";
    j << "          \"pcie_efficiency\": " << r.pcie_efficiency << "\n";
    j << "        }\n";
    j << "      }" << (i + 1 < rows.size() ? "," : "") << "\n";
  }
  j << "    ]";
}

#include <sys/utsname.h>

struct ArchInfo {
  std::string compute_capability;
  std::string architecture;
  std::string architecture_detail;
  std::string platform_type;
  std::string cpu_arch;
  std::vector<std::string> features;
};

static std::string detect_cpu_arch() {
  struct utsname u{};
  if (uname(&u) == 0) return std::string(u.machine);
  return "unknown";
}

static std::string arch_family(int major, int minor) {
  if (major >= 12) return "Blackwell";
  if (major == 10) return "Blackwell";
  if (major == 9)  return "Hopper";
  if (major == 8) {
    if (minor == 9) return "Ada Lovelace";
    return "Ampere";
  }
  if (major == 7) {
    if (minor == 5) return "Turing";
    return "Volta";
  }
  if (major == 6) return "Pascal";
  if (major == 5) return "Maxwell";
  if (major == 3) return "Kepler";
  return "Unknown";
}

static std::string arch_detail(int major, int minor) {
  if (major >= 12) return "Blackwell RTX";
  if (major == 10) return "Blackwell Datacenter";
  if (major == 9)  return "Hopper";
  if (major == 8) {
    if (minor == 0) return "Ampere A100";
    if (minor == 6) return "Ampere RTX30";
    if (minor == 7) return "Ampere Orin";
    if (minor == 9) return "Ada Lovelace";
    return "Ampere";
  }
  if (major == 7) {
    if (minor == 5) return "Turing";
    if (minor == 2) return "Volta Xavier";
    if (minor == 0) return "Volta";
    return "Volta";
  }
  if (major == 6) return "Pascal";
  if (major == 5) return "Maxwell";
  if (major == 3) return "Kepler";
  return "Unknown";
}

static ArchInfo build_arch_info(const cudaDeviceProp& prop) {
  ArchInfo a;
  std::ostringstream cc;
  cc << prop.major << "." << prop.minor;
  a.compute_capability = cc.str();
  a.architecture = arch_family(prop.major, prop.minor);
  a.architecture_detail = arch_detail(prop.major, prop.minor);
  a.cpu_arch = detect_cpu_arch();

  bool coherent = prop.pageableMemoryAccessUsesHostPageTables;
  bool unified  = prop.concurrentManagedAccess;

  if (coherent && a.cpu_arch == "aarch64")
    a.platform_type = "Grace coherent platform";
  else if (coherent)
    a.platform_type = "HW coherent unified memory";
  else if (prop.integrated)
    a.platform_type = "Integrated GPU";
  else
    a.platform_type = "Discrete GPU";

  if (unified) a.features.push_back("Unified Memory");
  if (coherent) a.features.push_back("HW Coherent UM");
  if (prop.concurrentManagedAccess) a.features.push_back("Concurrent UM");

  return a;
}

static double theoretical_bw(const cudaDeviceProp& prop) {
  // Fix #4: CUDA 13 removed memoryClockRate. On UMA platforms (DGX Spark GB10,
  // Grace Blackwell) the field is 0 or absent. Detect UMA via device properties
  // and return 0 — caller must treat 0 as "not queryable, do not display".
  // Rule: never return a hardcoded or guessed bandwidth as measured telemetry.
  if (prop.pageableMemoryAccessUsesHostPageTables && prop.concurrentManagedAccess) {
    return 0.0;  // UMA platform — bandwidth not queryable via CUDA props
  }
#if CUDART_VERSION < 13000
  if (prop.memoryClockRate > 0) {
    return 2.0 *
           (double)prop.memoryClockRate * 1000.0 *
           (prop.memoryBusWidth / 8.0) /
           1e9;
  }
#endif
  return 0.0;  // not queryable on this platform/CUDA version
}



static void write_run_json(const fs::path& run_dir,
                           const NvmlInfo& nv,
                           const HostInfo& host,
                           double host_cap_gib,
                           const std::vector<ResultRow>& cold,
                           const std::vector<ResultRow>& warm,
                           const std::vector<ResultRow>& pressure,
                           const KneeResult2& cold_k,
                           const KneeResult2& warm_k,
                           bool pressure_enabled,
                           int cold_child_failures,
                           int skipped_ratios,
                           const double* pressure_cv_history,
                           int pressure_repeats,
                           double pressure_cv_drop_pct,
                           int pressure_stable_pass,
                           bool pressure_is_fault_migration,
                           const HardwareState& hw,
                           const TransportInfo& transport,
                           const NVLinkCounters& nvlink_c,
                           const cudaDeviceProp& prop,
                           unsigned int pcie_replay_delta,
                           const PcieBwResult& pcie_bw,
                           const ThrashMetrics& thrash,
                           const ResidencyWindow& reswin,
                           double maf,
                           double migration_efficiency,
                           double oscillation_ratio,
                           double settle_ms,
                           double bpf_htod,
                           double bpf_total,
                           double direction_ratio,
                           const char* direction_trend,
                           double test_seconds,
                           double fault_max_window_rate,
                           double fault_burst_ratio,
                           double residency_half_life_ratio,
                           double kv_pressure_score,
                           const char* kv_pressure_level,
                           const PsiSnapshot& psi_start,
                           const PsiSnapshot& psi_end,
                           const std::string& psi_state_str) {
  std::ofstream j(run_dir / "run.json");
  j << "{\n";
  ArchInfo arch = build_arch_info(prop);
  double theoretical_bandwidth = theoretical_bw(prop);
  j << "  \"schema_version\": \"2.8\",\n";
  j << "  \"cold_child_failures\": " << cold_child_failures << ",\n";
  j << "  \"skipped_ratios\": " << skipped_ratios << ",\n";
  j << "  \"pressure_enabled\": " << (pressure_enabled ? "true" : "false") << ",\n";
  j << "  \"system\": {\n";
  j << "    \"timestamp_local\": \"" << now_local_compact() << "\",\n";
  j << "    \"thp\": \"" << json_escape(host.thp) << "\",\n";
  j << "    \"host_mem_total_bytes\": " << host.mem_total << ",\n";
  j << "    \"host_mem_avail_bytes\": " << host.mem_avail << ",\n";
  j << "    \"swap_total_bytes\": " << host.swap_total << ",\n";
  j << "    \"swap_free_bytes\": " << host.swap_free << "\n";
  j << "  },\n";
  j << "  \"gpu\": {\n";
  j << "    \"name\": \"" << json_escape(nv.name) << "\",\n";
  j << "    \"uuid\": \"" << json_escape(nv.uuid) << "\",\n";
  j << "    \"nvml_driver\": \"" << json_escape(nv.driver) << "\",\n";
  j << "    \"compute_capability\": \"" << prop.major << "." << prop.minor << "\",\n";
  j << "    \"architecture\": \"" << arch.architecture << "\",\n";
  j << "    \"architecture_detail\": \"" << arch.architecture_detail << "\",\n";
  j << "    \"platform_type\": \"" << arch.platform_type << "\",\n";
  j << "    \"cpu_arch\": \"" << arch.cpu_arch << "\",\n";
  j << "    \"theoretical_bandwidth_gbps\": " << std::fixed << std::setprecision(2) << theoretical_bandwidth << ",\n";
  j << "    \"sm_count\": " << prop.multiProcessorCount << ",\n";
  j << "    \"l2_cache_kib\": " << (prop.l2CacheSize / 1024) << ",\n";
  j << "    \"um_paradigm\": \"" << json_escape(nv.um_paradigm) << "\",\n";
  j << "    \"transport\": {\n";
  j << "      \"layer\": \"" << transport_str(transport.layer) << "\",\n";
  j << "      \"nvlink_links\": " << transport.nvlink_links << ",\n";
  j << "      \"nvlink_health\": \"" << nvlink_c.verdict << "\",\n";
  j << "      \"nvlink_crc_flit_total\": " << nvlink_c.total_crc_flit << ",\n";
  j << "      \"nvlink_replay_total\": "   << nvlink_c.total_replay   << ",\n";
  j << "      \"nvlink_err_dl_total\": "   << nvlink_c.total_err_dl   << ",\n";
  j << "      \"nvlink_util_tx_total\": "  << nvlink_c.total_util_tx  << ",\n";
  j << "      \"nvlink_util_rx_total\": "  << nvlink_c.total_util_rx  << "\n";
  j << "    },\n";
  j << "    \"pcie_curr\": \"Gen" << nv.pcie_gen_curr << " x" << nv.pcie_width_curr << "\",\n";
  j << "    \"pcie_max\": \"Gen" << nv.pcie_gen_max << " x" << nv.pcie_width_max << "\",\n";
  j << "    \"pcie_replay_delta\": " << pcie_replay_delta << ",\n";
  j << "    \"pcie_h2d_gbs\": " << std::fixed << std::setprecision(3) << pcie_bw.h2d_gbs << ",\n";
  j << "    \"pcie_d2h_gbs\": " << std::fixed << std::setprecision(3) << pcie_bw.d2h_gbs << ",\n";
  j << "    \"thrash_symmetry\": " << std::setprecision(3) << thrash.symmetry << ",\n";
  j << "    \"thrash_cv_mean\": " << std::setprecision(3) << thrash.cv_instability << ",\n";
  j << "    \"thrash_score\": " << std::setprecision(3) << thrash.thrash_score << ",\n";
  j << "    \"thrash_state\": \"" << thrash.state << "\",\n";
  j << "    \"maf\": " << std::fixed << std::setprecision(3) << maf << ",\n";
  j << "    \"migration_efficiency\": " << std::setprecision(3) << migration_efficiency << ",\n";
  j << "    \"migration_oscillation_ratio\": " << std::setprecision(3) << oscillation_ratio << ",\n";
  j << "    \"bpf_htod_bytes\": " << std::setprecision(1) << bpf_htod << ",\n";
  j << "    \"bpf_total_bytes\": " << std::setprecision(1) << bpf_total << ",\n";
  j << "    \"settled\": " << (thrash.settled ? "true" : "false") << ",\n";
  {
    const char* sc = "STABLE";
    if      (!thrash.settled)                    sc = "UNSTABLE";
    else if (thrash.thrash_score >= 0.15)        sc = "LATE_UNSTABLE";
    else                                         sc = "STABLE";
    j << "    \"settle_class\": \"" << sc << "\",\n";
  }
  j << "    \"settle_ms\": " << std::setprecision(1) << settle_ms << ",\n";
  j << "    \"cupti_available\": " << (g_cupti_ok ? "true" : "false") << ",\n";
  j << "    \"cupti_not_supported\": " << (g_cupti.not_supported ? "true" : "false") << ",\n";
  j << "    \"cupti_callbacks_failed\": " << (g_cupti.callbacks_failed ? "true" : "false") << ",\n";
  j << "    \"cupti_records_total\": " << g_cupti.records_total << ",\n";
  j << "    \"cupti_records_dropped\": " << g_cupti.records_dropped << ",\n";
  j << "    \"cupti_zero_end_ts_skipped\": " << g_cupti.zero_end_ts_skipped << ",\n";
  j << "    \"cupti_zeroed_record_skipped\": " << g_cupti.zeroed_record_skipped << ",\n";
  j << "    \"cupti_gpu_page_faults\": " << g_cupti.gpu_page_faults << ",\n";
  j << "    \"cupti_cpu_page_faults\": " << g_cupti.cpu_page_faults << ",\n";
  bool cupti_migration_ok = (g_cupti_ok && (g_cupti.bytes_htod + g_cupti.bytes_dtoh) > 0);
  j << "    \"cupti_migration_data_available\": " << (cupti_migration_ok ? "true" : "false") << ",\n";
  j << "    \"cupti_bytes_htod\": " << g_cupti.bytes_htod << ",\n";
  j << "    \"cupti_bytes_dtoh\": " << g_cupti.bytes_dtoh << ",\n";
  j << "    \"cupti_thrashing_events\": " << g_cupti.thrashing << ",\n";
  j << "    \"cupti_throttling_events\": " << g_cupti.throttling << ",\n";
  j << "    \"direction_ratio\": " << std::fixed << std::setprecision(3) << direction_ratio << ",\n";
  j << "    \"direction_trend\": \"" << direction_trend << "\",\n";
  // UM intent from last cold row with valid data
  {
    std::string pref = "n/a", last_pf = "n/a";
    for (auto it = cold.rbegin(); it != cold.rend(); ++it) {
      if (!it->um_preferred_location.empty() && it->um_preferred_location != "n/a") {
        pref    = it->um_preferred_location;
        last_pf = it->um_last_prefetch;
        break;
      }
    }
    j << "    \"um_preferred_location\": \"" << pref << "\",\n";
    j << "    \"um_last_prefetch_location\": \"" << last_pf << "\",\n";
  }
  j << "    \"test_seconds\": " << std::setprecision(3) << test_seconds << ",\n";
  j << "    \"fault_rate_per_sec\": " << std::setprecision(3) << ((double)g_cupti.gpu_page_faults / test_seconds) << ",\n";
  j << "    \"fault_max_window_rate_per_sec\": " << std::setprecision(3) << fault_max_window_rate << ",\n";
  j << "    \"fault_burst_ratio\": " << std::setprecision(3) << fault_burst_ratio << ",\n";
  j << "    \"fault_pressure_index\": " << std::setprecision(3) << fault_burst_ratio * ((double)g_cupti.gpu_page_faults / test_seconds) << ",\n";
  if (residency_half_life_ratio > 0.0)
    j << "    \"residency_half_life_ratio\": " << std::setprecision(3) << residency_half_life_ratio << ",\n";
  else
    j << "    \"residency_half_life_ratio\": null,\n";
  j << "    \"llm_pressure_score\": " << std::setprecision(3) << kv_pressure_score << ",\n";
  j << "    \"llm_pressure_level\": \"" << kv_pressure_level << "\",\n";
  j << "    \"memory_psi_state\": \"" << psi_state_str << "\",\n";
  if (psi_start.available && psi_end.available && psi_state_str != "LOW") {
    j << "    \"memory_psi_some_avg10_start\": " << std::setprecision(3) << psi_start.some_avg10 << ",\n";
    j << "    \"memory_psi_some_avg10_end\": "   << std::setprecision(3) << psi_end.some_avg10   << ",\n";
    j << "    \"memory_psi_full_avg10_end\": "   << std::setprecision(3) << psi_end.full_avg10   << ",\n";
    uint64_t delta = (psi_end.some_total >= psi_start.some_total)
                     ? (psi_end.some_total - psi_start.some_total) : 0;
    j << "    \"memory_psi_some_total_delta_us\": " << delta << ",\n";
  }
  j << "    \"residency_fault_onset_ratio\": " << std::setprecision(3) << reswin.fault_onset_ratio << ",\n";
  j << "    \"residency_resident_gib\": " << std::setprecision(3) << reswin.resident_gib << ",\n";
  j << "    \"residency_fault_onset_gib\": " << std::setprecision(3) << reswin.fault_onset_gib << ",\n";
  j << "    \"vram_total_bytes\": " << nv.vram_total << ",\n";
  j << "    \"vram_free_bytes\": " << nv.vram_free << ",\n";
  j << "    \"mem_type\": \"" << nv.mem_type << "\",\n";
  j << "    \"mem_bus_width_bits\": " << nv.mem_bus_width << "\n";
  j << "  },\n";
  j << "  \"host_cap_gib\": " << std::fixed << std::setprecision(3) << host_cap_gib << ",\n";
  {
    double hf = (double)host.mem_avail / (1024.0*1024.0*1024.0);
    double hr = (host_cap_gib > 0.0) ? (hf / host_cap_gib) : 0.0;
    j << "  \"um_headroom_ratio\": " << std::setprecision(3) << hr << ",\n";
  }
  j << "  \"knee\": {\n";
  j << "    \"cold\": {\"ratio\": " << cold_k.ratio << ", \"metric\": \"" << cold_k.metric << "\", \"confidence\": " << cold_k.confidence << "},\n";
  j << "    \"warm\": {\"ratio\": " << warm_k.ratio << ", \"metric\": \"" << warm_k.metric << "\", \"confidence\": " << warm_k.confidence << "}\n";
  j << "  },\n";
  j << "  \"passes\": {\n";
  dump_rows(j, "cold", cold); j << ",\n";
  dump_rows(j, "warm", warm);
  if (!pressure.empty()) { j << ",\n"; dump_rows(j, "pressure", pressure); }
  j << "\n";
  j << "  },\n";

  // Pressure stability block (schema 2.2 addition)
  j << "  \"pressure_stability\": {\n";
  j << "    \"repeats\": " << pressure_repeats << ",\n";
  j << "    \"cv_history\": [";
  for (int i = 0; i < pressure_repeats; ++i) {
    j << std::fixed << std::setprecision(4) << pressure_cv_history[i];
    if (i + 1 < pressure_repeats) j << ", ";
  }
  j << "],\n";
  j << "    \"cv_drop_pct\": " << std::setprecision(4) << pressure_cv_drop_pct * 100.0 << ",\n";
  j << "    \"stable_pass\": " << pressure_stable_pass << ",\n";
  j << "    \"fault_migration_regime\": " << (pressure_is_fault_migration ? "true" : "false") << ",\n";
  j << "    \"stability_meaningful\": " << ((pressure_is_fault_migration || pressure_cv_history[0] > 0.10 || pressure_cv_history[pressure_repeats-1] < 0.05) ? "true" : "false") << "\n";
  j << "  },\n";

  // Hardware state block (schema 2.3)
  j << "  \"hardware_state\": {\n";
  j << "    \"temp_idle_c\": "    << hw.temp_idle_c  << ",\n";
  j << "    \"temp_start_c\": "   << hw.temp_start_c << ",\n";
  j << "    \"temp_end_c\": "     << hw.temp_end_c   << ",\n";
  j << "    \"temp_drift_c\": "   << hw.temp_drift_c << ",\n";
  j << "    \"pstate_start\": "   << hw.pstate_start << ",\n";
  j << "    \"sm_clock_start_mhz\": "  << hw.sm_clock_start_mhz  << ",\n";
  j << "    \"mem_clock_start_mhz\": " << hw.mem_clock_start_mhz << ",\n";
  j << "    \"throttle_reasons_start\": \"" << throttle_reasons_str(hw.throttle_reasons_start) << "\",\n";
  j << "    \"throttle_reasons_end\": \""   << throttle_reasons_str(hw.throttle_reasons_end)   << "\",\n";
  j << "    \"throttle_threshold_c\": " << hw.throttle_threshold_c << ",\n";
  j << "    \"power_available\": " << (hw.power_available ? "true" : "false") << ",\n";
  if (hw.power_available) {
    j << "    \"power_draw_w\": " << std::fixed << std::setprecision(1)
      << (hw.power_draw_mw / 1000.0) << ",\n";
    j << "    \"power_limit_w\": " << std::setprecision(0)
      << (hw.power_limit_mw / 1000.0) << ",\n";
  } else {
    j << "    \"power_draw_w\": null,\n";
    j << "    \"power_limit_w\": null,\n";
  }
  j << "    \"thermal_verdict\": \""    << json_escape(hw.thermal_verdict) << "\"\n";
  j << "  },\n";

  // uma_diagnostics block — schema 2.4
  j << "  \"uma_diagnostics\": {\n";
  j << "    \"paradigm\": \""             << json_escape(nv.um_paradigm)                             << "\",\n";
  j << "    \"cudamemgetinfo_unreliable\": " << (host.cudamemgetinfo_unreliable ? "true" : "false") << ",\n";
  j << "    \"uma_allocatable_bytes\": "  << host.uma_allocatable                                   << ",\n";
  j << "    \"cudamemgetinfo_free_bytes\": " << host.cudamemgetinfo_free                            << ",\n";
  j << "    \"cudamemgetinfo_total_bytes\": " << host.cudamemgetinfo_total                          << ",\n";
  // Delta fields only meaningful on UMA — on discrete GPU the two pools are
  // physically separate and the "delta" is not an error, just different pools.
  if (host.cudamemgetinfo_unreliable) {
    j << "    \"cudamemgetinfo_delta_bytes\": " << host.cudamemgetinfo_delta                        << ",\n";
    j << "    \"cudamemgetinfo_error_pct\": " << std::fixed << std::setprecision(2)
                                             << host.cudamemgetinfo_error_pct                      << ",\n";
  } else {
    j << "    \"cudamemgetinfo_delta_bytes\": null,\n";
    j << "    \"cudamemgetinfo_error_pct\": null,\n";
  }
  j << "    \"ceiling_utilization\": "    << std::setprecision(4) << host.ceiling_utilization      << ",\n";
  j << "    \"overcommit\": "               << (host.overcommit ? "true" : "false")                  << ",\n";
  j << "    \"prerun_pressure_verdict\": \"" << json_escape(host.prerun_pressure_verdict)           << "\",\n";
  j << "    \"cache_recoverable_bytes\": " << host.cache_recoverable_bytes                          << ",\n";
  j << "    \"cached_bytes\": "           << host.cached                                            << ",\n";
  j << "    \"buffers_bytes\": "          << host.buffers                                           << ",\n";
  j << "    \"buffer_cache_pressure\": "  << (host.buffer_cache_pressure ? "true" : "false")        << ",\n";
  j << "    \"swap_disabled\": "          << (host.swap_disabled          ? "true" : "false")        << ",\n";
  j << "    \"zombie_oom_structural\": "  << (host.zombie_oom_structural  ? "true" : "false")        << ",\n";
  j << "    \"hugetlb_total_pages\": "    << host.hugetlb_total                                     << ",\n";
  j << "    \"hugetlb_free_pages\": "     << host.hugetlb_free                                      << "\n";
  j << "  }\n";
  j << "}\n";
}

// ------------------- CLI -------------------

// ─────────────────────────────────────────────────────────────────────────────
// P2P Topology Layer
//
// Per CUDA Programming Guide §3.4 "Programming Systems with Multiple GPUs":
//   - cudaDeviceCanAccessPeer() is the ONLY authoritative P2P capability query.
//     CC matching is a common reason for failure but not the rule — never infer.
//   - cudaDeviceEnablePeerAccess() must be called before cudaMemcpyPeerAsync().
//   - Access is UNIDIRECTIONAL — A→B ≠ B→A. Both directions queried and tested.
//   - Max 8 peer connections per device system-wide.
//   - cudaDeviceGetP2PAttribute() gives performance rank and NVLink atomic support.
//
// From NVIDIA forum thread (Robert Crovella, Nov 2015):
//   - PHB topology (host bridge) does not guarantee P2P failure — use the API.
//   - SOC (QPI/socket) topology DOES prevent P2P on most platforms.
//   - ECC errors on a device can cause P2P to silently fail as OOM — cross-check.
//
// Link type classification (derived from NVLink active links + P2P perf rank):
//   - NVLink:      nvlink_any_active == true on BOTH endpoints
//   - PCIe-Direct: cudaDeviceCanAccessPeer==1, no NVLink
//   - PCIe-via-Host: cudaDeviceCanAccessPeer==0, traffic routes through host CPU
// ─────────────────────────────────────────────────────────────────────────────

static constexpr size_t P2P_BW_BYTES     = 256ULL * 1024 * 1024; // 256 MiB per direction
static constexpr int    P2P_WARMUP_REPS  = 3;
static constexpr int    P2P_TIMING_REPS  = 7;

enum class P2PLinkType { NVLINK, PCIE_DIRECT, PCIE_HOST, UNKNOWN };

static const char* p2p_link_name(P2PLinkType t) {
  switch (t) {
    case P2PLinkType::NVLINK:      return "NVLink";
    case P2PLinkType::PCIE_DIRECT: return "PCIe-Direct";
    case P2PLinkType::PCIE_HOST:   return "PCIe-via-Host";
    default:                       return "Unknown";
  }
}

struct P2PPairResult {
  int  src = 0, dst = 0;
  bool can_access_src_dst = false;
  bool can_access_dst_src = false;
  int  perf_rank          = -1;
  bool native_atomics     = false;
  P2PLinkType link_type   = P2PLinkType::UNKNOWN;
  double bw_src_to_dst_gbs = 0.0;
  double bw_dst_to_src_gbs = 0.0;
  bool   bw_measured       = false;
  std::string verdict;
  std::string notes;
};

static double measure_p2p_bandwidth(int src, int dst, void* src_buf, void* dst_buf) {
  cudaStream_t stream;
  cudaSetDevice(src);
  if (cudaStreamCreate(&stream) != cudaSuccess) return 0.0;
  for (int i = 0; i < P2P_WARMUP_REPS; ++i)
    cudaMemcpyPeerAsync(dst_buf, dst, src_buf, src, P2P_BW_BYTES, stream);
  cudaStreamSynchronize(stream);
  std::vector<double> samples;
  for (int i = 0; i < P2P_TIMING_REPS; ++i) {
    auto t0 = clk::now();
    cudaMemcpyPeerAsync(dst_buf, dst, src_buf, src, P2P_BW_BYTES, stream);
    cudaStreamSynchronize(stream);
    double elapsed_ms = ms(t0, clk::now());
    if (elapsed_ms > 0.0) samples.push_back((double)P2P_BW_BYTES / (elapsed_ms * 1e6));
  }
  cudaStreamDestroy(stream);
  if (samples.empty()) return 0.0;
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2]; // median
}

static std::vector<P2PPairResult> run_p2p_topology(int num_devices,
                                                    const std::vector<NvmlInfo>& infos)
{
  std::vector<P2PPairResult> results;
  if (num_devices < 2) return results;
  std::cout << "\n[p2p-topology] scanning " << num_devices << " GPU(s)...\n";

  std::vector<void*> dev_bufs(num_devices, nullptr);
  for (int d = 0; d < num_devices; ++d) {
    cudaSetDevice(d);
    if (cudaMalloc(&dev_bufs[d], P2P_BW_BYTES) != cudaSuccess) {
      std::cerr << "[p2p] cudaMalloc failed on device " << d << "\n";
      dev_bufs[d] = nullptr;
    }
  }

  for (int i = 0; i < num_devices; ++i) {
    for (int j = i + 1; j < num_devices; ++j) {
      P2PPairResult pr;
      pr.src = i; pr.dst = j;

      int can_ij = 0, can_ji = 0;
      cudaDeviceCanAccessPeer(&can_ij, i, j);
      cudaDeviceCanAccessPeer(&can_ji, j, i);
      pr.can_access_src_dst = (can_ij == 1);
      pr.can_access_dst_src = (can_ji == 1);

      int perf_rank = -1, nat_atomic = 0;
      cudaDeviceGetP2PAttribute(&perf_rank,  cudaDevP2PAttrPerformanceRank,          i, j);
      cudaDeviceGetP2PAttribute(&nat_atomic, cudaDevP2PAttrNativeAtomicSupported,     i, j);
      pr.perf_rank      = perf_rank;
      pr.native_atomics = (nat_atomic == 1);

      // Link type: NVLink if both endpoints report active NVLink links
      if (infos[i].nvlink_any_active && infos[j].nvlink_any_active
          && pr.can_access_src_dst)
        pr.link_type = P2PLinkType::NVLINK;
      else if (pr.can_access_src_dst)
        pr.link_type = P2PLinkType::PCIE_DIRECT;
      else
        pr.link_type = P2PLinkType::PCIE_HOST;

      bool bufs_ok = (dev_bufs[i] && dev_bufs[j]);
      if (pr.can_access_src_dst && bufs_ok) {
        cudaSetDevice(i);
        cudaError_t e_ij = cudaDeviceEnablePeerAccess(j, 0);
        cudaSetDevice(j);
        cudaError_t e_ji = cudaDeviceEnablePeerAccess(i, 0);
        bool ok_ij = (e_ij == cudaSuccess || e_ij == cudaErrorPeerAccessAlreadyEnabled);
        bool ok_ji = (e_ji == cudaSuccess || e_ji == cudaErrorPeerAccessAlreadyEnabled);
        if (ok_ij) pr.bw_src_to_dst_gbs = measure_p2p_bandwidth(i, j, dev_bufs[i], dev_bufs[j]);
        if (ok_ji) pr.bw_dst_to_src_gbs = measure_p2p_bandwidth(j, i, dev_bufs[j], dev_bufs[i]);
        pr.bw_measured = (ok_ij || ok_ji);
        if (ok_ij) { cudaSetDevice(i); cudaDeviceDisablePeerAccess(j); }
        if (ok_ji) { cudaSetDevice(j); cudaDeviceDisablePeerAccess(i); }
      }

      // Verdict
      if (!pr.can_access_src_dst) {
        pr.verdict = "NO_DIRECT_P2P";
        pr.notes   = "Traffic routes through host CPU RAM. "
                     "Expect ~PCIe ceiling / 2 effective bandwidth.";
      } else if (pr.bw_measured) {
        double expected_floor = 0.0;
        if (pr.link_type == P2PLinkType::NVLINK) {
          expected_floor = 100.0; // conservative: 50% of NVLink3 floor
        } else {
          unsigned gen   = infos[i].pcie_gen_curr;
          unsigned width = infos[i].pcie_width_curr;
          double theoretical = (gen >= 4 ? 31.5 : 15.75) * (width / 16.0);
          expected_floor = theoretical * 0.40;
        }
        double min_bw = std::min(pr.bw_src_to_dst_gbs, pr.bw_dst_to_src_gbs);
        if (min_bw < expected_floor) {
          pr.verdict = "DEGRADED";
          pr.notes   = "P2P BW below expected floor ("
                     + std::to_string((int)expected_floor)
                     + " GB/s). Check: nvidia-smi topo -m, ACS settings, IOMMU.";
        } else {
          pr.verdict = "HEALTHY";
        }
      } else {
        pr.verdict = "UNKNOWN";
        pr.notes   = "Buffer allocation failed — could not measure bandwidth.";
      }

      std::cout << "[p2p] GPU" << i << "↔GPU" << j
                << "  link=" << p2p_link_name(pr.link_type)
                << "  can_access=" << (pr.can_access_src_dst ? "YES" : "NO")
                << "  perf_rank=" << pr.perf_rank;
      if (pr.bw_measured)
        std::cout << std::fixed << std::setprecision(1)
                  << "  " << i << "→" << j << "=" << pr.bw_src_to_dst_gbs << "GB/s"
                  << "  " << j << "→" << i << "=" << pr.bw_dst_to_src_gbs << "GB/s";
      std::cout << "  verdict=" << pr.verdict << "\n";
      results.push_back(pr);
    }
  }

  for (int d = 0; d < num_devices; ++d)
    if (dev_bufs[d]) { cudaSetDevice(d); cudaFree(dev_bufs[d]); }

  return results;
}

static void write_p2p_verdict(std::ostream& v, const std::vector<P2PPairResult>& pairs) {
  if (pairs.empty()) return;
  v << "\n" << std::string(108, '=') << "\n";
  v << "  P2P TOPOLOGY\n";
  v << std::string(108, '=') << "\n";
  int degraded = 0, no_p2p = 0, healthy = 0;
  for (auto& pr : pairs) {
    v << "\nGPU" << pr.src << " ↔ GPU" << pr.dst << "\n";
    v << "  Link type:      " << p2p_link_name(pr.link_type) << "\n";
    v << "  Direct P2P:     " << (pr.can_access_src_dst ? "YES (both directions)" : "NO") << "\n";
    v << "  Perf rank:      " << pr.perf_rank << "  (0=best)\n";
    v << "  Native atomics: " << (pr.native_atomics ? "YES" : "NO") << "\n";
    if (pr.bw_measured) {
      v << std::fixed << std::setprecision(1);
      v << "  BW GPU" << pr.src << "→GPU" << pr.dst << ": " << pr.bw_src_to_dst_gbs << " GB/s\n";
      v << "  BW GPU" << pr.dst << "→GPU" << pr.src << ": " << pr.bw_dst_to_src_gbs << " GB/s\n";
    }
    v << "  Verdict:        " << pr.verdict << "\n";
    if (!pr.notes.empty()) v << "  Notes:          " << pr.notes << "\n";
    if      (pr.verdict == "DEGRADED")      degraded++;
    else if (pr.verdict == "NO_DIRECT_P2P") no_p2p++;
    else if (pr.verdict == "HEALTHY")       healthy++;
  }
  v << "\nP2P SUMMARY\n";
  v << "  Healthy pairs:   " << healthy  << "\n";
  v << "  Degraded pairs:  " << degraded << "\n";
  v << "  No direct P2P:   " << no_p2p   << "\n";
  if (degraded > 0) {
    v << "\nACTION REQUIRED\n";
    v << "  Degraded P2P bandwidth detected.\n";
    v << "  1. nvidia-smi topo -m  — check topology path\n";
    v << "     PIX (same switch) > PXB (bridge) > PHB (host bridge) > SOC (QPI)\n";
    v << "  2. Check ACS: sudo lspci -vvv | grep ACSCtl\n";
    v << "  3. Check IOMMU: cat /proc/cmdline | grep iommu\n";
    v << "  4. Check ECC: nvidia-smi -q | grep -A4 'ECC Errors'\n";
  }
  if (no_p2p > 0) {
    v << "\nNOTE: " << no_p2p << " GPU pair(s) have no direct P2P access.\n";
    v << "  Traffic routes through host CPU RAM.\n";
    v << "  Common causes: mismatched architectures, SOC topology, IOMMU.\n";
  }
}

static void write_p2p_json(std::ostream& j, const std::vector<P2PPairResult>& pairs) {
  j << "  \"p2p_topology\": [\n";
  for (size_t k = 0; k < pairs.size(); ++k) {
    const auto& pr = pairs[k];
    j << "    {\n";
    j << "      \"src\": "              << pr.src                                           << ",\n";
    j << "      \"dst\": "              << pr.dst                                           << ",\n";
    j << "      \"can_access\": "       << (pr.can_access_src_dst ? "true" : "false")       << ",\n";
    j << "      \"perf_rank\": "        << pr.perf_rank                                     << ",\n";
    j << "      \"native_atomics\": "   << (pr.native_atomics ? "true" : "false")           << ",\n";
    j << "      \"link_type\": \""      << p2p_link_name(pr.link_type)                      << "\",\n";
    j << "      \"bw_src_to_dst_gbs\": "<< std::fixed << std::setprecision(2)
                                         << pr.bw_src_to_dst_gbs                            << ",\n";
    j << "      \"bw_dst_to_src_gbs\": "<< pr.bw_dst_to_src_gbs                            << ",\n";
    j << "      \"bw_measured\": "      << (pr.bw_measured ? "true" : "false")              << ",\n";
    j << "      \"verdict\": \""        << json_escape(pr.verdict)                          << "\",\n";
    j << "      \"notes\": \""          << json_escape(pr.notes)                            << "\"\n";
    j << "    }" << (k + 1 < pairs.size() ? "," : "") << "\n";
  }
  j << "  ],\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// P2P Unified Memory Coherence Test
//
// What this measures:
//   cudaMallocManaged allocates a buffer accessible from any device. On
//   FULL_HARDWARE_COHERENT (DGX Spark C2C, Grace Hopper) hardware cache
//   coherence guarantees pointer-valid cross-device access with no explicit
//   copy. On FULL_EXPLICIT (discrete PCIe GPUs) the driver handles migrations
//   via fault-and-migrate — coherence is software-managed but still required
//   to be correct.
//
//   This test verifies correctness, not performance:
//     1. GPU src writes a known u32 pattern to a managed buffer.
//     2. GPU dst reads it back via raw pointer — no cudaMemcpy.
//     3. Mismatch at any element → COHERENCE_FAIL.
//     4. Test runs both directions: src→dst and dst→src.
//
// Failure modes this catches:
//   - Driver bug causing stale TLB entries after migration
//   - Hardware coherence protocol failure (C2C link fault)
//   - ECC corruption silent enough to pass single-device tests
//   - cudaMemAdvise misconfiguration causing read from wrong physical page
//
// Buffer size: 16 MiB — large enough to stress the TLB, small enough to
// complete quickly on slow PCIe paths.
// ─────────────────────────────────────────────────────────────────────────────

static constexpr size_t  COHERENCE_BYTES    = 16ULL * 1024 * 1024;
static constexpr size_t  COHERENCE_ELEMS    = COHERENCE_BYTES / sizeof(uint32_t);
static constexpr uint32_t COHERENCE_PATTERN = 0xDEADBEEFu;

// GPU kernel: fill buffer with pattern XOR element index.
// Index-dependent pattern catches element transposition bugs.
__global__ void coherence_fill(uint32_t* buf, size_t n, uint32_t pattern) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[i] = pattern ^ (uint32_t)i;
}

// GPU kernel: verify buffer written by a different device.
// Writes mismatch count into result[0].
__global__ void coherence_verify(const uint32_t* buf, size_t n,
                                  uint32_t pattern, uint32_t* result) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && buf[i] != (pattern ^ (uint32_t)i))
    atomicAdd(result, 1u);
}

enum class CoherenceVerdict { PASS, FAIL, SKIP, ERROR };

static const char* coherence_verdict_str(CoherenceVerdict v) {
  switch (v) {
    case CoherenceVerdict::PASS:  return "PASS";
    case CoherenceVerdict::FAIL:  return "FAIL";
    case CoherenceVerdict::SKIP:  return "SKIP";
    case CoherenceVerdict::ERROR: return "ERROR";
    default:                      return "UNKNOWN";
  }
}

struct CoherencePairResult {
  int  src = 0, dst = 0;
  CoherenceVerdict src_to_dst = CoherenceVerdict::SKIP;
  CoherenceVerdict dst_to_src = CoherenceVerdict::SKIP;
  uint32_t mismatches_src_to_dst = 0;
  uint32_t mismatches_dst_to_src = 0;
  std::string notes;
};

// Test coherence in one direction: writer=writer_dev, reader=reader_dev.
// Returns PASS, FAIL, or ERROR. mismatch_count set on FAIL.
static CoherenceVerdict test_coherence_direction(int writer_dev, int reader_dev,
                                                  uint32_t* managed_buf,
                                                  uint32_t* result_buf,
                                                  uint32_t& mismatch_count) {
  const int  threads = 256;
  const int  blocks  = (int)((COHERENCE_ELEMS + threads - 1) / threads);

  // Writer fills the buffer
  cudaSetDevice(writer_dev);
  coherence_fill<<<blocks, threads>>>(managed_buf, COHERENCE_ELEMS, COHERENCE_PATTERN);
  if (cudaDeviceSynchronize() != cudaSuccess) return CoherenceVerdict::ERROR;

  // Ensure writer is done before reader starts — explicit sync across devices
  // cudaDeviceSynchronize on writer_dev is sufficient: managed memory migration
  // is triggered by the reader's first access, not by an explicit transfer.
  // No cudaMemcpy — that would defeat the purpose of the test.

  // Reader verifies — accesses managed buffer directly by pointer
  cudaSetDevice(reader_dev);
  result_buf[0] = 0u;
  coherence_verify<<<blocks, threads>>>(managed_buf, COHERENCE_ELEMS,
                                         COHERENCE_PATTERN, result_buf);
  if (cudaDeviceSynchronize() != cudaSuccess) return CoherenceVerdict::ERROR;

  mismatch_count = result_buf[0];
  return (mismatch_count == 0) ? CoherenceVerdict::PASS : CoherenceVerdict::FAIL;
}

static std::vector<CoherencePairResult> run_p2p_coherence(
    int num_devices, const std::vector<NvmlInfo>& infos)
{
  std::vector<CoherencePairResult> results;
  if (num_devices < 2) return results;

  std::cout << "\n[p2p-coherence] testing " << num_devices << " GPU(s)...\n";

  // Single managed buffer reused across all pairs.
  // cudaMallocManaged from device 0 — accessible from all devices.
  uint32_t* managed_buf = nullptr;
  cudaSetDevice(0);
  if (cudaMallocManaged(&managed_buf, COHERENCE_BYTES) != cudaSuccess) {
    std::cerr << "[coherence] cudaMallocManaged failed — skipping coherence test\n";
    return results;
  }

  // Per-device result buffers for mismatch counts (pinned host mem is fine here)
  std::vector<uint32_t*> result_bufs(num_devices, nullptr);
  for (int d = 0; d < num_devices; ++d) {
    cudaSetDevice(d);
    // Small allocation — just one u32 for the atomic mismatch counter
    if (cudaMallocManaged(&result_bufs[d], sizeof(uint32_t)) != cudaSuccess) {
      std::cerr << "[coherence] result buf alloc failed on device " << d << "\n";
      cudaFree(managed_buf);
      return results;
    }
  }

  for (int i = 0; i < num_devices; ++i) {
    for (int j = i + 1; j < num_devices; ++j) {
      CoherencePairResult cr;
      cr.src = i;
      cr.dst = j;

      // Direction 1: GPU i writes, GPU j reads
      cr.src_to_dst = test_coherence_direction(i, j, managed_buf,
                                                result_bufs[j],
                                                cr.mismatches_src_to_dst);
      // Direction 2: GPU j writes, GPU i reads
      cr.dst_to_src = test_coherence_direction(j, i, managed_buf,
                                                result_bufs[i],
                                                cr.mismatches_dst_to_src);

      if (cr.src_to_dst == CoherenceVerdict::FAIL ||
          cr.dst_to_src == CoherenceVerdict::FAIL) {
        cr.notes = "Managed memory value mismatch detected. "
                   "Check ECC errors (nvidia-smi -q | grep -A4 'ECC'), "
                   "driver version, and IOMMU configuration.";
      } else if (cr.src_to_dst == CoherenceVerdict::ERROR ||
                 cr.dst_to_src == CoherenceVerdict::ERROR) {
        cr.notes = "Kernel launch or sync failed. "
                   "Check device health: nvidia-smi -q | grep 'GPU Operation Mode'";
      }

      std::cout << "[coherence] GPU" << i << "↔GPU" << j
                << "  " << i << "→" << j << "=" << coherence_verdict_str(cr.src_to_dst);
      if (cr.src_to_dst == CoherenceVerdict::FAIL)
        std::cout << "(" << cr.mismatches_src_to_dst << " mismatches)";
      std::cout << "  " << j << "→" << i << "=" << coherence_verdict_str(cr.dst_to_src);
      if (cr.dst_to_src == CoherenceVerdict::FAIL)
        std::cout << "(" << cr.mismatches_dst_to_src << " mismatches)";
      std::cout << "\n";

      results.push_back(cr);
    }
  }

  // Cleanup
  cudaFree(managed_buf);
  for (int d = 0; d < num_devices; ++d)
    if (result_bufs[d]) cudaFree(result_bufs[d]);

  return results;
}

static void write_coherence_verdict(std::ostream& v,
                                     const std::vector<CoherencePairResult>& pairs)
{
  if (pairs.empty()) return;
  v << "\n" << std::string(108, '=') << "\n";
  v << "  P2P UM COHERENCE\n";
  v << std::string(108, '=') << "\n";

  int pass = 0, fail = 0, err = 0;
  for (auto& cr : pairs) {
    v << "\nGPU" << cr.src << " ↔ GPU" << cr.dst << "\n";
    v << "  " << cr.src << "→" << cr.dst << ": "
      << coherence_verdict_str(cr.src_to_dst);
    if (cr.src_to_dst == CoherenceVerdict::FAIL)
      v << "  mismatches=" << cr.mismatches_src_to_dst;
    v << "\n";
    v << "  " << cr.dst << "→" << cr.src << ": "
      << coherence_verdict_str(cr.dst_to_src);
    if (cr.dst_to_src == CoherenceVerdict::FAIL)
      v << "  mismatches=" << cr.mismatches_dst_to_src;
    v << "\n";
    if (!cr.notes.empty()) v << "  Note: " << cr.notes << "\n";

    bool pair_pass = (cr.src_to_dst == CoherenceVerdict::PASS &&
                      cr.dst_to_src == CoherenceVerdict::PASS);
    bool pair_fail = (cr.src_to_dst == CoherenceVerdict::FAIL ||
                      cr.dst_to_src == CoherenceVerdict::FAIL);
    bool pair_err  = (cr.src_to_dst == CoherenceVerdict::ERROR ||
                      cr.dst_to_src == CoherenceVerdict::ERROR);
    if      (pair_fail) fail++;
    else if (pair_err)  err++;
    else if (pair_pass) pass++;
  }

  v << "\nCOHERENCE SUMMARY\n";
  v << "  Pairs passed:  " << pass << "\n";
  v << "  Pairs failed:  " << fail << "\n";
  v << "  Pairs errored: " << err  << "\n";

  if (fail > 0) {
    v << "\nACTION REQUIRED\n";
    v << "  Coherence failure means managed memory returned wrong values\n";
    v << "  when read from a different GPU than the one that wrote it.\n";
    v << "  This is a correctness failure, not a performance issue.\n";
    v << "  Steps:\n";
    v << "    1. nvidia-smi -q | grep -A4 'ECC'  — check for uncorrectable errors\n";
    v << "    2. Check driver version: nvidia-smi — update if below current release\n";
    v << "    3. Check IOMMU: cat /proc/cmdline | grep iommu\n";
    v << "    4. Re-run with --p2p to check topology — degraded P2P can cause\n";
    v << "       coherence failures on some platforms\n";
  }
}

static void write_coherence_json(std::ostream& j,
                                  const std::vector<CoherencePairResult>& pairs)
{
  j << "  \"p2p_coherence\": [\n";
  for (size_t k = 0; k < pairs.size(); ++k) {
    const auto& cr = pairs[k];
    j << "    {\n";
    j << "      \"src\": "                   << cr.src                                              << ",\n";
    j << "      \"dst\": "                   << cr.dst                                              << ",\n";
    j << "      \"src_to_dst\": \""          << coherence_verdict_str(cr.src_to_dst)               << "\",\n";
    j << "      \"dst_to_src\": \""          << coherence_verdict_str(cr.dst_to_src)               << "\",\n";
    j << "      \"mismatches_src_to_dst\": " << cr.mismatches_src_to_dst                           << ",\n";
    j << "      \"mismatches_dst_to_src\": " << cr.mismatches_dst_to_src                           << ",\n";
    j << "      \"notes\": \""               << json_escape(cr.notes)                              << "\"\n";
    j << "    }" << (k + 1 < pairs.size() ? "," : "") << "\n";
  }
  j << "  ],\n";
}

static void print_usage() {
  std::cout
    << "UM Analyzer V8.3 (schema 2.8)\n\n"
    << "Usage:\n"
    << "  ./um_analyzer [--device N] [--all-devices] [--list-devices]\n"
    << "  ./um_analyzer --p2p                              (topology + bandwidth, 2+ GPUs)\n"
    << "  ./um_analyzer --coherence                        (UM coherence test, 2+ GPUs)\n"
    << "  ./um_analyzer --p2p --coherence                  (full P2P suite)\n"
    << "  ./um_analyzer --all-devices --p2p --coherence    (P2P suite + per-device UM)\n"
    << "  (internal) ./um_analyzer --child --device N --ratio R --bytes B\n\n";
}

static void list_devices() {
  int n = 0;
  cudaError_t ce = cudaGetDeviceCount(&n);
  if (ce != cudaSuccess) {
    std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(ce) << "\n";
    std::exit(1);
  }
  std::cout << "Detected GPUs (CUDA)\n\n";
  for (int i = 0; i < n; ++i) {
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, i) != cudaSuccess) continue;
    std::cout << "  " << i << ": " << p.name << " (CC " << p.major << "." << p.minor << ")\n";
  }
}

// ------------------- MAIN -------------------

int main(int argc, char** argv) {
  int device = 0;
  bool all_devices = false;
  bool list_only   = false;
  bool run_p2p       = false;
  bool run_coherence = false;

  bool child_mode = false;
  double child_ratio = 0.0;
  uint64_t child_bytes = 0;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if      (a == "--device"      && i + 1 < argc) device     = std::atoi(argv[++i]);
    else if (a == "--all-devices") all_devices = true;
    else if (a == "--list-devices") list_only  = true;
    else if (a == "--p2p")          run_p2p        = true;
    else if (a == "--coherence")    run_coherence  = true;
    else if (a == "--child")        child_mode = true;
    else if (a == "--cupti-debug")  g_cupti_debug = true;
    else if (a == "--ratio"       && i + 1 < argc) child_ratio  = std::stod(argv[++i]);
    else if (a == "--bytes"       && i + 1 < argc) child_bytes  = (uint64_t)std::strtoull(argv[++i], nullptr, 10);
    else if (a == "--help" || a == "-h") { print_usage(); return 0; }
    else { std::cerr << "Unknown arg: " << a << "\n"; print_usage(); return 2; }
  }

  if (list_only) { list_devices(); return 0; }

  // ── P2P suite (topology + coherence) ──────────────────────────────────────
  if (run_p2p || run_coherence) {
    int n = 0;
    cudaGetDeviceCount(&n);
    if (n < 2) {
      std::cout << "[p2p] only " << n << " GPU(s) detected — P2P requires 2 or more.\n";
      if (!all_devices) return 0;
    } else {
      // Build NvmlInfo for all devices — needed for NVLink detection and PCIe gen
      std::vector<NvmlInfo> all_infos;
      for (int d = 0; d < n; ++d) {
        cudaSetDevice(d);
        all_infos.push_back(read_nvml(d));
      }

      // Run requested subtests
      std::vector<P2PPairResult>    p2p_results;
      std::vector<CoherencePairResult> coh_results;

      if (run_p2p)       p2p_results = run_p2p_topology(n, all_infos);
      if (run_coherence) coh_results = run_p2p_coherence(n, all_infos);

      // Write combined report bundle
      fs::create_directories("runs");
      fs::path p2p_dir = fs::path("runs") / ("um_p2p_" + now_local_compact());
      fs::create_directories(p2p_dir);

      // VERDICT.txt — human-readable
      {
        std::ofstream vf(p2p_dir / "P2P_VERDICT.txt");
        vf << "UM Analyzer V3 — P2P Report\n";
        vf << "GPUs scanned: " << n << "\n";
        if (run_p2p)       write_p2p_verdict(vf, p2p_results);
        if (run_coherence) write_coherence_verdict(vf, coh_results);
      }

      // p2p.json — machine-readable
      {
        std::ofstream jf(p2p_dir / "p2p.json");
        jf << "{\n";
        jf << "  \"tool\": \"um_analyzer_v3\",\n";
        jf << "  \"schema\": \"2.8\",\n";
        jf << "  \"gpu_count\": " << n << ",\n";
        if (run_p2p)       write_p2p_json(jf, p2p_results);
        if (run_coherence) write_coherence_json(jf, coh_results);
        jf << "  \"timestamp\": \"" << now_local_compact() << "\"\n";
        jf << "}\n";
      }

      // Console summary
      std::cout << "\n";
      if (run_p2p) {
        int healthy = 0, degraded = 0, no_p2p = 0;
        for (auto& r : p2p_results) {
          if      (r.verdict == "HEALTHY")      healthy++;
          else if (r.verdict == "DEGRADED")     degraded++;
          else if (r.verdict == "NO_DIRECT_P2P") no_p2p++;
        }
        std::cout << std::left << std::setw(14) << "topology"
                  << std::setw(17) << (degraded > 0 ? "DEGRADED" : "HEALTHY")
                  << healthy << " healthy  " << degraded << " degraded  "
                  << no_p2p << " no-direct\n";
      }
      if (run_coherence) {
        int pass = 0, fail = 0;
        for (auto& r : coh_results) {
          bool ok = (r.src_to_dst == CoherenceVerdict::PASS &&
                     r.dst_to_src == CoherenceVerdict::PASS);
          ok ? pass++ : fail++;
        }
        std::cout << std::left << std::setw(14) << "coherence"
                  << std::setw(17) << (fail > 0 ? "FAIL" : "PASS")
                  << pass << " pairs passed  " << fail << " pairs failed\n";
      }
      std::cout << "P2P_VERDICT.txt: " << (p2p_dir / "P2P_VERDICT.txt").string() << "\n";
      std::cout << "p2p.json:        " << (p2p_dir / "p2p.json").string()        << "\n";

      if (!all_devices) return 0;
    }
  }

  if (!child_mode && all_devices) {
    int n = 0;
    cudaError_t ce = cudaGetDeviceCount(&n);
    if (ce != cudaSuccess || n <= 0) { std::cerr << "No CUDA devices.\n"; return 1; }

    HostInfo host = read_host_info();
    uint64_t host_cap_total = compute_host_cap_bytes(host);
    uint64_t per_gpu_cap = (n > 0) ? (host_cap_total / (uint64_t)n) : host_cap_total;

    std::cout << "[all-devices] n=" << n
              << " host_cap_total_gib=" << (double)host_cap_total/(1024.0*1024.0*1024.0)
              << " per_gpu_cap_gib=" << (double)per_gpu_cap/(1024.0*1024.0*1024.0) << "\n";

    int worst = 0;
    for (int d = 0; d < n; ++d) {
      std::ostringstream cmd;
      cmd << "UM_HOST_CAP_BYTES=" << per_gpu_cap << " ";
      cmd << "./um_analyzer --device " << d;

      std::cout << "\n[all-devices] " << cmd.str() << "\n";
      int rc = std::system(cmd.str().c_str());
      int ec = system_exit_code(rc);
      if (ec != 0) {
        std::cerr << "[all-devices] child failed exit_code=" << ec
                  << " raw_status=" << rc << " gpu=" << d << "\n";
        worst = ec;
      }
    }
    return worst ? 2 : 0;
  }

  // Capture temp_idle BEFORE CUDA context init — true pre-run thermal baseline.
  // NVML init happens inside read_nvml; we need a temporary init here just for idle temp.
  HardwareState hw;
  {
    nvmlDevice_t tmpdev{};
    if (nvmlInit() == NVML_SUCCESS) {
      if (nvmlDeviceGetHandleByIndex(device, &tmpdev) == NVML_SUCCESS)
        nvmlDeviceGetTemperature(tmpdev, NVML_TEMPERATURE_GPU, &hw.temp_idle_c);
      nvmlShutdown();
    }
  }

  cudaError_t ce = cudaSetDevice(device);
  if (ce != cudaSuccess) { std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(ce) << "\n"; return 1; }

  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device);
  ArchInfo arch = build_arch_info(prop);
  double theoretical_bandwidth = theoretical_bw(prop);

  NvmlInfo nv = read_nvml(device);  // opens g_nvml session, keeps it alive
  if (!nv.ok) { std::cerr << "ERROR: NVML initialization failed — GPU driver not responding.\n" << "If this is a DGX Spark, perform a full cold power cycle (wall disconnect, 60 seconds).\n" << "Run spark-gpu-throttle-check after reboot to verify power state before retrying.\n"; return 1; }

  nv.um_paradigm = query_um_paradigm(device);

  cupti_init(device, nv.um_paradigm, prop.major);  // sm_major gates CPU_PAGE_FAULT/THROTTLING (Volta+ only)
  if (g_cupti_debug) {
    uint32_t cupti_ver = 0;
    cuptiGetVersion(&cupti_ver);
    uint32_t counter_struct_ver = (cupti_ver >= 130000) ? 3 : 2;
    fprintf(stderr, "[CUPTI_DBG] CUPTI version: %u  (struct: CUpti_ActivityUnifiedMemoryCounter%u)\n", cupti_ver, counter_struct_ver);
  }

  // Transport classification — drives pass labels and metric suppression
  TransportInfo transport = classify_transport(nv.um_paradigm, nv.nvlink_active_links);
  NVLinkCounters nvlink_c = query_nvlink_counters(g_nvml.dev);
  PassLabels pass_labels  = make_pass_labels(transport.layer);


  // Derive arch name from compute capability for throttle threshold lookup
  {
    int maj = prop.major, min = prop.minor;
    if      (maj == 6)                   nv.arch_name = "Pascal";
    else if (maj == 7 && min == 0)       nv.arch_name = "Volta";
    else if (maj == 7 && min == 5)       nv.arch_name = "Turing";
    else if (maj == 8 && min == 0)       nv.arch_name = "Ampere";
    else if (maj == 8 && min == 6)       nv.arch_name = "Ampere_consumer";
    else if (maj == 8 && min == 9)       nv.arch_name = "Ada";
    else if (maj == 9)                   nv.arch_name = "Hopper";
    else if (maj == 10)                  nv.arch_name = "Blackwell";
    else                                 nv.arch_name = "UNKNOWN";
  }

  // Populate hardware state pre-run fields now that we have arch and g_nvml.dev
  read_hw_state_pre(hw, g_nvml.dev, nv.arch_name, nv.um_paradigm);

  HostInfo host = read_host_info();
  PsiSnapshot psi_start = read_psi_memory();  // PSI snapshot before test
  // Paradigm-aware allocatable memory — must happen after um_paradigm is known
  compute_uma_allocatable(host, nv.um_paradigm);
  // cudaMemGetInfo delta — quantifies how much CMG underreports on UMA.
  // Called here: after cudaSetDevice, before any allocation.
  measure_cudamemgetinfo_delta(host);
  EffectiveMemoryView emv =
      build_effective_memory_view(nv, host, nv.um_paradigm);
  uint64_t host_cap_bytes = compute_host_cap_bytes(host);
  double host_cap_gib = (double)host_cap_bytes / (1024.0*1024.0*1024.0);
  double vram_gib = (double)nv.vram_total / (1024.0*1024.0*1024.0);
  // Fix #6: ratio ladder must use the authoritative memory ceiling, not raw
  // NVML vram_total. On FULL_HARDWARE_COHERENT (DGX Spark), emv.total_bytes
  // is MemAvailable+SwapFree (host_fused). On discrete GPU it equals vram_total.
  double emv_gib   = (double)emv.total_bytes / (1024.0*1024.0*1024.0);
  double host_free_gib = (double)host.mem_avail / (1024.0*1024.0*1024.0);
  double um_headroom_ratio = (host_cap_gib > 0.0) ? (host_free_gib / host_cap_gib) : 0.0;

  // KV-cache / LLM pressure detector — purely derived, no new measurements
  // score = fault_pressure_index * (1 - migration_efficiency) / um_headroom_ratio
  // finalized after fault_pressure_index is computed below
  double kv_pressure_score = 0.0;
  const char* kv_pressure_level = "LOW";

  if (child_mode) {
    if (child_ratio <= 0.0 || child_bytes == 0) {
      std::cerr << "child mode requires --ratio and --bytes\n";
      return 2;
    }
    const int reps = 11;
    ResultRow row = measure_ratio(child_ratio, child_bytes, device, nv.pcie_gen_curr, nv.pcie_width_curr, reps, false, /*skip_prefetch=*/true);
    cupti_flush();  // flush CUPTI callbacks before reading g_cupti counters
    std::cout << "{"
              << "\"status\":\"" << json_escape(row.status) << "\","
              << "\"oversub_ratio\":" << std::fixed << std::setprecision(3) << row.oversub_ratio << ","
              << "\"bytes\":" << row.bytes << ","
              << "\"steady_p50_ms\":" << row.t.steady_p50_ms << ","
              << "\"steady_cv\":" << row.t.steady_cv << ","
              << "\"alloc_ms\":" << row.t.alloc_ms << ","
              << "\"cpu_init_ms\":" << row.t.cpu_init_ms << ","
              << "\"prefetch_to_gpu_ms\":" << row.t.prefetch_to_gpu_ms << ","
              << "\"gpu_first_touch_ms\":" << row.t.gpu_first_touch_ms << ","
              << "\"steady_p90_ms\":" << row.t.steady_p90_ms << ","
              << "\"steady_p99_ms\":" << row.t.steady_p99_ms << ","
              << "\"steady_max_ms\":" << row.t.steady_max_ms << ","
              << "\"steady_tail_ratio\":" << row.t.steady_tail_ratio << ","
              << "\"steady_mean_ms\":" << row.t.steady_mean_ms << ","
              << "\"prefetch_to_cpu_ms\":" << row.t.prefetch_to_cpu_ms << ","
              << "\"cpu_retouch_ms\":" << row.t.cpu_retouch_ms << ","
              << "\"prefetch_back_to_gpu_ms\":" << row.t.prefetch_back_to_gpu_ms << ","
              << "\"gpu_retouch_ms\":" << row.t.gpu_retouch_ms << ","
              << "\"pressure_p50_ms\":" << row.t.pressure_p50_ms << ","
              << "\"pressure_score\":" << row.t.pressure_score << ","
              << "\"dominant_total\":\"" << json_escape(row.t.dominant_total) << "\","
              << "\"dominant_um\":\"" << json_escape(row.t.dominant_um) << "\","
              << "\"um_preferred_location\":\"" << json_escape(row.um_preferred_location) << "\","
              << "\"um_last_prefetch\":\"" << json_escape(row.um_last_prefetch) << "\","
              << "\"child_gpu_faults\":" << g_cupti.gpu_page_faults << ","
              << "\"child_bytes_htod\":" << g_cupti.bytes_htod << ","
              << "\"child_bytes_dtoh\":" << g_cupti.bytes_dtoh
              << "}\n";
    return (row.status == "ok") ? 0 : 3;
  }

  const std::string HDR(108, '=');
  std::cout << "\n" << HDR << "\n";
  std::cout << "  CUDA Unified Memory Analyzer  V8.3.2\n";
  std::cout << HDR << "\n\n";

  std::cout << "  GPU      " << nv.name << "\n";
  std::cout << "  Arch     " << arch.architecture_detail
            << "  SM " << prop.major << "." << prop.minor
            << "  |  " << arch.platform_type
            << "  |  " << arch.cpu_arch << "\n";
  std::cout << std::fixed << std::setprecision(1)
            << "  Compute  " << theoretical_bandwidth << " GB/s theoretical"
            << "  |  " << prop.multiProcessorCount << " SM"
            << "  |  L2 " << (prop.l2CacheSize / 1024) << " KiB\n";
  std::cout << "\n";
  std::cout << std::fixed << std::setprecision(2)
            << "  " << std::left << std::setw(8) << emv.label
            << " " << gib(emv.total_bytes) << " GiB";
  if (!nv.mem_type.empty()) std::cout << "  " << nv.mem_type;
  if (std::string(emv.label) == "VRAM" && nv.mem_bus_width > 0)
    std::cout << "  " << nv.mem_bus_width << "-bit";
  std::cout << "\n";
  std::cout << "  Driver   " << nv.driver
            << "  |  PCIe Gen" << nv.pcie_gen_curr << " x" << nv.pcie_width_curr
            << "  (max Gen" << nv.pcie_gen_max << " x" << nv.pcie_width_max << ")\n";
  std::cout << "  UUID     " << nv.uuid << "\n";
  std::cout << "\n";
  std::cout << "  UM       " << nv.um_paradigm << "  —  ";
  if (transport.nvlink_links > 0)
    std::cout << "NVLink transport  (" << transport.nvlink_links << " active link(s)  health=" << nvlink_c.verdict << ")\n";
  else
    std::cout << transport_str(transport.layer) << " transport  |  " << nvlink_c.verdict << "\n";
  std::cout << std::fixed << std::setprecision(2)
            << "  Host     " << (double)host.mem_avail/(1024.0*1024.0*1024.0) << " GiB free"
            << "  |  Test ceiling: " << host_cap_gib << " GiB\n";
  std::cout << "\n" << HDR << "\n";
  std::cout << "  PASS RESULTS\n";
  std::cout << HDR << "\n";

  fs::path run_dir = make_run_dir(device, short_uuid(nv.uuid));

  std::vector<double> ratios = adaptive_ratios(emv_gib);
  std::vector<double> runnable;
  std::vector<uint64_t> runnable_bytes;
  int skipped_ratios = 0;
  int skipped_core_ratios = 0;  // ratios <= 1.00x skipped — real limitation

  for (double r : ratios) {
    uint64_t req = (uint64_t)(emv_gib * 1024.0*1024.0*1024.0 * r);
    if (host_cap_bytes == 0 || req > host_cap_bytes) {
      skipped_ratios++;
      if (r <= 1.001) skipped_core_ratios++;
      continue;
    }
    runnable.push_back(r);
    runnable_bytes.push_back(req);
  }

  if (runnable.empty()) {
    std::cerr << "No runnable ratios under clamp.\n";
    return 1;
  }

  const int reps = 11;

  // ── Progress display helpers ──────────────────────────────────────────────
  // Pass 1: spinner — no timing data yet, just signal tool is alive
  // Pass 2+: timing bar with estimated seconds remaining
  // All output uses \r + flush — updates in-place, clears before result line
  static const int BAR_WIDTH = 24;
  static const int PROGRESS_COL = 80;

  auto clear_progress = [&]() {
    std::cout << "\r" << std::string(PROGRESS_COL, ' ') << "\r" << std::flush;
  };

  // Spinner — cycles | / - \ on each call
  auto draw_spinner = [&](const std::string& tag, double ratio) {
    static const char frames[] = {'|', '/', '-', '\\'};
    static int frame = 0;
    std::ostringstream line;
    line << "  " << std::left << std::setw(10) << tag
         << std::fixed << std::setprecision(2) << ratio << "x"
         << "  running...  " << frames[frame++ % 4];
    std::string s = line.str();
    // Pad to PROGRESS_COL to overwrite any previous longer line
    if ((int)s.size() < PROGRESS_COL) s.append(PROGRESS_COL - s.size(), ' ');
    std::cout << "\r" << s << std::flush;
  };

  // Timing bar — [=========>          ]  ~Ns remaining
  auto draw_bar = [&](const std::string& tag, double ratio,
                      int done, int total, double pass_dur_ms) {
    int remaining_passes = total - done;
    double est_sec = (pass_dur_ms > 0.0)
                     ? (remaining_passes * pass_dur_ms / 1000.0) : 0.0;

    int filled = (total > 0) ? (done * BAR_WIDTH / total) : 0;
    filled = std::min(filled, BAR_WIDTH - 1);

    std::ostringstream bar;
    bar << "  " << std::left << std::setw(10) << tag
        << std::fixed << std::setprecision(2) << ratio << "x  [";
    for (int b = 0; b < BAR_WIDTH; ++b) {
      if (b < filled)       bar << '=';
      else if (b == filled) bar << '>';
      else                  bar << ' ';
    }
    bar << "]";
    if (est_sec > 0.5)
      bar << "  ~" << std::setprecision(0) << est_sec << "s remaining";
    std::string s = bar.str();
    if ((int)s.size() < PROGRESS_COL) s.append(PROGRESS_COL - s.size(), ' ');
    std::cout << "\r" << s << std::flush;
  };

  auto live_line = [&](const char* tag, const ResultRow& row) {
    clear_progress();
    bool is_cold = (std::string(tag) == "cold");
    std::ostringstream ratio_str;
    ratio_str << std::fixed << std::setprecision(2) << row.oversub_ratio << "x";
    std::cout << "  "
              << std::left  << std::setw(6)  << ratio_str.str()
              << std::setw(10) << row.regime
              << std::right
              << std::fixed << std::setprecision(2)
              << std::setw(5) << row.t.steady_p50_ms
              << std::setw(6) << row.t.steady_p90_ms
              << std::setw(6) << row.t.steady_p99_ms
              << std::setw(6) << row.t.steady_max_ms
              << std::setw(6) << row.t.steady_tail_ratio
              << std::setw(7) << std::setprecision(3) << row.t.steady_cv
              << std::setw(6) << std::setprecision(2) << row.steady_jump;
    if (is_cold && row.child_gpu_faults > 0) {
      std::cout << std::setw(8) << row.child_gpu_faults
                << std::setw(8) << std::setprecision(0)
                << (double)row.child_bytes_htod / (1024.0 * 1024.0)
                << " MB"
                << std::setw(6) << (double)row.child_bytes_dtoh / (1024.0 * 1024.0)
                << " MB";
    }
    std::cout << "\n";
    (void)tag;
  };

  // Capture temp_start — after context init, before first measurement
  read_hw_state_start(hw, g_nvml.dev);

  // COLD runs first — fresh child process per ratio resets page tables and
  // driver state before each measurement.  This is the correct baseline:
  // GPU has never touched these pages.  WARM runs second — in-process,
  // pages already resident, measures steady-state throughput.
  // PCIe replay counter — snapshot before measurement passes begin.
  // Delta at end quantifies replay events during the entire run.
  // Non-zero delta on a healthy system is normal under heavy DMA load;
  // large delta (>1000) during a short run indicates link instability.
  unsigned int pcie_replay_pre = 0;
  unsigned int pcie_replay_post = 0;
  nvmlDeviceGetPcieReplayCounter(g_nvml.dev, &pcie_replay_pre);
  clk::time_point test_start = clk::now();
  std::vector<FaultSample> fault_samples;
  fault_samples.reserve(64);
  fault_samples.push_back({test_start, g_cupti.gpu_page_faults});
  clk::time_point last_fault_sample = test_start;

  auto maybe_sample_faults = [&](clk::time_point now) {
    if (ms(last_fault_sample, now) >= 100.0) {
      fault_samples.push_back({now, g_cupti.gpu_page_faults});
      last_fault_sample = now;
    }
  };

  // COLD pass — fault-and-migrate path.
  // Each ratio runs entirely inside a fresh child process so the GPU has
  // never touched those pages.  The child measures via measure_ratio and
  // pipes a compact JSON line back.  Parent parses it — no in-process
  // measure_ratio call here at all.
  // Doc basis: cudaMallocManaged placement = "first touch", GPU-resident
  // page size = 2MB.  Child faults 4KB CPU pages into 2MB GPU pages during
  // measurement.  Process exit reclaims pages — next child starts clean.
  std::cout << "\n  COLD  (" << pass_labels.cold_note << ")\n";
  std::cout << "  " << std::left
            << std::setw(6)  << "SIZE"
            << std::setw(10) << "STATUS"
            << std::right
            << std::setw(5)  << "p50"
            << std::setw(6)  << "p90"
            << std::setw(6)  << "p99"
            << std::setw(6)  << "max"
            << std::setw(6)  << "tail"
            << std::setw(7)  << "cv"
            << std::setw(6)  << "jump"
            << std::setw(8)  << "faults"
            << std::setw(8)  << "htod"
            << std::setw(9)  << "dtoh"
            << "\n"
            << "  " << std::string(86, '-') << "\n";
  std::vector<ResultRow> cold_rows;
  int cold_child_failures = 0;
  double cold_base = 0.0;

  // PCIe throughput peak across cold pass — sampled while child DMA is active.
  // nvmlDeviceGetPcieThroughput returns KB/s over a 20ms rolling window.
  // Sampled once per ratio right after child launch so the window overlaps
  // the page fault migration burst.
  PcieBwResult pcie_bw = probe_pcie_bandwidth(device);

  double cold_pass_dur_ms = 0.0;
  for (size_t i = 0; i < runnable.size(); ++i) {
    double r = runnable[i];
    uint64_t b = runnable_bytes[i];

    // Progress: spinner on pass 0 (no timing data), bar on pass 1+ from pass 0 duration
    auto cold_pass_start = clk::now();
    {
      if (i == 0 || cold_pass_dur_ms <= 0.0)
        draw_spinner("COLD", r);
      else
        draw_bar("COLD", r, (int)i, (int)runnable.size(), cold_pass_dur_ms);
    }

    std::ostringstream cmd;
    // Forward full environment to child so CUPTI can init correctly.
    // popen uses /bin/sh which inherits the parent's environment — the only
    // var we need to prepend explicitly is UM_HOST_CAP_BYTES.
    cmd << "UM_HOST_CAP_BYTES=" << host_cap_bytes << " "
        << argv[0] << " --child --device " << device
        << " --ratio " << std::fixed << std::setprecision(3) << r
        << " --bytes " << b;

    FILE* pipe = popen(cmd.str().c_str(), "r");
    ResultRow row;
    row.oversub_ratio = r;
    row.bytes = b;
    row.status = "child_failed";

    if (pipe) {
      // Sample PCIe throughput ~10ms after child launch — window overlaps
      // the initial page fault migration burst (child touches pages immediately).
      // Sleep is short enough that fgets below won't miss the child output.
      char buf[4096] = {};
      if (fgets(buf, sizeof(buf), pipe)) {
        std::string line(buf);
        // Parse compact JSON: {"status":"ok","steady_p50_ms":X,"steady_cv":X,...}
        auto jget = [&](const std::string& key) -> double {
          auto pos = line.find("\"" + key + "\":");
          if (pos == std::string::npos) return 0.0;
          pos += key.size() + 3;
          return std::stod(line.substr(pos));
        };
        auto jstr = [&](const std::string& key) -> std::string {
          auto pos = line.find("\"" + key + "\":\"");
          if (pos == std::string::npos) return "";
          pos += key.size() + 4;
          auto end = line.find("\"", pos);
          return (end == std::string::npos) ? "" : line.substr(pos, end - pos);
        };
        row.status                    = jstr("status");
        row.t.steady_p50_ms           = jget("steady_p50_ms");
        row.t.steady_cv               = jget("steady_cv");
        row.t.alloc_ms                = jget("alloc_ms");
        row.t.cpu_init_ms             = jget("cpu_init_ms");
        row.t.prefetch_to_gpu_ms      = jget("prefetch_to_gpu_ms");
        row.t.gpu_first_touch_ms      = jget("gpu_first_touch_ms");
        row.t.steady_p90_ms           = jget("steady_p90_ms");
        row.t.steady_p99_ms           = jget("steady_p99_ms");
        row.t.steady_max_ms           = jget("steady_max_ms");
        row.t.steady_tail_ratio       = jget("steady_tail_ratio");
        if (row.t.steady_max_ms <= 0.0) row.t.steady_max_ms = row.t.steady_p99_ms; 
        if (row.t.steady_tail_ratio <= 0.0) row.t.steady_tail_ratio = row.t.steady_p99_ms / std::max(0.001, row.t.steady_p50_ms);
        row.t.steady_mean_ms          = jget("steady_mean_ms");
        row.t.prefetch_to_cpu_ms      = jget("prefetch_to_cpu_ms");
        row.t.cpu_retouch_ms          = jget("cpu_retouch_ms");
        row.t.prefetch_back_to_gpu_ms = jget("prefetch_back_to_gpu_ms");
        row.t.gpu_retouch_ms          = jget("gpu_retouch_ms");
        row.t.pressure_p50_ms         = jget("pressure_p50_ms");
        row.t.pressure_score          = jget("pressure_score");
        row.t.dominant_total          = jstr("dominant_total");
        row.t.dominant_um             = jstr("dominant_um");
        row.um_preferred_location     = jstr("um_preferred_location");
        row.um_last_prefetch          = jstr("um_last_prefetch");
        row.child_gpu_faults          = (uint64_t)jget("child_gpu_faults");
        row.child_bytes_htod          = (uint64_t)jget("child_bytes_htod");
        row.child_bytes_dtoh          = (uint64_t)jget("child_bytes_dtoh");
      }
      int prc = pclose(pipe);
      if (system_exit_code(prc) != 0 || row.status != "ok") {
        std::cerr << "[cold] child failed ratio=" << std::fixed << std::setprecision(2) << r << "\n";
        cold_child_failures++;
        continue;
      }
    } else {
      std::cerr << "[cold] popen failed ratio=" << std::fixed << std::setprecision(2) << r << "\n";
      cold_child_failures++;
      continue;
    }

    if (cold_base <= 0.0) cold_base = std::max(0.001, row.t.steady_p50_ms);
    row.steady_jump    = row.t.steady_p50_ms / cold_base;
    row.stability_index = 1.0 / (1.0 + row.t.steady_cv);
    row.regime = classify_regime(row.steady_jump, row.t.steady_cv,
                                 row.t.gpu_retouch_ms, row.t.steady_p50_ms, /*is_cold=*/true);
    cold_rows.push_back(row);
    maybe_sample_faults(clk::now());
    // Record first pass duration for adaptive bar on subsequent passes
    if (i == 0) cold_pass_dur_ms = ms(cold_pass_start, clk::now());
    live_line("cold", row);
  }

  std::vector<double> cold_r, cold_s;
  for (auto& rr : cold_rows) { cold_r.push_back(rr.oversub_ratio); cold_s.push_back(rr.t.steady_p50_ms); }
  KneeResult2 cold_k = detect_knee2(cold_r, cold_s);

  // WARM pass — resident path.
  // Per ratio: allocate in this process, CPU-touch (4KB pages committed),
  // prefetch to GPU (driver migrates to 2MB GPU pages), sync, then measure.
  // cudaFree after each ratio — next ratio starts with a clean allocation,
  // no cross-ratio allocator state.  No child process needed: pages are
  // resident in THIS process's CUDA context when measure_ratio runs.
  // Doc basis: cudaMallocManaged "first touch" placement; after
  // cudaMemPrefetchAsync + sync pages are GPU-resident at 2MB granularity.
  std::cout << "\n  WARM  (" << pass_labels.warm_note << ")\n";
  std::cout << "  " << std::left
            << std::setw(6)  << "SIZE"
            << std::setw(10) << "STATUS"
            << std::right
            << std::setw(5)  << "p50"
            << std::setw(6)  << "p90"
            << std::setw(6)  << "p99"
            << std::setw(6)  << "max"
            << std::setw(6)  << "tail"
            << std::setw(7)  << "cv"
            << std::setw(6)  << "jump"
            << "\n"
            << "  " << std::string(62, '-') << "\n";
  std::vector<ResultRow> warm_rows;
  double baseline = 0.0;
  double warm_pass_dur_ms = 0.0;

  for (size_t i = 0; i < runnable.size(); ++i) {
    double r = runnable[i];
    uint64_t b = runnable_bytes[i];
    size_t n_floats = b / sizeof(float);
    const size_t stride = 4096 / sizeof(float);

    auto warm_pass_start = clk::now();
    {
      if (i == 0 || warm_pass_dur_ms <= 0.0)
        draw_spinner("WARM", r);
      else
        draw_bar("WARM", r, (int)i, (int)runnable.size(), warm_pass_dur_ms);
    }

    // Step 1: allocate — timed
    float* p = nullptr;
    auto a0 = clk::now();
    if (cudaMallocManaged(&p, b) != cudaSuccess || !p) {
      std::cerr << "[warm] alloc failed ratio=" << std::fixed << std::setprecision(2) << r << "\n";
      continue;
    }
    auto a1 = clk::now();

    // Step 2: CPU first-touch — commits pages at 4KB CPU granularity — timed
    auto b0 = clk::now();
    for (size_t j = 0; j < n_floats; j += stride) p[j] = 1.0f;
    auto b1 = clk::now();

    // Step 3: prefetch to GPU — driver migrates to 2MB GPU pages — timed
    auto c0 = clk::now();
    umPrefetchToDevice(p, b, device, 0);
    cudaDeviceSynchronize();
    auto c1 = clk::now();
    // Pages are now GPU-resident. measure_ratio sees zero fault cost.

    // Step 4: measure — pages are GPU-resident, use gpu_touch_ms directly.
    // One throwaway touch to populate TLBs, then reps steady samples.
    double first_touch = gpu_touch_ms(p, n_floats, stride);
    std::vector<double> wsamp = steady_samples(p, n_floats, stride, reps);
    std::sort(wsamp.begin(), wsamp.end());
    double warm_p50  = pct_sorted(wsamp, 0.50);
    double warm_p90  = pct_sorted(wsamp, 0.90);
    double warm_p99  = pct_sorted(wsamp, 0.99);
    double warm_mean = std::accumulate(wsamp.begin(), wsamp.end(), 0.0) / wsamp.size();
    double warm_cv   = coeff_var(wsamp);

    // Step 5: prefetch back to CPU and retouch — timed
    auto rt0 = clk::now();
    umPrefetchToCPU(p, b, 0);
    cudaDeviceSynchronize();
    auto rt1 = clk::now();
    auto fb0 = clk::now();
    umPrefetchToDevice(p, b, device, 0);
    cudaDeviceSynchronize();
    auto fb1 = clk::now();
    double gpu_retouch = gpu_touch_ms(p, n_floats, stride);

    // Step 6: free — releases GPU pages, next ratio starts clean
    cudaFree(p);

    ResultRow row;
    row.oversub_ratio              = r;
    row.bytes                      = b;
    row.status                     = "ok";
    row.t.alloc_ms                 = ms(a0, a1);
    row.t.cpu_init_ms              = ms(b0, b1);
    row.t.prefetch_to_gpu_ms       = ms(c0, c1);
    row.t.gpu_first_touch_ms       = first_touch;
    row.t.steady_p50_ms            = warm_p50;
    row.t.steady_p90_ms            = warm_p90;
    row.t.steady_p99_ms            = warm_p99;
    row.t.steady_max_ms            = wsamp.empty() ? 0.0 : wsamp.back();
    row.t.steady_tail_ratio        = row.t.steady_p99_ms / std::max(0.001, row.t.steady_p50_ms);
    row.t.steady_mean_ms           = warm_mean;
    row.t.steady_cv                = warm_cv;
    row.t.prefetch_to_cpu_ms       = ms(rt0, rt1);
    row.t.prefetch_back_to_gpu_ms  = ms(fb0, fb1);
    row.t.gpu_retouch_ms           = gpu_retouch;

    if (baseline <= 0.0) baseline = std::max(0.001, row.t.steady_p50_ms);
    row.steady_jump     = row.t.steady_p50_ms / baseline;
    row.stability_index = 1.0 / (1.0 + row.t.steady_cv);
    row.regime = classify_regime(row.steady_jump, row.t.steady_cv,
                                 row.t.gpu_retouch_ms, row.t.steady_p50_ms, /*is_cold=*/false);
    warm_rows.push_back(row);
    maybe_sample_faults(clk::now());
    if (i == 0) warm_pass_dur_ms = ms(warm_pass_start, clk::now());
    live_line("warm", row);
  }

  std::vector<double> warm_r, warm_s;
  for (auto& rr : warm_rows) { warm_r.push_back(rr.oversub_ratio); warm_s.push_back(rr.t.steady_p50_ms); }
  KneeResult2 warm_k = detect_knee2(warm_r, warm_s);

  // ── Residency half-life ratio ─────────────────────────────────────────────
  // First warm ratio where residency decay score >= 1.5x baseline score.
  // decay_score = steady_tail_ratio * (1 + steady_cv)
  // Uses warm passes only. -1.0 = never reached / insufficient data.
  double residency_half_life_ratio = -1.0;
  {
    double baseline_score = -1.0;
    for (const auto& rr : warm_rows) {
      if (rr.regime == "RESIDENT") {
        baseline_score = rr.t.steady_tail_ratio * (1.0 + rr.t.steady_cv);
        break;
      }
    }
    if (baseline_score > 0.0) {
      double threshold = baseline_score * 1.5;
      for (const auto& rr : warm_rows) {
        if (rr.regime != "RESIDENT") continue;
        double score = rr.t.steady_tail_ratio * (1.0 + rr.t.steady_cv);
        if (score >= threshold) {
          residency_half_life_ratio = rr.oversub_ratio;
          break;
        }
      }
    }
  }

  bool pressure_enabled = true;
  std::vector<ResultRow> pressure_rows;

  // Pressure stability: 3 consecutive identical passes at max ratio.
  // Only FAULT_MIGRATION regimes (steady_jump >= 50) benefit from this;
  // RESIDENT regimes are stable by definition.  We always run all 3 to
  // collect data, but the stability verdict is only meaningful when the
  // cold pass at this ratio was MIGRATION_HEAVY or PINGPONG_SUSPECT.
  static const int PRESSURE_REPEATS = 3;
  double pressure_cv_history[PRESSURE_REPEATS] = {};

  std::cout << "\n" << std::flush;

  // settling_time_ms: wall-clock from first pressure pass start
  // until the pass where CV stabilises. Zero if never stable.
  double settling_time_ms  = 0.0;
  auto   pressure_t_start  = clk::now();   // start of pass 0
  auto   pressure_t_stable = clk::now();   // updated when stable pass found

  {
    size_t idx = runnable.size() - 1;
    double pressure_pass_dur_ms = 0.0;
    for (int pi = 0; pi < PRESSURE_REPEATS; ++pi) {
      // Pass 0: spinner (no timing). Pass 1+: bar with time estimate.
      if (pi == 0 || pressure_pass_dur_ms <= 0.0)
        draw_spinner("PRESSURE", runnable[idx]);
      else
        draw_bar("PRESSURE", runnable[idx], pi, PRESSURE_REPEATS, pressure_pass_dur_ms);

      auto pass_start = clk::now();
      ResultRow prow = measure_ratio(runnable[idx], runnable_bytes[idx], device,
                                     nv.pcie_gen_curr, nv.pcie_width_curr, reps, true);
      // Re-classify using the same rule as warm/cold: jump must confirm migration.
      prow.regime = classify_regime(prow.steady_jump, prow.t.steady_cv,
                                    prow.t.gpu_retouch_ms, prow.t.steady_p50_ms);

      pressure_cv_history[pi] = prow.t.steady_cv;
      pressure_rows.push_back(prow);
      maybe_sample_faults(clk::now());
      if (pi == 0) {
        pressure_t_start    = pass_start;
        pressure_pass_dur_ms = ms(pass_start, clk::now());
      }
      if (pressure_cv_history[pi] < 0.05 && settling_time_ms == 0.0)
        pressure_t_stable = clk::now();

      // Per-rep rows suppressed — cv progression shown in summary block.
    }
  }

  // Compute settling_time_ms after loop — valid only if pressure truly converged.
  // Requires final pass CV < 0.05 to avoid counting transient mid-loop dips.
  if (settling_time_ms == 0.0) {
    if (pressure_cv_history[PRESSURE_REPEATS-1] < 0.05) {
      for (int i = 0; i < PRESSURE_REPEATS; ++i) {
        if (pressure_cv_history[i] < 0.05) {
          settling_time_ms = ms(pressure_t_start, pressure_t_stable);
          break;
        }
      }
    }
  }

  // Single pressure line: cv progression across all repeats
  clear_progress();
  std::cout << "    "
            << std::fixed << std::setprecision(2) << runnable.back() << "x"
            << "  cv=";
  for (int pi = 0; pi < PRESSURE_REPEATS; ++pi) {
    if (pi > 0) std::cout << "→";
    std::cout << std::setprecision(3) << pressure_cv_history[pi];
  }

  // Convergence analysis.
  // Three complementary criteria — any one sufficient to declare stabilized:
  // 1. Absolute floor: all cv values < 0.05 — inherently stable, no learning needed.
  // 2. Per-step relative delta: |cv[i] - cv[i-1]| < 20% of cv[i-1].
  // 3. Overall drop fallback: cv_drop_pct > 50%.
  //
  // Criterion 1 catches RESIDENT workloads where cv is flat-low.
  // Criteria 2/3 catch MIGRATION workloads where cv drops as pages warm up.
  //
  // All measurement passes complete — capture final hardware state
  read_hw_state_post(hw, g_nvml.dev);

  // stable_pass is 1-based. -1 means no criterion satisfied.
  double pressure_cv_drop_pct = 0.0;
  int    pressure_stable_pass = -1;
  bool   pressure_is_fault_migration = false;

  if (PRESSURE_REPEATS >= 2) {
    double cv0 = pressure_cv_history[0];
    double cvN = pressure_cv_history[PRESSURE_REPEATS - 1];
    pressure_cv_drop_pct = (cv0 > 1e-9) ? (cv0 - cvN) / cv0 : 0.0;
    if (pressure_cv_history[PRESSURE_REPEATS-1] < 0.05 && pressure_stable_pass < 0) pressure_stable_pass = PRESSURE_REPEATS;

    // Criterion 1: all passes below absolute stable floor
    bool all_low = true;
    for (int i = 0; i < PRESSURE_REPEATS; ++i)
      if (pressure_cv_history[i] >= 0.05) { all_low = false; break; }
    if (all_low) {
      pressure_stable_pass = 1;  // stable from first pass
    } else {
      // Criterion 2: per-step delta within 20% of prior cv AND cv not rising above floor
      for (int i = 1; i < PRESSURE_REPEATS; ++i) {
        double delta     = std::fabs(pressure_cv_history[i] - pressure_cv_history[i - 1]);
        double threshold = 0.20 * std::max(pressure_cv_history[i - 1], 1e-9);
        // Only counts as stable if CV is not rising above 0.10
        if (delta < threshold && pressure_cv_history[i] < 0.10) { pressure_stable_pass = i + 1; break; }
      }
      // Criterion 3: overall drop > 50%
      if (pressure_stable_pass < 0 && pressure_cv_drop_pct > 0.50)
        pressure_stable_pass = PRESSURE_REPEATS;
    }
  }

  for (auto& rr : cold_rows) {
    if (std::fabs(rr.oversub_ratio - runnable.back()) < 0.01) {
      if (rr.regime == "MIGRATION_HEAVY" || rr.regime == "PINGPONG_SUSPECT")
        pressure_is_fault_migration = true;
    }
  }
  (void)pressure_is_fault_migration; // used in write_summary_txt / write_run_json

  // Complete the [pressure] line with stable/noise verdict
  {
    bool stability_meaningful = (pressure_is_fault_migration || pressure_cv_history[0] > 0.10 || pressure_cv_history[PRESSURE_REPEATS-1] < 0.05);
    bool all_low_cv = true;
    for (int i = 0; i < PRESSURE_REPEATS; ++i) {
      if (pressure_cv_history[i] >= 0.05) { all_low_cv = false; break; }
    }
    if (!stability_meaningful)
      std::cout << "  stable=no";
    else if (all_low_cv)
      std::cout << "  stable=LOW_CV_ALL";
    else if (pressure_stable_pass > 0)
      std::cout << "  stable=pass" << pressure_stable_pass;
    else
      std::cout << "  stable=no";
  }
  // Pressure regime: use last pass (most settled)
  std::string pressure_regime = pressure_rows.empty() ? "UNKNOWN"
                                                       : pressure_rows.back().regime;
  std::string pressure_noise = "UNKNOWN";
  {
    double cv_last = pressure_cv_history[PRESSURE_REPEATS - 1];
    if (cv_last < 0.05) pressure_noise = "LOW";
    else if (cv_last < 0.15) pressure_noise = "MEDIUM";
    else pressure_noise = "HIGH";
  }
  std::cout << "  " << pressure_regime
            << "  noise=" << pressure_noise << "\n";

  write_summary_txt(run_dir, nv, host, host_cap_gib, cold_k, warm_k, pressure_enabled,
                    cold_child_failures,
                    pressure_cv_history, PRESSURE_REPEATS,
                    pressure_cv_drop_pct, pressure_stable_pass, pressure_is_fault_migration,
                    hw);

  // Capture PCIe replay delta — all measurement passes complete
  clk::time_point test_end = clk::now();
  fault_samples.push_back({test_end, g_cupti.gpu_page_faults});
  double test_seconds = std::max(0.001, ms(test_start, test_end) / 1000.0);
  nvmlDeviceGetPcieReplayCounter(g_nvml.dev, &pcie_replay_post);
  unsigned int pcie_replay_delta = (pcie_replay_post >= pcie_replay_pre)
                                   ? (pcie_replay_post - pcie_replay_pre) : 0;

  // ── Thrash detection ─────────────────────────────────────────────────────
  ThrashMetrics thrash = classify_thrash(
      pressure_cv_history, PRESSURE_REPEATS,
      pressure_stable_pass,
      pcie_bw.h2d_gbs, pcie_bw.d2h_gbs,
      cold_rows);

  // ── Residency window ─────────────────────────────────────────────────────
  ResidencyWindow reswin = compute_residency_window(cold_rows, vram_gib);


  bool uma_platform = (nv.um_paradigm == "FULL_HARDWARE_COHERENT");

  // ── Summary block ─────────────────────────────────────────────────────────

  // ── MAF / BPF computation ─────────────────────────────────────────────────
  // Must be before verdict block — maf used in UM_THRASHING / MIGRATION_PRESSURE gates.
  double maf      = 0.0;
  double bpf_htod  = 0.0;
  double bpf_total = 0.0;
  {
    uint64_t total_pass_bytes = 0;
    for (uint64_t b : runnable_bytes) total_pass_bytes += b;  // cold passes
    for (uint64_t b : runnable_bytes) total_pass_bytes += b;  // warm passes
    if (!runnable_bytes.empty())
      total_pass_bytes += runnable_bytes.back() * (uint64_t)3; // pressure x3
    if (g_cupti_ok && total_pass_bytes > 0) {
      cupti_flush();
      maf = (double)(g_cupti.bytes_htod + g_cupti.bytes_dtoh) / (double)total_pass_bytes;
      if (g_cupti.gpu_page_faults > 0) {
        bpf_htod  = (double)g_cupti.bytes_htod / (double)g_cupti.gpu_page_faults;
        bpf_total = (double)(g_cupti.bytes_htod + g_cupti.bytes_dtoh) / (double)g_cupti.gpu_page_faults;
      }
    }
  }

  // ── Fault rate metrics ────────────────────────────────────────────────────
  double fault_rate_avg     = (double)g_cupti.gpu_page_faults / test_seconds;
  double migration_efficiency = (maf > 0.0) ? (1.0 / maf) : 0.0;
  double oscillation_ratio = 0.0;
  if (g_cupti.bytes_htod > 0 && g_cupti.bytes_dtoh > 0) {
    double min_dir = (double)std::min(g_cupti.bytes_htod, g_cupti.bytes_dtoh);
    double max_dir = (double)std::max(g_cupti.bytes_htod, g_cupti.bytes_dtoh);
    oscillation_ratio = min_dir / max_dir;
  }
  double fault_max_window_rate = 0.0;
  for (size_t i = 1; i < fault_samples.size(); ++i) {
    double dt = ms(fault_samples[i-1].ts, fault_samples[i].ts) / 1000.0;
    if (dt <= 1e-6) continue;
    uint64_t f0 = fault_samples[i-1].gpu_faults;
    uint64_t f1 = fault_samples[i].gpu_faults;
    if (f1 < f0) continue;
    double window_rate = (double)(f1 - f0) / dt;
    if (window_rate > fault_max_window_rate) fault_max_window_rate = window_rate;
  }
  double fault_burst_ratio = (fault_rate_avg > 1e-6)
                             ? (fault_max_window_rate / fault_rate_avg) : 0.0;
  double fault_pressure_index = fault_rate_avg * fault_burst_ratio;

  // Finalize KV-cache pressure score now that all three inputs are available
  if (um_headroom_ratio > 0.0)
    kv_pressure_score = fault_pressure_index * (1.0 - migration_efficiency) / um_headroom_ratio;
  if      (kv_pressure_score > 200.0) kv_pressure_level = "HIGH";
  else if (kv_pressure_score > 100.0) kv_pressure_level = "MODERATE";

  // Compute verdict strings first
  std::string thermal_v, memory_v, pressure_v, overall_verdict;
  {
    // thermal
    if      (hw.temp_end_c >= hw.throttle_threshold_c) thermal_v = "THERMAL_THROTTLE";
    else if (hw.power_available &&
             hw.power_draw_mw >= hw.power_limit_mw * 0.97) thermal_v = "POWER_CAP_ACTIVE";
    else if (hw.temp_start_c >= hw.throttle_threshold_c) thermal_v = "INVALID_STATE";
    else    thermal_v = "STABLE";

    // memory
    double ceil_pct = host.ceiling_utilization * 100.0;
    if      (ceil_pct >= 100.0) memory_v = "CRITICAL";
    else if (ceil_pct >=  85.0) memory_v = "DANGER";
    else if (ceil_pct >=  70.0) memory_v = "WARN";
    else                        memory_v = "OK";

    // pressure
    std::string last_regime = pressure_rows.empty() ? "UNKNOWN" : pressure_rows.back().regime;
    if      (last_regime == "FAULT_PATH")    pressure_v = "FAULT_PATH";
    else if (last_regime == "FAULT_HYBRID")  pressure_v = "FAULT_HYBRID";
    else                                     pressure_v = last_regime;

    // hardware clean = no physical subsystem fault
    bool hardware_clean = (thermal_v == "STABLE")
                          && (memory_v == "OK" || memory_v == "WARN")
                          && (pcie_replay_delta < 100);

    // cv_mean across pressure passes (skip pass 0 — first pass always noisy)
    double cv_mean = 0.0;
    {
      int n = 0;
      for (int i = 1; i < PRESSURE_REPEATS; i++) {
        cv_mean += pressure_cv_history[i];
        n++;
      }
      if (n > 0) cv_mean /= n;
    }

    // CRITICAL — fatal conditions
    bool is_critical = (cold_child_failures > 0)
                       || (thermal_v == "THERMAL_THROTTLE")
                       || (thermal_v == "INVALID_STATE")
                       || (memory_v == "CRITICAL")
                       || (memory_v == "DANGER");

    // DEGRADED — physical subsystem bad, not workload-induced
    bool is_degraded = !is_critical && (
                         (thermal_v == "POWER_CAP_ACTIVE")
                         || (pcie_replay_delta >= 100)
                         || (!hardware_clean));

    // UM_THRASHING — workload hit migration boundary, hardware clean
    // All three required — prevents single noisy pass from promoting
    bool is_um_thrashing = !is_critical && !is_degraded && hardware_clean
                           && (thrash.thrash_score >= 0.45)
                           && (maf >= 2.0)
                           && (!thrash.settled);

    // MIGRATION_PRESSURE — paired instability signals, hardware clean
    // maf alone is not sufficient — must be paired with thrash_score
    // OR settle=NO with sustained CV elevation
    bool is_migration_pressure = !is_critical && !is_degraded && !is_um_thrashing
                                 && hardware_clean
                                 && (((thrash.thrash_score >= 0.15) && (maf >= 1.5))
                                     || (!thrash.settled && cv_mean > 0.15));

    // HEALTHY_LIMITED — core ratios (<= 1.00x) clamped by host memory
    bool is_healthy_limited = !is_critical && !is_degraded
                              && !is_um_thrashing && !is_migration_pressure
                              && (skipped_core_ratios > 0);

    if      (is_critical)           overall_verdict = "CRITICAL";
    else if (is_degraded)           overall_verdict = "DEGRADED";
    else if (is_um_thrashing)       overall_verdict = "UM_THRASHING";
    else if (is_migration_pressure) overall_verdict = "MIGRATION_PRESSURE";
    else if (is_healthy_limited)    overall_verdict = "HEALTHY_LIMITED";
    else                            overall_verdict = "HEALTHY";
  }

  bool show_cache = false;
  {
    if (host.mem_avail > 0)
      show_cache = ((double)host.cache_recoverable_bytes / (double)host.mem_avail > 0.20);
  }

  PsiSnapshot psi_end = read_psi_memory();  // PSI snapshot after all passes complete
  std::string psi_state_str = psi_state(psi_start, psi_end);

  std::cout << "\n" << HDR << "\n";
  std::cout << "  VERDICT   " << overall_verdict << "\n";
  std::cout << HDR << "\n\n";

  // thermal line
  {
    std::ostringstream v;
    v << std::fixed << std::setprecision(0)
      << hw.temp_start_c << "→" << hw.temp_end_c << "°C"
      << "  " << (hw.temp_drift_c >= 0 ? "+" : "") << hw.temp_drift_c << "°C";
    if (hw.power_available)
      v << "  |  " << std::setprecision(1) << (hw.power_draw_mw/1000.0)
        << "W / " << std::setprecision(0) << (hw.power_limit_mw/1000.0) << "W"
        << "  (P" << hw.pstate_start << ")";
    std::cout << "  " << std::left << std::setw(10) << "thermal"
              << std::setw(12) << thermal_v << v.str() << "\n";
  }

  // memory line
  {
    std::ostringstream v;
    v << std::fixed << std::setprecision(2)
      << "free=" << gib(emv.free_bytes) << " GiB"
      << "  total=" << gib(emv.total_bytes) << " GiB"
      << "  src=" << emv.source
      << "  ceil=" << std::setprecision(0) << host.ceiling_utilization * 100.0 << "%"
      << (host.overcommit ? "  [overcommit]" : "");
    std::cout << "  " << std::left << std::setw(10) << "memory"
              << std::setw(12) << memory_v << v.str() << "\n";
  }

  // headroom line
  {
    const char* hr_label = (um_headroom_ratio >= 2.0) ? "LARGE" :
                           (um_headroom_ratio >= 1.3) ? "SAFE"  :
                           (um_headroom_ratio >= 1.0) ? "LOW"   : "RISK";
    std::cout << "  " << std::left << std::setw(10) << "headroom"
              << std::setw(12) << hr_label
              << "ratio=" << std::fixed << std::setprecision(2) << um_headroom_ratio << "\n";
  }

  // llm_pressure line
  std::cout << "  " << std::left << std::setw(10) << "llm_kv"
            << std::setw(12) << kv_pressure_level
            << "score=" << std::fixed << std::setprecision(1) << kv_pressure_score << "\n";

  // host PSI pressure line
  {
    std::string psi_disp = psi_state_str;
    if (psi_state_str == "n/a") psi_disp = "n/a (kernel)";
    std::cout << "  " << std::left << std::setw(10) << "host_psi"
              << std::setw(12) << (psi_state_str == "n/a" ? "" : psi_state_str)
              << "memory_psi=" << psi_disp << "\n";
  }

  // cache line
  if (show_cache) {
    std::ostringstream v;
    v << std::fixed << std::setprecision(2)
      << gib(host.cache_recoverable_bytes) << " GiB recoverable ("
      << std::setprecision(0) << host.cache_frac_of_total * 100.0
      << "%)  —  drop_caches";
    std::cout << "  " << std::left << std::setw(10) << "cache"
              << std::setw(12) << "INFO" << v.str() << "\n";
  }

  // oom line — always present, hardware-aware, no duplication of transport section
  {
    const char* oom_label;
    std::ostringstream oom_detail;

    if (host.zombie_oom_structural) {
      // UMA structural risk — swap state drives this, not fault metrics
      oom_label = "STRUCTURAL";
      oom_detail << "swap=NONE  cache="
                 << std::fixed << std::setprecision(2)
                 << gib(host.cache_recoverable_bytes) << " GiB  stall risk on large alloc";
    } else if (uma_platform) {
      // UMA, no structural risk — headroom is the only signal
      oom_label = (um_headroom_ratio >= 1.3) ? "SAFE" : (um_headroom_ratio >= 1.0) ? "LOW" : "RISK";
      oom_detail << "unified_pool  headroom=" << std::setprecision(2) << um_headroom_ratio << "x";
    } else {
      // Discrete GPU — label driven by burst+maf combination
      bool unstable = (thrash.thrash_score > 0.0 || !thrash.settled);
      if      (fault_burst_ratio >= 3.5 && maf >= 2.0 && unstable) oom_label = "ELEVATED";
      else if (fault_burst_ratio >= 3.0 && maf >= 2.0 && unstable) oom_label = "WATCH";
      else                                                           oom_label = "SAFE";
      oom_detail << "burst=" << std::fixed << std::setprecision(2) << fault_burst_ratio << "x"
                 << "  maf=" << std::setprecision(2) << maf
                 << "  →  " << (std::string(oom_label) == "SAFE" ? "nominal" :
                                std::string(oom_label) == "WATCH" ? "monitor under load" :
                                "elevated migration pressure");
    }
    std::cout << "  " << std::left << std::setw(10) << "oom"
              << std::setw(12) << oom_label
              << oom_detail.str() << "\n";
  }

  // pressure line
  {
    double cv_last = pressure_cv_history[PRESSURE_REPEATS-1];
    std::string pressure_noise = (cv_last < 0.05) ? "LOW" : (cv_last < 0.15) ? "MEDIUM" : "HIGH";
    std::ostringstream v;
    v << std::fixed << std::setprecision(2) << runnable.back() << "x"
      << "  cv=" << std::setprecision(3) << pressure_cv_history[0]
      << "→" << pressure_cv_history[PRESSURE_REPEATS-1];
    {
      bool stability_meaningful = (pressure_is_fault_migration || pressure_cv_history[0] > 0.10 || pressure_cv_history[PRESSURE_REPEATS-1] < 0.05);
      bool all_low_cv_summary = true;
      for (int i = 0; i < PRESSURE_REPEATS; ++i)
        if (pressure_cv_history[i] >= 0.05) { all_low_cv_summary = false; break; }
      if (!stability_meaningful)         v << "  stable=no";
      else if (all_low_cv_summary)       v << "  stable=LOW_CV_ALL";
      else if (pressure_stable_pass > 0) v << "  stable=pass" << pressure_stable_pass;
      else                               v << "  stable=no";
    }
    v << "  noise=" << pressure_noise;
    std::cout << "  " << std::left << std::setw(10) << "pressure"
              << std::setw(12) << pressure_v << v.str() << "\n";
  }

  // ── direction ratio ───────────────────────────────────────────────────────

  double um_htod_gb = (double)g_cupti.bytes_htod / (1024.0 * 1024.0 * 1024.0);
  double um_dtoh_gb = (double)g_cupti.bytes_dtoh / (1024.0 * 1024.0 * 1024.0);
  double um_dir_hi  = std::max(um_htod_gb, um_dtoh_gb);
  double um_dir_lo  = std::min(um_htod_gb, um_dtoh_gb);
  double um_direction_ratio = (um_dir_hi > 0.0001) ? (um_dir_hi / std::max(um_dir_lo, 0.0001)) : 1.0;
  // Platform-aware direction trend interpretation.
  // PCIe/NVLink:  H2D_DOMINANT = normal fault-driven inbound migration.
  //               D2H_DOMINANT = reverse migration, potential thrash.
  // Coherent C2C: no physical migration — asymmetry means coherence ownership
  //               churn (CPU retaining pages the GPU repeatedly requests).
  //               Labels reflect ownership pattern, not transfer direction.
  bool is_coherent_platform = (nv.um_paradigm == "FULL_HARDWARE_COHERENT");
  const char* um_direction_trend;
  if (um_direction_ratio <= 1.50) {
    um_direction_trend = "BALANCED";
  } else if (is_coherent_platform) {
    um_direction_trend = (um_htod_gb > um_dtoh_gb)
                         ? "GPU_OWNERSHIP_DEMAND"
                         : "CPU_RETENTION";
  } else {
    um_direction_trend = (um_htod_gb > um_dtoh_gb) ? "H2D_DOMINANT" : "D2H_DOMINANT";
  }

  write_run_json(run_dir, nv, host, host_cap_gib, cold_rows, warm_rows, pressure_rows, cold_k, warm_k, pressure_enabled, cold_child_failures,
                 skipped_ratios, pressure_cv_history, PRESSURE_REPEATS, pressure_cv_drop_pct, pressure_stable_pass, pressure_is_fault_migration,
                 hw, transport, nvlink_c, prop, pcie_replay_delta, pcie_bw, thrash, reswin, maf, migration_efficiency, oscillation_ratio, settling_time_ms, bpf_htod, bpf_total,
                 um_direction_ratio, um_direction_trend, test_seconds, fault_max_window_rate, fault_burst_ratio, residency_half_life_ratio, kv_pressure_score, kv_pressure_level, psi_start, psi_end, psi_state_str);

  // ── Transport & Migration ─────────────────────────────────────────────────
  std::cout << "\n" << HDR << "\n";
  std::cout << "  TRANSPORT & MIGRATION\n";
  std::cout << HDR << "\n\n";

  // UM intent — from last successful cold row (most representative)
  {
    std::string pref = "n/a", last_pf = "n/a";
    bool have_intent = false;
    for (auto it = cold_rows.rbegin(); it != cold_rows.rend(); ++it) {
      if (!it->um_preferred_location.empty() && it->um_preferred_location != "n/a") {
        pref    = it->um_preferred_location;
        last_pf = it->um_last_prefetch;
        have_intent = true;
        break;
      }
    }
    std::cout << "  UM intent preferred_location=" << pref;
    if (!have_intent) std::cout << " (arch)";
    std::cout << "  last_prefetch=" << last_pf;
    if (!have_intent) std::cout << " (arch)";
    std::cout << "\n\n";
  }

  if (transport.layer == TransportLayer::PCIe) {
    std::cout << "  PCIe      replay=";
    std::cout << pcie_replay_delta;
    if      (pcie_replay_delta == 0)       std::cout << " (clean)";
    else if (pcie_replay_delta < 100)      std::cout << " (low)";
    else                                   std::cout << " ! elevated";
    if (pcie_bw.ok)
      std::cout << std::fixed << std::setprecision(1)
                << "  |  H2D=" << pcie_bw.h2d_gbs << " GB/s"
                << "  D2H=" << pcie_bw.d2h_gbs << " GB/s";
    std::cout << "\n\n";
  }

  std::cout << std::fixed << std::setprecision(1)
            << "  migration H2D=" << thrash.h2d_gbs << " GB/s"
            << "  D2H=" << thrash.d2h_gbs << " GB/s"
            << "  symmetry=" << std::setprecision(2) << thrash.symmetry
            << "  oscillation_ratio=" << std::setprecision(2) << oscillation_ratio << "\n";
  std::cout << "            settle=";
  std::cout << (thrash.settled ? "YES" : "NO");
  if (thrash.settled && settling_time_ms > 0.0)
    std::cout << "  settle_ms=" << std::fixed << std::setprecision(0) << settling_time_ms;
  std::cout << "  thrash_score=" << std::setprecision(2) << thrash.thrash_score
            << "  state=" << thrash.state << "\n";
  if (g_cupti_ok && g_cupti.records_total > 0) {
    std::cout << std::setprecision(2)
              << "            maf=" << maf
              << "  migration_efficiency=" << std::setprecision(3) << migration_efficiency << "\n"
              << "            bpf_htod=" << std::setprecision(0) << bpf_htod / (1024.0*1024.0) << " MB/fault"
              << "  bpf_total=" << bpf_total / (1024.0*1024.0) << " MB/fault\n";
    std::cout << "            fault_density=" << std::setprecision(2) << thrash.repeat_fault_density;
    if (reswin.detected)
      std::cout << "  fault_at_first=" << std::setprecision(2) << reswin.fault_onset_ratio
                << "x  (" << reswin.fault_onset_gib << " GiB)";
    std::cout << "\n";
    if (residency_half_life_ratio > 0.0)
      std::cout << "            residency_half_life_ratio=" << std::setprecision(2)
                << residency_half_life_ratio << "x\n";
    else
      std::cout << "            residency_half_life_ratio=n/a\n";
    std::cout << "\n";

    // CUPTI line
    std::cout << "  CUPTI     gpu_faults=" << g_cupti.gpu_page_faults;
    if (g_cupti.volta_plus)
      std::cout << "  cpu_faults=" << g_cupti.cpu_page_faults;
    else
      std::cout << "  cpu_faults=n/a";
    std::cout << "  thrashing=" << g_cupti.thrashing;
    if (g_cupti.volta_plus)
      std::cout << "  throttling=" << g_cupti.throttling;
    else
      std::cout << "  throttling=n/a";
    std::cout << "\n";
    std::cout << std::fixed << std::setprecision(2)
              << "            H2D=" << (double)g_cupti.bytes_htod / (1024.0*1024.0*1024.0) << " GB"
              << "  D2H=" << (double)g_cupti.bytes_dtoh / (1024.0*1024.0*1024.0) << " GB";
    if (g_cupti.bytes_htod == 0 && g_cupti.bytes_dtoh == 0)
      std::cout << "  [note: migration bytes not captured — known CUPTI limitation on some platforms]";
    std::cout << "\n";
    std::cout << "            direction_ratio=" << std::setprecision(2) << um_direction_ratio
              << "  trend=" << um_direction_trend;
    if (is_coherent_platform && um_direction_ratio > 1.50)
      std::cout << "  [coherence ownership — not physical transfer]";
    std::cout << "\n";
    {
      std::cout << std::fixed << std::setprecision(1)
                << "            fault_rate=" << fault_rate_avg << " faults/sec\n"
                << "            fault_max_window_rate=" << fault_max_window_rate << " faults/sec\n"
                << std::setprecision(2)
                << "            fault_burst_ratio=" << fault_burst_ratio << "x\n"
                << std::setprecision(1)
                << "            fault_pressure_index=" << fault_pressure_index << "\n";
    }
    if (g_cupti.zero_end_ts_skipped > 0) {
      bool is_coherent = (nv.um_paradigm == "FULL_HARDWARE_COHERENT");
      std::cout << "            [note] " << g_cupti.zero_end_ts_skipped
                << " GPU_PAGE_FAULT records skipped (end=0 timestamp";
      if (is_coherent)
        std::cout << " bug — confirmed NVIDIA Jul 2025, no fix as of CUDA 13.2)\n";
      else
        std::cout << " — unexpected on this platform)\n";
    }
    if (g_cupti.records_dropped > 0)
      std::cout << "            [warn] " << g_cupti.records_dropped
                << " records dropped — buffer pressure, results may be incomplete\n";
  } else if (g_cupti.not_supported) {
    std::cout << "  CUPTI     not supported on this platform"
              << " — MAF/BPF/fault metrics unavailable\n";
  } else if (g_cupti.callbacks_failed) {
    std::cout << "  CUPTI     callback registration failed — counters unavailable\n";
  } else if (g_cupti_ok && g_cupti.records_total == 0) {
    std::cout << "  CUPTI     active but no UM counter records received";
    if (g_cupti.zero_end_ts_skipped > 0)
      std::cout << "  (" << g_cupti.zero_end_ts_skipped << " skipped: end=0 timestamp bug)";
    std::cout << "\n";
  }

  // ── Warnings ─────────────────────────────────────────────────────────────
  bool any_warning = (thermal_v == "THERMAL_THROTTLE" || thermal_v == "POWER_CAP_ACTIVE"
                      || thermal_v == "INVALID_STATE"
                      || memory_v == "DANGER" || memory_v == "CRITICAL"
                      || host.zombie_oom_structural);
  {
    bool cold_unreliable = false, warm_unreliable = false;
    for (const auto& r : cold_rows) if (r.t.steady_cv > 1.0) cold_unreliable = true;
    for (const auto& r : warm_rows) if (r.t.steady_cv > 1.0) warm_unreliable = true;
    if (cold_unreliable || warm_unreliable) any_warning = true;
  }
  if (uma_platform && host.cudamemgetinfo_unreliable
      && std::abs((double)host.cudamemgetinfo_delta) > 1e9) any_warning = true;

  if (any_warning) {
    std::cout << "\n" << HDR << "\n";
    std::cout << "  WARNINGS\n";
    std::cout << HDR << "\n\n";
    if (thermal_v == "THERMAL_THROTTLE")
      std::cout << "  ! thermal  Peak " << hw.temp_end_c << "°C hit throttle threshold ("
                << hw.throttle_threshold_c << "°C) — timing measurements affected\n";
    if (thermal_v == "POWER_CAP_ACTIVE")
      std::cout << "  ! thermal  Power cap enforced during run — migration BW may be reduced\n";
    if (thermal_v == "INVALID_STATE")
      std::cout << "  ! thermal  GPU throttled before measurements began — results invalid\n";
    if (memory_v == "DANGER" || memory_v == "CRITICAL")
      std::cout << "  ! memory   Reduce committed memory or add swap before workload\n";
    {
      bool cold_unreliable = false, warm_unreliable = false;
      for (const auto& r : cold_rows) if (r.t.steady_cv > 1.0) cold_unreliable = true;
      for (const auto& r : warm_rows) if (r.t.steady_cv > 1.0) warm_unreliable = true;
      if (cold_unreliable || warm_unreliable)
        std::cout << "  ! cv>1.0   OS memory pressure corrupting timing ("
                  << (cold_unreliable ? "cold" : "") << (cold_unreliable && warm_unreliable ? "+" : "")
                  << (warm_unreliable ? "warm" : "") << ") — free RAM and retry\n";
    }
    if (host.zombie_oom_structural)
      std::cout << "  ! oom      No swap on UMA — page allocator will stall at MemAvailable ceiling\n";
    if (uma_platform && host.cudamemgetinfo_unreliable
        && std::abs((double)host.cudamemgetinfo_delta) > 1e9)
      std::cout << "  ! memory   cudaMemGetInfo underreports by "
                << std::fixed << std::setprecision(2)
                << gib((uint64_t)std::abs(host.cudamemgetinfo_delta))
                << " GiB — use allocatable value for workload sizing\n";
    std::cout << "\n";
  }

  std::cout << "  Output    " << (run_dir / "run.json").string() << "\n";
  std::cout << HDR << "\n";

  cupti_teardown();
  return 0;
}
