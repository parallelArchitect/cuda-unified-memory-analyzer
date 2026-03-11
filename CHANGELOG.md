# Changelog

All notable changes to cuda-unified-memory-analyzer are documented here.

---

## v8 — schema 2.5

### Added

**CUPTI Activity API integration**
- `CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER` enabled at startup
- Hardware-level collection of: `BYTES_TRANSFER_HTOD`, `BYTES_TRANSFER_DTOH`, `GPU_PAGE_FAULT`, `THRASHING`
- Uses `CUpti_ActivityUnifiedMemoryCounter2` struct (CUPTI 12 — v1 struct produces incorrect values)
- `cupti_available`, `cupti_records_total`, `cupti_gpu_page_faults`, `cupti_bytes_htod`, `cupti_bytes_dtoh`, `cupti_thrashing_events` added to JSON
- `cupti_thrashing_events` silent on Pascal SM 6.x (hardware limitation), fires on Volta+

**Migration Amplification Factor (MAF)**
- `MAF = (htod + dtoh) / total_pass_bytes`
- total_pass_bytes accounts for cold + warm + pressure passes
- Printed in terminal migration line, stored in JSON

**Bytes Per Fault (BPF) — split into two fields**
- `bpf_htod = htod_bytes / gpu_page_faults` — fault-driven inbound cost
- `bpf_total = (htod + dtoh) / gpu_page_faults` — full round-trip cost
- Gap between the two reveals reverse migration volume
- Architecture-portable: PCIe migration efficiency on discrete GPU, coherence locality on DGX Spark C2C
- Printed in terminal, stored as `bpf_htod_bytes` and `bpf_total_bytes` in JSON

**Settling time**
- Wall-clock ms from first pressure pass start to first stable pass (CV < 0.05)
- Stored as `settle_ms` in JSON, printed when `settling=YES`

**Full pass timing for all pass types**
- Cold: all timing fields now transmitted via IPC (previously only `steady_p50_ms`, `steady_cv`, `prefetch_to_gpu_ms`)
- Warm: full timing instrumentation added inline (`alloc_ms`, `cpu_init_ms`, `prefetch_to_gpu_ms`, `gpu_first_touch_ms`, `prefetch_to_cpu_ms`, `prefetch_back_to_gpu_ms`, `gpu_retouch_ms`, full steady percentiles)
- Eliminates the cold/warm timing zeros that appeared in schema 2.4 JSON

### Fixed

**Stability label logic**
- `stable=passN` no longer printed when `stability_meaningful=false`
- `stability_meaningful = (fault_migration_regime || cv_history[0] > 0.10)`
- Runs where initial CV is below 0.10 with no fault migration now correctly print `stable=no`
- Convergence criterion 2 (per-step delta) now requires CV is not rising above 0.10 — prevents false stable detection on diverging sequences

**Verdict logic**
- `thrash_state=UNSTABLE` now forces `verdict=DEGRADED`
- `thrash_state=SEVERE_UNSTABLE` now forces `verdict=DEGRADED`
- Previously UNSTABLE could produce `HEALTHY_LIMITED` despite active migration instability

**DEGRADED conditions now documented explicitly** — see README and source header

### Schema changes (2.4 → 2.5)

Added to `gpu` block:
```
maf
bpf_htod_bytes
bpf_total_bytes
settle_ms
cupti_available
cupti_records_total
cupti_gpu_page_faults
cupti_bytes_htod
cupti_bytes_dtoh
cupti_thrashing_events
```

---

## v7 — schema 2.4

- P2P Unified Memory coherence test using `cudaMallocManaged` with cross-device pointer access
- Transport layer detection: PCIe / NVLink / C2C / coherent
- `FULL_EXPLICIT` vs `FULL_HARDWARE_COHERENT` paradigm detection at runtime
- NVLink counter telemetry
- Aggregate VERDICT rollup: HEALTHY / DEGRADED / CRITICAL
- False-positive fixes for PINGPONG_SUSPECT detection
- Rebuilt terminal output formatting

---

## v6 and earlier

- Residency window detection
- Knee detection (cold and warm)
- Pressure stability tracking with CV history
- Hardware state block: thermal, power, clocks, throttle reasons
- PCIe bandwidth probe (pinned H2D / D2H)
- NVML integration: driver version, VRAM, PCIe link gen/width
- Cold pass child process isolation
- Schema versioning introduced
