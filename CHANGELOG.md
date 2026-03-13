# Changelog

All notable changes to cuda-unified-memory-analyzer are documented here.

---

## v8.1 — schema 2.6

### Added

**6-state verdict tree**
- `HEALTHY` — all subsystems nominal, full ratio ladder executed
- `HEALTHY_LIMITED` — core ratios (≤ 1.00x) clamped by host memory, hardware clean
- `MIGRATION_PRESSURE` — hardware clean, paired instability signals: `thrash_score >= 0.15 AND maf >= 1.5`, OR `settle=NO AND cv_mean > 0.15`. MAF alone does not trigger — high MAF with settled pressure is normal operation on discrete PCIe.
- `UM_THRASHING` — hardware clean, all three required: `thrash_score >= 0.45 AND maf >= 2.0 AND settle=NO`. Workload-induced boundary condition, not hardware fault.
- `DEGRADED` — physical subsystem fault: thermal throttle, power cap active, PCIe replay >= 100 events
- `CRITICAL` — cold child failures, memory DANGER/CRITICAL, thermal fault

Previous 4-state tree (`HEALTHY` / `HEALTHY_LIMITED` / `DEGRADED` / `CRITICAL`) replaced. `DEGRADED` now reserved strictly for physical subsystem faults.

**Thrash state `SETTLED_UNSTABLE`**
- New state for runs where pressure settled but the steady region remains noisy
- `SETTLED_UNSTABLE` = `thrash_score` in 0.15–0.45 range AND `settled=true`
- `UNSTABLE` = same score range AND `settled=false`
- Removes the contradiction where `state=UNSTABLE` and `settle=YES` printed together

**Settling fields in JSON**
- `settled` — boolean: true if pressure converged to stable CV
- `settle_class` — `STABLE` / `LATE_UNSTABLE` / `UNSTABLE`
- `settle_ms` — 0.0 when `settled=false` (previously could show non-zero on transient CV dips)

**Direction ratio and trend — platform-aware**
- `direction_ratio = max(H2D, D2H) / min(H2D, D2H)`
- PCIe / NVLink platforms: `BALANCED` / `H2D_DOMINANT` / `D2H_DOMINANT`
- `FULL_HARDWARE_COHERENT` platforms (GB10 / DGX Spark): `BALANCED` / `GPU_OWNERSHIP_DEMAND` / `CPU_RETENTION` — reflects coherence ownership pattern, not physical transfer direction
- Added to JSON: `direction_ratio`, `direction_trend`

**CUPTI migration data availability flag**
- `cupti_migration_data_available` — false when CUPTI initialises but `bytes_htod + bytes_dtoh = 0`
- Addresses known GB10 / SM 12.1 limitation where `BYTES_TRANSFER_HTOD` / `BYTES_TRANSFER_DTOH` may not appear in traces despite CUPTI running
- Display note printed when bytes are zero: `[note: migration bytes not captured — known CUPTI limitation on some platforms]`

**Tail latency fields**
- `steady_max_ms` — worst observed latency in steady-state repeats
- `steady_tail_ratio` — p99 / p50, printed as `tail=Nx` in PASS RESULTS
- Propagated through cold child IPC — no longer zero in cold pass JSON rows

**Universal platform header**
- Source header updated to document all three paradigms: `FULL_EXPLICIT`, `FULL_HARDWARE_COHERENT`, `FULL_SOFTWARE_COHERENT`
- Explicit statement: no hardcoded architecture assumptions

### Fixed

**`HEALTHY_LIMITED` false positive on over-VRAM ratios**
- `HEALTHY_LIMITED` now only fires when core ratios (≤ 1.00x) are skipped
- Previously, over-VRAM ratios (1.25x, 1.50x, 2.00x) skipped due to host cap incorrectly triggered `HEALTHY_LIMITED` on a GTX 1080 running a full 1.00x test

**`stability_meaningful` inconsistency between display and JSON**
- `stability_meaningful` now fires when final pressure CV < 0.05, not only when `cv_history[0] > 0.10` or fault migration detected
- Both display and JSON now use the same criterion — previously the JSON field could show `false` while the display correctly showed `stable=passN`

**`settle_ms` transient dip false positive**
- `settling_time_ms` fallback now requires final CV < 0.05 before using any mid-loop dip
- Previously, a single pass dipping below 0.05 (e.g. `cv=0.376→0.032→0.386`) set `settle_ms` even when pressure never truly converged

**Schema version corrected**
- `schema_version` field in `run.json` now correctly emits `2.6`
- Coherence JSON (`--coherence` output) schema field also corrected to `2.6`

### Schema changes (2.5 → 2.6)

Added to `gpu` block:
```
settled
settle_class
direction_ratio
direction_trend
cupti_migration_data_available
cupti_cpu_page_faults        (was present in 2.5, now correctly gated Volta+)
cupti_throttling_events      (was present in 2.5, now correctly gated Volta+)
```

Added to pass `times_ms` blocks:
```
steady_max_ms
steady_tail_ratio
```

`settle_ms` semantics changed: now 0.0 when `settled=false`. Previously could be non-zero on transient CV dips.

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
