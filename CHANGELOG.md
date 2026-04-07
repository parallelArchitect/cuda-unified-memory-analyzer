# Changelog

All notable changes to cuda-unified-memory-analyzer are documented here.

---

## v8.3.2 — schema 2.8

### Fixed

**NVML initialization failure message — actionable for DGX Spark**
- Previous message: `"NVML failed."` — no diagnostic value
- New message identifies the failure, explains the likely cause on GB10, and provides
  the correct recovery steps:
  - Full cold power cycle (wall disconnect, 60 seconds)
  - Run `spark-gpu-throttle-check` after reboot to verify power state
- Backed by forensic analysis of a real GB10 sosreport (March 2026) confirming
  that a Grace firmware BERT reset can leave the PCIe link degraded to 2.5GT/s x1
  post-reboot, preventing driver initialization entirely. A warm reboot does not
  retrain the PCIe link. Only a cold power cycle forces fresh link negotiation.

### Schema
- No schema changes (2.8 compatible)

---

## v8.3.1 — schema 2.8

### Fixed

**`ceiling_utilization` overflow on memory-pressured systems**
- `ceiling_utilization` could exceed 1.0 (display as `ceil=126%`) on systems
  where committed RAM exceeded `uma_allocatable`
- Root cause: `committed = mem_total - mem_avail` was divided by `mem_avail`
  (FULL_EXPLICIT) — on systems with more than half of RAM in use, committed
  exceeds the allocatable pool and the ratio blows past 100%
- Fix: compute `raw_util` first, cap `ceiling_utilization` at `std::min(1.0, raw_util)`
- Added `overcommit` boolean flag — set true when `raw_util > 1.0`
- Terminal output appends `[overcommit]` when flag is set
- JSON adds `"overcommit": true/false` field after `ceiling_utilization`
- Previously this produced a false `VERDICT CRITICAL` on healthy systems under
  moderate host memory pressure — now correctly reflects actual state

### Schema
- No schema version bump — `overcommit` field added to JSON host block (2.8 compatible)

---


## v8.3 — schema 2.8

### Added

**Per-ratio cold pass CUPTI data**
- `child_gpu_faults`, `child_bytes_htod`, `child_bytes_dtoh` added to `ResultRow`
- Child process emits these via JSON IPC after `cupti_flush()`
- Parent parses from 4096-byte buffer (increased from 512 — root cause of missing per-ratio data)
- `live_line()` prints fault/htod/dtoh under each cold row when faults > 0

**COLD and WARM table column headers**
- `SIZE STATUS p50 p90 p99 max tail cv jump faults htod dtoh`
- Single line per ratio, aligned columns, separator line under headers

**GB10 / DGX Spark platform support**
- `arch_family()` and `arch_detail()`: `maj==12 && min==1` → `"Blackwell GB10"` (SM 12.1)
- Distinct from SM 12.0 (RTX 50xx Blackwell) and SM 10.0 (B200 datacenter)
- `throttle_threshold_from_arch()`: `"Blackwell_GB10"` → 95°C firmware cutoff
- Thermal band check on `FULL_HARDWARE_COHERENT`: `THERMAL_ELEVATED` at 85°C, `THERMAL_THROTTLE` at 95°C using raw temperature, not NVML throttle bits
- NVML throttle bits (`HwSlowdown`, `SwPowerCap`) gated out on UMA platforms — always set by SoC power management, not meaningful for fault detection

**OOM line — hardware-aware**
- Discrete GPU: `burst=Xx maf=X.X → nominal/monitor/elevated` based on thrash state
- UMA: headroom-based verdict, no burst/maf (not meaningful on coherent UMA)
- Structural zombie OOM preserved for UMA no-swap case
- WATCH threshold requires `thrash.thrash_score > 0 || !thrash.settled` — prevents false WATCH on healthy discrete GPU

**`--cupti-debug` flag**
- Dumps every raw CUPTI unified memory activity record to stderr
- Per record: counter kind, value, start/end timestamps, processId
- Zero cost when not set — gated by global flag
- Purpose: verify CUPTI UM counter behavior on GB10 coherent UMA before building kernel correlation

**`LD_LIBRARY_PATH` forwarding in child**
- Previously stripped from popen environment, breaking child CUPTI initialization
- Now forwarded correctly — resolves missing CUPTI data in cold pass children

### Fixed

- `cold_child_failures` correctly increments on popen failure (was silently dropped)
- `zero_end_ts_skipped` counter: GPU_PAGE_FAULT records with `end=0` skipped correctly; HTOD/DTOH records with `end=0` preserved (legitimate on Pascal and other platforms)
- `any_warning` gate: WATCH verdict no longer fires on settled healthy Pascal runs

### Removed

- `check_driver_regression()` function and all call sites — driver version warnings removed entirely. Driver issues are vendor territory; stale warnings damage tool credibility. Driver landscape documented in KB notes only.

### Community patches integrated (from adi-sonusflow fork)

- CUDA 13 prefetch API compatibility wrapper (`#if CUDART_VERSION >= 13000`)
- LPDDR5X memory type detection
- UMA throttle guard (NVML throttle bits gated on coherent UMA)
- `vram_total==0` fallback for NVML memory reporting on GB10
- Prefetch API wrappers for CUDA 13.x
- Ratio ladder correction

### Schema
- Bumped to `2.8`

---

## v8.2 — schema 2.7

### Added

**Fault pressure metrics**
- `fault_pressure_index` — `fault_rate_per_sec * fault_burst_ratio`; compact single-number roll-up for cross-run comparison. Baseline GTX 1080 ~197–205, danger band ~234–274.

**Migration quality metrics**
- `migration_efficiency` — `1 / maf`; 1.0 = ideal, <0.5 = moderate amplification, <0.3 = heavy
- `migration_oscillation_ratio` — `min(H2D,D2H) / max(H2D,D2H)`; 0.0 = one-direction, >0.6 = oscillation, >0.85 = severe CPU↔GPU ping-pong

**Residency decay**
- `residency_half_life_ratio` — first warm ratio where `steady_tail_ratio * (1 + steady_cv)` exceeds 1.5× baseline; `null` if threshold never reached

**Managed Memory Intent Inspector**
- `um_preferred_location` — `cudaMemRangeAttributePreferredLocation` queried in child after GPU work; `n/a (arch)` on Pascal (no automatic hint)
- `um_last_prefetch_location` — `cudaMemRangeAttributeLastPrefetchLocation`; reflects last `cudaMemPrefetchAsync` target
- Both fields populate on Volta+/UMA platforms (DGX Spark) where driver sets location hints automatically

**Unified Memory Headroom Predictor**
- `um_headroom_ratio` — `host_free_gib / host_cap_gib`; >2.0 LARGE, 1.3–2.0 SAFE, 1.0–1.3 LOW, <1.0 RISK
- Displayed in VERDICT block as `headroom` line
- Predicts UM exhaustion before workload launch; directly relevant to DGX Spark OOM freezes

**LLM KV-Cache Pressure Detector**
- `llm_pressure_score` — `fault_pressure_index * (1 - migration_efficiency) / um_headroom_ratio`
- `llm_pressure_level` — LOW (<100) / MODERATE (100–200) / HIGH (>200)
- Captures the LLM failure triangle: paging stress × migration cost ÷ remaining headroom
- Displayed in VERDICT block as `llm_kv` line

**PSI Memory Pressure Snapshot**
- `memory_psi_state` — Linux PSI memory stall classification: LOW / ELEVATED / HIGH / n/a
- `memory_psi_some_avg10_start/end` — % some-stall at run start and end
- `memory_psi_full_avg10_end` — % full-stall at run end
- `memory_psi_some_total_delta_us` — cumulative stall microseconds during run
- Read from `/proc/pressure/memory` when available (Linux 4.20+, PSI-enabled kernel)
- Displayed in VERDICT block as `host_psi` line; distinguishes GPU-side UM instability from host-side reclaim pressure

### Schema
- Bumped to `2.7`; all new fields added to `gpu` block in `run.json`

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
- `cupti_migration_data_available` — false when CUPTI initializes but `bytes_htod + bytes_dtoh = 0`
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
