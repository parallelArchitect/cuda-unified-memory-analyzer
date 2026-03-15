# CUDA Unified Memory Analyzer
![license MIT](https://img.shields.io/badge/license-MIT-blue)
![CUDA supported](https://img.shields.io/badge/CUDA-supported-green)
![platform Linux](https://img.shields.io/badge/platform-Linux-lightgrey)
![NVML supported](https://img.shields.io/badge/NVML-supported-green)
![CUPTI supported](https://img.shields.io/badge/CUPTI-supported-green)
![Unified Memory](https://img.shields.io/badge/CUDA-Unified%20Memory-orange)
![Architecture Adaptive](https://img.shields.io/badge/architecture-adaptive-green)

A hardware-aware CUDA diagnostic tool for analyzing Unified Memory migration behavior, residency stability, and transport performance on NVIDIA GPUs.

All measurements come from live CUDA execution and runtime hardware queries. No hardcoded architecture assumptions — platform detection, transport labels, verdict logic, and CUPTI counter gating all adapt at runtime.

---

## Use Cases

- Diagnose Unified Memory thrashing on discrete PCIe and coherent UMA systems
- Evaluate migration efficiency on PCIe vs NVLink systems
- Detect host memory pressure before launching LLM workloads
- Validate system stability before training runs
- Investigate migration oscillation between CPU and GPU
- Assess LLM KV-cache memory risk on DGX Spark and similar UMA platforms
- Generate structured JSON diagnostics for automated pipeline health checks

---

## Platform Support

| Paradigm | Hardware | Status |
|---|---|---|
| `FULL_EXPLICIT` | Discrete GPU, PCIe transport (Pascal, Ampere, Ada, Hopper, Blackwell) | Validated on GTX 1080 (SM 6.1) |
| `FULL_HARDWARE_COHERENT` | Coherent UMA, C2C interconnect (DGX Spark / GB10, Grace Blackwell, Grace Hopper) | Detection implemented with architecture-adaptive logic. Hardware validation pending. |
| `FULL_SOFTWARE_COHERENT` | HMM / software-managed coherence | Detection implemented with architecture-adaptive logic. Hardware validation pending. |

---

## Measurements and Diagnostics

**Memory behavior**
* Cold path — page-fault migration latency (child process isolated)
* Warm path — resident access latency after prefetch
* Pressure path — repeated oversubscription passes used to observe latency variance convergence, settling behavior, and migration stability
* Unified Memory paradigm detection at runtime
* Working-set residency boundary detection

**Migration stability**
* Thrash scoring and state classification (`STABLE` / `SETTLED_UNSTABLE` / `UNSTABLE` / `SEVERE_UNSTABLE`)
* Migration Amplification Factor (MAF) — total migration volume relative to working set
* Bytes Per Fault — `bpf_htod` and `bpf_total` — per-fault migration cost
* Settling time — wall-clock ms from first pressure pass to first stable pass
* Fault density, symmetry, settling detection

**CUPTI hardware instrumentation**
* GPU page fault counts — direct from CUDA driver
* H2D and D2H migration byte totals — hardware measured
* CPU page fault counter — Volta+ / coherent platforms
* Thrashing and throttling event counters — Volta+
* Architecture-portable — same API call on all platforms, behavior resolves at runtime

**Transport**
* Real transport bandwidth — pinned H2D / D2H transfer probe
* PCIe link health — replay counter delta
* NVLink telemetry — presence, link count, error counters, utilization

**System telemetry**
* Thermal and power state — temperature drift, power draw vs TDP, P-state
* VRAM characteristics — total, free, memory type, bus width
* Host free RAM — measured live from the operating system
* Host allocation cap — based on available host memory, coherent-platform corrected

**Verdict system**
* `HEALTHY` — all subsystems nominal, full ratio ladder executed
* `HEALTHY_LIMITED` — core ratios (≤ 1.00x) clamped by host memory, hardware clean
* `MIGRATION_PRESSURE` — hardware clean, paired instability: `thrash_score >= 0.15 AND maf >= 1.5`, OR `settle=NO AND cv_mean > 0.15`
* `UM_THRASHING` — hardware clean, workload at migration boundary: `thrash_score >= 0.45 AND maf >= 2.0 AND settle=NO`
* `DEGRADED` — physical subsystem fault: thermal throttle, power cap, PCIe replay elevated
* `CRITICAL` — cold child failure, thermal fault, or unsafe memory condition

> **Note:** Verdict thresholds (`thrash_score`, `maf`, `cv_mean`) were calibrated on Pascal SM 6.1 (GTX 1080) with PCIe Gen3 x16. These values are not universal — NVLink systems, Volta+, and coherent UMA platforms (DGX Spark) will exhibit different baseline MAF and fault rate ranges. Cross-architecture threshold refinement is ongoing.

---

## Key Metrics

### Migration Amplification Factor (MAF)
```
MAF = (htod_bytes + dtoh_bytes) / total_pass_bytes
```
~1.0 clean residency. 2–4 moderate overhead. >4 severe amplification.

High MAF with settled pressure and low thrash_score is normal operation on discrete PCIe — MAF alone does not indicate a problem.

### Bytes Per Fault (BPF)
```
bpf_htod  = htod_bytes / gpu_page_faults
bpf_total = (htod_bytes + dtoh_bytes) / gpu_page_faults
```
**Pascal / PCIe:** migration efficiency — larger = efficient bulk copy, small = thrash
**DGX Spark C2C:** coherence locality — larger = ownership churn across coherent fabric
The gap between `bpf_htod` and `bpf_total` reveals reverse migration volume.

### Thrash score
```
thrash_score = cv_instability × fault_density × settling_factor
settling_factor = 0.4 if pressure converged, 1.0 if not
```

| State | Score | Description |
|---|---|---|
| `STABLE` | < 0.15 | Clean, settled |
| `SETTLED_UNSTABLE` | 0.15–0.45 | Settled but noisy steady region |
| `UNSTABLE` | 0.15–0.45 | Did not settle |
| `SEVERE_UNSTABLE` | > 0.45 | Never settled, active instability |

### Direction ratio
```
direction_ratio = max(H2D, D2H) / min(H2D, D2H)
```
≤ 1.50 = `BALANCED`. Above threshold:
* **PCIe / NVLink:** `H2D_DOMINANT` (normal fault-driven inbound) or `D2H_DOMINANT` (reverse migration)
* **Coherent C2C (GB10):** `GPU_OWNERSHIP_DEMAND` or `CPU_RETENTION` — reflects coherence ownership pattern, not physical transfer direction

### Fault pressure metrics *(v8.2)*
```
fault_pressure_index = fault_rate_per_sec * fault_burst_ratio
```
Single-number roll-up for cross-run comparison. Baseline GTX 1080 ~197–205. Danger band ~234–274 indicating elevated Unified Memory fault burst pressure.

### Migration oscillation ratio *(v8.2)*
```
oscillation_ratio = min(H2D, D2H) / max(H2D, D2H)
```
0.0 = one-direction migration. >0.6 = oscillation. >0.85 = severe CPU↔GPU ping-pong.

### LLM KV-cache pressure detector *(v8.2)*
```
llm_pressure_score = fault_pressure_index * (1 - migration_efficiency) / um_headroom_ratio
```
Captures paging stress × migration cost ÷ remaining headroom. `LOW` < 100, `MODERATE` 100–200, `HIGH` > 200. Predicts system instability before launching large context models on DGX Spark.

### Unified Memory headroom *(v8.2)*
```
um_headroom_ratio = host_free_gib / host_cap_gib
```
Predicts UM pool exhaustion. `SAFE` ≥ 1.3. `LOW` 1.0–1.3. `RISK` < 1.0.

---

## Architecture Support

Supports NVIDIA GPU architectures from Pascal through Blackwell with runtime platform detection. Validated on Pascal (GTX 1080, SM 6.1). DGX Spark / GB10 (SM 12.1) support included with platform-aware detection — validation on Spark hardware pending.

Initial development and baseline calibration were performed on a GTX 1080 (Pascal SM 6.1), the primary hardware used during development. Pascal uses explicit Unified Memory migration over PCIe without hardware coherence, which makes migration faults, residency boundaries, and paging pressure easier to observe during diagnostics. The analyzer itself is architecture-adaptive and designed for newer platforms as well, including NVLink systems and coherent UMA platforms such as DGX Spark. Results from those systems help expand the cross-architecture Unified Memory diagnostic baseline.

---

## Build

**Requirements**
* Linux
* CUDA Toolkit 12+
* NVML (`libnvidia-ml`)
* CUPTI (`libcupti`) — ships with CUDA Toolkit
* C++17

**Compile**
```bash
nvcc -O2 -std=c++17 -o um_analyzer um_analyzer_v8_2.cu -lcupti -lnvidia-ml
```

**Run**
```bash
./um_analyzer
```

**DGX Spark / Grace (ARM64)**
```bash
nvcc -O2 -std=c++17 -o um_analyzer um_analyzer_v8_2.cu -lcupti -lnvidia-ml -arch=sm_120
```

---

## Command Line Options

```bash
./um_analyzer                  # run on default GPU
./um_analyzer --device 0       # run on specific GPU
./um_analyzer --all-devices    # run on all GPUs sequentially
./um_analyzer --list-devices   # list available GPUs
```

Each execution writes a structured JSON report to:
```
runs/<timestamp>_GPU<ID>_<UUID>/run.json
```

---

## JSON Output Schema

Current schema version: **2.7**

**Example output (Pascal reference system)**

Key fields in `gpu` block:
```json
{
  "thrash_score": 0.06,
  "thrash_state": "STABLE",
  "maf": 3.07,
  "migration_efficiency": 0.326,
  "migration_oscillation_ratio": 0.53,
  "bpf_htod_bytes": 49700000,
  "bpf_total_bytes": 87100000,
  "settled": true,
  "settle_class": "STABLE",
  "settle_ms": 10496.0,
  "cupti_available": true,
  "cupti_migration_data_available": true,
  "cupti_gpu_page_faults": 2764,
  "cupti_bytes_htod": 176412426240,
  "cupti_bytes_dtoh": 120176771072,
  "cupti_thrashing_events": 0,
  "fault_rate_per_sec": 74.6,
  "fault_max_window_rate_per_sec": 202.9,
  "fault_burst_ratio": 2.71,
  "fault_pressure_index": 201.9,
  "residency_half_life_ratio": 0.75,
  "direction_ratio": 1.39,
  "direction_trend": "BALANCED",
  "um_preferred_location": "GPU",
  "um_last_prefetch_location": "GPU",
  "um_headroom_ratio": 1.70,
  "llm_pressure_score": 70.2,
  "llm_pressure_level": "LOW",
  "memory_psi_state": "LOW"
}
```

`cupti_migration_data_available` is false when CUPTI initializes successfully but migration byte counters return zero — a known limitation on some platforms including GB10 (SM 12.1).

`cupti_thrashing_events` is always 0 on Pascal SM 6.x — hardware limitation, not a bug.

`um_preferred_location` / `um_last_prefetch_location` show `n/a` on Pascal (SM 6.x) — no automatic preferred location hint on discrete FULL_EXPLICIT platforms. Populated on Volta+/UMA systems.

`memory_psi_*` fields present only when `/proc/pressure/memory` is available (Linux 4.20+, PSI-enabled kernel).

---

## DGX Spark

The analyzer includes detection and adaptation logic for hardware-coherent Unified Memory platforms. This logic has been implemented and unit-tested without GB10 hardware using `test_coherent.cpp` (61/61 passing). End-to-end validation on real DGX Spark hardware is pending.

Implemented and unit-tested:
* `direction_trend` uses `GPU_OWNERSHIP_DEMAND` / `CPU_RETENTION` labels instead of H2D/D2H
* `cudaMemGetInfo` underreporting detected and corrected for allocation ceiling
* Structural OOM condition (`zombie_oom_structural`) detected pre-run
* CUPTI `cpu_faults` and `throttling` counters gated on Volta+ (SM 7.0+, includes GB10 SM 12.1)
* Pass labels adapt to C2C semantics — no PCIe migration on coherent fabric

Known CUPTI limitation on GB10: `BYTES_TRANSFER_HTOD` / `BYTES_TRANSFER_DTOH` may not appear in traces. `cupti_migration_data_available` flags this condition in JSON output.

Engineers running the analyzer on DGX Spark / GB10 systems are encouraged to share results to expand the cross-architecture Unified Memory diagnostic baseline.

---

## Related Work

- [pascal-um-benchmark](https://github.com/parallelArchitect/pascal-um-benchmark) — Pascal Unified Memory benchmark suite
- [gpu-pcie-path-validator](https://github.com/parallelArchitect/gpu-pcie-path-validator) — PCIe path validator for NVIDIA GPUs

---

## Field Usage

The analyzer has already been used in live debugging discussions around CUDA Unified Memory behavior and telemetry limitations.

Examples:

- NVIDIA Developer Forum — incorrect VRAM metrics with `cudaMallocManaged`
  https://forums.developer.nvidia.com/t/incorrect-vram-usage-metrics-using-unified-cudamallocmanaged/362491

---

## Author

Joe McLaren (parallelArchitect)
Human-directed GPU engineering with AI assistance.

---

## Contact

gpu.validation@gmail.com

---

## License

MIT License
Copyright (c) 2026 Joe McLaren
