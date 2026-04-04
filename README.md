# CUDA Unified Memory Analyzer

Linux tool for observing CUDA Unified Memory behavior at runtime.

Focus: page fault migration, host↔device data movement (HtoD / DtoH), and memory pressure signals.

---

## What it does

Runs controlled memory access patterns and collects runtime signals using CUDA, CUPTI, and NVML.

Exposes:

- GPU page faults
- Host ↔ Device migration (bytes_htod / bytes_dtoh)
- Fault distribution (steady vs burst)
- Migration overhead (MAF, BPF)
- Pressure and stability signals
- Residency vs migration behavior

---

## Why this exists

Unified Memory simplifies development, but obscures key runtime behaviors:

- when data is actually moving
- how often faults occur
- whether memory is stable or thrashing
- why systems slow down or become unresponsive under load

This tool makes those behaviors observable.

One observed failure mode on unified memory platforms is systems becoming
unresponsive under memory pressure instead of raising a CUDA OOM error.

This tool measures the memory conditions and signal patterns that precede
that state.

---

## How it works

The analyzer runs three passes:

### COLD pass
- Fresh process (no GPU context)
- First-touch page faults (CPU → GPU)
- Measures actual migration path

### WARM pass
- Prefetch to GPU
- Measures steady-state resident access
- Confirms residency stability

### PRESSURE pass
- Sustained access at highest safe ratio
- Detects thrashing, oscillation, and instability under load

Verdict is derived from combined signals, not a single metric.

---

## Key metrics

### Migration Amplification Factor (MAF)

```
MAF = (htod_bytes + dtoh_bytes) / total_pass_bytes
```

- ~1.0 → resident behavior
- 2–4 → moderate migration overhead
- >4 → heavy migration amplification

Note: On PCIe systems, elevated MAF alone does not indicate a failure condition.

High MAF values should be interpreted alongside fault rate, burst behavior, and
residency signals.

---

### Bytes Per Fault (BPF)

```
bpf_htod  = htod_bytes / gpu_page_faults
bpf_total = (htod_bytes + dtoh_bytes) / gpu_page_faults
```

Describes migration efficiency per fault.

---

### Fault pressure

```
fault_pressure_index = fault_rate_per_sec * fault_burst_ratio
```

Used to detect sustained pressure and burst behavior.

---

### Directional behavior

```
direction_ratio = total_htod / total_dtoh
```

Helps distinguish forward migration (CPU → GPU) from eviction / fallback (GPU → CPU).

---

## Output

### Terminal

- Per-pass summaries
- Ratio scaling behavior
- Final verdict

### JSON

```
runs/um_YYYYMMDD_HHMMSS_GPU0_<uuid>/run.json
```

Includes:

- gpu_page_faults
- bytes_htod / bytes_dtoh
- maf
- fault_rate_per_sec
- fault_burst_ratio
- pressure_score
- transport (PCIe / NVLink / UMA)
- platform_caps
- stability indicators

---

## Verdict system

| Verdict | Meaning |
|---|---|
| `HEALTHY` | Stable, resident behavior |
| `HEALTHY_LIMITED` | Stable but memory constrained |
| `MIGRATION_PRESSURE` | Elevated migration under load |
| `UM_THRASHING` | Active instability boundary |
| `DEGRADED` | Hardware-level issue |
| `CRITICAL` | Fatal / unrecoverable state |

---

## Platform model

Detected at runtime — no hardcoding.

| Paradigm | Meaning |
|---|---|
| `FULL_EXPLICIT` | Discrete GPU, PCIe migration |
| `FULL_HARDWARE_COHERENT` | Unified DRAM systems (DGX Spark GB10, Grace Blackwell) |
| `FULL_SOFTWARE_COHERENT` | OS-managed coherence |

Transport awareness:

- PCIe → migration cost dominated by interconnect bandwidth and latency
- NVLink → lower latency and higher bandwidth relative to PCIe
- UMA → shared memory pool; pressure replaces explicit migration

---

## Build

Requirements:

- Linux (x86_64 / aarch64)
- CUDA 12.x or 13.x
- NVML (`libnvidia-ml`)
- CUPTI (`libcupti`)
- C++17

Compile:

```bash
nvcc -O2 -std=c++17 \
  -I/usr/local/cuda/include \
  um_analyzer.cu \
  -o um_analyzer \
  -lcudart -lcupti -lnvidia-ml
```

---

## Run

```bash
./um_analyzer
```

Options:

```
--device N          test specific GPU index
--all-devices       test all GPUs
--list-devices      list available GPUs
--cupti-debug       dump raw CUPTI UM records to stderr
```

`--cupti-debug` writes every CUPTI unified memory activity record to stderr —
counter kind, value, timestamps. Useful for understanding what signals are
present on a given platform, particularly on unified memory architectures
where behavior differs from discrete PCIe systems.

---

## Status

Core functionality is implemented using CUDA, CUPTI, and NVML APIs.

Behavior varies by platform and driver. This tool is actively being exercised
across different systems to understand how Unified Memory signals behave under
real workloads.

Findings are used to investigate failure modes such as:

- unexpected slowdowns
- memory pressure instability
- systems becoming unresponsive instead of reporting CUDA OOM

The goal is to make these behaviors observable and explainable.

---

## Community data

Behavior varies across architectures, drivers, and system configurations.

Results from different systems are used to understand how Unified Memory signals
behave in practice, particularly on newer platforms where migration and memory
semantics differ from discrete PCIe systems.

Observed data is applied to investigate real failure modes (for example, systems
becoming unresponsive instead of reporting CUDA OOM) and to identify underlying
causes.

Logs and results from real workloads help improve interpretation and make these
behaviors more explainable as results are collected across architectures and
configurations.

---

## Validated hardware

| Platform | Paradigm | Status |
|---|---|---|
| Discrete GPU, PCIe | `FULL_EXPLICIT` | Validated — multiple generations |
| DGX Spark GB10, coherent UMA | `FULL_HARDWARE_COHERENT` | Detection implemented, runtime validation in progress |

---

## Companion tool

[spark-gpu-throttle-check](https://github.com/parallelArchitect/spark-gpu-throttle-check)  
GPU power and thermal diagnostics for DGX Spark systems. Helps identify power
delivery limits (such as USB PD constraints) that may reduce clocks under
sustained load.

---

## Acknowledgements

[cuda-unified-memory-analyzer fork](https://github.com/sonusflow/cuda-unified-memory-analyzer)  
Community fork by adi-sonusflow. Patches, CUPTI signal interpretation, and
execution model changes from this fork directly influenced the debugging approach
and runtime signal analysis used in this project.

---

## Author

parallelArchitect  
Human-directed GPU engineering with AI assistance.

---

## Contact

gpu.validation@gmail.com

---

## License

MIT
