# CUDA Unified Memory Analyzer

A hardware-aware CUDA diagnostic tool for analyzing Unified Memory migration behavior, residency stability, and transport performance on NVIDIA GPUs.

GB10 field data and confirmed baselines: https://forums.developer.nvidia.com/t/gb10-hardware-baseline-first-direct-measurements-and-findings/367851

All measurements come from live CUDA execution and runtime hardware queries.

---

## Measurements and Diagnostics

**Memory behavior**

* Cold path — page-fault migration latency (child process isolated)
* Warm path — resident access latency after prefetch
* Pressure path — sustained load with CV decay and settling detection
* Unified Memory paradigm detection — `FULL_EXPLICIT` / `FULL_HARDWARE_COHERENT`
* Working-set residency boundary detection

**Migration stability**

* Thrash scoring and state classification
* Migration stability metrics — fault density, symmetry, settling

**Transport**

* Real transport bandwidth — pinned H2D / D2H transfer probe
* PCIe link health — replay counter delta
* NVLink telemetry — presence, link count, error counters, utilization

**System telemetry**

* Thermal and power state — temperature drift, power draw vs TDP, P-state
* VRAM characteristics — total, free, memory type, bus width
* Host free RAM — measured live from the operating system
* Host allocation cap — allocation limit based on available host memory

**Verdict system**

* `HEALTHY` — all subsystems nominal, full ratio ladder executed
* `HEALTHY_LIMITED` — all subsystems nominal, ratios clamped by host memory
* `DEGRADED` — pressure instability detected
* `CRITICAL` — cold child failure, thermal fault, or unsafe memory condition

---

## Architecture Support

Supports NVIDIA GPU architectures from Pascal through Blackwell.

---

## Validation Platform

The analyzer was validated on NVIDIA Pascal (GeForce GTX 1080, Compute Capability 6.1).

Pascal uses GPU page-faulting with driver-managed Unified Memory migration and no hardware CPU–GPU cache coherence, making migration behavior directly observable.

Further exploration of Pascal Unified Memory migration behavior:
[https://github.com/parallelArchitect/pascal-um-benchmark](https://github.com/parallelArchitect/pascal-um-benchmark)

---

## DGX Spark

The analyzer includes detection logic for hardware-coherent Unified Memory platforms such as Grace-Blackwell DGX Spark.

Validation on Spark hardware is pending. Engineers running the analyzer on Spark systems are encouraged to report results.

DGX Spark requires a separate build because the system CPU architecture (Grace) is ARM64.

---

## Build

**Requirements**

* Linux
* CUDA Toolkit 12+
* NVML (`libnvidia-ml`)
* C++17

**Compile**

```bash
nvcc -O2 -std=c++17 -o um_analyzer um_analyzer_v7.cu -lnvidia-ml
```

**Run**

```bash
./um_analyzer
```

Each execution writes a structured JSON report to:

```
runs/<timestamp>_GPU<ID>_<UUID>/run.json
```

---

## Related Work

* [https://github.com/parallelArchitect/pascal-um-benchmark](https://github.com/parallelArchitect/pascal-um-benchmark) — Pascal Unified Memory benchmark
* [https://github.com/parallelArchitect/gpu-pcie-path-validator](https://github.com/parallelArchitect/gpu-pcie-path-validator) — PCIe path validator for NVIDIA GPUs

---

## Author

Joe McLaren (parallelArchitect)
Human-directed GPU engineering with AI assistance.

---

## License

MIT License

Copyright (c) 2026 Joe McLaren

This project is licensed under the MIT License — see the `LICENSE` file for details.



