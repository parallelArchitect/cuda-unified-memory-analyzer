// test_coherent.cpp — DGX Spark / FULL_HARDWARE_COHERENT platform test harness
// No GPU, no CUDA, no CUPTI required.
//
// Build:  g++ -O2 -std=c++17 -o test_coherent test_coherent.cpp
// Run:    ./test_coherent
//
// Covers:
//   [A] compute_uma_allocatable — coherent vs explicit paths
//   [B] zombie_oom_structural detection
//   [C] ceiling_utilization and prerun_pressure_verdict
//   [D] cudamemgetinfo_unreliable flag
//   [E] direction_trend coherent labels (GPU_OWNERSHIP_DEMAND / CPU_RETENTION)
//   [F] direction_trend PCIe labels (H2D_DOMINANT / D2H_DOMINANT) — regression
//   [G] pass labels for TransportLayer::Coherent
//   [H] volta_plus CUPTI counter gating (SM >= 7)
//   [I] cupti_migration_data_available flag
//   [J] BALANCED threshold — same on all platforms

#include <iostream>
#include <string>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <iomanip>
#include <sstream>
#include <algorithm>

// ── Replicated structs from um_analyzer_v8_1.cu ──────────────────────────────

struct HostInfo {
  uint64_t mem_total             = 0;
  uint64_t mem_avail             = 0;
  uint64_t swap_total            = 0;
  uint64_t swap_free             = 0;
  uint64_t cached                = 0;
  uint64_t buffers               = 0;
  uint64_t hugetlb_total         = 0;
  uint64_t hugetlb_free          = 0;
  uint64_t hugetlb_size          = 0;   // page size in KB
  uint64_t uma_allocatable       = 0;
  uint64_t cache_recoverable_bytes = 0;
  double   cache_frac_of_total   = 0.0;
  double   ceiling_utilization   = 0.0;
  bool     buffer_cache_pressure = false;
  bool     swap_disabled         = false;
  bool     zombie_oom_structural = false;
  bool     cudamemgetinfo_unreliable = false;
  std::string prerun_pressure_verdict = "UNKNOWN";
};

// ── Replicated logic from um_analyzer_v8_1.cu ────────────────────────────────

static void compute_uma_allocatable(HostInfo& h, const std::string& um_paradigm) {
  if (um_paradigm == "FULL_HARDWARE_COHERENT") {
    h.cudamemgetinfo_unreliable = true;
    if (h.hugetlb_total > 0) {
      h.uma_allocatable = (uint64_t)(h.hugetlb_free * h.hugetlb_size) * 1024ull;
    } else {
      h.uma_allocatable = h.mem_avail + h.swap_free;
    }
  } else {
    h.cudamemgetinfo_unreliable = false;
    h.uma_allocatable = h.mem_avail;
  }

  h.cache_recoverable_bytes = h.cached + h.buffers;
  if (h.mem_total > 0) {
    h.cache_frac_of_total = (double)h.cache_recoverable_bytes / (double)h.mem_total;
    h.buffer_cache_pressure = (h.cache_frac_of_total > 0.20);
  }

  if (h.uma_allocatable > 0) {
    uint64_t committed = (h.mem_total > h.mem_avail) ? (h.mem_total - h.mem_avail) : 0;
    h.ceiling_utilization = (double)committed / (double)h.uma_allocatable;
    if      (h.ceiling_utilization < 0.50) h.prerun_pressure_verdict = "CLEAR";
    else if (h.ceiling_utilization < 0.70) h.prerun_pressure_verdict = "ELEVATED";
    else if (h.ceiling_utilization < 0.85) h.prerun_pressure_verdict = "CRITICAL";
    else                                   h.prerun_pressure_verdict = "DANGER";
  }

  h.swap_disabled = (h.swap_total == 0);
  h.zombie_oom_structural = (um_paradigm == "FULL_HARDWARE_COHERENT")
                           && h.swap_disabled
                           && h.buffer_cache_pressure;
}

// direction trend logic
struct DirectionResult {
  double ratio;
  std::string trend;
};

static DirectionResult compute_direction(double htod_gb, double dtoh_gb,
                                         const std::string& paradigm) {
  double hi = std::max(htod_gb, dtoh_gb);
  double lo = std::min(htod_gb, dtoh_gb);
  double dr = (hi > 0.0001) ? (hi / std::max(lo, 0.0001)) : 1.0;
  std::string trend;
  bool is_coherent = (paradigm == "FULL_HARDWARE_COHERENT");
  if (dr <= 1.50) {
    trend = "BALANCED";
  } else if (is_coherent) {
    trend = (htod_gb > dtoh_gb) ? "GPU_OWNERSHIP_DEMAND" : "CPU_RETENTION";
  } else {
    trend = (htod_gb > dtoh_gb) ? "H2D_DOMINANT" : "D2H_DOMINANT";
  }
  return {dr, trend};
}

// pass labels
struct PassLabels {
  std::string cold, warm, cold_note, warm_note;
};

enum class TransportLayer { PCIe, NVLink, Coherent, Unknown };

static PassLabels make_pass_labels(TransportLayer layer) {
  PassLabels p;
  switch (layer) {
    case TransportLayer::PCIe:
      p.cold      = "COLD (fault path)";
      p.warm      = "WARM (resident path)";
      p.cold_note = "fault -> PCIe DMA -> GPU VRAM";
      p.warm_note = "pages GPU-resident at 2MB granularity";
      break;
    case TransportLayer::NVLink:
      p.cold      = "COLD (NVLink fault path)";
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

// volta_plus gating
static bool is_volta_plus(int sm_major) { return sm_major >= 7; }

// cupti_migration_data_available
static bool cupti_migration_data_available(bool cupti_ok,
                                           uint64_t bytes_htod,
                                           uint64_t bytes_dtoh) {
  return cupti_ok && (bytes_htod + bytes_dtoh) > 0;
}

// ── Test framework ─────────────────────────────────────────────────────────

static int tests_run = 0, tests_passed = 0;

void check(bool cond, const char* desc) {
  tests_run++;
  if (cond) { tests_passed++; std::cout << "  PASS  " << desc << "\n"; }
  else                        std::cout << "  FAIL  " << desc << "\n";
}

void check_near(double a, double b, double tol, const char* desc) {
  check(std::fabs(a - b) <= tol, desc);
}

// ── [A] compute_uma_allocatable — coherent vs explicit ─────────────────────

void test_A_uma_allocatable() {
  std::cout << "\n[A] compute_uma_allocatable\n";

  const uint64_t GiB = 1024ull * 1024ull * 1024ull;

  // Coherent — no HugeTLB: allocatable = mem_avail + swap_free
  {
    HostInfo h;
    h.mem_total  = 128 * GiB;
    h.mem_avail  = 100 * GiB;
    h.swap_total = 8 * GiB;
    h.swap_free  = 8 * GiB;
    compute_uma_allocatable(h, "FULL_HARDWARE_COHERENT");
    check(h.uma_allocatable == (100 + 8) * GiB,  "A1 coherent: allocatable = mem_avail + swap_free");
    check(h.cudamemgetinfo_unreliable == true,    "A2 coherent: cudamemgetinfo_unreliable=true");
  }

  // Coherent — HugeTLB active: allocatable = hugetlb_free * hugetlb_size * 1024
  {
    HostInfo h;
    h.mem_total    = 128 * GiB;
    h.mem_avail    = 100 * GiB;
    h.swap_total   = 0;
    h.swap_free    = 0;
    h.hugetlb_total = 32768;   // pages
    h.hugetlb_free  = 32768;
    h.hugetlb_size  = 2048;    // 2MB pages in KB
    compute_uma_allocatable(h, "FULL_HARDWARE_COHERENT");
    uint64_t expected = (uint64_t)(32768 * 2048) * 1024ull;
    check(h.uma_allocatable == expected,          "A3 coherent+hugetlb: allocatable from hugetlb");
    check(h.cudamemgetinfo_unreliable == true,    "A4 coherent+hugetlb: unreliable=true");
  }

  // Explicit — allocatable = mem_avail only
  {
    HostInfo h;
    h.mem_total  = 32 * GiB;
    h.mem_avail  = 14 * GiB;
    h.swap_total = 8 * GiB;
    h.swap_free  = 8 * GiB;
    compute_uma_allocatable(h, "FULL_EXPLICIT");
    check(h.uma_allocatable == 14 * GiB,         "A5 explicit: allocatable = mem_avail only");
    check(h.cudamemgetinfo_unreliable == false,   "A6 explicit: cudamemgetinfo_unreliable=false");
  }

  // Software coherent — same as explicit
  {
    HostInfo h;
    h.mem_total  = 64 * GiB;
    h.mem_avail  = 50 * GiB;
    h.swap_total = 0;
    h.swap_free  = 0;
    compute_uma_allocatable(h, "FULL_SOFTWARE_COHERENT");
    check(h.uma_allocatable == 50 * GiB,         "A7 sw-coherent: allocatable = mem_avail only");
    check(h.cudamemgetinfo_unreliable == false,   "A8 sw-coherent: unreliable=false");
  }
}

// ── [B] zombie_oom_structural ──────────────────────────────────────────────

void test_B_zombie_oom() {
  std::cout << "\n[B] zombie_oom_structural\n";

  const uint64_t GiB = 1024ull * 1024ull * 1024ull;

  // Fires: coherent + no swap + buffer cache pressure > 20%
  {
    HostInfo h;
    h.mem_total  = 128 * GiB;
    h.mem_avail  = 80 * GiB;
    h.swap_total = 0;
    h.swap_free  = 0;
    h.cached     = 30 * GiB;   // 30/128 = 23% > 20%
    h.buffers    = 0;
    compute_uma_allocatable(h, "FULL_HARDWARE_COHERENT");
    check(h.zombie_oom_structural == true,        "B1 coherent+no-swap+cache-pressure=true");
  }

  // Does NOT fire: coherent + swap present
  {
    HostInfo h;
    h.mem_total  = 128 * GiB;
    h.mem_avail  = 80 * GiB;
    h.swap_total = 8 * GiB;
    h.swap_free  = 8 * GiB;
    h.cached     = 30 * GiB;
    compute_uma_allocatable(h, "FULL_HARDWARE_COHERENT");
    check(h.zombie_oom_structural == false,       "B2 coherent+swap=false");
  }

  // Does NOT fire: coherent + no swap + cache pressure low
  {
    HostInfo h;
    h.mem_total  = 128 * GiB;
    h.mem_avail  = 120 * GiB;
    h.swap_total = 0;
    h.cached     = 5 * GiB;    // 5/128 = 4% < 20%
    compute_uma_allocatable(h, "FULL_HARDWARE_COHERENT");
    check(h.zombie_oom_structural == false,       "B3 coherent+no-swap+low-cache=false");
  }

  // Does NOT fire: explicit platform even with worst conditions
  {
    HostInfo h;
    h.mem_total  = 32 * GiB;
    h.mem_avail  = 10 * GiB;
    h.swap_total = 0;
    h.cached     = 10 * GiB;   // 31% — would trigger if coherent
    compute_uma_allocatable(h, "FULL_EXPLICIT");
    check(h.zombie_oom_structural == false,       "B4 explicit platform never fires");
  }
}

// ── [C] ceiling_utilization and prerun_pressure_verdict ───────────────────

void test_C_ceiling() {
  std::cout << "\n[C] ceiling_utilization + prerun_pressure_verdict\n";

  const uint64_t GiB = 1024ull * 1024ull * 1024ull;

  // CLEAR: committed < 50% of allocatable
  {
    HostInfo h;
    h.mem_total = 128 * GiB;
    h.mem_avail = 100 * GiB;   // committed = 28 GiB / 100 GiB = 28%
    h.swap_total = 0; h.swap_free = 0;
    compute_uma_allocatable(h, "FULL_HARDWARE_COHERENT");
    check(h.prerun_pressure_verdict == "CLEAR",   "C1 28% committed = CLEAR");
  }

  // ELEVATED: 50–70%
  {
    HostInfo h;
    h.mem_total = 128 * GiB;
    h.mem_avail = 68 * GiB;    // committed = 60/128 = 47% but allocatable=68 -> 60/68=88% wait
    // Let's be explicit: allocatable=68, committed=128-68=60, 60/68=88% -> DANGER
    // Redo: mem_total=100, mem_avail=40 -> committed=60, allocatable=40 -> 150% DANGER
    // For ELEVATED: committed/allocatable in 0.50-0.70
    // mem_total=100, mem_avail=45 -> committed=55, allocatable=45, 55/45=122% nope
    // coherent: allocatable = mem_avail + swap_free
    // mem_total=100, mem_avail=60, swap_free=40 -> allocatable=100, committed=40 -> 40% CLEAR
    // mem_total=100, mem_avail=40, swap_free=40 -> allocatable=80, committed=60 -> 75% CRITICAL
    // mem_total=100, mem_avail=50, swap_free=40 -> allocatable=90, committed=50 -> 55% ELEVATED
    h.mem_total  = 100 * GiB;
    h.mem_avail  = 50 * GiB;
    h.swap_total = 40 * GiB;
    h.swap_free  = 40 * GiB;
    compute_uma_allocatable(h, "FULL_HARDWARE_COHERENT");
    check(h.prerun_pressure_verdict == "ELEVATED","C2 55% committed = ELEVATED");
  }

  // CRITICAL: 70–85%
  {
    HostInfo h;
    h.mem_total  = 100 * GiB;
    h.mem_avail  = 40 * GiB;
    h.swap_total = 40 * GiB;
    h.swap_free  = 40 * GiB;   // allocatable=80, committed=60 -> 75% CRITICAL
    compute_uma_allocatable(h, "FULL_HARDWARE_COHERENT");
    check(h.prerun_pressure_verdict == "CRITICAL","C3 75% committed = CRITICAL");
  }

  // DANGER: >= 85%
  {
    HostInfo h;
    h.mem_total  = 100 * GiB;
    h.mem_avail  = 10 * GiB;
    h.swap_total = 0;
    h.swap_free  = 0;          // allocatable=10, committed=90 -> 900% DANGER
    compute_uma_allocatable(h, "FULL_HARDWARE_COHERENT");
    check(h.prerun_pressure_verdict == "DANGER",  "C4 900% committed = DANGER");
  }
}

// ── [D] cudamemgetinfo_unreliable ─────────────────────────────────────────

void test_D_cmg_unreliable() {
  std::cout << "\n[D] cudamemgetinfo_unreliable\n";

  const uint64_t GiB = 1024ull * 1024ull * 1024ull;

  auto make = [&](const std::string& paradigm) {
    HostInfo h;
    h.mem_total = 128 * GiB; h.mem_avail = 100 * GiB;
    compute_uma_allocatable(h, paradigm);
    return h;
  };

  check(make("FULL_HARDWARE_COHERENT").cudamemgetinfo_unreliable == true,  "D1 FULL_HARDWARE_COHERENT=unreliable");
  check(make("FULL_EXPLICIT").cudamemgetinfo_unreliable          == false, "D2 FULL_EXPLICIT=reliable");
  check(make("FULL_SOFTWARE_COHERENT").cudamemgetinfo_unreliable == false, "D3 FULL_SOFTWARE_COHERENT=reliable");
}

// ── [E] direction_trend coherent labels ───────────────────────────────────

void test_E_direction_coherent() {
  std::cout << "\n[E] direction_trend — FULL_HARDWARE_COHERENT\n";

  // GPU pulling pages from CPU — htod > dtoh, ratio > 1.50
  auto r1 = compute_direction(100.0, 34.5, "FULL_HARDWARE_COHERENT");
  check(r1.trend == "GPU_OWNERSHIP_DEMAND",   "E1 htod>dtoh, coherent = GPU_OWNERSHIP_DEMAND");
  check(r1.ratio > 1.50,                     "E2 ratio > 1.50");

  // CPU retaining pages GPU needs — dtoh > htod, ratio > 1.50
  auto r2 = compute_direction(34.5, 100.0, "FULL_HARDWARE_COHERENT");
  check(r2.trend == "CPU_RETENTION",          "E3 dtoh>htod, coherent = CPU_RETENTION");
  check(r2.ratio > 1.50,                     "E4 ratio > 1.50");

  // Balanced on coherent — same label as PCIe
  auto r3 = compute_direction(64.5, 60.0, "FULL_HARDWARE_COHERENT");
  check(r3.trend == "BALANCED",              "E5 balanced coherent = BALANCED");
  check(r3.ratio <= 1.50,                    "E6 balanced ratio <= 1.50");

  // Exactly at threshold — BALANCED (<=)
  auto r4 = compute_direction(1.50, 1.00, "FULL_HARDWARE_COHERENT");
  check(r4.trend == "BALANCED",              "E7 exactly 1.50 coherent = BALANCED");

  // Just over threshold
  auto r5 = compute_direction(1.51, 1.00, "FULL_HARDWARE_COHERENT");
  check(r5.trend == "GPU_OWNERSHIP_DEMAND",  "E8 1.51 coherent htod dominant = GPU_OWNERSHIP_DEMAND");
}

// ── [F] direction_trend PCIe labels — regression ──────────────────────────

void test_F_direction_pcie() {
  std::cout << "\n[F] direction_trend — FULL_EXPLICIT (regression)\n";

  auto r1 = compute_direction(100.0, 34.5, "FULL_EXPLICIT");
  check(r1.trend == "H2D_DOMINANT",          "F1 explicit htod>dtoh = H2D_DOMINANT");

  auto r2 = compute_direction(34.5, 100.0, "FULL_EXPLICIT");
  check(r2.trend == "D2H_DOMINANT",          "F2 explicit dtoh>htod = D2H_DOMINANT");

  auto r3 = compute_direction(64.5, 60.0, "FULL_EXPLICIT");
  check(r3.trend == "BALANCED",              "F3 explicit balanced = BALANCED");

  // Coherent labels must NOT appear on PCIe
  check(r1.trend != "GPU_OWNERSHIP_DEMAND",  "F4 PCIe never GPU_OWNERSHIP_DEMAND");
  check(r2.trend != "CPU_RETENTION",         "F5 PCIe never CPU_RETENTION");
}

// ── [G] pass labels for TransportLayer::Coherent ──────────────────────────

void test_G_pass_labels() {
  std::cout << "\n[G] pass labels — TransportLayer::Coherent\n";

  auto p = make_pass_labels(TransportLayer::Coherent);
  check(p.cold_note.find("no PCIe") != std::string::npos,     "G1 cold_note: no PCIe");
  check(p.cold_note.find("NVLink") != std::string::npos,      "G2 cold_note: no NVLink");
  check(p.cold_note.find("TLB miss") != std::string::npos,    "G3 cold_note: TLB miss cost");
  check(p.warm_note.find("ATS") != std::string::npos,         "G4 warm_note: ATS");
  check(p.warm_note.find("cached") != std::string::npos,      "G5 warm_note: cached");
  check(p.cold.find("TLB cold") != std::string::npos,         "G6 cold label: TLB cold");
  check(p.warm.find("ATS cached") != std::string::npos,       "G7 warm label: ATS cached");

  // PCIe labels must NOT appear on coherent path
  auto pcie = make_pass_labels(TransportLayer::PCIe);
  check(pcie.cold_note.find("PCIe DMA") != std::string::npos, "G8 PCIe cold_note: PCIe DMA");
  check(pcie.cold_note.find("TLB") == std::string::npos,      "G9 PCIe cold_note: no TLB");
}

// ── [H] volta_plus CUPTI counter gating ───────────────────────────────────

void test_H_volta_plus() {
  std::cout << "\n[H] volta_plus CUPTI gating\n";

  // Pascal SM 6.1 — cpu_faults and throttling NOT active
  check(is_volta_plus(6) == false,   "H1 SM 6 (Pascal) = not volta_plus");
  check(is_volta_plus(6) == false,   "H2 SM 6.1 major=6 = not volta_plus");

  // Volta SM 7.0 — active
  check(is_volta_plus(7) == true,    "H3 SM 7 (Volta) = volta_plus");

  // Turing SM 7.5
  check(is_volta_plus(7) == true,    "H4 SM 7 (Turing) = volta_plus");

  // Ampere SM 8.0
  check(is_volta_plus(8) == true,    "H5 SM 8 (Ampere) = volta_plus");

  // Hopper SM 9.0
  check(is_volta_plus(9) == true,    "H6 SM 9 (Hopper) = volta_plus");

  // GB10 SM 12.1 — major=12
  check(is_volta_plus(12) == true,   "H7 SM 12 (GB10/Blackwell) = volta_plus");
}

// ── [I] cupti_migration_data_available ────────────────────────────────────

void test_I_migration_data() {
  std::cout << "\n[I] cupti_migration_data_available\n";

  // CUPTI ok, bytes present — available
  check(cupti_migration_data_available(true, 1000, 500) == true,  "I1 cupti_ok+bytes=true");

  // CUPTI ok, both zero — GB10 known limitation
  check(cupti_migration_data_available(true, 0, 0) == false,      "I2 cupti_ok+zero_bytes=false");

  // CUPTI not ok — not available regardless
  check(cupti_migration_data_available(false, 1000, 500) == false, "I3 cupti_fail=false");
  check(cupti_migration_data_available(false, 0, 0) == false,      "I4 cupti_fail+zero=false");

  // One side zero, other has data — available
  check(cupti_migration_data_available(true, 1000, 0) == true,    "I5 htod_only=true");
  check(cupti_migration_data_available(true, 0, 500) == true,     "I6 dtoh_only=true");
}

// ── [J] BALANCED threshold consistent across platforms ────────────────────

void test_J_balanced_universal() {
  std::cout << "\n[J] BALANCED threshold — universal across platforms\n";

  // 1.50 = BALANCED on both
  auto c = compute_direction(1.50, 1.00, "FULL_HARDWARE_COHERENT");
  auto e = compute_direction(1.50, 1.00, "FULL_EXPLICIT");
  check(c.trend == "BALANCED",   "J1 1.50 coherent = BALANCED");
  check(e.trend == "BALANCED",   "J2 1.50 explicit = BALANCED");

  // 1.51 = dominant on both (different labels)
  auto c2 = compute_direction(1.51, 1.00, "FULL_HARDWARE_COHERENT");
  auto e2 = compute_direction(1.51, 1.00, "FULL_EXPLICIT");
  check(c2.trend != "BALANCED",  "J3 1.51 coherent != BALANCED");
  check(e2.trend != "BALANCED",  "J4 1.51 explicit != BALANCED");

  // Labels differ above threshold
  check(c2.trend != e2.trend,    "J5 above threshold: coherent and explicit labels differ");
  check(c2.trend == "GPU_OWNERSHIP_DEMAND", "J6 coherent dominant = GPU_OWNERSHIP_DEMAND");
  check(e2.trend == "H2D_DOMINANT",         "J7 explicit dominant = H2D_DOMINANT");
}

// ── Main ───────────────────────────────────────────────────────────────────

int main() {
  std::cout << "um_analyzer v8.1 — DGX Spark / FULL_HARDWARE_COHERENT test harness\n";
  std::cout << "====================================================================\n";

  test_A_uma_allocatable();
  test_B_zombie_oom();
  test_C_ceiling();
  test_D_cmg_unreliable();
  test_E_direction_coherent();
  test_F_direction_pcie();
  test_G_pass_labels();
  test_H_volta_plus();
  test_I_migration_data();
  test_J_balanced_universal();

  std::cout << "\n====================================================================\n";
  std::cout << "Results: " << tests_passed << "/" << tests_run << " passed\n";

  return (tests_passed == tests_run) ? 0 : 1;
}
