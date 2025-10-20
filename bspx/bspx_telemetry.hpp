// bspx_telemetry.hpp

#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace bspx::tel {

enum class Metric : uint8_t {
  BSP_mi_nodes,
  BSP_lp_nodes,
  BSP_tb_nodes,
  BSP_es_nodes,
  BSP_decision_cost_bits,

  TB_run,
  TB_sparse,
  TB_holes,
  TB_split,
  TB_smallbit,

  JUDY_nodes,
  JUDY_mode_eb,
  JUDY_mode_fb,
  JUDY_mode_b,
  JUDY_mode_efb,
  JUDY_child_e,
  JUDY_child_f,
  JUDY_child_b,
  JUDY_presence_maps,

  ROAR_pick_array,
  ROAR_pick_empty,
  ROAR_pick_runs,
  ROAR_pick_bits,

  XFORM_bits,
  NARROW_Unew,
  RECN_bits,
  LCC_term_bal,
  LCC_term_cl,
  LCC_term_co,
  LRRC_regions,
  LRRC_runs,
  LRRC_len_total,
  LRRC_len_min,
  LRRC_len_max,
  PROBE_hash,
  kCount // sentinel — keep as last
};

static constexpr size_t N_METRICS = static_cast<size_t>(Metric::kCount);

// No explicit size: deduce and check against enum count.
static constexpr const char* metric_labels[] = {
    "BSP_mi_nodes",    "BSP_lp_nodes",    "BSP_tb_nodes",   "BSP_es_nodes",       "BSP_decision_cost_bits",

    "TB_run",          "TB_sparse",       "TB_holes",       "TB_split",           "TB_smallbit",

    "JUDY_nodes",      "JUDY_mode_eb",    "JUDY_mode_fb",   "JUDY_mode_b",        "JUDY_mode_efb",
    "JUDY_child_e",    "JUDY_child_f",    "JUDY_child_b",   "JUDY_presence_maps",

    "ROAR_pick_array", "ROAR_pick_empty", "ROAR_pick_runs", "ROAR_pick_bits",

    "XFORM_bits",      "NARROW_Unew",     "RECN_bits",      "LCC_term_bal",       "LCC_term_cl",
    "LCC_term_co",     "LRRC_regions",    "LRRC_runs",      "LRRC_len_total",     "LRRC_len_min",
    "LRRC_len_max",    "PROBE_hash",
};
static_assert(std::size(metric_labels) == N_METRICS, "metric_labels size mismatch");

struct TelemetrySink {
  std::array<double, N_METRICS> counters_fast{};
  std::unordered_map<std::string, double> counters_slow;

  static constexpr size_t idx(Metric m) noexcept { return static_cast<size_t>(m); }

  void record(Metric m, double value) { counters_fast[(size_t)m] += value; }
  void record(std::string_view key, double value) { counters_slow[std::string(key)] += value; }

  std::string dump() {
    // Merge slow/fast counters
    for (size_t i = 0; i < N_METRICS; ++i) {
      const double v = counters_fast[i];
      counters_slow[metric_labels[i]] += v;
      counters_fast[i] = 0.0;
    }

    // Collect: drop value≅0 counters and sort
    std::vector<std::pair<std::string, double>> items;
    items.reserve(counters_slow.size());
    for (const auto& kv : counters_slow)
      if (std::llround(kv.second * 1000.0) != 0)
        items.emplace_back(kv.first, kv.second);
    std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    // Format: key=val|key=val...
    std::string out;
    out.reserve(items.size() * 8); // approx. (likely under)

    bool first = true;
    for (const auto& [k, v] : items) {
      if (!first)
        out.push_back('|');
      first = false;
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(3) << v;
      out += k + '=' + oss.str();
    }
    return out;
  }

  TelemetrySink& operator+=(const TelemetrySink& R) {
    for (size_t i = 0; i < N_METRICS; ++i) {
      counters_fast[i] += R.counters_fast[i];
    }
    for (const auto& [k, v] : R.counters_slow) {
      if (v != 0.0) { // skip injecting zeros
        counters_slow[k] += v;
      }
    }
    return *this;
  }
};

// =================== Policy-agnostic stats probe ===================

// Boxes are contiguous ranges where every value shares the same top-K bits.
// Visits each box at K=0,4,...,28 and calls fn(i, K, key, box_span)
// Skips empty and full boxes. For K=0, all values belong to one box.
template <class Fn>
inline void for_each_box(const std::vector<uint32_t>& V, Fn&& fn) {
  static constexpr std::array<uint32_t, 8> Ks{0, 4, 8, 12, 16, 20, 24, 28};

  // Per-band state
  std::array<bool, 8> has_box{};          // whether we've started a box
  std::array<uint32_t, 8> cur_key{};      // current box key
  std::array<std::size_t, 8> box_begin{}; // start index in V for current box

  auto finish = [&](std::size_t i, std::size_t end_idx, const std::vector<uint32_t>& VV) {
    if (!has_box[i])
      return;
    const std::size_t begin = box_begin[i];
    if (end_idx > begin) {
      std::span<const uint32_t> box{VV.data() + begin, end_idx - begin};
      fn(i, Ks[i], cur_key[i], box);
    }
  };

  for (std::size_t pos = 0; pos < V.size(); ++pos) {
    const uint32_t x = V[pos];
    for (std::size_t i = 0; i < Ks.size(); ++i) {
      const uint32_t K = Ks[i];
      // Top-K prefix; for K==0, the key must be 0
      const uint32_t key = (K == 0) ? 0u : (x >> (32u - K));

      if (!has_box[i]) {
        has_box[i] = true;
        cur_key[i] = key;
        box_begin[i] = pos;
      } else if (key != cur_key[i]) {
        // Close previous box [box_begin, pos) and open a new one at pos
        finish(i, pos, V);
        cur_key[i] = key;
        box_begin[i] = pos;
      }
    }
  }

  // Flush trailing boxes
  for (std::size_t i = 0; i < Ks.size(); ++i) {
    finish(i, V.size(), V);
  }
}

struct Probes {
  uint64_t boxes = 0;
  double sum_density = 0.0;   // ρ
  double sum_runs = 0.0;      // average run count per box
  double sum_blp = 0.0;       // LP header bits surrogate
  double sum_log2Klp = 0.0;   // log2(Klp)
  double sum_phi_lp = 0.0;    // center distance for LP
  double sum_bmed = 0.0;      // MI header bits surrogate
  double sum_log2Kmed = 0.0;  // log2(Kmed)
  double sum_delta_med = 0.0; // center distance for MI
  double sum_D01 = 0.0;       // next-bit KL to fair coin
};

// Expect U == 2^32; boxes are top-K prefixes, box spans are width 2^(32-K)
[[nodiscard]] inline TelemetrySink stats_probe(const std::vector<uint32_t>& V, uint64_t U) {
  (void)U;
  assert(U == uint64_t(std::numeric_limits<uint32_t>::max()) + 1);

  TelemetrySink tel;

  std::array<Probes, 8> A{}; // per-band aggregates

  // FNV-1a hash
  uint64_t hash = 14695981039346656037ull;
  for (uint32_t x : V) {
    hash ^= x;
    hash *= 1099511628211ull;
  }
  tel.record(Metric::PROBE_hash, hash);

  for_each_box(V, [&](std::size_t /*band_index*/, uint32_t K, uint32_t /*key*/, std::span<const uint32_t> box) {
    const uint64_t n = box.size();
    const uint64_t Ubox = 1ull << (32u - K);
    const uint64_t UL = Ubox >> 1, UR = Ubox >> 1;

    // skip full and empty
    if (!n || Ubox == n)
      return;

    // Count next-bit lefts & runs in one pass.
    const uint32_t shift = 31u - K; // next bit after top-K
    uint64_t nL = 0, runs = 0;
    uint32_t prev = 0;
    bool have_prev = false;
    for (uint32_t x : box) {
      nL += (((x >> shift) & 1u) == 0u);
      if (!have_prev) {
        prev = x;
        have_prev = true;
        runs = 1;
      } else {
        runs += (uint64_t)(x != prev + 1u);
        prev = x;
      }
    }

    // copied from bits_scalar
    auto bits_truncbin = [](uint64_t U) {
      if (U <= 1)
        return 0.0;
      const uint64_t b = 63u - __builtin_clzll(U - 1);
      const uint64_t pow2b = 1ull << b;
      return double(b) + 2.0 * double(U - pow2b) / double(U);
    };

    // LP feasibility and header surrogate
    const uint64_t xmin = (n > UR) ? (n - UR) : 0ull;
    const uint64_t xmax = (n < UL) ? n : UL;
    const uint64_t Klp = (xmax >= xmin) ? (xmax - xmin + 1ull) : 1ull;
    const uint64_t ylp = (nL >= xmin) ? (nL - xmin) : 0ull;
    const double blp = bits_truncbin(Klp);
    const double phi_lp = (Klp > 1) ? std::abs((double)ylp - 0.5 * (double)(Klp - 1)) / (double)(Klp - 1) : 0.0;

    // MI feasibility and header surrogate
    const uint64_t Kmed = Ubox - n + 1ull;
    const uint64_t rank = (uint64_t)(n / 2);
    const uint64_t mask = (32u - K == 32u) ? 0xFFFFFFFFull : ((1ull << (32u - K)) - 1ull);
    const uint64_t med_local = (uint64_t)box[rank] & mask; // local [0..Ubox)
    const uint64_t ymed = (med_local >= rank) ? (med_local - rank) : 0ull;
    const double bmed = bits_truncbin(Kmed);
    const double delta_med = (Kmed > 1) ? std::abs((double)ymed - 0.5 * (double)(Kmed - 1)) / (double)(Kmed - 1) : 0.0;

    auto log2z = [](double x) { return x > 0.0 ? log2(x) : 0.0; };

    // Node imbalance (KL to fair)
    const double pL = (double)nL / (double)n;
    const double pR = 1.0 - pL;
    const double D01 = pL * log2z(2.0 * pL) + pR * log2z(2.0 * pR);

    const int b = K / 4;
    auto& a = A[b];
    a.boxes += 1;
    a.sum_density += (double)n / (double)Ubox;
    a.sum_runs += (double)runs;
    a.sum_blp += blp;
    a.sum_log2Klp += log2((double)Klp);
    a.sum_phi_lp += phi_lp;
    a.sum_bmed += bmed;
    a.sum_log2Kmed += log2((double)Kmed);
    a.sum_delta_med += delta_med;
    a.sum_D01 += D01;
  });

  // Push into the telemetry fields
  for (int i = 0; i < 8; ++i) {
    const auto& a = A[i];
    if (!a.boxes)
      continue;
    std::string is = "[" + std::to_string(i * 4) + "]";
    tel.record("PROBE_boxes" + is, a.boxes);
    tel.record("PROBE_avg_density" + is, a.sum_density / (double)a.boxes);
    tel.record("PROBE_avg_runs" + is, a.sum_runs / (double)a.boxes);
    tel.record("PROBE_lp_bits" + is, a.sum_blp / (double)a.boxes); // mean LP header bits
    tel.record("PROBE_lp_log2K" + is, a.sum_log2Klp / (double)a.boxes);
    tel.record("PROBE_lp_center" + is, a.sum_phi_lp / (double)a.boxes); // 0..1 (edgeyness)
    tel.record("PROBE_mi_bits" + is, a.sum_bmed / (double)a.boxes);     // mean MI header bits
    tel.record("PROBE_mi_log2K" + is, a.sum_log2Kmed / (double)a.boxes);
    tel.record("PROBE_mi_center" + is, a.sum_delta_med / (double)a.boxes);
    tel.record("PROBE_imbalance_kl" + is, a.sum_D01 / (double)a.boxes);
  }

  return tel;
}

} // namespace bspx::tel
