// bspx_policy.hpp
#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "bspx_telemetry.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define bspx_ALWAYS_INLINE inline __attribute__((always_inline))
#define bspx_HOT inline __attribute__((hot))
#else
#define bspx_ALWAYS_INLINE inline
#define bspx_HOT inline
#endif

namespace bspx {

// ======================= Common types =======================

struct Dataset {
  std::string name;                              // display name
  std::shared_ptr<std::vector<uint64_t>> values; // sorted & unique (global coords unless transformed)
  uint64_t U = (uint64_t)UINT32_MAX + 1ull;      // universe size
};

struct TransformOutput {
  std::vector<Dataset> children; // zero or more child datasets (mapped to local coords if needed)
  double cost_bits = 0.0;        // overhead bits to store auxiliary structure/tags/headers
  bspx::tel::TelemetrySink tel;  // transform-local telemetry sink
};
using Transform = std::function<TransformOutput(const Dataset& in)>;

struct PolicyResult {
  double size_bits;
  bspx::tel::TelemetrySink tel;
};
using Policy = std::function<PolicyResult(const std::vector<uint64_t>&, uint64_t)>;

enum class IntCodec : uint8_t { Fixed, Uniform, TruncatedBinary };
static IntCodec g_intCodec = IntCodec::TruncatedBinary;

// ======================= Math / bit-cost helpers =======================

[[nodiscard]] inline uint64_t ceil_log2_u64(uint64_t x) {
  if (x <= 1)
    return 0;
  return 64u - __builtin_clzll(x - 1u);
}
[[nodiscard]] inline uint64_t floor_log2_u64(uint64_t x) {
  if (x == 0)
    return 0;
  return 63u - __builtin_clzll(x);
}
[[nodiscard]] inline double log2_binom(uint64_t n, uint64_t k) {
  if (k > n)
    return -INFINITY;
  k = std::min(k, n - k);
  long double lnC =
      lgammal((long double)n + 1.0L) - lgammal((long double)k + 1.0L) - lgammal((long double)(n - k) + 1.0L);
  return (double)(lnC / std::log(2.0L));
}

// Expected bits for a uniformly chosen symbol y in [0, U)
// (y is not used, but supplied because thinking about what the value should be creates clarity.)
[[nodiscard]] inline double bits_scalar(uint64_t U, uint64_t /*y*/) {
  if (U <= 1ull)
    return 0.0;
  switch (g_intCodec) {
  case IntCodec::Fixed: return (double)ceil_log2_u64(U);
  case IntCodec::Uniform: return std::log2((double)U);
  case IntCodec::TruncatedBinary: {
    const uint64_t b = floor_log2_u64(U); // 0..63
    const uint64_t pow2b = 1ull << b;
    const double invU = 1.0 / (double)U;
    // E[L] = b + 2*(U - 2^b)/U
    return (double)b + 2.0 * (double)(U - pow2b) * invU;
  }
  }
  return 0.0; // unreachable
}

// Variant for large-U using log2(U)
[[nodiscard]] inline double bits_scalar_log2U(double log2U) {
  if (!(log2U > 0.0))
    return 0.0;
  switch (g_intCodec) {
  case IntCodec::Fixed: return std::ceil(log2U);
  case IntCodec::Uniform: return log2U;
  case IntCodec::TruncatedBinary: {
    const double b = std::floor(log2U);
    const double frac = log2U - b; // in [0,1)
    return b + 2.0 - std::pow(2.0, 1.0 - frac);
  }
  }
  return 0.0; // unreachable
}

[[nodiscard]] inline double bits_enumerative_subset(uint64_t universe_size, uint64_t subset_count) {
  const double log2U = log2_binom(universe_size, subset_count);
  return bits_scalar_log2U(log2U);
}

// ======================= Pointer windows & DatasetView =======================
//
// DatasetView represents a half-open span [lo, hi) over a sorted unique vector,
// along with pointers to the subrange of present values. It provides canonical
// names for per-span capacity U(), present count n(), and holes m().

[[nodiscard]] inline std::pair<const uint64_t*, const uint64_t*> window_ptr64(const std::vector<uint64_t>& V,
                                                                              uint64_t lo, uint64_t hi) {
  const uint64_t* b = V.data();
  const uint64_t* e = b + V.size();
  const uint64_t* it_lo = std::lower_bound(b, e, (uint64_t)lo);
  const uint64_t* it_hi = std::lower_bound(it_lo, e, (uint64_t)hi);
  if (lo >= hi)
    it_hi = it_lo; // empty
  return {it_lo, it_hi};
}

[[nodiscard]] inline const uint64_t* lower_bound_u64(const uint64_t* lo, const uint64_t* hi, uint64_t x) {
  return std::lower_bound(lo, hi, (uint64_t)x);
}

struct DatasetView {
  const uint64_t* lo_ptr = nullptr;
  const uint64_t* hi_ptr = nullptr;
  uint64_t lo = 0, hi = 0; // half-open [lo, hi)

  // Derived
  uint64_t U() const { return (hi > lo) ? (hi - lo) : 0; }   // span width
  uint64_t n() const { return (uint64_t)(hi_ptr - lo_ptr); } // #present
  uint64_t m() const { return U() - n(); }                   // #holes

  // Convenience
  uint64_t mid() const { return lo + (U() >> 1); }
  std::pair<DatasetView, DatasetView> split_mid() const { return split(mid()); }

  // Build from full dataset (clamped to [0, U))
  static DatasetView whole(const std::vector<uint64_t>& V, uint64_t U) {
    auto [p0, p1] = window_ptr64(V, 0ull, U);
    return DatasetView{p0, p1, 0ull, U};
  }

  // Subspan with pointer narrowing
  DatasetView sub(uint64_t new_lo, uint64_t new_hi) const {
    if (new_lo >= new_hi)
      return DatasetView{hi_ptr, hi_ptr, new_lo, new_hi};
    const uint64_t L = std::max(lo, new_lo);
    const uint64_t H = std::min(hi, new_hi);
    if (H <= L)
      return DatasetView{hi_ptr, hi_ptr, L, H};
    const uint64_t* pL = lower_bound_u64(lo_ptr, hi_ptr, L);
    const uint64_t* pH = lower_bound_u64(pL, hi_ptr, H);
    return DatasetView{pL, pH, L, H};
  }

  // Split at pivot: [lo, pivot) and [pivot, hi)
  std::pair<DatasetView, DatasetView> split(uint64_t pivot) const {
    if (pivot <= lo)
      return {DatasetView{lo_ptr, lo_ptr, lo, lo}, DatasetView{lo_ptr, hi_ptr, lo, hi}};
    if (pivot >= hi)
      return {DatasetView{lo_ptr, hi_ptr, lo, hi}, DatasetView{hi_ptr, hi_ptr, hi, hi}};
    const uint64_t* midp = lower_bound_u64(lo_ptr, hi_ptr, pivot);
    return {DatasetView{lo_ptr, midp, lo, pivot}, DatasetView{midp, hi_ptr, pivot, hi}};
  }
};

// k-th empty (gap) in [view.lo, view.hi), k in [0, view.m())
static inline uint64_t kth_empty_in_span(uint64_t k, const DatasetView& view) {
  uint64_t cur = view.lo;
  for (const uint64_t* p = view.lo_ptr; p != view.hi_ptr; ++p) {
    const uint64_t v = *p;
    if (v > cur) {
      const uint64_t gap = v - cur;
      if (k < gap)
        return cur + k;
      k -= gap;
    }
    if (v + 1 > cur)
      cur = v + 1;
  }
  return cur + k;
}

// ======================= Transforms =======================

// t_slice(low, high)
//   Semantic: Partition values by their top-of-span chunk at bit boundary “high”,
//   and within each chunk project values onto the bit slice [low, high).
//   Output children are in local coordinates [0..2^(high-low)).
//
//   For each original v:
//     chunk = v >> high
//     local = (v >> low) & ((1<<(high-low))-1)
//   Children are deduplicated (input already sorted/unique).
inline Transform t_slice(int low, int high) {
  if (!(0 <= low && low < high && high <= 32))
    throw std::invalid_argument("t_slice: require 0 <= low < high <= 32");
  const int width = high - low;
  const uint64_t mask = (width >= 32) ? 0xFFFFFFFFull : ((1ull << width) - 1ull);
  return [low, high, mask, width](const Dataset& in) -> TransformOutput {
    TransformOutput out;
    const DatasetView root = in.values ? DatasetView::whole(*in.values, in.U) : DatasetView{};

    std::map<uint64_t, std::vector<uint64_t>> groups; // chunk -> local values
    for (const uint64_t* p = root.lo_ptr; p && p != root.hi_ptr; ++p) {
      const uint64_t v = *p;
      const uint64_t chunk = v >> high; // aligned chunk id (size 2^high)
      const uint64_t local = (low == 0) ? (v & mask) : ((v >> low) & mask);
      auto& vec = groups[chunk];
      if (vec.empty() || vec.back() != local)
        vec.push_back(local);
    }
    const uint64_t U2 = 1ull << width;
    out.children.reserve(groups.size());

    for (const auto& [chunk_id, vec] : groups) {
      auto ptr = std::make_shared<std::vector<uint64_t>>(vec);
      out.children.push_back(Dataset{in.name + "#c" + std::to_string(chunk_id), ptr, U2});
    }
    return out;
  };
}

// t_simple(which="UN", decide)
//
//   Semantic: Apply tiny headers/actions to a dataset and pass it through.
//   Flags:
//     'U' → narrow universe to next power-of-two above max value (header bits + telemetry)
//     'N' → record |set| (header bits + telemetry)
//   The optional decide(Dataset) hook can add constant cost_bits and/or drop the dataset.
//
//   Note: This transform modifies Dataset::U (when 'U' set), but does not change values.
//
struct TransformSimpleDecision {
  double cost_bits = 0.0;
  bool drop = false;
};

inline Transform t_simple(const char* which = "UN",
                          std::function<TransformSimpleDecision(const Dataset&)> decide = {}) {
  auto has_flag = [](const char* s, char ch) { return s && std::strchr(s, ch) != nullptr; };

  return [which, decide, has_flag](const Dataset& in) -> TransformOutput {
    TransformOutput o;
    Dataset cur = in;

    const bool trivial = (!cur.values || cur.values->empty() || cur.U == 0ull);
    if (!trivial) {
      // Universe narrowing header + telemetry
      if (has_flag(which, 'U')) {
        const uint64_t last = cur.values->back();
        const uint64_t Ueff = 1ull << ceil_log2_u64(last + 1ull);
        double cost_bits = bits_scalar(ceil_log2_u64(in.U + 1ull), Ueff);
        o.cost_bits += cost_bits;
        cur.U = Ueff;
        o.tel.record(bspx::tel::Metric::XFORM_bits, cost_bits);
        o.tel.record(bspx::tel::Metric::NARROW_Unew, (double)Ueff);
        o.tel.record(bspx::tel::Metric::NARROW_Unew, (double)Ueff);
      }

      // Record-N header + telemetry
      if (has_flag(which, 'N')) {
        const double rec_bits = bits_scalar(cur.U, (uint64_t)cur.values->size());
        o.cost_bits += rec_bits;
        o.tel.record(bspx::tel::Metric::XFORM_bits, rec_bits);
        o.tel.record(bspx::tel::Metric::RECN_bits, rec_bits);
        o.tel.record(bspx::tel::Metric::RECN_bits, rec_bits);
      }
    }

    if (decide) {
      TransformSimpleDecision d = decide(cur);
      o.cost_bits += d.cost_bits;
      o.tel.record(bspx::tel::Metric::XFORM_bits, d.cost_bits);
      if (d.drop)
        return o; // dropped (exit before children.push_bback)
    }

    o.children.push_back(cur);
    return o;
  };
}

// ======================= Left-Cluster-Count (LCC) =======================
//
// t_left_cluster_count(balanced_thresh, recn)
//
//   Semantic: Recursively split the span by midpoint. A node becomes a terminal
//   when its span width Uspan <= balanced_thresh, or when n is extreme relative
//   to Uspan (co/cluster), computed with the log2 rule below. For every
//   non-terminal, write the left-subtree terminal count sLeft (encoded as y in
//   [0..s-2], alphabet size K=s-1). At each terminal, optionally emit a compact
//   description of n via ClusterRecordNMethod.
//
//   Terminals classification used for telemetry:
//     • balanced: Uspan <= balanced_thresh
//     • cluster:  n > Uspan - log2(Uspan)
//     • co-clust: n < log2(Uspan)
//
enum class ClusterRecordNMethod : uint8_t { None, Plain, Narrow };

inline Transform t_left_cluster_count(uint64_t balanced_thresh = 1,
                                      ClusterRecordNMethod recn = ClusterRecordNMethod::Narrow) {
  auto is_terminal = [balanced_thresh](uint64_t Uspan, uint64_t n) -> bool {
    if (Uspan == 0)
      return true;
    if (Uspan <= balanced_thresh)
      return true;
    const double dS = (double)Uspan, lgS = std::log2(dS);
    return ((double)n > (dS - lgS)) || ((double)n < lgS);
  };

  return [is_terminal, balanced_thresh, recn](const Dataset& in) -> TransformOutput {
    TransformOutput out;
    if (!in.values || in.values->empty() || in.U == 0ull) {
      out.children.push_back(in);
      return out;
    }

    DatasetView root = DatasetView::whole(*in.values, in.U);
    if (root.n() == 0) {
      out.children.push_back(Dataset{in.name, std::make_shared<std::vector<uint64_t>>(), 0ull});
      return out;
    }

    struct Node {
      DatasetView view;
    };

    std::vector<Dataset> kids;
    kids.reserve(64);

    double tree_bits = 0.0;
    double recn_bits = 0.0;
    uint64_t term_bal = 0, term_cl = 0, term_co = 0;

    // Depth-first over [lo,hi); writes sLeft at interior nodes; emits terminals as children.
    auto dfs = [&](auto&& self, const Node& nd, const Dataset& src, std::vector<Dataset>& out_children,
                   double& tree_acc) -> uint64_t {
      const DatasetView& view = nd.view;
      const uint64_t Uspan = view.U();
      const uint64_t n = view.n();
      if (Uspan == 0)
        return 0;

      if (is_terminal(Uspan, n)) {
        auto vec = std::make_shared<std::vector<uint64_t>>();
        vec->reserve(n);
        for (const uint64_t* p = view.lo_ptr; p != view.hi_ptr; ++p)
          vec->push_back(*p - view.lo); // localize to [0,Uspan)
        out_children.push_back(Dataset{src.name, vec, Uspan});

        bool is_bal = (Uspan <= balanced_thresh), is_co = false, is_cl = false;
        if (!is_bal) {
          const double dS = (double)Uspan, lgS = std::log2(dS);
          if ((double)n > (dS - lgS))
            is_cl = true;
          else if ((double)n < lgS)
            is_co = true;
          else
            is_bal = true;
        }
        term_bal += is_bal;
        term_cl += is_cl;
        term_co += is_co;

        if (recn != ClusterRecordNMethod::None) {
          double bits_here = 0.0;
          if (recn == ClusterRecordNMethod::Plain) {
            bits_here = bits_scalar(Uspan, n);
          } else {
            const uint64_t L = floor_log2_u64(Uspan);
            if (L == 0ull || is_bal) {
              bits_here = bits_scalar(Uspan, n);
            } else {
              const uint64_t K = 2ull * L;
              uint64_t y = 0ull;
              if (is_co) {
                y = std::min<uint64_t>(n, L - 1ull);
              } else {
                const uint64_t dn = (Uspan > n) ? (Uspan - n) : 0ull;
                y = L + std::min<uint64_t>(dn, L - 1ull);
              }
              bits_here = bits_scalar(K, y);
            }
          }
          recn_bits += bits_here;
        }
        return 1;
      }

      auto [L, R] = view.split_mid();
      const uint64_t sL = self(self, Node{L}, src, out_children, tree_acc);
      const uint64_t sR = self(self, Node{R}, src, out_children, tree_acc);
      const uint64_t s = sL + sR;

      if (s >= 2) {
        // sLeft ∈ {1..s-1} → encode y=sLeft-1 in alphabet K=s-1
        tree_acc += bits_scalar(s - 1ull, sL ? (sL - 1ull) : 0ull);
      }
      return s;
    };

    const uint64_t s_root = dfs(dfs, Node{root}, in, kids, tree_bits);

    // Header for number of terminals
    const uint64_t Smax_raw = (balanced_thresh ? (in.U + balanced_thresh - 1ull) / balanced_thresh : in.U);
    const uint64_t Smax = std::max<uint64_t>(1ull, Smax_raw);
    const uint64_t y_hdr = (s_root ? (s_root - 1ull) : 0ull);
    const double hdr_bits = bits_scalar(Smax, y_hdr);

    out.children = std::move(kids);
    out.cost_bits = hdr_bits + ((s_root >= 2ull) ? tree_bits : 0.0) + recn_bits;

    out.tel.record(bspx::tel::Metric::XFORM_bits, (double)out.cost_bits);
    out.tel.record(bspx::tel::Metric::LCC_term_bal, (double)term_bal);
    out.tel.record(bspx::tel::Metric::LCC_term_cl, (double)term_cl);
    out.tel.record(bspx::tel::Metric::LCC_term_co, (double)term_co);
    out.tel.record(bspx::tel::Metric::RECN_bits, recn_bits);
    return out;
  };
}

// ======================= Left-Run-Region-Count (LRRC) =======================
//
// t_left_run_count(min_span)
//
//   Semantic: Recursively partition the span by midpoint until leaves reach
//   min_span. At each internal node, decide whether to “mark” the node as a
//   region (one container) or pass counts down to children. In a marked region,
//   the child pass collapses to 1 and the region is later rewritten as runs:
//
//     • pairs mode:   [a0,b0), [a1,b1), …  (emit starts and ends)
//     • lengths mode: a0, ℓ0, a1, ℓ1, …    (emit starts and lengths, with local alphabets)
//
//   The transform returns one rewritten child dataset; telemetry reports region
//   and run statistics.
//
template <bool UseStartsAndLengths = false>
inline Transform t_left_run_count(uint64_t min_span = 256) {
  auto is_pow2 = [](uint64_t x) { return x && !(x & (x - 1)); };
  if (!is_pow2(min_span))
    throw std::invalid_argument("t_left_run_count: min_span must be power of two");

  struct Stats {
    uint64_t n = 0, runs = 0, rle = 0; // present, #runs, “compressed” run count at node
    bool marked = false, first = false, last = false;
  };

  auto should_mark = [](uint64_t /*Uspan*/, uint64_t n, uint64_t runs, uint32_t d) -> bool {
    if (n < 4 || runs == 0)
      return false;
    if ((int64_t)n - 2 * (int64_t)runs - (int64_t)d <= 0)
      return false;
    constexpr uint64_t avg_vals_per_run_thresh = 64;
    return (n >= avg_vals_per_run_thresh * runs);
  };

  return [min_span, should_mark](const Dataset& in) -> TransformOutput {
    TransformOutput out;
    if (!in.values || in.values->empty() || in.U == 0ull) {
      out.children.push_back(in);
      return out;
    }

    const auto& V = *in.values;
    DatasetView root = DatasetView::whole(V, in.U);
    if (root.n() == 0) {
      out.children.push_back(Dataset{in.name, std::make_shared<std::vector<uint64_t>>(), 0ull});
      return out;
    }

    // Cover span: power-of-two multiple of min_span that encloses U
    const uint64_t cells = (in.U + min_span - 1ull) / min_span;
    const uint64_t Uroot = (1ull << ceil_log2_u64(std::max<uint64_t>(1, cells))) * min_span;
    const DatasetView cover{root.lo_ptr, root.hi_ptr, 0ull, Uroot};

    auto cap_cells = [&](uint64_t a, uint64_t b) -> uint64_t {
      if (a >= b || a >= in.U)
        return 0ull;
      const uint64_t hi = std::min<uint64_t>(b, in.U);
      if (hi <= a)
        return 0ull;
      return ((hi - a) + min_span - 1ull) / min_span;
    };

    // Pass 1: compute local run counts at leaves; aggregate upward; decide marks.
    std::function<Stats(DatasetView, uint32_t)> stats = [&](DatasetView view, uint32_t d) -> Stats {
      const uint64_t Uspan = view.U(), n = view.n();
      if (Uspan == 0 || n == 0)
        return {};

      if (Uspan <= min_span) {
        Stats s;
        s.n = n;
        s.first = (*view.lo_ptr == view.lo);
        s.last = (*(view.hi_ptr - 1) == view.hi - 1);
        uint64_t rr = 0;
        if (n) {
          rr = 1;
          for (const uint64_t* p = view.lo_ptr + 1; p != view.hi_ptr; ++p)
            if (*p != *(p - 1) + 1)
              ++rr;
        }
        s.runs = rr;
        s.marked = should_mark(Uspan, s.n, s.runs, d);
        s.rle = s.marked ? 1ull : 0ull;
        return s;
      }

      auto [L, R] = view.split_mid();
      Stats A = stats(L, d + 1);
      Stats B = stats(R, d + 1);
      if (A.n == 0 && B.n == 0)
        return {};

      Stats s;
      s.n = A.n + B.n;
      s.runs = A.runs + B.runs - ((A.last && B.first) ? 1ull : 0ull);
      s.first = A.n ? A.first : false;
      s.last = B.n ? B.last : false;
      s.marked = should_mark(view.U(), s.n, s.runs, d);
      s.rle = s.marked ? 1ull : (A.rle + B.rle);
      return s;
    };

    const Stats rootS = stats(cover, 0);
    const uint64_t s_root = rootS.rle;

    // Header: number of top-level regions
    const uint64_t Smax = (in.U + min_span - 1ull) / min_span;
    double bits = bits_scalar(std::max<uint64_t>(1, Smax), s_root);
    (void)s_root;

    if (s_root == 0ull) {
      out.children.push_back(in);
      out.cost_bits = bits;

      out.tel.record(bspx::tel::Metric::LRRC_regions, (double)s_root);
      return out;
    }

    // Pass 2: materialize marked regions; emit left counts where not marked.
    std::vector<std::pair<uint64_t, uint64_t>> regions;
    regions.reserve((size_t)s_root);

    std::function<uint64_t(DatasetView, uint32_t)> emit = [&](DatasetView view, uint32_t d) -> uint64_t {
      const uint64_t Uspan = view.U(), n = view.n();
      if (Uspan == 0 || n == 0)
        return 0ull;

      if (Uspan <= min_span) {
        Stats s = stats(view, d);
        if (s.rle == 1ull)
          regions.emplace_back(view.lo, view.hi);
        return s.rle;
      }

      auto [L, R] = view.split_mid();
      Stats SL = stats(L, d + 1);
      Stats SR = stats(R, d + 1);

      Stats P;
      P.n = SL.n + SR.n;
      P.runs = SL.runs + SR.runs - ((SL.n && SR.n && SL.last && SR.first) ? 1ull : 0ull);
      P.marked = should_mark(Uspan, P.n, P.runs, d);
      P.rle = P.marked ? 1ull : (SL.rle + SR.rle);

      if (P.rle == 0ull)
        return 0ull;
      if (P.marked) {
        regions.emplace_back(view.lo, view.hi);
        return 1ull;
      }

      const uint64_t capL = cap_cells(view.lo, view.mid());
      const uint64_t capR = cap_cells(view.mid(), view.hi);
      const uint64_t sLmin = (P.rle > capR) ? (P.rle - capR) : 0ull;
      const uint64_t sLmax = std::min<uint64_t>(P.rle, capL);
      const uint64_t sLeft = SL.rle; // actual
      const uint64_t K = (sLmax >= sLmin) ? (sLmax - sLmin + 1ull) : 1ull;
      const uint64_t y = (sLeft >= sLmin) ? (sLeft - sLmin) : 0ull;
      bits += bits_scalar(K, y);

      if (sLeft)
        emit(L, d + 1);
      const uint64_t sRight = P.rle - sLeft;
      if (sRight)
        emit(R, d + 1);
      return P.rle;
    };

    emit(cover, 0);

    // Pass 3: rewrite values inside collected regions
    auto outVals = std::make_shared<std::vector<uint64_t>>();
    outVals->reserve(V.size());
    auto push_unique = [&](uint64_t x) {
      if (outVals->empty() || outVals->back() != x)
        outVals->push_back(x);
    };

    double side_bits = 0.0;
    const uint64_t* cur = root.lo_ptr;

    uint64_t runs_total = 0;
    uint64_t run_len_total = 0, run_len_min = (uint64_t)UINT64_MAX, run_len_max = 0;

    for (const auto& [B, E] : regions) {
      auto [it_lo, it_hi] = window_ptr64(V, B, E);

      for (; cur != it_lo; ++cur)
        push_unique(*cur); // copy-through

      if (it_lo != it_hi) {
        const uint64_t* it = it_lo;
        uint64_t a = *it, prev = a;
        ++it;

        for (; it != it_hi; ++it) {
          const uint64_t v = *it;
          if (v != prev + 1ull) {
            const uint64_t b = prev + 1ull;
            if constexpr (UseStartsAndLengths) {
              push_unique(a);
              const uint64_t feasible = v - a; // a_next - a
              const uint64_t len = b - a;      // ℓ_i
              side_bits += bits_scalar(feasible, len ? (len - 1ull) : 0ull);
            } else {
              push_unique(a);
              push_unique(b);
            }
            const uint64_t len = b - a;
            runs_total += 1;
            run_len_total += len;
            run_len_min = std::min(run_len_min, len);
            run_len_max = std::max(run_len_max, len);
            a = v;
          }
          prev = v;
        }
        const uint64_t b = (prev + 1ull <= E) ? (prev + 1ull) : E;
        if constexpr (UseStartsAndLengths) {
          push_unique(a);
          const uint64_t feasible = E - a; // a_m - a, with a_m = E
          const uint64_t len = b - a;
          side_bits += bits_scalar(feasible, len ? (len - 1ull) : 0ull);
        } else {
          push_unique(a);
          push_unique(b);
        }
        const uint64_t len = b - a;
        runs_total += 1;
        run_len_total += len;
        run_len_min = std::min(run_len_min, len);
        run_len_max = std::max(run_len_max, len);
      }
      cur = it_hi;
    }
    for (; cur != root.hi_ptr; ++cur)
      push_unique(*cur); // tail

    Dataset kid = in;
    kid.values = outVals;
    if constexpr (!UseStartsAndLengths)
      kid.U = in.U + 1ull;

    out.children.clear();
    out.children.push_back(kid);
    out.cost_bits = bits + side_bits;

    out.tel.record(bspx::tel::Metric::XFORM_bits, (double)out.cost_bits);
    out.tel.record(bspx::tel::Metric::LRRC_regions, (double)s_root);
    out.tel.record(bspx::tel::Metric::LRRC_runs, (double)runs_total);
    out.tel.record(bspx::tel::Metric::LRRC_len_total, (double)run_len_total);
    out.tel.record(bspx::tel::Metric::LRRC_len_min, (double)(runs_total ? run_len_min : 0));
    out.tel.record(bspx::tel::Metric::LRRC_len_max, (double)run_len_max);
    return out;
  };
}

inline Transform t_left_run_count_pairs(uint64_t min_span = 256) {
  return t_left_run_count</*UseStartsAndLengths=*/false>(min_span);
}
inline Transform t_left_run_count_lengths(uint64_t min_span = 256) {
  return t_left_run_count</*UseStartsAndLengths=*/true>(min_span);
}

// ======================= Policies =======================
//
// The “two-function” structure below is used for every policy that recurses:
//   p_*           — public, stable signature (creates DatasetView, telemetry)
//   p_*_recurse<> — templated core (all hot logic; suitable for inlining)
//
// The recursion core accepts functors as template parameters so the compiler
// can inline/specialize policy decisions and node tags without virtual calls
// or erased lambdas.
//

// ---------- BSP-TREE (multi-policy driver over LeftPopcount / MedianIndex / MedianEmptyIndex / EarlyStop)
//
// Node semantics:
//   • LeftPopcount: split at midpoint; encode how many present values go left,
//                   with alphabet constrained by left/right capacities.
//   • MedianIndex:  encode the median value by feasible position; recurse on
//                   strict left / strict right subspans.
//   • MedianEmptyIndex: encode the median *empty* similarly; split around it.
//   • EarlyStop:    terminate and return accumulated cost.
//
enum class BspSubPolicy : uint8_t { LeftPopcount, MedianIndex, MedianEmptyIndex, EarlyStop };

struct BspDecision {
  BspSubPolicy policy;
  double cost_bits; // e.g., explicit tag bits
  uint64_t state;   // feed-forward parent→children (opaque to the driver)
};

// Legacy selector typedef (preserved)
using PolicySelectLambda = BspDecision (*)(uint64_t* it_lo, uint64_t* it_hi, uint64_t lo, uint64_t hi, uint64_t state);

// Telemetry glue: legacy hooks are deprecated; use TelemetrySink only.

template <class Select>
bspx_HOT double p_bsp_tree_recurse(const DatasetView& view, const Select& select, uint64_t state, uint64_t& nodes,
                                   bspx::tel::TelemetrySink& tel) {
  const uint64_t n = view.n();
  const uint64_t U = view.U();
  if (n == 0ull || U == 0ull)
    return 0.0;

  // Policy choice at this node (inlined via Select template param)
  BspDecision dec =
      select(const_cast<uint64_t*>(view.lo_ptr), const_cast<uint64_t*>(view.hi_ptr), view.lo, view.hi, state);
  double bits = dec.cost_bits;
  const uint64_t next_state = dec.state;

  tel.record(bspx::tel::Metric::BSP_decision_cost_bits, dec.cost_bits);

  switch (dec.policy) {
  case BspSubPolicy::EarlyStop: {
    tel.record(bspx::tel::Metric::BSP_es_nodes, 1.0);
    return bits;
  }
  case BspSubPolicy::LeftPopcount: {
    nodes += 1ull;
    tel.record(bspx::tel::Metric::BSP_lp_nodes, 1.0);
    auto [L, R] = view.split_mid();
    const uint64_t Luni = L.U(), Runi = R.U();
    if (Luni == 0 || Runi == 0)
      return bits; // degenerate

    const uint64_t left_cnt = L.n();

    const uint64_t xmin = (uint64_t)n > Runi ? ((uint64_t)n - Runi) : 0ull;
    const uint64_t xmax = std::min<uint64_t>(n, Luni);
    const uint64_t K = (xmax >= xmin) ? (xmax - xmin + 1ull) : 1ull;
    const uint64_t y = (left_cnt >= xmin) ? (left_cnt - xmin) : 0ull;

    bits += bits_scalar(K, y);
    if (left_cnt)
      bits += p_bsp_tree_recurse(L, select, next_state, nodes, tel);
    if (n - left_cnt)
      bits += p_bsp_tree_recurse(R, select, next_state, nodes, tel);
    return bits;
  }
  case BspSubPolicy::MedianIndex: {
    nodes += 1ull;
    tel.record(bspx::tel::Metric::BSP_mi_nodes, 1.0);

    const uint64_t rank = n / 2u;
    const uint64_t med = view.lo_ptr[rank];

    const uint64_t nL = rank;
    const uint64_t nR = n - rank - 1u;

    const uint64_t min_pos = view.lo + nL;
    const uint64_t max_pos_excl = view.hi - nR;
    const uint64_t K = (max_pos_excl > min_pos) ? (max_pos_excl - min_pos) : 0ull;
    const uint64_t y = (uint64_t)med - min_pos;

    bits += (K > 1ull) ? bits_scalar(K, y) : 0.0;

    if (nL > 0) {
      bits += p_bsp_tree_recurse(view.sub(view.lo, (uint64_t)med), select, next_state, nodes, tel);
    }
    if (nR > 0) {
      bits += p_bsp_tree_recurse(view.sub((uint64_t)med + 1ull, view.hi), select, next_state, nodes, tel);
    }
    return bits;
  }
  case BspSubPolicy::MedianEmptyIndex: {
    nodes += 1ull;

    const uint64_t m = U - n;
    if (m == 0ull)
      return bits;

    const uint64_t k = m / 2ull; // rank among empties
    const uint64_t mL = k, mR = m - k - 1ull;

    const uint64_t min_pos_empty = view.lo + mL;
    const uint64_t max_pos_empty_excl = view.hi - mR;
    const uint64_t K = (max_pos_empty_excl > min_pos_empty) ? (max_pos_empty_excl - min_pos_empty) : 0ull;

    const uint64_t e = kth_empty_in_span(k, view);
    const uint64_t y = e - min_pos_empty;

    bits += (K > 1ull) ? bits_scalar(K, y) : 0.0;

    const DatasetView L = view.sub(view.lo, e);
    const DatasetView R = view.sub(e + 1ull, view.hi);
    const uint64_t left_cnt = L.n(), right_cnt = R.n();

    if (left_cnt)
      bits += p_bsp_tree_recurse(L, select, next_state, nodes, tel);
    if (right_cnt)
      bits += p_bsp_tree_recurse(R, select, next_state, nodes, tel);
    return bits;
  }
  }
  return bits; // unreachable
}

template <class Select>
[[nodiscard]] inline PolicyResult p_bsp_tree(const std::vector<uint64_t>& V, uint64_t U, const Select& select) {
  if (U == 0ull || V.empty())
    return PolicyResult{0.0, {}};

  const DatasetView root = DatasetView::whole(V, U);
  uint64_t nodes = 0ull;
  bspx::tel::TelemetrySink tel;
  const double bits = p_bsp_tree_recurse(root, select, /*state=*/0ull, nodes, tel);

  (void)nodes;
  PolicyResult r{bits, {}};
  r.tel = tel;
  return r;
}

// ---------- TAGGED-BISECT
//
// p_tagged_bisect_recurse(view, Control, Tel)
//   Semantic:
//     • If U<=16: emit exact bitset for the span.
//     • If all present values form one contiguous run: tag “Run”, emit start and length.
//     • If n < log2(U): tag “Sparse”, emit |set| and each position in [0..U).
//     • If (U-n) < log2(U): tag “Holes”, emit |holes| and each hole position.
//     • Else: split by midpoint; tag “Split”; recurse left/right if non-empty.
//
struct TaggedBisectState {
  DatasetView view;
};

template <class Control>
bspx_HOT double p_tagged_bisect_recurse(const DatasetView& view, Control control, uint64_t& nodes,
                                        bspx::tel::TelemetrySink& tel) {
  if (control(view))
    return 0.0;

  const uint64_t U = view.U(), n = view.n();
  if (U == 0ull || n == 0ull)
    return 0.0;

  // Small base case: exact bitset
  if (U <= 16u) {
    tel.record(bspx::tel::Metric::TB_smallbit, 1.0);
    return (double)U;
  }

  double tag_bits = 1.0;

  // One contiguous run?
  const uint64_t first = *view.lo_ptr;
  const uint64_t last = *(view.hi_ptr - 1);
  if ((uint64_t)last - (uint64_t)first + 1ull == (uint64_t)n) {
    nodes += 1ull;
    tag_bits += bits_scalar(3, 0);
    tel.record(bspx::tel::Metric::TB_run, 1.0);
    const uint64_t Kstart = U - (uint64_t)n + 1ull;
    const uint64_t start = (uint64_t)first - view.lo;
    const uint64_t Klen = U; // length-1 in [0..U-1]
    const uint64_t lenm1 = (uint64_t)n - 1ull;
    return tag_bits + bits_scalar(Kstart, start) + bits_scalar(Klen, lenm1);
  }

  const double log2U = (double)ceil_log2_u64(U);

  // Sparse: emit present positions directly
  if ((double)n < log2U) {
    nodes += 1ull;
    tag_bits += bits_scalar(3, 1);
    tel.record(bspx::tel::Metric::TB_sparse, 1.0);
    double sum = 0.0;
    for (const uint64_t* p = view.lo_ptr; p != view.hi_ptr; ++p) {
      sum += bits_scalar(U, (uint64_t)(*p - (uint64_t)view.lo));
    }
    const double length_bits = bits_scalar(U, (uint64_t)n);
    return tag_bits + length_bits + sum;
  }

  // Co-sparse: emit hole positions
  const uint64_t cleared = (uint64_t)(U - (uint64_t)n);
  if ((double)cleared < log2U) {
    nodes += 1ull;
    tag_bits += bits_scalar(3, 2);
    tel.record(bspx::tel::Metric::TB_holes, 1.0);
    double sum = 0.0;
    uint64_t prev = view.lo;
    const uint64_t* p = view.lo_ptr;
    while (p != view.hi_ptr) {
      const uint64_t v = *p;
      for (uint64_t h = prev; h < (uint64_t)v; ++h)
        sum += bits_scalar(U, h - view.lo);
      prev = (uint64_t)v + 1ull;
      ++p;
    }
    for (uint64_t h = prev; h < view.hi; ++h)
      sum += bits_scalar(U, h - view.lo);
    const double length_bits = bits_scalar(U, (uint64_t)cleared);
    return tag_bits + length_bits + sum;
  }

  // Otherwise split by midpoint
  auto [L, R] = view.split_mid();
  nodes += 1ull;
  tel.record(bspx::tel::Metric::TB_split, 1.0);

  double bits = tag_bits;
  if (L.n())
    bits += p_tagged_bisect_recurse(L, control, nodes, tel);
  if (R.n())
    bits += p_tagged_bisect_recurse(R, control, nodes, tel);
  return bits;
}

[[nodiscard]] inline PolicyResult p_tagged_bisect(const std::vector<uint64_t>& V, uint64_t U) {
  if (U == 0ull || V.empty())
    return PolicyResult{0.0, {}};

  const DatasetView root_full = DatasetView::whole(V, U);

  // Narrow universe header + clamp to Ueff (as before)
  const uint64_t last = (root_full.hi_ptr > root_full.lo_ptr) ? (uint64_t)*(root_full.hi_ptr - 1) : 0ull;
  const uint64_t b_universe = ceil_log2_u64(last + 1ull);
  double bits = bits_scalar(33ull, b_universe);
  const uint64_t Ueff = std::min<uint64_t>(U, 1ull << b_universe);
  const uint64_t* p1_eff = lower_bound_u64(root_full.lo_ptr, root_full.hi_ptr, Ueff);
  const DatasetView root{root_full.lo_ptr, p1_eff, 0ull, Ueff};

  struct NeverControl {
    bspx_ALWAYS_INLINE bool operator()(const DatasetView&) const { return false; }
  } control;

  uint64_t nodes = 0ull;
  bspx::tel::TelemetrySink tel;
  bits += p_tagged_bisect_recurse(root, control, nodes, tel);

  (void)nodes;
  PolicyResult r{bits, {}};
  r.tel = tel;
  return r;
}

// ---------- JUDY
//
// p_judy_recurse<UseModeTags>(view, node_span)
//   Semantic:
//     • Partition the span into “node_span” equal-capacity children.
//     • Classify each child as E (empty), F (full), or B (balanced).
//     • Emit a small per-child descriptor depending on the chosen mode:
//         EB  : 1 bit per child (E=0, B=1; F treated as B in the descriptor)
//         FB  : 1 bit per child (F=0, B=1)
//         EFB : 2 bits per child (E=0, F=1, B=2)
//         B   : no descriptor (all B), just recurse
//       When UseModeTags=false, we force EB without emitting a mode tag.
//     • Recurse only into B-children (E and F terminate at this level).
//
struct JudyState {
  DatasetView view;
};
enum class JudyMode : uint8_t { EB = 0, FB = 1, B = 2, EFB = 3 };
enum class JChild : uint8_t { E = 0, F = 1, B = 2 };

template <bool UseModeTags, class Tel>
bspx_HOT double p_judy_recurse(const DatasetView& view, Tel& /*tel*/, uint32_t node_span, uint64_t& nodes) {
  const uint64_t U = view.U(), n = view.n();
  if (U == 0ull || n == 0ull)
    return 0.0;

  // Small leaf: exact bitmap
  if (U < (uint64_t)node_span)
    return (double)U;

  nodes += 1ull;

  struct ChildInfo {
    DatasetView view;
    uint64_t n = 0;
    JChild tag = JChild::E;
  };

  std::vector<ChildInfo> kids;
  kids.reserve(node_span);

  bool anyE = false, anyF = false, allB = true;

  for (uint32_t b = 0; b < node_span; ++b) {
    const uint64_t b_lo = view.lo + (U * (uint64_t)b) / (uint64_t)node_span;
    const uint64_t b_hi = view.lo + (U * (uint64_t)(b + 1u)) / (uint64_t)node_span;
    if (b_lo >= b_hi)
      continue; // degenerate

    DatasetView child = view.sub(b_lo, b_hi);
    const uint64_t cnt = child.n();
    const uint64_t Uchild = child.U();

    ChildInfo ci;
    ci.view = child;
    ci.n = cnt;

    if (cnt == 0ull) {
      ci.tag = JChild::E;
      anyE = true;
      allB = false;
    } else if (cnt == Uchild && UseModeTags) {
      ci.tag = JChild::F;
      anyF = true;
      allB = false;
    } else {
      ci.tag = JChild::B;
    }

    kids.push_back(ci);
  }

  auto pick_mode = [&]() -> JudyMode {
    if constexpr (!UseModeTags)
      return JudyMode::EB; // forced EB (no mode tag)
    if (allB)
      return JudyMode::B;
    if (!anyF)
      return JudyMode::EB;
    if (!anyE)
      return JudyMode::FB;
    return JudyMode::EFB;
  };

  const JudyMode mode = pick_mode();

  double bits = 0.0;
  if constexpr (UseModeTags)
    bits += 2.0; // 2-bit mode tag

  switch (mode) {
  case JudyMode::B: {
    // All B; no per-child storage
  } break;
  case JudyMode::EB: {
    for (const auto& c : kids) {
      const uint64_t y = (c.tag == JChild::E) ? 0ull : 1ull;
      bits += bits_scalar(2ull, y);
    }
  } break;
  case JudyMode::FB: {
    for (const auto& c : kids) {
      const uint64_t y = (c.tag == JChild::F) ? 0ull : 1ull;
      bits += bits_scalar(2ull, y);
    }
  } break;
  case JudyMode::EFB: {
    for (const auto& c : kids) {
      const uint64_t y = (c.tag == JChild::E) ? 0ull : (c.tag == JChild::F ? 1ull : 2ull);
      bits += bits_scalar(3ull, y);
    }
  } break;
  }

  // Recurse only into balanced children (true B, not F “treated as B”)
  for (const auto& c : kids) {
    if (c.tag == JChild::B) {
      bits += p_judy_recurse<UseModeTags>(c.view, *(Tel*)nullptr, node_span, nodes);
    }
  }

  return bits;
}

template <bool UseModeTags = true>
[[nodiscard]] inline PolicyResult p_judy(const std::vector<uint64_t>& V, uint64_t U, uint32_t node_span = 256u) {
  if (node_span == 0u)
    throw std::invalid_argument("p_judy: node_span must be > 0");

  if (U == 0ull || V.empty())
    return PolicyResult{0.0, {}};

  const DatasetView root = DatasetView::whole(V, U);
  uint64_t nodes = 0ull;
  bspx::tel::TelemetrySink tel; // currently unused by Judy hooks; reserved for future
  const double bits = p_judy_recurse<UseModeTags>(root, tel, node_span, nodes);

  (void)nodes;
  PolicyResult r{bits, {}};
  r.tel = tel;
  return r;
}

// ---------- PICK-BEST (per-span helper)
//
//   Semantic: For a given span, compute simple closed-form costs for a few
//   concrete encodings and return the best allowed by “available” (e.g. "BAR").
//     B: exact bitset  → cost = U
//     A: array of positions (present) → n × bits_scalar(U,•)
//     E: array of holes (absent)      → (U-n) × bits_scalar(U,•)
//     R: run endpoints                 → 2R × ceil_log2(U)
//
[[nodiscard]] inline PolicyResult p_pick_best(const std::vector<uint64_t>& V, uint64_t U, const char* available) {
  const DatasetView view = DatasetView::whole(V, U);
  if (view.U() == 0ull || view.n() == 0ull)
    return PolicyResult{0.0, {}};

  const double bitset_cost = (double)view.U();
  const uint64_t n = view.n();
  const double array_cost = (double)n * bits_scalar(view.U(), 0);
  const double empty_cost = (double)(view.U() - n) * bits_scalar(view.U(), 0);

  // Count runs in-order
  uint64_t runs = 0;
  if (n) {
    runs = 1;
    const uint64_t* k = view.lo_ptr;
    uint64_t prev = *k;
    ++k;
    for (; k != view.hi_ptr; ++k) {
      if (*k != prev + 1ull)
        ++runs;
      prev = *k;
    }
  }
  const double runs_cost = 2.0 * (double)runs * (double)ceil_log2_u64(view.U());

  assert(available && (std::strchr(available, 'B') || std::strchr(available, 'A') || std::strchr(available, 'R')));
  double best = std::numeric_limits<double>::infinity();
  if (std::strchr(available, 'B'))
    best = std::min(best, bitset_cost);
  if (std::strchr(available, 'A'))
    best = std::min(best, array_cost);
  if (std::strchr(available, 'E'))
    best = std::min(best, empty_cost);
  if (std::strchr(available, 'R'))
    best = std::min(best, runs_cost);

  return PolicyResult{best, {}};
}

// ---------- stats-probe (dummy policy; returns +inf size, detail = probes)
//
//   Semantic: Convert the active [0..U) window to 32-bit view (assumes U=2^32),
//   compute per-band box counts and skew stats; emit them into telemetry.
//
[[nodiscard]] inline PolicyResult stats_probe(const std::vector<uint64_t>& V, uint64_t U) {
  auto r = PolicyResult{std::numeric_limits<double>::infinity(), {}};
  r.tel = bspx::tel::stats_probe(std::vector<uint32_t>(V.begin(), V.end()), U);
  return r;
}

} // namespace bspx
