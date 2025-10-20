// bspx_main.cpp
//
// Program entry for the bspx experiments.
//
// High-level flow:
//   1) Parse CLI, load datasets from a directory (each dataset: sorted, unique uint64 values; U is capacity).
//   2) Define a set of *policies* (bit-cost calculators on a span) and *transforms* (Dataset -> children + side bits).
//      - Policies are pure evaluators: size_bits = f(values, U). They do not mutate input.
//      - Transforms rewrite a dataset into one or more local-coordinate children and may add side information bits.
//   3) Assemble *pipelines* = [Transform*] + Policy. Each pipeline runs on each dataset.
//   4) Execute all pipelines in parallel; collect rows {dataset, policy, bits, metrics}.
//   5) Postprocess results (compose some “sliced” results, order for presentation), write CSV, and print a summary.
//
// Notes for readers seeing this code for the first time:
//   • “LeftPopcount”, “MedianIndex”, “MedianEmptyIndex”, “Judy”, “Tagged-Bisect”, etc., are *policies* defined in
//     bspx_policy.hpp. Comments there describe *what* each policy encodes at a node/leaf.
//   • This file focuses on orchestration and on documenting *what each pipeline is doing* at a semantic level.
//

#include "bspx_policy.hpp"
#include "bspx_runtime.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <regex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace bspx;
using namespace rt;

// ------------------------- CLI -------------------------

struct CLI {
  std::string dir = "../.common/data-bitset"; // input directory (files -> datasets)
  std::string out = "report.csv";             // results table
};

// Parse a tiny CLI: --dir <path>, --out <file>, --help
static bool parse_cli(int argc, char** argv, CLI& a) {
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    if ((s == "--dir" || s == "-d") && i + 1 < argc) {
      a.dir = argv[++i];
    } else if (s == "--out" && i + 1 < argc) {
      a.out = argv[++i];
    } else if (s == "-h" || s == "--help") {
      std::cout << "Usage: bspx [--dir data] [--out report.csv]\n";
      return false;
    } else {
      std::cerr << "Unknown arg: " << s << "\n";
      return false;
    }
  }
  return true;
}

// ------------------------- Main -------------------------

int main(int argc, char** argv) {
  CLI cli;
  if (!parse_cli(argc, argv, cli))
    return 1;

  // Load datasets: runtime contract is that each dataset contains a sorted, duplicate-free vector<uint64_t>
  //     and an associated U (universe size / capacity). U may exceed the maximum present value.
  auto datasets = rt::load_datasets_from_dir(cli.dir);
  if (datasets.empty()) {
    std::cerr << "No datasets found in " << cli.dir << "\n";
    return 2;
  }

  // Define policies via light adapters.
  //     The lambdas here are intentionally thin wrappers to preserve readability at the callsites below.

  // Pick-best among a small fixed menu on a single span (bitset / array / holes / runs).
  auto make_pick_best = [](const char* available) {
    return [available](const std::vector<uint64_t>& V, uint64_t U) -> PolicyResult {
      return p_pick_best(V, U, available);
    };
  };

  // Judy family (no mode tag vs tagged mode).
  auto make_judy = [](const uint32_t node_span) {
    return [node_span](const std::vector<uint64_t>& V, uint64_t U) -> PolicyResult {
      return p_judy<false>(V, U, node_span);
    };
  };
  auto make_judy_with_mode_tags = [](const uint32_t node_span) {
    return [node_span](const std::vector<uint64_t>& V, uint64_t U) -> PolicyResult {
      return p_judy<true>(V, U, node_span);
    };
  };

  // Force a single BSP sub-policy at every node (useful for isolating behaviors).
  auto make_bsp_always = [](BspSubPolicy policy) {
    return [policy](const std::vector<uint64_t>& V, uint64_t U) -> PolicyResult {
      // The selector ignores the current span; it only returns the fixed policy.
      return p_bsp_tree(V, U,
                        [&](const uint64_t* /*it_lo*/, const uint64_t* /*it_hi*/, uint64_t /*lo*/, uint64_t /*hi*/,
                            uint64_t st) { return BspDecision{policy, 0.0, st}; });
    };
  };

  // Early-stop variant for LeftPopcount:
  //   If below a leaf threshold, end recursion and encode the leaf directly with:
  //     method=0 → enumerative subset
  //     method=1 → min(pos/holes) simple array
  //     method=2 → bitset (when span itself is small)
  auto make_leftpop_early_stop = [](uint64_t leaf_thresh,
                                    uint64_t leaf_method /* 0=enumerative rank, 1=array, 2=bitset */) {
    return [leaf_thresh, leaf_method](const std::vector<uint64_t>& V, uint64_t U) -> PolicyResult {
      return p_bsp_tree( //
          V, U, [&](const uint64_t* it_lo, const uint64_t* it_hi, uint64_t lo, uint64_t hi, uint64_t st) {
            const uint64_t Uloc = hi - lo;                // span width
            const uint64_t n = (uint64_t)(it_hi - it_lo); // #present
            if (leaf_method == 2) {
              if (Uloc <= leaf_thresh)
                return BspDecision{BspSubPolicy::EarlyStop, (double)Uloc, st};
            } else {
              if (n <= leaf_thresh || Uloc - n <= leaf_thresh) {
                const double b = (leaf_method == 0) ? bits_enumerative_subset(Uloc, n)
                                                    : std::min<uint64_t>(n, Uloc - n) * bits_scalar(Uloc, 0);
                return BspDecision{BspSubPolicy::EarlyStop, b, st};
              }
            }
            return BspDecision{BspSubPolicy::LeftPopcount, 0.0, st};
          });
    };
  };

  // Mixed BSP policy with a small finite-state controller (“st”):
  //   • Far from leaves: choose policy by measured balance at this node.
  //   • Every few levels: deterministically alternate to exercise MedianIndex as well.
  Policy p_mixed = +[](const std::vector<uint64_t>& V, uint64_t U) -> PolicyResult {
    return p_bsp_tree( //
        V, U, [](const uint64_t* it_lo, const uint64_t* it_hi, uint64_t lo, uint64_t hi, uint64_t st) {
          const uint64_t Uloc = hi - lo;
          const uint64_t n = (uint64_t)(it_hi - it_lo);

          // Periodically force alternation (countdown in low bits of st).
          if (st >= 2 || Uloc <= 256) {
            const bool is_balanced = (st & 1);
            auto policy = is_balanced ? BspSubPolicy::MedianIndex : BspSubPolicy::LeftPopcount;
            return BspDecision{policy, 0.0, st - 2};
          }

          // Balance heuristic: left fraction inside [1/3, 2/3].
          const uint64_t pivot = lo + (Uloc >> 1);
          const uint64_t* it_mid = lower_bound_u64(it_lo, it_hi, pivot);
          const uint64_t nL = (uint64_t)(it_mid - it_lo);
          const double left_frac = (n == 0) ? 0.0 : double(nL) / double(n);
          const bool is_balanced = (left_frac > 1.0 / 3.0) && (left_frac < 2.0 / 3.0);

          // Lightweight tag bits (illustrative).
          const double bit_cost = 0.5;
          auto policy = is_balanced ? BspSubPolicy::MedianIndex : BspSubPolicy::LeftPopcount;

          // Feed-forward: re-arm the alternation window, encoding balance parity in LSB.
          return BspDecision{policy, bit_cost, (uint64_t)(is_balanced ? 7 : 6)};
        });
  };

  // Convenience transform bundles:
  //
  //  t_bsp: U-narrowing and cardinality headers.
  //
  //  t_bsp_cluster: split dataset into clusters
  //    Recursively splits at midpoint; stops early on small or extreme spans.
  //    Stores left cluster counts instead of member counts (lower cardinality).
  //
  //  t_bsp_rle:
  //    Stores region marks, where runs are common, using left-region-counts
  //    Rewrites marked regions as run endpoints (a0,b0),(a1,b1),...
  auto t_bsp = {t_simple("UN")};
  auto t_bsp_cluster = {t_simple("U"), t_left_cluster_count(256, ClusterRecordNMethod::Narrow)};
  auto t_bsp_rle = {t_simple("U"), t_left_run_count_pairs(), t_simple("N")};

  // Compose pipelines.
  //
  //  Each Pipeline = {label, transforms..., policy}. Transforms are applied in order,
  //  producing zero or more child datasets (often in local coordinates); the policy then
  //  evaluates each child and the runtime sums bits + side bits to form a row.
  std::vector<Pipeline> pipes;

  // Baseline Judy policy (no mode tags).
  pipes.push_back(Pipeline{"judy256", {t_simple("U")}, make_judy(256)});
  pipes.push_back(Pipeline{"judy16", {t_simple("U")}, make_judy(16)});
  pipes.push_back(Pipeline{"judy4", {t_simple("U")}, make_judy(4)});
  pipes.push_back(Pipeline{"judy2", {t_simple("U")}, make_judy(2)});

  // Roaring-style policy: slice into 2^16 chunks (local coords), then charge a small per-child header.
  std::vector<Transform> t_roaring;
  t_roaring.push_back(t_slice(0, 16));                  // child per [v>>16], localize to low 16 bits
  t_roaring.push_back(t_simple("", [](const Dataset&) { // per-child fixed header (key+cardinality)
    return TransformSimpleDecision{.cost_bits = 32.0, .drop = false};
  }));

  pipes.push_back(Pipeline{"roaring-B+A", t_roaring, make_pick_best("BA")});
  pipes.push_back(Pipeline{"roaring-B+A+R", t_roaring, make_pick_best("BAR")});
  pipes.push_back(Pipeline{"tagged-bisect", {t_simple("N")}, p_tagged_bisect});
  pipes.push_back(Pipeline{"leftpop", t_bsp, make_bsp_always(BspSubPolicy::LeftPopcount)});
  pipes.push_back(Pipeline{"leftpop-then@N4-enum", t_bsp, make_leftpop_early_stop(/*thresh=*/4, /*method=*/0)});
  pipes.push_back(Pipeline{"leftpop-then@N1-array", t_bsp, make_leftpop_early_stop(/*thresh=*/1, /*method=*/1)});
  pipes.push_back(Pipeline{"leftpop-then@S8-bitset", t_bsp, make_leftpop_early_stop(/*thresh=*/8, /*method=*/2)});
  pipes.push_back(Pipeline{"median-index", t_bsp, make_bsp_always(BspSubPolicy::MedianIndex)});
  pipes.push_back(Pipeline{"bsp-mixed", t_bsp, p_mixed});

  // Cluster/RLE pre-passes followed by BSP baselines.
  pipes.push_back(Pipeline{"cluster-then-leftpop", t_bsp_cluster, make_bsp_always(BspSubPolicy::LeftPopcount)});
  pipes.push_back(Pipeline{"cluster-then-median", t_bsp_cluster, make_bsp_always(BspSubPolicy::MedianIndex)});
  pipes.push_back(Pipeline{"rle-then-leftpop", t_bsp_rle, make_bsp_always(BspSubPolicy::LeftPopcount)});
  pipes.push_back(Pipeline{"rle-then-median", t_bsp_rle, make_bsp_always(BspSubPolicy::MedianIndex)});

  // Helper: build a “slice to chunks → policy” pipeline with optional extras:
  //   extras flags:
  //     'U' → U-narrowing before slicing,
  //     'N' → record |set| per child,
  //     'D' → drop full children (when ds.U == |values|).
  //     '1' → add 1-bit fixed cost per-slice.
  auto add_slice_pipeline = [&](int lo, int hi, const char* label_base, Policy pol, const char* extras = "") {
    std::vector<Transform> ts;
    if (std::strchr(extras, 'U')) // narrow U before slicing
      ts.push_back(t_simple("U"));

    ts.push_back(t_slice(lo, hi)); // explode into per-chunk children in local coords

    ts.push_back(t_simple("", [extras](const Dataset& ds) {
      const bool allowN = std::strchr(extras, 'N');
      const bool wantDropFull = std::strchr(extras, 'D');
      const bool extraCost = std::strchr(extras, '1');
      const bool hasVals = (bool)ds.values;
      const uint64_t n = hasVals ? ds.values->size() : 0ull;

      const bool drop = wantDropFull && hasVals && (ds.U == n);
      const double rec_bits = (drop || !allowN || !hasVals) ? 0.0 : bits_scalar(ds.U, (uint64_t)n);
      return TransformSimpleDecision{.cost_bits = rec_bits + (double)extraCost, .drop = drop};
    }));

    auto label = std::string("slice[") + std::to_string(lo) + "to" + std::to_string(hi) + ")->" + label_base;
    pipes.push_back(Pipeline{label, ts, pol});
  };

  // Slice low bits to local and run BSPs there; also try Judy in the complementary high range.
  add_slice_pipeline(0, 8, "leftpop", make_bsp_always(BspSubPolicy::LeftPopcount), "N");
  add_slice_pipeline(0, 12, "leftpop", make_bsp_always(BspSubPolicy::LeftPopcount), "N");
  add_slice_pipeline(0, 16, "leftpop", make_bsp_always(BspSubPolicy::LeftPopcount), "N");
  add_slice_pipeline(0, 8, "median-index", make_bsp_always(BspSubPolicy::MedianIndex), "N");
  add_slice_pipeline(0, 12, "median-index", make_bsp_always(BspSubPolicy::MedianIndex), "N");
  add_slice_pipeline(0, 16, "median-index", make_bsp_always(BspSubPolicy::MedianIndex), "N");
  add_slice_pipeline(8, 32, "leftpop", make_bsp_always(BspSubPolicy::LeftPopcount), "UN");
  add_slice_pipeline(12, 32, "leftpop", make_bsp_always(BspSubPolicy::LeftPopcount), "UN");
  add_slice_pipeline(16, 32, "leftpop", make_bsp_always(BspSubPolicy::LeftPopcount), "UN");

  add_slice_pipeline(8, 32, "judy16", make_judy(16), "U");
  add_slice_pipeline(12, 32, "judy16", make_judy(16), "U");
  add_slice_pipeline(16, 32, "judy16", make_judy(16), "U");
  add_slice_pipeline(8, 32, "judyX", make_judy_with_mode_tags(16), "U");
  add_slice_pipeline(12, 32, "judyX", make_judy_with_mode_tags(16), "U");
  add_slice_pipeline(16, 32, "judyX", make_judy_with_mode_tags(16), "U");

  // Slice + drop full children before policy evaluation.
  add_slice_pipeline(0, 8, "leftpop-drop-full", make_bsp_always(BspSubPolicy::LeftPopcount), "UND");
  add_slice_pipeline(0, 12, "leftpop-drop-full", make_bsp_always(BspSubPolicy::LeftPopcount), "UND");
  add_slice_pipeline(8, 32, "judy4", make_judy(4), "U");
  add_slice_pipeline(0, 8, "judy4", make_judy(4), "D1");

  add_slice_pipeline(8, 32, "judy16", make_judy(16), "U");

  // Slice + pick-best on local 8-bit chunks; charge a tiny per-child header for cardinality.
  pipes.push_back(
      Pipeline{"slice[0to8)->pick-best",
               {t_slice(0, 8), t_simple("", [](const Dataset&) { return TransformSimpleDecision{.cost_bits = 8.0}; })},
               make_pick_best("BAE")});

  // Probe policy that records structural stats in telemetry (not a real compressor).
  pipes.push_back(Pipeline{"stats-probe", {}, stats_probe});

  // Execute all pipelines in parallel with default configuration.
  rt::ParallelConfig pcfg; // threads=hw_concurrency, batch=10, timeout=0 by default
  auto rows = rt::run_parallel(datasets, pipes, pcfg);

  // Try alternative scalar integer codecs for a canonical policy while holding everything else fixed.
  g_intCodec = IntCodec::Fixed;
  pipes.push_back(Pipeline{"leftpop-scalar-fixed", t_bsp, make_bsp_always(BspSubPolicy::LeftPopcount)});
  auto fixed_scalar_rows = rt::run_parallel(datasets, std::vector{pipes.back()}, pcfg);

  g_intCodec = IntCodec::Uniform;
  pipes.push_back(Pipeline{"leftpop-scalar-uniform", t_bsp, make_bsp_always(BspSubPolicy::LeftPopcount)});
  auto uniform_scalar_rows = rt::run_parallel(datasets, std::vector{pipes.back()}, pcfg);

  // Merge those variant rows into the main table.
  rows.insert(rows.end(), fixed_scalar_rows.begin(), fixed_scalar_rows.end());
  rows.insert(rows.end(), uniform_scalar_rows.begin(), uniform_scalar_rows.end());

  // Compose two-stage results formed by disjoint bit-slices:
  //   e.g., “judy16 on high bits” + “leftpop on low bits”.
  auto combineSlices = [&](std::string left, std::string right, std::string label) {
    // Index by dataset; require both pieces to exist.
    std::unordered_map<std::string, const Row*> left_map, right_map;
    for (const auto& r : rows) {
      if (r.policy == left)
        left_map[r.dataset] = &r;
      else if (r.policy == right)
        right_map[r.dataset] = &r;
    }
    for (const auto& [ds, rowL] : left_map) {
      auto it = right_map.find(ds);
      if (it != right_map.end()) {
        const Row* rowR = it->second;
        assert(rowL->dataset == rowR->dataset);
        Row merged;
        merged.dataset = rowL->dataset;
        merged.n = rowL->n;
        merged.policy = label;
        merged.bits = rowL->bits + rowR->bits;
        // Split metrics strings, prepend "L_" and "R_" to each part, then join with "|"
        std::vector<std::string> left_parts, right_parts, merged_parts;
        auto split_and_prefix = [](const std::string& s, const char* prefix) {
          std::vector<std::string> out;
          size_t start = 0, end;
          while ((end = s.find('|', start)) != std::string::npos) {
            out.push_back(std::string(prefix) + s.substr(start, end - start));
            start = end + 1;
          }
          if (start < s.size())
            out.push_back(std::string(prefix) + s.substr(start));
          return out;
        };
        left_parts = split_and_prefix(rowL->metrics, "L_");
        left_parts.insert(left_parts.begin(), "L_bits=" + std::to_string(rowL->bits));
        right_parts = split_and_prefix(rowR->metrics, "R_");
        right_parts.insert(right_parts.begin(), "R_bits=" + std::to_string(rowR->bits));
        merged_parts.reserve(left_parts.size() + right_parts.size());
        merged_parts.insert(merged_parts.end(), left_parts.begin(), left_parts.end());
        merged_parts.insert(merged_parts.end(), right_parts.begin(), right_parts.end());
        merged.metrics.clear();
        for (size_t i = 0; i < merged_parts.size(); ++i) {
          if (i > 0)
            merged.metrics += "|";
          merged.metrics += merged_parts[i];
        }
        rows.push_back(std::move(merged));
      }
    }
  };
  combineSlices("slice[8to32)->judy16", "slice[0to8)->leftpop", "judy16-then@D8-leftpop");
  combineSlices("slice[12to32)->judy16", "slice[0to12)->leftpop", "judy16-then@D12-leftpop");
  combineSlices("slice[16to32)->judy16", "slice[0to16)->leftpop", "judy16-then@D16-leftpop");
  combineSlices("slice[8to32)->judy16", "slice[0to8)->median-index", "judy16-then@D8-median-index");
  combineSlices("slice[12to32)->judy16", "slice[0to12)->median-index", "judy16-then@D12-median-index");
  combineSlices("slice[16to32)->judy16", "slice[0to16)->median-index", "judy16-then@D16-median-index");
  combineSlices("slice[8to32)->judyX", "slice[0to8)->leftpop-drop-full", "judyX-then@D8-leftpop");
  combineSlices("slice[12to32)->judyX", "slice[0to12)->leftpop-drop-full", "judyX-then@D12-leftpop");
  combineSlices("slice[16to32)->judyX", "slice[0to16)->leftpop-drop-full", "judyX-then@D16-leftpop");
  combineSlices("slice[8to32)->judy16", "slice[0to8)->pick-best", "judy16-then@D8-pickbest");
  combineSlices("slice[8to32)->judy4", "slice[0to8)->judy4", "judy4-then@D8-drop-full");

  for (const auto& ds : datasets) {
    Row shannon_row;
    auto n = ds.values->size();
    shannon_row.dataset = ds.name;
    shannon_row.n = n;
    shannon_row.policy = "shannon-uniform-best";
    shannon_row.bits = n * log2(n);
    rows.push_back(std::move(shannon_row));
  }

  // Presentation order for the report; policies absent for a dataset are simply skipped in the printout loop below.
  auto rowOrder = {
      "roaring-B+A",
      "roaring-B+A+R",
      "leftpop",
      "median-index",
      "tagged-bisect",
      "bsp-mixed",
      "leftpop-then@N4-enum",
      "leftpop-then@N1-array",
      "leftpop-then@S8-bitset",
      "leftpop-scalar-fixed",
      "leftpop-scalar-uniform",
      "cluster-then-leftpop",
      "cluster-then-median",
      "rle-then-leftpop",
      "rle-then-median",
      "judy256",
      "judy16",
      "judy4",
      "judy2",
      "judy16-then@D8-leftpop",
      "judy16-then@D12-leftpop",
      "judy16-then@D16-leftpop",
      "judy16-then@D8-median-index",
      "judy16-then@D12-median-index",
      "judy16-then@D16-median-index",
      "judyX-then@D8-leftpop",
      "judyX-then@D12-leftpop",
      "judyX-then@D16-leftpop",
      "judy16-then@D8-pickbest",
      "judy4-then@D8-drop-full",
      "stats-probe",
  };

  // Remove all slice-internals (intermediate per-chunk rows); we keep only named top-level policies.
  rows.erase(std::remove_if(rows.begin(), rows.end(), [](const Row& r) { return r.policy.rfind("slice[", 0) == 0; }),
             rows.end());

  // Stable sort by the presentation order above; everything else remains stable within each bucket.
  std::vector<std::string> order_vec(rowOrder.begin(), rowOrder.end());
  std::unordered_map<std::string, size_t> policy_order;
  for (size_t i = 0; i < order_vec.size(); ++i)
    policy_order[order_vec[i]] = i;

  std::stable_sort(rows.begin(), rows.end(), [&](const Row& a, const Row& b) {
    const auto ita = policy_order.find(a.policy);
    const auto itb = policy_order.find(b.policy);
    const size_t ia = (ita != policy_order.end()) ? ita->second : order_vec.size();
    const size_t ib = (itb != policy_order.end()) ? itb->second : order_vec.size();
    return ia < ib;
  });

  // Write CSV (dataset, policy, bits, metrics, etc.) via the runtime helper.
  rt::write_csv(cli.out, rows);

  // Human-readable summary to stdout.
  //
  // We aggregate total bits per policy across datasets (the same aggregates the CSV encodes but as a quick glance).
  std::cout.setf(std::ios::fixed);
  std::cout.precision(4);
  std::cout << "\nWrote " << cli.out << " (" << rows.size() << " rows)\n\n";

  struct Agg {
    uint64_t ds = 0;
    double bits = 0.0;
  };
  std::unordered_map<std::string, Agg> agg;
  for (const auto& r : rows) {
    auto& a = agg[r.policy];
    a.ds += 1;
    a.bits += r.bits;
  }

  // Align labels for a tidy table.
  size_t max_label_len = 0;
  for (const auto& r : rows)
    max_label_len = std::max(max_label_len, r.policy.size());

  for (auto& label : rowOrder) {
    auto it = agg.find(label);
    if (it == agg.end())
      continue;
    const auto& A = it->second;
    const double mb = (A.bits / 8.0) / 1e6; // total bytes → MB
    std::cout << std::left << std::setw(max_label_len + 2) << (std::string(label) + ":") << "MB.total=" << mb << "\n";
  }

  // Extract a simple histogram from the stats-probe policy’s telemetry.
  std::unordered_map<std::string, uint64_t> band_counts;
  static const std::regex re(R"(PROBE_boxes\[(\d+)\]=(\d+))");

  for (const auto& r : rows) {
    if (r.policy != "stats-probe")
      continue;
    for (std::sregex_iterator it(r.metrics.begin(), r.metrics.end(), re), end; it != end; ++it) {
      const auto& m = *it;
      band_counts[m[1].str()] += static_cast<uint64_t>(std::stoull(m[2].str()));
    }
  }

  std::vector<std::pair<std::string, uint64_t>> band_vec(band_counts.begin(), band_counts.end());
  std::sort(band_vec.begin(), band_vec.end(),
            [&](const auto& a, const auto& b) { return std::stoi(a.first) < std::stoi(b.first); });

  std::cout << "\nBoxes per band:\n";
  for (const auto& [band, cnt] : band_vec)
    std::cout << std::left << std::setw(4) << (band + ": ") << cnt << "\n";

  return 0;
}
