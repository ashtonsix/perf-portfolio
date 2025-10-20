// bspx_runtime.hpp
// Runtime utilities: dataset loading, transform pipelines, task management and CSV reporting.

#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <cxxabi.h>
#include <pthread.h>
#include <sched.h>
#include <string.h>
#endif

#include "bspx_policy.hpp" // for types and low-level helpers

namespace bspx {
namespace rt {

namespace fs = std::filesystem;

// ---- Dataset & loading ----

inline bool is_sorted_unique(const std::vector<uint64_t>& v) {
  auto it = std::adjacent_find(v.begin(), v.end(), std::greater_equal<uint64_t>());
  return it == v.end();
}

inline bool parse_uint64_csv(const std::string& text, std::vector<uint64_t>& out) {
  out.clear();
  const char* p = text.c_str();
  const char* end = p + text.size();
  auto skip_ws = [&](const char*& q) {
    while (q < end && std::isspace((unsigned char)*q))
      ++q;
  };
  while (p < end) {
    skip_ws(p);
    if (p >= end)
      break;
    uint64_t val = 0;
    bool any = false;
    while (p < end && std::isdigit((unsigned char)*p)) {
      any = true;
      val = val * 10 + (uint64_t)(*p - '0');
      ++p;
    }
    if (!any)
      return false;
    out.push_back(val);
    skip_ws(p);
    if (p < end) {
      if (*p == ',' || std::isspace((unsigned char)*p)) {
        ++p;
        continue;
      }
      return false;
    }
  }
  return true;
}

inline std::vector<Dataset> load_datasets_from_dir(const std::string& dir) {
  std::vector<Dataset> out;
  std::vector<fs::path> files;
  for (auto& e : fs::recursive_directory_iterator(dir))
    if (e.is_regular_file())
      files.push_back(e.path());
  for (const auto& p : files) {
    std::ifstream in(p, std::ios::binary);
    if (!in)
      continue;
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto vec = std::make_shared<std::vector<uint64_t>>();
    if (!parse_uint64_csv(text, *vec))
      continue;
    if (!is_sorted_unique(*vec))
      continue;
    out.push_back(Dataset{p.string(), vec, (uint64_t)UINT32_MAX + 1ull});
  }
  return out;
}

// ---- Pipelines (transforms + terminal policy) ----
struct Pipeline {
  std::string label;         // display label (e.g., "slice16->judy256")
  std::vector<Transform> ts; // ordered transforms
  bspx::Policy policy;       // terminal encoder policy
};

namespace detail {
inline bspx::PolicyResult eval_terminal(const Dataset& d, const bspx::Policy& pol) {
  return pol(*d.values, d.U);
}

inline bspx::PolicyResult eval_recursive(const Dataset& d, const Pipeline& P, size_t t_index,
                                         double accrued_transform_bits) {
  if (t_index >= P.ts.size()) {
    auto r = eval_terminal(d, P.policy);
    r.size_bits += accrued_transform_bits;
    return r;
  }
  const Transform& T = P.ts[t_index];
  TransformOutput out = T(d);

  double sum_bits = 0.0;
  bspx::tel::TelemetrySink tel;

  for (const auto& child : out.children) {
    auto r = eval_recursive(child, P, t_index + 1, accrued_transform_bits);
    sum_bits += r.size_bits;
    tel += r.tel;
  }

  sum_bits += out.cost_bits;
  // Accumulate transform-level telemetry
  tel += out.tel;

  bspx::PolicyResult R{sum_bits, {}};
  R.tel = tel;
  return R;
}
} // namespace detail

inline bspx::PolicyResult run_pipeline(const Dataset& d, const Pipeline& P) {
  return detail::eval_recursive(d, P, 0u, 0.0);
}

// ---- Reporting helpers ----
struct Row {
  std::string dataset;
  size_t n;
  std::string policy;
  double bits;
  std::string metrics;
};

inline void write_csv(const std::string& path, const std::vector<Row>& rows) {
  std::ofstream out(path);
  out.setf(std::ios::fixed);
  out.precision(6);
  out << "file,n,policy,bits,metrics\n";
  auto sorted = rows;
  std::stable_sort(sorted.begin(), sorted.end(),
                   [](const Row& a, const Row& b) { return strverscmp(a.dataset.c_str(), b.dataset.c_str()) < 0; });
  for (const auto& r : sorted) {
    out << r.dataset << "," << r.n << "," << r.policy << "," << r.bits << "," << r.metrics << "\n";
  }
}

// ---- Simple runner ----
struct RunConfig {
  std::string out_csv = "report.csv";
};

inline std::vector<Row> run(const std::vector<Dataset>& data, const std::vector<Pipeline>& pipes,
                            const RunConfig& cfg = {}) {
  (void)cfg;
  std::vector<Row> rows;
  rows.reserve(data.size() * pipes.size());

  auto last_print = std::chrono::steady_clock::now();
  for (size_t i = 0; i < data.size(); ++i) {
    const auto& ds = data[i];
    for (const auto& P : pipes) {
      auto r = run_pipeline(ds, P);
      const size_t n = ds.values ? ds.values->size() : 0ull;
      rows.push_back(Row{ds.name, n, P.label, r.size_bits, r.tel.dump()});
    }
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_print).count() >= 1) {
      std::cout << i << "/" << data.size() << " datasets processed" << std::endl;
      last_print = now;
    }
  }
  return rows;
}

// ---- Parallel runner ----
//
// Schedules contiguous dataset tiles to workers (tile t -> worker t % threads).
// Within each tile, we evaluate ALL pipelines per dataset (dataset-major) to
// maximize cache locality. On timeout or exception, a job is marked failed,
// skipped, and the responsible worker thread is restarted.
//
// Usage:
//   bspx::rt::ParallelConfig cfg;
//   cfg.threads = 8;
//   cfg.batch = 50;
//   cfg.timeout_ms = 2000;
//   auto rows = bspx::rt::run_parallel(datasets, pipelines, cfg);
//   bspx::rt::write_csv("report.csv", rows);

struct ParallelConfig {
  int threads = std::max(1u, std::thread::hardware_concurrency());
  int batch = 10;     // datasets per tile
  int timeout_ms = 0; // 0 = no timeout
  bool pin_workers = true;
};

#if defined(__linux__)
// ---- helpers (Linux) ----
namespace detail_parallel {

using Clock = std::chrono::steady_clock;
using std::chrono::milliseconds;

static inline int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

static std::vector<int> allowed_cpus() {
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) != 0) {
    unsigned n = std::max(1u, std::thread::hardware_concurrency());
    std::vector<int> v(n);
    for (unsigned i = 0; i < n; ++i)
      v[i] = (int)i;
    return v;
  }
  std::vector<int> v;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
    if (CPU_ISSET(cpu, &set))
      v.push_back(cpu);
  if (v.empty())
    v.push_back(0);
  return v;
}

static void pin_current_thread_to_cpu(int cpu) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

static void set_thread_name(const char* name) {
#if defined(__GLIBC__)
  pthread_setname_np(pthread_self(), name);
#endif
}

} // namespace detail_parallel
#endif // __linux__

// Public API
inline std::vector<Row> run_parallel(const std::vector<Dataset>& data, const std::vector<Pipeline>& pipes,
                                     const ParallelConfig& cfg = {}) {
  std::vector<Row> rows;
  rows.reserve(data.size() * pipes.size());

#if !defined(__linux__)
  // Fallback: no parallelism/timeout/pinning on non-Linux.
  RunConfig single_cfg;
  rows = run(data, pipes, single_cfg);
  return rows;
#else
  using namespace detail_parallel;

  if (data.empty() || pipes.empty())
    return rows;

  // ---- tiling over datasets ----
  const int D = (int)data.size();
  const int TILE = std::max(1, cfg.batch);
  const int num_tiles = (D + TILE - 1) / TILE;

  // ---- worker affinity ----
  const int W = std::max(1, cfg.threads);
  const auto cpus = allowed_cpus();
  const int C = (int)cpus.size();
  std::vector<int> worker_cpu(W);
  for (int w = 0; w < W; ++w)
    worker_cpu[w] = cpus[w % C];

  // ---- job model: one job == one dataset tile ----
  struct Job {
    int start, end, tile, cpu;
  };
  std::vector<Job> jobs;
  jobs.reserve(num_tiles);
  for (int t = 0; t < num_tiles; ++t) {
    int s = t * TILE;
    int e = std::min(s + TILE, D);
    int wid = t % W;
    jobs.push_back(Job{s, e, t, worker_cpu[wid]});
  }

  // Per-worker queues (dataset ownership)
  std::vector<std::vector<int>> jobs_by_worker(W);
  for (int j = 0; j < (int)jobs.size(); ++j) {
    int wid = jobs[j].tile % W;
    jobs_by_worker[wid].push_back(j);
  }

  // Job status: 0=pending, 1=running, 2=done, 3=failed/timeout
  std::vector<std::atomic<int>> jstatus(jobs.size());
  for (auto& s : jstatus)
    s.store(0, std::memory_order_relaxed);
  std::vector<std::atomic<int>> next_idx(W);
  for (int w = 0; w < W; ++w)
    next_idx[w].store(0, std::memory_order_relaxed);

  std::atomic<int> completed{0};
  std::atomic<int> datasets_done{0};
  std::mutex rows_mu, cout_mu;

  // Worker state for supervision
  struct WorkerState {
    std::thread thr;
    std::atomic<bool> running{false};
    std::atomic<int> current_job{-1};
    std::atomic<int64_t> start_ns{0};
    int id = -1;
    int cpu = -1;
  };

  // Core job body (throws on any dataset/pipeline exception to enforce restart)
  auto run_one_job = [&](int /*worker_id*/, int job_idx, int /*worker_cpu_id*/) {
    jstatus[job_idx].store(1, std::memory_order_release);

    const Job& J = jobs[job_idx];
    const int P = (int)pipes.size();
    const bool has_tmo = (cfg.timeout_ms > 0);
    const int64_t tmo_ns = (int64_t)cfg.timeout_ms * 1'000'000;
    const int64_t t_start = now_ns();

    for (int di = J.start; di < J.end; ++di) {
      if (has_tmo && (now_ns() - t_start) > tmo_ns) {
        throw std::runtime_error("tile timeout");
      }
      const Dataset& ds = data[di];
      const size_t n = ds.values ? ds.values->size() : 0ull;

      // Dataset-major: evaluate all pipelines before moving to the next dataset.
      for (int pi = 0; pi < P; ++pi) {
        const Pipeline& Pn = pipes[pi];
        double bits = 0.0;
        std::string mkv;

        try {
          auto r = run_pipeline(ds, Pn);
          bits = r.size_bits;
          mkv = r.tel.dump();
        } catch (...) {
          throw;
        }

        // Append CSV row immediately
        {
          std::lock_guard<std::mutex> lk(rows_mu);
          rows.push_back(Row{ds.name, n, Pn.label, bits, mkv});
        }
      }
      // Count dataset as processed once all pipelines are done
      datasets_done.fetch_add(1, std::memory_order_acq_rel);
    }

    jstatus[job_idx].store(2, std::memory_order_release);
    completed.fetch_add(1, std::memory_order_acq_rel);
  };

  // Worker thread entry (pulls jobs from its own queue)
  auto worker_main = [&](WorkerState* WS) {
    try {
      {
        char name[16];
        std::snprintf(name, sizeof(name), "wrk-%d", WS->id);
        set_thread_name(name);
      }
      if (cfg.pin_workers && WS->cpu >= 0)
        pin_current_thread_to_cpu(WS->cpu);

      WS->running.store(true, std::memory_order_release);
      auto& bucket = jobs_by_worker[WS->id];

      while (true) {
        int k = next_idx[WS->id].fetch_add(1, std::memory_order_acq_rel);
        if (k >= (int)bucket.size())
          break;
        int job_idx = bucket[k];

        WS->current_job.store(job_idx, std::memory_order_release);
        WS->start_ns.store(now_ns(), std::memory_order_release);

        run_one_job(WS->id, job_idx, WS->cpu);

        WS->current_job.store(-1, std::memory_order_release);
        WS->start_ns.store(0, std::memory_order_release);
      }
      WS->running.store(false, std::memory_order_release);
    } catch (const std::exception& e) {
      int ji = WS->current_job.load(std::memory_order_acquire);
      if (ji >= 0) {
        int expected = 1;
        if (jstatus[ji].compare_exchange_strong(expected, 3, std::memory_order_acq_rel)) {
          completed.fetch_add(1, std::memory_order_acq_rel);
        }
        std::lock_guard<std::mutex> lk(cout_mu);
        const Job& jb = jobs[ji];
        int y = std::max(jb.start, jb.end - 1);
        std::cerr << "Failed to process datasets " << jb.start << ".." << y << ": " << e.what() << "\n";
      }
      WS->current_job.store(-1, std::memory_order_release);
      WS->start_ns.store(0, std::memory_order_release);
      WS->running.store(false, std::memory_order_release);
    } catch (...) {
      int ji = WS->current_job.load(std::memory_order_acquire);
      if (ji >= 0) {
        int expected = 1;
        if (jstatus[ji].compare_exchange_strong(expected, 3, std::memory_order_acq_rel)) {
          completed.fetch_add(1, std::memory_order_acq_rel);
        }
        std::lock_guard<std::mutex> lk(cout_mu);
        const Job& jb = jobs[ji];
        int y = std::max(jb.start, jb.end - 1);
        std::cerr << "Failed to process datasets " << jb.start << ".." << y << ": unknown error\n";
      }
      WS->current_job.store(-1, std::memory_order_release);
      WS->start_ns.store(0, std::memory_order_release);
      WS->running.store(false, std::memory_order_release);
    }
  };

  // Launch workers
  std::vector<std::unique_ptr<WorkerState>> workers;
  workers.reserve(W);
  for (int w = 0; w < W; ++w) {
    auto ws = std::make_unique<WorkerState>();
    ws->id = w;
    ws->cpu = worker_cpu[w];
    ws->thr = std::thread(worker_main, ws.get());
    workers.push_back(std::move(ws));
  }

  // Supervisor (timeouts + dead worker respawn)
  std::atomic<bool> stop_supervisor{false};
  std::thread supervisor([&] {
    set_thread_name("supervisor");
    auto last_print = Clock::now();

    while (!stop_supervisor.load(std::memory_order_acquire)) {
      // Periodic progress print
      auto now = Clock::now();
      if (std::chrono::duration_cast<std::chrono::seconds>(now - last_print).count() >= 1) {
        std::lock_guard<std::mutex> lk(cout_mu);
        std::cout << datasets_done.load(std::memory_order_acquire) << "/" << D << " datasets processed" << std::endl;
        last_print = now;
      }

      if (completed.load(std::memory_order_acquire) >= (int)jobs.size())
        break;

      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  });

  // Wait for all jobs (done or failed)
  while (completed.load(std::memory_order_acquire) < (int)jobs.size()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  // Shutdown
  stop_supervisor.store(true, std::memory_order_release);
  if (supervisor.joinable())
    supervisor.join();

  for (auto& up : workers) {
    WorkerState* ws = up.get();
    if (ws->thr.joinable()) {
      ws->thr.join();
    }
  }

  return rows;
#endif // __linux__
}

} // namespace rt
} // namespace bspx