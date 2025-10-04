// bytepack_eval.cpp

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sched.h>
#include <thread>
#include <unistd.h>

// ---- external kernels -------------------------------------------------------
namespace bytepack {
template <int K>
void bytepack(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, size_t n);
template <int K>
void byteunpack(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, size_t n);
} // namespace bytepack

namespace BytepackBaseline {
void encodeArray(const uint32_t* in, size_t length_uint32, uint32_t* out, uint32_t bit);
const uint32_t* decodeArray(const uint32_t* in, size_t length_uint32, uint32_t* out, uint32_t bit);
} // namespace BytepackBaseline

// ---- affinity ---------------------------------------------------------------
static inline void pin_to_cpu0() {
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(0, &set);
  (void)sched_setaffinity(0, sizeof(set), &set);
#endif
}

// ---- small helpers ----------------------------------------------------------
template <int K>
constexpr size_t packed_size_bytes(size_t n_in_bytes) {
  return (n_in_bytes * static_cast<size_t>(K)) / 8;
}
static inline std::string as_gbps(double bytes_per_sec) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(2) << (bytes_per_sec / 1e9);
  return os.str();
}

// Simple aligned unique_ptr helper
template <class T>
using aligned_uptr = std::unique_ptr<T, void (*)(void*)>;

template <class T>
static aligned_uptr<T> make_aligned(size_t count, size_t align = 64) {
  void* p = nullptr;
  if (posix_memalign(&p, align, count * sizeof(T))) {
    std::perror("posix_memalign");
    std::exit(1);
  }
  return aligned_uptr<T>(static_cast<T*>(p), std::free);
}

// ---- Bursty timer: run in small active bursts with sleeps in between --------
struct Bursty {
  static constexpr size_t kBurstReps = 200; // active iterations per burst
  static constexpr int kBurstSleepMs = 5;   // idle between bursts

  template <class F>
  static std::pair<size_t, double> measure(F&& body, size_t total_reps) {
    using clock = std::chrono::steady_clock;
    size_t done = 0;
    double active_sec = 0.0;
    while (done < total_reps) {
      const size_t chunk = std::min(kBurstReps, total_reps - done);
      const auto t0 = clock::now();
      for (size_t i = 0; i < chunk; ++i)
        body();
      const auto t1 = clock::now();
      active_sec += std::chrono::duration<double>(t1 - t0).count();
      done += chunk;
      if (done < total_reps)
        std::this_thread::sleep_for(std::chrono::milliseconds(kBurstSleepMs));
    }
    return {total_reps, active_sec}; // only active time counts
  }
};

// ---- Benchmark for a fixed K ------------------------------------------------
template <int K>
struct BenchK {
  static void run(size_t workset_kib, size_t total_reps) {
    static_assert(K >= 1 && K <= 8, "K out of range");

    const size_t n_bytes = workset_kib * 1024;
    const size_t n_u32 = n_bytes / 4;

    const size_t pk_bytes = packed_size_bytes<K>(n_bytes);
    const size_t pk32_words = (n_u32 * K + 31) / 32; // round up

    // Allocate buffers
    auto in_b = make_aligned<uint8_t>(n_bytes);
    auto pk_b = make_aligned<uint8_t>(pk_bytes);
    auto out_b = make_aligned<uint8_t>(n_bytes);

    auto in32 = make_aligned<uint32_t>(n_u32);
    auto pk32 = make_aligned<uint32_t>(pk32_words);
    auto out32 = make_aligned<uint32_t>(n_u32);

    // Initialize inputs (deterministic)
    std::mt19937_64 rng(0x12345678ULL + K);
    const uint32_t maxv = (K == 8) ? 0xFFu : ((1u << K) - 1u);
    std::uniform_int_distribution<uint32_t> dist(0u, maxv);
    for (size_t i = 0; i < n_bytes; ++i)
      in_b.get()[i] = static_cast<uint8_t>(dist(rng));
    for (size_t i = 0; i < n_u32; ++i)
      in32.get()[i] = static_cast<uint32_t>(dist(rng));

    // Quick functional validation (one round-trip each path)
    bytepack::bytepack<K>(in_b.get(), pk_b.get(), n_bytes);
    bytepack::byteunpack<K>(pk_b.get(), out_b.get(), n_bytes);

    BytepackBaseline::encodeArray(in32.get(), n_u32, pk32.get(), K);
    BytepackBaseline::decodeArray(pk32.get(), n_u32, out32.get(), K);

    if (std::memcmp(in_b.get(), out_b.get(), n_bytes) != 0) {
      std::cerr << "NEON round-trip mismatch (K=" << K << ")\n";
      std::exit(2);
    }
    if (std::memcmp(in32.get(), out32.get(), n_u32 * sizeof(uint32_t)) != 0) {
      std::cerr << "Baseline round-trip mismatch (K=" << K << ")\n";
      std::exit(2);
    }

    // Measurements (bursty; only active time is counted)
    auto [iters_np, secs_np] =
        Bursty::measure([&] { bytepack::bytepack<K>(in_b.get(), pk_b.get(), n_bytes); }, total_reps);
    auto [iters_nu, secs_nu] =
        Bursty::measure([&] { bytepack::byteunpack<K>(pk_b.get(), out_b.get(), n_bytes); }, total_reps);
    auto [iters_bp, secs_bp] =
        Bursty::measure([&] { BytepackBaseline::encodeArray(in32.get(), n_u32, pk32.get(), K); }, total_reps);
    auto [iters_bu, secs_bu] =
        Bursty::measure([&] { BytepackBaseline::decodeArray(pk32.get(), n_u32, out32.get(), K); }, total_reps);

    const double neon_pack_in_bps = (double(n_bytes) * iters_np) / secs_np;
    const double neon_unpack_out_bps = (double(n_bytes) * iters_nu) / secs_nu;
    const double base_pack_in_bps = (double(n_u32 * sizeof(uint32_t)) * iters_bp) / secs_bp;
    const double base_unpack_out_bps = (double(n_u32 * sizeof(uint32_t)) * iters_bu) / secs_bu;

    // Output (same table format)
    std::cout << std::left << std::setw(1) << K << "  " << std::setw(11) << as_gbps(neon_pack_in_bps) << " "
              << std::setw(11) << as_gbps(neon_unpack_out_bps) << "  " << std::setw(15) << as_gbps(base_pack_in_bps)
              << " " << std::setw(15) << as_gbps(base_unpack_out_bps) << "\n";
  }
};

// ---- main -------------------------------------------------------------------
int main() {
  pin_to_cpu0();

  constexpr size_t kWorksetKiB = 16; // 16 KiB
  constexpr size_t kTotalReps = 20000;

  std::cout << "Bytepack Bench — " << kWorksetKiB << " KiB, reps=" << kTotalReps << " (pinned if available)\n";
  std::cout << "Throughput GB/s\n\n";
  std::cout << std::left << std::setw(1) << "K" << "  " << std::setw(11) << "NEON pack" << " " << std::setw(11)
            << "NEON unpack" << "  " << std::setw(15) << "Baseline pack" << " " << std::setw(15) << "Baseline unpack"
            << "\n";

  BenchK<1>::run(kWorksetKiB, kTotalReps);
  BenchK<2>::run(kWorksetKiB, kTotalReps);
  BenchK<3>::run(kWorksetKiB, kTotalReps);
  BenchK<4>::run(kWorksetKiB, kTotalReps);
  BenchK<5>::run(kWorksetKiB, kTotalReps);
  BenchK<6>::run(kWorksetKiB, kTotalReps);
  BenchK<7>::run(kWorksetKiB, kTotalReps);
  BenchK<8>::run(kWorksetKiB, kTotalReps);

  return 0;
}
