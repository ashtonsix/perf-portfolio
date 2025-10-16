#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <new>
#include <random>
#include <sched.h>
#include <thread>
#include <type_traits>
#include <utility>

static inline void pin_to_cpu0() {
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(0, &set);
  (void)sched_setaffinity(0, sizeof(set), &set);
#endif
}

// Allocates over-aligned buffer and optionally fills with random bits
template <class T>
std::unique_ptr<T[]> make_buf(std::size_t count, std::size_t align = 64, std::uint64_t seed = 0xDEADBEEFCAFEBABEull) {
  static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
  if (count == 0)
    return {};

  if ((align & (align - 1)) != 0)
    throw std::invalid_argument("align must be power of two");
  align = std::max<std::size_t>(align, alignof(T));

  T* p = ::new (std::align_val_t{align}) T[count];
  std::unique_ptr<T[]> ptr(p);

  if (seed == 0)
    return ptr;

  std::mt19937_64 rng(seed);
  std::size_t bytes = count * sizeof(T);
  unsigned char* dst = reinterpret_cast<unsigned char*>(p);

  while (bytes >= 8) {
    const std::uint64_t x = rng();
    std::memcpy(dst, &x, 8);
    dst += 8;
    bytes -= 8;
  }
  if (bytes) {
    const std::uint64_t x = rng();
    std::memcpy(dst, &x, bytes);
  }

  return ptr;
}

struct Timer {
  static constexpr long long kActiveNs = 0'500'000; // target active time per cycle, fits in IRQ time slice
  static constexpr long long kSleepNs = 2'000'000;  // sleep between cycles
  static constexpr int kWarmupIters = 8;            // fixed number of warm-up iterations

  template <class F>
  static std::pair<std::size_t, double> measure(F&& body, std::size_t total_reps, std::size_t reps_per_cycle = 0) {
    using clock = std::chrono::steady_clock;

    if (total_reps == 0)
      return {0, 0.0};

    // --- Warm-up: repeated linear extrapolation to decide reps_per_cycle ---
    // After calibration, pass reps_per_cycle as a parameter to improve consistency between runs.
    if (reps_per_cycle == 0) {
      reps_per_cycle = 64;
      for (int it = 0; it < kWarmupIters; ++it) {
        const auto t0 = clock::now();
        for (std::size_t i = 0; i < reps_per_cycle; ++i)
          body();
        const auto t1 = clock::now();

        const long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        if (ns <= 0) {
          reps_per_cycle = std::max<std::size_t>(reps_per_cycle * 10, reps_per_cycle + 1); // coarse fallback
        } else {
          const double scale = double(kActiveNs) / double(ns);
          const std::size_t next = static_cast<std::size_t>(std::ceil(double(reps_per_cycle) * scale));
          reps_per_cycle = std::max<std::size_t>(1, next);
        }
      }
      std::cout << "Time::measure: using " << reps_per_cycle << " reps per cycle\n";
    }

    // --- Measurement: fixed reps-per-cycle, count only active time ---
    std::size_t done = 0;
    double active_sec = 0.0;

    while (done < total_reps) {
      const std::size_t chunk = std::min(reps_per_cycle, total_reps - done);

      const auto t0 = clock::now();
      for (std::size_t i = 0; i < chunk; ++i)
        body();
      const auto t1 = clock::now();

      active_sec += std::chrono::duration<double>(t1 - t0).count();
      done += chunk;

      if (done < total_reps)
        std::this_thread::sleep_for(std::chrono::nanoseconds(kSleepNs));
    }

    return {total_reps, active_sec}; // only active time counts
  }
};
