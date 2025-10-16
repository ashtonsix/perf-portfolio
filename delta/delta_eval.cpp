#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

// ---- helpers -------------------------------------------------------
inline constexpr int kLabelWidth = 36;
inline constexpr int kValueWidth = 17;
inline constexpr int kTotalReps = 20000;

static std::vector<uint32_t> make_input(size_t n, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<uint32_t> dist(0u, 0xFFFFFFFFu);
  std::vector<uint32_t> v(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = dist(rng);
  return v;
}

template <class T, class MapFn>
void check(std::span<T> in, std::span<T> out, MapFn map_fn) {
  if (in.size() != out.size()) {
    std::cerr << "check(): size mismatch: in=" << in.size() << " out=" << out.size() << "\n";
    std::abort();
  }
  std::vector<T> buf(out.size());
  map_fn(in.data(), buf.data(), out.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    if (!(buf[i] == out[i])) {
      std::cerr << "check(): mismatch at i=" << i;
      if constexpr (std::is_arithmetic_v<T>) {
        std::cerr << " expected=" << out[i] << " got=" << buf[i];
      }
      std::cerr << "\n";

      auto print_first32 = [](const char* label, std::span<T> x) {
        std::cerr << label;
        size_t limit = std::min<size_t>(x.size(), 32);
        for (size_t j = 0; j < limit; ++j) {
          if constexpr (std::is_arithmetic_v<T>) {
            std::cerr << x[j];
          } else {
            std::cerr << "?";
          }
          if (j + 1 < limit)
            std::cerr << " ";
        }
        std::cerr << "\n";
      };

      print_first32("input:    ", in);
      print_first32("f(input): ", std::span<T>(buf.data(), buf.size()));
      print_first32("expected: ", out);

      std::abort();
    }
  }
}

struct Bursty {
  static constexpr size_t kBurstReps = 200;
  static constexpr int kBurstSleepMs = 5;

  template <class F>
  static std::pair<size_t, double> measure(F&& body, size_t total_reps) {
    using clock = std::chrono::steady_clock;
    size_t done = 0;
    double active_sec = 0.0;
    while (done < total_reps) {
      size_t chunk = std::min(kBurstReps, total_reps - done);
      auto t0 = clock::now();
      for (size_t i = 0; i < chunk; ++i)
        body();
      auto t1 = clock::now();
      active_sec += std::chrono::duration<double>(t1 - t0).count();
      done += chunk;
      if (done < total_reps)
        std::this_thread::sleep_for(std::chrono::milliseconds(kBurstSleepMs));
    }
    return {total_reps, active_sec};
  }
};

template <class T, class TestFn>
double measure(std::span<T> in, TestFn test_fn) {
  std::vector<T> out(in.size());

  test_fn(in.data(), out.data(), in.size()); // Warm-up

  auto fn = [&] { test_fn(in.data(), out.data(), in.size()); };
  auto [n, sec] = Bursty::measure(fn, kTotalReps);
  double bytes = static_cast<double>(in.size()) * sizeof(T) * kTotalReps;
  double gbps = bytes / sec / 1e9;
  return gbps;
}

inline void print(const std::string& label, double gbps, double comparison = 0.0) {
  std::cout << "| " << std::left << std::setw(kLabelWidth) << label << " | " << std::left << std::setw(kValueWidth);
  if (gbps == 0.0) {
    std::cout << "-";
  } else {
    std::cout << std::fixed << std::setprecision(2) << gbps;
  }
  std::cout << " | " << std::left << std::setw(kValueWidth);
  if (comparison == 0.0) {
    std::cout << "-";
  } else {
    std::ostringstream s;
    s << std::fixed << std::setprecision(1) << (gbps / comparison) << "x";
    std::cout << s.str();
  }
  std::cout << " |\n";
}

inline void print_decoration(char c) {
  std::cout << "| " << std::string(kLabelWidth, c) << " | " << std::string(kValueWidth, c) << " | "
            << std::string(kValueWidth, c) << " |\n";
}

// ---- external routines -------------------------------------------------------
extern "C" {
void delta_naive_W32(uint32_t* in, uint32_t* out, size_t n);
void prefix_naive_W32(uint32_t* in, uint32_t* out, size_t n);
void prefix_fastpfor_W32(uint32_t* in, uint32_t* out, size_t n);
void prefix_unrolled_W32(uint32_t* in, uint32_t* out, size_t n);
void prefix_pipelined_W32(uint32_t* in, uint32_t* out, size_t n);
void prefix_transpose_W32(uint32_t* in, uint32_t* out, size_t n);
void deltaOfDelta_naive_W32(uint32_t* in, uint32_t* out, size_t n);
void prefixOfPrefix_naive_W32(uint32_t* in, uint32_t* out, size_t n);
void prefixOfPrefix_pipelined_W32(uint32_t* in, uint32_t* out, size_t n);
void xor_naive_W32(uint32_t* in, uint32_t* out, size_t n);
void xorInv_naive_W32(uint32_t* in, uint32_t* out, size_t n);
void xorInv_pipelined_W32(uint32_t* in, uint32_t* out, size_t n);
void xorInv_transpose_W32(uint32_t* in, uint32_t* out, size_t n);
void delta_transpose_W32(uint32_t* in, uint32_t* out, size_t n);
void deltaOfDelta_transpose_W32(uint32_t* in, uint32_t* out, size_t n);
void prefixOfPrefix_transpose_W32(uint32_t* in, uint32_t* out, size_t n);
void xor_transpose_W32(uint32_t* in, uint32_t* out, size_t n);
}

using DeltaFunc = void (*)(uint32_t*, uint32_t*, size_t);
using PrefixFunc = void (*)(uint32_t*, uint32_t*, size_t);

int main() {
  constexpr size_t kWorksetKiB = 4;
  const size_t num_elements = (kWorksetKiB * 1024) / sizeof(uint32_t);

  // Prepare data
  auto input = make_input(num_elements, 0x12345678ULL);
  std::vector<uint32_t> delta_ref(num_elements);
  std::vector<uint32_t> dt_ref(num_elements);
  std::vector<uint32_t> d2_ref(num_elements);
  std::vector<uint32_t> d2t_ref(num_elements);
  std::vector<uint32_t> xor_ref(num_elements);
  std::vector<uint32_t> xt_ref(num_elements);
  delta_naive_W32(input.data(), delta_ref.data(), num_elements);
  delta_transpose_W32(input.data(), dt_ref.data(), num_elements);
  deltaOfDelta_naive_W32(input.data(), d2_ref.data(), num_elements);
  deltaOfDelta_transpose_W32(input.data(), d2t_ref.data(), num_elements);
  xor_naive_W32(input.data(), xor_ref.data(), num_elements);
  xor_transpose_W32(input.data(), xt_ref.data(), num_elements);

  // Correctness checks
  check<uint32_t>(delta_ref, input, prefix_naive_W32);
  check<uint32_t>(delta_ref, input, prefix_fastpfor_W32);
  check<uint32_t>(delta_ref, input, prefix_unrolled_W32);
  check<uint32_t>(delta_ref, input, prefix_pipelined_W32);
  check<uint32_t>(dt_ref, input, prefix_transpose_W32);
  check<uint32_t>(d2_ref, input, prefixOfPrefix_naive_W32);
  check<uint32_t>(d2_ref, input, prefixOfPrefix_pipelined_W32);
  check<uint32_t>(d2t_ref, input, prefixOfPrefix_transpose_W32);
  check<uint32_t>(xor_ref, input, xorInv_naive_W32);
  check<uint32_t>(xor_ref, input, xorInv_pipelined_W32);
  check<uint32_t>(xt_ref, input, xorInv_transpose_W32);

  // Benchmarks
  std::cout << "Delta / Prefix evaluation — " << kWorksetKiB << " KiB, reps=" << kTotalReps << "\n\n";

  std::cout << "| " << std::left << std::setw(kLabelWidth) << "Routine"
            << " | " << std::left << std::setw(kValueWidth) << "Throughput (GB/s)"
            << " | " << std::left << std::setw(kValueWidth) << "vs Naive" << " |\n";
  print_decoration('-');

  print("Delta  / naive", measure<uint32_t>(input, delta_naive_W32));

  double prefix_naive_gbps = measure<uint32_t>(input, prefix_naive_W32);
  print("Prefix / FastPFoR (SIMDe)", measure<uint32_t>(input, prefix_fastpfor_W32), prefix_naive_gbps);
  print("Prefix / naive", prefix_naive_gbps, prefix_naive_gbps);
  print("Prefix / unrolled", measure<uint32_t>(input, prefix_unrolled_W32), prefix_naive_gbps);
  print("Prefix / pipelined", measure<uint32_t>(input, prefix_pipelined_W32), prefix_naive_gbps);
  print("Delta  + 4x4 transpose", measure<uint32_t>(input, delta_transpose_W32));
  print("Prefix + 4x4 transpose", measure<uint32_t>(input, prefix_transpose_W32), prefix_naive_gbps);
  print_decoration(' ');

  double prefixOfPrefix_naive_gbps = measure<uint32_t>(input, prefixOfPrefix_naive_W32);
  print("Delta-of-delta   / naive", measure<uint32_t>(input, deltaOfDelta_naive_W32));
  print("Prefix-of-prefix / naive", prefixOfPrefix_naive_gbps, prefixOfPrefix_naive_gbps);
  print("Prefix-of-prefix / pipelined", measure<uint32_t>(input, prefixOfPrefix_pipelined_W32),
        prefixOfPrefix_naive_gbps);

  print("Delta-of-delta   + 4x4 transpose", measure<uint32_t>(input, deltaOfDelta_transpose_W32));
  print("Prefix-of-prefix + 4x4 transpose", measure<uint32_t>(input, prefixOfPrefix_transpose_W32),
        prefixOfPrefix_naive_gbps);
  print_decoration(' ');

  double xorInv_naive_gbps = measure<uint32_t>(input, xorInv_naive_W32);
  print("XOR  / naive", measure<uint32_t>(input, xor_naive_W32));
  print("XOR' / naive", xorInv_naive_gbps, xorInv_naive_gbps);
  print("XOR' / pipelined", measure<uint32_t>(input, xorInv_pipelined_W32), xorInv_naive_gbps);

  print("XOR  + 4x4 transpose", measure<uint32_t>(input, xor_transpose_W32));
  print("XOR' + 4x4 transpose", measure<uint32_t>(input, xorInv_transpose_W32), xorInv_naive_gbps);

  return 0;
}
