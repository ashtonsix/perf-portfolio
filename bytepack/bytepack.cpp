#include <arm_neon.h>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stddef.h>
#include <stdint.h>

namespace bytepack {
namespace detail {

#define INLINE static inline __attribute__((always_inline))
#define NALIAS __restrict__
#define ASSUME_ALIGNED(ptr, n) (decltype(ptr))__builtin_assume_aligned((ptr), (n))

// Load 32B with robust pair-fusion, while avoiding disruption on
// regalloc, address mode, and linear prefetch.
// Intended lowering:  LDP qA, qB, [base, #offset]
template <size_t offset>
INLINE uint8x16x2_t ld_(const uint8_t* NALIAS ptr) {
  const uint8_t* base = ASSUME_ALIGNED(ptr, 32);
  std::atomic_signal_fence(std::memory_order_acq_rel);
  uint8x16_t a = vld1q_u8(base + offset);
  uint8x16_t b = vld1q_u8(base + offset + 16);
  return uint8x16x2_t{a, b};
}

// Store 32B.
// Intended lowering:  STP qA, qB, [base, #offset]
template <size_t offset>
INLINE void st_(uint8_t* NALIAS ptr, uint8x16x2_t data) {
  uint8_t* base = ASSUME_ALIGNED(ptr, 32);
  std::atomic_signal_fence(std::memory_order_acq_rel);
  vst1q_u8(base + offset, data.val[0]);
  vst1q_u8(base + offset + 16, data.val[1]);
}

// Duplicate scalar mask across lanes of register.
template <int mask>
INLINE uint8x16_t mask_() {
  uint8x16_t m = vdupq_n_u8((uint8_t)(mask));
  asm volatile("" : "+w"(m)); // forces SSA
  return m;
}

// Bitwise insert (BIT): dst := (dst & ~mask) | (src & mask)
// Inline asm ensures no accidental mov-insertion (ACLE/compiler stopgap)
INLINE uint8x16x2_t bit_(uint8x16x2_t dst, const uint8x16x2_t src, const uint8x16_t mask) {
  asm("bit %0.16b, %1.16b, %2.16b" : "+w"(dst.val[0]) : "w"(src.val[0]), "w"(mask));
  asm("bit %0.16b, %1.16b, %2.16b" : "+w"(dst.val[1]) : "w"(src.val[1]), "w"(mask));
  return dst;
}

// Bitwise insert if false (BIF): dst := (dst & mask) | (src & ~mask)
INLINE uint8x16x2_t bif_(uint8x16x2_t dst, const uint8x16x2_t src, const uint8x16_t mask) {
  asm("bif %0.16b, %1.16b, %2.16b" : "+w"(dst.val[0]) : "w"(src.val[0]), "w"(mask));
  asm("bif %0.16b, %1.16b, %2.16b" : "+w"(dst.val[1]) : "w"(src.val[1]), "w"(mask));
  return dst;
}

// Bitwise AND with mask.
INLINE uint8x16x2_t and_(const uint8x16x2_t src, const uint8x16_t mask) {
  uint8x16_t a = vandq_u8(src.val[0], mask);
  uint8x16_t b = vandq_u8(src.val[1], mask);
  return uint8x16x2_t{a, b};
}

// Left shift by compile-time constant.
template <int shift>
INLINE uint8x16x2_t shl_(const uint8x16x2_t src) {
  uint8x16_t a = vshlq_n_u8(src.val[0], shift);
  uint8x16_t b = vshlq_n_u8(src.val[1], shift);
  return uint8x16x2_t{a, b};
}

// Right shift by compile-time constant.
template <int shift>
INLINE uint8x16x2_t shr_(const uint8x16x2_t src) {
  uint8x16_t a = vshrq_n_u8(src.val[0], shift);
  uint8x16_t b = vshrq_n_u8(src.val[1], shift);
  return uint8x16x2_t{a, b};
}

// Shift-left-insert.
template <int shift>
INLINE uint8x16x2_t sli_(uint8x16x2_t dst, const uint8x16x2_t src) {
  uint8x16_t d0 = dst.val[0], d1 = dst.val[1];
  d0 = vsliq_n_u8(d0, src.val[0], shift);
  d1 = vsliq_n_u8(d1, src.val[1], shift);
  return uint8x16x2_t{d0, d1};
}

// Shift-right-insert.
template <int shift>
INLINE uint8x16x2_t sri_(uint8x16x2_t dst, const uint8x16x2_t src) {
  uint8x16_t d0 = dst.val[0], d1 = dst.val[1];
  d0 = vsriq_n_u8(d0, src.val[0], shift);
  d1 = vsriq_n_u8(d1, src.val[1], shift);
  return uint8x16x2_t{d0, d1};
}

#define STRINGIFY_PRAGMA(x) #x
#define APPLY_PRAGMA(directive) _Pragma(STRINGIFY_PRAGMA(directive))

template <int UnrollCount, int InStride, int OutStride, typename Fn>
INLINE void loop_(const uint8_t* NALIAS in, uint8_t* NALIAS out, size_t n_iters, Fn&& fn) {
  in = ASSUME_ALIGNED(in, 64);
  out = ASSUME_ALIGNED(out, 64);
  __builtin_prefetch(in + 512, 0);
  __builtin_prefetch(out + 512, 1);
  APPLY_PRAGMA(clang loop unroll_count(UnrollCount))
  for (; n_iters; --n_iters, in += InStride, out += OutStride) {
    fn(in, out);
  }
}

} // namespace detail

using namespace detail;

template <int K>
void bytepack(const uint8_t* NALIAS in, uint8_t* NALIAS out, size_t n);

template <int K>
void byteunpack(const uint8_t* NALIAS in, uint8_t* NALIAS out, size_t n);

// K=1: 8 inputs -> 1 output, each input contributes 1 bit
template <>
void bytepack<1>(const uint8_t* NALIAS in, uint8_t* NALIAS out, size_t n) {
  auto fn = [&](const uint8_t* NALIAS in, uint8_t* NALIAS out) {
    // Merge tree reduces dependency chain length, 8->3

    auto x0 = ld_<0>(in), x1 = ld_<32>(in);    // ld
    auto h0 = sli_<1>(x0, x1);                 // x1 -> x0 (1b)
    auto x2 = ld_<64>(in), x3 = ld_<96>(in);   // ld
    auto h1 = sli_<1>(x2, x3);                 // x3 -> x2 (1b)
    auto x4 = ld_<128>(in), x5 = ld_<160>(in); // ld
    auto h2 = sli_<1>(x4, x5);                 // x5 -> x4 (1b)
    auto x6 = ld_<192>(in), x7 = ld_<224>(in); // ld
    auto h3 = sli_<1>(x6, x7);                 // x7 -> x6 (1b)

    auto h4 = sli_<2>(h0, h1);    // x2 -> x0 (2b)
    auto h5 = sli_<2>(h2, h3);    // x6 -> x4 (2b)
    st_<0>(out, sli_<4>(h4, h5)); // x4 -> x0 (4b)
  };
  loop_<2, 256, 32>(in, out, n / 256, fn);
}

template <>
void byteunpack<1>(const uint8_t* NALIAS in, uint8_t* NALIAS out, size_t n) {
  const auto m1 = mask_<0b0000'0001>();
  auto fn = [&](const uint8_t* NALIAS in, uint8_t* NALIAS out) {
    const auto y = ld_<0>(in);

    // x[i] = (y >> i)[0]
    st_<0>(out, and_(y, m1));
    st_<32>(out, and_(shr_<1>(y), m1));
    st_<64>(out, and_(shr_<2>(y), m1));
    st_<96>(out, and_(shr_<3>(y), m1));
    st_<128>(out, and_(shr_<4>(y), m1));
    st_<160>(out, and_(shr_<5>(y), m1));
    st_<192>(out, and_(shr_<6>(y), m1));
    st_<224>(out, shr_<7>(y));
  };
  loop_<2, 32, 256>(in, out, n / 256, fn);
}

// K=2: 4 inputs -> 1 output, each input contributes 2 bits
template <>
void bytepack<2>(const uint8_t* NALIAS in, uint8_t* NALIAS out, size_t n) {
  auto fn = [&](const uint8_t* NALIAS in, uint8_t* NALIAS out) {
    // Merge tree reduces dependency chain length, 4->2

    auto x0 = ld_<0>(in), x1 = ld_<32>(in);  // ld
    auto h0 = sli_<2>(x0, x1);               // x1 -> x0 (2b)
    auto x2 = ld_<64>(in), x3 = ld_<96>(in); // ld
    auto h1 = sli_<2>(x2, x3);               // x3 -> x2 (2b)
    st_<0>(out, sli_<4>(h0, h1));            // x2 -> x0 (4b)
  };
  loop_<4, 128, 32>(in, out, n / 128, fn);
}

template <>
void byteunpack<2>(const uint8_t* NALIAS in, uint8_t* NALIAS out, size_t n) {
  const auto m2 = mask_<0b0000'0011>();
  auto fn = [&](const uint8_t* NALIAS in, uint8_t* NALIAS out) {
    const auto y = ld_<0>(in);

    // x[i] = (y >> (2*i))[1:0]
    st_<0>(out, and_(y, m2));
    st_<32>(out, and_(shr_<2>(y), m2));
    st_<64>(out, and_(shr_<4>(y), m2));
    st_<96>(out, shr_<6>(y));
  };
  loop_<4, 32, 128>(in, out, n / 128, fn);
}

// K=3: 8 inputs -> 3 outputs
template <>
void bytepack<3>(const uint8_t* NALIAS in, uint8_t* NALIAS out, size_t n) {
  const auto m7 = mask_<0b0111'1111>();
  const auto m6 = mask_<0b0011'1111>();
  auto fn = [&](const uint8_t* NALIAS in, uint8_t* NALIAS out) {
    // y0 (10210210) : x0[2:0] | x1[2:0] << 3 | x2[1:0] << 6
    auto x0 = ld_<0>(in), x1 = ld_<32>(in), x2 = ld_<64>(in);
    st_<0>(out, sli_<6>(sli_<3>(x0, x1), x2));

    // h25 (22xxxxxx) := x2[2] << 4 | x5[2] << 5
    auto x3 = ld_<96>(in), x4 = ld_<128>(in), x5 = ld_<160>(in);
    auto h25 = bif_(shl_<4>(x2), shl_<5>(x5), m7);

    // y1 (22210210) : x3 + x4 then patch two MSBs from h25
    st_<32>(out, bif_(sli_<3>(x3, x4), h25, m6));

    // y2 (10210210) : x6 + x7 + tail of x5
    auto x6 = ld_<192>(in), x7 = ld_<224>(in);
    st_<64>(out, sli_<6>(sli_<3>(x6, x7), x5));
  };
  loop_<2, 256, 96>(in, out, n / 256, fn);
}

// K = 3 -> 8 outputs
template <>
void byteunpack<3>(const uint8_t* NALIAS in, uint8_t* NALIAS out, size_t n) {
  const auto m3 = mask_<0b0000'0111>();
  auto fn = [&](const uint8_t* NALIAS in, uint8_t* NALIAS out) {
    auto y0 = ld_<0>(in);                             // ld
    st_<0>(out, and_(y0, m3));                        // y0[2:0]
    st_<32>(out, and_(shr_<3>(y0), m3));              // y0[5:3]>>3
    auto y1 = ld_<32>(in);                            // ld
    st_<64>(out, sri_<6>(and_(shr_<4>(y1), m3), y0)); // y1[6]>>4 | y0[7:6]>>6
    st_<96>(out, and_(y1, m3));                       // y1[2:0]
    st_<128>(out, and_(shr_<3>(y1), m3));             // y1[5:3]>>3
    auto y2 = ld_<64>(in);                            // ld
    st_<160>(out, sri_<6>(shr_<5>(y1), y2));          // y1[7]>>5 | y2[7:6]>>6
    st_<192>(out, and_(y2, m3));                      // y2[2:0]
    st_<224>(out, and_(shr_<3>(y2), m3));             // y2[5:3]>>3
  };
  loop_<2, 96, 256>(in, out, n / 256, fn);
}

// K=4: 2 inputs -> 1 output, each input contributes 4 bits
template <>
void bytepack<4>(const uint8_t* NALIAS in, uint8_t* NALIAS out, size_t n) {
  auto fn = [&](const uint8_t* NALIAS in, uint8_t* NALIAS out) {
    auto x0 = ld_<0>(in), x1 = ld_<32>(in);
    st_<0>(out, sli_<4>(x0, x1)); // y = x0 | (x1 << 4)
  };
  loop_<8, 64, 32>(in, out, n / 64, fn);
}

template <>
void byteunpack<4>(const uint8_t* NALIAS in, uint8_t* NALIAS out, size_t n) {
  const auto m4 = mask_<0b0000'1111>();
  auto fn = [&](const uint8_t* NALIAS in, uint8_t* NALIAS out) {
    auto y = ld_<0>(in);
    st_<0>(out, and_(y, m4));
    st_<32>(out, shr_<4>(y));
  };
  loop_<8, 32, 64>(in, out, n / 64, fn);
}

// K=5: 8 inputs -> 5 outputs  (pattern (5,3)×4 + packed LSBs)
template <>
void bytepack<5>(const uint8_t* NALIAS in, uint8_t* NALIAS out, size_t n) {
  const auto m5 = mask_<0b0001'1111>();
  auto fn = [&](const uint8_t* NALIAS in, uint8_t* NALIAS out) {
    // Even keeps 5, odd donates 3 MSBs
    auto x0 = ld_<0>(in), x1 = ld_<32>(in);
    st_<0>(out, bif_(x0, shl_<3>(x1), m5));
    auto x2 = ld_<64>(in), x3 = ld_<96>(in);
    st_<32>(out, bif_(x2, shl_<3>(x3), m5));
    auto x4 = ld_<128>(in), x5 = ld_<160>(in);
    st_<64>(out, bif_(x4, shl_<3>(x5), m5));
    auto x6 = ld_<192>(in), x7 = ld_<224>(in);
    st_<96>(out, bif_(x6, shl_<3>(x7), m5));

    // Collect the leftover 2 LSBs from each odd input into y4
    auto h0 = sli_<2>(x1, x3);
    auto h1 = sli_<2>(x5, x7);
    st_<128>(out, sli_<4>(h0, h1));
  };
  loop_<1, 256, 160>(in, out, n / 256, fn);
}

template <>
void byteunpack<5>(const uint8_t* NALIAS in, uint8_t* NALIAS out, size_t n) {
  const auto m5 = mask_<0b0001'1111>();
  const auto m2 = mask_<0b0000'0011>();
  auto fn = [&](const uint8_t* NALIAS in, uint8_t* NALIAS out) {
    // Even needs mask only, odd extracts 3 MSBs from y[i/2] & 2 LSBs from y[4]
    auto y0 = ld_<0>(in);
    st_<0>(out, and_(y0, m5));
    auto y1 = ld_<32>(in), y2 = ld_<64>(in), y3 = ld_<96>(in), y4 = ld_<128>(in);
    st_<32>(out, bit_(shr_<3>(y0), y4, m2));
    st_<64>(out, and_(y1, m5));
    st_<96>(out, bit_(shr_<3>(y1), shr_<2>(y4), m2));
    st_<128>(out, and_(y2, m5));
    st_<160>(out, bit_(shr_<3>(y2), shr_<4>(y4), m2));
    st_<192>(out, and_(y3, m5));
    st_<224>(out, sri_<6>(shr_<3>(y3), y4));
  };
  loop_<2, 160, 256>(in, out, n / 256, fn);
}

// K=6: 4 inputs -> 3 outputs  (6,2) / (4,4) / (6,2)
template <>
void bytepack<6>(const uint8_t* NALIAS in, uint8_t* NALIAS out, size_t n) {
  const auto m6 = mask_<0b0011'1111>();
  auto fn = [&](const uint8_t* NALIAS in, uint8_t* NALIAS out) {
    auto x0 = ld_<0>(in), x1 = ld_<32>(in); // ld
    x0 = bif_(x0, shl_<2>(x1), m6);         // y0 : x0 + x1 top‑2
    st_<0>(out, x0);                        // st
    auto x2 = ld_<64>(in);                  // ld
    st_<32>(out, sli_<4>(x1, x2));          // y1 : x1 & x2 nybbles
    auto x3 = ld_<96>(in);                  // ld
    x3 = bif_(x3, shl_<2>(x2), m6);         // y2 : x3 + x2 top‑2
    st_<64>(out, x3);                       // st
  };
  loop_<4, 128, 96>(in, out, n / 128, fn);
}

// K = 6 -> 4 outputs
template <>
void byteunpack<6>(const uint8_t* NALIAS in, uint8_t* NALIAS out, size_t n) {
  const auto m6 = mask_<0b0011'1111>();
  const auto m4 = mask_<0b0000'1111>();
  auto fn = [&](const uint8_t* NALIAS in, uint8_t* NALIAS out) {
    auto y0 = ld_<0>(in);                    // ld
    st_<0>(out, and_(y0, m6));               // y0[5:0]
    auto y1 = ld_<32>(in);                   // ld
    st_<32>(out, bit_(shr_<2>(y0), y1, m4)); // y0[7:6]>>2 | y1[3:0]
    auto y2 = ld_<64>(in);                   // ld
    st_<64>(out, sri_<4>(shr_<2>(y2), y1));  // y2[7:6]>>2 | y1[7:4]>>4
    st_<96>(out, and_(y2, m6));              // y2[5:0]
  };
  loop_<4, 96, 128>(in, out, n / 128, fn);
}

// K=7: 8 inputs -> 7 outputs,
//      pattern:  (7,1) / (6,2) / (7,1) / (4,4) / (7,1) / (6,2) / (7,1)
template <>
void bytepack<7>(const uint8_t* NALIAS in, uint8_t* NALIAS out, size_t n) {
  const auto m7 = mask_<0b0111'1111>();
  const auto m6 = mask_<0b0011'1111>();
  auto fn = [&](const uint8_t* NALIAS in, uint8_t* NALIAS out) {
    auto x0 = ld_<0>(in), x1 = ld_<32>(in);                       // ld
    x0 = bif_(x0, shl_<1>(x1), m7);                               // x0[6:0] | x1[6:6]<<1
    st_<0>(out, x0);                                              // st
    auto x2 = ld_<64>(in);                                        // ld
    x1 = bif_(x1, shl_<2>(x2), m6);                               // x1[5:0] | x2[5:4]<<2
    st_<32>(out, x1);                                             // st
    auto x3 = ld_<96>(in);                                        // ld
    x3 = bif_(x3, shl_<1>(x2), m7);                               // x3[6:0] | x2[6:6]<<1
    st_<64>(out, x3);                                             // st
    auto x4 = ld_<128>(in), x5 = ld_<160>(in), x6 = ld_<192>(in); // ld
    st_<96>(out, sli_<4>(x2, x6));                                // x2[3:0] | x6[3:0]<<4
    x4 = bif_(x4, shl_<1>(x5), m7);                               // x4[6:0] | x5[6:6]<<1
    st_<128>(out, x4);                                            // st
    x5 = bif_(x5, shl_<2>(x6), m6);                               // x5[5:0] | x6[5:4]<<2
    st_<160>(out, x5);                                            // st
    auto x7 = ld_<224>(in);                                       // ld
    x7 = bif_(x7, shl_<1>(x6), m7);                               // x7[6:0] | x6[6:6]<<1
    st_<192>(out, x7);                                            // st
  };
  loop_<2, 256, 224>(in, out, n / 256, fn);
}

// K = 7 -> 8 outputs
template <>
void byteunpack<7>(const uint8_t* NALIAS in, uint8_t* NALIAS out, size_t n) {
  const auto m7 = mask_<0b0111'1111>();
  const auto m6 = mask_<0b0011'1111>();
  const auto m4 = mask_<0b0000'1111>();
  auto fn = [&](const uint8_t* NALIAS in, uint8_t* NALIAS out) {
    // Block order minimises register pressure (only y3 requires a long lifetime)

    auto y0 = ld_<0>(in);                                 // ld
    st_<0>(out, and_(y0, m7));                            // y0[6:0]
    auto y1 = ld_<32>(in);                                // ld
    st_<32>(out, bit_(shr_<1>(y0), y1, m6));              // y0[7]>>1 | y1[5:0]
    auto y2 = ld_<64>(in), y3 = ld_<96>(in);              // ld
    st_<64>(out, bit_(sri_<2>(shr_<1>(y2), y1), y3, m4)); // y2[7]>>1 | y1[7:6]>>2 | y3[3:0]
    st_<96>(out, and_(y2, m7));                           // y2[6:0]
    auto y4 = ld_<128>(in);                               // ld
    st_<128>(out, and_(y4, m7));                          // y4[6:0]
    auto y5 = ld_<160>(in);                               // ld
    st_<160>(out, bit_(shr_<1>(y4), y5, m6));             // y4[7]>>1 | y5[5:0]
    auto y6 = ld_<192>(in);                               // ld
    st_<192>(out, sri_<4>(sri_<2>(shr_<1>(y6), y5), y3)); // y6[7]>>1 | y5[7:6]>>2 | y3[7:4]>>4
    st_<224>(out, and_(y6, m7));                          // y6[6:0]
  };
  loop_<2, 224, 256>(in, out, n / 256, fn);
}

// K = 8 -> just copy
template <>
void bytepack<8>(const uint8_t* NALIAS in, uint8_t* NALIAS out, size_t n) {
  std::memcpy(out, in, n);
}

template <>
void byteunpack<8>(const uint8_t* NALIAS in, uint8_t* NALIAS out, size_t n) {
  std::memcpy(out, in, n);
}

} // namespace bytepack
