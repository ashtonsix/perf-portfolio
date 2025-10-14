#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef SIMDE_ENABLE_NATIVE_ALIASES
#define SIMDE_ENABLE_NATIVE_ALIASES
#endif

#ifdef __SSE4_1__
#include <smmintrin.h>
#else
#include <simde/x86/sse4.1.h>
#endif

void delta_naive_W32(const uint32_t* in, uint32_t* out, const size_t n) {
  if (n == 0)
    return;
  out[0] = in[0];
  for (size_t i = 1; i < n; ++i) {
    out[i] = in[i] - in[i - 1];
  }
}

// SIMD baseline originally authored by Daniel Lemire, modified by Ashton Six
// Based on: https://github.com/fast-pack/FastPFOR/blob/master/headers/deltautil.h (fastinverseDelta2)
void prefix_fastpfor_W32(uint32_t* in, uint32_t* out, const size_t n) {
  const size_t n4 = n / 4;
  __m128i acc = _mm_setzero_si128();
  __m128i* pi = (__m128i*)in;
  __m128i* po = (__m128i*)out;
  const __m128i* end = pi + n4;
  while (pi < end) {
    __m128i a0 = _mm_loadu_si128(pi++);
    __m128i a1 = _mm_add_epi32(_mm_slli_si128(a0, 8), a0);
    __m128i a2 = _mm_add_epi32(_mm_slli_si128(a1, 4), a1);
    a0 = _mm_add_epi32(a2, acc);
    acc = _mm_shuffle_epi32(a0, 0xFF);
    _mm_storeu_si128(po++, a0);
  }
}

void prefix_naive_W32(const uint32_t* in, uint32_t* out, const size_t n) {
  if (n == 0)
    return;
  out[0] = in[0];
  for (size_t i = 1; i < n; ++i) {
    out[i] = out[i - 1] + in[i];
  }
}

void prefix_unrolled_W32(const uint32_t* in, uint32_t* out, size_t n) {
  const uint32_t* pi = (const uint32_t*)__builtin_assume_aligned(in, 64);
  uint32_t* po = (uint32_t*)__builtin_assume_aligned(out, 64);
  __builtin_prefetch(in + 512, 0);
  __builtin_prefetch(out + 512, 1);

  size_t iters = n / 16;
  uint32x4_t acc = vdupq_n_u32(0);
  const uint32x4_t z = vdupq_n_u32(0);

  for (; iters; --iters, pi += 16, po += 16) {
    uint32x4_t l0 = vld1q_u32(pi);
    uint32x4_t l1 = vld1q_u32(pi + 4);
    uint32x4_t l2 = vld1q_u32(pi + 8);
    uint32x4_t l3 = vld1q_u32(pi + 12);

    // Hillis-Steele scan within each vector of 4 elements
    uint32x4_t s0 = vaddq_u32(l0, vextq_u32(z, l0, 3));
    uint32x4_t s1 = vaddq_u32(l1, vextq_u32(z, l1, 3));
    uint32x4_t s2 = vaddq_u32(l2, vextq_u32(z, l2, 3));
    uint32x4_t s3 = vaddq_u32(l3, vextq_u32(z, l3, 3));
    s0 = vaddq_u32(s0, vextq_u32(z, s0, 2));
    s1 = vaddq_u32(s1, vextq_u32(z, s1, 2));
    s2 = vaddq_u32(s2, vextq_u32(z, s2, 2));
    s3 = vaddq_u32(s3, vextq_u32(z, s3, 2));

    // Carry across vectors
    s0 = vaddq_u32(s0, acc);
    s1 = vaddq_u32(s1, vdupq_laneq_u32(s0, 3));
    s2 = vaddq_u32(s2, vdupq_laneq_u32(s1, 3));
    s3 = vaddq_u32(s3, vdupq_laneq_u32(s2, 3));

    acc = vdupq_laneq_u32(s3, 3);

    vst1q_u32(po, s0);
    vst1q_u32(po + 4, s1);
    vst1q_u32(po + 8, s2);
    vst1q_u32(po + 12, s3);
  }
}

void prefix_pipelined_W32(const uint32_t* in, uint32_t* out, size_t n) {
  const uint32_t* pi = (const uint32_t*)__builtin_assume_aligned(in, 64);
  uint32_t* po = (uint32_t*)__builtin_assume_aligned(out, 64);
  __builtin_prefetch(in + 512, 0);
  __builtin_prefetch(out + 512, 1);

  size_t iters = n / 16;
  uint32x4_t acc = vdupq_n_u32(0);
  const uint32x4_t z = vdupq_n_u32(0);

  for (; iters; --iters, pi += 16, po += 16) {
    uint32x4_t l0 = vld1q_u32(pi);
    uint32x4_t l1 = vld1q_u32(pi + 4);
    uint32x4_t l2 = vld1q_u32(pi + 8);
    uint32x4_t l3 = vld1q_u32(pi + 12);

    // Hillis-Steele scan within each vector of 4 elements
    uint32x4_t s0 = vaddq_u32(l0, vextq_u32(z, l0, 3));
    uint32x4_t s1 = vaddq_u32(l1, vextq_u32(z, l1, 3));
    uint32x4_t s2 = vaddq_u32(l2, vextq_u32(z, l2, 3));
    uint32x4_t s3 = vaddq_u32(l3, vextq_u32(z, l3, 3));
    s0 = vaddq_u32(s0, vextq_u32(z, s0, 2));
    s1 = vaddq_u32(s1, vextq_u32(z, s1, 2));
    s2 = vaddq_u32(s2, vextq_u32(z, s2, 2));
    s3 = vaddq_u32(s3, vextq_u32(z, s3, 2));

    // Carry across vectors
    s1 = vaddq_u32(s1, vdupq_laneq_u32(s0, 3));
    s3 = vaddq_u32(s3, vdupq_laneq_u32(s2, 3));

    // Late accumulator application
    uint32x4_t ac2 = vaddq_u32(acc, vdupq_laneq_u32(s1, 3));
    s0 = vaddq_u32(s0, acc);
    s1 = vaddq_u32(s1, acc);
    s2 = vaddq_u32(s2, ac2);
    s3 = vaddq_u32(s3, ac2);

    acc = vdupq_laneq_u32(s3, 3);

    vst1q_u32(po, s0);
    vst1q_u32(po + 4, s1);
    vst1q_u32(po + 8, s2);
    vst1q_u32(po + 12, s3);
  }
}

void delta_transpose_W32(const uint32_t* in, uint32_t* out, const size_t n) {
  assert(n % 16 == 0);
  if (n == 0)
    return;

  for (size_t base = 0; base < n; base += 16) {
    // Delta
    uint32_t d[16];
    if (base == 0) {
      d[0] = in[0];
      for (int j = 1; j < 16; ++j) {
        d[j] = in[j] - in[j - 1];
      }
    } else {
      for (int j = 0; j < 16; ++j) {
        d[j] = in[base + j] - in[base + j - 1];
      }
    }

    // Transpose
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        out[base + r * 4 + c] = d[c * 4 + r];
      }
    }
  }
}

void prefix_transpose_W32(const uint32_t* in, uint32_t* out, size_t n) {
  const uint32_t* pi = (const uint32_t*)__builtin_assume_aligned(in, 64);
  uint32_t* po = (uint32_t*)__builtin_assume_aligned(out, 64);
  int iters = n / 16;
  uint32x4_t acc = vdupq_n_u32(0);
  __builtin_prefetch(pi + 512, 0);
  __builtin_prefetch(po + 512, 1);

  // "AB" denotes the sum of values at indices A..B, in hexadecimal.
  // Each range corresponds to one element in the vector.
  for (; iters; --iters, pi += 16, po += 16) {
    uint32x4_t l00 = vld1q_u32(pi);      // 00 44 88 cc
    uint32x4_t l11 = vld1q_u32(pi + 4);  // 11 55 99 dd
    uint32x4_t l22 = vld1q_u32(pi + 8);  // 22 66 aa ee
    uint32x4_t l33 = vld1q_u32(pi + 12); // 33 77 bb ff

    // Scan within each vector and transpose simultaneously
    uint32x4_t l01 = vaddq_u32(l00, l11); // 01 45 89 cd
    uint32x4_t l23 = vaddq_u32(l22, l33); // 23 67 ab ef
    uint32x4_t l02 = vaddq_u32(l01, l22); // 02 46 8a ce
    uint32x4_t l03 = vaddq_u32(l01, l23); // 03 47 8b cf
    uint32x4_t a0 = vtrn1q_u32(l00, l01); // 00 01 88 89
    uint32x4_t a1 = vtrn2q_u32(l00, l01); // 44 45 cc cd
    uint32x4_t a2 = vtrn1q_u32(l02, l03); // 02 03 8a 8b
    uint32x4_t a3 = vtrn2q_u32(l02, l03); // 46 47 ce cf
    uint32x4_t s0 = vzip1q_u64(a0, a2);   // 00 01 02 03
    uint32x4_t s1 = vzip1q_u64(a1, a3);   // 44 45 46 47
    uint32x4_t s2 = vzip2q_u64(a0, a2);   // 88 89 8a 8b
    uint32x4_t s3 = vzip2q_u64(a1, a3);   // cc cd ce cf

    // Carry across vectors
    s1 = vaddq_u32(s1, vdupq_laneq_u32(s0, 3));
    s3 = vaddq_u32(s3, vdupq_laneq_u32(s2, 3));

    // Late accumulator application
    uint32x4_t ac2 = vaddq_u32(acc, vdupq_laneq_u32(s1, 3));
    s0 = vaddq_u32(s0, acc);
    s1 = vaddq_u32(s1, acc);
    s2 = vaddq_u32(s2, ac2);
    s3 = vaddq_u32(s3, ac2);

    acc = vdupq_laneq_u32(s3, 3);

    vst1q_u32(po, s0);
    vst1q_u32(po + 4, s1);
    vst1q_u32(po + 8, s2);
    vst1q_u32(po + 12, s3);
  }
}

void deltaOfDelta_naive_W32(const uint32_t* in, uint32_t* out, const size_t n) {
  if (n < 2) {
    if (n == 1)
      out[0] = in[0];
    return;
  }
  out[0] = in[0];
  out[1] = in[1] - in[0];
  for (size_t i = 2; i < n; ++i)
    out[i] = in[i] - 2 * in[i - 1] + in[i - 2];
}

void prefixOfPrefix_naive_W32(const uint32_t* in, uint32_t* out, const size_t n) {
  if (n < 2) {
    if (n == 1)
      out[0] = in[0];
    return;
  }
  out[0] = in[0];
  out[1] = in[1] + out[0];
  for (size_t i = 2; i < n; ++i)
    out[i] = in[i] + 2 * out[i - 1] - out[i - 2];
}

void prefixOfPrefix_pipelined_W32(const uint32_t* in, uint32_t* out, const size_t n) {
  // Process 16 elements in prelude so 64B alignment holds in main loop body
  if (n < 2) {
    if (n == 1)
      out[0] = in[0];
    return;
  }
  out[0] = in[0];
  out[1] = in[1] + out[0];
  const size_t prelude_end = (n < 16) ? n : 16;
  for (size_t i = 2; i < prelude_end; ++i)
    out[i] = in[i] + 2 * out[i - 1] - out[i - 2];
  if (n <= 16)
    return;
  uint32_t B_acc = out[15] - out[14];
  uint32x4_t Y_acc = vdupq_n_u32(out[15]);

  const uint32_t* pi = (const uint32_t*)__builtin_assume_aligned(in + 16, 64);
  uint32_t* po = (uint32_t*)__builtin_assume_aligned(out + 16, 64);
  __builtin_prefetch(pi + 512, 0);
  __builtin_prefetch(po + 512, 1);

  // Ramps for block-local prefix-of-prefix scan
  const uint32x4_t r0 = (uint32x4_t){1u, 2u, 3u, 4u};
  const uint32x4_t r1 = (uint32x4_t){5u, 6u, 7u, 8u};
  const uint32x4_t r2 = (uint32x4_t){9u, 10u, 11u, 12u};
  const uint32x4_t r3 = (uint32x4_t){13u, 14u, 15u, 16u};
  const uint32x4_t z = vdupq_n_u32(0);

  size_t iters = (n - 16) / 16;
  for (; iters; --iters, pi += 16, po += 16) {
    uint32x4_t l0 = vld1q_u32(pi + 0);
    uint32x4_t l1 = vld1q_u32(pi + 4);
    uint32x4_t l2 = vld1q_u32(pi + 8);
    uint32x4_t l3 = vld1q_u32(pi + 12);

    // --- First prefix (local) ---
    uint32x4_t p0 = vaddq_u32(l0, vextq_u32(z, l0, 3));
    uint32x4_t p1 = vaddq_u32(l1, vextq_u32(z, l1, 3));
    uint32x4_t p2 = vaddq_u32(l2, vextq_u32(z, l2, 3));
    uint32x4_t p3 = vaddq_u32(l3, vextq_u32(z, l3, 3));
    p0 = vaddq_u32(p0, vextq_u32(z, p0, 2));
    p1 = vaddq_u32(p1, vextq_u32(z, p1, 2));
    p2 = vaddq_u32(p2, vextq_u32(z, p2, 2));
    p3 = vaddq_u32(p3, vextq_u32(z, p3, 2));
    // carry across vectors
    p1 = vaddq_u32(p1, vdupq_laneq_u32(p0, 3));
    p2 = vaddq_u32(p2, vdupq_laneq_u32(p1, 3));
    p3 = vaddq_u32(p3, vdupq_laneq_u32(p2, 3));

    // --- Second prefix (local) ---
    uint32x4_t s0 = vaddq_u32(p0, vextq_u32(z, p0, 3));
    uint32x4_t s1 = vaddq_u32(p1, vextq_u32(z, p1, 3));
    uint32x4_t s2 = vaddq_u32(p2, vextq_u32(z, p2, 3));
    uint32x4_t s3 = vaddq_u32(p3, vextq_u32(z, p3, 3));
    s0 = vaddq_u32(s0, vextq_u32(z, s0, 2));
    s1 = vaddq_u32(s1, vextq_u32(z, s1, 2));
    s2 = vaddq_u32(s2, vextq_u32(z, s2, 2));
    s3 = vaddq_u32(s3, vextq_u32(z, s3, 2));
    // carry across vectors
    s1 = vaddq_u32(s1, vdupq_laneq_u32(s0, 3));
    s2 = vaddq_u32(s2, vdupq_laneq_u32(s1, 3));
    s3 = vaddq_u32(s3, vdupq_laneq_u32(s2, 3));

    // --- Late accumulator application ---
    s0 = vaddq_u32(s0, Y_acc);
    s1 = vaddq_u32(s1, Y_acc);
    s2 = vaddq_u32(s2, Y_acc);
    s3 = vaddq_u32(s3, Y_acc);
    s0 = vmlaq_n_u32(s0, r0, B_acc);
    s1 = vmlaq_n_u32(s1, r1, B_acc);
    s2 = vmlaq_n_u32(s2, r2, B_acc);
    s3 = vmlaq_n_u32(s3, r3, B_acc);

    vst1q_u32(po + 0, s0);
    vst1q_u32(po + 4, s1);
    vst1q_u32(po + 8, s2);
    vst1q_u32(po + 12, s3);

    B_acc += vgetq_lane_u32(p3, 3);
    Y_acc = vdupq_laneq_u32(s3, 3);
  }
}

void deltaOfDelta_transpose_W32(const uint32_t* in, uint32_t* out, const size_t n) {
  assert(n % 16 == 0);
  if (n == 0)
    return;

  for (size_t base = 0; base < n; base += 16) {
    // Delta-of-Delta
    uint32_t d[16];
    if (base == 0) {
      d[0] = in[0];
      d[1] = in[1] - in[0];
      for (int j = 2; j < 16; ++j) {
        d[j] = in[j] - 2 * in[j - 1] + in[j - 2];
      }
    } else {
      d[0] = in[base] - 2 * in[base - 1] + in[base - 2];
      d[1] = in[base + 1] - 2 * in[base] + in[base - 1];
      for (int j = 2; j < 16; ++j) {
        d[j] = in[base + j] - 2 * in[base + j - 1] + in[base + j - 2];
      }
    }

    // Transpose
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        out[base + r * 4 + c] = d[c * 4 + r];
      }
    }
  }
}

void prefixOfPrefix_transpose_W32(const uint32_t* in, uint32_t* out, size_t n) {
  assert((n % 16) == 0);
  if (n == 0)
    return;

  // --- Prelude: decode first tile to seed cross-tile accumulators ---
  {
    // De-transpose first 16 inputs back to row-major d[0..15]
    const uint32_t* t = in;
    uint32_t d[16];
    d[0] = t[0], d[1] = t[4], d[2] = t[8], d[3] = t[12];
    d[4] = t[1], d[5] = t[5], d[6] = t[9], d[7] = t[13];
    d[8] = t[2], d[9] = t[6], d[10] = t[10], d[11] = t[14];
    d[12] = t[3], d[13] = t[7], d[14] = t[11], d[15] = t[15];

    // Scalar inverse of delta-of-delta
    out[0] = d[0];
    out[1] = d[1] + out[0];
    for (int i = 2; i < 16; ++i)
      out[i] = d[i] + 2 * out[i - 1] - out[i - 2];
  }

  // Seed cross-tile accumulators from the prelude
  uint32_t B_acc = out[15] - out[14];      // last value of first prefix
  uint32x4_t Y_acc = vdupq_n_u32(out[15]); // last value of second prefix

  const uint32_t* pi = (const uint32_t*)__builtin_assume_aligned(in + 16, 64);
  uint32_t* po = (uint32_t*)__builtin_assume_aligned(out + 16, 64);
  size_t iters = (n - 16) / 16;

  __builtin_prefetch(pi + 512, 0);
  __builtin_prefetch(po + 512, 1);

  // Ramps for 1..16 (row-major across the 16 outputs of a tile)
  const uint32x4_t r0 = (uint32x4_t){1u, 2u, 3u, 4u};
  const uint32x4_t r1 = (uint32x4_t){5u, 6u, 7u, 8u};
  const uint32x4_t r2 = (uint32x4_t){9u, 10u, 11u, 12u};
  const uint32x4_t r3 = (uint32x4_t){13u, 14u, 15u, 16u};

  const uint32x4_t z = vdupq_n_u32(0);

  for (; iters; --iters, pi += 16, po += 16) {
    // Load columns
    uint32x4_t l00 = vld1q_u32(pi + 0);  // 00 44 88 cc
    uint32x4_t l11 = vld1q_u32(pi + 4);  // 11 55 99 dd
    uint32x4_t l22 = vld1q_u32(pi + 8);  // 22 66 aa ee
    uint32x4_t l33 = vld1q_u32(pi + 12); // 33 77 bb ff

    // ---- First prefix (transpose) ----
    uint32x4_t l01 = vaddq_u32(l00, l11); // 01 45 89 cd
    uint32x4_t l23 = vaddq_u32(l22, l33); // 23 67 ab ef
    uint32x4_t l02 = vaddq_u32(l01, l22); // 02 46 8a ce
    uint32x4_t l03 = vaddq_u32(l01, l23); // 03 47 8b cf
    uint32x4_t a0 = vtrn1q_u32(l00, l01); // 00 01 88 89
    uint32x4_t a1 = vtrn2q_u32(l00, l01); // 44 45 cc cd
    uint32x4_t a2 = vtrn1q_u32(l02, l03); // 02 03 8a 8b
    uint32x4_t a3 = vtrn2q_u32(l02, l03); // 46 47 ce cf
    uint32x4_t p0 = vzip1q_u64(a0, a2);   // 00 01 02 03
    uint32x4_t p1 = vzip1q_u64(a1, a3);   // 44 45 46 47
    uint32x4_t p2 = vzip2q_u64(a0, a2);   // 88 89 8a 8b
    uint32x4_t p3 = vzip2q_u64(a1, a3);   // cc cd ce cf

    // Carry across rows
    p1 = vaddq_u32(p1, vdupq_laneq_u32(p0, 3));
    p2 = vaddq_u32(p2, vdupq_laneq_u32(p1, 3));
    p3 = vaddq_u32(p3, vdupq_laneq_u32(p2, 3));

    uint32_t p_end = vgetq_lane_u32(p3, 3);

    // ---- Second prefix (Hillis-Steele) ----
    uint32x4_t s0 = vaddq_u32(p0, vextq_u32(z, p0, 3));
    uint32x4_t s1 = vaddq_u32(p1, vextq_u32(z, p1, 3));
    uint32x4_t s2 = vaddq_u32(p2, vextq_u32(z, p2, 3));
    uint32x4_t s3 = vaddq_u32(p3, vextq_u32(z, p3, 3));
    s0 = vaddq_u32(s0, vextq_u32(z, s0, 2));
    s1 = vaddq_u32(s1, vextq_u32(z, s1, 2));
    s2 = vaddq_u32(s2, vextq_u32(z, s2, 2));
    s3 = vaddq_u32(s3, vextq_u32(z, s3, 2));
    s1 = vaddq_u32(s1, vdupq_laneq_u32(s0, 3));
    s2 = vaddq_u32(s2, vdupq_laneq_u32(s1, 3));
    s3 = vaddq_u32(s3, vdupq_laneq_u32(s2, 3));

    // ---- Late accumulator application ----
    s0 = vaddq_u32(s0, Y_acc);
    s1 = vaddq_u32(s1, Y_acc);
    s2 = vaddq_u32(s2, Y_acc);
    s3 = vaddq_u32(s3, Y_acc);
    s0 = vmlaq_n_u32(s0, r0, B_acc);
    s1 = vmlaq_n_u32(s1, r1, B_acc);
    s2 = vmlaq_n_u32(s2, r2, B_acc);
    s3 = vmlaq_n_u32(s3, r3, B_acc);

    // Store
    vst1q_u32(po + 0, s0);
    vst1q_u32(po + 4, s1);
    vst1q_u32(po + 8, s2);
    vst1q_u32(po + 12, s3);

    // Update cross-tile accumulators
    B_acc += p_end;                 // next tile's linear term
    Y_acc = vdupq_laneq_u32(s3, 3); // next tile's constant term (last second-prefix value)
  }
}

void xor_naive_W32(const uint32_t* in, uint32_t* out, const size_t n) {
  if (n == 0)
    return;
  out[0] = in[0];
  for (size_t i = 1; i < n; ++i)
    out[i] = (uint32_t)(in[i] ^ in[i - 1]);
}

void xorInv_naive_W32(const uint32_t* in, uint32_t* out, const size_t n) {
  if (n == 0)
    return;
  out[0] = in[0];
  for (size_t i = 1; i < n; ++i)
    out[i] = (uint32_t)(out[i - 1] ^ in[i]);
}

void xorInv_pipelined_W32(const uint32_t* in, uint32_t* out, size_t n) {
  const uint32_t* pi = (const uint32_t*)__builtin_assume_aligned(in, 64);
  uint32_t* po = (uint32_t*)__builtin_assume_aligned(out, 64);
  __builtin_prefetch(in + 512, 0);
  __builtin_prefetch(out + 512, 1);

  size_t iters = n / 16;
  uint32x4_t acc = vdupq_n_u32(0);
  const uint32x4_t z = vdupq_n_u32(0);

  for (; iters; --iters, pi += 16, po += 16) {
    uint32x4_t l0 = vld1q_u32(pi);
    uint32x4_t l1 = vld1q_u32(pi + 4);
    uint32x4_t l2 = vld1q_u32(pi + 8);
    uint32x4_t l3 = vld1q_u32(pi + 12);

    // Hillis-Steele scan within each vector of 4 elements
    uint32x4_t s0 = veorq_u32(l0, vextq_u32(z, l0, 3));
    uint32x4_t s1 = veorq_u32(l1, vextq_u32(z, l1, 3));
    uint32x4_t s2 = veorq_u32(l2, vextq_u32(z, l2, 3));
    uint32x4_t s3 = veorq_u32(l3, vextq_u32(z, l3, 3));
    s0 = veorq_u32(s0, vextq_u32(z, s0, 2));
    s1 = veorq_u32(s1, vextq_u32(z, s1, 2));
    s2 = veorq_u32(s2, vextq_u32(z, s2, 2));
    s3 = veorq_u32(s3, vextq_u32(z, s3, 2));

    // Carry across vectors
    s1 = veorq_u32(s1, vdupq_laneq_u32(s0, 3));
    s3 = veorq_u32(s3, vdupq_laneq_u32(s2, 3));

    // Late accumulator application
    uint32x4_t ac2 = veorq_u32(acc, vdupq_laneq_u32(s1, 3));
    s0 = veorq_u32(s0, acc);
    s1 = veorq_u32(s1, acc);
    s2 = veorq_u32(s2, ac2);
    s3 = veorq_u32(s3, ac2);

    acc = vdupq_laneq_u32(s3, 3);

    vst1q_u32(po, s0);
    vst1q_u32(po + 4, s1);
    vst1q_u32(po + 8, s2);
    vst1q_u32(po + 12, s3);
  }
}

void xor_transpose_W32(const uint32_t* in, uint32_t* out, const size_t n) {
  assert(n % 16 == 0);
  if (n == 0)
    return;

  for (size_t base = 0; base < n; base += 16) {
    // Xor
    uint32_t d[16];
    if (base == 0) {
      d[0] = in[0];
      for (int j = 1; j < 16; ++j) {
        d[j] = in[j] ^ in[j - 1];
      }
    } else {
      for (int j = 0; j < 16; ++j) {
        d[j] = in[base + j] ^ in[base + j - 1];
      }
    }

    // Transpose
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        out[base + r * 4 + c] = d[c * 4 + r];
      }
    }
  }
}

void xorInv_transpose_W32(const uint32_t* in, uint32_t* out, size_t n) {
  const uint32_t* pi = (const uint32_t*)__builtin_assume_aligned(in, 64);
  uint32_t* po = (uint32_t*)__builtin_assume_aligned(out, 64);
  int iters = n / 16;
  uint32x4_t acc = vdupq_n_u32(0);
  __builtin_prefetch(pi + 512, 0);
  __builtin_prefetch(out + 512, 1);

  // "AB" denotes the sum of values at indices A..B, in hexadecimal.
  // Each range corresponds to one element in the vector.
  for (; iters; --iters, pi += 16, po += 16) {
    uint32x4_t l00 = vld1q_u32(pi);      // 00 44 88 cc
    uint32x4_t l11 = vld1q_u32(pi + 4);  // 11 55 99 dd
    uint32x4_t l22 = vld1q_u32(pi + 8);  // 22 66 aa ee
    uint32x4_t l33 = vld1q_u32(pi + 12); // 33 77 bb ff

    // Scan within each vector column
    uint32x4_t l01 = veorq_u32(l00, l11); // 01 45 89 cd
    uint32x4_t l23 = veorq_u32(l22, l33); // 23 67 ab ef
    uint32x4_t l02 = veorq_u32(l01, l22); // 02 46 8a ce
    uint32x4_t l03 = veorq_u32(l01, l23); // 03 47 8b cf

    // Transpose
    uint32x4_t a0 = vtrn1q_u32(l00, l01); // 00 01 88 89
    uint32x4_t a1 = vtrn2q_u32(l00, l01); // 44 45 cc cd
    uint32x4_t a2 = vtrn1q_u32(l02, l03); // 02 03 8a 8b
    uint32x4_t a3 = vtrn2q_u32(l02, l03); // 46 47 ce cf
    uint32x4_t s0 = vzip1q_u64(a0, a2);   // 00 01 02 03
    uint32x4_t s1 = vzip1q_u64(a1, a3);   // 44 45 46 47
    uint32x4_t s2 = vzip2q_u64(a0, a2);   // 88 89 8a 8b
    uint32x4_t s3 = vzip2q_u64(a1, a3);   // cc cd ce cf

    // Carry across vectors
    s1 = veorq_u32(s1, vdupq_laneq_u32(s0, 3));
    s3 = veorq_u32(s3, vdupq_laneq_u32(s2, 3));

    // Late accumulator application
    uint32x4_t ac2 = veorq_u32(acc, vdupq_laneq_u32(s1, 3));
    s0 = veorq_u32(s0, acc);
    s1 = veorq_u32(s1, acc);
    s2 = veorq_u32(s2, ac2);
    s3 = veorq_u32(s3, ac2);

    acc = vdupq_laneq_u32(s3, 3);

    vst1q_u32(po, s0);
    vst1q_u32(po + 4, s1);
    vst1q_u32(po + 8, s2);
    vst1q_u32(po + 12, s3);
  }
}