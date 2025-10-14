/**
 * Originally authored by Daniel Lemire, modified by Ashton Six
 * Based on: https://github.com/fast-pack/FastPFOR/blob/master/src/simdbitpacking.cpp and
 *           https://github.com/fast-pack/FastPFOR/blob/master/headers/simdbinarypacking.h
 */
#include <cstring>
#include <stdexcept>
#include <stdint.h>
#include <stdlib.h>

#ifndef SIMDE_ENABLE_NATIVE_ALIASES
#define SIMDE_ENABLE_NATIVE_ALIASES
#endif /* SIMDE_ENABLE_NATIVE_ALIASES */
#ifdef __SSE4_1__
#include <smmintrin.h>
#else
#include <simde/x86/sse4.1.h>
#endif

namespace BytepackBaseline {

void simdpack(const uint32_t* __restrict__ in, __m128i* __restrict__ out, uint32_t bit);
void simdunpack(const __m128i* __restrict__ in, uint32_t* __restrict__ out, uint32_t bit);

} // namespace BytepackBaseline

namespace BytepackBaseline {

namespace simd {

static void __SIMD_fastpack1_32(const uint32_t* __restrict__ _in, __m128i* __restrict__ out) {
  const __m128i* in = reinterpret_cast<const __m128i*>(_in);
  __m128i OutReg;

  const __m128i mask = _mm_set1_epi32((1U << 1) - 1);

  __m128i InReg = _mm_and_si128(_mm_loadu_si128(in), mask);
  OutReg = InReg;
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 1));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 2));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 3));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 4));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 5));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 6));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 7));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 8));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 9));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 10));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 11));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 12));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 13));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 14));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 15));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 16));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 17));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 18));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 19));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 20));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 21));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 22));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 23));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 24));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 25));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 26));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 27));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 28));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 29));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 30));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 31));
  _mm_storeu_si128(out, OutReg);
}

static void __SIMD_fastpack2_32(const uint32_t* __restrict__ _in, __m128i* __restrict__ out) {
  const __m128i* in = reinterpret_cast<const __m128i*>(_in);
  __m128i OutReg;

  const __m128i mask = _mm_set1_epi32((1U << 2) - 1);

  __m128i InReg = _mm_and_si128(_mm_loadu_si128(in), mask);
  OutReg = InReg;
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 2));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 4));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 6));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 8));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 10));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 12));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 14));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 16));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 18));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 20));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 22));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 24));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 26));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 28));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 30));
  _mm_storeu_si128(out, OutReg);
  ++out;
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = InReg;
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 2));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 4));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 6));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 8));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 10));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 12));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 14));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 16));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 18));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 20));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 22));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 24));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 26));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 28));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 30));
  _mm_storeu_si128(out, OutReg);
}

static void __SIMD_fastpack3_32(const uint32_t* __restrict__ _in, __m128i* __restrict__ out) {
  const __m128i* in = reinterpret_cast<const __m128i*>(_in);
  __m128i OutReg;

  const __m128i mask = _mm_set1_epi32((1U << 3) - 1);

  __m128i InReg = _mm_and_si128(_mm_loadu_si128(in), mask);
  OutReg = InReg;
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 3));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 6));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 9));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 12));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 15));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 18));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 21));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 24));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 27));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 30));
  _mm_storeu_si128(out, OutReg);
  ++out;
  OutReg = _mm_srli_epi32(InReg, 3 - 1);
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 1));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 4));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 7));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 10));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 13));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 16));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 19));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 22));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 25));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 28));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 31));
  _mm_storeu_si128(out, OutReg);
  ++out;
  OutReg = _mm_srli_epi32(InReg, 3 - 2);
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 2));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 5));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 8));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 11));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 14));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 17));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 20));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 23));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 26));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 29));
  _mm_storeu_si128(out, OutReg);
}

static void __SIMD_fastpack4_32(const uint32_t* __restrict__ _in, __m128i* __restrict__ out) {
  const __m128i* in = reinterpret_cast<const __m128i*>(_in);
  __m128i OutReg, InReg;
  const __m128i mask = _mm_set1_epi32((1U << 4) - 1);

  for (uint32_t outer = 0; outer < 4; ++outer) {
    InReg = _mm_and_si128(_mm_loadu_si128(in), mask);
    OutReg = InReg;

    InReg = _mm_and_si128(_mm_loadu_si128(in + 1), mask);
    OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 4));

    InReg = _mm_and_si128(_mm_loadu_si128(in + 2), mask);
    OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 8));

    InReg = _mm_and_si128(_mm_loadu_si128(in + 3), mask);
    OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 12));

    InReg = _mm_and_si128(_mm_loadu_si128(in + 4), mask);
    OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 16));

    InReg = _mm_and_si128(_mm_loadu_si128(in + 5), mask);
    OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 20));

    InReg = _mm_and_si128(_mm_loadu_si128(in + 6), mask);
    OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 24));

    InReg = _mm_and_si128(_mm_loadu_si128(in + 7), mask);
    OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 28));
    _mm_storeu_si128(out, OutReg);
    ++out;

    in += 8;
  }
}

static void __SIMD_fastpack5_32(const uint32_t* __restrict__ _in, __m128i* __restrict__ out) {
  const __m128i* in = reinterpret_cast<const __m128i*>(_in);
  __m128i OutReg;

  const __m128i mask = _mm_set1_epi32((1U << 5) - 1);

  __m128i InReg = _mm_and_si128(_mm_loadu_si128(in), mask);
  OutReg = InReg;
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 5));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 10));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 15));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 20));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 25));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 30));
  _mm_storeu_si128(out, OutReg);
  ++out;
  OutReg = _mm_srli_epi32(InReg, 5 - 3);
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 3));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 8));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 13));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 18));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 23));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 28));
  _mm_storeu_si128(out, OutReg);
  ++out;
  OutReg = _mm_srli_epi32(InReg, 5 - 1);
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 1));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 6));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 11));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 16));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 21));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 26));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 31));
  _mm_storeu_si128(out, OutReg);
  ++out;
  OutReg = _mm_srli_epi32(InReg, 5 - 4);
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 4));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 9));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 14));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 19));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 24));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 29));
  _mm_storeu_si128(out, OutReg);
  ++out;
  OutReg = _mm_srli_epi32(InReg, 5 - 2);
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 2));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 7));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 12));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 17));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 22));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 27));
  _mm_storeu_si128(out, OutReg);
}

static void __SIMD_fastpack6_32(const uint32_t* __restrict__ _in, __m128i* __restrict__ out) {
  const __m128i* in = reinterpret_cast<const __m128i*>(_in);
  __m128i OutReg;

  const __m128i mask = _mm_set1_epi32((1U << 6) - 1);

  __m128i InReg = _mm_and_si128(_mm_loadu_si128(in), mask);
  OutReg = InReg;
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 6));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 12));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 18));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 24));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 30));
  _mm_storeu_si128(out, OutReg);
  ++out;
  OutReg = _mm_srli_epi32(InReg, 6 - 4);
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 4));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 10));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 16));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 22));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 28));
  _mm_storeu_si128(out, OutReg);
  ++out;
  OutReg = _mm_srli_epi32(InReg, 6 - 2);
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 2));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 8));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 14));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 20));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 26));
  _mm_storeu_si128(out, OutReg);
  ++out;
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = InReg;
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 6));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 12));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 18));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 24));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 30));
  _mm_storeu_si128(out, OutReg);
  ++out;
  OutReg = _mm_srli_epi32(InReg, 6 - 4);
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 4));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 10));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 16));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 22));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 28));
  _mm_storeu_si128(out, OutReg);
  ++out;
  OutReg = _mm_srli_epi32(InReg, 6 - 2);
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 2));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 8));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 14));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 20));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 26));
  _mm_storeu_si128(out, OutReg);
}

static void __SIMD_fastpack7_32(const uint32_t* __restrict__ _in, __m128i* __restrict__ out) {
  const __m128i* in = reinterpret_cast<const __m128i*>(_in);
  __m128i OutReg;

  const __m128i mask = _mm_set1_epi32((1U << 7) - 1);

  __m128i InReg = _mm_and_si128(_mm_loadu_si128(in), mask);
  OutReg = InReg;
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 7));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 14));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 21));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 28));
  _mm_storeu_si128(out, OutReg);
  ++out;
  OutReg = _mm_srli_epi32(InReg, 7 - 3);
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 3));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 10));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 17));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 24));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 31));
  _mm_storeu_si128(out, OutReg);
  ++out;
  OutReg = _mm_srli_epi32(InReg, 7 - 6);
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 6));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 13));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 20));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 27));
  _mm_storeu_si128(out, OutReg);
  ++out;
  OutReg = _mm_srli_epi32(InReg, 7 - 2);
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 2));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 9));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 16));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 23));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 30));
  _mm_storeu_si128(out, OutReg);
  ++out;
  OutReg = _mm_srli_epi32(InReg, 7 - 5);
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 5));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 12));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 19));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 26));
  _mm_storeu_si128(out, OutReg);
  ++out;
  OutReg = _mm_srli_epi32(InReg, 7 - 1);
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 1));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 8));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 15));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 22));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 29));
  _mm_storeu_si128(out, OutReg);
  ++out;
  OutReg = _mm_srli_epi32(InReg, 7 - 4);
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 4));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 11));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 18));
  InReg = _mm_and_si128(_mm_loadu_si128(++in), mask);

  OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 25));
  _mm_storeu_si128(out, OutReg);
}

static void __SIMD_fastpack8_32(const uint32_t* __restrict__ _in, __m128i* __restrict__ out) {
  const __m128i* in = reinterpret_cast<const __m128i*>(_in);
  __m128i OutReg, InReg;
  const __m128i mask = _mm_set1_epi32((1U << 8) - 1);

  for (uint32_t outer = 0; outer < 8; ++outer) {
    InReg = _mm_and_si128(_mm_loadu_si128(in), mask);
    OutReg = InReg;

    InReg = _mm_and_si128(_mm_loadu_si128(in + 1), mask);
    OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 8));

    InReg = _mm_and_si128(_mm_loadu_si128(in + 2), mask);
    OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 16));

    InReg = _mm_and_si128(_mm_loadu_si128(in + 3), mask);
    OutReg = _mm_or_si128(OutReg, _mm_slli_epi32(InReg, 24));
    _mm_storeu_si128(out, OutReg);
    ++out;

    in += 4;
  }
}

static void SIMD_nullunpacker32(const __m128i* __restrict__, uint32_t* __restrict__ out) {
  memset(out, 0, 32 * 4 * 4);
}

static void __SIMD_fastunpack1_32(const __m128i* __restrict__ in, uint32_t* __restrict__ _out) {
  __m128i* out = reinterpret_cast<__m128i*>(_out);
  __m128i InReg1 = _mm_loadu_si128(in);
  __m128i InReg2 = InReg1;
  __m128i OutReg1, OutReg2, OutReg3, OutReg4;
  const __m128i mask = _mm_set1_epi32(1);
#if (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))) ||                                               \
    (defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_AMD64)))
  unsigned shift = 0;

  for (unsigned i = 0; i < 8; ++i) {
    OutReg1 = _mm_and_si128(_mm_srli_epi32(InReg1, shift++), mask);
    OutReg2 = _mm_and_si128(_mm_srli_epi32(InReg2, shift++), mask);
    OutReg3 = _mm_and_si128(_mm_srli_epi32(InReg1, shift++), mask);
    OutReg4 = _mm_and_si128(_mm_srli_epi32(InReg2, shift++), mask);
    _mm_storeu_si128(out++, OutReg1);
    _mm_storeu_si128(out++, OutReg2);
    _mm_storeu_si128(out++, OutReg3);
    _mm_storeu_si128(out++, OutReg4);
  }
#elif (defined(__GNUC__) && (defined(__aarch64__))) || (defined(_MSC_VER) && defined(_M_ARM64))
  OutReg1 = _mm_and_si128(_mm_srli_epi32(InReg1, 0), mask);
  OutReg2 = _mm_and_si128(_mm_srli_epi32(InReg2, 1), mask);
  OutReg3 = _mm_and_si128(_mm_srli_epi32(InReg1, 2), mask);
  OutReg4 = _mm_and_si128(_mm_srli_epi32(InReg2, 3), mask);
  _mm_store_si128(out++, OutReg1);
  _mm_store_si128(out++, OutReg2);
  _mm_store_si128(out++, OutReg3);
  _mm_store_si128(out++, OutReg4);
  OutReg1 = _mm_and_si128(_mm_srli_epi32(InReg1, 4), mask);
  OutReg2 = _mm_and_si128(_mm_srli_epi32(InReg2, 5), mask);
  OutReg3 = _mm_and_si128(_mm_srli_epi32(InReg1, 6), mask);
  OutReg4 = _mm_and_si128(_mm_srli_epi32(InReg2, 7), mask);
  _mm_store_si128(out++, OutReg1);
  _mm_store_si128(out++, OutReg2);
  _mm_store_si128(out++, OutReg3);
  _mm_store_si128(out++, OutReg4);
  OutReg1 = _mm_and_si128(_mm_srli_epi32(InReg1, 8), mask);
  OutReg2 = _mm_and_si128(_mm_srli_epi32(InReg2, 9), mask);
  OutReg3 = _mm_and_si128(_mm_srli_epi32(InReg1, 10), mask);
  OutReg4 = _mm_and_si128(_mm_srli_epi32(InReg2, 11), mask);
  _mm_store_si128(out++, OutReg1);
  _mm_store_si128(out++, OutReg2);
  _mm_store_si128(out++, OutReg3);
  _mm_store_si128(out++, OutReg4);
  OutReg1 = _mm_and_si128(_mm_srli_epi32(InReg1, 12), mask);
  OutReg2 = _mm_and_si128(_mm_srli_epi32(InReg2, 13), mask);
  OutReg3 = _mm_and_si128(_mm_srli_epi32(InReg1, 14), mask);
  OutReg4 = _mm_and_si128(_mm_srli_epi32(InReg2, 15), mask);
  _mm_store_si128(out++, OutReg1);
  _mm_store_si128(out++, OutReg2);
  _mm_store_si128(out++, OutReg3);
  _mm_store_si128(out++, OutReg4);
  OutReg1 = _mm_and_si128(_mm_srli_epi32(InReg1, 16), mask);
  OutReg2 = _mm_and_si128(_mm_srli_epi32(InReg2, 17), mask);
  OutReg3 = _mm_and_si128(_mm_srli_epi32(InReg1, 18), mask);
  OutReg4 = _mm_and_si128(_mm_srli_epi32(InReg2, 19), mask);
  _mm_store_si128(out++, OutReg1);
  _mm_store_si128(out++, OutReg2);
  _mm_store_si128(out++, OutReg3);
  _mm_store_si128(out++, OutReg4);
  OutReg1 = _mm_and_si128(_mm_srli_epi32(InReg1, 20), mask);
  OutReg2 = _mm_and_si128(_mm_srli_epi32(InReg2, 21), mask);
  OutReg3 = _mm_and_si128(_mm_srli_epi32(InReg1, 22), mask);
  OutReg4 = _mm_and_si128(_mm_srli_epi32(InReg2, 23), mask);
  _mm_store_si128(out++, OutReg1);
  _mm_store_si128(out++, OutReg2);
  _mm_store_si128(out++, OutReg3);
  _mm_store_si128(out++, OutReg4);
  OutReg1 = _mm_and_si128(_mm_srli_epi32(InReg1, 24), mask);
  OutReg2 = _mm_and_si128(_mm_srli_epi32(InReg2, 25), mask);
  OutReg3 = _mm_and_si128(_mm_srli_epi32(InReg1, 26), mask);
  OutReg4 = _mm_and_si128(_mm_srli_epi32(InReg2, 27), mask);
  _mm_store_si128(out++, OutReg1);
  _mm_store_si128(out++, OutReg2);
  _mm_store_si128(out++, OutReg3);
  _mm_store_si128(out++, OutReg4);
  OutReg1 = _mm_and_si128(_mm_srli_epi32(InReg1, 28), mask);
  OutReg2 = _mm_and_si128(_mm_srli_epi32(InReg2, 29), mask);
  OutReg3 = _mm_and_si128(_mm_srli_epi32(InReg1, 30), mask);
  OutReg4 = _mm_and_si128(_mm_srli_epi32(InReg2, 31), mask);
  _mm_store_si128(out++, OutReg1);
  _mm_store_si128(out++, OutReg2);
  _mm_store_si128(out++, OutReg3);
  _mm_store_si128(out++, OutReg4);
#endif
}

static void __SIMD_fastunpack2_32(const __m128i* __restrict__ in, uint32_t* __restrict__ _out) {
  __m128i* out = reinterpret_cast<__m128i*>(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 2) - 1);

  OutReg = _mm_and_si128(InReg, mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 14), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 18), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 22), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 24), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 26), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 28), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 14), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 18), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 22), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 24), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 26), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 28), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack3_32(const __m128i* __restrict__ in, uint32_t* __restrict__ _out) {
  __m128i* out = reinterpret_cast<__m128i*>(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 3) - 1);

  OutReg = _mm_and_si128(InReg, mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 3), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 9), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 15), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 18), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 21), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 24), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 27), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 3 - 1), mask));
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 1), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 7), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 13), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 19), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 22), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 25), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 28), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 31);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 3 - 2), mask));
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 5), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 11), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 14), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 17), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 23), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 26), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 29);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack4_32(const __m128i* __restrict__ in, uint32_t* __restrict__ _out) {
  __m128i* out = reinterpret_cast<__m128i*>(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 4) - 1);

  OutReg = _mm_and_si128(InReg, mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 24), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 24), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 24), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 24), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack5_32(const __m128i* __restrict__ in, uint32_t* __restrict__ _out) {
  __m128i* out = reinterpret_cast<__m128i*>(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 5) - 1);

  OutReg = _mm_and_si128(InReg, mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 5), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 15), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 25), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 5 - 3), mask));
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 3), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 13), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 18), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 23), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 5 - 1), mask));
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 1), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 11), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 21), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 26), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 31);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 5 - 4), mask));
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 9), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 14), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 19), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 24), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 29);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 5 - 2), mask));
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 7), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 17), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 22), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 27);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack6_32(const __m128i* __restrict__ in, uint32_t* __restrict__ _out) {
  __m128i* out = reinterpret_cast<__m128i*>(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 6) - 1);

  OutReg = _mm_and_si128(InReg, mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 18), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 24), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 6 - 4), mask));
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 22), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 6 - 2), mask));
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 14), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 18), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 24), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 6 - 4), mask));
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 22), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 6 - 2), mask));
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 14), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack7_32(const __m128i* __restrict__ in, uint32_t* __restrict__ _out) {
  __m128i* out = reinterpret_cast<__m128i*>(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 7) - 1);

  OutReg = _mm_and_si128(InReg, mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 7), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 14), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 21), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 28);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 7 - 3), mask));
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 3), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 10), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 17), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 24), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 31);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 7 - 6), mask));
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 6), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 13), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 20), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 27);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 7 - 2), mask));
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 2), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 9), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 23), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 30);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 7 - 5), mask));
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 5), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 12), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 19), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 26);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 7 - 1), mask));
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 1), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 15), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 22), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 29);
  InReg = _mm_loadu_si128(++in);

  OutReg = _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 7 - 4), mask));
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 4), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 11), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 18), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 25);
  _mm_storeu_si128(out++, OutReg);
}

static void __SIMD_fastunpack8_32(const __m128i* __restrict__ in, uint32_t* __restrict__ _out) {
  __m128i* out = reinterpret_cast<__m128i*>(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  const __m128i mask = _mm_set1_epi32((1U << 8) - 1);

  OutReg = _mm_and_si128(InReg, mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  InReg = _mm_loadu_si128(++in);

  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(InReg, mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 8), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_and_si128(_mm_srli_epi32(InReg, 16), mask);
  _mm_storeu_si128(out++, OutReg);

  OutReg = _mm_srli_epi32(InReg, 24);
  _mm_storeu_si128(out++, OutReg);
}

} // namespace simd

void simdunpack(const __m128i* __restrict__ in, uint32_t* __restrict__ out, const uint32_t bit) {
  using namespace simd;
  switch (bit) {
  case 0: SIMD_nullunpacker32(in, out); return;
  case 1: __SIMD_fastunpack1_32(in, out); return;
  case 2: __SIMD_fastunpack2_32(in, out); return;
  case 3: __SIMD_fastunpack3_32(in, out); return;
  case 4: __SIMD_fastunpack4_32(in, out); return;
  case 5: __SIMD_fastunpack5_32(in, out); return;
  case 6: __SIMD_fastunpack6_32(in, out); return;
  case 7: __SIMD_fastunpack7_32(in, out); return;
  case 8: __SIMD_fastunpack8_32(in, out); return;
  default: break;
  }
  throw std::logic_error("number of bits is unsupported");
}

/*assumes that integers fit in the prescribed number of bits*/
void simdpack(const uint32_t* __restrict__ in, __m128i* __restrict__ out, const uint32_t bit) {
  using namespace simd;
  switch (bit) {
  case 0: return;
  case 1: __SIMD_fastpack1_32(in, out); return;
  case 2: __SIMD_fastpack2_32(in, out); return;
  case 3: __SIMD_fastpack3_32(in, out); return;
  case 4: __SIMD_fastpack4_32(in, out); return;
  case 5: __SIMD_fastpack5_32(in, out); return;
  case 6: __SIMD_fastpack6_32(in, out); return;
  case 7: __SIMD_fastpack7_32(in, out); return;
  case 8: __SIMD_fastpack8_32(in, out); return;
  default: break;
  }
  throw std::logic_error("number of bits is unsupported");
}

void SIMD_fastunpack_32(const __m128i* __restrict__ in, uint32_t* __restrict__ out, const uint32_t bit) {
  simdunpack(in, out, bit);
}
void SIMD_fastpack_32(const uint32_t* __restrict__ in, __m128i* __restrict__ out, const uint32_t bit) {
  simdpack(in, out, bit);
}

static const uint32_t MiniBlockSize = 128;
static const uint32_t HowManyMiniBlocks = 16;

void encodeArray(const uint32_t* in, const size_t length, uint32_t* out, const uint32_t bit) {
  const uint32_t* const final = in + length;
  for (; in + HowManyMiniBlocks * MiniBlockSize <= final; in += HowManyMiniBlocks * MiniBlockSize) {
    for (uint32_t i = 0; i < HowManyMiniBlocks; ++i) {
      SIMD_fastpack_32(in + i * MiniBlockSize, reinterpret_cast<__m128i*>(out), bit);
      out += MiniBlockSize / 32 * bit;
    }
  }
  if (in < final) {
    const size_t howmany = (final - in) / MiniBlockSize;
    for (uint32_t i = 0; i < howmany; ++i) {
      SIMD_fastpack_32(in + i * MiniBlockSize, reinterpret_cast<__m128i*>(out), bit);
      out += MiniBlockSize / 32 * bit;
    }
  }
}

const uint32_t* decodeArray(const uint32_t* in, const size_t length, uint32_t* out, const uint32_t bit) {
  const uint32_t* const initout(out);
  for (; out < initout + length / (HowManyMiniBlocks * MiniBlockSize) * HowManyMiniBlocks * MiniBlockSize;
       out += HowManyMiniBlocks * MiniBlockSize) {
    for (uint32_t i = 0; i < HowManyMiniBlocks; ++i) {
      SIMD_fastunpack_32(reinterpret_cast<const __m128i*>(in), out + i * MiniBlockSize, bit);
      in += MiniBlockSize / 32 * bit;
    }
  }
  if (out < initout + length) {
    const size_t howmany = (initout + length - out) / MiniBlockSize;
    for (uint32_t i = 0; i < howmany; ++i) {
      SIMD_fastunpack_32(reinterpret_cast<const __m128i*>(in), out + i * MiniBlockSize, bit);
      in += MiniBlockSize / 32 * bit;
    }
  }
  return in;
}

} // namespace BytepackBaseline
