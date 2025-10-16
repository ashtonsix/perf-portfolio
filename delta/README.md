# NEON Delta Coding

> **Scope:** Research preview. Perf demo & reference only. Not a drop-in library.
> **Created by [Ashton Six](https://ashtonsix.com)** (available for hire).

## TL;DR

Delta coding stores differences between successive values instead of raw integers, making data more compressible. Reversing this transformation (via prefix sum) is typically slow due to serial data dependencies.

We achieve **19.8 GB/s** prefix sum throughput—**1.8x faster** than a naive implementation and **2.6x faster** than FastPFoR—by restructuring the compute to shorten dependency chains and improve instruction-level parallelism. We achieve similar acceleration when reversing delta-of-delta and xor-with-previous transforms.

_Conditions:_ L1-hot, 1 thread, Neoverse V2 (Graviton4).

_Example (delta coding):_ `[30, 33, 35, 40] → [30, 3, 2, 5]`

## Delta coding

Delta coding is simple and easily auto-vectorized by compilers. We recommend this naive implementation (**41.1 GB/s**):

```c
void delta_W32(const uint32_t* in, uint32_t* out, const size_t n) {
  if (n == 0) return;
  out[0] = in[0];
  for (size_t i = 1; i < n; ++i) {
    out[i] = in[i] - in[i - 1];
  }
}
```

**Notes:**

- For in-place delta coding, iterate backwards.
- For possibly decreasing sequences, combine with zig-zag coding.

## Prefix sum

Prefix sums are harder to accelerate due to serial data dependencies. Each output depends on all prior inputs:

```txt
out[i] = in[0] + in[1] + … + in[i]
```

Naive implementations consume inputs immediately after they're produced:

```c
for (size_t i = 1; i < n; ++i) {
  out[i] = out[i - 1] + in[i]; // out[i] must wait for out[i-1]
}
```

This is a worst-case scenario for compute scheduling: instruction latency leads to stalled execution.

### Baseline (prior art: Hillis–Steele scan)

FastPFoR (following Lemire, Boytsov and Kurz, 2014, §5) uses a vectorised Hillis–Steele scan. Excerpt from [FastPFoR/deltautil.h](https://github.com/fast-pack/FastPFOR/blob/039134b61d39d52a18c72717ee1ec2afb797a222/headers/deltautil.h#L296)::**fastinverseDelta2**:

```c
while (pCurr < pEnd) {
  __m128i a0 = _mm_loadu_si128(pCurr);
  __m128i a1 = _mm_add_epi32(_mm_slli_si128(a0, 8), a0);
  __m128i a2 = _mm_add_epi32(_mm_slli_si128(a1, 4), a1);
  a0 = _mm_add_epi32(a2, runningCount);
  runningCount = _mm_shuffle_epi32(a0, 0xFF);
  _mm_storeu_si128(pCurr++, a0);
}
```

FastPFoR is well-established in both industry and academia. However, on our target platform (Graviton4, SIMDe-compiled) it benchmarks at only ~7.7 GB/s, beneath a naive scalar loop at ~10.8 GB/s.

Neither approach exceeds ~27% of delta coding throughput.

### Our Contribution: shortening the cross-vector carry chain

We improve upon the FastPFoR baseline in two steps:

1. **Unroll and tune for ARM** (~10.4 GB/s)
   - Recovers naive-level performance via explicit loop unrolling and ARM-specific tuning.
2. **Delay accumulator application** (~19.8 GB/s)
   - **Key insight:** Instead of immediately adding the running accumulator to each vector, we:
     1. Compute local prefix sums for each semi-block (8 values)
     2. **Then** apply running accumulators to all vectors in parallel
   - This approach trades a slight increase in instruction count for dramatically shorter dependency chains
   - Out-of-order hardware can now issue multiple operations simultaneously instead of waiting for each result

Here's the unified unrolled→pipelined diff:

```diff
- void prefixSum_unrolled(const uint32_t* in, uint32_t* out, size_t n) {
+ void prefixSum_pipelined(const uint32_t* in, uint32_t* out, size_t n) {
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

      // Hillis–Steele scan within each vector of 4 elements
      uint32x4_t s0 = vaddq_u32(l0, vextq_u32(z, l0, 3));
      uint32x4_t s1 = vaddq_u32(l1, vextq_u32(z, l1, 3));
      uint32x4_t s2 = vaddq_u32(l2, vextq_u32(z, l2, 3));
      uint32x4_t s3 = vaddq_u32(l3, vextq_u32(z, l3, 3));
      s0 = vaddq_u32(s0, vextq_u32(z, s0, 2));
      s1 = vaddq_u32(s1, vextq_u32(z, s1, 2));
      s2 = vaddq_u32(s2, vextq_u32(z, s2, 2));
      s3 = vaddq_u32(s3, vextq_u32(z, s3, 2));

      // Carry across vectors
-     s0 = vaddq_u32(s0, acc);
      s1 = vaddq_u32(s1, vdupq_laneq_u32(s0, 3));
-     s2 = vaddq_u32(s2, vdupq_laneq_u32(s1, 3));
      s3 = vaddq_u32(s3, vdupq_laneq_u32(s2, 3));

+     // Late accumulator application
+     uint32x4_t ac2 = vaddq_u32(acc, vdupq_laneq_u32(s1, 3));
+     s0 = vaddq_u32(s0, acc);
+     s1 = vaddq_u32(s1, acc);
+     s2 = vaddq_u32(s2, ac2);
+     s3 = vaddq_u32(s3, ac2);

      acc = vdupq_laneq_u32(s3, 3);

      vst1q_u32(po, s0);
      vst1q_u32(po + 4, s1);
      vst1q_u32(po + 8, s2);
      vst1q_u32(po + 12, s3);
    }
  }
```

## Delta-of-delta (second difference)

Delta-of-delta encodes the differences between consecutive deltas. It's effective for smoothly changing gradients (timestamps, sensor readings, etc.):

```c
out[0] = in[0]
out[1] = in[1] - in[0]
out[i>=2] = (in[i] − in[i−1]) − (in[i−1] − in[i−2])
          = in[i] - 2*in[i-1] + in[i-2]
```

**Reverse:**

```c
out[0] = in[0]
out[1] = in[1] + out[0]
out[i>=2] = in[i] + 2*out[i-1] - out[i-2]
```

Our pipeline trick yields a 2.2x speed-up over a naive baseline (8.2 GB/s).

<details>

<summary>Delta-of-delta implementations</summary>

Delta-of-delta

```c
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
```

Prefix-of-prefix (naive)

```c
void prefixOfPrefix_naive(const uint32_t* in, uint32_t* out, const size_t n) {
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
```

Prefix-of-prefix (pipelined)

```c
void prefixOfPrefix_pipelined(const uint32_t* in, uint32_t* out, const size_t n) {
  // Process 16 elements in prelude so 64B alignment holds in main loop body
  if (n < 2) {
    if (n == 1)
      out[0] = in[0];
    return;
  }
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
```

</details>

## XOR-with-previous

XOR-with-previous encodes each element as the XOR of itself with the previous element. It's effective for floating point values (eg, Gorilla compression):

```c
out[i] = in[i] ^ in[i-1]
```

**Reverse (XOR prefix):** Each decoded element is the XOR of the previous decoded element with the current encoded element:

```c
out[i] = out[i-1] ^ in[i]
```

XOR-based prefix scans are less receptive to our pipelining optimization than addition-based scans (only yields a 1.5x improvement, 10.8 vs 16.6 GB/s). While ADD and XOR have identical latency and throughput on Graviton4, ADD operations can take advantage of dedicated fast-forward circuits in the execution pipeline, making them less sensitive to dependency chains. XOR lacks these shortcuts.

**Transpose-based alternative:** To address XOR's latency sensitivity, we developed
a transpose-based algorithm that outperforms Hillis-Steele for the inverse operation.
This approach:

- Breaks input ordering (outputs appear in transposed order)
- Improves reverse-direction throughput by 2.0x (10.8 vs 21.5 GB/s)
- May offer more consistent performance across CPU microarchitectures due to reduced dependency on architecture-specific optimizations

We include this variant for completeness, but recommend the pipelined approach for
most use cases.

<details>

<summary>XOR implementations</summary>

XOR:

```c
void xor_naive_W32(const uint32_t* in, uint32_t* out, const size_t n) {
  if (n == 0)
    return;
  out[0] = in[0];
  for (size_t i = 1; i < n; ++i)
    out[i] = (uint32_t)(in[i] ^ in[i - 1]);
}
```

XOR inverse (naive):

```c
void xorInv_naive_W32(const uint32_t* in, uint32_t* out, const size_t n) {
  if (n == 0)
    return;
  out[0] = in[0];
  for (size_t i = 1; i < n; ++i)
    out[i] = (uint32_t)(out[i - 1] ^ in[i]);
}
```

XOR inverse (pipelined):

```c
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
```

XOR + 4x4 transpose:

```c
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
```

XOR' + 4x4 transpose:

```c
void xorInv_transpose_W32(const uint32_t* in, uint32_t* out, size_t n) {
  const uint32_t* pi = (const uint32_t*)__builtin_assume_aligned(in, 64);
  uint32_t* po = (uint32_t*)__builtin_assume_aligned(out, 64);
  int iters = n / 16;
  uint32x4_t acc = vdupq_n_u32(0);
  __builtin_prefetch(in + 512, 0);
  __builtin_prefetch(out + 512, 1);

  // "AB" denotes the sum of values at indices A..B, in hexadecimal.
  // Each range corresponds to one element in the vector.
  for (; iters; --iters, in += 16, out += 16) {
    uint32x4_t l00 = vld1q_u32(in);      // 00 44 88 cc
    uint32x4_t l11 = vld1q_u32(in + 4);  // 11 55 99 dd
    uint32x4_t l22 = vld1q_u32(in + 8);  // 22 66 aa ee
    uint32x4_t l33 = vld1q_u32(in + 12); // 33 77 bb ff

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

    vst1q_u32(out, s0);
    vst1q_u32(out + 4, s1);
    vst1q_u32(out + 8, s2);
    vst1q_u32(out + 12, s3);
  }
}
```

</details>

## Other bit-widths

The techniques described extend naturally to 8, 16, and 64-bit integers.
Implementation details coming soon.

## Results

**Test conditions:** 4 KiB working set, 20,000 iterations, single-threaded, L1-hot on AWS Graviton4 (Neoverse V2), `-O3 -mcpu=neoverse-v2`.

```txt
| Routine                              | Throughput (GB/s) | vs Naive          |
| ------------------------------------ | ----------------- | ----------------- |
| Delta  / naive                       | 41.07             | -                 |
| Prefix / FastPFoR (SIMDe)            | 7.70              | 0.7x              |
| Prefix / naive                       | 10.80             | 1.0x              |
| Prefix / unrolled                    | 10.43             | 1.0x              |
| Prefix / pipelined                   | 19.76             | 1.8x              |
| Delta  + 4x4 transpose               | 24.43             | -                 |
| Prefix + 4x4 transpose               | 21.36             | 2.0x              |
|                                      |                   |                   |
| Delta-of-delta   / naive             | 29.59             | -                 |
| Prefix-of-prefix / naive             | 3.65              | 1.0x              |
| Prefix-of-prefix / pipelined         | 8.20              | 2.2x              |
| Delta-of-delta   + 4x4 transpose     | 20.16             | -                 |
| Prefix-of-prefix + 4x4 transpose     | 8.29              | 2.3x              |
|                                      |                   |                   |
| XOR  / naive                         | 41.43             | -                 |
| XOR' / naive                         | 10.81             | 1.0x              |
| XOR' / pipelined                     | 16.59             | 1.5x              |
| XOR  + 4x4 transpose                 | 24.33             | -                 |
| XOR' + 4x4 transpose                 | 21.53             | 2.0x              |
```

**Reproduction:**

Install dependencies per the top-level repository [README](../README.md), then:

```sh
make -f delta.mk && ./out/delta_eval
```
