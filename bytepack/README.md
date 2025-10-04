# NEON Bytepack

> **Scope:** Research preview. Perf demo & reference only. Not a drop-in library.
> **Audience:** Knowledge of SIMD, Arm ISA, and µarch basics assumed.
> **Created by [Ashton Six](https://ashtonsix.com)** (available for hire).

## TL;DR

NEON pack/unpack micro-kernels for moving K ∈ {1…7} bits per input byte into tightly packed outputs (and back). **86 GB/s** on L1-resident data, 2.0× over a plane-transpose baseline (FastPFoR-derived, compiled for NEON via SIMDe).

_Conditions:_ L1-hot, 1 thread, Neoverse V2 (Graviton4). Full setup in **Appendix A**.

## Why it’s fast

- **Byte-level parallelism.** More logical elements per instruction.
- **Aligned interleaving.** Contributors align to word boundaries; most moves collapse to one op. No cross-word spill.
- **Instruction shaping.** Small helpers (`ld/st/sli/sri/bit/bif`) coax paired load/store and fused shift-insert/bit-blend.
- **µarch scheduling/RA.** Split chains, trim live ranges, balance ports, unroll hot loops, align buffers, keep prefetch linear.
- **Minimal control.** Per-K specializations inline the hot loop; no per-iter calls, switches, or mask builds.

## How it works

Pack reduces merge trees; unpack reverses.

- **`K=1`** — 8→1. Successive merges (`<<1`): 8→4→2→1.
- **`K=2`** — 4→1. Two levels of `<<2`.
- **`K=3`** — 8→3. `(3,3,2)` / `(3,3,1,1)` / `(3,3,2)` pattern.
- **`K=4`** — 2→1. Nybble interleave (`<<4`).
- **`K=5`** — 8→5. Evens keep 5 LSBs; odds donate 3 MSBs to neighbors; leftover 2 LSBs from odds collected in a fifth stream.
- **`K=6`** — 4→3. `(6,2)` / `(4,4)` / `(6,2)` pattern.
- **`K=7`** — 8→7. Alternating `(7,1)` and `(6,2)` with a central `(4,4)`.
- **`K=8`** — `memcpy`.

**Pack rule:** right-aligned pieces are no-ops; left-aligned pieces use `SLI`.
**Unpack rule:** right-aligned pieces use `AND`; left-aligned pieces use `SHR` then `AND`.
Exact per-K layouts are in source.

## Integration: usage in a real codec

The micro-kernels are compute-cheap but memory-throughput-bound unless L1-hot. To beat plane-transpose on broader workloads, fuse bytepack into adjacent kernel stages so intermediates stay in-register and only the final artifact hits memory.

**Examples (inspiration).** Potential adjacent stages include delta/zigzag, dictionary lookup, or an entropy stage. See **Appendix B** for `K>8`.

## License

Apache 2.0.

## Author Note

I am available for hire. Contact me if you have a project or position you think I'd be a good fit for via [https://ashtonsix.com](https://ashtonsix.com).

---

## Appendix A — Benchmark

- **Hardware/flags.** Neoverse V2 (Graviton4), single thread, CPU pinned; `-O3 -mcpu=neoverse-v2`.
- **Working set.** 16 KiB (4 pages), resident in L1; warm runs only.
- **Inputs.** PRNG-generated `uint8_t[16384]`, values uniform in `[0, 2^K)`, seed fixed.
- **Reports.** Throughput in GB/s, pack (input) and unpack (output) byte rates.
- **Comparison.** Baseline operates on 32-bit inputs. For integers/s, NEON throughput is 8.0× baseline.

**Results:**

```txt
K  NEON pack   NEON unpack  Baseline pack   Baseline unpack
1  90.50       66.80        38.83           58.94
2  104.93      72.88        48.14           56.17
3  84.55       73.11        40.92           59.56
4  95.22       70.09        52.83           68.66
5  80.43       69.70        39.64           56.96
6  79.58       68.11        44.66           57.56
7  66.66       66.20        38.24           53.96
8  79.73       80.17        58.37           73.35
```

**Reproduction (Graviton4):**

```sh
make -f bytepack.mk && ./out/bytepack_eval
```

---

## Appendix B — Advisory: wider bit-widths

Keep dataflow in-register; form byte-plane intermediates with lane transpose (Arm TRN); use Bytepack only for partial-byte residue.

- **`K=9…15`**: Split 16-bit lanes into `[LSB bytes][MSB bytes]` via `TRN1/TRN2`.
- **`K=17…31`**: Emit 16-bit significands; collapse one (optionally two) MSB byte-planes from parked halves.
- **`K=33…63`**: Hierarchical transpose `32→16→8`; 48-bit special case forwards bytes 4–5 as 16-bit pairs.
- **Residue:** if `K%8≠0`, pack the final MSB plane with `K=K%8`.
- **`K=8,16,32,64`**: Copy.

_Pseudo-code sketch (TRN = lane-wise transpose):_

```py
if 9 ≤ K ≤ 15:
  # inputs: x0, x1
  emit trn1.16(x0, x1)   # LSB bytes across lanes
  emit trn2.16(x0, x1)   # MSB bytes across lanes

if 17 ≤ K ≤ 31:
  # park the high bytes, then collapse to global planes
  u01 := trn2.16(x0, x1)
  u23 := trn2.16(x2, x3)
  emit trn1.16(x0, x1)           # 16-bit significands
  emit trn1.16(x2, x3)
  emit trn1.8(u01, u23)          # next MSB byte
  if K ≥ 25: emit trn2.8(u01, u23)

if 33 ≤ K ≤ 63:
  # low 32 bits (bytes 0..3) of each pair
  emit trn1.32(x0, x1), trn1.32(x2, x3), trn1.32(x4, x5), trn1.32(x6, x7)

  # park high 32 bits (bytes 4..7) of each pair
  u01 := trn2.32(x0, x1); u23 := trn2.32(x2, x3)
  u45 := trn2.32(x4, x5); u67 := trn2.32(x6, x7)

  # form (4,5) and (6,7) 16-bit pairs per half
  p45_lo := trn1.16(u01, u23)   # bytes 4..5, lanes 0..3
  p45_hi := trn1.16(u45, u67)   # bytes 4..5, lanes 4..7
  p67_lo := trn2.16(u01, u23)   # bytes 6..7, lanes 0..3
  p67_hi := trn2.16(u45, u67)   # bytes 6..7, lanes 4..7

  # --- bytes 4..5 materialization (K=48 skipped, relayed directly) ---
  if 33 ≤ K ≤ 47:
    emit trn1.8(p45_lo, p45_hi)              # byte 4
    if K ≥ 41: emit trn2.8(p45_lo, p45_hi)   # byte 5
    stop

  emit p45_lo, p45_hi           # relay 16-bit pairs (bytes 4..5) for b≥48
  # consumer may later materialize:
  #   byte4 := trn1.8(p45_lo, p45_hi)
  #   byte5 := trn2.8(p45_lo, p45_hi)

  # --- bytes 6..7 ---
  if K ≥ 49:
    emit trn1.8(p67_lo, p67_hi)              # byte 6
    if K ≥ 57: emit trn2.8(p67_lo, p67_hi)   # byte 7 (MSB)
```
