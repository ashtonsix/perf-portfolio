#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------
constexpr unsigned BASE_WIRES = 4;
constexpr unsigned MAX_GATES = 12;
constexpr unsigned MAX_SIGNALS = BASE_WIRES + MAX_GATES;
constexpr unsigned MAX_SIGNALS_COV = MAX_SIGNALS + 4;
constexpr unsigned MAX_SOLUTIONS = 128;
constexpr unsigned UPDATE_MS = 150;
constexpr size_t MAX_FRONTIER = 50000000;
constexpr const char* OUTPUT_FILE = "p4_circuits.txt";
constexpr const char* STATS_FILE = "p4_stats.txt";

using IsaMask = uint16_t;
using OpMask = uint32_t;
using WireIdx = uint8_t;
using GateIdx = uint8_t;
using GateCount = uint16_t;
using CircHash = uint64_t;
using StateHash = uint64_t;

static_assert(MAX_SIGNALS <= 255, "WireIdx too small for MAX_SIGNALS");
static_assert(MAX_GATES <= 255, "GateIdx too small for MAX_GATES");
static_assert(MAX_GATES <= 65535, "GateCount too small for MAX_GATES");

constexpr GateCount kInfGateCount = std::numeric_limits<GateCount>::max();
constexpr uint32_t TT4_SPACE = 1u << 16;
constexpr uint8_t P4_PERMS = 24;

// -----------------------------------------------------------------------------
// TTY
// -----------------------------------------------------------------------------
namespace tty {
constexpr auto CLEAR = "\033[K", BOLD = "\033[1m", RESET = "\033[0m";
constexpr auto GREEN = "\033[32m", YELLOW = "\033[33m", CYAN = "\033[36m", RED = "\033[31m";
constexpr auto HIDE = "\033[?25l", SHOW = "\033[?25h";
} // namespace tty

// -----------------------------------------------------------------------------
// Truth tables and P/NP/NPN utilities
// -----------------------------------------------------------------------------
namespace tt {
using TT4 = uint16_t;
constexpr TT4 X0 = 0xAAAA, X1 = 0xCCCC, X2 = 0xF0F0, X3 = 0xFF00;
constexpr TT4 ZERO = 0x0000, ONE = 0xFFFF;
constexpr std::array<TT4, 4> PI = {X0, X1, X2, X3};

constexpr uint16_t COUNT_NPN = 222;
constexpr uint16_t COUNT_NP = 402;
constexpr uint16_t COUNT_P = 3984;

constexpr std::array<std::array<WireIdx, 4>, P4_PERMS> PERMS = {{
    {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1}, {1, 0, 2, 3}, {1, 0, 3, 2},
    {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0}, {2, 0, 1, 3}, {2, 0, 3, 1}, {2, 1, 0, 3}, {2, 1, 3, 0},
    {2, 3, 0, 1}, {2, 3, 1, 0}, {3, 0, 1, 2}, {3, 0, 2, 1}, {3, 1, 0, 2}, {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 2, 1, 0},
}};

inline std::string hex(TT4 v) {
  char buf[8];
  std::snprintf(buf, sizeof(buf), "%04x", v);
  return buf;
}

constexpr inline TT4 permute(TT4 f, const std::array<WireIdx, 4>& p) {
  TT4 r = 0;
  for (int i = 0; i < 16; ++i) {
    if (!((f >> i) & 1))
      continue;
    int ni = 0;
    for (int v = 0; v < 4; ++v)
      if ((i >> v) & 1)
        ni |= (1 << p[v]);
    r |= (1 << ni);
  }
  return r;
}

constexpr inline TT4 negate_inputs(TT4 f, uint8_t mask) {
  if (mask & 1)
    f = ((f & 0x5555) << 1) | ((f >> 1) & 0x5555);
  if (mask & 2)
    f = ((f & 0x3333) << 2) | ((f >> 2) & 0x3333);
  if (mask & 4)
    f = ((f & 0x0F0F) << 4) | ((f >> 4) & 0x0F0F);
  if (mask & 8)
    f = ((f & 0x00FF) << 8) | ((f >> 8) & 0x00FF);
  return f;
}

inline TT4 cofactor(TT4 f, WireIdx var, bool val) {
  constexpr TT4 masks[] = {0x5555, 0x3333, 0x0F0F, 0x00FF};
  constexpr uint8_t shifts[] = {1, 2, 4, 8};
  TT4 lo = f & masks[var], hi = (f >> shifts[var]) & masks[var];
  TT4 chosen = val ? hi : lo;
  return chosen | (chosen << shifts[var]);
}

std::array<TT4, TT4_SPACE> canon_p{};
std::array<TT4, TT4_SPACE> canon_np{};
std::array<TT4, TT4_SPACE> canon_npn{};

std::vector<TT4> reps_p{};
std::array<uint16_t, TT4_SPACE> index_p{};

std::array<std::array<TT4, P4_PERMS>, COUNT_P> perm_p{};
std::array<std::array<TT4, TT4_SPACE>, P4_PERMS> perm4{};
std::vector<std::vector<TT4>> orbit_p{};

void init() {
  // canon_p
  for (uint32_t i = 0; i < TT4_SPACE; ++i) {
    TT4 f = static_cast<TT4>(i);
    TT4 best = f;
    for (const auto& p : PERMS)
      best = std::min(best, permute(f, p));
    canon_p[i] = best;
  }

  // canon_np
  for (uint32_t i = 0; i < TT4_SPACE; ++i) {
    TT4 f = static_cast<TT4>(i);
    TT4 best = std::numeric_limits<TT4>::max();
    for (uint8_t n = 0; n < 16; ++n)
      best = std::min(best, canon_p[negate_inputs(f, n)]);
    canon_np[i] = best;
  }

  // canon_npn
  for (uint32_t i = 0; i < TT4_SPACE; ++i) {
    TT4 f = static_cast<TT4>(i);
    canon_npn[i] = std::min(canon_np[i], canon_np[static_cast<TT4>(~f)]);
  }

  // reps_p (unique canonical reps for P)
  reps_p.reserve(TT4_SPACE);
  std::array<bool, TT4_SPACE> seen{};
  seen.fill(false);

  for (uint32_t i = 0; i < TT4_SPACE; ++i) {
    TT4 c = canon_p[i];
    if (!seen[c]) {
      seen[c] = true;
      reps_p.push_back(c);
    }
  }

  // index_p
  index_p.fill(std::numeric_limits<uint16_t>::max());
  for (uint16_t idx = 0; idx < static_cast<uint16_t>(reps_p.size()); ++idx)
    index_p[reps_p[idx]] = idx;

  for (uint32_t i = 0; i < TT4_SPACE; ++i)
    index_p[i] = index_p[canon_p[i]];

  // perm_p: permute the *representative truth table* (not the index)
  for (uint16_t idx = 0; idx < static_cast<uint16_t>(reps_p.size()); ++idx) {
    TT4 rep = reps_p[idx];
    for (size_t j = 0; j < PERMS.size(); ++j)
      perm_p[idx][j] = permute(rep, PERMS[j]);
  }

  // perm4
  for (uint8_t pi = 0; pi < P4_PERMS; ++pi)
    for (uint32_t f = 0; f < TT4_SPACE; ++f)
      perm4[pi][f] = permute(static_cast<TT4>(f), PERMS[pi]);

  // orbit_p: one orbit per representative
  orbit_p.resize(reps_p.size());
  for (uint16_t idx = 0; idx < static_cast<uint16_t>(reps_p.size()); ++idx) {
    TT4 rep = reps_p[idx];
    auto& o = orbit_p[idx];
    o.reserve(PERMS.size());
    for (const auto& p : PERMS)
      o.push_back(permute(rep, p));
    std::sort(o.begin(), o.end());
    o.erase(std::unique(o.begin(), o.end()), o.end());
  }
}

// For lifted circuits (supercirc)
struct TT8 {
  std::array<uint64_t, 4> w{};
  TT8() { w.fill(0); }
  inline bool get(uint16_t row) const { return (w[row >> 6] >> (row & 63)) & 1ull; }
  inline void set(uint16_t row) { w[row >> 6] |= 1ull << (row & 63); }

  friend TT8 operator~(const TT8& a) {
    TT8 r;
    for (int i = 0; i < 4; ++i)
      r.w[i] = ~a.w[i];
    return r;
  }
  friend TT8 operator&(const TT8& a, const TT8& b) {
    TT8 r;
    for (int i = 0; i < 4; ++i)
      r.w[i] = a.w[i] & b.w[i];
    return r;
  }
  friend TT8 operator|(const TT8& a, const TT8& b) {
    TT8 r;
    for (int i = 0; i < 4; ++i)
      r.w[i] = a.w[i] | b.w[i];
    return r;
  }
  friend TT8 operator^(const TT8& a, const TT8& b) {
    TT8 r;
    for (int i = 0; i < 4; ++i)
      r.w[i] = a.w[i] ^ b.w[i];
    return r;
  }

  static inline const std::array<TT8, 8>& PI() {
    static const std::array<TT8, 8> inp = [] {
      std::array<TT8, 8> r{};
      for (int i = 0; i < 8; ++i) {
        TT8 t;
        for (uint16_t row = 0; row < 256; ++row)
          if ((row >> i) & 1u)
            t.set(row);
        r[i] = t;
      }
      return r;
    }();
    return inp;
  }
};
} // namespace tt

// =============================================================================
// circ: circuit + isa
// =============================================================================
namespace circ {

// Hashing (FNV-1a 64-bit), used for signature
struct Fnv1a64 {
  uint64_t h = 1469598103934665603ull;
  inline void mix(uint64_t x) {
    h ^= x;
    h *= 1099511628211ull;
  }
  inline uint64_t value() const { return h; }
};

// -----------------------------------------------------------------------------
// Circuit IR and gate evaluation
// -----------------------------------------------------------------------------
enum Gate : uint8_t {
  G_NOT = 0,
  G_AND = 1,
  G_OR = 2,
  G_XOR = 3,
  G_ANDN = 4,
  G_ORN = 5,
  G_XNOR = 6,
  G_NAND = 7,
  G_NOR = 8,
  G_MUX = 9,
  G_XOR3 = 10,
  G_BCAX = 11,
  G_PAND = 12,
  G_POR = 13,
  G_PXOR = 14,
  G_PANDN = 15,
  NUM_GATES
};

enum Wire : uint8_t {
  W_DATA = 0,
  W_DATA_FREE = 1, // Commutative
  W_PREDICATE = 2,
  W_ZERO = 3,
};

constexpr const char* GATE_NAMES[] = {"NOT", "AND", "OR",   "XOR",  "ANDN", "ORN", "XNOR", "NAND",
                                      "NOR", "MUX", "XOR3", "BCAX", "PAND", "POR", "PXOR", "PANDN"};

constexpr uint8_t GATE_WIRES[] = {
    W_DATA,      W_ZERO,      W_ZERO,      // NOT
    W_DATA_FREE, W_DATA_FREE, W_ZERO,      // AND
    W_DATA_FREE, W_DATA_FREE, W_ZERO,      // OR
    W_DATA_FREE, W_DATA_FREE, W_ZERO,      // XOR
    W_DATA,      W_DATA,      W_ZERO,      // ANDN
    W_DATA,      W_DATA,      W_ZERO,      // ORN
    W_DATA_FREE, W_DATA_FREE, W_ZERO,      // XNOR
    W_DATA_FREE, W_DATA_FREE, W_ZERO,      // NAND
    W_DATA_FREE, W_DATA_FREE, W_ZERO,      // NOR
    W_DATA,      W_DATA,      W_DATA,      // MUX
    W_DATA_FREE, W_DATA_FREE, W_DATA_FREE, // XOR3
    W_DATA,      W_DATA,      W_DATA,      // BCAX
    W_DATA_FREE, W_DATA_FREE, W_PREDICATE, // PAND
    W_DATA_FREE, W_DATA_FREE, W_PREDICATE, // POR
    W_DATA_FREE, W_DATA_FREE, W_PREDICATE, // PXOR
    W_DATA,      W_DATA,      W_PREDICATE, // PANDN
};

template <typename TT>
inline constexpr TT eval_gate(Gate g, const TT& a, const TT& b, const TT& c) {
  switch (g) {
  case G_NOT: return ~a;
  case G_AND: return a & b;
  case G_OR: return a | b;
  case G_XOR: return a ^ b;
  case G_ANDN: return a & ~b;
  case G_ORN: return a | ~b;
  case G_XNOR: return ~(a ^ b);
  case G_NAND: return ~(a & b);
  case G_NOR: return ~(a | b);
  case G_MUX: return (a & b) | (~a & c);
  case G_XOR3: return a ^ b ^ c;
  case G_BCAX: return (a & ~b) ^ c;
  case G_PAND: return (a & b) & c;
  case G_POR: return (a | b) & c;
  case G_PXOR: return (a ^ b) & c;
  case G_PANDN: return (a & ~b) & c;
  default: return TT{};
  }
}

inline constexpr tt::TT4 eval_gate(Gate g, tt::TT4 a, tt::TT4 b = 0, tt::TT4 c = 0) {
  return eval_gate<tt::TT4>(g, a, b, c);
}
inline constexpr tt::TT8 eval_gate(Gate g, tt::TT8 a, tt::TT8 b = tt::TT8{}, tt::TT8 c = tt::TT8{}) {
  return eval_gate<tt::TT8>(g, a, b, c);
}

struct GateOp {
  Gate op{};
  WireIdx i0{}, i1{}, i2{};
  GateOp() = default;
  GateOp(Gate g, WireIdx a, WireIdx b = 0, WireIdx c = 0) : op(g), i0(a), i1(b), i2(c) {}
};

struct Binding {
  enum Type : uint8_t { Free = 0, Const0 = 1, Const1 = 2 };
  Type type = Const0;
  uint8_t value = 0; // free: input index 0..3
  static Binding free(uint8_t v) { return {Free, v}; }
  static Binding c0() { return {Const0, 0}; }
  static Binding c1() { return {Const1, 1}; }
  bool operator==(const Binding& o) const { return type == o.type && value == o.value; }
};

struct Cofactor {
  tt::TT4 function_p = 0;
  IsaMask isa_optimal_mask = 0; // best gate count for this function+ISA
  std::vector<Binding> bindings;
};

struct Circuit {
  std::vector<GateOp> gates;
  std::vector<Cofactor> cofactors; // empty during search
  uint8_t num_inputs = 4;          // search: 4; supercirc: 8
  IsaMask isa_valid_mask = 0;      // validity mask across ISAs

  GateCount gate_count() const { return static_cast<GateCount>(gates.size()); }

  OpMask gates_used() const {
    OpMask mask = 0;
    for (const auto& g : gates)
      mask |= (OpMask(1) << (g.op));
    return mask;
  }

  CircHash signature() const {
    Fnv1a64 hf;
    hf.mix(num_inputs);
    hf.mix(gates.size());
    if (gates.empty()) {
      if (num_inputs == 0 && !cofactors.empty()) {
        hf.mix(uint64_t(cofactors[0].function_p));
      } else {
        hf.mix(uint64_t(0xC1C1u));
      }
    }
    for (const auto& g : gates) {
      hf.mix(uint64_t((g.op)));
      hf.mix(g.i0);
      hf.mix(g.i1);
      hf.mix(g.i2);
    }
    return hf.value();
  }

  template <class TT, std::size_t MAXV, class Init>
  inline void eval_impl(TT* v, TT empty, int base_wires, Init init) const {
    const std::size_t n = gates.size();
    if (n == 0) {
      v[0] = empty;
      return;
    }

    init(v);

    const auto* gs = gates.data();
    for (std::size_t i = 0; i < n; ++i) {
      const auto& g = gs[i];
      v[base_wires + static_cast<int>(i)] = eval_gate(g.op, v[g.i0], v[g.i1], v[g.i2]);
    }
  }

  std::array<tt::TT4, MAX_SIGNALS> eval_all4() const {
    std::array<tt::TT4, MAX_SIGNALS> v{};
    eval_impl<tt::TT4, MAX_SIGNALS>(v.data(), tt::TT4{0}, BASE_WIRES, [](tt::TT4* v) {
      v[0] = tt::X0;
      v[1] = tt::X1;
      v[2] = tt::X2;
      v[3] = tt::X3;
    });
    return v;
  }

  std::array<tt::TT8, MAX_SIGNALS_COV> eval_all8() const {
    std::array<tt::TT8, MAX_SIGNALS_COV> v{};
    eval_impl<tt::TT8, MAX_SIGNALS_COV>(v.data(), tt::TT8{}, num_inputs, [](tt::TT8* v) {
      auto PI = tt::TT8::PI();
      for (int i = 0; i < 8; ++i)
        v[i] = PI[i];
    });
    return v;
  }

  tt::TT4 eval4() const {
    const auto v = eval_all4();
    return v[(BASE_WIRES - 1) + static_cast<int>(gates.size())];
  }

  tt::TT8 eval8() const {
    const auto v = eval_all8();
    return v[(num_inputs - 1) + static_cast<int>(gates.size())];
  }

  std::string to_string() const {
    auto name = [&](WireIdx i, uint8_t w) -> std::string {
      if (i < num_inputs)
        return (w == W_PREDICATE ? "p" : "x") + std::to_string(i);
      return "g" + std::to_string(i - num_inputs);
    };

    if (gates.empty()) {
      if (num_inputs == 0) {
        return cofactors[0].function_p == tt::ZERO ? "0;" : "1;";
      } else {
        return "x0;";
      }
    }

    std::ostringstream ss;
    for (size_t i = 0; i < gates.size(); ++i) {
      if (i)
        ss << ' ';
      const auto& g = gates[i];
      uint8_t w0 = GATE_WIRES[g.op * 3];
      uint8_t w1 = GATE_WIRES[g.op * 3 + 1];
      uint8_t w2 = GATE_WIRES[g.op * 3 + 2];
      ss << "g" << i << " = " << GATE_NAMES[(g.op)] << "(" << name(g.i0, w0);
      if (GATE_WIRES[g.op * 3 + 1] != W_ZERO)
        ss << ", " << name(g.i1, w1);
      if (GATE_WIRES[g.op * 3 + 2] != W_ZERO)
        ss << ", " << name(g.i2, w2);
      ss << ");";
    }
    return ss.str();
  }

  size_t unique_function_count() const {
    std::unordered_set<tt::TT4> s;
    for (auto& c : cofactors)
      s.insert(c.function_p);
    return s.size();
  }
};

// Helper: evaluate circuit and return all intermediate signal truth tables
inline std::array<tt::TT4, MAX_SIGNALS> eval_all4(const Circuit& c) {
  std::array<tt::TT4, MAX_SIGNALS> v{};
  v[0] = tt::X0;
  v[1] = tt::X1;
  v[2] = tt::X2;
  v[3] = tt::X3;

  for (std::size_t i = 0; i < c.gates.size(); ++i) {
    const auto& g = c.gates[i];
    v[BASE_WIRES + i] = eval_gate(g.op, v[g.i0], v[g.i1], v[g.i2]);
  }
  return v;
}

// -----------------------------------------------------------------------------
// Circuit Canonicalization
// -----------------------------------------------------------------------------
inline std::array<WireIdx, 8> canonicalize(circ::Circuit& c, bool relaxed = false) {
  const int NI = c.num_inputs;
  const int NG = static_cast<int>(c.gates.size());
  std::array<WireIdx, 8> perm_out{};
  for (int i = 0; i < 8; ++i)
    perm_out[i] = static_cast<WireIdx>(i);
  if (NG == 0 || NI <= 1)
    return perm_out;

  auto normalize_gate = [](GateOp& g) {
    const uint8_t* wc = &GATE_WIRES[g.op * 3];
    if (wc[1] == W_ZERO)
      g.i1 = 0;
    if (wc[2] == W_ZERO)
      g.i2 = 0;
    WireIdx* w[3] = {&g.i0, &g.i1, &g.i2};
    int fp[3], fc = 0;
    for (int p = 0; p < 3; ++p)
      if (wc[p] == W_DATA_FREE)
        fp[fc++] = p;
    for (int i = 0; i < fc - 1; ++i)
      for (int j = i + 1; j < fc; ++j)
        if (*w[fp[i]] > *w[fp[j]])
          std::swap(*w[fp[i]], *w[fp[j]]);
  };

  auto pack_gate = [](const GateOp& g) -> uint32_t {
    return (uint32_t(g.op) << 24) | (uint32_t(g.i0) << 16) | (uint32_t(g.i1) << 8) | uint32_t(g.i2);
  };

  auto is_ready = [NI](const GateOp& g, const bool* placed) {
    const uint8_t* wc = &GATE_WIRES[g.op * 3];
    WireIdx in[3] = {g.i0, g.i1, g.i2};
    for (int p = 0; p < 3; ++p)
      if (wc[p] != W_ZERO && in[p] >= NI && !placed[in[p] - NI])
        return false;
    return true;
  };

  auto topo_rewrite = [&](const WireIdx* in_map) {
    bool placed[MAX_GATES] = {};
    WireIdx gate_out[MAX_GATES] = {};
    std::vector<GateOp> out;
    out.reserve(NG);
    for (int step = 0; step < NG; ++step) {
      int best = -1;
      uint32_t best_key = ~0u;
      GateOp best_g{};
      for (int gi = 0; gi < NG; ++gi) {
        if (placed[gi] || !is_ready(c.gates[gi], placed))
          continue;
        const auto& g = c.gates[gi];
        const uint8_t* wc = &GATE_WIRES[g.op * 3];
        GateOp cand;
        cand.op = g.op;
        cand.i0 = (g.i0 < NI) ? in_map[g.i0] : gate_out[g.i0 - NI];
        cand.i1 = (wc[1] != W_ZERO) ? ((g.i1 < NI) ? in_map[g.i1] : gate_out[g.i1 - NI]) : 0;
        cand.i2 = (wc[2] != W_ZERO) ? ((g.i2 < NI) ? in_map[g.i2] : gate_out[g.i2 - NI]) : 0;
        normalize_gate(cand);
        uint32_t key = pack_gate(cand);
        if (key < best_key) {
          best = gi;
          best_key = key;
          best_g = cand;
        }
      }
      placed[best] = true;
      gate_out[best] = static_cast<WireIdx>(NI + step);
      out.push_back(best_g);
    }
    return out;
  };

  auto mix = [](uint64_t h, uint64_t x) { return (h ^ x) * 1099511628211ull; };

  std::array<bool, 8> used{};
  for (int gi = 0; gi < NG; ++gi) {
    const auto& g = c.gates[gi];
    const uint8_t* wc = &GATE_WIRES[g.op * 3];
    WireIdx in[3] = {g.i0, g.i1, g.i2};
    for (int p = 0; p < 3; ++p)
      if (wc[p] != W_ZERO && in[p] < NI)
        used[in[p]] = true;
  }

  std::array<uint64_t, 8> lab;
  for (int i = 0; i < NI; ++i)
    lab[i] = used[i] ? 0xabcdef0123456789ull : 0x9876543210fedcbaull;

  for (int round = 0; round < 4; ++round) {
    std::array<uint64_t, MAX_GATES> glab;
    for (int gi = 0; gi < NG; ++gi) {
      const auto& g = c.gates[gi];
      const uint8_t* wc = &GATE_WIRES[g.op * 3];
      WireIdx in[3] = {g.i0, g.i1, g.i2};
      uint64_t h = mix(0, uint64_t(g.op));
      uint64_t free_h[3];
      int fc = 0;
      for (int p = 0; p < 3; ++p) {
        if (wc[p] == W_ZERO)
          continue;
        uint64_t inp_lab = (in[p] < NI) ? lab[in[p]] : glab[in[p] - NI];
        uint64_t f = (uint64_t(wc[p]) << 56) ^ inp_lab;
        if (wc[p] == W_DATA_FREE)
          free_h[fc++] = f;
        else
          h = mix(h, f);
      }
      std::sort(free_h, free_h + fc);
      for (int i = 0; i < fc; ++i)
        h = mix(h, free_h[i]);
      glab[gi] = h;
    }
    std::array<uint64_t, 8> nlab;
    for (int inp = 0; inp < NI; ++inp) {
      uint64_t h = lab[inp];
      uint64_t feats[MAX_GATES * 3];
      int fc = 0;
      for (int gi = 0; gi < NG; ++gi) {
        const auto& g = c.gates[gi];
        const uint8_t* wc = &GATE_WIRES[g.op * 3];
        WireIdx in[3] = {g.i0, g.i1, g.i2};
        for (int p = 0; p < 3; ++p) {
          if (wc[p] == W_ZERO || in[p] != inp)
            continue;
          uint64_t kind = (wc[p] == W_DATA_FREE) ? 0ull : uint64_t(wc[p]);
          feats[fc++] = (kind << 56) ^ glab[gi];
        }
      }
      std::sort(feats, feats + fc);
      for (int i = 0; i < fc; ++i)
        h = mix(h, feats[i]);
      nlab[inp] = h;
    }
    lab = nlab;
  }

  std::array<WireIdx, 8> sorted;
  for (int i = 0; i < NI; ++i)
    sorted[i] = i;
  std::stable_sort(sorted.begin(), sorted.begin() + NI, [&](WireIdx a, WireIdx b) {
    if (used[a] != used[b])
      return used[a] > used[b];
    if (lab[a] != lab[b])
      return lab[a] < lab[b];
    return a < b;
  });

  int num_used = 0;
  for (int i = 0; i < NI; ++i)
    if (used[sorted[i]])
      num_used++;

  std::vector<int> boundaries = {0};
  for (int i = 1; i < num_used; ++i)
    if (lab[sorted[i]] != lab[sorted[i - 1]])
      boundaries.push_back(i);
  boundaries.push_back(num_used);
  int num_classes = static_cast<int>(boundaries.size()) - 1;

  std::array<WireIdx, 8> perm;
  for (int i = num_used; i < NI; ++i)
    perm[sorted[i]] = static_cast<WireIdx>(i);

  if (relaxed) {
    for (int i = 0; i < num_used; ++i)
      perm[sorted[i]] = static_cast<WireIdx>(i);
    c.gates = topo_rewrite(perm.data());
    for (int i = 0; i < NI; ++i)
      perm_out[i] = perm[i];
    return perm_out;
  }

  std::vector<std::vector<WireIdx>> class_members(num_classes);
  for (int ci = 0; ci < num_classes; ++ci) {
    for (int i = boundaries[ci]; i < boundaries[ci + 1]; ++i)
      class_members[ci].push_back(sorted[i]);
    std::sort(class_members[ci].begin(), class_members[ci].end());
  }

  std::array<uint32_t, MAX_GATES> best_key;
  best_key.fill(~0u);
  std::vector<GateOp> best_gates;

  std::function<void(int)> enumerate = [&](int ci) {
    if (ci == num_classes) {
      auto cand = topo_rewrite(perm.data());
      bool dominated = false, dominated_by = false;
      for (int i = 0; i < NG && !dominated && !dominated_by; ++i) {
        uint32_t k = pack_gate(cand[i]);
        if (k < best_key[i])
          dominated = true;
        else if (k > best_key[i])
          dominated_by = true;
      }
      if (dominated || best_gates.empty()) {
        for (int i = 0; i < NG; ++i)
          best_key[i] = pack_gate(cand[i]);
        best_gates = std::move(cand);
      }
      return;
    }
    int start = boundaries[ci];
    auto& cm = class_members[ci];
    do {
      for (size_t i = 0; i < cm.size(); ++i)
        perm[cm[i]] = static_cast<WireIdx>(start + i);
      enumerate(ci + 1);
    } while (std::next_permutation(cm.begin(), cm.end()));
  };

  enumerate(0);
  c.gates = std::move(best_gates);
  for (int i = 0; i < NI; ++i)
    perm_out[i] = perm[i];
  return perm_out;
}

// -----------------------------------------------------------------------------
// ISA sets
// -----------------------------------------------------------------------------
enum ISA : uint8_t {
  x86 = 0,
  x86_BMI,
  AVX512_PRED,
  ARM,
  SVE_PRED,
  NEON,
  PRE_SVE2,
  NEON_SHA3,

  // Post-supercirc targets (predicate gates allowed structurally)
  SVE,
  SVE2,
  NEON_SVE,
  NEON_SVE2,

  NUM_ISAS
};

constexpr int NUM_SEARCH_ISAS = NUM_ISAS - 4;

constexpr const char* NAMES[] = {"x86",      "x86_BMI",   "AVX512_PRED", "ARM",  "SVE_PRED", "NEON",
                                 "PRE_SVE2", "NEON_SHA3", "SVE",         "SVE2", "NEON_SVE", "NEON_SVE2"};

constexpr OpMask GATE_MASK_COMMON = (1u << G_NOT) | (1u << G_AND) | (1u << G_OR) | (1u << G_XOR);
constexpr OpMask GATE_MASK_FEAT_SHA3 = (1u << G_XNOR) | (1u << G_XOR3) | (1u << G_BCAX);
constexpr OpMask GATE_MASK_FEAT_PRED = (1u << G_PAND) | (1u << G_POR) | (1u << G_PXOR) | (1u << G_PANDN);

// clang-format off
constexpr OpMask GATE_MASK[NUM_ISAS] = {
    /* x86 */         GATE_MASK_COMMON,
    /* x86_BMI */     GATE_MASK_COMMON | (1u << G_ANDN),
    /* AVX512_PRED */ GATE_MASK_COMMON | (1u << G_ANDN) | (1u << G_XNOR),
    /* ARM */         GATE_MASK_COMMON | (1u << G_ANDN) | (1u << G_ORN) | (1u << G_XNOR),
    /* SVE_PRED */    GATE_MASK_COMMON | (1u << G_ANDN) | (1u << G_ORN) | (1u << G_NAND) | (1u << G_NOR),
    /* NEON */        GATE_MASK_COMMON | (1u << G_ANDN) | (1u << G_ORN) | (1u << G_MUX),
    /* PRE_SVE2 */    GATE_MASK_COMMON | (1u << G_ANDN) | (1u << G_MUX) | GATE_MASK_FEAT_SHA3,
    /* NEON_SHA3 */   GATE_MASK_COMMON | (1u << G_ANDN) | (1u << G_ORN) | (1u << G_MUX) | GATE_MASK_FEAT_SHA3,

    /* SVE */         GATE_MASK_COMMON | (1u << G_ANDN) | GATE_MASK_FEAT_PRED,
    /* SVE2 */        GATE_MASK_COMMON | (1u << G_ANDN) | (1u << G_MUX) | GATE_MASK_FEAT_SHA3 | GATE_MASK_FEAT_PRED,
    /* NEON_SVE */    GATE_MASK_COMMON | (1u << G_ANDN) | (1u << G_ORN) | (1u << G_MUX) | GATE_MASK_FEAT_PRED,
    /* NEON_SVE2 */   GATE_MASK_COMMON | (1u << G_ANDN) | (1u << G_ORN) | (1u << G_MUX) | GATE_MASK_FEAT_SHA3 | GATE_MASK_FEAT_PRED,
};
// clang-format on

inline constexpr bool gate_valid(ISA isa, Gate g) {
  return (GATE_MASK[isa] >> (g)) & 1u;
}

inline bool circuit_valid_for(const Circuit& c, ISA isa) {
  const OpMask allowed = GATE_MASK[isa];
  return (c.gates_used() & ~allowed) == 0;
}

inline IsaMask circuit_isa_mask(OpMask used) {
  IsaMask mask = 0;
  for (int i = 0; i < NUM_ISAS; ++i)
    if ((used & ~GATE_MASK[i]) == 0)
      mask |= (IsaMask(1) << i);
  return mask;
}

} // namespace circ

// =============================================================================
// search: ISA-local enumeration
// =============================================================================
namespace search {

using tt::TT4;

// -----------------------------------------------------------------------------
// Progress UI
// -----------------------------------------------------------------------------
namespace prog {

struct State {
  uint64_t circuits{0}, states{0};
  GateCount current_gate_count{0};
  unsigned newly_solved{0};
  size_t frontier_size{0};
  int current_phase{0};

  std::chrono::steady_clock::time_point start_time{};
  std::array<double, circ::NUM_SEARCH_ISAS> phase_durations{};
  std::array<unsigned, circ::NUM_SEARCH_ISAS> solved_per_isa{};
  unsigned total_classes{0};

  std::chrono::steady_clock::time_point last_render{};
  bool first = true;
  int last_lines = 0;
};

static inline int count_newlines(const std::string& s) {
  int n = 0;
  for (char c : s)
    if (c == '\n')
      ++n;
  return n;
}

static inline void clear_previous(State& s) {
  if (s.first || s.last_lines <= 0)
    return;

  std::cout << "\033[" << s.last_lines << "A";
  for (int i = 0; i < s.last_lines; ++i) {
    std::cout << tty::CLEAR << "\r";
    if (i < s.last_lines - 1)
      std::cout << "\033[1B";
  }
  if (s.last_lines > 1)
    std::cout << "\033[" << (s.last_lines - 1) << "A";
}

inline void render(State& s, bool final = false) {
  auto now = std::chrono::steady_clock::now();
  if (!final && s.last_render.time_since_epoch().count() != 0) {
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - s.last_render).count();
    if (ms < (int)UPDATE_MS)
      return;
  }
  s.last_render = now;

  double elapsed = std::chrono::duration<double>(now - s.start_time).count();
  uint64_t circs = s.circuits;

  std::ostringstream ss;
  ss << tty::HIDE << tty::BOLD << "P-4 Circuit Synthesis" << tty::RESET;
  if (final)
    ss << " " << tty::GREEN << "[Complete]" << tty::RESET;
  ss << tty::CLEAR << "\n" << std::string(60, '-') << tty::CLEAR << "\n";

  for (int i = circ::NUM_SEARCH_ISAS - 1; i >= 0; --i) {
    unsigned solved = s.solved_per_isa[i];
    double pct = (s.total_classes == 0) ? 0.0 : (100.0 * double(solved) / double(s.total_classes));

    ss << " " << std::left << std::setw(14) << circ::NAMES[i] << ": " << std::right << std::setw(4) << solved << "/"
       << s.total_classes << " (" << std::fixed << std::setprecision(2) << pct << "%)";

    if (solved >= s.total_classes)
      ss << " " << tty::GREEN << "\u2713" << tty::RESET;
    else
      ss << " " << tty::YELLOW << "[" << (s.total_classes - solved) << " unsolved]" << tty::RESET;

    double dur = s.phase_durations[i];
    if (dur > 0)
      ss << " (" << int(dur * 1000) << "ms)";
    else if (!final && i == s.current_phase)
      ss << " " << tty::CYAN << "(running)" << tty::RESET;

    ss << tty::CLEAR << "\n";
  }

  ss << std::string(60, '-') << tty::CLEAR << "\n";
  ss << " ISA:       " << (final ? "Complete" : circ::NAMES[s.current_phase]) << tty::CLEAR << "\n";
  ss << " Gates:     " << s.current_gate_count << " (+" << s.newly_solved << " solved)" << tty::CLEAR << "\n";
  ss << " Circuits:  " << std::fixed << std::setprecision(1) << (circs / 1e6) << "M ("
     << uint64_t(circs / std::max(elapsed, 0.001)) << "/s)" << tty::CLEAR << "\n";
  ss << " Frontier:  " << s.frontier_size << tty::CLEAR << "\n";
  ss << " Time:      " << int(elapsed) << "s" << tty::CLEAR << "\n";

  std::string out = ss.str();
  int lines = count_newlines(out);

  clear_previous(s);
  std::cout << out << std::flush;

  s.last_lines = lines;
  s.first = false;

  if (final)
    std::cout << tty::SHOW << std::flush;
}

} // namespace prog

// -----------------------------------------------------------------------------
// Canonical state hash: multiset of derived signals under a single global P-perm
// -----------------------------------------------------------------------------
struct Node {
  std::array<TT4, MAX_SIGNALS> values{};
  std::array<circ::GateOp, MAX_GATES> ops{};
  StateHash canonical_hash = 0;
  OpMask gates_used = 0;
  WireIdx num_signals = BASE_WIRES;

  Node() {
    values[0] = tt::X0;
    values[1] = tt::X1;
    values[2] = tt::X2;
    values[3] = tt::X3;
    finalize_hash();
  }

  inline WireIdx derived_count() const { return (num_signals >= BASE_WIRES) ? WireIdx(num_signals - BASE_WIRES) : 0; }

  circ::Circuit to_circuit() const {
    circ::Circuit c;
    c.num_inputs = 4;
    const WireIdx k = derived_count();
    c.gates.reserve(k);
    for (WireIdx i = 0; i < k; ++i)
      c.gates.push_back(ops[i]);
    return c;
  }

  void finalize_hash() {
    const WireIdx k = derived_count();

    circ::Fnv1a64 hf;
    hf.mix(uint64_t(num_signals));
    hf.mix(uint64_t(k));

    if (k == 0) {
      canonical_hash = hf.value();
      return;
    }

    std::array<TT4, MAX_GATES> best{};
    bool have_best = false;

    for (uint8_t pi = 0; pi < P4_PERMS; ++pi) {
      std::array<TT4, MAX_GATES> tmp{};
      for (WireIdx i = 0; i < k; ++i)
        tmp[i] = tt::perm4[pi][values[BASE_WIRES + i]];

      std::sort(tmp.begin(), tmp.begin() + k);

      if (!have_best || std::lexicographical_compare(tmp.begin(), tmp.begin() + k, best.begin(), best.begin() + k)) {
        for (WireIdx i = 0; i < k; ++i)
          best[i] = tmp[i];
        have_best = true;
      }
    }

    for (WireIdx i = 0; i < k; ++i)
      hf.mix(uint64_t(best[i]));
    canonical_hash = hf.value();
  }

  void add_signal(TT4 result, const circ::GateOp& op) {
    ops[derived_count()] = op;
    values[num_signals] = result;
    ++num_signals;
    finalize_hash();
  }

  StateHash hash() const { return canonical_hash; }
};

// -----------------------------------------------------------------------------
// Solution storage (per-ISA): best gatecount + tied-best circuits + signature dedup
// -----------------------------------------------------------------------------
struct SolutionStore {
  struct Entry {
    GateCount best = kInfGateCount;
    std::vector<circ::Circuit> sols;   // tied-best (cap MAX_SOLUTIONS)
    std::unordered_set<CircHash> sigs; // dedup signatures for stored canonical circuits
  };

  circ::ISA target;
  prog::State& st;

  std::vector<Entry> tab;
  std::vector<GateCount> ub;

  explicit SolutionStore(circ::ISA t, prog::State& s)
      : target(t), st(s), tab(tt::COUNT_P), ub(tt::COUNT_P, kInfGateCount) {}

  static inline uint16_t idx_of(TT4 p_rep) { return tt::index_p[p_rep]; }

  // Returns true iff this call newly solved a previously-unsolved P-class.
  bool store_solution(TT4 f, GateCount gate_count, const Node& node) {
    const TT4 rep = tt::canon_p[f];
    const uint16_t pidx = idx_of(rep);
    if (pidx >= tt::COUNT_P)
      return false;

    // UB prune (ISA-local)
    if (ub[pidx] < gate_count)
      return false;
    if (ub[pidx] > gate_count)
      ub[pidx] = gate_count;

    Entry& e = tab[pidx];
    if (e.best < gate_count)
      return false;

    const bool newly_solved = (e.best == kInfGateCount);

    if (gate_count < e.best) {
      e.best = gate_count;
      e.sols.clear();
      e.sigs.clear();
    }

    if (e.best != gate_count)
      return false;
    if (e.sols.size() >= MAX_SOLUTIONS)
      return false;

    circ::Circuit c = node.to_circuit();
    circ::canonicalize(c);

    c.cofactors.clear();
    circ::Cofactor cf;
    cf.function_p = rep;
    cf.isa_optimal_mask = 0;
    cf.bindings.resize(4);
    for (uint8_t i = 0; i < 4; ++i)
      cf.bindings[i] = circ::Binding::free(i);
    c.cofactors.push_back(std::move(cf));

    c.isa_valid_mask = circ::circuit_isa_mask(c.gates_used());

    const CircHash sig = c.signature();
    if (!e.sigs.insert(sig).second)
      return false;

    e.sols.push_back(std::move(c));
    return newly_solved;
  }

  void seed_zero_gate(TT4 rep, uint8_t num_inputs, std::vector<circ::Binding> bindings) {
    const uint16_t pidx = idx_of(rep);
    if (pidx >= tt::COUNT_P)
      return;

    Entry& e = tab[pidx];
    if (e.best == kInfGateCount) {
      e.best = 0;
      e.sols.clear();
      e.sigs.clear();
      st.solved_per_isa[int(target)]++;
    }

    if (e.best != 0 || e.sols.size() >= MAX_SOLUTIONS)
      return;

    circ::Circuit c;
    c.num_inputs = num_inputs;
    c.gates.clear();
    c.cofactors.clear();

    circ::Cofactor cf;
    cf.function_p = rep;
    cf.isa_optimal_mask = 0;
    cf.bindings = std::move(bindings);
    c.cofactors.push_back(std::move(cf));

    c.isa_valid_mask = circ::circuit_isa_mask(c.gates_used());

    const CircHash sig = c.signature();
    if (e.sigs.insert(sig).second)
      e.sols.push_back(std::move(c));
  }

  void seed_basics() {
    // Constant 0
    seed_zero_gate(tt::ZERO, /*num_inputs=*/0, /*bindings=*/{});
    // Constant 1
    seed_zero_gate(tt::ONE, /*num_inputs=*/0, /*bindings=*/{});
    // Wire x0 (represented as num_inputs=1, no gates)
    seed_zero_gate(tt::canon_p[tt::X0], /*num_inputs=*/1, /*bindings=*/{circ::Binding::free(0)});
  }

  std::vector<circ::Circuit> extract() {
    std::vector<circ::Circuit> out;
    out.reserve(60000);
    for (TT4 rep : tt::reps_p) {
      const uint16_t pidx = idx_of(rep);
      if (pidx >= tt::COUNT_P)
        continue;
      for (auto& c : tab[pidx].sols)
        out.push_back(std::move(c));
    }
    return out;
  }
};

// -----------------------------------------------------------------------------
// Helper: compute isa_valid_mask + isa_optimal_mask.
// -----------------------------------------------------------------------------
static void set_optimal_and_valid(std::vector<circ::Circuit>& circs) {
  using BestTable = std::array<std::array<GateCount, tt::COUNT_P>, circ::NUM_ISAS>;
  BestTable best{};
  for (auto& row : best)
    row.fill(kInfGateCount);

  // First pass: recompute validity + best gatecount per (ISA, P-rep)
  for (auto& c : circs) {
    c.isa_valid_mask = circ::circuit_isa_mask(c.gates_used());
    const GateCount gc = c.gate_count();

    for (const auto& cf : c.cofactors) {
      const uint16_t pidx = tt::index_p[cf.function_p];
      if (pidx >= tt::COUNT_P)
        continue;

      for (int isa = 0; isa < circ::NUM_ISAS; ++isa) {
        if (!(c.isa_valid_mask & (IsaMask(1) << isa)))
          continue;
        auto& cell = best[isa][pidx];
        if (gc < cell)
          cell = gc;
      }
    }
  }

  // Second pass: set optimal mask, drop non-optimal cofactors/circuits
  std::vector<circ::Circuit> out;
  out.reserve(circs.size());

  for (auto& c : circs) {
    c.isa_valid_mask = circ::circuit_isa_mask(c.gates_used());

    std::vector<circ::Cofactor> kept;
    kept.reserve(c.cofactors.size());

    for (auto& cf : c.cofactors) {
      const uint16_t pidx = tt::index_p[cf.function_p];
      if (pidx >= tt::COUNT_P)
        continue;

      IsaMask opt = 0;
      for (int isa = 0; isa < circ::NUM_ISAS; ++isa) {
        if (!(c.isa_valid_mask & (IsaMask(1) << isa)))
          continue;
        if (best[isa][pidx] == c.gate_count())
          opt |= (IsaMask(1) << isa);
      }
      if (opt) {
        cf.isa_optimal_mask = opt;
        kept.push_back(std::move(cf));
      }
    }

    if (!kept.empty()) {
      c.cofactors = std::move(kept);
      out.push_back(std::move(c));
    }
  }

  circs = std::move(out);
}

// -----------------------------------------------------------------------------
// search::run_isa: BFS for one target ISA.
// -----------------------------------------------------------------------------
std::vector<circ::Circuit> run_isa(circ::ISA target, prog::State& st) {
  st.solved_per_isa[int(target)] = 0;

  SolutionStore store(target, st);
  store.seed_basics();

  if (st.solved_per_isa[int(target)] >= st.total_classes)
    return store.extract();

  std::mt19937 rng{std::random_device{}()};
  std::vector<Node> frontier;
  frontier.reserve(1024);
  frontier.push_back(Node());

  // scratch reused across depths
  std::unordered_set<StateHash> seen;
  std::vector<Node> next;

  auto phase_start = std::chrono::steady_clock::now();

  auto try_add = [&](const Node& state, GateCount gate_count, circ::Gate g, WireIdx a, WireIdx b, WireIdx c) {
    TT4 result = circ::eval_gate(g, state.values[a], state.values[b], state.values[c]);
    st.circuits++;

    const TT4 rep = tt::canon_p[result];
    const uint16_t pidx = SolutionStore::idx_of(rep);
    if (pidx >= tt::COUNT_P)
      return;

    // UB prune
    if (store.ub[pidx] < gate_count)
      return;

    // avoid duplicate signals within the state
    for (WireIdx i = 0; i < state.num_signals; ++i)
      if (state.values[i] == result)
        return;

    Node ns = state;
    ns.add_signal(result, circ::GateOp(g, a, b, c));
    ns.gates_used |= (OpMask(1) << uint32_t(g));

    const unsigned before = st.solved_per_isa[int(target)];
    if (store.store_solution(result, gate_count, ns))
      st.solved_per_isa[int(target)]++; // newly solved
    const unsigned after = st.solved_per_isa[int(target)];
    if (after > before)
      st.newly_solved += (after - before);

    if (seen.insert(ns.hash()).second)
      next.push_back(std::move(ns));
  };

  constexpr circ::Gate kBinComm[] = {circ::G_AND, circ::G_OR, circ::G_XOR, circ::G_NAND, circ::G_NOR, circ::G_XNOR};
  constexpr circ::Gate kBinAsym[] = {circ::G_ANDN, circ::G_ORN};
  constexpr circ::Gate kTernDistinct[] = {circ::G_MUX, circ::G_BCAX};

  for (GateCount gate_count = 1; gate_count <= (GateCount)MAX_GATES; ++gate_count) {
    if (st.solved_per_isa[int(target)] >= st.total_classes)
      break;

    st.current_gate_count = gate_count;
    st.newly_solved = 0;
    st.frontier_size = frontier.size();

    seen.clear();
    next.clear();
    seen.reserve(frontier.size() * 8 + 1024);
    next.reserve(frontier.size() * 2 + 1024);

    for (size_t i = 0; i < frontier.size(); ++i) {
      prog::render(st);
      if (st.solved_per_isa[int(target)] >= st.total_classes)
        break;

      const Node& state = frontier[i];
      const unsigned n = state.num_signals;
      st.states++;

      // NOT
      if (circ::gate_valid(target, circ::G_NOT)) {
        for (unsigned a = 0; a < n; ++a)
          try_add(state, gate_count, circ::G_NOT, (WireIdx)a, 0, 0);
      }

      // binary commutative (a<b)
      for (circ::Gate g : kBinComm) {
        if (!circ::gate_valid(target, g))
          continue;
        for (unsigned a = 0; a < n; ++a)
          for (unsigned b = a + 1; b < n; ++b)
            try_add(state, gate_count, g, (WireIdx)a, (WireIdx)b, 0);
      }

      // binary asymmetric (a!=b)
      for (circ::Gate g : kBinAsym) {
        if (!circ::gate_valid(target, g))
          continue;
        for (unsigned a = 0; a < n; ++a)
          for (unsigned b = 0; b < n; ++b)
            if (a != b)
              try_add(state, gate_count, g, (WireIdx)a, (WireIdx)b, 0);
      }

      // ternary distinct (a,b,c all distinct)
      for (circ::Gate g : kTernDistinct) {
        if (!circ::gate_valid(target, g))
          continue;
        for (unsigned a = 0; a < n; ++a)
          for (unsigned b = 0; b < n; ++b)
            if (a != b)
              for (unsigned c = 0; c < n; ++c)
                if (a != c && b != c)
                  try_add(state, gate_count, g, (WireIdx)a, (WireIdx)b, (WireIdx)c);
      }

      // XOR3 (a<b<c)
      if (circ::gate_valid(target, circ::G_XOR3)) {
        for (unsigned a = 0; a < n; ++a)
          for (unsigned b = a + 1; b < n; ++b)
            for (unsigned c = b + 1; c < n; ++c)
              try_add(state, gate_count, circ::G_XOR3, (WireIdx)a, (WireIdx)b, (WireIdx)c);
      }
    }

    if (next.size() > MAX_FRONTIER) {
      std::shuffle(next.begin(), next.end(), rng);
      next.resize(MAX_FRONTIER);
    }

    frontier = std::move(next);
    if (frontier.empty())
      break;
  }

  st.phase_durations[int(target)] =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - phase_start).count();

  return store.extract();
}

// -----------------------------------------------------------------------------
// search::run: driver over run_isa. Accumulate, canonicalize, dedup, then
// compute optimal/legal masks from the final circuit vector.
// -----------------------------------------------------------------------------
std::vector<circ::Circuit> run(prog::State& st) {
  st.start_time = std::chrono::steady_clock::now();
  st.phase_durations.fill(0.0);
  st.solved_per_isa.fill(0);
  st.circuits = 0;
  st.states = 0;
  st.first = true;
  st.last_lines = 0;
  st.last_render = {};

  std::vector<circ::Circuit> all;
  all.reserve(250000);

  auto sig_less = [](const circ::Circuit& a, const circ::Circuit& b) { return a.signature() < b.signature(); };
  auto sig_eq = [](const circ::Circuit& a, const circ::Circuit& b) { return a.signature() == b.signature(); };

  for (int phase = circ::NUM_SEARCH_ISAS - 1; phase >= 0; --phase) {
    st.current_phase = phase;

    auto batch = run_isa(static_cast<circ::ISA>(phase), st);

    // Append and dedup by signature
    for (auto& c : batch) {
      circ::canonicalize(c);
      c.isa_valid_mask = circ::circuit_isa_mask(c.gates_used());
      all.push_back(std::move(c));
    }

    std::sort(all.begin(), all.end(), sig_less);
    all.erase(std::unique(all.begin(), all.end(), sig_eq), all.end());

    prog::render(st);
  }

  st.current_phase = circ::NUM_SEARCH_ISAS;
  prog::render(st, true);

  set_optimal_and_valid(all);

  // One last canonical+dedup pass
  for (auto& c : all) {
    circ::canonicalize(c);
    c.isa_valid_mask = circ::circuit_isa_mask(c.gates_used());
  }
  std::sort(all.begin(), all.end(), sig_less);
  all.erase(std::unique(all.begin(), all.end(), sig_eq), all.end());

  return all;
}

} // namespace search

// =============================================================================
// Supercircuit
// =============================================================================
namespace supercirc {

using tt::TT4;
using tt::TT8;

// ------------------------------- Config --------------------------------------

constexpr int LIFT_MAX_GATECOUNT = 12;
constexpr int ATTEMPTS_PER_BASE = 16;
constexpr int N_LIFT_GATES_MAX = 4;

constexpr size_t MAX_BASES_PER_PAIR = 300000; // safety cap

constexpr size_t FLUSH_BATCH_TARGET = 120000;
constexpr size_t FLUSH_BATCH_HARD_MAX = 180000;

constexpr int MAX_COFACTORS_PER_CIRC = 256;

// Seed expansion for the existing pool (phase0).
constexpr int SEED_MIN_BUDGET_EXISTING = 12;
constexpr int SEED_MAX_BUDGET_EXISTING = 36;
constexpr int SEED_MAX_ADD_EXISTING = 160;

// Seed-on-create during lifting (phase1).
constexpr int SEED_BUDGET_ON_CREATE = 28;
constexpr int SEED_MAX_ADD_ON_CREATE = 160;
constexpr int SEED_COF_CAP_ON_CREATE = 256;

// Extra per-flush heavy enumeration for the best-looking circuits in that flush.
constexpr size_t EXTRA_ENUM_TOP_M_PER_FLUSH = 4096; // (0 disables)
constexpr int EXTRA_ENUM_BUDGET = 192;              // extra trials on those top-M
constexpr int EXTRA_ENUM_MAX_ADD = 384;             // try to fill cofactor slots

// Sampling policy for binding: alias vs set 0/1.
constexpr int COF_P_ALIAS_PCT = 45;

constexpr bool ENABLE_SIMPLIFY = true;
constexpr int SIMPLIFY_MAX_ITERS = 8;
constexpr int SIMPLIFY_MAX_VARIANTS = 16;

constexpr size_t POOL_GUARD_SIZE = 2500000;
constexpr size_t POOL_GUARD_CAP = 2100000;

// ------------------------------ Utilities ------------------------------------

using BestTable = std::array<std::array<GateCount, TT4_SPACE>, circ::NUM_ISAS>;

static inline CircHash raw_signature(const circ::Circuit& c) {
  circ::Fnv1a64 hf;
  hf.mix(uint64_t(c.num_inputs));
  hf.mix(uint64_t(c.gates.size()));
  for (const auto& g : c.gates) {
    hf.mix(uint64_t(g.op));
    hf.mix(uint64_t(g.i0));
    hf.mix(uint64_t(g.i1));
    hf.mix(uint64_t(g.i2));
  }
  return hf.value();
}

// For non-empty circuits we dedup by structure and merge cofactors.
// For empty-gate circuits we must keep 0; 1; x0; distinct.
static inline CircHash dedup_signature(const circ::Circuit& c) {
  if (!c.gates.empty())
    return raw_signature(c);

  circ::Fnv1a64 hf;
  hf.mix(uint64_t(c.num_inputs));
  hf.mix(uint64_t(0xE0E0E0E0u)); // tag empty-gate circuits

  std::array<TT4, MAX_COFACTORS_PER_CIRC> tmp{};
  size_t n = 0;
  for (const auto& cf : c.cofactors) {
    if (n >= tmp.size())
      break;
    tmp[n++] = cf.function_p;
  }
  std::sort(tmp.begin(), tmp.begin() + n);
  for (size_t i = 0; i < n; ++i)
    hf.mix(uint64_t(tmp[i]));

  return hf.value();
}

static inline IsaMask recompute_valid_mask(const circ::Circuit& c) {
  return circ::circuit_isa_mask(c.gates_used());
}

static inline bool has_predicated_op(const circ::Circuit& c) {
  for (const auto& g : c.gates) {
    if (circ::GATE_WIRES[g.op * 3 + 2] == circ::W_PREDICATE)
      return true;
  }
  return false;
}

static inline int binding_const_count(const std::vector<circ::Binding>& b) {
  int k = 0;
  for (const auto& x : b)
    if (x.type != circ::Binding::Free)
      ++k;
  return k;
}

static inline void canonicalize_free_vars(std::vector<circ::Binding>& b) {
  uint8_t map[4];
  std::fill(map, map + 4, 0xFF);
  uint8_t next = 0;

  for (auto& x : b) {
    if (x.type != circ::Binding::Free)
      continue;
    uint8_t v = x.value;
    if (v >= 4)
      continue;
    if (map[v] == 0xFF)
      map[v] = next++;
    x.value = map[v];
  }
}

static inline void apply_input_perm_to_bindings(std::vector<circ::Cofactor>& cofs, uint8_t num_inputs,
                                                const std::array<WireIdx, 8>& perm) {
  for (auto& cf : cofs) {
    if (cf.bindings.size() < num_inputs)
      cf.bindings.resize(num_inputs, circ::Binding::c0());

    std::vector<circ::Binding> nb(num_inputs, circ::Binding::c0());
    for (uint8_t i = 0; i < num_inputs && i < 8; ++i)
      nb[perm[i]] = cf.bindings[i];

    cf.bindings = std::move(nb);
    canonicalize_free_vars(cf.bindings);
  }
}

static inline TT4 induce_tt4(const TT8& f8, uint8_t num_inputs, const std::vector<circ::Binding>& bind) {
  TT4 out = 0;
  for (uint16_t asg = 0; asg < 16; ++asg) {
    uint16_t row = 0;
    for (uint8_t i = 0; i < 8; ++i) {
      uint8_t bit = 0;
      if (i < num_inputs && i < bind.size()) {
        const auto& bi = bind[i];
        if (bi.type == circ::Binding::Const1)
          bit = 1;
        else if (bi.type == circ::Binding::Const0)
          bit = 0;
        else
          bit = (asg >> bi.value) & 1u;
      }
      row |= uint16_t(bit) << i;
    }
    if (f8.get(row))
      out |= TT4(1u << asg);
  }
  return out;
}

// Merge cofactors by P-class only; prefer the binding with more constants.
static inline void merge_cofactors(circ::Circuit& dst, const circ::Circuit& src) {
  for (const auto& cf : src.cofactors) {
    bool found = false;
    for (auto& ex : dst.cofactors) {
      if (ex.function_p == cf.function_p) {
        ex.isa_optimal_mask |= cf.isa_optimal_mask;
        if (binding_const_count(cf.bindings) > binding_const_count(ex.bindings))
          ex.bindings = cf.bindings;
        found = true;
        break;
      }
    }
    if (!found) {
      if ((int)dst.cofactors.size() >= MAX_COFACTORS_PER_CIRC)
        continue;
      dst.cofactors.push_back(cf);
    }
  }
}

static inline uint16_t popcount16(uint16_t x) {
#if defined(__GNUG__) || defined(__clang__)
  return (uint16_t)__builtin_popcount((unsigned)x);
#else
  // portable fallback
  uint16_t c = 0;
  while (x) {
    x &= (uint16_t)(x - 1);
    ++c;
  }
  return c;
#endif
}

static inline int32_t circuit_score_fast(const circ::Circuit& c) {
  const int gc = (int)c.gate_count();
  const int cof = (int)c.cofactors.size();
  const int isas = (int)popcount16((uint16_t)c.isa_valid_mask);

  int32_t s = 0;
  s += std::max(0, 12 - gc) * 18;
  s += std::min(cof, 256) * 4;
  s += isas * 6;
  return s;
}

static inline size_t count_total_cofactors(const std::vector<circ::Circuit>& pool) {
  size_t total = 0;
  for (const auto& c : pool)
    total += c.cofactors.size();
  return total;
}

static inline size_t count_unique_funcs(const std::vector<circ::Circuit>& pool) {
  std::unordered_set<TT4> s;
  s.reserve(4096);
  for (const auto& c : pool)
    for (const auto& cf : c.cofactors)
      s.insert(cf.function_p);
  return s.size();
}

// Recompute cf.function_p from bindings; dedup by P-class; keep best binding.
static inline void fixup_cofactors(circ::Circuit& c) {
  if (c.cofactors.empty())
    return;

  if (c.num_inputs == 0) {
    // constants: leave as-is (distinct 0; and 1; must remain)
    return;
  }

  TT8 out8;
  if (c.gates.empty()) {
    // semantics: empty gates with num_inputs>0 means "x0;"
    out8 = TT8::PI()[0];
  } else {
    out8 = c.eval8();
  }

  for (auto& cf : c.cofactors) {
    if (cf.bindings.size() < c.num_inputs)
      cf.bindings.resize(c.num_inputs, circ::Binding::c0());
    canonicalize_free_vars(cf.bindings);

    TT4 t4 = induce_tt4(out8, c.num_inputs, cf.bindings);
    cf.function_p = tt::canon_p[t4];
  }

  // n<=256: sorting is simpler than hashing and plenty fast.
  std::sort(c.cofactors.begin(), c.cofactors.end(),
            [](const circ::Cofactor& a, const circ::Cofactor& b) { return a.function_p < b.function_p; });

  std::vector<circ::Cofactor> out;
  out.reserve(c.cofactors.size());

  for (auto& cf : c.cofactors) {
    if (out.empty() || out.back().function_p != cf.function_p) {
      out.push_back(std::move(cf));
    } else {
      auto& ex = out.back();
      ex.isa_optimal_mask |= cf.isa_optimal_mask;
      if (binding_const_count(cf.bindings) > binding_const_count(ex.bindings))
        ex.bindings = std::move(cf.bindings);
    }
  }

  c.cofactors = std::move(out);
}

// ----------------------------- Dedup / Pool ---------------------------------

static std::vector<circ::Circuit> dedup_merge(std::vector<circ::Circuit>&& v) {
  std::unordered_map<CircHash, size_t> idx;
  idx.reserve(v.size() * 2 + 64);

  std::vector<circ::Circuit> out;
  out.reserve(v.size());

  for (auto& c : v) {
    const CircHash sig = dedup_signature(c);
    auto it = idx.find(sig);
    if (it == idx.end()) {
      idx.emplace(sig, out.size());
      out.push_back(std::move(c));
    } else {
      merge_cofactors(out[it->second], c);
    }
  }
  return out;
}

static void merge_batch_into_pool(std::vector<circ::Circuit>& pool, std::vector<circ::Circuit>&& batch) {
  std::unordered_map<CircHash, size_t> idx;
  idx.reserve(pool.size() * 2 + 64);
  for (size_t i = 0; i < pool.size(); ++i)
    idx.emplace(dedup_signature(pool[i]), i);

  for (auto& c : batch) {
    const CircHash sig = dedup_signature(c);
    auto [it, inserted] = idx.try_emplace(sig, pool.size());
    if (inserted) {
      pool.push_back(std::move(c));
    } else {
      merge_cofactors(pool[it->second], c);
    }
  }
}

static void cap_pool(std::vector<circ::Circuit>& pool, size_t cap) {
  if (pool.size() <= cap)
    return;

  std::unordered_map<TT4, size_t> best_idx;
  best_idx.reserve(4096);

  std::vector<int32_t> score(pool.size(), 0);
  for (size_t i = 0; i < pool.size(); ++i) {
    score[i] = circuit_score_fast(pool[i]);
    for (const auto& cf : pool[i].cofactors) {
      auto it = best_idx.find(cf.function_p);
      if (it == best_idx.end())
        best_idx.emplace(cf.function_p, i);
      else if (score[i] > score[it->second])
        it->second = i;
    }
  }

  std::vector<size_t> keep;
  keep.reserve(std::min(cap, pool.size()));

  {
    std::unordered_set<CircHash> seen;
    seen.reserve(best_idx.size() * 2 + 64);
    for (auto& kv : best_idx) {
      size_t i = kv.second;
      CircHash s = dedup_signature(pool[i]);
      if (seen.emplace(s).second)
        keep.push_back(i);
    }
  }

  if (keep.size() < cap) {
    std::vector<size_t> idxs(pool.size());
    for (size_t i = 0; i < pool.size(); ++i)
      idxs[i] = i;

    std::vector<uint8_t> already(pool.size(), 0);
    for (auto i : keep)
      already[i] = 1;

    size_t nth = std::min(cap, idxs.size());
    std::nth_element(idxs.begin(), idxs.begin() + nth, idxs.end(),
                     [&](size_t a, size_t b) { return score[a] > score[b]; });

    for (size_t k = 0; k < idxs.size() && keep.size() < cap; ++k) {
      size_t i = idxs[k];
      if (!already[i])
        keep.push_back(i);
    }
  }

  std::vector<circ::Circuit> out;
  out.reserve(keep.size());
  for (auto i : keep)
    out.push_back(std::move(pool[i]));
  pool = std::move(out);
}

// --------------------------- Optimal pruning --------------------------------

static BestTable compute_best_table(const std::vector<circ::Circuit>& pool) {
  BestTable best{};
  for (auto& row : best)
    row.fill(kInfGateCount);

  for (const auto& c : pool) {
    const GateCount gc = c.gate_count();
    const IsaMask valid = recompute_valid_mask(c);

    for (const auto& cf : c.cofactors) {
      for (int isa = 0; isa < circ::NUM_ISAS; ++isa) {
        if (!(valid & (IsaMask(1) << isa)))
          continue;
        auto& cell = best[isa][cf.function_p];
        if (gc < cell)
          cell = gc;
      }
    }
  }
  return best;
}

static void prune_and_set_optimal(std::vector<circ::Circuit>& pool, const BestTable& best) {
  std::vector<circ::Circuit> out;
  out.reserve(pool.size());

  for (auto& c : pool) {
    c.isa_valid_mask = recompute_valid_mask(c);

    std::vector<circ::Cofactor> kept;
    kept.reserve(c.cofactors.size());

    for (auto& cf : c.cofactors) {
      IsaMask opt = 0;
      for (int isa = 0; isa < circ::NUM_ISAS; ++isa) {
        if (!(c.isa_valid_mask & (IsaMask(1) << isa)))
          continue;
        if (best[isa][cf.function_p] == c.gate_count())
          opt |= (IsaMask(1) << isa);
      }
      if (opt) {
        cf.isa_optimal_mask = opt;
        kept.push_back(std::move(cf));
      }
    }

    if (!kept.empty()) {
      c.cofactors = std::move(kept);
      out.push_back(std::move(c));
    }
  }

  pool = std::move(out);
}

// ------------------------- Cofactor enumeration ------------------------------
// Budgeted, sample-first, with the "few-input MUX" corner handled generically:
// if dN>1, we keep at least one non-free data input by forcing freeN<=dN-1.
static size_t enumerate_cofactors_budgeted(circ::Circuit& c, const TT8& out8, std::mt19937& rng, int budget,
                                           int max_add, int cofactor_cap) {
  if (c.num_inputs == 0)
    return 0;
  if (budget <= 0)
    return 0;
  if ((int)c.cofactors.size() >= cofactor_cap)
    return 0;

  // Assume inputs 0..num_inputs-1 are data for non-predicated supercirc work.
  const int dN = std::min<int>(c.num_inputs, 8);
  if (dN <= 0)
    return 0;

  const int freeN = (dN <= 1) ? dN : std::min(4, dN - 1);

  // Fast seen set via stamps (64k entries, no clearing).
  static thread_local std::vector<uint32_t> stamp;
  static thread_local uint32_t cur = 1;
  if (stamp.size() != TT4_SPACE)
    stamp.assign(TT4_SPACE, 0);
  if (++cur == 0) {
    std::fill(stamp.begin(), stamp.end(), 0);
    cur = 1;
  }

  for (const auto& cf : c.cofactors)
    stamp[cf.function_p] = cur;

  auto try_add = [&](std::vector<circ::Binding>&& b) -> bool {
    canonicalize_free_vars(b);

    TT4 t4 = induce_tt4(out8, c.num_inputs, b);
    TT4 rep = tt::canon_p[t4];

    if (stamp[rep] == cur)
      return false;
    stamp[rep] = cur;

    if ((int)c.cofactors.size() >= cofactor_cap)
      return false;

    circ::Cofactor cf;
    cf.function_p = rep;
    cf.isa_optimal_mask = 0;
    cf.bindings = std::move(b);
    c.cofactors.push_back(std::move(cf));
    return true;
  };

  size_t added = 0;

  std::array<uint8_t, 8> perm_idx{};
  for (int i = 0; i < dN; ++i)
    perm_idx[i] = (uint8_t)i;

  for (int t = 0; t < budget; ++t) {
    if ((int)c.cofactors.size() >= cofactor_cap)
      break;
    if ((int)added >= max_add)
      break;

    // Pick freeN distinct anchors.
    std::shuffle(perm_idx.begin(), perm_idx.begin() + dN, rng);
    std::array<uint8_t, 4> anchors{};
    for (int i = 0; i < freeN; ++i)
      anchors[i] = perm_idx[i];

    std::vector<circ::Binding> b(c.num_inputs, circ::Binding::c0());
    for (int i = 0; i < freeN; ++i)
      b[anchors[i]] = circ::Binding::free((uint8_t)i);

    for (int w = 0; w < dN; ++w) {
      bool is_anchor = false;
      for (int i = 0; i < freeN; ++i)
        if ((uint8_t)w == anchors[i])
          is_anchor = true;
      if (is_anchor)
        continue;

      uint32_t r = (uint32_t)rng();
      if (freeN > 0 && (int)(r % 100u) < COF_P_ALIAS_PCT) {
        b[w] = circ::Binding::free((uint8_t)(r % (uint32_t)freeN));
      } else {
        b[w] = (r & 1u) ? circ::Binding::c1() : circ::Binding::c0();
      }
    }

    if (try_add(std::move(b)))
      ++added;
  }

  return added;
}

// ----------------------------- Simplify --------------------------------------

struct ConstInfo {
  std::array<int8_t, 8> v{};
  ConstInfo() { v.fill(-1); }
};

static ConstInfo compute_const_info(const circ::Circuit& c) {
  ConstInfo ci;
  if (c.cofactors.empty())
    return ci;

  for (uint8_t i = 0; i < c.num_inputs && i < 8; ++i) {
    bool all0 = true, all1 = true;
    for (const auto& cf : c.cofactors) {
      if (i >= cf.bindings.size()) {
        all0 = all1 = false;
        break;
      }
      all0 &= (cf.bindings[i].type == circ::Binding::Const0);
      all1 &= (cf.bindings[i].type == circ::Binding::Const1);
    }
    if (all0)
      ci.v[i] = 0;
    else if (all1)
      ci.v[i] = 1;
  }
  return ci;
}

static inline bool input_is_const(const ConstInfo& ci, WireIdx w, int need) {
  return w < 8 && ci.v[w] == need;
}

static bool try_rewrite_gate(circ::GateOp& g, const ConstInfo& ci) {
  switch (g.op) {
  case circ::G_MUX:
    if (input_is_const(ci, g.i2, 0)) {
      g.op = circ::G_AND;
      g.i2 = 0;
      return true;
    }
    if (input_is_const(ci, g.i2, 1)) {
      g.op = circ::G_ORN;
      std::swap(g.i0, g.i1);
      g.i2 = 0;
      return true;
    }
    if (input_is_const(ci, g.i1, 0)) {
      g.op = circ::G_ANDN;
      WireIdx a = g.i0, c = g.i2;
      g.i0 = c;
      g.i1 = a;
      g.i2 = 0;
      return true;
    }
    if (input_is_const(ci, g.i1, 1)) {
      g.op = circ::G_OR;
      g.i1 = g.i2;
      g.i2 = 0;
      return true;
    }
    return false;

  case circ::G_XOR3:
    if (input_is_const(ci, g.i2, 0)) {
      g.op = circ::G_XOR;
      g.i2 = 0;
      return true;
    }
    if (input_is_const(ci, g.i2, 1)) {
      g.op = circ::G_XNOR;
      g.i2 = 0;
      return true;
    }
    return false;

  case circ::G_BCAX:
    if (input_is_const(ci, g.i2, 0)) {
      g.op = circ::G_ANDN;
      g.i2 = 0;
      return true;
    }
    if (input_is_const(ci, g.i2, 1)) {
      g.op = circ::G_ORN;
      std::swap(g.i0, g.i1);
      g.i2 = 0;
      return true;
    }
    if (input_is_const(ci, g.i1, 0)) {
      g.op = circ::G_XOR;
      g.i1 = g.i2;
      g.i2 = 0;
      return true;
    }
    if (input_is_const(ci, g.i0, 1)) {
      g.op = circ::G_XNOR;
      g.i0 = g.i1;
      g.i1 = g.i2;
      g.i2 = 0;
      return true;
    }
    return false;

  default: return false;
  }
}

static inline IsaMask active_isa_mask(const circ::Circuit& c) {
  IsaMask m = 0;
  for (const auto& cf : c.cofactors)
    m |= cf.isa_optimal_mask;
  return m;
}

static void restrict_cofactors(circ::Circuit& c, IsaMask keep) {
  std::vector<circ::Cofactor> out;
  out.reserve(c.cofactors.size());
  for (auto& cf : c.cofactors) {
    cf.isa_optimal_mask &= keep;
    if (cf.isa_optimal_mask)
      out.push_back(std::move(cf));
  }
  c.cofactors = std::move(out);
}

static void compact_inputs(circ::Circuit& c) {
  if (c.num_inputs == 0 || c.gates.empty())
    return;

  std::array<bool, 8> used{};
  for (const auto& g : c.gates) {
    const uint8_t* wc = &circ::GATE_WIRES[g.op * 3];
    if (wc[0] != circ::W_ZERO && g.i0 < c.num_inputs)
      used[g.i0] = true;
    if (wc[1] != circ::W_ZERO && g.i1 < c.num_inputs)
      used[g.i1] = true;
    if (wc[2] != circ::W_ZERO && g.i2 < c.num_inputs)
      used[g.i2] = true;
  }

  std::array<WireIdx, 8> remap{};
  uint8_t nn = 0;
  for (uint8_t i = 0; i < c.num_inputs && i < 8; ++i)
    if (used[i])
      remap[i] = nn++;

  if (nn == c.num_inputs)
    return;

  auto remap_wire = [&](WireIdx w) -> WireIdx {
    return (w < c.num_inputs) ? remap[w] : WireIdx(nn + (w - c.num_inputs));
  };

  for (auto& g : c.gates) {
    const uint8_t* wc = &circ::GATE_WIRES[g.op * 3];
    g.i0 = remap_wire(g.i0);
    g.i1 = (wc[1] != circ::W_ZERO) ? remap_wire(g.i1) : 0;
    g.i2 = (wc[2] != circ::W_ZERO) ? remap_wire(g.i2) : 0;
  }

  for (auto& cf : c.cofactors) {
    std::vector<circ::Binding> nb(nn, circ::Binding::c0());
    for (uint8_t i = 0; i < c.num_inputs && i < 8; ++i)
      if (used[i] && i < cf.bindings.size())
        nb[remap[i]] = cf.bindings[i];
    cf.bindings = std::move(nb);
  }

  c.num_inputs = nn;
}

static std::vector<circ::Circuit> simplify_circuit_fast(const circ::Circuit& in) {
  std::vector<circ::Circuit> vars;
  vars.push_back(in);

  for (int iter = 0; iter < SIMPLIFY_MAX_ITERS; ++iter) {
    bool changed = false;
    std::vector<circ::Circuit> next;
    next.reserve(vars.size());

    for (auto& v : vars) {
      IsaMask act = active_isa_mask(v);
      if (!act)
        continue;

      ConstInfo ci = compute_const_info(v);
      bool did = false;

      for (size_t gi = 0; gi < v.gates.size(); ++gi) {
        circ::GateOp g2 = v.gates[gi];
        if (!try_rewrite_gate(g2, ci))
          continue;

        circ::Circuit cand = v;
        cand.gates[gi] = g2;
        cand.isa_valid_mask = recompute_valid_mask(cand);

        IsaMask can_apply = act & cand.isa_valid_mask;
        if (!can_apply)
          continue;

        if (can_apply == act) {
          v = std::move(cand);
          changed = true;
          did = true;
          break;
        }

        circ::Circuit rew = std::move(cand);
        circ::Circuit orig = v;

        restrict_cofactors(rew, can_apply);
        restrict_cofactors(orig, act & ~can_apply);

        if (!rew.cofactors.empty())
          next.push_back(std::move(rew));
        if (!orig.cofactors.empty())
          next.push_back(std::move(orig));

        v.cofactors.clear();
        changed = true;
        did = true;
        break;
      }

      if (!did && !v.cofactors.empty())
        next.push_back(std::move(v));

      if (next.size() >= (size_t)SIMPLIFY_MAX_VARIANTS)
        break;
    }

    vars = std::move(next);
    if (!changed)
      break;
  }

  for (auto& v : vars) {
    if (v.cofactors.empty())
      continue;

    compact_inputs(v);
    auto perm = circ::canonicalize(v);
    apply_input_perm_to_bindings(v.cofactors, v.num_inputs, perm);

    v.isa_valid_mask = recompute_valid_mask(v);
    fixup_cofactors(v);
  }

  vars.erase(std::remove_if(vars.begin(), vars.end(), [](const circ::Circuit& c) { return c.cofactors.empty(); }),
             vars.end());
  return vars;
}

// ------------------------------- Lifting -------------------------------------

struct LiftOpt {
  circ::Gate new_op;
  int a_tok, b_tok, c_tok; // -1 const0, -2 const1, 0/1/2 refer to old operands
};

static inline const std::vector<LiftOpt>& lift_options_for(circ::Gate g) {
  static const std::array<std::vector<LiftOpt>, circ::NUM_GATES> tbl = [] {
    std::array<std::vector<LiftOpt>, circ::NUM_GATES> t{};
    t[circ::G_AND] = {{circ::G_MUX, 0, 1, -1}};
    t[circ::G_OR] = {{circ::G_MUX, 0, -2, 1}};
    t[circ::G_XOR] = {{circ::G_XOR3, 0, 1, -1}, {circ::G_BCAX, 0, -1, 1}};
    t[circ::G_XNOR] = {{circ::G_XOR3, 0, 1, -2}, {circ::G_BCAX, -2, 0, 1}};
    t[circ::G_ANDN] = {{circ::G_MUX, 1, -1, 0}, {circ::G_BCAX, 0, 1, -1}};
    t[circ::G_ORN] = {{circ::G_MUX, 1, 0, -2}, {circ::G_BCAX, 1, 0, -2}};
    return t;
  }();
  return tbl[g];
}

static inline const std::vector<std::pair<circ::ISA, circ::ISA>>& isa_lift_pairs() {
  static const std::vector<std::pair<circ::ISA, circ::ISA>> pairs = {
      {circ::NEON, circ::NEON},
      {circ::NEON_SHA3, circ::NEON_SHA3},
  };
  return pairs;
}

static inline bool has_source_optimal(const circ::Circuit& c, circ::ISA source) {
  const IsaMask bit = IsaMask(1) << source;
  for (const auto& cf : c.cofactors)
    if (cf.isa_optimal_mask & bit)
      return true;
  return false;
}

static inline int choose_L_biased(int maxL, std::mt19937& rng) {
  // Bias to larger L a bit (quadratic weights).
  int sum = 0;
  for (int L = 1; L <= maxL; ++L)
    sum += L * L;

  int r = int(rng() % std::max(1, sum));
  int acc = 0;
  for (int L = 1; L <= maxL; ++L) {
    acc += L * L;
    if (r < acc)
      return L;
  }
  return maxL;
}

struct LiftBuild {
  circ::Circuit c;
  CircHash sig = 0;
};

static bool build_lifted_candidate(const circ::Circuit& base, circ::ISA target, std::mt19937& rng, LiftBuild& out) {
  if (has_predicated_op(base))
    return false;

  std::vector<int> liftable;
  liftable.reserve(base.gates.size());
  for (int gi = 0; gi < (int)base.gates.size(); ++gi) {
    for (const auto& o : lift_options_for(base.gates[gi].op)) {
      if (circ::gate_valid(target, o.new_op)) {
        liftable.push_back(gi);
        break;
      }
    }
  }
  if (liftable.empty())
    return false;

  int maxL = std::min(N_LIFT_GATES_MAX, (int)liftable.size());
  int L = choose_L_biased(maxL, rng);

  std::shuffle(liftable.begin(), liftable.end(), rng);
  liftable.resize(L);
  std::sort(liftable.begin(), liftable.end());

  bool need0 = false, need1 = false;

  struct Choice {
    int gi;
    LiftOpt opt;
  };
  std::vector<Choice> chosen;
  chosen.reserve(L);

  for (int gi : liftable) {
    std::vector<LiftOpt> legal;
    for (const auto& o : lift_options_for(base.gates[gi].op))
      if (circ::gate_valid(target, o.new_op))
        legal.push_back(o);
    if (legal.empty())
      return false;

    const LiftOpt& opt = legal[rng() % legal.size()];
    auto mark = [&](int tok) {
      if (tok == -1)
        need0 = true;
      if (tok == -2)
        need1 = true;
    };
    mark(opt.a_tok);
    mark(opt.b_tok);
    mark(opt.c_tok);

    chosen.push_back({gi, opt});
  }

  uint8_t newNI = 4 + (need0 ? 1 : 0) + (need1 ? 1 : 0);
  if (newNI > 8)
    return false;

  int w0 = need0 ? 4 : -1;
  int w1 = need1 ? (need0 ? 5 : 4) : -1;

  circ::Circuit lifted;
  lifted.num_inputs = newNI;
  lifted.gates.reserve(base.gates.size());
  lifted.cofactors.clear();

  auto map_wire = [&](WireIdx w) -> WireIdx { return (w < 4) ? w : WireIdx(newNI + (w - 4)); };

  auto tok_wire = [&](int tok, const circ::GateOp& og) -> WireIdx {
    if (tok == -1)
      return (WireIdx)w0;
    if (tok == -2)
      return (WireIdx)w1;
    if (tok == 0)
      return map_wire(og.i0);
    if (tok == 1)
      return map_wire(og.i1);
    return map_wire(og.i2);
  };

  size_t ci = 0;
  for (int gi = 0; gi < (int)base.gates.size(); ++gi) {
    const auto& og = base.gates[gi];
    circ::GateOp ng{};

    if (ci < chosen.size() && chosen[ci].gi == gi) {
      const auto& opt = chosen[ci].opt;
      ++ci;
      ng.op = opt.new_op;
      ng.i0 = tok_wire(opt.a_tok, og);
      ng.i1 = tok_wire(opt.b_tok, og);
      ng.i2 = tok_wire(opt.c_tok, og);
    } else {
      ng.op = og.op;
      ng.i0 = map_wire(og.i0);
      const uint8_t* wc = &circ::GATE_WIRES[og.op * 3];
      ng.i1 = (wc[1] != circ::W_ZERO) ? map_wire(og.i1) : 0;
      ng.i2 = (wc[2] != circ::W_ZERO) ? map_wire(og.i2) : 0;
    }

    lifted.gates.push_back(ng);
  }

  // Seed one cofactor: preserve base function_p; bindings: x0..x3 free; added inputs const.
  circ::Cofactor cf;
  cf.function_p = base.cofactors.empty() ? tt::canon_p[base.eval4()] : base.cofactors[0].function_p;
  cf.isa_optimal_mask = 0;
  cf.bindings.assign(newNI, circ::Binding::c0());
  for (uint8_t i = 0; i < 4; ++i)
    cf.bindings[i] = circ::Binding::free(i);
  if (w0 >= 0)
    cf.bindings[w0] = circ::Binding::c0();
  if (w1 >= 0)
    cf.bindings[w1] = circ::Binding::c1();
  lifted.cofactors.push_back(std::move(cf));

  lifted.isa_valid_mask = recompute_valid_mask(lifted);

  out.c = std::move(lifted);
  out.sig = raw_signature(out.c);
  return true;
}

// ------------------------------- Progress ------------------------------------

namespace prog {

struct State {
  std::chrono::steady_clock::time_point start{};
  std::chrono::steady_clock::time_point last_render{};
  int phase = 0;

  // Phase text
  std::string title;
  std::string sub;

  // Work counters
  uint64_t candidates = 0;
  uint64_t flushes = 0;
  size_t batch_size = 0;

  // Pool snapshot (updated at checkpoints)
  size_t pool = 0;
  size_t cof = 0;
  size_t uniqP = 0;

  // Lift-specific progress
  size_t bases_total = 0;
  size_t bases_done = 0;

  bool first = true;
  int last_lines = 0;
};

static inline int count_newlines(const std::string& s) {
  int n = 0;
  for (char c : s)
    if (c == '\n')
      ++n;
  return n;
}

static void clear_previous(State& s) {
  if (s.first || s.last_lines <= 0)
    return;

  // Move up to the top of the previous panel.
  std::cout << "\033[" << s.last_lines << "A";

  // Clear each line, moving down.
  for (int i = 0; i < s.last_lines; ++i) {
    std::cout << tty::CLEAR << "\r";
    if (i < s.last_lines - 1)
      std::cout << "\033[1B";
  }

  // Move back up to the top line so the next print lands in the same region.
  if (s.last_lines > 1)
    std::cout << "\033[" << (s.last_lines - 1) << "A";
}

static void checkpoint(State& s, const std::vector<circ::Circuit>& pool) {
  s.pool = pool.size();
  s.cof = count_total_cofactors(pool);
  s.uniqP = count_unique_funcs(pool);
}

static void render(State& s, bool final = false) {
  auto now = std::chrono::steady_clock::now();
  if (!final) {
    if (s.last_render.time_since_epoch().count() != 0) {
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - s.last_render).count();
      if (ms < (int)UPDATE_MS)
        return;
    }
  }
  s.last_render = now;

  double elapsed = std::chrono::duration<double>(now - s.start).count();
  double cand_ps = (elapsed > 0.001) ? (double(s.candidates) / elapsed) : 0.0;

  std::ostringstream ss;
  ss << tty::HIDE << tty::BOLD << "Supercircuits" << tty::RESET;
  if (final)
    ss << " " << tty::GREEN << "[Complete]" << tty::RESET;
  ss << tty::CLEAR << "\n" << std::string(60, '-') << tty::CLEAR << "\n";

  ss << " Phase:      " << tty::CYAN << s.title << tty::RESET;
  if (!s.sub.empty())
    ss << " " << tty::YELLOW << s.sub << tty::RESET;
  ss << tty::CLEAR << "\n";

  if (s.bases_total) {
    double pct = 100.0 * (double)s.bases_done / std::max<double>(1.0, (double)s.bases_total);
    ss << " Bases:      " << std::setw(6) << s.bases_done << " / " << s.bases_total << " (" << std::fixed
       << std::setprecision(2) << pct << "%)" << tty::CLEAR << "\n";
  } else {
    ss << " Bases:      " << tty::YELLOW << "-" << tty::RESET << tty::CLEAR << "\n";
  }

  ss << " Pool:       " << s.pool << " circs" << tty::CLEAR << "\n";
  ss << " Cofactors:  " << s.cof << "  (uniqP=" << s.uniqP << ")" << tty::CLEAR << "\n";
  ss << " Generated:  " << std::fixed << std::setprecision(1) << (s.candidates / 1e3) << "K candidates ("
     << int(cand_ps) << "/s)" << tty::CLEAR << "\n";
  ss << " Batch:      " << s.batch_size << "  flushes=" << s.flushes << tty::CLEAR << "\n";
  ss << " Time:       " << int(elapsed) << "s" << tty::CLEAR << "\n";

  std::string out = ss.str();
  int lines = count_newlines(out);

  clear_previous(s);
  std::cout << out << std::flush;

  s.last_lines = lines;
  s.first = false;

  if (final)
    std::cout << tty::SHOW << std::flush;
}

} // namespace prog

// ------------------------------- Phases --------------------------------------

// Phase 0: seed expansion for the existing pool, with a smooth budget curve.
static void phase0_seed_expand(std::vector<circ::Circuit>& pool, prog::State& ps) {
  std::mt19937 rng(12345);

  ps.title = "Seed expansion";
  ps.sub.clear();
  ps.bases_total = ps.bases_done = 0;

  for (size_t i = 0; i < pool.size(); ++i) {
    auto& c = pool[i];
    if (c.num_inputs == 0 || has_predicated_op(c))
      continue;

    int gc = (int)c.gate_count();
    int budget = SEED_MAX_BUDGET_EXISTING - 4 * gc;
    budget = std::max(SEED_MIN_BUDGET_EXISTING, std::min(SEED_MAX_BUDGET_EXISTING, budget));

    TT8 out8 = c.gates.empty() ? TT8::PI()[0] : c.eval8();
    enumerate_cofactors_budgeted(c, out8, rng, budget, SEED_MAX_ADD_EXISTING, MAX_COFACTORS_PER_CIRC);
    fixup_cofactors(c);

    if ((i & 1023) == 0) {
      ps.sub = "(pool scan)";
      ps.batch_size = 0;
      prog::render(ps);
    }
  }
}

// Phase 1: lift + seed-on-create cofactor exploration (no top-M sorting).
static void phase1_lift(std::vector<circ::Circuit>& pool, const std::vector<circ::Circuit>& base_pool,
                        prog::State& ps) {
  std::mt19937 rng(12345);

  auto maybe_guard = [&]() {
    if (pool.size() > POOL_GUARD_SIZE)
      cap_pool(pool, POOL_GUARD_CAP);
  };

  auto flush_batch = [&](std::vector<LiftBuild>& batch) {
    if (batch.empty())
      return;

    std::vector<circ::Circuit> circs;
    circs.reserve(batch.size());
    for (auto& b : batch)
      circs.push_back(std::move(b.c));
    batch.clear();

    // Canonicalize + binding perm + seed enumerate.
    for (auto& c : circs) {
      auto perm = circ::canonicalize(c);
      apply_input_perm_to_bindings(c.cofactors, c.num_inputs, perm);
      c.isa_valid_mask = recompute_valid_mask(c);
      fixup_cofactors(c);

      // Seed-on-create: small budget, capped to avoid explosive temporary cofactors.
      if (c.num_inputs != 0 && !has_predicated_op(c) && !c.gates.empty()) {
        TT8 out8 = c.eval8();
        int cap = std::min(MAX_COFACTORS_PER_CIRC, SEED_COF_CAP_ON_CREATE);
        enumerate_cofactors_budgeted(c, out8, rng, SEED_BUDGET_ON_CREATE, SEED_MAX_ADD_ON_CREATE, cap);
        fixup_cofactors(c);
      }
    }

    // Extra pass: spend more budget on the most promising circuits in this flush.
    // This is intentionally simple: score+topM, then enumerate again.
    if (EXTRA_ENUM_TOP_M_PER_FLUSH > 0 && !circs.empty()) {
      std::vector<int32_t> sc(circs.size(), 0);
      std::vector<size_t> ix(circs.size(), 0);
      for (size_t i = 0; i < circs.size(); ++i) {
        // ensure valid mask is set for scoring
        circs[i].isa_valid_mask = recompute_valid_mask(circs[i]);
        sc[i] = circuit_score_fast(circs[i]);
        ix[i] = i;
      }

      const size_t M = std::min(EXTRA_ENUM_TOP_M_PER_FLUSH, ix.size());
      std::nth_element(ix.begin(), ix.begin() + M, ix.end(), [&](size_t a, size_t b) { return sc[a] > sc[b]; });

      for (size_t k = 0; k < M; ++k) {
        auto& c = circs[ix[k]];
        if (c.num_inputs == 0 || c.gates.empty() || has_predicated_op(c))
          continue;
        TT8 out8 = c.eval8();
        enumerate_cofactors_budgeted(c, out8, rng, EXTRA_ENUM_BUDGET, EXTRA_ENUM_MAX_ADD, MAX_COFACTORS_PER_CIRC);
        fixup_cofactors(c);
      }
    }

    // Shrink duplicates before merging into the big pool.
    circs = dedup_merge(std::move(circs));
    merge_batch_into_pool(pool, std::move(circs));
    maybe_guard();

    ps.flushes++;
    ps.batch_size = 0;
    prog::render(ps);
  };

  const auto& pairs = isa_lift_pairs();

  for (size_t pi = 0; pi < pairs.size(); ++pi) {
    circ::ISA source = pairs[pi].first;
    circ::ISA target = pairs[pi].second;

    ps.title = "Lift";
    {
      std::ostringstream s;
      s << circ::NAMES[source] << " -> " << circ::NAMES[target];
      ps.sub = s.str();
    }

    // Collect eligible bases; simple single cap.
    std::vector<size_t> bases;
    bases.reserve(131072);
    for (size_t i = 0; i < base_pool.size(); ++i) {
      const auto& c = base_pool[i];
      if (c.num_inputs != 4)
        continue;
      if (c.gate_count() > LIFT_MAX_GATECOUNT)
        continue;
      if (!has_source_optimal(c, source))
        continue;
      if (has_predicated_op(c))
        continue;
      if (c.gates.empty())
        continue;
      bases.push_back(i);
    }

    if (bases.size() > MAX_BASES_PER_PAIR) {
      std::shuffle(bases.begin(), bases.end(), rng);
      bases.resize(MAX_BASES_PER_PAIR);
    }

    ps.bases_total = bases.size();
    ps.bases_done = 0;
    ps.batch_size = 0;
    prog::render(ps);

    std::vector<LiftBuild> batch;
    batch.reserve(FLUSH_BATCH_HARD_MAX);

    std::unordered_set<CircHash> raw_seen;
    raw_seen.reserve(FLUSH_BATCH_HARD_MAX * 2);

    for (size_t bi = 0; bi < bases.size(); ++bi) {
      ps.bases_done = bi + 1;

      const auto& base = base_pool[bases[bi]];

      for (int att = 0; att < ATTEMPTS_PER_BASE; ++att) {
        LiftBuild cand;
        if (!build_lifted_candidate(base, target, rng, cand))
          continue;

        if (!raw_seen.emplace(cand.sig).second)
          continue;

        batch.push_back(std::move(cand));
        ps.candidates++;
        ps.batch_size = batch.size();

        if (batch.size() >= FLUSH_BATCH_HARD_MAX) {
          flush_batch(batch);
          raw_seen.clear();
        }
      }

      if (!batch.empty() && batch.size() >= FLUSH_BATCH_TARGET) {
        flush_batch(batch);
        raw_seen.clear();
      }

      if ((bi & 1023) == 0)
        prog::render(ps);
    }

    flush_batch(batch);
    prog::checkpoint(ps, pool);
    prog::render(ps);
  }
}

// Phase 2: prune (optimal marking)
static void phase2_prune(std::vector<circ::Circuit>& pool, prog::State& ps) {
  ps.title = "Prune / optimalize";
  ps.sub.clear();
  ps.bases_total = ps.bases_done = 0;
  ps.batch_size = 0;
  prog::render(ps);

  pool = dedup_merge(std::move(pool));
  BestTable best = compute_best_table(pool);
  prune_and_set_optimal(pool, best);

  prog::checkpoint(ps, pool);
  prog::render(ps);
}

// Phase 3: simplify (rewrite under const-in-cofactors)
static void phase3_simplify(std::vector<circ::Circuit>& pool, prog::State& ps) {
  if (!ENABLE_SIMPLIFY)
    return;

  ps.title = "Simplify";
  ps.sub.clear();
  ps.bases_total = ps.bases_done = 0;
  ps.batch_size = 0;
  prog::render(ps);

  std::vector<circ::Circuit> out;
  out.reserve(pool.size());

  for (size_t i = 0; i < pool.size(); ++i) {
    auto vars = simplify_circuit_fast(pool[i]);
    for (auto& v : vars)
      out.push_back(std::move(v));

    if ((i & 2047) == 0)
      prog::render(ps);
  }

  pool = dedup_merge(std::move(out));
  BestTable best = compute_best_table(pool);
  prune_and_set_optimal(pool, best);

  prog::checkpoint(ps, pool);
  prog::render(ps);
}

// Phase 4: final canonicalize + binding fixup + prune
static void phase4_final(std::vector<circ::Circuit>& pool, prog::State& ps) {
  ps.title = "Final fixup";
  ps.sub.clear();
  ps.bases_total = ps.bases_done = 0;
  ps.batch_size = 0;
  prog::render(ps);

  for (size_t i = 0; i < pool.size(); ++i) {
    auto& c = pool[i];
    auto perm = circ::canonicalize(c);
    apply_input_perm_to_bindings(c.cofactors, c.num_inputs, perm);
    c.isa_valid_mask = recompute_valid_mask(c);
    fixup_cofactors(c);

    if ((i & 4095) == 0)
      prog::render(ps);
  }

  pool = dedup_merge(std::move(pool));
  BestTable best = compute_best_table(pool);
  prune_and_set_optimal(pool, best);

  prog::checkpoint(ps, pool);
  prog::render(ps);
}

// --------------------------------- API --------------------------------------

std::vector<circ::Circuit> run(std::vector<circ::Circuit> circs) {
  // Predicate-gate circuits are passthrough.
  std::vector<circ::Circuit> passthrough;
  std::vector<circ::Circuit> work;
  passthrough.reserve(1024);
  work.reserve(circs.size());

  for (auto& c : circs) {
    if (has_predicated_op(c))
      passthrough.push_back(std::move(c));
    else
      work.push_back(std::move(c));
  }

  // Base pool snapshot (pre-lift) so lifting remains stable.
  const std::vector<circ::Circuit> base_pool = work;

  // Progress panel
  prog::State ps;
  ps.start = std::chrono::steady_clock::now();
  prog::checkpoint(ps, work);
  ps.title = "Start";
  ps.sub.clear();
  prog::render(ps);

  // Phase 0: seed expansion
  phase0_seed_expand(work, ps);
  prog::checkpoint(ps, work);
  prog::render(ps);

  // Phase 1: lifting + seed-on-create enumeration
  phase1_lift(work, base_pool, ps);

  // Phase 2: prune/optimalize
  phase2_prune(work, ps);

  // Phase 3: simplify (may split by ISA coverage)
  phase3_simplify(work, ps);

  // Phase 4: final fixup/canonicalize + prune
  phase4_final(work, ps);

  // Append passthrough circuits untouched.
  if (!passthrough.empty()) {
    work.reserve(work.size() + passthrough.size());
    for (auto& c : passthrough)
      work.push_back(std::move(c));
  }

  // Final render (leave cursor shown)
  prog::checkpoint(ps, work);
  prog::render(ps, true);

  // Ensure 0-gate circuits remain distinct after all dedup:
  // enforced by dedup_signature().
  return work;
}

} // namespace supercirc

// =============================================================================
// output: report writer
// =============================================================================
namespace output {

inline std::string format_isa_tags(IsaMask mask) {
  std::ostringstream ss;
  bool first = true;
  for (int i = 0; i < circ::NUM_ISAS; ++i) {
    if (mask & (IsaMask(1) << i)) {
      if (!first)
        ss << " ";
      ss << circ::NAMES[i];
      first = false;
    }
  }
  return ss.str();
}

enum class InputKind : uint8_t { Data = 0, Pred = 1 };

inline void compute_input_kinds(const circ::Circuit& c, std::array<InputKind, 8>& kinds) {
  kinds.fill(InputKind::Data);
  for (auto& g : c.gates) {
    const bool pp = circ::GATE_WIRES[g.op * 3 + 2] == circ::W_PREDICATE;
    if (pp) {
      WireIdx w = g.i2;
      if (w < c.num_inputs && w < 8)
        kinds[w] = InputKind::Pred;
    }
  }
}

inline std::string format_bindings(const circ::Circuit& c, const std::vector<circ::Binding>& b) {
  std::array<InputKind, 8> kinds{};
  compute_input_kinds(c, kinds);

  auto in_name = [&](uint8_t i) -> std::string {
    return std::string(kinds[i] == InputKind::Pred ? "p" : "x") + std::to_string(i);
  };

  std::ostringstream ss;
  for (uint8_t i = 0; i < c.num_inputs; ++i) {
    if (i)
      ss << " ";
    ss << in_name(i) << "=";
    const auto& bi = b[i];
    switch (bi.type) {
    case circ::Binding::Const0: ss << "0"; break;
    case circ::Binding::Const1: ss << "1"; break;
    case circ::Binding::Free: ss << static_cast<char>('a' + bi.value); break;
    }
  }
  return ss.str();
}

inline void save_circuits(const std::string& path, const std::vector<circ::Circuit>& entries) {
  std::ofstream f(path);
  if (!f) {
    std::cerr << "Failed to open " << path << "\n";
    return;
  }

  for (const auto& circ : entries) {
    f << "[g=" << circ.gate_count() << " cf=" << circ.cofactors.size() << "] " << circ.to_string() << "\n";

    std::unordered_map<tt::TT4, std::vector<const circ::Cofactor*>> by_p;
    for (const auto& cf : circ.cofactors)
      by_p[cf.function_p].push_back(&cf);

    std::vector<tt::TT4> p_sorted;
    p_sorted.reserve(by_p.size());
    for (auto& kv : by_p)
      p_sorted.push_back(kv.first);
    std::sort(p_sorted.begin(), p_sorted.end());

    for (tt::TT4 p : p_sorted) {
      for (const circ::Cofactor* cf : by_p[p]) {
        tt::TT4 np = tt::canon_np[p];
        tt::TT4 npn = tt::canon_npn[p];

        const uint16_t skip_isas =
            (1u << circ::PRE_SVE2) | (1u << circ::SVE) | (1u << circ::NEON_SVE) | (1u << circ::NEON_SVE2);
        IsaMask isa_mask = cf->isa_optimal_mask & ~skip_isas;
        f << "  <" << format_bindings(circ, cf->bindings) << "> [" << format_isa_tags(isa_mask) << "]";
        f << " NPN:0x" << tt::hex(npn) << " NP:0x" << tt::hex(np) << " P:0x" << tt::hex(p) << "\n";
      }
    }
    f << "\n";
  }
}

inline void print_stats(const std::vector<circ::Circuit>& entries) {
  using tt::TT4;

  // --------------------------- build best(P-rep) table ------------------------
  // best_p[isa][p_rep] = optimal gatecount for that P-class on that ISA
  std::array<std::array<GateCount, TT4_SPACE>, circ::NUM_ISAS> best_p{};
  for (auto& row : best_p)
    row.fill(kInfGateCount);

  for (const auto& c : entries) {
    const GateCount gc = c.gate_count();
    for (const auto& cf : c.cofactors) {
      IsaMask m = cf.isa_optimal_mask;
      while (m) {
        const int bit = __builtin_ctz((unsigned)m);
        m &= IsaMask(m - 1);
        auto& cell = best_p[bit][cf.function_p];
        if (gc < cell)
          cell = gc;
      }
    }
  }

  // --------------------------- NP/NPN rep lists + indices ---------------------
  // Dense indices for greedy cover / hist loops (no hashing, O(1) lookup).
  std::array<uint16_t, TT4_SPACE> index_np{}, index_npn{};
  index_np.fill(std::numeric_limits<uint16_t>::max());
  index_npn.fill(std::numeric_limits<uint16_t>::max());

  std::vector<TT4> reps_np;
  std::vector<TT4> reps_npn;
  reps_np.reserve(512);
  reps_npn.reserve(256);

  {
    std::array<uint8_t, TT4_SPACE> seen_np{};
    std::array<uint8_t, TT4_SPACE> seen_npn{};
    seen_np.fill(0);
    seen_npn.fill(0);

    for (uint32_t i = 0; i < TT4_SPACE; ++i) {
      TT4 f = (TT4)i;

      TT4 rnp = tt::canon_np[f];
      if (!seen_np[rnp]) {
        seen_np[rnp] = 1;
        reps_np.push_back(rnp);
      }

      TT4 rnpn = tt::canon_npn[f];
      if (!seen_npn[rnpn]) {
        seen_npn[rnpn] = 1;
        reps_npn.push_back(rnpn);
      }
    }

    std::sort(reps_np.begin(), reps_np.end());
    std::sort(reps_npn.begin(), reps_npn.end());

    // Build indices after sort
    for (uint16_t i = 0; i < (uint16_t)reps_np.size(); ++i)
      index_np[reps_np[i]] = i;
    for (uint16_t i = 0; i < (uint16_t)reps_npn.size(); ++i)
      index_npn[reps_npn[i]] = i;
  }

  // --------------------------- histogram helpers ------------------------------
  auto bump_hist = [&](std::array<uint32_t, MAX_GATES + 1>& h, GateCount gc, uint32_t w = 1) {
    if (gc == kInfGateCount)
      return;
    if (gc <= (GateCount)MAX_GATES)
      h[(size_t)gc] += w;
  };

  auto sum_hist = [&](const std::array<uint32_t, MAX_GATES + 1>& h) -> uint32_t {
    uint32_t s = 0;
    for (auto v : h)
      s += v;
    return s;
  };

  auto write_row = [&](const std::string& isa, const std::string& type, const std::array<uint32_t, MAX_GATES + 1>& h) {
    std::cout << "| " << std::left << std::setw(11) << isa << " | " << std::left << std::setw(6) << type << " |";
    for (int g = 0; g <= (int)MAX_GATES; ++g)
      std::cout << " " << std::right << std::setw(7) << h[(size_t)g] << " |";
    std::cout << "\n";
  };

  auto write_table_header = [&]() {
    std::cout << "| ISA         | Type   |";
    for (int g = 0; g <= (int)MAX_GATES; ++g)
      std::cout << " " << std::right << std::setw(7) << ("g=" + std::to_string(g)) << " |";
    std::cout << "\n";

    std::cout << "|:------------|:-------|";
    for (int g = 0; g <= (int)MAX_GATES; ++g)
      std::cout << "--------:|";
    std::cout << "\n";
  };

  // --------------------------- greedy set cover -------------------------------
  struct CoverResult {
    uint32_t chosen = 0;
    uint32_t covered = 0;
    uint32_t total = 0;
  };

  auto greedy_cover = [&](const std::vector<std::vector<uint16_t>>& sets, uint32_t n_elems) -> CoverResult {
    CoverResult r;
    r.total = n_elems;

    if (n_elems == 0)
      return r;

    const uint32_t n_sets = (uint32_t)sets.size();

    std::vector<uint8_t> covered(n_elems, 0);
    std::vector<uint8_t> active(n_sets, 1);
    std::vector<uint16_t> score(n_sets, 0);

    uint16_t max_score = 0;
    for (uint32_t i = 0; i < n_sets; ++i) {
      uint16_t s = (uint16_t)sets[i].size();
      score[i] = s;
      if (s > max_score)
        max_score = s;
    }

    std::vector<std::vector<uint32_t>> bucket((size_t)max_score + 1);
    for (uint32_t i = 0; i < n_sets; ++i)
      if (score[i])
        bucket[score[i]].push_back(i);

    // post[e] = list of sets containing element e
    std::vector<std::vector<uint32_t>> post(n_elems);
    for (uint32_t i = 0; i < n_sets; ++i)
      for (uint16_t e : sets[i])
        post[e].push_back(i);

    uint32_t covered_count = 0;
    int best = (int)max_score;

    while (covered_count < n_elems && best > 0) {
      uint32_t pick = std::numeric_limits<uint32_t>::max();
      auto& b = bucket[(size_t)best];
      while (!b.empty()) {
        uint32_t i = b.back();
        b.pop_back();
        if (!active[i])
          continue;
        if (score[i] != (uint16_t)best)
          continue;
        pick = i;
        break;
      }
      if (pick == std::numeric_limits<uint32_t>::max()) {
        --best;
        continue;
      }

      active[pick] = 0;
      r.chosen++;

      for (uint16_t e : sets[pick]) {
        if (covered[e])
          continue;
        covered[e] = 1;
        covered_count++;

        for (uint32_t j : post[e]) {
          if (!active[j])
            continue;
          uint16_t old = score[j];
          if (!old)
            continue;
          uint16_t now = (uint16_t)(old - 1);
          score[j] = now;
          if (now)
            bucket[now].push_back(j);
        }
      }
    }

    r.covered = covered_count;
    return r;
  };

  // --------------------------- cover-set builder ------------------------------
  auto build_sets_for = [&](IsaMask require_opt_mask, int kind /*0=P,1=NP,2=NPN*/) {
    std::vector<std::vector<uint16_t>> sets;
    sets.reserve(entries.size());

    // per-circuit dedup by representative TT4 value
    std::array<uint32_t, TT4_SPACE> stamp{};
    stamp.fill(0);
    uint32_t cur = 1;

    for (const auto& c : entries) {
      if (++cur == 0) {
        stamp.fill(0);
        cur = 1;
      }

      std::vector<uint16_t> elems;
      elems.reserve(std::min<size_t>(c.cofactors.size(), 256));

      for (const auto& cf : c.cofactors) {
        if (!(cf.isa_optimal_mask & require_opt_mask))
          continue;

        TT4 p = cf.function_p;
        TT4 rep = p;
        uint16_t idx = std::numeric_limits<uint16_t>::max();

        if (kind == 0) {
          idx = tt::index_p[rep];
        } else if (kind == 1) {
          rep = tt::canon_np[p];
          idx = index_np[rep];
        } else {
          rep = tt::canon_npn[p];
          idx = index_npn[rep];
        }

        if (idx == std::numeric_limits<uint16_t>::max())
          continue;

        if (stamp[rep] == cur)
          continue;
        stamp[rep] = cur;

        elems.push_back(idx);
      }

      if (!elems.empty()) {
        std::sort(elems.begin(), elems.end());
        elems.erase(std::unique(elems.begin(), elems.end()), elems.end());
        sets.push_back(std::move(elems));
      }
    }

    return sets;
  };

  // --------------------------- header -----------------------------------------
  std::cout << "Statistics\n";
  std::cout << "============================================================\n\n";
  std::cout << "Types:\n";
  std::cout << "  P-4   : " << tt::reps_p.size() << " P-classes\n";
  std::cout << "  NP-4  : " << reps_np.size() << " NP-classes\n";
  std::cout << "  NPN-4 : " << reps_npn.size() << " NPN-classes\n";
  std::cout << "  ID-4  : 65536 exact functions\n\n";

  // --------------------------- histogram table --------------------------------
  write_table_header();

  const uint16_t skip_isas =
      (1u << circ::PRE_SVE2) | (1u << circ::SVE) | (1u << circ::NEON_SVE) | (1u << circ::NEON_SVE2);
  for (int iisa = circ::NUM_ISAS - 1; iisa >= 0; --iisa) {
    if (skip_isas & (1u << iisa))
      continue;

    std::array<uint32_t, MAX_GATES + 1> h_p{}, h_np{}, h_npn{}, h_id{};
    h_p.fill(0);
    h_np.fill(0);
    h_npn.fill(0);
    h_id.fill(0);

    // P histogram
    for (TT4 rep : tt::reps_p)
      bump_hist(h_p, best_p[iisa][rep], 1);

    // NP histogram (min over all P reps mapping into that NP rep)
    std::array<GateCount, TT4_SPACE> best_np{};
    best_np.fill(kInfGateCount);
    for (TT4 pr : tt::reps_p) {
      TT4 nr = tt::canon_np[pr];
      GateCount g = best_p[iisa][pr];
      if (g < best_np[nr])
        best_np[nr] = g;
    }
    for (TT4 nr : reps_np)
      bump_hist(h_np, best_np[nr], 1);

    // NPN histogram (min over all P reps mapping into that NPN rep)
    std::array<GateCount, TT4_SPACE> best_npn{};
    best_npn.fill(kInfGateCount);
    for (TT4 pr : tt::reps_p) {
      TT4 nr = tt::canon_npn[pr];
      GateCount g = best_p[iisa][pr];
      if (g < best_npn[nr])
        best_npn[nr] = g;
    }
    for (TT4 nr : reps_npn)
      bump_hist(h_npn, best_npn[nr], 1);

    // ID histogram: weight each P rep by its orbit size under input permutations
    for (TT4 pr : tt::reps_p) {
      GateCount g = best_p[iisa][pr];
      const uint16_t pidx = tt::index_p[pr];
      const uint32_t w = (pidx < tt::orbit_p.size()) ? (uint32_t)tt::orbit_p[pidx].size() : 0;
      bump_hist(h_id, g, w);
    }

    write_row(circ::NAMES[iisa], "NPN-4", h_npn);
    write_row("", "NP-4", h_np);
    write_row("", "P-4", h_p);
    write_row("", "ID-4", h_id);
    if (iisa > 0) {
      std::cout << "|-------------|--------|";
      for (int g = 0; g <= (int)MAX_GATES; ++g)
        std::cout << "---------|";
      std::cout << "\n";
    }
  }

  // --------------------------- greedy coverage (6 figures) --------------------
  const uint32_t N_P = (uint32_t)tt::reps_p.size();
  const uint32_t N_NP = (uint32_t)reps_np.size();
  const uint32_t N_NPN = (uint32_t)reps_npn.size();

  auto run_cover_triplet = [&](const char* isa_name, IsaMask isa_bit) {
    const auto sets_npn = build_sets_for(isa_bit, 2);
    const auto sets_np = build_sets_for(isa_bit, 1);
    const auto sets_p = build_sets_for(isa_bit, 0);

    const auto cov_npn = greedy_cover(sets_npn, N_NPN);
    const auto cov_np = greedy_cover(sets_np, N_NP);
    const auto cov_p = greedy_cover(sets_p, N_P);

    std::cout << "  " << isa_name << "\n";
    auto pr = [&](const char* name, const CoverResult& r) {
      std::cout << "    " << std::left << std::setw(6) << name << ": chosen=" << std::setw(4) << r.chosen
                << "  covered=" << r.covered << "/" << r.total;
      if (r.covered != r.total)
        std::cout << "  (UNREACHABLE=" << (r.total - r.covered) << ")";
      std::cout << "\n";
    };
    pr("NPN-4", cov_npn);
    pr("NP-4", cov_np);
    pr("P-4", cov_p);
  };

  std::cout << "\nGreedy upper-bounds: #circuits needed (optimal per ISA)\n";
  std::cout << "------------------------------------------------------\n";
  run_cover_triplet("NEON", IsaMask(1) << circ::NEON);
  std::cout << "\n";
  run_cover_triplet("NEON_SHA3", IsaMask(1) << circ::NEON_SHA3);
  std::cout << "\n";
  run_cover_triplet("SVE2", IsaMask(1) << circ::SVE2);
  std::cout << "\n";
}

} // namespace output

// =============================================================================
// Main
// =============================================================================
int main(int /*argc*/, char* /*argv*/[]) {
  tt::init();

  search::prog::State st;
  st.total_classes = static_cast<unsigned>(tt::COUNT_P);

  std::cout << "\n" << tty::BOLD << "Search" << tty::RESET << "\n" << std::string(60, '=') << "\n\n";
  std::vector<circ::Circuit> entries = search::run(st);

  std::cout << "\n" << entries.size() << " circuits found" << "\n";

  std::cout << "\n" << tty::BOLD << "Supercircuits" << tty::RESET << "\n" << std::string(60, '=') << "\n\n";
  entries = supercirc::run(std::move(entries));

  std::sort(entries.begin(), entries.end(), [](const circ::Circuit& a, const circ::Circuit& b) {
    if (a.gates.size() != b.gates.size())
      return a.gates.size() < b.gates.size();
    return a.cofactors.size() > b.cofactors.size();
  });

  output::save_circuits(OUTPUT_FILE, entries);
  std::cout << "\nCircuits saved: " << OUTPUT_FILE << "\n\n";
  output::print_stats(entries);
  return 0;
}
