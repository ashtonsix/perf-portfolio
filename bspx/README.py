# %% [markdown]
# # BSP-inspired bitsets: 46% smaller than Roaring
#
# > **Scope:** Pre-engineering research measuring compression only. Performance estimated.
# > **Created by [Ashton Six](https://ashtonsix.com)** (available for hire).
#
# ---
#
# **Context:** Search, OLAP databases, and stream analytics (Lucene, Spark, Kylin, Druid, ClickHouse, …) rely on compressed bitsets for fast filtering, joins, set algebra, and counts. Roaring-style bitset implementations dominate, which balance compression with operational speed.
#
# **Problem:** Bitset performance is typically memory-bound—once the memory bus saturates (~6 GB/s), extra compute won't accelerate operations (e.g., AND/OR/XOR). This motivates _operational compression_: keeping sets compressed in memory to stream more data through the CPU while maintaining set algebra speed.
#
# **Approach:** We compare Roaring and Judy against Binary Space Partitioning (BSP)-style encoders and two-stage hybrids. **This work measures compression ratios only**—we estimate throughput characteristics through system modeling, but do not implement or benchmark actual codecs.
#
# **Findings:**
#
# * **BSP dominates compression.** **Left-popcount (LP)** and **median-index (MI)** yield bitsets ~**46% smaller** than the best non-BSP baseline (Roaring+RLE) on aggregate.
# * **Hybrid policies don't help compression.** Mixing LP+MI and/or adding preprocessing passes doesn't pay off; control overhead erases gains and adds complexity.
# * **Entropy concentrates in the low 8-12 bits.** Two-stage designs can maintain single-stage compression ratios while improving codec practicality.
#
# **Outcome:** We identify a promising two-stage architecture:
#
#   * **Stage 1 (high bits):** Performance-oriented indexing (Judy16 or similar)
#   * **Stage 2 (low 8-12 bits):** Compression-oriented encoding (LP, MI, or micro-containers)
#
# LP/MI offer the best compression; simpler micro-containers (e.g., pickbest) sacrifice ~20% compression but should prove easier to accelerate.
#
# ---
#
# # Preamble
#
# This notebook distills results from an experimental C++ workbench (`bspx_*.cpp`). Each experiment composes transforms (restructure data and/or add sidechains) with a terminal policy (bit-count model). Pipelines run per dataset; results are aggregated to CSV and analysed here.
#
# **Workbench layout**
#
# * `bspx_main.cpp` — declares pipelines, runs in parallel, writes `report.csv`.
# * `bspx_policy.hpp` — policies (LP, MI, …), transforms (cluster, RLE, …), helpers.
# * `bspx_runtime.hpp` — data loader, pipeline executor, CSV/report utilities.
# * `bspx_telemetry.hpp` — metrics emitted by transforms/policies.
#
# **Running**
#
# ```sh
# make -f bspx.mk && ./out/bspx
# ```
#
# **Datasets**
#
# * CRoaring “realdata” (`census-income`, `census1881`, `uscensus2000`, `weather_sept_85`, `wikileaks-noquotes`).
# * Contains 1800 files, each with a sorted `uint32_t` set.
# * Some files are exact duplicates, we evaluate the 1570 unique sets.

# %%
# Imports, constants, report.csv import
from matplotlib.ticker import FixedLocator, FuncFormatter
from pathlib import Path
import bspx_analysis_helpers as bspx
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display

PROJECT_ROOT = Path.cwd()
REPORT_PATH = PROJECT_ROOT / "report.csv"

df = bspx.load_report(REPORT_PATH)
bitsets_per_dataset = df.groupby("dataset")["bitset"].nunique()
print(
    f"Loaded {len(df):,} rows x {len(df.columns):,} columns across {df['file'].nunique():,} datasets and {df['policy'].nunique():,} policies.\n")
print("Unique bitsets per dataset:")
print(bitsets_per_dataset)

df.head(3)

# %% [markdown]
# # Background
#
# ## Prior Art
#
# * **Judy (~2001–2004).** High-fanout radix arrays; influential but pointer-heavy.
# * **RLE family (WAH→PLWAH/CONCISE/EWAH, ~2004–2010).** Word-aligned run-length encoders; historically important, superseded by Roaring.
# * **Roaring (2014) → run containers (2016).** Two-level partitioning (high-16-bit) with per-container encoding (array/bitmap). Initial results showed Roaring as **4–5× faster** than WAH for intersections; later **run containers** yielded consistently smaller bitmaps than RLE-only schemes.
# * **BSP / quasi-succinct (BIC/PEF, 2000s–2010s).** State-of-the-art compression for bitsets via recursive bisection. Dominate research-grade inverted indices (PISA); rarely used for industry DB engines due to engineering pitfalls.
#
# ## Our contribution
#
# We revive Judy’s high-bit partitioning and apply BSP-inspired coding to the low bits within each partition.
#
# ---
#
# # Method and Results: single-stage policies at a glance
#
# All BSP policies descend by recursive bisection from the universe:
#
# * **Left-popcount (LP):** split at midpoint; encode the left-side popcount under a feasible alphabet; recurse.
# * **Median-index (MI):** encode a feasible median position; recurse strictly left/right.
# * **Tagged-bisect:** tag local regime (run/sparse/co-sparse/balanced); recurse if balanced.
#
# Let us compare aggregate sizes:
#
# - **Roaring vs Judy (baselines).** Plain Roaring is ~5.4% smaller than Judy256 (119.37 vs 126.16 Mb). Adding Run Length Encoding (RLE) to Roaring trims ~37.0% (→ 75.16). Reducing Judy fanout trims ~24.8% / ~29.5% (→ 94.92 / 88.88) for 16-ary / 4-ary fanout respectively. Roaring+RLE wins overall.
# - **BSP family (LP / MI / Tagged-bisect).** **MI** edges the best total (39.59) but **LP** (40.65) wins more datasets (6/9), i.e., it’s steadier. **Tagged-bisect** is only competitive on run-heavy data and loses overall (49.60).
# - **BSP vs Roaring+RLE.** **LP** beats Roaring+RLE on 9/9 datasets; **MI** wins 7/9 with two close losses on `wikileaks-noquotes` (both variations). Net: BSP policies are \~45.9–47.3% smaller on aggregate than Roaring+RLE.

# %%


def megabit_table(df: pd.DataFrame, policies: list[str], quiet=False) -> pd.DataFrame:
    df_baselines = df[df["policy"].isin(policies)].copy()
    total_Mb = df_baselines.groupby("policy")["bits"].sum().div(1e6)
    per_dataset_Mb = df_baselines.groupby(["policy", "dataset"])[
        "bits"].sum().div(1e6).unstack("dataset", fill_value=0)
    table = pd.DataFrame({"total": total_Mb}).reindex(
        policies).join(per_dataset_Mb, how="left").fillna(0)
    assert table.shape[0] == len(policies)
    if not quiet:
        print("Each column shows aggregate megabits across bitsets in a dataset:")
    return table


megabit_table(df, ["roaring-B+A", "roaring-B+A+R", "judy256",
              "judy16", "judy4", "leftpop", "median-index", "tagged-bisect"])

# %% [markdown]
# # Left-Popcount vs Median-Index: deeper look

# %%
# Join to form one row per file
df_lp = df[df["policy"] == "leftpop"]
df_mi = df[df["policy"] == "median-index"]
df_vs = df_lp.set_index("file").join(df_mi.set_index("file")[["bits", "BSP_mi_nodes"]].rename(columns={
    "bits": "mi_bits", "BSP_mi_nodes": "mi_nodes"}), how="inner").reset_index().rename(columns={"bits": "lp_bits", "BSP_lp_nodes": "lp_nodes"})
df_vs = df_vs[["file", "dataset", "bitset", "n", "lp_bits", "mi_bits", "lp_nodes",
               "mi_nodes", "P{total}.boxes"] + [c for c in df_vs.columns if c.startswith("P{avg}")]]

df_vs.head()

# %% [markdown]
# MI edges LP on near-uniform (hard-to-compress) bit patterns, while LP wins—and usually by larger *percentage* margins—on structured (compressible) patterns.
#
# This creates a relative-absolute reversal: LP wins more often and by bigger percentages, MI’s wins are rarer but larger in absolute bits because they occur on big rows.

# %%
wins_lp = df_vs["lp_bits"] < df_vs["mi_bits"]
wins_mi = df_vs["mi_bits"] < df_vs["lp_bits"]

num_lp_wins = int(wins_lp.sum())
num_total = len(df_vs)
total_lp_Mb = df_vs["lp_bits"].sum() / 1e6
total_mi_Mb = df_vs["mi_bits"].sum() / 1e6
print(
    f"Head-to-head, lp beats mi on {num_lp_wins}/{num_total} files ({num_lp_wins / num_total * 100:.1f}%).")
print(
    f"On aggregate, lp uses {total_lp_Mb:.2f} Mb, and mi uses {total_mi_Mb:.2f} Mb.\n")

bits_best_lp_win = df_vs.loc[wins_lp, ["lp_bits", "mi_bits"]].min(axis=1)
bits_best_mi_win = df_vs.loc[wins_mi, ["lp_bits", "mi_bits"]].min(axis=1)
best_bpe_lp_win = bits_best_lp_win / df_vs.loc[wins_lp, "n"]
best_bpe_mi_win = bits_best_mi_win / df_vs.loc[wins_mi, "n"]

avg_best_bits_lp_win = bits_best_lp_win.mean()
avg_best_bits_mi_win = bits_best_mi_win.mean()
avg_best_bpe_lp_win = best_bpe_lp_win.mean()
avg_best_bpe_mi_win = best_bpe_mi_win.mean()

print(f"Average `lp_bits`   where LP beats MI: {avg_best_bits_lp_win:.3f}")
print(f"Average `mi_bits`   where MI beats LP: {avg_best_bits_mi_win:.3f}")
print(f"Average `lp_bits/n` where LP beats MI: {avg_best_bpe_lp_win:.5f}")
print(f"Average `mi_bits/n` where MI beats LP: {avg_best_bpe_mi_win:.5f}\n")


def percentile_table(series, qs=(0.50, 0.75, 0.90, 0.95, 0.99)):
    q_vals = series.quantile(qs)
    q_vals.index = [f"p{int(q*100)}" for q in qs]
    return q_vals


def win_margins(df, winner_mask, winner_col, loser_col):
    pct = ((df.loc[winner_mask, loser_col] - df.loc[winner_mask, winner_col])
           / df.loc[winner_mask, loser_col] * 100)
    abits = (df.loc[winner_mask, loser_col] - df.loc[winner_mask, winner_col])
    return pct.dropna(), abits.dropna()


lp_pct, lp_abs = win_margins(df_vs, wins_lp, "lp_bits", "mi_bits")
mi_pct, mi_abs = win_margins(df_vs, wins_mi, "mi_bits", "lp_bits")

df_win_table = (
    pd.concat(
        [
            percentile_table(lp_pct).rename("lp win margin (%)"),
            percentile_table(mi_pct).rename("mi win margin (%)"),
            percentile_table(lp_abs).rename("lp win margin (bits)"),
            percentile_table(mi_abs).rename("mi win margin (bits)"),
        ],
        axis=1,
    )
    .round(2)
)

print(df_win_table.to_string())

# %% [markdown]
# At a span covering $s$ positions with popcount $n$ and co-popcount $m=s-n$ exists feasible alphabet $K_{\text{LP}} = \min(n, m)+1$ for $n_L$. LP exploits structure because these feasibility alphabets narrow rapidly when mass skews locally; entire sub-trees can be quickly eliminated.
#
# For MI the feasible alphabet is $K_{\text{MI}} = m+1$. Its narrowing comes from log concavity under the median split:
#
# $$\log _2(m_L+1)+\log _2(m_R+1) \leq 2 \log _2(m/2+1)$$
#
# which narrows slowly versus LP on structured data. Under a near-uniform distribution ($n \approx m$), $K_{\text{MI}}$ and $K_{\text{LP}}$ coincide, erasing LP's advantage and letting MI occasionally eke wins on hard-to-compress data.
#
# Exploratory data analysis supports this thesis to a _certain_ extent. MI’s wins cluster in the high-node/near-balanced regions, but plots show heavy overlap. Setting aggregates aside, it’s surprisingly hard to look at a bitset and reliably predict which of MI or LP will be smaller without actually running both.

# %%

df2 = df_vs.copy()
df2["mi_win_margin"] = (df2["lp_bits"] - df2["mi_bits"]) / \
    (df2["lp_bits"] + df2["mi_bits"])
df2["lp_nodes_per_value"] = df2["lp_nodes"] / df2["n"]
df2["dataset"] = df2["dataset"].astype("category")

ds_list = df2["dataset"].cat.categories.tolist()
color_map = {ds: matplotlib.colormaps["tab10"](
    i % 10) for i, ds in enumerate(ds_list)}


def scatter(ax, xcol, ycol="mi_win_margin", title=None, xlabel=None, ylabel=None):
    for ds in ds_list:
        mask = (df2["dataset"] == ds)
        ax.scatter(df2.loc[mask, xcol], df2.loc[mask, ycol],
                   s=10, alpha=0.35, label=str(ds), color=color_map[ds])
    ax.axhline(0, color="k", linestyle="--", linewidth=0.75, alpha=0.5)
    ax.set_xlabel(xlabel or xcol)
    ax.set_ylabel(ylabel or ycol)
    ax.set_title(title)
    ax.set_ylim(-0.52, 0.12)


fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
scatter(axes[0, 0], "P{avg}.avg_density",  title="",
        xlabel="Avg. density per node", ylabel="MI win margin (%)")
scatter(axes[1, 0], "lp_nodes_per_value",  title="",
        xlabel="LP nodes per value", ylabel="MI win margin (%)")
axes[1, 0].set_xlim(33, 1.5)
axes[1, 0].set_xscale("log")
major_ticks = [2, 3, 4, 6, 8, 12, 16, 24, 32]
axes[1, 0].xaxis.set_major_locator(FixedLocator(major_ticks))
axes[1, 0].xaxis.set_major_formatter(FuncFormatter(
    lambda x, _: f"{int(x):,}" if x in major_ticks else ""))
scatter(axes[0, 1], "lp_nodes",  title="",
        xlabel="LP node count", ylabel="MI win margin (%)")
axes[0, 1].set_xscale("log")
axes[0, 1].set_xlim(0.5, 5e6)
scatter(axes[1, 1], "mi_nodes",  title="",
        xlabel="MI node count", ylabel="MI win margin (%)")
axes[1, 1].set_xscale("log")
axes[1, 1].set_xlim(0.5, 5e6)

handles, labels = axes[0, 0].get_legend_handles_labels()
leg = fig.legend(handles, labels, loc="upper center",
                 ncol=min(6, len(labels)), frameon=False)
for lh in leg.legend_handles:
    lh.set_alpha(1.0)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
# # Failed Experiments: attempts to improve baseline compression
#
# - Mixed BSP controller `bsp-mixed` (tagged policy switch) did not beat MI (≈+0.2%); the tag overhead erased wins.
# - Cluster prepass `cluster-then-{LP,MI}` inflated bitsets by ~4–5.5% versus baselines.
# - RLE prepass `rle-then-{LP,MI}` gave tiny wins only: ~0.7–1.1% better; not enough to justify the complexity.
# - Early-stop leaf experiments for LP `leftpop-then-{enum,array}` had negligible effect, neither hurting or helping.
#
# **Bottom line:** baseline LP/MI already sit near the practical Pareto frontier for size on these datasets. Further augmentations and/or tweaks are unlikely to yield meaningful improvement.

# %%
policies = [
    "leftpop",
    "median-index",
    "bsp-mixed",
    "leftpop-then@N4-enum",
    "leftpop-then@N1-array",
    "cluster-then-leftpop",
    "cluster-then-median",
    "rle-then-leftpop",
    "rle-then-median"
]

megabit_table(df, policies)

# %% [markdown]
# # Scalar Values: truncated binary codes
#
# In BSP, each node emits a scalar $y\in[0,K)$ whose alphabet $K$ is rarely a power-of-two. Fixed codes using $\lceil \log_2 K\rceil$ bits per value are viable but leave compression on the table.
#
# Let $b=\lfloor\log_2 K\rfloor$ and $t=2^{b+1}-K$ (count of short codewords). **Truncated binary (TB)** encodes:
#
# * If $y<t$: emit $b$ bits for $y$.
# * Else: emit $(b{+}1)$ bits for $y+t$.
#
# **Decoding:** read $b$ bits $\to x$. If $x<t$, output $y=x$; otherwise read one deciding bit $d$ and output $y=(2x+d)-t$.
#
# Empirically, TB shrinks our aggregate from **43.36→40.65 Mb** (~6.25%). An ideal uniform entropy code would reach **40.13 Mb**, so TB captures **~84%** of the possible gain.
#
# Heavier entropy coding (e.g., ANS with an adaptive context model) could squeeze a few more points, but would add state/renormalization and serial dependencies, undermining random access, update locality, and SIMD parallelism. TB gets most of the benefit while staying simple.

# %%
policies = [
    "leftpop",  # truncated binary
    "leftpop-scalar-fixed",
    "leftpop-scalar-uniform"
]

megabit_table(df, policies)

# %% [markdown]
# # Operational requirements for high-performance bitsets
#
# For a compression scheme to be _operationally viable_ at high throughput, we need:
#
# * **Short-circuitability.** Trivialise absorbing/neutral spans (all-zero/all-one) and surface cardinality/membership without full decode.
# * **Slice locality.** Address and operate on ranges (e.g., `key in [L,R)`) with bounded work and good update locality.
# * **Zipped traversal.** Tight `load chunk → operate → write` loops, streaming multiple inputs in lockstep and avoiding inflate-to-RAM.
#
# Single-stage BSP (LP/MI) fails these requirements: subtree size is unknown until descent; random access touches many nodes; feasible alphabets depend on ancestor state. These don't preclude fast bulk operations, but significantly complicate implementation and limit operational flexibility.
#
# ## Solution Shape: two-stage layout
#
# We split the universe by high (Stage-1) and low (Stage-2) bits:
#
# - **Stage-1:** Performance-oriented map for non-empty spans (e.g., Judy16 over high 16 bits). Provides fast skip/copy for trivial cases and bounded access cost.
# - **Stage-2:** Compression-oriented encoding for low bits (e.g., LP/MI, or simpler micro-containers). Blocks live in small, independent byte ranges and decode only for non-trivial work.
#
# In aggregate, Stage-1 overhead is small: Judy16 indexing adds just 2.66 Mb (5.9% of total) when applied to low-8 LP-coded bits. The larger cost comes from metadata per Stage-2 root—recording cardinality alone requires 6.03 Mb (13.4% of total). Smaller Stage-2 block size covering low-$D$ bits improves throughput potential (at the cost of metadata overhead), while larger $D$ minimizes overhead (at the cost of operational flexibility).

# %%
policies = [
    "leftpop",
    "judy16-then@D16-leftpop",
    "judy16-then@D12-leftpop",
    "judy16-then@D8-leftpop",
    "median-index",
    "judy16-then@D16-median-index",
    "judy16-then@D12-median-index",
    "judy16-then@D8-median-index",
]

table = (
    df[df["policy"].isin(policies)]
    .assign(bsp_total=lambda d: d["R_bits"].where(d["R_bits"].ne(0), d["bits"]))
    .assign(bsp_n=lambda d: d["R_XFORM_bits"].where(d["R_XFORM_bits"].ne(0), d["XFORM_bits"]))
    .assign(bsp_body=lambda d: d["bsp_total"] - d["bsp_n"])
    .groupby("policy")[["bits", "L_bits", "bsp_n", "bsp_body"]]
    .sum().div(1e6).reindex(policies)
    .rename(columns={"bits": "Total (Mb)", "L_bits": "Judy (Mb)", "bsp_n": "BSP popcount (Mb)", "bsp_body": "BSP body (Mb)"})
    [["Total (Mb)", "Judy (Mb)", "BSP popcount (Mb)", "BSP body (Mb)"]]
)

table

# %% [markdown]
# ## Performance potential
#
# > **Speculative performance modeling. Not measured results.**
#
# Assume a 3 GHz core with a 6 GB/s memory bus (~2 bytes/cycle round-trip). For "smaller → faster" scaling, our pipeline must process ≥16 bit positions/cycle to break-even with simple bitsets, and can then continue improving until 42 bit positions/cycle (LP's average efficiency: 12 bytes covers 256 positions).
#
# Consider a two-input AND operation over low-8 bits (256-bit spans):
#
# **Compute budget for break-even:**
# - Simple bitsets: 2×32 bytes read + 32 bytes write = 96 bytes → 48 cycles
# - Compressed bitsets: memory delivers data in 18 cycles, but we have 48 cycles total
# - **Compute budget: 48 cycles** for 2× decode + operate + encode + overhead
# - Allocation: ~15 cycles per decode, ~15 cycles for encode, ~3 cycles for load/store/operate (issue-only)
#
# Even with instructions like bext/bdep 15 cycles per transcode is insufficient for our scheme's complexity level. The bitstream manipulation and control flow involved would substantially exceed this budget.
#
# **Mitigating factors:**
#
# Up to ~50% of work may be memory-bound (copy passthrough from Stage-1 triviality tests), effectively doubling the compute budget to ~30 cycles/transcode. For this to amortize effectively (assume 4096 byte store buffer), Stage-2 blocks should span ≲4096 bit positions: larger blocks would drain the buffer before completing. This favours $D∈\{8,12\}$ over $D=16$. Although even with a doubled compute budget, the margins remain tight given LP/MI transcode complexity.
#
# **Takeaway:**
#
# Beating simple bitset throughput with LP/MI through better memory bus utilization appears challenging. A well-optimized implementation might achieve:
#
# - Comparable throughput to simple bitsets (break-even or similar)
# - Superior performance for disk/network-bound workloads
# - Significant memory footprint reduction (~46%)
#
# This falls short of "smaller → proportionally faster" but may still provide value depending on the deployment context.
#
# **Miscellaneous experiment: tags for full BSP blocks**
#
# Instead of storing BSP root popcount for full blocks (all ones), it's possible to short-circuit with a 1-bit "full tag". This brings a small compression improvement, but the bigger advantage comes from accelerating triviality testing.

# %%
judy16_mb = df[df["policy"] == "judy16-then@D8-leftpop"]["bits"].sum() / 1e6
judyx_mb = df[df["policy"] == "judyX-then@D8-leftpop"]["bits"].sum() / 1e6

print(
    f"Using tags to skip full BSP blocks reduces the aggregate size by {(1 - judyx_mb / judy16_mb) * 100:.2f}% ({judy16_mb:.3f} → {judyx_mb:.3f} Mb)")

# %% [markdown]
# # Performance Solution: Simpler Micro-Containers (PickBest)
#
# If BSP decode complexity proves prohibitive for target throughput, simpler PickBest micro-containers over low-8 bits offer a pragmatic fallback:
#
# 1. Record popcount $p$ (1 byte)
# 2. Pick best storage based on $p$:
#     - If $p < 32$, emit an array.
#     - If $p > 224$, emit a co-array (holes).
#     - Otherwise, emit a 256-bit bitmap.
#
# Popcount gives reservation size, operations vectorize cleanly, streaming performance is predictable. With Judy16 for high bits this lands at 57.60 Mb, smaller than Roaring+RLE (by ~23.35%), and realising about half the compression improvement of single-stage LP.

# %%
policies = [
    "leftpop",
    "judy16-then@D8-pickbest",
    "roaring-B+A+R",
]


megabit_table(df, policies)

# %% [markdown]
# # Conclusion
#
# BSP-inspired policies (LP/MI) establish a new compression baseline: **~46% smaller than Roaring+RLE** on standard benchmarks. However, performance modeling suggests the decode complexity may preclude throughput gains from reduced memory traffic—implementations will likely match (not exceed) simple bitset speeds.
#
# ## Recommended Architecture
#
# The two-stage pattern remains promising:
#
# **Stage 1 (high bits):** Performance-optimized indexing
# * Example: Judy16 (16-ary radix tree over high 16 bits)
# * Role: Fast skip/copy, bounded access cost, minimal overhead (~6%)
#
# **Stage 2 (low 8-12 bits):** Compression-optimized encoding
# * **Compression-first:** LP or MI (~46% smaller vs Roaring+RLE)
#   * Requires complex decode; performance likely break-even at best
#   * Best for memory-constrained or disk/network-bound workloads
# * **Throughput-first:** Micro-containers like PickBest (~23% smaller vs Roaring+RLE)
#   * Simpler decode, predictable performance
#   * Better bet for CPU-bound in-memory workloads
#
# **Block size tradeoff:** D∈{8,12,16} balances metadata overhead (lower D = higher overhead) against operational flexibility (lower D = better locality, easier to amortize decode cost).
#
# ## What This Settles (and Doesn't)
#
# **Settled:**
# * BSP methods can compress bitsets substantially better than Roaring
# * Two-stage designs maintain compression while enabling practical operations
# * Judy16 + LP achieves ~46% improvement, Judy16 + PickBest achieves ~23%
#
# **Open questions requiring implementation:**
# * Can LP/MI decode be optimized to break-even throughput in practice?
# * Do micro-containers like PickBest actually deliver their theoretical performance advantage?
# * In real workloads, how often is data memory-bound vs CPU-bound?
#
# The compression improvements are real and measured. Whether they translate to performance gains remains an implementation question—but the potential is credible enough to warrant the attempt.
