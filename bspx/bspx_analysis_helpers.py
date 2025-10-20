"""Utilities for parsing and reshaping bspx analysis reports.

Public API:
    - load_report(path: Path) -> pd.DataFrame

Changes vs previous version:
    - Rename the aggregated box-count column from `P{avg}.boxes` to `P{total}.boxes`.
      (Kept in the exact same relative position in the final column order.)
    - Internal helpers refactored/simplified; behavior preserved.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

# ---- Constants & Regex -------------------------------------------------------------------------
# Bands we know how to aggregate over for probe metrics
KS = (4, 8, 12, 16, 20, 24, 28, 32)

# New probe column naming: PROBE_<metric>[K]
_probe_col_re = re.compile(r"^PROBE_(.+)\[(\d+)\]$")


# ---- Parsing -----------------------------------------------------------------------------------
def parse_metrics_series(metrics_col: pd.Series) -> pd.DataFrame:
    """
    Parse 'k=v|k=v|...' strings into a wide DataFrame of floats.
    Vectorized via split → explode → last-value-per-key → pivot.
    Missing keys are filled with 0.0.
    """
    s = metrics_col.astype("string").fillna("")
    if (s == "").all():
        return pd.DataFrame(index=s.index)

    # Split and explode keeping original row index
    exploded = s.str.split("|").explode()
    exploded = exploded[exploded.notna() & (
        exploded != "") & exploded.str.contains("=")]

    # Split into key/value
    kv = exploded.str.split("=", n=1, expand=True)
    kv.columns = ["key", "val"]
    kv["row"] = exploded.index
    kv["val"] = pd.to_numeric(kv["val"], errors="coerce")

    # Keep the last value per (row,key)
    kv_last = kv.groupby(["row", "key"], sort=False, as_index=False).last()

    # Pivot back to wide; align to original index and fill missing with 0.0
    wide = kv_last.pivot(index="row", columns="key", values="val")
    wide = wide.reindex(index=s.index).fillna(0.0)

    # Stable, sorted columns
    wide = wide.reindex(columns=sorted(wide.columns))
    return wide


# ---- Aggregation across bands ------------------------------------------------------------------
def compute_stats_probe_pavgs(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given stats-probe rows of the expanded report df, compute:
      - totals across bands for boxes:   ➊ `P{total}.boxes`
      - weighted averages across bands for every other metric present as
        `PROBE_<metric>[K]` → `P{avg}.<metric>`, using weights = `PROBE_boxes[K]`.
    Returns a DataFrame with those derived columns only.
    """
    if stats_df.empty:
        return pd.DataFrame(index=stats_df.index)

    # Discover all per-band metrics of the form PROBE_<metric>[K]
    metrics: set[str] = set()
    for col in stats_df.columns:
        m = _probe_col_re.match(col)
        if not m:
            continue
        metric_name = m.group(1)
        K = int(m.group(2))
        if K in KS:
            metrics.add(metric_name)

    out = pd.DataFrame(index=stats_df.index)

    # Sum of boxes across available bands → P{total}.boxes
    box_cols = [
        f"PROBE_boxes[{K}]" for K in KS if f"PROBE_boxes[{K}]" in stats_df.columns]
    if box_cols:
        with np.errstate(invalid="ignore"):
            out["P{total}.boxes"] = np.nansum(
                stats_df[box_cols].to_numpy(copy=False), axis=1)

    # Weighted averages per metric (vectorized with numpy)
    for metric in sorted(metrics):
        if metric == "boxes":
            continue
        ks_avail = [
            K for K in KS
            if f"PROBE_{metric}[{K}]" in stats_df.columns and f"PROBE_boxes[{K}]" in stats_df.columns
        ]
        if not ks_avail:
            continue

        vcols = [f"PROBE_{metric}[{K}]" for K in ks_avail]
        wcols = [f"PROBE_boxes[{K}]" for K in ks_avail]

        V = stats_df[vcols].to_numpy(copy=False)
        W = stats_df[wcols].to_numpy(copy=False)

        with np.errstate(invalid="ignore"):
            num = np.nansum(V * W, axis=1)
            den = np.nansum(W, axis=1)
            pav = np.divide(num, den, out=np.full_like(
                num, np.nan, dtype=float), where=den != 0)

        out[f"P{{avg}}.{metric}"] = pav

    return out


# ---- Column ordering ----------------------------------------------------------------------------
def reorder_probe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order columns as:
      [file,dataset,bitset,n,policy,bits],
      [non-probe metrics...],
      [P{total}.boxes, P{avg}.*],
      [PROBE_*[32], PROBE_*[28], ..., PROBE_*[4]],
    - The P{avg}.* metric ordering is alphabetical.
    - P{total}.boxes is placed exactly where P{avg}.boxes used to be (front of the P{avg} block).
    - Per-band blocks use the metric order: boxes first, then the same alphabetical metrics.
    Only columns that exist are included.
    """
    # 1) Front identifiers (keep only those present)
    front = [c for c in ["file", "dataset", "bitset",
                         "n", "policy", "bits"] if c in df.columns]

    # 2) Determine metric order from P{avg}.* (excluding boxes which no longer exists there)
    pavg_cols = [c for c in df.columns if c.startswith("P{avg}.")]
    # e.g., avg_density, avg_runs, ...
    metrics = sorted({c.split(".", 1)[1] for c in pavg_cols})

    # P{avg} block (with P{total}.boxes at the front, if present)
    pavg_ordered = []
    if "P{total}.boxes" in df.columns:
        pavg_ordered.append("P{total}.boxes")
    pavg_ordered.extend(
        [c for c in (f"P{{avg}}.{m}" for m in metrics) if c in df.columns])

    # 3) Per-band blocks, highest K → lowest K, using [boxes] + metrics order
    Ks_present = sorted({
        int(m.group(2))
        for col in df.columns
        for m in [_probe_col_re.match(col)] if m and int(m.group(2)) in KS
    }, reverse=True)
    band_ordered: list[str] = []
    metric_order_for_bands = ["boxes"] + metrics
    for K in Ks_present:  # e.g., 32,28,...,4
        for m in metric_order_for_bands:
            col = f"PROBE_{m}[{K}]"
            if col in df.columns:
                band_ordered.append(col)

    # 4) Everything else, in their existing relative order
    used = set(front) | set(pavg_ordered) | set(band_ordered)
    the_rest = [c for c in df.columns if c not in used]

    new_cols = front + the_rest + pavg_ordered + band_ordered
    # Safety: dedupe while preserving order
    new_cols = list(dict.fromkeys(new_cols))
    return df[new_cols]


# ---- Public API ---------------------------------------------------------------------------------
def load_report(path: Path) -> pd.DataFrame:
    """
    Read report.csv, expand the 'metrics' column, derive dataset/bitset.
    Expects file names like 'dataset.csv7.txt' -> dataset='dataset', bitset=7.
    Computes per-band aggregates for stats-probe rows, then broadcasts those
    metrics to other policies sharing the same file and drops the stats-probe rows.
    """
    df = pd.read_csv(path)

    # Types
    for col in ("file", "policy"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "bits" in df.columns:
        df["bits"] = pd.to_numeric(df["bits"], errors="coerce")

    # Derive dataset & bitset from file name
    file_last = df["file"].astype(str).str.rsplit("/", n=1).str[-1]
    extracted = file_last.str.extract(
        r"^(?P<dataset>[^.]+)\.csv(?P<bitset>\d+)\.", expand=True)
    assert extracted.notna().all().all(), "Expected file like '<dataset>.csv<bitset>.txt'"
    df["dataset"] = extracted["dataset"]
    df["bitset"] = extracted["bitset"].astype(int)

    # Expand metrics column
    metrics_df = parse_metrics_series(
        df["metrics"]) if "metrics" in df.columns else pd.DataFrame(index=df.index)
    df = pd.concat([df.drop(columns=[c for c in ["metrics"]
                   if c in df.columns]), metrics_df], axis=1)
    metric_cols = list(metrics_df.columns)

    # Mask for stats-probe rows (or all rows if policy is absent)
    has_policy = "policy" in df.columns
    stats_mask = df["policy"].eq(
        "stats-probe") if has_policy else pd.Series(True, index=df.index)

    # Compute P{avg}.* and P{total}.boxes on stats-probe rows only, then attach
    pavgs_and_total = compute_stats_probe_pavgs(df.loc[stats_mask])
    if not pavgs_and_total.empty:
        df.loc[stats_mask, pavgs_and_total.columns] = pavgs_and_total.to_numpy(
            copy=False)

    # Identify metric columns supplied by stats-probe: (non-stats rows have 0.0 defaults)
    non_stats_vals = df.loc[~stats_mask, metric_cols].replace(0.0, np.nan)
    stats_given_mask = ~non_stats_vals.notna().any(axis=0)
    stats_given_cols = list(non_stats_vals.columns[stats_given_mask])

    # Include derived columns for broadcasting
    derived_cols = list(
        pavgs_and_total.columns) if not pavgs_and_total.empty else []
    broadcast_cols = sorted(set(stats_given_cols).union(derived_cols))

    # Broadcast stats-probe metrics to other policies sharing the same file, then drop stats-probe rows
    # Collapse stats-probe rows to one row per file for broadcast (take first fast)
    source = (
        df.loc[stats_mask, ["file"] + broadcast_cols]
        .drop_duplicates(subset="file", keep="first")
    )

    # Apply to non-stats rows via merge; overwrite broadcast cols in one go
    non_stats = df.loc[~stats_mask].copy()
    merged = non_stats.merge(
        source, on="file", how="left", suffixes=("", "__stats"))

    present = [c for c in broadcast_cols if f"{c}__stats" in merged.columns]
    if present:
        merged[present] = merged[[
            f"{c}__stats" for c in present]].to_numpy(copy=False)
        merged.drop(columns=[f"{c}__stats" for c in present],
                    inplace=True, errors="ignore")

    df = merged  # stats-probe rows dropped

    # Drop identical sets
    df = df.drop_duplicates(
        subset=["policy", "dataset", "PROBE_hash"], keep="first")

    # Reorder: key identifiers first, then metrics/blocks
    df = reorder_probe_columns(df)
    return df
