"""
analyze_data.py — Data distribution and characteristics analysis for TPC-H Q3.

This script examines the actual data to inform optimization decisions.
Understanding your data is essential BEFORE optimizing — you need to know:

1. TABLE SIZES — How big is each table? Which dominates I/O?
2. PARQUET STRUCTURE — Row groups, compression ratios, column widths
3. COLUMN CARDINALITY — How many distinct values? Affects hash table sizing.
4. FILTER SELECTIVITY — How much data does each predicate eliminate?
5. SORT ORDER / CLUSTERING — Are columns sorted? Determines row-group skip potential.
6. JOIN FAN-OUT — Orders per customer, lineitems per order → join amplification.
7. KEY DISTRIBUTIONS — Are keys dense/sparse? Uniform/skewed? Affects hash table design.
8. VALUE DISTRIBUTIONS — Price/discount ranges → arithmetic precision considerations.

Usage:
    python analyze_data.py                  # analyze SF=1 (default)
    python analyze_data.py --sf data/sf1    # analyze specific SF
    python analyze_data.py --all            # analyze all SFs and compare scaling
"""

import os
import sys
import argparse
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from datetime import date


def file_size_mb(path: str) -> float:
    """Get file size in megabytes."""
    return os.path.getsize(path) / (1024 * 1024)


def analyze_parquet_structure(data_dir: str) -> None:
    """
    Analyze Parquet file structure: sizes, row groups, compression.
    
    This tells you:
    - Which table dominates I/O (lineitem is typically >50% of total)
    - How many row groups → how many potential skip opportunities
    - Compression ratio → how much decompression work
    """
    print("=" * 80)
    print("1. PARQUET FILE STRUCTURE")
    print("=" * 80)

    tables = ["customer", "orders", "lineitem"]
    total_size = 0

    for table in tables:
        path = os.path.join(data_dir, f"{table}.parquet")
        size = file_size_mb(path)
        total_size += size

        pf = pq.ParquetFile(path)
        meta = pf.metadata

        print(f"\n  {table}.parquet")
        print(f"    File size:      {size:>8.2f} MB")
        print(f"    Total rows:     {meta.num_rows:>10,}")
        print(f"    Total columns:  {meta.num_columns:>10}")
        print(f"    Row groups:     {meta.num_row_groups:>10}")

        # Row group details
        rg_sizes = []
        for rg_idx in range(meta.num_row_groups):
            rg = meta.row_group(rg_idx)
            rg_sizes.append(rg.num_rows)

        print(f"    Rows per group: {min(rg_sizes):,} – {max(rg_sizes):,} "
              f"(avg {np.mean(rg_sizes):,.0f})")

        # Column-level sizes (compressed vs uncompressed) for Q3 columns
        q3_cols = {
            "customer": ["c_custkey", "c_mktsegment"],
            "orders": ["o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"],
            "lineitem": ["l_orderkey", "l_extendedprice", "l_discount", "l_shipdate"],
        }

        if table in q3_cols:
            print(f"    Q3-relevant columns ({len(q3_cols[table])} of {meta.num_columns}):")
            schema = pf.schema_arrow
            col_names = [schema.field(i).name for i in range(len(schema))]

            q3_compressed = 0
            q3_uncompressed = 0
            total_compressed = 0
            total_uncompressed = 0

            for rg_idx in range(meta.num_row_groups):
                rg = meta.row_group(rg_idx)
                for col_idx in range(rg.num_columns):
                    col = rg.column(col_idx)
                    total_compressed += col.total_compressed_size
                    total_uncompressed += col.total_uncompressed_size
                    if col.path_in_schema in q3_cols[table]:
                        q3_compressed += col.total_compressed_size
                        q3_uncompressed += col.total_uncompressed_size

            q3_pct = (q3_compressed / total_compressed * 100) if total_compressed > 0 else 0
            ratio = total_uncompressed / total_compressed if total_compressed > 0 else 1
            print(f"    Compression ratio:     {ratio:.2f}x")
            print(f"    Q3 columns size:       {q3_compressed / 1024 / 1024:.2f} MB "
                  f"({q3_pct:.0f}% of file)")
            print(f"    Non-Q3 columns saved:  {(total_compressed - q3_compressed) / 1024 / 1024:.2f} MB "
                  f"(skipped by projection)")

    print(f"\n  TOTAL file size (3 Q3 tables): {total_size:.2f} MB")


def analyze_column_cardinality(data_dir: str) -> None:
    """
    Analyze distinct value counts for join keys and filter columns.
    
    This tells you:
    - Hash table sizes needed for joins
    - Whether keys are unique (1:1) or have duplicates (1:N)
    - Filter selectivity potential
    """
    print("\n" + "=" * 80)
    print("2. COLUMN CARDINALITY (distinct value counts)")
    print("=" * 80)

    # Customer
    cust = pq.read_table(os.path.join(data_dir, "customer.parquet"),
                         columns=["c_custkey", "c_mktsegment"])
    c_custkey = cust.column("c_custkey").to_numpy()
    c_mktsegment = cust.column("c_mktsegment").to_numpy()

    print(f"\n  customer ({len(c_custkey):,} rows):")
    print(f"    c_custkey:     {len(np.unique(c_custkey)):>10,} distinct  "
          f"(range: {c_custkey.min()} – {c_custkey.max()})")
    segments, seg_counts = np.unique(c_mktsegment, return_counts=True)
    print(f"    c_mktsegment:  {len(segments):>10} distinct")
    for seg, cnt in sorted(zip(segments, seg_counts), key=lambda x: -x[1]):
        pct = cnt / len(c_custkey) * 100
        marker = " ← Q3 filter" if seg == "BUILDING" else ""
        print(f"      {seg:15s}: {cnt:>8,} ({pct:5.1f}%){marker}")

    # Orders
    orders = pq.read_table(os.path.join(data_dir, "orders.parquet"),
                           columns=["o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"])
    o_orderkey = orders.column("o_orderkey").to_numpy()
    o_custkey = orders.column("o_custkey").to_numpy()
    o_orderdate = orders.column("o_orderdate").to_numpy()
    o_shippriority = orders.column("o_shippriority").to_numpy()

    print(f"\n  orders ({len(o_orderkey):,} rows):")
    print(f"    o_orderkey:      {len(np.unique(o_orderkey)):>10,} distinct  "
          f"(range: {o_orderkey.min()} – {o_orderkey.max()}, "
          f"density: {len(o_orderkey) / (o_orderkey.max() - o_orderkey.min() + 1):.2f})")
    print(f"    o_custkey:       {len(np.unique(o_custkey)):>10,} distinct  "
          f"(range: {o_custkey.min()} – {o_custkey.max()})")
    print(f"    o_orderdate:     {len(np.unique(o_orderdate)):>10,} distinct  "
          f"(range: {o_orderdate.min()} – {o_orderdate.max()})")
    print(f"    o_shippriority:  {len(np.unique(o_shippriority)):>10,} distinct  "
          f"(values: {np.unique(o_shippriority).tolist()})")

    # Lineitem
    li = pq.read_table(os.path.join(data_dir, "lineitem.parquet"),
                       columns=["l_orderkey", "l_extendedprice", "l_discount", "l_shipdate"])
    l_orderkey = li.column("l_orderkey").to_numpy()
    l_extendedprice = li.column("l_extendedprice").cast(pa.float64()).to_numpy()
    l_discount = li.column("l_discount").cast(pa.float64()).to_numpy()
    l_shipdate = li.column("l_shipdate").to_numpy()

    print(f"\n  lineitem ({len(l_orderkey):,} rows):")
    print(f"    l_orderkey:       {len(np.unique(l_orderkey)):>10,} distinct  "
          f"(range: {l_orderkey.min()} – {l_orderkey.max()})")
    print(f"    l_extendedprice:  {len(np.unique(l_extendedprice)):>10,} distinct  "
          f"(range: {l_extendedprice.min():.2f} – {l_extendedprice.max():.2f})")
    print(f"    l_discount:       {len(np.unique(l_discount)):>10,} distinct  "
          f"(values: {sorted(np.unique(l_discount).tolist())})")
    print(f"    l_shipdate:       {len(np.unique(l_shipdate)):>10,} distinct  "
          f"(range: {l_shipdate.min()} – {l_shipdate.max()})")


def analyze_filter_selectivity(data_dir: str) -> None:
    """
    Analyze the selectivity of each Q3 filter predicate.
    
    This tells you:
    - How much each filter eliminates
    - Combined selectivity after all filters
    - Whether filters are independent or correlated
    """
    print("\n" + "=" * 80)
    print("3. FILTER SELECTIVITY ANALYSIS")
    print("=" * 80)

    cutoff_date = date(1995, 3, 15)

    # Customer
    cust = pq.read_table(os.path.join(data_dir, "customer.parquet"),
                         columns=["c_custkey", "c_mktsegment"])
    c_total = len(cust)
    c_building = cust.filter(
        pa.compute.equal(cust.column("c_mktsegment"), "BUILDING")
    )
    c_pass = len(c_building)
    print(f"\n  c_mktsegment = 'BUILDING':")
    print(f"    {c_total:>10,} → {c_pass:>10,}  "
          f"(selectivity: {c_pass/c_total:.1%}, eliminates {c_total-c_pass:,})")

    # Orders — date filter only
    orders = pq.read_table(os.path.join(data_dir, "orders.parquet"),
                           columns=["o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"])
    o_total = len(orders)
    o_orderdate = orders.column("o_orderdate").to_numpy()
    date_mask = o_orderdate < np.datetime64(cutoff_date)
    o_date_pass = int(np.sum(date_mask))
    print(f"\n  o_orderdate < '1995-03-15':")
    print(f"    {o_total:>10,} → {o_date_pass:>10,}  "
          f"(selectivity: {o_date_pass/o_total:.1%}, eliminates {o_total-o_date_pass:,})")

    # Orders — date filter + semi-join
    building_keys = set(c_building.column("c_custkey").to_numpy().tolist())
    o_custkey = orders.column("o_custkey").to_numpy()
    semi_mask = np.isin(o_custkey, np.array(list(building_keys)))
    combined_mask = date_mask & semi_mask
    o_combined_pass = int(np.sum(combined_mask))
    print(f"\n  o_orderdate < '1995-03-15' AND c_mktsegment = 'BUILDING' (combined):")
    print(f"    {o_total:>10,} → {o_combined_pass:>10,}  "
          f"(selectivity: {o_combined_pass/o_total:.1%})")
    print(f"    Semi-join additional reduction: {o_date_pass:,} → {o_combined_pass:,} "
          f"({1 - o_combined_pass/o_date_pass:.1%} eliminated)")

    # Lineitem — shipdate filter only
    li = pq.read_table(os.path.join(data_dir, "lineitem.parquet"),
                       columns=["l_orderkey", "l_shipdate"])
    l_total = len(li)
    l_shipdate = li.column("l_shipdate").to_numpy()
    ship_mask = l_shipdate > np.datetime64(cutoff_date)
    l_ship_pass = int(np.sum(ship_mask))
    print(f"\n  l_shipdate > '1995-03-15':")
    print(f"    {l_total:>10,} → {l_ship_pass:>10,}  "
          f"(selectivity: {l_ship_pass/l_total:.1%}, eliminates {l_total-l_ship_pass:,})")

    # Lineitem — shipdate + join to qualifying orders
    qualifying_orderkeys = set(orders.column("o_orderkey").to_numpy()[combined_mask].tolist())
    l_orderkey = li.column("l_orderkey").to_numpy()
    join_mask = np.isin(l_orderkey[ship_mask], np.array(list(qualifying_orderkeys)))
    l_final = int(np.sum(join_mask))
    print(f"\n  l_shipdate > '1995-03-15' AND orderkey in qualifying orders (final):")
    print(f"    {l_total:>10,} → {l_final:>10,}  "
          f"(selectivity: {l_final/l_total:.1%})")
    print(f"    Join additional reduction: {l_ship_pass:,} → {l_final:,} "
          f"({1 - l_final/l_ship_pass:.1%} eliminated)")

    print(f"\n  Overall pipeline selectivity: {l_total:,} lineitem rows → "
          f"{l_final:,} joined rows ({l_final/l_total:.2%})")


def analyze_join_fanout(data_dir: str) -> None:
    """
    Analyze join fan-out: how many orders per customer, lineitems per order.
    
    This tells you:
    - Expected hash table probe hit rate
    - Memory amplification during joins
    - Whether pre-aggregation can help
    """
    print("\n" + "=" * 80)
    print("4. JOIN FAN-OUT ANALYSIS")
    print("=" * 80)

    # Orders per customer
    orders = pq.read_table(os.path.join(data_dir, "orders.parquet"),
                           columns=["o_orderkey", "o_custkey"])
    o_custkey = orders.column("o_custkey").to_numpy()
    _, cust_counts = np.unique(o_custkey, return_counts=True)

    print(f"\n  Orders per customer (all orders):")
    print(f"    min:    {cust_counts.min():>5}")
    print(f"    max:    {cust_counts.max():>5}")
    print(f"    mean:   {cust_counts.mean():>8.1f}")
    print(f"    median: {np.median(cust_counts):>8.1f}")
    print(f"    stddev: {cust_counts.std():>8.1f}")

    # Distribution histogram
    hist_edges = [0, 5, 10, 15, 20, 25, 30, 50, 100]
    hist, _ = np.histogram(cust_counts, bins=hist_edges)
    print(f"    Distribution:")
    for i in range(len(hist)):
        lo, hi = hist_edges[i], hist_edges[i + 1]
        pct = hist[i] / len(cust_counts) * 100
        bar = "#" * int(pct)
        print(f"      {lo:>3d}–{hi:>3d} orders: {hist[i]:>6,} customers ({pct:5.1f}%) {bar}")

    # Lineitems per order
    li = pq.read_table(os.path.join(data_dir, "lineitem.parquet"),
                       columns=["l_orderkey"])
    l_orderkey = li.column("l_orderkey").to_numpy()
    _, order_counts = np.unique(l_orderkey, return_counts=True)

    print(f"\n  Lineitems per order (all orders):")
    print(f"    min:    {order_counts.min():>5}")
    print(f"    max:    {order_counts.max():>5}")
    print(f"    mean:   {order_counts.mean():>8.1f}")
    print(f"    median: {np.median(order_counts):>8.1f}")

    hist_edges = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    hist, _ = np.histogram(order_counts, bins=hist_edges)
    print(f"    Distribution:")
    for i in range(len(hist)):
        lo, hi = hist_edges[i], hist_edges[i + 1]
        pct = hist[i] / len(order_counts) * 100
        bar = "#" * int(pct / 2)
        print(f"      {lo}–{hi} items: {hist[i]:>8,} orders ({pct:5.1f}%) {bar}")


def analyze_sort_order(data_dir: str) -> None:
    """
    Analyze whether columns are sorted/clustered in the Parquet files.
    
    If data is sorted by a filter column, row-group statistics can skip
    entire row groups. If NOT sorted (random order), every row group
    contains the full value range and NO groups can be skipped.
    
    This is crucial for understanding predicate pushdown effectiveness.
    """
    print("\n" + "=" * 80)
    print("5. SORT ORDER & CLUSTERING ANALYSIS")
    print("=" * 80)
    print("  (Determines whether Parquet row-group skipping can help)")

    for table, cols in [
        ("customer", ["c_custkey", "c_mktsegment"]),
        ("orders", ["o_orderkey", "o_custkey", "o_orderdate"]),
        ("lineitem", ["l_orderkey", "l_shipdate"]),
    ]:
        path = os.path.join(data_dir, f"{table}.parquet")
        pf = pq.ParquetFile(path)
        meta = pf.metadata
        n_rg = meta.num_row_groups

        print(f"\n  {table}.parquet ({n_rg} row groups):")

        # Get schema to find column indices
        schema = pf.schema_arrow
        col_names = [schema.field(i).name for i in range(len(schema))]

        for col in cols:
            if col not in col_names:
                continue
            col_idx = col_names.index(col)

            # Collect min/max per row group
            mins, maxs = [], []
            for rg_idx in range(n_rg):
                rg = meta.row_group(rg_idx)
                stats = rg.column(col_idx).statistics
                if stats and stats.has_min_max:
                    mins.append(stats.min)
                    maxs.append(stats.max)

            if not mins:
                print(f"    {col}: no statistics available")
                continue

            # Check if row groups are sorted (non-overlapping min/max ranges)
            sorted_flag = True
            overlap_count = 0
            for i in range(1, len(mins)):
                if mins[i] <= maxs[i - 1]:
                    sorted_flag = False
                    overlap_count += 1

            if sorted_flag:
                status = "SORTED ✓ (row groups have non-overlapping ranges → skipping possible)"
            elif overlap_count == n_rg - 1:
                status = "UNSORTED ✗ (ALL row groups overlap → NO skipping possible)"
            else:
                status = f"PARTIAL ({overlap_count}/{n_rg-1} overlap → limited skipping)"

            print(f"    {col:20s}: {status}")
            print(f"      RG ranges: [{mins[0]}..{maxs[0]}]", end="")
            if n_rg > 1:
                print(f", [{mins[1]}..{maxs[1]}]", end="")
            if n_rg > 2:
                print(f", ... [{mins[-1]}..{maxs[-1]}]", end="")
            print()

            # For the filter columns, estimate how many RGs could be skipped
            if col == "o_orderdate":
                cutoff = date(1995, 3, 15)
                skippable = sum(1 for m in mins if m >= cutoff)
                print(f"      Filter o_orderdate < {cutoff}: "
                      f"{skippable}/{n_rg} row groups skippable")
            elif col == "l_shipdate":
                cutoff = date(1995, 3, 15)
                skippable = sum(1 for m in maxs if m <= cutoff)
                print(f"      Filter l_shipdate > {cutoff}: "
                      f"{skippable}/{n_rg} row groups skippable")


def analyze_key_distribution(data_dir: str) -> None:
    """
    Analyze key density and distribution for hash table design.
    
    Tells you:
    - Are orderkeys dense (good for array-based lookup) or sparse (need hashing)?
    - What's the key range vs count ratio?
    - Memory implications for different hash table strategies.
    """
    print("\n" + "=" * 80)
    print("6. KEY DISTRIBUTION (hash table design implications)")
    print("=" * 80)

    orders = pq.read_table(os.path.join(data_dir, "orders.parquet"),
                           columns=["o_orderkey"])
    o_orderkey = orders.column("o_orderkey").to_numpy()

    n_keys = len(o_orderkey)
    n_unique = len(np.unique(o_orderkey))
    key_min = o_orderkey.min()
    key_max = o_orderkey.max()
    key_range = key_max - key_min + 1
    density = n_unique / key_range

    print(f"\n  o_orderkey:")
    print(f"    Count:      {n_keys:>12,}")
    print(f"    Unique:     {n_unique:>12,}")
    print(f"    Min:        {key_min:>12,}")
    print(f"    Max:        {key_max:>12,}")
    print(f"    Range:      {key_range:>12,}")
    print(f"    Density:    {density:>12.4f}  ({density:.1%} of range occupied)")

    # Check if keys follow a pattern (TPC-H typically uses multiples)
    sample = np.sort(o_orderkey[:100])
    diffs = np.diff(sample)
    unique_diffs, diff_counts = np.unique(diffs, return_counts=True)
    print(f"    Key spacing (first 100 sorted): "
          f"most common gaps = {unique_diffs[np.argsort(-diff_counts)][:5].tolist()}")

    # Memory analysis for different hash table strategies
    print(f"\n  Hash table strategy comparison (for ~{n_unique // 10:,} qualifying orders):")
    est_keys = n_unique // 10  # ~10% after filters
    print(f"    Python dict:     ~{est_keys * 100 // 1024 // 1024 + 1:>5} MB  "
          f"(~100 bytes/entry overhead)")
    print(f"    Direct array:    ~{key_range * 16 // 1024 // 1024:>5} MB  "
          f"(16 bytes × key_range, {density:.0%} utilization)")
    print(f"    Sorted array:    ~{est_keys * 8 // 1024 // 1024 + 1:>5} MB  "
          f"(8 bytes/key, binary search probe)")

    # c_custkey analysis
    cust = pq.read_table(os.path.join(data_dir, "customer.parquet"),
                         columns=["c_custkey"])
    c_custkey = cust.column("c_custkey").to_numpy()
    c_range = c_custkey.max() - c_custkey.min() + 1
    c_density = len(c_custkey) / c_range

    print(f"\n  c_custkey:")
    print(f"    Count:      {len(c_custkey):>12,}")
    print(f"    Range:      {c_custkey.min()} – {c_custkey.max()} ({c_range:,})")
    print(f"    Density:    {c_density:.4f}  ({c_density:.1%} of range occupied)")


def analyze_value_distributions(data_dir: str) -> None:
    """
    Analyze value distributions for Q3's arithmetic columns.
    
    This matters for:
    - Numeric precision: are there values that could cause float64 issues?
    - Revenue range: does top-K early termination help?
    """
    print("\n" + "=" * 80)
    print("7. VALUE DISTRIBUTIONS (arithmetic & aggregation)")
    print("=" * 80)

    li = pq.read_table(os.path.join(data_dir, "lineitem.parquet"),
                       columns=["l_extendedprice", "l_discount"])
    prices = li.column("l_extendedprice").cast(pa.float64()).to_numpy()
    discounts = li.column("l_discount").cast(pa.float64()).to_numpy()

    revenue_per_item = prices * (1.0 - discounts)

    print(f"\n  l_extendedprice:")
    print(f"    min:    {prices.min():>12.2f}")
    print(f"    max:    {prices.max():>12.2f}")
    print(f"    mean:   {prices.mean():>12.2f}")
    print(f"    stddev: {prices.std():>12.2f}")

    print(f"\n  l_discount:")
    print(f"    min:    {discounts.min():>12.2f}")
    print(f"    max:    {discounts.max():>12.2f}")
    print(f"    values: {sorted(np.unique(discounts).tolist())}")

    print(f"\n  revenue per item (extprice * (1 - discount)):")
    print(f"    min:    {revenue_per_item.min():>12.2f}")
    print(f"    max:    {revenue_per_item.max():>12.2f}")
    print(f"    mean:   {revenue_per_item.mean():>12.2f}")

    # Estimate per-group revenue (grouped by orderkey)
    # This tells us the range of possible top-K values
    l_orderkey = pq.read_table(os.path.join(data_dir, "lineitem.parquet"),
                               columns=["l_orderkey"]).column("l_orderkey").to_numpy()
    unique_keys, inverse = np.unique(l_orderkey, return_inverse=True)
    group_revenue = np.zeros(len(unique_keys))
    np.add.at(group_revenue, inverse, revenue_per_item)

    print(f"\n  Aggregate revenue per order (all orders, before any filter):")
    print(f"    min:    {group_revenue.min():>14.2f}")
    print(f"    max:    {group_revenue.max():>14.2f}")
    print(f"    mean:   {group_revenue.mean():>14.2f}")
    print(f"    median: {np.median(group_revenue):>14.2f}")

    # Top-K insight
    sorted_rev = np.sort(group_revenue)[::-1]
    print(f"    10th highest: {sorted_rev[9]:>14.2f}")
    print(f"    100th highest: {sorted_rev[99]:>14.2f}")
    print(f"    → Orders with revenue < {sorted_rev[9]:.0f} can never make top-10")


def analyze_orderdate_distribution(data_dir: str) -> None:
    """
    Analyze the order date distribution around the Q3 cutoff.
    
    The cutoff (1995-03-15) is near the midpoint of the date range.
    Understanding the distribution tells us if the filter is balanced.
    """
    print("\n" + "=" * 80)
    print("8. ORDER DATE DISTRIBUTION AROUND CUTOFF")
    print("=" * 80)

    orders = pq.read_table(os.path.join(data_dir, "orders.parquet"),
                           columns=["o_orderdate"])
    dates = orders.column("o_orderdate").to_numpy()

    cutoff = np.datetime64("1995-03-15")
    before = np.sum(dates < cutoff)
    after = np.sum(dates >= cutoff)
    total = len(dates)

    print(f"\n  Total orders: {total:,}")
    print(f"  Before 1995-03-15: {before:>10,} ({before/total:.1%})")
    print(f"  On/After 1995-03-15: {after:>10,} ({after/total:.1%})")

    # Monthly distribution around cutoff
    print(f"\n  Monthly distribution (1994-10 to 1995-08):")
    for year in [1994, 1995]:
        for month in range(1, 13):
            start = np.datetime64(f"{year}-{month:02d}-01")
            if month < 12:
                end = np.datetime64(f"{year}-{month+1:02d}-01")
            else:
                end = np.datetime64(f"{year+1}-01-01")
            count = np.sum((dates >= start) & (dates < end))
            if year == 1994 and month < 10:
                continue
            if year == 1995 and month > 8:
                continue
            pct = count / total * 100
            bar = "#" * int(pct * 2)
            marker = " ← cutoff" if year == 1995 and month == 3 else ""
            print(f"    {year}-{month:02d}: {count:>7,} ({pct:4.1f}%) {bar}{marker}")


def analyze_scaling(all_data: bool = False) -> None:
    """
    Compare characteristics across scale factors.
    Shows how table sizes, selectivities, and group counts scale.
    """
    if not all_data:
        return

    print("\n" + "=" * 80)
    print("9. SCALING ANALYSIS (across scale factors)")
    print("=" * 80)

    cutoff = date(1995, 3, 15)

    print(f"\n  {'SF':>6s}  {'Customer':>10s}  {'Orders':>12s}  {'Lineitem':>12s}  "
          f"{'After Filters':>15s}  {'Groups':>10s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*15}  {'-'*10}")

    for sf in [0.5, 1, 2, 5]:
        sf_label = f"sf{sf:g}"
        data_dir = os.path.join("data", sf_label)
        if not os.path.exists(data_dir):
            continue

        # Quick row counts
        c_rows = pq.ParquetFile(os.path.join(data_dir, "customer.parquet")).metadata.num_rows
        o_rows = pq.ParquetFile(os.path.join(data_dir, "orders.parquet")).metadata.num_rows
        l_rows = pq.ParquetFile(os.path.join(data_dir, "lineitem.parquet")).metadata.num_rows

        # Count qualifying rows (quick scan with filters)
        from scanner import scan_parquet
        li = scan_parquet(
            os.path.join(data_dir, "lineitem.parquet"),
            columns=["l_orderkey"],
            filters=[("l_shipdate", ">", cutoff)],
        )
        l_filtered = len(li["l_orderkey"])

        print(f"  {sf:>6g}  {c_rows:>10,}  {o_rows:>12,}  {l_rows:>12,}  "
              f"  {l_filtered:>13,}  {'—':>10s}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze TPC-H data for Q3 optimization")
    parser.add_argument("--sf", type=str, default="data/sf1",
                        help="Data directory (default: data/sf1)")
    parser.add_argument("--all", action="store_true",
                        help="Also show scaling analysis across all SFs")
    args = parser.parse_args()

    print(f"TPC-H Q3 Data Analysis — {args.sf}")
    print(f"{'='*80}\n")

    analyze_parquet_structure(args.sf)
    analyze_column_cardinality(args.sf)
    analyze_filter_selectivity(args.sf)
    analyze_join_fanout(args.sf)
    analyze_sort_order(args.sf)
    analyze_key_distribution(args.sf)
    analyze_value_distributions(args.sf)
    analyze_orderdate_distribution(args.sf)
    analyze_scaling(args.all)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
