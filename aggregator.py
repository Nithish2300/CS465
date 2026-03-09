"""
aggregator.py — Fused hash-join probe + group-by aggregation for Q3.

This is the most performance-critical module in our pipeline. It combines
two relational operators into a single pass over the lineitem data:

1. HASH JOIN PROBE: For each lineitem row, look up l_orderkey in the orders
   hash map to find the matching order's (o_orderdate, o_shippriority).
   
2. GROUP-BY AGGREGATION: Accumulate SUM(l_extendedprice * (1 - l_discount))
   grouped by (l_orderkey, o_orderdate, o_shippriority).

WHY FUSE THESE?
If we did them separately:
  Step A: Join lineitem (3.2M rows) × orders (147K) → ~1.5M matched rows
  Step B: Group the 1.5M joined rows → ~147K groups

The intermediate result (1.5M rows with all columns) wastes memory and
cache bandwidth. By fusing, we never materialize it — we go directly from
lineitem scan → aggregated groups.

The aggregation uses Q3's expression:
    revenue = SUM(l_extendedprice * (1 - l_discount))

This is computed per group, where each group is identified by:
    (l_orderkey, o_orderdate, o_shippriority)

IMPLEMENTATION STRATEGY:
We provide two implementations you can compare:

A) DICT-BASED (default): Use a Python dict as the aggregation hash map.
   - Key: l_orderkey (int) — since o_orderdate and o_shippriority are 
     functionally determined by l_orderkey (one order → one date/priority),
     we can use orderkey alone as the group key.
   - Value: running sum of revenue contributions.
   - Pro: Simple, correct, reasonable performance.
   - Con: Python-level loop over each lineitem row.

B) NUMPY VECTORIZED: Use NumPy's vectorized operations to avoid Python loops.
   - Compute revenue array = extendedprice * (1 - discount) in one vector op.
   - Use np.unique + np.add.at for group-by sum.
   - Pro: Much faster — stays in C/NumPy for all computation.
   - Con: More complex code, requires understanding NumPy advanced indexing.
"""

import numpy as np


def fused_join_aggregate_dict(
    order_map: dict,
    lineitem_data: dict[str, np.ndarray],
) -> dict:
    """
    Fused hash-join probe + group-by aggregation using a Python dict.
    
    This is the straightforward implementation. For each lineitem row:
    1. Check if l_orderkey exists in order_map (join probe)
    2. If yes, compute revenue contribution and add to running sum
    
    Args:
        order_map:     Dict from build_hash_join(): {orderkey → (date, priority)}
        lineitem_data: Dict of lineitem arrays from scanner, must contain:
                       l_orderkey, l_extendedprice, l_discount
    
    Returns:
        Dict: {orderkey → (revenue_sum, o_orderdate, o_shippriority)}
        Only contains groups with at least one matching lineitem.
    """
    orderkeys = lineitem_data["l_orderkey"]
    extprices = lineitem_data["l_extendedprice"]
    discounts = lineitem_data["l_discount"]

    # Aggregation map: orderkey → [revenue_sum, o_orderdate, o_shippriority]
    # We use a list as the value so we can mutate revenue_sum in-place.
    agg = {}

    n_rows = len(orderkeys)
    n_matched = 0

    for i in range(n_rows):
        okey = orderkeys[i].item()  # Convert numpy int to Python int

        # JOIN PROBE: look up in the orders hash map
        order_info = order_map.get(okey)
        if order_info is None:
            continue  # This lineitem's order didn't pass the filters → skip

        n_matched += 1

        # AGGREGATE: compute revenue contribution and accumulate
        revenue_contribution = extprices[i].item() * (1.0 - discounts[i].item())

        existing = agg.get(okey)
        if existing is not None:
            # Group already exists — add to running sum
            existing[0] += revenue_contribution
        else:
            # New group — initialize with order metadata from hash map
            agg[okey] = [revenue_contribution, order_info[0], order_info[1]]

    return agg


def fused_join_aggregate_numpy(
    order_map: dict,
    lineitem_data: dict[str, np.ndarray],
) -> dict:
    """
    Fused hash-join probe + group-by aggregation using NumPy vectorized ops.
    
    This is the optimized implementation. The key insight: we can vectorize
    BOTH the join probe and the aggregation by:
    1. Pre-filtering lineitem to only rows with matching orderkeys (vectorized)
    2. Computing revenue for all matching rows in one vector operation
    3. Using np.add.at for scatter-add (group-by sum without Python loops)
    
    Args:
        order_map:     Dict from build_hash_join(): {orderkey → (date, priority)}
        lineitem_data: Dict of lineitem arrays from scanner.
    
    Returns:
        Dict: {orderkey → (revenue_sum, o_orderdate, o_shippriority)}
    """
    orderkeys = lineitem_data["l_orderkey"]
    extprices = lineitem_data["l_extendedprice"]
    discounts = lineitem_data["l_discount"]

    # --- Step 1: Vectorized join probe ---
    # Create a boolean mask: True if the lineitem's orderkey is in order_map
    order_key_set = np.array(list(order_map.keys()))
    mask = np.isin(orderkeys, order_key_set)

    # Filter to matching rows only
    matched_orderkeys = orderkeys[mask]
    matched_extprices = extprices[mask]
    matched_discounts = discounts[mask]

    # --- Step 2: Vectorized revenue computation ---
    # revenue = extendedprice * (1 - discount)  for all matched rows at once
    revenue = matched_extprices * (1.0 - matched_discounts)

    # --- Step 3: Group-by SUM using NumPy ---
    # Get unique orderkeys and map each row to its group index
    unique_keys, group_indices = np.unique(matched_orderkeys, return_inverse=True)

    # Scatter-add: sum revenue into each group
    # np.add.at is an unbuffered operation that handles duplicate indices correctly
    revenue_sums = np.zeros(len(unique_keys), dtype=np.float64)
    np.add.at(revenue_sums, group_indices, revenue)

    # --- Step 4: Build result dict with order metadata ---
    agg = {}
    for i, okey in enumerate(unique_keys):
        okey_int = okey.item()
        order_info = order_map[okey_int]
        agg[okey_int] = [revenue_sums[i], order_info[0], order_info[1]]

    return agg


def top_k(agg: dict, k: int = 10) -> dict[str, np.ndarray]:
    """
    Extract the top-K groups by revenue, with tiebreaking by o_orderdate ASC.
    
    Q3's ORDER BY: revenue DESC, o_orderdate ASC, LIMIT 10.
    
    We use np.argpartition for O(n) partial sort to find the top-K candidates,
    then do a full sort on just those K elements for the final ordering.
    
    Args:
        agg: Aggregation result dict: {orderkey → [revenue, date, priority]}
        k:   Number of top results to return (default: 10).
    
    Returns:
        Dict with arrays: {
            "l_orderkey": np.array,
            "revenue": np.array,
            "o_orderdate": np.array,
            "o_shippriority": np.array,
        }
        Sorted by revenue DESC, o_orderdate ASC. Length = min(k, num_groups).
    """
    if not agg:
        return {
            "l_orderkey": np.array([], dtype=np.int64),
            "revenue": np.array([], dtype=np.float64),
            "o_orderdate": np.array([]),
            "o_shippriority": np.array([], dtype=np.int32),
        }

    # Unpack the aggregation dict into arrays
    orderkeys = np.array(list(agg.keys()), dtype=np.int64)
    values = list(agg.values())
    revenues = np.array([v[0] for v in values], dtype=np.float64)
    orderdates = np.array([v[1] for v in values])
    priorities = np.array([v[2] for v in values], dtype=np.int32)

    n = len(orderkeys)
    k = min(k, n)

    if n <= k:
        # Fewer groups than K — just sort all of them
        candidates = np.arange(n)
    else:
        # O(n) partial sort: find indices of top-K by revenue (highest = smallest negative)
        candidates = np.argpartition(-revenues, k)[:k]

    # Full sort of the K candidates: revenue DESC, then o_orderdate ASC
    # We sort by (-revenue, orderdate) to get the correct tiebreaking
    cand_revenues = revenues[candidates]
    cand_dates = orderdates[candidates]
    
    # Create a structured array for multi-key sort
    sort_keys = np.argsort(
        np.lexsort((cand_dates, -cand_revenues))
    )
    # Wait — np.lexsort sorts by LAST key first, then second-to-last, etc.
    # lexsort((dates, -revenues)) sorts by -revenues first (primary), dates second.
    # But we need: primary = revenue DESC, secondary = orderdate ASC.
    # lexsort keys are applied last-to-first, so:
    #   lexsort((secondary, primary)) → sorts by primary first, then secondary.
    # So lexsort((dates, -revenues)) is correct: sort by -revenues (DESC) first,
    # then by dates (ASC) for ties. ✓

    final_idx = candidates[np.lexsort((cand_dates, -cand_revenues))]

    return {
        "l_orderkey": orderkeys[final_idx],
        "revenue": revenues[final_idx],
        "o_orderdate": orderdates[final_idx],
        "o_shippriority": priorities[final_idx],
    }


# ---------------------------------------------------------------------------
# Self-test: run the full scan → join → aggregate → top-K pipeline
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import time
    from datetime import date
    from scanner import scan_parquet
    from joins import build_semi_join_set, probe_semi_join, build_hash_join

    data_dir = "data/sf1"

    # --- Scan phase ---
    print("Scanning tables...")
    t0 = time.perf_counter()

    customer = scan_parquet(
        os.path.join(data_dir, "customer.parquet"),
        columns=["c_custkey", "c_mktsegment"],
        filters=[("c_mktsegment", "=", "BUILDING")],
    )
    orders = scan_parquet(
        os.path.join(data_dir, "orders.parquet"),
        columns=["o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"],
        filters=[("o_orderdate", "<", date(1995, 3, 15))],
    )
    lineitem = scan_parquet(
        os.path.join(data_dir, "lineitem.parquet"),
        columns=["l_orderkey", "l_extendedprice", "l_discount", "l_shipdate"],
        filters=[("l_shipdate", ">", date(1995, 3, 15))],
    )
    t_scan = time.perf_counter() - t0
    print(f"  Scan time: {t_scan:.3f}s")

    # --- Join phase ---
    print("Running semi-join + hash join build...")
    t0 = time.perf_counter()

    custkey_set = build_semi_join_set(customer["c_custkey"])
    orders_filtered = probe_semi_join(custkey_set, orders)
    del orders_filtered["o_custkey"]  # No longer needed
    order_map = build_hash_join(orders_filtered)
    t_join_build = time.perf_counter() - t0
    print(f"  Join build time: {t_join_build:.3f}s")
    print(f"  Order hash map: {len(order_map):,} entries")

    # --- Aggregate phase: compare both implementations ---
    print()
    print("=" * 70)
    print("AGGREGATION: Comparing dict vs numpy implementations")
    print("=" * 70)

    # Dict-based
    print("\nMethod A: Python dict loop...")
    t0 = time.perf_counter()
    agg_dict = fused_join_aggregate_dict(order_map, lineitem)
    t_dict = time.perf_counter() - t0
    print(f"  Time: {t_dict:.3f}s")
    print(f"  Groups: {len(agg_dict):,}")

    # NumPy vectorized
    print("\nMethod B: NumPy vectorized...")
    t0 = time.perf_counter()
    agg_numpy = fused_join_aggregate_numpy(order_map, lineitem)
    t_numpy = time.perf_counter() - t0
    print(f"  Time: {t_numpy:.3f}s")
    print(f"  Groups: {len(agg_numpy):,}")

    print(f"\n  Speedup (dict / numpy): {t_dict / t_numpy:.1f}x")

    # --- Top-K ---
    print()
    print("=" * 70)
    print("TOP-10 RESULT (using numpy aggregation)")
    print("=" * 70)

    result = top_k(agg_numpy, k=10)
    print(f"\n{'l_orderkey':>12s}  {'revenue':>14s}  {'o_orderdate':>12s}  {'o_shippriority':>14s}")
    print("-" * 58)
    for i in range(len(result["l_orderkey"])):
        print(f"{result['l_orderkey'][i]:>12d}  "
              f"{result['revenue'][i]:>14.4f}  "
              f"{result['o_orderdate'][i]!s:>12s}  "
              f"{result['o_shippriority'][i]:>14d}")

    # --- Timing summary ---
    print()
    print("=" * 70)
    print("TIMING BREAKDOWN")
    print("=" * 70)
    t_total_dict = t_scan + t_join_build + t_dict
    t_total_numpy = t_scan + t_join_build + t_numpy
    print(f"  Scan:            {t_scan:.3f}s")
    print(f"  Join build:      {t_join_build:.3f}s")
    print(f"  Aggregate (dict): {t_dict:.3f}s  → total: {t_total_dict:.3f}s")
    print(f"  Aggregate (numpy):{t_numpy:.3f}s  → total: {t_total_numpy:.3f}s")
