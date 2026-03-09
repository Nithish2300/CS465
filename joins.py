"""
joins.py — Hash-based join operators for the Q3 pipeline.

This module implements two join strategies used in Q3:

1. SEMI-JOIN (customer → orders):
   Since no customer columns appear in Q3's output, we don't need a full join.
   We only need to know "is this order's customer in the BUILDING segment?"
   A semi-join answers this with a simple set membership test.
   
   Implementation: build a Python set of qualifying c_custkey values, then
   use NumPy to filter the orders arrays by checking o_custkey ∈ custkey_set.

2. HASH JOIN (orders → lineitem):
   We need to match lineitem rows to their parent orders and carry forward
   o_orderdate and o_shippriority. This is a classic hash join:
   - BUILD phase: insert qualifying orders into a hash map (o_orderkey → payload)
   - PROBE phase: for each lineitem row, look up l_orderkey in the hash map
   
   Implementation: Python dict with int keys (C-implemented, fast for int keys).

Why these data structures?
- Python's set/dict are implemented in C as hash tables with open addressing.
  For integer keys, the hash function is essentially the identity (Python ints
  hash to themselves), making lookups very fast.
- At SF-1: customer set has ~30K entries (< 1 MB), orders dict has ~150K entries
  (~10 MB). Both fit comfortably in CPU L3 cache.
- We avoid pure-Python loops over individual rows where possible by using NumPy
  vectorized operations (np.isin, boolean indexing).

Alternative approaches you could experiment with:
- np.searchsorted: Sort the key array and use binary search. O(n log k) per probe.
- Direct-address array: If keys are dense integers, use a flat array indexed by key.
  TPC-H orderkeys are sparse (multiples of some base), so this wastes memory.
- Bloom filter: Probabilistic membership test as a pre-filter before dict lookup.
  Adds complexity; useful if dict probe is the bottleneck.
"""

import numpy as np


def build_semi_join_set(keys: np.ndarray) -> set:
    """
    Build a hash set for semi-join filtering.
    
    This is the BUILD phase of the customer → orders semi-join.
    We create a Python set of customer keys that passed the segment filter.
    
    Args:
        keys: NumPy array of qualifying c_custkey values (e.g., BUILDING customers).
    
    Returns:
        A Python set of int keys for O(1) membership testing.
    
    Example:
        custkey_set = build_semi_join_set(customer_data["c_custkey"])
        # custkey_set = {1, 8, 11, 13, 18, ...}  (~30K entries at SF-1)
    """
    # .tolist() converts NumPy ints to Python ints, which is faster for set
    # construction than iterating over numpy scalar objects.
    return set(keys.tolist())


def probe_semi_join(
    custkey_set: set,
    orders_data: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Filter orders by customer key membership (PROBE phase of semi-join).
    
    For each order, we check: is o_custkey in custkey_set?
    Only orders belonging to qualifying customers are kept.
    
    This replaces a full inner join — since we don't need any customer columns
    in the output, we just need to know "does this order match a BUILDING customer?"
    
    Args:
        custkey_set:  Set of qualifying c_custkey values (from build_semi_join_set).
        orders_data:  Dict of orders arrays from the scanner, must include "o_custkey".
    
    Returns:
        Filtered orders_data dict with only rows where o_custkey ∈ custkey_set.
    
    Implementation note:
        We use np.isin() which vectorizes the membership test in C.
        For ~730K orders probing against ~30K customer keys, this is efficient.
        
        Alternative: convert set to sorted array and use np.searchsorted():
            sorted_keys = np.sort(np.array(list(custkey_set)))
            idx = np.searchsorted(sorted_keys, o_custkey)
            mask = (idx < len(sorted_keys)) & (sorted_keys[idx] == o_custkey)
        This is O(n log k) vs np.isin's O(n * k) worst case, but np.isin uses
        a hash set internally for large arrays, so it's typically fast enough.
    """
    o_custkey = orders_data["o_custkey"]

    # Convert set to array for np.isin (it handles set→array internally,
    # but being explicit makes the code clearer)
    custkey_array = np.array(list(custkey_set))
    
    # Vectorized membership test: returns boolean mask
    mask = np.isin(o_custkey, custkey_array)

    # Apply mask to ALL order columns
    filtered = {}
    for col_name, col_array in orders_data.items():
        filtered[col_name] = col_array[mask]

    return filtered


def build_hash_join(orders_data: dict[str, np.ndarray]) -> dict:
    """
    Build a hash map for the orders → lineitem join (BUILD phase).
    
    We create a Python dict mapping:
        o_orderkey → (o_orderdate, o_shippriority)
    
    These payload values are needed in the GROUP BY and SELECT of Q3:
        GROUP BY l_orderkey, o_orderdate, o_shippriority
        SELECT   l_orderkey, revenue, o_orderdate, o_shippriority
    
    Args:
        orders_data: Filtered orders dict (after semi-join), must contain
                     o_orderkey, o_orderdate, o_shippriority columns.
    
    Returns:
        A dict: {orderkey_int: (orderdate, shippriority), ...}
    
    Performance notes:
        - At SF-1 after both filters + semi-join: ~150K entries.
        - Memory: ~150K * ~40 bytes (key + value + dict overhead) ≈ 6 MB.
        - Python dicts with int keys use the int itself as hash → very fast.
        - Building the dict requires a Python loop (can't fully vectorize dict
          construction), but 150K iterations is fast (~10ms).
    """
    orderkeys = orders_data["o_orderkey"]
    orderdates = orders_data["o_orderdate"]
    shippriorities = orders_data["o_shippriority"]

    # Build the hash map. We use .item() to convert numpy scalars to Python
    # native types for faster dict operations.
    hash_map = {}
    for i in range(len(orderkeys)):
        hash_map[orderkeys[i].item()] = (orderdates[i], shippriorities[i].item())

    return hash_map


# ---------------------------------------------------------------------------
# Self-test: run this file directly to see the joins in action on SF=1
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    from datetime import date
    from scanner import scan_parquet

    data_dir = "data/sf1"

    # --- Phase 1: Scan ---
    print("=" * 70)
    print("PHASE 1: Scanning tables with pushdown filters")
    print("=" * 70)

    customer = scan_parquet(
        os.path.join(data_dir, "customer.parquet"),
        columns=["c_custkey", "c_mktsegment"],
        filters=[("c_mktsegment", "=", "BUILDING")],
    )
    print(f"  customer (BUILDING): {len(customer['c_custkey']):,} rows")

    orders = scan_parquet(
        os.path.join(data_dir, "orders.parquet"),
        columns=["o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"],
        filters=[("o_orderdate", "<", date(1995, 3, 15))],
    )
    print(f"  orders (before 1995-03-15): {len(orders['o_orderkey']):,} rows")

    # --- Phase 2: Semi-join (customer → orders) ---
    print()
    print("=" * 70)
    print("PHASE 2: Semi-join — filter orders by BUILDING customer keys")
    print("=" * 70)

    custkey_set = build_semi_join_set(customer["c_custkey"])
    print(f"  Customer key set size: {len(custkey_set):,}")

    orders_filtered = probe_semi_join(custkey_set, orders)
    print(f"  Orders before semi-join: {len(orders['o_orderkey']):,}")
    print(f"  Orders after semi-join:  {len(orders_filtered['o_orderkey']):,}")
    print(f"  Rows eliminated: {len(orders['o_orderkey']) - len(orders_filtered['o_orderkey']):,}")

    # --- Phase 3: Build hash join for orders → lineitem ---
    print()
    print("=" * 70)
    print("PHASE 3: Build hash map for orders → lineitem join")
    print("=" * 70)

    # Drop o_custkey — we don't need it anymore after the semi-join
    del orders_filtered["o_custkey"]

    order_map = build_hash_join(orders_filtered)
    print(f"  Hash map entries: {len(order_map):,}")
    
    # Show a sample entry
    sample_key = next(iter(order_map))
    sample_val = order_map[sample_key]
    print(f"  Sample: orderkey={sample_key} → (date={sample_val[0]}, priority={sample_val[1]})")

    # --- Summary ---
    print()
    print("=" * 70)
    print("DATA FLOW SUMMARY (SF=1)")
    print("=" * 70)
    print(f"  customer scan:     150,000 → {len(customer['c_custkey']):>10,}  (BUILDING filter)")
    print(f"  orders scan:     1,500,000 → {len(orders['o_orderkey']):>10,}  (date filter)")
    print(f"  orders semi-join:  {len(orders['o_orderkey']):>9,} → {len(orders_filtered['o_orderkey']):>10,}  (customer membership)")
    print(f"  order hash map:              {len(order_map):>10,} entries  (ready for lineitem probe)")
    print()
    print("  Next step: scan lineitem and probe into this hash map (aggregator.py)")
