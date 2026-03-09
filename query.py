"""
query.py — TPC-H Q3 query pipeline.

This module wires together the scanner, joins, and aggregator into a single
callable pipeline that implements the full Q3 query:

    scan_customer(filter: BUILDING)
        → build_semi_join_set(c_custkey)                    [~30K keys]
    scan_orders(filter: date < 1995-03-15)
        → probe_semi_join(custkey_set)                      [727K → 147K rows]
        → build_hash_join(o_orderkey → date, priority)      [147K entries]
    scan_lineitem(filter: shipdate > 1995-03-15)
        → fused_join_aggregate(order_map)                   [3.2M → 11.6K groups]
        → top_k(10, order by revenue DESC, date ASC)        [10 rows]

The pipeline is parameterized by:
  - data_dir: path to the scale-factor directory (e.g., "data/sf1")
  - agg_method: which aggregation implementation to use ("numpy" or "dict")

This makes it easy to experiment with different strategies while keeping
the overall pipeline structure fixed.

Usage:
    from query import run_q3
    
    result = run_q3("data/sf1")
    # result = {"l_orderkey": array, "revenue": array, "o_orderdate": array, "o_shippriority": array}
    
    result, timings = run_q3("data/sf1", return_timings=True)
    # timings = {"scan": 0.5, "join_build": 0.1, "aggregate": 0.05, "top_k": 0.001, "total": 0.65}
"""

import os
import time
import numpy as np
from datetime import date

from scanner import scan_parquet
from joins import build_semi_join_set, probe_semi_join, build_hash_join
from aggregator import fused_join_aggregate_dict, fused_join_aggregate_numpy, top_k


# Q3 constants — hardcoded for the specialized processor
# (These are the TPC-H Q3 default substitution parameters)
SEGMENT = "BUILDING"
ORDER_DATE_CUTOFF = date(1995, 3, 15)
SHIP_DATE_CUTOFF = date(1995, 3, 15)
TOP_K = 10


def run_q3(
    data_dir: str,
    agg_method: str = "numpy",
    return_timings: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, float]]:
    """
    Execute the full TPC-H Q3 pipeline on Parquet data.
    
    This is the entry point for the specialized query processor.
    
    Args:
        data_dir:       Path to scale-factor directory (e.g., "data/sf1").
        agg_method:     "numpy" (vectorized, fast) or "dict" (simple loop).
        return_timings: If True, also return a dict of per-phase timings.
    
    Returns:
        result: Dict with 4 arrays (l_orderkey, revenue, o_orderdate, o_shippriority),
                10 rows, sorted by revenue DESC, o_orderdate ASC.
        timings (optional): Dict of phase timings in seconds.
    """
    timings = {}

    # ===================================================================
    # PHASE 1: SCAN — Read Parquet files with projection + pushdown
    # ===================================================================
    # This is typically the dominant cost (>60% of total time).
    # We read only the columns needed and push filters into the reader.
    # ===================================================================
    t0 = time.perf_counter()

    # Scan customer: only need custkey + segment (2 of 8 columns)
    customer = scan_parquet(
        os.path.join(data_dir, "customer.parquet"),
        columns=["c_custkey", "c_mktsegment"],
        filters=[("c_mktsegment", "=", SEGMENT)],
    )

    # Scan orders: need orderkey, custkey (for join), date, priority (4 of 9 columns)
    orders = scan_parquet(
        os.path.join(data_dir, "orders.parquet"),
        columns=["o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"],
        filters=[("o_orderdate", "<", ORDER_DATE_CUTOFF)],
    )

    # Scan lineitem: need orderkey, price, discount, shipdate (4 of 16 columns)
    lineitem = scan_parquet(
        os.path.join(data_dir, "lineitem.parquet"),
        columns=["l_orderkey", "l_extendedprice", "l_discount", "l_shipdate"],
        filters=[("l_shipdate", ">", SHIP_DATE_CUTOFF)],
    )

    timings["scan"] = time.perf_counter() - t0

    # ===================================================================
    # PHASE 2: JOIN BUILD — Semi-join + hash join construction
    # ===================================================================
    # Semi-join: customer → orders (membership test only, no payload)
    # Hash join: orders → lineitem (carry date + priority as payload)
    # ===================================================================
    t0 = time.perf_counter()

    # Build customer key set for semi-join
    custkey_set = build_semi_join_set(customer["c_custkey"])

    # Probe: filter orders to only BUILDING customers
    orders_filtered = probe_semi_join(custkey_set, orders)

    # Drop o_custkey — no longer needed after semi-join
    del orders_filtered["o_custkey"]

    # Build hash map for orders → lineitem join
    order_map = build_hash_join(orders_filtered)

    timings["join_build"] = time.perf_counter() - t0

    # ===================================================================
    # PHASE 3: FUSED JOIN PROBE + AGGREGATION
    # ===================================================================
    # For each lineitem row: probe order_map, compute revenue, accumulate.
    # This is where the "query processing" happens.
    # ===================================================================
    t0 = time.perf_counter()

    if agg_method == "numpy":
        agg = fused_join_aggregate_numpy(order_map, lineitem)
    elif agg_method == "dict":
        agg = fused_join_aggregate_dict(order_map, lineitem)
    else:
        raise ValueError(f"Unknown agg_method: {agg_method!r}. Use 'numpy' or 'dict'.")

    timings["aggregate"] = time.perf_counter() - t0

    # ===================================================================
    # PHASE 4: TOP-K SORT
    # ===================================================================
    # Extract top 10 by revenue DESC, o_orderdate ASC.
    # Uses partial sort (O(n)) + full sort on K elements.
    # ===================================================================
    t0 = time.perf_counter()

    result = top_k(agg, k=TOP_K)

    timings["top_k"] = time.perf_counter() - t0

    timings["total"] = sum(timings.values())

    if return_timings:
        return result, timings
    return result


def format_result(result: dict[str, np.ndarray]) -> str:
    """
    Format the Q3 result as a readable table string.
    
    Matches the output format of DuckDB for visual comparison.
    """
    lines = []
    header = f"{'l_orderkey':>12s}  {'revenue':>14s}  {'o_orderdate':>12s}  {'o_shippriority':>14s}"
    lines.append(header)
    lines.append("-" * len(header))

    for i in range(len(result["l_orderkey"])):
        lines.append(
            f"{result['l_orderkey'][i]:>12d}  "
            f"{result['revenue'][i]:>14.4f}  "
            f"{str(result['o_orderdate'][i]):>12s}  "
            f"{result['o_shippriority'][i]:>14d}"
        )

    return "\n".join(lines)


def result_to_csv(result: dict[str, np.ndarray], filepath: str) -> None:
    """Write the Q3 result to a CSV file."""
    with open(filepath, "w") as f:
        f.write("l_orderkey,revenue,o_orderdate,o_shippriority\n")
        for i in range(len(result["l_orderkey"])):
            f.write(
                f"{result['l_orderkey'][i]},"
                f"{result['revenue'][i]:.4f},"
                f"{result['o_orderdate'][i]},"
                f"{result['o_shippriority'][i]}\n"
            )


# ---------------------------------------------------------------------------
# Self-test: run Q3 pipeline and show result + timings
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/sf1"

    print(f"Running TPC-H Q3 on {data_dir} ...\n")

    # Run with both methods to compare
    for method in ["numpy", "dict"]:
        result, timings = run_q3(data_dir, agg_method=method, return_timings=True)

        print(f"=== Method: {method} ===")
        print(format_result(result))
        print(f"\nTimings:")
        for phase, t in timings.items():
            pct = (t / timings["total"]) * 100 if timings["total"] > 0 else 0
            print(f"  {phase:15s}: {t:.4f}s  ({pct:5.1f}%)")
        print()
