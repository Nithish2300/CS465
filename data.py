"""
data.py — Dataset generation and DuckDB reference baseline for TPC-H Q3.

This module provides two capabilities:
1. Generate TPC-H Parquet files using DuckDB's built-in tpch extension.
2. Run the official TPC-H Q3 SQL in DuckDB (single-threaded) to produce
   a reference answer for correctness validation.

Usage:
    python data.py                  # generate data for all scale factors
    python data.py --sf 1           # generate data for SF=1 only
    python data.py --reference sf1  # run DuckDB Q3 on SF=1, print result
"""

import os
import argparse
import duckdb

# ---------------------------------------------------------------------------
# TPC-H Q3 SQL — the exact query we're implementing as a specialized processor.
#
# This query finds the top-10 unshipped orders with the highest revenue for
# customers in the 'BUILDING' market segment, where:
#   - The order was placed BEFORE 1995-03-15
#   - The lineitem was shipped AFTER 1995-03-15
#
# Tables involved:
#   customer  — filtered by c_mktsegment = 'BUILDING'
#   orders    — filtered by o_orderdate < '1995-03-15', joined to customer
#   lineitem  — filtered by l_shipdate > '1995-03-15', joined to orders
#
# The join path is:  customer --[c_custkey]--> orders --[o_orderkey]--> lineitem
# Output: top 10 by revenue DESC, then o_orderdate ASC
# ---------------------------------------------------------------------------

Q3_SQL = """
SELECT
    l_orderkey,
    SUM(l_extendedprice * (1 - l_discount)) AS revenue,
    o_orderdate,
    o_shippriority
FROM
    customer,
    orders,
    lineitem
WHERE
    c_mktsegment = 'BUILDING'
    AND c_custkey = o_custkey
    AND l_orderkey = o_orderkey
    AND o_orderdate < DATE '1995-03-15'
    AND l_shipdate > DATE '1995-03-15'
GROUP BY
    l_orderkey,
    o_orderdate,
    o_shippriority
ORDER BY
    revenue DESC,
    o_orderdate
LIMIT 10;
"""

# The 8 TPC-H tables that get exported to Parquet
TPCH_TABLES = [
    "region", "nation", "supplier", "part",
    "partsupp", "customer", "orders", "lineitem",
]

SCALE_FACTORS = [0.5, 1, 2, 5]


def generate_data(sf: float, data_dir: str = "data") -> None:
    """
    Generate TPC-H tables at the given scale factor and export to Parquet.
    
    Each table becomes one Parquet file in data/sf{sf}/.
    Skips generation if all 8 files already exist.
    
    Args:
        sf: Scale factor (0.5, 1, 2, or 5).
        data_dir: Root data directory.
    """
    # Build the output folder name: sf0.5, sf1, sf2, sf5
    sf_label = f"sf{sf:g}"  # :g removes trailing zeros (1.0 -> "1")
    out_dir = os.path.join(data_dir, sf_label)
    os.makedirs(out_dir, exist_ok=True)

    # Check if data already exists
    existing = [f for f in TPCH_TABLES if os.path.exists(os.path.join(out_dir, f"{f}.parquet"))]
    if len(existing) == len(TPCH_TABLES):
        print(f"[data] SF={sf:g}: all {len(TPCH_TABLES)} Parquet files already exist in {out_dir}/, skipping.")
        return

    print(f"[data] SF={sf:g}: generating TPC-H data into {out_dir}/ ...")

    # Use an in-memory DuckDB connection
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute(f"CALL dbgen(sf={sf});")

    for table in TPCH_TABLES:
        out_path = os.path.join(out_dir, f"{table}.parquet")
        con.execute(f"COPY {table} TO '{out_path}' (FORMAT 'parquet');")
        print(f"  -> {out_path}")

    con.close()
    print(f"[data] SF={sf:g}: done.")


def run_duckdb_q3(data_dir: str) -> "duckdb.DuckDBPyRelation":
    """
    Run TPC-H Q3 in DuckDB (single-threaded) reading from Parquet files.
    
    This is our ground-truth reference. We read directly from the Parquet
    files (not from DuckDB's internal tables) so the comparison is fair —
    both DuckDB and our engine read the same files.
    
    Args:
        data_dir: Path to a scale-factor directory, e.g. "data/sf1".
    
    Returns:
        A Pandas DataFrame with the Q3 result (10 rows).
    """
    con = duckdb.connect()
    con.execute("PRAGMA threads=1;")  # single-threaded baseline

    # Register each Parquet file as a DuckDB view so the SQL references
    # table names (customer, orders, lineitem) naturally.
    for table in ["customer", "orders", "lineitem"]:
        parquet_path = os.path.join(data_dir, f"{table}.parquet")
        con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{parquet_path}');")

    result_df = con.execute(Q3_SQL).fetchdf()
    con.close()
    return result_df


# ---------------------------------------------------------------------------
# CLI: run directly to generate data or get reference results
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPC-H data generation & DuckDB Q3 reference")
    parser.add_argument("--sf", type=float, default=None,
                        help="Generate data for a single scale factor (e.g. 1). Default: all.")
    parser.add_argument("--reference", type=str, default=None,
                        help="Run DuckDB Q3 on this data dir (e.g. data/sf1) and print result.")
    args = parser.parse_args()

    if args.reference:
        print(f"\n[DuckDB Q3 Reference] Running on {args.reference} ...\n")
        df = run_duckdb_q3(args.reference)
        print(df.to_string(index=False))
        print()
    else:
        factors = [args.sf] if args.sf else SCALE_FACTORS
        for sf in factors:
            generate_data(sf)
