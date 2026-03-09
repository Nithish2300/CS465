"""
scanner.py — Parquet scan operator with column projection and predicate pushdown.

This is the storage/IO layer of our query processor. It reads Parquet files
using PyArrow and applies two key optimizations:

1. COLUMN PROJECTION — Only read the columns needed for Q3, not all columns.
   For example, lineitem has 16 columns but we only need 4.
   This reduces I/O bandwidth and decompression work dramatically.

2. PREDICATE PUSHDOWN — Pass filter conditions to the Parquet reader so it
   can use row-group statistics (min/max per column per row group) to skip
   entire row groups without reading/decompressing them.
   Example: if a row group's min(o_orderdate) >= 1995-03-15, we know no rows
   in that group can pass our filter, so we skip it entirely.

The scanner returns data as a dict of NumPy arrays (one per column), which
is the format our join and aggregation operators expect. We use NumPy arrays
rather than PyArrow tables downstream because:
  - NumPy operations (arithmetic, indexing) are the building blocks for our
    manual join/aggregation operators
  - Zero-copy conversion from Arrow to NumPy is possible for numeric columns
  - It gives us full control over the execution (vs. using PyArrow's built-in
    join/aggregate which would hide the query processing logic)

Usage:
    from scanner import scan_parquet
    
    # Read only 2 columns from customer, filtering to 'BUILDING' segment
    data = scan_parquet(
        "data/sf1/customer.parquet",
        columns=["c_custkey", "c_mktsegment"],
        filters=[("c_mktsegment", "=", "BUILDING")]
    )
    # data = {"c_custkey": np.array([...]), "c_mktsegment": np.array([...])}
"""

import os
import numpy as np
import pyarrow.parquet as pq


def scan_parquet(
    filepath: str,
    columns: list[str],
    filters: list[tuple] | None = None,
    use_memory_map: bool = True,
) -> dict[str, np.ndarray]:
    """
    Read a Parquet file with column projection and optional predicate pushdown.

    This function is the "scan operator" of our query processor. In a real DBMS,
    this would be the table-scan or index-scan at the leaf of the query plan.

    Args:
        filepath:       Path to a .parquet file.
        columns:        List of column names to read (projection).
                        Only these columns will be decompressed and returned.
        filters:        Optional list of (column, op, value) tuples for pushdown.
                        PyArrow uses these to skip row groups whose statistics
                        prove no rows can match. Supported ops: '=', '<', '>', '<=', '>=', '!='.
                        Example: [("o_orderdate", "<", date(1995, 3, 15))]
        use_memory_map: If True, memory-map the file instead of reading it all
                        into memory. This lets the OS page in only the column
                        chunks we actually access.

    Returns:
        A dict mapping column_name -> numpy array for each requested column.
        All arrays have the same length (number of rows that passed the filter).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Parquet file not found: {filepath}")

    # Read the Parquet file with projection and optional pushdown filters.
    #
    # Under the hood, PyArrow will:
    #   1. Read the file footer to get the schema and row-group metadata.
    #   2. For each row group, check column statistics against our filters.
    #      If a row group can't contain matching rows, skip it entirely.
    #   3. For non-skipped row groups, read only our requested columns.
    #   4. Apply the filter predicate row-by-row on the read data to get
    #      the final filtered result.
    #
    # This two-level filtering (row-group skip + row-level filter) is exactly
    # how production databases use "zone maps" / "min-max indexes".
    table = pq.read_table(
        filepath,
        columns=columns,
        filters=filters,
        memory_map=use_memory_map,
    )

    # Convert each Arrow column to a NumPy array.
    # For numeric types (int, float, date-as-epoch), this can be zero-copy.
    # For string columns, Arrow stores them as variable-length binary, so
    # conversion to NumPy object arrays involves a copy.
    #
    # IMPORTANT: TPC-H stores prices and discounts as Decimal128 (fixed-point).
    # PyArrow preserves these as Python decimal.Decimal objects in NumPy arrays,
    # which are VERY slow for arithmetic. We cast Decimal columns to float64
    # during the scan so downstream operators can use fast NumPy math.
    # This matches DuckDB's internal behavior (it also uses float64 for TPC-H).
    import pyarrow as pa

    result = {}
    for col_name in columns:
        arrow_col = table.column(col_name)

        # Cast Decimal128/256 columns to float64 for fast arithmetic
        if pa.types.is_decimal(arrow_col.type):
            arrow_col = arrow_col.cast(pa.float64())

        result[col_name] = arrow_col.to_numpy()

    return result


def get_row_group_info(filepath: str) -> None:
    """
    Print row-group statistics for a Parquet file.
    
    This is a diagnostic/learning tool — it shows you the min/max statistics
    that PyArrow uses for predicate pushdown. Understanding these stats helps
    you see why some filters can skip row groups and others can't.
    
    For example, if lineitem.parquet has row groups sorted by l_shipdate,
    then the filter `l_shipdate > '1995-03-15'` can skip early row groups
    where max(l_shipdate) <= 1995-03-15.
    """
    pf = pq.ParquetFile(filepath)
    metadata = pf.metadata

    print(f"File: {filepath}")
    print(f"  Total rows: {metadata.num_rows:,}")
    print(f"  Row groups: {metadata.num_row_groups}")
    print(f"  Columns: {metadata.num_columns}")
    print()

    for rg_idx in range(metadata.num_row_groups):
        rg = metadata.row_group(rg_idx)
        print(f"  Row Group {rg_idx}: {rg.num_rows:,} rows")
        for col_idx in range(rg.num_columns):
            col = rg.column(col_idx)
            stats = col.statistics
            if stats and stats.has_min_max:
                print(f"    {col.path_in_schema:30s}  min={stats.min}  max={stats.max}")
    print()


# ---------------------------------------------------------------------------
# Self-test: run this file directly to see the scanner in action
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import date

    data_dir = "data/sf1"

    # --- Demo 1: Inspect row-group statistics ---
    print("=" * 70)
    print("ROW GROUP STATISTICS (what enables predicate pushdown)")
    print("=" * 70)
    get_row_group_info(os.path.join(data_dir, "customer.parquet"))
    get_row_group_info(os.path.join(data_dir, "orders.parquet"))

    # --- Demo 2: Scan customer with projection + filter ---
    print("=" * 70)
    print("SCAN: customer — columns=[c_custkey, c_mktsegment], filter=BUILDING")
    print("=" * 70)
    cust = scan_parquet(
        os.path.join(data_dir, "customer.parquet"),
        columns=["c_custkey", "c_mktsegment"],
        filters=[("c_mktsegment", "=", "BUILDING")],
    )
    print(f"  Rows returned: {len(cust['c_custkey']):,}")
    print(f"  Sample c_custkey values: {cust['c_custkey'][:5]}")
    print()

    # --- Demo 3: Scan orders with date filter ---
    print("=" * 70)
    print("SCAN: orders — filter: o_orderdate < 1995-03-15")
    print("=" * 70)
    orders = scan_parquet(
        os.path.join(data_dir, "orders.parquet"),
        columns=["o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"],
        filters=[("o_orderdate", "<", date(1995, 3, 15))],
    )
    print(f"  Rows returned: {len(orders['o_orderkey']):,}")
    print(f"  Date range: {orders['o_orderdate'].min()} to {orders['o_orderdate'].max()}")
    print()

    # --- Demo 4: Scan lineitem with shipdate filter ---
    print("=" * 70)
    print("SCAN: lineitem — filter: l_shipdate > 1995-03-15")
    print("=" * 70)
    li = scan_parquet(
        os.path.join(data_dir, "lineitem.parquet"),
        columns=["l_orderkey", "l_extendedprice", "l_discount", "l_shipdate"],
        filters=[("l_shipdate", ">", date(1995, 3, 15))],
    )
    print(f"  Rows returned: {len(li['l_orderkey']):,}")
    print(f"  Date range: {li['l_shipdate'].min()} to {li['l_shipdate'].max()}")
    print()

    # --- Summary ---
    print("=" * 70)
    print("SUMMARY: Column projection + predicate pushdown results")
    print("=" * 70)
    print(f"  customer:  {len(cust['c_custkey']):>10,} rows  (BUILDING segment only)")
    print(f"  orders:    {len(orders['o_orderkey']):>10,} rows  (before 1995-03-15)")
    print(f"  lineitem:  {len(li['l_orderkey']):>10,} rows  (shipped after 1995-03-15)")
