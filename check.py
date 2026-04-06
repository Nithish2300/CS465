#!/usr/bin/env python3
"""Correctness check: compare Rust Q17 processor output against DuckDB."""

import os
import subprocess
import sys

try:
    import duckdb
except ImportError:
    print("ERROR: duckdb Python package required. Install with: pip3 install duckdb")
    sys.exit(1)

Q17_SQL = """
SELECT SUM(l_extendedprice) / 7.0 AS avg_yearly
FROM read_parquet('{data_dir}/lineitem.parquet') AS lineitem,
     read_parquet('{data_dir}/part.parquet') AS part
WHERE p_partkey = l_partkey
  AND p_brand = 'Brand#23'
  AND p_container = 'MED BOX'
  AND l_quantity < (
    SELECT 0.2 * AVG(l_quantity)
    FROM read_parquet('{data_dir}/lineitem.parquet')
    WHERE l_partkey = p_partkey
  )
"""

RUST_BINARY = "target/release/q17"
DATA_ROOT = "data"
SCALE_FACTORS = ["sf0.5", "sf1", "sf2", "sf5"]
TOLERANCE = 0.01


def get_duckdb_result(data_dir):
    con = duckdb.connect()
    con.execute("PRAGMA threads=1")
    sql = Q17_SQL.format(data_dir=data_dir)
    result = con.execute(sql).fetchall()
    con.close()
    return result[0][0]


def get_rust_result(data_dir):
    proc = subprocess.run(
        [RUST_BINARY, "--data", data_dir],
        capture_output=True, text=True
    )
    if proc.returncode != 0:
        print(f"  Rust binary failed: {proc.stderr}")
        return None
    lines = proc.stdout.strip().split("\n")
    if len(lines) < 2 or lines[1].strip() == "":
        return None  # NULL result
    return float(lines[1])


def main():
    if not os.path.exists(RUST_BINARY):
        print(f"Building {RUST_BINARY}...")
        subprocess.run(
            ["bash", "-c", 'RUSTFLAGS="-C target-cpu=native" cargo build --release'],
            check=True
        )

    all_pass = True
    print(f"{'SF':<8} {'DuckDB':<20} {'Rust':<20} {'Diff':<12} {'Status'}")
    print("-" * 72)

    for sf in SCALE_FACTORS:
        data_dir = os.path.join(DATA_ROOT, sf)
        if not os.path.exists(data_dir):
            print(f"{sf:<8} SKIPPED (directory not found)")
            continue

        duckdb_val = get_duckdb_result(data_dir)
        rust_val = get_rust_result(data_dir)

        if duckdb_val is None and rust_val is None:
            status = "PASS"
            diff = "N/A"
            duckdb_str = "NULL"
            rust_str = "NULL"
        elif duckdb_val is None or rust_val is None:
            status = "FAIL"
            diff = "N/A"
            duckdb_str = str(duckdb_val)
            rust_str = str(rust_val)
            all_pass = False
        else:
            d = abs(duckdb_val - rust_val)
            diff = f"{d:.6f}"
            status = "PASS" if d < TOLERANCE else "FAIL"
            duckdb_str = f"{duckdb_val:.2f}"
            rust_str = f"{rust_val:.2f}"
            if status == "FAIL":
                all_pass = False

        print(f"{sf:<8} {duckdb_str:<20} {rust_str:<20} {diff:<12} {status}")

    print()
    if all_pass:
        print("All scale factors PASSED correctness check.")
    else:
        print("Some scale factors FAILED. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
