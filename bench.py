#!/usr/bin/env python3
"""Benchmark: compare Rust Q17 processor against DuckDB (single-threaded)."""

import os
import re
import subprocess
import sys
import time

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
RUNS = 6  # 1 warmup + 5 measured


def parse_duration_to_secs(s):
    """Parse Rust's Debug-formatted Duration (e.g. '157.13ms') to seconds."""
    m = re.match(r'^([\d.]+)(ns|[µμu]s|ms|s)$', s.strip())
    if not m:
        raise ValueError(f"Cannot parse duration: {s!r}")
    value = float(m.group(1))
    unit = m.group(2)
    if unit == 'ns':
        return value * 1e-9
    elif unit in ('µs', 'μs', 'us'):
        return value * 1e-6
    elif unit == 'ms':
        return value * 1e-3
    else:
        return value


def benchmark_duckdb(data_dir, n_runs=RUNS):
    times = []
    for _ in range(n_runs):
        con = duckdb.connect()
        con.execute("PRAGMA threads=1")
        sql = Q17_SQL.format(data_dir=data_dir)
        t0 = time.perf_counter()
        con.execute(sql).fetchall()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        con.close()
    return times[1:]  # skip warmup


def benchmark_rust(data_dir, n_runs=RUNS):
    """Run Rust binary with --bench and parse per-run times from stderr."""
    proc = subprocess.run(
        [RUST_BINARY, "--data", data_dir, "--bench", str(n_runs)],
        capture_output=True, text=True, check=True
    )
    times = []
    for line in proc.stderr.splitlines():
        m = re.match(r'^Run \d+: (.+)$', line.strip())
        if m:
            times.append(parse_duration_to_secs(m.group(1)))
    if len(times) < n_runs:
        raise RuntimeError(
            f"Expected {n_runs} run times, got {len(times)}:\n{proc.stderr}"
        )
    return times[1:]  # skip warmup


def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0


def main():
    if not os.path.exists(RUST_BINARY):
        print(f"Building {RUST_BINARY}...")
        subprocess.run(
            ["bash", "-c", 'RUSTFLAGS="-C target-cpu=native" cargo build --release'],
            check=True
        )

    print(f"Benchmark: {RUNS} runs per engine ({RUNS-1} measured, 1 warmup)")
    print(f"DuckDB: single-threaded (PRAGMA threads=1)")
    print()
    print(f"{'SF':<8} {'DuckDB (ms)':<15} {'Rust (ms)':<15} {'Speedup':<10}")
    print("-" * 48)

    for sf in SCALE_FACTORS:
        data_dir = os.path.join(DATA_ROOT, sf)
        if not os.path.exists(data_dir):
            print(f"{sf:<8} SKIPPED")
            continue

        duckdb_times = benchmark_duckdb(data_dir)
        rust_times = benchmark_rust(data_dir)

        duckdb_avg = avg(duckdb_times) * 1000  # to ms
        rust_avg = avg(rust_times) * 1000

        speedup = duckdb_avg / rust_avg if rust_avg > 0 else float('inf')

        print(f"{sf:<8} {duckdb_avg:<15.1f} {rust_avg:<15.1f} {speedup:<10.2f}x")

    print()
    print("Speedup = DuckDB time / Rust time (higher is better for Rust)")


if __name__ == "__main__":
    main()
