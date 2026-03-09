"""
benchmark.py — Benchmarking harness for TPC-H Q3: DuckDB vs custom engine.

Implements the required evaluation protocol:
  - Run at least 6 times per scale factor
  - Discard the first run (warm-up / cold cache)
  - Report the average of the remaining 5 runs
  - Both systems run single-threaded on the same machine and data

The warm-up run matters because:
  - First run includes Parquet file I/O from disk (cold OS page cache)
  - Subsequent runs benefit from OS file caching (warm page cache)
  - We want to measure query processing time, not disk seek time
  - Both DuckDB and our engine benefit equally from warm cache

Usage:
    python benchmark.py                 # benchmark all scale factors
    python benchmark.py --sf 1          # benchmark SF=1 only
    python benchmark.py --runs 11       # 11 runs (1 warm-up + 10 measured)
"""

import os
import sys
import time
import argparse
import duckdb
import platform

from data import Q3_SQL
from query import run_q3


SCALE_FACTORS = [0.5, 1, 2, 5]
DEFAULT_RUNS = 6  # 1 warm-up + 5 measured


def get_machine_info() -> dict:
    """Collect machine specifications for reproducible reporting."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }


def benchmark_duckdb(data_dir: str, n_runs: int = DEFAULT_RUNS) -> dict:
    """
    Benchmark DuckDB single-threaded on TPC-H Q3.
    
    DuckDB reads directly from Parquet files (same files as our engine).
    We use PRAGMA threads=1 to ensure single-threaded execution.
    
    Args:
        data_dir: Path to scale-factor directory (e.g., "data/sf1").
        n_runs:   Total runs (first is warm-up, rest are measured).
    
    Returns:
        Dict with "times" (all run times), "avg" (mean of measured runs),
        "runs" (number of measured runs).
    """
    times = []

    for run_idx in range(n_runs):
        con = duckdb.connect()
        con.execute("PRAGMA threads=1;")

        # Register Parquet files as views
        for table in ["customer", "orders", "lineitem"]:
            path = os.path.join(data_dir, f"{table}.parquet")
            con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path}');")

        t0 = time.perf_counter()
        con.execute(Q3_SQL).fetchall()
        elapsed = time.perf_counter() - t0

        con.close()
        times.append(elapsed)

    # Discard first run (warm-up)
    measured = times[1:]
    avg = sum(measured) / len(measured)

    return {
        "times": times,
        "measured": measured,
        "avg": avg,
        "warmup": times[0],
    }


def benchmark_custom(
    data_dir: str,
    n_runs: int = DEFAULT_RUNS,
    agg_method: str = "numpy",
) -> dict:
    """
    Benchmark our custom Q3 processor.
    
    Args:
        data_dir:   Path to scale-factor directory.
        n_runs:     Total runs (first is warm-up, rest are measured).
        agg_method: Aggregation method ("numpy" or "dict").
    
    Returns:
        Dict with "times", "avg", "warmup", and per-phase "phase_timings".
    """
    times = []
    all_phase_timings = []

    for run_idx in range(n_runs):
        t0 = time.perf_counter()
        result, phase_timings = run_q3(data_dir, agg_method=agg_method, return_timings=True)
        elapsed = time.perf_counter() - t0

        times.append(elapsed)
        all_phase_timings.append(phase_timings)

    # Discard first run
    measured = times[1:]
    measured_phases = all_phase_timings[1:]
    avg = sum(measured) / len(measured)

    # Average per-phase timings (excluding warm-up)
    avg_phases = {}
    for key in measured_phases[0]:
        avg_phases[key] = sum(p[key] for p in measured_phases) / len(measured_phases)

    return {
        "times": times,
        "measured": measured,
        "avg": avg,
        "warmup": times[0],
        "avg_phases": avg_phases,
    }


def run_benchmark(
    scale_factors: list[float] = None,
    n_runs: int = DEFAULT_RUNS,
    agg_method: str = "numpy",
) -> list[dict]:
    """
    Run the full benchmark suite across scale factors.
    
    Returns a list of result dicts, one per scale factor.
    """
    if scale_factors is None:
        scale_factors = SCALE_FACTORS

    results = []

    for sf in scale_factors:
        sf_label = f"sf{sf:g}"
        data_dir = os.path.join("data", sf_label)

        if not os.path.exists(data_dir):
            print(f"[SKIP] {data_dir} not found")
            continue

        print(f"\n{'='*70}")
        print(f"BENCHMARK: SF={sf:g}  ({data_dir})  —  {n_runs} runs (1 warm-up + {n_runs-1} measured)")
        print(f"{'='*70}")

        # DuckDB baseline
        print(f"\n  DuckDB (single-threaded)...")
        duck = benchmark_duckdb(data_dir, n_runs)
        print(f"    Warm-up: {duck['warmup']:.4f}s")
        print(f"    Measured: {['%.4f' % t for t in duck['measured']]}")
        print(f"    Average:  {duck['avg']:.4f}s")

        # Custom engine
        print(f"\n  Custom engine (agg={agg_method})...")
        custom = benchmark_custom(data_dir, n_runs, agg_method)
        print(f"    Warm-up: {custom['warmup']:.4f}s")
        print(f"    Measured: {['%.4f' % t for t in custom['measured']]}")
        print(f"    Average:  {custom['avg']:.4f}s")

        # Phase breakdown
        print(f"\n    Phase breakdown (average of measured runs):")
        for phase, t in custom["avg_phases"].items():
            pct = (t / custom["avg_phases"]["total"]) * 100
            print(f"      {phase:15s}: {t:.4f}s  ({pct:5.1f}%)")

        # Comparison
        ratio = custom["avg"] / duck["avg"] if duck["avg"] > 0 else float("inf")
        print(f"\n  Ratio (custom/DuckDB): {ratio:.2f}x", end="")
        if ratio < 1:
            print("  ← custom is FASTER!")
        elif ratio < 2:
            print("  ← within 2x")
        elif ratio < 5:
            print("  ← within 5x")
        else:
            print(f"  ← {ratio:.1f}x slower")

        results.append({
            "sf": sf,
            "duckdb_avg": duck["avg"],
            "custom_avg": custom["avg"],
            "ratio": ratio,
            "duckdb": duck,
            "custom": custom,
        })

    return results


def print_summary_table(results: list[dict]) -> None:
    """Print a summary comparison table."""
    print(f"\n{'='*70}")
    print("SUMMARY: DuckDB vs Custom Engine (average of measured runs)")
    print(f"{'='*70}")
    print(f"{'SF':>6s}  {'DuckDB (s)':>12s}  {'Custom (s)':>12s}  {'Ratio':>8s}")
    print(f"{'-'*6}  {'-'*12}  {'-'*12}  {'-'*8}")
    for r in results:
        print(f"{r['sf']:>6g}  {r['duckdb_avg']:>12.4f}  {r['custom_avg']:>12.4f}  {r['ratio']:>7.2f}x")

    print(f"\nMachine: {platform.processor() or platform.machine()}, "
          f"{os.cpu_count()} cores, "
          f"Python {platform.python_version()}, "
          f"{platform.platform()}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TPC-H Q3: DuckDB vs custom engine")
    parser.add_argument("--sf", type=float, nargs="+", default=None,
                        help="Scale factor(s) to benchmark (default: 0.5 1 2 5)")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help=f"Total runs per SF (default: {DEFAULT_RUNS}, first is warm-up)")
    parser.add_argument("--method", type=str, default="numpy", choices=["numpy", "dict"],
                        help="Aggregation method (default: numpy)")
    args = parser.parse_args()

    results = run_benchmark(
        scale_factors=args.sf,
        n_runs=args.runs,
        agg_method=args.method,
    )

    if results:
        print_summary_table(results)
