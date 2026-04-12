#!/usr/bin/env python3
"""Benchmark: compare Rust Q17 processor against DuckDB (single-threaded).

Produces a runtime comparison table and a plot saved to benchmark_results.png.
"""

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

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

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
SF_NUMERIC = [0.5, 1, 2, 5]
RUNS = 6  # 1 warmup + 5 measured
PLOT_FILE = "benchmark_results.png"


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


def generate_plot(sf_labels, duckdb_ms, rust_ms, speedups, out_path):
    """Generate a two-panel benchmark plot and save to out_path."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel 1: Runtime vs Scale Factor (line plot) ---
    sf_nums = [0.5, 1, 2, 5][:len(sf_labels)]
    ax1.plot(sf_nums, duckdb_ms, 'o-', color='#e74c3c', linewidth=2.5,
             markersize=9, label='DuckDB', zorder=5)
    ax1.plot(sf_nums, rust_ms, 's-', color='#2ecc71', linewidth=2.5,
             markersize=9, label='Rust Q17', zorder=5)

    # Annotate points with values
    for i, (sf, d, r) in enumerate(zip(sf_nums, duckdb_ms, rust_ms)):
        ax1.annotate(f'{d:.0f}ms', (sf, d), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=9, color='#e74c3c')
        ax1.annotate(f'{r:.0f}ms', (sf, r), textcoords="offset points",
                     xytext=(0, -18), ha='center', fontsize=9, color='#2ecc71')

    ax1.set_xlabel('Scale Factor', fontsize=12)
    ax1.set_ylabel('Average Runtime (ms)', fontsize=12)
    ax1.set_title('TPC-H Q17: Runtime vs Scale Factor', fontsize=13, fontweight='bold')
    ax1.set_xticks(sf_nums)
    ax1.set_xticklabels([f'SF {s}' for s in sf_nums])
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5.5)
    ax1.set_ylim(0, max(duckdb_ms) * 1.25)

    # --- Panel 2: Speedup bar chart ---
    x = range(len(sf_labels))
    bars = ax2.bar(x, speedups, color='#3498db', width=0.5, edgecolor='#2c3e50',
                   linewidth=0.8, zorder=5)
    ax2.axhline(y=1.0, color='#e74c3c', linestyle='--', linewidth=1.5,
                label='1.0x (parity)', zorder=3)

    # Annotate bars
    for bar, sp in zip(bars, speedups):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                 f'{sp:.2f}x', ha='center', va='bottom', fontsize=11,
                 fontweight='bold', color='#2c3e50')

    ax2.set_xlabel('Scale Factor', fontsize=12)
    ax2.set_ylabel('Speedup (DuckDB / Rust)', fontsize=12)
    ax2.set_title('Speedup: Rust Q17 vs DuckDB', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'SF {s}' for s in sf_nums[:len(sf_labels)]])
    ax2.legend(fontsize=10)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_ylim(0, max(speedups) * 1.3)

    fig.tight_layout(pad=3.0)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nPlot saved to: {out_path}")


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

    sf_labels = []
    duckdb_results = []
    rust_results = []
    speedups = []

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

        sf_labels.append(sf)
        duckdb_results.append(duckdb_avg)
        rust_results.append(rust_avg)
        speedups.append(speedup)

    print()
    print("Speedup = DuckDB time / Rust time (higher is better for Rust)")

    # Generate plot
    if sf_labels:
        if HAS_MPL:
            generate_plot(sf_labels, duckdb_results, rust_results, speedups, PLOT_FILE)
        else:
            print(f"\nWARNING: matplotlib not installed. Skipping plot generation.")
            print(f"Install with: pip3 install matplotlib")


if __name__ == "__main__":
    main()
