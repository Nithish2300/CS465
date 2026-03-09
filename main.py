"""
main.py — CLI entry point for the TPC-H Q3 specialized query processor.

This is the "one-command runner" required by the project specification.

Usage examples:
    # Run Q3 on SF=1, print result to stdout
    python main.py --data data/sf1

    # Run Q3 and save result to CSV
    python main.py --data data/sf1 --out result.csv
    
    # Run Q3 and validate against DuckDB
    python main.py --data data/sf1 --validate

    # Benchmark mode: 6 runs, report average of last 5
    python main.py --data data/sf1 --benchmark

    # Benchmark all scale factors
    python main.py --benchmark-all

    # Use dict-based aggregation instead of numpy
    python main.py --data data/sf1 --method dict
"""

import argparse
import sys

from query import run_q3, format_result, result_to_csv


def main():
    parser = argparse.ArgumentParser(
        description="TPC-H Q3 Specialized Query Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --data data/sf1                    Run Q3, print result
  python main.py --data data/sf1 --out result.csv   Run Q3, save to CSV
  python main.py --data data/sf1 --validate         Run Q3, check vs DuckDB
  python main.py --data data/sf1 --benchmark        Benchmark mode (6 runs)
  python main.py --benchmark-all                    Benchmark all scale factors
        """,
    )
    parser.add_argument("--data", type=str, default=None,
                        help="Path to scale-factor data directory (e.g., data/sf1)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV file path (default: print to stdout)")
    parser.add_argument("--method", type=str, default="numpy", choices=["numpy", "dict"],
                        help="Aggregation method: numpy (fast) or dict (simple)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate result against DuckDB output")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark mode: run multiple times and report average")
    parser.add_argument("--benchmark-all", action="store_true",
                        help="Run full benchmark across all scale factors")
    parser.add_argument("--runs", type=int, default=6,
                        help="Number of runs for benchmark mode (default: 6)")
    args = parser.parse_args()

    # --- Benchmark all scale factors ---
    if args.benchmark_all:
        from benchmark import run_benchmark, print_summary_table
        sf_list = [0.5, 1, 2, 5]
        results = run_benchmark(scale_factors=sf_list, n_runs=args.runs, agg_method=args.method)
        if results:
            print_summary_table(results)
        return

    # --- Everything else requires --data ---
    if not args.data:
        parser.error("--data is required (or use --benchmark-all)")

    # --- Benchmark single SF ---
    if args.benchmark:
        from benchmark import run_benchmark, print_summary_table
        # Extract SF from path for display (e.g., "data/sf1" → 1.0)
        results = run_benchmark(
            scale_factors=None,  # will be inferred
            n_runs=args.runs,
            agg_method=args.method,
        )
        # Actually, let's just benchmark the specified directory directly
        from benchmark import benchmark_duckdb, benchmark_custom
        print(f"\nBenchmarking on {args.data} — {args.runs} runs (1 warm-up + {args.runs - 1} measured)\n")
        
        duck = benchmark_duckdb(args.data, args.runs)
        print(f"DuckDB avg:  {duck['avg']:.4f}s")

        custom = benchmark_custom(args.data, args.runs, args.method)
        print(f"Custom avg:  {custom['avg']:.4f}s")

        ratio = custom["avg"] / duck["avg"] if duck["avg"] > 0 else float("inf")
        print(f"Ratio:       {ratio:.2f}x")

        print(f"\nPhase breakdown:")
        for phase, t in custom["avg_phases"].items():
            pct = (t / custom["avg_phases"]["total"]) * 100
            print(f"  {phase:15s}: {t:.4f}s  ({pct:5.1f}%)")
        return

    # --- Validate against DuckDB ---
    if args.validate:
        from test import validate_q3
        success = validate_q3(args.data, agg_method=args.method)
        sys.exit(0 if success else 1)

    # --- Normal run: execute Q3 and output result ---
    result, timings = run_q3(args.data, agg_method=args.method, return_timings=True)

    if args.out:
        result_to_csv(result, args.out)
        print(f"Result written to {args.out}")
    else:
        print(format_result(result))

    print(f"\nTotal time: {timings['total']:.4f}s")


if __name__ == "__main__":
    main()
