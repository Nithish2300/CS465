"""
test.py — Correctness validation for the TPC-H Q3 query processor.

Compares our custom engine's output against DuckDB's output (ground truth).

Checks:
  1. Same number of rows (should be 10)
  2. Same l_orderkey values in the same order
  3. Revenue values match within floating-point tolerance
  4. Same o_orderdate values
  5. Same o_shippriority values

Usage:
    python test.py                  # validate on all scale factors
    python test.py --sf data/sf1    # validate on SF=1 only
    python test.py --method dict    # validate with dict aggregation
"""

import argparse
import sys
import os
import numpy as np

from data import run_duckdb_q3
from query import run_q3


def validate_q3(
    data_dir: str,
    agg_method: str = "numpy",
    tolerance: float = 0.01,
    verbose: bool = True,
) -> bool:
    """
    Validate our Q3 result against DuckDB's reference output.
    
    Args:
        data_dir:  Path to scale-factor directory (e.g., "data/sf1").
        agg_method: Aggregation method to test ("numpy" or "dict").
        tolerance: Maximum allowed absolute difference in revenue values.
                   We use 0.01 because TPC-H revenues have 4 decimal places,
                   and float64 arithmetic may introduce ~1e-4 rounding errors.
        verbose:   Print detailed comparison output.
    
    Returns:
        True if all checks pass, False otherwise.
    """
    if verbose:
        print(f"\n[validate] Testing {data_dir} with method={agg_method} ...")

    # Get DuckDB reference
    ref_df = run_duckdb_q3(data_dir)

    # Get our result
    result = run_q3(data_dir, agg_method=agg_method)

    passed = True
    checks = []

    # Check 1: Row count
    n_expected = len(ref_df)
    n_actual = len(result["l_orderkey"])
    ok = n_expected == n_actual
    checks.append(("Row count", ok, f"expected={n_expected}, got={n_actual}"))
    if not ok:
        passed = False

    # Check 2: l_orderkey values and order
    ref_keys = ref_df["l_orderkey"].values
    our_keys = result["l_orderkey"]
    ok = np.array_equal(ref_keys, our_keys)
    checks.append(("l_orderkey match", ok,
                    f"ref={ref_keys.tolist()}, ours={our_keys.tolist()}" if not ok else ""))
    if not ok:
        passed = False

    # Check 3: Revenue values (within tolerance)
    ref_rev = ref_df["revenue"].values.astype(np.float64)
    our_rev = result["revenue"]
    max_diff = np.max(np.abs(ref_rev - our_rev)) if len(ref_rev) == len(our_rev) else float("inf")
    ok = max_diff <= tolerance
    checks.append(("Revenue match", ok, f"max_diff={max_diff:.6f}, tolerance={tolerance}"))
    if not ok:
        passed = False

    # Check 4: o_orderdate values
    # DuckDB may return timestamps (datetime64) while we return date objects.
    # Normalize both to "YYYY-MM-DD" strings for comparison.
    ref_dates = np.array([str(d)[:10] for d in ref_df["o_orderdate"].values])
    our_dates = np.array([str(d)[:10] for d in result["o_orderdate"]])
    ok = np.array_equal(ref_dates, our_dates)
    checks.append(("o_orderdate match", ok,
                    f"ref={ref_dates.tolist()}, ours={our_dates.tolist()}" if not ok else ""))
    if not ok:
        passed = False

    # Check 5: o_shippriority values
    ref_pri = ref_df["o_shippriority"].values
    our_pri = result["o_shippriority"]
    ok = np.array_equal(ref_pri, our_pri)
    checks.append(("o_shippriority match", ok, ""))
    if not ok:
        passed = False

    # Print results
    if verbose:
        for name, ok, detail in checks:
            status = "PASS" if ok else "FAIL"
            detail_str = f"  ({detail})" if detail else ""
            print(f"  [{status}] {name}{detail_str}")

        if passed:
            print(f"  All checks passed for {data_dir}")
        else:
            print(f"  Some checks FAILED for {data_dir}")

    return passed


# ---------------------------------------------------------------------------
# CLI: validate across scale factors
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate TPC-H Q3 correctness")
    parser.add_argument("--sf", type=str, default=None,
                        help="Data directory to validate (e.g., data/sf1). Default: all SFs.")
    parser.add_argument("--method", type=str, default="numpy", choices=["numpy", "dict"],
                        help="Aggregation method to test")
    args = parser.parse_args()

    all_passed = True

    if args.sf:
        all_passed = validate_q3(args.sf, agg_method=args.method)
    else:
        # Validate all scale factors
        for sf in [0.5, 1, 2, 5]:
            sf_label = f"sf{sf:g}"
            data_dir = os.path.join("data", sf_label)
            if os.path.exists(data_dir):
                ok = validate_q3(data_dir, agg_method=args.method)
                if not ok:
                    all_passed = False
            else:
                print(f"[SKIP] {data_dir} not found")

    print()
    if all_passed:
        print("=== ALL VALIDATIONS PASSED ===")
    else:
        print("=== SOME VALIDATIONS FAILED ===")

    sys.exit(0 if all_passed else 1)