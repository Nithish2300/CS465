/// Phase 3: Compute the final Q17 result from the SoA histogram accumulators.
///
/// For each qualifying partkey, we need to evaluate:
///   l_quantity < 0.2 * AVG(l_quantity)
///
/// where AVG(l_quantity) = sum_qty_raw / count (in raw Decimal units, i.e. value * 100).
///
/// **Integer threshold arithmetic** (avoiding floating-point):
///   The predicate `qty_raw < 0.2 * (sum_qty_raw / count)` is equivalent to:
///     5 * qty_raw * count < sum_qty_raw
///   Since `qty_raw = qty_int * 100` (TPC-H quantities are always whole numbers):
///     500 * qty_int * count < sum_qty_raw
///
///   This comparison uses only integer multiplication and comparison —
///   no floating-point division or rounding in the hot path.
///
/// For each quantity bucket q (1..50) that passes the threshold, we add
/// the accumulated price from that bucket to the running total.
///
/// Finally, the total price (stored in raw Decimal cents) is converted to
/// the result: total_price / 100.0 / 7.0

use crate::types::AccumulatorsSoA;

pub fn compute_result(accumulators: &AccumulatorsSoA) -> Option<f64> {
    let mut total_price: i64 = 0;
    let mut any_qualifying = false;

    let num = accumulators.counts.len();

    for i in 0..num {
        let count = accumulators.counts[i];
        if count == 0 {
            continue;
        }

        let count = count as i64;
        let sum_qty_raw = accumulators.sum_qty_raws[i];
        let hist = &accumulators.price_by_qty[i];

        // Check each quantity bucket against the threshold:
        //   500 * q * count < sum_qty_raw
        // This is the integer-rearranged form of:
        //   q < 0.2 * AVG(l_quantity)
        for q in 1..=50usize {
            if 500 * (q as i64) * count < sum_qty_raw {
                any_qualifying = true;
                total_price += unsafe { *hist.get_unchecked(q) };
            }
        }
    }

    if any_qualifying {
        // Convert from raw Decimal cents to dollars, then divide by 7.0
        Some(total_price as f64 / 100.0 / 7.0)
    } else {
        None
    }
}
