/// Shared types for the TPC-H Q17 specialized processor.

// ---------------------------------------------------------------------------
// AccumulatorsSoA – Struct-of-Arrays layout for histogram accumulators
// ---------------------------------------------------------------------------
//
// For each qualifying part (Brand#23 + MED BOX), we accumulate:
//   - counts[i]:       number of lineitem rows for qualifying part i
//   - sum_qty_raws[i]: sum of l_quantity in raw Decimal units (value * 100)
//   - price_by_qty[i]: price histogram — bucket q (1..=50) holds the sum of
//                      l_extendedprice for all rows where qty_int == q
//                      (index 0 is unused; TPC-H quantities are integers in [1, 50])
//
// **Why SoA instead of AoS?**
// With Array-of-Structs, each accumulator is ~420 bytes (4 + 8 + 51*8).
// Random access to any accumulator pollutes the cache with 6-7 cache lines,
// most of which contain price_by_qty buckets we don't need yet.
//
// With Struct-of-Arrays:
//   - counts (4B each) and sum_qty_raws (8B each) are packed into small,
//     contiguous arrays that fit entirely in L1/L2 cache (~200 parts at SF1
//     → ~800B counts + ~1.6KB sums = ~2.4KB total).
//   - price_by_qty is in a separate array, only touching the one cache line
//     containing the specific bucket we need.
//
// This reduces cache pressure on the hot path, where ~99.9% of rows only
// touch the bitmap (no accumulator at all), and the remaining 0.1% benefit
// from having counts/sums warm in L1.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct AccumulatorsSoA {
    pub counts: Vec<u32>,
    pub sum_qty_raws: Vec<i64>,
    pub price_by_qty: Vec<[i64; 51]>,
}

impl AccumulatorsSoA {
    pub fn new(size: usize) -> Self {
        AccumulatorsSoA {
            counts: vec![0u32; size],
            sum_qty_raws: vec![0i64; size],
            price_by_qty: vec![[0i64; 51]; size],
        }
    }
}

// ---------------------------------------------------------------------------
// PartScanResult – output of Phase 1
// ---------------------------------------------------------------------------
//
// After scanning the part table and filtering for Brand#23 + MED BOX:
//   - bitmap:    bit vector indexed by partkey; bit is set if the part qualifies
//   - pk_to_idx: maps partkey → dense index into the accumulators arrays
//   - num_qualifying: total number of qualifying parts
//   - min_qualifying_pk / max_qualifying_pk: bounds of qualifying partkey range,
//     used for row-group skipping in Phase 2
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct PartScanResult {
    pub bitmap: Vec<u64>,
    pub pk_to_idx: Vec<u32>,
    pub num_qualifying: usize,
    pub min_qualifying_pk: i64,
    pub max_qualifying_pk: i64,
}
