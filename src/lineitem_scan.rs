/// Phase 2: Single-pass lineitem scan with bitmap-probed histogram accumulation.
///
/// This is the performance-critical phase. The lineitem table is the largest
/// table in TPC-H (6M rows at SF1, 30M at SF5), so every cycle matters.
///
/// # Optimizations applied
///
/// **mmap I/O**: The file is memory-mapped via `memmap2` and wrapped in a
/// zero-copy `bytes::Bytes`. The Parquet reader accesses column chunks directly
/// from the mapped region — the OS pages in only the data touched, avoiding
/// userspace buffer copies and redundant syscalls.
///
/// **Row-group skipping**: Before reading each row group, its `l_partkey`
/// column statistics (min/max from the Parquet footer) are checked against the
/// qualifying-parts bitmap. If no qualifying partkey falls within the row
/// group's partkey range, the entire row group is skipped without decompressing
/// any data. This leverages Parquet's built-in zone-map metadata.
///
/// **Column projection**: Reads only 3 of 16 lineitem columns
/// (`l_partkey`, `l_quantity`, `l_extendedprice`), cutting I/O by ~75%.
///
/// **Bitmap probe**: For each row, a single bit test in the qualifying-parts
/// bitmap eliminates ~99.9% of rows (only Brand#23 + MED BOX parts qualify).
///
/// **SoA accumulation**: Qualifying rows update Struct-of-Arrays accumulators
/// (`counts`, `sum_qty_raws`, `price_by_qty`) for better cache behavior.
///
/// **Hot loop**: Uses `unsafe get_unchecked` to eliminate bounds checks and
/// accesses Arrow's raw value buffers directly (no per-element null checks).

use arrow::array::{Array, Decimal128Array, Int64Array};
use bytes::Bytes;
use memmap2::Mmap;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use std::fs::File;
use std::path::Path;

use crate::types::AccumulatorsSoA;

/// Check whether any bit in `bitmap` is set within the range [min_pk, max_pk].
///
/// Used for row-group skipping: if no qualifying partkey falls in a row group's
/// l_partkey range, the entire row group can be skipped.
fn bitmap_has_any_in_range(bitmap: &[u64], min_pk: usize, max_pk: usize) -> bool {
    let bitmap_len = bitmap.len();
    let start_word = min_pk >> 6;
    let end_word = max_pk >> 6;

    if start_word >= bitmap_len {
        return false;
    }
    let end_word = end_word.min(bitmap_len - 1);

    if start_word == end_word {
        // Both endpoints in the same bitmap word
        let lo_mask = !((1u64 << (min_pk & 63)) - 1); // bits >= min_pk
        let hi_bits = (max_pk & 63) + 1;
        let hi_mask = if hi_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << hi_bits) - 1
        }; // bits <= max_pk
        return bitmap[start_word] & lo_mask & hi_mask != 0;
    }

    // Check first partial word (bits >= min_pk within this word)
    let lo_mask = !((1u64 << (min_pk & 63)) - 1);
    if bitmap[start_word] & lo_mask != 0 {
        return true;
    }

    // Check full words in between
    for w in (start_word + 1)..end_word {
        if bitmap[w] != 0 {
            return true;
        }
    }

    // Check last partial word (bits <= max_pk within this word)
    let hi_bits = (max_pk & 63) + 1;
    let hi_mask = if hi_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << hi_bits) - 1
    };
    bitmap[end_word] & hi_mask != 0
}

pub fn scan_lineitem(
    path: &Path,
    bitmap: &[u64],
    pk_to_idx: &[u32],
    min_qualifying_pk: i64,
    max_qualifying_pk: i64,
    accumulators: &mut AccumulatorsSoA,
) {
    // -----------------------------------------------------------------------
    // mmap the file for zero-copy Parquet reads
    // -----------------------------------------------------------------------
    let file = File::open(path).expect("Failed to open lineitem.parquet");
    let mmap = unsafe { Mmap::map(&file).expect("Failed to mmap lineitem.parquet") };
    let bytes = Bytes::from_owner(mmap);

    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)
        .expect("Failed to read lineitem parquet metadata");

    let metadata = builder.metadata();
    let parquet_schema = builder.parquet_schema();
    let arrow_schema = builder.schema();

    // -----------------------------------------------------------------------
    // Row-group skipping: identify which row groups may contain qualifying rows
    // -----------------------------------------------------------------------
    let pk_parquet_idx = parquet_schema
        .columns()
        .iter()
        .position(|c| c.name() == "l_partkey")
        .expect("l_partkey column not found in parquet schema");

    let mut qualifying_row_groups: Vec<usize> = Vec::new();
    let mut skipped_row_groups = 0usize;

    for rg_idx in 0..metadata.num_row_groups() {
        let rg_meta = metadata.row_group(rg_idx);
        let mut should_include = true;

        if let Some(stats) = rg_meta.column(pk_parquet_idx).statistics() {
            if let parquet::file::statistics::Statistics::Int64(ref s) = stats {
                if let (Some(&rg_min), Some(&rg_max)) = (s.min_opt(), s.max_opt()) {
                    // Quick range check: does this row group overlap with qualifying range?
                    if rg_max < min_qualifying_pk || rg_min > max_qualifying_pk {
                        should_include = false;
                    } else {
                        // Precise bitmap check: does any qualifying partkey fall in this range?
                        let check_min = rg_min.max(0) as usize;
                        let check_max = rg_max as usize;
                        if !bitmap_has_any_in_range(bitmap, check_min, check_max) {
                            should_include = false;
                        }
                    }
                }
            }
        }

        if should_include {
            qualifying_row_groups.push(rg_idx);
        } else {
            skipped_row_groups += 1;
        }
    }

    if skipped_row_groups > 0 {
        eprintln!(
            "  Row-group skipping: {}/{} row groups skipped",
            skipped_row_groups,
            metadata.num_row_groups()
        );
    }

    // -----------------------------------------------------------------------
    // Column projection: only read l_partkey, l_quantity, l_extendedprice
    // -----------------------------------------------------------------------
    let pk_arrow_idx = arrow_schema.index_of("l_partkey").unwrap();
    let qty_arrow_idx = arrow_schema.index_of("l_quantity").unwrap();
    let price_arrow_idx = arrow_schema.index_of("l_extendedprice").unwrap();

    let mask = ProjectionMask::roots(
        parquet_schema,
        [pk_arrow_idx, qty_arrow_idx, price_arrow_idx],
    );

    let reader = builder
        .with_row_groups(qualifying_row_groups)
        .with_projection(mask)
        .with_batch_size(16384)
        .build()
        .expect("Failed to build lineitem reader");

    let bitmap_len = bitmap.len();

    // Cache projected column indices (constant across all batches)
    let mut pk_col_idx = 0;
    let mut qty_col_idx = 1;
    let mut price_col_idx = 2;
    let mut indices_resolved = false;

    // -----------------------------------------------------------------------
    // Main scan loop
    // -----------------------------------------------------------------------
    for batch in reader {
        let batch = batch.expect("Failed to read lineitem batch");

        // Resolve column indices from the projected schema on the first batch
        if !indices_resolved {
            let projected_schema = batch.schema();
            pk_col_idx = projected_schema.index_of("l_partkey").unwrap();
            qty_col_idx = projected_schema.index_of("l_quantity").unwrap();
            price_col_idx = projected_schema.index_of("l_extendedprice").unwrap();
            indices_resolved = true;
        }

        let pk_col = batch
            .column(pk_col_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("l_partkey must be Int64");
        let qty_col = batch
            .column(qty_col_idx)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("l_quantity must be Decimal128");
        let price_col = batch
            .column(price_col_idx)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("l_extendedprice must be Decimal128");

        // Direct access to underlying Arrow buffers for maximum throughput
        let pk_values = pk_col.values();
        let qty_values = qty_col.values();
        let price_values = price_col.values();
        let n = batch.num_rows();

        // -------------------------------------------------------------------
        // Hot loop: bitmap probe + SoA histogram accumulation
        //
        // For each row:
        //   1. Read l_partkey and compute bitmap word index
        //   2. Bounds-check the word index (handles partkeys beyond bitmap)
        //   3. Test the bit — ~99.9% of rows exit here
        //   4. For qualifying rows: update SoA accumulators
        // -------------------------------------------------------------------
        for i in 0..n {
            let pk = unsafe { *pk_values.get_unchecked(i) } as usize;

            // Bitmap bounds check + bit test
            let word_idx = pk >> 6;
            if word_idx >= bitmap_len {
                continue;
            }
            let word = unsafe { *bitmap.get_unchecked(word_idx) };
            if word & (1u64 << (pk & 63)) == 0 {
                continue;
            }

            // Qualifying row — accumulate into SoA histograms
            let idx = unsafe { *pk_to_idx.get_unchecked(pk) } as usize;
            let qty_raw = unsafe { *qty_values.get_unchecked(i) } as i64;
            let qty_int = (qty_raw / 100) as usize; // Decimal(15,2) → integer
            let price_raw = unsafe { *price_values.get_unchecked(i) } as i64;

            unsafe {
                *accumulators.counts.get_unchecked_mut(idx) += 1;
                *accumulators.sum_qty_raws.get_unchecked_mut(idx) += qty_raw;
                let hist = accumulators.price_by_qty.get_unchecked_mut(idx);
                *hist.get_unchecked_mut(qty_int) += price_raw;
            }
        }
    }
}
