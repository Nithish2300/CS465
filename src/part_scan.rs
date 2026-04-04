/// Phase 1: Scan the part table and build a bitmap of qualifying partkeys.
///
/// TPC-H Q17 filters parts by:
///   p_brand = 'Brand#23' AND p_container = 'MED BOX'
///
/// Only ~0.1% of parts match these predicates. We exploit this by building:
///   1. A **bitmap** over the dense partkey space (1..N) for O(1) probe in Phase 2.
///   2. A **pk_to_idx** mapping from partkey → dense accumulator index.
///   3. **min/max qualifying partkeys** for row-group skipping in Phase 2.
///
/// Column projection: reads only 3 of 9 part columns (p_partkey, p_brand, p_container).
///
/// **mmap I/O**: The file is memory-mapped via `memmap2`, then wrapped in a
/// zero-copy `bytes::Bytes` handle. The Parquet reader slices column chunks
/// directly from the mapped region — no userspace buffer copies, and the OS
/// pages in only the data that is actually accessed.
///
/// The maximum partkey value is determined from Parquet row-group statistics
/// (zone maps) to size the bitmap correctly without a preliminary scan.

use arrow::array::{Array, Int64Array, StringArray};
use bytes::Bytes;
use memmap2::Mmap;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use std::fs::File;
use std::path::Path;

use crate::types::PartScanResult;

pub fn scan_part(path: &Path) -> PartScanResult {
    // -----------------------------------------------------------------------
    // mmap the file for zero-copy Parquet reads
    // -----------------------------------------------------------------------
    let file = File::open(path).expect("Failed to open part.parquet");
    let mmap = unsafe { Mmap::map(&file).expect("Failed to mmap part.parquet") };
    let bytes = Bytes::from_owner(mmap);

    let builder =
        ParquetRecordBatchReaderBuilder::try_new(bytes).expect("Failed to read parquet metadata");

    let metadata = builder.metadata();
    let parquet_schema = builder.parquet_schema();
    let arrow_schema = builder.schema();

    // -----------------------------------------------------------------------
    // Determine max partkey from row-group statistics (zone maps)
    // This avoids an extra scan and lets us allocate the bitmap up front.
    // -----------------------------------------------------------------------
    let pk_parquet_idx = parquet_schema
        .columns()
        .iter()
        .position(|c| c.name() == "p_partkey")
        .expect("p_partkey column not found");

    let mut max_pk: i64 = 0;
    for rg in 0..metadata.num_row_groups() {
        let rg_meta = metadata.row_group(rg);
        if let Some(stats) = rg_meta.column(pk_parquet_idx).statistics() {
            if let parquet::file::statistics::Statistics::Int64(ref s) = stats {
                if let Some(&v) = s.max_opt() {
                    max_pk = max_pk.max(v);
                }
            }
        }
    }
    // Fallback: estimate based on total row count
    if max_pk == 0 {
        max_pk = metadata
            .row_groups()
            .iter()
            .map(|rg| rg.num_rows())
            .sum::<i64>()
            * 10;
    }

    let max_pk = max_pk as usize;
    let mut bitmap = vec![0u64; (max_pk + 64) / 64];
    let mut pk_to_idx = vec![u32::MAX; max_pk + 1];
    let mut num_qualifying: u32 = 0;
    let mut min_qualifying_pk: i64 = i64::MAX;
    let mut max_qualifying_pk: i64 = 0;

    // -----------------------------------------------------------------------
    // Set up column projection — read only the 3 columns we need
    // -----------------------------------------------------------------------
    let pk_arrow_idx = arrow_schema.index_of("p_partkey").unwrap();
    let brand_arrow_idx = arrow_schema.index_of("p_brand").unwrap();
    let container_arrow_idx = arrow_schema.index_of("p_container").unwrap();

    let mask = ProjectionMask::roots(
        parquet_schema,
        [pk_arrow_idx, brand_arrow_idx, container_arrow_idx],
    );

    let reader = builder
        .with_projection(mask)
        .with_batch_size(8192)
        .build()
        .expect("Failed to build part reader");

    // -----------------------------------------------------------------------
    // Scan batches: filter by brand + container, populate bitmap and index map
    // -----------------------------------------------------------------------
    for batch in reader {
        let batch = batch.expect("Failed to read part batch");
        let projected_schema = batch.schema();

        let pk_col_idx = projected_schema.index_of("p_partkey").unwrap();
        let brand_col_idx = projected_schema.index_of("p_brand").unwrap();
        let container_col_idx = projected_schema.index_of("p_container").unwrap();

        let pk_col = batch
            .column(pk_col_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("p_partkey must be Int64");
        let brand_col = batch
            .column(brand_col_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("p_brand must be String");
        let container_col = batch
            .column(container_col_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("p_container must be String");

        for i in 0..batch.num_rows() {
            let brand = brand_col.value(i);
            let container = container_col.value(i);
            if brand == "Brand#23" && container == "MED BOX" {
                let pk = pk_col.value(i);
                let pk_usize = pk as usize;
                bitmap[pk_usize >> 6] |= 1u64 << (pk_usize & 63);
                pk_to_idx[pk_usize] = num_qualifying;
                num_qualifying += 1;
                min_qualifying_pk = min_qualifying_pk.min(pk);
                max_qualifying_pk = max_qualifying_pk.max(pk);
            }
        }
    }

    PartScanResult {
        bitmap,
        pk_to_idx,
        num_qualifying: num_qualifying as usize,
        min_qualifying_pk,
        max_qualifying_pk,
    }
}
