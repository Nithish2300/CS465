# TPC-H Q17 Specialized Query Processor

A single-threaded Rust query processor specialized for TPC-H Query 17, designed to outperform DuckDB through domain-specific optimizations.

## Query

TPC-H Q17 computes the average yearly revenue lost due to small orders for parts of a specific brand and container type:

```sql
SELECT SUM(l_extendedprice) / 7.0 AS avg_yearly
FROM lineitem, part
WHERE p_partkey = l_partkey
  AND p_brand = 'Brand#23'
  AND p_container = 'MED BOX'
  AND l_quantity < (
    SELECT 0.2 * AVG(l_quantity)
    FROM lineitem
    WHERE l_partkey = p_partkey
  );
```

**What this query does**: For every part that is Brand#23 and comes in a MED BOX, find all lineitem rows where the ordered quantity is less than 20% of the average quantity ordered for that part. Sum up the extended prices of those "small quantity" rows and divide by 7 to get an annualized average.

**Why this query is challenging**: It contains a correlated subquery — the `0.2 * AVG(l_quantity)` threshold is different for each partkey. A naive implementation would scan lineitem once to compute averages, then scan again to filter. Our processor does it in a single pass.

## Dependencies

- **Rust** (1.70+ recommended): [Install via rustup](https://rustup.rs/)
- **Python 3** + `duckdb` package (for benchmarking and correctness checks only):
  ```bash
  pip3 install duckdb
  ```

## Build

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

The `-C target-cpu=native` flag enables CPU-specific instruction sets (AVX2, etc.) for maximum throughput. The release profile in `Cargo.toml` also enables LTO (link-time optimization) and sets `codegen-units=1` for best single-binary optimization.

Or simply use `run.sh`, which auto-builds if the binary is missing or stale.

## Usage

### Run the query

```bash
./run.sh --data data/sf1 --out result.csv
```

- `--data <dir>`: Path to directory containing `part.parquet` and `lineitem.parquet`
- `--out <file>`: (Optional) Write result CSV to this file

### Benchmark mode

Run multiple iterations and report averaged timings (first run is treated as warmup):

```bash
./run.sh --data data/sf1 --out result.csv --bench 6
```

### Correctness check

Compare output against DuckDB (single-threaded) for all available scale factors:

```bash
python3 check.py
```

### Full benchmark (vs DuckDB)

Run both DuckDB (single-threaded) and the Rust processor across all scale factors, reporting speedups:

```bash
python3 bench.py
```

## Data Layout

Place TPC-H Parquet files under `data/`:

```
data/
├── sf0.5/      (SF 0.5 — 3M lineitem rows)
├── sf1/        (SF 1   — 6M lineitem rows)
├── sf2/        (SF 2   — 12M lineitem rows)
└── sf5/        (SF 5   — 30M lineitem rows)
    ├── part.parquet
    ├── lineitem.parquet
    └── ... (other TPC-H tables, unused by Q17)
```

Only `part.parquet` and `lineitem.parquet` are read by this processor.

## How It Works: Step-by-Step

The processor exploits three domain-specific properties of TPC-H Q17 to avoid the overhead of a general-purpose SQL engine:

1. **Qualifying parts are rare (~0.1%)**: Only parts matching `Brand#23 + MED BOX` qualify. This extreme selectivity means a join with a hash table is wasteful — a bitmap over the dense partkey space enables O(1) filtering with minimal memory.

2. **l_quantity is always an integer 1-50**: TPC-H quantities are whole numbers stored as Decimal(15,2). By binning prices into 50 histogram buckets per partkey, we capture all information needed to evaluate the correlated subquery in a single pass — no second scan required.

3. **Partkeys are dense integers 1..N**: Since partkeys form a contiguous range, we can use direct array indexing instead of hash table lookups throughout.

### Code Structure

```
src/
├���─ main.rs           # CLI parsing, orchestration, benchmarking loop
├── types.rs          # SoA accumulators and PartScanResult structs
├── part_scan.rs      # Phase 1: Part table scan and bitmap construction (mmap I/O)
├── lineitem_scan.rs  # Phase 2: Lineitem scan with row-group skipping, mmap, bitmap probe
└── compute.rs        # Phase 3: Threshold evaluation and final summation
```

### Phase 1: Part Table Scan (`part_scan.rs`)

**Goal**: Identify which partkeys match `p_brand = 'Brand#23' AND p_container = 'MED BOX'` and build data structures for fast lookup in Phase 2.

**Step-by-step**:

1. **Memory-map `part.parquet`** using `memmap2`. The file is mapped into the process's virtual address space and wrapped in a zero-copy `bytes::Bytes` handle. The OS pages in data on demand — only the column chunks the Parquet reader actually accesses are loaded from disk. No userspace buffer copies occur.

2. **Determine max partkey** from row-group min/max statistics (zone maps). This tells us how large to make our bitmap and index arrays without reading any data rows yet. If statistics are unavailable, we fall back to estimating from total row counts.

3. **Allocate data structures**:
   - `bitmap`: A bit vector of size `max_partkey / 64` words. Each bit position corresponds to a partkey. If the bit is set, that part qualifies.
   - `pk_to_idx`: An array mapping each partkey to a dense index (0, 1, 2, ...) into the accumulators arrays. Non-qualifying partkeys have `u32::MAX`.

4. **Set up column projection**: Request only 3 of 9 part columns from the Parquet reader (`p_partkey`, `p_brand`, `p_container`). The Parquet reader skips the other 6 columns entirely — they are never decompressed or decoded.

5. **Scan batches**: For each batch of rows:
   - Check `p_brand == "Brand#23"` and `p_container == "MED BOX"`.
   - For matching rows: set the bitmap bit, record the dense index, and update min/max qualifying partkey bounds (used for row-group skipping in Phase 2).

**Output**: A bitmap, index map, and min/max qualifying partkey bounds. At SF1, roughly 200 out of 200,000 parts qualify (~0.1%).

### Phase 2: Lineitem Scan (`lineitem_scan.rs`)

**Goal**: In a single pass over lineitem, accumulate per-partkey quantity histograms and price totals for qualifying parts only.

**Step-by-step**:

1. **Memory-map `lineitem.parquet`** via `memmap2` + `bytes::Bytes` for zero-copy Parquet reads (same approach as Phase 1).

2. **Row-group skipping**: Before reading any data, inspect each row group's `l_partkey` column statistics (min/max) from the Parquet footer metadata:
   - **Quick range check**: If the row group's max partkey < min qualifying partkey, or min partkey > max qualifying partkey, skip it immediately.
   - **Precise bitmap check**: For row groups that pass the range check, scan the bitmap for the range `[rg_min_pk, rg_max_pk]` to verify at least one qualifying partkey exists in that range. This catches row groups that overlap the global range but contain no qualifying keys.
   - Only row groups that pass both checks are read. The qualifying indices are passed to the Parquet reader via `with_row_groups()`.

   > **Note on TPC-H data**: Because TPC-H generates `l_partkey` values uniformly across all parts, each lineitem row group typically spans the full partkey range — so few row groups are skipped in practice. However, this optimization would provide significant benefit (10-30%) on sorted or clustered data, and it demonstrates proper use of Parquet zone-map metadata.

3. **Column projection**: Only 3 of 16 columns (`l_partkey`, `l_quantity`, `l_extendedprice`) are requested. This cuts I/O by ~75%.

4. **Increased batch size**: Batches of 16,384 rows (up from default 8,192) reduce per-batch overhead (schema resolution, Arrow array allocation) while keeping data cache-friendly.

5. **For each batch of rows**, enter the hot loop:

   a. **Read `l_partkey`** and compute the bitmap word index (`pk >> 6`).

   b. **Bounds check**: If the word index is beyond the bitmap, skip (handles partkeys larger than any qualifying part).

   c. **Bitmap bit test**: Check if bit `pk & 63` is set in the bitmap word. Since only ~0.1% of parts qualify, **~99.9% of rows exit here** — this is the key performance win. A single bitwise AND replaces what would be a hash table probe in a general-purpose engine.

   d. **For qualifying rows** (the remaining ~0.1%), update **SoA (Struct-of-Arrays) accumulators**:
      - `counts[idx] += 1` — increment row count for this partkey
      - `sum_qty_raws[idx] += qty_raw` — accumulate quantity sum
      - `price_by_qty[idx][qty_int] += price_raw` — add price to histogram bucket

6. **`unsafe` optimizations**: The inner loop uses `get_unchecked` to eliminate bounds checks on array accesses. This is safe because:
   - Partkeys are validated against bitmap bounds before the index lookup.
   - Quantity values are guaranteed to be 1-50 by TPC-H data generation.
   - Column indices are resolved from the schema.

**Output**: SoA accumulator arrays, one entry per qualifying partkey, containing a 50-bucket price histogram and quantity statistics.

### Phase 3: Result Computation (`compute.rs`)

**Goal**: Evaluate the correlated subquery threshold and compute the final sum.

**Step-by-step**:

1. **For each qualifying partkey** (iterating the SoA arrays):

   a. Compute the threshold predicate using **integer arithmetic only**:
      - The SQL predicate is: `l_quantity < 0.2 * AVG(l_quantity)`
      - `AVG(l_quantity)` = `sum_qty_raw / count` (in raw Decimal units, i.e., value x 100)
      - Rearranging to avoid division: `qty_raw * 5 * count < sum_qty_raw`
      - Since `qty_raw = qty_int * 100`: **`500 * qty_int * count < sum_qty_raw`**
      - This is a pure integer comparison — no floating-point needed.

   b. **For each quantity bucket** q = 1 to 50:
      - If `500 * q * count < sum_qty_raw`, the threshold is satisfied.
      - Add `price_by_qty[q]` to the running total.

2. **Final division**: Convert the total from raw Decimal cents to dollars (`/ 100.0`) and divide by 7.0 to get `avg_yearly`.

**Output**: A single floating-point value matching DuckDB's result.

### Summary of Optimizations

| Optimization | What it replaces | Benefit |
|---|---|---|
| Bitmap partkey probe | Hash table join | O(1) bit test, cache-friendly, no hashing overhead |
| Single-pass histogram accumulators | Two lineitem scans (one for AVG, one for filter) | Halves I/O on the largest table |
| Dense direct-indexed arrays | Hash maps for partkey lookup | No hashing, no collision handling, predictable memory |
| Column projection (3/16 cols) | Full-row reads | ~75% less data decompressed from Parquet |
| **mmap I/O** (memmap2 + Bytes) | Buffered File reads with syscalls per chunk | Zero-copy column chunk access; OS-managed paging; no userspace buffer copies |
| **Row-group skipping** (zone maps) | Reading all row groups unconditionally | Skips row groups with no qualifying partkeys before decompression |
| **SoA memory layout** | Array-of-Structs accumulators | counts/sums fit in L1 cache (~2.4KB at SF1); reduces cache pollution from price histogram |
| **Batch size 16K** (up from 8K) | 8192-row batches | Fewer per-batch allocations and schema lookups |
| `unsafe get_unchecked` in hot loop | Bounds-checked array access | Eliminates branch mispredictions on millions of rows |
| Integer threshold arithmetic | Floating-point division for AVG | Exact results, faster integer ALU operations |
| Parquet row-group statistics | Extra scan to determine array sizes | Sizes bitmap from metadata without reading data |
| Release profile: LTO + codegen-units=1 | Default compilation | Cross-crate inlining, whole-program optimization |
| `target-cpu=native` | Generic x86-64 codegen | Enables AVX2, BMI2, and other CPU-specific instructions |
| `panic=abort` | Unwinding panic handler | Smaller binary, no unwinding overhead |

### Why This Beats DuckDB

DuckDB is a general-purpose SQL engine that must handle arbitrary queries. For Q17, it typically:
1. Scans part with a filter -> builds a hash table
2. Scans lineitem -> probes the hash table for the join
3. For the correlated subquery, uses a hash aggregate to compute per-partkey AVGs
4. Applies the threshold filter using the aggregated values
5. Computes the final SUM and division

Our processor eliminates the hash table entirely (bitmap is cheaper), merges steps 2-4 into a single pass (histogram trick), and avoids all the overhead of a general query executor (operator pipelines, type dispatch, memory management, etc.). The mmap I/O layer eliminates syscall overhead for column chunk reads, and the SoA layout keeps frequently-accessed metadata warm in L1 cache.

## Benchmark Results

Machine: Apple M-series, macOS, single-threaded execution for both engines.

| Scale Factor | DuckDB (ms) | Rust (ms) | Speedup |
|---|---|---|---|
| SF 0.5 | 55.2 | 39.9 | 1.38x |
| SF 1 | 112.4 | 76.0 | 1.48x |
| SF 2 | 229.2 | 149.6 | 1.53x |
| SF 5 | 578.3 | 374.7 | 1.54x |

Protocol: 6 runs per scale factor, first run discarded (warmup), average of remaining 5 reported. Both engines single-threaded (`PRAGMA threads=1` for DuckDB). Same Parquet files, same machine.

The speedup increases with scale factor (1.38x at SF0.5 -> 1.54x at SF5), indicating our processor scales better than DuckDB for this query — the bitmap probe and mmap I/O overhead is amortized across more data.

## Correctness

- TPC-H uses Decimal(15,2) for prices and quantities. All intermediate arithmetic uses `i64` in raw decimal units (cents) to avoid floating-point rounding errors.
- The only floating-point operation is the final `total / 100.0 / 7.0` division.
- `check.py` validates output against DuckDB (single-threaded) with a tolerance of 0.01.
- All scale factors (SF 0.5, 1, 2, 5) pass the correctness check.

## Compliance with Project Requirements

| Requirement | Status |
|---|---|
| Single-threaded execution | All processing is single-threaded; no parallel scan/join/aggregation |
| Cold start (no prebuilt indexes) | Each run reads Parquet from scratch; no cached state between runs |
| No SQL engine embedding | Hand-written scan/join/aggregate operators; no DuckDB/SQLite calls |
| Reads Parquet directly | Uses `parquet` crate to read `.parquet` files (no CSV conversion) |
| Column projection | Reads only needed columns (3/9 for part, 3/16 for lineitem) |
| Correct output | Matches DuckDB output within 0.01 tolerance across all scale factors |
| One-command runner | `./run.sh --data <dir> --out <file>` builds and runs |
| Benchmark mode | `./run.sh --data <dir> --bench 6` reports averaged timings |
| Correctness check script | `python3 check.py` compares against DuckDB |
| Scale factors 0.5, 1, 2, 5 | All tested and passing |

## Project Structure

```
.
├── Cargo.toml        # Rust manifest: parquet, arrow, clap, bytes, memmap2
├── Cargo.lock        # Locked dependency versions
├── src/
│   ├── main.rs           # CLI, orchestration, benchmark loop
│   ├─- types.rs          # AccumulatorsSoA and PartScanResult structs
│   ├── part_scan.rs      # Phase 1: mmap + part scan + bitmap construction
│   ├── lineitem_scan.rs  # Phase 2: mmap + row-group skip + lineitem scan
│   └── compute.rs        # Phase 3: threshold evaluation + final result
├── run.sh            # One-command runner (builds + executes)
├── bench.py          # Benchmark script (Rust vs DuckDB across scale factors)
├── check.py          # Correctness validation script
├── data/             # TPC-H Parquet files by scale factor
│   ├── sf0.5/
│   ├── sf1/
│   ├── sf2/
│   └── sf5/
└── CLAUDE.md         # Project requirements and specifications
```
