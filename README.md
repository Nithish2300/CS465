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
- **Python 3** (3.8+) with the following packages (for benchmarking, plotting, and correctness checks):
  - `duckdb` — DuckDB baseline for benchmarking and correctness validation
  - `matplotlib` — benchmark plot generation (`benchmark_results.png`)

Install all Python dependencies at once:
```bash
pip3 install duckdb matplotlib
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

- `--data <dir>`: Path to directory containing `part.parquet` and `lineitem.parquet`/lo
- `--out <file>`: (Optional) Write result CSV to this file

### Benchmark mode

Run multiple iterations and report averaged timings (first run is treated as warmup):

```bash
./run.sh --data data/sf1 --out result.csv --bench 5
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
v
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

## Project Structure

```
.
├── Cargo.toml        # Rust manifest: parquet, arrow, clap, bytes, memmap2
├── Cargo.lock        # Locked dependency versions
├── src/
│   ├── main.rs           # CLI, orchestration, benchmark loop
│   ├── types.rs          # AccumulatorsSoA and PartScanResult structs
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
```
