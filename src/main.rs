//! TPC-H Q17 Specialized Query Processor
//!
//! A single-threaded Rust query processor that computes TPC-H Query 17
//! by exploiting domain-specific properties of the data to beat DuckDB.
//!
//! # Three-phase execution
//!
//! - **Phase 1** (`part_scan`): Scan part.parquet via mmap, filter Brand#23 +
//!   MED BOX, build a bitmap and dense index map of qualifying partkeys.
//! - **Phase 2** (`lineitem_scan`): Single-pass scan of lineitem.parquet via
//!   mmap, with row-group skipping based on Parquet zone-map statistics.
//!   Uses the bitmap for O(1) join probing. Qualifying rows are accumulated
//!   into SoA (Struct-of-Arrays) per-partkey quantity histograms.
//! - **Phase 3** (`compute`): Evaluate the correlated subquery threshold
//!   (0.2 * AVG) using integer arithmetic on the histograms, sum qualifying
//!   prices, and divide by 7.0.

mod compute;
mod lineitem_scan;
mod part_scan;
mod types;

use clap::Parser;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use crate::types::AccumulatorsSoA;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "q17", about = "TPC-H Q17 specialized processor")]
struct Cli {
    /// Path to the data directory containing part.parquet and lineitem.parquet
    #[arg(long)]
    data: String,

    /// Output file path for the result CSV
    #[arg(long)]
    out: Option<String>,

    /// Benchmark mode: number of runs (reports averaged timings to stderr)
    #[arg(long)]
    bench: Option<usize>,
}

// ---------------------------------------------------------------------------
// Run one complete query execution (Phase 1 → Phase 2 → Phase 3)
// ---------------------------------------------------------------------------

fn run_query(data_dir: &Path) -> (Option<f64>, std::time::Duration) {
    let t0 = Instant::now();

    // Phase 1: Scan part table → qualifying bitmap + index map
    let part_path = data_dir.join("part.parquet");
    let part_result = part_scan::scan_part(&part_path);

    if part_result.num_qualifying == 0 {
        return (None, t0.elapsed());
    }

    // Phase 2: Single-pass lineitem scan → SoA histogram accumulators
    let mut accumulators = AccumulatorsSoA::new(part_result.num_qualifying);
    let lineitem_path = data_dir.join("lineitem.parquet");
    lineitem_scan::scan_lineitem(
        &lineitem_path,
        &part_result.bitmap,
        &part_result.pk_to_idx,
        part_result.min_qualifying_pk,
        part_result.max_qualifying_pk,
        &mut accumulators,
    );

    // Phase 3: Compute final result from accumulators
    let result = compute::compute_result(&accumulators);
    (result, t0.elapsed())
}

// ---------------------------------------------------------------------------
// Output formatting and file writing
// ---------------------------------------------------------------------------

fn format_result(result: Option<f64>) -> String {
    match result {
        Some(val) => format!("avg_yearly\n{:.2}", val),
        None => "avg_yearly\n".to_string(),
    }
}

fn write_output(output: &str, out_path: &str) {
    let mut f = File::create(out_path).expect("Failed to create output file");
    writeln!(f, "{}", output).expect("Failed to write output");
}

// ---------------------------------------------------------------------------
// Benchmark reporting
// ---------------------------------------------------------------------------

fn report_benchmark(times: &[std::time::Duration], bench_runs: usize) {
    if times.len() > 1 {
        let measured: std::time::Duration = times[1..].iter().sum();
        let measured_avg = measured / (bench_runs - 1) as u32;
        eprintln!(
            "\nBenchmark: {} runs, avg (excl. warmup): {:?}",
            bench_runs, measured_avg
        );
    }
    let total: std::time::Duration = times.iter().sum();
    eprintln!("Overall avg: {:?}", total / bench_runs as u32);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    let data_dir = Path::new(&cli.data);

    let bench_runs = cli.bench.unwrap_or(1);

    let mut result: Option<f64> = None;
    let mut times = Vec::with_capacity(bench_runs);

    for run in 0..bench_runs {
        let (r, elapsed) = run_query(data_dir);
        if run == 0 {
            result = r;
        }
        times.push(elapsed);
        if bench_runs > 1 {
            eprintln!("Run {}: {:?}", run + 1, elapsed);
        }
    }

    // Output result
    let output = format_result(result);
    println!("{}", output);

    if let Some(ref out_path) = cli.out {
        write_output(&output, out_path);
    }

    // Report timing
    if bench_runs > 1 {
        report_benchmark(&times, bench_runs);
    } else {
        eprintln!("Time: {:?}", times[0]);
    }
}
