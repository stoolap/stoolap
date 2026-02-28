// Copyright 2026 Stoolap Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Self-contained ANN benchmark for Stoolap HNSW vector search.
//!
//! Downloads the Fashion-MNIST dataset, computes exact ground truth,
//! inserts vectors into Stoolap, and measures recall + throughput
//! through the full SQL query path.
//!
//! ```bash
//! # Quick demo (16 configs: 4 m-values x 4 ef-values)
//! cargo run --release --example ann_benchmark --features ann-benchmark
//!
//! # Specific configuration
//! cargo run --release --example ann_benchmark --features ann-benchmark -- --m 16 --ef-search 32
//!
//! # Full parameter sweep with CSV output
//! cargo run --release --example ann_benchmark --features ann-benchmark -- --sweep --csv sweep.csv
//! ```

use anyhow::{anyhow, Context, Result};
use flate2::read::GzDecoder;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;
use stoolap::api::Database;

// ─── Constants ─────────────────────────────────────────────────────────

const FASHION_MNIST_TRAIN_URL: &str =
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz";
const FASHION_MNIST_TEST_URL: &str =
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz";

const CACHE_DIR_NAME: &str = "stoolap-ann-benchmark";
const DATASET_SUBDIR: &str = "fashion-mnist-784-euclidean";

const DEFAULT_MAX_QUERIES: usize = 300;
const DEFAULT_K: usize = 10;
const DEFAULT_RUNS: usize = 3;
// Match ann-benchmarks.com hnswlib default (ef_construction=500)
const DEFAULT_EF_CONSTRUCTION: usize = 500;
const WARMUP_QUERIES: usize = 10;
const INSERT_BATCH_SIZE: usize = 1000;

const DEFAULT_M_VALUES: &[usize] = &[12, 16, 24, 48];
const DEFAULT_EF_VALUES: &[usize] = &[10, 40, 120, 400];
// Exact ann-benchmarks.com hnswlib parameters
const SWEEP_M_VALUES: &[usize] = &[4, 8, 12, 16, 24, 36, 48, 64, 96];
const SWEEP_EF_VALUES: &[usize] = &[10, 20, 40, 80, 120, 200, 400, 600, 800];

const CSV_HEADER: &str = "dataset,base_count,dims,query_count,k,m,ef_search,build_time_s,\
                          build_vecps,bf_mean_ms,bf_p50_ms,bf_p95_ms,bf_p99_ms,bf_qps,\
                          bf_recall_pct,hnsw_mean_ms,hnsw_p50_ms,hnsw_p95_ms,hnsw_p99_ms,\
                          hnsw_qps,hnsw_recall_pct,speedup_x";

// ─── CLI ───────────────────────────────────────────────────────────────

struct Config {
    dataset_name: String,
    cache_dir: PathBuf,
    max_queries: usize,
    k: usize,
    runs: usize,
    ef_construction: usize,
    m: Option<usize>,
    ef_search: Option<usize>,
    sweep: bool,
    csv_output: Option<PathBuf>,
    skip_brute_force: bool,
}

fn print_help() {
    println!("Stoolap ANN Benchmark");
    println!();
    println!("Self-contained benchmark that downloads Fashion-MNIST, computes ground");
    println!("truth, and measures HNSW recall + throughput through the SQL query path.");
    println!();
    println!("Usage:");
    println!("  cargo run --release --example ann_benchmark --features ann-benchmark -- [OPTIONS]");
    println!();
    println!("Modes:");
    println!("  (default)     Run 16 configs (m=12,16,24,48 x ef_search=10,40,120,400)");
    println!("  --m N --ef-search N   Run a single configuration");
    println!("  --sweep               Full 81-config sweep (9 m-values x 9 ef_search values)");
    println!();
    println!("Options:");
    println!("  --max-queries N       Query count limit (default: {DEFAULT_MAX_QUERIES})");
    println!("  --k N                 Recall@k and LIMIT k (default: {DEFAULT_K})");
    println!("  --runs N              Best-of-N runs per config (default: {DEFAULT_RUNS})");
    println!("  --ef-construction N   HNSW build quality (default: {DEFAULT_EF_CONSTRUCTION})");
    println!("  --skip-brute-force    Skip brute-force baseline (saves ~20 min per run)");
    println!("  --csv PATH            Write CSV results to file");
    println!("  --cache-dir PATH      Dataset cache directory (default: $TMPDIR/{CACHE_DIR_NAME})");
    println!("  -h, --help            Show this help");
}

fn parse_args() -> Result<Config> {
    let args: Vec<String> = std::env::args().collect();
    let mut config = Config {
        dataset_name: String::from("fashion-mnist-784-euclidean"),
        cache_dir: std::env::temp_dir()
            .join(CACHE_DIR_NAME)
            .join(DATASET_SUBDIR),
        max_queries: DEFAULT_MAX_QUERIES,
        k: DEFAULT_K,
        runs: DEFAULT_RUNS,
        ef_construction: DEFAULT_EF_CONSTRUCTION,
        m: None,
        ef_search: None,
        sweep: false,
        csv_output: None,
        skip_brute_force: false,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "--sweep" => {
                config.sweep = true;
                i += 1;
            }
            "--skip-brute-force" => {
                config.skip_brute_force = true;
                i += 1;
            }
            flag @ ("--m" | "--ef-search" | "--max-queries" | "--k" | "--runs"
            | "--ef-construction" | "--csv" | "--cache-dir") => {
                if i + 1 >= args.len() {
                    return Err(anyhow!("{flag} requires a value"));
                }
                let val = &args[i + 1];
                match flag {
                    "--m" => config.m = Some(val.parse().context("invalid --m")?),
                    "--ef-search" => {
                        config.ef_search = Some(val.parse().context("invalid --ef-search")?)
                    }
                    "--max-queries" => {
                        config.max_queries = val.parse().context("invalid --max-queries")?
                    }
                    "--k" => config.k = val.parse().context("invalid --k")?,
                    "--runs" => config.runs = val.parse().context("invalid --runs")?,
                    "--ef-construction" => {
                        config.ef_construction = val.parse().context("invalid --ef-construction")?
                    }
                    "--csv" => config.csv_output = Some(PathBuf::from(val)),
                    "--cache-dir" => config.cache_dir = PathBuf::from(val),
                    _ => unreachable!(),
                }
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
    if config.runs == 0 {
        config.runs = 1;
    }
    Ok(config)
}

// ─── Download ──────────────────────────────────────────────────────────

fn download_gz(url: &str) -> Result<Vec<u8>> {
    eprintln!("  Downloading {url}");
    let response = ureq::get(url)
        .call()
        .map_err(|e| anyhow!("HTTP request failed for {url}: {e}"))?;

    let mut compressed = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut compressed)
        .context("reading response body")?;

    let mut decoder = GzDecoder::new(&compressed[..]);
    let mut decompressed = Vec::new();
    decoder
        .read_to_end(&mut decompressed)
        .context("gzip decompression")?;

    eprintln!(
        "  Downloaded {:.1} MB, decompressed to {:.1} MB",
        compressed.len() as f64 / 1_048_576.0,
        decompressed.len() as f64 / 1_048_576.0,
    );
    Ok(decompressed)
}

fn ensure_cached(cache_dir: &Path, filename: &str, url: &str) -> Result<Vec<u8>> {
    let cached_path = cache_dir.join(filename);
    if cached_path.exists() {
        eprintln!("  Using cached: {}", cached_path.display());
        return fs::read(&cached_path)
            .with_context(|| format!("reading {}", cached_path.display()));
    }
    fs::create_dir_all(cache_dir)
        .with_context(|| format!("creating cache dir: {}", cache_dir.display()))?;
    let data = download_gz(url)?;
    fs::write(&cached_path, &data)
        .with_context(|| format!("caching to {}", cached_path.display()))?;
    Ok(data)
}

// ─── IDX Format Parser ────────────────────────────────────────────────

fn parse_idx3_images(data: &[u8]) -> Result<(Vec<Vec<f32>>, usize)> {
    if data.len() < 16 {
        return Err(anyhow!("IDX file too short ({} bytes)", data.len()));
    }
    let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    if magic != 0x0000_0803 {
        return Err(anyhow!(
            "invalid IDX3 magic: 0x{magic:08x} (expected 0x00000803)"
        ));
    }
    let count = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let rows = u32::from_be_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let cols = u32::from_be_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let dims = rows * cols;

    let payload = &data[16..];
    let expected = count * dims;
    if payload.len() < expected {
        return Err(anyhow!(
            "IDX payload too short: {} bytes for {count}x{dims} (need {expected})",
            payload.len()
        ));
    }

    let mut vectors = Vec::with_capacity(count);
    for i in 0..count {
        let offset = i * dims;
        let mut vec = Vec::with_capacity(dims);
        for &byte in &payload[offset..offset + dims] {
            vec.push(byte as f32);
        }
        vectors.push(vec);
    }
    Ok((vectors, dims))
}

// ─── Ground Truth Computation ──────────────────────────────────────────

#[inline(always)]
fn l2_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        let d0 = a[base] - b[base];
        let d1 = a[base + 1] - b[base + 1];
        let d2 = a[base + 2] - b[base + 2];
        let d3 = a[base + 3] - b[base + 3];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    let base = chunks * 4;
    for i in 0..remainder {
        let d = a[base + i] - b[base + i];
        sum += d * d;
    }
    sum
}

fn compute_ground_truth(base: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
    queries
        .par_iter()
        .map(|query| {
            let mut distances: Vec<(usize, f32)> = base
                .iter()
                .enumerate()
                .map(|(id, vec)| (id, l2_distance_sq(query, vec)))
                .collect();

            let k_clamped = k.min(distances.len());
            distances.select_nth_unstable_by(k_clamped - 1, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
            });

            let mut top_k: Vec<(usize, f32)> = distances[..k_clamped].to_vec();
            top_k.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            top_k.iter().map(|(id, _)| *id).collect()
        })
        .collect()
}

// ─── SQL Helpers ───────────────────────────────────────────────────────

fn vec_to_sql_literal(v: &[f32]) -> String {
    let inner: Vec<String> = v.iter().map(|x| format!("{x:.6}")).collect();
    format!("[{}]", inner.join(","))
}

fn run_knn_query(db: &Database, sql: &str) -> Result<(Vec<usize>, f64)> {
    let start = Instant::now();
    let mut ids = Vec::new();
    for row in db.query(sql, ()).context("query failed")? {
        let row = row.context("row decode failed")?;
        let id = row.get::<i64>(0).context("missing id column")? as usize;
        ids.push(id);
    }
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok((ids, ms))
}

fn batch_insert(db: &Database, vectors: &[Vec<f32>]) -> Result<()> {
    let mut inserted = 0usize;
    while inserted < vectors.len() {
        let end = (inserted + INSERT_BATCH_SIZE).min(vectors.len());
        db.execute("BEGIN", ()).context("begin failed")?;
        for (i, vector) in vectors[inserted..end].iter().enumerate() {
            let id = inserted + i;
            let lit = vec_to_sql_literal(vector);
            let sql = format!("INSERT INTO vectors (id, embedding) VALUES ({id}, '{lit}')");
            db.execute(&sql, ()).context("insert failed")?;
        }
        db.execute("COMMIT", ()).context("commit failed")?;
        inserted = end;
    }
    Ok(())
}

// ─── Statistics ────────────────────────────────────────────────────────

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let idx = ((p / 100.0) * (sorted.len().saturating_sub(1)) as f64).round() as usize;
    sorted[idx]
}

fn recall_at_k(ground_truth: &[usize], approx: &[usize]) -> f64 {
    if ground_truth.is_empty() {
        return 0.0;
    }
    let mut hits = 0usize;
    for id in approx {
        if ground_truth.contains(id) {
            hits += 1;
        }
    }
    hits as f64 / ground_truth.len() as f64
}

// ─── Benchmark Results ─────────────────────────────────────────────────

struct BruteForceBaseline {
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    qps: f64,
    recall_pct: f64,
}

struct BenchResult {
    m: usize,
    ef_search: usize,
    build_time_s: f64,
    build_vecps: f64,
    hnsw_mean_ms: f64,
    hnsw_p50_ms: f64,
    hnsw_p95_ms: f64,
    hnsw_p99_ms: f64,
    hnsw_qps: f64,
    hnsw_recall_pct: f64,
    speedup: f64,
}

// ─── Benchmark Runners ─────────────────────────────────────────────────

fn run_brute_force_benchmark(
    db: &Database,
    query_sqls: &[String],
    ground_truth: &[Vec<usize>],
    k: usize,
    runs: usize,
) -> Result<BruteForceBaseline> {
    let warmup_n = query_sqls.len().min(WARMUP_QUERIES);
    eprintln!("  Warming up brute-force ({warmup_n} queries)...");
    for sql in query_sqls.iter().take(warmup_n) {
        let _ = run_knn_query(db, sql)?;
    }

    let mut best_latencies = Vec::new();
    let mut best_recalls = Vec::new();
    let mut best_total = f64::INFINITY;

    for run_idx in 0..runs {
        let mut run_latencies = Vec::with_capacity(query_sqls.len());
        let mut run_recalls = Vec::with_capacity(query_sqls.len());
        for (qi, sql) in query_sqls.iter().enumerate() {
            let (ids, ms) = run_knn_query(db, sql)?;
            run_latencies.push(ms);
            run_recalls.push(recall_at_k(&ground_truth[qi][..k], &ids));
        }
        let run_total: f64 = run_latencies.iter().sum();
        eprintln!(
            "  brute-force run {}/{}: {:.1} QPS",
            run_idx + 1,
            runs,
            query_sqls.len() as f64 / (run_total / 1000.0)
        );
        if run_total < best_total {
            best_total = run_total;
            best_latencies = run_latencies;
            best_recalls = run_recalls;
        }
    }

    Ok(BruteForceBaseline {
        mean_ms: mean(&best_latencies),
        p50_ms: percentile(&best_latencies, 50.0),
        p95_ms: percentile(&best_latencies, 95.0),
        p99_ms: percentile(&best_latencies, 99.0),
        qps: query_sqls.len() as f64 / (best_total / 1000.0),
        recall_pct: mean(&best_recalls) * 100.0,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_hnsw_benchmark(
    db: &Database,
    query_sqls: &[String],
    ground_truth: &[Vec<usize>],
    k: usize,
    runs: usize,
    base_count: usize,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    bf: &BruteForceBaseline,
) -> Result<BenchResult> {
    let build_start = Instant::now();
    db.execute(
        &format!(
            "CREATE INDEX idx_emb ON vectors(embedding) USING HNSW \
             WITH (m = {m}, ef_construction = {ef_construction}, ef_search = {ef_search})"
        ),
        (),
    )
    .context("create hnsw index failed")?;
    let build_s = build_start.elapsed().as_secs_f64();
    let build_vecps = base_count as f64 / build_s.max(1e-9);

    let warmup_n = query_sqls.len().min(WARMUP_QUERIES);
    for sql in query_sqls.iter().take(warmup_n) {
        let _ = run_knn_query(db, sql)?;
    }

    let mut best_latencies = Vec::new();
    let mut best_recalls = Vec::new();
    let mut best_total = f64::INFINITY;

    for run_idx in 0..runs {
        let mut run_latencies = Vec::with_capacity(query_sqls.len());
        let mut run_recalls = Vec::with_capacity(query_sqls.len());
        for (qi, sql) in query_sqls.iter().enumerate() {
            let (ids, ms) = run_knn_query(db, sql)?;
            run_latencies.push(ms);
            run_recalls.push(recall_at_k(&ground_truth[qi][..k], &ids));
        }
        let run_total: f64 = run_latencies.iter().sum();
        eprintln!(
            "    HNSW run {}/{}: {:.1} QPS, recall={:.2}%",
            run_idx + 1,
            runs,
            query_sqls.len() as f64 / (run_total / 1000.0),
            mean(&run_recalls) * 100.0,
        );
        if run_total < best_total {
            best_total = run_total;
            best_latencies = run_latencies;
            best_recalls = run_recalls;
        }
    }

    db.execute("DROP INDEX idx_emb ON vectors", ())
        .context("drop index failed")?;

    let hnsw_qps = query_sqls.len() as f64 / (best_total / 1000.0);
    Ok(BenchResult {
        m,
        ef_search,
        build_time_s: build_s,
        build_vecps,
        hnsw_mean_ms: mean(&best_latencies),
        hnsw_p50_ms: percentile(&best_latencies, 50.0),
        hnsw_p95_ms: percentile(&best_latencies, 95.0),
        hnsw_p99_ms: percentile(&best_latencies, 99.0),
        hnsw_qps,
        hnsw_recall_pct: mean(&best_recalls) * 100.0,
        speedup: hnsw_qps / bf.qps.max(1e-9),
    })
}

// ─── Output ────────────────────────────────────────────────────────────

fn format_csv_row(
    dataset: &str,
    base_count: usize,
    dims: usize,
    query_count: usize,
    k: usize,
    bf: &BruteForceBaseline,
    r: &BenchResult,
) -> String {
    format!(
        "{},{},{},{},{},{},{},{:.6},{:.1},{:.6},{:.6},{:.6},{:.6},{:.1},{:.3},\
         {:.6},{:.6},{:.6},{:.6},{:.1},{:.3},{:.3}",
        dataset,
        base_count,
        dims,
        query_count,
        k,
        r.m,
        r.ef_search,
        r.build_time_s,
        r.build_vecps,
        bf.mean_ms,
        bf.p50_ms,
        bf.p95_ms,
        bf.p99_ms,
        bf.qps,
        bf.recall_pct,
        r.hnsw_mean_ms,
        r.hnsw_p50_ms,
        r.hnsw_p95_ms,
        r.hnsw_p99_ms,
        r.hnsw_qps,
        r.hnsw_recall_pct,
        r.speedup,
    )
}

fn print_scorecard(results: &[BenchResult], bf: &BruteForceBaseline, k: usize) {
    println!();
    println!("Brute-force baseline:");
    println!(
        "  QPS={:.1}  p95={:.3}ms  p99={:.3}ms  recall@{k}={:.2}%",
        bf.qps, bf.p95_ms, bf.p99_ms, bf.recall_pct,
    );
    println!();
    println!(
        "{:>4}  {:>10}  {:>8}  {:>10}  {:>8}  {:>8}  {:>8}",
        "m", "ef_search", "recall%", "QPS", "p95 ms", "p99 ms", "speedup"
    );
    println!("{}", "-".repeat(68));
    for r in results {
        println!(
            "{:>4}  {:>10}  {:>7.2}%  {:>10.1}  {:>7.3}  {:>7.3}  {:>7.1}x",
            r.m,
            r.ef_search,
            r.hnsw_recall_pct,
            r.hnsw_qps,
            r.hnsw_p95_ms,
            r.hnsw_p99_ms,
            r.speedup,
        );
    }
    println!();

    // Best config per recall target
    let targets = [95.0, 99.0, 99.5, 99.9, 100.0];
    println!("Best configuration per recall target:");
    println!(
        "{:>8}  {:>6}  {:>10}  {:>10}  {:>8}  {:>8}",
        "target", "config", "recall%", "QPS", "p95 ms", "speedup"
    );
    println!("{}", "-".repeat(60));
    for target in &targets {
        let best = results
            .iter()
            .filter(|r| r.hnsw_recall_pct >= *target)
            .max_by(|a, b| {
                a.hnsw_qps
                    .partial_cmp(&b.hnsw_qps)
                    .unwrap_or(Ordering::Equal)
            });
        if let Some(r) = best {
            println!(
                ">={:>5.1}%  m={:<2} e={:<3}  {:>7.2}%  {:>10.1}  {:>7.3}  {:>7.1}x",
                target, r.m, r.ef_search, r.hnsw_recall_pct, r.hnsw_qps, r.hnsw_p95_ms, r.speedup,
            );
        } else {
            println!(">={:>5.1}%  (not reached)", target);
        }
    }
    println!();
}

// ─── Main ──────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let config = parse_args()?;

    // 1. Download / load dataset
    eprintln!("Loading Fashion-MNIST dataset...");
    let train_bytes = ensure_cached(
        &config.cache_dir,
        "train-images-idx3-ubyte",
        FASHION_MNIST_TRAIN_URL,
    )?;
    let test_bytes = ensure_cached(
        &config.cache_dir,
        "t10k-images-idx3-ubyte",
        FASHION_MNIST_TEST_URL,
    )?;

    let (base_vectors, dims) = parse_idx3_images(&train_bytes)?;
    let (all_queries, query_dims) = parse_idx3_images(&test_bytes)?;
    if dims != query_dims {
        return Err(anyhow!(
            "dimension mismatch: base={dims}, query={query_dims}"
        ));
    }

    let query_count = config.max_queries.min(all_queries.len());
    let queries: Vec<Vec<f32>> = all_queries.into_iter().take(query_count).collect();

    eprintln!(
        "Dataset: {} | base={} dims={} queries={} k={}",
        config.dataset_name,
        base_vectors.len(),
        dims,
        queries.len(),
        config.k,
    );

    // 2. Compute ground truth
    eprintln!(
        "Computing ground truth ({} queries x {} vectors)...",
        queries.len(),
        base_vectors.len(),
    );
    let gt_start = Instant::now();
    let ground_truth = compute_ground_truth(&base_vectors, &queries, config.k);
    eprintln!(
        "  Ground truth computed in {:.2}s",
        gt_start.elapsed().as_secs_f64()
    );

    // 3. Create database and insert vectors
    let db = Database::open_in_memory().context("failed to open database")?;
    db.execute(
        &format!("CREATE TABLE vectors (id INTEGER PRIMARY KEY, embedding VECTOR({dims}))"),
        (),
    )
    .context("create table failed")?;

    eprintln!("Inserting {} vectors...", base_vectors.len());
    let insert_start = Instant::now();
    batch_insert(&db, &base_vectors)?;
    let insert_s = insert_start.elapsed().as_secs_f64();
    eprintln!(
        "  Insert: {insert_s:.1}s ({:.0} vec/s)",
        base_vectors.len() as f64 / insert_s
    );

    // 4. Pre-build SQL queries
    let query_sqls: Vec<String> = queries
        .iter()
        .map(|q| {
            let lit = vec_to_sql_literal(q);
            format!(
                "SELECT id, VEC_DISTANCE_L2(embedding, '{lit}') AS dist \
                 FROM vectors ORDER BY dist LIMIT {}",
                config.k
            )
        })
        .collect();

    // 5. Brute-force baseline (single run; deterministic scan doesn't benefit from best-of-N)
    let bf = if config.skip_brute_force {
        eprintln!("Skipping brute-force baseline (--skip-brute-force)");
        BruteForceBaseline {
            mean_ms: 0.0,
            p50_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
            qps: 0.0,
            recall_pct: 0.0,
        }
    } else {
        eprintln!("Running brute-force baseline (1 run)...");
        let baseline = run_brute_force_benchmark(&db, &query_sqls, &ground_truth, config.k, 1)?;
        eprintln!(
            "  Brute-force: {:.1} QPS, recall={:.2}%",
            baseline.qps, baseline.recall_pct
        );
        baseline
    };

    // 6. Determine configurations
    let configs: Vec<(usize, usize)> = if config.sweep {
        let mut cfgs = Vec::new();
        for &m in SWEEP_M_VALUES {
            for &ef in SWEEP_EF_VALUES {
                cfgs.push((m, ef));
            }
        }
        cfgs
    } else if let (Some(m), Some(ef)) = (config.m, config.ef_search) {
        vec![(m, ef)]
    } else {
        let m_vals: Vec<usize> = config.m.map_or(DEFAULT_M_VALUES.to_vec(), |m| vec![m]);
        let ef_vals: Vec<usize> = config
            .ef_search
            .map_or(DEFAULT_EF_VALUES.to_vec(), |ef| vec![ef]);
        let mut cfgs = Vec::new();
        for &m in &m_vals {
            for &ef in &ef_vals {
                cfgs.push((m, ef));
            }
        }
        cfgs
    };

    // 7. Run HNSW benchmarks
    let total = configs.len();
    eprintln!("Running {total} HNSW configuration(s)...");
    let mut results = Vec::with_capacity(total);
    for (i, &(m, ef_search)) in configs.iter().enumerate() {
        eprintln!("[{}/{}] m={m} ef_search={ef_search}", i + 1, total);
        let result = run_hnsw_benchmark(
            &db,
            &query_sqls,
            &ground_truth,
            config.k,
            config.runs,
            base_vectors.len(),
            m,
            config.ef_construction,
            ef_search,
            &bf,
        )?;
        results.push(result);
    }

    // 8. Output
    print_scorecard(&results, &bf, config.k);

    if let Some(csv_path) = &config.csv_output {
        let mut csv_content = String::from(CSV_HEADER);
        csv_content.push('\n');
        for r in &results {
            csv_content.push_str(&format_csv_row(
                &config.dataset_name,
                base_vectors.len(),
                dims,
                queries.len(),
                config.k,
                &bf,
                r,
            ));
            csv_content.push('\n');
        }
        fs::write(csv_path, &csv_content)
            .with_context(|| format!("writing CSV to {}", csv_path.display()))?;
        eprintln!("Wrote {}", csv_path.display());
    } else if total > 1 {
        // Print CSV to stdout for piping
        println!("{CSV_HEADER}");
        for r in &results {
            println!(
                "{}",
                format_csv_row(
                    &config.dataset_name,
                    base_vectors.len(),
                    dims,
                    queries.len(),
                    config.k,
                    &bf,
                    r,
                )
            );
        }
    }

    eprintln!("Cache dir: {}", config.cache_dir.display());
    eprintln!("Done.");
    Ok(())
}
