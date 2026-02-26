// Copyright 2025 Stoolap Contributors
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

//! Vector Search Performance Benchmark
//!
//! Measures real end-to-end performance of Stoolap's vector search through SQL:
//! 1. Brute force k-NN (no index) — VEC_DISTANCE_L2 + ORDER BY + LIMIT
//! 2. HNSW index k-NN — same query, accelerated by HNSW graph
//! 3. Multi-metric comparison — L2 vs Cosine vs Inner Product
//! 4. Hybrid filtered search — WHERE clause + vector k-NN
//!
//! Run:  cargo run --example vector_search_bench --release
//! Args: cargo run --example vector_search_bench --release -- --vectors 10000 --dims 128

use std::time::Instant;

use stoolap::api::Database;
use stoolap::storage::index::{default_ef_construction, default_ef_search, default_m_for_dims};

// ─────────────────────────────────────────────────────────────
// Distance Function (for ground-truth validation)
// ─────────────────────────────────────────────────────────────

#[inline(always)]
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
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
    sum.sqrt()
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

fn format_duration(ms: f64) -> String {
    if ms < 0.01 {
        format!("{:.1} us", ms * 1000.0)
    } else if ms < 1.0 {
        format!("{:.2} ms", ms)
    } else if ms < 1000.0 {
        format!("{:.1} ms", ms)
    } else {
        format!("{:.2} s", ms / 1000.0)
    }
}

fn format_size(bytes: usize) -> String {
    if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

fn format_with_commas(n: usize) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut result = String::new();
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            result.push(',');
        }
        result.push(b as char);
    }
    result
}

fn vec_to_sql_literal(v: &[f32]) -> String {
    let inner: Vec<String> = v.iter().map(|x| format!("{:.6}", x)).collect();
    format!("[{}]", inner.join(","))
}

fn print_separator(width: usize) {
    println!("{}", "=".repeat(width));
}

fn compute_recall(ground_truth: &[usize], approximate: &[usize]) -> f64 {
    let gt: std::collections::HashSet<usize> = ground_truth.iter().copied().collect();
    let approx: std::collections::HashSet<usize> = approximate.iter().copied().collect();
    let matches = gt.intersection(&approx).count();
    matches as f64 / ground_truth.len().max(1) as f64
}

// ─────────────────────────────────────────────────────────────
// Vector Generation
// ─────────────────────────────────────────────────────────────

fn generate_vectors(num_vectors: usize, dims: usize, num_clusters: usize) -> Vec<Vec<f32>> {
    let centers: Vec<Vec<f32>> = (0..num_clusters)
        .map(|c| {
            (0..dims)
                .map(|d| {
                    let base = ((c * 7 + d * 13) as f32).sin() * 3.0;
                    let decay = 1.0 / (1.0 + d as f32 * 0.01);
                    base * decay
                })
                .collect()
        })
        .collect();

    (0..num_vectors)
        .map(|i| {
            let center = &centers[i % num_clusters];
            center
                .iter()
                .map(|&c| {
                    let u1: f32 = rand::random::<f32>().max(1e-10);
                    let u2: f32 = rand::random();
                    let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                    c + noise * 0.5
                })
                .collect()
        })
        .collect()
}

fn generate_queries(num_queries: usize, dims: usize, num_clusters: usize) -> Vec<Vec<f32>> {
    let centers: Vec<Vec<f32>> = (0..num_clusters)
        .map(|c| {
            (0..dims)
                .map(|d| {
                    let base = ((c * 7 + d * 13) as f32).sin() * 3.0;
                    let decay = 1.0 / (1.0 + d as f32 * 0.01);
                    base * decay
                })
                .collect()
        })
        .collect();

    (0..num_queries)
        .map(|i| {
            let center = &centers[i % num_clusters];
            center
                .iter()
                .map(|&c| {
                    let u1: f32 = rand::random::<f32>().max(1e-10);
                    let u2: f32 = rand::random();
                    let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                    c + noise * 0.6
                })
                .collect()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────
// Benchmark Helper
// ─────────────────────────────────────────────────────────────

/// Run a k-NN query and return (ids, latency_ms)
fn run_knn_query(db: &Database, sql: &str) -> (Vec<usize>, f64) {
    let start = Instant::now();
    let mut ids = Vec::new();
    for row in db.query(sql, ()).expect("query failed") {
        let row = row.expect("row error");
        let id = row.get::<i64>(0).unwrap() as usize;
        ids.push(id);
    }
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    (ids, ms)
}

/// Batch insert vectors into a table
fn batch_insert(db: &Database, table: &str, vectors: &[Vec<f32>], categories: Option<&[usize]>) {
    let batch_size = 1000;
    let mut inserted = 0;
    while inserted < vectors.len() {
        let end = (inserted + batch_size).min(vectors.len());
        db.execute("BEGIN", ()).unwrap();
        for idx in inserted..end {
            let vec_literal = vec_to_sql_literal(&vectors[idx]);
            if let Some(cats) = categories {
                db.execute(
                    &format!(
                        "INSERT INTO {table} (id, category, embedding) VALUES ({idx}, {}, '{vec_literal}')",
                        cats[idx]
                    ),
                    (),
                )
                .unwrap();
            } else {
                db.execute(
                    &format!("INSERT INTO {table} (id, embedding) VALUES ({idx}, '{vec_literal}')"),
                    (),
                )
                .unwrap();
            }
        }
        db.execute("COMMIT", ()).unwrap();
        inserted = end;
    }
}

fn main() {
    // Parse arguments
    let args: Vec<String> = std::env::args().collect();
    let mut num_vectors: usize = 10_000;
    let mut dims: usize = 128;
    let mut k: usize = 10;
    let mut hnsw_m: Option<usize> = None;
    let mut ef_construction: Option<usize> = None;
    let mut ef_search: Option<usize> = None;
    let mut num_queries: usize = 10;
    let mut bf_only = false;
    let mut skip_cosine = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--vectors" | "-n" => {
                num_vectors = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--dims" | "-d" => {
                dims = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--k" => {
                k = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--m" => {
                hnsw_m = Some(args[i + 1].parse().unwrap());
                i += 2;
            }
            "--ef-construction" => {
                ef_construction = Some(args[i + 1].parse().unwrap());
                i += 2;
            }
            "--ef-search" => {
                ef_search = Some(args[i + 1].parse().unwrap());
                i += 2;
            }
            "--queries" | "-q" => {
                num_queries = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--bf-only" => {
                bf_only = true;
                i += 1;
            }
            "--skip-cosine" => {
                skip_cosine = true;
                i += 1;
            }
            "--help" | "-h" => {
                println!("Stoolap Vector Search Performance Benchmark");
                println!();
                println!("Usage: cargo run --example vector_search_bench --release -- [OPTIONS]");
                println!();
                println!("Options:");
                println!("  -n, --vectors N          Number of vectors (default: 10000)");
                println!("  -d, --dims N             Vector dimensions (default: 128)");
                println!("  --k N                    Nearest neighbors to find (default: 10)");
                println!("  --m N                    HNSW M parameter (default: auto from dims)");
                println!("  --ef-construction N      HNSW build quality (default: auto from M)");
                println!("  --ef-search N            HNSW search quality (default: auto from M)");
                println!("  -q, --queries N          Number of queries to average (default: 10)");
                println!("  --bf-only                Skip HNSW index, brute force only");
                println!("  --skip-cosine            Skip cosine HNSW table (faster for large N)");
                println!();
                println!("Presets:");
                println!("  Quick:   -n 5000  -d 64    (fast, ~30s total)");
                println!("  Small:   -n 10000 -d 128   (default, ~2min total)");
                println!("  Medium:  -n 20000 -d 256   (~10min total)");
                println!("  Large:   -n 50000 -d 128 --skip-cosine (~10min)");
                println!("  BF-100K: -n 100000 -d 128 --bf-only    (~2min)");
                return;
            }
            _ => {
                i += 1;
            }
        }
    }

    // Resolve HNSW parameters: use index defaults when not explicitly set
    let hnsw_m = hnsw_m.unwrap_or_else(|| default_m_for_dims(dims));
    let ef_construction = ef_construction.unwrap_or_else(|| default_ef_construction(hnsw_m));
    let ef_search = ef_search.unwrap_or_else(|| default_ef_search(hnsw_m));

    let num_clusters = 50;
    let num_categories: usize = 20;
    let data_size = num_vectors * dims * 4;

    println!();
    print_separator(76);
    println!("  STOOLAP VECTOR SEARCH BENCHMARK (End-to-End SQL)");
    print_separator(76);
    println!();
    println!("  Configuration:");
    println!(
        "    Vectors:        {:>10}",
        format_with_commas(num_vectors)
    );
    println!("    Dimensions:     {:>10}", dims);
    println!("    k (neighbors):  {:>10}", k);
    println!("    Data size:      {:>10}", format_size(data_size));
    println!(
        "    HNSW params:    M={}, ef_construction={}, ef_search={}",
        hnsw_m, ef_construction, ef_search
    );
    println!("    Queries:        {:>10} (averaged)", num_queries);
    println!();

    // ── Generate vectors ──────────────────────────────────────
    print!(
        "  Generating {} vectors ({} dims)...",
        format_with_commas(num_vectors),
        dims,
    );
    let gen_start = Instant::now();
    let vectors = generate_vectors(num_vectors, dims, num_clusters);
    let queries = generate_queries(num_queries, dims, num_clusters);
    let categories: Vec<usize> = (0..num_vectors).map(|i| i % num_categories).collect();
    println!(" done ({:.1}s)", gen_start.elapsed().as_secs_f64());

    // ── Compute ground truth ──────────────────────────────────
    print!("  Computing ground truth...");
    let mut ground_truth: Vec<Vec<usize>> = Vec::new();
    for query in &queries {
        let mut dists: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, l2_distance(v, query)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        ground_truth.push(dists.iter().take(k).map(|&(id, _)| id).collect());
    }
    println!(" done");

    // ── Create database and insert ────────────────────────────
    let db = Database::open_in_memory().expect("failed to open database");

    db.execute(
        &format!(
            "CREATE TABLE vectors (
                id INTEGER PRIMARY KEY,
                category INTEGER,
                embedding VECTOR({dims})
            )"
        ),
        (),
    )
    .unwrap();

    print!("  Inserting {} vectors...", format_with_commas(num_vectors));
    let insert_start = Instant::now();
    batch_insert(&db, "vectors", &vectors, Some(&categories));
    let insert_time = insert_start.elapsed();
    println!(
        " done ({:.1}s, {:.0} rows/s)",
        insert_time.as_secs_f64(),
        num_vectors as f64 / insert_time.as_secs_f64()
    );

    println!();
    print_separator(76);
    println!("  BENCHMARK RESULTS");
    print_separator(76);

    // ══════════════════════════════════════════════════════════
    // [1] Brute Force k-NN (no index)
    // ══════════════════════════════════════════════════════════
    println!();
    println!("  [1] Brute Force k-NN (no HNSW index)");
    println!("  {}", "-".repeat(60));
    println!("      SELECT id, VEC_DISTANCE_L2(embedding, '...') AS dist");
    println!("      FROM vectors ORDER BY dist LIMIT {k}");
    println!();

    let mut bf_total_ms = 0.0;
    let mut bf_total_recall = 0.0;

    for (qi, query) in queries.iter().enumerate() {
        let vec_lit = vec_to_sql_literal(query);
        let sql = format!(
            "SELECT id, VEC_DISTANCE_L2(embedding, '{vec_lit}') AS dist \
             FROM vectors ORDER BY dist LIMIT {k}"
        );
        let (ids, ms) = run_knn_query(&db, &sql);
        bf_total_ms += ms;
        bf_total_recall += compute_recall(&ground_truth[qi], &ids);
    }

    let bf_avg_ms = bf_total_ms / num_queries as f64;
    let bf_recall = bf_total_recall / num_queries as f64;

    println!("    Avg latency:   {}", format_duration(bf_avg_ms));
    println!("    QPS:           {:.1}", 1000.0 / bf_avg_ms);
    println!("    Recall:        {:.1}%", bf_recall * 100.0);

    let mut hnsw_avg_ms = 0.0;
    let mut hnsw_recall = 0.0;
    let mut hnsw_speedup = 0.0;
    let mut build_time = std::time::Duration::ZERO;

    if !bf_only {
        // ══════════════════════════════════════════════════════════
        // [2] HNSW Index Build
        // ══════════════════════════════════════════════════════════
        println!();
        println!("  [2] Build HNSW Index");
        println!("  {}", "-".repeat(60));
        println!(
            "      CREATE INDEX ... USING HNSW WITH (m={hnsw_m}, ef_construction={ef_construction}, ef_search={ef_search})"
        );
        println!();

        let build_start = Instant::now();
        db.execute(
            &format!(
                "CREATE INDEX idx_emb ON vectors(embedding) USING HNSW \
                 WITH (m = {hnsw_m}, ef_construction = {ef_construction}, ef_search = {ef_search})"
            ),
            (),
        )
        .expect("create index failed");
        build_time = build_start.elapsed();

        println!(
            "    Build time:    {:.1}s ({:.0} vectors/sec)",
            build_time.as_secs_f64(),
            num_vectors as f64 / build_time.as_secs_f64()
        );

        // ══════════════════════════════════════════════════════════
        // [3] HNSW k-NN Search (L2)
        // ══════════════════════════════════════════════════════════
        println!();
        println!("  [3] HNSW k-NN Search (L2, same query as brute force)");
        println!("  {}", "-".repeat(60));
        println!();

        let mut hnsw_total_ms = 0.0;
        let mut hnsw_total_recall = 0.0;

        for (qi, query) in queries.iter().enumerate() {
            let vec_lit = vec_to_sql_literal(query);
            let sql = format!(
                "SELECT id, VEC_DISTANCE_L2(embedding, '{vec_lit}') AS dist \
                 FROM vectors ORDER BY dist LIMIT {k}"
            );
            let (ids, ms) = run_knn_query(&db, &sql);
            hnsw_total_ms += ms;
            hnsw_total_recall += compute_recall(&ground_truth[qi], &ids);
        }

        hnsw_avg_ms = hnsw_total_ms / num_queries as f64;
        hnsw_recall = hnsw_total_recall / num_queries as f64;
        hnsw_speedup = bf_avg_ms / hnsw_avg_ms;

        println!("    Avg latency:   {}", format_duration(hnsw_avg_ms));
        println!("    QPS:           {:.0}", 1000.0 / hnsw_avg_ms);
        println!("    Recall@{}:     {:.1}%", k, hnsw_recall * 100.0);
        println!("    Speedup:       {:.0}x vs brute force", hnsw_speedup);
    }

    if !bf_only {
        // ══════════════════════════════════════════════════════════
        // [4] Distance Metric Comparison (brute force)
        // ══════════════════════════════════════════════════════════
        println!();
        println!("  [4] Distance Metrics (brute force, no metric-specific HNSW)");
        println!("  {}", "-".repeat(60));
        println!();

        for (metric_name, func_name) in [
            ("L2 (Euclidean)", "VEC_DISTANCE_L2"),
            ("Cosine", "VEC_DISTANCE_COSINE"),
            ("Inner Product", "VEC_DISTANCE_IP"),
        ] {
            let mut total_ms = 0.0;
            for query in &queries {
                let vec_lit = vec_to_sql_literal(query);
                let sql = format!(
                    "SELECT id, {func_name}(embedding, '{vec_lit}') AS dist \
                     FROM vectors ORDER BY dist LIMIT {k}"
                );
                let (_, ms) = run_knn_query(&db, &sql);
                total_ms += ms;
            }
            let avg_ms = total_ms / num_queries as f64;
            println!(
                "    {:<20} {:>10}  ({:.0} QPS)",
                metric_name,
                format_duration(avg_ms),
                1000.0 / avg_ms
            );
        }

        println!();
        println!("    Note: L2 uses HNSW index. Cosine/IP use brute force (no matching HNSW).");
    }

    let mut cos_hnsw_avg_ms = 0.0;
    let mut hybrid_avg_ms = 0.0;

    if !bf_only && !skip_cosine {
        // ══════════════════════════════════════════════════════════
        // [5] HNSW Cosine Index
        // ══════════════════════════════════════════════════════════
        println!();
        println!("  [5] HNSW Cosine Index (separate table)");
        println!("  {}", "-".repeat(60));
        println!();

        db.execute(
            &format!("CREATE TABLE vectors_cos (id INTEGER PRIMARY KEY, embedding VECTOR({dims}))"),
            (),
        )
        .unwrap();

        print!("    Inserting...");
        let cos_ins_start = Instant::now();
        batch_insert(&db, "vectors_cos", &vectors, None);
        println!(" done ({:.1}s)", cos_ins_start.elapsed().as_secs_f64());

        print!("    Building HNSW cosine index...");
        let cos_build_start = Instant::now();
        db.execute(
            &format!(
                "CREATE INDEX idx_cos ON vectors_cos(embedding) USING HNSW \
                 WITH (m = {hnsw_m}, ef_construction = {ef_construction}, ef_search = {ef_search}, metric = 'cosine')"
            ),
            (),
        )
        .unwrap();
        let cos_build_time = cos_build_start.elapsed();
        println!(
            " done ({:.1}s, {:.0} vec/s)",
            cos_build_time.as_secs_f64(),
            num_vectors as f64 / cos_build_time.as_secs_f64()
        );

        let mut cos_hnsw_total_ms = 0.0;
        for query in &queries {
            let vec_lit = vec_to_sql_literal(query);
            let sql = format!(
                "SELECT id, VEC_DISTANCE_COSINE(embedding, '{vec_lit}') AS dist \
                 FROM vectors_cos ORDER BY dist LIMIT {k}"
            );
            let (_, ms) = run_knn_query(&db, &sql);
            cos_hnsw_total_ms += ms;
        }
        cos_hnsw_avg_ms = cos_hnsw_total_ms / num_queries as f64;

        // Compare with cosine brute force on 'vectors' table (which has no cosine HNSW)
        let mut cos_bf_total_ms = 0.0;
        for query in &queries {
            let vec_lit = vec_to_sql_literal(query);
            let sql = format!(
                "SELECT id, VEC_DISTANCE_COSINE(embedding, '{vec_lit}') AS dist \
                 FROM vectors ORDER BY dist LIMIT {k}"
            );
            let (_, ms) = run_knn_query(&db, &sql);
            cos_bf_total_ms += ms;
        }
        let cos_bf_avg_ms = cos_bf_total_ms / num_queries as f64;

        println!();
        println!(
            "    HNSW cosine:       {} ({:.0} QPS)",
            format_duration(cos_hnsw_avg_ms),
            1000.0 / cos_hnsw_avg_ms
        );
        println!(
            "    Brute force cos:   {} ({:.0} QPS)",
            format_duration(cos_bf_avg_ms),
            1000.0 / cos_bf_avg_ms
        );
        if cos_hnsw_avg_ms < cos_bf_avg_ms {
            println!(
                "    Speedup:           {:.0}x",
                cos_bf_avg_ms / cos_hnsw_avg_ms
            );
        }
    }

    if !bf_only {
        // ══════════════════════════════════════════════════════════
        // [6] Hybrid Filtered Search
        // ══════════════════════════════════════════════════════════
        let target_cat: usize = 5;
        let filtered_count = categories.iter().filter(|&&c| c == target_cat).count();

        println!();
        println!(
            "  [6] Hybrid Search: WHERE category = {} ({} of {} vectors, {:.0}% selectivity)",
            target_cat,
            format_with_commas(filtered_count),
            format_with_commas(num_vectors),
            100.0 * filtered_count as f64 / num_vectors as f64,
        );
        println!("  {}", "-".repeat(60));
        println!();

        let mut hybrid_total_ms = 0.0;
        for query in &queries {
            let vec_lit = vec_to_sql_literal(query);
            let sql = format!(
                "SELECT id, VEC_DISTANCE_L2(embedding, '{vec_lit}') AS dist \
                 FROM vectors WHERE category = {target_cat} ORDER BY dist LIMIT {k}"
            );
            let (_, ms) = run_knn_query(&db, &sql);
            hybrid_total_ms += ms;
        }
        hybrid_avg_ms = hybrid_total_ms / num_queries as f64;

        println!(
            "    HNSW + filter:     {} ({:.0} QPS)",
            format_duration(hybrid_avg_ms),
            1000.0 / hybrid_avg_ms
        );
    }

    // ══════════════════════════════════════════════════════════
    // Summary
    // ══════════════════════════════════════════════════════════
    println!();
    print_separator(76);
    println!("  SUMMARY");
    print_separator(76);
    println!();
    println!(
        "  {:46} {:>10} {:>8} {:>8}",
        "Method", "Latency", "QPS", "Recall"
    );
    println!("  {}", "-".repeat(74));
    println!(
        "  {:46} {:>10} {:>8.1} {:>6.1}%",
        "Brute force (VEC_DISTANCE_L2 + ORDER BY)",
        format_duration(bf_avg_ms),
        1000.0 / bf_avg_ms,
        bf_recall * 100.0,
    );
    if !bf_only {
        println!(
            "  {:46} {:>10} {:>8.0} {:>6.1}%",
            format!("HNSW L2 (ef_search={})", ef_search),
            format_duration(hnsw_avg_ms),
            1000.0 / hnsw_avg_ms,
            hnsw_recall * 100.0,
        );
        if !skip_cosine {
            println!(
                "  {:46} {:>10} {:>8.0} {:>8}",
                "HNSW Cosine",
                format_duration(cos_hnsw_avg_ms),
                1000.0 / cos_hnsw_avg_ms,
                "~",
            );
        }
        println!(
            "  {:46} {:>10} {:>8.0} {:>8}",
            "Hybrid L2 (HNSW + WHERE filter)",
            format_duration(hybrid_avg_ms),
            1000.0 / hybrid_avg_ms,
            "~",
        );
    }

    println!();
    println!("  Key results:");
    println!(
        "    Insert rate:     {:.0} rows/s",
        num_vectors as f64 / insert_time.as_secs_f64(),
    );
    if !bf_only {
        println!("    HNSW speedup:    {:.0}x vs brute force", hnsw_speedup);
        println!("    HNSW recall@{}:  {:.1}%", k, hnsw_recall * 100.0);
        println!(
            "    HNSW build:      {:.1}s ({:.0} vec/s)",
            build_time.as_secs_f64(),
            num_vectors as f64 / build_time.as_secs_f64(),
        );
    }

    println!();
    print_separator(76);
    println!();
}
