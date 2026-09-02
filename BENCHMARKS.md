# Stoolap Benchmark Results

Performance comparison between **Stoolap**, **SQLite**, and **DuckDB** using identical workloads.

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Rows | 10,000 |
| Iterations | 500 (point queries), 250 (medium), 50 (heavy) |
| Mode | In-memory |
| Platform | Apple Silicon |
| SQLite | rusqlite v0.40.2 |
| DuckDB | duckdb v1.10501.0 |

## Overall Score

```
+---------------------------------------------------------------+
|                                                               |
|   STOOLAP vs SQLite:    46 wins / 7 losses    (87% win rate)  |
|   STOOLAP vs DuckDB:    52 wins / 1 loss     (98% win rate)  |
|                                                               |
+---------------------------------------------------------------+
```

---

## Basic Operations

| Operation | Stoolap (us) | SQLite (us) | DuckDB (us) | Best |
|-----------|-------------|-------------|-------------|------|
| SELECT by ID | **0.13** | 0.21 | 72.54 | Stoolap |
| SELECT by index (exact) | **3.27** | 27.44 | 1283.59 | Stoolap |
| SELECT by index (range) | **27.20** | 249.63 | 314.86 | Stoolap |
| SELECT complex | **102.42** | 482.93 | 170.17 | Stoolap |
| SELECT * (full scan) | **84.70** | 485.73 | 622.47 | Stoolap |
| UPDATE by ID | 0.60 | **0.55** | 86.91 | SQLite |
| UPDATE complex | **56.61** | 394.35 | 145.22 | Stoolap |
| INSERT single | **0.82** | 1.40 | 150.74 | Stoolap |
| DELETE by ID | **0.69** | 1.18 | 99.71 | Stoolap |
| DELETE complex | **2.29** | 345.08 | 116.55 | Stoolap |
| Aggregation (GROUP BY) | **43.73** | 1154.25 | 100.39 | Stoolap |

---

## Advanced Operations

| Operation | Stoolap (us) | SQLite (us) | DuckDB (us) | Best |
|-----------|-------------|-------------|-------------|------|
| INNER JOIN | 20.61 | **13.67** | 352.72 | SQLite |
| LEFT JOIN + GROUP BY | **45.47** | 54.18 | 1404.65 | Stoolap |
| Scalar subquery | **8.49** | 332.67 | 255.46 | Stoolap |
| IN subquery | **93.63** | 1630.64 | 624.48 | Stoolap |
| EXISTS subquery | **2.60** | 30.25 | 977.45 | Stoolap |
| CTE + JOIN | **31.53** | 65.19 | 1097.58 | Stoolap |
| Window ROW_NUMBER | **240.48** | 1549.46 | 797.31 | Stoolap |
| Window ROW_NUMBER (PK) | **5.91** | 17.97 | 481.57 | Stoolap |
| Window PARTITION BY | **8.13** | 49.13 | 708.42 | Stoolap |
| UNION ALL | **4.89** | 5.85 | 220.02 | Stoolap |
| CASE expression | **4.29** | 4.39 | 225.71 | Stoolap |
| Complex JOIN+GROUP+HAVING | **48.73** | 71.83 | 2174.61 | Stoolap |
| Batch INSERT (100 rows) | **51.71** | 64.75 | 9153.49 | Stoolap |

---

## Bottleneck Hunters

| Operation | Stoolap (us) | SQLite (us) | DuckDB (us) | Best |
|-----------|-------------|-------------|-------------|------|
| DISTINCT (no ORDER) | **4.40** | 97.75 | 258.64 | Stoolap |
| DISTINCT + ORDER BY | **4.73** | 123.02 | 335.87 | Stoolap |
| COUNT DISTINCT | **0.26** | 91.07 | 269.60 | Stoolap |
| LIKE prefix (User_1%) | **3.51** | 8.19 | 253.31 | Stoolap |
| LIKE contains (%50%) | **30.14** | 133.87 | 243.19 | Stoolap |
| OR conditions (3 vals) | **2.92** | 12.68 | 207.39 | Stoolap |
| IN list (7 values) | **2.05** | 12.50 | 8600.29 | Stoolap |
| NOT IN subquery | **67.48** | 1693.86 | 621.88 | Stoolap |
| NOT EXISTS subquery | **21.77** | 86.66 | 1042.73 | Stoolap |
| OFFSET pagination (5000) | **11.62** | 21.24 | 542.62 | Stoolap |
| Multi-col ORDER BY (3) | **141.29** | 360.62 | 344.13 | Stoolap |
| Self JOIN (same age) | 12.65 | **9.43** | 419.84 | SQLite |
| Multi window funcs (3) | **399.55** | 1579.36 | 828.94 | Stoolap |
| Nested subquery (3 lvl) | **323.75** | 5585.09 | 967.97 | Stoolap |
| Multi aggregates (6) | **114.25** | 760.42 | 355.11 | Stoolap |
| COALESCE + IS NOT NULL | 3.68 | **2.94** | 82.59 | SQLite |
| Expr in WHERE (funcs) | **5.41** | 14.06 | 324.14 | Stoolap |
| Math expressions | 12.36 | **11.74** | 216.67 | SQLite |
| String concat (\|\|) | 6.13 | **5.25** | 251.33 | SQLite |
| Large result (no LIMIT) | **234.14** | 465.87 | 372.65 | Stoolap |
| Multiple CTEs (2) | 20.12 | **17.41** | 312.59 | SQLite |
| Correlated in SELECT | **251.33** | 523.02 | 1306.52 | Stoolap |
| BETWEEN (non-indexed) | **3.00** | 8.90 | 171.87 | Stoolap |
| GROUP BY (2 columns) | **140.68** | 2370.70 | 404.03 | Stoolap |
| CROSS JOIN (limited) | **89.24** | 1462.06 | 332.74 | Stoolap |
| Derived table (FROM sub) | 379.44 | 943.29 | **327.14** | DuckDB |
| Window ROWS frame | **354.91** | 2109.35 | 2461.63 | Stoolap |
| HAVING complex | **88.77** | 1492.41 | 94.98 | Stoolap |
| Compare with subquery | **5.10** | 1642.36 | 250.43 | Stoolap |

---

## Summary by Category

| Category | Stoolap Wins | SQLite Wins | DuckDB Wins |
|----------|-------------|-------------|-------------|
| Basic Operations | 10 | 1 | 0 |
| Advanced Operations | 12 | 1 | 0 |
| Bottleneck Hunters | 23 | 5 | 1 |
| **Total** | **45** | **7** | **1** |

---

## Top Stoolap Wins vs SQLite

| Operation | Stoolap | SQLite | Speedup |
|-----------|---------|--------|---------|
| COUNT DISTINCT | 0.26 us | 91.07 us | **352x** |
| Compare with subquery | 5.10 us | 1642.36 us | **322x** |
| DELETE complex | 2.29 us | 345.08 us | **151x** |
| Scalar subquery | 8.49 us | 332.67 us | **39x** |
| Aggregation (GROUP BY) | 43.73 us | 1154.25 us | **26x** |
| DISTINCT + ORDER BY | 4.73 us | 123.02 us | **26x** |
| NOT IN subquery | 67.48 us | 1693.86 us | **25x** |
| DISTINCT (no ORDER) | 4.40 us | 97.75 us | **22x** |
| IN subquery | 93.63 us | 1630.64 us | **17x** |
| Nested subquery (3 lvl) | 323.75 us | 5585.09 us | **17x** |
| GROUP BY (2 columns) | 140.68 us | 2370.70 us | **17x** |
| HAVING complex | 88.77 us | 1492.41 us | **17x** |
| CROSS JOIN (limited) | 89.24 us | 1462.06 us | **16x** |
| EXISTS subquery | 2.60 us | 30.25 us | **12x** |
| SELECT by index (range) | 27.20 us | 249.63 us | **9x** |

---

## Top Stoolap Wins vs DuckDB

| Operation | Stoolap | DuckDB | Speedup |
|-----------|---------|--------|---------|
| IN list (7 values) | 2.05 us | 8600.29 us | **4193x** |
| COUNT DISTINCT | 0.26 us | 269.60 us | **1041x** |
| SELECT by ID | 0.13 us | 72.54 us | **545x** |
| SELECT by index (exact) | 3.27 us | 1283.59 us | **392x** |
| EXISTS subquery | 2.60 us | 977.45 us | **377x** |
| INSERT single | 0.82 us | 150.74 us | **185x** |
| Batch INSERT (100 rows) | 51.71 us | 9153.49 us | **177x** |
| DELETE by ID | 0.69 us | 99.71 us | **144x** |
| UPDATE by ID | 0.60 us | 86.91 us | **144x** |
| Window PARTITION BY | 8.13 us | 708.42 us | **87x** |
| Window ROW_NUMBER (PK) | 5.91 us | 481.57 us | **81x** |
| LIKE prefix (User_1%) | 3.51 us | 253.31 us | **72x** |
| OR conditions (3 vals) | 2.92 us | 207.39 us | **71x** |
| DISTINCT + ORDER BY | 4.73 us | 335.87 us | **71x** |
| Expr in WHERE (funcs) | 5.41 us | 324.14 us | **60x** |
| DISTINCT (no ORDER) | 4.40 us | 258.64 us | **59x** |
| BETWEEN (non-indexed) | 3.00 us | 171.87 us | **57x** |
| CASE expression | 4.29 us | 225.71 us | **53x** |
| DELETE complex | 2.29 us | 116.55 us | **51x** |
| Compare with subquery | 5.10 us | 250.43 us | **49x** |
| NOT EXISTS subquery | 21.77 us | 1042.73 us | **48x** |
| OFFSET pagination (5000) | 11.62 us | 542.62 us | **47x** |
| UNION ALL | 4.89 us | 220.02 us | **45x** |
| Complex JOIN+GROUP+HAVING | 48.73 us | 2174.61 us | **45x** |
| String concat (\|\|) | 6.13 us | 251.33 us | **41x** |

---

## Where Others Win

### SQLite Advantages

| Operation | SQLite | Stoolap | Factor |
|-----------|--------|---------|--------|
| INNER JOIN | 13.67 us | 20.61 us | 1.5x |
| Self JOIN (same age) | 9.43 us | 12.65 us | 1.3x |
| COALESCE + IS NOT NULL | 2.94 us | 3.68 us | 1.3x |
| String concat (\|\|) | 5.25 us | 6.13 us | 1.2x |
| Multiple CTEs (2) | 17.41 us | 20.12 us | 1.2x |
| UPDATE by ID | 0.55 us | 0.60 us | 1.1x |
| Math expressions | 11.74 us | 12.36 us | 1.1x |

### DuckDB Advantages

| Operation | DuckDB | Stoolap | Factor |
|-----------|--------|---------|--------|
| Derived table (FROM sub) | 327.14 us | 379.44 us | 1.2x |

---

## Architecture Comparison

| Feature | Stoolap | SQLite | DuckDB |
|---------|---------|--------|--------|
| Storage Model | Row-based | Row-based | Columnar |
| Concurrency | MVCC | WAL/locking | MVCC |
| Query Optimizer | Cost-based | Rule-based | Cost-based |
| Parallel Execution | Yes (Rayon) | No | Yes |
| Language | Pure Rust | C | C++ |
| Memory Safety | Guaranteed | Manual | Manual |

---

## Performance Characteristics

```
Stoolap Strengths:
  Point Queries (ID):     ████████████████████  545x vs DuckDB
  Subquery Compare:       ████████████████████  322x vs SQLite
  DISTINCT Operations:    ████████████████████  22-352x vs SQLite
  Semi-joins (EXISTS):    ████████████████████  4-377x faster
  Batch Inserts:          ████████████████████  177x vs DuckDB
  Window (PARTITION BY):  ████████████████████  87x vs DuckDB
  OFFSET Pagination:      ████████████████████  47x vs DuckDB
  Complex DML:            ████████████████████  3-151x faster
  Aggregations:           ████████████████████  26x vs SQLite

SQLite Strengths:
  Simple JOINs:           ████████              SQLite ~1.3-1.5x faster
  Simple Expressions:     ██████                SQLite ~1.1-1.3x faster

DuckDB Strengths:
  Derived Tables:         ██████                DuckDB ~1.2x faster
```

---

## Best Use Cases

### Choose Stoolap for:
- **OLTP workloads** - Point queries, updates, deletes
- **Real-time analytics** - Fast aggregations with DISTINCT
- **Semi-join patterns** - EXISTS, IN subqueries
- **Time-travel queries** - AS OF temporal queries
- **Embedded applications** - Pure Rust, memory-safe
- **Edge computing** - Low-latency, low-memory operations

### Choose SQLite for:
- **Simple nested loop joins** - Highly optimized
- **Single-threaded simplicity** - No concurrency needs
- **Maximum compatibility** - Industry standard

### Choose DuckDB for:
- **Large columnar scans** - Vectorized execution
- **Data science workflows** - DataFrame integration
- **Derived table queries** - Slightly faster for FROM subqueries

---

## Running the Benchmarks

```bash
# Stoolap benchmark (no external dependencies)
cargo build --release --example benchmark
./target/release/examples/benchmark

# SQLite benchmark (requires sqlite feature)
cargo build --release --example benchmark_sqlite --features sqlite
./target/release/examples/benchmark_sqlite

# DuckDB benchmark (requires duckdb feature)
cargo build --release --example benchmark_duckdb --features duckdb
./target/release/examples/benchmark_duckdb

# Build all benchmarks at once
cargo build --release --example benchmark --example benchmark_sqlite --example benchmark_duckdb --features "sqlite duckdb"
```

---

*Benchmarks performed on Apple Silicon, in-memory mode, best of 10 runs. Results are point-in-time for v0.4.0; re-run on your hardware and workload for current numbers.*
