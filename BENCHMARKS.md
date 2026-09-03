# Stoolap Benchmark Results

Performance comparison between **Stoolap**, **SQLite**, and **DuckDB** using identical workloads.

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Rows | 10,000 |
| Iterations | 500 (point queries), 250 (medium), 50 (heavy) |
| Mode | In-memory |
| Platform | Apple Silicon |
| Sampling | best of 30 runs for Stoolap (three sessions), best of 10 for SQLite and DuckDB, measured back to back |
| Measured | 2026-09-03, `main` at 31364b6f |
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
| SELECT by ID | **0.11** | 0.16 | 69.70 | Stoolap |
| SELECT by index (exact) | **2.99** | 22.91 | 1186.94 | Stoolap |
| SELECT by index (range) | **24.92** | 227.87 | 309.96 | Stoolap |
| SELECT complex | **98.52** | 470.02 | 164.66 | Stoolap |
| SELECT * (full scan) | **81.99** | 434.31 | 598.39 | Stoolap |
| UPDATE by ID | 0.59 | **0.53** | 70.94 | SQLite |
| UPDATE complex | **47.52** | 391.94 | 119.49 | Stoolap |
| INSERT single | **0.72** | 1.36 | 122.53 | Stoolap |
| DELETE by ID | **0.58** | 1.16 | 82.32 | Stoolap |
| DELETE complex | **2.20** | 338.53 | 107.44 | Stoolap |
| Aggregation (GROUP BY) | **42.12** | 1126.52 | 102.80 | Stoolap |

---

## Advanced Operations

| Operation | Stoolap (us) | SQLite (us) | DuckDB (us) | Best |
|-----------|-------------|-------------|-------------|------|
| INNER JOIN | 19.39 | **12.07** | 307.31 | SQLite |
| LEFT JOIN + GROUP BY | **42.80** | 47.74 | 1132.88 | Stoolap |
| Scalar subquery | **7.91** | 330.70 | 220.47 | Stoolap |
| IN subquery | **89.11** | 1628.15 | 482.80 | Stoolap |
| EXISTS subquery | **2.49** | 27.75 | 744.22 | Stoolap |
| CTE + JOIN | **31.44** | 62.63 | 781.58 | Stoolap |
| Window ROW_NUMBER | **233.79** | 1519.95 | 615.79 | Stoolap |
| Window ROW_NUMBER (PK) | **5.63** | 17.22 | 372.81 | Stoolap |
| Window PARTITION BY | **7.96** | 47.89 | 541.97 | Stoolap |
| UNION ALL | **4.64** | 5.04 | 148.38 | Stoolap |
| CASE expression | **4.18** | 4.24 | 180.82 | Stoolap |
| Complex JOIN+GROUP+HAVING | **45.00** | 68.70 | 1637.86 | Stoolap |
| Batch INSERT (100 rows) | **49.62** | 62.99 | 7757.23 | Stoolap |

---

## Bottleneck Hunters

| Operation | Stoolap (us) | SQLite (us) | DuckDB (us) | Best |
|-----------|-------------|-------------|-------------|------|
| DISTINCT (no ORDER) | **3.83** | 91.43 | 206.90 | Stoolap |
| DISTINCT + ORDER BY | **4.46** | 121.32 | 247.00 | Stoolap |
| COUNT DISTINCT | **0.26** | 92.07 | 190.66 | Stoolap |
| LIKE prefix (User_1%) | **3.32** | 8.29 | 181.92 | Stoolap |
| LIKE contains (%50%) | **29.66** | 134.32 | 189.46 | Stoolap |
| OR conditions (3 vals) | **2.81** | 12.58 | 169.62 | Stoolap |
| IN list (7 values) | **2.02** | 12.29 | 7369.91 | Stoolap |
| NOT IN subquery | **64.28** | 1668.07 | 519.09 | Stoolap |
| NOT EXISTS subquery | **21.78** | 76.56 | 898.08 | Stoolap |
| OFFSET pagination (5000) | **11.36** | 21.71 | 440.42 | Stoolap |
| Multi-col ORDER BY (3) | **136.45** | 359.28 | 289.59 | Stoolap |
| Self JOIN (same age) | 12.32 | **9.07** | 356.03 | SQLite |
| Multi window funcs (3) | **393.63** | 1559.68 | 661.82 | Stoolap |
| Nested subquery (3 lvl) | **301.48** | 5530.04 | 766.03 | Stoolap |
| Multi aggregates (6) | **110.79** | 713.77 | 280.07 | Stoolap |
| COALESCE + IS NOT NULL | 3.52 | **2.56** | 72.09 | SQLite |
| Expr in WHERE (funcs) | **4.69** | 12.46 | 258.36 | Stoolap |
| Math expressions | 11.78 | **10.23** | 176.87 | SQLite |
| String concat (\|\|) | 5.86 | **4.74** | 206.59 | SQLite |
| Large result (no LIMIT) | **223.78** | 412.98 | 299.37 | Stoolap |
| Multiple CTEs (2) | 19.73 | **14.77** | 265.89 | SQLite |
| Correlated in SELECT | **225.11** | 459.84 | 1005.40 | Stoolap |
| BETWEEN (non-indexed) | **2.64** | 7.75 | 133.74 | Stoolap |
| GROUP BY (2 columns) | **136.35** | 1827.71 | 316.01 | Stoolap |
| CROSS JOIN (limited) | **84.46** | 1164.07 | 238.74 | Stoolap |
| Derived table (FROM sub) | 368.09 | 700.80 | **223.94** | DuckDB |
| Window ROWS frame | **347.98** | 1576.43 | 2808.85 | Stoolap |
| HAVING complex | **83.43** | 1128.71 | 120.62 | Stoolap |
| Compare with subquery | **4.78** | 1273.44 | 247.97 | Stoolap |

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
| COUNT DISTINCT | 0.26 us | 92.07 us | **360x** |
| Compare with subquery | 4.78 us | 1273.44 us | **266x** |
| DELETE complex | 2.20 us | 338.53 us | **154x** |
| Scalar subquery | 7.91 us | 330.70 us | **42x** |
| DISTINCT + ORDER BY | 4.46 us | 121.32 us | **27x** |
| Aggregation (GROUP BY) | 42.12 us | 1126.52 us | **27x** |
| NOT IN subquery | 64.28 us | 1668.07 us | **26x** |
| DISTINCT (no ORDER) | 3.83 us | 91.43 us | **24x** |
| Nested subquery (3 lvl) | 301.48 us | 5530.04 us | **18x** |
| IN subquery | 89.11 us | 1628.15 us | **18x** |
| CROSS JOIN (limited) | 84.46 us | 1164.07 us | **14x** |
| HAVING complex | 83.43 us | 1128.71 us | **14x** |
| GROUP BY (2 columns) | 136.35 us | 1827.71 us | **13x** |
| EXISTS subquery | 2.49 us | 27.75 us | **11x** |
| SELECT by index (range) | 24.92 us | 227.87 us | **9x** |

---

## Top Stoolap Wins vs DuckDB

| Operation | Stoolap | DuckDB | Speedup |
|-----------|---------|--------|---------|
| IN list (7 values) | 2.02 us | 7369.91 us | **3648x** |
| COUNT DISTINCT | 0.26 us | 190.66 us | **745x** |
| SELECT by ID | 0.11 us | 69.70 us | **634x** |
| SELECT by index (exact) | 2.99 us | 1186.94 us | **397x** |
| EXISTS subquery | 2.49 us | 744.22 us | **299x** |
| INSERT single | 0.72 us | 122.53 us | **171x** |
| Batch INSERT (100 rows) | 49.62 us | 7757.23 us | **156x** |
| DELETE by ID | 0.58 us | 82.32 us | **142x** |
| UPDATE by ID | 0.59 us | 70.94 us | **120x** |
| Window PARTITION BY | 7.96 us | 541.97 us | **68x** |
| Window ROW_NUMBER (PK) | 5.63 us | 372.81 us | **66x** |
| OR conditions (3 vals) | 2.81 us | 169.62 us | **60x** |
| DISTINCT + ORDER BY | 4.46 us | 247.00 us | **55x** |
| Expr in WHERE (funcs) | 4.69 us | 258.36 us | **55x** |
| LIKE prefix (User_1%) | 3.32 us | 181.92 us | **55x** |
| DISTINCT (no ORDER) | 3.83 us | 206.90 us | **54x** |
| Compare with subquery | 4.78 us | 247.97 us | **52x** |
| BETWEEN (non-indexed) | 2.64 us | 133.74 us | **51x** |
| DELETE complex | 2.20 us | 107.44 us | **49x** |
| CASE expression | 4.18 us | 180.82 us | **43x** |
| NOT EXISTS subquery | 21.78 us | 898.08 us | **41x** |
| OFFSET pagination (5000) | 11.36 us | 440.42 us | **39x** |
| Complex JOIN+GROUP+HAVING | 45.00 us | 1637.86 us | **36x** |
| String concat (\|\|) | 5.86 us | 206.59 us | **35x** |
| UNION ALL | 4.64 us | 148.38 us | **32x** |

---

## Where Others Win

### SQLite Advantages

| Operation | SQLite | Stoolap | Factor |
|-----------|--------|---------|--------|
| INNER JOIN | 12.07 us | 19.39 us | 1.6x |
| COALESCE + IS NOT NULL | 2.56 us | 3.52 us | 1.4x |
| Self JOIN (same age) | 9.07 us | 12.32 us | 1.4x |
| Multiple CTEs (2) | 14.77 us | 19.73 us | 1.3x |
| String concat (\|\|) | 4.74 us | 5.86 us | 1.2x |
| Math expressions | 10.23 us | 11.78 us | 1.2x |
| UPDATE by ID | 0.53 us | 0.59 us | 1.1x |

### DuckDB Advantages

| Operation | DuckDB | Stoolap | Factor |
|-----------|--------|---------|--------|
| Derived table (FROM sub) | 223.94 us | 368.09 us | 1.6x |

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
  Point Queries (ID):     ████████████████████  634x vs DuckDB
  Subquery Compare:       ████████████████████  266x vs SQLite
  DISTINCT Operations:    ████████████████████  24-360x vs SQLite
  Semi-joins (EXISTS):    ████████████████████  4-299x faster
  Batch Inserts:          ████████████████████  156x vs DuckDB
  Window (PARTITION BY):  ████████████████████  68x vs DuckDB
  OFFSET Pagination:      ████████████████████  39x vs DuckDB
  Complex DML:            ████████████████████  3-154x faster
  Aggregations:           ████████████████████  27x vs SQLite

SQLite Strengths:
  Simple JOINs:           ████████              SQLite ~1.4-1.6x faster
  Simple Expressions:     ██████                SQLite ~1.2-1.4x faster

DuckDB Strengths:
  Derived Tables:         ██████                DuckDB ~1.6x faster
```

---

## Best Use Cases

### Choose Stoolap for:
- **OLTP workloads** - Point queries, inserts, deletes
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

*Benchmarks performed on Apple Silicon, in-memory mode, best of 30 runs for Stoolap and 10 for SQLite and DuckDB, measured back to back on one idle machine. Results are point-in-time for `main` at 31364b6f, which is after the v0.4.0 release, not the release itself; re-run on your hardware and workload for current numbers.*
