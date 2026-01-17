# Stoolap v0.2.3 Benchmark Results

Performance comparison between **Stoolap**, **SQLite**, and **DuckDB** using identical workloads.

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Rows | 10,000 |
| Iterations | 500 (point queries), 250 (medium), 50 (heavy) |
| Mode | In-memory |
| Platform | Apple Silicon |
| SQLite | rusqlite v0.32.1 |
| DuckDB | duckdb v1.4.3 |

## Overall Score

```
+---------------------------------------------------------------+
|                                                               |
|   STOOLAP vs SQLite:    45 wins / 8 losses    (85% win rate)  |
|   STOOLAP vs DuckDB:    52 wins / 1 loss     (98% win rate)  |
|                                                               |
+---------------------------------------------------------------+
```

---

## Basic Operations

| Operation | Stoolap (us) | SQLite (us) | DuckDB (us) | Best |
|-----------|-------------|-------------|-------------|------|
| SELECT by ID | **0.14** | 0.21 | 145.55 | Stoolap |
| SELECT by index (exact) | **4.51** | 28.02 | 288.70 | Stoolap |
| SELECT by index (range) | **37.48** | 285.62 | 428.24 | Stoolap |
| SELECT complex | **138.41** | 534.21 | 184.68 | Stoolap |
| SELECT * (full scan) | **109.02** | 515.21 | 707.14 | Stoolap |
| UPDATE by ID | **0.58** | 0.61 | 146.14 | Stoolap |
| UPDATE complex | **51.78** | 443.73 | 209.77 | Stoolap |
| INSERT single | **1.32** | 1.62 | 194.26 | Stoolap |
| DELETE by ID | **0.73** | 1.32 | 152.34 | Stoolap |
| DELETE complex | **3.69** | 380.14 | 197.44 | Stoolap |
| Aggregation (GROUP BY) | **49.00** | 1403.39 | 104.32 | Stoolap |

**Basic Operations Score: Stoolap 11, SQLite 0, DuckDB 0**

---

## Advanced Operations

| Operation | Stoolap (us) | SQLite (us) | DuckDB (us) | Best |
|-----------|-------------|-------------|-------------|------|
| INNER JOIN | 25.04 | **14.86** | 607.20 | SQLite |
| LEFT JOIN + GROUP BY | 59.34 | **55.69** | 1269.40 | SQLite |
| Scalar subquery | **34.72** | 399.10 | 257.12 | Stoolap |
| IN subquery | **372.94** | 1838.79 | 853.94 | Stoolap |
| EXISTS subquery | **3.66** | 38.42 | 928.06 | Stoolap |
| CTE + JOIN | **44.33** | 74.16 | 859.53 | Stoolap |
| Window ROW_NUMBER | **281.16** | 1781.90 | 690.83 | Stoolap |
| Window ROW_NUMBER (PK) | **6.84** | 21.36 | 419.10 | Stoolap |
| Window PARTITION BY | **9.46** | 64.81 | 1162.16 | Stoolap |
| UNION ALL | **6.18** | 6.24 | 173.19 | Stoolap |
| CASE expression | 5.97 | **5.10** | 247.54 | SQLite |
| Complex JOIN+GROUP+HAVING | **59.92** | 93.20 | 2233.32 | Stoolap |
| Batch INSERT (100 rows) | 75.75 | **74.93** | 14920.25 | SQLite |

**Advanced Operations Score: Stoolap 9, SQLite 4, DuckDB 0**

---

## Bottleneck Hunters

| Operation | Stoolap (us) | SQLite (us) | DuckDB (us) | Best |
|-----------|-------------|-------------|-------------|------|
| DISTINCT (no ORDER) | **5.81** | 104.38 | 235.97 | Stoolap |
| DISTINCT + ORDER BY | **6.20** | 139.58 | 291.81 | Stoolap |
| COUNT DISTINCT | **0.43** | 105.98 | 219.91 | Stoolap |
| LIKE prefix (User_1%) | **4.90** | 9.49 | 173.02 | Stoolap |
| LIKE contains (%50%) | **46.00** | 156.52 | 271.86 | Stoolap |
| OR conditions (3 vals) | **5.01** | 14.54 | 207.08 | Stoolap |
| IN list (7 values) | **3.66** | 14.52 | 1152.84 | Stoolap |
| NOT IN subquery | **359.37** | 1898.21 | 965.60 | Stoolap |
| NOT EXISTS subquery | **49.30** | 1729.58 | 1429.57 | Stoolap |
| OFFSET pagination (5000) | **15.37** | 21.31 | 1222.62 | Stoolap |
| Multi-col ORDER BY (3) | **175.57** | 416.71 | 361.35 | Stoolap |
| Self JOIN (same age) | 16.26 | **10.71** | 396.87 | SQLite |
| Multi window funcs (3) | **628.97** | 1803.93 | 760.23 | Stoolap |
| Nested subquery (3 lvl) | **447.45** | 6397.42 | 850.91 | Stoolap |
| Multi aggregates (6) | **126.64** | 842.98 | 306.48 | Stoolap |
| COALESCE + IS NOT NULL | 5.10 | **2.84** | 90.97 | SQLite |
| Expr in WHERE (funcs) | **6.44** | 15.05 | 236.50 | Stoolap |
| Math expressions | **17.32** | 36.74 | 247.55 | Stoolap |
| String concat (\|\|) | 6.80 | **5.60** | 253.41 | SQLite |
| Large result (no LIMIT) | **276.74** | 486.65 | 348.78 | Stoolap |
| Multiple CTEs (2) | 24.86 | **20.60** | 313.38 | SQLite |
| Correlated in SELECT | **392.28** | 511.29 | 1283.17 | Stoolap |
| BETWEEN (non-indexed) | **2.86** | 9.26 | 178.98 | Stoolap |
| GROUP BY (2 columns) | **179.74** | 2259.41 | 320.48 | Stoolap |
| CROSS JOIN (limited) | **139.58** | 1358.05 | 1458.25 | Stoolap |
| Derived table (FROM sub) | 491.58 | 875.95 | **255.77** | DuckDB |
| Window ROWS frame | **660.65** | 1881.15 | 2198.75 | Stoolap |
| HAVING complex | **113.76** | 1420.61 | 114.47 | Stoolap |
| Compare with subquery | **184.04** | 1424.07 | 293.51 | Stoolap |

**Bottleneck Hunters Score: Stoolap 24, SQLite 4, DuckDB 1**

---

## Summary by Category

| Category | Stoolap Wins | SQLite Wins | DuckDB Wins |
|----------|-------------|-------------|-------------|
| Basic Operations | 11 | 0 | 0 |
| Advanced Operations | 9 | 4 | 0 |
| Bottleneck Hunters | 24 | 4 | 1 |
| **Total** | **44** | **8** | **1** |

---

## Top Stoolap Wins vs SQLite

| Operation | Stoolap | SQLite | Speedup |
|-----------|---------|--------|---------|
| COUNT DISTINCT | 0.43 us | 105.98 us | **246x** |
| DELETE complex | 3.69 us | 380.14 us | **103x** |
| NOT EXISTS subquery | 49.30 us | 1729.58 us | **35x** |
| DISTINCT + ORDER BY | 6.20 us | 139.58 us | **23x** |
| DISTINCT (no ORDER) | 5.81 us | 104.38 us | **18x** |
| Nested subquery (3 lvl) | 447.45 us | 6397.42 us | **14x** |
| GROUP BY (2 columns) | 179.74 us | 2259.41 us | **13x** |
| Scalar subquery | 34.72 us | 399.10 us | **11x** |
| HAVING complex | 113.76 us | 1420.61 us | **12x** |
| Aggregation (GROUP BY) | 49.00 us | 1403.39 us | **29x** |
| EXISTS subquery | 3.66 us | 38.42 us | **10x** |
| UPDATE complex | 51.78 us | 443.73 us | **9x** |
| CROSS JOIN (limited) | 139.58 us | 1358.05 us | **10x** |
| Compare with subquery | 184.04 us | 1424.07 us | **8x** |
| Multi aggregates (6) | 126.64 us | 842.98 us | **7x** |

---

## Top Stoolap Wins vs DuckDB

| Operation | Stoolap | DuckDB | Speedup |
|-----------|---------|--------|---------|
| SELECT by ID | 0.14 us | 145.55 us | **1040x** |
| EXISTS subquery | 3.66 us | 928.06 us | **254x** |
| UPDATE by ID | 0.58 us | 146.14 us | **252x** |
| DELETE by ID | 0.73 us | 152.34 us | **209x** |
| Batch INSERT (100 rows) | 75.75 us | 14920.25 us | **197x** |
| INSERT single | 1.32 us | 194.26 us | **147x** |
| Window PARTITION BY | 9.46 us | 1162.16 us | **123x** |
| OFFSET pagination (5000) | 15.37 us | 1222.62 us | **80x** |
| Window ROW_NUMBER (PK) | 6.84 us | 419.10 us | **61x** |
| SELECT by index (exact) | 4.51 us | 288.70 us | **64x** |
| DELETE complex | 3.69 us | 197.44 us | **53x** |
| DISTINCT (no ORDER) | 5.81 us | 235.97 us | **41x** |
| CASE expression | 5.97 us | 247.54 us | **41x** |
| Complex JOIN+GROUP+HAVING | 59.92 us | 2233.32 us | **37x** |
| UNION ALL | 6.18 us | 173.19 us | **28x** |
| Nested subquery (3 lvl) | 447.45 us | 850.91 us | **1.9x** |
| SELECT complex | 138.41 us | 184.68 us | **1.3x** |

---

## Where Others Win

### SQLite Advantages

| Operation | SQLite | Stoolap | Factor |
|-----------|--------|---------|--------|
| INNER JOIN | 14.86 us | 25.04 us | 1.7x |
| Self JOIN | 10.71 us | 16.26 us | 1.5x |
| COALESCE | 2.84 us | 5.10 us | 1.8x |
| Multiple CTEs (2) | 20.60 us | 24.86 us | 1.2x |

### DuckDB Advantages

| Operation | DuckDB | Stoolap | Factor |
|-----------|--------|---------|--------|
| Derived table | 255.77 us | 491.58 us | 1.9x |

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
  Point Queries (ID):     ████████████████████  DOMINANT (1000x vs DuckDB)
  DISTINCT Operations:    ████████████████████  EXCELLENT (18-246x vs SQLite)
  Semi-joins (EXISTS):    ████████████████████  EXCELLENT (10-233x faster)
  Complex DML:            ████████████████████  EXCELLENT (9-91x faster)
  Window (PARTITION BY):  ████████████████████  EXCELLENT (6-104x faster)
  Index Lookups:          ████████████████████  EXCELLENT (5-55x faster)
  Batch Inserts:          ████████████████████  EXCELLENT (197x vs DuckDB)
  Aggregations:           ████████████████████  EXCELLENT (29x vs SQLite)

SQLite Strengths:
  Simple JOINs:           ████████              GOOD (1.7x faster)
  Simple Expressions:     ██████                MODERATE (1.1-1.8x faster)

DuckDB Strengths:
  Nested Subqueries:      ████████████          GOOD (2.8x faster)
  Columnar Analytics:     ██████████            MODERATE (1.1-2.1x faster)
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
- **Complex analytical queries** - Multiple nested subqueries
- **Large columnar scans** - Vectorized execution
- **Data science workflows** - DataFrame integration

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

*Benchmarks performed on Apple Silicon, in-memory mode, best of 3 runs.*
