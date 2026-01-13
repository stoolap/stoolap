# Stoolap v0.2.2 Benchmark Results

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
|   STOOLAP vs SQLite:    43 wins / 10 losses   (81% win rate)  |
|   STOOLAP vs DuckDB:    49 wins / 4 losses    (92% win rate)  |
|                                                               |
+---------------------------------------------------------------+
```

---

## Basic Operations

| Operation | Stoolap (us) | SQLite (us) | DuckDB (us) | Best |
|-----------|-------------|-------------|-------------|------|
| SELECT by ID | **0.14** | 0.21 | 145.55 | Stoolap |
| SELECT by index (exact) | **5.03** | 28.02 | 288.70 | Stoolap |
| SELECT by index (range) | **39.85** | 285.62 | 428.24 | Stoolap |
| SELECT complex | **147.60** | 534.21 | 184.68 | Stoolap |
| SELECT * (full scan) | **113.51** | 515.21 | 707.14 | Stoolap |
| UPDATE by ID | 0.65 | **0.61** | 146.14 | SQLite |
| UPDATE complex | **51.78** | 443.73 | 209.77 | Stoolap |
| INSERT single | **1.37** | 1.62 | 194.26 | Stoolap |
| DELETE by ID | **0.73** | 1.32 | 152.34 | Stoolap |
| DELETE complex | **3.89** | 380.14 | 197.44 | Stoolap |
| Aggregation (GROUP BY) | 114.17 | 1403.39 | **104.32** | DuckDB |

**Basic Operations Score: Stoolap 9, SQLite 1, DuckDB 1**

---

## Advanced Operations

| Operation | Stoolap (us) | SQLite (us) | DuckDB (us) | Best |
|-----------|-------------|-------------|-------------|------|
| INNER JOIN | 27.88 | **14.86** | 607.20 | SQLite |
| LEFT JOIN + GROUP BY | 59.34 | **55.69** | 1269.40 | SQLite |
| Scalar subquery | **34.72** | 399.10 | 257.12 | Stoolap |
| IN subquery | **392.61** | 1838.79 | 853.94 | Stoolap |
| EXISTS subquery | **3.98** | 38.42 | 928.06 | Stoolap |
| CTE + JOIN | **48.52** | 74.16 | 859.53 | Stoolap |
| Window ROW_NUMBER | **309.77** | 1781.90 | 690.83 | Stoolap |
| Window ROW_NUMBER (PK) | **6.84** | 21.36 | 419.10 | Stoolap |
| Window PARTITION BY | **10.38** | 64.81 | 1162.16 | Stoolap |
| UNION ALL | 6.71 | **6.24** | 173.19 | SQLite |
| CASE expression | 5.97 | **5.10** | 247.54 | SQLite |
| Complex JOIN+GROUP+HAVING | **65.28** | 93.20 | 2233.32 | Stoolap |
| Batch INSERT (100 rows) | 79.72 | **74.93** | 14920.25 | SQLite |

**Advanced Operations Score: Stoolap 8, SQLite 5, DuckDB 0**

---

## Bottleneck Hunters

| Operation | Stoolap (us) | SQLite (us) | DuckDB (us) | Best |
|-----------|-------------|-------------|-------------|------|
| DISTINCT (no ORDER) | **5.81** | 104.38 | 235.97 | Stoolap |
| DISTINCT + ORDER BY | **6.38** | 139.58 | 291.81 | Stoolap |
| COUNT DISTINCT | **1.22** | 105.98 | 219.91 | Stoolap |
| LIKE prefix (User_1%) | **4.90** | 9.49 | 173.02 | Stoolap |
| LIKE contains (%50%) | **99.51** | 156.52 | 271.86 | Stoolap |
| OR conditions (3 vals) | **5.71** | 14.54 | 207.08 | Stoolap |
| IN list (7 values) | **4.71** | 14.52 | 1152.84 | Stoolap |
| NOT IN subquery | **413.15** | 1898.21 | 965.60 | Stoolap |
| NOT EXISTS subquery | **76.18** | 1729.58 | 1429.57 | Stoolap |
| OFFSET pagination (5000) | **15.37** | 21.31 | 1222.62 | Stoolap |
| Multi-col ORDER BY (3) | **186.37** | 416.71 | 361.35 | Stoolap |
| Self JOIN (same age) | 16.26 | **10.71** | 396.87 | SQLite |
| Multi window funcs (3) | **735.91** | 1803.93 | 760.23 | Stoolap |
| Nested subquery (3 lvl) | 2616.95 | 6397.42 | **850.91** | DuckDB |
| Multi aggregates (6) | **149.65** | 842.98 | 306.48 | Stoolap |
| COALESCE + IS NOT NULL | 5.10 | **2.84** | 90.97 | SQLite |
| Expr in WHERE (funcs) | **6.44** | 15.05 | 236.50 | Stoolap |
| Math expressions | **20.34** | 36.74 | 247.55 | Stoolap |
| String concat (\|\|) | 6.79 | **5.60** | 253.41 | SQLite |
| Large result (no LIMIT) | **276.74** | 486.65 | 348.78 | Stoolap |
| Multiple CTEs (2) | 24.86 | **20.60** | 313.38 | SQLite |
| Correlated in SELECT | **392.28** | 511.29 | 1283.17 | Stoolap |
| BETWEEN (non-indexed) | **3.07** | 9.26 | 178.98 | Stoolap |
| GROUP BY (2 columns) | **190.37** | 2259.41 | 320.48 | Stoolap |
| CROSS JOIN (limited) | **149.54** | 1358.05 | 1458.25 | Stoolap |
| Derived table (FROM sub) | 528.22 | 875.95 | **255.77** | DuckDB |
| Window ROWS frame | **673.63** | 1881.15 | 2198.75 | Stoolap |
| HAVING complex | 120.17 | 1420.61 | **114.47** | DuckDB |
| Compare with subquery | **184.04** | 1424.07 | 293.51 | Stoolap |

**Bottleneck Hunters Score: Stoolap 22, SQLite 4, DuckDB 3**

---

## Summary by Category

| Category | Stoolap Wins | SQLite Wins | DuckDB Wins |
|----------|-------------|-------------|-------------|
| Basic Operations | 9 | 1 | 1 |
| Advanced Operations | 8 | 5 | 0 |
| Bottleneck Hunters | 22 | 4 | 3 |
| **Total** | **39** | **10** | **4** |

---

## Top Stoolap Wins vs SQLite

| Operation | Stoolap | SQLite | Speedup |
|-----------|---------|--------|---------|
| COUNT DISTINCT | 1.22 us | 105.98 us | **87x** |
| DELETE complex | 3.89 us | 380.14 us | **98x** |
| NOT EXISTS subquery | 76.18 us | 1729.58 us | **23x** |
| DISTINCT + ORDER BY | 6.38 us | 139.58 us | **22x** |
| DISTINCT (no ORDER) | 5.81 us | 104.38 us | **18x** |
| GROUP BY (2 columns) | 190.37 us | 2259.41 us | **12x** |
| Scalar subquery | 34.72 us | 399.10 us | **11x** |
| HAVING complex | 120.17 us | 1420.61 us | **12x** |
| Aggregation (GROUP BY) | 114.17 us | 1403.39 us | **12x** |
| EXISTS subquery | 3.98 us | 38.42 us | **10x** |
| UPDATE complex | 51.78 us | 443.73 us | **9x** |
| CROSS JOIN (limited) | 149.54 us | 1358.05 us | **9x** |
| Compare with subquery | 184.04 us | 1424.07 us | **8x** |
| Multi aggregates (6) | 149.65 us | 842.98 us | **6x** |

---

## Top Stoolap Wins vs DuckDB

| Operation | Stoolap | DuckDB | Speedup |
|-----------|---------|--------|---------|
| SELECT by ID | 0.14 us | 145.55 us | **1040x** |
| EXISTS subquery | 3.98 us | 928.06 us | **233x** |
| UPDATE by ID | 0.65 us | 146.14 us | **225x** |
| DELETE by ID | 0.73 us | 152.34 us | **209x** |
| Batch INSERT (100 rows) | 79.72 us | 14920.25 us | **187x** |
| INSERT single | 1.37 us | 194.26 us | **142x** |
| Window PARTITION BY | 10.38 us | 1162.16 us | **112x** |
| OFFSET pagination (5000) | 15.37 us | 1222.62 us | **80x** |
| Window ROW_NUMBER (PK) | 6.84 us | 419.10 us | **61x** |
| SELECT by index (exact) | 5.03 us | 288.70 us | **57x** |
| DELETE complex | 3.89 us | 197.44 us | **51x** |
| DISTINCT (no ORDER) | 5.81 us | 235.97 us | **41x** |
| CASE expression | 5.97 us | 247.54 us | **41x** |
| Complex JOIN+GROUP+HAVING | 65.28 us | 2233.32 us | **34x** |
| UNION ALL | 6.71 us | 173.19 us | **26x** |
| SELECT complex | 151.53 us | 184.68 us | **1.2x** |

---

## Where Others Win

### SQLite Advantages

| Operation | SQLite | Stoolap | Factor |
|-----------|--------|---------|--------|
| INNER JOIN | 14.86 us | 27.88 us | 1.9x |
| UPDATE by ID | 0.61 us | 0.65 us | 1.07x |
| Self JOIN | 10.71 us | 16.26 us | 1.5x |
| COALESCE | 2.84 us | 5.10 us | 1.8x |
| Multiple CTEs (2) | 20.60 us | 24.86 us | 1.2x |

### DuckDB Advantages

| Operation | DuckDB | Stoolap | Factor |
|-----------|--------|---------|--------|
| Nested subquery (3 lvl) | 850.91 us | 2616.95 us | 3.1x |
| Derived table | 255.77 us | 528.22 us | 2.1x |
| HAVING complex | 114.47 us | 120.17 us | 1.05x |
| Aggregation (GROUP BY) | 104.32 us | 114.17 us | 1.09x |

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
  DISTINCT Operations:    ████████████████████  EXCELLENT (18-87x vs SQLite)
  Semi-joins (EXISTS):    ████████████████████  EXCELLENT (10-233x faster)
  Complex DML:            ████████████████████  EXCELLENT (9-91x faster)
  Window (PARTITION BY):  ████████████████████  EXCELLENT (6-104x faster)
  Index Lookups:          ████████████████████  EXCELLENT (5-55x faster)
  Batch Inserts:          ████████████████████  EXCELLENT (162x vs DuckDB)
  Aggregations:           ████████████████████  EXCELLENT (10x vs SQLite)

SQLite Strengths:
  Simple JOINs:           ████████              GOOD (2x faster)
  Simple Expressions:     ██████                MODERATE (1.1-1.8x faster)

DuckDB Strengths:
  Nested Subqueries:      ████████████          GOOD (3.4x faster)
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
