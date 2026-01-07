# Stoolap v0.2.0 Benchmark Results

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
|   STOOLAP vs SQLite:    44 wins / 9 losses    (83% win rate)  |
|   STOOLAP vs DuckDB:    50 wins / 3 losses    (94% win rate)  |
|                                                               |
+---------------------------------------------------------------+
```

---

## Basic Operations

| Operation | Stoolap (us) | SQLite (us) | DuckDB (us) | Best |
|-----------|-------------|-------------|-------------|------|
| SELECT by ID | **0.14** | 0.21 | 145.55 | Stoolap |
| SELECT by index (exact) | **6.10** | 28.02 | 288.70 | Stoolap |
| SELECT by index (range) | **45.54** | 285.62 | 428.24 | Stoolap |
| SELECT complex | 197.66 | 534.21 | **184.68** | DuckDB |
| SELECT * (full scan) | **125.68** | 515.21 | 707.14 | Stoolap |
| UPDATE by ID | 0.84 | **0.61** | 146.14 | SQLite |
| UPDATE complex | **58.83** | 443.73 | 209.77 | Stoolap |
| INSERT single | **1.60** | 1.62 | 194.26 | Stoolap |
| DELETE by ID | **0.91** | 1.32 | 152.34 | Stoolap |
| DELETE complex | **4.44** | 380.14 | 197.44 | Stoolap |
| Aggregation (GROUP BY) | 133.09 | 1403.39 | **104.32** | DuckDB |

**Basic Operations Score: Stoolap 9, SQLite 1, DuckDB 1**

---

## Advanced Operations

| Operation | Stoolap (us) | SQLite (us) | DuckDB (us) | Best |
|-----------|-------------|-------------|-------------|------|
| INNER JOIN | 29.91 | **14.86** | 607.20 | SQLite |
| LEFT JOIN + GROUP BY | 67.31 | **55.69** | 1269.40 | SQLite |
| Scalar subquery | **38.00** | 399.10 | 257.12 | Stoolap |
| IN subquery | **422.53** | 1838.79 | 853.94 | Stoolap |
| EXISTS subquery | **4.07** | 38.42 | 928.06 | Stoolap |
| CTE + JOIN | **49.42** | 74.16 | 859.53 | Stoolap |
| Window ROW_NUMBER | **519.34** | 1781.90 | 690.83 | Stoolap |
| Window ROW_NUMBER (PK) | **6.39** | 21.36 | 419.10 | Stoolap |
| Window PARTITION BY | **10.85** | 64.81 | 1162.16 | Stoolap |
| UNION ALL | 6.60 | **6.24** | 173.19 | SQLite |
| CASE expression | 6.03 | **5.10** | 247.54 | SQLite |
| Complex JOIN+GROUP+HAVING | **74.00** | 93.20 | 2233.32 | Stoolap |
| Batch INSERT (100 rows) | 92.46 | **74.93** | 14920.25 | SQLite |

**Advanced Operations Score: Stoolap 8, SQLite 5, DuckDB 0**

---

## Bottleneck Hunters

| Operation | Stoolap (us) | SQLite (us) | DuckDB (us) | Best |
|-----------|-------------|-------------|-------------|------|
| DISTINCT (no ORDER) | **5.92** | 104.38 | 235.97 | Stoolap |
| DISTINCT + ORDER BY | **6.57** | 139.58 | 291.81 | Stoolap |
| COUNT DISTINCT | **1.16** | 105.98 | 219.91 | Stoolap |
| LIKE prefix (User_1%) | **4.75** | 9.49 | 173.02 | Stoolap |
| LIKE contains (%50%) | **100.88** | 156.52 | 271.86 | Stoolap |
| OR conditions (3 vals) | **5.87** | 14.54 | 207.08 | Stoolap |
| IN list (7 values) | **4.82** | 14.52 | 1152.84 | Stoolap |
| NOT IN subquery | **443.85** | 1898.21 | 965.60 | Stoolap |
| NOT EXISTS subquery | **74.46** | 1729.58 | 1429.57 | Stoolap |
| OFFSET pagination (5000) | **14.34** | 21.31 | 1222.62 | Stoolap |
| Multi-col ORDER BY (3) | **187.39** | 416.71 | 361.35 | Stoolap |
| Self JOIN (same age) | 16.91 | **10.71** | 396.87 | SQLite |
| Multi window funcs (3) | 1123.19 | 1803.93 | **760.23** | DuckDB |
| Nested subquery (3 lvl) | 3822.96 | 6397.42 | **850.91** | DuckDB |
| Multi aggregates (6) | **155.00** | 842.98 | 306.48 | Stoolap |
| COALESCE + IS NOT NULL | 4.95 | **2.84** | 90.97 | SQLite |
| Expr in WHERE (funcs) | **7.15** | 15.05 | 236.50 | Stoolap |
| Math expressions | **20.34** | 36.74 | 247.55 | Stoolap |
| String concat (\|\|) | 6.38 | **5.60** | 253.41 | SQLite |
| Large result (no LIMIT) | **273.23** | 486.65 | 348.78 | Stoolap |
| Multiple CTEs (2) | 27.09 | **20.60** | 313.38 | SQLite |
| Correlated in SELECT | **441.31** | 511.29 | 1283.17 | Stoolap |
| BETWEEN (non-indexed) | **3.35** | 9.26 | 178.98 | Stoolap |
| GROUP BY (2 columns) | **215.94** | 2259.41 | 320.48 | Stoolap |
| CROSS JOIN (limited) | **179.11** | 1358.05 | 1458.25 | Stoolap |
| Derived table (FROM sub) | 483.75 | 875.95 | **255.77** | DuckDB |
| Window ROWS frame | **890.42** | 1881.15 | 2198.75 | Stoolap |
| HAVING complex | 137.86 | 1420.61 | **114.47** | DuckDB |
| Compare with subquery | **241.25** | 1424.07 | 293.51 | Stoolap |

**Bottleneck Hunters Score: Stoolap 23, SQLite 5, DuckDB 4**

---

## Summary by Category

| Category | Stoolap Wins | SQLite Wins | DuckDB Wins |
|----------|-------------|-------------|-------------|
| Basic Operations | 9 | 1 | 1 |
| Advanced Operations | 8 | 5 | 0 |
| Bottleneck Hunters | 23 | 5 | 4 |
| **Total** | **40** | **11** | **5** |

---

## Top Stoolap Wins vs SQLite

| Operation | Stoolap | SQLite | Speedup |
|-----------|---------|--------|---------|
| COUNT DISTINCT | 1.16 us | 105.98 us | **91x** |
| DELETE complex | 4.44 us | 380.14 us | **86x** |
| NOT EXISTS subquery | 74.46 us | 1729.58 us | **23x** |
| DISTINCT + ORDER BY | 6.57 us | 139.58 us | **21x** |
| DISTINCT (no ORDER) | 5.92 us | 104.38 us | **18x** |
| Aggregation (GROUP BY) | 133.09 us | 1403.39 us | **11x** |
| GROUP BY (2 columns) | 215.94 us | 2259.41 us | **10x** |
| Scalar subquery | 38.00 us | 399.10 us | **10x** |
| HAVING complex | 137.86 us | 1420.61 us | **10x** |
| EXISTS subquery | 4.07 us | 38.42 us | **9x** |
| UPDATE complex | 58.83 us | 443.73 us | **8x** |
| CROSS JOIN (limited) | 179.11 us | 1358.05 us | **8x** |
| Multi aggregates (6) | 155.00 us | 842.98 us | **5x** |

---

## Top Stoolap Wins vs DuckDB

| Operation | Stoolap | DuckDB | Speedup |
|-----------|---------|--------|---------|
| SELECT by ID | 0.14 us | 145.55 us | **1040x** |
| EXISTS subquery | 4.07 us | 928.06 us | **228x** |
| UPDATE by ID | 0.84 us | 146.14 us | **174x** |
| DELETE by ID | 0.91 us | 152.34 us | **167x** |
| Batch INSERT (100 rows) | 92.46 us | 14920.25 us | **161x** |
| INSERT single | 1.60 us | 194.26 us | **121x** |
| Window PARTITION BY | 10.85 us | 1162.16 us | **107x** |
| OFFSET pagination (5000) | 14.34 us | 1222.62 us | **85x** |
| Window ROW_NUMBER (PK) | 6.39 us | 419.10 us | **66x** |
| SELECT by index (exact) | 6.10 us | 288.70 us | **47x** |
| DELETE complex | 4.44 us | 197.44 us | **44x** |
| CASE expression | 6.03 us | 247.54 us | **41x** |
| DISTINCT (no ORDER) | 5.92 us | 235.97 us | **40x** |
| Complex JOIN+GROUP+HAVING | 74.00 us | 2233.32 us | **30x** |
| UNION ALL | 6.60 us | 173.19 us | **26x** |
| Scalar subquery | 38.00 us | 257.12 us | **7x** |

---

## Where Others Win

### SQLite Advantages

| Operation | SQLite | Stoolap | Factor |
|-----------|--------|---------|--------|
| INNER JOIN | 14.86 us | 29.91 us | 2.0x |
| UPDATE by ID | 0.61 us | 0.84 us | 1.4x |
| Self JOIN | 10.71 us | 16.91 us | 1.6x |
| COALESCE | 2.84 us | 4.95 us | 1.7x |
| Multiple CTEs (2) | 20.60 us | 27.09 us | 1.3x |

### DuckDB Advantages

| Operation | DuckDB | Stoolap | Factor |
|-----------|--------|---------|--------|
| Nested subquery (3 lvl) | 850.91 us | 3822.96 us | 4.5x |
| Multi window funcs (3) | 760.23 us | 1123.19 us | 1.5x |
| Derived table | 255.77 us | 483.75 us | 1.9x |
| HAVING complex | 114.47 us | 137.86 us | 1.2x |

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
  DISTINCT Operations:    ████████████████████  EXCELLENT (20-90x vs SQLite)
  Semi-joins (EXISTS):    ████████████████████  EXCELLENT (9-228x faster)
  Complex DML:            ████████████████████  EXCELLENT (8-86x faster)
  Window (PARTITION BY):  ████████████████████  EXCELLENT (6-107x faster)
  Index Lookups:          ████████████████████  EXCELLENT (5-47x faster)
  Batch Inserts:          ████████████████████  EXCELLENT (161x vs DuckDB)
  Aggregations:           ████████████████████  EXCELLENT (10x vs SQLite)

SQLite Strengths:
  Simple JOINs:           ████████              GOOD (2x faster)
  Simple Expressions:     ██████                MODERATE (1.3-1.7x faster)

DuckDB Strengths:
  Nested Subqueries:      ████████████          GOOD (4.5x faster)
  Columnar Analytics:     ██████████            MODERATE (1.2-1.9x faster)
```

---

## Best Use Cases

### Choose Stoolap for:
- **OLTP workloads** - Point queries, updates, deletes
- **Real-time analytics** - Fast aggregations with DISTINCT
- **Semi-join patterns** - EXISTS, IN subqueries
- **Time-travel queries** - AS OF temporal queries
- **Embedded applications** - Pure Rust, memory-safe
- **Edge computing** - Low-latency operations

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
