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
| SELECT by ID | **0.13** | 0.21 | 145.55 | Stoolap |
| SELECT by index (exact) | **4.39** | 28.02 | 288.70 | Stoolap |
| SELECT by index (range) | **33.33** | 285.62 | 428.24 | Stoolap |
| SELECT complex | **122.10** | 534.21 | 184.68 | Stoolap |
| SELECT * (full scan) | **91.12** | 515.21 | 707.14 | Stoolap |
| UPDATE by ID | **0.60** | 0.61 | 146.14 | Stoolap |
| UPDATE complex | **57.04** | 443.73 | 209.77 | Stoolap |
| INSERT single | **1.26** | 1.62 | 194.26 | Stoolap |
| DELETE by ID | **0.73** | 1.32 | 152.34 | Stoolap |
| DELETE complex | **3.56** | 380.14 | 197.44 | Stoolap |
| Aggregation (GROUP BY) | **49.17** | 1403.39 | 104.32 | Stoolap |

**Basic Operations Score: Stoolap 11, SQLite 0, DuckDB 0**

---

## Advanced Operations

| Operation | Stoolap (us) | SQLite (us) | DuckDB (us) | Best |
|-----------|-------------|-------------|-------------|------|
| INNER JOIN | 25.04 | **14.86** | 607.20 | SQLite |
| LEFT JOIN + GROUP BY | 59.34 | **55.69** | 1269.40 | SQLite |
| Scalar subquery | **9.32** | 399.10 | 257.12 | Stoolap |
| IN subquery | **119.80** | 1838.79 | 853.94 | Stoolap |
| EXISTS subquery | **3.38** | 38.42 | 928.06 | Stoolap |
| CTE + JOIN | **43.00** | 74.16 | 859.53 | Stoolap |
| Window ROW_NUMBER | **257.52** | 1781.90 | 690.83 | Stoolap |
| Window ROW_NUMBER (PK) | **6.29** | 21.36 | 419.10 | Stoolap |
| Window PARTITION BY | **8.80** | 64.81 | 1162.16 | Stoolap |
| UNION ALL | **5.87** | 6.24 | 173.19 | Stoolap |
| CASE expression | 5.43 | **5.10** | 247.54 | SQLite |
| Complex JOIN+GROUP+HAVING | **58.87** | 93.20 | 2233.32 | Stoolap |
| Batch INSERT (100 rows) | 75.75 | **74.93** | 14920.25 | SQLite |

**Advanced Operations Score: Stoolap 9, SQLite 4, DuckDB 0**

---

## Bottleneck Hunters

| Operation | Stoolap (us) | SQLite (us) | DuckDB (us) | Best |
|-----------|-------------|-------------|-------------|------|
| DISTINCT (no ORDER) | **4.75** | 104.38 | 235.97 | Stoolap |
| DISTINCT + ORDER BY | **5.15** | 139.58 | 291.81 | Stoolap |
| COUNT DISTINCT | **0.43** | 105.98 | 219.91 | Stoolap |
| LIKE prefix (User_1%) | **4.23** | 9.49 | 173.02 | Stoolap |
| LIKE contains (%50%) | **35.81** | 156.52 | 271.86 | Stoolap |
| OR conditions (3 vals) | **4.69** | 14.54 | 207.08 | Stoolap |
| IN list (7 values) | **3.49** | 14.52 | 1152.84 | Stoolap |
| NOT IN subquery | **88.32** | 1898.21 | 965.60 | Stoolap |
| NOT EXISTS subquery | **25.39** | 1729.58 | 1429.57 | Stoolap |
| OFFSET pagination (5000) | **15.14** | 21.31 | 1222.62 | Stoolap |
| Multi-col ORDER BY (3) | **137.78** | 416.71 | 361.35 | Stoolap |
| Self JOIN (same age) | 14.18 | **10.71** | 396.87 | SQLite |
| Multi window funcs (3) | **569.05** | 1803.93 | 760.23 | Stoolap |
| Nested subquery (3 lvl) | **353.29** | 6397.42 | 850.91 | Stoolap |
| Multi aggregates (6) | **117.81** | 842.98 | 306.48 | Stoolap |
| COALESCE + IS NOT NULL | 4.39 | **2.84** | 90.97 | SQLite |
| Expr in WHERE (funcs) | **5.69** | 15.05 | 236.50 | Stoolap |
| Math expressions | **16.29** | 36.74 | 247.55 | Stoolap |
| String concat (\|\|) | 6.80 | **5.60** | 253.41 | SQLite |
| Large result (no LIMIT) | **254.77** | 486.65 | 348.78 | Stoolap |
| Multiple CTEs (2) | 21.94 | **20.60** | 313.38 | SQLite |
| Correlated in SELECT | **297.46** | 511.29 | 1283.17 | Stoolap |
| BETWEEN (non-indexed) | **2.68** | 9.26 | 178.98 | Stoolap |
| GROUP BY (2 columns) | **155.01** | 2259.41 | 320.48 | Stoolap |
| CROSS JOIN (limited) | **104.07** | 1358.05 | 1458.25 | Stoolap |
| Derived table (FROM sub) | 406.82 | 875.95 | **255.77** | DuckDB |
| Window ROWS frame | **543.45** | 1881.15 | 2198.75 | Stoolap |
| HAVING complex | **92.37** | 1420.61 | 114.47 | Stoolap |
| Compare with subquery | **5.25** | 1424.07 | 293.51 | Stoolap |

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
| Compare with subquery | 5.25 us | 1424.07 us | **271x** |
| COUNT DISTINCT | 0.43 us | 105.98 us | **246x** |
| DELETE complex | 3.56 us | 380.14 us | **107x** |
| NOT EXISTS subquery | 25.39 us | 1729.58 us | **68x** |
| Scalar subquery | 9.32 us | 399.10 us | **43x** |
| Aggregation (GROUP BY) | 49.17 us | 1403.39 us | **29x** |
| DISTINCT + ORDER BY | 5.15 us | 139.58 us | **27x** |
| DISTINCT (no ORDER) | 4.75 us | 104.38 us | **22x** |
| NOT IN subquery | 88.32 us | 1898.21 us | **21x** |
| Nested subquery (3 lvl) | 353.29 us | 6397.42 us | **18x** |
| IN subquery | 119.80 us | 1838.79 us | **15x** |
| GROUP BY (2 columns) | 155.01 us | 2259.41 us | **15x** |
| HAVING complex | 92.37 us | 1420.61 us | **15x** |
| EXISTS subquery | 3.38 us | 38.42 us | **11x** |

---

## Top Stoolap Wins vs DuckDB

| Operation | Stoolap | DuckDB | Speedup |
|-----------|---------|--------|---------|
| SELECT by ID | 0.13 us | 145.55 us | **1120x** |
| EXISTS subquery | 3.38 us | 928.06 us | **275x** |
| UPDATE by ID | 0.60 us | 146.14 us | **244x** |
| DELETE by ID | 0.73 us | 152.34 us | **209x** |
| Batch INSERT (100 rows) | 75.75 us | 14920.25 us | **197x** |
| INSERT single | 1.26 us | 194.26 us | **154x** |
| Window PARTITION BY | 8.80 us | 1162.16 us | **132x** |
| OFFSET pagination (5000) | 15.14 us | 1222.62 us | **81x** |
| Window ROW_NUMBER (PK) | 6.29 us | 419.10 us | **67x** |
| SELECT by index (exact) | 4.39 us | 288.70 us | **66x** |
| Compare with subquery | 5.25 us | 293.51 us | **56x** |
| NOT EXISTS subquery | 25.39 us | 1429.57 us | **56x** |
| DELETE complex | 3.56 us | 197.44 us | **55x** |
| DISTINCT (no ORDER) | 4.75 us | 235.97 us | **50x** |
| CASE expression | 5.43 us | 247.54 us | **46x** |
| Complex JOIN+GROUP+HAVING | 58.87 us | 2233.32 us | **38x** |
| UNION ALL | 5.87 us | 173.19 us | **30x** |
| Scalar subquery | 9.32 us | 257.12 us | **28x** |
| NOT IN subquery | 88.32 us | 965.60 us | **11x** |
| IN subquery | 119.80 us | 853.94 us | **7.1x** |
| Nested subquery (3 lvl) | 353.29 us | 850.91 us | **2.4x** |
| SELECT complex | 122.10 us | 184.68 us | **1.5x** |

---

## Where Others Win

### SQLite Advantages

| Operation | SQLite | Stoolap | Factor |
|-----------|--------|---------|--------|
| INNER JOIN | 14.86 us | 25.04 us | 1.7x |
| COALESCE | 2.84 us | 4.39 us | 1.5x |
| Self JOIN | 10.71 us | 14.18 us | 1.3x |
| String concat | 5.60 us | 6.80 us | 1.2x |
| CASE expression | 5.10 us | 5.43 us | 1.1x |
| LEFT JOIN + GROUP BY | 55.69 us | 59.34 us | 1.1x |
| Multiple CTEs (2) | 20.60 us | 21.94 us | 1.1x |
| Batch INSERT | 74.93 us | 75.75 us | 1.0x |

### DuckDB Advantages

| Operation | DuckDB | Stoolap | Factor |
|-----------|--------|---------|--------|
| Derived table | 255.77 us | 406.82 us | 1.6x |

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
  Point Queries (ID):     ████████████████████  DOMINANT (1120x vs DuckDB)
  Subquery Compare:       ████████████████████  EXCELLENT (271x vs SQLite)
  DISTINCT Operations:    ████████████████████  EXCELLENT (22-246x vs SQLite)
  Semi-joins (EXISTS):    ████████████████████  EXCELLENT (11-275x faster)
  Batch Inserts:          ████████████████████  EXCELLENT (197x vs DuckDB)
  Window (PARTITION BY):  ████████████████████  EXCELLENT (132x vs DuckDB)
  OFFSET Pagination:      ████████████████████  EXCELLENT (81x vs DuckDB)
  Complex DML:            ████████████████████  EXCELLENT (55-107x faster)
  Aggregations:           ████████████████████  EXCELLENT (29x vs SQLite)

SQLite Strengths:
  Simple JOINs:           ████████              GOOD (1.7x faster)
  Simple Expressions:     ██████                MODERATE (1.1-1.5x faster)

DuckDB Strengths:
  Derived Tables:         ██████                MODERATE (1.6x faster)
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

*Benchmarks performed on Apple Silicon, in-memory mode, best of 3 runs.*
