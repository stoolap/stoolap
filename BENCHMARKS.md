# Stoolap v0.3.3 Benchmark Results

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
|   STOOLAP vs SQLite:    46 wins / 7 losses    (87% win rate)  |
|   STOOLAP vs DuckDB:    52 wins / 1 loss     (98% win rate)  |
|                                                               |
+---------------------------------------------------------------+
```

---

## Basic Operations

| Operation | Stoolap (us) | SQLite (us) | DuckDB (us) | Best |
|-----------|-------------|-------------|-------------|------|
| SELECT by ID | **0.12** | 0.21 | 145.55 | Stoolap |
| SELECT by index (exact) | **3.81** | 28.02 | 288.70 | Stoolap |
| SELECT by index (range) | **29.80** | 285.62 | 428.24 | Stoolap |
| SELECT complex | **111.88** | 534.21 | 184.68 | Stoolap |
| SELECT * (full scan) | **87.00** | 515.21 | 707.14 | Stoolap |
| UPDATE by ID | **0.54** | 0.61 | 146.14 | Stoolap |
| UPDATE complex | **59.01** | 443.73 | 209.77 | Stoolap |
| INSERT single | **1.14** | 1.62 | 194.26 | Stoolap |
| DELETE by ID | **0.65** | 1.32 | 152.34 | Stoolap |
| DELETE complex | **2.90** | 380.14 | 197.44 | Stoolap |
| Aggregation (GROUP BY) | **48.81** | 1403.39 | 104.32 | Stoolap |

**Basic Operations Score: Stoolap 11, SQLite 0, DuckDB 0**

---

## Advanced Operations

| Operation | Stoolap (us) | SQLite (us) | DuckDB (us) | Best |
|-----------|-------------|-------------|-------------|------|
| INNER JOIN | 20.35 | **14.86** | 607.20 | SQLite |
| LEFT JOIN + GROUP BY | **51.61** | 55.69 | 1269.40 | Stoolap |
| Scalar subquery | **9.08** | 399.10 | 257.12 | Stoolap |
| IN subquery | **110.94** | 1838.79 | 853.94 | Stoolap |
| EXISTS subquery | **3.06** | 38.42 | 928.06 | Stoolap |
| CTE + JOIN | **38.54** | 74.16 | 859.53 | Stoolap |
| Window ROW_NUMBER | **260.10** | 1781.90 | 690.83 | Stoolap |
| Window ROW_NUMBER (PK) | **5.98** | 21.36 | 419.10 | Stoolap |
| Window PARTITION BY | **7.56** | 64.81 | 1162.16 | Stoolap |
| UNION ALL | **5.48** | 6.24 | 173.19 | Stoolap |
| CASE expression | 5.25 | **5.10** | 247.54 | SQLite |
| Complex JOIN+GROUP+HAVING | **51.00** | 93.20 | 2233.32 | Stoolap |
| Batch INSERT (100 rows) | 77.94 | **74.93** | 14920.25 | SQLite |

**Advanced Operations Score: Stoolap 10, SQLite 3, DuckDB 0**

---

## Bottleneck Hunters

| Operation | Stoolap (us) | SQLite (us) | DuckDB (us) | Best |
|-----------|-------------|-------------|-------------|------|
| DISTINCT (no ORDER) | **4.81** | 104.38 | 235.97 | Stoolap |
| DISTINCT + ORDER BY | **5.18** | 139.58 | 291.81 | Stoolap |
| COUNT DISTINCT | **0.37** | 105.98 | 219.91 | Stoolap |
| LIKE prefix (User_1%) | **4.25** | 9.49 | 173.02 | Stoolap |
| LIKE contains (%50%) | **37.52** | 156.52 | 271.86 | Stoolap |
| OR conditions (3 vals) | **3.43** | 14.54 | 207.08 | Stoolap |
| IN list (7 values) | **2.47** | 14.52 | 1152.84 | Stoolap |
| NOT IN subquery | **81.75** | 1898.21 | 965.60 | Stoolap |
| NOT EXISTS subquery | **24.88** | 1729.58 | 1429.57 | Stoolap |
| OFFSET pagination (5000) | **14.53** | 21.31 | 1222.62 | Stoolap |
| Multi-col ORDER BY (3) | **144.68** | 416.71 | 361.35 | Stoolap |
| Self JOIN (same age) | 13.76 | **10.71** | 396.87 | SQLite |
| Multi window funcs (3) | **568.68** | 1803.93 | 760.23 | Stoolap |
| Nested subquery (3 lvl) | **339.88** | 6397.42 | 850.91 | Stoolap |
| Multi aggregates (6) | **125.83** | 842.98 | 306.48 | Stoolap |
| COALESCE + IS NOT NULL | 4.30 | **2.84** | 90.97 | SQLite |
| Expr in WHERE (funcs) | **5.53** | 15.05 | 236.50 | Stoolap |
| Math expressions | **15.96** | 36.74 | 247.55 | Stoolap |
| String concat (\|\|) | 7.13 | **5.60** | 253.41 | SQLite |
| Large result (no LIMIT) | **244.04** | 486.65 | 348.78 | Stoolap |
| Multiple CTEs (2) | 20.64 | **20.60** | 313.38 | SQLite |
| Correlated in SELECT | **265.93** | 511.29 | 1283.17 | Stoolap |
| BETWEEN (non-indexed) | **2.64** | 9.26 | 178.98 | Stoolap |
| GROUP BY (2 columns) | **161.17** | 2259.41 | 320.48 | Stoolap |
| CROSS JOIN (limited) | **95.77** | 1358.05 | 1458.25 | Stoolap |
| Derived table (FROM sub) | 411.49 | 875.95 | **255.77** | DuckDB |
| Window ROWS frame | **548.16** | 1881.15 | 2198.75 | Stoolap |
| HAVING complex | **99.57** | 1420.61 | 114.47 | Stoolap |
| Compare with subquery | **5.52** | 1424.07 | 293.51 | Stoolap |

**Bottleneck Hunters Score: Stoolap 24, SQLite 4, DuckDB 1**

---

## Summary by Category

| Category | Stoolap Wins | SQLite Wins | DuckDB Wins |
|----------|-------------|-------------|-------------|
| Basic Operations | 11 | 0 | 0 |
| Advanced Operations | 10 | 3 | 0 |
| Bottleneck Hunters | 24 | 4 | 1 |
| **Total** | **45** | **7** | **1** |

---

## Top Stoolap Wins vs SQLite

| Operation | Stoolap | SQLite | Speedup |
|-----------|---------|--------|---------|
| COUNT DISTINCT | 0.37 us | 105.98 us | **286x** |
| Compare with subquery | 5.52 us | 1424.07 us | **258x** |
| DELETE complex | 2.90 us | 380.14 us | **131x** |
| NOT EXISTS subquery | 24.88 us | 1729.58 us | **70x** |
| Scalar subquery | 9.08 us | 399.10 us | **44x** |
| Aggregation (GROUP BY) | 48.81 us | 1403.39 us | **29x** |
| DISTINCT + ORDER BY | 5.18 us | 139.58 us | **27x** |
| NOT IN subquery | 81.75 us | 1898.21 us | **23x** |
| DISTINCT (no ORDER) | 4.81 us | 104.38 us | **22x** |
| Nested subquery (3 lvl) | 339.88 us | 6397.42 us | **19x** |
| IN subquery | 110.94 us | 1838.79 us | **17x** |
| CROSS JOIN (limited) | 95.77 us | 1358.05 us | **14x** |
| GROUP BY (2 columns) | 161.17 us | 2259.41 us | **14x** |
| HAVING complex | 99.57 us | 1420.61 us | **14x** |
| EXISTS subquery | 3.06 us | 38.42 us | **13x** |

---

## Top Stoolap Wins vs DuckDB

| Operation | Stoolap | DuckDB | Speedup |
|-----------|---------|--------|---------|
| SELECT by ID | 0.12 us | 145.55 us | **1213x** |
| EXISTS subquery | 3.06 us | 928.06 us | **303x** |
| UPDATE by ID | 0.54 us | 146.14 us | **271x** |
| DELETE by ID | 0.65 us | 152.34 us | **234x** |
| Batch INSERT (100 rows) | 77.94 us | 14920.25 us | **191x** |
| INSERT single | 1.14 us | 194.26 us | **170x** |
| Window PARTITION BY | 7.56 us | 1162.16 us | **154x** |
| OFFSET pagination (5000) | 14.53 us | 1222.62 us | **84x** |
| SELECT by index (exact) | 3.81 us | 288.70 us | **76x** |
| Window ROW_NUMBER (PK) | 5.98 us | 419.10 us | **70x** |
| DELETE complex | 2.90 us | 197.44 us | **68x** |
| NOT EXISTS subquery | 24.88 us | 1429.57 us | **57x** |
| Compare with subquery | 5.52 us | 293.51 us | **53x** |
| DISTINCT (no ORDER) | 4.81 us | 235.97 us | **49x** |
| CASE expression | 5.25 us | 247.54 us | **47x** |
| Complex JOIN+GROUP+HAVING | 51.00 us | 2233.32 us | **44x** |
| UNION ALL | 5.48 us | 173.19 us | **32x** |
| Scalar subquery | 9.08 us | 257.12 us | **28x** |
| LEFT JOIN + GROUP BY | 51.61 us | 1269.40 us | **25x** |
| NOT IN subquery | 81.75 us | 965.60 us | **12x** |
| IN subquery | 110.94 us | 853.94 us | **7.7x** |
| Nested subquery (3 lvl) | 339.88 us | 850.91 us | **2.5x** |
| SELECT complex | 111.88 us | 184.68 us | **1.7x** |

---

## Where Others Win

### SQLite Advantages

| Operation | SQLite | Stoolap | Factor |
|-----------|--------|---------|--------|
| COALESCE | 2.84 us | 4.30 us | 1.5x |
| INNER JOIN | 14.86 us | 20.35 us | 1.4x |
| Self JOIN | 10.71 us | 13.76 us | 1.3x |
| String concat | 5.60 us | 7.13 us | 1.3x |
| Batch INSERT | 74.93 us | 77.94 us | 1.0x |
| CASE expression | 5.10 us | 5.25 us | 1.0x |
| Multiple CTEs (2) | 20.60 us | 20.64 us | 1.0x |

### DuckDB Advantages

| Operation | DuckDB | Stoolap | Factor |
|-----------|--------|---------|--------|
| Derived table | 255.77 us | 411.49 us | 1.6x |

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
  Point Queries (ID):     ████████████████████  1213x vs DuckDB
  Subquery Compare:       ████████████████████  258x vs SQLite
  DISTINCT Operations:    ████████████████████  22-286x vs SQLite
  Semi-joins (EXISTS):    ████████████████████  13-303x faster
  Batch Inserts:          ████████████████████  191x vs DuckDB
  Window (PARTITION BY):  ████████████████████  154x vs DuckDB
  OFFSET Pagination:      ████████████████████  84x vs DuckDB
  Complex DML:            ████████████████████  68-131x faster
  Aggregations:           ████████████████████  29x vs SQLite

SQLite Strengths:
  Simple JOINs:           ████████              SQLite ~1.4x faster
  Simple Expressions:     ██████                SQLite ~1.0-1.5x faster

DuckDB Strengths:
  Derived Tables:         ██████                DuckDB ~1.6x faster
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

*Benchmarks performed on Apple Silicon, in-memory mode, best of 10 runs. Results are point-in-time for v0.3.3; re-run on your hardware and workload for current numbers.*
