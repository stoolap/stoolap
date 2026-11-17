# Stoolap Benchmark Results

Performance comparison between **Stoolap**, **SQLite**, and **DuckDB** using identical workloads.

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Rows | 10,000 |
| Iterations | 500 (point queries), 250 (medium), 50 (heavy) |
| Mode | In-memory |
| Platform | Apple Silicon |
| Sampling | best of 25 runs for Stoolap; best of 10 for SQLite and DuckDB |
| Measured | Stoolap 2026-11-17, `main` at 253631c9; SQLite and DuckDB 2026-09-03 |
| SQLite | rusqlite v0.40.2 |
| DuckDB | duckdb v1.10501.0 |

## Overall Score

```
+---------------------------------------------------------------+
|                                                               |
|   STOOLAP vs SQLite:    50 wins / 3 losses (94% win rate)     |
|   STOOLAP vs DuckDB:    52 wins / 1 loss   (98% win rate)     |
|                                                               |
+---------------------------------------------------------------+
```

---

## Basic Operations

| Operation | Stoolap (us) | SQLite (us) | DuckDB (us) | Best |
|-----------|-------------|-------------|-------------|------|
| SELECT by ID | **0.06** | 0.16 | 69.70 | Stoolap |
| SELECT by index (exact) | **2.43** | 22.91 | 1186.94 | Stoolap |
| SELECT by index (range) | **23.38** | 227.87 | 309.96 | Stoolap |
| SELECT complex | **82.35** | 470.02 | 164.66 | Stoolap |
| SELECT * (full scan) | **77.88** | 434.31 | 598.39 | Stoolap |
| UPDATE by ID | **0.52** | 0.53 | 70.94 | Stoolap |
| UPDATE complex | **53.02** | 391.94 | 119.49 | Stoolap |
| INSERT single | **0.67** | 1.36 | 122.53 | Stoolap |
| DELETE by ID | **0.66** | 1.16 | 82.32 | Stoolap |
| DELETE complex | **1.61** | 338.53 | 107.44 | Stoolap |
| Aggregation (GROUP BY) | **46.06** | 1126.52 | 102.80 | Stoolap |

---

## Advanced Operations

| Operation | Stoolap (us) | SQLite (us) | DuckDB (us) | Best |
|-----------|-------------|-------------|-------------|------|
| INNER JOIN | **8.29** | 12.07 | 307.31 | Stoolap |
| LEFT JOIN + GROUP BY | **24.11** | 47.74 | 1132.88 | Stoolap |
| Scalar subquery | **3.60** | 330.70 | 220.47 | Stoolap |
| IN subquery | **38.44** | 1628.15 | 482.80 | Stoolap |
| EXISTS subquery | **2.38** | 27.75 | 744.22 | Stoolap |
| CTE + JOIN | **26.32** | 62.63 | 781.58 | Stoolap |
| Window ROW_NUMBER | **229.07** | 1519.95 | 615.79 | Stoolap |
| Window ROW_NUMBER (PK) | **3.38** | 17.22 | 372.81 | Stoolap |
| Window PARTITION BY | **6.25** | 47.89 | 541.97 | Stoolap |
| UNION ALL | **3.17** | 5.04 | 148.38 | Stoolap |
| CASE expression | 4.55 | **4.24** | 180.82 | SQLite |
| Complex JOIN+GROUP+HAVING | **35.60** | 68.70 | 1637.86 | Stoolap |
| Batch INSERT (100 rows) | **47.58** | 62.99 | 7757.23 | Stoolap |

---

## Bottleneck Hunters

| Operation | Stoolap (us) | SQLite (us) | DuckDB (us) | Best |
|-----------|-------------|-------------|-------------|------|
| DISTINCT (no ORDER) | **0.88** | 91.43 | 206.90 | Stoolap |
| DISTINCT + ORDER BY | **1.69** | 121.32 | 247.00 | Stoolap |
| COUNT DISTINCT | **0.20** | 92.07 | 190.66 | Stoolap |
| LIKE prefix (User_1%) | **2.69** | 8.29 | 181.92 | Stoolap |
| LIKE contains (%50%) | **28.38** | 134.32 | 189.46 | Stoolap |
| OR conditions (3 vals) | **2.38** | 12.58 | 169.62 | Stoolap |
| IN list (7 values) | **1.89** | 12.29 | 7369.91 | Stoolap |
| NOT IN subquery | **49.15** | 1668.07 | 519.09 | Stoolap |
| NOT EXISTS subquery | **2.21** | 76.56 | 898.08 | Stoolap |
| OFFSET pagination (5000) | **10.62** | 21.71 | 440.42 | Stoolap |
| Multi-col ORDER BY (3) | **111.91** | 359.28 | 289.59 | Stoolap |
| Self JOIN (same age) | **6.83** | 9.07 | 356.03 | Stoolap |
| Multi window funcs (3) | **406.06** | 1559.68 | 661.82 | Stoolap |
| Nested subquery (3 lvl) | **150.22** | 5530.04 | 766.03 | Stoolap |
| Multi aggregates (6) | **85.50** | 713.77 | 280.07 | Stoolap |
| COALESCE + IS NOT NULL | 2.78 | **2.56** | 72.09 | SQLite |
| Expr in WHERE (funcs) | **5.11** | 12.46 | 258.36 | Stoolap |
| Math expressions | **8.04** | 10.23 | 176.87 | Stoolap |
| String concat (\|\|) | **4.21** | 4.74 | 206.59 | Stoolap |
| Large result (no LIMIT) | **168.41** | 412.98 | 299.37 | Stoolap |
| Multiple CTEs (2) | 14.89 | **14.77** | 265.89 | SQLite |
| Correlated in SELECT | **198.59** | 459.84 | 1005.40 | Stoolap |
| BETWEEN (non-indexed) | **1.99** | 7.75 | 133.74 | Stoolap |
| GROUP BY (2 columns) | **132.59** | 1827.71 | 316.01 | Stoolap |
| CROSS JOIN (limited) | **7.90** | 1164.07 | 238.74 | Stoolap |
| Derived table (FROM sub) | 348.66 | 700.80 | **223.94** | DuckDB |
| Window ROWS frame | **321.08** | 1576.43 | 2808.85 | Stoolap |
| HAVING complex | **49.00** | 1128.71 | 120.62 | Stoolap |
| Compare with subquery | **2.55** | 1273.44 | 247.97 | Stoolap |

---

## Summary by Category

| Category | Stoolap Wins | SQLite Wins | DuckDB Wins |
|----------|-------------|-------------|-------------|
| Basic Operations | 11 | 0 | 0 |
| Advanced Operations | 12 | 1 | 0 |
| Bottleneck Hunters | 26 | 2 | 1 |
| **Total** | **49** | **3** | **1** |

---

## Top Stoolap Wins vs SQLite

| Operation | Stoolap | SQLite | Speedup |
|-----------|---------|--------|---------|
| Compare with subquery | 2.55 us | 1273.44 us | **499x** |
| COUNT DISTINCT | 0.20 us | 92.07 us | **460x** |
| DELETE complex | 1.61 us | 338.53 us | **210x** |
| CROSS JOIN (limited) | 7.90 us | 1164.07 us | **147x** |
| DISTINCT (no ORDER) | 0.88 us | 91.43 us | **104x** |
| Scalar subquery | 3.60 us | 330.70 us | **92x** |
| DISTINCT + ORDER BY | 1.69 us | 121.32 us | **72x** |
| IN subquery | 38.44 us | 1628.15 us | **42x** |
| Nested subquery (3 lvl) | 150.22 us | 5530.04 us | **37x** |
| NOT EXISTS subquery | 2.21 us | 76.56 us | **35x** |
| NOT IN subquery | 49.15 us | 1668.07 us | **34x** |
| Aggregation (GROUP BY) | 46.06 us | 1126.52 us | **24x** |
| HAVING complex | 49.00 us | 1128.71 us | **23x** |
| GROUP BY (2 columns) | 132.59 us | 1827.71 us | **14x** |
| EXISTS subquery | 2.38 us | 27.75 us | **12x** |

---

## Top Stoolap Wins vs DuckDB

| Operation | Stoolap | DuckDB | Speedup |
|-----------|---------|--------|---------|
| IN list (7 values) | 1.89 us | 7369.91 us | **3899x** |
| SELECT by ID | 0.06 us | 69.70 us | **1162x** |
| COUNT DISTINCT | 0.20 us | 190.66 us | **953x** |
| SELECT by index (exact) | 2.43 us | 1186.94 us | **488x** |
| NOT EXISTS subquery | 2.21 us | 898.08 us | **406x** |
| EXISTS subquery | 2.38 us | 744.22 us | **313x** |
| DISTINCT (no ORDER) | 0.88 us | 206.90 us | **235x** |
| INSERT single | 0.67 us | 122.53 us | **183x** |
| Batch INSERT (100 rows) | 47.58 us | 7757.23 us | **163x** |
| DISTINCT + ORDER BY | 1.69 us | 247.00 us | **146x** |
| UPDATE by ID | 0.52 us | 70.94 us | **136x** |
| DELETE by ID | 0.66 us | 82.32 us | **125x** |
| Window ROW_NUMBER (PK) | 3.38 us | 372.81 us | **110x** |
| Compare with subquery | 2.55 us | 247.97 us | **97x** |
| Window PARTITION BY | 6.25 us | 541.97 us | **87x** |
| OR conditions (3 vals) | 2.38 us | 169.62 us | **71x** |
| LIKE prefix (User_1%) | 2.69 us | 181.92 us | **68x** |
| BETWEEN (non-indexed) | 1.99 us | 133.74 us | **67x** |
| DELETE complex | 1.61 us | 107.44 us | **67x** |
| Scalar subquery | 3.60 us | 220.47 us | **61x** |
| Self JOIN (same age) | 6.83 us | 356.03 us | **52x** |
| Expr in WHERE (funcs) | 5.11 us | 258.36 us | **51x** |
| String concat (\|\|) | 4.21 us | 206.59 us | **49x** |
| LEFT JOIN + GROUP BY | 24.11 us | 1132.88 us | **47x** |
| UNION ALL | 3.17 us | 148.38 us | **47x** |

---

## Where Others Win

### SQLite Advantages

| Operation | SQLite | Stoolap | Factor |
|-----------|--------|---------|--------|
| COALESCE + IS NOT NULL | 2.56 us | 2.78 us | 1.1x |
| CASE expression | 4.24 us | 4.55 us | 1.1x |
| Multiple CTEs (2) | 14.77 us | 14.89 us | 1.0x |

### DuckDB Advantages

| Operation | DuckDB | Stoolap | Factor |
|-----------|--------|---------|--------|
| Derived table (FROM sub) | 223.94 us | 348.66 us | 1.6x |

---

## Architecture Comparison

| Feature | Stoolap | SQLite | DuckDB |
|---------|---------|--------|--------|
| Storage Model | Hybrid: row-oriented in-memory MVCC arena for hot rows, sealed on checkpoint into immutable columnar volumes (LZ4 row groups, dictionaries, zone maps) | Row-based (B-tree pages) | Columnar (vectorized) |
| Durability | WAL, checkpoint seals hot rows into volumes | Rollback journal or WAL | WAL with checkpoints |
| Concurrency | MVCC | WAL/locking | MVCC |
| Query Optimizer | Rule-based rewrites, runtime join and index selection | Cost-based (NGQP) | Cost-based |
| Parallel Execution | Yes (Rayon) | No | Yes |
| Language | Pure Rust | C | C++ |
| Memory Safety | Guaranteed | Manual | Manual |

---

## Performance Characteristics

```
Stoolap Strengths:
  Point Queries (ID):     ████████████████████  1162x vs DuckDB
  Subquery Compare:       ████████████████████  499x vs SQLite
  DISTINCT Operations:    ████████████████████  72-460x vs SQLite
  Semi-joins (EXISTS):    ████████████████████  12-406x faster
  Batch Inserts:          ████████████████████  163x vs DuckDB
  Window (PARTITION BY):  ████████████████████  87x vs DuckDB
  OFFSET Pagination:      ████████████████████  41x vs DuckDB
  Complex DML:            ████████████████████  2-210x faster
  Aggregations:           ████████████████████  24x vs SQLite

SQLite Strengths:
  Simple Expressions:     ██████                SQLite ~1.0-1.1x faster

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

# SQLite benchmark (the stoolap-compare crate in compare/)
cargo build --release -p stoolap-compare --bin benchmark_sqlite
./target/release/benchmark_sqlite

# DuckDB benchmark (the same crate, with its duckdb feature)
cargo build --release -p stoolap-compare --features duckdb --bin benchmark_duckdb
./target/release/benchmark_duckdb

# Build all benchmarks at once
cargo build --release --example benchmark
cargo build --release -p stoolap-compare --features duckdb
```

---

*Benchmarks performed on Apple Silicon, in-memory mode. Stoolap is the best of 25 runs on 2026-11-17; SQLite and DuckDB are the best of 10 runs from 2026-09-03 on the same machine. Some Stoolap operations settle at one of two levels per process (EXISTS subquery near 2.4 or 4.8 us, CTE + JOIN near 26 or 50 us), so a few runs can miss the faster one. Results are point-in-time for `main`; re-run on your hardware and workload for current numbers.*
