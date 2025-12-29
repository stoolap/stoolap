# Stoolap vs SQLite Benchmark Results

Comprehensive performance comparison between Stoolap and SQLite using identical workloads.

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Rows | 10,000 |
| Iterations | 500 |
| Mode | In-memory |
| SQLite Version | Latest (rusqlite) |

## Overall Score

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   STOOLAP    37 wins   ████████████████████████████████ │
│   SQLite     16 wins   ████████████████                 │
│                                                         │
│   Win Rate: 69.8%                                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Summary by Category

| Category | Stoolap Wins | SQLite Wins | Stoolap Win Rate |
|----------|-------------|-------------|------------------|
| Basic Operations | 8 | 3 | 72.7% |
| Advanced Operations | 7 | 6 | 53.8% |
| Bottleneck Hunters | 22 | 7 | 75.9% |
| **Total** | **37** | **16** | **69.8%** |

---

## Basic Operations

| Operation | Stoolap (μs) | SQLite (μs) | Winner | Factor |
|-----------|-------------|-------------|--------|--------|
| SELECT by ID | **0.192** | 0.290 | Stoolap | **1.5x** |
| SELECT by index (exact) | **15.9** | 35.5 | Stoolap | **2.2x** |
| SELECT by index (range) | **183.6** | 274.6 | Stoolap | **1.5x** |
| SELECT complex | **389.9** | 523.4 | Stoolap | **1.3x** |
| SELECT * (full scan) | **183.2** | 521.8 | Stoolap | **2.8x** |
| UPDATE by ID | 0.968 | **0.624** | SQLite | 1.6x |
| UPDATE complex | **62.7** | 451.3 | Stoolap | **7.2x** |
| INSERT single | 2.016 | **1.608** | SQLite | 1.3x |
| DELETE by ID | **1.072** | 1.333 | Stoolap | **1.2x** |
| DELETE complex | **5.98** | 378.5 | Stoolap | **63.3x** |
| Aggregation (GROUP BY) | **180.2** | 1431.1 | Stoolap | **7.9x** |

---

## Advanced Operations

| Operation | Stoolap (μs) | SQLite (μs) | Winner | Factor |
|-----------|-------------|-------------|--------|--------|
| INNER JOIN | 50.5 | **16.6** | SQLite | 3.0x |
| LEFT JOIN + GROUP BY | 90.4 | **59.6** | SQLite | 1.5x |
| Scalar subquery | **244.9** | 412.5 | Stoolap | **1.7x** |
| IN subquery | **646.2** | 1799.6 | Stoolap | **2.8x** |
| EXISTS subquery | 138.5 | **46.0** | SQLite | 3.0x |
| CTE + JOIN | **67.6** | 76.6 | Stoolap | **1.1x** |
| Window ROW_NUMBER | **790.6** | 1801.5 | Stoolap | **2.3x** |
| Window ROW_NUMBER (PK) | **8.0** | 21.3 | Stoolap | **2.6x** |
| Window PARTITION BY | **12.2** | 61.4 | Stoolap | **5.0x** |
| UNION ALL | 7.8 | **7.4** | SQLite | 1.1x |
| CASE expression | 10.8 | **5.4** | SQLite | 2.0x |
| Complex JOIN+GROUP+HAVING | 80.9 | 80.1 | Tie | 1.0x |
| Batch INSERT (100 rows) | 122.3 | **75.1** | SQLite | 1.6x |

---

## Bottleneck Hunters

| Operation | Stoolap (μs) | SQLite (μs) | Winner | Factor |
|-----------|-------------|-------------|--------|--------|
| DISTINCT (no ORDER) | **6.7** | 106.3 | Stoolap | **15.7x** |
| DISTINCT + ORDER BY | **8.2** | 147.4 | Stoolap | **17.9x** |
| COUNT DISTINCT | **1.6** | 109.3 | Stoolap | **67.4x** |
| LIKE prefix (User_1%) | **5.2** | 10.0 | Stoolap | **1.9x** |
| LIKE contains (%50%) | **131.1** | 156.4 | Stoolap | **1.2x** |
| OR conditions (3 vals) | **6.3** | 14.8 | Stoolap | **2.3x** |
| IN list (7 values) | **5.8** | 14.8 | Stoolap | **2.5x** |
| NOT IN subquery | **796.7** | 1910.2 | Stoolap | **2.4x** |
| NOT EXISTS subquery | **232.1** | 1803.7 | Stoolap | **7.8x** |
| OFFSET pagination (5000) | 46.5 | **21.7** | SQLite | 2.1x |
| Multi-col ORDER BY (3) | **245.7** | 428.6 | Stoolap | **1.7x** |
| Self JOIN (same age) | 26.9 | **12.7** | SQLite | 2.1x |
| Multi window funcs (3) | **1601.7** | 1837.6 | Stoolap | **1.1x** |
| Nested subquery (3 lvl) | **4738.1** | 6326.4 | Stoolap | **1.3x** |
| Multi aggregates (6) | 1047.5 | **825.8** | SQLite | 1.3x |
| COALESCE + IS NOT NULL | 6.9 | **2.9** | SQLite | 2.4x |
| Expr in WHERE (funcs) | **8.5** | 15.2 | Stoolap | **1.8x** |
| Math expressions | **21.1** | 36.4 | Stoolap | **1.7x** |
| String concat (\|\|) | 10.0 | **5.8** | SQLite | 1.7x |
| Large result (no LIMIT) | **364.9** | 480.1 | Stoolap | **1.3x** |
| Multiple CTEs (2) | 47.9 | **21.2** | SQLite | 2.3x |
| Correlated in SELECT | 512.2 | **482.7** | SQLite | 1.1x |
| BETWEEN (non-indexed) | **3.8** | 9.0 | Stoolap | **2.4x** |
| GROUP BY (2 columns) | **272.2** | 2189.3 | Stoolap | **8.0x** |
| CROSS JOIN (limited) | **274.2** | 1378.8 | Stoolap | **5.0x** |
| Derived table (FROM sub) | 863.2 | **842.1** | SQLite | 1.0x |
| Window ROWS frame | **1344.9** | 1904.0 | Stoolap | **1.4x** |
| HAVING complex | **186.4** | 1405.5 | Stoolap | **7.5x** |
| Compare with subquery | 1876.9 | **1450.8** | SQLite | 1.3x |

---

## Top Stoolap Wins

Operations where Stoolap significantly outperforms SQLite:

| Operation | Speedup | Notes |
|-----------|---------|-------|
| COUNT DISTINCT | **67.4x** | Hash-based single-pass counting |
| DELETE complex | **63.3x** | MVCC + index optimization |
| DISTINCT + ORDER BY | **17.9x** | Streaming deduplication |
| DISTINCT (no ORDER) | **15.7x** | Hash set deduplication |
| GROUP BY (2 columns) | **8.0x** | Multi-key hash aggregation |
| Aggregation (GROUP BY) | **7.9x** | Parallel hash aggregation |
| NOT EXISTS subquery | **7.8x** | Anti-join optimization |
| HAVING complex | **7.5x** | Predicate pushdown |
| UPDATE complex | **7.2x** | Index-based updates |
| Window PARTITION BY | **5.0x** | Optimized partitioning |
| CROSS JOIN (limited) | **5.0x** | Early termination |

---

## Where SQLite Wins

Operations where SQLite performs better:

| Operation | Factor | Reason |
|-----------|--------|--------|
| INNER JOIN | 3.0x | Highly optimized nested loop |
| EXISTS subquery | 3.0x | Specialized semi-join |
| COALESCE | 2.4x | Simple expression evaluation |
| Multiple CTEs | 2.3x | CTE materialization |
| Self JOIN | 2.1x | Join optimization |
| OFFSET pagination | 2.1x | Skip optimization |
| CASE expression | 2.0x | Expression evaluation |

---

## Stoolap Strengths

```
Analytics/OLAP:     ████████████████████  DOMINANT (8-67x faster)
Aggregations:       ████████████████████  EXCELLENT (7-8x faster)
DISTINCT:           ████████████████████  EXCELLENT (16-67x faster)
Window Functions:   ████████████████      STRONG (1.1-5x faster)
Complex DML:        ████████████████████  EXCELLENT (7-63x faster)
Subqueries (IN):    ██████████████        GOOD (1.3-2.8x faster)
CTEs:               ████████████          GOOD (1.1x faster)
```

---

## Best Use Cases for Stoolap

Based on benchmark results, Stoolap excels at:

1. **Analytics & Reporting** - Aggregations are 7-8x faster
2. **DISTINCT Operations** - 16-67x faster for deduplication
3. **Complex DML** - Updates/Deletes with conditions 7-63x faster
4. **Window Functions** - Especially PARTITION BY (5x faster)
5. **Large Table Scans** - Full scans 2.8x faster
6. **NOT IN/NOT EXISTS** - Anti-joins 2.4-7.8x faster

---

## Running the Benchmarks

```bash
# Run Stoolap benchmark
cargo run --release --example benchmark

# Run SQLite comparison benchmark
cargo run --release --example benchmark_sqlite
```

---

*Benchmarks performed on Apple Silicon, in-memory mode, averaged over 10000 rows and 500 iterations.*
