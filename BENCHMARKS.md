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
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   STOOLAP    40 wins   ██████████████████████████████████████ │
│   SQLite     13 wins   █████████████                        │
│                                                             │
│   Win Rate: 75.5%                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Summary by Category

| Category | Stoolap Wins | SQLite Wins | Stoolap Win Rate |
|----------|-------------|-------------|------------------|
| Basic Operations | 9 | 2 | 81.8% |
| Advanced Operations | 8 | 5 | 61.5% |
| Bottleneck Hunters | 23 | 6 | 79.3% |
| **Total** | **40** | **13** | **75.5%** |

---

## Basic Operations

| Operation | Stoolap (μs) | SQLite (μs) | Winner | Factor |
|-----------|-------------|-------------|--------|--------|
| SELECT by ID | **0.186** | 0.289 | Stoolap | **1.6x** |
| SELECT by index (exact) | **6.4** | 36.4 | Stoolap | **5.7x** |
| SELECT by index (range) | **167.6** | 278.1 | Stoolap | **1.7x** |
| SELECT complex | **382.0** | 528.5 | Stoolap | **1.4x** |
| SELECT * (full scan) | **119.5** | 527.2 | Stoolap | **4.4x** |
| UPDATE by ID | 0.939 | **0.616** | SQLite | 1.5x |
| UPDATE complex | **73.2** | 443.1 | Stoolap | **6.1x** |
| INSERT single | 1.96 | **1.62** | SQLite | 1.2x |
| DELETE by ID | **1.04** | 1.34 | Stoolap | **1.3x** |
| DELETE complex | **4.5** | 374.7 | Stoolap | **83.3x** |
| Aggregation (GROUP BY) | **122.1** | 1406.5 | Stoolap | **11.5x** |

---

## Advanced Operations

| Operation | Stoolap (μs) | SQLite (μs) | Winner | Factor |
|-----------|-------------|-------------|--------|--------|
| INNER JOIN | 29.7 | **16.6** | SQLite | 1.8x |
| LEFT JOIN + GROUP BY | 77.7 | **63.6** | SQLite | 1.2x |
| Scalar subquery | **191.2** | 392.8 | Stoolap | **2.1x** |
| IN subquery | **574.2** | 1825.2 | Stoolap | **3.2x** |
| EXISTS subquery | **4.2** | 41.3 | Stoolap | **9.9x** |
| CTE + JOIN | **51.0** | 69.8 | Stoolap | **1.4x** |
| Window ROW_NUMBER | **699.3** | 1766.7 | Stoolap | **2.5x** |
| Window ROW_NUMBER (PK) | **7.5** | 21.6 | Stoolap | **2.9x** |
| Window PARTITION BY | **11.7** | 63.6 | Stoolap | **5.4x** |
| UNION ALL | 7.1 | **6.8** | SQLite | 1.0x |
| CASE expression | 10.6 | **5.5** | SQLite | 1.9x |
| Complex JOIN+GROUP+HAVING | **73.1** | 99.1 | Stoolap | **1.4x** |
| Batch INSERT (100 rows) | 122.6 | **75.6** | SQLite | 1.6x |

---

## Bottleneck Hunters

| Operation | Stoolap (μs) | SQLite (μs) | Winner | Factor |
|-----------|-------------|-------------|--------|--------|
| DISTINCT (no ORDER) | **6.0** | 102.8 | Stoolap | **17.1x** |
| DISTINCT + ORDER BY | **6.8** | 143.0 | Stoolap | **21.0x** |
| COUNT DISTINCT | **1.5** | 104.5 | Stoolap | **70.6x** |
| LIKE prefix (User_1%) | **4.7** | 9.8 | Stoolap | **2.1x** |
| LIKE contains (%50%) | **99.5** | 155.1 | Stoolap | **1.6x** |
| OR conditions (3 vals) | **5.5** | 14.5 | Stoolap | **2.6x** |
| IN list (7 values) | **4.6** | 14.7 | Stoolap | **3.2x** |
| NOT IN subquery | **612.9** | 1928.6 | Stoolap | **3.1x** |
| NOT EXISTS subquery | **71.8** | 1814.8 | Stoolap | **25.3x** |
| OFFSET pagination (5000) | **14.7** | 22.8 | Stoolap | **1.6x** |
| Multi-col ORDER BY (3) | **186.4** | 430.3 | Stoolap | **2.3x** |
| Self JOIN (same age) | 16.3 | **11.6** | SQLite | 1.4x |
| Multi window funcs (3) | **1531.3** | 1782.4 | Stoolap | **1.2x** |
| Nested subquery (3 lvl) | **4076.0** | 6313.1 | Stoolap | **1.5x** |
| Multi aggregates (6) | **827.5** | 835.0 | Stoolap | **1.0x** |
| COALESCE + IS NOT NULL | 7.1 | **2.9** | SQLite | 2.4x |
| Expr in WHERE (funcs) | **8.7** | 15.3 | Stoolap | **1.8x** |
| Math expressions | **21.7** | 35.1 | Stoolap | **1.6x** |
| String concat (\|\|) | 12.0 | **5.7** | SQLite | 2.1x |
| Large result (no LIMIT) | **344.8** | 484.4 | Stoolap | **1.4x** |
| Multiple CTEs (2) | 27.2 | **22.4** | SQLite | 1.2x |
| Correlated in SELECT | 528.5 | **497.0** | SQLite | 1.06x |
| BETWEEN (non-indexed) | **3.6** | 9.9 | Stoolap | **2.8x** |
| GROUP BY (2 columns) | **223.8** | 2363.5 | Stoolap | **10.6x** |
| CROSS JOIN (limited) | **205.0** | 1354.6 | Stoolap | **6.6x** |
| Derived table (FROM sub) | **823.8** | 858.3 | Stoolap | **1.0x** |
| Window ROWS frame | **1288.1** | 1827.1 | Stoolap | **1.4x** |
| HAVING complex | **125.9** | 1374.3 | Stoolap | **10.9x** |
| Compare with subquery | 1639.3 | **1404.4** | SQLite | 1.2x |

---

## Top Stoolap Wins

Operations where Stoolap significantly outperforms SQLite:

| Operation | Speedup | Notes |
|-----------|---------|-------|
| DELETE complex | **83.3x** | MVCC + index optimization |
| COUNT DISTINCT | **70.6x** | Hash-based single-pass counting |
| NOT EXISTS subquery | **25.3x** | Semi-join with early termination |
| DISTINCT + ORDER BY | **21.0x** | Streaming deduplication |
| DISTINCT (no ORDER) | **17.1x** | Hash set deduplication |
| Aggregation (GROUP BY) | **11.5x** | Parallel hash aggregation |
| HAVING complex | **10.9x** | Predicate pushdown |
| GROUP BY (2 columns) | **10.6x** | Multi-key hash aggregation |
| EXISTS subquery | **9.9x** | Semi-join optimization |
| CROSS JOIN (limited) | **6.6x** | Early termination |
| UPDATE complex | **6.1x** | Index-based updates |
| SELECT by index (exact) | **5.7x** | Optimized index lookup |
| Window PARTITION BY | **5.4x** | Optimized partitioning |
| SELECT * (full scan) | **4.4x** | Columnar scan optimization |

---

## Where SQLite Wins

Operations where SQLite performs better:

| Operation | Factor | Reason |
|-----------|--------|--------|
| COALESCE | 2.4x | Simple expression evaluation |
| String concat | 2.1x | String operations |
| CASE expression | 1.9x | Expression evaluation |
| INNER JOIN | 1.8x | Highly optimized nested loop |
| Batch INSERT | 1.6x | Insert path optimization |
| UPDATE by ID | 1.5x | Simple update path |
| Self JOIN | 1.4x | Join optimization |
| Multiple CTEs | 1.2x | CTE materialization |
| Correlated in SELECT | 1.06x | Nearly tied |

---

## Stoolap Strengths

```
Analytics/OLAP:     ████████████████████  DOMINANT (11-71x faster)
DISTINCT:           ████████████████████  EXCELLENT (17-71x faster)
Aggregations:       ████████████████████  EXCELLENT (10-11x faster)
Complex DML:        ████████████████████  EXCELLENT (6-83x faster)
Semi-joins:         ████████████████████  EXCELLENT (10-25x faster)
Index Lookups:      ████████████████████  EXCELLENT (5.7x faster)
Full Table Scans:   ████████████████████  EXCELLENT (4.4x faster)
Window Functions:   ████████████████      STRONG (1.2-5.4x faster)
Subqueries (IN):    ██████████████        GOOD (2.1-3.2x faster)
Anti-joins:         ████████████████████  EXCELLENT (3-25x faster)
```

---

## Best Use Cases for Stoolap

Based on benchmark results, Stoolap excels at:

1. **Analytics & Reporting** - Aggregations are 10-11x faster
2. **DISTINCT Operations** - 17-71x faster for deduplication
3. **Complex DML** - Updates/Deletes with conditions 6-83x faster
4. **Semi-joins (EXISTS)** - 10x faster with early termination
5. **Index Lookups** - Exact match queries 5.7x faster
6. **Large Table Scans** - Full scans 4.4x faster
7. **Window Functions** - Especially PARTITION BY (5.4x faster)
8. **NOT IN/NOT EXISTS** - Anti-joins 3-25x faster
9. **Pagination** - OFFSET queries 1.6x faster

---

## Running the Benchmarks

```bash
# Run Stoolap benchmark
cargo run --release --example benchmark

# Run SQLite comparison benchmark
cargo run --release --example benchmark_sqlite
```

---

*Benchmarks performed on Apple Silicon, in-memory mode, averaged over 10,000 rows and 500 iterations.*
