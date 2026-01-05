# Stoolap (v0.2.0) vs SQLite Benchmark Results

Performance comparison between Stoolap and SQLite using identical workloads.

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Rows | 10,000 |
| Iterations | 500 |
| Mode | In-memory |
| SQLite Version | rusqlite (v0.32.1) |

## Overall Score

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   STOOLAP    44 wins   ████████████████████████████████     │
│   SQLite      9 wins   █████████                            │
│                                                             │
│   Win Rate: 83.0%                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Summary by Category

| Category | Stoolap Wins | SQLite Wins | Stoolap Win Rate |
|----------|-------------|-------------|------------------|
| Basic Operations | 10 | 1 | 90.9% |
| Advanced Operations | 9 | 4 | 69.2% |
| Bottleneck Hunters | 25 | 4 | 86.2% |
| **Total** | **44** | **9** | **83.0%** |

---

## Basic Operations

| Operation | Stoolap (μs) | SQLite (μs) | Winner | Factor |
|-----------|-------------|-------------|--------|--------|
| SELECT by ID | **0.195** | 0.289 | Stoolap | **1.5x** |
| SELECT by index (exact) | **7.9** | 36.4 | Stoolap | **4.6x** |
| SELECT by index (range) | **53.4** | 278.1 | Stoolap | **5.2x** |
| SELECT complex | **189.4** | 528.5 | Stoolap | **2.8x** |
| SELECT * (full scan) | **124.3** | 527.2 | Stoolap | **4.2x** |
| UPDATE by ID | 0.82 | **0.616** | SQLite | 1.3x |
| UPDATE complex | **58.4** | 443.1 | Stoolap | **7.6x** |
| INSERT single | **1.57** | 1.62 | Stoolap | **1.03x** |
| DELETE by ID | **0.92** | 1.34 | Stoolap | **1.5x** |
| DELETE complex | **4.9** | 374.7 | Stoolap | **76.5x** |
| Aggregation (GROUP BY) | **135.3** | 1406.5 | Stoolap | **10.4x** |

---

## Advanced Operations

| Operation | Stoolap (μs) | SQLite (μs) | Winner | Factor |
|-----------|-------------|-------------|--------|--------|
| INNER JOIN | 29.1 | **16.6** | SQLite | 1.8x |
| LEFT JOIN + GROUP BY | 72.2 | **63.6** | SQLite | 1.1x |
| Scalar subquery | **72.2** | 392.8 | Stoolap | **5.4x** |
| IN subquery | **425.9** | 1825.2 | Stoolap | **4.3x** |
| EXISTS subquery | **3.9** | 41.3 | Stoolap | **10.6x** |
| CTE + JOIN | **49.1** | 69.8 | Stoolap | **1.4x** |
| Window ROW_NUMBER | **509.6** | 1766.7 | Stoolap | **3.5x** |
| Window ROW_NUMBER (PK) | **6.4** | 21.6 | Stoolap | **3.4x** |
| Window PARTITION BY | **11.5** | 63.6 | Stoolap | **5.5x** |
| UNION ALL | **6.7** | 6.8 | Stoolap | **1.01x** |
| CASE expression | 5.9 | **5.5** | SQLite | 1.07x |
| Complex JOIN+GROUP+HAVING | **75.8** | 99.1 | Stoolap | **1.3x** |
| Batch INSERT (100 rows) | 89.7 | **75.6** | SQLite | 1.2x |

---

## Bottleneck Hunters

| Operation | Stoolap (μs) | SQLite (μs) | Winner | Factor |
|-----------|-------------|-------------|--------|--------|
| DISTINCT (no ORDER) | **6.0** | 102.8 | Stoolap | **17.1x** |
| DISTINCT + ORDER BY | **6.8** | 143.0 | Stoolap | **21.0x** |
| COUNT DISTINCT | **1.2** | 104.5 | Stoolap | **87.1x** |
| LIKE prefix (User_1%) | **4.9** | 9.8 | Stoolap | **2.0x** |
| LIKE contains (%50%) | **100.6** | 155.1 | Stoolap | **1.5x** |
| OR conditions (3 vals) | **6.1** | 14.5 | Stoolap | **2.4x** |
| IN list (7 values) | **5.0** | 14.7 | Stoolap | **2.9x** |
| NOT IN subquery | **446.5** | 1928.6 | Stoolap | **4.3x** |
| NOT EXISTS subquery | **73.8** | 1814.8 | Stoolap | **24.6x** |
| OFFSET pagination (5000) | **14.7** | 22.8 | Stoolap | **1.6x** |
| Multi-col ORDER BY (3) | **195.4** | 430.3 | Stoolap | **2.2x** |
| Self JOIN (same age) | 17.3 | **11.6** | SQLite | 1.5x |
| Multi window funcs (3) | **1112.9** | 1782.4 | Stoolap | **1.6x** |
| Nested subquery (3 lvl) | **3587.3** | 6313.1 | Stoolap | **1.8x** |
| Multi aggregates (6) | **289.7** | 835.0 | Stoolap | **2.9x** |
| COALESCE + IS NOT NULL | 5.1 | **2.9** | SQLite | 1.8x |
| Expr in WHERE (funcs) | **7.3** | 15.3 | Stoolap | **2.1x** |
| Math expressions | **20.7** | 35.1 | Stoolap | **1.7x** |
| String concat (\|\|) | 6.7 | **5.7** | SQLite | 1.2x |
| Large result (no LIMIT) | **268.0** | 484.4 | Stoolap | **1.8x** |
| Multiple CTEs (2) | 29.8 | **22.4** | SQLite | 1.3x |
| Correlated in SELECT | **429.3** | 497.0 | Stoolap | **1.2x** |
| BETWEEN (non-indexed) | **3.8** | 9.9 | Stoolap | **2.6x** |
| GROUP BY (2 columns) | **219.1** | 2363.5 | Stoolap | **10.8x** |
| CROSS JOIN (limited) | **179.7** | 1354.6 | Stoolap | **7.5x** |
| Derived table (FROM sub) | **485.9** | 858.3 | Stoolap | **1.8x** |
| Window ROWS frame | **864.8** | 1827.1 | Stoolap | **2.1x** |
| HAVING complex | **140.1** | 1374.3 | Stoolap | **9.8x** |
| Compare with subquery | **208.6** | 1026.0 | Stoolap | **4.9x** |

---

## Top Stoolap Wins

Operations where Stoolap significantly outperforms SQLite:

| Operation | Speedup | Notes |
|-----------|---------|-------|
| COUNT DISTINCT | **87.1x** | Hash-based single-pass counting |
| DELETE complex | **76.5x** | MVCC + index optimization |
| NOT EXISTS subquery | **24.6x** | Semi-join with early termination |
| DISTINCT + ORDER BY | **21.0x** | Streaming deduplication |
| DISTINCT (no ORDER) | **17.1x** | Hash set deduplication |
| GROUP BY (2 columns) | **10.8x** | Multi-key hash aggregation |
| EXISTS subquery | **10.6x** | Semi-join optimization |
| Aggregation (GROUP BY) | **10.4x** | Parallel hash aggregation |
| HAVING complex | **9.8x** | Predicate pushdown |
| UPDATE complex | **7.6x** | Index-based updates |
| CROSS JOIN (limited) | **7.5x** | Early termination |
| Window PARTITION BY | **5.5x** | Optimized partitioning |
| Scalar subquery | **5.4x** | Optimized subquery execution |
| SELECT by index (range) | **5.2x** | Efficient range scans |
| Compare with subquery | **4.9x** | Subquery optimization |
| SELECT by index (exact) | **4.6x** | Optimized index lookup |

---

## Where SQLite Wins

Operations where SQLite performs better:

| Operation | Factor | Reason |
|-----------|--------|--------|
| INNER JOIN | 1.8x | Highly optimized nested loop |
| COALESCE | 1.8x | Simple expression evaluation |
| Self JOIN | 1.5x | Join optimization |
| Multiple CTEs | 1.3x | CTE materialization |
| UPDATE by ID | 1.3x | Simple update path |
| Batch INSERT | 1.2x | Insert path optimization |
| String concat | 1.2x | String operation overhead |
| LEFT JOIN + GROUP BY | 1.1x | Join optimization |
| CASE expression | 1.07x | Simple expression evaluation |

---

## Stoolap Strengths

```
Analytics/OLAP:     ████████████████████  DOMINANT (10-87x faster)
DISTINCT:           ████████████████████  EXCELLENT (17-87x faster)
Aggregations:       ████████████████████  EXCELLENT (10-11x faster)
Complex DML:        ████████████████████  EXCELLENT (7-77x faster)
Semi-joins:         ████████████████████  EXCELLENT (10-25x faster)
Index Lookups:      ████████████████████  EXCELLENT (4.6x faster)
Full Table Scans:   ████████████████████  EXCELLENT (4.2x faster)
Window Functions:   ████████████████      STRONG (1.6-5.5x faster)
Subqueries (IN):    ██████████████        GOOD (4.3x faster)
Anti-joins:         ████████████████████  EXCELLENT (4-25x faster)
```

---

## Best Use Cases for Stoolap

Based on benchmark results, Stoolap excels at:

1. **Analytics & Reporting** - Aggregations are 10-11x faster
2. **DISTINCT Operations** - 17-87x faster for deduplication
3. **Complex DML** - Updates/Deletes with conditions 7-77x faster
4. **Semi-joins (EXISTS)** - 10x faster with early termination
5. **Index Lookups** - Exact match queries 4.6x faster
6. **Large Table Scans** - Full scans 4.2x faster
7. **Window Functions** - Especially PARTITION BY (5.5x faster)
8. **NOT IN/NOT EXISTS** - Anti-joins 4-25x faster
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
