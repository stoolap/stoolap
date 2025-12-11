---
title: Parallel Execution
category: Performance
order: 3
---

# Parallel Execution

Stoolap uses Rayon's work-stealing scheduler to automatically parallelize CPU-intensive query operations. This provides significant performance improvements on multi-core systems for large datasets.

## Overview

Parallel execution is automatic and transparent:
- Queries on small datasets run sequentially (lower overhead)
- Queries on large datasets automatically use multiple CPU cores
- The optimizer decides when parallelization is beneficial

## Parallelized Operations

### Parallel Filter (WHERE clause)

When filtering large result sets, Stoolap processes rows in parallel chunks:

```sql
-- Automatically parallelized for tables with 10,000+ rows
SELECT * FROM large_table WHERE value > 100 AND status = 'active';
```

**Threshold**: 10,000+ rows

**How it works**:
1. Rows are split into chunks (default: 2,048 rows per chunk)
2. Each chunk is filtered independently using multiple threads
3. Results are merged back together

### Parallel Sort (ORDER BY)

Large result sets are sorted using parallel merge sort:

```sql
-- Automatically parallelized for 50,000+ rows
SELECT * FROM large_table ORDER BY created_at DESC;
```

**Threshold**: 50,000+ rows

### Parallel Hash Join

Hash join operations parallelize both the build and probe phases:

```sql
-- Build phase uses parallel hash map construction
-- Probe phase processes rows in parallel
SELECT o.*, c.name
FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE o.amount > 1000;
```

**Threshold**: 5,000+ rows in the build table

Uses a concurrent hash map (DashMap) for thread-safe parallel operations.

### Parallel Distinct

DISTINCT operations with large result sets use two-phase parallel deduplication:

```sql
-- Parallelized for 10,000+ rows
SELECT DISTINCT category FROM products;
```

**Threshold**: 10,000+ rows

## Viewing Parallel Execution

Use `EXPLAIN ANALYZE` to see when parallel execution is used:

```sql
EXPLAIN ANALYZE SELECT * FROM large_table WHERE value > 100;
```

Output shows parallel execution:
```
plan
----
SELECT (actual time=45.2ms, rows=50000)
  Columns: *
  -> Parallel Seq Scan on large_table (workers=8) (actual rows=50000)
       Filter: (value > 100)
```

The `workers=N` indicates the number of parallel workers used.

## Thresholds

| Operation | Minimum Rows | Description |
|-----------|--------------|-------------|
| Filter | 10,000 | WHERE clause evaluation |
| Sort | 50,000 | ORDER BY processing |
| Hash Join | 5,000 | Join build table size |
| Distinct | 10,000 | Deduplication |

These thresholds ensure parallelization overhead doesn't exceed the benefit.

## How It Works

### Work-Stealing Scheduler

Stoolap uses Rayon's work-stealing scheduler:

1. **Task Distribution**: Work is divided into chunks
2. **Thread Pool**: A global thread pool processes chunks
3. **Work Stealing**: Idle threads steal work from busy threads
4. **Load Balancing**: Automatically handles varying chunk processing times

### Chunk Size

The default chunk size is 2,048 rows, optimized for:
- L2 cache efficiency
- Task scheduling overhead
- Load balancing

### Thread Pool

Rayon creates a thread pool sized to the number of CPU cores:
- Automatically scales to available cores
- Shared across all parallel operations
- No configuration needed

## Performance Considerations

### When Parallel Helps

Parallel execution provides the most benefit when:
- Processing large datasets (above thresholds)
- Performing CPU-intensive operations (complex WHERE, sorting)
- Running on multi-core systems

### When Parallel May Not Help

Parallel execution may have minimal benefit when:
- Dataset is small (below thresholds)
- I/O bound operations (disk reads)
- Single-core systems

### Index Usage with Parallel

Indexes and parallel execution work together:
- Index lookups are sequential (fast, no parallelization needed)
- Post-filter operations on indexed results may still parallelize

```sql
-- Index used for category lookup, parallel filter for value
SELECT * FROM products
WHERE category = 'Electronics' AND value > complex_calculation(price);
```

## Partial Pushdown

Stoolap supports "partial pushdown" where:
- Simple predicates are pushed to storage (index usage)
- Complex predicates are evaluated with parallel filtering

```sql
-- indexed_col uses index, complex_func uses parallel filter
SELECT * FROM data
WHERE indexed_col = 5 AND complex_func(x) > 0;
```

This hybrid approach combines index efficiency with parallel computation power.

## Memory Considerations

Parallel execution uses more memory than sequential:
- Each thread maintains its own working set
- Results are collected from all threads

For very large result sets, consider:
- Adding more restrictive WHERE clauses
- Using LIMIT to reduce result size
- Ensuring adequate system memory

## Monitoring

### Query Timing

Use `EXPLAIN ANALYZE` to compare sequential vs parallel performance:

```sql
-- Shows actual execution time
EXPLAIN ANALYZE SELECT * FROM large_table WHERE value > 100;
```

### CPU Usage

During parallel queries, you should see:
- Multiple CPU cores utilized
- Higher overall CPU usage
- Faster query completion

## Best Practices

1. **Let the optimizer decide**: Parallel execution is automatic
2. **Use EXPLAIN ANALYZE**: Verify parallelization is being used
3. **Index appropriately**: Combine indexes with parallel filtering
4. **Monitor memory**: Large parallel queries use more memory
5. **Test with production data sizes**: Parallelization benefits depend on scale
