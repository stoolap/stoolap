---
layout: doc
title: Parallel Execution
category: Performance
order: 5
---

# Parallel Execution in Stoolap

## Overview

Stoolap's query execution engine is designed to accelerate SQL query processing by operating on data in parallel batches, allowing for better CPU utilization across multiple cores. This document explains the architecture and components of the parallel execution system.

## Key Concepts

### Batch Processing

The parallel execution model processes data in batches to improve throughput. Instead of processing one row at a time, the engine processes multiple rows simultaneously using Rayon's work-stealing scheduler.

Key characteristics:
- Automatic parallelization based on data size
- Work-stealing for optimal load balancing
- Configurable thresholds for parallel execution

### Parallelization Thresholds

Stoolap automatically parallelizes operations based on data size:

| Operation | Threshold | Description |
|-----------|-----------|-------------|
| **Filter (WHERE)** | 10,000 rows | Parallel predicate evaluation |
| **Hash Join** | 5,000 rows | Parallel hash build and probe |
| **ORDER BY** | 50,000 rows | Parallel sorting |
| **DISTINCT** | 10,000 rows | Two-phase parallel deduplication |

## Architecture Components

### 1. Parallel Filter

For large tables, WHERE clause evaluation is parallelized:

- Data is divided into chunks
- Each chunk is processed by a separate thread
- Results are merged using efficient concurrent data structures

### 2. Parallel Hash Join

Hash joins are parallelized in two phases:

**Build Phase:**
- The build side is partitioned across threads
- Hash table is constructed using DashMap (concurrent hash map)

**Probe Phase:**
- The probe side is processed in parallel
- Each thread looks up matches in the shared hash table

### 3. Parallel Sort

Large ORDER BY operations use parallel sorting:

- Uses Rayon's `par_sort_by()` for efficient multi-threaded sorting
- Automatically falls back to sequential sort for small datasets

### 4. Parallel Distinct

DISTINCT operations use two-phase deduplication:

- First phase: parallel identification of unique values per chunk
- Second phase: merge of partial results

## Query Flow

1. **Query Planning**: The planner estimates cardinality and decides on parallel execution.

2. **Data Fetching**: Data is fetched from storage with index acceleration where possible.

3. **Parallel Processing**: Operations exceeding thresholds are parallelized.

4. **Result Merging**: Partial results are combined efficiently.

5. **Result Formation**: Final results are returned to the caller.

## Performance Benefits

Parallel execution provides significant benefits for analytical queries:

- **Filter Operations**: 2-5x speedup for complex predicates
- **Hash Joins**: 2-4x speedup for large joins
- **Sorting**: 3-6x speedup for large ORDER BY
- **Aggregations**: Linear speedup with core count

## EXPLAIN ANALYZE Output

Use `EXPLAIN ANALYZE` to see parallel execution in action:

```sql
EXPLAIN ANALYZE SELECT * FROM large_table WHERE value > 100;
-- Output shows: Parallel Seq Scan on large_table (workers=N)
```

The output indicates how many worker threads were used for each operation.

## Query Types That Benefit Most

Parallel execution provides the greatest benefit for:

1. **Analytical queries** that process large amounts of data
2. **Filter-heavy operations** with complex conditions
3. **Large joins** between tables
4. **Large aggregations** over many rows
5. **Sorting large result sets**

## Best Practices

For optimal performance with Stoolap's parallel execution:

1. **Ensure sufficient data**: Parallel overhead only pays off for larger datasets
2. **Use appropriate indexes**: Even with parallelism, indexes are still important
3. **Analyze tables**: Run ANALYZE to give the optimizer accurate cardinality estimates
4. **Check EXPLAIN output**: Verify parallel execution is being used as expected

## Implementation Details

### Rayon Integration

Stoolap uses Rayon for parallel execution:

- Work-stealing scheduler for optimal load balancing
- Automatic thread pool management
- Zero-cost abstraction over parallelism

### Memory Management

Parallel execution uses efficient memory patterns:

- DashMap for concurrent hash tables
- Chunk-based processing to limit memory usage
- Efficient result merging to avoid copies

## Current Capabilities

The parallel execution engine supports:

- Parallel filter evaluation
- Parallel hash join (build and probe)
- Parallel sorting
- Parallel distinct/deduplication
- Parallel aggregation
