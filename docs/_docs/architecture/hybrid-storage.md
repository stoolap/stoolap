---
title: Storage Architecture
category: Architecture
order: 1
---

# Storage Architecture

This document explains Stoolap's storage architecture, including how data is stored, indexed, and accessed.

## Storage Overview

Stoolap uses a row-based storage design optimized for both transactional and analytical workloads:

- **Row-Based Version Store** - Primary storage with MVCC for transactions
- **Multiple Index Types** - B-tree, Hash, and Bitmap indexes for different query patterns
- **Write-Ahead Logging** - Durability through WAL and snapshots

## Row-Based Storage

Stoolap's storage is designed for efficient transactional operations:

### Advantages

- **Efficient Record Access** - All fields of a record are stored together
- **Low-latency Updates** - Fast for targeted modifications
- **Transaction Efficiency** - Optimized for ACID transaction processing
- **Write Optimization** - Efficient for inserting complete records

### Implementation

Stoolap's storage consists of:

- **Version Store** - Tracks row versions for MVCC
- **Transaction Management** - Ensures ACID properties
- **In-Memory Tables** - Primary working set kept in memory
- **Disk Persistence** - Optional WAL and snapshots for durability

## Index System

Stoolap supports multiple index types, each optimized for different query patterns:

### B-tree Indexes

- **Default for**: INTEGER, FLOAT, TIMESTAMP columns
- **Strengths**: Range queries, equality lookups, sorting
- **Use cases**: Price ranges, date ranges, numeric comparisons

```sql
CREATE INDEX idx_price ON products(price) USING BTREE;
```

### Hash Indexes

- **Default for**: TEXT, JSON columns
- **Strengths**: O(1) equality lookups
- **Use cases**: Email lookups, exact string matches

```sql
CREATE INDEX idx_email ON users(email) USING HASH;
```

### Bitmap Indexes

- **Default for**: BOOLEAN columns
- **Strengths**: Low-cardinality columns, fast AND/OR operations
- **Use cases**: Status flags, boolean fields

```sql
CREATE INDEX idx_active ON users(active) USING BITMAP;
```

### Multi-Column Indexes

- **Strengths**: Queries filtering on multiple columns
- **Use cases**: Composite lookups, unique constraints

```sql
CREATE INDEX idx_cust_date ON orders(customer_id, order_date);
```

## Query Optimization

Stoolap's query optimizer can route queries to the appropriate index:

- **Index Selection** - Cost-based selection of best index
- **Index Intersection** - Combining multiple indexes for AND conditions
- **Statistics-Based** - Uses ANALYZE data for better estimates

### Consistency

- All data views remain consistent through the MVCC mechanism
- Queries see a transactionally consistent snapshot
- No synchronization delay between reads and writes

## Implementation Details

Key components implementing this architecture:

- `src/storage/mvcc/table.rs` - The table implementation with row-based storage
- `src/storage/mvcc/version_store.rs` - Manages row versions and MVCC
- `src/storage/mvcc/btree_index.rs` - B-tree index implementation
- `src/storage/mvcc/hash_index.rs` - Hash index implementation
- `src/storage/mvcc/bitmap_index.rs` - Bitmap index implementation
- `src/storage/mvcc/multi_column_index.rs` - Multi-column index implementation

## Optimizations

Several optimizations improve performance:

### For Transactions

- **Optimistic Concurrency Control** - Reduces locking overhead
- **Transaction Batching** - Processes multiple operations at once
- **Efficient Version Chain** - Optimized layout for version traversal

### For Queries

- **Predicate Pushdown** - Filters applied at the storage level
- **Index Intersection** - Combining multiple indexes
- **Parallel Execution** - Multi-threaded query processing for large datasets

## Performance Considerations

### OLTP Performance

- **Point Lookups** - Fast due to row-based storage and indexes
- **Small Transactions** - Low overhead for common operations
- **High Concurrency** - Efficient handling of many simultaneous transactions

### Analytical Performance

- **Aggregations** - Optimized through parallel execution
- **Complex Filtering** - Accelerated by appropriate index selection
- **Large Scans** - Parallelized for better throughput

## Best Practices

To get the most out of Stoolap's storage architecture:

1. **Index Design** - Create indexes on frequently filtered columns
2. **Index Type Selection** - Use USING clause when default isn't optimal
3. **Transaction Sizing** - Keep transactions appropriately sized
4. **ANALYZE Usage** - Run ANALYZE for better query planning
5. **Data Types** - Use appropriate data types for better index performance
