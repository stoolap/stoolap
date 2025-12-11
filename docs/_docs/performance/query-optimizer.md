---
layout: doc
title: Query Optimizer
category: Performance
order: 2
---

# Query Optimizer

Stoolap includes a sophisticated cost-based query optimizer that analyzes queries and chooses efficient execution plans. The optimizer uses table statistics, index information, and cost models to make intelligent decisions.

## Overview

The query optimizer performs several tasks:
- **Cost estimation**: Calculates the expected cost of different execution strategies
- **Access method selection**: Chooses between table scans, index lookups, and primary key access
- **Join ordering**: Determines the optimal order to join multiple tables
- **Join algorithm selection**: Chooses Hash Join, Merge Join, or Nested Loop Join
- **Predicate pushdown**: Moves filters close to data sources
- **Expression simplification**: Optimizes boolean and arithmetic expressions

## Cost Model

### Cost Components

The optimizer considers two main cost components:

1. **I/O Cost**: Reading data from storage
   - Sequential page reads (table scans)
   - Random page reads (index access)

2. **CPU Cost**: Processing data
   - Tuple processing
   - Predicate evaluation
   - Join operations

### Access Methods

The optimizer chooses from these access methods:

| Method | Use Case | Cost |
|--------|----------|------|
| Primary Key Lookup | WHERE pk = value | Lowest (O(1)) |
| Index Scan | WHERE indexed_col = value | Low (O(log n)) |
| Index Range Scan | WHERE indexed_col > value | Medium |
| Sequential Scan | No usable index | Highest (O(n)) |

### Example

```sql
-- The optimizer evaluates multiple strategies
SELECT * FROM users WHERE id = 123;

-- Strategy 1: Sequential scan - O(n) rows
-- Strategy 2: PK lookup - O(1)
-- Winner: PK lookup (much lower cost)
```

## Table Statistics

The optimizer uses statistics collected by `ANALYZE` to make better decisions.

### ANALYZE Command

```sql
-- Collect statistics for a table
ANALYZE users;

-- Statistics are stored in system tables
SELECT * FROM _sys_table_stats WHERE table_name = 'users';
SELECT * FROM _sys_column_stats WHERE table_name = 'users';
```

### Statistics Collected

| Statistic | Description |
|-----------|-------------|
| Row count | Total number of rows in the table |
| Distinct count | Number of unique values per column |
| NULL count | Number of NULL values per column |
| Min/Max values | Range of values per column |
| Histograms | Distribution of values for selectivity estimation |

### Selectivity Estimation

Statistics help estimate how selective predicates are:

```sql
-- Without statistics: assume 10% selectivity
-- With statistics: use actual data distribution
SELECT * FROM orders WHERE status = 'pending';
```

## Join Optimization

### Join Ordering

For queries joining multiple tables, the optimizer uses dynamic programming to find the optimal join order:

```sql
-- The optimizer considers all possible join orders
SELECT *
FROM orders o
JOIN customers c ON o.customer_id = c.id
JOIN products p ON o.product_id = p.id;

-- Possible orders:
-- 1. (orders JOIN customers) JOIN products
-- 2. (orders JOIN products) JOIN customers
-- 3. (customers JOIN orders) JOIN products
-- etc.
```

### Join Algorithms

The optimizer selects the best join algorithm:

| Algorithm | Best For | Requirements |
|-----------|----------|--------------|
| Hash Join | Large tables, equality joins | Memory for hash table |
| Merge Join | Pre-sorted inputs | Sorted data |
| Nested Loop | Small tables, any join condition | None |

### Semi-Join Optimization

For EXISTS and IN subqueries, the optimizer may use semi-join:

```sql
-- Can use semi-join optimization
SELECT * FROM customers c
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id);
```

## Adaptive Query Execution

Stoolap implements Adaptive Query Execution (AQE), which can adjust the execution plan at runtime based on actual data characteristics.

### How AQE Works

1. **Initial Plan**: Optimizer creates plan using statistics
2. **Runtime Monitoring**: Actual row counts are tracked during execution
3. **Plan Adjustment**: If actual differs significantly from estimated, the plan may be adjusted

### Cardinality Feedback

The optimizer learns from query execution:

```sql
-- First execution: uses estimated cardinality
SELECT * FROM orders WHERE amount > 1000;

-- Subsequent executions: uses feedback from actual row counts
-- This improves future estimates for similar queries
```

## Bloom Filter Propagation

For join-heavy queries, the optimizer can use Bloom filters to reduce data movement:

```sql
-- Bloom filter can pre-filter orders before join
SELECT * FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE c.region = 'US';
```

The filter on `customers.region` creates a Bloom filter that filters `orders` early.

## Expression Simplification

The optimizer simplifies expressions when possible:

### Constant Folding

```sql
-- Before: SELECT * FROM t WHERE x > 1 + 1
-- After:  SELECT * FROM t WHERE x > 2
```

### Boolean Optimization

```sql
-- Before: SELECT * FROM t WHERE true AND x > 5
-- After:  SELECT * FROM t WHERE x > 5

-- Before: SELECT * FROM t WHERE false OR x > 5
-- After:  SELECT * FROM t WHERE x > 5
```

### Predicate Merging

```sql
-- Before: WHERE x > 5 AND x > 10
-- After:  WHERE x > 10

-- Before: WHERE x = 5 AND x = 5
-- After:  WHERE x = 5
```

## Viewing Query Plans

### EXPLAIN

Shows the planned execution strategy:

```sql
EXPLAIN SELECT * FROM users WHERE id = 1;
```

Output:
```
plan
----
SELECT
  Columns: *
  -> PK Lookup on users
       id = 1
```

### EXPLAIN ANALYZE

Shows the plan with actual runtime statistics:

```sql
EXPLAIN ANALYZE SELECT * FROM users WHERE status = 'active';
```

Output:
```
plan
----
SELECT (actual time=2.5ms, rows=1500)
  Columns: *
  -> Seq Scan on users (actual rows=1500)
       Filter: (status = 'active')
```

## Best Practices

### Keep Statistics Updated

Run ANALYZE after significant data changes:

```sql
-- After bulk insert
INSERT INTO orders SELECT * FROM staging_orders;
ANALYZE orders;
```

### Create Appropriate Indexes

The optimizer can only use indexes that exist:

```sql
-- Create index for common query patterns
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date);
```

### Use EXPLAIN to Understand Plans

Before optimizing, understand current behavior:

```sql
-- Check what plan is being used
EXPLAIN ANALYZE SELECT * FROM orders WHERE customer_id = 100;
```

### Consider Query Patterns

Design indexes based on how queries filter and join:

```sql
-- If queries often filter by status and date together
CREATE INDEX idx_orders_status_date ON orders(status, order_date);
```

## Cost Constants

The optimizer uses these tunable constants:

| Constant | Description | Default |
|----------|-------------|---------|
| cpu_tuple_cost | Cost to process one row | 0.01 |
| cpu_operator_cost | Cost to evaluate one predicate | 0.0025 |
| seq_page_cost | Cost for sequential I/O | 1.0 |
| random_page_cost | Cost for random I/O | 2.0 |
| pk_lookup_cost | Cost for primary key lookup | 0.1 |

These values are tuned for in-memory operation and may differ from disk-based databases.
