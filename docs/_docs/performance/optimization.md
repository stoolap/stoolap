---
layout: doc
title: Performance Optimization
category: Performance
order: 1
---

# Performance Optimization in Stoolap

This document provides guidelines and best practices for optimizing performance in Stoolap, including database design, query optimization, and system configuration.

## Database Design Optimization

### Table Design

- **Choose appropriate data types** - Use the smallest data type that can reliably store your data
- **Normalize when appropriate** - Balance normalization for data integrity with denormalization for query performance
- **Use primary keys** - Every table should have a primary key, preferably a simple integer
- **Consider column order** - Place frequently used columns first
- **Limit column count** - Tables with fewer columns generally perform better

### Indexing Strategy

- **Index selection** - Create indexes on columns used in WHERE, JOIN, and ORDER BY clauses
- **Multi-column indexes** - Create multi-column indexes for frequently combined filters
- **Index order matters** - For multi-column indexes, place high-selectivity columns first
- **Avoid over-indexing** - Each index increases write overhead and storage requirements
- **Monitor index usage** - Periodically review which indexes are being used

```sql
-- Create an index on a frequently filtered column
CREATE INDEX idx_user_email ON users (email);

-- Create a multi-column index for common query patterns
CREATE INDEX idx_product_category_price ON products (category_id, price);
```

## Query Optimization

### Recent Performance Improvements

Stoolap includes significant performance optimizations:

- **Hash-Based IN Subqueries** - Converts large IN/NOT IN lookups to hash probes for substantial speedups on many workloads
- **Array-Based Row Storage** - Eliminated map allocations throughout query execution
- **Efficient Aggregations** - Optimized operations for COUNT, SUM, AVG, MIN, MAX
- **Parallel Execution** - Rayon-based parallelism for filter, join, sort, and distinct operations

### SELECT Statement Optimization

- **Select only needed columns** - Avoid `SELECT *` when possible
- **Use WHERE clauses effectively** - Apply filters early to reduce the result set
- **Leverage indexes** - Ensure queries can use available indexes
- **Minimize function calls** - Avoid functions on indexed columns in WHERE clauses
- **Use LIMIT for large result sets** - Apply LIMIT to prevent excessive memory usage

```sql
-- Instead of this:
SELECT * FROM large_table WHERE status = 'active';

-- Do this:
SELECT id, name, created_at FROM large_table WHERE status = 'active' LIMIT 1000;
```

### JOIN Optimization

- **Join order** - Join smaller tables first when possible
- **Use appropriate join types** - Choose INNER, LEFT, RIGHT joins as needed
- **Index join columns** - Ensure columns used in join conditions are indexed
- **Consider denormalization** - For critical queries, strategic denormalization may help

```sql
-- Ensure both user_id in orders and id in users are indexed
SELECT u.name, o.order_date FROM orders o
JOIN users u ON o.user_id = u.id
WHERE o.status = 'completed';
```

### Aggregate Query Optimization

- **Filter before aggregating** - Apply WHERE clauses before GROUP BY
- **Index GROUP BY columns** - Ensure columns used in GROUP BY are indexed
- **Use HAVING efficiently** - Apply HAVING only for conditions on aggregated results

```sql
-- Filter first, then aggregate
SELECT category_id, COUNT(*) FROM products
WHERE price > 100
GROUP BY category_id
HAVING COUNT(*) > 10;
```

## Prepared Statements

Use prepared statements for repeated queries to leverage query caching:

```rust
// Prepare once, execute many times
let stmt = db.prepare("SELECT * FROM users WHERE id = $1")?;

for id in 1..=100 {
    for row in stmt.query((id,))? {
        let row = row?;
        // Process row...
    }
}
```

## Parallel Execution

Stoolap's parallel execution engine processes data using Rayon's work-stealing scheduler:

- **Automatic Parallelization** - Operations exceeding thresholds are automatically parallelized
- **Work-Stealing** - Optimal load balancing across CPU cores
- **Configurable Thresholds** - Filter (10K rows), Hash Join (5K), ORDER BY (50K), DISTINCT (10K)

## Transaction Management

- **Keep transactions short** - Long-running transactions can impact concurrency
- **Choose appropriate isolation level** - Use the minimum isolation level needed
- **Batch operations** - Group related operations within a single transaction
- **Handle conflicts** - Implement retry logic for optimistic concurrency conflicts

```sql
-- Example of a focused transaction
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
```

## Memory Management

- **Buffer pool sizing** - Configure buffer pool size based on available memory
- **Query memory limits** - Set appropriate memory limits for complex queries
- **Monitor memory usage** - Watch for excessive memory consumption

## Bulk Operations

Use bulk operations for better performance:

```sql
-- Bulk insert example
INSERT INTO products (name, price, category_id) VALUES
  ('Product A', 10.99, 1),
  ('Product B', 15.99, 1),
  ('Product C', 20.99, 2);
```

## Performance Monitoring

Stoolap provides several performance monitoring features:

- **EXPLAIN** - Shows the query execution plan
- **EXPLAIN ANALYZE** - Shows the plan with actual runtime statistics
- **Query timing** - The CLI displays execution time for each query
- **Row counts** - Result sets show the number of rows affected/returned

```sql
-- View query plan
EXPLAIN SELECT * FROM orders WHERE amount > 1000;

-- View plan with actual execution statistics
EXPLAIN ANALYZE SELECT * FROM products WHERE category = 'Electronics';
```

See [EXPLAIN]({% link _docs/sql-features/explain.md %}) for detailed documentation.

## Implementation-Specific Optimizations

Stoolap includes several specialized optimizations:

### Parallel Processing

Stoolap uses Rayon for parallel query execution:

- **Parallel Filtering** - Predicate evaluation across multiple threads
- **Parallel Aggregation** - Concurrent aggregation operations with DashMap
- **Parallel Sorting** - Multi-threaded sorting via `par_sort_by()`

### Custom Data Structures

Stoolap uses specialized data structures for better performance:

- **DashMap** - Concurrent hash tables for parallel operations
- **Efficient Index Structures** - B-tree, Hash, and Bitmap indexes
- **Row Version Chains** - MVCC implementation with version chains

## Advanced Optimization Techniques

### Expression Pushdown

Stoolap pushes down expressions to minimize data processing:

```sql
-- Filter and projection will be pushed down to the storage layer
SELECT name, price FROM products WHERE price > 100;
```

### Join Algorithms

Stoolap selects among several join algorithms:

- **Hash Join** - For equality joins with large tables
- **Merge Join** - For pre-sorted data
- **Nested Loop Join** - For small tables or when using indexes

### Parallel Execution

Stoolap can execute some operations in parallel:

- **Parallel scans** - Multiple segments scanned concurrently
- **Parallel aggregations** - Divided work for faster aggregations
- **Concurrent index operations** - Background index building

## CTE and Subquery Optimization

Stoolap includes advanced optimizations for CTEs and subqueries:

### Common Table Expressions (CTEs)

- **Single Materialization**: CTEs are evaluated once and reused throughout the query
- **Efficient Storage**: CTE results are stored efficiently in memory
- **Optimized Aggregations**: Fast operations for COUNT, SUM, AVG, MIN, MAX

```sql
-- Efficient: CTE materializes once
WITH summary AS (
    SELECT region, product_id, SUM(amount) as total
    FROM sales
    WHERE year = 2024
    GROUP BY region, product_id
)
SELECT
    s1.region,
    COUNT(*) as product_count,
    SUM(s1.total) as region_total
FROM summary s1
GROUP BY s1.region;
```

### Subquery Optimization

- **Hash-Based IN/NOT IN**: Automatic conversion to hash lookups
- **Performance Gains**: Can be substantial for large IN/NOT IN lists; verify with `EXPLAIN ANALYZE` on your workload
- **Memory Efficiency**: O(1) lookups instead of O(n×m) comparisons

```sql
-- Automatically optimized with hash table
DELETE FROM orders
WHERE customer_id IN (
    SELECT id FROM customers
    WHERE last_order < '2024-01-01'
);

-- Scalar subqueries are optimized
SELECT name, salary,
    (SELECT AVG(salary) FROM employees) as company_avg
FROM employees;
```

### Best Practices for CTEs and Subqueries

1. **Use CTEs for Complex Queries**: Break down complex logic into readable CTEs
2. **Filter Early**: Apply WHERE clauses in CTE definitions
3. **Avoid Redundant Subqueries**: Use CTEs when the same subquery is needed multiple times

## Best Practices Summary

1. **Design schema carefully** - Choose appropriate data types and normalization level
2. **Create targeted indexes** - Index columns used in filters, joins, and sorts
3. **Write optimized queries** - Select only needed columns and filter early
4. **Use prepared statements** - Leverage the query cache for repeated queries
5. **Manage transactions efficiently** - Keep transactions short and focused
6. **Monitor performance** - Use query timing and row counts to identify bottlenecks
7. **Rebuild indexes** - Recreate indexes periodically for optimal performance
8. **Consider bulk operations** - Use bulk inserts and updates for better throughput
9. **Leverage parallel execution** - Structure operations to benefit from multi-threaded processing
10. **Configure for your workload** - Adjust memory settings based on your specific needs
11. **Use CTEs for complex queries** - Benefit from single materialization
12. **Optimize subqueries** - Leverage automatic hash-based IN/NOT IN optimization
