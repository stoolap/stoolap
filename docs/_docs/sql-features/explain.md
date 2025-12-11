---
title: EXPLAIN
category: SQL Features
order: 10
---

# EXPLAIN

The EXPLAIN command shows the query execution plan that Stoolap will use to execute a query. EXPLAIN ANALYZE additionally shows actual runtime statistics after executing the query.

## Basic Syntax

```sql
-- Show planned execution strategy
EXPLAIN query;

-- Show plan with actual runtime statistics
EXPLAIN ANALYZE query;
```

## EXPLAIN

Shows the execution plan without running the query.

### Sequential Scan

When no suitable index exists:

```sql
EXPLAIN SELECT * FROM users;
```

Output:
```
plan
----
SELECT
  Columns: *
  -> Seq Scan on users
```

### Primary Key Lookup

When filtering by primary key:

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

### Index Scan

When using an index for equality:

```sql
CREATE INDEX idx_category ON products(category);
EXPLAIN SELECT * FROM products WHERE category = 'Electronics';
```

Output:
```
plan
----
SELECT
  Columns: *
  -> Index Scan using idx_category on products
       Index Cond: category = Electronics
```

### Index Range Scan

When using an index for range queries:

```sql
CREATE INDEX idx_value ON products(value);
EXPLAIN SELECT * FROM products WHERE value > 100;
```

Output:
```
plan
----
SELECT
  Columns: *
  -> Index Scan using idx_value on products
       Index Cond: value > 100
```

### Hash Join

When joining tables:

```sql
EXPLAIN SELECT o.id, c.name
FROM orders o
JOIN customers c ON o.customer_id = c.id;
```

Output:
```
plan
----
SELECT
  Columns: o.id, c.name
  -> Hash Join (INNER Join) (cost=1.10 rows=1)
     Join Cond: (o.customer_id = c.id)
    -> Seq Scan on orders
       Alias: o
    -> Seq Scan on customers
       Alias: c
```

### Aggregation

When using GROUP BY:

```sql
EXPLAIN SELECT category, SUM(value) FROM products GROUP BY category;
```

Output:
```
plan
----
SELECT
  Columns: category, SUM(value)
  -> Seq Scan on products
  Group By: category
```

### Sorting

When using ORDER BY:

```sql
EXPLAIN SELECT * FROM products ORDER BY value DESC;
```

Output:
```
plan
----
SELECT
  Columns: *
  -> Seq Scan on products
  Order By: value DESC
```

## EXPLAIN ANALYZE

Executes the query and shows actual runtime statistics.

```sql
EXPLAIN ANALYZE SELECT * FROM products WHERE value > 15;
```

Output:
```
plan
----
SELECT (actual time=1.53ms, rows=2)
  Columns: *
  -> Seq Scan on products (actual rows=2)
       Filter: (value > 15)
```

### Understanding EXPLAIN ANALYZE Output

| Field | Description |
|-------|-------------|
| actual time | Total execution time |
| rows | Number of rows returned |
| actual rows | Rows processed at each step |
| Filter | WHERE condition applied |

### Comparing Estimated vs Actual

EXPLAIN ANALYZE helps identify inaccurate estimates:

```sql
EXPLAIN ANALYZE SELECT * FROM large_table WHERE rare_condition = true;
```

If estimated rows differs significantly from actual rows:
- Run `ANALYZE table_name` to update statistics
- Consider adding an index for the condition

## Plan Components

### Access Methods

| Component | Description |
|-----------|-------------|
| Seq Scan | Full table scan (reads all rows) |
| PK Lookup | Direct access by primary key |
| Index Scan | Uses index to find rows |
| Index Cond | Condition pushed to index |
| Filter | Condition applied after scan |

### Join Methods

| Component | Description |
|-----------|-------------|
| Hash Join | Builds hash table, probes with other table |
| Merge Join | Joins sorted inputs |
| Nested Loop | Loops over each row combination |
| Join Cond | Join condition |

### Modifiers

| Component | Description |
|-----------|-------------|
| Group By | Columns for aggregation |
| Order By | Sort specification |
| Limit | Row count limit |
| Alias | Table alias in query |

## Cost Information

EXPLAIN shows cost estimates for joins:

```
-> Hash Join (INNER Join) (cost=1.10 rows=1)
```

- **cost**: Estimated total cost (relative units)
- **rows**: Estimated number of rows

Lower cost generally means faster execution.

## Use Cases

### Query Optimization

Use EXPLAIN to understand query behavior:

```sql
-- Check if index is being used
EXPLAIN SELECT * FROM orders WHERE customer_id = 100;

-- If showing Seq Scan, consider adding an index
CREATE INDEX idx_customer ON orders(customer_id);

-- Verify index is now used
EXPLAIN SELECT * FROM orders WHERE customer_id = 100;
```

### Performance Debugging

Use EXPLAIN ANALYZE to find slow operations:

```sql
-- Find where time is spent
EXPLAIN ANALYZE SELECT o.*, c.name
FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE o.amount > 1000;
```

Look for:
- Large "actual rows" values
- Operations taking significant time
- Mismatch between estimated and actual rows

### Index Effectiveness

Compare plans with and without indexes:

```sql
-- Without index
EXPLAIN ANALYZE SELECT * FROM products WHERE category = 'Electronics';

-- Add index
CREATE INDEX idx_category ON products(category);

-- With index
EXPLAIN ANALYZE SELECT * FROM products WHERE category = 'Electronics';
```

## Best Practices

1. **Use EXPLAIN first**: Check plan before running expensive queries
2. **Use EXPLAIN ANALYZE for optimization**: Get actual numbers for tuning
3. **Check for Seq Scan on large tables**: Consider adding indexes
4. **Verify index usage**: Ensure expected indexes are being used
5. **Update statistics**: Run ANALYZE after significant data changes

## Limitations

- EXPLAIN shows the planned execution, not necessarily the exact runtime behavior
- Cost values are relative, not absolute time measurements
- Some runtime optimizations may not appear in the plan
