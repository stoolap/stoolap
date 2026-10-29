---
layout: doc
title: EXPLAIN
category: SQL Features
order: 13
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

A SELECT over one table is planned once, when it is prepared, and EXPLAIN prints that plan: the steps the rows go through, outermost first, each indented under the step it feeds. The access at the bottom is the one the run takes, and the indexes it names are the ones it reads. Other statements (joins, GROUP BY, window functions, DISTINCT, EXPLAIN ANALYZE) print the format shown in their own sections below.

### Sequential Scan

When no suitable index exists:

```sql
EXPLAIN SELECT * FROM users;
```

Output:
```
plan
----
Project id, name, email
  SeqScan users
```

### Primary Key Lookup

When filtering by primary key, the row is read by its key:

```sql
EXPLAIN SELECT * FROM users WHERE id = 1;
```

Output:
```
plan
----
Project id, name, email
  PkGet users id = 1
  else IndexScan users (id = 1)
    PkKey
```

A run whose key is not an integer, or one in a transaction with a timeout, takes the read after `else`.

### Index Scan

When using an index for equality, the scan line shows the condition every row it reads is judged by, and the line under it the index that finds the rows:

```sql
CREATE INDEX idx_category ON products(category);
EXPLAIN SELECT * FROM products WHERE category = 'Electronics';
```

Output:
```
plan
----
Project id, name, category, value
  IndexScan products (category = 'Electronics')
    Index idx_category (category =)
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
Project id, name, category, value
  IndexScan products (value > 100)
    Index idx_value (value range)
```

### Several Indexes

Conditions on several indexed columns joined by AND read the rows every index names; joined by OR, the rows any index names. An OR with a branch no index answers is scanned:

```sql
EXPLAIN SELECT name FROM products WHERE category = 'Electronics' AND value > 15;
EXPLAIN SELECT * FROM products WHERE category = 'Books' OR value > 25;
```

Output:
```
plan
----
Project name
  IndexScan products ((category = 'Electronics') AND (value > 15))
    Intersect
      Index idx_category (category =)
      Index idx_value (value range)

plan
----
Project id, name, category, value
  IndexScan products ((category = 'Books') OR (value > 25))
    Union
      Index idx_category (category =)
      Index idx_value (value range)
```

A multi-column index shows as `CompositeIndex name (a =, b range)`: equalities on its leading columns and a range, with an upper bound, on the next one.

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
Sort value DESC
  Project id, name, category, value
    SeqScan products
```

Under a LIMIT, an ORDER BY of one column reads the rows in that order instead of sorting them all: through the column's index or the primary key (`IndexOrder`), from a lower bound on the primary key (`Keyset`), or by the table's top-k scan (`TopK`). A run the table cannot answer that way takes the next line after `else`: the top-k scan, then a `TopN` over the read beneath it, which keeps only the rows the LIMIT keeps:

```sql
EXPLAIN SELECT * FROM products ORDER BY value DESC LIMIT 5;
EXPLAIN SELECT * FROM products WHERE id > 1 ORDER BY id LIMIT 5;
```

Output:
```
plan
----
Limit 5
  Project id, name, category, value
    IndexOrder products value DESC
    else TopK products value DESC
    else TopN value DESC
      SeqScan products

plan
----
Limit 5
  Project id, name, category, value
    Keyset products id ASC (id > 1)
    else TopK products id ASC (id > 1)
    else TopN id ASC
      IndexScan products (id > 1)
        Index __pk_products_id (id range)
```

A LIMIT or OFFSET given as a parameter or a subquery is read on each run and takes the same plan; EXPLAIN shows it as written:

```sql
EXPLAIN SELECT * FROM products ORDER BY value DESC LIMIT $1;
```

Output:
```
plan
----
Limit $1
  Project id, name, category, value
    IndexOrder products value DESC
    else TopK products value DESC
    else TopN value DESC
      SeqScan products
```

A WHERE with a term that reads no column, such as a parameter or a literal comparison, reads no row on a run where that term is not true:

```sql
EXPLAIN SELECT * FROM products WHERE 1 = 0 AND value > 15;
```

Output:
```
plan
----
Project id, name, category, value
  Empty unless (1 = 0)
  else IndexScan products ((1 = 0) AND (value > 15))
    Index idx_value (value range)
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

For a planned single-table SELECT:

| Component | Description |
|-----------|-------------|
| SeqScan | Full table scan; the condition after the table name is the one each row is judged by |
| PkGet | Reads the one row the primary key names |
| IndexScan | Reads the rows the indexes under it name, each judged by the condition after the table name |
| Index | One index and what it reads: an equality (`=`), an `IN` list, a `LIKE prefix` or a `range` |
| CompositeIndex | A multi-column index: equalities on its leading columns, a range on the next one |
| PkKey, PkKeys | Rows a primary key equality or IN list names, inside an IndexScan |
| Intersect, Union | The rows every index under it names, or any of them names |
| IndexOrder, Keyset, TopK | Rows read in ORDER BY order under a LIMIT, with no sort of every row |
| Empty unless | The WHERE's terms that read no column; a run where they are not true reads no row |
| else | The read a run takes when the table cannot answer the line above it |
| Filter | The part of the WHERE judged after the scan |
| Project | The output columns |
| TopN | The ORDER BY under a LIMIT, keeping only the rows the LIMIT keeps |
| Sort | The ORDER BY over every row |
| Limit | The LIMIT and OFFSET |

For other statements:

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
| Nested Loop | Loops over each row combination |
| Join Cond | Join condition |

### Modifiers

| Component | Description |
|-----------|-------------|
| Group By | Columns for aggregation |
| Order By | Sort specification |
| Limit | Row count limit |
| Alias | Table alias in query |
| Prepared | For a WHERE with a subquery, or a join whose ON holds one: `yes` when the statement runs as a [prepared subquery]({% link _docs/sql-features/subqueries.md %}#prepared-subqueries), otherwise `no` and the reason |

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
