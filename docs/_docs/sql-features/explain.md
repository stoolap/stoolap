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

A SELECT over one table, view, CTE, derived table, VALUES list or table function is planned once, when it is prepared, and EXPLAIN prints that plan: the steps the rows go through, outermost first, each indented under the step it feeds. The access at the bottom is the one the run takes, and the indexes it names are the ones it reads. Other statements (joins, a GROUP BY other than of one column with no WHERE, window functions, DISTINCT or a set operation over anything but a table, a WHERE or select list with a subquery, an aggregate that reads its rows with no source of its own, EXPLAIN ANALYZE) print the format shown in their own sections below.

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

An equality on a column with a unique index of its own reads the one row that index names. The row is still judged by the whole WHERE:

```sql
CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE);
EXPLAIN SELECT id, name FROM users WHERE email = 'alice@example.com';
```

Output:
```
plan
----
Project id, name
  Filter
    UniqueGet users email = 'alice@example.com'
    else IndexScan users (email = 'alice@example.com')
      Index unique_users_email (email =)
```

A run inside a transaction, or one while a commit is publishing to the table, takes the read after `else`.

An UPDATE or DELETE whose WHERE is the primary key's equality reads its row the same way, when it has no RETURNING or WITH, keeps no foreign key, and runs no CHECK on a column it sets. Its SET values show in column order:

```sql
EXPLAIN UPDATE users SET email = 'new@example.com' WHERE id = 1;
```

Output:
```
plan
----
Update users SET email = 'new@example.com'
  PkGet users id = 1
  else IndexScan users (id = 1)
    PkKey
```

Any other UPDATE or DELETE shows its clauses:

```sql
EXPLAIN UPDATE users SET name = 'x' WHERE email = 'a';
```

Output:
```
plan
----
UPDATE users
  Set: 1 column(s)
  Filter: (email = 'a')
```

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

### Aggregates Over One Table

COUNT, SUM, AVG, MIN and MAX over a whole table, with no WHERE or GROUP BY, are answered from what the table keeps without reading its rows: its row count, an index's count of distinct values, a column's sum and bounds (`Stats`). A single MIN or MAX of a column with a B-tree index, or of the primary key, reads the index's first or last key (`IndexBound`). A run the table cannot answer that way takes the line after `else`, down to reading every row:

```sql
EXPLAIN SELECT COUNT(*), SUM(value) FROM products;
EXPLAIN SELECT MAX(id) FROM products;
```

Output:
```
plan
----
Aggregate COUNT(*), SUM(value)
  Stats products
  else SeqScan products

plan
----
Aggregate MAX(id)
  IndexBound products (id)
  else Stats products
  else SeqScan products
```

With a WHERE, the table computes the calls over the rows the WHERE passes (`StorageFiltered`) when its volumes can; otherwise the rows are read by the scan after `else`, through the indexes the WHERE can use:

```sql
EXPLAIN SELECT SUM(value) FROM products WHERE value > 15;
```

Output:
```
plan
----
Aggregate SUM(value)
  StorageFiltered products (value > 15)
  else IndexScan products (value > 15)
    Index idx_value (value range)
```

### Aggregation

A GROUP BY of one column with no WHERE shows its groups after `by`. The table computes the groups itself when it can (`StorageGrouped`); groups read in the order of the column's index, each by its rows, stop once a LIMIT's groups pass the HAVING (`SortedStream`); otherwise every row is read and grouped:

```sql
EXPLAIN SELECT category, SUM(value) FROM products GROUP BY category;
EXPLAIN SELECT category, COUNT(*) FROM products GROUP BY category HAVING COUNT(*) > 1;
```

Output:
```
plan
----
Aggregate category, SUM(value) by category
  StorageGrouped products (category)
  else SeqScan products

plan
----
Having (COUNT(*) > 1)
  Aggregate category, COUNT(*) by category
    SortedStream products (category)
    else SeqScan products
```

Other grouped statements print the format of the path that runs them:

```sql
EXPLAIN SELECT category, SUM(value) FROM products WHERE value > 0 GROUP BY category;
```

Output:
```
plan
----
SELECT
  Columns: category, SUM(value)
  -> IndexScan products (value > 0)
       Index idx_value (value range)
  Group By: category
```

The scan line is the same one a planned statement prints: the access rule that chooses its keys or indexes is one rule for every path.

### Views, CTEs and Derived Tables

A statement over a view, a CTE, a derived table, a VALUES list or a table function reads the rows its source gives and judges each by the whole WHERE, shown after the source's name. The plan of the statement the rows come from is indented under it, in the format of the path that runs it:

```sql
EXPLAIN SELECT name FROM (SELECT id, name, value FROM products) p WHERE value > 15 ORDER BY name LIMIT 2;
```

Output:
```
plan
----
Limit 2
  TopN name ASC
    Project name
      Derived p (value > 15)
        SELECT
          Columns: id, name, value
          -> Seq Scan on products
```

A CTE read once, with no column list, runs as a derived table and shows as one. Any other CTE runs before the statement and shows as `Cte`, with its query's plan under it.

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

A DISTINCT is taken over the projected rows, before the ORDER BY. A DISTINCT of one column with no WHERE reads the distinct values the table keeps for that column in its index or its volumes, and scans when it keeps none, or when the transaction running it has written to the table:

```sql
EXPLAIN SELECT DISTINCT category FROM products;
```

Output:
```
plan
----
Distinct
  Project category
    DistinctValues products (category)
    else SeqScan products
```

A DISTINCT ON keeps the first row of each key once the rows are sorted:

```sql
EXPLAIN SELECT DISTINCT ON (category) category, name FROM products ORDER BY category, value DESC;
```

Output:
```
plan
----
DistinctOn category
  Sort category ASC, value DESC
    Project category, name
      SeqScan products
```

Window functions are computed over every row the read passes, and the SELECT list with them. With no WHERE, a read through an index may give the rows already grouped by the one PARTITION BY column every window shares, or ordered by the first window's ORDER BY column when that column holds no NULL. Under a LIMIT with no ORDER BY, DISTINCT or DISTINCT ON, the partitions are read one at a time until the LIMIT has its rows:

```sql
EXPLAIN SELECT name, RANK() OVER (PARTITION BY category ORDER BY value DESC) AS r FROM products LIMIT 5;
```

Output:
```
plan
----
Limit 5
  Window RANK() OVER (PARTITION BY category ORDER BY value DESC)
    PartitionsByIndex products (category) one at a time to the LIMIT
    else SeqScan products
```

The ORDER BY, DISTINCT and LIMIT are taken over the window step's rows:

```sql
EXPLAIN SELECT name, SUM(value) OVER (PARTITION BY category) AS total FROM products WHERE value > 10 ORDER BY total DESC;
```

Output:
```
plan
----
Sort total DESC
  Window SUM(value) OVER (PARTITION BY category)
    IndexScan products (value > 10)
      Index idx_value (value range)
```

A set operation over a table shows its first branch's plan and, beside it, each other branch as EXPLAIN shows that statement on its own. The ORDER BY and LIMIT above it are the whole statement's:

```sql
EXPLAIN SELECT name FROM products WHERE value > 15
UNION SELECT category FROM products ORDER BY name LIMIT 5;
```

Output:
```
plan
----
Limit 5
  TopN name ASC
    UNION
      Project name
        IndexScan products (value > 15)
          Index idx_value (value range)
      SELECT
        Columns: category
        -> Seq Scan on products
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

### Subqueries

For a statement that runs its subqueries as [prepared subqueries]({% link _docs/sql-features/subqueries.md %}#prepared-subqueries), EXPLAIN ANALYZE ends with a `Subqueries:` section: one line per subquery, numbered from 1, nested ones included, and so are the subqueries of its `LIMIT` and `OFFSET` and of its derived tables, set operation branches, CTEs and join sides that the run prepared. A view keeps its subqueries in its own plan, which EXPLAIN ANALYZE of the view's query lists. A subquery that runs as a statement of its own lists that statement's subqueries under its line, numbered after it, such as `1.1`. Each line says what the subquery tests, what it reads and how it answers:

```sql
EXPLAIN ANALYZE SELECT id FROM t WHERE EXISTS (SELECT 1 FROM w WHERE w.k = t.k);
```

```
SELECT (actual time=1.58ms, rows=2)
  Columns: id
  -> Seq Scan on t (actual rows=2)
       Filter: EXISTS (SELECT 1 FROM w WHERE (w.k = t.k))
  Subqueries:
    1: EXISTS over w, per call: a keyed set, or where the index on k makes probing cheaper: its pushed conditions
```

| Answer | Meaning |
|--------|---------|
| once per run | It reads no enclosing query: computed once per execution |
| once per row of N | It reads only values of subquery N's enclosing row: computed once for each such row |
| per call | Computed for each row that asks it |
| a statement | It runs as a statement of its own, such as one over a join |
| a keyed set | The rows of every key are read once per execution, and each call looks its key up |
| a keyed set, or where the index on k makes probing cheaper: ... | Each execution chooses on its first call: a probe per call when an index on `k` answers it and the enclosing rows are few or its `LIMIT` is small, else the keyed set |
| the count the index on k holds, else a scan | A `COUNT(*)` of one equality is counted from the index where it answers for the call |
| the row its key names | The equality on the primary key reads that one row |
| its pushed conditions, a scan | The rows the conditions the scan takes pass, or every row |

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
| UniqueGet | Reads the one row a unique index of one column names for its equality |
| IndexScan | Reads the rows the indexes under it name, each judged by the condition after the table name |
| Index | One index and what it reads: an equality (`=`), an `IN` list, a `LIKE prefix` or a `range` |
| CompositeIndex | A multi-column index: equalities on its leading columns, a range on the next one |
| PkKey, PkKeys | Rows a primary key equality or IN list names, inside an IndexScan |
| Intersect, Union | The rows every index under it names, or any of them names |
| MixedOr | A read that stops at a LIMIT, of a WHERE that is one OR: the rows its indexed operands name, then a scan asking the others (`Scan`) |
| IndexOrder, Keyset, TopK | Rows read in ORDER BY order under a LIMIT, with no sort of every row; for window functions, rows in their ORDER BY column's order |
| DistinctValues | The distinct values the table keeps for one column, for a DISTINCT of that column with no WHERE |
| PartitionsByIndex | Rows grouped by the window functions' PARTITION BY column through its index, or a partition at a time to the LIMIT |
| Derived, View, Cte | The rows of a derived table, a view or a CTE, each judged by the condition after its name; the plan of its query is under it |
| Values, Function | The rows of a VALUES list or a table function, each judged by the condition after its name |
| OneRow | The one row a SELECT without FROM reads |
| Aggregate | The one row of COUNT, SUM, AVG, MIN and MAX over a table, or its groups after `by`, from the line under it |
| Stats | Those values from what the table keeps: row count, distinct count, column sum and bounds |
| IndexBound | A MIN or MAX from the first or last key of the column's index |
| StorageFiltered | The calls the table computes over the rows the WHERE passes |
| StorageGrouped | The groups the table computes over its rows |
| SortedStream | The groups in the order of the column's index, each read by its rows |
| Having | The groups the HAVING keeps |
| Empty unless | The WHERE's terms that read no column; a run where they are not true reads no row |
| else | The read a run takes when the table cannot answer the line above it |
| Filter | The part of the WHERE judged after the scan |
| Project | The output columns |
| Window | The window functions over the rows the read passes, and the SELECT list over them |
| Distinct | The first row of each output value |
| DistinctOn | The first row of each DISTINCT ON key, over the sorted rows |
| UNION, UNION ALL, INTERSECT, EXCEPT | A set operation over its first branch's rows and each other branch's |
| TopN | The ORDER BY under a LIMIT, keeping only the rows the LIMIT keeps |
| Sort | The ORDER BY over every row |
| Limit | The LIMIT and OFFSET |

For other statements:

| Component | Description |
|-----------|-------------|
| Seq Scan | Full table scan (reads all rows) |
| IndexScan | Reads the rows its keys or indexes name, listed under it |
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
