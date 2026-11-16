---
layout: doc
title: Query Planning
category: Performance
order: 2
---

# Query Planning

Stoolap plans a statement once, when it is prepared, and the run reads that plan. The choices are made by rules over the statement's shape, the schema and the indexes. Where a choice depends on what only a run knows, such as a parameter's value or how many rows a side of a join holds, the plan names the possible reads and the run picks one; `EXPLAIN` prints them, the alternative after `else`.

Stoolap does not have a cost-based optimizer yet: no choice below is made by comparing estimated costs, and the join order is the one the statement is written in. [Limitations](#limitations) lists what this means for queries.

## Reading One Table

| Read | When |
|------|------|
| Primary key | `WHERE pk = value`, `pk IN (...)` or a range on the primary key |
| Index | an equality, a range or an `IN` list on an indexed column, or on the leading columns of a composite index |
| Several indexes | the `AND` of terms on different indexed columns: the read starts from the index holding the fewest rows for the run's values; an `OR` of indexed terms reads each and merges them |
| Ordered read | `ORDER BY` an indexed column with a `LIMIT`: rows are read in index order and the read stops at the limit |
| Sequential scan | no term an index or the primary key can answer |

Every term of the WHERE that the read does not answer is checked on each row it returns.

```sql
EXPLAIN SELECT id FROM users WHERE status = 'active' AND age = 30;
```

```
plan
----
Project id
  Filter
    IndexGet users by the fewest of status = 'active', age = 30
    else IndexScan users ((status = 'active') AND (age = 30))
      Intersect, fewest rows first
        Index idx_status (status =)
        Index idx_age (age =)
```

Each run counts, from the indexes themselves, how many rows each term's key holds for the values it binds, and starts from the fewest. The count needs no `ANALYZE` and follows the parameters of a prepared statement. An index holding more than four times the rows already found is not read: its term is checked on those rows. On sealed data each volume makes the same choice from its own side file. The order the terms are written in does not matter.

An ordered read names the sort it falls back to:

```sql
EXPLAIN SELECT id FROM users WHERE age > 30 ORDER BY age LIMIT 5;
```

```
plan
----
Limit 5
  Project id
    TopK users age ASC (age > 30)
    else TopN age ASC
      IndexScan users (age > 30)
        Index idx_age (age range)
```

On sealed (cold) data, zone maps and per-column bloom filters skip row groups that cannot hold a matching row, whatever the read.

## Joins

| Method | When |
|--------|------|
| Index nested loop | one side is a table whose primary key or an index covers its join column: the other side is read and each of its rows probes that key |
| Hash join | an equality condition: a hash table is built on one side and the other side probes it; on the general path a bloom filter of the build keys skips probe rows early |
| Nested loop | no equality condition, a side with no rows, or a side so small that comparing every pair costs less than building a hash table |

Between a hash join and a nested loop, the run decides from the number of rows each side actually returned. For an outer join the hash table is built on the side whose rows may go unmatched (the right side of a `LEFT` or `FULL JOIN`, the left side of a `RIGHT JOIN`), so the kept side streams through it; for an inner join, on the smaller side.

Tables are joined in the order the statement writes them. Writing the most selective table first can make a large difference:

```sql
-- The filter on customers keeps few rows: reading customers first means
-- only those rows probe orders
SELECT o.id, c.name
FROM customers c
JOIN orders o ON o.customer_id = c.id
WHERE c.region = 'EU';
```

## Statistics

`ANALYZE` collects, per table, the row count and, per column, the number of distinct values, NULL count, minimum and maximum, and a histogram:

```sql
ANALYZE users;

SELECT * FROM _sys_table_stats WHERE table_name = 'users';
SELECT * FROM _sys_column_stats WHERE table_name = 'users';
```

`ANALYZE` without a table name analyzes every table, each one on its own. It keeps the query timeout: past it, the statement stops with a query cancelled error, the table it was reading keeps its old statistics, and the tables it finished before keep their new ones.

The statistics are one source of the [row estimates](#row-estimates). `ANALYZE` also builds zone maps: see [Expression Simplification](#expression-simplification).

## Row Estimates

Where a choice depends on how many rows a side of a join keeps under its own conditions, the planner estimates them on each run, for the values that run binds, parameters included:

- A comparison of an indexed column with a value is counted by the index: an equality or an `IN` list exactly, a range while it holds at most 4,096 rows. This needs no `ANALYZE`. It is used while all of the table's rows are in memory; once some are sealed to volumes, the statistics answer instead.
- Otherwise the statistics `ANALYZE` collected estimate it: the histogram, the distinct values, the minimum and maximum.
- Without either, a default share: an equality keeps a tenth of the rows.

`AND` multiplies the shares of its terms; `OR` adds them, less their overlap.

Two choices read the estimates: a join under a small `LIMIT` that streams through a hash table builds that table on the side estimated smaller, and an `INNER JOIN` whose right side is estimated small beside its left one reads the right side first and probes the left table's index for each of its rows. `EXPLAIN` prints a join's estimate as `rows=`.

`EXPLAIN ANALYZE` leaves nothing behind: what one run found changes no later estimate.

## Expression Simplification

The planner simplifies expressions before it plans the read.

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

A WHERE that keeps no row, because one of its `AND` terms is a constant that is false or NULL (`1 = 0`, `NULL`, `$1 = 0` with a parameter that makes it false), reads no row of the table at all. Aggregates and window functions still answer their empty input, so `SELECT COUNT(*) FROM t WHERE 1 = 0` returns `0`.

`ANALYZE` also builds zone maps, the smallest and largest value of each column, which let a query without aggregates, window functions or `GROUP BY` skip the scan when no row can match its WHERE. They describe the committed rows `ANALYZE` read, so a query uses them only while no commit has landed on the table since, and never inside a transaction that has written to the table or reads an older snapshot. Run `ANALYZE` again after changes to use them again.

### Predicate Merging

```sql
-- Before: WHERE x > 5 AND x > 10
-- After:  WHERE x > 10

-- Before: WHERE x = 5 AND x = 5
-- After:  WHERE x = 5
```

## Viewing Query Plans

### EXPLAIN

Shows the plan the run reads:

```sql
EXPLAIN SELECT * FROM users WHERE id = 1;
```

```
plan
----
Project id, name, email, status, age
  PkGet users id = 1
  else IndexScan users (id = 1)
    PkKey
```

See [EXPLAIN]({% link _docs/sql-features/explain.md %}) for every plan line.

### EXPLAIN ANALYZE

Runs the statement and shows the plan with the time and rows of the run:

```sql
EXPLAIN ANALYZE SELECT * FROM users WHERE status = 'active';
```

```
plan
----
SELECT (actual time=2.5ms, rows=1500)
  Columns: *
  -> Seq Scan on users (actual rows=1500)
       Filter: (status = 'active')
```

## Limitations

These choices are made by rule, not by estimated cost:

- **Join order** is the written order. A selective table written last makes every earlier join read rows the filter later drops.
- **Index or scan**: an index is read even when the one holding the fewest rows holds most of the table's rows, where a scan would be cheaper on hot data.
- **Ordered reads under LIMIT** walk the index even when the WHERE matches rows only far along it, where filtering first and sorting the few matches would be cheaper.

## Best Practices

### Keep Statistics Updated

Run `ANALYZE` after significant data changes, so zone maps and row estimates match the data:

```sql
INSERT INTO orders SELECT * FROM staging_orders;
ANALYZE orders;
```

### Create Indexes for Selective Columns

An index helps when its term keeps few rows. A column with few distinct values gains little from an index of its own: when a query also tests a more selective indexed column, the read starts from that one:

```sql
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date);
```

### Write the Most Selective Table First

Joins run in the written order, so put the table with the most selective filter first.

### Use EXPLAIN to Understand Plans

```sql
EXPLAIN SELECT * FROM orders WHERE customer_id = 100;
```
