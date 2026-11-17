---
layout: doc
title: Query Planning
category: Performance
order: 2
---

# Query Planning

Stoolap plans a statement once, when it is prepared, and the run reads that plan. The choices are made by rules over the statement's shape, the schema and the indexes. Where a choice depends on what only a run knows, such as a parameter's value or how many rows a side of a join holds, the plan names the possible reads and the run picks one; `EXPLAIN` prints them, the alternative after `else`.

One choice is made by comparing estimated costs: the order of a run of three or more `INNER JOIN`s over tables, described in [Join Order](#join-order). The others are made by rules; [Limitations](#limitations) lists what this means for queries.

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

### Join Order

Three or more tables joined by `INNER JOIN ... ON` are joined from the table estimated to keep the fewest rows under its own `WHERE` terms. Each next table is one an equality of the `ON` conditions reaches, the one whose step is estimated cheapest: a step that probes the next table's primary key or an index reads a key per row so far, any other step reads the table whole. The order is taken only when it is estimated to cost at most half of the written one; otherwise the tables are joined as written.

```sql
-- Written with the selective table last, joined s, u, t: the two rows of s
-- probe u's index on sid, and those rows probe t's primary key
SELECT COUNT(*)
FROM t
JOIN u ON t.id = u.tid
JOIN s ON u.sid = s.id
WHERE s.x = 1;
```

The order is decided when the statement first runs and kept until `ANALYZE` runs, an index or a table changes, or a table's rows grow or shrink about fourfold. The estimates read literals; a parameter counts as an unknown value. `EXPLAIN` prints the order taken beside the written one:

```
SELECT
  Join order: s, u, t (written t, u, s)
```

The answer is the written statement's, columns and names included; without `ORDER BY` the rows may come in another order. The written order is kept for `SELECT *` and `t.*`, whose columns follow the tables' order, and for a run that holds an outer join, a `CROSS` or comma join, `USING` or `NATURAL`, a subquery, view, CTE or `AS OF` side, a window function, or a subquery anywhere in the statement. Tables joined by an outer join above an `INNER` run keep their places, and the run below may still be reordered.

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

- A comparison of an indexed column with a value is counted by the index: an equality or an `IN` list exactly, a range while it holds at most 4,096 rows. This needs no `ANALYZE`. Once some rows are sealed to volumes, an equality or an `IN` list still counts them through each volume's side file for the index, exactly where its key pages are cached or can be read in, else by the side file's directory. The side files also hold rows since deleted or updated: when those could make up half of a count, or volumes overlap, or a seal moves rows while it counts, or an estimate would read more than 256 side file keys, the statistics estimate it instead. A range over sealed rows is estimated from the statistics.
- Otherwise the statistics `ANALYZE` collected estimate it: the histogram, the distinct values, the minimum and maximum.
- Without either, a default share: an equality keeps a tenth of the rows.

`AND` multiplies the shares of its terms; `OR` adds them, less their overlap.

An equality join of two sides is estimated to give the product of their rows over the larger of the two keys' distinct counts, and no more than one side's rows where the other side's key is a primary key or a unique index. A distinct count comes from the key's index while the table's rows are all in memory, else from `ANALYZE`; without either, the key is taken to hold a value per row.

Three choices read the estimates: the [join order](#join-order), a join under a small `LIMIT` that streams through a hash table, which builds that table on the side estimated smaller, and an `INNER JOIN` whose right side is estimated small beside its left one, which reads the right side first and probes the left table's index for each of its rows. `EXPLAIN` prints a join's estimate as `rows=`.

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

Runs the statement and shows the plan EXPLAIN shows, with the time and rows of the run on its top line:

```sql
EXPLAIN ANALYZE SELECT id FROM users WHERE status = 'active';
```

```
plan
----
Project id (actual time=2.5ms, rows=1500)
  SeqScan users (status = 'active')
```

## Limitations

These choices are made by rule, not by estimated cost:

- **Join order** is the written order outside a run of `INNER JOIN`s over tables (see [Join Order](#join-order)), and for every table of a `SELECT *`. There, a selective table written last makes every earlier join read rows the filter later drops.
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

Where the written order is kept (see [Limitations](#limitations)), put the table with the most selective filter first.

### Use EXPLAIN to Understand Plans

```sql
EXPLAIN SELECT * FROM orders WHERE customer_id = 100;
```
