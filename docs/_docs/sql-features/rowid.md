---
layout: doc
title: Row Identity (_rowid_)
category: SQL Features
order: 22
---

# Row Identity (_rowid_)

Every row has an identity, a 64-bit integer that names it for as long as it lives. The pseudo-column `_rowid_` reads it.

- In a table with an `INTEGER PRIMARY KEY`, the identity is the key's value: `_rowid_` reads the same value as the key column.
- In a table without a primary key, the identity is the id the engine assigned when the row was inserted, counting up from 1.

```sql
CREATE TABLE events (kind TEXT, payload TEXT);
INSERT INTO events VALUES ('open', 'a'), ('close', 'b')
RETURNING _rowid_;
-- Returns: 1, 2

SELECT _rowid_, kind FROM events WHERE _rowid_ = 2;
-- Returns: 2, 'close'
```

## A reserved name

`_rowid_` names the row identity and nothing else. No column and no alias may take it: `CREATE TABLE`, `ALTER TABLE ... ADD COLUMN` or `RENAME COLUMN`, `CREATE TABLE ... AS SELECT`, a select-list or `RETURNING` alias, and the column list of a CTE, a `VALUES` list or a table function all refuse it. `rowid` is an ordinary name: a table may have a column named `rowid`, and a query may alias to it.

A database written before the name was reserved, holding a table with a column named `_rowid_`, refuses to open; the error names the table.

## Where _rowid_ can be used

`_rowid_` reads like any column: in the select list, `WHERE`, `ORDER BY`, `GROUP BY`, `HAVING`, `CASE`, window functions, subqueries (correlated ones included), set operations, joins (as `t._rowid_` of a table with a primary key), and the `WHERE` and `RETURNING` of `UPDATE`, `DELETE` and `INSERT`.

A `WHERE _rowid_ = ?` or `WHERE _rowid_ IN (...)` reads the rows it names by their ids, on rows in memory and on rows already written to disk alike, as a primary key lookup does.

## Name rules

- A quoted `"_rowid_"` is not the row identity.
- `SELECT *` never shows `_rowid_`.
- `t._rowid_` reads the identity of table `t`. A bare `_rowid_` over two tables is ambiguous.
- A derived table, view or CTE has a `_rowid_` only when its query selects one as an output column. A bare `_rowid_` counts it with the tables beside it, so a table joined with such a derived table makes the name ambiguous.
- In `INSERT ... ON CONFLICT DO UPDATE`, the `SET` values, the `WHERE` and the `RETURNING` read the `_rowid_` of the row met, in a table with or without a primary key; `excluded._rowid_` is an error.

## What _rowid_ does not do

- `_rowid_` is never written: `UPDATE t SET _rowid_ = ...`, `INSERT INTO t (_rowid_, ...)` and `ON CONFLICT DO UPDATE SET _rowid_ = ...` are errors. In a table with a primary key, the key is not updatable either; delete and insert instead.
- In a join, a table without a primary key has no `_rowid_`: the name is an error there. A table with a primary key reads its key as usual.
- A table read `AS OF` a past point has no `_rowid_`.
- A primary key named `TRUE` or `FALSE` is read through `_rowid_` everywhere but inside an aggregate; write the key's quoted name there.
- Under `SELECT DISTINCT`, an `ORDER BY` key is matched to the select list as written: `_rowid_` and the primary key column it reads count as different, so `SELECT DISTINCT id FROM t ORDER BY _rowid_` is an error. Sort by what the list selects.

## Stability

A row keeps its identity through updates, through being written to disk and compacted, and across restarts. The id of a deleted row is not given to another row while the database stays open. After a restart, a table without a primary key continues from the largest id it still holds, so ids of the newest rows deleted before the restart can be given again. `TRUNCATE` starts the ids again from 1.
