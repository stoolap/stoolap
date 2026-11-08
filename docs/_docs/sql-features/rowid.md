---
layout: doc
title: Row Identity (rowid)
category: SQL Features
order: 22
---

# Row Identity (rowid)

Every row has an identity, a 64-bit integer that names it for as long as it lives. The pseudo-columns `rowid` and `_rowid_` read it.

- In a table with an `INTEGER PRIMARY KEY`, the identity is the key's value: `rowid` reads the same value as the key column.
- In a table without a primary key, the identity is the id the engine assigned when the row was inserted, counting up from 1.

```sql
CREATE TABLE events (kind TEXT, payload TEXT);
INSERT INTO events VALUES ('open', 'a'), ('close', 'b')
RETURNING rowid;
-- Returns: 1, 2

SELECT rowid, kind FROM events WHERE rowid = 2;
-- Returns: 2, 'close'
```

## Where rowid can be used

`rowid` reads like any column: in the select list, `WHERE`, `ORDER BY`, `GROUP BY`, `HAVING`, `CASE`, window functions, subqueries (correlated ones included), set operations, and the `WHERE` and `RETURNING` of `UPDATE` and `DELETE`, and an `INSERT ... RETURNING`.

A `WHERE rowid = ?` or `WHERE rowid IN (...)` reads the rows it names by their ids, on rows in memory and on rows already written to disk alike, as a primary key lookup does.

## Name rules

- A real column named `rowid` or `_rowid_` wins: the name then reads that column. The other spelling still reads the identity.
- A quoted `"rowid"` names only a real column.
- `SELECT *` never shows `rowid`.
- A select-list alias named `rowid` wins in `ORDER BY`, as any alias does.
- `t.rowid` reads the identity of table `t`. A bare `rowid` over two tables is ambiguous.
- A derived table, view or CTE has a `rowid` only when its query selects one as an output column.

## What rowid does not do

- `rowid` is never written: `UPDATE t SET rowid = ...` and `INSERT INTO t (rowid, ...)` are errors. In a table with a primary key, the key is not updatable either; delete and insert instead.
- In a join, a table without a primary key has no `rowid`: the name is an error there. A table with a primary key reads its key as usual.
- A table read `AS OF` a past point has no `rowid`.
- The `RETURNING` of `INSERT ... ON CONFLICT DO UPDATE` does not read the `rowid` of a table without a primary key.

## Stability

A row keeps its identity through updates, through being written to disk and compacted, and across restarts. The id of a deleted row is not given to another row while the database stays open. After a restart, a table without a primary key continues from the largest id it still holds, so ids of the newest rows deleted before the restart can be given again. `TRUNCATE` starts the ids again from 1.
