---
layout: doc
title: Migration Guide v0.4.2 to v0.5.0
category: Development
order: 1
---

# Migration Guide: v0.4.2 to v0.5.0

v0.5.0 opens a v0.4.2 database without any manual step. The storage format moves forward on its own as the database runs, and there is no way back to v0.4.2 afterwards.

Most of the work an upgrade asks for is in SQL. v0.4.2 answered many statements it should have refused, often with a NULL, a missing row or the wrong column, and v0.5.0 refuses them with an error that names the problem. A smaller set of queries returns a different, correct answer. This guide lists what to look for, grouped so you can search your application's queries for each pattern.

## Before You Upgrade

1. **Back up the database directory** (or run `PRAGMA SNAPSHOT`). Once v0.5.0 has written a manifest, v0.4.2 refuses to open the database.
2. **Run your test suite against v0.5.0** before switching production. Most changes surface as errors at prepare or run time, so a test run finds them.
3. **Check for a column named `_rowid_`.** It is now a reserved name (see [Storage and Files](#storage-and-files)).
4. **Check for TIMESTAMP values before 1677-09-21 or after 2262-04-11** and for TIME values written by v0.4.2 (see [Dates, Times and Intervals](#dates-times-and-intervals)).
5. **Rust embedders that install their own global allocator** need `default-features = false` (see [Build and Dependencies](#build-and-dependencies)).
6. **Builds with the `semantic` feature need Rust 1.94** or newer; the rest of the crate still builds on 1.88.

## Storage and Files

### Format

| File | v0.4.2 | v0.5.0 |
|------|--------|--------|
| Volume files (`vol_*.vol`) | V4 | V5, with a CRC-32C on every section and block |
| Manifest (`manifest.bin`) | version 6 | version 8, recording each volume file's identity and ending in a CRC-32C |
| Secondary index side files (`vol_*.sidx`) | version 4 | version 6, which also hold FLOAT keys and TEXT B-tree and Hash keys |
| WAL | version 2 | version 2 (unchanged) |

- v0.5.0 reads V4 volumes and version 6 and 7 manifests. Compaction rewrites V4 volumes as V5, one volume per table each cycle, so an upgraded database converts gradually with no downtime.
- **No downgrade.** v0.4.2 refuses a database once v0.5.0 has written a manifest, with `unsupported manifest version 8 (expected 6)`. Restore the backup you took before upgrading if you need to go back.
- `PRAGMA VERIFY` (new) checks every volume file's checksums, structure and identity, and returns one row per volume with `ok`, `damaged`, `unreadable` or `failed`.

### Open and Recovery

- **A table whose listed volume is missing or unreadable is unavailable.** v0.4.2 answered from the remaining volumes, silently missing rows. Now every statement on that table fails with an error naming the segments, other tables keep working, `DROP TABLE` still works, and putting the file back then reopening restores the table. While a table is unavailable, `PRAGMA SNAPSHOT` fails, and seal, compaction and WAL truncation wait.
- **Recovery stops at a committed WAL record it cannot apply**, naming the operation, table and LSN. v0.4.2 skipped such records and opened with the data missing.
- **The open runs a checkpoint after replaying the WAL**, and fails if that checkpoint fails. `PRAGMA CHECKPOINT` returns an error when a manifest write fails instead of reporting success.
- **A damaged block in a V5 volume** fails only the statements that read it, and the error names the file, column and row group.
- **A `CREATE TABLE` inside a transaction is written to the WAL**, so the table and its committed rows survive a crash.
- **AUTO_INCREMENT ids of deleted rows are not handed out again after a reopen.** Insert 1, 2, 3, delete 3, reopen: the next insert gets 4.
- Several crash windows around checkpoints that lost committed rows or brought dropped tables back are closed.

### Reserved `_rowid_`

`_rowid_` is a pseudo-column giving a row's identity: its INTEGER primary key, or the id the engine assigned in a table without one. See [Row Identity]({{ '/docs/sql-features/rowid/' | relative_url }}).

A database with a table that has a column named `_rowid_` still opens, with a warning. That table refuses every statement except `ALTER TABLE`, `DROP TABLE`, `DROP INDEX`, `SHOW CREATE TABLE`, `SHOW INDEXES` and `DESCRIBE` until the column is renamed:

```sql
ALTER TABLE t RENAME COLUMN _rowid_ TO row_ref;
```

`rowid` without underscores stays an ordinary name.

### Two Index Names That Differ Only in Case

Index names are now matched without regard to case. A v0.4.2 database holding both `idxk` and `IdxK` on one table still opens and keeps both, but `DROP INDEX` then needs the exact spelling; any other spelling is refused as ambiguous.

## SQL That Now Returns an Error

Each item below was accepted by v0.4.2, usually with a silently wrong answer. The fix is almost always to write what the query meant.

### Names and Scope

| Statement | v0.4.2 | v0.5.0 |
|-----------|--------|--------|
| `SELECT id FROM t JOIN u ON t.k = u.k` (both have `id`) | read `t.id` | `column reference 'id' is ambiguous`; write `t.id` |
| `SELECT MAX(x) FROM (SELECT 1 AS x, 2 AS x) s` | read one of them | ambiguous |
| `SELECT zz.x FROM u` (no FROM entry `zz`) | read `x` | `unknown table or alias 'zz'` |
| `SELECT t.a FROM t AS x` | read `a` | `table or alias 't' is not visible here`: an aliased table is named only by its alias |
| `SELECT SUM(missing) FROM t` | NULL | `column 'missing' not found` |
| `SELECT b, COUNT(*) FROM t GROUP BY a` | `b` was NULL | `Column 'b' not found`: read `b` through GROUP BY or an aggregate |
| `SELECT COUNT(*) FROM t GROUP BY nosuchcol` | one group over every row | `Column 'nosuchcol' not found` |
| `GROUP BY COUNT(*)`, or an aggregate by position or alias | `Function not found: COUNT` | `an aggregate function cannot be a GROUP BY key` |
| `SELECT a FROM t ORDER BY 2` | unsorted | `ORDER BY position 2 is not in the select list` |
| `SELECT DISTINCT customer FROM orders ORDER BY order_date` | arbitrary order | error: every ORDER BY key must follow from the selected items |
| `SELECT DISTINCT ON (nosuchcol) a FROM t` | key dropped, one row | `column 'nosuchcol' not found` |

Names are compared by one rule everywhere, Unicode lowercasing. A table created as `"äö"` is found as `"ÄÖ"`, and savepoint names are no longer case-sensitive. See [Names]({{ '/docs/sql-features/names/' | relative_url }}).

### Statement Shape

- **Text after a complete statement** that is not `;` is a parse error. v0.4.2 ran the statement and ignored the rest: `UPDATE t SET v = 1 WHERE id = 1 2` wrote the row.
- **A statement that writes one column more than once** is refused: `UPDATE t SET v = 5, v = 6`, a repeated column in an INSERT or COPY column list, or in an upsert's assignments.
- **Statements nested too deeply** are refused when parsed: more than 32 nested SELECTs (subqueries, derived tables, CTE bodies) or 64 nested expressions in a release build. Deeper statements used to crash the process.
- **DISTINCT, an inner ORDER BY and FILTER** are allowed only on aggregate calls: `ABS(DISTINCT a)` is an error.
- **A subquery must return as many values as its place takes**, with `*` counting every column: `a IN (SELECT x, y FROM u)` is an error. A row compared with a row of another width, `(1, 2) = (1, 2, 3)`, is an error (v0.4.2 answered true), and `SELECT (1, 2)` is an error (v0.4.2 answered 1).
- **A grouped query's subquery** that reads an outer column not in GROUP BY fails as an ungrouped column.
- **AS OF** works only on a table: on a view or a CTE it is an error (v0.4.2 ignored it), and `CREATE VIEW` refuses a query that reads a table `AS OF`.
- **Set operations**: in the ORDER BY of a UNION, INTERSECT or EXCEPT, a name two result columns carry is ambiguous and a qualified name is an error. Use a position.

### Parameters

- A parameter with no value is an error on every path, before any row is read or written. v0.4.2 mostly read it as NULL.
- Mixing positional (`$1`, `?`) and named (`:name`) parameters in one statement is a parse error, and so is `$0`.
- Two anonymous `?` are two different expressions: `SELECT g + ?, COUNT(*) FROM t GROUP BY g + ?` fails as an ungrouped column. Repeat a value with `$1` or `:name`.
- `CREATE VIEW` refuses parameters.

### Evaluation Errors

A runtime error in an expression now fails the statement wherever the expression sits: the SELECT list, WHERE, ORDER BY, GROUP BY, join conditions, window arguments, UPDATE SET, DEFAULT and CHECK. v0.4.2 returned NULL, skipped the row or read the condition as false, so an UPDATE or DELETE could change every row but the failing one.

- **INTEGER overflow** is an `Integer overflow` error for every operator. Addition already was; newly, `ABS` and unary minus of the smallest INTEGER (v0.4.2 gave NULL), the smallest INTEGER divided by -1 (v0.4.2 crashed the process), and `1 << 64` (v0.4.2 gave 1; a shift outside 0 to 63 is an error). Division by zero still gives NULL.
- **SUM over INTEGER values** that leaves the INTEGER range is `Integer overflow in SUM` (v0.4.2 turned it into FLOAT or wrapped).
- **FLOAT to INTEGER** outside the INTEGER range, or NaN or infinity, is an error: `CAST(1e30 AS INTEGER)` (v0.4.2 gave the largest INTEGER).
- **LIMIT and OFFSET** must be an INTEGER of 0 or more, or a FLOAT without a fraction, on every path: `LIMIT 2.5` is an error (v0.4.2 returned 2 rows).
- **Window arguments**: ROWS and LAG/LEAD offsets must be INTEGER >= 0, NTILE and NTH_VALUE counts INTEGER >= 1, RANGE offsets a number >= 0.
- **REPEAT, LPAD, RPAD and REPLACE** refuse a result over 10 MB.
- **`generate_series`** past 10,000,000 rows is an error instead of a cut-short answer.
- **A vector distance** with mismatched dimensions, or a non-vector argument, is an error in `ORDER BY VEC_DISTANCE_*(...) LIMIT k` and with `<=>` (v0.4.2 gave NULL).

### DDL, Constraints and Foreign Keys

- **Table-level CHECK constraints are enforced** (v0.4.2 parsed and dropped them), a column CHECK may name other columns, and a CHECK now runs on a NULL value: only a NULL result passes, so `CHECK (v IS NOT NULL)` refuses a NULL. A CHECK that fails to evaluate fails the statement.
- **A DEFAULT or CHECK holding a subquery, aggregate or window function** is refused.
- **`ALTER TABLE ... ADD COLUMN`** refuses CHECK, UNIQUE, PRIMARY KEY, AUTO_INCREMENT and REFERENCES (v0.4.2 accepted them without enforcing them), and refuses `NOT NULL` without a non-NULL DEFAULT while the table holds rows.
- **`ALTER TABLE ... DROP COLUMN`** refuses the primary key column, a column a foreign key references, and a column another CHECK reads.
- **`ALTER TABLE ... MODIFY COLUMN`** refuses the primary key column, foreign key columns, and changes whose cast values would turn NOT NULL values into NULL or break UNIQUE. It now rewrites hot rows, indexes and sealed volumes.
- **`DROP TABLE`** is refused while another table's foreign key references the table. `DROP TABLE ... CASCADE` (new) drops the table and the referencing keys.
- **An UPDATE that sets NULL in a NOT NULL column** is refused for every row (v0.4.2 stored it for rows still in memory).
- **Foreign keys are checked by `ON CONFLICT DO UPDATE` and `ON DUPLICATE KEY UPDATE`** (v0.4.2 skipped them).
- **A foreign key check holds the parent key until its transaction ends**: a child INSERT racing an uncommitted parent DELETE fails with a write conflict instead of both committing.
- **Names**: any name, CHECK or DEFAULT text over 65,535 bytes, a table name over 255 bytes, and a table name containing `/` or `..` are refused.
- **Views**: `CREATE VIEW` runs its query once and refuses one that cannot run or that gives two columns the same name. Reading a view whose query now returns another number of columns asks for the view to be created again. `ALTER TABLE ... RENAME TO` a name a view has is refused.

### Transactions and Concurrency

- **Unique key violations inside an explicit transaction** are reported when the statement ends, not at COMMIT. The failing statement is undone and the transaction stays open.
- **A statement that overlaps a concurrent ADD, DROP or RENAME COLUMN** fails with `SchemaChanged` (v0.4.2 could read wrong positions).
- **A READ COMMITTED write** that a TRUNCATE or DROP overtakes fails with `ViewOvertaken`; a SELECT retries up to 3 times.
- **A query timeout stops every kind of work**: writes, joins, sorts, windows, aggregation, recursive CTEs, storage scans and ANALYZE, with `QueryCancelled`. A stopped write leaves no row changed. v0.4.2 timed only part of a SELECT, so an UPDATE with a 20 ms timeout committed all of its rows.

## Queries Whose Answers Change

### Comparison and Ordering

v0.5.0 compares and orders values by one rule on every plan, in memory or on disk. See [Operators and Expressions]({{ '/docs/sql-features/operators-expressions/' | relative_url }}) and [NULL Handling]({{ '/docs/sql-features/null-handling/' | relative_url }}).

- **Numbers and TEXT**: TEXT that reads as a number is compared as that number; any other TEXT makes the comparison UNKNOWN. `1 = 'abc'` is NULL. A number and TEXT are never a join key.
- **BOOLEAN and TIMESTAMP are not numbers**: `i = TRUE` no longer matches `i = 1`, and `<`, `>` and BETWEEN between them are UNKNOWN.
- **JSON and VECTOR** are only equal or unequal by value. A JSON column `= 'text'` matches nothing; write `CAST(... AS JSON)`.
- **FLOAT uses one total order**: NaN equals NaN and is greater than every number, and `0.0` equals `-0.0`. `MAX` is NaN when a NaN is present, and GROUP BY puts every NaN in one group.
- **Sorting across types** (ORDER BY, MIN, MAX, GREATEST, LEAST, windows) uses one order: BOOLEAN, numbers, TEXT, TIMESTAMP. Mixed columns used to sort by arrival order.
- **Row comparisons compare every value**: `(1, 2) = (1, 3)` is false (it was true).
- **ORDER BY on an indexed column with LIMIT** puts NULLs last ascending and first descending, as the full sort does.
- **LIKE, ILIKE, GLOB and REGEXP** read both sides as their `CAST AS TEXT` form; `n LIKE '1%'` on an INTEGER column matches the same rows in memory and on disk. `UPPER(s) LIKE 'john%'` is case-sensitive again, and ILIKE folds non-ASCII case.

### Aggregates and Grouping

- **SUM of INTEGER values is an exact INTEGER**; it is FLOAT once any value is FLOAT. Some GROUP BY paths returned FLOAT for integer sums.
- **Repeated aggregates keep every column**: `SELECT SUM(a), SUM(a) FROM t` returns two columns.
- **Aggregates are matched by their whole expression**: `STRING_AGG(v, ',')` and `STRING_AGG(v, ';')` no longer share one result.
- **GROUP BY treats 1 and 1.0 as one group.**
- **HAVING alone** (no GROUP BY, no aggregate in the SELECT list) aggregates the statement into one group.
- **Window functions over ROLLUP, CUBE or GROUPING SETS** see every group, subtotal rows included.
- **STRING_AGG**: the separator may be any expression that reads no column. A NULL separator adds nothing (v0.4.2 used `,`), and `STRING_AGG(s, '-' || '')` joins with `-` (v0.4.2 answered NULL).
- **A grouped column is read from its GROUP BY item** even when an aggregate's alias has the same name.
- **COUNT(DISTINCT)** and DISTINCT keep NULLs held by sealed rows and count re-inserted rows correctly.

### Result Column Names

An unaliased SELECT item is named as written on every path: `ROW_NUMBER() OVER (...)`, `SUM(f) OVER (PARTITION BY g)`, `COUNT(DISTINCT g)`. v0.4.2 produced names such as `expr6`, `column_2` or `SUM()`. Code that reads result columns by position is unaffected; code that reads them by these generated names needs an alias.

A derived table, CTE or view over `SELECT *` from a join names its columns by their plain names (`k`, not `u.k`).

### Joins

Many join answers are fixed. If your application relied on any of these, check its results:

- USING and NATURAL joins: each name joins on the first column of that name on each side; RIGHT and FULL joins read the merged column from the side that has the row (`COALESCE(left, right)` for FULL).
- An index join applies its whole ON clause; a CTE side, an AS OF side and a VALUES side keep their conditions.
- A large merge of nearly sorted inputs no longer loses matches (merge join is gone).
- Joins with LIMIT, DISTINCT or DISTINCT ON return the full number of rows.
- Without ORDER BY, the row order of a join may differ, because INNER joins of three or more tables may now be reordered. Add ORDER BY where order matters.

### Dates, Times and Intervals

- **TIMESTAMP range**: a table stores TIMESTAMP values from `1677-09-21` to `2262-04-11`. Storing a value outside it is an error. A row written by v0.4.2 with such a value and still in memory makes the checkpoint report an error naming the table and row; v0.4.2 silently corrupted it when sealing. Computing outside the range (years 0000 to 9999) still works.
- **TIME values**: a time of day alone (`'12:00:00'`) now reads as `1970-01-01 12:00:00`. TIME values stored by v0.4.2 carry the date they were written on and no longer equal the same time written now. See [Date and Time]({{ '/docs/data-types/date-and-time/' | relative_url }}) for moving such a column onto 1970-01-01.
- **Timestamp text**: `'2024/03/04'` and `'03/04/2024'` read as dates (the US form first when both read); a two-digit year such as `'24-03-04'` no longer reads as a timestamp; a `TIMESTAMP 'not a date'` literal is a parse error.
- **Intervals**: `n * INTERVAL '1 second'` and `-INTERVAL '1 hour'` evaluate (v0.4.2 gave NULL). Compound literals such as `INTERVAL '1 month 1 day'` are an error (v0.4.2 kept the first unit). Month and year steps move on the calendar with the day clamped.
- **Text conversion**: every conversion of a value to text follows `CAST AS TEXT`, so timestamps are written in RFC 3339.

### AS OF

- History lasts from the last open: a point from before it is an error naming the oldest readable point, where v0.4.2 returned missing rows. The same holds for points cut off by VACUUM, TRUNCATE, column changes, a seal, compaction, and the per-row version and 24-hour limits. See [Temporal Queries]({{ '/docs/sql-features/temporal-queries/' | relative_url }}).
- Tables with sealed volumes answer AS OF for any point after their newest seal; v0.4.2 refused every historical read on such tables.
- `AS OF TIMESTAMP t` includes only transactions whose commit completed by `t`; `AS OF TRANSACTION n` means right after `n` committed, in commit order.
- AS OF inside an explicit transaction is honoured (v0.4.2 read current rows).

### Writes

- **A DELETE re-checks its WHERE** against the version it deletes, so a row a concurrent commit stopped matching is not deleted.
- **Subqueries in RETURNING and ON CONFLICT** read tables as the statement found them, before its own cascades.
- **`ALTER TABLE ... RENAME` and `RENAME COLUMN`** re-point the foreign keys that reference the table or column.

## Changed Defaults

| Setting | v0.4.2 | v0.5.0 |
|---------|--------|--------|
| `CREATE INDEX` on a TEXT column without `USING` | Hash | B-tree (JSON columns still get Hash; existing indexes keep their type; write `USING HASH` for the old behaviour) |
| `SET isolation_level` and `set_default_isolation_level` | engine-wide | per session; a session that sets none uses the engine's level |
| Opening a DSN that is already open | the opener's handle and its transaction | a new session on the same engine, with its own transaction state |
| Engine close | when the original handle drops | when the last handle drops, clones included; a `Transaction` keeps its database open until commit or rollback |
| Threads sharing one `Database` handle | serialised | run statements concurrently outside an explicit transaction |
| `sync_mode=none` | each schema change synced | schema changes buffered; checkpoint catalog copies still synced |
| Query timeout | parts of SELECT | every statement kind; `PRAGMA query_timeout` and the `query_timeout` DSN key set a per-connection default (0, no timeout) |
| EXPLAIN ANALYZE | recorded feedback that changed later plans | no feedback; prints EXPLAIN's plan lines with the run's time and rows on the top line |

## New Features

- `_rowid_` pseudo-column. See [Row Identity]({{ '/docs/sql-features/rowid/' | relative_url }}).
- `UPDATE` may change an INTEGER primary key; `ON UPDATE CASCADE` into a child's primary key works.
- Self-referencing foreign keys, `DROP TABLE ... CASCADE`, `NATURAL FULL JOIN`, `NATURAL INNER JOIN`.
- `ALTER TABLE t AUTO_INCREMENT = n`; `ALTER TABLE t DROP CLUSTER BY`.
- `PRAGMA query_timeout`, `PRAGMA VERIFY`.
- Subqueries in INSERT VALUES, RETURNING, ON CONFLICT, JOIN ON, LIMIT and OFFSET, window arguments, VALUES rows and table-function arguments; a subquery may start with its own WITH.
- Row value comparisons: `(a, b) IN (SELECT x, y ...)`, `(a, b) = (SELECT x, y ...)`.
- `RETURNING` items accept aliases.
- Interval arithmetic: `n * INTERVAL '...'`, negated intervals.
- `execute_counts` returns `ExecCounts { matched, changed }`, so an UPDATE reports the rows it actually changed.
- serde deserialization of result rows: `ResultRow::deserialize`, `Rows::deserialize`, `Database::query_as_serde`.
- EXPLAIN shows the planner's plan tree; see [EXPLAIN]({{ '/docs/sql-features/explain/' | relative_url }}). Its output format differs from v0.4.2.

## Rust API Changes

The engine internals are public modules, so this list names what an embedder is most likely to touch. A crate that only uses `Database`, `Statement`, `Transaction`, `Rows` and `Value` needs few or no changes.

### Removed or Made Private

- Modules `stoolap::optimizer` and `stoolap::planner` are crate-private.
- Crate-root exports removed: `BufferPool`, `PoolStats`, `SemVer`, `RangeExpr`, `EmptyResult`, `EmptyScanner`, `VecScanner`, `RangeScanner`, `SingleRowScanner`, `DEFAULT_CHECKPOINT_INTERVAL`, `DEFAULT_KEEP_SNAPSHOTS`.
- `Transaction::execute_prepared_named` and `query_prepared_named`: use `Statement::execute_named_in` and `query_named_in`.
- `Executor::with_function_registry` and `function_registry`: every path uses one global registry.
- `TimeoutGuard`, `CancellationHandle` and the `ExecutionContext` cancellation methods: use `ExecutionContext::deadline()` and `set_deadline()`.
- AST variants `Expression::Distinct`, `InHashSet` and `CteReference`; `TokenType::Date`, `Time` and `Timestamp`; parser `KEYWORDS` and `is_keyword(&str)` (use the `Keyword` enum).

### Changed

- `Error`: `LockAcquisitionFailed` is removed. New variants: `TableCheckViolation`, `TableHasOpenTransactionRows`, `ViewOvertaken`, `WalOutcomeUnknown`. `Error::UniqueConstraint` carries `ConflictValues`. A `match` that lists every variant needs updating.
- `FunctionRegistry::register_scalar`, `register_aggregate`, `register_window` and `register_tvf` return `Result<()>` and refuse a built-in function's name, so built-ins can no longer be overridden.
- `FunctionInfo` carries `Volatility` (`Fixed`, `PerStatement`, `PerCall`) instead of a deterministic flag.
- Custom aggregates: `AggregateFunction::result` returns `Result<Value>`, and `accumulate` loses its `distinct` parameter.
- `set_default_isolation_level` takes `&self`.
- The storage `Table` and `Index` traits changed: several methods take a `Deadline`; `delete` takes the access the caller decides; `select_as_of` is replaced by `rows_at`; `has_visible_row` and `for_each_unique_non_pk_index` must be implemented; `Index::find_with_operator` returns an error for an operator it cannot answer.
- Library locks no longer poison after a panic.

### Added

`ExecCounts`, `Volatility`, `Statement::execute_in`, `query_in`, `execute_named`, `query_named`, `execute_named_in`, `query_named_in`, `Database::execute_counts`, `ResultRow::deserialize`, `Rows::deserialize`, `Database::query_as_serde`, `Database::query_as_serde_named`, `Value::vector_f32_iter`.

## C API Changes

New functions in `include/stoolap.h`:

- `stoolap_exec_params_counts`, `stoolap_stmt_exec_counts`, `stoolap_tx_exec_params_counts`
- `stoolap_stmt_exec_named`, `stoolap_stmt_query_named`
- `stoolap_stmt_query_all`, `stoolap_query_params_all`, `stoolap_query_named_all`, `stoolap_stmt_query_named_all`, `stoolap_rows_fetch_chunk`

Changed behaviour:

- A runtime error part way through a fetch-all is returned as an error, not a partial result.
- A non-UTF-8 text parameter or invalid JSON parameter fails the call (v0.4.2 bound NULL).
- `stoolap_stmt_exec_batch` refuses a negative `params_per_row` or NULL params.
- Functions returning the packed result format return `STOOLAP_ERROR` for a column name longer than 65,535 bytes.
- `stoolap_rows_next` and the batch fetch return an error for a TIMESTAMP outside the stored range.

## Build and Dependencies

- **jemalloc is the default global allocator** (the `jemalloc` feature, on by default). An embedder that installs its own `#[global_allocator]` must depend on `stoolap` with `default-features = false`, adding back `parallel` (and `cli` for the binary) as needed. The `mimalloc` feature is removed.
- The `heap-profile` feature enables `PRAGMA heap_profile`.
- The two comparison-benchmark features and `dhat-heap` are removed; the comparison benchmarks live in the unpublished `stoolap-compare` crate.
- The minimum Rust version stays 1.88, except for the optional `semantic` feature, which needs **Rust 1.94** or newer: it moved to candle 0.11 (for zip 9, which fixes zip 7's advisories), and candle's x86_64 code needs intrinsics that are stable or safe only from 1.94.
- The browser WASM build has its own size-optimized profile: `wasm-pack build --target web --profile wasm -- --no-default-features --features wasm`.

## Drivers

The Python, Node.js, Go, PHP, Ruby, Java, C#, Swift and MCP drivers are released separately. Upgrade a driver to its release built on Stoolap 0.5 to get this engine; the driver pages name the versions.

## Rollback

There is no automatic downgrade. To return to v0.4.2:

1. Stop the application.
2. Restore the database directory from the backup taken before the upgrade.
3. Reinstall v0.4.2.

Writes made under v0.5.0 are not in that backup. To carry them back, export them from v0.5.0 (for example with `SELECT` or `COPY`) and load them into the restored database.

## Troubleshooting

### "column reference 'x' is ambiguous"

Two sources of a join (or two columns of a derived table) carry the name. Qualify it: `t.x`. v0.4.2 silently read one of them.

### "unknown table or alias" or "table or alias 't' is not visible here"

The qualifier names no FROM entry in scope, or names a table that has an alias. v0.4.2 read the bare column instead. Use the alias, or the FROM entry that holds the column.

### "Column 'x' not found"

The name does not exist where it is written. v0.4.2 often answered NULL instead. In a grouped query the same error means the query reads a column outside GROUP BY and outside any aggregate: add it to GROUP BY or wrap it in an aggregate such as `MAX(...)`.

### "Integer overflow"

An INTEGER computation left the INTEGER range. For ABS, negation, division and SUM, v0.4.2 returned NULL, a FLOAT or crashed. Cast to FLOAT before the operation if the larger range is intended.

### A table refuses every statement after the upgrade

Either a listed volume file is missing or unreadable (the error names the segments; restore the file and reopen), or the table has a column named `_rowid_` (rename it).

### The database does not open after a crash

The error names the WAL record, table and LSN that recovery could not apply. Restore from a backup snapshot with the CLI:

```bash
stoolap -d file:///path/to/db --restore                    # latest snapshot
stoolap -d file:///path/to/db --reset-volumes --restore    # when volumes or manifests are damaged
```

See [Persistence]({{ '/docs/architecture/persistence/' | relative_url }}).
