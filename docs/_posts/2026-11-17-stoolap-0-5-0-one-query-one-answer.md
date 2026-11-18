---
layout: post
title: "Stoolap 0.5.0: One Query, One Answer"
author: Semih Alev
date: 2026-11-17
category: engineering
---

Stoolap 0.5.0 is out. It is the largest release so far: 670 merged pull requests since 0.4.2, 182 closed issues, and a test suite that grew from 6,032 to 8,580 tests. The [release notes](https://github.com/stoolap/stoolap/releases/tag/v0.5.0) list what changed. This post is about why most of it changed, because the theme of the release is not a feature.

The theme is this: a query should give the same answer no matter which path inside the engine computes it. In 0.4.2 that was not always true, and I want to start by showing it.

## Four outputs from 0.4.2

Every output in this section was produced today by the `v0.4.2` tag and by `v0.5.0`, running the same program.

**An index changes the answer.**

```sql
CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER);
INSERT INTO t VALUES (1, 1), (2, 2);

SELECT id FROM t WHERE k IN (1, 1.0);   -- 0.4.2: 1
CREATE INDEX ik ON t(k);
SELECT id FROM t WHERE k IN (1, 1.0);   -- 0.4.2: 1, 1
```

Without the index, the scan tested each row once against the list. With the index, the list was answered by probing the index once per member, and nothing removed a row that two members found. Even `k IN (1, 1)` returned the row twice. 0.5.0 returns one row both times.

**A checkpoint changes the answer.**

```sql
CREATE TABLE f (id INTEGER PRIMARY KEY, x FLOAT);
INSERT INTO f VALUES (1, 1.0), (2, CAST('NaN' AS FLOAT)), (3, 4.0);

SELECT MAX(x) FROM f;    -- 0.4.2: NaN
PRAGMA CHECKPOINT;
SELECT MAX(x) FROM f;    -- 0.4.2: 4.0
```

Stoolap keeps recent rows in a row-oriented hot store and seals older rows into compressed columnar volumes. A checkpoint moves the rows from one to the other. The rows did not change, but the code computing `MAX` did, and the two paths disagreed about where NaN sorts. 0.5.0 answers NaN both times.

**One flipped byte loses a table.**

I wrote 100,000 rows to a file database, checkpointed, closed it, flipped one byte in the volume file, and opened it again.

```text
before:        SELECT COUNT(*), SUM(v) FROM big  -> 100000, 5000050000
after reopen:  SELECT COUNT(*), SUM(v) FROM big  -> 0, NULL
stderr:        Warning: Failed to read volume ...: V4: CRC mismatch. Skipping file.
```

The checksum caught the damage. The engine then skipped the file, printed a warning to stderr and answered from what was left, which was nothing. No error reached the application.

**An error becomes a false.**

```sql
CREATE TABLE a (id INTEGER PRIMARY KEY, v INTEGER);
INSERT INTO a VALUES (1, 1), (2, 9223372036854775807);

DELETE FROM a WHERE v + 1 > 0;   -- 0.4.2: ok, 1 row deleted
SELECT id FROM a;                -- 0.4.2: 2
```

For row 2, `v + 1` overflows. 0.4.2 read that error as false, deleted row 1, and reported success. 0.5.0 fails the statement with `Integer overflow in arithmetic operation` and deletes nothing.

None of these are exotic. They are the ordinary result of an engine that grew fast paths faster than it grew rules.

## Why it happened: one rule, many copies

A database engine has a small number of semantic rules: how two values compare, how they sort, when two values are the same group key, how arithmetic overflows, how a value converts to another type. It also has a large number of paths that apply those rules: the expression VM, filters pushed into storage, index probes, hash joins, semi-joins, the columnar aggregates over sealed volumes, top-k, keyset pagination.

Every fast path is written by someone who needs one more comparison and writes a small one for the types in front of them. Each copy is correct for the case its author tested. Together they disagree.

When we went looking, we found this pattern everywhere:

- Function volatility (is `NOW()` a constant within a statement?) was decided in eight places. `NOW()` was frozen in storage pushdown and volatile for window functions.
- Comparing a number with a TEXT had a different answer in the VM, in pushdown, in index probes and in joins. A test over generated statements found 282 wrong result cells before the rule was unified.
- `ORDER BY` an alias answered wrong in 448 of 1,270 generated cases.
- In a join, a bare column name present on both sides silently read the left one. The test suite found 924 statement shapes that did this.
- Aggregate modifiers (`DISTINCT`, `FILTER`, `ORDER BY` inside an aggregate) were applied by each shortcut on its own, with 30 wrong answers between them. `COUNT(DISTINCT ts)` gave 0.

You cannot fix this class of bug one bug at a time. You fix one path, and the sibling path doing the same job still disagrees.

## What 0.5.0 does instead

The rule we adopted is short: **a semantic rule lives in one function, and a fast path calls that function.** It never carries its own copy, not even a simpler one for the types it handles.

Concretely:

- **Comparison.** One function decides how any two values compare. A TEXT that reads as a number compares as that number, and any other TEXT makes the comparison UNKNOWN, so `1 = '01'` is true and `1 = 'abc'` is NULL. BOOLEAN and TIMESTAMP are separate domains from numbers. JSON and VECTOR only take `=` and `<>`.
- **Order.** FLOAT has one total order: NaN equals NaN and sorts after every number, and `0.0` equals `-0.0`. Sorting across types follows one order, BOOLEAN, numbers, TEXT, TIMESTAMP. Zone maps, bloom filters and side-file indexes were adjusted so they never prune a row the rule keeps.
- **Keys.** One key equality is used by GROUP BY, DISTINCT, IN sets and hash joins, so `1` and `1.0` are one key everywhere.
- **Arithmetic.** One module does INTEGER arithmetic and raises `Integer overflow` outside the range. In 0.4.2, `i64::MIN / -1` crashed the process and `ABS(i64::MIN)` returned NULL.
- **Sums.** `SUM` and `AVG` use one exact accumulator, summing integers in `i128`. `SUM` over INTEGER returns an INTEGER on every path, where it used to wrap on one path and return FLOAT on another.

Unifying the rule is half the work. The other half is keeping it unified. Every optimized path has a name in the code (`KeyReadPath`, `OrderLimitPath`, `DistinctPath`, `JoinPath` and so on), and every name has an equivalence test that runs its cases and compares each answer, including values, types and column names, with a recorded answer. An answer is recorded only when another path, run on the same query, gives the same rows. The test fails when a registered path has no case, so a new fast path cannot ship without one. There are 47 answer files today.

One detail matters more than it looks: a test's expected answer is written down, not computed at test time by running the query on another path of the engine. A test that compares two paths of the same engine passes when both are wrong in the same way.

## The engine stopped guessing

The `DELETE` above shows the other half of the problem. Even with consistent rules, an engine can hide an error by turning it into a value: an overflow read as NULL, a subquery in an unsupported position read as FALSE, an unknown qualifier quietly resolved to some other column.

0.5.0 removes those guesses. A runtime error fails the statement wherever it appears, in the SELECT list, WHERE, ORDER BY, joins, window arguments, UPDATE SET, DEFAULT and CHECK. A qualifier must name a FROM entry: `SELECT zz.x FROM u` used to answer `u.x`, and `DELETE FROM u WHERE zz.id = 3` deleted a row. A token after a complete statement is a parse error: `UPDATE t SET v = 1 WHERE id = 1 2` used to write the row and then answer 2. A subquery that returns two columns where one is expected is an error instead of silently reading the first.

The cost is visible: the [migration guide](/docs/development/migration-v042-to-v050/) is long, because some queries that ran on 0.4.2 now fail. I think that is the right trade. A query that fails tells you something is wrong today. A query that answers wrong tells you nothing, and you find out from your data later.

## The disk: checksums that do something

The flipped byte above was caught by a checksum and still lost the table. Detecting damage is not enough; what the engine does next is the part that matters.

0.5.0 writes volumes in a new format, V5. Every column block and every metadata section has its own CRC-32C, checked every time it is read. The manifest records each volume file's identity (length, a random nonce and the section table's checksum), so a file copied in from another table or another database is refused at open.

Here is the same experiment on 0.5.0:

```text
SELECT COUNT(*), SUM(v) FROM big        -> 100000, 5000050000
SELECT v FROM big WHERE id = 50000      -> IO error: ...vol: block CRC mismatch at column 1, group 0
SELECT SUM(v + 0) FROM big              -> IO error: ...vol: block CRC mismatch at column 1, group 0
PRAGMA VERIFY                           -> big | damaged | ...: block CRC mismatch at column 1 (v), group 0
```

The first query is still right: the row count and the per-group sums live in checksummed metadata sections, and those were intact. A query that needs the damaged block fails with an error naming the file, the column and the row group. The new `PRAGMA VERIFY` reads every volume, past the cache, and reports each one as `ok`, `damaged`, `unreadable` or `failed`. It changes nothing.

When a volume is missing or unreadable at open, its table becomes unavailable: every statement on it fails with an error naming the table and the segments, other tables keep working, and checkpoints stop truncating the WAL until it is resolved. In a test on a 1.1M-row table, one flipped byte used to make the table answer 845,000 rows with no error.

The same principle went through recovery. A committed WAL record that recovery cannot apply now stops the open with the record's operation, table and LSN, where 0.4.2 skipped it. And we closed the crash windows that lost committed data:

- A failed WAL write in the default sync mode cut the log back to the last synced offset, erasing commits that had already been acknowledged.
- An INSERT committed while a checkpoint was writing its catalog copies vanished after a crash, and an UPDATE of a sealed row in that window lost the row.
- A table created inside a transaction wrote no WAL record, so the table and its rows were gone after a crash.
- A refused TRUNCATE had already removed the sealed rows, and compaction could bring a deleted row back.

## Plan once, run many

The last large change is structural, and it serves the same theme.

0.4.2 had a general executor that interpreted statements, plus a growing set of fast paths that intercepted shapes they recognized. Each fast path was a place where the answer could differ from the general path. In 0.5.0 every statement is planned once, when it is prepared, and runs from that plan on one runner: single-table reads, aggregates, windows, set operations, DISTINCT, joins, UPDATE and DELETE. The general executor and its name-based compile cache are deleted.

The plan is the thing EXPLAIN prints, so EXPLAIN now shows what will run, with each fallback after `else`:

```text
EXPLAIN SELECT id, name, email FROM users WHERE id = 1;

Project id, name, email
  PkGet users id = 1
  else IndexScan users (id = 1)
    PkKey
```

`EXPLAIN ANALYZE` prints the same lines with the time and rows on the top line, and ends with the paths the run actually took:

```text
Project id, name, email (actual time=1.05ms, rows=1)
  PkGet users id = 1
  else IndexScan users (id = 1)
    PkKey
Paths: key_read.pk_get
```

Fewer paths also turned out to be faster ones. Some numbers from the pull requests, each measured on the shape it names:

- A primary-key SELECT inside an explicit transaction: about 800 ns and 36 allocations before, about 180 ns and 5 now.
- A prepared lookup on a unique index: 74% fewer instructions.
- `COUNT(*)` over a 10M-row join used to materialize the join and peaked at 1.2 GB. Aggregates over joins now consume the joined rows in parts of 262,144.
- Parsing a two-join query: 95 allocations before, 33 now.

## What 0.5.0 is not

It is not a promise that there are no wrong answers left. It is a structure in which a wrong answer has one place to live, and a test that names every path that could carry it.

It is also not compatible backwards on disk. Once 0.5.0 has written a manifest, 0.4.2 cannot open the database. Back up the directory before you upgrade, and read the [migration guide](/docs/development/migration-v042-to-v050/): every changed answer in it was run on both versions.

```toml
[dependencies]
stoolap = "0.5"
```

If you find a query whose answer depends on an index, a LIMIT or a checkpoint, that is a bug by definition now, and I would like to see it on [GitHub](https://github.com/stoolap/stoolap/issues).
