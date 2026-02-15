---
layout: doc
title: Transactions
category: SQL Features
order: 17
---

# Transactions

Stoolap provides full ACID transactions with multi-version concurrency control (MVCC). Transactions ensure that a group of operations either all succeed or all fail, maintaining data consistency.

## Auto-Commit Mode

By default, each SQL statement runs in its own implicit transaction and is automatically committed:

```sql
-- Each statement is independently committed
INSERT INTO users VALUES (1, 'Alice');
INSERT INTO users VALUES (2, 'Bob');
-- If the second INSERT fails, the first is already committed
```

## Explicit Transactions

Use `BEGIN` and `COMMIT` to group multiple statements into a single atomic operation:

```sql
BEGIN;
INSERT INTO accounts VALUES (1, 'Alice', 1000);
INSERT INTO accounts VALUES (2, 'Bob', 500);
UPDATE accounts SET balance = balance - 200 WHERE id = 1;
UPDATE accounts SET balance = balance + 200 WHERE id = 2;
COMMIT;
-- All four statements succeed or fail together
```

## ROLLBACK

Use `ROLLBACK` to discard all changes made within the current transaction:

```sql
BEGIN;
DELETE FROM important_data WHERE id = 1;
-- Oops, wrong row!
ROLLBACK;
-- The DELETE is undone, data is intact
```

## Savepoints

Savepoints allow partial rollback within a transaction:

```sql
BEGIN;
INSERT INTO orders VALUES (1, 'Order A');
SAVEPOINT sp1;
INSERT INTO orders VALUES (2, 'Order B');
-- Undo only Order B
ROLLBACK TO SAVEPOINT sp1;
-- Order A is still pending
COMMIT;
-- Only Order A is committed
```

For full savepoint documentation, see [Savepoints](../sql-features/savepoints).

## Isolation Levels

Stoolap supports two isolation levels:

### READ COMMITTED (Default)

Each statement sees data committed before the statement began. Different statements within the same transaction may see different snapshots:

```sql
BEGIN;
SELECT * FROM accounts; -- Sees data as of this moment
-- Another transaction commits changes here
SELECT * FROM accounts; -- May see the new changes
COMMIT;
```

### SNAPSHOT (Repeatable Read)

The entire transaction sees a consistent snapshot from when it began. No changes from other transactions are visible:

```sql
BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT;
SELECT * FROM accounts; -- Sees data as of BEGIN
-- Another transaction commits changes here
SELECT * FROM accounts; -- Still sees data as of BEGIN
COMMIT;
```

#### Isolation Level Aliases

The following SQL-standard isolation levels are accepted as aliases:

| Alias | Maps To |
|-------|---------|
| `REPEATABLE READ` | `SNAPSHOT` |
| `SERIALIZABLE` | `SNAPSHOT` |
| `READ UNCOMMITTED` | `READ COMMITTED` |

```sql
-- All equivalent - start a snapshot isolation transaction
BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT;
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
```

## Transaction Behavior with DDL

DDL statements (CREATE TABLE, DROP TABLE, ALTER TABLE) can participate in explicit transactions:

```sql
BEGIN;
CREATE TABLE temp_results (id INTEGER PRIMARY KEY, value TEXT);
INSERT INTO temp_results VALUES (1, 'result');
-- Both the table creation and insert are committed together
COMMIT;
```

### Setting Default Isolation Level

Use `SET` to change the default isolation level for the session:

```sql
-- Set default to SNAPSHOT for all subsequent transactions
SET isolation_level = 'SNAPSHOT';

-- Reset to default
SET isolation_level = 'READ COMMITTED';
```

Both `isolation_level` and `transaction_isolation` are accepted as variable names. If called within an active transaction, the isolation level is changed for that transaction only.

## Transaction Behavior Summary

| Statement | Behavior |
|-----------|----------|
| `BEGIN` | Starts an explicit transaction with default isolation |
| `BEGIN TRANSACTION ISOLATION LEVEL ...` | Starts a transaction with specified isolation level |
| `COMMIT` | Commits all pending changes |
| `ROLLBACK` | Discards all pending changes |
| `SAVEPOINT name` | Creates a named savepoint |
| `ROLLBACK TO SAVEPOINT name` | Rolls back to the savepoint |
| `RELEASE SAVEPOINT name` | Removes a savepoint |
| `SET isolation_level = '...'` | Sets default isolation level for the session |

## Concurrency

Stoolap uses MVCC (Multi-Version Concurrency Control) to handle concurrent transactions:

- Readers never block writers
- Writers never block readers
- Write conflicts are detected at commit time
- Each transaction sees a consistent view of the data based on its isolation level

For architectural details on MVCC, see [MVCC Implementation](../architecture/mvcc-implementation).
