---
layout: doc
title: Temporal Queries (AS OF)
category: SQL Features
order: 1
---

# Temporal Queries (AS OF)

Stoolap supports temporal queries using the SQL:2011 standard `AS OF` clause, allowing you to query historical data at a specific point in time. This feature leverages Stoolap's MVCC (Multi-Version Concurrency Control) architecture to provide time travel capabilities.

## Overview

The `AS OF` clause enables you to view data as it existed at a specific transaction or timestamp. This is particularly useful for:

- Auditing and compliance
- Debugging data issues
- Analyzing historical trends
- Implementing point-in-time recovery
- Building Git-like branching for data (future feature)

## Syntax

Stoolap supports two types of temporal queries:

### AS OF TRANSACTION

Query data as it existed at a specific transaction ID:

```sql
SELECT * FROM table_name AS OF TRANSACTION transaction_id
```

### AS OF TIMESTAMP

Query data as it existed at a specific timestamp:

```sql
SELECT * FROM table_name AS OF TIMESTAMP 'timestamp_string'
```

The text is read in any form a TIMESTAMP accepts. A time outside the range a table stores, `1677-09-21` to `2262-04-11`, is an error rather than an empty answer: for "as of now or later", write the current time or a time inside the range.

## Usage Examples

### Basic AS OF TRANSACTION Query

```sql
-- View orders table as of transaction 42
SELECT * FROM orders AS OF TRANSACTION 42;

-- With WHERE clause
SELECT * FROM users AS OF TRANSACTION 100 
WHERE status = 'active';

-- With specific columns, each as the row held it then
SELECT id, name, email FROM users AS OF TRANSACTION 50;
SELECT email, name FROM users AS OF TRANSACTION 50;
```

### Basic AS OF TIMESTAMP Query

```sql
-- View data as of a specific timestamp
SELECT * FROM events AS OF TIMESTAMP '2025-06-10 10:30:00';

-- Query data from yesterday
SELECT * FROM products AS OF TIMESTAMP '2025-06-09 23:59:59';

-- With filtering
SELECT * FROM transactions AS OF TIMESTAMP '2025-06-10 09:00:00'
WHERE amount > 1000;
```

### Using Table Aliases

```sql
-- With explicit alias
SELECT u.id, u.name
FROM users AS u AS OF TRANSACTION 75
WHERE u.created_at < '2025-01-01';

-- Without explicit alias (implicit alias)
SELECT users.id, users.name
FROM users AS OF TIMESTAMP '2025-06-10 12:00:00'
WHERE users.status = 'active';
```

### AS OF in JOIN Clauses

You can use `AS OF TIMESTAMP` and `AS OF TRANSACTION` directly in JOIN clauses to compare current data with historical data in a single query:

```sql
-- Compare current prices with historical prices
SELECT
    c.id,
    c.name,
    c.price AS current_price,
    h.price AS old_price,
    c.price - h.price AS price_change
FROM products c
JOIN products AS OF TIMESTAMP '2024-01-01 00:00:00' h ON c.id = h.id
WHERE c.price != h.price;

-- Find users whose status changed since a specific transaction
SELECT
    current.id,
    current.name,
    current.status AS current_status,
    old.status AS previous_status
FROM users current
JOIN users AS OF TRANSACTION 500 old ON current.id = old.id
WHERE current.status != old.status;

-- LEFT JOIN to include new records not in historical data
SELECT
    c.id,
    c.name,
    c.price AS current_price,
    h.price AS historical_price
FROM products c
LEFT JOIN products AS OF TIMESTAMP '2024-06-01 00:00:00' h ON c.id = h.id;
```

This feature enables powerful use cases:
- **Price change detection**: Compare current vs historical prices
- **Audit trails**: Track what changed between two points in time
- **Data reconciliation**: Verify data consistency across time periods
- **Trend analysis**: Analyze how values evolved over time

## How It Works

A point is a moment in the order commits complete, the order in which they become visible.

1. **Timestamp-based Queries**: `AS OF TIMESTAMP t` reads, for each row, the newest version whose transaction had completed its commit at or before `t`. A transaction that wrote before `t` and committed after it is not part of the answer. A time that has not passed yet when the statement starts reads the newest committed state at its start.

2. **Transaction-based Queries**: `AS OF TRANSACTION n` reads the state right after transaction `n` completed its commit, including every commit that completed before it, whatever the transaction ids. A transaction that is still active, was rolled back, wrote nothing, or is no longer in the history the database keeps is an error.

3. **Deletion Handling**: A row deleted by a commit that completed at or before the point does not appear; one deleted later does.

4. **Columns**: Each version is read whole, so a query answers the columns it names, in any order and with or without a `WHERE`, as the row held them at that point. A column added after a version was written reads in it as the column's default, or NULL when it has none.

5. **Stable answers**: The answer for a given transaction, and for a time that has passed, never changes: a commit that completes later is never part of an earlier point.

## History Window

Stoolap answers a point exactly or not at all. When it no longer holds what a point needs, the statement fails with an error that names the oldest point it can still read. A point is out of reach when it is:

- before the database was last opened: history does not survive a restart;
- older than the version history a table keeps: a row keeps up to 10 earlier versions, and versions older than 24 hours are let go in the background, so a table updated often keeps a shorter window;
- before a `TRUNCATE`, an `ALTER TABLE` that adds, drops or changes a column, or the creation of the table;
- before the table's newest checkpoint moved rows into its volumes, or before a compaction of them;
- older than the database's commit log, which keeps the last 262,144 commits.

The window is a property of each table: 24 hours is the longest it keeps, not a guarantee.

## Important Notes

### Timestamp Format

- Timestamps should be provided in UTC to match Stoolap's internal timestamp handling
- The timestamp string format is flexible and supports ISO 8601 and common date/time formats
- Common formats include:
  - `'2025-06-10 14:30:00'`
  - `'2025-06-10T14:30:00Z'`
  - `'2025-06-10 14:30:00.123456'`

### Performance Considerations

- An AS OF query reads every row version of the table and the volumes it holds; it uses no index
- A transaction-based point is found by scanning the commit log, a timestamp-based one by a binary search in it

### VACUUM Warning

Running `VACUUM` (or `PRAGMA vacuum`) permanently removes all historical row versions not needed by currently active transactions. After a VACUUM, an AS OF query naming a point before it fails with the history error. If you rely on time-travel queries, leave history to the background cleanup instead of running VACUUM.

### Limitations

- Subqueries with AS OF are not yet supported
- Inside an explicit transaction, a table read AS OF in a statement without subqueries is read at its current state
- The timestamp resolution depends on the system clock precision

## Use Cases

### Auditing

```sql
-- See what a user's profile looked like before an update
SELECT * FROM user_profiles AS OF TRANSACTION 1000
WHERE user_id = 123;
```

### Debugging

```sql
-- Check inventory levels at a specific time
SELECT product_id, quantity FROM inventory 
AS OF TIMESTAMP '2025-06-10 09:00:00'
WHERE product_id IN (101, 102, 103);
```

### Historical Analysis

```sql
-- Compare current prices with last week's prices
SELECT
    c.product_id,
    c.price AS current_price,
    h.price AS last_week_price,
    c.price - h.price AS price_difference
FROM products c
JOIN products AS OF TIMESTAMP '2025-06-03 00:00:00' h ON c.product_id = h.product_id;
```

## Future Enhancements

The AS OF feature is the foundation for Stoolap's planned "Git for Data" functionality, which will include:

- Named branches for data versioning
- Data merging capabilities
- Conflict resolution
- Tagged versions

## See Also

- [MVCC Implementation]({% link _docs/architecture/mvcc-implementation.md %})
- [Transaction Isolation]({% link _docs/architecture/transaction-isolation.md %})