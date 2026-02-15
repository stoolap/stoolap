---
layout: doc
title: ALTER TABLE
category: SQL Features
order: 18
---

# ALTER TABLE

ALTER TABLE modifies the structure of an existing table. All ALTER TABLE operations are crash-safe with full WAL durability and snapshot recovery support.

## ADD COLUMN

Add a new column to an existing table:

```sql
ALTER TABLE users ADD COLUMN last_login TIMESTAMP;
ALTER TABLE users ADD COLUMN score INTEGER;
```

Existing rows receive NULL for the new column. Subsequent inserts can provide values for the new column:

```sql
INSERT INTO users (id, name, score) VALUES (1, 'Alice', 100);
```

## DROP COLUMN

Remove a column from a table:

```sql
ALTER TABLE users DROP COLUMN last_login;
```

The column data is physically removed. Queries referencing the dropped column will return an error.

## RENAME COLUMN

Rename an existing column:

```sql
ALTER TABLE users RENAME COLUMN old_name TO new_name;
```

Data is preserved. The old name is no longer accessible after the rename.

## MODIFY COLUMN

Change a column's data type or nullability:

```sql
-- Change column type
ALTER TABLE products MODIFY COLUMN price TEXT;

-- Allow NULL values on a previously NOT NULL column
ALTER TABLE config MODIFY COLUMN value INTEGER;
```

## RENAME TABLE

Rename an entire table:

```sql
ALTER TABLE users RENAME TO system_users;
```

All data is accessible via the new name. The old table name is no longer valid.

## Persistence

All ALTER TABLE operations are recorded in the WAL and survive crash recovery:

1. The schema change is applied immediately
2. A DDL entry is written to the WAL
3. On recovery, the WAL replays the ALTER TABLE operation
4. Snapshots include the updated schema

This means ALTER TABLE changes persist correctly even if the database crashes immediately after the operation.

## Limitations

- ALTER TABLE operations may temporarily block concurrent writes
- Adding constraints to existing columns (e.g., adding NOT NULL to a column with NULL values) is not supported
- Composite primary key modifications are not supported
