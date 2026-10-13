---
layout: doc
title: ALTER TABLE
category: SQL Features
order: 18
---

# ALTER TABLE

ALTER TABLE modifies the structure of an existing table. All ALTER TABLE operations are crash-safe with full WAL durability and volume recovery support.

## ADD COLUMN

Add a new column to an existing table:

```sql
ALTER TABLE users ADD COLUMN last_login TIMESTAMP;
ALTER TABLE users ADD COLUMN score INTEGER;
```

Existing rows receive the column's DEFAULT, or NULL without one, whether they are still in memory or already in a frozen volume. Subsequent inserts can provide values for the new column:

```sql
INSERT INTO users (id, name, score) VALUES (1, 'Alice', 100);
```

## DROP COLUMN

Remove a column from a table:

```sql
ALTER TABLE users DROP COLUMN last_login;
```

The column data is physically removed from the rows in memory at once; a frozen volume drops it at its next compaction. Queries referencing the dropped column will return an error.

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

Every existing value is converted to the new type, in memory and in the frozen volumes, before the statement returns. A value the new type cannot hold becomes NULL. The change is refused, with the table left as it was, when:

- the column is the primary key, a foreign key column, or a column a foreign key references;
- an index on the column cannot hold the new type (an HNSW index holds only vectors of a known dimension);
- a converted value is NULL and the column is NOT NULL;
- two rows would hold the same key of a UNIQUE index once converted (for example `'7'` and `'07'` changed to INTEGER);
- the column becomes a VECTOR of another dimension and a row holds a vector of the old one (values are never padded or cut).

The table is closed while the change runs: other statements on it return `SchemaChanged` until it finishes. Once the statement returns, the change survives a crash whatever the `sync_mode`.

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

If recording an ADD, DROP, RENAME, or MODIFY COLUMN operation in the WAL fails, the engine restores the previous schema and frozen-volume mappings. Reads can continue against that schema. A WAL write or sync failure still requires closing and reopening the database before further writes.

A MODIFY COLUMN whose WAL record may or may not have been written (the failed write could not be cut back), or whose table manifest could not be written after the record, returns an error and leaves the table closed until the database is reopened. The reopen finishes the change from the WAL, or finds it was never written.

## Concurrent statements

In a file database, a statement can return `SchemaChanged` if column DDL overlaps its capture of the schema and frozen-volume mappings. The engine rejects the mixed view instead of waiting for the ALTER to finish. A new statement can proceed after the ALTER completes.

A transaction that wrote rows to a table before another connection added or dropped one of its columns cannot commit those rows: its COMMIT returns `SchemaChanged` and the transaction rolls back. The rows were written with the columns as they were, and the table now lays its rows out as the columns are.

`SchemaChanged` is a public Rust error variant. The C API and drivers report it as a database error and do not automatically retry statements. Applications should coordinate column DDL with concurrent work and handle this error explicitly.

## Limitations

- ALTER TABLE operations may temporarily block concurrent writes
- Composite primary key modifications are not supported
- **Frozen volume tables**: ALTER TABLE works on tables with frozen volumes. Volume data is normalized on read (new columns return NULL/DEFAULT, dropped columns are skipped). The volume is rebuilt automatically on the next checkpoint cycle
