---
layout: doc
title: Upsert (ON CONFLICT / ON DUPLICATE KEY)
category: SQL Features
order: 11
---

# Upsert (ON CONFLICT / ON DUPLICATE KEY)

Stoolap supports both PostgreSQL-style `ON CONFLICT` and MySQL-style `ON DUPLICATE KEY UPDATE` for upsert operations. Both work with INSERT ... VALUES and INSERT ... SELECT.

## Syntax

### PostgreSQL Style (ON CONFLICT)

```sql
-- Upsert: update on conflict
INSERT INTO table_name (column1, column2, ...)
VALUES (value1, value2, ...)
ON CONFLICT (conflict_column1, ...) DO UPDATE SET
    column1 = EXCLUDED.column1,
    column2 = expression;

-- Skip duplicates silently
INSERT INTO table_name (column1, column2, ...)
VALUES (value1, value2, ...)
ON CONFLICT (conflict_column1, ...) DO NOTHING;

-- DO NOTHING without specifying conflict target (any constraint)
INSERT INTO table_name (column1, column2, ...)
VALUES (value1, value2, ...)
ON CONFLICT DO NOTHING;
```

### MySQL Style (ON DUPLICATE KEY UPDATE)

```sql
INSERT INTO table_name (column1, column2, ...)
VALUES (value1, value2, ...)
ON DUPLICATE KEY UPDATE
    column1 = EXCLUDED.column1,
    column2 = expression;
```

Both styles are triggered when:
- A primary key conflict occurs
- A unique index conflict occurs (single or composite)

## EXCLUDED Pseudo-Table

Use `EXCLUDED.column_name` to reference the values from the attempted INSERT row. This works with both syntax styles.

```sql
-- Use incoming values in the update
INSERT INTO products (id, name, price)
VALUES (1, 'Updated Name', 29.99)
ON CONFLICT (id) DO UPDATE SET
    name = EXCLUDED.name,
    price = EXCLUDED.price;

-- Mix EXCLUDED with expressions
INSERT INTO inventory (product_id, quantity)
VALUES (101, 25)
ON CONFLICT (product_id) DO UPDATE SET
    quantity = quantity + EXCLUDED.quantity;
```

## Examples

### ON CONFLICT DO UPDATE SET

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL,
    email TEXT
);

INSERT INTO users VALUES (1, 'alice', 'alice@example.com');

-- Upsert with PostgreSQL syntax
INSERT INTO users VALUES (1, 'alice_new', 'newalice@example.com')
ON CONFLICT (id) DO UPDATE SET
    username = EXCLUDED.username,
    email = EXCLUDED.email;
-- Result: id=1, username='alice_new', email='newalice@example.com'
```

### ON CONFLICT DO NOTHING

```sql
CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT);

INSERT INTO items VALUES (1, 'apple');
INSERT INTO items VALUES (2, 'banana');

-- Silently skip the duplicate
INSERT INTO items VALUES (1, 'cherry')
ON CONFLICT DO NOTHING;
-- Result: 2 rows (apple, banana), cherry was skipped
```

### With Composite Unique Constraint

```sql
CREATE TABLE metrics (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    host TEXT NOT NULL,
    metric TEXT NOT NULL,
    value FLOAT NOT NULL,
    UNIQUE(host, metric)
);

INSERT INTO metrics (host, metric, value) VALUES ('server1', 'cpu', 45.0);

-- Same (host, metric) pair triggers update
INSERT INTO metrics (host, metric, value)
VALUES ('server1', 'cpu', 88.0)
ON CONFLICT (host, metric) DO UPDATE SET value = EXCLUDED.value;
-- Result: value updated to 88.0
```

### Updating with Expressions

```sql
CREATE TABLE counters (id INTEGER PRIMARY KEY, name TEXT, count INTEGER);

INSERT INTO counters VALUES (1, 'visits', 10);

-- Increment count on conflict
INSERT INTO counters VALUES (1, 'visits', 5)
ON CONFLICT (id) DO UPDATE SET count = count + EXCLUDED.count;
-- Result: count = 15 (10 + 5)
```

### INSERT ... SELECT with Upsert

```sql
CREATE TABLE staging (host TEXT, metric TEXT, value FLOAT);
CREATE TABLE metrics (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    host TEXT NOT NULL,
    metric TEXT NOT NULL,
    value FLOAT NOT NULL,
    UNIQUE(host, metric)
);

INSERT INTO metrics (host, metric, value) VALUES ('s1', 'cpu', 50.0);

INSERT INTO staging VALUES ('s1', 'cpu', 88.0);
INSERT INTO staging VALUES ('s1', 'mem', 72.0);

-- Bulk upsert from staging
INSERT INTO metrics (host, metric, value)
SELECT host, metric, value FROM staging
ON CONFLICT (host, metric) DO UPDATE SET value = EXCLUDED.value;
-- Result: cpu updated to 88.0, mem inserted as 72.0
```

### INSERT ... SELECT with DO NOTHING

```sql
-- Skip duplicates during bulk import
INSERT INTO dst (id, name)
SELECT id, name FROM src
ON CONFLICT DO NOTHING;
```

### With CTE

```sql
INSERT INTO target (id, name, score)
WITH src AS (
    SELECT id, name, score FROM source WHERE active = TRUE
)
SELECT id, name, score FROM src
ON CONFLICT (id) DO UPDATE SET
    name = EXCLUDED.name,
    score = EXCLUDED.score;
```

### MySQL-Style Syntax

Both styles produce the same behavior:

```sql
-- These are equivalent:
INSERT INTO t (id, val) VALUES (1, 'x')
ON CONFLICT (id) DO UPDATE SET val = EXCLUDED.val;

INSERT INTO t (id, val) VALUES (1, 'x')
ON DUPLICATE KEY UPDATE val = EXCLUDED.val;
```

## How It Works

1. Stoolap attempts the INSERT operation normally
2. If a unique constraint violation occurs (primary key, unique index, or composite unique):
   - **DO UPDATE SET**: identifies the conflicting row and updates specified columns
   - **DO NOTHING**: silently skips the row
3. `EXCLUDED.column` references resolve to the values from the attempted insert row

## Use Cases

1. **Upsert Operations**: Insert if a record doesn't exist, otherwise update it
2. **Bulk Data Import**: Use INSERT ... SELECT with ON CONFLICT for efficient batch upserts
3. **Metrics Collection**: Accumulate or overwrite time-series data with composite unique keys
4. **Idempotent Operations**: Use DO NOTHING for safe retry/replay of INSERT batches
5. **Data Synchronization**: Merge data from staging tables into production tables

## Considerations

1. **Multiple Unique Constraints**: A violation of any unique constraint triggers the conflict action.

2. **Performance**: Upsert performs better than separate SELECT + INSERT/UPDATE, as it avoids multiple roundtrips.

3. **Auto-increment Behavior**: When an insert is converted to an update, it does not consume an auto-increment value.

4. **EXCLUDED pseudo-table**: Use `EXCLUDED.column` to reference incoming insert values. Without EXCLUDED, column names in the SET clause refer to the existing row in the target table.

5. **Conflict target**: The column list in ON CONFLICT (...) is optional for DO NOTHING. For DO UPDATE SET, it documents which constraint you expect to conflict on.
