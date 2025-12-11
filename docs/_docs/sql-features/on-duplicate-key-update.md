---
layout: doc
title: ON DUPLICATE KEY UPDATE
category: SQL Features
order: 11
---

# ON DUPLICATE KEY UPDATE

This document explains the ON DUPLICATE KEY UPDATE feature in Stoolap based on the implementation and test files.

## Overview

Stoolap supports the ON DUPLICATE KEY UPDATE clause for INSERT statements. This feature allows you to insert a new row or update an existing one if the insertion would violate a unique constraint (primary key or unique index), all in a single statement.

## Syntax

```sql
INSERT INTO table_name (column1, column2, ...)
VALUES (value1, value2, ...)
ON DUPLICATE KEY UPDATE
    column1 = new_value1,
    column2 = new_value2,
    ...
```

The ON DUPLICATE KEY UPDATE clause is executed when:
- A primary key conflict occurs
- A unique index conflict occurs

## Examples

### With Primary Key Constraint

```sql
-- Create a test table with primary key
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL,
    email TEXT,
    age INTEGER
);

-- Insert initial data
INSERT INTO users (id, username, email, age) VALUES (1, 'user1', 'user1@example.com', 30);

-- Insert with the same primary key, triggering update
INSERT INTO users (id, username, email, age) 
VALUES (1, 'different_user', 'new_email@example.com', 40)
ON DUPLICATE KEY UPDATE 
    username = 'updated_user',
    email = 'updated@example.com', 
    age = 45;

-- The row now has the updated values
-- id=1, username='updated_user', email='updated@example.com', age=45
```

### With Unique Index Constraint

```sql
-- Create a table with a unique constraint on a non-primary key column
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    code TEXT NOT NULL,
    name TEXT,
    price FLOAT
);

-- Create unique index on code column
CREATE UNIQUE INDEX idx_product_code ON products(code);

-- Insert initial data
INSERT INTO products (id, code, name, price) VALUES (1, 'PROD-001', 'Original Product', 19.99);

-- Insert with the same product code, triggering update due to unique constraint
INSERT INTO products (id, code, name, price)
VALUES (999, 'PROD-001', 'Duplicate Code Product', 29.99)
ON DUPLICATE KEY UPDATE
    name = 'Updated Product',
    price = 39.99;

-- The row now has:
-- id=1 (original PK), code='PROD-001', name='Updated Product', price=39.99
```

### Updating with Expressions

```sql
-- Create inventory table
CREATE TABLE inventory (
    id INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL
);

-- Create unique index on product_id
CREATE UNIQUE INDEX idx_inventory_product ON inventory(product_id);

-- Insert initial inventory
INSERT INTO inventory (id, product_id, quantity) VALUES (1, 101, 50);

-- Update with an expression that changes quantity
INSERT INTO inventory (id, product_id, quantity)
VALUES (999, 101, 25)
ON DUPLICATE KEY UPDATE
    quantity = quantity + 25;

-- The row now has:
-- id=1, product_id=101, quantity=75 (50 + 25)
```

## How It Works

1. Stoolap attempts the INSERT operation normally
2. If a unique constraint violation occurs:
   - The system identifies the conflicting row
   - Instead of returning an error, it performs an UPDATE on that row
   - Only the columns specified in the ON DUPLICATE KEY UPDATE clause are modified

## Use Cases

This feature is particularly useful for:

1. **Upsert Operations**: Insert if a record doesn't exist, otherwise update it
2. **Data Import**: Handle potential duplicates gracefully during batch imports
3. **Distributed Systems**: Handle potential race conditions and retries
4. **API Endpoints**: Implement idempotent PUT/POST operations

## Considerations and Limitations

1. **Multiple Unique Constraints**: If a table has multiple unique constraints, a violation of any of them will trigger the update.

2. **Performance**: ON DUPLICATE KEY UPDATE typically performs better than separate SELECT then INSERT or UPDATE operations, as it avoids multiple roundtrips.

3. **Auto-increment Behavior**: When an insert is converted to an update, it doesn't consume an auto-increment value.

4. **Column Selection**: Only specify columns that need updating in the ON DUPLICATE KEY UPDATE clause for better performance.

5. **No Inserted Value Reference**: Unlike some databases, Stoolap doesn't provide special syntax to reference values from the failed insert (like MySQL's VALUES() function).

## Implementation Details

Internally, Stoolap:

1. Attempts the INSERT operation
2. Catches unique constraint violations
3. Identifies the conflicting row using an appropriate search expression
4. Creates and applies an update function with the specified column values
5. Returns success if the update succeeds

This implementation allows for efficient "upsert" operations without requiring separate transactions or error handling code.