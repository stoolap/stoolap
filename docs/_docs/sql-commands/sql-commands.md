---
title: SQL Commands
category: SQL Commands
order: 1
---

# SQL Commands

This document provides a comprehensive reference to SQL commands supported by Stoolap.

## Data Manipulation Language (DML)

### SELECT

The SELECT statement retrieves data from one or more tables.

#### Basic Syntax

```sql
SELECT [DISTINCT] column1, column2, ...
FROM table_name
[WHERE condition]
[GROUP BY column1, column2, ...]
[HAVING condition]
[ORDER BY column1 [ASC|DESC], column2 [ASC|DESC], ...]
[LIMIT count [OFFSET offset]]
```

#### Parameters

- **DISTINCT**: Optional keyword that removes duplicate rows from the result
- **column1, column2, ...**: Columns to retrieve; use `*` for all columns
- **table_name**: The table to query
- **WHERE condition**: Optional filter condition
- **GROUP BY**: Optional grouping of rows by specified columns
- **HAVING**: Optional filter applied to groups
- **ORDER BY**: Optional sorting of results
- **LIMIT**: Optional restriction on the number of rows returned
- **OFFSET**: Optional number of rows to skip

#### Examples

Basic query:

```sql
SELECT id, name, price FROM products;
```

Filtering with WHERE:

```sql
SELECT * FROM products WHERE price > 50.00;
```

Sorting with ORDER BY:

```sql
SELECT * FROM products ORDER BY price DESC;
```

Limiting results:

```sql
SELECT * FROM customers LIMIT 10;
```

Unique values with DISTINCT:

```sql
SELECT DISTINCT category FROM products;
```

Aggregation with GROUP BY:

```sql
SELECT category, AVG(price) AS avg_price 
FROM products 
GROUP BY category;
```

Filtering groups with HAVING:

```sql
SELECT category, COUNT(*) AS product_count 
FROM products 
GROUP BY category 
HAVING COUNT(*) > 5;
```

#### JOIN Operations

Combining data from multiple tables:

```sql
-- INNER JOIN
SELECT p.id, p.name, c.name AS category
FROM products p
INNER JOIN categories c ON p.category_id = c.id;

-- LEFT JOIN
SELECT c.id, c.name, o.id AS order_id
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id;
```

#### Subqueries

Using subqueries in WHERE clause:

```sql
-- Find customers who have placed orders
SELECT * FROM customers 
WHERE id IN (
    SELECT DISTINCT customer_id FROM orders
);

-- Find products with above-average price
SELECT * FROM products 
WHERE price > (
    SELECT AVG(price) FROM products
);
```

#### Common Table Expressions (CTEs)

Using WITH clause to define temporary named result sets:

```sql
-- Simple CTE
WITH high_value_orders AS (
    SELECT * FROM orders WHERE total > 1000
)
SELECT * FROM high_value_orders;

-- Multiple CTEs
WITH 
customer_totals AS (
    SELECT customer_id, SUM(total) as total_spent
    FROM orders
    GROUP BY customer_id
),
vip_customers AS (
    SELECT * FROM customer_totals WHERE total_spent > 10000
)
SELECT c.name, ct.total_spent
FROM customers c
JOIN vip_customers ct ON c.id = ct.customer_id;
```

### INSERT

The INSERT statement adds new rows to a table.

#### Basic Syntax

```sql
-- Single row insertion
INSERT INTO table_name [(column1, column2, ...)]
VALUES (value1, value2, ...);

-- Multiple row insertion
INSERT INTO table_name [(column1, column2, ...)]
VALUES 
  (value1_1, value1_2, ...),
  (value2_1, value2_2, ...),
  ...;

-- With ON DUPLICATE KEY UPDATE
INSERT INTO table_name [(column1, column2, ...)]
VALUES (value1, value2, ...)
ON DUPLICATE KEY UPDATE
  column1 = new_value1,
  column2 = new_value2,
  ...;
```

#### Parameters

- **table_name**: The table to insert data into
- **column1, column2, ...**: Optional list of columns to insert into
- **value1, value2, ...**: Values to insert corresponding to the columns
- **ON DUPLICATE KEY UPDATE**: Optional clause to update existing rows on conflict

#### Examples

Basic insertion:

```sql
INSERT INTO customers (id, name, email) 
VALUES (1, 'John Doe', 'john@example.com');
```

Multiple row insertion:

```sql
INSERT INTO products (id, name, price) VALUES 
(1, 'Laptop', 1200.00),
(2, 'Smartphone', 800.00),
(3, 'Tablet', 500.00);
```

With ON DUPLICATE KEY UPDATE:

```sql
INSERT INTO inventory (product_id, quantity) 
VALUES (101, 50)
ON DUPLICATE KEY UPDATE
  quantity = quantity + 50;
```

### UPDATE

The UPDATE statement modifies existing data in a table.

#### Basic Syntax

```sql
UPDATE table_name
SET column1 = value1, column2 = value2, ...
[WHERE condition];
```

#### Parameters

- **table_name**: The table to update
- **column1, column2, ...**: Columns to update
- **value1, value2, ...**: New values for the columns
- **WHERE condition**: Optional condition to specify which rows to update

#### Examples

Update a single row:

```sql
UPDATE customers 
SET email = 'new.email@example.com' 
WHERE id = 1;
```

Update multiple rows:

```sql
UPDATE products 
SET price = price * 1.1 
WHERE category = 'Electronics';
```

Update all rows:

```sql
UPDATE settings 
SET last_updated = NOW();
```

Update using a subquery:

```sql
-- Apply discount to products in premium categories
UPDATE products 
SET discount = 0.15 
WHERE category IN (
    SELECT name FROM categories WHERE is_premium = true
);

-- Reset prices for discontinued items
UPDATE products 
SET price = 0, status = 'discontinued'
WHERE id NOT IN (
    SELECT product_id FROM active_inventory
);
```

### DELETE

The DELETE statement removes rows from a table.

#### Basic Syntax

```sql
DELETE FROM table_name
[WHERE condition];
```

#### Parameters

- **table_name**: The table to delete from
- **WHERE condition**: Optional condition to specify which rows to delete

#### Examples

Delete a single row:

```sql
DELETE FROM customers WHERE id = 1;
```

Delete multiple rows with condition:

```sql
DELETE FROM orders WHERE order_date < '2023-01-01';
```

Delete all rows:

```sql
DELETE FROM temporary_logs;
```

Delete using a subquery:

```sql
-- Delete orders from inactive customers
DELETE FROM orders 
WHERE customer_id IN (
    SELECT id FROM customers WHERE status = 'inactive'
);

-- Delete orphaned records
DELETE FROM order_items 
WHERE order_id NOT IN (
    SELECT id FROM orders
);
```

## Data Definition Language (DDL)

### CREATE TABLE

Creates a new table with specified columns and constraints.

#### Basic Syntax

```sql
CREATE TABLE [IF NOT EXISTS] table_name (
    column_name data_type [constraints...],
    column_name data_type [constraints...],
    ...
);
```

#### Parameters

- **IF NOT EXISTS**: Optional clause that prevents an error if the table already exists
- **table_name**: Name of the table to create
- **column_name**: Name of a column in the table
- **data_type**: Data type of the column (INTEGER, TEXT, FLOAT, BOOLEAN, TIMESTAMP, JSON)
- **constraints**: Optional column constraints (NOT NULL, PRIMARY KEY)

#### Examples

Basic table creation:

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL,
    email TEXT,
    age INTEGER,
    created_at TIMESTAMP
);
```

Using IF NOT EXISTS:

```sql
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    price FLOAT NOT NULL,
    category TEXT
);
```

### ALTER TABLE

Modifies an existing table.

#### Basic Syntax

```sql
ALTER TABLE table_name operation;
```

Where `operation` is one of:

```sql
ADD COLUMN column_name data_type [constraints...]
DROP COLUMN column_name
RENAME COLUMN column_name TO new_column_name
RENAME TO new_table_name
```

#### Parameters

- **table_name**: Name of the table to alter
- **column_name**: Name of the column to alter
- **new_column_name**: New name for the column when renaming
- **new_table_name**: New name for the table when renaming
- **data_type**: Data type for a new column
- **constraints**: Optional constraints for a new column

#### Examples

Adding a column:

```sql
ALTER TABLE users ADD COLUMN last_login TIMESTAMP;
```

Dropping a column:

```sql
ALTER TABLE users DROP COLUMN age;
```

Renaming a column:

```sql
ALTER TABLE users RENAME COLUMN username TO user_name;
```

Renaming a table:

```sql
ALTER TABLE users RENAME TO customers;
```

### DROP TABLE

Removes a table and all its data.

#### Basic Syntax

```sql
DROP TABLE [IF EXISTS] table_name;
```

#### Parameters

- **IF EXISTS**: Optional clause that prevents an error if the table doesn't exist
- **table_name**: Name of the table to drop

#### Examples

Basic table drop:

```sql
DROP TABLE temporary_data;
```

Using IF EXISTS:

```sql
DROP TABLE IF EXISTS temporary_logs;
```

### CREATE INDEX

Creates an index on table columns.

#### Basic Syntax

```sql
-- Regular index
CREATE [UNIQUE] INDEX [IF NOT EXISTS] index_name 
ON table_name (column_name [, column_name...]);

-- Columnar index
CREATE [UNIQUE] COLUMNAR INDEX [IF NOT EXISTS] 
ON table_name (column_name);
```

#### Parameters

- **UNIQUE**: Optional keyword that creates a unique index
- **IF NOT EXISTS**: Optional clause that prevents an error if the index already exists
- **index_name**: Name of the index to create
- **table_name**: Name of the table on which to create the index
- **column_name**: Name of the column(s) to include in the index

#### Examples

Single-column index:

```sql
CREATE INDEX idx_user_email ON users (email);
```

Multi-column index:

```sql
CREATE INDEX idx_name_category ON products (name, category);
```

Unique index:

```sql
CREATE UNIQUE INDEX idx_unique_username ON users (username);
```

Columnar index:

```sql
CREATE COLUMNAR INDEX ON products (category);
```

### DROP INDEX

Removes an index from a table.

#### Basic Syntax

```sql
-- Regular index
DROP INDEX [IF EXISTS] index_name ON table_name;

-- Columnar index
DROP COLUMNAR INDEX [IF EXISTS] ON table_name (column_name);
```

#### Parameters

- **IF EXISTS**: Optional clause that prevents an error if the index doesn't exist
- **index_name**: Name of the index to drop
- **table_name**: Name of the table containing the index
- **column_name**: Name of the column with the columnar index

#### Examples

Basic index drop:

```sql
DROP INDEX idx_user_email ON users;
```

Columnar index drop:

```sql
DROP COLUMNAR INDEX ON products (category);
```


## Transaction Control

### BEGIN TRANSACTION

Starts a new transaction.

#### Basic Syntax

```sql
BEGIN TRANSACTION;
```

#### Example

```sql
BEGIN TRANSACTION;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
```

### COMMIT

Commits the current transaction, making all changes permanent.

#### Basic Syntax

```sql
COMMIT;
```

### ROLLBACK

Rolls back the current transaction, discarding all changes.

#### Basic Syntax

```sql
ROLLBACK;
```

## Other Commands

### SHOW INDEXES

Lists all indexes for a table.

#### Basic Syntax

```sql
SHOW INDEXES FROM table_name;
```

#### Parameters

- **table_name**: The table to show indexes for

#### Example

```sql
SHOW INDEXES FROM users;
```

### PRAGMA

Sets or gets configuration options.

#### Basic Syntax

```sql
-- Set a configuration option
PRAGMA name = value;

-- Get the current value
PRAGMA name;
```

#### Supported PRAGMAs

- **sync_mode**: Controls WAL synchronization (0=None, 1=Normal, 2=Full)
- **snapshot_interval**: Controls how often snapshots are taken (in seconds)
- **keep_snapshots**: Controls how many snapshots to retain
- **wal_flush_trigger**: Controls when WAL is flushed to disk (number of operations)

#### Examples

```sql
-- Configure snapshot interval
PRAGMA snapshot_interval = 60;

-- Configure WAL sync mode
PRAGMA sync_mode = 2;

-- Configure WAL flush trigger
PRAGMA wal_flush_trigger = 1000;

-- Read current values
PRAGMA snapshot_interval;
PRAGMA sync_mode;
```

## Notes and Limitations

1. **Transactions**: Stoolap provides MVCC-based transactions for concurrent operations

2. **JOIN Support**: LEFT JOIN and INNER JOIN are supported, but RIGHT JOIN and FULL JOIN are not

3. **VIEW Support**: CREATE VIEW and DROP VIEW syntax is parsed but not yet implemented in the execution engine

4. **Advanced Features**: Some SQL features like window functions are implemented but with limitations

5. **Parameters**: Use `?` placeholder for parameter binding in prepared statements

6. **NULL Handling**: Stoolap follows standard SQL NULL semantics; NULL values are not equal to any value, including another NULL, and require IS NULL or IS NOT NULL operators for testing

7. **Type Conversion**: Stoolap performs implicit type conversions in some contexts, but explicit CAST is recommended for clarity