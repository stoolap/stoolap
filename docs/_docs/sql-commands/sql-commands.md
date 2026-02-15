---
layout: doc
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
[GROUP BY column1, ... | ROLLUP(column1, ...) | CUBE(column1, ...)]
[HAVING condition]
[ORDER BY column1 [ASC|DESC] [NULLS FIRST|NULLS LAST], ...]
[LIMIT count [OFFSET offset]]
```

#### Parameters

- **DISTINCT**: Removes duplicate rows from the result
- **column1, column2, ...**: Columns to retrieve; use `*` for all columns
- **table_name**: The table to query
- **WHERE condition**: Filter condition
- **GROUP BY**: Groups rows by specified columns
- **ROLLUP/CUBE**: Multi-dimensional aggregation (see [ROLLUP and CUBE](../sql-features/rollup-cube))
- **HAVING**: Filter applied to groups
- **ORDER BY**: Sorting of results (`NULLS FIRST` or `NULLS LAST` to control NULL placement)
- **LIMIT**: Maximum rows to return
- **OFFSET**: Number of rows to skip

#### Examples

```sql
-- Basic query
SELECT id, name, price FROM products;

-- Filtering
SELECT * FROM products WHERE price > 50.00 AND category = 'Electronics';

-- Sorting
SELECT * FROM products ORDER BY price DESC, name ASC;

-- Pagination
SELECT * FROM customers LIMIT 10 OFFSET 20;

-- Unique values
SELECT DISTINCT category FROM products;

-- Aggregation
SELECT category, AVG(price) AS avg_price, COUNT(*) as count
FROM products
GROUP BY category;

-- Filtering groups
SELECT category, COUNT(*) AS product_count
FROM products
GROUP BY category
HAVING COUNT(*) > 5;
```

#### JOIN Operations

Stoolap supports all standard JOIN types:

```sql
-- INNER JOIN
SELECT p.name, c.name AS category
FROM products p
INNER JOIN categories c ON p.category_id = c.id;

-- LEFT JOIN
SELECT c.name, o.id AS order_id
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id;

-- RIGHT JOIN
SELECT c.name, o.id AS order_id
FROM customers c
RIGHT JOIN orders o ON c.id = o.customer_id;

-- FULL OUTER JOIN
SELECT c.name, o.id
FROM customers c
FULL OUTER JOIN orders o ON c.id = o.customer_id;

-- CROSS JOIN
SELECT p.name, c.name
FROM products p
CROSS JOIN colors c;
```

See [JOIN Operations](../sql-features/join-operations) for detailed documentation.

#### Subqueries

Using subqueries in various clauses (both correlated and non-correlated):

```sql
-- Scalar subquery
SELECT name, price,
       (SELECT AVG(price) FROM products) as avg_price
FROM products;

-- IN subquery
SELECT * FROM customers
WHERE id IN (SELECT DISTINCT customer_id FROM orders);

-- EXISTS subquery (correlated)
SELECT * FROM customers c
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id);

-- NOT IN subquery
SELECT * FROM products
WHERE id NOT IN (SELECT product_id FROM discontinued_items);

-- Correlated subquery in WHERE
SELECT * FROM employees e1
WHERE salary > (SELECT AVG(salary) FROM employees e2 WHERE e2.department = e1.department);

-- ANY/ALL subquery
SELECT * FROM products WHERE price > ALL (SELECT price FROM products WHERE category = 'Books');

-- Derived table (subquery in FROM)
SELECT * FROM (SELECT id, name FROM products WHERE price > 100) AS expensive;
```

See [Subqueries](../sql-features/subqueries) for detailed documentation.

#### Common Table Expressions (CTEs)

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

-- Recursive CTE
WITH RECURSIVE numbers AS (
    SELECT 1 as n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < 10
)
SELECT * FROM numbers;
```

See [Common Table Expressions](../sql-features/common-table-expressions) for detailed documentation.

#### Set Operations

```sql
-- UNION (removes duplicates)
SELECT name FROM customers
UNION
SELECT name FROM suppliers;

-- UNION ALL (keeps duplicates)
SELECT name FROM customers
UNION ALL
SELECT name FROM suppliers;

-- INTERSECT
SELECT id FROM table1
INTERSECT
SELECT id FROM table2;

-- EXCEPT
SELECT id FROM table1
EXCEPT
SELECT id FROM table2;
```

#### VALUES as Table Source

The VALUES clause can be used as an inline table in queries:

```sql
-- Basic usage with column aliases
SELECT * FROM (VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')) AS t(id, name);

-- With filtering
SELECT * FROM (VALUES (1, 10), (2, 20), (3, 30)) AS t(id, val)
WHERE val >= 20;

-- With expressions
SELECT id, val * 2 AS doubled
FROM (VALUES (1, 10), (2, 20), (3, 30)) AS t(id, val);

-- With aggregation
SELECT SUM(x) FROM (VALUES (1), (2), (3), (4), (5)) AS t(x);

-- Without column aliases (uses column1, column2, ...)
SELECT * FROM (VALUES (10, 20), (30, 40)) AS t;
```

#### Temporal Queries (AS OF)

Query historical data at a specific point in time:

```sql
-- Query data as of a specific timestamp
SELECT * FROM orders AS OF TIMESTAMP '2024-01-15 10:30:00';

-- Query data as of current time
SELECT * FROM inventory AS OF TIMESTAMP NOW();
```

See [Temporal Queries](../sql-features/temporal-queries) for detailed documentation.

### INSERT

The INSERT statement adds new rows to a table.

#### Basic Syntax

```sql
-- Single row
INSERT INTO table_name [(column1, column2, ...)]
VALUES (value1, value2, ...)
[RETURNING *|column1, column2, ...];

-- Multiple rows
INSERT INTO table_name [(column1, column2, ...)]
VALUES
  (value1_1, value1_2, ...),
  (value2_1, value2_2, ...);

-- With ON DUPLICATE KEY UPDATE
INSERT INTO table_name [(column1, column2, ...)]
VALUES (value1, value2, ...)
ON DUPLICATE KEY UPDATE
  column1 = new_value1,
  column2 = new_value2;
```

#### Examples

```sql
-- Basic insertion
INSERT INTO customers (id, name, email)
VALUES (1, 'John Doe', 'john@example.com');

-- Multiple rows
INSERT INTO products (id, name, price) VALUES
(1, 'Laptop', 1200.00),
(2, 'Smartphone', 800.00),
(3, 'Tablet', 500.00);

-- With RETURNING clause
INSERT INTO users (name, email)
VALUES ('Alice', 'alice@example.com')
RETURNING id, name;

-- Upsert with ON DUPLICATE KEY UPDATE
INSERT INTO inventory (product_id, quantity)
VALUES (101, 50)
ON DUPLICATE KEY UPDATE
  quantity = quantity + 50;
```

See [ON DUPLICATE KEY UPDATE](../sql-features/on-duplicate-key-update) for detailed documentation.

#### INSERT INTO ... SELECT

Inserts rows from a query result into a table:

```sql
-- Copy rows from another table
INSERT INTO archive (id, name, amount)
SELECT id, name, amount FROM orders WHERE status = 'completed';

-- With expressions and aggregation
INSERT INTO summary (category, total)
SELECT category, SUM(amount) FROM sales GROUP BY category;

-- With JOIN
INSERT INTO user_totals (user_name, total_amount)
SELECT u.name, SUM(o.amount)
FROM users u
JOIN orders o ON u.id = o.user_id
GROUP BY u.name;

-- With UNION
INSERT INTO combined (val)
SELECT val FROM table1
UNION ALL
SELECT val FROM table2;

-- With CTE
INSERT INTO results (id, value)
WITH doubled AS (SELECT id, val * 2 AS val FROM source)
SELECT id, val FROM doubled;

-- With LIMIT
INSERT INTO top_items (name, price)
SELECT name, price FROM products ORDER BY price DESC LIMIT 10;
```

Columns not specified in the INSERT column list receive their DEFAULT values.

### UPDATE

The UPDATE statement modifies existing data.

#### Basic Syntax

```sql
UPDATE table_name
SET column1 = value1, column2 = value2, ...
[WHERE condition]
[RETURNING *|column1, column2, ...];
```

#### Examples

```sql
-- Update single row
UPDATE customers
SET email = 'new.email@example.com'
WHERE id = 1;

-- Update multiple rows
UPDATE products
SET price = price * 1.1
WHERE category = 'Electronics';

-- Update all rows
UPDATE settings
SET last_updated = NOW();

-- With RETURNING clause
UPDATE accounts
SET balance = balance + 100
WHERE id = 1
RETURNING id, balance;

-- Update using subquery
UPDATE products
SET discount = 0.15
WHERE category IN (
    SELECT name FROM categories WHERE is_premium = true
);
```

### DELETE

The DELETE statement removes rows from a table.

#### Basic Syntax

```sql
DELETE FROM table_name
[WHERE condition]
[RETURNING *|column1, column2, ...];
```

#### Examples

```sql
-- Delete single row
DELETE FROM customers WHERE id = 1;

-- Delete multiple rows
DELETE FROM orders WHERE order_date < '2023-01-01';

-- Delete all rows
DELETE FROM temporary_logs;

-- With RETURNING clause
DELETE FROM users WHERE inactive = true
RETURNING id, name;

-- Delete using subquery
DELETE FROM orders
WHERE customer_id IN (
    SELECT id FROM customers WHERE status = 'inactive'
);
```

### TRUNCATE

The TRUNCATE statement removes all rows from a table efficiently.

#### Basic Syntax

```sql
TRUNCATE TABLE table_name;
```

#### Example

```sql
-- Remove all rows (faster than DELETE)
TRUNCATE TABLE logs;
```

**Important notes:**
- TRUNCATE is faster than DELETE because it doesn't log individual row deletions
- Unlike DELETE, TRUNCATE **cannot be rolled back** — ROLLBACK will not restore truncated rows
- TRUNCATE will fail if another transaction has uncommitted changes on the table
- TRUNCATE will fail if the table is referenced by foreign key constraints with existing child rows

## Data Definition Language (DDL)

### CREATE TABLE

Creates a new table.

#### Basic Syntax

```sql
CREATE TABLE [IF NOT EXISTS] table_name (
    column_name data_type [constraints...],
    column_name data_type [constraints...],
    ...
);
```

#### Data Types

| Type | Description |
|------|-------------|
| INTEGER | 64-bit signed integer |
| FLOAT | 64-bit floating point |
| TEXT | Variable-length string |
| BOOLEAN | true/false |
| TIMESTAMP | Date and time |
| JSON | JSON data |

#### Column Constraints

| Constraint | Description |
|------------|-------------|
| PRIMARY KEY | Unique identifier, cannot be NULL |
| NOT NULL | Column cannot contain NULL values |
| AUTO_INCREMENT | Automatically generates sequential values |

#### Examples

```sql
-- Basic table
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL,
    email TEXT,
    age INTEGER,
    created_at TIMESTAMP
);

-- With AUTO_INCREMENT
CREATE TABLE posts (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    title TEXT NOT NULL,
    content TEXT,
    author_id INTEGER
);

-- With IF NOT EXISTS
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    price FLOAT NOT NULL
);
```

### CREATE TABLE AS SELECT

Creates a new table from the result of a SELECT query.

#### Basic Syntax

```sql
CREATE TABLE table_name AS SELECT ...;
```

#### Example

```sql
-- Create table from query result
CREATE TABLE active_users AS
SELECT id, name, email
FROM users
WHERE last_login > '2024-01-01';

-- Create summary table
CREATE TABLE daily_sales AS
SELECT DATE_TRUNC('day', order_date) as day,
       SUM(amount) as total
FROM orders
GROUP BY DATE_TRUNC('day', order_date);
```

### ALTER TABLE

Modifies an existing table.

#### Basic Syntax

```sql
ALTER TABLE table_name operation;
```

#### Operations

```sql
-- Add a column
ALTER TABLE users ADD COLUMN last_login TIMESTAMP;

-- Drop a column
ALTER TABLE users DROP COLUMN age;

-- Rename a column
ALTER TABLE users RENAME COLUMN username TO user_name;

-- Rename table
ALTER TABLE users RENAME TO customers;
```

### DROP TABLE

Removes a table and all its data.

#### Basic Syntax

```sql
DROP TABLE [IF EXISTS] table_name;
```

#### Examples

```sql
DROP TABLE temporary_data;
DROP TABLE IF EXISTS old_logs;
```

### CREATE VIEW

Creates a virtual table based on a SELECT statement.

#### Basic Syntax

```sql
CREATE VIEW view_name AS SELECT ...;
```

#### Examples

```sql
-- Simple view
CREATE VIEW active_products AS
SELECT * FROM products WHERE in_stock = true;

-- View with joins
CREATE VIEW order_details AS
SELECT o.id, c.name as customer, p.name as product, o.quantity
FROM orders o
JOIN customers c ON o.customer_id = c.id
JOIN products p ON o.product_id = p.id;

-- Query the view
SELECT * FROM active_products WHERE price > 100;
```

### DROP VIEW

Removes a view.

#### Basic Syntax

```sql
DROP VIEW [IF EXISTS] view_name;
```

#### Example

```sql
DROP VIEW active_products;
DROP VIEW IF EXISTS old_report;
```

### CREATE INDEX

Creates an index on table columns for faster queries.

#### Basic Syntax

```sql
CREATE [UNIQUE] INDEX [IF NOT EXISTS] index_name
ON table_name (column_name [, column_name...]);
```

#### Index Type Selection

Stoolap automatically selects the optimal index type based on column data type:

| Data Type | Index Type | Best For |
|-----------|------------|----------|
| INTEGER, FLOAT, TIMESTAMP | B-tree | Range queries, equality, sorting |
| TEXT, JSON | Hash | Equality lookups, IN clauses |
| BOOLEAN | Bitmap | Low-cardinality columns |

#### Examples

```sql
-- Single-column index
CREATE INDEX idx_user_email ON users (email);

-- Multi-column index
CREATE INDEX idx_order_customer_date ON orders (customer_id, order_date);

-- Unique index
CREATE UNIQUE INDEX idx_unique_email ON users (email);

-- With IF NOT EXISTS
CREATE INDEX IF NOT EXISTS idx_name ON products (name);
```

See [Indexing](../architecture/indexing) for detailed documentation.

### DROP INDEX

Removes an index from a table.

#### Basic Syntax

```sql
DROP INDEX [IF EXISTS] index_name ON table_name;
```

#### Example

```sql
DROP INDEX idx_user_email ON users;
DROP INDEX IF EXISTS idx_old ON products;
```

## Transaction Control

### BEGIN TRANSACTION

Starts a new transaction.

```sql
BEGIN TRANSACTION;
-- or simply
BEGIN;
```

### COMMIT

Commits the current transaction, making all changes permanent.

```sql
COMMIT;
```

### ROLLBACK

Rolls back the current transaction, discarding all changes.

```sql
ROLLBACK;
```

### SAVEPOINT

Creates a savepoint within a transaction for partial rollback.

```sql
-- Create a savepoint
SAVEPOINT savepoint_name;

-- Rollback to a savepoint
ROLLBACK TO SAVEPOINT savepoint_name;

-- Release a savepoint
RELEASE SAVEPOINT savepoint_name;
```

#### Example

```sql
BEGIN TRANSACTION;

INSERT INTO accounts (id, balance) VALUES (1, 1000);
SAVEPOINT after_insert;

UPDATE accounts SET balance = 500 WHERE id = 1;
-- Oops, wrong update
ROLLBACK TO SAVEPOINT after_insert;

-- Continue with correct update
UPDATE accounts SET balance = 900 WHERE id = 1;
COMMIT;
```

See [Savepoints](../sql-features/savepoints) for detailed documentation.

## Query Analysis

### EXPLAIN

Shows the query execution plan.

```sql
EXPLAIN SELECT * FROM users WHERE id = 1;
```

Output:
```
plan
----
SELECT
  Columns: *
  -> PK Lookup on users
       id = 1
```

### EXPLAIN ANALYZE

Shows the execution plan with actual runtime statistics.

```sql
EXPLAIN ANALYZE SELECT * FROM products WHERE price > 100;
```

Output:
```
plan
----
SELECT (actual time=1.2ms, rows=150)
  Columns: *
  -> Seq Scan on products (actual rows=150)
       Filter: (price > 100)
```

See [EXPLAIN](../sql-features/explain) for detailed documentation.

### ANALYZE

Collects statistics for the query optimizer.

```sql
-- Analyze a specific table
ANALYZE table_name;
```

Statistics are used by the cost-based optimizer to choose efficient query plans.

## Utility Commands

### SHOW TABLES

Lists all tables in the database.

```sql
SHOW TABLES;
```

### SHOW INDEXES

Lists all indexes for a table.

```sql
SHOW INDEXES FROM table_name;
```

### SHOW CREATE TABLE

Shows the CREATE TABLE statement for a table.

```sql
SHOW CREATE TABLE table_name;
```

### PRAGMA

Sets or gets configuration options.

```sql
-- Set a value
PRAGMA name = value;

-- Get current value
PRAGMA name;
```

#### Supported PRAGMAs

| PRAGMA | Description | Default |
|--------|-------------|---------|
| sync_mode | WAL sync mode (0=None, 1=Normal, 2=Full) | 1 |
| snapshot_interval | Snapshot interval in seconds | 300 |
| keep_snapshots | Number of snapshots to retain | 3 |
| wal_flush_trigger | Operations before WAL flush | 1000 |
| create_snapshot | Manually create a snapshot | - |

See [PRAGMA Commands](pragma-commands) for detailed documentation.

## Parameter Binding

Use `$N` positional placeholders for parameter binding:

```sql
SELECT * FROM users WHERE id = $1;
INSERT INTO products (name, price) VALUES ($1, $2);
UPDATE orders SET status = $1 WHERE id = $2;
```

See [Parameter Binding](../sql-features/parameter-binding) for detailed documentation.

## Notes

1. **Transactions**: Stoolap provides MVCC-based transactions for concurrent operations
2. **NULL Handling**: Follows standard SQL NULL semantics; use IS NULL or IS NOT NULL for testing
3. **Type Conversion**: Explicit CAST is recommended for clarity
4. **Case Sensitivity**: SQL keywords are case-insensitive; identifiers are case-sensitive
