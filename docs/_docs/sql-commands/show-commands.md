---
layout: doc
title: SHOW Commands
category: SQL Commands
order: 5
---

# SHOW Commands

SHOW commands display metadata about the database schema, including tables, views, indexes, and their definitions.

## SHOW TABLES

Lists all tables in the database.

```sql
SHOW TABLES;
```

Output column:

| Column | Description |
|--------|-------------|
| **table_name** | Name of each table |

## SHOW VIEWS

Lists all views in the database.

```sql
SHOW VIEWS;
```

Output column:

| Column | Description |
|--------|-------------|
| **view_name** | Name of each view |

## SHOW CREATE TABLE

Displays the CREATE TABLE statement that would recreate a table.

```sql
SHOW CREATE TABLE products;
```

Output columns:

| Column | Description |
|--------|-------------|
| **Table** | The table name |
| **Create Table** | Full CREATE TABLE statement |

The generated statement includes column types, written in capitals as `INTEGER`, `TEXT` or `VECTOR(3)`, PRIMARY KEY, NOT NULL, UNIQUE, AUTO_INCREMENT, DEFAULT, CHECK, FOREIGN KEY and CLUSTER BY. A unique index over several columns, including the one a composite `PRIMARY KEY (a, b)` is kept by, is printed as a table constraint `UNIQUE (a, b)` beside the NOT NULL of its columns. A CHECK condition is printed in one pair of parentheses: `CHECK (age >= 0)`.

Every name is printed so the statement parses again: a name holding a character a bare name cannot, such as a space or a `"`, or spelling a reserved keyword such as `order` or `select`, is double-quoted, with an embedded `"` doubled. Running the output again creates a table that refuses the same rows.

## SHOW CREATE VIEW

Displays the CREATE VIEW statement that defines a view.

```sql
SHOW CREATE VIEW active_products;
```

Output columns:

| Column | Description |
|--------|-------------|
| **View** | The view name |
| **Create View** | Full CREATE VIEW statement with the original query; the view name and every name in the query are quoted as in SHOW CREATE TABLE |

## SHOW INDEXES

Lists all indexes on a specified table.

```sql
SHOW INDEXES FROM products;
```

Output columns:

| Column | Description | Examples |
|--------|-------------|----------|
| **table_name** | The table name | `products` |
| **index_name** | The index name | `idx_price`, `pk_products` |
| **column_name** | Indexed column(s) | `price`, `(customer_id, order_date)` |
| **index_type** | Index type | `BTREE`, `HASH`, `BITMAP`, `MULTICOLUMN`, `PRIMARYKEY` |
| **is_unique** | Whether the index enforces uniqueness | `true`, `false` |
| **options** | An HNSW index's parameters, empty for other indexes | `metric=l2, m=8, ef_construction=64, ef_search=99` |

Multi-column indexes show column names in parentheses: `(col1, col2, col3)`.

## DESCRIBE

`DESCRIBE products` lists each column with its `Type`, written as CREATE TABLE takes it: `INTEGER`, or a VECTOR column with its dimension, `VECTOR(3)`. Its `Extra` shows `auto_increment` and the column's CHECK, as [DESCRIBE]({% link _docs/sql-commands/describe.md %}) lists.

## Example

```sql
CREATE TABLE orders (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    customer_id INTEGER NOT NULL,
    amount FLOAT DEFAULT 0,
    status TEXT CHECK (status IN ('pending', 'shipped', 'delivered'))
);

CREATE INDEX idx_customer ON orders(customer_id);
CREATE INDEX idx_status ON orders(status);

-- View the table structure
SHOW CREATE TABLE orders;

-- List all indexes
SHOW INDEXES FROM orders;
```
