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

The generated statement includes column types, PRIMARY KEY, NOT NULL, UNIQUE, AUTO_INCREMENT, DEFAULT, and CHECK constraints.

## SHOW CREATE VIEW

Displays the CREATE VIEW statement that defines a view.

```sql
SHOW CREATE VIEW active_products;
```

Output columns:

| Column | Description |
|--------|-------------|
| **View** | The view name |
| **Create View** | Full CREATE VIEW statement with the original query |

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

Multi-column indexes show column names in parentheses: `(col1, col2, col3)`.

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
