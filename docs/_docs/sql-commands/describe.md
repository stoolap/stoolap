---
layout: doc
title: DESCRIBE
category: SQL Commands
order: 3
---

# DESCRIBE

DESCRIBE displays the structure of a table, including column names, types, nullability, key constraints, and default values.

## Syntax

```sql
DESCRIBE table_name;
DESC table_name;
DESCRIBE TABLE table_name;
```

All three forms are equivalent. `DESC` is shorthand for `DESCRIBE`.

## Output Columns

| Column | Description | Examples |
|--------|-------------|----------|
| **Field** | Column name | `id`, `name`, `price` |
| **Type** | Data type, as CREATE TABLE takes it | `INTEGER`, `TEXT`, `FLOAT`, `BOOLEAN`, `TIMESTAMP`, `JSON`, `VECTOR(3)` |
| **Null** | Whether NULL is allowed | `YES` or `NO` |
| **Key** | Constraint type | `PRI` for PRIMARY KEY, `UNI` for a single-column UNIQUE, `MUL` for a column of a multi-column index or a FOREIGN KEY, empty otherwise |
| **Default** | Default value expression | `'unnamed'`, `0`, empty if none |
| **Extra** | `auto_increment` and the column's CHECK, joined by `, ` | `auto_increment`, `CHECK (age >= 0)` |

A CHECK written as a table constraint, such as `CHECK (a < b)` after the columns, belongs to no one column, so DESCRIBE does not show it; [SHOW CREATE TABLE]({% link _docs/sql-commands/show-commands.md %}) does. A FOREIGN KEY shows only as `MUL`; the table and column it references are in SHOW CREATE TABLE.

## Example

```sql
CREATE TABLE products (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    name TEXT NOT NULL,
    price FLOAT CHECK (price >= 0),
    active BOOLEAN DEFAULT TRUE
);

DESCRIBE products;
```

Result:

| Field | Type | Null | Key | Default | Extra |
|-------|------|------|-----|---------|-------|
| id | INTEGER | NO | PRI | | auto_increment |
| name | TEXT | NO | | | |
| price | FLOAT | YES | | | CHECK (price >= 0) |
| active | BOOLEAN | YES | | TRUE | |
