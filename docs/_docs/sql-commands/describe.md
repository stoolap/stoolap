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
| **Type** | Data type | `Integer`, `Text`, `Float`, `Boolean`, `Timestamp` |
| **Null** | Whether NULL is allowed | `YES` or `NO` |
| **Key** | Constraint type | `PRI` for PRIMARY KEY, empty otherwise |
| **Default** | Default value expression | `'unnamed'`, `0`, empty if none |
| **Extra** | Additional information | Currently reserved |

## Example

```sql
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    price FLOAT,
    active BOOLEAN DEFAULT TRUE
);

DESCRIBE products;
```

Result:

| Field | Type | Null | Key | Default | Extra |
|-------|------|------|-----|---------|-------|
| id | Integer | NO | PRI | | |
| name | Text | NO | | | |
| price | Float | YES | | | |
| active | Boolean | YES | | true | |
