---
layout: doc
title: DISTINCT Operations
category: SQL Features
order: 8
---

# DISTINCT Operations

Stoolap supports the DISTINCT keyword to eliminate duplicate rows from query results, including `DISTINCT ON` for per-group deduplication.

## Overview

Stoolap supports DISTINCT in two forms:
- **DISTINCT**: Eliminates fully duplicate rows from the result set.
- **DISTINCT ON (expr, ...)**: Keeps only the first row per unique combination of the specified expressions, as determined by the ORDER BY clause. Inspired by PostgreSQL's extension, but Stoolap does not require the DISTINCT ON expressions to match the leftmost ORDER BY columns.

## Syntax

Stoolap supports these DISTINCT syntax patterns:

```sql
-- Basic DISTINCT on single column
SELECT DISTINCT column FROM table;

-- DISTINCT on multiple columns
SELECT DISTINCT column1, column2, ... FROM table;

-- DISTINCT with ORDER BY
SELECT DISTINCT column FROM table ORDER BY column;

-- COUNT with DISTINCT
SELECT COUNT(DISTINCT column) FROM table;

-- DISTINCT ON: first row per group
SELECT DISTINCT ON (expr1, expr2, ...) column1, column2, ...
FROM table
ORDER BY expr1, expr2, ..., other_columns;
```

## Examples

### Basic DISTINCT on Single Column

```sql
-- Create a test table
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    price FLOAT
);

-- Insert sample data with duplicate categories
INSERT INTO products (id, name, category, price) VALUES
(1, 'Laptop', 'Electronics', 999.99),
(2, 'Smartphone', 'Electronics', 499.99),
(3, 'Headphones', 'Electronics', 99.99),
(4, 'T-shirt', 'Clothing', 19.99),
(5, 'Jeans', 'Clothing', 49.99),
(6, 'Novel', 'Books', 14.99),
(7, 'Textbook', 'Books', 79.99);

-- Select distinct categories
SELECT DISTINCT category FROM products;

-- Result: 3 rows
-- Electronics
-- Clothing
-- Books
```

### DISTINCT on Multiple Columns

```sql
-- Create a table with region information
CREATE TABLE sales (
    id INTEGER PRIMARY KEY,
    product TEXT,
    category TEXT,
    region TEXT,
    amount FLOAT
);

-- Insert data with duplicate combinations
INSERT INTO sales (id, product, category, region, amount) VALUES
(1, 'Laptop', 'Electronics', 'North', 999.99),
(2, 'Smartphone', 'Electronics', 'South', 499.99),
(3, 'Headphones', 'Electronics', 'North', 99.99),
(4, 'T-shirt', 'Clothing', 'East', 19.99),
(5, 'Jeans', 'Clothing', 'West', 49.99),
(6, 'T-shirt', 'Clothing', 'North', 19.99),
(7, 'Novel', 'Books', 'South', 14.99);

-- Select distinct category and region combinations
SELECT DISTINCT category, region FROM sales;

-- Result: 6 rows (unique combinations)
-- Electronics, North
-- Electronics, South
-- Clothing, East
-- Clothing, West
-- Clothing, North
-- Books, South
```

### DISTINCT with ORDER BY

```sql
-- Select distinct regions ordered alphabetically
SELECT DISTINCT region FROM sales ORDER BY region;

-- Result: 4 rows in alphabetical order
-- East
-- North
-- South
-- West
```

### COUNT with DISTINCT

```sql
-- Count distinct categories
SELECT COUNT(DISTINCT category) FROM sales;

-- Result: 3 (Electronics, Clothing, Books)
```

### DISTINCT ON: First Row Per Group

`DISTINCT ON` keeps only the first row for each unique combination of the specified expressions. The ORDER BY clause determines which row is "first" within each group.

```sql
-- Get the most recent order per customer
SELECT DISTINCT ON (customer) customer, amount, order_date
FROM orders
ORDER BY customer, order_date DESC;

-- Result: one row per customer, each with their latest order
-- Alice   200.00   2024-02-01
-- Bob     250.00   2024-02-05
-- Charlie 225.00   2024-03-01
```

### DISTINCT ON with Multiple Keys

```sql
-- First sale per (region, category) combination
SELECT DISTINCT ON (region, category) region, category, amount
FROM sales
ORDER BY region, category, sale_date;
```

### DISTINCT ON with LIMIT

DISTINCT ON composes naturally with LIMIT and OFFSET. Deduplication happens first, then LIMIT is applied to the deduplicated result.

```sql
-- Top 5 customers by their highest order amount
SELECT DISTINCT ON (customer) customer, amount
FROM orders
ORDER BY customer, amount DESC
LIMIT 5;
```

### DISTINCT ON without ORDER BY

When no ORDER BY is specified, Stoolap automatically sorts by the DISTINCT ON columns to ensure correct deduplication. However, which row is kept per group is non-deterministic in this case.

```sql
-- One row per customer (arbitrary selection within each group)
SELECT DISTINCT ON (customer) customer, amount
FROM orders;
```

## How DISTINCT Works in Stoolap

Stoolap implements DISTINCT operations through the following mechanism:

1. The query executor detects the DISTINCT keyword during SQL parsing
2. After retrieving the base result set, but before applying ORDER BY or LIMIT, the result is filtered for uniqueness
3. A map tracks unique rows to eliminate duplicates
4. For multiple columns, the uniqueness is based on the combination of all column values

### How DISTINCT ON Works

DISTINCT ON follows a different pipeline order than regular DISTINCT:

1. The base result set is computed (with WHERE filtering)
2. ORDER BY sorts the full result set
3. DISTINCT ON uses a hash-based filter to keep only the first row seen per unique key combination
4. LIMIT/OFFSET is applied to the deduplicated result

The hash-based approach uses O(groups) memory, where "groups" is the number of distinct key combinations. This correctly handles all ORDER BY patterns, including cases where the DISTINCT ON columns are not the leading sort keys.

## DISTINCT with NULL Values

NULL values are considered distinct values in DISTINCT operations:

```sql
-- Create a table with NULL values
CREATE TABLE null_test (
    id INTEGER PRIMARY KEY,
    value TEXT
);

-- Insert data with NULLs
INSERT INTO null_test (id, value) VALUES
(1, 'A'),
(2, 'B'),
(3, NULL),
(4, 'A'),
(5, NULL);

-- Select distinct values
SELECT DISTINCT value FROM null_test;

-- Result: 3 rows
-- A
-- B
-- NULL
```

## Performance Considerations

Based on the implementation:

1. **Memory Usage**: Regular DISTINCT requires memory proportional to the number of unique rows. DISTINCT ON uses memory proportional to the number of unique groups (the distinct key combinations), which is typically much smaller than the full result set
2. **Large Result Sets**: For very large tables, regular DISTINCT can be memory-intensive. DISTINCT ON is more efficient when you only need one representative row per group
3. **Column Count**: DISTINCT on multiple columns requires more processing than single columns
4. **Data Cardinality**: Performance depends on the ratio of unique values to total rows
5. **DISTINCT ON vs GROUP BY**: For "first/last row per group" queries, DISTINCT ON avoids the overhead of aggregation functions and is typically simpler and faster than equivalent GROUP BY + window function workarounds

## Best Practices

1. **Be Selective**: Only use DISTINCT when you actually need to remove duplicates
2. **Consider Alternatives**: For counting unique values, unique indexes may be more efficient
3. **Column Order**: For multi-column DISTINCT, put highest-cardinality columns first if you're also using ORDER BY
4. **Index Usage**: Properly indexed columns can improve DISTINCT operations
5. **Use DISTINCT ON for "first per group"**: When you need one row per group (e.g., latest order per customer), prefer `DISTINCT ON` over `GROUP BY` with `MIN`/`MAX` or window functions. It is both simpler and more efficient
6. **ORDER BY with DISTINCT ON**: Always specify ORDER BY when using DISTINCT ON to control which row is selected per group. The DISTINCT ON columns should appear at the beginning of the ORDER BY clause

## Implementation Details

Internally, Stoolap:

1. Creates a map to track unique rows (regular DISTINCT)
2. Generates a unique key for each row based on its values
3. Only passes through rows that haven't been seen before
4. Optimizes COUNT(DISTINCT) operations with specialized processing
5. For DISTINCT ON, uses a hash-based streaming filter that tracks seen key groups, requiring O(groups) memory. This approach correctly handles any ORDER BY pattern, including non-leading sort orders and computed key expressions