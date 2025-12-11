---
title: DISTINCT Operations
category: SQL Features
order: 1
---

# DISTINCT Operations

This document explains DISTINCT operations in Stoolap based on the implementation and test files.

## Overview

Stoolap supports the DISTINCT keyword to eliminate duplicate rows from query results. This feature is essential for retrieving unique values or combinations of values from tables.

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

## How DISTINCT Works in Stoolap

Stoolap implements DISTINCT operations through the following mechanism:

1. The query executor detects the DISTINCT keyword during SQL parsing
2. After retrieving the base result set, but before applying ORDER BY or LIMIT, the result is filtered for uniqueness
3. A map tracks unique rows to eliminate duplicates
4. For multiple columns, the uniqueness is based on the combination of all column values

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

1. **Memory Usage**: DISTINCT operations require memory to track unique rows
2. **Large Result Sets**: For very large tables, DISTINCT can be memory-intensive
3. **Column Count**: DISTINCT on multiple columns requires more processing than single columns
4. **Data Cardinality**: Performance depends on the ratio of unique values to total rows

## Best Practices

1. **Be Selective**: Only use DISTINCT when you actually need to remove duplicates
2. **Consider Alternatives**: For counting unique values, unique indexes may be more efficient
3. **Column Order**: For multi-column DISTINCT, put highest-cardinality columns first if you're also using ORDER BY
4. **Index Usage**: Properly indexed columns can improve DISTINCT operations

## Implementation Details

Internally, Stoolap:

1. Creates a map to track unique rows
2. Generates a unique key for each row based on its values
3. Only passes through rows that haven't been seen before
4. Optimizes COUNT(DISTINCT) operations with specialized processing

These optimizations ensure that DISTINCT operations are both correct and reasonably efficient for most use cases.