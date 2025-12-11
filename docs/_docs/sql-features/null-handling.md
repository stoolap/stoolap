---
title: NULL Handling
category: SQL Features
order: 1
---

# NULL Handling

This document explains how NULL values are handled in Stoolap based on test files and implementation details.

## Overview

Stoolap follows standard SQL semantics for NULL values, representing the absence of a value. NULL is distinct from zero, empty string, or any other value.

## NULL Behavior in SQL

In Stoolap, NULL follows these standard SQL behaviors:

- NULL is not equal to any value, including another NULL
- Comparisons with NULL generally yield NULL, not TRUE or FALSE
- NULL requires special operators (IS NULL, IS NOT NULL) for testing
- Functions and expressions propagate NULL values
- Aggregate functions generally ignore NULL values

## Column NULL Constraints

Columns in Stoolap can be defined as nullable (the default) or NOT NULL:

```sql
-- Create a table with both nullable and NOT NULL columns
CREATE TABLE users (
    id INTEGER PRIMARY KEY,  -- Primary key implies NOT NULL
    name TEXT NOT NULL,      -- Can never be NULL
    email TEXT,              -- Can be NULL (default)
    age INTEGER              -- Can be NULL (default)
);
```

## Testing for NULL Values

To test for NULL values, use the IS NULL and IS NOT NULL operators:

```sql
-- Find rows with NULL values
SELECT * FROM users WHERE email IS NULL;

-- Find rows without NULL values
SELECT * FROM users WHERE email IS NOT NULL;
```

These operators are properly optimized and can use indexes.

## Example

```sql
-- Create a test table
CREATE TABLE test_null (
    id INTEGER PRIMARY KEY,
    text_val TEXT,
    int_val INTEGER,
    float_val FLOAT,
    bool_val BOOLEAN
);

-- Insert data with NULL values
INSERT INTO test_null VALUES (1, 'Text', 10, 3.14, TRUE);
INSERT INTO test_null VALUES (2, NULL, NULL, NULL, NULL);
INSERT INTO test_null VALUES (3, 'Other', NULL, 2.71, FALSE);

-- Query with IS NULL
SELECT id FROM test_null WHERE text_val IS NULL;  -- Returns 2

-- Query with IS NOT NULL
SELECT id FROM test_null WHERE int_val IS NOT NULL;  -- Returns 1

-- Multiple NULL conditions
SELECT id FROM test_null 
WHERE text_val IS NULL AND int_val IS NULL;  -- Returns 2

-- Mix of NULL and regular conditions
SELECT id FROM test_null 
WHERE float_val IS NOT NULL AND bool_val = FALSE;  -- Returns 3
```

## NULL in Indexes

Stoolap supports indexing columns that contain NULL values:

```sql
-- Create a table with nullable columns
CREATE TABLE test_index_null (
    id INTEGER PRIMARY KEY,
    category TEXT,
    value INTEGER
);

-- Create an index on a nullable column
CREATE INDEX idx_category ON test_index_null(category);
CREATE INDEX idx_value ON test_index_null(value);

-- Insert data with NULL values
INSERT INTO test_index_null VALUES (1, 'A', 10);
INSERT INTO test_index_null VALUES (2, NULL, 20);
INSERT INTO test_index_null VALUES (3, 'B', NULL);
INSERT INTO test_index_null VALUES (4, NULL, NULL);

-- Query with IS NULL using index
SELECT id FROM test_index_null WHERE category IS NULL;  -- Returns 2, 4
```

The tests confirm that IS NULL and IS NOT NULL conditions can use indexes for efficient filtering.

## NULL in Expressions

NULL values propagate through expressions according to standard SQL rules:

```sql
-- Any operation with NULL generally yields NULL
SELECT 1 + NULL;             -- Result: NULL
SELECT 'text' || NULL;       -- Result: NULL
SELECT column1 = NULL;       -- Result: NULL (not FALSE!)

-- Exceptions for logical operators
SELECT TRUE OR NULL;         -- Result: TRUE
SELECT FALSE AND NULL;       -- Result: FALSE
```

## NULL in Joins

NULL values in join columns affect the matching behavior:

```sql
-- Inner join (NULL doesn't match anything)
SELECT a.id, b.id 
FROM table_a a
INNER JOIN table_b b ON a.value = b.value;
-- Rows with NULL in value don't match

-- Left join (preserves all rows from left table)
SELECT a.id, b.id 
FROM table_a a
LEFT JOIN table_b b ON a.value = b.value;
-- Rows with NULL in value appear with NULL for b columns
```

## NULL in GROUP BY and DISTINCT

NULL is considered a single value for grouping and distinct operations:

```sql
-- NULLs are grouped together
SELECT category, COUNT(*) 
FROM products 
GROUP BY category;
-- All NULL categories form a single group

-- NULLs count as one distinct value
SELECT COUNT(DISTINCT category) FROM products;
-- Counts NULL as one distinct value if present
```

## NULL in Aggregation Functions

NULL handling in aggregate functions:

```sql
-- COUNT(*) counts all rows regardless of NULL
SELECT COUNT(*) FROM users;  -- Counts all rows

-- COUNT(column) skips NULL values
SELECT COUNT(email) FROM users;  -- Counts only non-NULL emails

-- Other aggregates (SUM, AVG, MAX, MIN) ignore NULL values
SELECT AVG(age) FROM users;  -- Average of non-NULL ages
```

## COALESCE Function

To provide default values for NULL, use the COALESCE function:

```sql
-- Return the first non-NULL value
SELECT COALESCE(email, 'No Email') FROM users;

-- Can check multiple values in order
SELECT COALESCE(preferred_name, first_name, 'Anonymous') FROM users;
```

## Implementation Details

From the test files and code inspection:

- NULL values are represented distinctly in the storage engine
- The expression evaluator handles NULL propagation
- Index structures store and retrieve NULL values efficiently
- IS NULL and IS NOT NULL operators are optimized for performance
- Comparisons with NULL follow three-valued logic (TRUE, FALSE, NULL)