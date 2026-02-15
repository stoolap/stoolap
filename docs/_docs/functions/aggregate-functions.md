---
layout: doc
title: Aggregate Functions
category: Functions
order: 3
---

# Aggregate Functions

Stoolap supports standard SQL aggregate functions that operate on multiple rows to calculate a single result. These functions are typically used with GROUP BY clauses for summarizing data.

## Basic Aggregate Functions

### COUNT

Counts the number of rows or non-NULL values:

```sql
-- Count all rows
SELECT COUNT(*) FROM orders;

-- Count non-NULL values in a column
SELECT COUNT(status) FROM orders;

-- Count distinct values
SELECT COUNT(DISTINCT category) FROM products;
```

COUNT is optimized to use available indexes when possible.

### SUM

Calculates the sum of numeric values:

```sql
SELECT SUM(amount) FROM orders;

-- With grouping
SELECT category, SUM(amount) FROM orders GROUP BY category;
```

### AVG

Calculates the average (mean) of numeric values:

```sql
SELECT AVG(price) FROM products;

-- With grouping
SELECT category, AVG(price) FROM products GROUP BY category;
```

### MIN

Finds the minimum value:

```sql
SELECT MIN(price) FROM products;

-- Works with strings (alphabetical order)
SELECT MIN(name) FROM products;

-- With grouping
SELECT category, MIN(price) FROM products GROUP BY category;
```

### MAX

Finds the maximum value:

```sql
SELECT MAX(price) FROM products;

-- Works with strings
SELECT MAX(name) FROM products;

-- With grouping
SELECT category, MAX(price) FROM products GROUP BY category;
```

## Statistical Functions

### MEDIAN

Returns the median (middle value) of all non-NULL values. For an even number of values, returns the average of the two middle values.

```sql
SELECT MEDIAN(salary) FROM employees;

-- Odd count: values 1, 3, 5, 7, 9 → median = 5
-- Even count: values 1, 3, 5, 7 → median = (3+5)/2 = 4
```

### STDDEV / STDDEV_SAMP

Calculates the sample standard deviation:

```sql
SELECT STDDEV(value) FROM measurements;
SELECT STDDEV_SAMP(value) FROM measurements;  -- same as STDDEV

-- For values: 10, 20, 30, 40, 50
SELECT STDDEV(value) FROM data;  -- Returns ~15.81
```

### STDDEV_POP

Calculates the population standard deviation (divides by N instead of N-1):

```sql
SELECT STDDEV_POP(value) FROM measurements;
```

### VARIANCE / VAR_SAMP

Calculates the sample variance:

```sql
SELECT VARIANCE(value) FROM measurements;
SELECT VAR_SAMP(value) FROM measurements;  -- same as VARIANCE

-- For values: 10, 20, 30, 40, 50
SELECT VARIANCE(value) FROM data;  -- Returns ~250
```

### VAR_POP

Calculates the population variance (divides by N instead of N-1):

```sql
SELECT VAR_POP(value) FROM measurements;
```

## String and Array Aggregation

### STRING_AGG

Concatenates values into a single string with a separator:

```sql
-- Basic usage
SELECT STRING_AGG(name, ', ') FROM employees;
-- Returns: "Alice, Bob, Charlie, Diana"

-- With grouping
SELECT department, STRING_AGG(name, ', ') as team_members
FROM employees
GROUP BY department;

-- With ORDER BY in subquery for ordered results
SELECT STRING_AGG(name, ' | ') FROM (
    SELECT name FROM employees ORDER BY name
) sub;
```

### GROUP_CONCAT

Alias for STRING_AGG. Concatenates values with a separator:

```sql
SELECT GROUP_CONCAT(name, ', ') FROM employees;
-- Returns: "Alice, Bob, Charlie, Diana"
```

### ARRAY_AGG

Aggregates values into a JSON array:

```sql
-- Basic usage
SELECT ARRAY_AGG(name) FROM employees;
-- Returns: ["Alice", "Bob", "Charlie", "Diana"]

-- With ORDER BY clause
SELECT ARRAY_AGG(name ORDER BY salary DESC) FROM employees;
-- Returns names ordered by salary descending

-- With grouping
SELECT department, ARRAY_AGG(name ORDER BY name) as members
FROM employees
GROUP BY department;
```

## Position Functions

### FIRST

Returns the first value in a group based on row order:

```sql
SELECT FIRST(name) FROM employees;

-- With grouping
SELECT department, FIRST(name) FROM employees GROUP BY department;
```

Note: FIRST depends on the order of rows. Use ORDER BY in a subquery for deterministic results.

### LAST

Returns the last value in a group based on row order:

```sql
SELECT LAST(name) FROM employees;

-- With grouping
SELECT department, LAST(name) FROM employees GROUP BY department;
```

Note: LAST depends on the order of rows. Use ORDER BY in a subquery for deterministic results.

## Using GROUP BY

The GROUP BY clause groups rows with the same values and applies aggregate functions to each group:

```sql
SELECT
    category,
    COUNT(*) as count,
    SUM(amount) as total,
    AVG(amount) as average,
    MIN(amount) as minimum,
    MAX(amount) as maximum
FROM sales
GROUP BY category;
```

Result:
```
category    | count | total   | average | minimum | maximum
------------+-------+---------+---------+---------+--------
Electronics | 3     | 2150.00 | 716.67  | 150.00  | 1200.00
Clothing    | 3     | 145.00  | 48.33   | 25.00   | 70.00
```

## Using HAVING

The HAVING clause filters groups based on aggregate results:

```sql
-- Find categories with more than 2 products and total > 100
SELECT category, COUNT(*) as count, SUM(amount) as total
FROM sales
GROUP BY category
HAVING COUNT(*) > 2 AND SUM(amount) > 100;
```

## Multi-Dimensional Aggregation

### ROLLUP

Creates subtotals rolling up from the most detailed level to a grand total:

```sql
SELECT
    region,
    category,
    SUM(sales) as total_sales
FROM orders
GROUP BY ROLLUP(region, category);
```

Result includes:
- Detail rows for each region/category combination
- Subtotals for each region (category = NULL)
- Grand total (region = NULL, category = NULL)

### CUBE

Creates subtotals for all combinations of grouping columns:

```sql
SELECT
    region,
    category,
    SUM(sales) as total_sales
FROM orders
GROUP BY CUBE(region, category);
```

Result includes:
- Detail rows for each region/category combination
- Subtotals for each region
- Subtotals for each category
- Grand total

### GROUPING SETS

Specifies exactly which grouping combinations to generate:

```sql
SELECT
    region,
    category,
    SUM(sales) as total_sales
FROM orders
GROUP BY GROUPING SETS ((region, category), (region), ());
```

Result includes only the specified groupings:
- Detail rows for each region/category combination
- Subtotals for each region
- Grand total

GROUPING SETS provides full control over which aggregation levels appear in the result.

### GROUPING() Function

The GROUPING() function identifies super-aggregate rows (subtotals and grand totals):

```sql
SELECT
    region,
    category,
    SUM(sales) as total_sales,
    GROUPING(region) as is_region_agg,
    GROUPING(category) as is_category_agg
FROM orders
GROUP BY ROLLUP(region, category);
```

Returns:
- `0` if the column is part of the current grouping (normal grouped row)
- `1` if the column is aggregated (super-aggregate row)

Use GROUPING() to distinguish between actual NULL values and NULLs that indicate subtotal rows.

See [ROLLUP, CUBE, and GROUPING SETS](../sql-features/rollup-cube) for detailed documentation.

## NULL Handling

Stoolap follows standard SQL NULL handling for aggregate functions:

- NULL values are ignored by most aggregate functions (SUM, AVG, MIN, MAX)
- COUNT(*) includes all rows regardless of NULL values
- COUNT(column) only counts non-NULL values
- If all inputs to an aggregate function are NULL, the result is NULL (except COUNT which returns 0)

```sql
-- Example with NULL values
CREATE TABLE test (id INTEGER, value INTEGER);
INSERT INTO test VALUES (1, 10), (2, NULL), (3, 30);

SELECT COUNT(*) FROM test;        -- Returns 3
SELECT COUNT(value) FROM test;    -- Returns 2
SELECT SUM(value) FROM test;      -- Returns 40
SELECT AVG(value) FROM test;      -- Returns 20
```

## Performance Considerations

- COUNT(*) is optimized to use the smallest available index
- COUNT DISTINCT can be expensive for large datasets with many unique values
- Aggregations benefit from indexes on grouped columns
- Use WHERE clauses to reduce input size before aggregation
- ROLLUP and CUBE can produce many result rows for high-cardinality columns

## Complete Example

```sql
-- Create sample data
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer TEXT,
    category TEXT,
    amount FLOAT,
    order_date TEXT
);

INSERT INTO orders VALUES
(1, 'Alice', 'Electronics', 1200.00, '2024-01-15'),
(2, 'Bob', 'Electronics', 800.00, '2024-01-16'),
(3, 'Alice', 'Clothing', 150.00, '2024-01-17'),
(4, 'Charlie', 'Electronics', 450.00, '2024-01-18'),
(5, 'Bob', 'Clothing', 75.00, '2024-01-19');

-- Multiple aggregates in one query
SELECT
    category,
    COUNT(*) as order_count,
    COUNT(DISTINCT customer) as unique_customers,
    SUM(amount) as total_sales,
    AVG(amount) as avg_order,
    MIN(amount) as smallest_order,
    MAX(amount) as largest_order,
    STRING_AGG(customer, ', ') as customers
FROM orders
GROUP BY category
ORDER BY total_sales DESC;
```

Result:
```
category    | order_count | unique_customers | total_sales | avg_order | smallest_order | largest_order | customers
------------+-------------+------------------+-------------+-----------+----------------+---------------+--------------------
Electronics | 3           | 3                | 2450.00     | 816.67    | 450.00         | 1200.00       | Alice, Bob, Charlie
Clothing    | 2           | 2                | 225.00      | 112.50    | 75.00          | 150.00        | Alice, Bob
```
