---
title: SQL Functions Reference
category: Functions
order: 1
---

# SQL Functions Reference

This document provides a comprehensive reference for the SQL functions supported by Stoolap, categorized by function type.

## Aggregate Functions

Aggregate functions perform a calculation on a set of values and return a single value.

### AVG

Calculates the average of a numeric column.

```sql
SELECT AVG(price) FROM products;
```

### COUNT

Counts the number of rows or non-NULL values.

```sql
-- Count all rows
SELECT COUNT(*) FROM users;

-- Count non-NULL values in a column
SELECT COUNT(email) FROM users;
```

### FIRST

Returns the first value in a group.

```sql
SELECT category, FIRST(name) FROM products GROUP BY category;
```

### LAST

Returns the last value in a group.

```sql
SELECT category, LAST(name) FROM products GROUP BY category;
```

### MAX

Returns the maximum value from a column.

```sql
SELECT MAX(price) FROM products;
```

### MIN

Returns the minimum value from a column.

```sql
SELECT MIN(price) FROM products;
```

### SUM

Calculates the sum of values in a numeric column.

```sql
SELECT SUM(quantity * price) FROM order_items;
```

## Scalar Functions

Scalar functions operate on a single value and return a single value.

### String Functions

#### CONCAT

Concatenates two or more strings.

```sql
SELECT CONCAT(first_name, ' ', last_name) FROM users;
```

#### LENGTH

Returns the length of a string.

```sql
SELECT name, LENGTH(name) FROM products;
```

#### LOWER

Converts a string to lowercase.

```sql
SELECT LOWER(email) FROM users;
```

#### UPPER

Converts a string to uppercase.

```sql
SELECT UPPER(country_code) FROM locations;
```

#### SUBSTRING

Extracts a portion of a string.

```sql
-- Syntax: SUBSTRING(string, start_position, length)
SELECT SUBSTRING(description, 1, 100) FROM products;
```

#### COLLATE

Compares strings using specific collation rules.

```sql
SELECT * FROM users ORDER BY name COLLATE NOCASE;
```

### Numeric Functions

#### ABS

Returns the absolute value of a number.

```sql
SELECT ABS(temperature) FROM weather_data;
```

#### CEILING

Rounds a number up to the nearest integer.

```sql
SELECT CEILING(price) FROM products;
```

#### FLOOR

Rounds a number down to the nearest integer.

```sql
SELECT FLOOR(price) FROM products;
```

#### ROUND

Rounds a number to a specified number of decimal places.

```sql
-- Round to nearest integer
SELECT ROUND(price) FROM products;

-- Round to 2 decimal places
SELECT ROUND(price, 2) FROM products;
```

### Date and Time Functions

#### NOW

Returns the current date and time.

```sql
SELECT NOW();
```

#### DATE_TRUNC

Truncates a timestamp to a specified precision.

```sql
-- Truncate to day (removes time component)
SELECT DATE_TRUNC('day', timestamp) FROM events;

-- Truncate to month
SELECT DATE_TRUNC('month', timestamp) FROM events;
```

#### TIME_TRUNC

Truncates a time or timestamp to a specified precision.

```sql
-- Truncate to hour
SELECT TIME_TRUNC('hour', timestamp) FROM events;

-- Truncate to minute
SELECT TIME_TRUNC('minute', timestamp) FROM events;
```

### Type Conversion Functions

#### CAST

Converts a value from one data type to another.

```sql
-- Convert string to integer
SELECT CAST(value AS INT) FROM data;

-- Convert string to timestamp
SELECT CAST(date_string AS TIMESTAMP) FROM events;
```

### Conditional Functions

#### COALESCE

Returns the first non-NULL value from a list of expressions.

```sql
SELECT COALESCE(preferred_name, first_name, 'Unknown') FROM users;
```

## Window Functions

Window functions perform calculations across a set of rows related to the current row.

### ROW_NUMBER

Assigns a unique sequential integer to each row within a partition.

```sql
SELECT name, department, salary,
       ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as rank
FROM employees;
```

## Advanced Usage

### Function Chaining

Functions can be nested to perform complex operations:

```sql
SELECT ROUND(AVG(price), 2) FROM products;
```

### Functions in WHERE Clauses

Functions can be used in WHERE clauses to filter data:

```sql
SELECT * FROM products WHERE LOWER(name) LIKE '%organic%';
```

### Functions in GROUP BY and HAVING

Functions can be used in GROUP BY and HAVING clauses:

```sql
SELECT DATE_TRUNC('month', order_date) as month, SUM(total) as monthly_sales
FROM orders
GROUP BY DATE_TRUNC('month', order_date)
HAVING SUM(total) > 10000;
```

## Implementation Details

Stoolap's function implementation is modular and extensible:

- **Function Registry** - Central registry of all available functions
- **Type Checking** - Functions validate argument types at parse time
- **Function Categories** - Organized into scalar, aggregate, and window functions
- **Custom Implementations** - Each function has a specialized implementation for performance

Functions are defined in:
- `src/functions/aggregate/` - Aggregate function implementations
- `src/functions/scalar/` - Scalar function implementations
- `src/functions/window/` - Window function implementations
- `src/functions/registry.rs` - Function registration system

## Performance Considerations

- Avoid using functions on indexed columns in WHERE clauses, as this may prevent index usage
- Some functions can be pushed down to the storage layer for better performance
- Window functions may require multiple passes over the data
- Complex function chains may impact query performance
