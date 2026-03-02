---
layout: doc
title: Table-Valued Functions
category: Functions
order: 5
---

# Table-Valued Functions

Table-valued functions (TVFs) are functions that return a set of rows rather than a single value. They are used in the `FROM` clause of a query, just like a regular table. Stoolap supports TVFs with full SQL integration: WHERE filtering, ORDER BY, LIMIT, JOINs, subqueries, CTEs, and aggregation all work seamlessly.

## GENERATE_SERIES

Generates a series of values from `start` to `stop` (inclusive) with an optional `step`. Supports integer, float, and timestamp/date types.

### Syntax

```sql
GENERATE_SERIES(start, stop)
GENERATE_SERIES(start, stop, step)
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `start` | The first value in the series |
| `stop` | The last value in the series (inclusive) |
| `step` | The increment between values (optional, auto-detected if omitted) |

**Return value:** A table with a single column named `value`.

### Integer Series

Generate a sequence of integers.

```sql
-- Basic ascending series
SELECT * FROM generate_series(1, 5);
-- Returns: 1, 2, 3, 4, 5

-- With explicit step
SELECT * FROM generate_series(0, 10, 2);
-- Returns: 0, 2, 4, 6, 8, 10

-- Descending with negative step
SELECT * FROM generate_series(5, 1, -1);
-- Returns: 5, 4, 3, 2, 1

-- Auto-detect descending (no step needed)
SELECT * FROM generate_series(5, 1);
-- Returns: 5, 4, 3, 2, 1

-- Negative range
SELECT * FROM generate_series(-3, 3);
-- Returns: -3, -2, -1, 0, 1, 2, 3

-- Single value (start equals stop)
SELECT * FROM generate_series(3, 3);
-- Returns: 3
```

### Float Series

When any argument is a float, the function generates floating-point values. Index-based generation is used internally to avoid floating-point drift.

```sql
-- Float series with fractional step
SELECT * FROM generate_series(0.0, 1.0, 0.5);
-- Returns: 0.0, 0.5, 1.0

-- Mixed integer and float arguments
SELECT * FROM generate_series(0, 2, 0.5);
-- Returns: 0.0, 0.5, 1.0, 1.5, 2.0
```

### Timestamp/Date Series

Generate a sequence of timestamps or dates. The `start` and `stop` values can be date strings (`'YYYY-MM-DD'`) or timestamp strings (`'YYYY-MM-DD HH:MM:SS'`). The `step` is an interval string.

**Supported interval units:**

| Unit | Examples |
|------|----------|
| year/years | `'1 year'`, `'2 years'` |
| month/months | `'1 month'`, `'3 months'` |
| week/weeks | `'1 week'`, `'2 weeks'` |
| day/days | `'1 day'`, `'7 days'` |
| hour/hours | `'1 hour'`, `'6 hours'` |
| minute/minutes/min | `'1 minute'`, `'30 min'` |
| second/seconds/sec | `'1 second'`, `'15 sec'` |
| millisecond/milliseconds/ms | `'100 ms'` |
| microsecond/microseconds/us | `'500 us'` |

```sql
-- Daily series
SELECT * FROM generate_series('2024-01-01', '2024-01-05', '1 day');
-- Returns: 2024-01-01, 2024-01-02, 2024-01-03, 2024-01-04, 2024-01-05

-- Auto-detect daily step
SELECT * FROM generate_series('2024-01-01', '2024-01-03');
-- Returns: 2024-01-01, 2024-01-02, 2024-01-03

-- Hourly series
SELECT * FROM generate_series(
    '2024-01-01 00:00:00',
    '2024-01-01 06:00:00',
    '2 hours'
);
-- Returns: 00:00, 02:00, 04:00, 06:00

-- Every 10 minutes
SELECT * FROM generate_series(
    '2024-01-01 00:00:00',
    '2024-01-01 00:30:00',
    '10 minutes'
);
-- Returns: 00:00, 00:10, 00:20, 00:30

-- Weekly series
SELECT * FROM generate_series('2024-01-01', '2024-01-29', '1 week');
-- Returns: Jan 1, Jan 8, Jan 15, Jan 22, Jan 29

-- Monthly series (1 month = 30 days)
SELECT * FROM generate_series('2024-01-01', '2024-04-01', '1 month');

-- Descending date series
SELECT * FROM generate_series('2024-01-05', '2024-01-01', '-1 day');
-- Returns: Jan 5, Jan 4, Jan 3, Jan 2, Jan 1
```

### Column Aliases

The default output column is named `value`. You can rename it using standard SQL aliasing.

```sql
-- Table alias with column alias
SELECT n FROM generate_series(1, 5) AS gs(n);

-- Implicit alias (without AS keyword)
SELECT n FROM generate_series(1, 5) gs(n);

-- Using default column name
SELECT value FROM generate_series(1, 5);
```

### Scalar Mode

When used in a `SELECT` expression (without `FROM`), `GENERATE_SERIES` returns a JSON array string. This matches DuckDB behavior.

```sql
SELECT generate_series(1, 5);
-- Returns: '[1, 2, 3, 4, 5]'

SELECT generate_series(0, 10, 2);
-- Returns: '[0, 2, 4, 6, 8, 10]'

SELECT generate_series('2024-01-01', '2024-01-03', '1 day');
-- Returns: '["2024-01-01T00:00:00+00:00", "2024-01-02T00:00:00+00:00", "2024-01-03T00:00:00+00:00"]'
```

## Using with SQL Clauses

### WHERE

```sql
SELECT * FROM generate_series(1, 10) AS g(value)
WHERE value > 7;
-- Returns: 8, 9, 10

SELECT * FROM generate_series('2024-01-01', '2024-01-10', '1 day') AS g(value)
WHERE value > '2024-01-07';
-- Returns: Jan 8, Jan 9, Jan 10
```

### ORDER BY

```sql
SELECT * FROM generate_series(1, 5) AS g(value)
ORDER BY value DESC;
-- Returns: 5, 4, 3, 2, 1
```

### LIMIT and OFFSET

```sql
SELECT * FROM generate_series(1, 100) AS g(value)
LIMIT 5;
-- Returns: 1, 2, 3, 4, 5

-- LIMIT with OFFSET using CTE
WITH gs AS (
    SELECT * FROM generate_series(1, 10) AS g(value)
)
SELECT * FROM gs LIMIT 3 OFFSET 2;
-- Returns: 3, 4, 5
```

### Aggregation

```sql
-- Sum of 1 to 100
SELECT SUM(value) FROM generate_series(1, 100) AS g(value);
-- Returns: 5050

-- Count
SELECT COUNT(*) FROM generate_series(1, 1000) AS g(value);
-- Returns: 1000

-- Count days in a year
SELECT COUNT(*) FROM generate_series('2024-01-01', '2024-12-31', '1 day') AS g(value);
-- Returns: 366 (2024 is a leap year)
```

### JOINs

```sql
-- Join with a regular table
SELECT g.n, t.name
FROM generate_series(1, 3) AS g(n)
JOIN users t ON g.n = t.id
ORDER BY g.n;

-- Cross join two series (generates a grid)
SELECT a.value AS x, b.value AS y
FROM generate_series(1, 3) AS a(value)
CROSS JOIN generate_series(1, 2) AS b(value)
ORDER BY x, y;
-- Returns: (1,1), (1,2), (2,1), (2,2), (3,1), (3,2)
```

### Subqueries and CTEs

```sql
-- In a subquery
SELECT * FROM (
    SELECT * FROM generate_series(1, 5) AS g(n)
) sub
ORDER BY n;

-- In a CTE
WITH numbers AS (
    SELECT value FROM generate_series(1, 10) AS g(value)
)
SELECT value, value * value AS squared
FROM numbers
WHERE value <= 5;
```

## Practical Examples

### Generate a Calendar Table

```sql
SELECT value AS date
FROM generate_series('2024-01-01', '2024-12-31', '1 day') AS g(value)
ORDER BY date;
```

### Generate Hourly Time Slots

```sql
SELECT value AS slot
FROM generate_series(
    '2024-01-01 08:00:00',
    '2024-01-01 17:00:00',
    '1 hour'
) AS g(value);
```

### Fill Gaps in Time Series Data

```sql
-- Generate all dates, then left join with actual data
WITH dates AS (
    SELECT value AS date
    FROM generate_series('2024-01-01', '2024-01-31', '1 day') AS g(value)
)
SELECT d.date, COALESCE(o.total, 0) AS total
FROM dates d
LEFT JOIN daily_orders o ON d.date = o.order_date
ORDER BY d.date;
```

### Number Table for Testing

```sql
-- Generate test IDs
INSERT INTO test_data (id, value)
SELECT value, value * 10
FROM generate_series(1, 100) AS g(value);
```

### Multiplication Table

```sql
SELECT a.value AS x, b.value AS y, a.value * b.value AS product
FROM generate_series(1, 10) AS a(value)
CROSS JOIN generate_series(1, 10) AS b(value)
ORDER BY x, y;
```

## Behavior Notes

- **Inclusive bounds**: Both `start` and `stop` are included in the output when the step aligns.
- **Auto-detect direction**: When `step` is omitted, the function automatically determines the direction. For integers and floats, it uses `+1` or `-1`. For timestamps, it uses `+1 day` or `-1 day`.
- **Direction mismatch**: If the step goes in the opposite direction from start to stop (e.g., `generate_series(1, 5, -1)`), an empty result is returned. This follows PostgreSQL behavior.
- **Zero step**: A step of zero is an error.
- **Safety limit**: A maximum of 10,000,000 rows can be generated per call to prevent out-of-memory conditions.
- **Case insensitive**: `GENERATE_SERIES`, `generate_series`, and `Generate_Series` all work.
- **Month approximation**: The `month` interval unit is approximated as 30 days. For exact calendar month arithmetic, use application-level logic.
