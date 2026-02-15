---
layout: doc
title: Window Functions
category: Functions
order: 4
---

# Window Functions

Window functions perform calculations across a set of table rows that are related to the current row. Unlike regular aggregate functions, window functions do not cause rows to become grouped into a single output row - the rows retain their separate identities.

## Syntax

```sql
function_name([expression]) OVER (
    [PARTITION BY partition_expression [, ...]]
    [ORDER BY sort_expression [ASC | DESC] [, ...]]
    [frame_clause]
)
```

- **PARTITION BY**: Divides rows into groups (partitions) that share the same values
- **ORDER BY**: Defines the order of rows within each partition
- **frame_clause**: Defines which rows within the partition to include (see [Window Frames](#window-frames))

## Available Window Functions

### Ranking Functions

#### ROW_NUMBER()
Assigns a unique sequential number to each row within a partition.

```sql
SELECT name, dept, salary,
       ROW_NUMBER() OVER (ORDER BY salary DESC) as row_num
FROM employees;
```

Result:
```
name   | dept        | salary | row_num
-------+-------------+--------+--------
Diana  | Engineering | 80000  | 1
Frank  | Engineering | 75000  | 2
Charlie| Engineering | 70000  | 3
Bob    | Sales       | 60000  | 4
Eve    | Sales       | 55000  | 5
Alice  | Sales       | 50000  | 6
```

#### RANK()
Assigns a rank to each row within a partition. Rows with equal values receive the same rank, with gaps in the sequence.

```sql
SELECT name, salary,
       RANK() OVER (ORDER BY salary DESC) as rank
FROM employees;
```

If two employees have the same salary and are ranked 2, the next rank would be 4 (not 3).

#### DENSE_RANK()
Similar to RANK(), but without gaps in the ranking sequence.

```sql
SELECT name, salary,
       DENSE_RANK() OVER (ORDER BY salary DESC) as dense_rank
FROM employees;
```

If two employees have the same salary and are ranked 2, the next rank would be 3.

#### NTILE(n)
Distributes rows into a specified number of buckets.

```sql
SELECT name, salary,
       NTILE(3) OVER (ORDER BY salary DESC) as tertile
FROM employees;
```

Divides employees into 3 groups based on salary.

#### PERCENT_RANK()
Returns the relative rank of the current row as a percentage (0 to 1).

```sql
SELECT name, salary,
       PERCENT_RANK() OVER (ORDER BY salary) as pct_rank
FROM employees;
```

#### CUME_DIST()
Returns the cumulative distribution of a value (fraction of rows with values <= current row).

```sql
SELECT name, salary,
       CUME_DIST() OVER (ORDER BY salary) as cume_dist
FROM employees;
```

### Value Functions

#### LAG(expression [, offset [, default]])
Accesses a value from a previous row within the partition.

```sql
-- Get previous row's salary
SELECT name, salary,
       LAG(salary) OVER (ORDER BY salary) as prev_salary
FROM employees;

-- Get salary from 2 rows back, with default of 0
SELECT name, salary,
       LAG(salary, 2, 0) OVER (ORDER BY salary) as prev2_salary
FROM employees;
```

#### LEAD(expression [, offset [, default]])
Accesses a value from a following row within the partition.

```sql
-- Get next row's salary
SELECT name, salary,
       LEAD(salary) OVER (ORDER BY salary) as next_salary
FROM employees;

-- Get salary from 2 rows ahead, with default of 0
SELECT name, salary,
       LEAD(salary, 2, 0) OVER (ORDER BY salary) as next2_salary
FROM employees;
```

#### FIRST_VALUE(expression)
Returns the first value within the partition.

```sql
SELECT name, dept, salary,
       FIRST_VALUE(salary) OVER (PARTITION BY dept ORDER BY salary DESC) as max_in_dept
FROM employees;
```

#### LAST_VALUE(expression)
Returns the last value within the current window frame.

```sql
SELECT name, dept, salary,
       LAST_VALUE(salary) OVER (PARTITION BY dept ORDER BY salary DESC) as current_last
FROM employees;
```

#### NTH_VALUE(expression, n)
Returns the nth value within the partition.

```sql
SELECT name, salary,
       NTH_VALUE(salary, 2) OVER (ORDER BY salary DESC) as second_highest
FROM employees;
```

## Using PARTITION BY

PARTITION BY divides the result set into partitions, and window functions are applied to each partition separately.

```sql
-- Row numbers within each department
SELECT name, dept, salary,
       ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) as dept_rank
FROM employees
ORDER BY dept, dept_rank;
```

Result:
```
name   | dept        | salary | dept_rank
-------+-------------+--------+----------
Diana  | Engineering | 80000  | 1
Frank  | Engineering | 75000  | 2
Charlie| Engineering | 70000  | 3
Bob    | Sales       | 60000  | 1
Eve    | Sales       | 55000  | 2
Alice  | Sales       | 50000  | 3
```

## Aggregate Functions as Window Functions

Standard aggregate functions can also be used as window functions:

```sql
-- Running total
SELECT name, salary,
       SUM(salary) OVER (ORDER BY salary) as running_total
FROM employees
ORDER BY salary;
```

Result:
```
name   | salary | running_total
-------+--------+--------------
Alice  | 50000  | 50000
Eve    | 55000  | 105000
Bob    | 60000  | 165000
Charlie| 70000  | 235000
Frank  | 75000  | 310000
Diana  | 80000  | 390000
```

## Common Use Cases

### Top N per Group

Find the highest paid employee in each department:

```sql
SELECT * FROM (
    SELECT name, dept, salary,
           ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) as rn
    FROM employees
) ranked
WHERE rn = 1;
```

### Running Totals

Calculate cumulative sales:

```sql
SELECT order_date, amount,
       SUM(amount) OVER (ORDER BY order_date) as cumulative_sales
FROM orders;
```

### Compare to Previous/Next

Calculate month-over-month change:

```sql
SELECT month, revenue,
       revenue - LAG(revenue) OVER (ORDER BY month) as change
FROM monthly_sales;
```

### Percentile Calculation

Find employees in the top 10% of salaries:

```sql
SELECT * FROM (
    SELECT name, salary,
           PERCENT_RANK() OVER (ORDER BY salary DESC) as pct
    FROM employees
) ranked
WHERE pct <= 0.10;
```

## Window Frames

Window frames control exactly which rows within a partition are included in the calculation. There are two frame types: `ROWS` (physical row offsets) and `RANGE` (logical value ranges).

### Syntax

```sql
{ ROWS | RANGE } BETWEEN frame_start AND frame_end
{ ROWS | RANGE } frame_start
```

### Frame Bounds

| Bound | Description |
|-------|-------------|
| `UNBOUNDED PRECEDING` | Start of the partition |
| `n PRECEDING` | n rows/values before the current row |
| `CURRENT ROW` | The current row |
| `n FOLLOWING` | n rows/values after the current row |
| `UNBOUNDED FOLLOWING` | End of the partition |

### Default Frame

When ORDER BY is specified without an explicit frame, the default is:

```sql
RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
```

When ORDER BY is omitted, the entire partition is used.

### Examples

```sql
-- Running total (default frame with ORDER BY)
SELECT name, salary,
       SUM(salary) OVER (ORDER BY salary) as running_total
FROM employees;

-- 3-row moving average
SELECT name, salary,
       AVG(salary) OVER (
           ORDER BY salary
           ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
       ) as moving_avg
FROM employees;

-- Cumulative sum across entire partition
SELECT name, dept, salary,
       SUM(salary) OVER (
           PARTITION BY dept
           ORDER BY salary
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ) as dept_running_total
FROM employees;

-- Total of current and all following rows
SELECT name, salary,
       SUM(salary) OVER (
           ORDER BY salary
           ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
       ) as remaining_total
FROM employees;
```

### ROWS vs RANGE

- **ROWS**: Counts physical rows. `1 PRECEDING` means exactly one row before.
- **RANGE**: Uses logical value ranges. `1 PRECEDING` means all rows whose ORDER BY value is within 1 of the current row's value.

```sql
-- ROWS: exactly 2 preceding physical rows
SUM(salary) OVER (ORDER BY salary ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)

-- RANGE: all rows with value within the preceding range
SUM(salary) OVER (ORDER BY salary RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
```

## Named Windows (WINDOW Clause)

The WINDOW clause defines reusable window specifications to avoid repetition when multiple window functions share the same partitioning and ordering.

### Syntax

```sql
SELECT
    function1() OVER w,
    function2() OVER w
FROM table_name
WINDOW w AS (PARTITION BY col ORDER BY col2);
```

### Example

```sql
-- Without WINDOW clause (repetitive)
SELECT name, dept, salary,
       ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) as rn,
       RANK() OVER (PARTITION BY dept ORDER BY salary DESC) as rnk,
       SUM(salary) OVER (PARTITION BY dept ORDER BY salary DESC) as running
FROM employees;

-- With WINDOW clause (cleaner)
SELECT name, dept, salary,
       ROW_NUMBER() OVER w as rn,
       RANK() OVER w as rnk,
       SUM(salary) OVER w as running
FROM employees
WINDOW w AS (PARTITION BY dept ORDER BY salary DESC);
```

## Notes

- When ORDER BY is omitted, the entire partition is treated as a single group
- NULL values are handled according to SQL standards
- Window functions can only appear in SELECT and ORDER BY clauses
- Multiple window functions can be used in the same query
