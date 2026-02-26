---
layout: doc
title: SQL Functions Reference
category: Functions
order: 1
---

# SQL Functions Reference

Quick reference for all 117 built-in SQL functions in Stoolap, organized by category. For detailed documentation with examples, see [Scalar Functions]({% link _docs/functions/scalar-functions.md %}), [Aggregate Functions]({% link _docs/functions/aggregate-functions.md %}), and [Window Functions]({% link _docs/functions/window-functions.md %}).

## Aggregate Functions (17)

Aggregate functions operate on a set of rows and return a single value. Used with `GROUP BY` or over entire result sets.

| Function | Syntax | Description |
|----------|--------|-------------|
| `COUNT` | `COUNT(*)`, `COUNT(expr)`, `COUNT(DISTINCT expr)` | Count rows or non-NULL values |
| `SUM` | `SUM(expr)`, `SUM(DISTINCT expr)` | Sum of numeric values |
| `AVG` | `AVG(expr)`, `AVG(DISTINCT expr)` | Average of numeric values |
| `MIN` | `MIN(expr)` | Minimum value |
| `MAX` | `MAX(expr)` | Maximum value |
| `FIRST` | `FIRST(expr)` | First value in a group |
| `LAST` | `LAST(expr)` | Last value in a group |
| `MEDIAN` | `MEDIAN(expr)` | Median value (50th percentile) |
| `STRING_AGG` | `STRING_AGG(expr, delimiter)` | Concatenate values with delimiter |
| `GROUP_CONCAT` | `GROUP_CONCAT(expr, delimiter)` | Alias for STRING_AGG |
| `ARRAY_AGG` | `ARRAY_AGG(expr)` | Collect values into a JSON array |
| `STDDEV` | `STDDEV(expr)` | Sample standard deviation (alias for STDDEV_SAMP) |
| `STDDEV_SAMP` | `STDDEV_SAMP(expr)` | Sample standard deviation |
| `STDDEV_POP` | `STDDEV_POP(expr)` | Population standard deviation |
| `VARIANCE` | `VARIANCE(expr)` | Sample variance (alias for VAR_SAMP) |
| `VAR_SAMP` | `VAR_SAMP(expr)` | Sample variance |
| `VAR_POP` | `VAR_POP(expr)` | Population variance |

## Scalar Functions (89)

### String Functions (24)

| Function | Syntax | Description |
|----------|--------|-------------|
| `UPPER` | `UPPER(str)` | Convert to uppercase |
| `LOWER` | `LOWER(str)` | Convert to lowercase |
| `LENGTH` | `LENGTH(str)` | String length in characters |
| `CHAR_LENGTH` | `CHAR_LENGTH(str)` | String length in characters (alias for LENGTH) |
| `CHAR` | `CHAR(code)` | Character from Unicode code point |
| `CONCAT` | `CONCAT(str1, str2, ...)` | Concatenate strings |
| `CONCAT_WS` | `CONCAT_WS(sep, str1, str2, ...)` | Concatenate with separator |
| `SUBSTRING` | `SUBSTRING(str, pos, len)` | Extract substring |
| `SUBSTR` | `SUBSTR(str, pos, len)` | Alias for SUBSTRING |
| `TRIM` | `TRIM(str)` | Remove leading/trailing whitespace |
| `LTRIM` | `LTRIM(str)` | Remove leading whitespace |
| `RTRIM` | `RTRIM(str)` | Remove trailing whitespace |
| `REPLACE` | `REPLACE(str, from, to)` | Replace occurrences |
| `REVERSE` | `REVERSE(str)` | Reverse a string |
| `LEFT` | `LEFT(str, n)` | First n characters |
| `RIGHT` | `RIGHT(str, n)` | Last n characters |
| `REPEAT` | `REPEAT(str, n)` | Repeat string n times |
| `SPLIT_PART` | `SPLIT_PART(str, delim, n)` | Extract nth part after splitting |
| `POSITION` | `POSITION(substr IN str)` | Position of substring (1-based) |
| `STRPOS` | `STRPOS(str, substr)` | Position of substring (1-based) |
| `INSTR` | `INSTR(str, substr)` | Position of substring (1-based) |
| `LOCATE` | `LOCATE(substr, str)` | Position of substring (1-based) |
| `LPAD` | `LPAD(str, len, pad)` | Left-pad to length |
| `RPAD` | `RPAD(str, len, pad)` | Right-pad to length |

### Math Functions (22)

| Function | Syntax | Description |
|----------|--------|-------------|
| `ABS` | `ABS(n)` | Absolute value |
| `ROUND` | `ROUND(n)`, `ROUND(n, decimals)` | Round to nearest integer or decimal places |
| `FLOOR` | `FLOOR(n)` | Round down to nearest integer |
| `CEILING` | `CEILING(n)` | Round up to nearest integer |
| `CEIL` | `CEIL(n)` | Alias for CEILING |
| `MOD` | `MOD(a, b)` | Modulo (remainder) |
| `POWER` | `POWER(base, exp)` | Exponentiation |
| `POW` | `POW(base, exp)` | Alias for POWER |
| `SQRT` | `SQRT(n)` | Square root |
| `LOG` | `LOG(n)`, `LOG(base, n)` | Base-10 log, or log with specified base |
| `LOG10` | `LOG10(n)` | Base-10 logarithm |
| `LOG2` | `LOG2(n)` | Base-2 logarithm |
| `LN` | `LN(n)` | Natural logarithm |
| `EXP` | `EXP(n)` | e raised to the power n |
| `SIGN` | `SIGN(n)` | Sign of number (-1, 0, or 1) |
| `TRUNCATE` | `TRUNCATE(n, decimals)` | Truncate to decimal places |
| `TRUNC` | `TRUNC(n, decimals)` | Alias for TRUNCATE |
| `PI` | `PI()` | Value of Pi (3.14159...) |
| `RANDOM` | `RANDOM()` | Random float between 0 and 1 |
| `SIN` | `SIN(n)` | Sine (radians) |
| `COS` | `COS(n)` | Cosine (radians) |
| `TAN` | `TAN(n)` | Tangent (radians) |

### Date/Time Functions (18)

| Function | Syntax | Description |
|----------|--------|-------------|
| `NOW` | `NOW()` | Current date and time |
| `CURRENT_DATE` | `CURRENT_DATE` | Current date |
| `CURRENT_TIME` | `CURRENT_TIME` | Current time as HH:MM:SS |
| `CURRENT_TIMESTAMP` | `CURRENT_TIMESTAMP` | Current date and time (alias for NOW) |
| `DATE_TRUNC` | `DATE_TRUNC(unit, timestamp)` | Truncate timestamp to unit |
| `TIME_TRUNC` | `TIME_TRUNC(unit, timestamp)` | Truncate time to unit |
| `EXTRACT` | `EXTRACT(field FROM timestamp)` | Extract date/time field |
| `YEAR` | `YEAR(timestamp)` | Extract year |
| `MONTH` | `MONTH(timestamp)` | Extract month |
| `DAY` | `DAY(timestamp)` | Extract day |
| `HOUR` | `HOUR(timestamp)` | Extract hour |
| `MINUTE` | `MINUTE(timestamp)` | Extract minute |
| `SECOND` | `SECOND(timestamp)` | Extract second |
| `DATE_ADD` | `DATE_ADD(timestamp, n [, unit])` | Add interval to date |
| `DATE_SUB` | `DATE_SUB(timestamp, n [, unit])` | Subtract interval from date |
| `DATEDIFF` | `DATEDIFF(date1, date2)` | Difference between dates in days |
| `DATE_DIFF` | `DATE_DIFF(date1, date2)` | Alias for DATEDIFF |
| `TO_CHAR` | `TO_CHAR(timestamp, format)` | Format timestamp as string |

### JSON Functions (8)

| Function | Syntax | Description |
|----------|--------|-------------|
| `JSON_EXTRACT` | `JSON_EXTRACT(json, path)` | Extract value at JSON path |
| `JSON_ARRAY_LENGTH` | `JSON_ARRAY_LENGTH(json [, path])` | Length of JSON array |
| `JSON_ARRAY` | `JSON_ARRAY(val1, val2, ...)` | Create JSON array |
| `JSON_OBJECT` | `JSON_OBJECT(key1, val1, ...)` | Create JSON object |
| `JSON_TYPE` | `JSON_TYPE(json)` | Type of JSON value |
| `JSON_TYPEOF` | `JSON_TYPEOF(json)` | Alias for JSON_TYPE |
| `JSON_VALID` | `JSON_VALID(str)` | Check if string is valid JSON |
| `JSON_KEYS` | `JSON_KEYS(json)` | Get keys of JSON object |

JSON shorthand operators are also supported:

| Operator | Description | Example |
|----------|-------------|---------|
| `->` | Extract JSON value (returns JSON) | `col -> 'key'` |
| `->>` | Extract JSON value (returns TEXT) | `col ->> 'key'` |

### Conditional Functions (4)

| Function | Syntax | Description |
|----------|--------|-------------|
| `COALESCE` | `COALESCE(expr1, expr2, ...)` | First non-NULL value |
| `NULLIF` | `NULLIF(expr1, expr2)` | NULL if expr1 = expr2 |
| `IFNULL` | `IFNULL(expr, default)` | Default value if NULL |
| `IIF` | `IIF(condition, true_val, false_val)` | Inline conditional |

### Type and Comparison Functions (5)

| Function | Syntax | Description |
|----------|--------|-------------|
| `CAST` | `CAST(expr AS type)` | Convert to another data type |
| `TYPEOF` | `TYPEOF(expr)` | Return the data type name |
| `COLLATE` | `COLLATE(expr, collation)` | Apply collation for comparison |
| `GREATEST` | `GREATEST(val1, val2, ...)` | Largest value from list |
| `LEAST` | `LEAST(val1, val2, ...)` | Smallest value from list |

### Vector Functions (7)

| Function | Syntax | Description |
|----------|--------|-------------|
| `VEC_DISTANCE_L2` | `VEC_DISTANCE_L2(vec_a, vec_b)` | Euclidean (L2) distance between vectors |
| `VEC_DISTANCE_COSINE` | `VEC_DISTANCE_COSINE(vec_a, vec_b)` | Cosine distance (1 - cosine similarity) |
| `VEC_DISTANCE_IP` | `VEC_DISTANCE_IP(vec_a, vec_b)` | Negative inner product distance (-dot product) |
| `VEC_DIMS` | `VEC_DIMS(vec)` | Number of dimensions in a vector |
| `VEC_NORM` | `VEC_NORM(vec)` | L2 norm (magnitude) of a vector |
| `VEC_TO_TEXT` | `VEC_TO_TEXT(vec)` | Convert vector to text representation |
| `EMBED`* | `EMBED(text)` | Convert text to 384-dim semantic embedding (MiniLM-L6-v2) |

\* Requires `--features semantic`. See [Semantic Search]({% link _docs/data-types/semantic-search.md %}) for details.

See [Vector Search]({% link _docs/data-types/vector-search.md %}) for usage examples and HNSW index documentation.

### System Functions (2)

| Function | Syntax | Description |
|----------|--------|-------------|
| `VERSION` | `VERSION()` | Database version string |
| `SLEEP` | `SLEEP(seconds)` | Pause execution (for testing) |

## Window Functions (11)

Window functions perform calculations across a set of rows related to the current row. Used with the `OVER` clause.

| Function | Syntax | Description |
|----------|--------|-------------|
| `ROW_NUMBER` | `ROW_NUMBER() OVER (...)` | Sequential row number |
| `RANK` | `RANK() OVER (...)` | Rank with gaps for ties |
| `DENSE_RANK` | `DENSE_RANK() OVER (...)` | Rank without gaps for ties |
| `NTILE` | `NTILE(n) OVER (...)` | Distribute rows into n groups |
| `LEAD` | `LEAD(expr, offset, default) OVER (...)` | Value from following row |
| `LAG` | `LAG(expr, offset, default) OVER (...)` | Value from preceding row |
| `FIRST_VALUE` | `FIRST_VALUE(expr) OVER (...)` | First value in window frame |
| `LAST_VALUE` | `LAST_VALUE(expr) OVER (...)` | Last value in window frame |
| `NTH_VALUE` | `NTH_VALUE(expr, n) OVER (...)` | Nth value in window frame |
| `PERCENT_RANK` | `PERCENT_RANK() OVER (...)` | Relative rank (0 to 1) |
| `CUME_DIST` | `CUME_DIST() OVER (...)` | Cumulative distribution (0 to 1) |

### Window Frame Syntax

```sql
function OVER (
    [PARTITION BY col1, col2, ...]
    [ORDER BY col1 [ASC|DESC], ...]
    [frame_clause]
)
```

Frame clause options:

| Frame | Description |
|-------|-------------|
| `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` | Default frame |
| `ROWS BETWEEN n PRECEDING AND n FOLLOWING` | Fixed-size sliding window |
| `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` | Entire partition |
| `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` | Range-based frame |

### Named Windows

```sql
SELECT
    ROW_NUMBER() OVER w,
    SUM(amount) OVER w
FROM orders
WINDOW w AS (PARTITION BY customer_id ORDER BY order_date);
```

## CASE Expression

While not a function, `CASE` is commonly used alongside functions:

```sql
-- Simple CASE
CASE status WHEN 'A' THEN 'Active' WHEN 'I' THEN 'Inactive' END

-- Searched CASE
CASE WHEN age >= 18 THEN 'Adult' ELSE 'Minor' END
```

## Common Patterns

### Nesting Functions

```sql
SELECT ROUND(AVG(price), 2) FROM products;
SELECT UPPER(TRIM(name)) FROM users;
SELECT COALESCE(NULLIF(value, ''), 'default') FROM config;
```

### Functions in WHERE

```sql
SELECT * FROM events WHERE YEAR(created_at) = 2024;
SELECT * FROM products WHERE LOWER(name) LIKE '%organic%';
```

### Functions with GROUP BY

```sql
SELECT DATE_TRUNC('month', order_date) AS month,
       COUNT(*) AS orders,
       ROUND(AVG(total), 2) AS avg_total
FROM orders
GROUP BY DATE_TRUNC('month', order_date)
HAVING SUM(total) > 10000;
```
