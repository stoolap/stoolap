---
layout: doc
title: Scalar Functions
category: Functions
order: 2
---

# Scalar Functions

Stoolap provides a comprehensive set of scalar functions that operate on individual values and return a single result. This document covers all available scalar functions organized by category.

## String Functions

### UPPER
Converts a string to uppercase.

```sql
SELECT UPPER('hello');                     -- Returns 'HELLO'
SELECT UPPER(name) FROM users;
```

### LOWER
Converts a string to lowercase.

```sql
SELECT LOWER('HELLO');                     -- Returns 'hello'
SELECT LOWER(email) FROM users;
```

### LENGTH / CHAR_LENGTH
Returns the number of characters in a string.

```sql
SELECT LENGTH('hello');                    -- Returns 5
SELECT CHAR_LENGTH('hello');               -- Same as LENGTH
```

### SUBSTRING / SUBSTR
Extracts a substring from a string.

```sql
-- SUBSTRING(string, start [, length])
SELECT SUBSTRING('hello world', 1, 5);     -- Returns 'hello'
SELECT SUBSTRING('hello world', 7);        -- Returns 'world'
SELECT SUBSTR('hello', 2, 3);              -- Returns 'ell'
```

Note: Position is 1-indexed (first character is at position 1).

### CONCAT
Concatenates two or more strings.

```sql
SELECT CONCAT('hello', ' ', 'world');      -- Returns 'hello world'
SELECT CONCAT(first_name, ' ', last_name) AS full_name FROM users;
```

### CONCAT_WS
Concatenates strings with a separator (Concatenate With Separator).

```sql
-- CONCAT_WS(separator, string1, string2, ...)
SELECT CONCAT_WS(', ', 'apple', 'banana', 'cherry');  -- Returns 'apple, banana, cherry'
SELECT CONCAT_WS('-', city, state, zip) AS address FROM customers;
```

Note: NULL values are skipped.

### TRIM / LTRIM / RTRIM
Removes whitespace from strings.

```sql
SELECT TRIM('  hello  ');                  -- Returns 'hello'
SELECT LTRIM('  hello');                   -- Returns 'hello'
SELECT RTRIM('hello  ');                   -- Returns 'hello'
```

### LPAD / RPAD
Pads a string to a specified length.

```sql
-- LPAD(string, length [, pad_string])
SELECT LPAD('42', 5, '0');                 -- Returns '00042'
SELECT RPAD('hello', 10, '-');             -- Returns 'hello-----'
```

### LEFT / RIGHT
Returns characters from the left or right side of a string.

```sql
SELECT LEFT('hello world', 5);             -- Returns 'hello'
SELECT RIGHT('hello world', 5);            -- Returns 'world'
```

### REPLACE
Replaces occurrences of a substring.

```sql
SELECT REPLACE('hello world', 'world', 'there');  -- Returns 'hello there'
```

### REVERSE
Reverses a string.

```sql
SELECT REVERSE('hello');                   -- Returns 'olleh'
```

### REPEAT
Repeats a string a specified number of times.

```sql
SELECT REPEAT('ab', 3);                    -- Returns 'ababab'
```

### LOCATE
Finds the position of a substring within a string.

```sql
-- LOCATE(substring, string [, start_position])
SELECT LOCATE('l', 'hello');               -- Returns 3 (1-based)
SELECT LOCATE('o', 'hello world', 6);      -- Returns 8 (second 'o', starting from position 6)
```

Returns 0 if not found.

### POSITION
SQL standard syntax for finding substring position.

```sql
SELECT POSITION('l' IN 'hello');           -- Returns 3
```

### STRPOS
PostgreSQL-style function for finding substring position.

```sql
-- STRPOS(string, substring)
SELECT STRPOS('hello', 'l');               -- Returns 3
```

### INSTR
Finds the position of a substring (string first, then substring).

```sql
-- INSTR(string, substring)
SELECT INSTR('hello', 'l');                -- Returns 3
```

### SPLIT_PART
Splits a string by a delimiter and returns a specific part.

```sql
-- SPLIT_PART(string, delimiter, part_number)
SELECT SPLIT_PART('a,b,c', ',', 2);        -- Returns 'b'
```

### CHAR
Returns the character for an ASCII/Unicode code point.

```sql
SELECT CHAR(65);                           -- Returns 'A'
SELECT CHAR(97);                           -- Returns 'a'
```

### COLLATE
Applies a specific collation for sorting and comparison.

```sql
SELECT COLLATE('Hello', 'NOCASE');
SELECT * FROM users ORDER BY COLLATE(name, 'NOCASE');
```

Supported collations:
- `BINARY` - Case-sensitive, accent-sensitive
- `NOCASE`, `CASE_INSENSITIVE` - Case-insensitive
- `NOACCENT`, `ACCENT_INSENSITIVE` - Accent-insensitive
- `NUMERIC` - Compare strings as numbers

## Numeric Functions

### ABS
Returns the absolute value.

```sql
SELECT ABS(-10);                           -- Returns 10
SELECT ABS(-3.14);                         -- Returns 3.14
```

### ROUND
Rounds a number to specified decimal places.

```sql
SELECT ROUND(3.14159);                     -- Returns 3.0
SELECT ROUND(3.14159, 2);                  -- Returns 3.14
```

### CEIL / CEILING
Returns the smallest integer >= the number.

```sql
SELECT CEIL(3.14);                         -- Returns 4.0
SELECT CEILING(-3.14);                     -- Returns -3.0
```

### FLOOR
Returns the largest integer <= the number.

```sql
SELECT FLOOR(3.99);                        -- Returns 3.0
SELECT FLOOR(-3.14);                       -- Returns -4.0
```

### TRUNC / TRUNCATE
Truncates a number to specified decimal places (towards zero).

```sql
SELECT TRUNC(3.99);                        -- Returns 3.0
SELECT TRUNCATE(3.14159, 2);               -- Returns 3.14
```

### SQRT
Returns the square root.

```sql
SELECT SQRT(16);                           -- Returns 4.0
SELECT SQRT(2);                            -- Returns 1.4142...
```

### POWER / POW
Raises a number to a power.

```sql
SELECT POWER(2, 3);                        -- Returns 8.0
SELECT POW(10, 2);                         -- Returns 100.0
```

### MOD
Returns the remainder of division.

```sql
SELECT MOD(10, 3);                         -- Returns 1
```

### SIGN
Returns the sign of a number (-1, 0, or 1).

```sql
SELECT SIGN(-15);                          -- Returns -1
SELECT SIGN(0);                            -- Returns 0
SELECT SIGN(42);                           -- Returns 1
```

### EXP
Returns e raised to the specified power.

```sql
SELECT EXP(1);                             -- Returns 2.7183...
```

### LN
Returns the natural logarithm.

```sql
SELECT LN(2.718281828);                    -- Returns ~1
```

### LOG
Returns logarithm. With one argument, returns base-10 log. With two arguments, uses first as base.

```sql
SELECT LOG(10);                            -- Returns 1.0 (base 10)
SELECT LOG(10, 100);                       -- Returns 2.0 (log base 10 of 100)
```

### LOG10
Returns the base-10 logarithm.

```sql
SELECT LOG10(100);                         -- Returns 2.0
SELECT LOG10(1000);                        -- Returns 3.0
```

### LOG2
Returns the base-2 logarithm.

```sql
SELECT LOG2(8);                            -- Returns 3.0
```

### PI
Returns the value of pi.

```sql
SELECT PI();                               -- Returns 3.1416...
```

### RANDOM
Returns a random number between 0 and 1.

```sql
SELECT RANDOM();                           -- Returns random float 0-1
SELECT FLOOR(RANDOM() * 100);              -- Random integer 0-99
```

### SIN / COS / TAN
Trigonometric functions (input in radians).

```sql
SELECT SIN(0);                             -- Returns 0.0
SELECT COS(0);                             -- Returns 1.0
SELECT TAN(0);                             -- Returns 0.0
```

### GREATEST
Returns the largest value from a list.

```sql
SELECT GREATEST(1, 5, 3);                  -- Returns 5
SELECT GREATEST(price, min_price) FROM products;
```

### LEAST
Returns the smallest value from a list.

```sql
SELECT LEAST(1, 5, 3);                     -- Returns 1
SELECT LEAST(price, max_price) FROM products;
```

## Date and Time Functions

### NOW / CURRENT_TIMESTAMP
Returns the current date and time.

```sql
SELECT NOW();                              -- Returns current timestamp
SELECT CURRENT_TIMESTAMP;                  -- Same as NOW()
```

### CURRENT_DATE
Returns the current date (at midnight UTC).

```sql
SELECT CURRENT_DATE;                       -- Returns today's date
```

### CURRENT_TIME
Returns the current time as a string in HH:MM:SS format.

```sql
SELECT CURRENT_TIME;                       -- Returns e.g. '14:30:45'
```

### DATE_TRUNC
Truncates a timestamp to specified precision.

```sql
SELECT DATE_TRUNC('year', '2024-03-15 10:30:45');    -- '2024-01-01T00:00:00Z'
SELECT DATE_TRUNC('month', '2024-03-15 10:30:45');   -- '2024-03-01T00:00:00Z'
SELECT DATE_TRUNC('day', NOW());                     -- Start of today
SELECT DATE_TRUNC('hour', NOW());                    -- Current hour
```

Supported units: `year`, `quarter`, `month`, `week`, `day`, `hour`, `minute`, `second`

### TIME_TRUNC
Truncates a timestamp to a duration interval.

```sql
SELECT TIME_TRUNC('15m', '2024-03-15 10:37:45');     -- '2024-03-15T10:30:00Z'
SELECT TIME_TRUNC('1h', '2024-03-15 10:37:45');      -- '2024-03-15T10:00:00Z'
```

Supported intervals: `ns`, `us`, `ms`, `s`, `m`, `h` (with numeric prefix, e.g., `15m`, `4h`)

### EXTRACT
Extracts a field from a timestamp. Uses SQL standard syntax.

```sql
SELECT EXTRACT(YEAR FROM '2024-03-15');              -- 2024
SELECT EXTRACT(MONTH FROM '2024-03-15');             -- 3
SELECT EXTRACT(DAY FROM '2024-03-15');               -- 15
SELECT EXTRACT(HOUR FROM '2024-03-15 14:30:00');     -- 14
SELECT EXTRACT(MINUTE FROM '2024-03-15 14:30:00');   -- 30
SELECT EXTRACT(SECOND FROM '2024-03-15 14:30:45');   -- 45
SELECT EXTRACT(DOW FROM '2024-03-15');               -- 5 (Friday, 0=Sunday)
SELECT EXTRACT(DOY FROM '2024-03-15');               -- 75 (day of year)
SELECT EXTRACT(WEEK FROM '2024-03-15');              -- 11 (ISO week)
SELECT EXTRACT(QUARTER FROM '2024-05-15');           -- 2
```

Supported fields: `YEAR`, `MONTH`, `DAY`, `HOUR`, `MINUTE`, `SECOND`, `DOW` (day of week), `DOY` (day of year), `WEEK`, `QUARTER`

### YEAR / MONTH / DAY
Shorthand functions to extract date parts.

```sql
SELECT YEAR('2024-03-15');                 -- Returns 2024
SELECT MONTH('2024-03-15');                -- Returns 3
SELECT DAY('2024-03-15');                  -- Returns 15
```

### HOUR / MINUTE / SECOND
Shorthand functions to extract time parts.

```sql
SELECT HOUR('2024-03-15 14:30:45');        -- Returns 14
SELECT MINUTE('2024-03-15 14:30:45');      -- Returns 30
SELECT SECOND('2024-03-15 14:30:45');      -- Returns 45
```

### DATE_ADD
Adds an interval to a timestamp.

```sql
-- DATE_ADD(timestamp, amount [, unit])
SELECT DATE_ADD('2024-03-15', 10);                   -- Add 10 days (default)
SELECT DATE_ADD('2024-03-15', 2, 'month');           -- Add 2 months
```

Supported units: `year`, `month`, `week`, `day`, `hour`, `minute`, `second`

### DATE_SUB
Subtracts an interval from a timestamp.

```sql
SELECT DATE_SUB('2024-03-15', 10);                   -- Subtract 10 days
SELECT DATE_SUB('2024-03-15', 1, 'month');           -- Subtract 1 month
```

### DATEDIFF
Returns the difference between two dates in days.

```sql
SELECT DATEDIFF('2024-03-15', '2024-03-01');         -- Returns 14
```

### TO_CHAR
Formats a timestamp as a string.

```sql
SELECT TO_CHAR('2024-03-15 14:30:45', 'YYYY-MM-DD');         -- '2024-03-15'
SELECT TO_CHAR('2024-03-15 14:30:45', 'DD MON YYYY');        -- '15 MAR 2024'
SELECT TO_CHAR('2024-03-15 14:30:45', 'HH24:MI:SS');         -- '14:30:45'
```

Format patterns:
- `YYYY` - 4-digit year
- `YY` - 2-digit year
- `MM` - Month as 01-12
- `MON` - Abbreviated month (JAN, FEB, ...)
- `MONTH` - Full month name
- `DD` - Day of month (01-31)
- `DY` - Abbreviated day name (SUN, MON, ...)
- `DAY` - Full day name
- `HH24` - Hour (00-23)
- `HH` or `HH12` - Hour (01-12)
- `MI` - Minutes (00-59)
- `SS` - Seconds (00-59)

## Conversion Functions

### CAST
Converts a value from one type to another.

```sql
SELECT CAST('123' AS INTEGER);             -- Returns 123
SELECT CAST(3.14 AS INTEGER);              -- Returns 3
SELECT CAST(42 AS TEXT);                   -- Returns '42'
SELECT CAST('true' AS BOOLEAN);            -- Returns true
SELECT CAST('2024-03-15' AS TIMESTAMP);    -- Returns timestamp
```

Supported types: `INTEGER`/`INT`, `FLOAT`/`REAL`/`DOUBLE`, `TEXT`/`STRING`/`VARCHAR`, `BOOLEAN`/`BOOL`, `TIMESTAMP`/`DATETIME`/`DATE`, `JSON`, `VECTOR(N)`

### COALESCE
Returns the first non-NULL value.

```sql
SELECT COALESCE(NULL, NULL, 'default');    -- Returns 'default'
SELECT COALESCE(nickname, first_name, 'Anonymous') FROM users;
```

### NULLIF
Returns NULL if two values are equal, otherwise returns the first value.

```sql
SELECT NULLIF(10, 10);                     -- Returns NULL
SELECT NULLIF(10, 20);                     -- Returns 10
```

### IFNULL
Returns the second value if the first is NULL.

```sql
SELECT IFNULL(NULL, 'default');            -- Returns 'default'
SELECT IFNULL(nickname, 'No nickname') FROM users;
```

### IIF
Inline conditional (if-then-else).

```sql
-- IIF(condition, true_value, false_value)
SELECT IIF(5 > 3, 'yes', 'no');            -- Returns 'yes'
SELECT IIF(quantity > 0, 'In Stock', 'Out of Stock') FROM products;
```

### TYPEOF
Returns the data type name of a value.

```sql
SELECT TYPEOF(123);                        -- Returns 'INTEGER'
SELECT TYPEOF(3.14);                       -- Returns 'FLOAT'
SELECT TYPEOF('hello');                    -- Returns 'TEXT'
SELECT TYPEOF(true);                       -- Returns 'BOOLEAN'
SELECT TYPEOF(NULL);                       -- Returns 'NULL'
```

## JSON Functions

Stoolap provides comprehensive JSON support for storing and querying JSON data.

### JSON_EXTRACT
Extracts a value from JSON using a path expression.

```sql
-- JSON_EXTRACT(json, path)
SELECT JSON_EXTRACT('{"name": "Alice", "age": 30}', '$.name');    -- 'Alice'
SELECT JSON_EXTRACT('{"user": {"email": "a@b.com"}}', '$.user.email');
SELECT JSON_EXTRACT('{"items": [1, 2, 3]}', '$.items[0]');        -- 1
```

### JSON_TYPE / JSON_TYPEOF
Returns the type of a JSON value.

```sql
SELECT JSON_TYPE('{"a": 1}');              -- 'object'
SELECT JSON_TYPE('[1, 2, 3]');             -- 'array'
SELECT JSON_TYPEOF('"hello"');             -- 'string'
SELECT JSON_TYPEOF('123');                 -- 'number'
SELECT JSON_TYPEOF('true');                -- 'boolean'
SELECT JSON_TYPEOF('null');                -- 'null'
```

### JSON_VALID
Checks if a string is valid JSON.

```sql
SELECT JSON_VALID('{"a": 1}');             -- Returns 1 (true)
SELECT JSON_VALID('not json');             -- Returns 0 (false)
```

### JSON_KEYS
Returns the keys of a JSON object as a JSON array.

```sql
SELECT JSON_KEYS('{"a": 1, "b": 2, "c": 3}');   -- '["a","b","c"]'
```

### JSON_ARRAY_LENGTH
Returns the number of elements in a JSON array.

```sql
SELECT JSON_ARRAY_LENGTH('[1, 2, 3, 4, 5]');   -- 5
```

### JSON_ARRAY
Creates a JSON array from values.

```sql
SELECT JSON_ARRAY(1, 2, 3);                -- '[1,2,3]'
SELECT JSON_ARRAY('a', 'b', 'c');          -- '["a","b","c"]'
```

### JSON_OBJECT
Creates a JSON object from key-value pairs.

```sql
SELECT JSON_OBJECT('name', 'Alice', 'age', 30);   -- '{"age":30,"name":"Alice"}'
```

## Vector Functions

Stoolap provides functions for computing distances between vectors and inspecting vector values. These are used with the `VECTOR(N)` data type for similarity search. See [Vector Search]({% link _docs/data-types/vector-search.md %}) for complete documentation.

### VEC_DISTANCE_L2
Computes the Euclidean (L2) distance between two vectors.

```sql
-- VEC_DISTANCE_L2(vector_a, vector_b)
SELECT VEC_DISTANCE_L2(embedding, '[0.1, 0.2, 0.3]') AS dist FROM items;

-- k-nearest neighbor search
SELECT id, VEC_DISTANCE_L2(embedding, '[0.1, 0.2, 0.3]') AS dist
FROM items ORDER BY dist LIMIT 10;
```

Returns a FLOAT value. Both arguments must have the same number of dimensions.

### VEC_DISTANCE_COSINE
Computes the cosine distance (1 - cosine similarity) between two vectors.

```sql
-- VEC_DISTANCE_COSINE(vector_a, vector_b)
SELECT VEC_DISTANCE_COSINE(embedding, '[0.1, 0.2, 0.3]') AS dist FROM items;
```

Returns 0.0 for identical directions, 1.0 for orthogonal vectors, 2.0 for opposite directions. Returns 1.0 if either vector is a zero vector.

### VEC_DISTANCE_IP
Computes the negative inner product distance (-dot product) between two vectors.

```sql
-- VEC_DISTANCE_IP(vector_a, vector_b)
SELECT VEC_DISTANCE_IP(embedding, '[0.1, 0.2, 0.3]') AS dist FROM items;
```

### VEC_DIMS
Returns the number of dimensions in a vector.

```sql
SELECT VEC_DIMS(embedding) FROM items WHERE id = 1;  -- Returns 384
```

### VEC_NORM
Returns the L2 norm (magnitude) of a vector.

```sql
SELECT VEC_NORM(embedding) FROM items WHERE id = 1;  -- Returns 1.0 for normalized vectors
```

### VEC_TO_TEXT
Converts a vector to its text representation.

```sql
SELECT VEC_TO_TEXT(embedding) FROM items WHERE id = 1;  -- Returns '[0.1, 0.2, 0.3, ...]'
```

### EMBED

> **Requires:** `--features semantic`

Converts text into a 384-dimensional semantic embedding vector using the built-in all-MiniLM-L6-v2 sentence-transformer model. The model runs in pure Rust and is automatically downloaded on first use.

```sql
-- Generate an embedding from text
SELECT EMBED('How to reset my password');

-- Insert with auto-generated embedding
INSERT INTO docs (content, embedding)
VALUES ('Hello world', EMBED('Hello world'));

-- Semantic search
SELECT content,
       VEC_DISTANCE_COSINE(embedding, EMBED('greeting')) AS dist
FROM docs ORDER BY dist LIMIT 5;
```

Returns a `VECTOR(384)` value. Accepts TEXT, INTEGER, or FLOAT arguments. Returns NULL for NULL input. See [Semantic Search]({% link _docs/data-types/semantic-search.md %}) for complete documentation.

## Utility Functions

### VERSION
Returns the Stoolap version string.

```sql
SELECT VERSION();                          -- Returns version info
```

### SLEEP
Pauses execution for a specified number of seconds.

```sql
SELECT SLEEP(1);                           -- Pauses for 1 second, returns 0
SELECT SLEEP(0.5);                         -- Pauses for 500ms
```

## Example Queries

### Data Cleaning
```sql
SELECT
    id,
    TRIM(UPPER(name)) AS clean_name,
    COALESCE(email, 'no-email@example.com') AS email,
    IFNULL(phone, 'N/A') AS phone
FROM customers;
```

### Date Analysis
```sql
SELECT
    DATE_TRUNC('month', order_date) AS month,
    COUNT(*) AS orders,
    ROUND(SUM(total), 2) AS revenue
FROM orders
WHERE order_date >= DATE_SUB(NOW(), 12, 'month')
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY month;
```

### JSON Processing
```sql
SELECT
    id,
    JSON_EXTRACT(metadata, '$.category') AS category,
    JSON_EXTRACT(metadata, '$.tags[0]') AS first_tag,
    JSON_ARRAY_LENGTH(metadata, '$.tags') AS tag_count
FROM products
WHERE JSON_VALID(metadata) = 1;
```

### Conditional Logic
```sql
SELECT
    product_name,
    price,
    IIF(quantity > 0, 'In Stock', 'Out of Stock') AS availability,
    GREATEST(price * 0.9, min_price) AS sale_price
FROM products;
```

## Performance Notes

- Scalar functions execute for each row; consider filtering first to reduce row count
- Using functions in WHERE clauses may prevent index usage
- JSON functions parse JSON on each call; consider storing frequently accessed values in regular columns
