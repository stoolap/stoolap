---
layout: doc
title: CAST Operations
category: SQL Features
order: 7
---

# CAST Operations

Stoolap supports both explicit type casting using the CAST function and implicit type conversion.

## Overview

Stoolap supports both explicit type casting using the CAST function and implicit type conversion in certain contexts. Type casting allows you to convert a value from one data type to another.

## Supported Data Types

Stoolap supports casting between these data types:

- `INTEGER` or `INT`: 64-bit signed integers
- `FLOAT`: 64-bit floating-point numbers
- `TEXT`, `STRING`, `VARCHAR`, or `CHAR`: Text strings
- `BOOLEAN` or `BOOL`: Boolean values
- `TIMESTAMP`, `DATETIME`, `DATE`, or `TIME`: Date and time values
- `JSON`: JSON data format

## Explicit CAST Syntax

To explicitly cast a value, use the CAST function:

```sql
CAST(value AS type)
```

Where:
- `value` can be a column reference, literal, or expression
- `type` is one of the supported data types

## Examples

### Basic CAST Operations

```sql
-- String to integer
SELECT CAST('123' AS INTEGER);                -- Returns 123

-- Integer to string
SELECT CAST(42 AS TEXT);                      -- Returns '42'

-- Float to integer (truncates)
SELECT CAST(123.45 AS INTEGER);               -- Returns 123

-- Boolean to integer
SELECT CAST(TRUE AS INTEGER);                 -- Returns 1
SELECT CAST(FALSE AS INTEGER);                -- Returns 0

-- String to boolean
SELECT CAST('true' AS BOOLEAN);               -- Returns TRUE
SELECT CAST('yes' AS BOOLEAN);                -- Returns TRUE
SELECT CAST('false' AS BOOLEAN);              -- Returns FALSE
SELECT CAST('0' AS BOOLEAN);                  -- Returns FALSE

-- Number to boolean
SELECT CAST(0 AS BOOLEAN);                    -- Returns FALSE
SELECT CAST(1 AS BOOLEAN);                    -- Returns TRUE
SELECT CAST(42 AS BOOLEAN);                   -- Returns TRUE

-- Integer to float
SELECT CAST(123 AS FLOAT);                    -- Returns 123.0

-- String to timestamp
SELECT CAST('2023-05-15 14:30:00' AS TIMESTAMP); -- Returns timestamp

-- Timestamp to string
SELECT CAST(NOW() AS TEXT);                   -- Returns formatted timestamp
```

### Using CAST in WHERE Clauses

```sql
-- Create a test table with mixed data types
CREATE TABLE cast_test (
    id INTEGER PRIMARY KEY,
    text_val TEXT,
    int_val INTEGER,
    float_val FLOAT,
    bool_val BOOLEAN
);

-- Insert test data
INSERT INTO cast_test VALUES 
    (1, '100', 100, 100.5, TRUE),
    (2, '200', 200, 200.5, FALSE),
    (3, 'abc', 300, 300.5, TRUE);

-- Cast text column to integer in WHERE clause
SELECT id FROM cast_test WHERE CAST(text_val AS INTEGER) > 150;  -- Returns 2
```

### Complex CAST Operations

```sql
-- Nested CAST operations
SELECT CAST(CAST(123.45 AS INTEGER) AS TEXT);  -- Returns '123'

-- CAST with expressions
SELECT CAST(int_val * 2 AS TEXT) FROM cast_test;

-- CAST with NULL values (preserves NULL with target type)
SELECT CAST(NULL AS INTEGER);                  -- Returns NULL (typed as INTEGER)
SELECT CAST(NULL AS TEXT);                     -- Returns NULL (typed as TEXT)
SELECT CAST(NULL AS BOOLEAN);                  -- Returns NULL (typed as BOOLEAN)
```

## Type Conversion Rules

### To INTEGER

- **From FLOAT**: Truncates toward zero (123.45 → 123, -3.7 → -3). A value outside the INTEGER range, NaN or infinity is an error
- **From TEXT**: Parses an integer, ignoring the spaces around it (" 123 " → 123). A number outside the INTEGER range is an error; any other text, including "3.7" and "", becomes NULL
- **From BOOLEAN**: TRUE → 1, FALSE → 0
- **From TIMESTAMP**: NULL
- **From NULL**: Returns NULL (typed as INTEGER)

The same conversion applies wherever a value becomes an INTEGER: a CAST, a value written into an INTEGER column by INSERT, UPDATE, upsert, a DEFAULT or COPY, and an INTEGER argument of a function such as LEFT or SUBSTRING. A FLOAT outside the range is never clamped to the largest INTEGER. `ALTER TABLE ... MODIFY COLUMN ... INTEGER` is the exception: a value it cannot convert becomes NULL, as described for that statement.

### To FLOAT

- **From INTEGER**: Direct conversion (123 → 123.0)
- **From TEXT**: Parses numeric string ("123.45" → 123.45), non-numeric strings become 0.0
- **From BOOLEAN**: TRUE → 1.0, FALSE → 0.0
- **From TIMESTAMP**: Converts to Unix timestamp with fractional seconds
- **From NULL**: Returns NULL (typed as FLOAT)

### To TEXT

- **From INTEGER**: String representation (123 → "123")
- **From FLOAT**: String representation (123.45 → "123.45")
- **From BOOLEAN**: "true" or "false"
- **From TIMESTAMP**: RFC 3339 in UTC ("2023-05-15T14:30:00+00:00"), with the fractional seconds the value holds, up to nanoseconds ("2023-05-15T14:30:00.123456789+00:00")
- **From NULL**: Returns NULL (typed as TEXT)

The same conversion applies wherever a value becomes a TEXT: a CAST, and a value written into a TEXT column by INSERT, UPDATE, upsert or a DEFAULT. A typed literal such as `TIMESTAMP '2023-05-15 14:30:00'` or `DATE '2023-05-15'` is a TIMESTAMP, so written into a TEXT column it is stored as `'2023-05-15T14:30:00+00:00'`, not as the text between its quotes. A plain string literal is stored as written.

### To BOOLEAN

- **From INTEGER**: 0 → FALSE, non-zero → TRUE
- **From FLOAT**: 0.0 → FALSE, non-zero → TRUE
- **From TEXT**: "true", "t", "yes", "y", "1" → TRUE (case-insensitive)
                "false", "f", "no", "n", "0" → FALSE (case-insensitive)
                any other text, including "", becomes NULL
- **From NULL**: Returns NULL (typed as BOOLEAN)

### To TIMESTAMP

- **From INTEGER**: Interpreted as Unix timestamp
- **From FLOAT**: Interpreted as Unix timestamp with fractional seconds
- **From TEXT**: Parses date/time string in various formats
- **From NULL**: Returns NULL (typed as TIMESTAMP)

### To JSON

- **From TEXT**: Validated as JSON string
- **From other types**: Converted to JSON representation

## Implicit Type Conversion

Stoolap performs implicit type conversion in these contexts:

1. **Arithmetic operations**: When mixing numeric types
   ```sql
   SELECT 1 + 2.5;  -- INTEGER implicitly converted to FLOAT
   ```

2. **Comparison operations**: When a number is compared with a TEXT, the TEXT is read as a number
   ```sql
   SELECT * FROM table WHERE id = '100';  -- '100' is read as the number 100
   SELECT 1 = '01';                       -- true
   SELECT 1 < '1.5';                      -- true
   SELECT '01' = '1';                     -- false: two TEXT values compare as text
   SELECT 1 = 'abc';                      -- NULL: 'abc' is not a number
   ```
   A TEXT reads as a number when it has the form `[+-]digits[.digits][e[+-]digits]`, with no spaces. Integral values that fit a 64-bit integer are compared exactly; others are compared as FLOAT. A TEXT that is not a number makes the comparison unknown (NULL), so `1 IS DISTINCT FROM 'abc'` is true and `1 IN ('abc', 2)` is NULL. The same rule applies to `=`, `<>`, `<`, `<=`, `>`, `>=`, `IS [NOT] DISTINCT FROM`, `BETWEEN`, `IN` lists and subqueries, `EXISTS`, and join conditions, whichever plan or index the query uses.

   A BOOLEAN or TIMESTAMP and an INTEGER or FLOAT are different domains, and neither is read as the other: `=` between them is false, `<>` is true, and `<`, `<=`, `>`, `>=` and `BETWEEN` are unknown (NULL), in either operand order, in WHERE, HAVING and the select list alike, whether or not the comparison reaches an index or the table scan. A quoted TEXT is still read as the column's type, so `b = 'false'` and `ts > '2025-01-01'` compare as written.
   ```sql
   SELECT TRUE > 0;                                    -- NULL
   SELECT CAST('2025-01-01' AS TIMESTAMP) >= 0;        -- NULL
   SELECT TRUE = 1;                                    -- false
   SELECT * FROM t WHERE b = 0;                        -- no rows, for a BOOLEAN column b
   ```

   A TEXT compared with a BOOLEAN is read the way `CAST(... AS BOOLEAN)` reads it, and a TEXT compared with a TIMESTAMP is parsed as a timestamp. A TEXT that does not read makes the comparison unknown (NULL). BOOLEAN values order FALSE before TRUE. Two JSON values, or two VECTOR values, are equal or not by value; ordering them, and comparing either with any other type, is unknown. These rules hold for every operator and context listed above.
   ```sql
   SELECT TRUE = 'yes';                                -- true
   SELECT FALSE < TRUE;                                -- true
   SELECT TRUE = 'maybe';                              -- NULL
   SELECT CAST('2024-01-01' AS TIMESTAMP) < '2024-01-02';  -- true
   SELECT CAST('2024-01-01' AS TIMESTAMP) = 'junk';    -- NULL
   SELECT CAST('{"a":1}' AS JSON) = CAST('{"a":1}' AS JSON);  -- true
   SELECT CAST('{"a":1}' AS JSON) < CAST('{"a":2}' AS JSON);  -- NULL
   SELECT * FROM t WHERE j = '{"a":1}';                -- no rows, for a JSON column j
   ```

3. **Function arguments**: When a function expects specific types
   ```sql
   SELECT ABS('-123');  -- String implicitly converted to number
   ```

## NULL Handling in CAST Operations

Following the SQL standard, casting NULL preserves NULL with the target type:

```sql
SELECT CAST(NULL AS INTEGER);    -- NULL (typed as INTEGER)
SELECT CAST(NULL AS TEXT);       -- NULL (typed as TEXT)
SELECT CAST(NULL AS BOOLEAN);    -- NULL (typed as BOOLEAN)
SELECT CAST(NULL AS TIMESTAMP);  -- NULL (typed as TIMESTAMP)
```

Use `COALESCE` to provide default values when NULL is not desired:

```sql
SELECT COALESCE(CAST(NULL AS INTEGER), 0);  -- 0
```

## Best Practices

1. **Use explicit casts** for clarity when type conversion is intentional

2. **Be aware of data loss** when casting from higher precision to lower precision types (FLOAT to INTEGER loses decimal portion)

3. **Validate inputs** when casting from TEXT to ensure they're in the expected format

4. **Consider NULL handling** when casting columns that might contain NULL values

## Implementation Details

Stoolap implements CAST operations in various components:

- The parser (`src/parser/`) recognizes CAST expressions
- The scalar function implementation (`src/functions/scalar/cast.rs`) handles the CAST logic
- Type-specific cast expressions (`src/storage/expression/cast_expression.rs`) optimize CAST operations in storage layer

Parallel execution is applied where possible to improve CAST performance for large result sets.