---
layout: doc
title: Operators and Expressions
category: SQL Features
order: 6
---

# Operators and Expressions

Stoolap supports a comprehensive set of SQL operators for building expressions in SELECT, WHERE, HAVING, and other clauses.

## Comparison Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `=` | Equal | `WHERE id = 5` |
| `<>` or `!=` | Not equal | `WHERE status <> 'deleted'` |
| `<` | Less than | `WHERE price < 100` |
| `<=` | Less than or equal | `WHERE age <= 65` |
| `>` | Greater than | `WHERE score > 90` |
| `>=` | Greater than or equal | `WHERE date >= '2024-01-01'` |

## Logical Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `AND` | Both conditions true | `WHERE a > 1 AND b < 10` |
| `OR` | Either condition true | `WHERE status = 'active' OR status = 'pending'` |
| `XOR` | Exactly one condition true | `WHERE a XOR b` |
| `NOT` | Negates a condition | `WHERE NOT deleted` |

An error a condition raises on a row, such as an `Integer overflow` from `ABS`, fails the statement, for `SELECT`, `UPDATE` and `DELETE`, on recent and on checkpointed rows alike. `AND` and `OR` evaluate their operands left to right and stop at the first one that decides the result, so `WHERE id = 1 OR ABS(x) >= 0` raises nothing for the row with `id = 1`, while `WHERE ABS(x) >= 0 OR id = 1` fails on it. `NOT` passes an error on. A row an index or a `LIMIT` leaves unread is never asked, so it raises nothing.

`XOR` returns true when exactly one operand is true:

```sql
SELECT true XOR false;   -- true
SELECT true XOR true;    -- false
SELECT false XOR false;  -- false
```

## Arithmetic Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `+` | Addition | `SELECT price + tax` |
| `-` | Subtraction | `SELECT total - discount` |
| `*` | Multiplication | `SELECT quantity * price` |
| `/` | Division | `SELECT total / count` |
| `%` | Modulo (remainder) | `SELECT id % 10` |

INTEGER arithmetic is exact. A result outside the INTEGER range is an `Integer overflow` error, for every operator and for `ABS`: `9223372036854775807 + 1`, `-(-9223372036854775807 - 1)`, `(-9223372036854775807 - 1) / -1` and `ABS(-9223372036854775807 - 1)` are errors. Division truncates toward zero (`7 / 2` is 3, `-7 / 2` is -3), and the remainder takes the sign of the dividend (`-7 % 2` is -1). Dividing by zero, with `/`, `%` or `MOD`, returns NULL.

## Bitwise Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `&` | Bitwise AND | `SELECT flags & 0x0F` |
| `\|` | Bitwise OR | `SELECT flags \| 0x10` |
| `^` | Bitwise XOR | `SELECT a ^ b` |
| `~` | Bitwise NOT | `SELECT ~flags` |
| `<<` | Left shift | `SELECT 1 << 4` (returns 16) |
| `>>` | Right shift | `SELECT 16 >> 2` (returns 4) |

A shift moves the bits of the 64-bit INTEGER by 0 to 63 places: `1 << 63` is the smallest INTEGER, and `>>` keeps the sign (`-8 >> 1` is -4). A negative shift amount, or one of 64 or more, is an error.

## String Operators

### Concatenation

```sql
-- Using || operator
SELECT first_name || ' ' || last_name AS full_name FROM users;

-- Using CONCAT function
SELECT CONCAT(first_name, ' ', last_name) AS full_name FROM users;
```

### Pattern Matching

LIKE, ILIKE, GLOB and REGEXP read both the value and the pattern as text, as `CAST(... AS TEXT)` gives it, whatever their type. So `15 LIKE '1%'` is true, a TIMESTAMP is matched against its text `'2024-01-01T12:00:00+00:00'`, and `DATE '2024-01-01'` as a pattern is the text `'2024-01-01T00:00:00+00:00'`. A NULL on either side gives NULL, so `NOT (x LIKE p)` keeps no row where `x` is NULL. ILIKE folds case for any character, not only ASCII ones: `'Émile' ILIKE 'é%'` is true.

#### LIKE (Case-Sensitive)

```sql
-- % matches any sequence of characters
SELECT * FROM products WHERE name LIKE 'Apple%';      -- Starts with 'Apple'
SELECT * FROM products WHERE name LIKE '%Phone';      -- Ends with 'Phone'
SELECT * FROM products WHERE name LIKE '%Pro%';       -- Contains 'Pro'

-- _ matches any single character
SELECT * FROM products WHERE code LIKE 'A_C';         -- Matches 'ABC', 'A1C', etc.

-- NOT LIKE
SELECT * FROM products WHERE name NOT LIKE '%test%';
```

#### ILIKE (Case-Insensitive)

```sql
-- Same as LIKE but ignores case
SELECT * FROM products WHERE name ILIKE 'apple%';     -- Matches 'Apple', 'APPLE', 'apple'
SELECT * FROM users WHERE email ILIKE '%@gmail.com';

-- NOT ILIKE
SELECT * FROM users WHERE name NOT ILIKE 'admin%';
```

#### LIKE with ESCAPE

Use the ESCAPE clause when your pattern needs to match literal `%` or `_` characters:

```sql
-- Match values containing a literal '%' character
SELECT * FROM metrics WHERE label LIKE '%!%%' ESCAPE '!';

-- Match values containing a literal '_' character
SELECT * FROM codes WHERE code LIKE 'A!_B' ESCAPE '!';
```

The escape character can be any single character. The character immediately after the escape is treated as a literal instead of a wildcard.

#### GLOB (Shell-Style Patterns)

```sql
-- * matches any sequence of characters (like % in LIKE)
SELECT * FROM files WHERE name GLOB '*.txt';

-- ? matches any single character (like _ in LIKE)
SELECT * FROM files WHERE name GLOB 'file?.dat';

-- [...] matches any character in the set
SELECT * FROM files WHERE name GLOB '[abc]*';

-- NOT GLOB
SELECT * FROM files WHERE name NOT GLOB '*.tmp';
```

#### REGEXP (Regular Expressions)

```sql
-- Full regex pattern matching (Rust regex syntax)
SELECT * FROM logs WHERE message REGEXP 'error|warning';
SELECT * FROM users WHERE email REGEXP '^[a-z]+@[a-z]+\.[a-z]+$';
SELECT * FROM data WHERE value REGEXP '[0-9]{3}-[0-9]{4}';

-- NOT REGEXP
SELECT * FROM logs WHERE message NOT REGEXP 'debug|trace';
```

Invalid regex patterns return an error instead of silently matching nothing.

#### Parameterized Patterns

All pattern matching operators support parameterized patterns. The pattern is compiled once and reused for every row:

```sql
-- LIKE with parameter
SELECT * FROM products WHERE name LIKE $1;

-- REGEXP with parameter
SELECT * FROM logs WHERE message REGEXP $1;

-- GLOB with parameter
SELECT * FROM files WHERE name GLOB $1;
```

## Range Operators

### BETWEEN

```sql
-- Inclusive range check
SELECT * FROM products WHERE price BETWEEN 10 AND 100;

-- Equivalent to:
SELECT * FROM products WHERE price >= 10 AND price <= 100;

-- Works with dates
SELECT * FROM orders WHERE order_date BETWEEN '2024-01-01' AND '2024-12-31';

-- NOT BETWEEN
SELECT * FROM products WHERE price NOT BETWEEN 10 AND 100;
```

### IN

```sql
-- Check if value is in a list
SELECT * FROM products WHERE category IN ('Electronics', 'Computers', 'Phones');

-- With numbers
SELECT * FROM orders WHERE status_id IN (1, 2, 3);

-- NOT IN
SELECT * FROM products WHERE category NOT IN ('Discontinued', 'Archived');

-- With subquery
SELECT * FROM customers WHERE id IN (SELECT customer_id FROM orders WHERE total > 1000);
```

## JSON Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `->` | JSON access (returns JSON) | `data -> 'key'` |
| `->>` | JSON access (returns TEXT) | `data ->> 'key'` |

```sql
-- Extract as JSON (preserves JSON type)
SELECT data -> 'address' FROM users;

-- Extract as text (always returns a string)
SELECT data ->> 'name' FROM users;

-- Nested access
SELECT data -> 'address' ->> 'city' FROM users;

-- In WHERE clause
SELECT * FROM products WHERE attributes ->> 'color' = 'red';
```

These are shorthand for `JSON_EXTRACT`. See [JSON Support]({% link _docs/data-types/json-support.md %}) for more JSON functions.

## NULL Operators

```sql
-- Check for NULL
SELECT * FROM users WHERE deleted_at IS NULL;

-- Check for NOT NULL
SELECT * FROM users WHERE email IS NOT NULL;

-- Note: = and <> don't work with NULL
SELECT * FROM users WHERE value = NULL;    -- Always returns no rows!
SELECT * FROM users WHERE value IS NULL;   -- Correct way
```

### IS DISTINCT FROM / IS NOT DISTINCT FROM

NULL-safe comparison operators that treat NULL as a comparable value:

```sql
-- IS DISTINCT FROM: returns true when values differ (treats NULLs as equal)
SELECT * FROM t WHERE a IS DISTINCT FROM b;

-- NULL IS DISTINCT FROM NULL → false (they are "the same")
-- NULL IS DISTINCT FROM 1   → true  (they differ)
-- 1 IS DISTINCT FROM 1      → false (they are equal)
-- 1 IS DISTINCT FROM 2      → true  (they differ)

-- IS NOT DISTINCT FROM: NULL-safe equality
SELECT * FROM t WHERE a IS NOT DISTINCT FROM b;

-- NULL IS NOT DISTINCT FROM NULL → true
-- NULL IS NOT DISTINCT FROM 1   → false
-- 1 IS NOT DISTINCT FROM 1      → true
```

Unlike `=` and `<>`, these operators never return NULL. They always produce true or false.

## CASE Expression

```sql
-- Simple CASE
SELECT name,
    CASE status
        WHEN 'A' THEN 'Active'
        WHEN 'I' THEN 'Inactive'
        WHEN 'P' THEN 'Pending'
        ELSE 'Unknown'
    END AS status_name
FROM users;

-- Searched CASE
SELECT name, salary,
    CASE
        WHEN salary >= 100000 THEN 'Executive'
        WHEN salary >= 70000 THEN 'Senior'
        WHEN salary >= 40000 THEN 'Mid-level'
        ELSE 'Entry-level'
    END AS level
FROM employees;
```

## Operator Precedence

From highest to lowest:

1. `()` - Parentheses
2. `~` - Bitwise NOT
3. `*`, `/`, `%` - Multiplication, division, modulo
4. `+`, `-` - Addition, subtraction
5. `<<`, `>>` - Bit shifts
6. `&` - Bitwise AND
7. `^` - Bitwise XOR
8. `|` - Bitwise OR
9. `=`, `<>`, `<`, `<=`, `>`, `>=` - Comparisons
10. `NOT` - Logical NOT
11. `AND` - Logical AND
12. `OR` - Logical OR

Use parentheses to control evaluation order:

```sql
-- Without parentheses: AND has higher precedence than OR
SELECT * FROM products WHERE category = 'A' OR category = 'B' AND price > 100;
-- Equivalent to: category = 'A' OR (category = 'B' AND price > 100)

-- With parentheses: explicit grouping
SELECT * FROM products WHERE (category = 'A' OR category = 'B') AND price > 100;
```

## Examples

### Complex WHERE Clauses

```sql
SELECT * FROM orders
WHERE status IN ('pending', 'processing')
  AND total BETWEEN 100 AND 1000
  AND customer_name ILIKE '%smith%'
  AND created_at >= DATE_SUB(NOW(), 30, 'day');
```

### Computed Columns

```sql
SELECT
    product_name,
    quantity,
    unit_price,
    quantity * unit_price AS line_total,
    CASE WHEN quantity >= 10 THEN 0.1 ELSE 0 END AS discount_rate,
    quantity * unit_price * (1 - CASE WHEN quantity >= 10 THEN 0.1 ELSE 0 END) AS final_price
FROM order_items;
```

### Bitwise Flags

```sql
-- Check if specific bit is set
SELECT * FROM permissions WHERE (flags & 0x04) = 0x04;  -- Check bit 2

-- Set a bit
UPDATE permissions SET flags = flags | 0x04 WHERE user_id = 1;

-- Clear a bit
UPDATE permissions SET flags = flags & ~0x04 WHERE user_id = 1;
```
