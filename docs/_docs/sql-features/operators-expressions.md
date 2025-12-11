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
| `NOT` | Negates a condition | `WHERE NOT deleted` |

## Arithmetic Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `+` | Addition | `SELECT price + tax` |
| `-` | Subtraction | `SELECT total - discount` |
| `*` | Multiplication | `SELECT quantity * price` |
| `/` | Division | `SELECT total / count` |
| `%` | Modulo (remainder) | `SELECT id % 10` |

## Bitwise Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `&` | Bitwise AND | `SELECT flags & 0x0F` |
| `\|` | Bitwise OR | `SELECT flags \| 0x10` |
| `^` | Bitwise XOR | `SELECT a ^ b` |
| `~` | Bitwise NOT | `SELECT ~flags` |
| `<<` | Left shift | `SELECT 1 << 4` (returns 16) |
| `>>` | Right shift | `SELECT 16 >> 2` (returns 4) |

## String Operators

### Concatenation

```sql
-- Using || operator
SELECT first_name || ' ' || last_name AS full_name FROM users;

-- Using CONCAT function
SELECT CONCAT(first_name, ' ', last_name) AS full_name FROM users;
```

### Pattern Matching

#### LIKE (Case-Sensitive)

```sql
-- % matches any sequence of characters
SELECT * FROM products WHERE name LIKE 'Apple%';      -- Starts with 'Apple'
SELECT * FROM products WHERE name LIKE '%Phone';      -- Ends with 'Phone'
SELECT * FROM products WHERE name LIKE '%Pro%';       -- Contains 'Pro'

-- _ matches any single character
SELECT * FROM products WHERE code LIKE 'A_C';         -- Matches 'ABC', 'A1C', etc.
```

#### ILIKE (Case-Insensitive)

```sql
-- Same as LIKE but ignores case
SELECT * FROM products WHERE name ILIKE 'apple%';     -- Matches 'Apple', 'APPLE', 'apple'
SELECT * FROM users WHERE email ILIKE '%@gmail.com';
```

#### GLOB (Shell-Style Patterns)

```sql
-- * matches any sequence of characters (like % in LIKE)
SELECT * FROM files WHERE name GLOB '*.txt';

-- ? matches any single character (like _ in LIKE)
SELECT * FROM files WHERE name GLOB 'file?.dat';

-- [...] matches any character in the set
SELECT * FROM files WHERE name GLOB '[abc]*';
```

#### REGEXP (Regular Expressions)

```sql
-- Full regex pattern matching
SELECT * FROM logs WHERE message REGEXP 'error|warning';
SELECT * FROM users WHERE email REGEXP '^[a-z]+@[a-z]+\.[a-z]+$';
SELECT * FROM data WHERE value REGEXP '[0-9]{3}-[0-9]{4}';
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
