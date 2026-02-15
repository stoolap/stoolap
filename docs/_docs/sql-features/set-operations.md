---
layout: doc
title: Set Operations
category: SQL Features
order: 16
---

# Set Operations

Set operations combine the results of two or more SELECT queries. Stoolap supports UNION, UNION ALL, INTERSECT, and EXCEPT.

## UNION

Combines results from two queries, removing duplicate rows:

```sql
SELECT name FROM employees
UNION
SELECT name FROM contractors;
```

### UNION ALL

Keeps all rows including duplicates (faster since no deduplication is needed):

```sql
SELECT name FROM employees
UNION ALL
SELECT name FROM contractors;
```

## INTERSECT

Returns only rows that appear in both result sets:

```sql
SELECT name FROM current_employees
INTERSECT
SELECT name FROM bonus_recipients;
-- Returns employees who received a bonus
```

### INTERSECT ALL

Keeps duplicate rows in the intersection (a duplicate appears as many times as it exists in both inputs):

```sql
SELECT name FROM current_employees
INTERSECT ALL
SELECT name FROM bonus_recipients;
```

## EXCEPT

Returns rows from the first query that do not appear in the second:

```sql
SELECT name FROM all_employees
EXCEPT
SELECT name FROM terminated_employees;
-- Returns only active employees
```

Order matters: `A EXCEPT B` is different from `B EXCEPT A`.

### EXCEPT ALL

Keeps duplicates: each matching row from the second query removes one occurrence from the first:

```sql
SELECT name FROM all_employees
EXCEPT ALL
SELECT name FROM terminated_employees;
```

## Chaining Multiple Operations

Set operations can be chained:

```sql
SELECT id FROM table1
UNION
SELECT id FROM table2
UNION
SELECT id FROM table3
UNION
SELECT id FROM table4;
```

## ORDER BY and LIMIT

ORDER BY and LIMIT apply to the entire combined result set:

```sql
SELECT name, salary FROM employees
UNION
SELECT name, salary FROM contractors
ORDER BY salary DESC
LIMIT 10;
```

## With WHERE Clauses

Each SELECT in a set operation can have its own WHERE clause:

```sql
SELECT val FROM sales WHERE val > 1000
UNION
SELECT val FROM returns WHERE val < 500;
```

## Column Requirements

All queries in a set operation must have the same number of columns. Column names are taken from the first query:

```sql
-- Both queries must return 2 columns
SELECT id, name FROM employees
UNION
SELECT id, name FROM contractors;
```

## Duplicate Handling Summary

| Operation | Duplicates |
|-----------|-----------|
| `UNION` | Removed |
| `UNION ALL` | Kept |
| `INTERSECT` | Removed |
| `INTERSECT ALL` | Kept (min count from both sides) |
| `EXCEPT` | Removed |
| `EXCEPT ALL` | Kept (subtracts one per match) |
