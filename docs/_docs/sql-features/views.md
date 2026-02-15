---
layout: doc
title: Views
category: SQL Features
order: 15
---

# Views

Views are named saved queries that act as virtual tables. They simplify complex queries, provide abstraction over underlying table structure, and can be referenced anywhere a table can be used.

## Creating Views

```sql
CREATE VIEW engineering AS
SELECT id, name, salary
FROM employees
WHERE department = 'Engineering';
```

Use `IF NOT EXISTS` to avoid errors when the view already exists:

```sql
CREATE VIEW IF NOT EXISTS engineering AS
SELECT * FROM employees WHERE department = 'Engineering';
```

## Dropping Views

```sql
DROP VIEW engineering;

-- Safe drop (no error if view doesn't exist)
DROP VIEW IF EXISTS engineering;
```

Dropping a view does not affect the underlying tables or their data.

## Querying Views

Views support all standard query clauses:

```sql
-- Filter view results
SELECT * FROM engineering WHERE salary > 80000;

-- Sort and limit
SELECT * FROM engineering ORDER BY salary DESC LIMIT 10;

-- DISTINCT
SELECT DISTINCT department FROM all_employees;

-- Aggregation
SELECT COUNT(*), AVG(salary) FROM engineering;
```

## Views with Joins

Views can encapsulate join logic:

```sql
CREATE VIEW order_details AS
SELECT o.id, o.order_date, c.name AS customer_name, o.amount
FROM orders o
INNER JOIN customers c ON o.customer_id = c.id;

-- Query the view like a simple table
SELECT * FROM order_details WHERE amount > 100;
```

Views can also be joined with other tables or views:

```sql
-- Join a view with a table
SELECT v.customer_name, p.product_name
FROM order_details v
JOIN products p ON v.product_id = p.id;

-- Join two views
SELECT a.name, b.total_orders
FROM active_customers a
JOIN customer_stats b ON a.id = b.customer_id;
```

## Views with Aggregation

```sql
CREATE VIEW dept_stats AS
SELECT department,
       COUNT(*) AS emp_count,
       AVG(salary) AS avg_salary
FROM employees
GROUP BY department;

SELECT * FROM dept_stats WHERE emp_count > 5;
```

## Views with Expressions

Views can include function calls and expressions:

```sql
CREATE VIEW formatted_employees AS
SELECT id,
       UPPER(name) AS name_upper,
       COALESCE(department, 'Unassigned') AS dept,
       LENGTH(name) AS name_length
FROM employees;
```

## Nested Views

Views can reference other views, up to 32 levels deep:

```sql
CREATE VIEW all_employees AS
SELECT * FROM employees;

CREATE VIEW senior_employees AS
SELECT * FROM all_employees WHERE salary > 100000;

CREATE VIEW senior_engineers AS
SELECT * FROM senior_employees WHERE department = 'Engineering';
```

The maximum nesting depth of 32 levels prevents infinite recursion from circular view definitions.

## View Metadata

```sql
-- List all views
SHOW VIEWS;

-- Show the CREATE VIEW statement for a view
SHOW CREATE VIEW engineering;
```

`SHOW CREATE VIEW` returns two columns: `View` (the view name) and `Create View` (the full CREATE VIEW statement).

## Constraints

- View names and table names share the same namespace and cannot conflict
- Views are read-only (INSERT, UPDATE, DELETE on views are not supported)
- View definitions are stored in the WAL and persist across restarts
