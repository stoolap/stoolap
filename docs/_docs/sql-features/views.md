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

## Column Names

`CREATE VIEW` runs the view's query once and records the names of its columns with the view. Rows are not fetched, but a query that has to read its input before giving its first row, such as one with GROUP BY or ORDER BY, does that work once. A query that cannot run, because it reads a table or a column that does not exist, is refused, and so is a query that names two columns alike:

```sql
CREATE VIEW v AS SELECT a, a FROM t;        -- error: column 'a' is named more than once
CREATE VIEW v AS SELECT a, a AS a2 FROM t;  -- columns a, a2
```

A view created by an earlier release has no recorded names: the first query that reads it records the ones its query gives, and a statement that prepares a subquery over it runs it once more for that, the only time it does. The recorded names are the view's columns from then on. Renaming a column of a table the view reads keeps the view's name for it. If the view's query later gives a different number of columns, for example `SELECT *` after `ALTER TABLE ... ADD COLUMN`, reading the view is an error until the view is created again.

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

A view's rows are read as the query needs them. In a statement that only filters and projects them, with no subquery, aggregate, window function, DISTINCT or set operation, a LIMIT without ORDER BY stops reading once it has its rows, so a row past it is never computed, and a value that would fail in that row raises no error. Other statements may read every row first.

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
-- Join a view with a table (the view must include the join column)
SELECT v.customer_name, v.amount
FROM order_details v
WHERE v.amount > 500;

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

The maximum nesting depth of 32 levels prevents infinite recursion from circular view definitions. Since `CREATE VIEW` runs the view's query, a view nested deeper than that is refused when it is created.

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
- View definitions and their column names are stored in the WAL and persist across restarts
- A view cannot take parameters (`$1`, `?`, `:name`): it is stored SQL with no values of its own, so `CREATE VIEW` refuses one, and reading a view stored with one is an error
- A view's query follows the qualifier rule of every statement: `CREATE VIEW` refuses a qualified name that names no FROM entry of the view's query, and reading a view stored with one is an error. A query that reads the view names it by the view's name or alias, never by a table inside the view
- A view's query reads the tables and views it names, never a CTE of the statement that reads the view: in `WITH u AS (...) SELECT * FROM v`, a `u` inside `v` is still the table `u`
