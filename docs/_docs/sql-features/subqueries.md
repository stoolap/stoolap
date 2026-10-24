---
layout: doc
title: Subqueries
category: SQL Features
order: 3
---

# Subqueries in Stoolap

Stoolap supports both correlated and non-correlated subqueries in SQL statements, allowing you to use the results of one query within another query. This enables complex data operations and filtering based on dynamic conditions.

## Supported Subquery Types

### EXISTS/NOT EXISTS Subqueries

The EXISTS operator returns true if the subquery returns at least one row, and false otherwise. NOT EXISTS returns the opposite.

#### Basic Syntax

```sql
-- Check if any orders exist
SELECT * FROM customers
WHERE EXISTS (
    SELECT 1 FROM orders
);

-- Check if no high-value orders exist
SELECT * FROM products
WHERE NOT EXISTS (
    SELECT 1 FROM orders WHERE amount > 1000
);

-- DELETE when any discontinued items exist
DELETE FROM promotions
WHERE EXISTS (
    SELECT 1 FROM discontinued_items
);

-- UPDATE when no pending orders exist
UPDATE products
SET in_stock = true
WHERE NOT EXISTS (
    SELECT 1 FROM orders WHERE status = 'pending'
);
```

#### EXISTS Optimization

- EXISTS stops scanning as soon as it finds the first matching row
- The SELECT list in the EXISTS subquery is ignored - you can use `SELECT 1` or `SELECT *`
- EXISTS returns FALSE if the subquery returns no rows
- A row whose WHERE fails with an error does not decide EXISTS when another row passes: in a [prepared subquery](#prepared-subqueries), EXISTS is TRUE when a row passes, else the first error a row raised is the statement's error, else FALSE. The equalities that correlate the subquery with the outer row (`u.k = t.k`) choose the rows its WHERE is asked of, so a row with another key, or a NULL one, raises no error for this outer row

#### Common Use Cases

```sql
-- Check if any high-value orders exist
SELECT 'High value orders exist' as status
WHERE EXISTS (SELECT 1 FROM orders WHERE amount > 1000);

-- Find all products when no returns exist
SELECT * FROM products
WHERE NOT EXISTS (SELECT 1 FROM returns);

-- Delete temporary data when processing is complete
DELETE FROM temp_data
WHERE EXISTS (SELECT 1 FROM job_status WHERE status = 'completed');
```

### IN Subqueries

The IN subquery allows you to filter rows based on values returned by another query.

#### Basic Syntax

```sql
-- Select rows where a column matches values from a subquery
SELECT * FROM orders 
WHERE customer_id IN (
    SELECT id FROM customers WHERE country = 'USA'
);

-- Delete rows based on subquery results
DELETE FROM orders 
WHERE customer_id IN (
    SELECT id FROM customers WHERE total_spent > 1000
);

-- Update rows based on subquery results
UPDATE products 
SET discount = 0.15 
WHERE category IN (
    SELECT name FROM categories WHERE is_premium = true
);
```

#### NOT IN Subqueries

You can also use NOT IN to exclude rows that match the subquery results:

```sql
-- Find customers who haven't placed any orders
SELECT * FROM customers 
WHERE id NOT IN (
    SELECT DISTINCT customer_id FROM orders
);

-- Delete inactive users
DELETE FROM users 
WHERE id NOT IN (
    SELECT user_id FROM login_history 
    WHERE login_date > DATE_SUB(NOW(), 30, 'day')
);
```

### Scalar Subqueries

Scalar subqueries return a single value and can be used in:
- WHERE clauses with comparison operators
- SELECT expressions to compute values

#### Basic Syntax

```sql
-- Compare with aggregate functions
SELECT * FROM products 
WHERE price > (SELECT AVG(price) FROM products);

-- Use in SELECT expressions
SELECT name, price, 
       (SELECT AVG(price) FROM products) as avg_price
FROM products;

DELETE FROM orders 
WHERE amount < (SELECT MIN(amount) FROM orders WHERE status = 'completed');

UPDATE employees 
SET bonus = 1000 
WHERE sales > (SELECT AVG(sales) FROM employees WHERE department = 'Sales');
```

#### Comparison Operators

Scalar subqueries work with all standard comparison operators:
- `=` (equal)
- `!=` or `<>` (not equal)  
- `>` (greater than)
- `<` (less than)
- `>=` (greater than or equal)
- `<=` (less than or equal)

#### NULL Handling

When a scalar subquery returns no rows, it evaluates to NULL:

```sql
-- This returns no rows because NULL comparisons are always false
SELECT * FROM products 
WHERE price > (SELECT MAX(price) FROM products WHERE category = 'NonExistent');

-- Use COALESCE to handle NULL results
SELECT * FROM products 
WHERE price > COALESCE((SELECT MAX(price) FROM products WHERE category = 'Books'), 0);
```

### ANY/SOME and ALL Subqueries

Use ANY (or SOME) and ALL to compare a value against all values returned by a subquery.

#### ANY/SOME

Returns true if the comparison is true for at least one value returned by the subquery. SOME is a synonym for ANY.

```sql
-- Find products more expensive than at least one electronics product
SELECT * FROM products
WHERE price > ANY (SELECT price FROM products WHERE category = 'Electronics');

-- Find employees earning more than some manager
SELECT * FROM employees
WHERE salary > SOME (SELECT salary FROM employees WHERE role = 'Manager');
```

#### ALL

Returns true if the comparison is true for all values returned by the subquery.

```sql
-- Find products more expensive than all books
SELECT * FROM products
WHERE price > ALL (SELECT price FROM products WHERE category = 'Books');

-- Find the highest paid employee in each department
SELECT * FROM employees e
WHERE salary >= ALL (SELECT salary FROM employees WHERE department = e.department);
```

#### Comparison Operators with ANY/ALL

Both ANY and ALL work with all comparison operators:

```sql
-- Equal to any value (equivalent to IN)
WHERE value = ANY (SELECT ...)

-- Not equal to all values (equivalent to NOT IN)
WHERE value <> ALL (SELECT ...)

-- Greater than at least one value
WHERE value > ANY (SELECT ...)

-- Greater than all values
WHERE value > ALL (SELECT ...)
```

The value on the left of `IN`, `ANY` or `ALL` may be a `SELECT` alias in a `WHERE` or a `HAVING`, and an aggregate in a statement with a `GROUP BY`, such as `HAVING COUNT(*) IN (SELECT n FROM limits)`.

#### Error Handling

Scalar subqueries must return at most one row. If multiple rows are returned, an error occurs:

```sql
-- Error: Subquery returns more than one row
SELECT * FROM products
WHERE price > (SELECT price FROM products WHERE category = 'Electronics');
-- Use aggregate functions to ensure single value
SELECT * FROM products
WHERE price > (SELECT MAX(price) FROM products WHERE category = 'Electronics');
```

#### Subqueries without FROM

A subquery without FROM reads one row that has no columns. Its WHERE, GROUP BY, HAVING and aggregates apply to that row as they would to a table holding it, and a bare name in it refers to the enclosing query:

```sql
-- 2, for each row of t
SELECT * FROM t WHERE a = (SELECT 2);

-- TRUE where a > 1: the one row passes HAVING only there
SELECT * FROM t WHERE EXISTS (SELECT 1 HAVING t.a > 1);

-- TRUE for every row: an aggregate without GROUP BY answers one row
SELECT * FROM t WHERE EXISTS (SELECT COUNT(*) WHERE t.a > 5);

-- reads t.a
SELECT * FROM t WHERE a = (SELECT a);
```

### Correlated Subqueries

Correlated subqueries reference columns from the outer query, allowing row-by-row evaluation. The subquery is re-executed for each row of the outer query.

#### Basic Syntax

```sql
-- Find employees earning above their department average
SELECT name, department, salary
FROM employees e1
WHERE salary > (
    SELECT AVG(salary)
    FROM employees e2
    WHERE e2.department = e1.department
);

-- Find products priced above category average
SELECT p1.name, p1.category, p1.price
FROM products p1
WHERE p1.price > (
    SELECT AVG(p2.price)
    FROM products p2
    WHERE p2.category = p1.category
);
```

#### Correlated EXISTS

EXISTS with correlation is powerful for checking related records:

```sql
-- Find customers who have placed orders over $100
SELECT * FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o
    WHERE o.customer_id = c.id AND o.amount > 100
);

-- Find products that have never been ordered
SELECT * FROM products p
WHERE NOT EXISTS (
    SELECT 1 FROM order_items oi
    WHERE oi.product_id = p.id
);
```

#### Correlated Scalar Subqueries in SELECT

Use correlated subqueries to compute per-row values:

```sql
-- Get each customer's maximum order amount
SELECT
    c.name,
    c.email,
    (SELECT MAX(amount) FROM orders o WHERE o.customer_id = c.id) as max_order
FROM customers c;

-- Get each product's order count
SELECT
    p.name,
    p.price,
    (SELECT COUNT(*) FROM order_items oi WHERE oi.product_id = p.id) as times_ordered
FROM products p;
```

#### Correlated IN Subqueries

```sql
-- Find customers in the same city as VIP customers
SELECT * FROM customers c1
WHERE c1.city IN (
    SELECT c2.city FROM customers c2 WHERE c2.status = 'VIP'
);
```

### Derived Tables (Subqueries in FROM)

Subqueries in the FROM clause create temporary result sets called derived tables:

```sql
-- Use a derived table to pre-filter data
SELECT * FROM (
    SELECT id, name, price
    FROM products
    WHERE price > 100
) AS expensive_products
WHERE name LIKE '%Pro%';

-- Aggregate then join
SELECT c.name, order_summary.total_orders, order_summary.total_spent
FROM customers c
JOIN (
    SELECT customer_id, COUNT(*) as total_orders, SUM(amount) as total_spent
    FROM orders
    GROUP BY customer_id
) AS order_summary ON c.id = order_summary.customer_id;
```

### Subqueries in UPDATE SET Clause

Scalar subqueries can be used in the SET clause of UPDATE statements:

```sql
-- Update salary to department average
UPDATE employees
SET salary = (SELECT AVG(salary) FROM employees WHERE department = 'Engineering')
WHERE department = 'Engineering' AND salary < 50000;

-- Update price based on category average
UPDATE products
SET price = (SELECT AVG(price) FROM products WHERE category = 'Electronics') * 1.1
WHERE id = 5;
```

### Subqueries in ON CONFLICT

A subquery in `ON CONFLICT ... DO UPDATE SET`, in its `WHERE`, or in `ON DUPLICATE KEY UPDATE` sees the tables as the statement found them, as a subquery in `VALUES` does. A value that reads neither the row met nor the `excluded` row is read once, before the statement writes a row; one that reads them is read for each conflict, and a bare column name in it is the row met's:

```sql
-- The highest score the table held before this statement
INSERT INTO scores VALUES (1, 0)
ON CONFLICT (id) DO UPDATE SET best = (SELECT MAX(best) FROM scores);

-- A lookup by the row that conflicted
INSERT INTO orders VALUES (7, 'B', 0)
ON CONFLICT (id) DO UPDATE SET price = (SELECT price FROM catalog WHERE catalog.code = excluded.code);
```

Outside an explicit transaction the statement reads one snapshot. Inside one, a value read for each conflict would see the rows this statement has already written, so such a value may not read the target table there, directly or through a view; the statement fails with an error instead.

### Subqueries in RETURNING

A subquery in the `RETURNING` of an `INSERT`, `UPDATE` or `DELETE` sees the tables as the statement found them, and a column of the target in it names the row returned: the row written, or for `DELETE` the row deleted. A value that reads no row returned is read once, before the statement writes a row; one that reads it is read for each row returned:

```sql
-- The category's name for each product the statement changed
UPDATE products SET price = price * 1.1 WHERE category_id = 3
RETURNING id, (SELECT name FROM categories WHERE categories.id = products.category_id);

-- How many orders the table held before this statement
DELETE FROM orders WHERE created < '2024-01-01'
RETURNING id, (SELECT COUNT(*) FROM orders);
```

A subquery that reads the target table reads it as the statement found it, so for an `UPDATE` it returns a row's old values. Inside an explicit transaction a value read for each row returned may not read the target table, as in `ON CONFLICT`; the statement fails with an error instead.

### Subqueries in LIMIT and OFFSET

A subquery in `LIMIT` or `OFFSET` is read once, before the query reads a row, under the same snapshot as the query. Its value counts as the same value written as a literal: a whole number of zero or more, and anything else, NULL included (a subquery with no row is NULL), is an error. It reads no row of the query, or of an enclosing one:

```sql
-- As many products as there are featured categories
SELECT * FROM products ORDER BY price DESC
LIMIT (SELECT COUNT(*) FROM categories WHERE featured = true);

-- Error: LIMIT must be an integer (the subquery's MAX over no row is NULL)
SELECT * FROM products LIMIT (SELECT MAX(id) FROM categories WHERE 1 = 0);
```

A subquery block that runs once for each row of its enclosing query, because its `WHERE` reads that row, cannot hold a `LIMIT` or `OFFSET` subquery. A `LIMIT` or `OFFSET` subquery may read a CTE of the statement's `WITH`, which comes before a table of its name.

## How Subqueries Work

Stoolap processes subqueries differently based on their type:

### Non-Correlated Subqueries

For non-correlated subqueries, Stoolap uses a two-phase approach:

1. **Evaluation Phase**: The subquery is executed first and its results are collected
2. **Substitution Phase**: The subquery is replaced with its results before the main query executes

This approach ensures that subqueries are properly evaluated before being used in the main query's WHERE clause. Non-correlated subqueries are executed only once.

### Correlated Subqueries

For correlated subqueries (those that reference the outer query):

1. **Outer Row Processing**: For each row in the outer query
2. **Subquery Execution**: The subquery is executed with the current outer row's values substituted
3. **Result Application**: The subquery result is used to filter or compute values for that outer row

This means correlated subqueries may be executed multiple times (once per outer row), so they can be slower for large datasets. Consider using JOINs for better performance when possible.

### Prepared Subqueries

A SELECT of one shape is prepared once per cached plan instead of rebuilt for every outer row. The shape:

- `FROM` one table, view or CTE of the statement's `WITH` (not a join, no `AS OF`), one derived table, one `VALUES` list without a subquery in its rows, or one table function such as `generate_series`, and a `WHERE`, a SELECT list or an `ORDER BY` that holds at least one scalar subquery, `EXISTS`, `NOT EXISTS`, `IN (subquery)`, `NOT IN (subquery)`, or a comparison with `ANY`, `SOME` or `ALL`, with one value, not a tuple, on the left
- no `DISTINCT`, `GROUP BY`, `HAVING`, window functions or set operations; a `WITH` runs its CTEs first, as statements of their own; no aggregate in the SELECT list or `ORDER BY`, and no `ORDER BY` key holding a subquery that names a SELECT alias; `LIMIT` and `OFFSET` as literals or parameters
- each subquery reads one table, one view, one CTE of a `WITH` around the statement (not of the statement's own `WITH`), one `VALUES` list whose values read no column, or one table function whose arguments read no column, with an optional `WHERE`; it may hold further subqueries of these forms, up to 32 scopes in one chain, and no `DISTINCT`. Unless it aggregates, it may hold an `ORDER BY` of expressions, not positions, and a `LIMIT` and `OFFSET` as literals or parameters. A scalar subquery and the subquery of an `IN`, `ANY` or `ALL` select one expression; an `EXISTS` subquery may select anything, and its SELECT list is never evaluated
- a subquery may aggregate: aggregate calls in its SELECT list and `HAVING`, with `DISTINCT`, `FILTER` or `ORDER BY` and any argument after the first a literal, and a `GROUP BY` of columns, expressions, SELECT aliases or positions, without `ROLLUP`, `CUBE` or `GROUPING SETS`. A name in its `HAVING` reads a SELECT alias as any grouped statement's `HAVING` does: the first item of that name, ahead of a column the `GROUP BY` does not name. Such a subquery holds further subqueries only in its `WHERE`
- the expressions are columns, literals, parameters, arithmetic, comparisons, `AND`, `OR`, `NOT`, `IS [NOT] NULL`, `IS [NOT] DISTINCT FROM`, `BETWEEN`, `IN (list)`, `LIKE`, `CASE`, `CAST` and scalar function calls

`EXPLAIN` shows `Prepared: yes` for such a statement, or `Prepared: no (reason)` for a WHERE, SELECT list or `ORDER BY` with a subquery that does not fit. For a prepared statement:

- every value a subquery reads from an enclosing query becomes a parameter of the subquery, so the subquery's own comparisons narrow its scan, through its primary key or an index
- a name resolves to the nearest scope that has it: a qualified name `q.c` to the nearest query whose table or alias is `q`, an unqualified name to the subquery's own table first and then to each enclosing query outward
- a qualifier that names no scope is an error, as is a column its table does not have, and a named parameter the statement is given no value for. The qualifier rule is the one every statement follows, described under [SELECT]({% link _docs/sql-commands/sql-commands.md %})
- a value compared with a column, or tested against one with `IS [NOT] DISTINCT FROM`, is read as the column's type, the way an index or table scan reads it, whether or not the scan takes the comparison: `b = 'true'` matches a `TRUE` row of a `BOOLEAN` column and `s = TRUE` matches `'true'` in a `TEXT` one. A `TEXT` that is no number against a number column, and a number against a `TEXT` column, stay as they are and compare by the numeric rules above. A number against a `BOOLEAN` or `TIMESTAMP` column, and a `BOOLEAN` against a number column, stay as they are too: the two never compare equal, and their ordering is unknown. A `JSON` or `VECTOR` comparison is never handed to the scan
- a view or a derived table runs its own query once per execution, under the statement's snapshot, and the statement's `WHERE` is asked of each row it returns. A view is read as it is defined when the statement runs, so a view dropped and created again with another query answers from the new one. A `VALUES` list's values are evaluated once per execution the same way, a parameter among them taking the execution's value. A CTE's rows are the ones its `WITH` gave it, run under the same snapshot as the statement's subqueries. A table function's arguments are evaluated once per execution, and it generates only the range the `WHERE`'s constant comparisons on its column allow. Their columns have no column type, so a value compared with one is not read as a column's type: it compares by the rules above as it stands. A view, a derived table, a `VALUES` list or a CTE may return two columns with one name: `*` returns each by its position, and a reference to that name, qualified or not, in the statement or in one of its subqueries, is an error (`column reference 'x' is ambiguous`)
- a subquery that reads no enclosing query runs at most once per execution, and only when a row needs it: a `CASE` arm or the right side of an `OR` that no row reaches does not run it. A subquery compared directly with a column in a top-level `AND` of the statement runs once before the scan, so the scan can use the column's index. So does one a column is tested against with `IN`, `NOT IN`, `ANY` or `ALL` in a top-level `AND`: the scan tests its rows against the members, and an error a member raised still belongs to the row that needs the answer
- a subquery in the SELECT list or the `ORDER BY` is asked of each row the statement returns, a correlated one reading that row: one in a SELECT item is asked only of the rows `LIMIT` and `OFFSET` keep, after the `ORDER BY` chose them, unless an `ORDER BY` key reads that item; one in an `ORDER BY` key is asked of every row the sort reads. Every other computed SELECT item of such a statement is computed for those rows too, so an expression that fails on a row the `LIMIT` drops raises no error. One that reads no enclosing query is read once before the statement reads a row, so it raises its error even when no row is returned. A SELECT item holding a subquery is named as the SELECT list names any computed expression, by its position (`expr2` for the second item), unless it carries an alias
- an `EXISTS` that reads its enclosing query only through equalities `own column = enclosing column`, each pair of one type among `INTEGER`, `TEXT`, `BOOLEAN` and `TIMESTAMP` or an enclosing column without a declared type (one of a derived table, a view, a CTE, a `VALUES` list or a join), and calls no volatile function, may read its table once per execution, grouped by those columns, and answer each outer row by one lookup; an enclosing value of another type than its own column's is answered by running the subquery for that row. It does so unless an index on its column serves a probe per outer row more cheaply: for a `LIMIT` of at most 500 or for few outer rows, when its `WHERE` holds nothing besides the equalities. Both give the same answer
- such an `EXISTS` with one equality in a top-level `AND` of the statement reads its table before the scan, and the statement reads only the rows whose column holds one of its keys, through the primary key, an index, or without a `LIMIT` the table's own candidates
- the values of `x IN (subquery)`, `x op ANY (subquery)` and `x op ALL (subquery)` are read as `x`'s column type when `x` is a column, as a value compared with a column is. A subquery that reads no enclosing query reads its values once per execution; a correlated `IN` that meets the rule for `EXISTS` above groups its values by key the same way; any other is run for each row, stopping at the first value that settles the answer
- a column `IN` an uncorrelated subquery in a top-level `AND` of the statement reads the subquery before the scan, and the statement reads only the rows holding one of its values, as for `EXISTS`, when the column's type reads them: a `TEXT` column is not looked up for number values. A `NOT IN` on the primary key reads the keys the values leave out when no value is NULL, and no row at all when one is
- a subquery that aggregates feeds each row its `WHERE` passes to its group, then answers from its groups: a scalar one gives the value of the one group `HAVING` passes, NULL for none, and an error for more than one. One that reads its enclosing query only through equalities as for `EXISTS` above, the rest of it reading no enclosing value, may read its table once per execution, grouped by those columns and its `GROUP BY` items, and answer each outer row by one lookup; a key without rows answers as no rows do, so `COUNT(*)` gives 0. One that reads no enclosing query and holds no subquery runs once per execution as a query of its own, so it takes the same aggregate paths as that query, such as an index for `MIN` or `MAX`
- a subquery with a `LIMIT` or `OFFSET` answers from the rows its `WHERE` passes, in the order of its `ORDER BY`, rows equal on every key in the order they were read, past `OFFSET` and up to `LIMIT`: `(SELECT x FROM u WHERE u.k = t.id ORDER BY x DESC NULLS LAST LIMIT 1)` gives the largest `x` of the outer row's key, and with `LIMIT 2` it fails as any scalar subquery of two rows does. A key binds as a statement's `ORDER BY` does: a bare name that is the SELECT alias, or the SELECT expression as written, sorts by the value the subquery answers with, computed once per row, and inside an expression a column of the subquery's table comes before an alias of its name. An `ORDER BY` key may read the enclosing query. An error a row it reads raises is the outer row's error, and it is never grouped by key as described above. An `EXISTS` stops at the first row past its `OFFSET`, and a table function read with no `WHERE` or `ORDER BY` generates only the rows the `LIMIT` and `OFFSET` keep. An `ORDER BY` without a `LIMIT` or `OFFSET`, or in an `EXISTS`, changes no answer: its names are bound, so a column it names must exist, but its keys are not computed
- an error a row of an `EXISTS`, `IN`, `ANY` or `ALL` subquery raises is the statement's error only when no other row settles the answer: TRUE for `EXISTS`, `IN` and `ANY` once a row passes or a value matches, FALSE for `ALL` once a value compares FALSE. A row of another key, or of a NULL key, raises nothing for the outer row. A subquery that aggregates reads every row its `WHERE` passes, so an error one raises is the outer row's error, and the same holds for an error in a value it reads from an enclosing query
- a subquery over a view, a CTE, a `VALUES` list or a table function reads its rows once per execution, before the statement reads a row, under the statement's snapshot, so an error they raise is the statement's error even when no row reaches the subquery. A view is read as it is defined when the statement runs. A table function generates only the range the subquery's comparisons of its column with values that read no column allow, and nothing when a condition that reads no column is FALSE or NULL. Their columns have no column type, as above, so the subquery's comparisons do not narrow its read and it is never grouped by key as described above: a correlated one is asked of its rows for each outer row
- the value of a scalar subquery, the answer of an `EXISTS` and the values of an `IN`, `ANY` or `ALL` read once per execution are kept for the next execution of the same statement with the same parameters, until a write to one of the tables they read; not inside a transaction, and not for a subquery that reads a view, a CTE, a `VALUES` list or a table function

An `UPDATE` or `DELETE` is prepared the same way when its `WHERE`, a `SET` value or its `RETURNING` holds a subquery and every expression and subquery in them fits the shape above; the statement's target table is the enclosing query of its subqueries. Its `SET` values may mix subqueries that read the target row with subqueries that do not, in one value or across several. A `SET` value that reads no column of the target row and holds no volatile call, such as `RANDOM()`, runs once per execution, before the statement reads a row.

Every subquery of an `UPDATE` or `DELETE`, the ones that read the target table included, sees the tables as the statement found them: a row the statement updates or deletes does not change what a subquery answers for the next row.

An `INSERT ... VALUES` is prepared the same way when a value holds a subquery and every value fits the shape above. Every value is read before the statement writes a row, so a subquery that reads the target table sees it as the statement found it, without the rows the statement inserts. A scalar subquery that answers more than one row fails the statement and writes no row.

A join of two tables, neither a view, a CTE or a derived table and neither read `AS OF`, is prepared the same way when its `ON`, its `WHERE`, its `SELECT` list or its `ORDER BY` holds a subquery and every subquery fits the shape above. A subquery in the `ON`, in a condition that reads both tables, or a correlated one is asked of the pair of rows the join tests: its enclosing query is that pair, and an unqualified name both tables have resolves to the left one. A condition on one table alone holding an uncorrelated subquery is asked of that table's rows as its scan reads them, by the rules above for one table: a value a column is compared with is read once before the join reads a row, so the scan can use the column's index. The rules above hold for the rest, so an uncorrelated subquery runs at most once per execution and only when a row needs it: one that no row reaches raises no error. A subquery in the `ON` of an outer join decides whether a pair matches, as the rest of the `ON` does. A subquery in the `SELECT` list or the `ORDER BY` is asked of each joined row the join returns, a correlated one reading that row; an uncorrelated one is read once before the join reads a row, so it raises its error even when the join returns no row. A join whose `SELECT` list or `ORDER BY` holds a subquery is not prepared when it is a `NATURAL` or `USING` join, or when an `ORDER BY` key holding a subquery names a `SELECT` alias; a join that groups its rows runs its `SELECT` list's and `HAVING`'s subqueries as a grouped statement does, below. `EXPLAIN` shows `Prepared: yes` for such a join, or `Prepared: no (reason)` for a join whose `ON`, `WHERE`, `SELECT` list or `ORDER BY` holds a subquery outside these rules. The sides and the subqueries read one snapshot of the tables. Any other join with a subquery in its `WHERE` reads an uncorrelated subquery before the join and runs a correlated one for each joined row.

A statement that groups its rows is prepared the same way when its `SELECT` list, its `HAVING`, its `GROUP BY`, its `DISTINCT ON`, its `ORDER BY` or, unless it reads a join, its `WHERE` holds a subquery and every subquery fits the shape above, unless an `ORDER BY` key holding a subquery names a `SELECT` alias or the statement also has window functions and its `ORDER BY` holds a subquery; its `WHERE` may hold subqueries too, and a `GROUP BY` alias or position counts as the item it names. Its rows may come from a table not read `AS OF`, a view, a CTE, a derived table, a `VALUES` list, a table function or a join other than a `NATURAL` or `USING` one; it may use `ROLLUP`, `CUBE`, `GROUPING SETS` or `DISTINCT`, and it may be the first branch of a set operation, whose `ORDER BY`, `LIMIT` and `OFFSET` belong to the whole set operation. A join's `ON` and `WHERE` are asked of its rows as the paragraph above says, before they group. A subquery in a `GROUP BY` item is read for each row, so a group rolled up by `ROLLUP`, `CUBE` or `GROUPING SETS` reads NULL for that item, as for any other. A subquery in the `SELECT` list, the `HAVING`, a `DISTINCT ON` key or an `ORDER BY` key outside an aggregate call is asked of each group: a correlated one's enclosing query is the group's row, so it may read only the columns the `GROUP BY` names. A subquery in the `WHERE`, in a `GROUP BY` item, or in an aggregate call's first argument, `FILTER` or `ORDER BY` is asked of each row the statement reads, a correlated one reading that row, and only of a row that reaches it: a row the `WHERE` or a `FILTER` drops, or a `CASE` arm it does not take, raises no error from it. A subquery in an aggregate call's argument after the first, such as the separator of `STRING_AGG`, is read once before the statement reads a row; one there that reads the row is an error. A `SELECT` item written as a `GROUP BY` item reads the group's value of that item. An uncorrelated subquery is read once before the statement reads a row, so it raises its error even when there is no row or group, and the statement reads its value like a parameter: an expression around it is computed for each row or group, and the statement takes the aggregate path it takes with that value written in its place. A `SELECT` item holding a subquery is named after the item as it is written, such as `(SELECT MAX(x) FROM u)`, unless it carries an alias. `EXPLAIN` shows `Prepared: yes` for such a statement, or `Prepared: no (reason)` for a grouped statement whose `SELECT` list, `HAVING`, `GROUP BY` or `ORDER BY` holds a subquery outside these rules. A `WHERE` subquery that reads no row is read once before the statement reads a row, and the scan takes its value as it takes a parameter, so a comparison of an indexed column with it can read through the index.

A statement with window functions is prepared the same way when its `SELECT` list holds a subquery, inside a window function or outside one, or its `DISTINCT ON` does, and every subquery fits the shape above, unless its `ORDER BY` holds a subquery or a window function names a window of a `WINDOW` clause. One that does not group is prepared for a subquery in its `WHERE` alone too, unless it reads a join. Its rows may come from the sources a grouped statement reads. In a statement that does not group, a subquery is asked of each row the statement reads, a correlated one reading that row, before the window functions run over the rows; one in a window function's argument, `FILTER`, `PARTITION BY` or `ORDER BY` gives that function's value for the row. In a statement that groups, the window functions run over the groups, and a subquery is asked of each row or each group as for a grouped statement above. A subquery in the argument of `NTILE`, in a window function's argument after the first or in a frame offset, and one in an aggregate call's argument after the first, is read once before the statement reads a row, since the function reads one value for the whole call; one there that reads the row is an error. A `SELECT` item holding a subquery is named after the item as it is written, as for a grouped statement.

## Examples

### Example 1: Delete Orders from High-Value Customers

```sql
-- Create sample data
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT,
    total_spent FLOAT
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    amount FLOAT
);

-- Insert sample data
INSERT INTO customers VALUES 
    (1, 'Alice', 1000.0),
    (2, 'Bob', 500.0),
    (3, 'Charlie', 2000.0);

INSERT INTO orders VALUES 
    (1, 1, 200.0),
    (2, 2, 100.0),
    (3, 3, 500.0);

-- Delete orders for customers who have spent more than 1000
DELETE FROM orders 
WHERE customer_id IN (
    SELECT id FROM customers WHERE total_spent > 1000
);
-- This will delete order #3 (Charlie's order)
```

### Example 2: Update Products in Premium Categories

```sql
-- Create sample data
CREATE TABLE categories (
    id INTEGER PRIMARY KEY,
    name TEXT,
    is_premium BOOLEAN
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT,
    category TEXT,
    price FLOAT,
    discount FLOAT
);

-- Insert sample data
INSERT INTO categories VALUES 
    (1, 'Electronics', true),
    (2, 'Books', false),
    (3, 'Clothing', true);

INSERT INTO products VALUES 
    (1, 'Laptop', 'Electronics', 1000.0, 0.0),
    (2, 'Novel', 'Books', 20.0, 0.0),
    (3, 'Shirt', 'Clothing', 50.0, 0.0);

-- Apply discount to products in premium categories
UPDATE products 
SET discount = 0.15 
WHERE category IN (
    SELECT name FROM categories WHERE is_premium = true
);
-- This will update Laptop and Shirt with 15% discount
```

### Example 3: Scalar Subquery Examples

```sql
-- Find products priced above average
SELECT name, price FROM products 
WHERE price > (SELECT AVG(price) FROM products);

-- Delete orders below average for completed orders
DELETE FROM orders 
WHERE amount < (
    SELECT AVG(amount) FROM orders WHERE status = 'completed'
) AND status = 'pending';

-- Update salaries below company average
UPDATE employees 
SET salary = salary * 1.1 
WHERE salary < (
    SELECT AVG(salary) FROM employees
);
```

### Example 4: EXISTS/NOT EXISTS Examples

```sql
-- Check if any premium products exist
SELECT 'Premium products available' as message
WHERE EXISTS (
    SELECT 1 FROM products WHERE price > 1000
);

-- List all customers when no active orders exist
SELECT * FROM customers
WHERE NOT EXISTS (
    SELECT 1 FROM orders WHERE status = 'active'
);

-- Delete old logs when archive is complete
DELETE FROM logs
WHERE NOT EXISTS (
    SELECT 1 FROM archive_status WHERE status = 'in_progress'
);

-- Update all customers when recent orders exist
UPDATE customers
SET status = 'active'
WHERE EXISTS (
    SELECT 1 FROM orders 
    WHERE order_date > DATE_SUB(NOW(), 30, 'day')
);
```

### Example 5: Find Orphaned Records

```sql
-- Find products that reference non-existent categories
SELECT * FROM products 
WHERE category_id NOT IN (
    SELECT id FROM categories
);

-- Find orders without valid customers
SELECT * FROM orders 
WHERE customer_id NOT IN (
    SELECT id FROM customers
);
```

## Performance Considerations

1. **Subquery Size**: IN subqueries that return very large result sets may consume significant memory. Consider using JOINs for better performance in such cases.

2. **Indexing**: Ensure that columns used in subquery WHERE clauses are properly indexed for optimal performance.

3. **Execution Order**: Non-correlated subqueries are executed once per statement. Correlated subqueries are executed once per outer row.

4. **Correlated Subquery Performance**: Correlated subqueries execute the inner query for each outer row, which can be slow for large tables. Consider rewriting as JOINs when performance is critical.

## Limitations

Currently, Stoolap supports the following subquery features:

**Supported:**
- EXISTS and NOT EXISTS operators (both correlated and non-correlated)
- IN and NOT IN subqueries in WHERE clauses
- Scalar subqueries in WHERE clauses with comparison operators (=, !=, <>, >, <, >=, <=)
- Scalar subqueries in SELECT expressions
- Correlated subqueries (subqueries that reference columns from the outer query)
- Derived tables (subqueries in FROM clause)
- Subqueries in DELETE WHERE clauses
- Subqueries in UPDATE WHERE clauses
- Subqueries in UPDATE SET clauses
- Subqueries in INSERT VALUES
- Subqueries in ON CONFLICT DO UPDATE SET and WHERE, and in ON DUPLICATE KEY UPDATE
- Subqueries in RETURNING
- Subqueries in LIMIT and OFFSET
- Proper NULL handling for subqueries returning no rows
- Error detection for scalar subqueries returning multiple rows
- Optimization for EXISTS (stops on first row found)
- ANY/SOME and ALL operators for subquery comparisons

**Not Yet Supported:**

A subquery in these positions is refused with an error rather than answered. Support arrives as each position moves to prepared plans:

- A subquery in `RETURNING` outside the [prepared](#prepared-subqueries) shape
- Inside an explicit transaction, a subquery in `ON CONFLICT ... DO UPDATE` that reads the target table, in a value that reads the row met or the excluded row
- Inside an explicit transaction, a subquery in `RETURNING` that reads the target table, in a value that reads the row returned
- An `UPDATE` that sets one column from a correlated subquery and another from an uncorrelated one, when one of its subqueries is outside the [prepared](#prepared-subqueries) shape
- `DISTINCT ON` expressions in a query that neither groups nor has window functions, a `DISTINCT ON` subquery in the first `SELECT` of a set operation, and a `GROUP BY` expression outside the [prepared](#prepared-subqueries) shape
- `HAVING` in a query that also has window functions, outside the [prepared](#prepared-subqueries) shape
- A grouped `SELECT` list that holds both a correlated and an uncorrelated subquery
- A subquery inside a row of a `VALUES` list, and one in the `WHERE`, the `SELECT` list or the `ORDER BY` of a query over a view, a derived table or a `VALUES` list outside the [prepared](#prepared-subqueries) shape
- A correlated subquery in a `JOIN ... ON` condition, unless the join is [prepared](#prepared-subqueries)
- The argument of `NTILE`, the offset of `LAG` and `LEAD`, or the position of `NTH_VALUE`, when the subquery reads the current row: the argument is one value per partition

The error reads, for example:

```
Compile error: Unsupported expression: scalar subquery outside a prepared plan
```

with `EXISTS`, `IN (subquery)` or `ANY or ALL` in place of `scalar subquery` for those forms.

A column `DEFAULT` and a `CHECK` constraint never hold a subquery: `CREATE TABLE` and `ALTER TABLE ... ADD COLUMN` refuse one when the constraint is declared.

## Best Practices

1. **Use Subqueries for Dynamic Filtering**: Subqueries are ideal when you need to filter based on conditions that require querying other tables.

2. **Consider JOINs for Large Result Sets**: If a subquery returns many rows, a JOIN might be more efficient.

3. **Index Subquery Columns**: Ensure columns used in subquery WHERE clauses are indexed.

4. **Test with Representative Data**: Always test subqueries with realistic data volumes to ensure acceptable performance.

## Error Handling

Common errors when using subqueries:

```sql
-- Error: Subquery returns multiple columns (only first column is used)
DELETE FROM orders WHERE customer_id IN (
    SELECT id, name FROM customers  -- Only 'id' will be used
);

-- Error: Scalar subquery returns more than one row
UPDATE products SET discount = 0.1
WHERE price > (
    SELECT price FROM products WHERE category = 'Electronics'  
    -- Error if multiple products in Electronics category
);

-- Correct: Use aggregate function
UPDATE products SET discount = 0.1
WHERE price > (
    SELECT MAX(price) FROM products WHERE category = 'Electronics'
);
```

## Future Enhancements

Stoolap's subquery support is comprehensive. Planned enhancements include:

- Performance optimizations for correlated subqueries (caching, decorrelation)
- Lateral subqueries (LATERAL JOIN)

For the latest updates on subquery support, check the [Stoolap GitHub repository](https://github.com/stoolap/stoolap).