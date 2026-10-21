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

- `FROM` one table (not a view, CTE, join or derived table, no `AS OF`), and a `WHERE` that holds at least one scalar subquery, `EXISTS`, `NOT EXISTS`, `IN (subquery)`, `NOT IN (subquery)`, or a comparison with `ANY`, `SOME` or `ALL`, with one value, not a tuple, on the left
- no `WITH`, `DISTINCT`, `GROUP BY`, `HAVING`, window functions or set operations; no subquery or aggregate in the SELECT list or `ORDER BY`; `LIMIT` and `OFFSET` as literals or parameters
- each subquery reads one table, with an optional `WHERE`; it may hold further subqueries of these forms, up to 32 scopes in one chain, and no `ORDER BY`, `LIMIT` or `DISTINCT`. A scalar subquery and the subquery of an `IN`, `ANY` or `ALL` select one expression; an `EXISTS` subquery may select anything, and its SELECT list is never evaluated
- a subquery may aggregate: aggregate calls in its SELECT list and `HAVING`, with `DISTINCT`, `FILTER` or `ORDER BY` and any argument after the first a literal, and a `GROUP BY` of columns, expressions, SELECT aliases or positions, without `ROLLUP`, `CUBE` or `GROUPING SETS`. Such a subquery holds further subqueries only in its `WHERE`
- the expressions are columns, literals, parameters, arithmetic, comparisons, `AND`, `OR`, `NOT`, `IS [NOT] NULL`, `IS [NOT] DISTINCT FROM`, `BETWEEN`, `IN (list)`, `LIKE`, `CASE`, `CAST` and scalar function calls

`EXPLAIN` shows `Prepared: yes` for such a statement, or `Prepared: no (reason)` for a WHERE with a subquery that does not fit. For a prepared statement:

- every value a subquery reads from an enclosing query becomes a parameter of the subquery, so the subquery's own comparisons narrow its scan, through its primary key or an index
- a name resolves to the nearest scope that has it: a qualified name `q.c` to the nearest query whose table or alias is `q`, an unqualified name to the subquery's own table first and then to each enclosing query outward
- a qualifier that names no scope is an error, as is a column its table does not have, and a named parameter the statement is given no value for
- a value compared with a column, or tested against one with `IS [NOT] DISTINCT FROM`, is read as the column's type, the way an index or table scan reads it, whether or not the scan takes the comparison: `b = 'true'` matches a `TRUE` row of a `BOOLEAN` column and `s = TRUE` matches `'true'` in a `TEXT` one. A `TEXT` that is no number against a number column, and a number against a `TEXT` column, stay as they are and compare by the numeric rules above. A number against a `BOOLEAN` or `TIMESTAMP` column, and a `BOOLEAN` against a number column, stay as they are too: the two never compare equal, and their ordering is unknown. A `JSON` or `VECTOR` comparison is never handed to the scan
- a subquery that reads no enclosing query runs at most once per execution, and only when a row needs it: a `CASE` arm or the right side of an `OR` that no row reaches does not run it. A subquery compared directly with a column in a top-level `AND` of the statement runs once before the scan, so the scan can use the column's index
- an `EXISTS` that reads its enclosing query only through equalities `own column = enclosing column`, each pair of one type among `INTEGER`, `TEXT`, `BOOLEAN` and `TIMESTAMP`, and calls no volatile function, may read its table once per execution, grouped by those columns, and answer each outer row by one lookup. It does so unless an index on its column serves a probe per outer row more cheaply: for a `LIMIT` of at most 500 or for few outer rows, when its `WHERE` holds nothing besides the equalities. Both give the same answer
- such an `EXISTS` with one equality in a top-level `AND` of the statement reads its table before the scan, and the statement reads only the rows whose column holds one of its keys, through the primary key, an index, or without a `LIMIT` the table's own candidates
- the values of `x IN (subquery)`, `x op ANY (subquery)` and `x op ALL (subquery)` are read as `x`'s column type when `x` is a column, as a value compared with a column is. A subquery that reads no enclosing query reads its values once per execution; a correlated `IN` that meets the rule for `EXISTS` above groups its values by key the same way; any other is run for each row, stopping at the first value that settles the answer
- a column `IN` an uncorrelated subquery in a top-level `AND` of the statement reads the subquery before the scan, and the statement reads only the rows holding one of its values, as for `EXISTS`, when the column's type reads them: a `TEXT` column is not looked up for number values. A `NOT IN` on the primary key reads the keys the values leave out when no value is NULL, and no row at all when one is
- a subquery that aggregates feeds each row its `WHERE` passes to its group, then answers from its groups: a scalar one gives the value of the one group `HAVING` passes, NULL for none, and an error for more than one. One that reads its enclosing query only through equalities as for `EXISTS` above, the rest of it reading no enclosing value, may read its table once per execution, grouped by those columns and its `GROUP BY` items, and answer each outer row by one lookup; a key without rows answers as no rows do, so `COUNT(*)` gives 0. One that reads no enclosing query and holds no subquery runs once per execution as a query of its own, so it takes the same aggregate paths as that query, such as an index for `MIN` or `MAX`
- an error a row of an `EXISTS`, `IN`, `ANY` or `ALL` subquery raises is the statement's error only when no other row settles the answer: TRUE for `EXISTS`, `IN` and `ANY` once a row passes or a value matches, FALSE for `ALL` once a value compares FALSE. A row of another key, or of a NULL key, raises nothing for the outer row. A subquery that aggregates reads every row its `WHERE` passes, so an error one raises is the outer row's error, and the same holds for an error in a value it reads from an enclosing query
- the value of a scalar subquery, the answer of an `EXISTS` and the values of an `IN`, `ANY` or `ALL` read once per execution are kept for the next execution of the same statement with the same parameters, until a write to one of the tables they read; not inside a transaction

An `UPDATE` or `DELETE` is prepared the same way when its `WHERE` or a `SET` value holds a subquery and every expression and subquery in them fits the shape above; the statement's target table is the enclosing query of its subqueries. Its `SET` values may mix subqueries that read the target row with subqueries that do not, in one value or across several. A `SET` value that reads no column of the target row and holds no volatile call, such as `RANDOM()`, runs once per execution, before the statement reads a row.

Every subquery of an `UPDATE` or `DELETE`, the ones that read the target table included, sees the tables as the statement found them: a row the statement updates or deletes does not change what a subquery answers for the next row.

An `INSERT ... VALUES` is prepared the same way when a value holds a subquery and every value fits the shape above. Every value is read before the statement writes a row, so a subquery that reads the target table sees it as the statement found it, without the rows the statement inserts. A scalar subquery that answers more than one row fails the statement and writes no row.

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
- Proper NULL handling for subqueries returning no rows
- Error detection for scalar subqueries returning multiple rows
- Optimization for EXISTS (stops on first row found)
- ANY/SOME and ALL operators for subquery comparisons

**Not Yet Supported:**

A subquery in these positions is refused with an error rather than answered. Support arrives as each position moves to prepared plans:

- `ON CONFLICT ... DO UPDATE SET` and `RETURNING`, and `INSERT ... VALUES` in a statement with `ON CONFLICT`
- An `UPDATE` that sets one column from a correlated subquery and another from an uncorrelated one, when one of its subqueries is outside the [prepared](#prepared-subqueries) shape
- `GROUP BY` and `DISTINCT ON` expressions
- `HAVING` in a query that also has window functions
- A grouped `SELECT` list that holds both a correlated and an uncorrelated subquery
- An uncorrelated subquery in the `WHERE` or `SELECT` list of a query over a view, a derived table or a `VALUES` list
- A correlated subquery in a `JOIN ... ON` condition
- The argument of `NTILE`, the offset of `LAG` and `LEAD`, or the position of `NTH_VALUE`, when the subquery reads the current row: the argument is one value per partition
- `LIMIT` and `OFFSET`

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