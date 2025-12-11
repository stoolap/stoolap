---
layout: doc
title: Subqueries
category: SQL Features
order: 8
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
- EXISTS returns NULL (treated as false) if the subquery returns no rows

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
    WHERE login_date > DATE('now', '-30 days')
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
    WHERE order_date > DATE('now', '-30 days')
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
- Proper NULL handling for subqueries returning no rows
- Error detection for scalar subqueries returning multiple rows
- Optimization for EXISTS (stops on first row found)
- ANY/SOME and ALL operators for subquery comparisons

**Not Yet Supported:**
- None - all major subquery features are supported!

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