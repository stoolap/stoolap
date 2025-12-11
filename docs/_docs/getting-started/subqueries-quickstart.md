---
layout: doc
title: Subqueries Quick Start
category: Getting Started
order: 5
---

# Quick Start: Using Subqueries in Stoolap

This guide provides a quick introduction to using subqueries in Stoolap SQL statements.

## What are Subqueries?

Subqueries are SQL queries nested within another query. They allow you to use the result of one query as input for another, enabling more complex data operations.

## Basic IN Subquery Example

Here's a simple example that demonstrates the power of subqueries:

```sql
-- Create sample tables
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT,
    country TEXT
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    total FLOAT
);

-- Insert sample data
INSERT INTO customers VALUES 
    (1, 'Alice', 'USA'),
    (2, 'Bob', 'Canada'),
    (3, 'Charlie', 'USA');

INSERT INTO orders VALUES 
    (1, 1, 100.0),
    (2, 2, 200.0),
    (3, 1, 150.0);

-- Find all orders from US customers using a subquery
SELECT * FROM orders 
WHERE customer_id IN (
    SELECT id FROM customers WHERE country = 'USA'
);
-- Returns orders 1 and 3 (Alice's orders)
```

## EXISTS/NOT EXISTS Example

Check if related records exist without retrieving them:

```sql
-- Find customers who have placed at least one order
SELECT name FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o WHERE o.customer_id = c.id
);

-- Find customers who have never ordered
SELECT name FROM customers c
WHERE NOT EXISTS (
    SELECT 1 FROM orders o WHERE o.customer_id = c.id
);
```

## Scalar Subqueries Example

Use subqueries that return a single value in comparisons:

```sql
-- Find orders above average
SELECT * FROM orders
WHERE total > (SELECT AVG(total) FROM orders);

-- Delete old records below a threshold
DELETE FROM logs
WHERE timestamp < (SELECT MIN(timestamp) FROM logs WHERE priority = 'high');
```

## Common Use Cases

### 1. Filtering Based on Another Table

```sql
-- Delete inactive user data
DELETE FROM user_sessions 
WHERE user_id IN (
    SELECT id FROM users WHERE last_login < DATE('now', '-90 days')
);
```

### 2. Bulk Updates Based on Conditions

```sql
-- Apply discount to premium category products
UPDATE products 
SET discount = 0.20 
WHERE category_id IN (
    SELECT id FROM categories WHERE tier = 'premium'
);
```

### 3. Finding Missing Records

```sql
-- Find customers without any orders
SELECT * FROM customers 
WHERE id NOT IN (
    SELECT DISTINCT customer_id FROM orders
);
```

### 4. Existence Checks

```sql
-- Update product availability
UPDATE products
SET in_stock = false
WHERE NOT EXISTS (
    SELECT 1 FROM inventory WHERE product_id = products.id AND quantity > 0
);
```

### 5. Correlated Subqueries

```sql
-- Find employees earning above their department average
SELECT name, department, salary
FROM employees e1
WHERE salary > (
    SELECT AVG(salary) FROM employees e2 WHERE e2.department = e1.department
);

-- Get each customer's total order amount
SELECT
    c.name,
    (SELECT SUM(amount) FROM orders o WHERE o.customer_id = c.id) as total_spent
FROM customers c;
```

### 6. Derived Tables (Subqueries in FROM)

```sql
-- Join with aggregated data
SELECT c.name, stats.order_count, stats.total_spent
FROM customers c
JOIN (
    SELECT customer_id, COUNT(*) as order_count, SUM(amount) as total_spent
    FROM orders
    GROUP BY customer_id
) AS stats ON c.id = stats.customer_id;
```

## Best Practices

1. **Keep subqueries simple**: Complex subqueries can impact performance
2. **Use indexes**: Ensure columns used in subquery WHERE clauses are indexed
3. **Consider alternatives**: Sometimes a JOIN might be more efficient than a subquery

## Supported Features

Stoolap fully supports:
- ✅ IN and NOT IN subqueries
- ✅ EXISTS and NOT EXISTS operators
- ✅ Scalar subqueries (returning single values)
- ✅ Correlated subqueries (referencing outer query)
- ✅ Derived tables (subqueries in FROM clause)
- ✅ Subqueries in SELECT, UPDATE, and DELETE statements
- ✅ Subqueries in UPDATE SET clause
- ✅ ANY/SOME and ALL operators

## Next Steps

- Read the full [Subqueries documentation](../sql-features/subqueries)
- Learn about [JOIN operations](../sql-features/join-operations) as an alternative
- Explore [SQL Commands reference](../sql-commands/sql-commands)