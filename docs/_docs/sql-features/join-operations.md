---
title: JOIN Operations
category: SQL Features
order: 1
---

# JOIN Operations

This document explains JOIN operations in Stoolap, their syntax, and how to use them effectively based on the implementation and test files.

## Overview

Stoolap supports SQL JOIN operations to combine data from multiple tables based on related columns. These operations are fundamental for relational data operations and complex queries.

## Supported JOIN Types

Stoolap supports the following JOIN types:

### INNER JOIN

An INNER JOIN returns rows when there is a match in both tables.

```sql
SELECT a.id, a.name, b.category
FROM table_a AS a
INNER JOIN table_b AS b ON a.id = b.a_id;
```

### LEFT JOIN (or LEFT OUTER JOIN)

A LEFT JOIN returns all rows from the left table, and the matched rows from the right table. When no match is found, NULL values are returned for columns from the right table.

```sql
SELECT c.id, c.name, o.order_date
FROM customers AS c
LEFT JOIN orders AS o ON c.id = o.customer_id;
```

## JOIN Syntax

The basic syntax for JOIN operations in Stoolap is:

```sql
SELECT column1, column2, ...
FROM table1
[JOIN_TYPE] JOIN table2
ON table1.column_name = table2.column_name;
```

Where:
- `JOIN_TYPE` is the type of join (INNER, LEFT)
- The `ON` clause specifies the join condition

## Examples

### Basic JOIN Example

```sql
-- Create test tables
CREATE TABLE categories (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category_id INTEGER,
    price FLOAT
);

-- Insert test data
INSERT INTO categories (id, name) VALUES 
(1, 'Electronics'),
(2, 'Clothing'),
(3, 'Food');

INSERT INTO products (id, name, category_id, price) VALUES 
(1, 'Smartphone', 1, 899.99),
(2, 'Laptop', 1, 1299.99),
(3, 'T-Shirt', 2, 24.99),
(4, 'Jeans', 2, 49.99),
(5, 'Bread', 3, 3.49);

-- Simple INNER JOIN
SELECT p.id, p.name, p.price, c.name AS category
FROM products p
INNER JOIN categories c ON p.category_id = c.id
ORDER BY p.id;
```

### Comprehensive JOIN Example

```sql
-- Create test tables
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    order_date TIMESTAMP,
    total FLOAT
);

CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    price FLOAT
);

-- Insert test data
INSERT INTO customers (id, name, email) VALUES 
(1, 'John Doe', 'john@example.com'),
(2, 'Jane Smith', 'jane@example.com'),
(3, 'Robert Johnson', 'robert@example.com');

INSERT INTO orders (id, customer_id, order_date, total) VALUES 
(101, 1, '2023-01-15', 126.49),
(102, 1, '2023-02-20', 89.99),
(103, 2, '2023-01-25', 54.98),
(104, 3, '2023-03-10', 199.99);

-- LEFT JOIN to include all customers
SELECT c.id, c.name, o.id AS order_id, o.order_date
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
ORDER BY c.id, o.id;
```

### Multi-Table JOINs

```sql
-- Multi-table joins
SELECT c.name, o.id AS order_id, o.order_date, oi.product_id, oi.quantity, oi.price
FROM customers c
INNER JOIN orders o ON c.id = o.customer_id
INNER JOIN order_items oi ON o.id = oi.order_id
ORDER BY c.id, o.id, oi.id;
```

### Self JOIN Example

```sql
-- Create employees table with self-referencing manager_id
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    title TEXT,
    manager_id INTEGER
);

-- Insert test data for hierarchy
INSERT INTO employees (id, name, title, manager_id) VALUES 
(1, 'John Smith', 'CEO', NULL),
(2, 'Jane Doe', 'CTO', 1),
(3, 'Robert Johnson', 'CFO', 1),
(4, 'Emily Brown', 'Engineering Manager', 2),
(5, 'Michael Wilson', 'Developer', 4);

-- Self JOIN to show employee-manager relationship
SELECT e.id, e.name, e.title, m.name AS manager_name, m.title AS manager_title
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id
ORDER BY e.id;
```

## JOIN Performance Considerations

Based on the implementation and test files:

- **Indexing Join Columns**: For optimal performance, create indexes on columns used in JOIN conditions
- **Parallel Processing**: Stoolap uses parallel execution for large joins
- **Join Order**: The order of joined tables can impact performance
- **Column Selection**: Only select columns you need to minimize data transfer
- **Filter Early**: Apply WHERE conditions before joins when possible to reduce the number of rows processed

## JOIN with NULL Values

- In INNER JOIN, rows with NULL values in the join columns do not match
- In LEFT JOIN, all rows from the left table are included, with NULLs for non-matching right table values
- The IS NULL operator can be used in join conditions for special cases

## Best Practices

1. **Use Table Aliases**: Improves readability, especially in complex queries
2. **Column Qualification**: Always qualify column names with table aliases in multi-table queries
3. **Join Conditions**: Use correct and meaningful join conditions
4. **Consider JOIN Type**: Choose the appropriate JOIN type for your query requirements
5. **Index Join Columns**: Ensure join columns are indexed for performance
6. **Test with Representative Data**: Validate join behavior with varied test data

## Limitations

- Large joins can be memory-intensive, especially without proper filtering
- Complex joins with many tables may require careful optimization