---
layout: doc
title: Quick Start Tutorial
category: Getting Started
order: 2
---

# Quick Start Tutorial

This tutorial will guide you through creating your first database with Stoolap and performing basic operations.

## Installation

Before starting, ensure you have Stoolap installed. If not, follow the [Installation Guide](installation).

```bash
# Install with Cargo
cargo install stoolap

# Or build from source
git clone https://github.com/stoolap/stoolap.git
cd stoolap
cargo build --release
```

## Starting the CLI

Stoolap includes a command-line interface (CLI) for interactive use:

```bash
# Start with an in-memory database (data is lost when the CLI exits)
./target/release/stoolap

# Or with persistent storage (data is saved to disk)
./target/release/stoolap --db "file:///path/to/data"

# Execute a query directly
./target/release/stoolap -q "SELECT 1 + 1"
```

## Creating a Table

Let's create a simple table to store product information:

```sql
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    price FLOAT NOT NULL,
    category TEXT,
    in_stock BOOLEAN,
    created_at TIMESTAMP
);
```

## Inserting Data

Now let's add some sample products:

```sql
-- Insert a single product
INSERT INTO products (id, name, description, price, category, in_stock, created_at)
VALUES (1, 'Laptop', 'High-performance laptop with 16GB RAM', 1299.99, 'Electronics', TRUE, NOW());

-- Insert multiple products
INSERT INTO products (id, name, description, price, category, in_stock, created_at) VALUES 
(2, 'Smartphone', '5G smartphone with 128GB storage', 799.99, 'Electronics', TRUE, NOW()),
(3, 'Headphones', 'Wireless noise-cancelling headphones', 249.99, 'Accessories', TRUE, NOW()),
(4, 'Monitor', '27-inch 4K monitor', 349.99, 'Electronics', FALSE, NOW()),
(5, 'Keyboard', 'Mechanical gaming keyboard', 129.99, 'Accessories', TRUE, NOW());
```

## Querying Data

### Basic SELECT

Retrieve all products:

```sql
SELECT * FROM products;
```

### Filtering with WHERE

Retrieve products in a specific category:

```sql
SELECT name, price FROM products WHERE category = 'Electronics';
```

### Sorting with ORDER BY

Sort products by price from highest to lowest:

```sql
SELECT name, price FROM products ORDER BY price DESC;
```

### Limiting Results

Get only the 3 most expensive products:

```sql
SELECT name, price FROM products ORDER BY price DESC LIMIT 3;
```

## Updating Data

Let's update the price of a product:

```sql
UPDATE products SET price = 1199.99 WHERE id = 1;
```

Update multiple fields:

```sql
UPDATE products 
SET price = 349.99, description = 'Updated description'
WHERE id = 2;
```

## Deleting Data

Remove a product from the database:

```sql
DELETE FROM products WHERE id = 5;
```

## Creating an Index

Indexes speed up queries on frequently searched columns:

```sql
-- Create an index on the category column
CREATE INDEX idx_category ON products(category);

-- Create a unique index on the name column
CREATE UNIQUE INDEX idx_name ON products(name);
```

## Working with Transactions

Transactions ensure that multiple operations succeed or fail as a unit:

```sql
-- Start a transaction
BEGIN TRANSACTION;

-- Perform operations
UPDATE products SET price = price * 0.9 WHERE category = 'Electronics';
INSERT INTO products (id, name, price, category) VALUES (6, 'Tablet', 499.99, 'Electronics');

-- Commit the transaction to save changes
COMMIT;

-- Or roll back to discard changes
-- ROLLBACK;
```

## Using Joins

Let's create a categories table and join it with our products:

```sql
-- Create categories table
CREATE TABLE categories (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT
);

-- Add some categories
INSERT INTO categories (id, name, description) VALUES
(1, 'Electronics', 'Electronic devices and gadgets'),
(2, 'Accessories', 'Peripherals and accessories for devices');

-- Update products to use category ids
ALTER TABLE products ADD COLUMN category_id INTEGER;
UPDATE products SET category_id = 1 WHERE category = 'Electronics';
UPDATE products SET category_id = 2 WHERE category = 'Accessories';

-- Join tables to get category information
SELECT p.id, p.name, p.price, c.name AS category_name, c.description AS category_description
FROM products p
JOIN categories c ON p.category_id = c.id;
```

## Using Aggregation Functions

Get summary statistics for your products:

```sql
-- Count products by category
SELECT category, COUNT(*) AS product_count
FROM products
GROUP BY category;

-- Get average price by category
SELECT category, AVG(price) AS avg_price
FROM products
GROUP BY category;

-- Get price range by category
SELECT 
    category,
    MIN(price) AS min_price,
    MAX(price) AS max_price,
    AVG(price) AS avg_price
FROM products
GROUP BY category;
```

## Working with Common Table Expressions (CTEs)

CTEs make complex queries more readable:

```sql
-- Find top products by category
WITH category_stats AS (
    SELECT 
        category,
        AVG(price) as avg_price,
        MAX(price) as max_price
    FROM products
    GROUP BY category
)
SELECT 
    p.name,
    p.price,
    cs.avg_price,
    ROUND((p.price / cs.avg_price - 1) * 100, 2) as pct_above_avg
FROM products p
JOIN category_stats cs ON p.category = cs.category
WHERE p.price > cs.avg_price
ORDER BY pct_above_avg DESC;
```

## Next Steps

Now that you've learned the basics, you might want to explore:

- [Connection Strings](/docs/getting-started/connection-strings/) - More connection options
- [SQL Commands](/docs/sql-commands/sql-commands/) - Comprehensive SQL reference
- [Data Types](/docs/data-types/data-types/) - Detailed information on data types
- [Indexing](/docs/architecture/indexing/) - How to optimize queries with indexes
- [Transaction Isolation](/docs/architecture/transaction-isolation/) - How transactions work

For a more comprehensive reference, browse the [Documentation](/docs/).