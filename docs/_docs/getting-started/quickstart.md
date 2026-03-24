---
layout: doc
title: Quick Start Tutorial
category: Getting Started
order: 2
---

# Quick Start Tutorial

This tutorial will guide you through creating your first database with Stoolap and performing basic operations.

## Installation

Before starting, ensure you have Stoolap installed. If not, follow the [Installation Guide]({% link _docs/getting-started/installation.md %}).

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
./target/release/stoolap -e "SELECT 1 + 1"
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

## Persistence and Backup

When using a persistent database (`file://`), Stoolap automatically manages data durability through WAL and cold volumes. You can also create manual backups:

### Checkpoint

Seal hot data to cold volumes and truncate the WAL:

```sql
PRAGMA CHECKPOINT;
```

### Backup Snapshot

Create a point-in-time backup:

```sql
PRAGMA SNAPSHOT;
```

Or from the command line:

```bash
stoolap -d "file:///path/to/data" --snapshot
```

### Restore from Backup

If your database is corrupted or you need to roll back, restore from a backup snapshot:

```bash
# Restore from latest backup (works even with corrupted volumes)
stoolap -d "file:///path/to/data" --restore

# Restore from a specific backup by timestamp
stoolap -d "file:///path/to/data" --restore "20260315-100000.000"
```

The `--restore` command works at the filesystem level. It removes corrupted volumes and WAL files, then rebuilds from the backup snapshot. No running database is required.

### Configuration

```sql
-- Set checkpoint interval (seconds)
PRAGMA checkpoint_interval = 60;

-- Set WAL sync mode (0=none, 1=normal, 2=full)
PRAGMA sync_mode = 2;

-- Read current configuration
PRAGMA checkpoint_interval;
PRAGMA sync_mode;
```

See the [Connection Strings]({% link _docs/getting-started/connection-strings.md %}) reference for all configuration options.

## CLI Reference

```bash
# Start interactive mode
stoolap -d "file:///path/to/data"

# Execute a single query
stoolap -d "file:///path/to/data" -e "SELECT COUNT(*) FROM users"

# Execute from a SQL file
stoolap -d "file:///path/to/data" -f script.sql

# JSON output mode
stoolap -d "file:///path/to/data" -j -e "SELECT * FROM users"

# Create backup snapshot
stoolap -d "file:///path/to/data" --snapshot

# Restore from backup
stoolap -d "file:///path/to/data" --restore

# Recovery from corrupted volumes
stoolap -d "file:///path/to/data" --reset-volumes --restore

# Set persistence options
stoolap -d "file:///path/to/data" --sync full --checkpoint-interval 30

# Query timeout (milliseconds)
stoolap -d "file:///path/to/data" -t 5000 -e "SELECT * FROM large_table"

# Suppress connection messages
stoolap -d "file:///path/to/data" -q -e "SELECT 1"
```

Run `stoolap --help` for the full list of options.

## Next Steps

Now that you've learned the basics, you might want to explore:

- [Connection Strings]({% link _docs/getting-started/connection-strings.md %}) - More connection options
- [SQL Commands]({% link _docs/sql-commands/sql-commands.md %}) - Comprehensive SQL reference
- [Data Types]({% link _docs/data-types/data-types.md %}) - Detailed information on data types
- [Indexing]({% link _docs/architecture/indexing.md %}) - How to optimize queries with indexes
- [Transaction Isolation]({% link _docs/architecture/transaction-isolation.md %}) - How transactions work

For a more comprehensive reference, browse the [Documentation]({{ site.baseurl }}/docs/).