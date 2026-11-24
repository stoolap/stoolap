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

# Run a script from a file, or from standard input
./target/release/stoolap -f setup.sql
cat setup.sql | ./target/release/stoolap -q
```

`-e`, `-f` and standard input run each statement in turn and print each result. A statement ends at a semicolon outside a string, a quoted name or a comment, read as the SQL parser reads them, or at the end of the input; from standard input a statement runs as soon as its semicolon arrives. A statement that fails prints `Error (statement 2, line 5): ...` on standard error, the run goes on with the next statement, and the CLI exits with status 1 when any statement failed.

A result shows its first 40 rows (`-l N` for another number, `-l 0` for all) and ends with `(N more rows, M in total)` when rows were left out; a plan from EXPLAIN is shown whole. A statement without rows prints its count, `2 rows affected`. In the table, NULL is `NULL`, and a text that could be read as another value or that would break the row, such as `'NULL'`, `'1'`, `''`, `'x|y'` or a text holding a newline, is printed in single quotes with `''`, `\n`, `\r`, `\t` and `\\` for those characters. A FLOAT is printed in the shortest form that reads back as the same number, with `.0` on a whole number. `-j` prints one JSON object per result, `{"columns":[...],"rows":[...],"count":N}`, with `"shown"` when the limit left rows out, and NaN and the infinities as the strings `"NaN"`, `"Infinity"` and `"-Infinity"`.

Without `-q`, the CLI prints how long it took to open the database, `Query executed in 1.234 ms (output 2.1 µs)` after each statement, and `Database closed in ...` at the end. A statement's time is the time the engine took to run it and produce its rows. The time the CLI took to print the rows it showed is measured on its own and given after it, so a large result does not count its printing as query time; rows past the limit are counted without being timed one by one. Opening a file database reads its volumes and closing it seals the rows still in memory, so either can take longer than the statements did, most of all after writes: the time of a whole `stoolap -e ...` process, as `time` reports it, is open, statements and close together. A debug build says so after each of these times, since it runs many times slower than a release build.

`-t MS` (`--timeout`) sets the connection's `query_timeout`, so it bounds every statement the CLI runs, those inside a transaction included; a statement past it fails with `query cancelled`. `-t 0` turns it off. Without `-t` there is no timeout, or the one the connection string gives.

### Agent mode

`--agent`, or the environment variable `STOOLAP_AGENT=1`, writes output for a program to read: JSON Lines on standard output and nothing else there. It is never turned on by detecting a pipe or another tool; `--no-agent` turns it off whatever the variable says. In agent mode the CLI never prompts: it reads `-e`, `-f` or standard input as a script.

```bash
stoolap --agent -e "SELECT 1 AS a; SELECT nope"
```

```
{"stoolap":"0.5.1","commit":"91f751ec...","build":"release","features":["parallel","jemalloc"],"dsn":"memory://"}
{"stmt":1,"line":1,"ok":true,"columns":["a"],"types":["INTEGER"],"rows":[[1]],"row_count":1}
{"stmt":2,"line":1,"ok":false,"error":{"kind":"column_not_found","message":"column 'nope' not found"}}
{"done":true,"statements":2,"failed":1,"first_failed":2,"open_transaction":false}
```

- The first line names the version, the commit, whether the build is a debug or a release build, its features and the connection string.
- Each statement gives one line, in input order, with its number and the line of the input it starts on. A statement without rows gives `"rows_affected"`, EXPLAIN gives its plan as one `"plan"` string, and a query gives `"columns"`, `"types"`, `"rows"` and `"row_count"`, the number of rows it produced.
- Values are JSON: NULL is `null`, a JSON value is written as itself, a VECTOR as an array, a TIMESTAMP as an RFC 3339 string, and a FLOAT's NaN and infinities as the strings `"NaN"`, `"Infinity"` and `"-Infinity"`. `"types"` gives each column's type over every row the statement produced, `null` when they hold only NULL. A column whose values have more than one type is `"MIXED"`, and only its values are written with their type, `{"t":"TEXT","v":"x"}`.
- The first 40 rows are shown (`-l N`, `-l 0` for all); when rows were left out the line adds `"shown"` and `"truncated":true`. A text or JSON value longer than 1,000 characters is cut, and the line adds `"cells_truncated":true`; `--full` keeps them whole. The limit is applied to the rows the statement produced: it does not change the statement.
- `"hash"` is a 64-bit hash of every row the statement produced, the ones the limit left out included, over the values and their types, so `1` and `'1'`, or NULL and `'NULL'`, differ. It is the same in every build: two builds, or two runs, that give a statement the same hash gave it the same answer. A statement with an ORDER BY outside every parenthesis has `"order":"ordered"` and a hash that follows the rows' order; any other has `"order":"unordered"` and a hash the order of its rows does not change.
- A line `-- @case <name>` between statements starts a case: the statements after it, up to the next case, run in a new in-memory database of their own, and their lines carry `"case":"<name>"`. The statements before the first case run in the `-d` database. One process can so run many scenarios that each need a fresh database; the summary adds `"cases"`, each with its statements, its failed ones and whether it left a transaction open. `--on-error stop` skips the rest of the case only.
- A failed statement gives `"ok":false` and an `"error"` with its `"kind"`, one word per kind of engine error, such as `parse`, `column_not_found`, `unique_constraint` or `query_cancelled`, and its `"message"`; a parse error adds the `"line"` and `"column"` of the input it is at.
- The last line sums up the run: the statements, the failed ones, the first that failed, and whether a transaction was still open at the end, which the close rolls back.
- A failed statement does not stop the run; `--on-error stop` stops at the first, and the summary counts the statements it skipped. The exit status is 1 when a statement failed.
- Every statement has a timeout of 30 seconds unless `-t` or the connection string gives one. `--timing` adds `"elapsed_us"`, the engine's time for each statement; without it the output holds no times.

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
# Restore from a specific backup by timestamp (recommended)
stoolap -d "file:///path/to/data" --restore "20260315-100000.000"

# Restore from latest backup snapshot
stoolap -d "file:///path/to/data" --restore

# Recovery from corrupted volumes/manifests (cleans up first)
stoolap -d "file:///path/to/data" --reset-volumes --restore
```

The `--restore` command requires the database to open successfully. If volumes or manifests are corrupted, use `--reset-volumes --restore` which removes bad on-disk state before restoring.

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