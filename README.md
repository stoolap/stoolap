<div align="center">
  <img src="logo.svg" alt="Stoolap Logo" width="360">

  <h3>A Modern Embedded SQL Database in Pure Rust</h3>

  <p>
    <a href="https://github.com/stoolap/stoolap/releases"><img src="https://img.shields.io/github/v/release/stoolap/stoolap?style=flat-square" alt="GitHub release"></a>
    <a href="https://github.com/stoolap/stoolap/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="Apache License 2.0"></a>
    <a href="https://crates.io/crates/stoolap"><img src="https://img.shields.io/crates/v/stoolap?style=flat-square" alt="Crates.io"></a>
  </p>

  <p><strong>104K lines of Rust | 101+ Functions | Full ACID | No C Dependencies</strong></p>
</div>

---

## Why Stoolap?

| Feature | Stoolap | SQLite | DuckDB |
|---------|:-------:|:------:|:------:|
| MVCC Transactions | ✅ | ❌ | ✅ |
| Time-Travel Queries (AS OF) | ✅ | ❌ | ❌ |
| Adaptive Query Execution | ✅ | ❌ | ❌ |
| Semantic Query Caching | ✅ | ❌ | ❌ |
| Pure Rust (Memory Safe) | ✅ | ❌ | ❌ |

Stoolap combines the simplicity of SQLite with advanced features found in PostgreSQL and DuckDB. It's the only embedded database with **built-in time-travel queries**, **adaptive query execution**, and **semantic caching** - all in pure, memory-safe Rust.

---

## Key Features

- **Pure Rust**: Memory-safe with no C dependencies
- **ACID Transactions**: Full MVCC with READ COMMITTED and SNAPSHOT isolation
- **Cost-Based Optimizer**: PostgreSQL-style optimizer with adaptive query execution
- **Rich SQL Support**: JOINs, CTEs, window functions, subqueries, and 101+ built-in functions
- **Multiple Index Types**: B-tree, Hash, and Bitmap indexes with automatic type selection
- **Temporal Queries**: Built-in AS OF time-travel queries without extensions
- **Parallel Execution**: Rayon-based parallelism for filters, joins, sorts, and aggregations
- **Persistence**: WAL + snapshots with crash recovery

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
stoolap = "0.1"
```

Or build from source:

```bash
git clone https://github.com/stoolap/stoolap.git
cd stoolap
cargo build --release
```

## Quick Start

### Command Line Interface

```bash
# Start with in-memory database
./target/release/stoolap

# Start with persistent storage
./target/release/stoolap --db "file:///path/to/data"

# Execute queries directly
./target/release/stoolap -q "SELECT 1 + 1"
```

### Rust Application

```rust
use stoolap::Database;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create in-memory database
    let db = Database::open_in_memory()?;

    // Create a table
    db.execute("
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ", ())?;

    // Insert data
    db.execute("INSERT INTO users (id, name, email) VALUES (1, 'Alice', 'alice@example.com')", ())?;

    // Query data
    let results = db.query("SELECT * FROM users WHERE id = 1", ())?;
    for row in results {
        println!("{:?}", row);
    }

    Ok(())
}
```

## SQL Features

### Data Types

| Type | Description |
|------|-------------|
| `INTEGER` | 64-bit signed integers |
| `FLOAT` | 64-bit floating point |
| `TEXT` | UTF-8 strings |
| `BOOLEAN` | TRUE/FALSE |
| `TIMESTAMP` | Date and time |
| `JSON` | JSON data with query functions |

### DDL Commands

```sql
-- Tables
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    age INTEGER CHECK (age >= 0 AND age <= 150),
    active BOOLEAN DEFAULT true
);

ALTER TABLE users ADD COLUMN phone TEXT;
ALTER TABLE users DROP COLUMN phone;
DROP TABLE users;
TRUNCATE TABLE users;

-- Indexes (type auto-selected or explicit)
CREATE INDEX idx_name ON users(name);
CREATE INDEX idx_age ON users(age) USING BTREE;
CREATE INDEX idx_email ON users(email) USING HASH;
CREATE INDEX idx_active ON users(active) USING BITMAP;
CREATE INDEX idx_composite ON users(name, age);  -- Multi-column
CREATE UNIQUE INDEX idx_unique ON users(email);
DROP INDEX idx_name ON users;

-- Views
CREATE VIEW active_users AS SELECT * FROM users WHERE active = true;
DROP VIEW active_users;
```

### Query Features

```sql
-- JOINs (INNER, LEFT, RIGHT, FULL OUTER, CROSS)
SELECT u.name, o.total
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;

-- Subqueries (scalar, IN, EXISTS, correlated)
SELECT name, (SELECT AVG(total) FROM orders) as avg_order
FROM users
WHERE id IN (SELECT user_id FROM orders WHERE total > 100);

-- Common Table Expressions (WITH and WITH RECURSIVE)
WITH RECURSIVE countdown(n) AS (
    SELECT 10
    UNION ALL
    SELECT n - 1 FROM countdown WHERE n > 1
)
SELECT * FROM countdown;

-- Window Functions
SELECT name, salary,
    ROW_NUMBER() OVER (ORDER BY salary DESC) as rank,
    LAG(salary) OVER (ORDER BY salary) as prev_salary,
    SUM(salary) OVER (PARTITION BY dept) as dept_total
FROM employees;

-- Aggregations with ROLLUP, CUBE, GROUPING SETS
SELECT region, product, SUM(sales) as total
FROM sales
GROUP BY ROLLUP(region, product);

-- Set Operations
SELECT id FROM table1
UNION ALL
SELECT id FROM table2
INTERSECT
SELECT id FROM table3;

-- Pattern Matching
SELECT * FROM users WHERE name LIKE 'A%';
SELECT * FROM users WHERE name ILIKE 'alice';  -- Case-insensitive
SELECT * FROM users WHERE email REGEXP '^[a-z]+@';
```

### DML with RETURNING

```sql
-- INSERT with upsert and RETURNING
INSERT INTO users (id, name) VALUES (1, 'Alice')
ON DUPLICATE KEY UPDATE name = 'Alice Updated'
RETURNING *;

-- UPDATE with RETURNING
UPDATE users SET name = 'Bob' WHERE id = 1 RETURNING id, name;

-- DELETE with RETURNING
DELETE FROM users WHERE id = 1 RETURNING *;
```

### Transactions

```sql
-- Basic transaction
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;

-- With isolation level
BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT;
SELECT * FROM accounts;
COMMIT;

-- Rollback
BEGIN;
DELETE FROM users;
ROLLBACK;
```

### Temporal Queries (Time Travel)

```sql
-- Query historical data
SELECT * FROM users AS OF TIMESTAMP '2024-01-01 00:00:00';
SELECT * FROM orders AS OF TRANSACTION 42;
```

## Built-in Functions (101+)

### String Functions
`UPPER`, `LOWER`, `LENGTH`, `TRIM`, `LTRIM`, `RTRIM`, `CONCAT`, `CONCAT_WS`, `SUBSTRING`, `SUBSTR`, `REPLACE`, `REVERSE`, `LEFT`, `RIGHT`, `LPAD`, `RPAD`, `LOCATE`, `POSITION`, `STRPOS`, `REPEAT`, `SPACE`

### Math Functions
`ABS`, `CEIL`, `FLOOR`, `ROUND`, `SQRT`, `POWER`, `MOD`, `SIGN`, `GREATEST`, `LEAST`, `PI`, `EXP`, `LN`, `LOG`, `LOG10`, `SIN`, `COS`, `TAN`, `ASIN`, `ACOS`, `ATAN`, `DEGREES`, `RADIANS`, `RAND`

### Date/Time Functions
`NOW`, `CURRENT_DATE`, `CURRENT_TIMESTAMP`, `EXTRACT`, `DATE_TRUNC`, `TO_CHAR`, `YEAR`, `MONTH`, `DAY`, `HOUR`, `MINUTE`, `SECOND`, `DATE_ADD`, `DATE_SUB`, `DATEDIFF`

### JSON Functions
`JSON_EXTRACT`, `JSON_TYPE`, `JSON_TYPEOF`, `JSON_VALID`, `JSON_KEYS`

### Aggregate Functions
`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, `STDDEV`, `VARIANCE`, `STRING_AGG`, `ARRAY_AGG`, `GROUP_CONCAT`

### Window Functions
`ROW_NUMBER`, `RANK`, `DENSE_RANK`, `NTILE`, `LAG`, `LEAD`, `FIRST_VALUE`, `LAST_VALUE`, `NTH_VALUE`

### Other Functions
`COALESCE`, `NULLIF`, `CAST`, `CASE WHEN`, `IIF`

## Index Types

Stoolap automatically selects the optimal index type based on column data type, or you can specify explicitly with the `USING` clause:

| Data Type | Default Index | Best For |
|-----------|---------------|----------|
| INTEGER, FLOAT, TIMESTAMP | B-tree | Range queries, sorting |
| TEXT, JSON | Hash | O(1) equality lookups |
| BOOLEAN | Bitmap | Low-cardinality columns |

```sql
-- Auto-selection (recommended)
CREATE INDEX idx_price ON products(price);

-- Explicit selection
CREATE INDEX idx_status ON orders(status) USING HASH;
CREATE INDEX idx_active ON users(active) USING BITMAP;
```

## Query Optimizer

Stoolap includes a sophisticated cost-based query optimizer:

- **Cost-Based Planning**: PostgreSQL-style cost model with I/O and CPU costs
- **Join Optimization**: Dynamic programming for optimal join ordering
- **Multiple Join Algorithms**: Hash Join, Merge Join, Nested Loop (auto-selected)
- **Adaptive Query Execution**: Runtime re-optimization based on actual cardinalities
- **Cardinality Feedback**: Self-learning optimizer that improves over time
- **Bloom Filter Propagation**: Runtime bloom filters for join acceleration
- **Zone Maps**: Min/max statistics for partition pruning
- **Semantic Query Caching**: Intelligent result caching with predicate subsumption

```sql
-- View query plan
EXPLAIN SELECT * FROM orders WHERE amount > 100;

-- View execution statistics
EXPLAIN ANALYZE SELECT * FROM orders WHERE amount > 100;

-- Collect statistics for optimizer
ANALYZE orders;
```

## Parallel Execution

Stoolap automatically parallelizes large queries using Rayon:

| Operation | Parallel Threshold |
|-----------|-------------------|
| Filter (WHERE) | 10,000 rows |
| Hash Join | 5,000 rows |
| ORDER BY | 50,000 rows |
| DISTINCT | 10,000 rows |

## Persistence

```bash
# In-memory (fast, non-persistent)
./stoolap --db "memory://"

# File-based (durable with WAL)
./stoolap --db "file:///path/to/database"
```

Features:
- Write-Ahead Logging (WAL) for crash recovery
- Periodic snapshots for faster recovery
- All indexes fully persisted and recovered

## Development Status

Stoolap is under active development and is feature-complete for most embedded database use cases. While it provides ACID compliance and a rich feature set, please evaluate thoroughly for production use.

## Documentation

- [Official Documentation](https://stoolap.io)
- [API Reference](https://docs.rs/stoolap)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

```
Copyright 2025 Stoolap Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```
