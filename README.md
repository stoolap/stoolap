<div align="center">
  <img src="logo.svg" alt="Stoolap Logo" width="360">

  <h3>A Modern Embedded SQL Database in Pure Rust</h3>

  <p>
    <a href="https://stoolap.io">Website</a> •
    <a href="https://stoolap.io/docs">Documentation</a> •
    <a href="https://github.com/stoolap/stoolap/releases">Releases</a> •
    <a href="BENCHMARKS.md">Benchmarks</a>
  </p>

  <p>
    <a href="https://github.com/stoolap/stoolap/actions/workflows/ci.yml"><img src="https://github.com/stoolap/stoolap/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://codecov.io/gh/stoolap/stoolap"><img src="https://codecov.io/gh/stoolap/stoolap/branch/main/graph/badge.svg" alt="codecov"></a>
    <a href="https://crates.io/crates/stoolap"><img src="https://img.shields.io/crates/v/stoolap.svg" alt="Crates.io"></a>
    <a href="https://github.com/stoolap/stoolap/releases"><img src="https://img.shields.io/github/v/release/stoolap/stoolap" alt="GitHub release"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  </p>
</div>

---

## Why Stoolap?

Stoolap is a **feature-rich embedded SQL database** with capabilities that rival established databases like PostgreSQL and DuckDB - all in a single dependency with zero external requirements.

### Performance

Stoolap is optimized for OLTP workloads: point queries, transactional updates, and real-time analytics. It uses parallel execution via Rayon for large scans and a cost-based optimizer for query planning.

See [BENCHMARKS.md](BENCHMARKS.md) for detailed comparisons against SQLite and DuckDB.

### Unique Features

| Feature | Stoolap | SQLite | DuckDB | PostgreSQL |
|---------|:-------:|:------:|:------:|:----------:|
| **AS OF Time-Travel Queries** | ✅ | ❌ | ❌ | ❌* |
| **MVCC Transactions** | ✅ | ❌ | ✅ | ✅ |
| **Cost-Based Optimizer** | ✅ | ❌ | ✅ | ✅ |
| **Adaptive Query Execution** | ✅ | ❌ | ❌ | ❌ |
| **Semantic Query Caching** | ✅ | ❌ | ❌ | ❌ |
| **Parallel Query Execution** | ✅ | ❌ | ✅ | ✅ |
| **Pure Rust (Memory Safe)** | ✅ | ❌ | ❌ | ❌ |
| **No C/C++ Required** | ✅ | ❌ | ❌ | ❌ |

*PostgreSQL requires extensions for temporal queries

---

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
stoolap = "0.2"
```

Or build from source:

```bash
git clone https://github.com/stoolap/stoolap.git
cd stoolap
cargo build --release
```

### Library Usage

```rust
use stoolap::api::Database;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // In-memory database
    let db = Database::open_in_memory()?;

    // Or persistent storage
    // let db = Database::open("file:///path/to/data")?;

    // Create table
    db.execute("CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )", ())?;

    // Insert with parameters
    db.execute("INSERT INTO users (id, name, email) VALUES (?, ?, ?)",
        (1, "Alice", "alice@example.com"))?;

    // Query with iteration
    for row in db.query("SELECT * FROM users WHERE id = ?", (1,))? {
        let row = row?;
        println!("User: {} <{}>",
            row.get::<String>(1)?,  // name
            row.get::<String>(2)?   // email
        );
    }

    Ok(())
}
```

### Command Line Interface

```bash
# Interactive REPL (in-memory)
./stoolap

# Persistent database
./stoolap --db "file:///var/lib/stoolap/data"

# Execute query directly
./stoolap -q "SELECT version()"

# Execute SQL file
./stoolap --db "file://./mydb" < schema.sql
```

---

## Features

### MVCC Transactions

Full multi-version concurrency control with isolation levels:

```sql
-- Read Committed (default)
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;

-- Snapshot Isolation (repeatable reads)
BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT;
SELECT * FROM accounts;  -- Consistent view throughout transaction
COMMIT;
```

### Time-Travel Queries

Query historical data at any point in time - a feature typically only found in enterprise databases:

```sql
-- Query data as it existed at a specific timestamp
SELECT * FROM orders AS OF TIMESTAMP '2024-01-15 10:30:00';

-- Query data as of a specific transaction
SELECT * FROM inventory AS OF TRANSACTION 1234;

-- Compare current vs historical data
SELECT
    c.price AS current_price,
    h.price AS old_price,
    c.price - h.price AS change
FROM products c
JOIN products AS OF TIMESTAMP '2024-01-01 00:00:00' h ON c.id = h.id
WHERE c.price != h.price;
```

### Smart Indexes

Automatic index type selection based on data characteristics:

```sql
-- B-tree (auto-selected for INTEGER, FLOAT, TIMESTAMP)
-- Best for: range queries, sorting, prefix matching
CREATE INDEX idx_date ON orders(created_at);
SELECT * FROM orders WHERE created_at BETWEEN '2024-01-01' AND '2024-12-31';

-- Hash (auto-selected for TEXT, JSON)
-- Best for: O(1) equality lookups
CREATE INDEX idx_email ON users(email);
SELECT * FROM users WHERE email = 'alice@example.com';

-- Bitmap (auto-selected for BOOLEAN)
-- Best for: low-cardinality columns, efficient AND/OR
CREATE INDEX idx_status ON orders(status) USING BITMAP;

-- Multi-column composite indexes
CREATE INDEX idx_lookup ON events(user_id, event_type);
CREATE UNIQUE INDEX idx_unique ON orders(customer_id, order_date);
```

### Window Functions

Full analytical query support:

```sql
SELECT
    employee_name,
    department,
    salary,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS dept_rank,
    LAG(salary) OVER (ORDER BY hire_date) AS prev_salary,
    AVG(salary) OVER (PARTITION BY department) AS dept_avg,
    SUM(salary) OVER (ORDER BY hire_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
FROM employees;
```

### Common Table Expressions (CTEs)

Including recursive queries for hierarchical data:

```sql
-- Recursive CTE: organizational hierarchy
WITH RECURSIVE org_chart AS (
    -- Base case: top-level managers
    SELECT id, name, manager_id, 1 AS level
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive case: employees under managers
    SELECT e.id, e.name, e.manager_id, oc.level + 1
    FROM employees e
    JOIN org_chart oc ON e.manager_id = oc.id
)
SELECT * FROM org_chart ORDER BY level, name;
```

### Advanced Aggregations

```sql
-- ROLLUP: hierarchical subtotals
SELECT region, product, SUM(sales)
FROM sales GROUP BY ROLLUP(region, product);

-- CUBE: all dimension combinations
SELECT region, product, SUM(sales)
FROM sales GROUP BY CUBE(region, product);

-- GROUPING SETS: custom combinations
SELECT region, product, SUM(sales)
FROM sales GROUP BY GROUPING SETS ((region, product), (region), ());
```

### Subqueries

Scalar, correlated, EXISTS, IN, ANY/ALL:

```sql
-- Correlated subquery
SELECT * FROM employees e
WHERE salary > (
    SELECT AVG(salary) FROM employees
    WHERE department = e.department
);

-- EXISTS with correlation
SELECT * FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o
    WHERE o.customer_id = c.id AND o.amount > 1000
);
```

### Query Optimizer

PostgreSQL-style cost-based optimizer with runtime adaptation:

```sql
-- Collect statistics for better query plans
ANALYZE orders;
ANALYZE customers;

-- View query plan with cost estimates
EXPLAIN SELECT * FROM orders WHERE customer_id = 100;

-- View plan with actual execution statistics
EXPLAIN ANALYZE
SELECT o.*, c.name
FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE c.country = 'US';
```

---

## Data Types

| Type | Description | Example |
|------|-------------|---------|
| `INTEGER` | 64-bit signed integer | `42`, `-100` |
| `FLOAT` | 64-bit floating point | `3.14`, `-0.001` |
| `TEXT` | UTF-8 string | `'hello'`, `'日本語'` |
| `BOOLEAN` | true/false | `TRUE`, `FALSE` |
| `TIMESTAMP` | Date and time | `'2024-01-15 10:30:00'` |
| `JSON` | JSON data | `'{"key": "value"}'` |

---

## Built-in Functions (100+)

<details>
<summary><b>String Functions</b></summary>

`UPPER`, `LOWER`, `LENGTH`, `TRIM`, `LTRIM`, `RTRIM`, `CONCAT`, `SUBSTRING`, `REPLACE`, `REVERSE`, `LEFT`, `RIGHT`, `LPAD`, `RPAD`, `REPEAT`, `POSITION`, `LOCATE`, `INSTR`, `SPLIT_PART`, `INITCAP`, `ASCII`, `CHR`, `TRANSLATE`
</details>

<details>
<summary><b>Math Functions</b></summary>

`ABS`, `CEIL`, `FLOOR`, `ROUND`, `TRUNC`, `SQRT`, `POWER`, `MOD`, `SIGN`, `GREATEST`, `LEAST`, `EXP`, `LN`, `LOG`, `LOG10`, `LOG2`, `SIN`, `COS`, `TAN`, `ASIN`, `ACOS`, `ATAN`, `ATAN2`, `DEGREES`, `RADIANS`, `PI`, `RAND`, `RANDOM`
</details>

<details>
<summary><b>Date/Time Functions</b></summary>

`NOW`, `CURRENT_DATE`, `CURRENT_TIME`, `CURRENT_TIMESTAMP`, `EXTRACT`, `DATE_TRUNC`, `DATE_ADD`, `DATE_SUB`, `DATEDIFF`, `YEAR`, `MONTH`, `DAY`, `HOUR`, `MINUTE`, `SECOND`, `DAYOFWEEK`, `DAYOFYEAR`, `WEEK`, `QUARTER`, `TO_CHAR`, `TO_DATE`, `TO_TIMESTAMP`
</details>

<details>
<summary><b>JSON Functions</b></summary>

`JSON_EXTRACT`, `JSON_EXTRACT_PATH`, `JSON_TYPE`, `JSON_TYPEOF`, `JSON_VALID`, `JSON_KEYS`, `JSON_ARRAY_LENGTH`
</details>

<details>
<summary><b>Aggregate Functions</b></summary>

`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, `STDDEV`, `STDDEV_POP`, `STDDEV_SAMP`, `VARIANCE`, `VAR_POP`, `VAR_SAMP`, `STRING_AGG`, `ARRAY_AGG`, `FIRST`, `LAST`, `BIT_AND`, `BIT_OR`, `BIT_XOR`, `BOOL_AND`, `BOOL_OR`
</details>

<details>
<summary><b>Window Functions</b></summary>

`ROW_NUMBER`, `RANK`, `DENSE_RANK`, `NTILE`, `LAG`, `LEAD`, `FIRST_VALUE`, `LAST_VALUE`, `NTH_VALUE`, `PERCENT_RANK`, `CUME_DIST`
</details>

<details>
<summary><b>Utility Functions</b></summary>

`COALESCE`, `NULLIF`, `CAST`, `CASE`, `IF`, `IIF`, `NVL`, `NVL2`, `DECODE`, `GENERATE_SERIES`
</details>

---

## Storage & Persistence

```bash
# In-memory (fastest, data lost on exit)
./stoolap --db "memory://"

# File-based (durable storage with WAL)
./stoolap --db "file:///var/lib/stoolap/data"
```

**Durability features:**
- **Write-Ahead Logging (WAL)**: All changes logged before applied
- **Periodic Snapshots**: Fast recovery from crashes
- **Index Persistence**: All indexes saved and restored automatically

---

## Architecture

```
src/
├── api/           # Public API (Database, Connection, Rows)
├── core/          # Core types (Value, Row, Schema, Error)
├── parser/        # SQL lexer and parser
├── optimizer/     # Cost-based query optimizer
│   ├── cost.rs        # Cost model with I/O and CPU costs
│   ├── join.rs        # Join optimization (dynamic programming)
│   ├── bloom.rs       # Bloom filter propagation
│   └── aqe.rs         # Adaptive query execution
├── executor/      # Query execution engine
│   ├── operators/     # Volcano-style operators
│   ├── parallel.rs    # Parallel execution (Rayon)
│   └── expression/    # Expression VM
├── functions/     # 100+ built-in functions
│   ├── scalar/        # String, math, date, JSON
│   ├── aggregate/     # COUNT, SUM, AVG, etc.
│   └── window/        # ROW_NUMBER, RANK, LAG, etc.
└── storage/       # Storage engine
    ├── mvcc/          # Multi-version concurrency control
    └── index/         # B-tree, Hash, Bitmap indexes
```

---

## Development

### Building

```bash
cargo build              # Debug build
cargo build --release    # Optimized release build
```

### Testing

```bash
cargo nextest run        # Run all tests (recommended)
cargo test               # Standard test runner
```

### Code Quality

```bash
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

### Documentation

```bash
cargo doc --open         # Generate and open API docs
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
