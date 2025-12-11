<div align="center">
  <img src="logo.svg" alt="Stoolap Logo" width="360">

  <p>An embedded SQL database written in Rust.</p>

  <p>
    <a href="https://stoolap.io">Website</a> •
    <a href="https://stoolap.io/docs">Docs</a> •
    <a href="https://github.com/stoolap/stoolap/releases">Releases</a>
  </p>

  <p>
    <a href="https://github.com/stoolap/stoolap/actions/workflows/ci.yml"><img src="https://github.com/stoolap/stoolap/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://codecov.io/gh/stoolap/stoolap"><img src="https://codecov.io/gh/stoolap/stoolap/branch/main/graph/badge.svg" alt="codecov"></a>
    <a href="https://crates.io/crates/stoolap"><img src="https://img.shields.io/crates/v/stoolap.svg" alt="Crates.io"></a>
    <a href="https://github.com/stoolap/stoolap/releases"><img src="https://img.shields.io/github/v/release/stoolap/stoolap" alt="GitHub release"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  </p>
</div>

## Overview

Stoolap is an embedded SQL database with MVCC transactions, written entirely in Rust. It can be used as an in-memory database or with file-based persistence.

## Installation

```bash
# From source
git clone https://github.com/stoolap/stoolap.git
cd stoolap
cargo build --release

# Or add to Cargo.toml
[dependencies]
stoolap = "0.1"
```

## Usage

### Command Line

```bash
# In-memory database
./target/release/stoolap

# Persistent database
./target/release/stoolap --db "file:///path/to/data"

# Execute a query directly
./target/release/stoolap -q "SELECT 1 + 1"
```

### As a Library

```rust
use stoolap::Database;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = Database::open_in_memory()?;

    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())?;
    db.execute("INSERT INTO users VALUES (1, 'Alice')", ())?;

    for row in db.query("SELECT * FROM users", ())? {
        let row = row?;
        let id: i64 = row.get(0)?;
        let name: String = row.get(1)?;
        println!("{}: {}", id, name);
    }

    Ok(())
}
```

## Features

### SQL Support

- **DDL**: CREATE/DROP/ALTER TABLE, CREATE/DROP INDEX, CREATE/DROP VIEW
- **DML**: SELECT, INSERT, UPDATE, DELETE with RETURNING clause
- **Joins**: INNER, LEFT, RIGHT, FULL OUTER, CROSS
- **Subqueries**: Scalar, IN, EXISTS, correlated
- **CTEs**: WITH and WITH RECURSIVE
- **Window functions**: ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, etc.
- **Aggregations**: GROUP BY with ROLLUP, CUBE, GROUPING SETS
- **Set operations**: UNION, INTERSECT, EXCEPT

### Data Types

- `INTEGER` - 64-bit signed integer
- `FLOAT` - 64-bit floating point
- `TEXT` - UTF-8 string
- `BOOLEAN` - true/false
- `TIMESTAMP`, `DATE`, `TIME` - temporal types
- `JSON` - JSON data

### Indexes

```sql
CREATE INDEX idx_name ON table(column);              -- Auto-selects type
CREATE INDEX idx_name ON table(column) USING BTREE;  -- B-tree (range queries)
CREATE INDEX idx_name ON table(column) USING HASH;   -- Hash (equality)
CREATE INDEX idx_name ON table(column) USING BITMAP; -- Bitmap (low cardinality)
CREATE INDEX idx_name ON table(col1, col2);          -- Multi-column
```

### Transactions

```sql
BEGIN;
-- or: BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
```

Supported isolation levels: READ COMMITTED, SNAPSHOT.

### Temporal Queries

Query historical data using AS OF:

```sql
SELECT * FROM users AS OF TIMESTAMP '2024-01-01 00:00:00';
SELECT * FROM orders AS OF TRANSACTION 42;
```

### Query Analysis

```sql
EXPLAIN SELECT * FROM orders WHERE amount > 100;
EXPLAIN ANALYZE SELECT * FROM orders WHERE amount > 100;
ANALYZE table_name;  -- Collect statistics
```

## Built-in Functions

**String**: UPPER, LOWER, LENGTH, TRIM, CONCAT, SUBSTRING, REPLACE, REVERSE, LEFT, RIGHT, LPAD, RPAD, LOCATE, POSITION, REPEAT

**Math**: ABS, CEIL, FLOOR, ROUND, SQRT, POWER, MOD, SIGN, GREATEST, LEAST, EXP, LN, LOG, SIN, COS, TAN, PI, RAND

**Date/Time**: NOW, CURRENT_DATE, CURRENT_TIMESTAMP, EXTRACT, DATE_TRUNC, DATE_ADD, DATE_SUB, YEAR, MONTH, DAY

**JSON**: JSON_EXTRACT, JSON_TYPE, JSON_VALID, JSON_KEYS

**Aggregate**: COUNT, SUM, AVG, MIN, MAX, STDDEV, VARIANCE, STRING_AGG, ARRAY_AGG

**Window**: ROW_NUMBER, RANK, DENSE_RANK, NTILE, LAG, LEAD, FIRST_VALUE, LAST_VALUE, NTH_VALUE

**Other**: COALESCE, NULLIF, CAST, CASE

## Persistence

- Write-ahead logging (WAL) for durability
- Periodic snapshots for faster recovery
- All indexes persisted

```bash
# In-memory (default)
./stoolap --db "memory://"

# File-based
./stoolap --db "file:///var/lib/stoolap/data"
```

## Architecture

```
src/
├── api/        # Public API (Database, Rows, Statement)
├── core/       # Types (Value, Row, Schema, Error)
├── parser/     # SQL lexer and parser
├── executor/   # Query execution
├── optimizer/  # Cost-based query optimizer
├── functions/  # Built-in functions (scalar, aggregate, window)
└── storage/    # Storage engine with MVCC
```

## Building

```bash
cargo build           # Debug build
cargo build --release # Release build
cargo test            # Run tests
cargo clippy          # Lint
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
