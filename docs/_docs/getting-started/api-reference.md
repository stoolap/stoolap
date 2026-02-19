---
layout: doc
title: API Reference
category: Getting Started
order: 4
---

# Rust API Reference

Complete reference for the Stoolap Rust API (`stoolap` crate).

```toml
[dependencies]
stoolap = "0.3"
```

## Database

The `Database` struct is the main entry point. It is thread-safe and can be shared across threads via cloning.

### Opening a Database

```rust
use stoolap::Database;

// In-memory (unique instance per call)
let db = Database::open_in_memory()?;

// In-memory (shared instance — same DSN returns same engine)
let db = Database::open("memory://")?;

// File-based (persistent)
let db = Database::open("file:///path/to/database")?;

// File-based with configuration
let db = Database::open("file:///path/to/db?sync=full&snapshot_interval=60")?;
```

`open_in_memory()` creates a unique, isolated instance each time. `open("memory://")` returns the same shared engine for the same DSN.

### Connection String Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sync` / `sync_mode` | `normal` | Sync mode: `none`, `normal`, `full` (or `0`, `1`, `2`) |
| `snapshot_interval` | `300` | Seconds between automatic snapshots |
| `keep_snapshots` | `5` | Number of snapshot files to retain |
| `wal_flush_trigger` | `32768` | WAL flush trigger size in bytes |
| `wal_buffer_size` | `65536` | WAL buffer size in bytes |
| `wal_max_size` | `67108864` | Max WAL file size before rotation (64 MB) |
| `commit_batch_size` | `100` | Commits to batch before syncing (normal mode) |
| `sync_interval_ms` | `10` | Minimum ms between syncs (normal mode) |
| `wal_compression` | `on` | LZ4 compression for WAL entries |
| `snapshot_compression` | `on` | LZ4 compression for snapshots |
| `compression` | -- | Set both `wal_compression` and `snapshot_compression` |
| `compression_threshold` | `64` | Minimum bytes before compressing an entry |

### execute()

Execute DDL or DML statements. Returns the number of rows affected.

```rust
use stoolap::params;

// DDL — no parameters
db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)", ())?;

// DML with tuple parameters
db.execute("INSERT INTO users VALUES ($1, $2, $3)", (1, "Alice", 30))?;

// DML with params! macro
db.execute("INSERT INTO users VALUES ($1, $2, $3)", params![2, "Bob", 25])?;

// Returns rows affected
let affected = db.execute("UPDATE users SET name = $1 WHERE id = $2", ("Charlie", 1))?;
```

### query()

Execute a SELECT query and iterate over results.

```rust
// Iterate rows
for row in db.query("SELECT * FROM users", ())? {
    let row = row?;
    let id: i64 = row.get(0)?;
    let name: String = row.get_by_name("name")?;
}

// With parameters
for row in db.query("SELECT * FROM users WHERE age > $1", (18,))? {
    let row = row?;
    // ...
}

// Collect into Vec
let users: Vec<_> = db.query("SELECT * FROM users", ())?
    .collect::<Result<Vec<_>, _>>()?;
```

### query_one()

Return a single value. Errors if no rows are returned.

```rust
let count: i64 = db.query_one("SELECT COUNT(*) FROM users", ())?;
let name: String = db.query_one("SELECT name FROM users WHERE id = $1", (1,))?;
```

### query_opt()

Return an optional single value. Returns `None` if no rows match.

```rust
let name: Option<String> = db.query_opt("SELECT name FROM users WHERE id = $1", (999,))?;
match name {
    Some(n) => println!("Found: {}", n),
    None => println!("Not found"),
}
```

### Named Parameters

Use `:name` syntax with `named_params!` macro or the `NamedParams` builder.

```rust
use stoolap::named_params;

// Execute with named params
db.execute_named(
    "INSERT INTO users VALUES (:id, :name, :age)",
    named_params!{ id: 1, name: "Alice", age: 30 }
)?;

// Query with named params
for row in db.query_named(
    "SELECT * FROM users WHERE age > :min_age",
    named_params!{ min_age: 25 }
)? {
    // ...
}

// Query single value with named params
let count: i64 = db.query_one_named(
    "SELECT COUNT(*) FROM users WHERE age > :min_age",
    named_params!{ min_age: 18 }
)?;

// Builder API (alternative to macro)
use stoolap::NamedParams;
let params = NamedParams::new()
    .add("id", 1)
    .add("name", "Alice")
    .add("age", 30);
db.execute_named("INSERT INTO users VALUES (:id, :name, :age)", params)?;
```

### Timeout Methods

Cancel queries that exceed a time limit. Timeout is in milliseconds; use 0 for no timeout.

```rust
// Execute with 5 second timeout
db.execute_with_timeout("DELETE FROM large_table WHERE old = true", (), 5000)?;

// Query with 10 second timeout
for row in db.query_with_timeout("SELECT * FROM large_table", (), 10000)? {
    let row = row?;
    // ...
}
```

### Struct Mapping with FromRow

Map query results directly to structs.

```rust
use stoolap::{Database, FromRow, ResultRow, Result};

struct User {
    id: i64,
    name: String,
    email: Option<String>,
}

impl FromRow for User {
    fn from_row(row: &ResultRow) -> Result<Self> {
        Ok(User {
            id: row.get(0)?,
            name: row.get(1)?,
            email: row.get(2)?,
        })
    }
}

// Query and map to structs
let users: Vec<User> = db.query_as("SELECT id, name, email FROM users", ())?;

// With named params
let users: Vec<User> = db.query_as_named(
    "SELECT id, name, email FROM users WHERE id = :id",
    named_params!{ id: 1 }
)?;
```

### Cached Plans

Parse SQL once and execute many times with zero cache-lookup overhead.

```rust
// Create a cached plan (parse once)
let insert_plan = db.cached_plan("INSERT INTO users VALUES ($1, $2, $3)")?;

// Execute many times (no parsing, no cache lookup)
db.execute_plan(&insert_plan, (1, "Alice", 30))?;
db.execute_plan(&insert_plan, (2, "Bob", 25))?;

// Query with cached plan
let query_plan = db.cached_plan("SELECT * FROM users WHERE id = $1")?;
let rows = db.query_plan(&query_plan, (1,))?;

// Named params with cached plan
let plan = db.cached_plan("INSERT INTO users VALUES (:id, :name, :age)")?;
db.execute_named_plan(&plan, named_params!{ id: 3, name: "Charlie", age: 35 })?;
let rows = db.query_named_plan(
    &db.cached_plan("SELECT * FROM users WHERE id = :id")?,
    named_params!{ id: 3 }
)?;
```

### Utility Methods

```rust
// Close the database (releases file lock immediately)
db.close()?;

// Check if a table exists
if db.table_exists("users")? {
    // ...
}

// Get the DSN this database was opened with
let dsn = db.dsn();  // "memory://" or "file:///path"

// Set default isolation level for new transactions
use stoolap::IsolationLevel;
db.set_default_isolation_level(IsolationLevel::Snapshot)?;

// Create a point-in-time snapshot (for file-based databases)
db.create_snapshot()?;

// Semantic cache stats
let stats = db.semantic_cache_stats()?;
println!("Hits: {}, Subsumption: {}", stats.hits, stats.subsumption_hits);

// Clear the semantic cache
db.clear_semantic_cache()?;
```

## Prepared Statements

Prepare a statement for repeated execution with different parameters. The SQL is validated at prepare time.

```rust
// Prepare once
let stmt = db.prepare("SELECT * FROM users WHERE id = $1")?;

// Execute multiple times
for id in 1..=10 {
    for row in stmt.query((id,))? {
        let row = row?;
        // ...
    }
}

// Single value
let name: String = stmt.query_one((1,))?;

// Optional value
let name: Option<String> = stmt.query_opt((999,))?;

// DML prepared statement
let insert = db.prepare("INSERT INTO users VALUES ($1, $2, $3)")?;
insert.execute((1, "Alice", 30))?;
insert.execute((2, "Bob", 25))?;

// Get the SQL text
assert_eq!(insert.sql(), "INSERT INTO users VALUES ($1, $2, $3)");
```

Statement holds a weak reference to the Database. It becomes invalid after the Database is dropped.

## Transactions

### Programmatic API

```rust
// Begin with default isolation (ReadCommitted)
let mut tx = db.begin()?;

// Begin with specific isolation level
use stoolap::IsolationLevel;
let mut tx = db.begin_with_isolation(IsolationLevel::Snapshot)?;

// Execute within transaction
tx.execute("INSERT INTO users VALUES ($1, $2, $3)", (1, "Alice", 30))?;
tx.execute("UPDATE accounts SET balance = balance - $1 WHERE id = $2", (100, 1))?;

// Query within transaction (sees uncommitted changes)
for row in tx.query("SELECT * FROM users", ())? {
    let row = row?;
    // ...
}

// Single value query
let count: i64 = tx.query_one("SELECT COUNT(*) FROM users", ())?;

// Optional value query
let name: Option<String> = tx.query_opt("SELECT name FROM users WHERE id = $1", (999,))?;

// Named parameters
tx.execute_named(
    "INSERT INTO users VALUES (:id, :name, :age)",
    named_params!{ id: 3, name: "Charlie", age: 35 }
)?;

for row in tx.query_named(
    "SELECT * FROM users WHERE age > :min_age",
    named_params!{ min_age: 25 }
)? {
    let row = row?;
    // ...
}

// Commit or rollback
tx.commit()?;
// Or: tx.rollback()?;
```

Transactions auto-rollback on drop if not committed.

### Pre-parsed Statements in Transactions

For batch operations, parse SQL once and execute many times within a transaction.

```rust
use stoolap::parser::Parser;

let stmt = Parser::new("INSERT INTO users VALUES ($1, $2, $3)")
    .parse_program()?
    .statements
    .into_iter()
    .next()
    .unwrap();

let mut tx = db.begin()?;
for (id, name, age) in data {
    tx.execute_prepared(&stmt, (id, name, age))?;
}
tx.commit()?;
```

### SQL-based Transactions

```rust
db.execute("BEGIN", ())?;
db.execute("INSERT INTO users VALUES ($1, $2)", (1, "Alice"))?;
db.execute("COMMIT", ())?;
// Or: db.execute("ROLLBACK", ())?;
```

### Savepoints

```rust
db.execute("BEGIN", ())?;
db.execute("INSERT INTO users VALUES ($1, $2)", (1, "Alice"))?;
db.execute("SAVEPOINT sp1", ())?;
db.execute("INSERT INTO users VALUES ($1, $2)", (2, "Bob"))?;
db.execute("ROLLBACK TO SAVEPOINT sp1", ())?;  // Undo Bob
db.execute("RELEASE SAVEPOINT sp1", ())?;       // Release savepoint
db.execute("COMMIT", ())?;  // Only Alice is inserted
```

### Isolation Levels

```rust
// Programmatic
let mut tx = db.begin_with_isolation(IsolationLevel::Snapshot)?;

// SQL-based
db.execute("BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT", ())?;

// Set default for all new transactions
db.set_default_isolation_level(IsolationLevel::Snapshot)?;
```

Available levels: `ReadCommitted` (default), `Snapshot`, `RepeatableRead`, `Serializable`, `ReadUncommitted`.

## Working with Rows

### Rows Iterator

`query()` returns a `Rows` struct that implements `Iterator<Item = Result<ResultRow>>`.

```rust
let mut rows = db.query("SELECT * FROM users", ())?;

// Column metadata
let columns: &[String] = rows.columns();
let count: usize = rows.column_count();

// Iterate
for row in rows {
    let row = row?;
    // ...
}

// Or collect
let all: Vec<ResultRow> = db.query("SELECT * FROM users", ())?.collect_vec()?;
```

### Zero-Clone Cursor

For bulk serialization, `advance()` / `current_row()` avoids per-row cloning.

```rust
let mut rows = db.query("SELECT * FROM users", ())?;
while rows.advance() {
    let row: &Row = rows.current_row();
    // Access row by reference — no clone
    if let Some(value) = row.get(0) {
        // ...
    }
}
```

### ResultRow Accessors

```rust
let row = db.query("SELECT id, name, active FROM users", ())?
    .next().unwrap()?;

// By index (0-based)
let id: i64 = row.get(0)?;

// By column name (case-insensitive)
let name: String = row.get_by_name("name")?;

// Optional values (for nullable columns)
let active: Option<bool> = row.get_by_name("active")?;

// Raw Value access
let value: Option<&Value> = row.get_value(0);

// NULL check
if row.is_null(2) {
    println!("active is NULL");
}

// Metadata
let columns: &[String] = row.columns();
let len: usize = row.len();
let empty: bool = row.is_empty();

// Get underlying Row
let inner: &Row = row.as_row();
let owned: Row = row.into_inner();
```

### Type Conversions (FromValue -- reading)

Types you can use with `row.get::<T>()`:

| SQL Type | Rust Type |
|----------|-----------|
| INTEGER | `i64`, `i32`, `f64`, `bool`, `String`, `Value` |
| FLOAT | `f64`, `i64`, `i32`, `String`, `Value` |
| TEXT | `String`, `Value` |
| BOOLEAN | `bool`, `Value` |
| TIMESTAMP | `String` (ISO format), `Value` |
| JSON | `String`, `Value` |
| NULL | `Option<T>` for any supported `T` |

### Parameter Types (ToParam -- writing)

Types you can pass as SQL parameters:

| Rust Type | SQL Value |
|-----------|-----------|
| `i8`, `i16`, `i32`, `i64` | INTEGER |
| `u8`, `u16`, `u32`, `usize` | INTEGER |
| `f32`, `f64` | FLOAT |
| `bool` | BOOLEAN |
| `&str`, `String`, `Arc<str>` | TEXT |
| `DateTime<Utc>` | TIMESTAMP |
| `Value` | as-is |
| `Option<T>` | T or NULL |

Tuple parameters support up to 12 elements. For more, use `params![]` macro or `Vec<Value>`.

## Error Handling

```rust
use stoolap::{Database, Error};

match db.execute("INSERT INTO users VALUES (1, 'Alice')", ()) {
    Ok(affected) => println!("Inserted {} rows", affected),
    Err(e) => {
        let msg = e.to_string();
        if msg.contains("UNIQUE constraint") {
            println!("Duplicate key");
        } else if msg.contains("write-write conflict") {
            println!("Transaction conflict — retry");
        } else {
            return Err(e.into());
        }
    }
}
```

## Thread Safety

`Database` is `Send + Sync`. Clone to share across threads — each clone has its own executor with independent transaction state but shares the same storage engine.

```rust
use std::sync::Arc;
use std::thread;

let db = Arc::new(Database::open("memory://")?);
db.execute("CREATE TABLE counter (id INTEGER PRIMARY KEY, value INTEGER)", ())?;
db.execute("INSERT INTO counter VALUES (1, 0)", ())?;

let handles: Vec<_> = (0..4).map(|_| {
    let db = Arc::clone(&db);
    thread::spawn(move || {
        for _ in 0..100 {
            db.execute("UPDATE counter SET value = value + 1 WHERE id = 1", ()).unwrap();
        }
    })
}).collect();

for h in handles {
    h.join().unwrap();
}

let count: i64 = db.query_one("SELECT value FROM counter WHERE id = 1", ())?;
```

## Complete Example

```rust
use stoolap::{Database, FromRow, ResultRow, Result, named_params, params};

struct User {
    id: i64,
    name: String,
    email: Option<String>,
}

impl FromRow for User {
    fn from_row(row: &ResultRow) -> Result<Self> {
        Ok(User {
            id: row.get(0)?,
            name: row.get(1)?,
            email: row.get(2)?,
        })
    }
}

fn main() -> Result<()> {
    let db = Database::open("file:///tmp/myapp.db")?;

    // Schema
    db.execute("
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            name TEXT NOT NULL,
            email TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    ", ())?;

    db.execute("CREATE INDEX IF NOT EXISTS idx_email ON users(email)", ())?;

    // Batch insert with cached plan
    let plan = db.cached_plan("INSERT INTO users (name, email) VALUES ($1, $2)")?;
    db.execute_plan(&plan, ("Alice", "alice@example.com"))?;
    db.execute_plan(&plan, ("Bob", "bob@example.com"))?;

    // Query with struct mapping
    let users: Vec<User> = db.query_as("SELECT id, name, email FROM users", ())?;
    for user in &users {
        println!("{}: {} ({:?})", user.id, user.name, user.email);
    }

    // Aggregation
    let count: i64 = db.query_one("SELECT COUNT(*) FROM users", ())?;
    println!("Total users: {}", count);

    // Window function
    for row in db.query("
        SELECT name, ROW_NUMBER() OVER (ORDER BY created_at) as row_num
        FROM users
    ", ())? {
        let row = row?;
        let name: String = row.get_by_name("name")?;
        let num: i64 = row.get_by_name("row_num")?;
        println!("{}: {}", num, name);
    }

    db.close()?;
    Ok(())
}
```

## Method Reference

### Database

| Method | Returns | Description |
|--------|---------|-------------|
| `open(dsn)` | `Result<Database>` | Open or reuse a database by DSN |
| `open_in_memory()` | `Result<Database>` | Open a unique in-memory database |
| `execute(sql, params)` | `Result<i64>` | Execute DDL/DML, return rows affected |
| `query(sql, params)` | `Result<Rows>` | Execute SELECT, return row iterator |
| `query_one(sql, params)` | `Result<T>` | Query single value |
| `query_opt(sql, params)` | `Result<Option<T>>` | Query optional value |
| `execute_named(sql, params)` | `Result<i64>` | Execute with named params |
| `query_named(sql, params)` | `Result<Rows>` | Query with named params |
| `query_one_named(sql, params)` | `Result<T>` | Single value with named params |
| `query_as(sql, params)` | `Result<Vec<T>>` | Query and map to structs |
| `query_as_named(sql, params)` | `Result<Vec<T>>` | Map to structs with named params |
| `execute_with_timeout(sql, params, ms)` | `Result<i64>` | Execute with timeout |
| `query_with_timeout(sql, params, ms)` | `Result<Rows>` | Query with timeout |
| `cached_plan(sql)` | `Result<CachedPlanRef>` | Create a cached execution plan |
| `execute_plan(plan, params)` | `Result<i64>` | Execute cached plan |
| `query_plan(plan, params)` | `Result<Rows>` | Query with cached plan |
| `execute_named_plan(plan, params)` | `Result<i64>` | Execute cached plan with named params |
| `query_named_plan(plan, params)` | `Result<Rows>` | Query cached plan with named params |
| `prepare(sql)` | `Result<Statement>` | Create a prepared statement |
| `begin()` | `Result<Transaction>` | Begin transaction (ReadCommitted) |
| `begin_with_isolation(level)` | `Result<Transaction>` | Begin with isolation level |
| `close()` | `Result<()>` | Close database, release file lock |
| `table_exists(name)` | `Result<bool>` | Check if table exists |
| `dsn()` | `&str` | Get the DSN |
| `set_default_isolation_level(level)` | `Result<()>` | Set default isolation |
| `create_snapshot()` | `Result<()>` | Create point-in-time snapshot |
| `semantic_cache_stats()` | `Result<Stats>` | Get cache statistics |
| `clear_semantic_cache()` | `Result<()>` | Clear query cache |

### Statement

| Method | Returns | Description |
|--------|---------|-------------|
| `execute(params)` | `Result<i64>` | Execute DML |
| `query(params)` | `Result<Rows>` | Query rows |
| `query_one(params)` | `Result<T>` | Single value |
| `query_opt(params)` | `Result<Option<T>>` | Optional value |
| `sql()` | `&str` | Get the SQL text |

### Transaction

| Method | Returns | Description |
|--------|---------|-------------|
| `execute(sql, params)` | `Result<i64>` | Execute DML |
| `query(sql, params)` | `Result<Rows>` | Query rows |
| `query_one(sql, params)` | `Result<T>` | Single value |
| `query_opt(sql, params)` | `Result<Option<T>>` | Optional value |
| `execute_named(sql, params)` | `Result<i64>` | Execute with named params |
| `query_named(sql, params)` | `Result<Rows>` | Query with named params |
| `execute_prepared(stmt, params)` | `Result<i64>` | Execute pre-parsed statement |
| `commit()` | `Result<()>` | Commit |
| `rollback()` | `Result<()>` | Rollback |
| `id()` | `i64` | Transaction ID |

### Rows

| Method | Returns | Description |
|--------|---------|-------------|
| `next()` | `Option<Result<ResultRow>>` | Iterator next |
| `advance()` | `bool` | Zero-clone cursor advance |
| `current_row()` | `&Row` | Current row reference |
| `columns()` | `&[String]` | Column names |
| `column_count()` | `usize` | Number of columns |
| `rows_affected()` | `i64` | DML rows affected |
| `collect_vec()` | `Result<Vec<ResultRow>>` | Collect all rows |
| `close()` | `()` | Close result set |

### ResultRow

| Method | Returns | Description |
|--------|---------|-------------|
| `get(index)` | `Result<T>` | Value by column index |
| `get_by_name(name)` | `Result<T>` | Value by column name |
| `get_value(index)` | `Option<&Value>` | Raw Value reference |
| `is_null(index)` | `bool` | Check if NULL |
| `columns()` | `&[String]` | Column names |
| `len()` | `usize` | Number of columns |
| `is_empty()` | `bool` | Empty check |
| `as_row()` | `&Row` | Reference to inner Row |
| `into_inner()` | `Row` | Consume into Row |
