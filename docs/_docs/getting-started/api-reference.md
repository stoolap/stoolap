---
layout: doc
title: API Reference
category: Getting Started
order: 4
---

# Stoolap API Reference

This document provides a comprehensive reference for the Stoolap Rust API.

## Database API

The `Database` struct is the main entry point for using Stoolap.

### Opening a Database

```rust
use stoolap::Database;

// Open an in-memory database (unique instance)
let db = Database::open_in_memory()?;

// Open an in-memory database (shared instance)
let db = Database::open("memory://")?;

// Open a persistent database
let db = Database::open("file:///path/to/database")?;

// Open with configuration options
let db = Database::open("file:///path/to/database?sync_mode=full&snapshot_interval=60")?;
```

### Connection String Options

| Parameter | Description | Values |
|-----------|-------------|--------|
| sync_mode | WAL sync mode | none, normal, full (or 0, 1, 2) |
| snapshot_interval | Snapshot interval in seconds | Integer |
| keep_snapshots | Number of snapshots to keep | Integer |
| wal_flush_trigger | Operations before WAL flush | Integer |
| compression | Enable compression | on, off |

## Executing Queries

### execute()

Execute DDL or DML statements that don't return rows.

```rust
use stoolap::{Database, params};

let db = Database::open("memory://")?;

// DDL - no parameters
db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())?;

// DML with positional parameters ($1, $2, ...)
db.execute("INSERT INTO users VALUES ($1, $2)", (1, "Alice"))?;

// DML with params! macro
db.execute("INSERT INTO users VALUES ($1, $2)", params![2, "Bob"])?;

// Returns rows affected
let affected = db.execute("UPDATE users SET name = $1 WHERE id = $2", ("Charlie", 1))?;
println!("Updated {} rows", affected);
```

### query()

Execute a SELECT query and iterate over results.

```rust
// Query all rows
for row in db.query("SELECT * FROM users", ())? {
    let row = row?;
    let id: i64 = row.get(0)?;           // By index
    let name: String = row.get_by_name("name")?; // By column name
    println!("User {}: {}", id, name);
}

// Query with parameters
for row in db.query("SELECT * FROM users WHERE id > $1", (10,))? {
    let row = row?;
    // Process row...
}

// Collect into Vec
let users: Vec<_> = db.query("SELECT * FROM users", ())?
    .collect::<Result<Vec<_>, _>>()?;
```

### execute_with_timeout()

Execute a write statement with a timeout. The query is cancelled if it exceeds the specified timeout.

```rust
// Execute with 5 second timeout (timeout in milliseconds)
db.execute_with_timeout(
    "DELETE FROM large_table WHERE created_at < $1",
    ("2020-01-01",),
    5000  // 5000ms = 5 seconds
)?;

// Use 0 for no timeout
db.execute_with_timeout("UPDATE users SET active = true", (), 0)?;
```

### query_with_timeout()

Execute a query with a timeout. The query is cancelled if it exceeds the specified timeout.

```rust
// Query with 10 second timeout
for row in db.query_with_timeout(
    "SELECT * FROM large_table WHERE complex_condition",
    (),
    10000  // 10000ms = 10 seconds
)? {
    let row = row?;
    // Process row...
}

// Useful for preventing long-running queries from blocking
let results = db.query_with_timeout(
    "SELECT * FROM orders WHERE status = $1",
    ("pending",),
    3000  // 3 second timeout
)?;
```

### query_one()

Execute a query that returns a single value.

```rust
// Get a single value
let count: i64 = db.query_one("SELECT COUNT(*) FROM users", ())?;

// With parameters
let name: String = db.query_one("SELECT name FROM users WHERE id = $1", (1,))?;
```

### query_opt()

Execute a query that returns an optional single value.

```rust
// Returns None if no rows match
let name: Option<String> = db.query_opt(
    "SELECT name FROM users WHERE id = $1",
    (999,)
)?;

match name {
    Some(n) => println!("Found: {}", n),
    None => println!("Not found"),
}
```

### Named Parameters

Use named parameters with `:name` syntax.

```rust
use stoolap::{Database, named_params};

let db = Database::open("memory://")?;
db.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)", ())?;

// Insert with named parameters
db.execute_named(
    "INSERT INTO users VALUES (:id, :name, :age)",
    named_params!{ id: 1, name: "Alice", age: 30 }
)?;

// Query with named parameters
for row in db.query_named(
    "SELECT * FROM users WHERE age > :min_age",
    named_params!{ min_age: 25 }
)? {
    // Process rows...
}
```

## Prepared Statements

Prepare a statement for repeated execution.

```rust
// Prepare once
let stmt = db.prepare("SELECT * FROM users WHERE id = $1")?;

// Execute multiple times with different parameters
for id in 1..=10 {
    for row in stmt.query((id,))? {
        let row = row?;
        // Process row...
    }
}
```

## Transactions

### Basic Transactions

```rust
// Begin transaction
db.execute("BEGIN", ())?;

// Execute statements
db.execute("INSERT INTO users VALUES ($1, $2)", (1, "Alice"))?;
db.execute("UPDATE accounts SET balance = balance - 100 WHERE id = $1", (1,))?;

// Commit or rollback
db.execute("COMMIT", ())?;
// Or: db.execute("ROLLBACK", ())?;
```

### Savepoints

```rust
db.execute("BEGIN", ())?;

db.execute("INSERT INTO users VALUES ($1, $2)", (1, "Alice"))?;
db.execute("SAVEPOINT sp1", ())?;

db.execute("INSERT INTO users VALUES ($1, $2)", (2, "Bob"))?;
// Oops, undo the second insert
db.execute("ROLLBACK TO SAVEPOINT sp1", ())?;

db.execute("COMMIT", ())?;
// Only Alice is inserted
```

### Isolation Levels

```rust
// Snapshot isolation (prevents lost updates)
db.execute("BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT", ())?;
// ... your queries ...
db.execute("COMMIT", ())?;

// Read committed (default, higher concurrency)
db.execute("BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED", ())?;
// ... your queries ...
db.execute("COMMIT", ())?;
```

## Working with Rows

### Accessing Column Values

```rust
for row in db.query("SELECT id, name, active FROM users", ())? {
    let row = row?;

    // By index (0-based)
    let id: i64 = row.get(0)?;

    // By column name
    let name: String = row.get_by_name("name")?;

    // Optional values (for nullable columns)
    let active: Option<bool> = row.get_by_name("active")?;
}
```

### Supported Types

| SQL Type | Rust Type |
|----------|-----------|
| INTEGER | i64, i32, i16, i8 |
| FLOAT | f64, f32 |
| TEXT | String, &str |
| BOOLEAN | bool |
| TIMESTAMP | String (ISO format) |
| JSON | String |
| NULL | Option<T> |

## Error Handling

```rust
use stoolap::{Database, Error};

let db = Database::open("memory://")?;

match db.execute("INSERT INTO users VALUES (1, 'Alice')", ()) {
    Ok(affected) => println!("Inserted {} rows", affected),
    Err(e) => {
        let msg = e.to_string();
        if msg.contains("UNIQUE constraint") {
            println!("Duplicate key error");
        } else if msg.contains("write-write conflict") {
            println!("Transaction conflict - should retry");
        } else {
            return Err(e.into());
        }
    }
}
```

## Thread Safety

The `Database` struct is thread-safe and can be shared across threads:

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
println!("Final count: {}", count);
```

## Complete Example

```rust
use stoolap::{Database, params};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open persistent database
    let db = Database::open("file:///tmp/myapp.db")?;

    // Create schema
    db.execute("
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            name TEXT NOT NULL,
            email TEXT,
            created_at TIMESTAMP
        )
    ", ())?;

    // Create index
    db.execute("CREATE INDEX IF NOT EXISTS idx_email ON users(email)", ())?;

    // Insert data in a transaction
    db.execute("BEGIN", ())?;
    db.execute(
        "INSERT INTO users (name, email, created_at) VALUES ($1, $2, NOW())",
        ("Alice", "alice@example.com")
    )?;
    db.execute(
        "INSERT INTO users (name, email, created_at) VALUES ($1, $2, NOW())",
        ("Bob", "bob@example.com")
    )?;
    db.execute("COMMIT", ())?;

    // Query with aggregation
    let count: i64 = db.query_one("SELECT COUNT(*) FROM users", ())?;
    println!("Total users: {}", count);

    // Query with window function
    for row in db.query("
        SELECT name, email,
               ROW_NUMBER() OVER (ORDER BY created_at) as row_num
        FROM users
    ", ())? {
        let row = row?;
        let name: String = row.get_by_name("name")?;
        let row_num: i64 = row.get_by_name("row_num")?;
        println!("{}: {}", row_num, name);
    }

    Ok(())
}
```

## Best Practices

1. **Use transactions for related operations**: Wrap multiple statements in BEGIN/COMMIT
2. **Handle conflicts in SNAPSHOT isolation**: Be prepared to retry on write-write conflicts
3. **Use parameterized queries**: Prevents SQL injection
4. **Create indexes for frequently filtered columns**: Improves query performance
5. **Use `query_one` for single values**: Simpler than iterating
6. **Close database properly**: The database is automatically closed when dropped
7. **Run ANALYZE after bulk inserts**: Updates optimizer statistics
8. **Use timeouts for untrusted queries**: Use `execute_with_timeout` and `query_with_timeout` to prevent long-running queries from blocking your application
