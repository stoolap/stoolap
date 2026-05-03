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
stoolap = "0.4"
```

## Database

The `Database` struct is the main entry point. It is thread-safe and can be shared across threads via cloning.

### Opening a Database

```rust
use stoolap::Database;

// In-memory (unique instance per call)
let db = Database::open_in_memory()?;

// In-memory (shared instance, same DSN returns same engine)
let db = Database::open("memory://")?;

// File-based (persistent)
let db = Database::open("file:///path/to/database")?;

// File-based with configuration
let db = Database::open("file:///path/to/db?sync=full&checkpoint_interval=60")?;

// Read-only handles come from the dedicated entry point — returns
// `ReadOnlyDatabase`, which has no `execute` / `begin` methods so write
// SQL is a compile-time error rather than a runtime ReadOnlyViolation.
let ro = Database::open_read_only("file:///path/to/database")?;
// The DSN flag is accepted as redundant (driver DSN strings stay valid):
let ro = Database::open_read_only("file:///path/to/database?read_only=true")?;
// SQLite-style alias also accepted on this entry point:
let ro = Database::open_read_only("file:///path/to/database?mode=ro")?;

// `Database::open(dsn)` REJECTS read-only DSN flags with a clear
// migration error pointing to `open_read_only`. The type system
// enforces the read-only contract instead of runtime checks.

// Wrap an existing writable handle as read-only (shares the engine,
// in-process — no SWMR coordination concerns).
let ro = db.as_read_only();
```

`open_in_memory()` creates a unique, isolated instance each time. `open("memory://")` returns the same shared engine for the same DSN.

Read-only opens take a *shared* file lock (`LOCK_SH`) so multiple reader processes can coexist; a writable open is refused while any reader is alive (and vice versa). `open_read_only` refuses to materialize a fresh database: the path must already exist as a stoolap directory. Read-only opens succeed on directories mounted read-only at the kernel level AND on chmod-read-only directories (where they hold a long-lived `LOCK_SH` so a privileged writer cannot reclaim WAL/volumes under the reader).

### Connection String Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sync` / `sync_mode` | `normal` | Sync mode: `none`, `normal`, `full` (or `0`, `1`, `2`) |
| `checkpoint_interval` | `60` | Seconds between automatic checkpoint cycles |
| `compact_threshold` | `4` | Sub-target volumes per table before merging |
| `keep_snapshots` | `3` | Backup snapshots to retain per table |
| `wal_flush_trigger` | `32768` | WAL flush trigger size in bytes |
| `wal_buffer_size` | `65536` | WAL buffer size in bytes |
| `wal_max_size` | `67108864` | Max WAL file size before rotation (64 MB) |
| `commit_batch_size` | `100` | Commits to batch before syncing (normal mode) |
| `sync_interval_ms` | `1000` | Minimum ms between syncs (normal mode) |
| `wal_compression` | `on` | LZ4 compression for WAL entries |
| `compression` | -- | Alias that sets both `wal_compression` and `volume_compression` |
| `compression_threshold` | `64` | Minimum bytes before compressing an entry |
| `volume_compression` | `on` | LZ4 compression for cold volume files |
| `checkpoint_on_close` | `on` | Seal all hot rows to volumes on clean shutdown |
| `target_volume_rows` | `1048576` | Target rows per cold volume. Controls compaction split boundary. |
| `read_only` / `readonly` | `false` | Read-only mode. Pass to `Database::open_read_only(dsn)` — `Database::open(dsn)` rejects this flag with a migration error. |
| `mode` | `rw` | SQLite-style alias for `read_only`. `mode=ro` matches `read_only=true`. Same routing rule. |

### execute()

Execute DDL or DML statements. Returns the number of rows affected.

```rust
use stoolap::params;

// DDL, no parameters
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
db.set_default_isolation_level(IsolationLevel::SnapshotIsolation)?;

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
let mut tx = db.begin_with_isolation(IsolationLevel::SnapshotIsolation)?;

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
// Prepare once (via Database)
let insert = db.prepare("INSERT INTO users VALUES ($1, $2, $3)")?;

let mut tx = db.begin()?;
for (id, name, age) in data {
    tx.execute_prepared(&insert, (id, name, age))?;
}
tx.commit()?;
```

Query with a prepared statement inside a transaction:

```rust
let lookup = db.prepare("SELECT * FROM users WHERE id = $1")?;

let mut tx = db.begin()?;
// Reads within the transaction see uncommitted changes
let rows = tx.query_prepared(&lookup, (42,))?;
tx.commit()?;
```

Pre-parsed statements also work with named parameters, combining parse-once performance with `:name`-style bindings:

```rust
let insert = db.prepare("INSERT INTO users VALUES (:id, :name, :age)")?;

let mut tx = db.begin()?;
tx.execute_prepared_named(&insert, named_params!{ id: 1, name: "Alice", age: 30 })?;
tx.execute_prepared_named(&insert, named_params!{ id: 2, name: "Bob", age: 25 })?;
tx.commit()?;

// Query variant
let lookup = db.prepare("SELECT * FROM users WHERE id = :id")?;
let mut tx = db.begin()?;
let rows = tx.query_prepared_named(&lookup, named_params!{ id: 1 })?;
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
let mut tx = db.begin_with_isolation(IsolationLevel::SnapshotIsolation)?;

// SQL-based
db.execute("BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT", ())?;

// Set default for all new transactions
db.set_default_isolation_level(IsolationLevel::SnapshotIsolation)?;
```

The Rust API has two enum variants: `ReadCommitted` (default) and `SnapshotIsolation`. At the SQL level, `SNAPSHOT`, `SERIALIZABLE`, `REPEATABLE READ` all map to `SnapshotIsolation`, and `READ UNCOMMITTED` maps to `ReadCommitted`.

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
    // Access row by reference, no clone
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

### Type Conversions (FromValue, reading)

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

### Parameter Types (ToParam, writing)

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
            println!("Transaction conflict, retry");
        } else {
            return Err(e.into());
        }
    }
}
```

## Thread Safety

`Database` is `Send + Sync`. Clone to share across threads. Each clone has its own executor with independent transaction state but shares the same storage engine.

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
| `open(dsn)` | `Result<Database>` | Open or reuse a writable database by DSN. **Rejects `?read_only=true` / `?readonly=true` / `?mode=ro`** with a migration error pointing to `open_read_only`. |
| `open_in_memory()` | `Result<Database>` | Open a unique in-memory database |
| `open_read_only(dsn)` | `Result<ReadOnlyDatabase>` | Open an existing database read-only (shared file lock; refuses to create a fresh DB). Accepts `?read_only=true` / `?mode=ro` as redundant; rejects `?read_only=false` / `?mode=rw` (contradicts the function name). |
| `as_read_only()` | `ReadOnlyDatabase` | Return an in-process read-only view sharing this Database's engine |
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
| `is_read_only()` | `bool` | Always `false` — `Database` is always writable. Read-only handles are `ReadOnlyDatabase` (whose `is_read_only()` always returns `true`). |
| `set_default_isolation_level(level)` | `Result<()>` | Set default isolation |
| `create_snapshot()` | `Result<()>` | Create point-in-time snapshot |
| `semantic_cache_stats()` | `Result<Stats>` | Get cache statistics |
| `clear_semantic_cache()` | `Result<()>` | Clear query cache |

### ReadOnlyDatabase

Returned by `Database::open_read_only(dsn)` and `Database::as_read_only()`. The type has NO `execute` / `begin` / `prepare` methods — write SQL is a *compile-time* error rather than a runtime `Error::ReadOnlyViolation`. Read SQL (SELECT, SHOW, EXPLAIN) goes through `query` / `query_named` / `cached_plan` + `query_plan`. Read-only transactions inside an explicit `BEGIN ... COMMIT` work via SQL — the Rust `begin()` API is intentionally absent.

`ReadOnlyDatabase` is a *view*, not a connection sharing a session with the source `Database`. Each handle owns its own executor and transaction state, so an uncommitted `BEGIN` on the source `Database` is **not** visible through `as_read_only()`. To observe uncommitted writes, run the read SQL inside the same `Transaction`.

For cross-process readers (`Database::open_read_only` against a `file://` DSN), the handle picks up writer **checkpoints** via the manifest-epoch poll. Typed must-reopen errors (`Error::SwmrPendingDdl` for post-attach DDL, `Error::SwmrWriterReincarnated` for writer crash + restart) are sticky once raised; reopen the handle to apply.

**Visibility is checkpoint-bounded.** A read-only handle sees writer state as of the writer's most recent checkpoint. Commits the writer has accepted but not yet checkpointed (rows in the writer's hot buffer + WAL tail) are NOT visible to query execution on the read-only handle. To make a writer commit visible across processes, the writer must run `PRAGMA CHECKPOINT` (or wait for the periodic background checkpoint at `checkpoint_interval`, default 60s). The WAL-tail overlay infrastructure exists on the read-only side but is not yet wired into query execution; sub-checkpoint visibility ships in a follow-up phase.

For prepared-statement ergonomics, use `cached_plan(sql)` plus `query_plan` / `query_named_plan` — same parse-once / execute-many shape `Database::prepare` provides.

| Method | Returns | Description |
|--------|---------|-------------|
| `query(sql, params)` | `Result<Rows>` | Execute a read-only query |
| `query_named(sql, params)` | `Result<Rows>` | Read-only query with named params |
| `cached_plan(sql)` | `Result<CachedPlanRef>` | Parse once and cache; refuses write SQL |
| `query_plan(plan, params)` | `Result<Rows>` | Execute a cached plan with positional params |
| `query_named_plan(plan, params)` | `Result<Rows>` | Execute a cached plan with named params |
| `dsn()` | `&str` | Get the DSN |
| `is_read_only()` | `bool` | Always `true`. |
| `table_exists(name)` | `Result<bool>` | Check if a table exists |
| `refresh()` | `Result<bool>` | Force a manifest reload to pick up any new writer checkpoint; returns `true` if state advanced. Visibility is checkpoint-bounded — uncheckpointed writer commits are not yet exposed to query execution. |
| `set_auto_refresh(enabled)` | `()` | Master switch for implicit refresh. `false` pauses BOTH the per-query auto-refresh path AND the background ticker (if any) — the snapshot only moves on explicit `refresh()`. WAL pin advancement also stalls; keep stable windows short. |
| `auto_refresh_enabled()` | `bool` | Read the auto-refresh flag |
| `set_refresh_interval(Option<Duration>)` | `Result<()>` | Configure the background refresh ticker. `Some(d)` spawns a thread calling `refresh()` every `d` (min 100ms); `None` stops it. Use for idle handles so the WAL pin advances. Pauses while `auto_refresh=false` or a `BEGIN` is active. Equivalent DSN flag: `?refresh_interval=30s`. |
| `refresh_interval()` | `Option<Duration>` | Currently configured ticker interval, or `None` if no ticker is running. |
| `try_clone()` | `Self` | Clone for multi-threaded use. Each clone has its own executor, WAL pin, auto_refresh flag, and ticker (inherits parent's interval). |
| `set_swmr_overlay_enabled(enabled)` | `Result<()>` | Opt into per-row WAL-tail materialization. Off by default; DDL detection is always on. |
| `swmr_overlay_enabled()` | `bool` | Whether overlay materialization is enabled |
| `read_engine()` | `Arc<dyn ReadEngine>` | Get the underlying read engine for libraries accepting `&dyn ReadEngine` |

```rust
let ro = Database::open_read_only("file:///data/mydb")?;
let plan = ro.cached_plan("SELECT name FROM users WHERE age > $1")?;
for age in [18, 25, 40] {
    for row in ro.query_plan(&plan, (age,))? {
        let row = row?;
        println!("{}", row.get::<String>("name")?);
    }
}
```

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
| `query_prepared(stmt, params)` | `Result<Rows>` | Query with pre-parsed statement |
| `execute_prepared_named(stmt, params)` | `Result<i64>` | Execute pre-parsed statement with named params |
| `query_prepared_named(stmt, params)` | `Result<Rows>` | Query pre-parsed statement with named params |
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
