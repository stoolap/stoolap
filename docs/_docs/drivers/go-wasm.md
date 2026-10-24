---
layout: doc
title: Go WASM Driver
category: Drivers
order: 11
icon: go
---

# Go WASM Driver

Pure Go driver for Stoolap using WebAssembly. No CGO, no shared libraries, no platform-specific binaries. The full Stoolap engine runs as a WASM module inside your Go process via [wazero](https://wazero.io/).

For the native driver, which loads `libstoolap` as a shared library without CGO, see [Go Driver]({{ '/docs/drivers/go/' | relative_url }}).

## When to Use

| | Native Driver | WASM Driver |
|---|---|---|
| **Package** | `github.com/stoolap/stoolap-go` | `github.com/stoolap/stoolap-go/wasm` |
| **CGO required** | No | No |
| **Native dependency** | `libstoolap` shared library (bundled for macOS arm64, Linux x64, Windows x64) | None |
| **Threading** | Full (parallel queries) | Single-threaded |
| **File persistence** | Full (automatic maintenance) | Works (manual maintenance) |
| **Cross-compile** | Needs `libstoolap` for the target | Anywhere Go compiles |
| **Best for** | Production, max throughput | Portability, zero dependencies |

The two drivers expose the same API: positional and named parameters, prepared statements with batches, transactions, chunked row iteration, one-call result sets and `database/sql`.

## Requirements

- Go 1.25+
- A `stoolap.wasm` built from Stoolap 0.5.0 or newer. `NewEngine` reports which export an older binary lacks.

## Installation

```bash
go get github.com/stoolap/stoolap-go/wasm
```

A prebuilt `stoolap.wasm` (5 MB) is included in the module and available on the [releases page](https://github.com/stoolap/stoolap-go/releases).

## Quick Start

### Direct API

```go
package main

import (
    "context"
    "fmt"
    "os"

    "github.com/stoolap/stoolap-go/wasm"
)

func main() {
    ctx := context.Background()

    wasmBytes, _ := os.ReadFile("stoolap.wasm")
    engine, _ := wasm.NewEngine(ctx, wasmBytes)
    defer engine.Close(ctx)

    db, _ := engine.OpenMemory(ctx)
    defer db.Close()

    db.Exec(ctx, "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    db.Exec(ctx, "INSERT INTO users VALUES (1, 'Alice', 30), (2, 'Bob', 25)")

    rows, _ := db.Query(ctx, "SELECT id, name, age FROM users ORDER BY id")
    defer rows.Close()

    for rows.Next() {
        var id int64
        var name string
        var age int64
        rows.Scan(&id, &name, &age)
        fmt.Printf("id=%d name=%s age=%d\n", id, name, age)
    }
}
```

### database/sql Driver

The WASM driver registers as `"stoolap-wasm"` for use with `database/sql`. Call `SetWASM()` once before opening connections:

```go
package main

import (
    "context"
    "database/sql"
    "fmt"
    "os"

    "github.com/stoolap/stoolap-go/wasm"
)

func main() {
    ctx := context.Background()
    wasmBytes, _ := os.ReadFile("stoolap.wasm")
    wasm.SetWASM(ctx, wasmBytes)

    db, _ := sql.Open("stoolap-wasm", "memory://")
    defer db.Close()

    db.ExecContext(ctx, "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    db.ExecContext(ctx, "INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')")

    rows, _ := db.QueryContext(ctx, "SELECT id, name FROM users ORDER BY id")
    defer rows.Close()

    for rows.Next() {
        var id int64
        var name string
        rows.Scan(&id, &name)
        fmt.Printf("id=%d name=%s\n", id, name)
    }
}
```

## Connection Strings

| DSN | Description |
|-----|-------------|
| `memory://` | In-memory database (unique, isolated instance) |
| `memory://mydb` | Named in-memory database (same name shares the engine) |
| `file:///data/mydb` | File-based persistent database (requires `NewEngineWithFS`) |

See [Connection String Reference]({{ '/docs/getting-started/connection-strings/' | relative_url }}) for all options.

## Parameters

Positional parameters are `$1`, `$2`, etc. or `?`. Named parameters are `:name`; a name may be used more than once in a statement. A statement uses one style or the other, and every parameter it names must be supplied.

```go
// Direct API, positional
db.ExecParams(ctx, "INSERT INTO users VALUES ($1, $2)", []any{int64(1), "Alice"})
rows, _ := db.QueryParams(ctx, "SELECT * FROM users WHERE id = $1", []any{int64(1)})

// Direct API, named
db.ExecNamed(ctx, "INSERT INTO users VALUES (:id, :name)",
    []sql.NamedArg{sql.Named("id", int64(2)), sql.Named("name", "Bob")})
rows, _ := db.QueryNamed(ctx, "SELECT * FROM users WHERE id = :id",
    []sql.NamedArg{sql.Named("id", int64(2))})

// database/sql
db.ExecContext(ctx, "INSERT INTO users VALUES ($1, $2)", 1, "Alice")
rows, _ := db.QueryContext(ctx, "SELECT * FROM users WHERE id = :id", sql.Named("id", 1))
```

Parameter values may be `nil`, `int64`, `int`, `int32`, `float64`, `float32`, `bool`, `string`, `[]byte` (VECTOR or BLOB) and `time.Time`. A nil `[]byte` binds as NULL; an empty `[]byte{}` or an empty string binds as an empty value. There is no size limit on a parameter: the WASM arena grows to fit it.

## Transactions

```go
// Direct API
tx, _ := db.Begin(ctx)
tx.Exec(ctx, "INSERT INTO users VALUES (1, 'Alice')")
tx.ExecNamed(ctx, "INSERT INTO users VALUES (:id, :name)",
    []sql.NamedArg{sql.Named("id", int64(2)), sql.Named("name", "Bob")})
tx.Commit()

// Snapshot isolation
tx, _ := db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSnapshot})
rows, _ := tx.Query(ctx, "SELECT * FROM users")
// All reads see the same consistent snapshot
tx.Commit()

// Rollback
tx, _ := db.Begin(ctx)
tx.Exec(ctx, "DELETE FROM users")
tx.Rollback() // changes discarded
```

`Commit` and `Rollback` consume the transaction; using it afterwards returns an error.

## Prepared Statements

```go
stmt, _ := db.Prepare(ctx, "INSERT INTO users VALUES ($1, $2)")
defer stmt.Close()

// One row per call
stmt.ExecContext(ctx, []any{int64(1), "Alice"})

// Many rows in one WASM call, inside one transaction. A failing row rolls
// back the whole batch.
batch := make([][]any, 1000)
for i := range batch {
    batch[i] = []any{int64(i + 2), fmt.Sprintf("user_%d", i+2)}
}
res, _ := stmt.ExecBatch(ctx, batch)
n, _ := res.RowsAffected() // 1000

// Named parameters
upd, _ := db.Prepare(ctx, "UPDATE users SET name = :name WHERE id = :id")
upd.ExecNamed(ctx, []sql.NamedArg{sql.Named("name", "Carol"), sql.Named("id", int64(1))})
```

In `database/sql`, a statement prepared on a `*sql.Tx` runs inside that transaction.

## Reading Results

`Query` returns a `*Rows` cursor. Rows arrive from the engine in packed chunks (32 rows first, doubling to 256), so a scan crosses the WASM boundary once per chunk rather than once per row and column. Values are decoded in Go when you read them.

```go
rows, _ := db.Query(ctx, "SELECT id, name, score FROM users")
defer rows.Close()
for rows.Next() {
    var id int64
    var name string
    var score float64
    rows.Scan(&id, &name, &score)
}
if err := rows.Err(); err != nil {
    // the error that ended iteration
}
```

`Scan` accepts `*int64`, `*float64`, `*string`, `*bool`, `*time.Time`, `*[]byte`, the `sql.Null*` types and `*any`. The destination count must match the column count. Scanning a non-text column into a string gives the engine's own text form, the same as `CAST(x AS TEXT)`. A TEXT column in a recognised timestamp form scans into `time.Time`; anything else into `time.Time` is an error.

Typed accessors read the current row without a `Scan`: `rows.IsNull(i)`, `rows.Int64(i)`, `rows.Float64(i)`, `rows.Bool(i)` and `rows.Timestamp(i)`. They convert between numeric types the way the engine does.

### One-Call Result Sets

When the whole result is wanted at once, `QueryAll` fetches it in a single WASM call. There is no cursor to iterate or close. The `database/sql` driver takes this path for every query.

```go
rs, _ := db.QueryAll(ctx, "SELECT id, name FROM users WHERE id > $1", []any{int64(10)})
fmt.Println(rs.Columns) // [id name]
for _, row := range rs.Rows {
    fmt.Printf("id=%v name=%v\n", row[0], row[1])
}

// Named, prepared and transactional forms
rs, _ = db.QueryNamedAll(ctx, "SELECT * FROM users WHERE id = :id", []sql.NamedArg{sql.Named("id", int64(1))})
rs, _ = stmt.QueryAll(ctx, []any{int64(1)})
rs, _ = tx.QueryAll(ctx, "SELECT COUNT(*) FROM users", nil)
```

`rows.FetchAll()` does the same for the rows a cursor has not yet returned.

## File Persistence

Create an engine with filesystem access for persistent databases:

```go
engine, _ := wasm.NewEngineWithFS(ctx, wasmBytes, "/path/to/data")
db, _ := engine.Open(ctx, "file:///data/mydb")
```

The directory is mounted at `/data` inside the WASM sandbox. Use `file:///data/...` paths in the DSN.

### Manual Maintenance

WASM does not support background threads, so the automatic checkpoint cycle and cleanup do not run. You must call these commands periodically:

```go
// Checkpoint: seal hot rows to cold volumes, persist manifests, truncate WAL.
// Without this, all data stays in the hot buffer and WAL grows unbounded.
db.Exec(ctx, "PRAGMA checkpoint")

// Cleanup: remove deleted rows, old versions, compact indexes
db.Exec(ctx, "VACUUM")

// Backup: create a full .bin snapshot for disaster recovery
db.Exec(ctx, "PRAGMA snapshot")

// Statistics: update optimizer cost estimates
db.Exec(ctx, "ANALYZE my_table")
```

**Important**: `PRAGMA checkpoint` is the most critical command for file-based WASM databases. Without it, data stays in the hot buffer (high memory) and the WAL grows indefinitely. Call it periodically (e.g., every 60 seconds or after bulk writes).

`hot_max_bytes` does not hold commits back on WASM. Natively a commit over the limit waits for the background seal, which WASM does not have, so the hot buffer only shrinks when `PRAGMA checkpoint` runs.

For production file-based workloads with automatic background maintenance, use the [native driver]({{ '/docs/drivers/go/' | relative_url }}).

## Type Mapping

| SQL Type | Go Type | Nullable Go Type |
|----------|---------|------------------|
| INTEGER | `int64` | `sql.NullInt64` |
| FLOAT | `float64` | `sql.NullFloat64` |
| TEXT | `string` | `sql.NullString` |
| BOOLEAN | `bool` | `sql.NullBool` |
| TIMESTAMP | `time.Time` | `sql.NullTime` |
| JSON | `string` | `sql.NullString` |
| VECTOR/BLOB | `[]byte` | `[]byte` (nil for NULL) |

## Thread Safety

All WASM operations are serialized through a mutex. The engine is safe for concurrent use from multiple goroutines, but operations execute sequentially.

- **Engine**: Thread-safe (mutex-protected)
- **DB, Tx, Stmt**: Use from any goroutine (engine serializes access)
- **Rows**: One goroutine at a time; the cursor and its decoded chunk live in Go memory
- **database/sql**: Thread-safe by default (connection pool creates cloned handles)

## Building the WASM Binary from Source

Requires: Rust toolchain, `wasm32-wasip1` target, and [binaryen](https://github.com/WebAssembly/binaryen) (for `wasm-opt`).

```bash
# Install WASI target
rustup target add wasm32-wasip1

# Build (from the stoolap engine repo)
cd stoolap
cargo build --profile max --target wasm32-wasip1 --features ffi --no-default-features

# Optimize (31 MB -> 5 MB)
wasm-opt -Oz target/wasm32-wasip1/max/stoolap.wasm -o stoolap.wasm
```

## Performance

Every WASM call costs a boundary crossing, so the driver makes as few as it can:

- **Chunked cursors**: `Rows` reads packed chunks of up to 256 rows per call and decodes them in Go. Scanning 1000 rows of four columns takes about 80us, against 370us when each column was a call.
- **One-call results**: `QueryAll` and the `database/sql` driver receive the whole result packed from a single call, with no rows handle to open or close.
- **Batches**: `ExecBatch` sends every row of a batch in one call inside one transaction.
- **CallWithStack**: A pre-allocated call stack eliminates per-call allocations.
- **Arena allocator**: SQL text and parameters are written into a pre-allocated WASM memory region with no malloc or free per operation. The arena grows when an operation needs more than 256 KB.

Read-heavy workloads (SELECT, aggregation, JOINs) perform close to the native driver. Write-heavy workloads have roughly 1.5 to 2x overhead from the boundary crossing per operation; batches remove most of it.

## Direct API Reference

### Engine

| Method | Description |
|--------|-------------|
| `NewEngine(ctx, wasmBytes)` | Create engine for in-memory databases |
| `NewEngineWithFS(ctx, wasmBytes, rootDir)` | Create engine with filesystem access |
| `engine.Close(ctx)` | Release all resources |
| `engine.Version(ctx)` | Get engine version string |
| `engine.Open(ctx, dsn)` | Open database by DSN |
| `engine.OpenMemory(ctx)` | Open new in-memory database |

### DB

| Method | Description |
|--------|-------------|
| `db.Close()` | Close the connection |
| `db.Clone(ctx)` | Clone handle (shares engine) |
| `db.Exec(ctx, query)` | Execute without parameters |
| `db.ExecParams(ctx, query, args)` | Execute with positional parameters |
| `db.ExecNamed(ctx, query, args)` | Execute with named parameters |
| `db.Query(ctx, query)` | Query without parameters |
| `db.QueryParams(ctx, query, args)` | Query with positional parameters |
| `db.QueryNamed(ctx, query, args)` | Query with named parameters |
| `db.QueryAll(ctx, query, args)` | Whole result in one call, as a `ResultSet` |
| `db.QueryNamedAll(ctx, query, args)` | Same with named parameters |
| `db.Prepare(ctx, query)` | Create prepared statement |
| `db.Begin(ctx)` | Begin transaction (Read Committed) |
| `db.BeginTx(ctx, opts)` | Begin transaction with options |

### Rows

| Method | Description |
|--------|-------------|
| `rows.Next()` | Advance to next row |
| `rows.Scan(dest...)` | Read current row |
| `rows.Err()` | Error that ended iteration |
| `rows.Columns()` | Get column names |
| `rows.IsNull(i)`, `rows.Int64(i)`, `rows.Float64(i)`, `rows.Bool(i)`, `rows.Timestamp(i)` | Read one column of the current row |
| `rows.FetchAll()` | Remaining rows as `[][]any` in one call |
| `rows.Close()` | Close result set |

### ResultSet

| Field | Description |
|-------|-------------|
| `rs.Columns` | Column names |
| `rs.Rows` | Every row as `[]any` |

### Stmt

| Method | Description |
|--------|-------------|
| `stmt.ExecContext(ctx, args)` | Execute with positional parameters |
| `stmt.ExecNamed(ctx, args)` | Execute with named parameters |
| `stmt.ExecBatch(ctx, rows)` | Execute once per row in one transaction |
| `stmt.QueryContext(ctx, args)` | Query with positional parameters |
| `stmt.QueryNamed(ctx, args)` | Query with named parameters |
| `stmt.QueryAll(ctx, args)` | Whole result in one call |
| `stmt.QueryNamedAll(ctx, args)` | Same with named parameters |
| `stmt.Close()` | Destroy the statement |

### Tx

| Method | Description |
|--------|-------------|
| `tx.Exec(ctx, query)` | Execute within transaction |
| `tx.ExecParams(ctx, query, args)` | Execute with positional parameters |
| `tx.ExecNamed(ctx, query, args)` | Execute with named parameters |
| `tx.Query(ctx, query)` | Query within transaction |
| `tx.QueryParams(ctx, query, args)` | Query with positional parameters |
| `tx.QueryNamed(ctx, query, args)` | Query with named parameters |
| `tx.QueryAll(ctx, query, args)` | Whole result in one call |
| `tx.QueryNamedAll(ctx, query, args)` | Same with named parameters |
| `tx.Commit()` | Commit |
| `tx.Rollback()` | Rollback |
