---
layout: doc
title: Go WASM Driver
category: Drivers
order: 8
icon: go
---

# Go WASM Driver

Pure Go driver for Stoolap using WebAssembly. No CGO, no shared libraries, no platform-specific binaries. The full Stoolap engine runs as a WASM module inside your Go process via [wazero](https://wazero.io/).

For the native CGO driver, see [Go Driver]({{ '/docs/drivers/go/' | relative_url }}).

## When to Use

| | CGO Driver | WASM Driver |
|---|---|---|
| **Package** | `github.com/stoolap/stoolap-go` | `github.com/stoolap/stoolap-go/wasm` |
| **CGO required** | Yes | No |
| **Threading** | Full (parallel queries) | Single-threaded |
| **File persistence** | Full (automatic maintenance) | Works (manual maintenance) |
| **Cross-compile** | Needs C toolchain per target | Anywhere Go compiles |
| **Best for** | Production, max throughput | Portability, zero dependencies |

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

Use positional parameters `$1`, `$2`, etc.:

```go
// Direct API
db.ExecParams(ctx, "INSERT INTO users VALUES ($1, $2)", []any{int64(1), "Alice"})
rows, _ := db.QueryParams(ctx, "SELECT * FROM users WHERE id = $1", []any{int64(1)})

// database/sql
db.ExecContext(ctx, "INSERT INTO users VALUES ($1, $2)", 1, "Alice")
rows, _ := db.QueryContext(ctx, "SELECT * FROM users WHERE id = $1", 1)
```

## Transactions

```go
// Direct API
tx, _ := db.Begin(ctx)
tx.Exec(ctx, "INSERT INTO users VALUES (1, 'Alice')")
tx.Exec(ctx, "INSERT INTO users VALUES (2, 'Bob')")
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

## Prepared Statements

```go
stmt, _ := db.Prepare(ctx, "INSERT INTO users VALUES ($1, $2)")
defer stmt.Close()

for i := int64(1); i <= 1000; i++ {
    stmt.ExecContext(ctx, []any{i, "User"})
}
```

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

For production file-based workloads with automatic background maintenance, use the [CGO driver]({{ '/docs/drivers/go/' | relative_url }}).

## FetchAll (Bulk Read)

The `FetchAll()` method retrieves all remaining rows in a single WASM call and returns them as parsed Go values. The `database/sql` driver uses this automatically.

```go
rows, _ := db.Query(ctx, "SELECT id, name, age FROM users")
defer rows.Close()

allRows, _ := rows.FetchAll()
for _, row := range allRows {
    fmt.Printf("id=%v name=%v age=%v\n", row[0], row[1], row[2])
}
```

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
- **DB, Tx, Stmt, Rows**: Use from any goroutine (engine serializes access)
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

The WASM driver uses several optimizations to minimize overhead:

- **FetchAll**: All result rows are fetched in a single WASM call as packed binary, then parsed in Go. The `database/sql` driver uses this automatically.
- **CallWithStack**: Pre-allocated call stack eliminates per-call allocations.
- **Arena allocator**: SQL strings and parameters are written to a pre-allocated WASM memory region with zero malloc/free overhead.

Read-heavy workloads (SELECT, aggregation, JOINs) perform close to the native CGO driver. Write-heavy workloads have ~1.5-2x overhead due to WASM boundary crossing per operation.

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
| `db.ExecParams(ctx, query, args)` | Execute with parameters |
| `db.Query(ctx, query)` | Query without parameters |
| `db.QueryParams(ctx, query, args)` | Query with parameters |
| `db.Prepare(ctx, query)` | Create prepared statement |
| `db.Begin(ctx)` | Begin transaction (Read Committed) |
| `db.BeginTx(ctx, opts)` | Begin transaction with options |

### Rows

| Method | Description |
|--------|-------------|
| `rows.Next()` | Advance to next row |
| `rows.Scan(dest...)` | Read current row |
| `rows.Close()` | Close result set |
| `rows.Columns()` | Get column names |
| `rows.FetchAll()` | Fetch all rows as `[][]any` |

### Stmt

| Method | Description |
|--------|-------------|
| `stmt.ExecContext(ctx, args)` | Execute with parameters |
| `stmt.QueryContext(ctx, args)` | Query with parameters |
| `stmt.Close()` | Destroy the statement |

### Tx

| Method | Description |
|--------|-------------|
| `tx.Exec(ctx, query)` | Execute within transaction |
| `tx.ExecParams(ctx, query, args)` | Execute with parameters |
| `tx.Query(ctx, query)` | Query within transaction |
| `tx.QueryParams(ctx, query, args)` | Query with parameters |
| `tx.Commit()` | Commit |
| `tx.Rollback()` | Rollback |
