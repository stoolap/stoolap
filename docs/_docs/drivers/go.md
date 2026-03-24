---
layout: doc
title: Go Driver
category: Drivers
order: 4
icon: go
---

# Go Driver

Go driver for Stoolap built on the Rust engine via C FFI (cgo). Provides two ways to use Stoolap from Go:

- **Direct API** for maximum performance and control
- **`database/sql`** driver for standard Go database access

For a pure Go driver with no CGO dependency, see [Go WASM Driver]({{ '/docs/drivers/go-wasm/' | relative_url }}).

## Requirements

- Go 1.24+
- CGO enabled (`CGO_ENABLED=1`, the default)

## Installation

```bash
go get github.com/stoolap/stoolap-go
```

Prebuilt shared libraries for macOS (arm64), Linux (x64), and Windows (x64) are bundled
in the module. No extra downloads or environment variables needed, just `go get` and build.

The compiled Go binary dynamically links against `libstoolap`. For deployment, place the
shared library next to your executable or in a system library path.

### Other Platforms

For platforms without a bundled library (e.g. Linux arm64, macOS x64), download from the
[releases page](https://github.com/stoolap/stoolap-go/releases) or build from source,
then build with the `stoolap_use_lib` tag:

```bash
export LIBRARY_PATH=/path/to/stoolap/target/release
go build -tags stoolap_use_lib ./...
```

## Quick Start

### Direct API

```go
package main

import (
    "context"
    "fmt"

    stoolap "github.com/stoolap/stoolap-go"
)

func main() {
    db, err := stoolap.Open("memory://")
    if err != nil {
        panic(err)
    }
    defer db.Close()

    ctx := context.Background()

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

```go
package main

import (
    "context"
    "database/sql"
    "fmt"

    _ "github.com/stoolap/stoolap-go/pkg/driver"
)

func main() {
    db, err := sql.Open("stoolap", "memory://")
    if err != nil {
        panic(err)
    }
    defer db.Close()

    ctx := context.Background()

    db.ExecContext(ctx, "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    db.ExecContext(ctx, "INSERT INTO users VALUES (1, 'Alice', 30), (2, 'Bob', 25)")

    rows, _ := db.QueryContext(ctx, "SELECT id, name, age FROM users ORDER BY id")
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

## Connection Strings

| DSN | Description |
|-----|-------------|
| `memory://` | In-memory database (unique, isolated instance) |
| `memory://mydb` | Named in-memory database (same name shares the engine) |
| `file:///path/to/db` | File-based persistent database |
| `file:///path/to/db?sync_mode=full` | File-based with configuration options |

See [Connection String Reference]({{ '/docs/getting-started/connection-strings/' | relative_url }}) for all configuration options.

## Parameters

Use positional parameters `$1`, `$2`, etc. with `driver.NamedValue`:

```go
import "database/sql/driver"

ctx := context.Background()

db.ExecContext(ctx, "INSERT INTO users VALUES ($1, $2, $3)",
    driver.NamedValue{Ordinal: 1, Value: int64(1)},
    driver.NamedValue{Ordinal: 2, Value: "Alice"},
    driver.NamedValue{Ordinal: 3, Value: int64(30)},
)

row := db.QueryRow(ctx, "SELECT name FROM users WHERE id = $1",
    driver.NamedValue{Ordinal: 1, Value: int64(1)},
)
var name string
row.Scan(&name)
```

With `database/sql`, use standard positional arguments:

```go
db.ExecContext(ctx, "INSERT INTO users VALUES ($1, $2, $3)", 1, "Alice", 30)
rows, _ := db.QueryContext(ctx, "SELECT name FROM users WHERE id = $1", 1)
```

## Transactions

### Default Isolation (Read Committed)

```go
tx, err := db.Begin()
if err != nil {
    panic(err)
}

tx.ExecContext(ctx, "INSERT INTO users VALUES ($1, $2)",
    driver.NamedValue{Ordinal: 1, Value: int64(1)},
    driver.NamedValue{Ordinal: 2, Value: "Alice"},
)

if err := tx.Commit(); err != nil {
    panic(err)
}
```

### Snapshot Isolation

```go
tx, err := db.BeginTx(ctx, &sql.TxOptions{
    Isolation: sql.LevelSnapshot,
})
if err != nil {
    panic(err)
}
defer tx.Rollback()

// All reads within this transaction see the same snapshot
rows, _ := tx.QueryContext(ctx, "SELECT * FROM users")
// ...
tx.Commit()
```

| Level | Description |
|-------|-------------|
| Read Committed (default) | Each statement sees data committed before it started |
| Snapshot | The transaction sees a consistent snapshot from when it began |

## Prepared Statements

Parse once, execute many times:

```go
stmt, err := db.Prepare("INSERT INTO users VALUES ($1, $2)")
if err != nil {
    panic(err)
}
defer stmt.Close()

for i := int64(1); i <= 1000; i++ {
    stmt.ExecContext(ctx,
        driver.NamedValue{Ordinal: 1, Value: i},
        driver.NamedValue{Ordinal: 2, Value: "User"},
    )
}
```

### Prepared Statements in Transactions

For transactional atomicity with parse-once performance, prepare statements via `Tx.Prepare()`. This uses `stoolap_tx_stmt_exec`/`stoolap_tx_stmt_query` internally, ensuring all operations participate in the transaction's commit/rollback.

```go
stmt, err := db.Prepare("INSERT INTO orders VALUES ($1, $2, $3)")
if err != nil {
    panic(err)
}
defer stmt.Close()

tx, err := db.Begin()
if err != nil {
    panic(err)
}

txStmt, err := tx.Prepare("INSERT INTO orders VALUES ($1, $2, $3)")
if err != nil {
    tx.Rollback()
    panic(err)
}
defer txStmt.Close()

for i := int64(0); i < 1000; i++ {
    txStmt.ExecContext(ctx,
        driver.NamedValue{Ordinal: 1, Value: i},
        driver.NamedValue{Ordinal: 2, Value: int64(1)},
        driver.NamedValue{Ordinal: 3, Value: 99.99},
    )
}

tx.Commit() // all 1000 rows committed atomically
```

**Important**: Do not use `stmt.ExecContext()` (DB-level prepared statement) inside a transaction block. It creates its own standalone auto-committing transaction per call, so rollback will not undo those operations. Always use `tx.Prepare()` for transaction-bound statements.

## NULL Handling

Use `sql.Null*` types for nullable columns:

```go
var (
    name   sql.NullString
    age    sql.NullInt64
    score  sql.NullFloat64
    active sql.NullBool
    ts     sql.NullTime
)
row := db.QueryRow(ctx, "SELECT name, age, score, active, created_at FROM users WHERE id = $1",
    driver.NamedValue{Ordinal: 1, Value: int64(1)},
)
row.Scan(&name, &age, &score, &active, &ts)

if name.Valid {
    fmt.Println("Name:", name.String)
} else {
    fmt.Println("Name is NULL")
}
```

## Scanning into `any`

```go
rows, _ := db.Query(ctx, "SELECT id, name, age FROM users")
defer rows.Close()

for rows.Next() {
    var id, name, age any
    rows.Scan(&id, &name, &age)
    fmt.Printf("id=%v name=%v age=%v\n", id, name, age)
}
```

## JSON

JSON values are stored and retrieved as strings:

```go
db.Exec(ctx, "CREATE TABLE docs (id INTEGER PRIMARY KEY, data JSON)")
db.Exec(ctx, `INSERT INTO docs VALUES (1, '{"name":"Alice","age":30}')`)

var data string
db.QueryRow(ctx, "SELECT data FROM docs WHERE id = 1").Scan(&data)
// data = `{"name":"Alice","age":30}`
```

## Vector Search

Vectors are stored as packed little-endian f32 bytes:

```go
import (
    "encoding/binary"
    "math"
)

db.Exec(ctx, "CREATE TABLE vectors (id INTEGER PRIMARY KEY, embedding VECTOR(3))")

// Encode a float32 vector to bytes
vec := []float32{1.0, 2.0, 3.0}
buf := make([]byte, len(vec)*4)
for i, f := range vec {
    binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
}

db.ExecContext(ctx, "INSERT INTO vectors VALUES ($1, $2)",
    driver.NamedValue{Ordinal: 1, Value: int64(1)},
    driver.NamedValue{Ordinal: 2, Value: buf},
)

// Read back
var blob []byte
db.QueryRow(ctx, "SELECT embedding FROM vectors WHERE id = 1").Scan(&blob)

// Decode packed f32 bytes back to float32 slice
result := make([]float32, len(blob)/4)
for i := range result {
    result[i] = math.Float32frombits(binary.LittleEndian.Uint32(blob[i*4:]))
}
```

## Bulk Fetch

`FetchAll()` fetches all remaining rows into a single packed binary buffer, avoiding per-row FFI overhead:

```go
rows, _ := db.Query(ctx, "SELECT id, name, age FROM users")
defer rows.Close()

buf, err := rows.FetchAll()
if err != nil {
    panic(err)
}
// buf contains all rows in packed binary format
// See the C API docs for the binary format specification
```

## Cloning for Concurrency

A single `DB` handle must not be used from multiple goroutines simultaneously. Use `Clone()` to create per-goroutine handles that share the underlying engine:

```go
db, _ := stoolap.Open("memory://mydb")
defer db.Close()

db.Exec(ctx, "CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")

var wg sync.WaitGroup
for i := 0; i < 4; i++ {
    wg.Add(1)
    go func(workerID int) {
        defer wg.Done()

        clone, _ := db.Clone()
        defer clone.Close()

        clone.Exec(ctx, fmt.Sprintf("INSERT INTO t VALUES (%d, 'worker-%d')", workerID, workerID))
    }(i)
}
wg.Wait()
```

The `database/sql` driver handles this automatically. Each connection in the pool gets its own cloned handle.

## Type Mapping

| SQL Type | Go Type | Nullable Go Type |
|----------|---------|------------------|
| INTEGER | `int64`, `int`, `int32` | `sql.NullInt64` |
| FLOAT | `float64`, `float32` | `sql.NullFloat64` |
| TEXT | `string` | `sql.NullString` |
| BOOLEAN | `bool` | `sql.NullBool` |
| TIMESTAMP | `time.Time` | `sql.NullTime` |
| JSON | `string` | `sql.NullString` |
| VECTOR/BLOB | `[]byte` | `[]byte` (nil for NULL) |

Scan supports type coercion: INTEGER columns can scan into `*string`, FLOAT into `*int64`, etc.

## Thread Safety

- **Direct API**: A single `DB` handle must not be shared across goroutines. Use `Clone()` for per-goroutine handles.
- **`database/sql`**: Thread-safe by default. The connection pool creates cloned handles automatically.
- **Tx, Stmt, Rows**: Must remain on the goroutine that created them.

## Direct API Reference

### Package Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `Version()` | `string` | Stoolap library version |
| `Open(dsn)` | `*DB, error` | Open a database connection |

### DB

| Method | Returns | Description |
|--------|---------|-------------|
| `Close()` | `error` | Close the connection |
| `Clone()` | `*DB, error` | Clone handle for multi-goroutine use |
| `Exec(ctx, query)` | `sql.Result, error` | Execute without parameters |
| `ExecContext(ctx, query, args...)` | `sql.Result, error` | Execute with parameters |
| `Query(ctx, query)` | `Rows, error` | Query without parameters |
| `QueryContext(ctx, query, args...)` | `Rows, error` | Query with parameters |
| `QueryRow(ctx, query, args...)` | `Row` | Query expecting at most one row |
| `Begin()` | `Tx, error` | Begin transaction (Read Committed) |
| `BeginTx(ctx, opts)` | `Tx, error` | Begin transaction with options |
| `Prepare(query)` | `Stmt, error` | Create a prepared statement |
| `PrepareContext(ctx, query)` | `Stmt, error` | Create a prepared statement with context |

### Rows

| Method | Returns | Description |
|--------|---------|-------------|
| `Next()` | `bool` | Advance to next row |
| `Scan(dest...)` | `error` | Read current row columns |
| `Close()` | `error` | Close result set |
| `Columns()` | `[]string` | Get column names |
| `FetchAll()` | `[]byte, error` | Fetch all remaining rows as packed binary |

### Row

| Method | Returns | Description |
|--------|---------|-------------|
| `Scan(dest...)` | `error` | Read the row columns (`sql.ErrNoRows` if empty) |

### Tx

| Method | Returns | Description |
|--------|---------|-------------|
| `Commit()` | `error` | Commit the transaction |
| `Rollback()` | `error` | Rollback the transaction |
| `ExecContext(ctx, query, args...)` | `sql.Result, error` | Execute within the transaction |
| `QueryContext(ctx, query, args...)` | `Rows, error` | Query within the transaction |
| `Prepare(query)` | `Stmt, error` | Prepare statement bound to the transaction |
| `ID()` | `int64` | Get the transaction ID |

### Stmt

| Method | Returns | Description |
|--------|---------|-------------|
| `ExecContext(ctx, args...)` | `sql.Result, error` | Execute the prepared statement |
| `QueryContext(ctx, args...)` | `Rows, error` | Query with the prepared statement |
| `SQL()` | `string` | Get the SQL text |
| `Close()` | `error` | Destroy the prepared statement |

## database/sql Driver

The driver is registered as `"stoolap"` and implements the following `database/sql/driver` interfaces:

| Interface | Description |
|-----------|-------------|
| `driver.Driver` | Basic driver |
| `driver.DriverContext` | Connector-based driver |
| `driver.Connector` | Connection factory with pooling |
| `driver.Conn` | Connection |
| `driver.ConnBeginTx` | Transaction with isolation levels |
| `driver.ExecerContext` | Direct exec (bypasses prepare) |
| `driver.QueryerContext` | Direct query (bypasses prepare) |
| `driver.ConnPrepareContext` | Prepared statements |
| `driver.Pinger` | Connection health check |
| `driver.SessionResetter` | Session reset on pool return |
| `driver.Validator` | Connection validation |
| `driver.Tx` | Transaction commit/rollback |
| `driver.Stmt` | Prepared statement |
| `driver.StmtExecContext` | Prepared exec with context |
| `driver.StmtQueryContext` | Prepared query with context |
| `driver.Rows` | Result set iteration |

## Building from Source

```bash
git clone https://github.com/stoolap/stoolap-go.git
cd stoolap-go
go test -v ./...
```
