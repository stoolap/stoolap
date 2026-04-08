---
layout: doc
title: Go Driver
category: Drivers
order: 2
icon: go
---

# Go Driver

Native Go driver for Stoolap. Loads `libstoolap` at runtime via direct ABI calls. No C compiler, no CGO required. Provides two ways to use Stoolap from Go:

- **Direct API** for maximum performance and control
- **`database/sql`** driver for standard Go database access

For a pure Go driver with no shared library dependency, see [Go WASM Driver]({{ '/docs/drivers/go-wasm/' | relative_url }}).

## Requirements

- Go 1.24+
- `CGO_ENABLED=0` works (no C compiler needed)
- `CGO_ENABLED=1` also works (if linked with other CGO code)

## Installation

```bash
go get github.com/stoolap/stoolap-go
```

Prebuilt shared libraries for macOS (arm64), Linux (x64), and Windows (x64) are bundled
in the module. No extra downloads or environment variables needed, just `go get` and build.

The binary dynamically loads `libstoolap` at runtime via `dlopen`. For deployment, place the
shared library next to your executable or set the `STOOLAP_LIB` environment variable.

### Other Platforms

For platforms without a bundled library (e.g. Linux arm64, macOS x64), download from the
[releases page](https://github.com/stoolap/stoolap-go/releases) or build from source, then:

```bash
export STOOLAP_LIB=/path/to/libstoolap.so
go build ./...
```

## Quick Start

### Direct API

```go
package main

import (
    "context"
    "fmt"

    "github.com/stoolap/stoolap-go"
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
        var id, age int64
        var name string
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

    _ "github.com/stoolap/stoolap-go"
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
        var id, age int64
        var name string
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

Use positional parameters `$1`, `$2`, etc. with the direct API:

```go
db.ExecParams(ctx, "INSERT INTO users VALUES ($1, $2, $3)",
    []any{int64(1), "Alice", int64(30)})

rows, _ := db.QueryParams(ctx, "SELECT name FROM users WHERE id = $1",
    []any{int64(1)})
```

With `database/sql`, use standard `?` positional arguments:

```go
db.ExecContext(ctx, "INSERT INTO users VALUES (?, ?, ?)", 1, "Alice", 30)
rows, _ := db.QueryContext(ctx, "SELECT name FROM users WHERE id = ?", 1)
```

## Transactions

### Default Isolation (Read Committed)

```go
tx, err := db.Begin(ctx)
if err != nil {
    panic(err)
}

tx.ExecParams(ctx, "INSERT INTO users VALUES ($1, $2)",
    []any{int64(1), "Alice"})

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
rows, _ := tx.Query(ctx, "SELECT * FROM users")
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
stmt, err := db.Prepare(ctx, "INSERT INTO users VALUES ($1, $2)")
if err != nil {
    panic(err)
}
defer stmt.Close()

for i := int64(1); i <= 1000; i++ {
    stmt.ExecContext(ctx, []any{i, "User"})
}
```

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
rows, _ := db.QueryParams(ctx,
    "SELECT name, age, score, active, created_at FROM users WHERE id = $1",
    []any{int64(1)})
defer rows.Close()

if rows.Next() {
    rows.Scan(&name, &age, &score, &active, &ts)
    if name.Valid {
        fmt.Println("Name:", name.String)
    } else {
        fmt.Println("Name is NULL")
    }
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

rows, _ := db.Query(ctx, "SELECT data FROM docs WHERE id = 1")
defer rows.Close()

var data string
if rows.Next() {
    rows.Scan(&data)
    // data = `{"name":"Alice","age":30}`
}
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

db.ExecParams(ctx, "INSERT INTO vectors VALUES ($1, $2)",
    []any{int64(1), buf})

// Read back
var blob []byte
rows, _ := db.QueryParams(ctx, "SELECT embedding FROM vectors WHERE id = $1",
    []any{int64(1)})
defer rows.Close()
if rows.Next() {
    rows.Scan(&blob)
}

// Decode packed f32 bytes back to float32 slice
result := make([]float32, len(blob)/4)
for i := range result {
    result[i] = math.Float32frombits(binary.LittleEndian.Uint32(blob[i*4:]))
}
```

## Bulk Fetch

`FetchAll()` fetches all remaining rows in a single native call and returns them as `[][]any`, minimizing per-row FFI overhead:

```go
rows, _ := db.Query(ctx, "SELECT id, name, age FROM users")
defer rows.Close()

allRows, err := rows.FetchAll()
if err != nil {
    panic(err)
}
for _, row := range allRows {
    // row[0] = id, row[1] = name, row[2] = age
}
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

        clone.ExecParams(ctx, "INSERT INTO t VALUES ($1, $2)",
            []any{int64(workerID), fmt.Sprintf("worker-%d", workerID)})
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

## Thread Safety

- **Direct API**: A single `DB` handle must not be shared across goroutines. Use `Clone()` for per-goroutine handles.
- **`database/sql`**: Thread-safe by default. The connection pool creates cloned handles automatically.
- **Tx, Stmt, Rows**: Must remain on the goroutine that created them.

## Direct API Reference

### Package Functions

```go
func Version() (string, error)
func Open(dsn string) (*DB, error)
func OpenMemory() (*DB, error)
```

### DB

```go
func (db *DB) Close() error
func (db *DB) Clone() (*DB, error)
func (db *DB) Exec(ctx context.Context, query string) (sql.Result, error)
func (db *DB) ExecParams(ctx context.Context, query string, args []any) (sql.Result, error)
func (db *DB) Query(ctx context.Context, query string) (*Rows, error)
func (db *DB) QueryParams(ctx context.Context, query string, args []any) (*Rows, error)
func (db *DB) Prepare(ctx context.Context, query string) (*Stmt, error)
func (db *DB) Begin(ctx context.Context) (*Tx, error)
func (db *DB) BeginTx(ctx context.Context, opts *sql.TxOptions) (*Tx, error)
```

### Rows

```go
func (r *Rows) Next() bool
func (r *Rows) Scan(dest ...any) error
func (r *Rows) Columns() []string
func (r *Rows) Close() error
func (r *Rows) FetchAll() ([][]any, error)
```

### Stmt

```go
func (s *Stmt) ExecContext(ctx context.Context, args []any) (sql.Result, error)
func (s *Stmt) QueryContext(ctx context.Context, args []any) (*Rows, error)
func (s *Stmt) Close() error
```

### Tx

```go
func (tx *Tx) Exec(ctx context.Context, query string) (sql.Result, error)
func (tx *Tx) ExecParams(ctx context.Context, query string, args []any) (sql.Result, error)
func (tx *Tx) Query(ctx context.Context, query string) (*Rows, error)
func (tx *Tx) QueryParams(ctx context.Context, query string, args []any) (*Rows, error)
func (tx *Tx) Commit() error
func (tx *Tx) Rollback() error
```

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

## Architecture

The driver loads `libstoolap` at runtime via `dlopen` and dispatches FFI calls through hand-written assembly trampolines entered via `runtime.asmcgocall`. This bypasses the standard CGO overhead (~40ns per call) and achieves near-native performance (~3ns per call).

On Linux, a minimal fake-cgo runtime preserves glibc thread-local storage so the shared library's thread-local state works correctly without requiring `CGO_ENABLED=1`. On macOS and Windows, the native OS loader handles this directly.

## Building from Source

```bash
git clone https://github.com/stoolap/stoolap-go.git
cd stoolap-go
go test -v ./...
```
