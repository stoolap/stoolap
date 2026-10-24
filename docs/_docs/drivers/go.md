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

- Go 1.25+
- Stoolap 0.5.0 or later (`libstoolap` built with the `ffi` feature)
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
    if err := rows.Err(); err != nil {
        panic(err)
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

### Positional

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

### Named

Named parameters use `:name` placeholders and take `sql.NamedArg` values. A name may appear more than once in the query; every occurrence reads the same value.

```go
db.ExecNamed(ctx, "INSERT INTO users VALUES (:id, :name, :age)", []sql.NamedArg{
    sql.Named("id", int64(1)),
    sql.Named("name", "Alice"),
    sql.Named("age", int64(30)),
})

rows, _ := db.QueryNamed(ctx, "SELECT name FROM users WHERE age > :min",
    []sql.NamedArg{sql.Named("min", int64(18))})
```

With `database/sql`, pass `sql.Named` values. Named and positional arguments cannot be mixed in one call. Prepared statements accept both forms.

```go
db.ExecContext(ctx, "INSERT INTO users VALUES (:id, :name, :age)",
    sql.Named("id", 1), sql.Named("name", "Alice"), sql.Named("age", 30))
```

Every parameter the statement reads must have a value; a missing one is an error before any row is read or written. A name in the argument list that the statement does not use is ignored.

## Reading Results

### Rows

`Rows` pulls results from the engine in packed chunks, starting at 32 rows and growing to 256, so a scan crosses the native boundary once per chunk rather than once per row and column. `Next` returns false at the end of the result or on an error; check `Err` afterwards.

Besides `Scan`, `Rows` has typed accessors that read the current row without allocating:

```go
rows, _ := db.Query(ctx, "SELECT id, name, score, active, created_at FROM users")
defer rows.Close()

for rows.Next() {
    if rows.IsNull(1) {
        continue
    }
    id := rows.Int64(0)
    score := rows.Float64(2)
    active := rows.Bool(3)
    created := rows.Timestamp(4)
    // ...
}
if err := rows.Err(); err != nil {
    panic(err)
}
```

`Int64`, `Float64` and `Bool` convert other column types the way the engine does, so numeric text reads as a number and an integer reads as a boolean. `Timestamp` reads a TIMESTAMP, an INTEGER as nanoseconds since the Unix epoch, or TEXT in a common timestamp form.

### QueryAll and ResultSet

`QueryAll` and `QueryNamedAll` on `DB`, `Tx` and `Stmt` run a query and decode every row with a single native call. No rows handle is created, so a point lookup costs one crossing. The `database/sql` driver uses this path for every query.

```go
rs, err := db.QueryAll(ctx, "SELECT id, name FROM users WHERE age > $1", []any{int64(18)})
if err != nil {
    panic(err)
}
fmt.Println(rs.Columns) // [id name]
for _, row := range rs.Rows {
    fmt.Println(row[0].(int64), row[1].(string))
}
```

Use `Query` and `Rows` instead when a result is too large to hold in memory.

### FetchAll

`FetchAll()` on `Rows` returns every remaining row as `[][]any` with one native call. Rows already consumed by `Next` are not included.

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

With `database/sql`, a statement prepared on a `*sql.Tx` runs inside that transaction, and `tx.Stmt(stmt)` rebinds a connection level statement to the transaction.

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

A prepared statement accepts positional or named parameters, depending on how it was written:

```go
stmt, _ := db.Prepare(ctx, "SELECT name FROM users WHERE id = :id")
rs, _ := stmt.QueryNamedAll(ctx, []sql.NamedArg{sql.Named("id", int64(1))})
```

### Batch Execution

`ExecBatch` runs the statement once per row inside a single transaction with one native call. Every row must have the same number of parameters. If any row fails, the whole batch is rolled back.

```go
stmt, _ := db.Prepare(ctx, "INSERT INTO users VALUES ($1, $2, $3)")
res, err := stmt.ExecBatch(ctx, [][]any{
    {int64(1), "Alice", int64(30)},
    {int64(2), "Bob", int64(25)},
})
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

A `nil` parameter binds as NULL. A nil `[]byte` also binds as NULL, while an empty `[]byte{}` or an empty string binds as an empty value.

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

Scanning a vector into a `string` gives its text form, for example `[1.0, 2.0, 3.0]`.

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

| SQL Type | Parameter Types | Scan Types |
|----------|-----------------|------------|
| INTEGER | `int64`, `int`, `int32` | `int64`, `sql.NullInt64` |
| FLOAT | `float64`, `float32` | `float64`, `sql.NullFloat64` |
| TEXT | `string` | `string`, `sql.NullString` |
| BOOLEAN | `bool` | `bool`, `sql.NullBool` |
| TIMESTAMP | `time.Time` | `time.Time`, `sql.NullTime` |
| JSON | `string` | `string`, `sql.NullString` |
| VECTOR/BLOB | `[]byte` | `[]byte` (nil for NULL) |

Every column can also be scanned into `string` or `any`. Scanning into a `string` gives the engine's text form, the same as `CAST(x AS TEXT)`, except that a float whose shortest representation ends in an exact decimal tie may round its last digit differently. Scanning TEXT into `time.Time` accepts RFC 3339 with Z or an offset, ISO and SQL style with an optional fraction, a bare date, and `YYYY/MM/DD HH:MM:SS`; text in another form is a Scan error.

## Error Handling

All errors returned by the stoolap engine are typed as `*stoolap.Error`, which carries a categorized error code:

```go
var stErr *stoolap.Error
if errors.As(err, &stErr) {
    switch stErr.Code() {
    case stoolap.ErrUniqueConstraint:
        // handle duplicate
    case stoolap.ErrPrimaryKeyConstraint:
        // handle primary key violation
    case stoolap.ErrTableNotFound:
        // handle missing table
    }
}
```

Use `IsConstraintViolation()` to check for any constraint error (unique, primary key, not null, check, foreign key):

```go
if errors.As(err, &stErr) && stErr.IsConstraintViolation() {
    // handle any constraint violation
}
```

| ErrorCode | Description |
|-----------|-------------|
| `ErrGeneral` | Uncategorized error |
| `ErrUniqueConstraint` | Unique index violation |
| `ErrPrimaryKeyConstraint` | Primary key violation |
| `ErrNotNullConstraint` | NOT NULL violation |
| `ErrCheckConstraint` | CHECK constraint violation |
| `ErrForeignKeyViolation` | Foreign key violation |
| `ErrTableNotFound` | Table does not exist |
| `ErrTableExists` | Table already exists |

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
func (db *DB) ExecNamed(ctx context.Context, query string, args []sql.NamedArg) (sql.Result, error)
func (db *DB) Query(ctx context.Context, query string) (*Rows, error)
func (db *DB) QueryParams(ctx context.Context, query string, args []any) (*Rows, error)
func (db *DB) QueryNamed(ctx context.Context, query string, args []sql.NamedArg) (*Rows, error)
func (db *DB) QueryAll(ctx context.Context, query string, args []any) (*ResultSet, error)
func (db *DB) QueryNamedAll(ctx context.Context, query string, args []sql.NamedArg) (*ResultSet, error)
func (db *DB) Prepare(ctx context.Context, query string) (*Stmt, error)
func (db *DB) Begin(ctx context.Context) (*Tx, error)
func (db *DB) BeginTx(ctx context.Context, opts *sql.TxOptions) (*Tx, error)
```

### Rows

```go
func (r *Rows) Next() bool
func (r *Rows) Scan(dest ...any) error
func (r *Rows) Err() error
func (r *Rows) Columns() []string
func (r *Rows) Close() error
func (r *Rows) FetchAll() ([][]any, error)
func (r *Rows) IsNull(i int) bool
func (r *Rows) Int64(i int) int64
func (r *Rows) Float64(i int) float64
func (r *Rows) Bool(i int) bool
func (r *Rows) Timestamp(i int) time.Time
```

### ResultSet

```go
type ResultSet struct {
    Columns []string
    Rows    [][]any
}
```

### Stmt

```go
func (s *Stmt) ExecContext(ctx context.Context, args []any) (sql.Result, error)
func (s *Stmt) ExecNamed(ctx context.Context, args []sql.NamedArg) (sql.Result, error)
func (s *Stmt) ExecBatch(ctx context.Context, rows [][]any) (sql.Result, error)
func (s *Stmt) QueryContext(ctx context.Context, args []any) (*Rows, error)
func (s *Stmt) QueryNamed(ctx context.Context, args []sql.NamedArg) (*Rows, error)
func (s *Stmt) QueryAll(ctx context.Context, args []any) (*ResultSet, error)
func (s *Stmt) QueryNamedAll(ctx context.Context, args []sql.NamedArg) (*ResultSet, error)
func (s *Stmt) Close() error
```

### Tx

```go
func (tx *Tx) Exec(ctx context.Context, query string) (sql.Result, error)
func (tx *Tx) ExecParams(ctx context.Context, query string, args []any) (sql.Result, error)
func (tx *Tx) ExecNamed(ctx context.Context, query string, args []sql.NamedArg) (sql.Result, error)
func (tx *Tx) Query(ctx context.Context, query string) (*Rows, error)
func (tx *Tx) QueryParams(ctx context.Context, query string, args []any) (*Rows, error)
func (tx *Tx) QueryNamed(ctx context.Context, query string, args []sql.NamedArg) (*Rows, error)
func (tx *Tx) QueryAll(ctx context.Context, query string, args []any) (*ResultSet, error)
func (tx *Tx) QueryNamedAll(ctx context.Context, query string, args []sql.NamedArg) (*ResultSet, error)
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

Every query through `database/sql` is answered with one native call that returns the whole result, so a `*sql.Rows` holds its rows in memory. Use the direct API's `Rows` for results too large to hold at once.

## Architecture

The driver loads `libstoolap` at runtime via `dlopen` and dispatches FFI calls through hand-written assembly trampolines entered via `runtime.cgocall`. No cgo wrapper code is generated or compiled, and the Go runtime still sees each call as a system call, so goroutine scheduling and stack growth behave as they do with cgo.

With `CGO_ENABLED=0` on Linux and macOS, a minimal fake cgo runtime initializes threads so the shared library's thread-local storage works correctly. On Windows the OS loader handles this directly.

## Building from Source

```bash
git clone https://github.com/stoolap/stoolap-go.git
cd stoolap-go
go test -v ./...
```
