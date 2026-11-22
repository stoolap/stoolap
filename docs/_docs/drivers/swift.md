---
layout: doc
title: Swift Driver
category: Drivers
order: 8
icon: swift
---

# Swift Driver

High-performance Swift driver for Stoolap. Calls the Rust engine directly through the official C ABI with zero intermediate wrappers. Provides sync, async, and streaming cursor APIs. Targets macOS 12+ and iOS 15+.

## Installation

### Swift Package Manager

```swift
dependencies: [
    .package(url: "https://github.com/stoolap/stoolap-swift.git", from: "0.5.1")
]
```

The package links `StoolapC.xcframework`, the engine's C API built as a static library for macOS and iOS, which Swift Package Manager downloads from the [release](https://github.com/stoolap/stoolap-swift/releases). No Rust toolchain is needed.

Driver 0.5.1 is built on Stoolap engine 0.5.1. Requires macOS 12+ or iOS 15+. To build the engine from source instead, see [Building from Source](#building-from-source).

## Quick Start

```swift
import Stoolap

let db = try Database.open(":memory:")

try db.exec("""
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT
    )
""")

try db.execute(
    "INSERT INTO users (id, name, email) VALUES ($1, $2, $3)",
    [.integer(1), .text("Alice"), .text("alice@example.com")]
)

let users = try db.query("SELECT * FROM users ORDER BY id")
for user in users {
    print(user["name"]?.stringValue ?? "")
}

let one = try db.queryOne("SELECT * FROM users WHERE id = $1", [.integer(1)])
print(one?["email"]?.stringValue ?? "")
```

## Opening a Database

```swift
// In-memory
let db = try Database.open(":memory:")
let db = try Database.open("memory://")

// File-based (data persists across restarts)
let db = try Database.open("./mydata")
let db = try Database.open("file:///absolute/path/to/db")

// With configuration
let db = try Database.open("./mydata?sync=full&compression=on")
```

`:memory:`, an empty string and `memory://` all name the same in-memory database: while one `Database` on it is open, a second open of the same DSN shares its engine and data. Use a name such as `memory://cache` for a separate in-memory database. The data of an in-memory database is gone once the last `Database` on it is released.

See [Connection Strings]({{ '/docs/getting-started/connection-strings/' | relative_url }}) for every configuration option.

## Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `execute(sql, params?)` | `Int64` | Execute one statement, return rows affected (discardable) |
| `exec(sql)` | `Void` | Execute a script of semicolon-separated statements, without parameters |
| `query(sql, params?)` | `[Row]` | Query rows with named column access |
| `queryOne(sql, params?)` | `Row?` | Query single row |
| `queryRaw(sql, params?)` | `ColumnarResult` | Query in flat columnar format |
| `queryCursor(sql, params?)` | `RowCursor` | Streaming cursor over result set |
| `prepare(sql)` | `PreparedStatement` | Create a prepared statement |
| `begin()` | `Transaction` | Begin a transaction |
| `withTransaction(_:)` | Generic | Auto-commit/rollback closure |
| `Database.version` | `String` | Engine version, e.g. `"0.5.1"` |

`exec` parses the whole script before it runs anything: a syntax error anywhere runs none of it. After a runtime error, the statements before the failing one stay applied.

## Row Access

`Row` is a zero-allocation struct backed by an `ArraySlice<Value>` into a shared result-wide cell array.

```swift
let rows = try db.query("SELECT id, name, email FROM users ORDER BY id")
for row in rows {
    // By name (linear scan, fast for typical column counts)
    let name = row["name"]?.stringValue

    // By index (zero-based)
    let id = row[0].int64Value

    // Iterate all columns
    row.forEach { column, value in
        print("\(column): \(value)")
    }
}
```

## Columnar Results

`queryRaw()` returns a `ColumnarResult` with all cells in a single flat array. Zero per-row heap allocations.

```swift
let raw = try db.queryRaw("SELECT id, name FROM users ORDER BY id")

// Row access (ArraySlice view)
let firstRow = raw[row: 0]

// Cell access
let name = raw[row: 0, column: 1]

// Column-wise access (strided RandomAccessCollection, zero allocation)
let ids = raw.column(at: 0)         // ColumnarColumn
let names = raw.column(named: "name")  // ColumnarColumn?

for id in ids {
    print(id.int64Value ?? 0)
}
```

## Streaming Cursor

For large result sets where you want bounded memory instead of bulk materialization. The cursor reads rows from the engine in packed chunks of 32 rows, doubling up to 256, so it holds at most one chunk at any result size:

```swift
let cursor = try db.queryCursor("SELECT * FROM users ORDER BY id")

while try cursor.next() {
    // Read individual cells without materializing the whole row
    let id = try cursor.value(at: 0)
    let name = try cursor.value(named: "name")

    // Or materialize the current row
    let row = try cursor.row()
    print(row["email"]?.stringValue ?? "")
}

// Or drain with a closure
let cursor = try db.queryCursor("SELECT * FROM large_table")
try cursor.forEachRemaining { row in
    process(row)
}
```

`cursor.columns` holds the column names. The cursor closes itself when it runs out of rows; call `close()` to stop early. `PreparedStatement` and `Transaction` have `queryCursor` too. An error partway through a chunk is thrown before that chunk's earlier rows are returned.

## Prepared Statements

Prepared statements parse SQL once and reuse the cached execution plan. Column names are reused across calls, eliminating per-call string allocations; each result's column header is checked against them, so a schema change after the prepare (such as `ALTER TABLE ... ADD COLUMN` under a `SELECT *`) is picked up.

```swift
let insert = try db.prepare("INSERT INTO users VALUES ($1, $2, $3)")
try insert.execute([.integer(1), .text("Alice"), .text("alice@example.com")])
try insert.execute([.integer(2), .text("Bob"), .text("bob@example.com")])

let lookup = try db.prepare("SELECT * FROM users WHERE id = $1")
let user = try lookup.queryOne([.integer(1)])
```

### Prepared Statement Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `execute(params?)` | `Int64` | Execute DML statement |
| `query(params?)` | `[Row]` | Query rows |
| `queryOne(params?)` | `Row?` | Query single row |
| `queryRaw(params?)` | `ColumnarResult` | Query in columnar format |
| `queryCursor(params?)` | `RowCursor` | Streaming cursor |
| `executeBatch(paramsList)` | `Int64` | Batch execute in single FFI call |
| `sql` | `String` | The statement's SQL text |

## Batch Execution

Execute multiple parameter sets in a single FFI call. The engine runs them in one transaction: if any row fails, none of the rows are applied. Every parameter set must have the same number of values.

```swift
let insert = try db.prepare("INSERT INTO users VALUES ($1, $2, $3)")
let changes = try insert.executeBatch([
    [.integer(1), .text("Alice"), .text("alice@example.com")],
    [.integer(2), .text("Bob"), .text("bob@example.com")],
    [.integer(3), .text("Charlie"), .text("charlie@example.com")],
])
// changes == 3
```

## Transactions

### Auto-commit/rollback

```swift
try db.withTransaction { tx in
    try tx.execute("INSERT INTO users VALUES ($1, $2, $3)",
                   [.integer(1), .text("Alice"), .text("alice@example.com")])
    try tx.execute("INSERT INTO users VALUES ($1, $2, $3)",
                   [.integer(2), .text("Bob"), .text("bob@example.com")])
}
```

### Manual control

```swift
let tx = try db.begin()
do {
    try tx.execute("INSERT INTO users VALUES ($1, $2, $3)",
                   [.integer(1), .text("Alice"), .text("alice@example.com")])
    try tx.commit()
} catch {
    try? tx.rollback()
    throw error
}
```

A `Transaction` released without `commit()` or `rollback()` is rolled back. After either call, every other call on it throws.

### Transaction Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `execute(sql, params?)` | `Int64` | Execute DML statement |
| `query(sql, params?)` | `[Row]` | Query rows |
| `queryOne(sql, params?)` | `Row?` | Query single row |
| `queryRaw(sql, params?)` | `ColumnarResult` | Query in columnar format |
| `queryCursor(sql, params?)` | `RowCursor` | Streaming cursor |
| `commit()` | `Void` | Commit the transaction |
| `rollback()` | `Void` | Rollback the transaction |

## Async API

`AsyncDatabase` wraps `Database` with Swift concurrency. Each call runs the synchronous engine call in a detached task, off the calling actor. The task still occupies a thread of the cooperative pool while the engine works.

```swift
let db = try await AsyncDatabase.open(":memory:")

try await db.exec("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
try await db.execute("INSERT INTO users VALUES ($1, $2)",
                     [.integer(1), .text("Alice")])

let users = try await db.query("SELECT * FROM users")
let one = try await db.queryOne("SELECT * FROM users WHERE id = $1", [.integer(1)])
let raw = try await db.queryRaw("SELECT * FROM users")
let stmt = try await db.prepare("SELECT * FROM users WHERE id = $1")
```

`AsyncDatabase` has `execute`, `exec`, `query`, `queryOne`, `queryRaw` and `prepare`. For transactions and cursors, use the wrapped `Database`: `let inner = await db.db`.

## Parameters

Parameters are positional (`$1`, `$2`, ...) and passed as `[Value]`. Every parameter the statement uses needs a value: a missing one is an error, not `NULL`.

```swift
// Throws: $2 has no value
_ = try db.query("SELECT * FROM users WHERE id = $1 AND name = $2", [.integer(1)])
```

## Error Handling

All methods throw `StoolapError` on failure:

```swift
do {
    // id 1 already exists
    try db.execute("INSERT INTO users VALUES ($1, $2, $3)",
                   [.integer(1), .text("Alice"), .null])
} catch let error as StoolapError {
    print("Database error: \(error.message)")
}
```

## Type Mapping

| Swift Value | Stoolap SQL | Notes |
|-------------|-------------|-------|
| `.integer(Int64)` | `INTEGER` | 64-bit signed |
| `.float(Double)` | `FLOAT` | 64-bit double |
| `.text(String)` | `TEXT` | UTF-8 encoded |
| `.boolean(Bool)` | `BOOLEAN` | |
| `.null` | `NULL` | Any type |
| `.timestamp(Date)` | `TIMESTAMP` | Stored in nanoseconds; `Date` holds about 0.1 µs at current dates, years 1677-2262 |
| `.json(String)` | `JSON` | Pre-serialized JSON string |
| `.blob(Data)` | `VECTOR` | Packed little-endian f32 bytes (length a multiple of 4); reads back as `.vector` |
| `.vector([Float])` | `VECTOR` | Packed f32 for similarity search |

Values read back as the same case they were written with, except `.blob`, which reads back as `.vector`. A `.json` parameter that is not valid JSON is an error. A `.timestamp` outside 1677-2262, or a `.blob` whose length is not a multiple of 4, throws before the statement runs.

## Vector Search

Bind a `[Float]` as `.vector` to store it in a `VECTOR(N)` column or to pass it to a distance function:

```swift
try db.exec("""
    CREATE TABLE documents (
        id INTEGER PRIMARY KEY,
        title TEXT,
        embedding VECTOR(3)
    );
    CREATE INDEX idx_emb ON documents(embedding) USING HNSW WITH (metric = 'cosine');
""")

try db.execute("INSERT INTO documents VALUES ($1, $2, $3)",
               [.integer(1), .text("Hello world"), .vector([0.1, 0.2, 0.3])])
try db.execute("INSERT INTO documents VALUES ($1, $2, $3)",
               [.integer(2), .text("Goodbye world"), .vector([0.9, 0.1, 0.0])])

// 5 nearest neighbours by cosine distance
let nearest = try db.query("""
    SELECT id, title, VEC_DISTANCE_COSINE(embedding, $1) AS dist
    FROM documents
    ORDER BY dist
    LIMIT 5
""", [.vector([0.1, 0.2, 0.3])])

// Vectors read back as .vector([Float])
let embedding = try db.queryOne("SELECT embedding FROM documents WHERE id = 1")?["embedding"]
```

See [Vector Search]({{ '/docs/data-types/vector-search/' | relative_url }}) for the distance functions and HNSW index options.

## Persistence

File-based databases persist data using WAL and immutable cold volumes.

```swift
do {
    let db = try Database.open("./mydata?sync=full")
    try db.exec("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)")
    try db.execute("INSERT INTO notes VALUES ($1, $2)", [.integer(1), .text("hello")])
}   // the last Database on ./mydata is released here, which closes it

// Reopen: data is still there
let db = try Database.open("./mydata")
let row = try db.queryOne("SELECT body FROM notes WHERE id = $1", [.integer(1)])
```

## Thread Safety

`Database` and `PreparedStatement` are `@unchecked Sendable` and safe to share across threads. The engine keeps each handle's last error on the handle, so the driver serializes the calls made on one `Database` or `PreparedStatement`. For queries that run in parallel, open the same DSN once per thread: the opens share one engine and its data. `Transaction` and `RowCursor` belong to the thread that uses them.

## Building from Source

Requires [Rust](https://rustup.rs) (stable) and Swift 6.0+.

```bash
git clone https://github.com/stoolap/stoolap-swift.git
cd stoolap-swift

# Build the Rust shared library
./scripts/build-rust.sh

# Build and test against the library built above
STOOLAP_LOCAL_BUILD=1 swift build
STOOLAP_LOCAL_BUILD=1 swift test
```

Without `STOOLAP_LOCAL_BUILD=1`, the package links the prebuilt `StoolapC.xcframework` of the release named in `Package.swift`.
