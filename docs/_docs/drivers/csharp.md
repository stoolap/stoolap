---
layout: doc
title: C# Driver
category: Drivers
order: 5
icon: csharp
---

# C# Driver

High-performance .NET driver for Stoolap. Binds directly to `libstoolap` through source-generated `[LibraryImport]` P/Invoke (no C/C++ shim, no JNI layer). Ships with a full ADO.NET provider for drop-in use with `System.Data.Common`, Dapper, and anything that speaks `DbConnection`.

Targets **.NET 8** and **.NET 9**. Supported platforms: macOS (arm64/x64), Linux (x64/arm64), Windows (x64).

## Installation

```bash
dotnet add package Stoolap
```

Or in your project file:

```xml
<PackageReference Include="Stoolap" Version="0.5.0" />
```

Driver 0.5.0 is built on Stoolap engine 0.5.0. The package ships the native library for every supported platform under `runtimes/<rid>/native/`. Requires **.NET 8 SDK or newer**. To build from source instead, see [Building from Source](#building-from-source).

## Native Library Loading

The driver searches for `libstoolap.{dylib,so,dll}` in this order. On Windows the native file is `libstoolap.dll` rather than `stoolap.dll`, so it cannot collide with the managed `Stoolap.dll` on the case-insensitive filesystem.

1. Absolute path in the `STOOLAP_LIB_PATH` environment variable.
2. The RID-specific folder `runtimes/<rid>/native/` next to the running assembly. This is the canonical NuGet layout and is the resolver's preferred location, both for packed-package consumers and for project-reference consumers (the build targets copy the host platform's binary into this subfolder).
3. The assembly's base directory next to `Stoolap.dll`, where a RID-specific build or publish places it.
4. The OS loader search path (`LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`, `PATH`, `/usr/local/lib`, etc.).

```bash
export STOOLAP_LIB_PATH=/absolute/path/to/libstoolap.dylib
dotnet run
```

Resolution is wired through `NativeLibrary.SetDllImportResolver` so there is no platform-specific loader code in user projects.

## Quick Start

```csharp
using Stoolap;

using var db = Database.OpenInMemory();

db.Execute("""
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT
    )
""");

db.Execute("INSERT INTO users (id, name, email) VALUES (?, ?, ?)",
    1, "Alice", "alice@example.com");

var result = db.Query("SELECT id, name, email FROM users ORDER BY id");
foreach (var row in result.Rows)
{
    long id = (long)row[0]!;
    string name = (string)row[1]!;
    string? email = row[2] as string;
    Console.WriteLine($"{id} {name} {email}");
}
```

## Opening a Database

```csharp
// In-memory (isolated, each call creates a new instance)
using var db = Database.OpenInMemory();

// In-memory via DSN (one engine per DSN string)
using var db = Database.Open("memory://");

// File-based (data persists across restarts)
using var db = Database.Open("file:///absolute/path/to/db");
```

`Database.OpenInMemory()` always returns a fresh engine. `Database.Open(dsn)` routes through a global DSN registry: opening a DSN that is already open gives a new session on the same engine, with its own transaction state, so two handles on `memory://` see the same tables. For isolated in-memory databases, prefer `OpenInMemory()` or a unique DSN suffix like `memory://test-{guid}`.

## Core API

| Method | Returns | Description |
|--------|---------|-------------|
| `Database.Open(dsn)` | `Database` | Open a database by DSN |
| `Database.OpenInMemory()` | `Database` | Open a fresh in-memory database |
| `Database.Version` | `string` | Version of the underlying libstoolap |
| `db.Execute(sql)` | `long` | Execute DDL/DML, returns rows affected |
| `db.Execute(sql, params)` | `long` | Execute with positional parameters |
| `db.Query(sql)` | `QueryResult` | Materialized query (one-call path) |
| `db.Query(sql, params)` | `QueryResult` | Materialized parameterized query |
| `db.QueryStream(sql)` | `Rows` | Streaming row reader |
| `db.QueryStream(sql, params)` | `Rows` | Streaming parameterized query |
| `db.Prepare(sql)` | `PreparedStatement` | Create a prepared statement |
| `db.Begin()` | `Transaction` | Begin a transaction (READ COMMITTED) |
| `db.Begin(isolation)` | `Transaction` | Begin with explicit isolation level |
| `db.Clone()` | `Database` | Clone for multi-threaded use |
| `db.Dispose()` | `void` | Close the database |

### QueryResult vs Rows

The driver exposes two read paths so callers can pick per workload:

- **`QueryResult`**. Fully materialized. Backed by `stoolap_query_params_all` / `stoolap_stmt_query_all`, which run the statement and pack the entire result set into one binary buffer in a single P/Invoke call, with no rows handle to open or close. `BinaryRowParser` then decodes it in one pass. Use for small-to-medium result sets where the ergonomic `row[0]` / `result.Columns` API is useful.
- **`Rows`**. Streaming reader. One P/Invoke call per row advance, per-cell accessors (`GetInt64`, `GetString`, etc.). Use for large result sets, LINQ-style iteration, or when you want to stop early.

```csharp
// Materialized (one call, all rows decoded up-front)
var result = db.Query("SELECT id, name FROM users LIMIT 100");
foreach (var row in result.Rows)
{
    Console.WriteLine(row[0]);
}

// Streaming (per-row iteration, no full materialization)
using var rows = db.QueryStream("SELECT id, name FROM users");
while (rows.Read())
{
    long id = rows.GetInt64(0);
    string? name = rows.GetString(1);
    Console.WriteLine($"{id} {name}");
}
```

## Parameters

Positional `?` placeholders are the native parameter style:

```csharp
db.Execute("INSERT INTO t VALUES (?, ?, ?)", 1, "hello", 3.14);
db.Query("SELECT * FROM t WHERE id = ? AND name = ?", 1, "hello");
```

Parameters are marshalled with zero heap allocations in the driver layer: a stack-allocated `Span<StoolapValue>` holds the FFI-ready values, and a stack-allocated 1 KiB `byte*` scratch buffer receives UTF-8 payloads for short string/blob parameters. Only oversized payloads fall back to `Marshal.AllocHGlobal`, which is freed at the end of the call.

Every placeholder needs a value: a missing one is an error, and so is mixing positional (`?`, `$1`) and named (`:name`) placeholders in one statement.

Supported parameter types: `null` / `DBNull`, `bool`, `sbyte`/`byte`/`short`/`ushort`/`int`/`uint`/`long`/`ulong`, `float`/`double`/`decimal`, `string`, `DateTime`/`DateTimeOffset` (as `TIMESTAMP` with nanosecond precision), `byte[]` / `ReadOnlyMemory<byte>` (as a `VECTOR` payload of packed little-endian `float32` values; any other length is an error, since Stoolap has no byte BLOB type), `float[]` (as `VECTOR`), `Guid` (as `TEXT`).

## Prepared Statements

```csharp
using var db = Database.OpenInMemory();
db.Execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)");

using var insert = db.Prepare("INSERT INTO users VALUES (?, ?, ?)");
insert.Execute(1, "Alice", "alice@example.com");
insert.Execute(2, "Bob",   "bob@example.com");

using var lookup = db.Prepare("SELECT * FROM users WHERE id = ?");
var result = lookup.Query(1);
Console.WriteLine(result[0, 1]); // "Alice"
```

Prepared statements cache the parsed AST on the Rust side, so every subsequent `Execute`/`Query` skips SQL parsing entirely. The C# side remembers the last result header of each statement and reuses its decoded column names while the header bytes stay the same; a schema change after `Prepare` produces a different header, which is decoded afresh.

### PreparedStatement Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `Execute(params)` | `long` | Execute DDL/DML, returns rows affected |
| `Query(params)` | `QueryResult` | Materialized query |
| `QueryStream(params)` | `Rows` | Streaming query |
| `Sql` | `string` | The SQL text used to prepare |
| `Dispose()` | `void` | Finalize the native handle |

## Transactions

```csharp
using var db = Database.OpenInMemory();
db.Execute("CREATE TABLE accounts (id INTEGER, balance INTEGER)");
db.Execute("INSERT INTO accounts VALUES (1, 100)");
db.Execute("INSERT INTO accounts VALUES (2, 0)");

using (var tx = db.Begin())
{
    tx.Execute("UPDATE accounts SET balance = balance - 50 WHERE id = 1");
    tx.Execute("UPDATE accounts SET balance = balance + 50 WHERE id = 2");
    tx.Commit();
}
// Disposing without Commit automatically rolls back.
```

### Snapshot Isolation

```csharp
// StoolapIsolationLevel lives in the Stoolap.Native namespace.
using (var tx = db.Begin(StoolapIsolationLevel.Snapshot))
{
    // Sees a consistent view from the transaction's start point.
    // Writes from other connections are invisible until commit.
    var snapshot = tx.Query("SELECT * FROM t");
    tx.Commit();
}
```

### Prepared statements inside a transaction

```csharp
using var stmt = db.Prepare("INSERT INTO t VALUES (?, ?)");
using var tx = db.Begin();
for (int i = 0; i < 1000; i++)
{
    tx.Execute(stmt, i, $"row-{i}");
}
tx.Commit();
```

### Transaction Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `Execute(sql, params)` | `long` | Execute DDL/DML |
| `Query(sql, params)` | `QueryResult` | Materialized query (one-call path) |
| `QueryStream(sql, params)` | `Rows` | Streaming row reader inside the transaction |
| `Execute(stmt, params)` | `long` | Execute a prepared statement in this transaction |
| `Query(stmt, params)` | `QueryResult` | Query a prepared statement in this transaction |
| `Commit()` | `void` | Commit the transaction |
| `Rollback()` | `void` | Rollback the transaction (idempotent) |
| `Dispose()` | `void` | Auto-rollback if not committed |

A transaction keeps its database's native handle alive until it commits, rolls back or is disposed, so disposing the `Database` first does not cut it off.

## Multi-Threaded Use

A single `Database` instance owns one query executor and is intended to be used from a single thread at a time. For parallel workloads, call `Clone()` once per worker thread:

```csharp
using var main = Database.Open("file:///var/data/mydb");

void Worker()
{
    using var local = main.Clone();  // per-thread handle, shared engine
    var r = local.Query("SELECT COUNT(*) FROM t");
    Console.WriteLine(r[0, 0]);
}

var t1 = new Thread(Worker);
var t2 = new Thread(Worker);
t1.Start(); t2.Start();
t1.Join();  t2.Join();
```

Clones share the underlying engine (data, indexes, WAL) but each has its own executor and error state. Cloning is cheap.

Objects that come from a handle (`Transaction`, `PreparedStatement`, `Rows`, and the ADO.NET connection, command and reader) are not thread-safe either, but they are not tied to the thread that created them: using one from different threads one after another, for example across an `await`, is fine. A transaction that is never disposed is rolled back, and releases its connection, on the finalizer thread.

## ADO.NET Provider

A full `System.Data.Common` implementation lives in the `Stoolap.Ado` namespace. Any library that accepts a `DbConnection`, like Dapper, LINQ to DB, Entity Framework Core (with an adapter), or custom code, works out of the box.

### Connection String

| Keyword | Description |
|---------|-------------|
| `Data Source` | DSN to pass to `Database.Open` (e.g. `memory://`, `file:///path/to/db`) |
| `DataSource` | Alias for `Data Source` |
| `DSN` | Alias for `Data Source` |

```csharp
using Stoolap.Ado;

using var conn = new StoolapConnection("Data Source=file:///var/data/mydb");
conn.Open();

using var cmd = conn.CreateCommand();
cmd.CommandText = "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)";
cmd.ExecuteNonQuery();
```

### Named Parameters

ADO.NET idiom is named parameters (`@name`, `:name`, `$name`). The command layer rewrites these into numbered placeholders (`$1`, `$2`, ...), one per distinct name, before sending them to the engine, so a name used twice is one value (two anonymous `?` are two different expressions in Stoolap 0.5). String literals, double-quoted and backtick-quoted identifiers, line/block comments and the `::` cast operator are left alone. A command whose SQL names a parameter missing from its `Parameters` collection throws before it runs.

```csharp
using var cmd = conn.CreateCommand();
cmd.CommandText = "INSERT INTO users VALUES (@id, @name)";
cmd.Parameters.Add(new StoolapParameter("@id", 1));
cmd.Parameters.Add(new StoolapParameter("@name", "Alice"));
cmd.ExecuteNonQuery();

using var read = conn.CreateCommand();
read.CommandText = "SELECT name FROM users WHERE id = @id";
read.Parameters.Add(new StoolapParameter("@id", 1));
var name = (string?)read.ExecuteScalar();
```

The leading sigil (`@`, `:`, or `$`) is stripped when `ParameterName` is normalized, so `"@id"` and `"id"` refer to the same parameter in the collection.

### DataReader

```csharp
using var cmd = conn.CreateCommand();
cmd.CommandText = "SELECT id, name FROM users ORDER BY id";
using var reader = cmd.ExecuteReader();
while (reader.Read())
{
    long id = reader.GetInt64(0);
    string name = reader.GetString(1);
    Console.WriteLine($"{id} {name}");
}
```

`StoolapDataReader` wraps a streaming `Rows` handle, for connection-level and transaction-level commands alike: one P/Invoke call per `Read` and per-cell native accessors. `ExecuteScalar` uses the materialized one-call path.

Behaviour that ADO.NET consumers such as Dapper rely on:

- **Column types are learned from the data.** The engine reports each value's type, not a result column's declared type. The reader reads the first row when it opens, and when a column is NULL there it reads ahead (up to 64 rows, replayed in order) until every column has shown a value, so `GetFieldType` is right before the first `Read` even when the first rows hold NULLs. A column that is NULL in all of those rows, and every column of an empty result, reports `object`. Dapper copes with that when a class property's type is exactly the value's type (`long?`, `double?`, `string`, `DateTime?`, `bool?`): an empty result maps to an empty list, and so does `QueryFirstOrDefault`. Two cases still fail, because the engine does not report column types on its own: constructor mapping of a positional `record` on an empty result, and a narrower property (such as `int?` for an INTEGER column) whose column's first value comes after the first 64 rows. Prefer classes with exact property types for queries that can return no rows or long runs of NULLs.
- **Typed getters are strict.** `GetInt64`, `GetString` and the other typed getters throw `InvalidCastException` on a NULL value; check `IsDBNull` or read the value with `GetValue`, which returns `DBNull.Value`. A getter reads a value of its own type, and converts between INTEGER, FLOAT, BOOLEAN and numeric TEXT (`GetString` also formats numbers and booleans); any other conversion, such as `GetString` on a TIMESTAMP or `GetInt64` on non-numeric TEXT, throws `InvalidCastException`. An ordinal outside the result throws `IndexOutOfRangeException`.
- **`RecordsAffected`** is the number of rows a DML statement changed and -1 for a query, and stays readable after the reader is closed.
- **Closing the connection** closes its open readers and rolls back its transactions that have not ended. `CommandBehavior.CloseConnection` closes the connection when the reader closes.
- **`GetFieldValue<T>`** converts through the typed getters (`GetFieldValue<int>` works on an INTEGER column), and **`GetSchemaTable`** describes the result, so `DataTable.Load` works.
- **Not supported:** a command returns one result set (`NextResult` is always false, so Dapper's `QueryMultiple` cannot be used); `CommandTimeout` is not applied (set a per-connection limit with the `query_timeout` DSN key or `PRAGMA query_timeout`); a `Guid` is stored as TEXT, so Dapper needs a type handler (`SqlMapper.AddTypeHandler`) to map it back to a `Guid` property.

All standard accessors are implemented: `GetInt32`, `GetInt64`, `GetString`, `GetDouble`, `GetBoolean`, `GetDateTime`, `GetFieldValue<T>`, `GetValues`, `IsDBNull`, `GetOrdinal`, `GetName`, `GetFieldType`, `GetSchemaTable`, string indexer, integer indexer, `NextResult`.

### Transactions via ADO.NET

```csharp
using var conn = new StoolapConnection("Data Source=memory://test");
conn.Open();

using var tx = conn.BeginTransaction();
using (var cmd = conn.CreateCommand())
{
    cmd.Transaction = tx;
    cmd.CommandText = "INSERT INTO accounts VALUES (@id, @bal)";
    cmd.Parameters.Add(new StoolapParameter("@id", 1));
    cmd.Parameters.Add(new StoolapParameter("@bal", 100));
    cmd.ExecuteNonQuery();
}
tx.Commit();
```

`IsolationLevel.Snapshot` and `IsolationLevel.ReadCommitted` are supported; `IsolationLevel.Unspecified` defaults to READ COMMITTED.

### Dapper

```csharp
using Stoolap.Ado;
using Dapper;

await using var conn = new StoolapConnection("Data Source=memory://");
conn.Open();

await conn.ExecuteAsync("""
    CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)
""");

await conn.ExecuteAsync(
    "INSERT INTO users VALUES (@id, @name)",
    new { id = 1, name = "Alice" });

var users = await conn.QueryAsync<User>(
    "SELECT id, name FROM users WHERE id >= @min",
    new { min = 1 });

record User(long Id, string Name);
```

The provider has no native async I/O: the async methods are the `System.Data.Common` defaults, which run synchronously on the calling thread and return a completed task.

## Type Mapping

| .NET (write) | Stoolap | .NET (read) |
|--------------|---------|-------------|
| `long`, `int`, `short`, `sbyte`, `byte`, `ushort`, `uint`, `ulong` | `INTEGER` | `long` |
| `double`, `float`, `decimal` | `FLOAT` | `double` |
| `string` | `TEXT` | `string` |
| `bool` | `BOOLEAN` | `bool` |
| `DateTime`, `DateTimeOffset` | `TIMESTAMP` (nanos UTC) | `DateTime` (UTC) |
| `string` (JSON) | `JSON` | `string` |
| `byte[]`, `ReadOnlyMemory<byte>` (packed little-endian f32, length a multiple of 4) | `VECTOR` | `float[]` |
| `float[]` | `VECTOR` | `float[]` |
| `Guid` | `TEXT` (string form) | `string` |
| `null`, `DBNull.Value` | `NULL` | `null` |
| `enum` | `INTEGER` (underlying value) | `long` |
| `char` | `TEXT` | `string` |

Any other parameter type throws `NotSupportedException` rather than being stored in some string form. `ulong` values above `long.MaxValue` throw `OverflowException`, and `DateTime` values outside 1677-09-21 to 2262-04-11 (the range of an i64 nanosecond `TIMESTAMP`) throw `ArgumentOutOfRangeException`.

Aggregate results (`SUM`, `AVG` over integer columns) may be returned as `long` or `double` depending on the planner's promotion rules. Use `Convert.ToInt64` / `Convert.ToDouble` when reading aggregate output.

## Performance

The driver is built around six performance principles:

1. **Source-generated P/Invoke.** Every native entry point uses `[LibraryImport]`, which generates marshalling stubs at compile time instead of at JIT time. No per-call IL stubs, full AOT compatibility.
2. **UTF-8 end to end.** Stoolap is UTF-8 throughout, and `StringMarshalling.Utf8` skips the UTF-16 round trip a plain `DllImport` would force.
3. **Zero-allocation parameter binding.** `Span<StoolapValue>` via `stackalloc` holds FFI values on the stack, and a 1 KiB `byte*` scratch buffer receives short UTF-8 payloads. `[SkipLocalsInit]` on the hot path avoids zeroing the scratch buffer. The driver itself allocates **zero bytes per call**; all per-call heap traffic is at the caller (the `params object?[]` array, value-type boxes, and string interpolations).
4. **One-call read path.** `Query()` on `Database`, `Transaction` and `PreparedStatement` issues one P/Invoke call: `stoolap_query_params_all` / `stoolap_stmt_query_all` run the statement and pack the whole result into a buffer owned by the handle, so there is no rows handle to open, no second call to fetch, and nothing to free. The buffer is decoded in one pass over a `ReadOnlySpan<byte>`. A prepared point lookup costs about a quarter less than the v0.4 open-fetch-close sequence.
5. **Cached column headers.** Each connection and prepared statement remembers the last result header and reuses its decoded column names when the next result has the same bytes.
6. **`SafeHandle` everywhere.** Every opaque pointer (`StoolapDB*`, `StoolapStmt*`, `StoolapRows*`, `StoolapTx*`) is wrapped in a `SafeHandle` subclass, so handles are freed on every exception path and during AppDomain teardown without leaking.

### Running the Benchmark

The driver ships with `benchmark/Stoolap.Benchmark.csproj`, which runs a fixed set of operations against both Stoolap and Microsoft.Data.Sqlite on the same in-memory dataset. Run it yourself:

```bash
dotnet run --project benchmark/Stoolap.Benchmark.csproj -c Release
```

## Error Handling

All driver errors surface as `StoolapException`, which extends `Exception`:

```csharp
using Stoolap;

using var db = Database.OpenInMemory();
try
{
    db.Execute("CREATE TABLE t (id INTEGER PRIMARY KEY)");
    db.Execute("INSERT INTO t VALUES (1)");
    db.Execute("INSERT INTO t VALUES (1)"); // duplicate PK
}
catch (StoolapException ex)
{
    Console.Error.WriteLine($"Database error ({ex.StatusCode}): {ex.Message}");
}
```

Inside the ADO.NET layer, the same exception propagates through `DbCommand.ExecuteNonQuery` / `ExecuteReader` calls and can be caught directly as a `StoolapException` or as its base `Exception`.

## Architecture

```
+------------------------------------------------------+
|               Your .NET application                  |
+------------------------------------------------------+
|  Stoolap.Ado.*  (ADO.NET)  |  Stoolap.*  (core)      |
|  +-- StoolapConnection     |  +-- Database           |
|  +-- StoolapCommand        |  +-- PreparedStatement  |
|  +-- StoolapDataReader     |  +-- Transaction        |
|  +-- StoolapParameter      |  +-- Rows / QueryResult |
|  +-- NamedParameterRewriter|  +-- ParameterBinder    |
|  +-- StoolapTransaction    |  +-- BinaryRowParser    |
+------------------------------------------------------+
|  Stoolap.Native (bindings internal; public enums)     |
|  +-- NativeMethods  [LibraryImport] bindings          |
|  +-- StoolapValue   [StructLayout] tagged union       |
|  +-- SafeHandle wrappers                              |
|  +-- LibraryResolver (STOOLAP_LIB_PATH + RID lookup)  |
+------------------------------------------------------+
                          |
                          | P/Invoke (stable C ABI)
                          v
+------------------------------------------------------+
|    libstoolap.{dylib,so,dll}  (Rust, --features ffi) |
|    src/ffi/{database,statement,transaction,rows}.rs  |
+------------------------------------------------------+
                          |
                          v
+------------------------------------------------------+
|                 stoolap crate (Rust)                 |
|  MVCC, columnar indexes, volume storage, WAL         |
+------------------------------------------------------+
```

## Testing

The repository ships with a full xUnit suite across thirteen test files, covering both target frameworks (`net8.0` and `net9.0`):

- `SmokeTests.cs`. Open/close, execute, query, streaming, prepared statements, transactions, clone, NULL parameters.
- `ParameterBinderTests.cs`. Scratch-buffer fast path, HGlobal slow path, boundary cases, all primitive types, stackalloc capacity transitions.
- `NamedParameterRewriterTests.cs`. `@/:/$` sigils, string literals, escaped quotes, quoted identifiers, line/block comments, emails, duplicate names.
- `ConnectionStringBuilderTests.cs`. `DataSource` round-trip, `DSN` alias normalization, indexer.
- `CommandAndParameterTests.cs`. Command lifecycle, parameter collection, named parameter rewriting, scalar/non-query execution, positional fallback, missing-parameter errors.
- `DataReaderTests.cs`. `FieldCount`, `GetName`/`GetOrdinal`, numeric getters, `GetValues`, `IsDBNull`, `NextResult`, empty result, `GetFieldType`, indexers.
- `SqlFeatureTests.cs`. Aggregates, `GROUP BY`, `HAVING`, `ORDER BY`, `LIMIT`/`OFFSET`, `INNER`/`LEFT JOIN`, `DISTINCT`, `IN`, `LIKE`, CTEs, subqueries, `CASE`, `DROP TABLE`.
- `ErrorHandlingTests.cs`. Invalid SQL, missing tables, duplicate tables, disposed objects, transaction-after-end, null arguments.
- `TypeRoundTripTests.cs`. Every Stoolap type through both the binary and streaming read paths, including 100 K-byte strings, Unicode payloads, timestamps, vectors, JSON, and NULL.
- `AdoTests.cs`. ADO.NET connection lifecycle, reader streaming, transaction rollback.
- `FastPathTests.cs`. One-call queries on every handle, the column header cache across schema changes, transaction reads of uncommitted writes, a transaction outliving its `Database`, engine error messages, streaming every value type.
- `AdoContractTests.cs`. Dapper mapping with NULL-first rows, column type look-ahead, connection close and `CloseConnection`, `RecordsAffected`, ordinal and NULL errors, `GetFieldValue`, `GetSchemaTable`, parameter type rules.
- `RegressionTests.cs`. Driver-contract regressions: `HasRows` accuracy on empty results, `GetFieldType` schema stability before `Read()`, transaction-foreign-connection rejection, transactional `ExecuteReader` streaming behavior.

Run the full suite:

```bash
dotnet test -c Release
```

## Building from Source

Requires:

- [Rust](https://rustup.rs) (stable)
- [.NET 8 SDK](https://dotnet.microsoft.com/download) or newer

```bash
git clone https://github.com/stoolap/stoolap-csharp.git
cd stoolap-csharp

# Build the native library for the host platform.
# Auto-clones the stoolap engine at the pinned ref if no source is found.
./build/build-native.sh

# Build and test everything
dotnet build -c Release
dotnet test  -c Release

# Run the comparison benchmark
dotnet run --project benchmark/Stoolap.Benchmark.csproj -c Release
```

The build script resolves the stoolap source in this order:

1. `$STOOLAP_ROOT` if set and points at a Cargo project.
2. `../stoolap` (a sibling checkout next to this repo).
3. Auto-clones `github.com/stoolap/stoolap` at the version pinned in `STOOLAP_ENGINE_REF` (default `v0.5.0`) into `build/.stoolap-engine/`. The clone is gitignored and reused on subsequent runs.

It then runs `cargo build --release --features ffi`, detects the host OS/arch, and drops the resulting binary into `runtimes/<rid>/native/`, the standard NuGet convention. This is the same layout the published package uses to ship per-platform binaries.

The repo also includes a `global.json` pinning the .NET SDK to 9.0 with `latestFeature` rollforward, so the multi-target `net8.0;net9.0` build works from a single SDK install.

## License

Apache-2.0.
