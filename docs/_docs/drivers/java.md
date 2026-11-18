---
layout: doc
title: Java Driver
category: Drivers
order: 4
icon: java
---

# Java Driver

High-performance Java driver for Stoolap. Built with [jni-rs](https://github.com/jni-rs/jni-rs) to call the Rust API directly (no C FFI layer). Ships with a full JDBC driver so it works with any standard Java SQL code, HikariCP, JPA, and other JDBC tooling.

## Installation

The driver is published to Maven Central. The JAR bundles the native library for every supported platform, so no Rust toolchain is needed.

```xml
<dependency>
    <groupId>io.stoolap</groupId>
    <artifactId>stoolap-java</artifactId>
    <version>0.5.0</version>
</dependency>
```

Gradle:

```groovy
implementation 'io.stoolap:stoolap-java:0.5.0'
```

Driver 0.5.x runs the Stoolap 0.5 engine. It opens a database written by 0.4.x and moves its files forward on their own; see the [migration guide]({{ '/docs/development/migration-v042-to-v050/' | relative_url }}) before upgrading.

Requires **Java 17+** (tested on Java 17, 21, 25). Supported platforms: macOS (aarch64/x86_64), Linux (x86_64/aarch64), Windows (x86_64).

## Native Library Loading

The driver searches for `libstoolap_jni.{dylib,so,dll}` in this order:

1. Path in `STOOLAP_LIB` environment variable
2. Bundled JAR resource at `/native/{os}-{arch}/libstoolap_jni.{ext}`
3. System library path (`java.library.path`)

```bash
export STOOLAP_LIB=/path/to/stoolap-java/jni/target/release/libstoolap_jni.dylib
java -cp stoolap-java-0.5.0.jar:your-app.jar your.Main
```

## Quick Start

```java
import io.stoolap.StoolapDB;
import io.stoolap.internal.BulkDecoder;

try (StoolapDB db = StoolapDB.openInMemory()) {
    db.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT
        )
    """);

    // Insert with positional parameters ($1, $2, ...)
    db.execute(
        "INSERT INTO users (id, name, email) VALUES ($1, $2, $3)",
        1L, "Alice", "alice@example.com"
    );

    // Query rows — result contains column names and row data
    BulkDecoder.Result users = db.query("SELECT * FROM users ORDER BY id");
    for (Object[] row : users.rows()) {
        long id = (Long) row[0];
        String name = (String) row[1];
        String email = (String) row[2];
        System.out.println(id + " " + name + " " + email);
    }
}
```

## Opening a Database

```java
// In-memory (isolated, each call creates a new instance)
StoolapDB db = StoolapDB.openInMemory();

// In-memory via DSN
StoolapDB db = StoolapDB.open("memory://");

// File-based (data persists across restarts)
StoolapDB db = StoolapDB.open("file:///absolute/path/to/db");
```

## Core API Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `execute(sql)` | `long` | Execute DDL/DML statement, returns rows affected |
| `execute(sql, params...)` | `long` | Execute with positional parameters |
| `execute(sql, Map)` | `long` | Execute with named parameters |
| `query(sql)` | `BulkDecoder.Result` | Execute query, returns all rows |
| `query(sql, params...)` | `BulkDecoder.Result` | Query with positional parameters |
| `query(sql, Map)` | `BulkDecoder.Result` | Query with named parameters |
| `cursor(sql, params...)` | `StoolapCursor` | Query, reading the result in chunks |
| `cursor(sql, Map)` | `StoolapCursor` | Query with named parameters, in chunks |
| `prepare(sql)` | `StoolapStmt` | Create a prepared statement |
| `begin()` | `StoolapTx` | Begin a transaction (READ COMMITTED) |
| `begin(boolean snapshot)` | `StoolapTx` | Begin with SNAPSHOT isolation if `true` |
| `cloneHandle()` | `StoolapDB` | Clone for multi-threaded use |
| `close()` | `void` | Close the database |

### BulkDecoder.Result

`query()` returns a `BulkDecoder.Result` record with:

```java
public record Result(String[] columnNames, List<Object[]> rows) {
    int getColumnCount();
    int getRowCount();
}
```

Each row is an `Object[]` containing typed values. See [Type Mapping](#type-mapping) for how SQL types map to Java types.

### Cursors

`query()` returns the whole result at once. For a large result, `cursor()` reads it 1024 rows per JNI call:

```java
try (StoolapCursor c = db.cursor("SELECT * FROM events WHERE day = $1", day)) {
    String[] columns = c.columnNames();
    for (Object[] row; (row = c.next()) != null; ) {
        // ...
    }
}
```

`next()` returns `null` once the result is exhausted, and the cursor releases its native result then or on `close()`. A runtime error part way through a result, such as an invalid parameterized `REGEXP` pattern, is raised from `next()`; `query()` raises it too instead of returning the rows read so far. Read a cursor opened in a transaction before the transaction ends.

## Persistence

File-based databases persist data to disk using WAL (Write-Ahead Logging) and immutable cold volumes. A background checkpoint cycle seals hot rows into columnar volume files, compacts them, and truncates the WAL. Data survives process restarts.

```java
try (StoolapDB db = StoolapDB.open("file:///tmp/mydata")) {
    db.execute("CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT)");
    db.execute("INSERT INTO kv VALUES ($1, $2)", "hello", "world");
}

// Reopen: data is still there
try (StoolapDB db = StoolapDB.open("file:///tmp/mydata")) {
    BulkDecoder.Result result = db.query("SELECT * FROM kv WHERE key = $1", "hello");
    System.out.println(result.rows().get(0)[1]); // "world"
}
```

### Configuration

Pass configuration as query parameters in the DSN:

```java
// Maximum durability (fsync on every WAL write)
StoolapDB db = StoolapDB.open("file:///tmp/mydata?sync_mode=full");

// Custom checkpoint interval with compression
StoolapDB db = StoolapDB.open(
    "file:///tmp/mydata?checkpoint_interval=60&wal_compression=on&volume_compression=on"
);

// Multiple options
StoolapDB db = StoolapDB.open(
    "file:///tmp/mydata?sync_mode=full&checkpoint_interval=120&compact_threshold=8"
);
```

All configuration options from the core engine are supported (see the [Python driver](python#persistence) for the full table — the same keys work here).

## Prepared Statements

Prepared statements parse SQL once and run on their compiled plan on every call, on the database and inside a transaction alike.

```java
try (StoolapDB db = StoolapDB.openInMemory()) {
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)");

    try (StoolapStmt insert = db.prepare("INSERT INTO users VALUES ($1, $2, $3)")) {
        insert.execute(1L, "Alice", "alice@example.com");
        insert.execute(2L, "Bob", "bob@example.com");
    }

    try (StoolapStmt lookup = db.prepare("SELECT * FROM users WHERE id = $1")) {
        BulkDecoder.Result result = lookup.query(1L);
        if (result.getRowCount() > 0) {
            Object[] row = result.rows().get(0);
            System.out.println("Found: " + row[1]);
        }
    }
}
```

### StoolapStmt Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `execute(params...)` | `long` | Execute DDL/DML, returns rows affected |
| `execute(Map)` | `long` | Execute with named parameters |
| `query(params...)` | `BulkDecoder.Result` | Execute query |
| `query(Map)` | `BulkDecoder.Result` | Query with named parameters |
| `cursor(params...)` / `cursor(Map)` | `StoolapCursor` | Query, reading the result in chunks |
| `getSql()` | `String` | Return the original SQL text |
| `close()` | `void` | Finalize the native handle |

## Transactions

```java
try (StoolapDB db = StoolapDB.openInMemory()) {
    db.execute("CREATE TABLE accounts (id INTEGER, balance INTEGER)");
    db.execute("INSERT INTO accounts VALUES (1, 100)");
    db.execute("INSERT INTO accounts VALUES (2, 0)");

    try (StoolapTx tx = db.begin()) {
        tx.execute("UPDATE accounts SET balance = balance - 50 WHERE id = 1");
        tx.execute("UPDATE accounts SET balance = balance + 50 WHERE id = 2");
        tx.commit();
        // Auto-rollback on close if commit wasn't called
    }
}
```

### Snapshot Isolation

```java
try (StoolapTx tx = db.begin(true /* snapshot */)) {
    // Sees a consistent view from the transaction's start point.
    // Writes from other connections are invisible until commit.
    BulkDecoder.Result snapshot = tx.query("SELECT * FROM t");
    tx.commit();
}
```

### Transaction Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `execute(sql, params...)` / `execute(sql, Map)` | `long` | Execute DDL/DML |
| `query(sql, params...)` / `query(sql, Map)` | `BulkDecoder.Result` | Execute query |
| `execute(stmt, params...)` / `execute(stmt, Map)` | `long` | Execute a prepared statement within the transaction |
| `query(stmt, params...)` / `query(stmt, Map)` | `BulkDecoder.Result` | Query a prepared statement within the transaction |
| `cursor(sql, params...)` / `cursor(stmt, params...)` | `StoolapCursor` | Query within the transaction, in chunks |
| `commit()` | `void` | Commit; a failed commit leaves the transaction active so it can be rolled back |
| `rollback()` | `void` | Rollback the transaction |
| `close()` | `void` | Auto-rollback if not committed |

## Multi-Threaded Use

Every handle is thread-safe. Several threads may run statements on one `StoolapDB` or `StoolapStmt` at once, and calls on one `StoolapTx` or `StoolapCursor` run one at a time. `close()` waits for the calls in flight, and a call on a closed handle raises `StoolapException` instead of touching freed memory. A handle that is never closed is freed once it becomes unreachable, but close handles explicitly: an open database keeps its files locked. `cloneHandle()` gives another handle on the same engine (data, indexes, transactions), closed on its own:

```java
StoolapDB main = StoolapDB.open("file:///tmp/mydata");

Runnable worker = () -> {
    try (StoolapDB local = main.cloneHandle()) {
        BulkDecoder.Result r = local.query("SELECT COUNT(*) FROM t");
        System.out.println(r.rows().get(0)[0]);
    } catch (Exception e) {
        e.printStackTrace();
    }
};

Thread t1 = new Thread(worker);
Thread t2 = new Thread(worker);
t1.start(); t2.start();
t1.join();  t2.join();

main.close();
```

Cloning is cheap, and each JDBC connection holds its own handle.

## JDBC Driver

Full `java.sql` implementation for drop-in compatibility with existing JDBC code.

### Registration

The driver auto-registers via the `META-INF/services/java.sql.Driver` SPI. Just use the URL:

```java
import java.sql.*;

try (Connection conn = DriverManager.getConnection("jdbc:stoolap:memory://");
     Statement stmt = conn.createStatement()) {
    stmt.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)");
    stmt.executeUpdate("INSERT INTO users VALUES (1, 'Alice')");

    try (ResultSet rs = stmt.executeQuery("SELECT id, name FROM users")) {
        while (rs.next()) {
            System.out.println(rs.getInt("id") + " " + rs.getString("name"));
        }
    }
}
```

### JDBC URL Formats

| URL | Description |
|-----|-------------|
| `jdbc:stoolap:memory://` | In-memory database |
| `jdbc:stoolap:file:///path/to/db` | File-based |
| `jdbc:stoolap:file:///path/to/db?sync_mode=full` | File-based with DSN options |

### PreparedStatement

```java
try (Connection conn = DriverManager.getConnection("jdbc:stoolap:memory://")) {
    conn.createStatement().execute("CREATE TABLE t (id INTEGER, name TEXT, score FLOAT)");

    try (PreparedStatement ps = conn.prepareStatement("INSERT INTO t VALUES ($1, $2, $3)")) {
        ps.setLong(1, 1);
        ps.setString(2, "Alice");
        ps.setDouble(3, 95.5);
        ps.executeUpdate();
    }
}
```

### Batch Execution

In auto-commit mode a batch runs in a transaction of its own and every row runs on the statement's compiled plan: all rows commit, or none do. A failing row raises a `BatchUpdateException` naming it (`batch row 2: ...`). With auto-commit off, the batch runs in the connection's transaction.

```java
try (Connection conn = DriverManager.getConnection("jdbc:stoolap:memory://")) {
    conn.createStatement().execute("CREATE TABLE t (id INTEGER, name TEXT)");

    try (PreparedStatement ps = conn.prepareStatement("INSERT INTO t VALUES ($1, $2)")) {
        for (int i = 0; i < 1000; i++) {
            ps.setLong(1, i);
            ps.setString(2, "item_" + i);
            ps.addBatch();
        }
        ps.executeBatch();
    }
}
```

### Transaction Control via JDBC

```java
try (Connection conn = DriverManager.getConnection("jdbc:stoolap:memory://")) {
    conn.setAutoCommit(false);
    try (Statement stmt = conn.createStatement()) {
        stmt.executeUpdate("INSERT INTO t VALUES (1, 'a')");
        Savepoint sp = conn.setSavepoint();
        stmt.executeUpdate("INSERT INTO t VALUES (2, 'b')");
        conn.rollback(sp);              // keeps row 1
        conn.commit();
    } catch (SQLException e) {
        conn.rollback();
        throw e;
    } finally {
        conn.setAutoCommit(true);
    }
}
```

With auto-commit off, DML runs in a transaction that begins with the first statement. A Stoolap transaction accepts only DML and savepoints, so DDL commits the open transaction and runs on its own, as in MySQL and Oracle. `SHOW`, `DESCRIBE`, `EXPLAIN` and every `DatabaseMetaData` lookup run outside the transaction and leave it open. The SQL statements `BEGIN` and `COMMIT` commit the open transaction and `ROLLBACK` rolls it back.

`setTransactionIsolation` takes `TRANSACTION_READ_COMMITTED` (the default) or `TRANSACTION_REPEATABLE_READ`, which runs each transaction on a snapshot; `SERIALIZABLE` is refused.

### Generated Keys

`RETURN_GENERATED_KEYS` adds a `RETURNING` clause for the target table's primary key, and the column-name and column-index forms return those columns:

```java
try (PreparedStatement ps = conn.prepareStatement(
        "INSERT INTO users (name) VALUES (?)", Statement.RETURN_GENERATED_KEYS)) {
    ps.setString(1, "Alice");
    ps.executeUpdate();
    try (ResultSet keys = ps.getGeneratedKeys()) {
        keys.next();
        long id = keys.getLong(1);
    }
}
```

### Result Sets and Metadata

A `ResultSet` is forward-only and streams from the engine one fetch-size chunk at a time (1024 rows by default); `setMaxRows` stops it early. `Statement.execute()` reports a result set exactly when the statement returns columns, so `INSERT ... RETURNING` and a query with a leading comment are result sets. `ResultSetMetaData` reports column types, `getObject(column, type)` converts to the boxed numbers, `BigDecimal`, the `java.sql` date types and the `java.time` types, and a value a getter cannot represent raises an `SQLException` instead of returning 0. `setQueryTimeout` is enforced outside a transaction; `cancel()` is not supported.

`DatabaseMetaData` answers `getTables`, `getColumns`, `getPrimaryKeys`, `getIndexInfo`, `getImportedKeys`, `getExportedKeys`, `getTypeInfo` and `getTableTypes` from `SHOW TABLES`, `SHOW VIEWS`, `DESCRIBE`, `SHOW INDEXES` and `SHOW CREATE TABLE`. Stoolap has no catalogs or schemas, so those columns are null.

### Connection Pool Integration

Each `StoolapConnection` holds its own handle, so it pools well. Closing a connection closes its statements and result sets and rolls back an open transaction. Example with HikariCP:

```java
HikariConfig config = new HikariConfig();
config.setJdbcUrl("jdbc:stoolap:file:///var/data/mydb");
config.setMaximumPoolSize(10);

try (HikariDataSource ds = new HikariDataSource(config);
     Connection conn = ds.getConnection()) {
    // ...
}
```

## Type Mapping

| Java (write) | Stoolap | Java (read) |
|--------------|---------|-------------|
| `Long`, `Integer`, `Short`, `Byte`, `BigInteger` | `INTEGER` | `Long` |
| `Double`, `Float`, `BigDecimal` (as double) | `FLOAT` | `Double` |
| `String` | `TEXT` | `String` |
| `Boolean` | `BOOLEAN` | `Boolean` |
| `Instant`, `java.sql.Timestamp`, `java.util.Date`, `OffsetDateTime`, `ZonedDateTime`, `LocalDateTime` (as UTC) | `TIMESTAMP` | `java.time.Instant` (JDBC `getObject`: `java.sql.Timestamp`) |
| `String` (JSON) | `JSON` | `String` |
| `float[]` | `VECTOR(n)` | `float[]` |
| `null` | `NULL` | `null` |

A `TIMESTAMP` holds 1677-09-21 to 2262-04-11 UTC, and a value outside that range is refused. A `java.sql.Timestamp` binds as its instant, so it reads back unchanged in any JVM time zone. `java.sql.Date` and `java.sql.Time` bind as `yyyy-MM-dd` and `HH:mm:ss` text, which a `TIMESTAMP` column reads as UTC, and JDBC `getDate` and `getTime` read a timestamp's UTC date and time, so they round-trip too. Stoolap has no BLOB type: a `byte[]` parameter is refused, and JDBC `setBytes` throws `SQLFeatureNotSupportedException`. Any other object binds as its `toString()` text.

Aggregate results (`SUM`, `AVG`, `MIN`, `MAX` over integer columns) may be returned as `Double` depending on the engine's promotion rules. Use `((Number) value).longValue()` when reading aggregate output.

## Parameters

Positional parameters (`$1`, `$2`, ... or `?`) take varargs:

```java
db.query("SELECT * FROM users WHERE id = $1 AND name = $2", 1L, "Alice");
db.execute("INSERT INTO t VALUES ($1, $2, $3)", 42L, "hello", 3.14);
```

Named parameters (`:name`) take a `Map`, on `StoolapDB`, `StoolapStmt` and `StoolapTx`:

```java
db.execute("INSERT INTO users VALUES (:id, :name)", Map.of("id", 1, "name", "Alice"));

try (StoolapStmt byName = db.prepare("SELECT id FROM users WHERE name = :name")) {
    byName.query(Map.of("name", "Alice"));
}
```

A leading `:`, `@` or `$` on a key is ignored. A placeholder without a value raises `StoolapException`.

Params are encoded to a compact binary format on the Java side by `ParamEncoder` and decoded in Rust without any JNI reflection, keeping the per-call overhead minimal. Rust refuses a parameter buffer that is truncated, holds text that is not UTF-8, or carries an unknown type.

## Error Handling

The core API throws `StoolapException`, which extends `java.sql.SQLException` for JDBC compatibility:

```java
import io.stoolap.StoolapDB;
import io.stoolap.StoolapException;

try (StoolapDB db = StoolapDB.openInMemory()) {
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)");
    db.execute("INSERT INTO t VALUES (1)");
    db.execute("INSERT INTO t VALUES (1)"); // duplicate PK
} catch (StoolapException e) {
    System.err.println("Database error: " + e.getMessage());
}
```

JDBC code catches `SQLException` the normal way. Errors carry the SQLState of the engine error: `23505` for a unique or primary key violation, `23502` NOT NULL, `23503` foreign key, `40001` a write conflict worth retrying, `42P01` an unknown table, `42601` a syntax error. Using a closed database, statement or transaction raises an exception rather than crashing the JVM.

## Architecture

```
+------------------------------------------------------+
|              Your Java application                   |
+------------------------------------------------------+
|  java.sql.*  (JDBC)       |  io.stoolap.*  (core)   |
+------------------------------------------------------+
|  io.stoolap.internal                                |
|  +-- NativeBridge  (static native methods)           |
|  +-- ParamEncoder  (Java -> byte[])                  |
|  +-- BulkDecoder   (byte[] -> Object[])              |
+------------------------------------------------------+
           |          JNI (jni-rs)         |
           v                                v
+------------------------------------------------------+
|         stoolap-jni  (Rust crate in jni/)            |
|  +-- Wraps stoolap::api::{Database, Statement, Tx}   |
|  +-- Statement runs on its compiled plan             |
|  +-- packs results, whole or in cursor chunks        |
+------------------------------------------------------+
                          |
                          v
+------------------------------------------------------+
|                 stoolap crate (Rust)                 |
|  MVCC, columnar indexes, volume storage, WAL         |
+------------------------------------------------------+
```

Query results are encoded once in Rust and transferred to Java as a single `byte[]`, which is decoded lazily by `BulkDecoder`. This eliminates per-row JNI crossings, which is the main performance bottleneck in naive JNI drivers.

## Building from Source

Requires:

- [Rust](https://rustup.rs) (stable)
- [Java 17+](https://adoptium.net) (Java 21 LTS or later recommended)
- [Maven 3.9+](https://maven.apache.org)

```bash
git clone https://github.com/stoolap/stoolap-java.git
cd stoolap-java

# Build the native library
cd jni && cargo build --release && cd ..

# Build, test, and install to local Maven repo
export STOOLAP_LIB=$(pwd)/jni/target/release/libstoolap_jni.dylib  # or .so / .dll
mvn verify
```

Run the full benchmark against SQLite JDBC:

```bash
java -Djava.library.path=jni/target/release \
     -cp "target/classes:target/test-classes:$(mvn dependency:build-classpath -q -DincludeScope=test -Dmdep.outputFile=/dev/stdout)" \
     io.stoolap.Benchmark
```
