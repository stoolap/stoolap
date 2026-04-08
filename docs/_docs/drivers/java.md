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

Until the first release is published to Maven Central, build from source:

```bash
git clone https://github.com/stoolap/stoolap-java.git
cd stoolap-java

# 1. Build the native library (requires Rust)
cd jni && cargo build --release && cd ..

# 2. Build the Java artifact
mvn package

# 3. Install to your local Maven repo
mvn install
```

In your `pom.xml`:

```xml
<dependency>
    <groupId>io.stoolap</groupId>
    <artifactId>stoolap-java</artifactId>
    <version>0.4.0</version>
</dependency>
```

Requires **Java 17+** (tested on Java 17, 21, 25). Supported platforms: macOS (aarch64/x86_64), Linux (x86_64/aarch64), Windows (x86_64).

## Native Library Loading

The driver searches for `libstoolap_jni.{dylib,so,dll}` in this order:

1. Path in `STOOLAP_LIB` environment variable
2. Bundled JAR resource at `/native/{os}-{arch}/libstoolap_jni.{ext}`
3. System library path (`java.library.path`)

```bash
export STOOLAP_LIB=/path/to/stoolap-java/jni/target/release/libstoolap_jni.dylib
java -cp stoolap-java-0.4.0.jar:your-app.jar your.Main
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
| `query(sql)` | `BulkDecoder.Result` | Execute query, returns all rows |
| `query(sql, params...)` | `BulkDecoder.Result` | Query with positional parameters |
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

Prepared statements parse SQL once and reuse the cached execution plan on every call. The Java driver stores a `CachedPlanRef` internally, so `execute`/`query` bypass SQL parsing and plan-cache lookup entirely.

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
| `query(params...)` | `BulkDecoder.Result` | Execute query |
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
| `execute(sql, params...)` | `long` | Execute DDL/DML |
| `query(sql, params...)` | `BulkDecoder.Result` | Execute query |
| `execute(stmt, params...)` | `long` | Execute a prepared statement within the transaction |
| `query(stmt, params...)` | `BulkDecoder.Result` | Query a prepared statement within the transaction |
| `commit()` | `void` | Commit the transaction |
| `rollback()` | `void` | Rollback the transaction |
| `close()` | `void` | Auto-rollback if not committed |

## Multi-Threaded Use

A single `StoolapDB` handle must not be used concurrently from multiple threads. Use `cloneHandle()` to create a per-thread handle that shares the underlying engine (data, indexes, transactions):

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

Cloning is cheap and integrates naturally with JDBC connection pools like HikariCP.

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

The JDBC batch path auto-wraps in a transaction and uses `execute_prepared` on the cached plan AST, so per-row overhead is minimal.

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
        stmt.executeUpdate("INSERT INTO t VALUES (2, 'b')");
        conn.commit();
    } catch (SQLException e) {
        conn.rollback();
        throw e;
    } finally {
        conn.setAutoCommit(true);
    }
}
```

### Connection Pool Integration

`StoolapConnection` clones the underlying handle, making it safe to pool. Example with HikariCP:

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
| `Long`, `Integer`, `Short`, `Byte` | `INTEGER` | `Long` |
| `Double`, `Float`, `BigDecimal` | `FLOAT` | `Double` |
| `String` | `TEXT` | `String` |
| `Boolean` | `BOOLEAN` | `Boolean` |
| `java.time.Instant`, `java.sql.Timestamp` | `TIMESTAMP` | `java.time.Instant` |
| `String` (JSON) | `JSON` | `String` |
| `byte[]` | `BLOB` | `byte[]` |
| `null` | `NULL` | `null` |

Aggregate results (`SUM`, `AVG`, `MIN`, `MAX` over integer columns) may be returned as `Double` depending on the engine's promotion rules. Use `((Number) value).longValue()` when reading aggregate output.

## Parameters

The Java driver currently supports positional parameters only (`$1`, `$2`, ...):

```java
db.query("SELECT * FROM users WHERE id = $1 AND name = $2", 1L, "Alice");
db.execute("INSERT INTO t VALUES ($1, $2, $3)", 42L, "hello", 3.14);
```

Params are encoded to a compact binary format on the Java side by `ParamEncoder` and decoded in Rust without any JNI reflection, keeping the per-call overhead minimal.

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

JDBC code catches `SQLException` the normal way.

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
|  +-- CachedPlanRef for zero-parse execution          |
|  +-- encode_rows / decode_binary_params              |
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
