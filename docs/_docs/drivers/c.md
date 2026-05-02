---
layout: doc
title: C API (FFI)
category: Drivers
order: 9
icon: c
---

# C API (FFI)

C API for Stoolap with opaque handles, step-based result iteration, and per-handle error messages. No external dependencies beyond the shared library and header file.

## Building

```bash
cargo build --release --features ffi
```

This produces a shared library and a C header:

| Platform | Library | Header |
|----------|---------|--------|
| Linux | `target/release/libstoolap.so` | `include/stoolap.h` |
| macOS | `target/release/libstoolap.dylib` | `include/stoolap.h` |
| Windows | `target/release/stoolap.dll` | `include/stoolap.h` |

## Linking

```bash
# Compile and link
cc -O2 -o myapp myapp.c -I include -L target/release -lstoolap

# Run (set library path)
# macOS:
DYLD_LIBRARY_PATH=target/release ./myapp
# Linux:
LD_LIBRARY_PATH=target/release ./myapp
```

For system-wide installation, copy the shared library to `/usr/local/lib` and the header to `/usr/local/include`, then run `ldconfig` (Linux) or no extra step (macOS).

## Quick Start

```c
#include "stoolap.h"
#include <stdio.h>

int main(void) {
    StoolapDB* db = NULL;
    StoolapRows* rows = NULL;

    /* Open an in-memory database */
    if (stoolap_open_in_memory(&db) != STOOLAP_OK) {
        fprintf(stderr, "Open failed: %s\n", stoolap_errmsg(NULL));
        return 1;
    }

    /* Create a table and insert data */
    stoolap_exec(db, "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)", NULL);
    stoolap_exec(db, "INSERT INTO users VALUES (1, 'Alice', 30), (2, 'Bob', 25)", NULL);

    /* Query */
    if (stoolap_query(db, "SELECT id, name, age FROM users ORDER BY id", &rows) != STOOLAP_OK) {
        fprintf(stderr, "Query failed: %s\n", stoolap_errmsg(db));
        stoolap_close(db);
        return 1;
    }

    /* Iterate rows */
    while (stoolap_rows_next(rows) == STOOLAP_ROW) {
        int64_t id  = stoolap_rows_column_int64(rows, 0);
        const char* name = stoolap_rows_column_text(rows, 1, NULL);
        int64_t age = stoolap_rows_column_int64(rows, 2);
        printf("id=%lld name=%s age=%lld\n", (long long)id, name, (long long)age);
    }

    stoolap_rows_close(rows);
    stoolap_close(db);
    return 0;
}
```

## Handles and Ownership

The API uses five opaque handle types. Each handle is created by a specific function and must be freed by its corresponding cleanup function.

| Handle | Created by | Freed by | Notes |
|--------|-----------|----------|-------|
| `StoolapDB` | `stoolap_open`, `stoolap_open_in_memory`, `stoolap_clone` | `stoolap_close` | NULL-safe close (no-op) |
| `StoolapRoDB` | `stoolap_open_read_only` | `stoolap_ro_close` | NULL-safe close. Read-only handle: no `_exec` / `_begin` entry points exist, so write SQL through this handle is a link error on the C side. |
| `StoolapStmt` | `stoolap_prepare` | `stoolap_stmt_finalize` | NULL-safe finalize (no-op). Keeps the database engine alive. |
| `StoolapTx` | `stoolap_begin`, `stoolap_begin_with_isolation` | `stoolap_tx_commit` or `stoolap_tx_rollback` | Handle consumed on commit/rollback regardless of success or failure. Keeps the database engine alive. |
| `StoolapRows` | `stoolap_query*`, `stoolap_ro_query*`, `stoolap_stmt_query`, `stoolap_tx_query*`, `stoolap_tx_stmt_query*` | `stoolap_rows_close` | Must be closed even after `STOOLAP_DONE`. NULL-safe close (no-op) |

**Important**: `stoolap_tx_commit()` and `stoolap_tx_rollback()` free the transaction handle whether they succeed or fail. After calling either function, the `StoolapTx` pointer is invalid.

## Status Codes

| Constant | Value | Meaning |
|----------|:-----:|---------|
| `STOOLAP_OK` | 0 | Operation succeeded |
| `STOOLAP_ERROR` | 1 | Operation failed. Call the appropriate `*_errmsg()` for details |
| `STOOLAP_ROW` | 100 | `stoolap_rows_next()`: a row is available for reading |
| `STOOLAP_DONE` | 101 | `stoolap_rows_next()`: no more rows |

## Opening a Database

```c
StoolapDB* db = NULL;

/* In-memory (unique, isolated instance) */
stoolap_open_in_memory(&db);

/* In-memory via DSN */
stoolap_open("memory://", &db);

/* Named in-memory (same name shares the engine) */
stoolap_open("memory://mydb", &db);

/* File-based (persistent) */
stoolap_open("file:///path/to/mydb", &db);

/* File-based with configuration */
stoolap_open("file:///path/to/mydb?sync_mode=full&compression=on", &db);
```

See [Connection String Reference]({{ '/docs/getting-started/connection-strings/' | relative_url }}) for all configuration options.

Always check the return code. On failure, `*db` is set to NULL and the error is available via `stoolap_errmsg(NULL)`:

```c
StoolapDB* db = NULL;
if (stoolap_open("file:///nonexistent/path", &db) != STOOLAP_OK) {
    fprintf(stderr, "Failed to open: %s\n", stoolap_errmsg(NULL));
}
```

When done, close the handle:

```c
stoolap_close(db);  /* safe to call with NULL */
```

## Executing SQL

Use `stoolap_exec` for DDL and DML statements that do not return rows.

```c
/* Without parameters */
int64_t affected = 0;
stoolap_exec(db, "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)", NULL);
stoolap_exec(db, "INSERT INTO t VALUES (1, 'Alice'), (2, 'Bob')", &affected);
/* affected == 2 */

/* With positional parameters ($1, $2, ...) */
StoolapValue params[2] = {
    { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = 3 } },
    { .value_type = STOOLAP_TYPE_TEXT,    0, { .text = { "Charlie", 7 } } },
};
stoolap_exec_params(db, "INSERT INTO t VALUES ($1, $2)", params, 2, &affected);
/* affected == 1 */
```

The `rows_affected` pointer may be NULL if you do not need the count.

## Querying Data

Use `stoolap_query` for SELECT statements. Iterate the result set with `stoolap_rows_next()`, then close it.

```c
StoolapRows* rows = NULL;
int32_t rc = stoolap_query(db, "SELECT id, name FROM t ORDER BY id", &rows);
if (rc != STOOLAP_OK) {
    fprintf(stderr, "Query error: %s\n", stoolap_errmsg(db));
    /* rows is NULL on error, no need to close */
}

while (stoolap_rows_next(rows) == STOOLAP_ROW) {
    int64_t id = stoolap_rows_column_int64(rows, 0);
    const char* name = stoolap_rows_column_text(rows, 1, NULL);
    printf("%lld: %s\n", (long long)id, name);
}

/* Must always close, even after STOOLAP_DONE */
stoolap_rows_close(rows);
```

With parameters:

```c
StoolapValue p = { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = 1 } };
StoolapRows* rows = NULL;
stoolap_query_params(db, "SELECT name FROM t WHERE id = $1", &p, 1, &rows);
```

## Table Row Count

`stoolap_table_count()` returns the number of rows visible to the current autocommit handle without parsing or executing SQL. It uses the SegmentedTable fast path so hot rows in memory and sealed cold volumes are both counted, with a single atomic load when the snapshot-isolation fallback is not required.

```c
uint64_t count = 0;
if (stoolap_table_count(db, "users", &count) == STOOLAP_OK) {
    printf("users: %llu rows\n", (unsigned long long)count);
}
```

Inside an explicit transaction, use `stoolap_tx_table_count()` instead. It returns the count visible to that transaction, including uncommitted local INSERTs and DELETEs done in the same transaction:

```c
StoolapTx* tx = NULL;
stoolap_begin(db, &tx);
stoolap_tx_exec(tx, "INSERT INTO users VALUES (99, 'Tx-only')", NULL);

uint64_t in_tx = 0;
stoolap_tx_table_count(tx, "users", &in_tx);    /* sees the new row */

uint64_t outside = 0;
stoolap_table_count(db, "users", &outside);     /* does not see it (autocommit) */

stoolap_tx_rollback(tx);
```

Both functions return `STOOLAP_ERROR` and record `STOOLAP_ERR_TABLE_NOT_FOUND` on a missing table.

## Column Access

After a successful `stoolap_rows_next()` call, read column values by 0-based index:

| Function | Returns | Default on NULL |
|----------|---------|:---------------:|
| `stoolap_rows_column_int64(rows, i)` | `int64_t` | 0 |
| `stoolap_rows_column_double(rows, i)` | `double` | 0.0 |
| `stoolap_rows_column_text(rows, i, &len)` | `const char*` (len = full byte length) | NULL |
| `stoolap_rows_column_bool(rows, i)` | `int32_t` | 0 |
| `stoolap_rows_column_timestamp(rows, i)` | `int64_t` (nanos since epoch) | 0 |
| `stoolap_rows_column_blob(rows, i, &len)` | `const uint8_t*` (VECTOR only, packed f32) | NULL |
| `stoolap_rows_column_is_null(rows, i)` | `int32_t` (1=NULL, 0=not) | 1 |
| `stoolap_rows_column_type(rows, i)` | `int32_t` (`STOOLAP_TYPE_*`) | `STOOLAP_TYPE_NULL` |

Metadata (available before first `stoolap_rows_next()`):

| Function | Returns |
|----------|---------|
| `stoolap_rows_column_count(rows)` | Number of columns |
| `stoolap_rows_column_name(rows, i)` | Column name (NULL if out of bounds) |
| `stoolap_rows_affected(rows)` | Rows affected (for DML results) |

### Pointer Lifetimes

| Pointer from | Valid until |
|-------------|-------------|
| `stoolap_rows_column_text()` | Next `stoolap_rows_next()` call |
| `stoolap_rows_column_blob()` | Next `stoolap_rows_next()` call |
| `stoolap_rows_column_name()` | `stoolap_rows_close()` |
| `stoolap_errmsg()` | Next API call on the same handle |
| `stoolap_stmt_sql()` | `stoolap_stmt_finalize()` |
| `stoolap_version()` | Forever (static) |

Do **not** free any of these pointers. They are managed by their parent handle.

## Prepared Statements

Prepared statements parse SQL once and reuse the cached plan. This avoids parse overhead when executing the same SQL repeatedly with different parameters.

A prepared statement keeps the underlying database engine alive. You can safely close the originating `StoolapDB` handle before finalizing the statement. The engine resources are released only when all statements (and other handles) referencing it have been finalized or closed.

```c
/* Prepare */
StoolapStmt* stmt = NULL;
stoolap_prepare(db, "INSERT INTO t VALUES ($1, $2)", &stmt);

/* Execute repeatedly */
for (int i = 100; i < 110; i++) {
    char name[32];
    snprintf(name, sizeof(name), "User_%d", i);
    StoolapValue params[2] = {
        { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = i } },
        { .value_type = STOOLAP_TYPE_TEXT,    0, { .text = { name, strlen(name) } } },
    };
    stoolap_stmt_exec(stmt, params, 2, NULL);
}

/* Finalize when done */
stoolap_stmt_finalize(stmt);
```

Prepared queries work the same way:

```c
StoolapStmt* lookup = NULL;
stoolap_prepare(db, "SELECT name FROM t WHERE id = $1", &lookup);

StoolapValue p = { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = 100 } };
StoolapRows* rows = NULL;
stoolap_stmt_query(lookup, &p, 1, &rows);

if (stoolap_rows_next(rows) == STOOLAP_ROW) {
    printf("Name: %s\n", stoolap_rows_column_text(rows, 0, NULL));
}
stoolap_rows_close(rows);

/* Retrieve the SQL text */
printf("SQL: %s\n", stoolap_stmt_sql(lookup));

stoolap_stmt_finalize(lookup);
```

### Batch Execution

`stoolap_stmt_exec_batch()` executes a prepared statement once per parameter row inside a single transaction. It replaces `2 + N` FFI calls (begin + N executions + commit) with a single call, reducing per-call overhead for bulk inserts and updates.

Parameters are passed as a flat row-major array: all values for row 0, then all values for row 1, and so on.

```c
StoolapStmt* stmt = NULL;
stoolap_prepare(db, "INSERT INTO t VALUES ($1, $2)", &stmt);

/* 3 rows, 2 params each = 6 StoolapValue structs */
StoolapValue params[6] = {
    /* Row 0 */
    { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = 1 } },
    { .value_type = STOOLAP_TYPE_TEXT,    0, { .text = { "Alice", 5 } } },
    /* Row 1 */
    { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = 2 } },
    { .value_type = STOOLAP_TYPE_TEXT,    0, { .text = { "Bob", 3 } } },
    /* Row 2 */
    { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = 3 } },
    { .value_type = STOOLAP_TYPE_TEXT,    0, { .text = { "Charlie", 7 } } },
};

int64_t total = 0;
if (stoolap_stmt_exec_batch(db, stmt, params, 2, 3, &total) != STOOLAP_OK) {
    fprintf(stderr, "Batch failed: %s\n", stoolap_errmsg(db));
}
/* total == 3 */

stoolap_stmt_finalize(stmt);
```

On success, all rows are committed atomically. On any error, the entire batch is rolled back and the error is available via `stoolap_errmsg(db)`.

The `total_affected` pointer may be NULL if you do not need the count. When `row_count` is 0, the function returns `STOOLAP_OK` immediately without opening a transaction.

### Statement Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `stoolap_prepare(db, sql, &stmt)` | `int32_t` | Prepare a SQL statement |
| `stoolap_stmt_exec(stmt, params, len, &affected)` | `int32_t` | Execute with parameters |
| `stoolap_stmt_exec_batch(db, stmt, params, params_per_row, row_count, &total)` | `int32_t` | Execute batch in a single transaction |
| `stoolap_stmt_query(stmt, params, len, &rows)` | `int32_t` | Query with parameters |
| `stoolap_stmt_sql(stmt)` | `const char*` | Get the SQL text |
| `stoolap_stmt_finalize(stmt)` | `void` | Destroy the statement (NULL-safe) |
| `stoolap_stmt_errmsg(stmt)` | `const char*` | Last error message |

## Transactions

Like prepared statements, a transaction keeps the underlying database engine alive. You can safely close the originating `StoolapDB` handle while a transaction is still open. The engine resources are released only after the transaction is committed or rolled back (and all other handles are closed).

### Default Isolation (Read Committed)

```c
StoolapTx* tx = NULL;
stoolap_begin(db, &tx);

stoolap_tx_exec(tx, "INSERT INTO t VALUES (10, 'In Transaction')", NULL);

/* Query within the transaction */
StoolapRows* rows = NULL;
stoolap_tx_query(tx, "SELECT * FROM t WHERE id = 10", &rows);
if (stoolap_rows_next(rows) == STOOLAP_ROW) {
    printf("Name: %s\n", stoolap_rows_column_text(rows, 1, NULL));
}
stoolap_rows_close(rows);

/* Commit (frees the tx handle) */
int32_t rc = stoolap_tx_commit(tx);
if (rc != STOOLAP_OK) {
    fprintf(stderr, "Commit failed: %s\n", stoolap_errmsg(NULL));
}
/* tx is now invalid, do not use it */
```

### Snapshot Isolation

```c
StoolapTx* tx = NULL;
stoolap_begin_with_isolation(db, STOOLAP_ISOLATION_SNAPSHOT, &tx);
/* ... operations ... */
stoolap_tx_commit(tx);
```

### Rollback

```c
StoolapTx* tx = NULL;
stoolap_begin(db, &tx);
stoolap_tx_exec(tx, "DELETE FROM t", NULL);
stoolap_tx_rollback(tx);  /* changes discarded, tx handle freed */
```

### Transaction with Parameters

```c
StoolapTx* tx = NULL;
stoolap_begin(db, &tx);

StoolapValue params[2] = {
    { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = 20 } },
    { .value_type = STOOLAP_TYPE_TEXT,    0, { .text = { "TxUser", 6 } } },
};
stoolap_tx_exec_params(tx, "INSERT INTO t VALUES ($1, $2)", params, 2, NULL);

stoolap_tx_commit(tx);
```

### Prepared Statements in Transactions

Use `stoolap_tx_stmt_exec()` and `stoolap_tx_stmt_query()` to execute a prepared statement within a transaction. This gives both parse-once performance and transactional atomicity (all-or-nothing commit/rollback).

**Important**: Do not use `stoolap_stmt_exec()` inside a transaction block. It creates its own standalone auto-committing transaction per call, so rollback will not undo those operations.

```c
StoolapStmt* stmt = NULL;
stoolap_prepare(db, "INSERT INTO orders VALUES ($1, $2, $3)", &stmt);

StoolapTx* tx = NULL;
stoolap_begin(db, &tx);

for (int i = 0; i < 1000; i++) {
    StoolapValue params[3] = {
        { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = i } },
        { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = 1 } },
        { .value_type = STOOLAP_TYPE_FLOAT,   0, { .float64 = 99.99 } },
    };
    stoolap_tx_stmt_exec(tx, stmt, params, 3, NULL);
}

stoolap_tx_commit(tx);   /* all 1000 rows committed atomically */
stoolap_stmt_finalize(stmt);
```

Queries work the same way:

```c
StoolapStmt* lookup = NULL;
stoolap_prepare(db, "SELECT name FROM users WHERE id = $1", &lookup);

StoolapTx* tx = NULL;
stoolap_begin(db, &tx);

StoolapValue p = { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = 42 } };
StoolapRows* rows = NULL;
stoolap_tx_stmt_query(tx, lookup, &p, 1, &rows);
/* ... iterate rows ... */
stoolap_rows_close(rows);

stoolap_tx_commit(tx);
stoolap_stmt_finalize(lookup);
```

### Savepoints

Savepoints record a point inside an open transaction so you can later discard work done after that point without rolling back the entire transaction. The three operations follow the SQL standard: create, release (forget the savepoint, keep the work), and rollback to (undo work done after the savepoint).

```c
StoolapTx* tx = NULL;
stoolap_begin(db, &tx);

stoolap_tx_exec(tx, "INSERT INTO orders VALUES (1, 100)", NULL);

stoolap_tx_savepoint(tx, "before_items", -1);
stoolap_tx_exec(tx, "INSERT INTO order_items VALUES (1, 'A')", NULL);
stoolap_tx_exec(tx, "INSERT INTO order_items VALUES (1, 'B')", NULL);

/* Something is wrong: undo the items, keep the order. */
stoolap_tx_rollback_to_savepoint(tx, "before_items", -1);

stoolap_tx_commit(tx);   /* the order is committed; items are not */
```

The `name_len` argument is the byte length of the savepoint name. Pass `-1` to treat `name` as a NUL-terminated C string, or pass an explicit positive length when interoperating with non-NUL-terminated buffers (for example, MariaDB handlerton savepoint chunks).

```c
const char* raw = "sp_aBOGUS";    /* explicit length clips the trailing junk */
stoolap_tx_savepoint(tx, raw, 4);
stoolap_tx_release_savepoint(tx, raw, 4);
```

Re-using an existing savepoint name overwrites it. `stoolap_tx_release_savepoint()` and `stoolap_tx_rollback_to_savepoint()` return `STOOLAP_ERROR` on an unknown name; the typed code is `STOOLAP_ERR_INVALID_ARGUMENT` and the message identifies the missing savepoint.

### Transaction Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `stoolap_begin(db, &tx)` | `int32_t` | Begin with READ COMMITTED |
| `stoolap_begin_with_isolation(db, level, &tx)` | `int32_t` | Begin with specific isolation |
| `stoolap_tx_exec(tx, sql, &affected)` | `int32_t` | Execute without params |
| `stoolap_tx_exec_params(tx, sql, params, len, &affected)` | `int32_t` | Execute with params |
| `stoolap_tx_query(tx, sql, &rows)` | `int32_t` | Query without params |
| `stoolap_tx_query_params(tx, sql, params, len, &rows)` | `int32_t` | Query with params |
| `stoolap_tx_stmt_exec(tx, stmt, params, len, &affected)` | `int32_t` | Execute prepared statement in transaction |
| `stoolap_tx_stmt_query(tx, stmt, params, len, &rows)` | `int32_t` | Query with prepared statement in transaction |
| `stoolap_tx_table_count(tx, table, &count)` | `int32_t` | Snapshot-correct row count visible to this tx |
| `stoolap_tx_savepoint(tx, name, name_len)` | `int32_t` | Create or overwrite a savepoint |
| `stoolap_tx_release_savepoint(tx, name, name_len)` | `int32_t` | Forget a savepoint, keep the work |
| `stoolap_tx_rollback_to_savepoint(tx, name, name_len)` | `int32_t` | Undo work done after the savepoint |
| `stoolap_tx_commit(tx)` | `int32_t` | Commit and free handle |
| `stoolap_tx_rollback(tx)` | `int32_t` | Rollback and free handle |
| `stoolap_tx_errmsg(tx)` | `const char*` | Last error message |
| `stoolap_tx_errcode(tx)` | `int32_t` | Last error code (`STOOLAP_ERR_*`) |
| `stoolap_tx_errdetails(tx, &out)` | `int32_t` | Fill structured error details |

**Important**: After `stoolap_tx_commit()` or `stoolap_tx_rollback()`, the `tx` pointer is invalid regardless of the return code. On commit/rollback failure, retrieve the error with `stoolap_errmsg(NULL)` (the global thread-local error).

## Parameters (StoolapValue)

Pass parameters to SQL statements using an array of `StoolapValue` structs. Each value is a tagged union:

```c
typedef struct StoolapValue {
    int32_t value_type;   /* STOOLAP_TYPE_* constant */
    int32_t _padding;     /* must be 0 */
    union {
        int64_t  integer;
        double   float64;
        int32_t  boolean;
        struct { const char*    ptr; int64_t len; } text;
        struct { const uint8_t* ptr; int64_t len; } blob;
        int64_t  timestamp_nanos;
    } v;
} StoolapValue;
```

### Constructing Values

```c
/* NULL */
StoolapValue v_null = { .value_type = STOOLAP_TYPE_NULL, 0, { .integer = 0 } };

/* Integer */
StoolapValue v_int = { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = 42 } };

/* Float */
StoolapValue v_float = { .value_type = STOOLAP_TYPE_FLOAT, 0, { .float64 = 3.14 } };

/* Text (pointer + byte length, does not need null termination) */
const char* name = "Alice";
StoolapValue v_text = { .value_type = STOOLAP_TYPE_TEXT, 0, { .text = { name, 5 } } };

/* Boolean (0 = false, non-zero = true) */
StoolapValue v_bool = { .value_type = STOOLAP_TYPE_BOOLEAN, 0, { .boolean = 1 } };

/* Timestamp (nanoseconds since Unix epoch, UTC) */
StoolapValue v_ts = { .value_type = STOOLAP_TYPE_TIMESTAMP, 0, { .timestamp_nanos = 1705312200000000000LL } };

/* JSON (pointer + byte length, valid JSON text) */
const char* json = "{\"key\": \"value\"}";
StoolapValue v_json = { .value_type = STOOLAP_TYPE_JSON, 0, { .text = { json, 16 } } };

/* BLOB / Vector (packed little-endian f32 bytes, length must be a multiple of 4) */
float vec[] = { 1.0f, 2.0f, 3.0f };
StoolapValue v_blob = { .value_type = STOOLAP_TYPE_BLOB, 0, { .blob = { (const uint8_t*)vec, sizeof(vec) } } };
```

### Named Parameters (StoolapNamedParam)

Named parameters use `:name` syntax in SQL. Pass an array of `StoolapNamedParam` structs instead of positional `StoolapValue` arrays:

```c
typedef struct StoolapNamedParam {
    const char* name;       /* Parameter name (without ':' prefix) */
    int32_t     name_len;   /* Length of name in bytes */
    int32_t     _padding;   /* must be 0 */
    StoolapValue value;     /* Parameter value */
} StoolapNamedParam;
```

```c
/* Execute with named parameters */
StoolapNamedParam params[2] = {
    { "id",   2, 0, { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = 1 } } },
    { "name", 4, 0, { .value_type = STOOLAP_TYPE_TEXT,    0, { .text = { "Alice", 5 } } } },
};
stoolap_exec_named(db, "INSERT INTO t VALUES (:id, :name)", params, 2, NULL);

/* Query with named parameters */
StoolapNamedParam qp[1] = {
    { "id", 2, 0, { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = 1 } } },
};
StoolapRows* rows = NULL;
stoolap_query_named(db, "SELECT name FROM t WHERE id = :id", qp, 1, &rows);
/* ... iterate rows ... */
stoolap_rows_close(rows);
```

Named parameters also work within transactions and with prepared statements:

```c
/* Transaction with named params */
StoolapTx* tx = NULL;
stoolap_begin(db, &tx);
stoolap_tx_exec_named(tx, "INSERT INTO t VALUES (:id, :name)", params, 2, NULL);
stoolap_tx_commit(tx);

/* Prepared statement + named params in a transaction */
StoolapStmt* stmt = NULL;
stoolap_prepare(db, "INSERT INTO t VALUES (:id, :name)", &stmt);

StoolapTx* tx2 = NULL;
stoolap_begin(db, &tx2);
for (int i = 0; i < 100; i++) {
    char name[32];
    snprintf(name, sizeof(name), "User_%d", i);
    StoolapNamedParam p[2] = {
        { "id",   2, 0, { .value_type = STOOLAP_TYPE_INTEGER, 0, { .integer = i } } },
        { "name", 4, 0, { .value_type = STOOLAP_TYPE_TEXT,    0, { .text = { name, strlen(name) } } } },
    };
    stoolap_tx_stmt_exec_named(tx2, stmt, p, 2, NULL);
}
stoolap_tx_commit(tx2);
stoolap_stmt_finalize(stmt);
```

The parameter name must be valid UTF-8 and match the `:name` placeholder in the SQL (without the `:` prefix). Names with invalid UTF-8 or zero length are silently skipped.

Text and JSON parameters do not need to be null-terminated. The `len` field specifies the byte length. Both must be valid UTF-8.

JSON parameters are validated with a full parse. Malformed JSON (including empty strings) is rejected and treated as NULL.

BLOB parameters must be packed little-endian f32 data. The byte length must be a multiple of 4. Non-conforming payloads are treated as NULL.

## Error Handling

Each handle type has its own error message function. The error message is valid until the next API call on the same handle.

```c
/* Database handle errors */
if (stoolap_exec(db, "INVALID SQL", NULL) != STOOLAP_OK) {
    fprintf(stderr, "Error: %s\n", stoolap_errmsg(db));
}

/* Statement handle errors */
if (stoolap_stmt_exec(stmt, params, len, NULL) != STOOLAP_OK) {
    fprintf(stderr, "Error: %s\n", stoolap_stmt_errmsg(stmt));
}

/* Transaction handle errors (before commit/rollback) */
if (stoolap_tx_exec(tx, sql, NULL) != STOOLAP_OK) {
    fprintf(stderr, "Error: %s\n", stoolap_tx_errmsg(tx));
}

/* Global error (for stoolap_open failures, or after commit/rollback) */
if (stoolap_open("bad://dsn", &db) != STOOLAP_OK) {
    fprintf(stderr, "Error: %s\n", stoolap_errmsg(NULL));
}
```

| Function | Use when |
|----------|----------|
| `stoolap_errmsg(db)` | After `stoolap_exec*`, `stoolap_query*`, `stoolap_prepare` fail |
| `stoolap_errmsg(NULL)` | After `stoolap_open*` fails, or after `stoolap_tx_commit`/`stoolap_tx_rollback` fails |
| `stoolap_stmt_errmsg(stmt)` | After `stoolap_stmt_exec`, `stoolap_stmt_query` fail |
| `stoolap_tx_errmsg(tx)` | After `stoolap_tx_exec*`, `stoolap_tx_query*`, `stoolap_tx_stmt_*_named` fail |
| `stoolap_rows_errmsg(rows)` | After `stoolap_rows_next` returns `STOOLAP_ERROR` |

If no error has occurred, all `*_errmsg()` functions return an empty string (`""`), never NULL.

### Typed Error Codes

For programmatic error handling, every handle also exposes a typed error code and a structured detail struct. Use the code to branch (for example, retry on `STOOLAP_ERR_DB_LOCKED`, surface the conflicting column for `STOOLAP_ERR_UNIQUE`) instead of grepping the message text.

```c
int32_t rc = stoolap_exec(db, "INSERT INTO users VALUES (1, 'a@x.com')", NULL);
if (rc == STOOLAP_ERROR) {
    switch (stoolap_errcode(db)) {
        case STOOLAP_ERR_UNIQUE: {
            StoolapErrorDetails det = {0};
            stoolap_errdetails(db, &det);
            /* det.column = "email", det.constraint = "<index name>",
               det.detail = the conflicting value, det.message = full text. */
            fprintf(stderr, "duplicate %s: %s\n", det.column, det.detail);
            break;
        }
        case STOOLAP_ERR_DB_LOCKED:
            /* retry with backoff */
            break;
        default:
            fprintf(stderr, "%s\n", stoolap_errmsg(db));
    }
}
```

`stoolap_errdetails()` fills the caller's struct; the `const char*` fields are valid until the next API call on the same handle. `message` is never NULL (empty string on success). All other pointer fields are NULL when the field does not apply to this error code.

```c
typedef struct StoolapErrorDetails {
    int32_t code;          /* one of STOOLAP_ERR_* */
    int32_t _padding;
    const char* message;   /* never NULL: empty string on success */
    const char* table;     /* table name, or NULL */
    const char* column;    /* column name, or NULL */
    const char* constraint;/* index name (UNIQUE) or referenced table (FK), or NULL */
    const char* detail;    /* free-form: conflicting value, CHECK expr, FK detail */
} StoolapErrorDetails;
```

The same surface is available on every handle:

| Function | Use when |
|----------|----------|
| `stoolap_errcode(db)` / `stoolap_errdetails(db, &out)` | After `stoolap_exec*`, `stoolap_query*`, `stoolap_prepare`, `stoolap_table_count` fail |
| `stoolap_tx_errcode(tx)` / `stoolap_tx_errdetails(tx, &out)` | After `stoolap_tx_exec*`, `stoolap_tx_query*`, `stoolap_tx_table_count`, `stoolap_tx_savepoint*` fail |
| `stoolap_stmt_errcode(stmt)` / `stoolap_stmt_errdetails(stmt, &out)` | After `stoolap_stmt_exec`, `stoolap_stmt_query` fail |
| `stoolap_rows_errcode(rows)` / `stoolap_rows_errdetails(rows, &out)` | After `stoolap_rows_next` returns `STOOLAP_ERROR` |
| `stoolap_ro_errcode(ro)` / `stoolap_ro_errdetails(ro, &out)` | After `stoolap_ro_*` calls fail |
| `stoolap_errcode(NULL)` / `stoolap_errdetails(NULL, &out)` | After `stoolap_open*`, `stoolap_open_read_only`, or `stoolap_tx_commit`/`rollback` fail (thread-local fallback) |

Codes are appended-only and stable across releases. Plugins should default to generic handling for unknown codes so future additions do not require a recompile.

| Constant | Value | Meaning |
|----------|:-----:|---------|
| `STOOLAP_ERR_OK` | 0 | No error |
| `STOOLAP_ERR_GENERIC` | 1 | Unclassified error (use the message) |
| `STOOLAP_ERR_NOT_NULL` | 2 | NOT NULL constraint violated; `column` set |
| `STOOLAP_ERR_UNIQUE` | 3 | UNIQUE violated; `column`, `constraint`, `detail` set |
| `STOOLAP_ERR_PRIMARY_KEY` | 4 | PRIMARY KEY violated |
| `STOOLAP_ERR_FOREIGN_KEY` | 5 | FK violated; `table`, `column`, `constraint` (= referenced table), `detail` set |
| `STOOLAP_ERR_CHECK` | 6 | CHECK constraint violated; `column`, `detail` (= expression) set |
| `STOOLAP_ERR_TABLE_NOT_FOUND` | 7 | `table` set |
| `STOOLAP_ERR_TABLE_EXISTS` | 8 | `table` set |
| `STOOLAP_ERR_COLUMN_NOT_FOUND` | 9 | `column` set |
| `STOOLAP_ERR_INDEX_NOT_FOUND` | 10 | `constraint` (= index name) set |
| `STOOLAP_ERR_INDEX_EXISTS` | 11 | `constraint` set |
| `STOOLAP_ERR_TYPE_MISMATCH` | 12 | Type conversion or comparison error |
| `STOOLAP_ERR_INVALID_ARGUMENT` | 13 | Bad parameter, malformed savepoint name, etc. |
| `STOOLAP_ERR_PARSE` | 14 | SQL parse error |
| `STOOLAP_ERR_TX_ABORTED` | 15 | Transaction was aborted by the engine |
| `STOOLAP_ERR_TX_CLOSED` | 16 | Transaction not started, already ended, or closed |
| `STOOLAP_ERR_READ_ONLY` | 17 | Write rejected on a read-only handle or read-only mount |
| `STOOLAP_ERR_DB_LOCKED` | 18 | Another process holds the file lock |
| `STOOLAP_ERR_IO` | 19 | I/O failure |
| `STOOLAP_ERR_NOT_SUPPORTED` | 20 | Operation not supported |
| `STOOLAP_ERR_INTERNAL` | 21 | Internal invariant failure (file a bug) |
| `STOOLAP_ERR_QUERY_CANCELLED` | 22 | Query cancelled |
| `STOOLAP_ERR_DIVISION_BY_ZERO` | 23 | Division by zero in expression |
| `STOOLAP_ERR_VALUE_TOO_LONG` | 24 | Value exceeds column length limit; `column` set |
| `STOOLAP_ERR_VIEW_NOT_FOUND` | 25 | `table` (= view name) set |
| `STOOLAP_ERR_VIEW_EXISTS` | 26 | `table` (= view name) set |
| `STOOLAP_ERR_REOPEN_REQUIRED` | 27 | SWMR reader: cached state stale; close and reopen the handle |

## Thread Safety

A single `StoolapDB` handle must not be used from multiple threads simultaneously. For multi-threaded use, clone the handle with `stoolap_clone()`. Each clone shares the underlying engine (data, indexes, transactions) but has its own executor and error state.

```c
StoolapDB* db = NULL;
stoolap_open("memory://shared", &db);

stoolap_exec(db, "CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)", NULL);

/* Clone for each worker thread */
StoolapDB* thread_db = NULL;
stoolap_clone(db, &thread_db);

/* Pass thread_db to the worker thread */
/* ... use thread_db exclusively in that thread ... */

/* Each clone must be closed independently */
stoolap_close(thread_db);
stoolap_close(db);
```

### Thread Safety Rules

- **StoolapDB**: Do not share across threads. Use `stoolap_clone()` for per-thread handles.
- **StoolapStmt**: Do not use concurrently from multiple threads.
- **StoolapTx**: Must remain on the thread that created it.
- **StoolapRows**: Must remain on the thread that created it.

## Read-Only Handle

`stoolap_open_read_only()` returns a `StoolapRoDB*` whose surface is exclusively read-only. There are no `stoolap_ro_exec` or `stoolap_ro_begin` entry points, so write SQL routed through this handle is a link error on the C side rather than a runtime check. Write SQL passed to `stoolap_ro_query()` (which routes through the read engine) is rejected with `STOOLAP_ERR_READ_ONLY` at runtime.

Multiple processes can hold a read-only handle on the same database concurrently. A writable open is rejected while any reader is active, and vice versa.

```c
StoolapRoDB* ro = NULL;
if (stoolap_open_read_only("file:///data/mydb", &ro) != STOOLAP_OK) {
    fprintf(stderr, "open failed: %s\n", stoolap_errmsg(NULL));
    return 1;
}

uint64_t n = 0;
stoolap_ro_table_count(ro, "users", &n);
printf("users visible to this snapshot: %llu\n", (unsigned long long)n);

StoolapRows* rows = NULL;
stoolap_ro_query(ro, "SELECT id, name FROM users WHERE id < 100", &rows);
while (stoolap_rows_next(rows) == STOOLAP_ROW) {
    /* ... read columns ... */
}
stoolap_rows_close(rows);

stoolap_ro_close(ro);
```

### DSN Flag Routing

`stoolap_open()` REJECTS the read-only DSN flags (`?read_only=true`, `?readonly=true`, `?mode=ro`) with `STOOLAP_ERR_INVALID_ARGUMENT` and a message pointing the caller to `stoolap_open_read_only`. The intent is to surface the type-system mismatch loudly rather than silently downgrade to a writable handle.

`stoolap_open_read_only()` accepts those flags as redundant no-ops, so existing driver DSN strings continue to work unchanged:

```c
StoolapRoDB* ro = NULL;
stoolap_open_read_only("file:///data/mydb",                &ro);  /* preferred */
stoolap_open_read_only("file:///data/mydb?read_only=1",    &ro);  /* also fine */
stoolap_open_read_only("file:///data/mydb?mode=ro",        &ro);  /* also fine */
```

In-memory DSNs (`memory://`) are not supported for `stoolap_open_read_only` (there is no on-disk state to share). Read-only opens against directories on read-only mounts and chmod-read-only directories are supported; the engine acquires a long-lived shared lock to prevent a privileged writer from reclaiming files under the reader.

### Cross-Process Visibility

By default each `stoolap_ro_query*` call polls the on-disk manifest epoch (one 8-byte read) and, if the writer has advanced, refreshes manifests before executing. This costs roughly 1 microsecond when nothing has changed; the manifest reload only fires after a writer checkpoint.

For stable visibility across multiple queries (for example, inside an application-level "report" block) disable auto-refresh, run the queries, then re-enable:

```c
stoolap_ro_set_auto_refresh(ro, 0);
/* ... run a series of queries against a stable snapshot ... */
stoolap_ro_set_auto_refresh(ro, 1);
```

Call `stoolap_ro_refresh()` to advance manually. It returns `1` if the snapshot moved, `0` if it was already current, or `STOOLAP_ERROR` on a must-reopen condition. The latter surfaces with the typed code `STOOLAP_ERR_REOPEN_REQUIRED` and indicates the caller MUST close this handle and call `stoolap_open_read_only` again. Causes include the writer process being replaced, a checkpoint truncating the WAL window the reader was tailing, or DDL that the read-only engine cannot replay live.

```c
int32_t r = stoolap_ro_refresh(ro);
if (r == STOOLAP_ERROR && stoolap_ro_errcode(ro) == STOOLAP_ERR_REOPEN_REQUIRED) {
    fprintf(stderr, "snapshot expired: %s\n", stoolap_ro_errmsg(ro));
    stoolap_ro_close(ro);
    stoolap_open_read_only(dsn, &ro);
}
```

### Read-Only Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `stoolap_open_read_only(dsn, &ro)` | `int32_t` | Open a read-only handle |
| `stoolap_ro_close(ro)` | `void` | Close and free (NULL-safe) |
| `stoolap_ro_query(ro, sql, &rows)` | `int32_t` | Query without parameters |
| `stoolap_ro_query_params(ro, sql, params, len, &rows)` | `int32_t` | Query with positional parameters |
| `stoolap_ro_query_named(ro, sql, params, len, &rows)` | `int32_t` | Query with named parameters |
| `stoolap_ro_table_exists(ro, name)` | `int32_t` | 1 if exists, 0 if not, -1 on error |
| `stoolap_ro_table_count(ro, table, &count)` | `int32_t` | Snapshot row count for `table` |
| `stoolap_ro_refresh(ro)` | `int32_t` | Advance to writer's latest visible state |
| `stoolap_ro_set_auto_refresh(ro, enabled)` | `void` | Toggle automatic refresh on every query |
| `stoolap_ro_dsn(ro)` | `const char*` | DSN string (cached, valid for handle lifetime) |
| `stoolap_ro_errmsg(ro)` | `const char*` | Last error message |
| `stoolap_ro_errcode(ro)` | `int32_t` | Last error code (`STOOLAP_ERR_*`) |
| `stoolap_ro_errdetails(ro, &out)` | `int32_t` | Fill structured error details |

## Type Mapping

| SQL Type | Type Constant | C Accessor | C Type |
|----------|:-------------:|------------|--------|
| `NULL` | `STOOLAP_TYPE_NULL` | `stoolap_rows_column_is_null()` | `int32_t` (1 or 0) |
| `INTEGER` | `STOOLAP_TYPE_INTEGER` | `stoolap_rows_column_int64()` | `int64_t` |
| `FLOAT` / `REAL` | `STOOLAP_TYPE_FLOAT` | `stoolap_rows_column_double()` | `double` |
| `TEXT` | `STOOLAP_TYPE_TEXT` | `stoolap_rows_column_text()` | `const char*` |
| `BOOLEAN` | `STOOLAP_TYPE_BOOLEAN` | `stoolap_rows_column_bool()` | `int32_t` (1 or 0) |
| `TIMESTAMP` | `STOOLAP_TYPE_TIMESTAMP` | `stoolap_rows_column_timestamp()` | `int64_t` (nanos since epoch) |
| `JSON` | `STOOLAP_TYPE_JSON` | `stoolap_rows_column_text()` | `const char*` (JSON string) |
| `VECTOR` | `STOOLAP_TYPE_BLOB` | `stoolap_rows_column_blob()` | `const uint8_t*` (packed little-endian f32) |

Any column can also be read as text via `stoolap_rows_column_text()`, which performs type coercion (integers, floats, booleans, timestamps, and vectors are converted to their string representation).

`stoolap_rows_column_blob()` only returns data for VECTOR columns. For JSON and other extension types it returns NULL. The returned bytes are the raw packed f32 payload without any internal headers.

**Interior NUL bytes**: `stoolap_rows_column_text()` always sets `out_len` to the full byte length of the value, which may exceed `strlen()` if the text contains embedded `\0` bytes. Callers using `out_len` can access the complete data. Callers treating the pointer as a C string will see a truncated view at the first `\0`.

## Memory Management

The C API is designed so that callers never need to free individual strings. All returned `const char*` pointers are managed by their parent handle and become invalid when the handle is closed or the next row is fetched.

`stoolap_string_free()` is provided for future use. Currently, no public API function returns a string that requires explicit freeing.

**Summary of rules:**

- Never free pointers returned by `stoolap_errmsg()`, `stoolap_rows_column_text()`, `stoolap_rows_column_name()`, `stoolap_stmt_sql()`, or `stoolap_version()`.
- Always call `stoolap_rows_close()` on every `StoolapRows` handle, even after `STOOLAP_DONE`.
- Always call `stoolap_stmt_finalize()` on every `StoolapStmt` handle.
- Always call `stoolap_close()` on every `StoolapDB` handle.
- Transaction handles are freed by `stoolap_tx_commit()` or `stoolap_tx_rollback()`.

## Bulk Fetch

`stoolap_rows_fetch_all()` consumes all remaining rows from a result set into a single packed binary buffer. This is useful for language bindings that need to transfer entire result sets across the FFI boundary in one call, avoiding per-row overhead.

```c
StoolapRows* rows = NULL;
stoolap_query(db, "SELECT id, name, age FROM users", &rows);

uint8_t* buf = NULL;
int64_t buf_len = 0;
if (stoolap_rows_fetch_all(rows, &buf, &buf_len) == STOOLAP_OK) {
    /* Parse the binary buffer (see format below) */
    /* ... */
    stoolap_buffer_free(buf, buf_len);
}
stoolap_rows_close(rows);
```

### Binary Format

The buffer layout is:

```
[column_count: u32 LE]
[for each column: name_len:u16 LE, name_bytes:u8[name_len]]
[row_count: u32 LE]
[for each row, for each column:
  type_tag: u8
  payload:
    NULL(0):      (empty)
    INTEGER(1):   i64 LE (8 bytes)
    FLOAT(2):     f64 LE (8 bytes)
    TEXT(3):      len:u32 LE + bytes
    BOOLEAN(4):   u8 (0 or 1)
    TIMESTAMP(5): i64 LE (8 bytes, nanos since epoch)
    JSON(6):      len:u32 LE + bytes
    BLOB(7):      len:u32 LE + bytes (packed f32 for vectors)
]
```

The caller must free the buffer with `stoolap_buffer_free(buf, buf_len)`. The rows handle must still be closed with `stoolap_rows_close()` after the fetch.

## API Reference

### Library

| Function | Returns | Description |
|----------|---------|-------------|
| `stoolap_version()` | `const char*` | Version string (static, never free) |

### Database Lifecycle

| Function | Returns | Description |
|----------|---------|-------------|
| `stoolap_open(dsn, &db)` | `int32_t` | Open by DSN string (REJECTS `?read_only=*` flags) |
| `stoolap_open_in_memory(&db)` | `int32_t` | Open a unique in-memory database |
| `stoolap_open_read_only(dsn, &ro)` | `int32_t` | Open a read-only handle (`StoolapRoDB*`) |
| `stoolap_clone(db, &out_db)` | `int32_t` | Clone handle for multi-threaded use |
| `stoolap_close(db)` | `int32_t` | Close and free (NULL-safe) |
| `stoolap_errmsg(db)` | `const char*` | Last error (pass NULL for global error) |
| `stoolap_errcode(db)` | `int32_t` | Last error code (`STOOLAP_ERR_*`; pass NULL for global) |
| `stoolap_errdetails(db, &out)` | `int32_t` | Fill structured error details (pass NULL for global) |

### Execute (DDL/DML)

| Function | Returns | Description |
|----------|---------|-------------|
| `stoolap_exec(db, sql, &affected)` | `int32_t` | Execute without parameters |
| `stoolap_exec_params(db, sql, params, len, &affected)` | `int32_t` | Execute with positional parameters |
| `stoolap_exec_named(db, sql, params, len, &affected)` | `int32_t` | Execute with named parameters |

### Query

| Function | Returns | Description |
|----------|---------|-------------|
| `stoolap_query(db, sql, &rows)` | `int32_t` | Query without parameters |
| `stoolap_query_params(db, sql, params, len, &rows)` | `int32_t` | Query with positional parameters |
| `stoolap_query_named(db, sql, params, len, &rows)` | `int32_t` | Query with named parameters |
| `stoolap_table_count(db, table, &count)` | `int32_t` | O(1) autocommit row count for `table` |

### Prepared Statements

| Function | Returns | Description |
|----------|---------|-------------|
| `stoolap_prepare(db, sql, &stmt)` | `int32_t` | Prepare a SQL statement |
| `stoolap_stmt_exec(stmt, params, len, &affected)` | `int32_t` | Execute prepared statement |
| `stoolap_stmt_exec_batch(db, stmt, params, params_per_row, row_count, &total)` | `int32_t` | Execute batch in a single transaction |
| `stoolap_stmt_query(stmt, params, len, &rows)` | `int32_t` | Query with prepared statement |
| `stoolap_stmt_sql(stmt)` | `const char*` | Get SQL text (valid until finalize) |
| `stoolap_stmt_finalize(stmt)` | `void` | Destroy statement (NULL-safe) |
| `stoolap_stmt_errmsg(stmt)` | `const char*` | Last error message |
| `stoolap_stmt_errcode(stmt)` | `int32_t` | Last error code (`STOOLAP_ERR_*`) |
| `stoolap_stmt_errdetails(stmt, &out)` | `int32_t` | Fill structured error details |

### Transactions

| Function | Returns | Description |
|----------|---------|-------------|
| `stoolap_begin(db, &tx)` | `int32_t` | Begin with READ COMMITTED |
| `stoolap_begin_with_isolation(db, level, &tx)` | `int32_t` | Begin with specific isolation level |
| `stoolap_tx_exec(tx, sql, &affected)` | `int32_t` | Execute in transaction |
| `stoolap_tx_exec_params(tx, sql, params, len, &affected)` | `int32_t` | Execute with positional parameters in transaction |
| `stoolap_tx_exec_named(tx, sql, params, len, &affected)` | `int32_t` | Execute with named parameters in transaction |
| `stoolap_tx_query(tx, sql, &rows)` | `int32_t` | Query in transaction |
| `stoolap_tx_query_params(tx, sql, params, len, &rows)` | `int32_t` | Query with positional parameters in transaction |
| `stoolap_tx_query_named(tx, sql, params, len, &rows)` | `int32_t` | Query with named parameters in transaction |
| `stoolap_tx_stmt_exec(tx, stmt, params, len, &affected)` | `int32_t` | Execute prepared statement in transaction |
| `stoolap_tx_stmt_exec_named(tx, stmt, params, len, &affected)` | `int32_t` | Execute prepared statement with named params in transaction |
| `stoolap_tx_stmt_query(tx, stmt, params, len, &rows)` | `int32_t` | Query with prepared statement in transaction |
| `stoolap_tx_stmt_query_named(tx, stmt, params, len, &rows)` | `int32_t` | Query with prepared statement and named params in transaction |
| `stoolap_tx_table_count(tx, table, &count)` | `int32_t` | Snapshot row count visible to this tx |
| `stoolap_tx_savepoint(tx, name, name_len)` | `int32_t` | Create or overwrite a savepoint |
| `stoolap_tx_release_savepoint(tx, name, name_len)` | `int32_t` | Forget a savepoint, keep the work |
| `stoolap_tx_rollback_to_savepoint(tx, name, name_len)` | `int32_t` | Undo work done after the savepoint |
| `stoolap_tx_commit(tx)` | `int32_t` | Commit and free handle |
| `stoolap_tx_rollback(tx)` | `int32_t` | Rollback and free handle |
| `stoolap_tx_errmsg(tx)` | `const char*` | Last error message |
| `stoolap_tx_errcode(tx)` | `int32_t` | Last error code (`STOOLAP_ERR_*`) |
| `stoolap_tx_errdetails(tx, &out)` | `int32_t` | Fill structured error details |

### Read-Only Handle

| Function | Returns | Description |
|----------|---------|-------------|
| `stoolap_open_read_only(dsn, &ro)` | `int32_t` | Open a read-only handle on `dsn` |
| `stoolap_ro_close(ro)` | `void` | Close and free (NULL-safe) |
| `stoolap_ro_query(ro, sql, &rows)` | `int32_t` | Query without parameters |
| `stoolap_ro_query_params(ro, sql, params, len, &rows)` | `int32_t` | Query with positional parameters |
| `stoolap_ro_query_named(ro, sql, params, len, &rows)` | `int32_t` | Query with named parameters |
| `stoolap_ro_table_exists(ro, name)` | `int32_t` | 1 if exists, 0 if not, -1 on error |
| `stoolap_ro_table_count(ro, table, &count)` | `int32_t` | Snapshot row count for `table` |
| `stoolap_ro_refresh(ro)` | `int32_t` | Advance to writer's latest visible state |
| `stoolap_ro_set_auto_refresh(ro, enabled)` | `void` | Toggle automatic refresh on every query |
| `stoolap_ro_dsn(ro)` | `const char*` | DSN string (cached for handle lifetime) |
| `stoolap_ro_errmsg(ro)` | `const char*` | Last error message |
| `stoolap_ro_errcode(ro)` | `int32_t` | Last error code (`STOOLAP_ERR_*`) |
| `stoolap_ro_errdetails(ro, &out)` | `int32_t` | Fill structured error details |

### Result Set

| Function | Returns | Description |
|----------|---------|-------------|
| `stoolap_rows_next(rows)` | `int32_t` | Advance to next row (`STOOLAP_ROW`, `STOOLAP_DONE`, or `STOOLAP_ERROR`) |
| `stoolap_rows_column_count(rows)` | `int32_t` | Number of columns |
| `stoolap_rows_column_name(rows, i)` | `const char*` | Column name by index (NULL if out of bounds) |
| `stoolap_rows_column_type(rows, i)` | `int32_t` | Column type (`STOOLAP_TYPE_*`) in current row |
| `stoolap_rows_column_int64(rows, i)` | `int64_t` | Integer value (0 if NULL) |
| `stoolap_rows_column_double(rows, i)` | `double` | Float value (0.0 if NULL) |
| `stoolap_rows_column_text(rows, i, &len)` | `const char*` | Text value (NULL if NULL column) |
| `stoolap_rows_column_bool(rows, i)` | `int32_t` | Boolean value (0 if NULL) |
| `stoolap_rows_column_timestamp(rows, i)` | `int64_t` | Nanoseconds since epoch (0 if NULL) |
| `stoolap_rows_column_blob(rows, i, &len)` | `const uint8_t*` | Vector payload as packed f32 (NULL if not VECTOR) |
| `stoolap_rows_column_is_null(rows, i)` | `int32_t` | 1 if NULL, 0 otherwise |
| `stoolap_rows_affected(rows)` | `int64_t` | Rows affected (for DML results) |
| `stoolap_rows_close(rows)` | `void` | Close and free (NULL-safe, must always be called) |
| `stoolap_rows_errmsg(rows)` | `const char*` | Last error message |
| `stoolap_rows_errcode(rows)` | `int32_t` | Last error code (`STOOLAP_ERR_*`) |
| `stoolap_rows_errdetails(rows, &out)` | `int32_t` | Fill structured error details |

### Typed Error Codes

| Function | Returns | Description |
|----------|---------|-------------|
| `stoolap_errcode(db)` | `int32_t` | Last error code on `db`. Pass NULL for thread-local (open / commit / rollback failures). |
| `stoolap_errdetails(db, &out)` | `int32_t` | Fill `StoolapErrorDetails` from `db`. Pass NULL for thread-local. |

### Bulk Fetch

| Function | Returns | Description |
|----------|---------|-------------|
| `stoolap_rows_fetch_all(rows, &buf, &len)` | `int32_t` | Fetch all remaining rows into a packed binary buffer |
| `stoolap_buffer_free(buf, len)` | `void` | Free a buffer from `stoolap_rows_fetch_all` (NULL-safe) |

### Memory

| Function | Returns | Description |
|----------|---------|-------------|
| `stoolap_string_free(s)` | `void` | Free a library-allocated string (NULL-safe). Reserved for future use. |
