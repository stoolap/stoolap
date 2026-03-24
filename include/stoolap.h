/*
 * Copyright 2025 Stoolap Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file stoolap.h
 * @brief Stoolap C API: high-performance embedded SQL database.
 *
 * C interface with opaque handles, step-based result iteration,
 * and per-handle error messages.
 *
 * Build: cargo build --release --features ffi
 * Link:  -lstoolap (libstoolap.so / libstoolap.dylib / stoolap.dll)
 */

#ifndef STOOLAP_H
#define STOOLAP_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Opaque handles
 * ========================================================================= */

/** Database connection handle. */
typedef struct StoolapDB     StoolapDB;

/** Prepared statement handle. */
typedef struct StoolapStmt   StoolapStmt;

/** Transaction handle. */
typedef struct StoolapTx     StoolapTx;

/** Result set iterator handle. */
typedef struct StoolapRows   StoolapRows;

/* =========================================================================
 * Status codes
 * ========================================================================= */

/** Operation succeeded. */
#define STOOLAP_OK      0

/** Operation failed. Call the appropriate *_errmsg() for details. */
#define STOOLAP_ERROR   1

/** stoolap_rows_next(): a row is available. */
#define STOOLAP_ROW     100

/** stoolap_rows_next(): no more rows. */
#define STOOLAP_DONE    101

/* =========================================================================
 * Value type codes
 * ========================================================================= */

#define STOOLAP_TYPE_NULL       0
#define STOOLAP_TYPE_INTEGER    1
#define STOOLAP_TYPE_FLOAT      2
#define STOOLAP_TYPE_TEXT       3
#define STOOLAP_TYPE_BOOLEAN    4
#define STOOLAP_TYPE_TIMESTAMP  5
#define STOOLAP_TYPE_JSON       6
#define STOOLAP_TYPE_BLOB       7

/* =========================================================================
 * Isolation levels
 * ========================================================================= */

#define STOOLAP_ISOLATION_READ_COMMITTED    0
#define STOOLAP_ISOLATION_SNAPSHOT          1

/* =========================================================================
 * Parameter value struct
 * ========================================================================= */

/** Tagged union for passing parameter values to SQL statements. */
typedef struct StoolapValue {
    /** One of STOOLAP_TYPE_* constants. */
    int32_t value_type;
    int32_t _padding;
    union {
        int64_t  integer;
        double   float64;
        int32_t  boolean;
        struct { const char*    ptr; int64_t len; } text;   /**< STOOLAP_TYPE_TEXT / _JSON */
        struct { const uint8_t* ptr; int64_t len; } blob;  /**< STOOLAP_TYPE_BLOB: packed f32, len must be multiple of 4 */
        /** Nanoseconds since Unix epoch (UTC). */
        int64_t  timestamp_nanos;
    } v;
} StoolapValue;

/* =========================================================================
 * Library info
 * ========================================================================= */

/**
 * Returns the stoolap version string (e.g. "0.3.7").
 * The returned pointer is static and must NOT be freed.
 */
const char* stoolap_version(void);

/* =========================================================================
 * Thread safety
 * =========================================================================
 *
 * A single StoolapDB handle must NOT be used concurrently from multiple
 * threads. To use stoolap from multiple threads, clone the handle with
 * stoolap_clone(). Each clone shares the underlying engine (data, indexes,
 * transactions) but has its own executor and error state.
 *
 * StoolapRows and StoolapTx handles must remain on the thread that created
 * them. StoolapStmt handles must not be used concurrently.
 *
 * Recommended pattern:
 *   StoolapDB* db;
 *   stoolap_open("file:///path/to/db", &db);
 *
 *   // Per worker thread:
 *   StoolapDB* thread_db;
 *   stoolap_clone(db, &thread_db);
 *   // ... use thread_db exclusively in this thread ...
 *   stoolap_close(thread_db);
 *
 *   stoolap_close(db);
 */

/* =========================================================================
 * Database lifecycle
 * ========================================================================= */

/**
 * Open a database connection.
 *
 * @param dsn     DSN string. Supported schemes:
 *                  "memory://"           - in-memory database
 *                  "file:///path/to/db"  - persistent database
 *                File DSN supports query parameters:
 *                  sync_mode=none|normal|full
 *                  compression=on|off
 *                  wal_flush_trigger=<bytes>
 *                  checkpoint_interval=<seconds>
 *                Example: "file:///tmp/mydb?sync_mode=full&compression=on"
 * @param out_db  On success, receives the database handle.
 * @return STOOLAP_OK on success. On failure, *out_db is NULL.
 *         Use stoolap_errmsg(NULL) to get the error message.
 */
int32_t stoolap_open(const char* dsn, StoolapDB** out_db);

/**
 * Open a new in-memory database.
 * Each call creates a unique, isolated instance.
 */
int32_t stoolap_open_in_memory(StoolapDB** out_db);

/**
 * Clone a database handle for multi-threaded use.
 *
 * The new handle shares the same underlying engine but has its own
 * executor and error state. Each clone must be closed independently.
 *
 * @param db      Source database handle.
 * @param out_db  On success, receives the cloned handle.
 */
int32_t stoolap_clone(const StoolapDB* db, StoolapDB** out_db);

/**
 * Close a database connection and free all associated resources.
 * Safe to call with NULL (no-op).
 * After this call, the db pointer is invalid.
 */
int32_t stoolap_close(StoolapDB* db);

/**
 * Get the last error message for a database handle.
 *
 * @param db  Database handle, or NULL for the last global error
 *            (e.g. from stoolap_open failures).
 * @return Pointer valid until the next API call on this handle.
 *         Returns "" if no error. Must NOT be freed.
 */
const char* stoolap_errmsg(const StoolapDB* db);

/* =========================================================================
 * Execute (DDL/DML)
 * ========================================================================= */

/**
 * Execute a SQL statement without parameters.
 *
 * @param rows_affected  If non-NULL, receives the number of affected rows.
 */
int32_t stoolap_exec(StoolapDB* db, const char* sql, int64_t* rows_affected);

/**
 * Execute a SQL statement with positional parameters ($1, $2, ...).
 *
 * @param params      Array of parameter values. May be NULL if params_len is 0.
 * @param params_len  Number of parameters.
 * @param rows_affected  If non-NULL, receives the number of affected rows.
 */
int32_t stoolap_exec_params(
    StoolapDB* db,
    const char* sql,
    const StoolapValue* params,
    int32_t params_len,
    int64_t* rows_affected
);

/* =========================================================================
 * Query (returns result rows)
 * ========================================================================= */

/**
 * Execute a query without parameters.
 * On success, *out_rows must be closed with stoolap_rows_close().
 */
int32_t stoolap_query(StoolapDB* db, const char* sql, StoolapRows** out_rows);

/**
 * Execute a query with positional parameters.
 * On success, *out_rows must be closed with stoolap_rows_close().
 */
int32_t stoolap_query_params(
    StoolapDB* db,
    const char* sql,
    const StoolapValue* params,
    int32_t params_len,
    StoolapRows** out_rows
);

/* =========================================================================
 * Prepared statements
 * ========================================================================= */

/**
 * Prepare a SQL statement for repeated execution.
 * On success, *out_stmt must be finalized with stoolap_stmt_finalize().
 *
 * The statement keeps the underlying database engine alive even if the
 * originating StoolapDB handle is closed before the statement is finalized.
 */
int32_t stoolap_prepare(StoolapDB* db, const char* sql, StoolapStmt** out_stmt);

/**
 * Execute a prepared statement with parameters.
 */
int32_t stoolap_stmt_exec(
    StoolapStmt* stmt,
    const StoolapValue* params,
    int32_t params_len,
    int64_t* rows_affected
);

/**
 * Query using a prepared statement with parameters.
 * On success, *out_rows must be closed with stoolap_rows_close().
 */
int32_t stoolap_stmt_query(
    StoolapStmt* stmt,
    const StoolapValue* params,
    int32_t params_len,
    StoolapRows** out_rows
);

/**
 * Get the SQL text of a prepared statement.
 * Returns a pointer valid for the lifetime of the statement. Must NOT be freed.
 */
const char* stoolap_stmt_sql(const StoolapStmt* stmt);

/**
 * Finalize (destroy) a prepared statement and free resources.
 * Safe to call with NULL (no-op).
 */
void stoolap_stmt_finalize(StoolapStmt* stmt);

/** Get the last error message for a statement handle. */
const char* stoolap_stmt_errmsg(const StoolapStmt* stmt);

/* =========================================================================
 * Transactions
 * ========================================================================= */

/**
 * Begin a transaction with the default isolation level (READ COMMITTED).
 * The transaction must be ended with stoolap_tx_commit() or stoolap_tx_rollback().
 *
 * The transaction keeps the underlying database engine alive even if the
 * originating StoolapDB handle is closed before the transaction ends.
 */
int32_t stoolap_begin(StoolapDB* db, StoolapTx** out_tx);

/**
 * Begin a transaction with a specific isolation level.
 * @param isolation  STOOLAP_ISOLATION_READ_COMMITTED or STOOLAP_ISOLATION_SNAPSHOT.
 */
int32_t stoolap_begin_with_isolation(
    StoolapDB* db,
    int32_t isolation,
    StoolapTx** out_tx
);

/** Execute within a transaction (no parameters). */
int32_t stoolap_tx_exec(StoolapTx* tx, const char* sql, int64_t* rows_affected);

/** Execute within a transaction (with parameters). */
int32_t stoolap_tx_exec_params(
    StoolapTx* tx,
    const char* sql,
    const StoolapValue* params,
    int32_t params_len,
    int64_t* rows_affected
);

/** Query within a transaction (no parameters). */
int32_t stoolap_tx_query(StoolapTx* tx, const char* sql, StoolapRows** out_rows);

/** Query within a transaction (with parameters). */
int32_t stoolap_tx_query_params(
    StoolapTx* tx,
    const char* sql,
    const StoolapValue* params,
    int32_t params_len,
    StoolapRows** out_rows
);

/**
 * Execute a prepared statement within a transaction (with parameters).
 *
 * Combines parse-once performance with transaction atomicity.
 * The statement must have been created via stoolap_prepare().
 *
 * @param tx             Transaction handle.
 * @param stmt           Prepared statement handle (not consumed).
 * @param params         Array of parameter values. May be NULL if params_len is 0.
 * @param params_len     Number of parameters.
 * @param rows_affected  If non-NULL, receives the number of affected rows.
 */
int32_t stoolap_tx_stmt_exec(
    StoolapTx* tx,
    const StoolapStmt* stmt,
    const StoolapValue* params,
    int32_t params_len,
    int64_t* rows_affected
);

/**
 * Query using a prepared statement within a transaction (with parameters).
 *
 * Combines parse-once performance with transaction atomicity.
 * On success, *out_rows must be closed with stoolap_rows_close().
 *
 * @param tx          Transaction handle.
 * @param stmt        Prepared statement handle (not consumed).
 * @param params      Array of parameter values. May be NULL if params_len is 0.
 * @param params_len  Number of parameters.
 * @param out_rows    On success, receives the result set handle.
 */
int32_t stoolap_tx_stmt_query(
    StoolapTx* tx,
    const StoolapStmt* stmt,
    const StoolapValue* params,
    int32_t params_len,
    StoolapRows** out_rows
);

/**
 * Commit a transaction.
 * The tx handle is consumed (freed) regardless of success or failure.
 */
int32_t stoolap_tx_commit(StoolapTx* tx);

/**
 * Rollback a transaction.
 * The tx handle is consumed (freed) regardless of success or failure.
 */
int32_t stoolap_tx_rollback(StoolapTx* tx);

/** Get the last error message for a transaction handle. */
const char* stoolap_tx_errmsg(const StoolapTx* tx);

/* =========================================================================
 * Result set iteration
 * ========================================================================= */

/**
 * Advance to the next row.
 * @return STOOLAP_ROW if a row is available.
 *         STOOLAP_DONE when exhausted.
 *         STOOLAP_ERROR on error (check stoolap_rows_errmsg).
 */
int32_t stoolap_rows_next(StoolapRows* rows);

/** Get the number of columns in the result set. */
int32_t stoolap_rows_column_count(const StoolapRows* rows);

/**
 * Get the name of a column by index (0-based).
 * Returns a pointer valid until stoolap_rows_close(). Must NOT be freed.
 * Returns NULL if index is out of bounds.
 */
const char* stoolap_rows_column_name(const StoolapRows* rows, int32_t index);

/**
 * Get the type of a column value in the current row.
 * Returns a STOOLAP_TYPE_* constant.
 * Only valid after a successful stoolap_rows_next().
 */
int32_t stoolap_rows_column_type(const StoolapRows* rows, int32_t index);

/** Get an integer value. Returns 0 if NULL or not convertible. */
int64_t stoolap_rows_column_int64(const StoolapRows* rows, int32_t index);

/** Get a float value. Returns 0.0 if NULL or not convertible. */
double stoolap_rows_column_double(const StoolapRows* rows, int32_t index);

/**
 * Get a text value from the current row.
 * @param out_len  If non-NULL, receives the full byte length (excluding the
 *                 trailing NUL terminator). This may exceed strlen() if the
 *                 value contains embedded NUL bytes. Use out_len to read the
 *                 complete data in that case.
 * @return Pointer valid until the next stoolap_rows_next() call.
 *         Returns NULL if the column is NULL. Must NOT be freed.
 */
const char* stoolap_rows_column_text(StoolapRows* rows, int32_t index, int64_t* out_len);

/** Get a boolean value. Returns 0 (false) if NULL or not convertible. */
int32_t stoolap_rows_column_bool(const StoolapRows* rows, int32_t index);

/** Get a timestamp as nanoseconds since Unix epoch (UTC). Returns 0 if NULL. */
int64_t stoolap_rows_column_timestamp(const StoolapRows* rows, int32_t index);

/**
 * Get a VECTOR column as packed little-endian f32 bytes.
 * Only returns data for VECTOR columns; returns NULL for JSON and other types.
 * @param out_len  Receives the byte length of the f32 payload.
 * @return Pointer valid until the next stoolap_rows_next() call.
 *         Returns NULL if not a VECTOR column or if NULL. Must NOT be freed.
 */
const uint8_t* stoolap_rows_column_blob(const StoolapRows* rows, int32_t index, int64_t* out_len);

/** Check if the current row's column is NULL. Returns 1 if NULL, 0 otherwise. */
int32_t stoolap_rows_column_is_null(const StoolapRows* rows, int32_t index);

/** Get the number of rows affected (for DML results). */
int64_t stoolap_rows_affected(const StoolapRows* rows);

/**
 * Close the result set and free resources.
 * Safe to call with NULL (no-op).
 * Must be called even after STOOLAP_DONE.
 */
void stoolap_rows_close(StoolapRows* rows);

/** Get the last error message for a rows handle. */
const char* stoolap_rows_errmsg(const StoolapRows* rows);

/* =========================================================================
 * Bulk fetch
 * ========================================================================= */

/**
 * Fetch all remaining rows into a packed binary buffer (single call).
 *
 * Format:
 *   [column_count: u32 LE]
 *   [for each column: name_len:u16 LE, name_bytes:u8[name_len]]
 *   [row_count: u32 LE]
 *   [for each row, for each column:
 *     type_tag: u8
 *     payload (varies):
 *       NULL(0):      (empty)
 *       INTEGER(1):   i64 LE (8 bytes)
 *       FLOAT(2):     f64 LE (8 bytes)
 *       TEXT(3):      len:u32 LE + bytes
 *       BOOLEAN(4):   u8 (0 or 1)
 *       TIMESTAMP(5): i64 LE (8 bytes, nanos since epoch)
 *       JSON(6):      len:u32 LE + bytes
 *       BLOB(7):      len:u32 LE + bytes (packed f32 for vectors)
 *   ]
 *
 * On success, *out_buf receives a heap-allocated buffer and *out_len its
 * byte length.  Caller MUST free with stoolap_buffer_free().
 * The rows handle's data is consumed; call stoolap_rows_close() afterward.
 */
int32_t stoolap_rows_fetch_all(
    StoolapRows* rows,
    uint8_t** out_buf,
    int64_t* out_len
);

/**
 * Free a buffer allocated by stoolap_rows_fetch_all().
 * Safe to call with NULL (no-op).
 */
void stoolap_buffer_free(uint8_t* buf, int64_t len);

/* =========================================================================
 * Memory management
 * ========================================================================= */

/**
 * Free a string allocated by the library.
 * Safe to call with NULL (no-op).
 * Only use for strings explicitly documented as "must be freed".
 */
void stoolap_string_free(char* s);

#ifdef __cplusplus
}
#endif

#endif /* STOOLAP_H */
