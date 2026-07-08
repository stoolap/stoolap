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

/** stoolap_ro_refresh(): snapshot advanced. STOOLAP_OK = unchanged. */
#define STOOLAP_REFRESH_ADVANCED    102

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

/**
 * Named parameter binding for `:name` placeholders.
 * `name` need not be NUL-terminated; must be valid UTF-8. Invalid/empty names are silently skipped.
 */
typedef struct StoolapNamedParam {
    /** Parameter name (without the ':' prefix). */
    const char*  name;
    /** Length of `name` in bytes. */
    int32_t      name_len;
    int32_t      _padding;
    /** Parameter value. */
    StoolapValue value;
} StoolapNamedParam;

/* =========================================================================
 * Library info
 * ========================================================================= */

/** Stoolap version string (e.g. "0.4.0"). Static pointer; do NOT free. */
const char* stoolap_version(void);

/* =========================================================================
 * Thread safety
 * =========================================================================
 *
 * A single handle is not concurrent-safe. Use stoolap_clone() per thread.
 * StoolapRows/StoolapTx/StoolapStmt are single-thread only.
 */

/* =========================================================================
 * Database lifecycle
 * ========================================================================= */

/**
 * Open a database connection.
 * DSN: "memory://" or "file:///path?sync_mode=...&compression=...&wal_flush_trigger=...&checkpoint_interval=..."
 * Returns STOOLAP_OK or STOOLAP_ERROR; on failure *out_db is NULL, see stoolap_errmsg(NULL).
 */
int32_t stoolap_open(const char* dsn, StoolapDB** out_db);

/** Open a new isolated in-memory database. */
int32_t stoolap_open_in_memory(StoolapDB** out_db);

/** Clone a handle for multi-threaded use; each clone must be closed independently. */
int32_t stoolap_clone(const StoolapDB* db, StoolapDB** out_db);

/** Close a database and free resources. NULL-safe. */
int32_t stoolap_close(StoolapDB* db);

/**
 * Last error message for a handle (NULL = last global error from stoolap_open).
 * Pointer valid until next API call on this handle. Returns "" if no error. Do NOT free.
 */
const char* stoolap_errmsg(const StoolapDB* db);

/* =========================================================================
 * Execute (DDL/DML)
 * ========================================================================= */

/** Execute a SQL statement. `rows_affected` may be NULL. */
int32_t stoolap_exec(StoolapDB* db, const char* sql, int64_t* rows_affected);

/** Execute a SQL statement with positional parameters ($1, $2, ...). */
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

/** Execute a query. *out_rows must be closed with stoolap_rows_close(). */
int32_t stoolap_query(StoolapDB* db, const char* sql, StoolapRows** out_rows);

/** Execute a query with positional parameters. *out_rows must be closed with stoolap_rows_close(). */
int32_t stoolap_query_params(
    StoolapDB* db,
    const char* sql,
    const StoolapValue* params,
    int32_t params_len,
    StoolapRows** out_rows
);

/* =========================================================================
 * Named parameters (`:name` placeholders)
 * ========================================================================= */

/** Execute a SQL statement with named parameters. */
int32_t stoolap_exec_named(
    StoolapDB* db,
    const char* sql,
    const StoolapNamedParam* params,
    int32_t params_len,
    int64_t* rows_affected
);

/** Execute a query with named parameters. */
int32_t stoolap_query_named(
    StoolapDB* db,
    const char* sql,
    const StoolapNamedParam* params,
    int32_t params_len,
    StoolapRows** out_rows
);

/* =========================================================================
 * Prepared statements
 * ========================================================================= */

/**
 * Prepare a SQL statement. *out_stmt must be finalized with stoolap_stmt_finalize().
 * The statement keeps the engine alive even if the originating StoolapDB is closed first.
 */
int32_t stoolap_prepare(StoolapDB* db, const char* sql, StoolapStmt** out_stmt);

/** Execute a prepared statement with parameters. */
int32_t stoolap_stmt_exec(
    StoolapStmt* stmt,
    const StoolapValue* params,
    int32_t params_len,
    int64_t* rows_affected
);

/**
 * Execute a prepared statement as a batch inside a single transaction.
 * `params` is a flat row-major array of row_count * params_per_row values.
 * On error the transaction is rolled back.
 */
int32_t stoolap_stmt_exec_batch(
    StoolapDB* db,
    const StoolapStmt* stmt,
    const StoolapValue* params,
    int32_t params_per_row,
    int32_t row_count,
    int64_t* total_affected
);

/** Query using a prepared statement with parameters. *out_rows must be closed with stoolap_rows_close(). */
int32_t stoolap_stmt_query(
    StoolapStmt* stmt,
    const StoolapValue* params,
    int32_t params_len,
    StoolapRows** out_rows
);

/** Get the SQL text of a prepared statement. Pointer valid for the statement's lifetime. Do NOT free. */
const char* stoolap_stmt_sql(const StoolapStmt* stmt);

/** Finalize a prepared statement. NULL-safe. */
void stoolap_stmt_finalize(StoolapStmt* stmt);

/** Last error message for a statement handle. */
const char* stoolap_stmt_errmsg(const StoolapStmt* stmt);

/* =========================================================================
 * Transactions
 * ========================================================================= */

/**
 * Begin a transaction (READ COMMITTED). End with stoolap_tx_commit/rollback.
 * The transaction keeps the engine alive even if the originating StoolapDB is closed first.
 */
int32_t stoolap_begin(StoolapDB* db, StoolapTx** out_tx);

/** Begin a transaction with STOOLAP_ISOLATION_READ_COMMITTED or STOOLAP_ISOLATION_SNAPSHOT. */
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

/** Execute a prepared statement within a transaction. `stmt` is not consumed. */
int32_t stoolap_tx_stmt_exec(
    StoolapTx* tx,
    const StoolapStmt* stmt,
    const StoolapValue* params,
    int32_t params_len,
    int64_t* rows_affected
);

/** Query using a prepared statement within a transaction. *out_rows must be closed with stoolap_rows_close(). */
int32_t stoolap_tx_stmt_query(
    StoolapTx* tx,
    const StoolapStmt* stmt,
    const StoolapValue* params,
    int32_t params_len,
    StoolapRows** out_rows
);

/** Execute within a transaction (named parameters). */
int32_t stoolap_tx_exec_named(
    StoolapTx* tx,
    const char* sql,
    const StoolapNamedParam* params,
    int32_t params_len,
    int64_t* rows_affected
);

/** Query within a transaction (named parameters). */
int32_t stoolap_tx_query_named(
    StoolapTx* tx,
    const char* sql,
    const StoolapNamedParam* params,
    int32_t params_len,
    StoolapRows** out_rows
);

/** Execute a prepared statement within a transaction (named parameters). */
int32_t stoolap_tx_stmt_exec_named(
    StoolapTx* tx,
    const StoolapStmt* stmt,
    const StoolapNamedParam* params,
    int32_t params_len,
    int64_t* rows_affected
);

/** Query using a prepared statement within a transaction (named parameters). */
int32_t stoolap_tx_stmt_query_named(
    StoolapTx* tx,
    const StoolapStmt* stmt,
    const StoolapNamedParam* params,
    int32_t params_len,
    StoolapRows** out_rows
);

/** Commit a transaction. The tx handle is consumed regardless of result. */
int32_t stoolap_tx_commit(StoolapTx* tx);

/** Rollback a transaction. The tx handle is consumed regardless of result. */
int32_t stoolap_tx_rollback(StoolapTx* tx);

/** Last error message for a transaction handle. */
const char* stoolap_tx_errmsg(const StoolapTx* tx);

/* =========================================================================
 * Result set iteration
 * ========================================================================= */

/** Advance to the next row. Returns STOOLAP_ROW, STOOLAP_DONE, or STOOLAP_ERROR. */
int32_t stoolap_rows_next(StoolapRows* rows);

/** Number of columns in the result set. */
int32_t stoolap_rows_column_count(const StoolapRows* rows);

/** Column name by 0-based index. Pointer valid until stoolap_rows_close(). NULL if out of bounds. */
const char* stoolap_rows_column_name(const StoolapRows* rows, int32_t index);

/** STOOLAP_TYPE_* of a column value in the current row. Only valid after stoolap_rows_next() returns STOOLAP_ROW. */
int32_t stoolap_rows_column_type(const StoolapRows* rows, int32_t index);

/** Get an integer value. Returns 0 if NULL or not convertible. */
int64_t stoolap_rows_column_int64(const StoolapRows* rows, int32_t index);

/** Get a float value. Returns 0.0 if NULL or not convertible. */
double stoolap_rows_column_double(const StoolapRows* rows, int32_t index);

/**
 * Get a text value. *out_len (if non-NULL) is the byte length excluding trailing NUL
 * (may exceed strlen() with embedded NULs). Pointer valid until next stoolap_rows_next().
 * NULL if column is NULL. Do NOT free.
 */
const char* stoolap_rows_column_text(StoolapRows* rows, int32_t index, int64_t* out_len);

/** Get a boolean (0 if NULL or not convertible). */
int32_t stoolap_rows_column_bool(const StoolapRows* rows, int32_t index);

/** Get a timestamp as nanoseconds since Unix epoch UTC (0 if NULL). */
int64_t stoolap_rows_column_timestamp(const StoolapRows* rows, int32_t index);

/**
 * Get a VECTOR column as packed little-endian f32 bytes. NULL for non-VECTOR or NULL.
 * Pointer valid until next stoolap_rows_next(). Do NOT free.
 */
const uint8_t* stoolap_rows_column_blob(const StoolapRows* rows, int32_t index, int64_t* out_len);

/** Returns 1 if column is NULL, 0 otherwise. */
int32_t stoolap_rows_column_is_null(const StoolapRows* rows, int32_t index);

/** Number of rows affected (for DML results). */
int64_t stoolap_rows_affected(const StoolapRows* rows);

/** Close the result set. NULL-safe. Must be called even after STOOLAP_DONE. */
void stoolap_rows_close(StoolapRows* rows);

/** Last error message for a rows handle. */
const char* stoolap_rows_errmsg(const StoolapRows* rows);

/* =========================================================================
 * Bulk fetch
 * ========================================================================= */

/**
 * Fetch all remaining rows into a packed binary buffer. Caller MUST free *out_buf with stoolap_buffer_free().
 * The rows handle is consumed; call stoolap_rows_close() afterward.
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
 */
int32_t stoolap_rows_fetch_all(
    StoolapRows* rows,
    uint8_t** out_buf,
    int64_t* out_len
);

/** Free a buffer allocated by stoolap_rows_fetch_all(). NULL-safe. */
void stoolap_buffer_free(uint8_t* buf, int64_t len);

/* =========================================================================
 * Memory management
 * ========================================================================= */

/** Free a string returned by the library. NULL-safe. Only use for strings documented as "must be freed". */
void stoolap_string_free(char* s);

/* =========================================================================
 * Typed error codes
 * =========================================================================
 *
 * errdetails() fills StoolapErrorDetails; char* fields valid until next FFI
 * call on the same handle. NULL = field not applicable. Codes are
 * append-only; treat unknown codes as STOOLAP_ERR_GENERIC.
 */

#define STOOLAP_ERR_OK                  0
#define STOOLAP_ERR_GENERIC             1
#define STOOLAP_ERR_NOT_NULL            2
#define STOOLAP_ERR_UNIQUE              3
#define STOOLAP_ERR_PRIMARY_KEY         4
#define STOOLAP_ERR_FOREIGN_KEY         5
#define STOOLAP_ERR_CHECK               6
#define STOOLAP_ERR_TABLE_NOT_FOUND     7
#define STOOLAP_ERR_TABLE_EXISTS        8
#define STOOLAP_ERR_COLUMN_NOT_FOUND    9
#define STOOLAP_ERR_INDEX_NOT_FOUND    10
#define STOOLAP_ERR_INDEX_EXISTS       11
#define STOOLAP_ERR_TYPE_MISMATCH      12
#define STOOLAP_ERR_INVALID_ARGUMENT   13
#define STOOLAP_ERR_PARSE              14
#define STOOLAP_ERR_TX_ABORTED         15
#define STOOLAP_ERR_TX_CLOSED          16
#define STOOLAP_ERR_READ_ONLY          17
#define STOOLAP_ERR_DB_LOCKED          18
#define STOOLAP_ERR_IO                 19
#define STOOLAP_ERR_NOT_SUPPORTED      20
#define STOOLAP_ERR_INTERNAL           21
#define STOOLAP_ERR_QUERY_CANCELLED    22
#define STOOLAP_ERR_DIVISION_BY_ZERO   23
#define STOOLAP_ERR_VALUE_TOO_LONG     24
#define STOOLAP_ERR_VIEW_NOT_FOUND     25
#define STOOLAP_ERR_VIEW_EXISTS        26
#define STOOLAP_ERR_REOPEN_REQUIRED    27 /**< SWMR readers: hard-reopen required (schema change,
                                                writer reincarnation, snapshot expiry, pending DDL,
                                                partial reload). Transient WAL-tail decode errors
                                                surface as STOOLAP_ERR_IO and may be retried. */

typedef struct StoolapErrorDetails {
    int32_t code;
    int32_t _padding;
    /** Always non-NULL. Empty C string on success. */
    const char* message;
    /** Table name, or NULL if not applicable. */
    const char* table;
    /** Column name, or NULL if not applicable. */
    const char* column;
    /** Index name (UNIQUE) or referenced table (FK), or NULL. */
    const char* constraint;
    /** Free-form: conflicting value (UNIQUE), CHECK expression, FK detail. */
    const char* detail;
} StoolapErrorDetails;

int32_t  stoolap_errcode      (const StoolapDB*    db);
int32_t  stoolap_errdetails   (const StoolapDB*    db, StoolapErrorDetails* out);
int32_t  stoolap_tx_errcode   (const StoolapTx*    tx);
int32_t  stoolap_tx_errdetails(const StoolapTx*    tx, StoolapErrorDetails* out);
int32_t  stoolap_stmt_errcode (const StoolapStmt*  stmt);
int32_t  stoolap_stmt_errdetails(const StoolapStmt* stmt, StoolapErrorDetails* out);
int32_t  stoolap_rows_errcode (const StoolapRows*  rows);
int32_t  stoolap_rows_errdetails(const StoolapRows* rows, StoolapErrorDetails* out);

/* =========================================================================
 * MVCC-safe table count
 * ========================================================================= */

/** Visible row count of `table`. Autocommit-correct; safe in hot loop. */
int32_t stoolap_table_count   (StoolapDB* db,    const char* table, uint64_t* out_count);
/** Visible row count within a transaction; includes same-tx uncommitted INSERT/DELETE. */
int32_t stoolap_tx_table_count(StoolapTx* tx,    const char* table, uint64_t* out_count);

/* =========================================================================
 * Savepoints
 * =========================================================================
 *
 * `name_len` is the byte length; pass -1 for a NUL-terminated C string.
 */

int32_t stoolap_tx_savepoint            (StoolapTx* tx, const char* name, int32_t name_len);
int32_t stoolap_tx_release_savepoint    (StoolapTx* tx, const char* name, int32_t name_len);
int32_t stoolap_tx_rollback_to_savepoint(StoolapTx* tx, const char* name, int32_t name_len);

/* =========================================================================
 * Read-only handle
 * =========================================================================
 *
 * StoolapRoDB exposes only read entry points. Write SQL passed to
 * stoolap_ro_query is rejected with STOOLAP_ERR_READ_ONLY. The DSN flags
 * ?read_only=true / ?readonly=true / ?mode=ro are accepted here as
 * no-ops, but rejected by stoolap_open with STOOLAP_ERR_INVALID_ARGUMENT.
 */

/** Read-only database handle. */
typedef struct StoolapRoDB StoolapRoDB;

int32_t      stoolap_open_read_only(const char* dsn, StoolapRoDB** out_db);
/** Clone a read-only handle for multi-threaded use. Each clone has its own WAL pin and must be closed independently. */
int32_t      stoolap_ro_clone          (const StoolapRoDB* db, StoolapRoDB** out_db);
void         stoolap_ro_close          (StoolapRoDB* db);
const char*  stoolap_ro_errmsg         (const StoolapRoDB* db);
int32_t      stoolap_ro_errcode        (const StoolapRoDB* db);
int32_t      stoolap_ro_errdetails     (const StoolapRoDB* db, StoolapErrorDetails* out);
const char*  stoolap_ro_dsn            (const StoolapRoDB* db);

/** Returns 1 if `name` exists, 0 if not, -1 on error (use stoolap_ro_errmsg). */
int32_t      stoolap_ro_table_exists   (StoolapRoDB* db, const char* name);

int32_t      stoolap_ro_table_count    (StoolapRoDB* db, const char* table, uint64_t* out_count);

/**
 * Advance the snapshot to the writer's latest visible state.
 * Returns STOOLAP_REFRESH_ADVANCED on advance, STOOLAP_OK if already
 * current, STOOLAP_ERROR on failure (STOOLAP_ERR_REOPEN_REQUIRED
 * means the caller MUST close and reopen this handle; transient
 * WAL-tail decode errors surface as STOOLAP_ERR_IO and may be retried).
 */
int32_t      stoolap_ro_refresh        (StoolapRoDB* db);

/** Toggle auto-refresh on every query (default enabled). Disable for stable cross-query snapshot. */
void         stoolap_ro_set_auto_refresh(StoolapRoDB* db, int32_t enabled);

/**
 * Configure background refresh ticker. 0 stops it; >0 (re)starts at that cadence.
 * Minimum 100ms; smaller values fail with STOOLAP_ERROR
 * (STOOLAP_ERR_INVALID_ARGUMENT via stoolap_ro_errcode()).
 * The ticker advances the WAL pin so idle readers do not block writer WAL truncation.
 * On must-reopen the ticker exits and the next query/refresh surfaces the error.
 * Equivalent DSN flag: `?refresh_interval=30s`.
 */
int32_t      stoolap_ro_set_refresh_interval(StoolapRoDB* db, uint64_t interval_millis);

int32_t      stoolap_ro_query          (StoolapRoDB* db, const char* sql, StoolapRows** out_rows);
int32_t      stoolap_ro_query_params   (StoolapRoDB* db, const char* sql,
                                         const StoolapValue* params, int32_t params_len,
                                         StoolapRows** out_rows);
int32_t      stoolap_ro_query_named    (StoolapRoDB* db, const char* sql,
                                         const StoolapNamedParam* params, int32_t params_len,
                                         StoolapRows** out_rows);

#ifdef __cplusplus
}
#endif

#endif /* STOOLAP_H */
