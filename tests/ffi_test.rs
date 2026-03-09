// Copyright 2025 Stoolap Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Tests for the C FFI layer.

#![cfg(feature = "ffi")]

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use stoolap::ffi::*;

/// Helper to create a CString and return its pointer.
fn cstr(s: &str) -> CString {
    CString::new(s).unwrap()
}

/// Helper to read a C string pointer as &str.
unsafe fn read_cstr(ptr: *const c_char) -> &'static str {
    if ptr.is_null() {
        return "";
    }
    CStr::from_ptr(ptr).to_str().unwrap_or("")
}

// =========================================================================
// Database lifecycle
// =========================================================================

#[test]
fn test_version() {
    let ver = stoolap_version();
    let ver_str = unsafe { read_cstr(ver) };
    assert!(!ver_str.is_empty());
    assert!(ver_str.contains('.'));
}

#[test]
fn test_open_close_in_memory() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        let rc = stoolap_open_in_memory(&mut db);
        assert_eq!(rc, STOOLAP_OK);
        assert!(!db.is_null());

        let rc = stoolap_close(db);
        assert_eq!(rc, STOOLAP_OK);
    }
}

#[test]
fn test_open_with_dsn() {
    unsafe {
        let dsn = cstr("memory://test_open_dsn");
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        let rc = stoolap_open(dsn.as_ptr(), &mut db);
        assert_eq!(rc, STOOLAP_OK);
        assert!(!db.is_null());

        let rc = stoolap_close(db);
        assert_eq!(rc, STOOLAP_OK);
    }
}

#[test]
fn test_close_null_is_noop() {
    unsafe {
        let rc = stoolap_close(std::ptr::null_mut());
        assert_eq!(rc, STOOLAP_OK);
    }
}

#[test]
fn test_errmsg_no_error() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let msg = read_cstr(stoolap_errmsg(db));
        assert_eq!(msg, "");

        stoolap_close(db);
    }
}

// =========================================================================
// Execute and Query
// =========================================================================

#[test]
fn test_exec_create_table() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT, score FLOAT)");
        let rc = stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());
        assert_eq!(rc, STOOLAP_OK);

        stoolap_close(db);
    }
}

#[test]
fn test_exec_insert_and_count() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        let sql = cstr("INSERT INTO t VALUES (1, 'hello'), (2, 'world')");
        let mut affected: i64 = 0;
        let rc = stoolap_exec(db, sql.as_ptr(), &mut affected);
        assert_eq!(rc, STOOLAP_OK);
        assert_eq!(affected, 2);

        // Query count
        let sql = cstr("SELECT COUNT(*) FROM t");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        let rc = stoolap_query(db, sql.as_ptr(), &mut rows);
        assert_eq!(rc, STOOLAP_OK);
        assert!(!rows.is_null());

        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_ROW);
        assert_eq!(stoolap_rows_column_int64(rows, 0), 2);

        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_DONE);

        stoolap_rows_close(rows);
        stoolap_close(db);
    }
}

#[test]
fn test_exec_error() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("SELECT * FROM nonexistent_table");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        let rc = stoolap_query(db, sql.as_ptr(), &mut rows);
        assert_eq!(rc, STOOLAP_ERROR);
        assert!(rows.is_null());

        let msg = read_cstr(stoolap_errmsg(db));
        assert!(!msg.is_empty());

        stoolap_close(db);
    }
}

// =========================================================================
// Parameters
// =========================================================================

#[test]
fn test_exec_with_params() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql =
            cstr("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, score FLOAT, active BOOLEAN)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        let name = cstr("Alice");
        let params = [
            StoolapValue {
                value_type: STOOLAP_TYPE_INTEGER,
                _padding: 0,
                v: StoolapValueData { integer: 1 },
            },
            StoolapValue {
                value_type: STOOLAP_TYPE_TEXT,
                _padding: 0,
                v: StoolapValueData {
                    text: StoolapTextData {
                        ptr: name.as_ptr(),
                        len: 5,
                    },
                },
            },
            StoolapValue {
                value_type: STOOLAP_TYPE_FLOAT,
                _padding: 0,
                v: StoolapValueData { float64: 95.5 },
            },
            StoolapValue {
                value_type: STOOLAP_TYPE_BOOLEAN,
                _padding: 0,
                v: StoolapValueData { boolean: 1 },
            },
        ];

        let sql = cstr("INSERT INTO t VALUES ($1, $2, $3, $4)");
        let mut affected: i64 = 0;
        let rc = stoolap_exec_params(db, sql.as_ptr(), params.as_ptr(), 4, &mut affected);
        assert_eq!(rc, STOOLAP_OK);
        assert_eq!(affected, 1);

        // Verify the data
        let sql = cstr("SELECT id, name, score, active FROM t WHERE id = 1");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        let rc = stoolap_query(db, sql.as_ptr(), &mut rows);
        assert_eq!(rc, STOOLAP_OK);

        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_ROW);

        assert_eq!(stoolap_rows_column_int64(rows, 0), 1);

        let mut text_len: i64 = 0;
        let text_ptr = stoolap_rows_column_text(rows, 1, &mut text_len);
        assert!(!text_ptr.is_null());
        assert_eq!(read_cstr(text_ptr), "Alice");
        assert_eq!(text_len, 5);

        assert!((stoolap_rows_column_double(rows, 2) - 95.5).abs() < f64::EPSILON);
        assert_eq!(stoolap_rows_column_bool(rows, 3), 1);

        stoolap_rows_close(rows);
        stoolap_close(db);
    }
}

// =========================================================================
// Column metadata
// =========================================================================

#[test]
fn test_rows_column_metadata() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER, name TEXT)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());
        let sql = cstr("INSERT INTO t VALUES (1, 'hello')");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        let sql = cstr("SELECT id, name FROM t");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        stoolap_query(db, sql.as_ptr(), &mut rows);

        assert_eq!(stoolap_rows_column_count(rows), 2);
        assert_eq!(read_cstr(stoolap_rows_column_name(rows, 0)), "id");
        assert_eq!(read_cstr(stoolap_rows_column_name(rows, 1)), "name");
        assert!(stoolap_rows_column_name(rows, 99).is_null());

        // Check types after stepping
        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_ROW);
        assert_eq!(stoolap_rows_column_type(rows, 0), STOOLAP_TYPE_INTEGER);
        assert_eq!(stoolap_rows_column_type(rows, 1), STOOLAP_TYPE_TEXT);

        stoolap_rows_close(rows);
        stoolap_close(db);
    }
}

// =========================================================================
// NULL handling
// =========================================================================

#[test]
fn test_null_values() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER, val TEXT)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());
        let sql = cstr("INSERT INTO t VALUES (1, NULL)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        let sql = cstr("SELECT id, val FROM t");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        stoolap_query(db, sql.as_ptr(), &mut rows);

        stoolap_rows_next(rows);

        assert_eq!(stoolap_rows_column_is_null(rows, 0), 0);
        assert_eq!(stoolap_rows_column_is_null(rows, 1), 1);
        assert_eq!(stoolap_rows_column_type(rows, 1), STOOLAP_TYPE_NULL);
        assert!(stoolap_rows_column_text(rows, 1, std::ptr::null_mut()).is_null());

        stoolap_rows_close(rows);
        stoolap_close(db);
    }
}

// =========================================================================
// Prepared statements
// =========================================================================

#[test]
fn test_prepared_statement() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        // Prepare insert
        let sql = cstr("INSERT INTO t VALUES ($1, $2)");
        let mut stmt: *mut StoolapStmt = std::ptr::null_mut();
        let rc = stoolap_prepare(db, sql.as_ptr(), &mut stmt);
        assert_eq!(rc, STOOLAP_OK);
        assert!(!stmt.is_null());

        // Check SQL text
        assert_eq!(
            read_cstr(stoolap_stmt_sql(stmt)),
            "INSERT INTO t VALUES ($1, $2)"
        );

        // Execute with different params
        for (id, name_str) in [(1i64, "Alice"), (2, "Bob"), (3, "Charlie")] {
            let name = cstr(name_str);
            let params = [
                StoolapValue {
                    value_type: STOOLAP_TYPE_INTEGER,
                    _padding: 0,
                    v: StoolapValueData { integer: id },
                },
                StoolapValue {
                    value_type: STOOLAP_TYPE_TEXT,
                    _padding: 0,
                    v: StoolapValueData {
                        text: StoolapTextData {
                            ptr: name.as_ptr(),
                            len: name_str.len() as i64,
                        },
                    },
                },
            ];
            let mut affected: i64 = 0;
            let rc = stoolap_stmt_exec(stmt, params.as_ptr(), 2, &mut affected);
            assert_eq!(rc, STOOLAP_OK);
            assert_eq!(affected, 1);
        }

        stoolap_stmt_finalize(stmt);

        // Verify count
        let sql = cstr("SELECT COUNT(*) FROM t");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        stoolap_query(db, sql.as_ptr(), &mut rows);
        stoolap_rows_next(rows);
        assert_eq!(stoolap_rows_column_int64(rows, 0), 3);
        stoolap_rows_close(rows);

        stoolap_close(db);
    }
}

#[test]
fn test_prepared_query() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());
        let sql = cstr("INSERT INTO t VALUES (1, 'Alice'), (2, 'Bob')");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        let sql = cstr("SELECT name FROM t WHERE id = $1");
        let mut stmt: *mut StoolapStmt = std::ptr::null_mut();
        stoolap_prepare(db, sql.as_ptr(), &mut stmt);

        // Query for id=1
        let params = [StoolapValue {
            value_type: STOOLAP_TYPE_INTEGER,
            _padding: 0,
            v: StoolapValueData { integer: 1 },
        }];
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        let rc = stoolap_stmt_query(stmt, params.as_ptr(), 1, &mut rows);
        assert_eq!(rc, STOOLAP_OK);

        stoolap_rows_next(rows);
        assert_eq!(
            read_cstr(stoolap_rows_column_text(rows, 0, std::ptr::null_mut())),
            "Alice"
        );
        stoolap_rows_close(rows);

        // Query for id=2
        let params = [StoolapValue {
            value_type: STOOLAP_TYPE_INTEGER,
            _padding: 0,
            v: StoolapValueData { integer: 2 },
        }];
        let mut rows2: *mut StoolapRows = std::ptr::null_mut();
        stoolap_stmt_query(stmt, params.as_ptr(), 1, &mut rows2);
        stoolap_rows_next(rows2);
        assert_eq!(
            read_cstr(stoolap_rows_column_text(rows2, 0, std::ptr::null_mut())),
            "Bob"
        );
        stoolap_rows_close(rows2);

        stoolap_stmt_finalize(stmt);
        stoolap_close(db);
    }
}

// =========================================================================
// Transactions
// =========================================================================

#[test]
fn test_transaction_commit() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        // Begin transaction
        let mut tx: *mut StoolapTx = std::ptr::null_mut();
        let rc = stoolap_begin(db, &mut tx);
        assert_eq!(rc, STOOLAP_OK);
        assert!(!tx.is_null());

        // Insert within transaction
        let sql = cstr("INSERT INTO t VALUES (1, 100)");
        let rc = stoolap_tx_exec(tx, sql.as_ptr(), std::ptr::null_mut());
        assert_eq!(rc, STOOLAP_OK);

        // Commit
        let rc = stoolap_tx_commit(tx);
        assert_eq!(rc, STOOLAP_OK);

        // Verify data persisted
        let sql = cstr("SELECT val FROM t WHERE id = 1");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        stoolap_query(db, sql.as_ptr(), &mut rows);
        stoolap_rows_next(rows);
        assert_eq!(stoolap_rows_column_int64(rows, 0), 100);
        stoolap_rows_close(rows);

        stoolap_close(db);
    }
}

#[test]
fn test_transaction_rollback() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());
        let sql = cstr("INSERT INTO t VALUES (1, 100)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        // Begin transaction
        let mut tx: *mut StoolapTx = std::ptr::null_mut();
        stoolap_begin(db, &mut tx);

        // Update within transaction
        let sql = cstr("UPDATE t SET val = 999 WHERE id = 1");
        stoolap_tx_exec(tx, sql.as_ptr(), std::ptr::null_mut());

        // Rollback
        let rc = stoolap_tx_rollback(tx);
        assert_eq!(rc, STOOLAP_OK);

        // Verify original data
        let sql = cstr("SELECT val FROM t WHERE id = 1");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        stoolap_query(db, sql.as_ptr(), &mut rows);
        stoolap_rows_next(rows);
        assert_eq!(stoolap_rows_column_int64(rows, 0), 100);
        stoolap_rows_close(rows);

        stoolap_close(db);
    }
}

#[test]
fn test_transaction_with_isolation() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        let mut tx: *mut StoolapTx = std::ptr::null_mut();
        let rc = stoolap_begin_with_isolation(db, STOOLAP_ISOLATION_SNAPSHOT, &mut tx);
        assert_eq!(rc, STOOLAP_OK);

        let sql = cstr("INSERT INTO t VALUES (1)");
        stoolap_tx_exec(tx, sql.as_ptr(), std::ptr::null_mut());
        stoolap_tx_commit(tx);

        stoolap_close(db);
    }
}

#[test]
fn test_transaction_query() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());
        let sql = cstr("INSERT INTO t VALUES (1, 'Alice')");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        let mut tx: *mut StoolapTx = std::ptr::null_mut();
        stoolap_begin(db, &mut tx);

        let sql = cstr("SELECT name FROM t WHERE id = 1");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        let rc = stoolap_tx_query(tx, sql.as_ptr(), &mut rows);
        assert_eq!(rc, STOOLAP_OK);

        stoolap_rows_next(rows);
        assert_eq!(
            read_cstr(stoolap_rows_column_text(rows, 0, std::ptr::null_mut())),
            "Alice"
        );
        stoolap_rows_close(rows);

        stoolap_tx_commit(tx);
        stoolap_close(db);
    }
}

// =========================================================================
// Rows affected
// =========================================================================

#[test]
fn test_rows_affected() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());
        let sql = cstr("INSERT INTO t VALUES (1), (2), (3)");
        let mut affected: i64 = 0;
        stoolap_exec(db, sql.as_ptr(), &mut affected);
        assert_eq!(affected, 3);

        let sql = cstr("DELETE FROM t WHERE id > 1");
        stoolap_exec(db, sql.as_ptr(), &mut affected);
        assert_eq!(affected, 2);

        stoolap_close(db);
    }
}

// =========================================================================
// Clone for multi-threaded use
// =========================================================================

#[test]
fn test_clone_shares_data() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());
        let sql = cstr("INSERT INTO t VALUES (1, 'hello'), (2, 'world')");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        // Clone the handle
        let mut db2: *mut StoolapDB = std::ptr::null_mut();
        let rc = stoolap_clone(db, &mut db2);
        assert_eq!(rc, STOOLAP_OK);
        assert!(!db2.is_null());

        // Query through the clone — should see the same data
        let sql = cstr("SELECT COUNT(*) FROM t");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        let rc = stoolap_query(db2, sql.as_ptr(), &mut rows);
        assert_eq!(rc, STOOLAP_OK);
        stoolap_rows_next(rows);
        assert_eq!(stoolap_rows_column_int64(rows, 0), 2);
        stoolap_rows_close(rows);

        // Insert through clone, visible from original
        let sql = cstr("INSERT INTO t VALUES (3, 'from_clone')");
        stoolap_exec(db2, sql.as_ptr(), std::ptr::null_mut());

        let sql = cstr("SELECT COUNT(*) FROM t");
        let mut rows2: *mut StoolapRows = std::ptr::null_mut();
        stoolap_query(db, sql.as_ptr(), &mut rows2);
        stoolap_rows_next(rows2);
        assert_eq!(stoolap_rows_column_int64(rows2, 0), 3);
        stoolap_rows_close(rows2);

        // Independent error state
        let bad_sql = cstr("SELECT * FROM nonexistent");
        let mut bad_rows: *mut StoolapRows = std::ptr::null_mut();
        stoolap_query(db2, bad_sql.as_ptr(), &mut bad_rows);
        let err2 = read_cstr(stoolap_errmsg(db2));
        assert!(!err2.is_empty());
        let err1 = read_cstr(stoolap_errmsg(db));
        assert_eq!(err1, ""); // original handle unaffected

        // Both must be closed independently
        stoolap_close(db2);
        stoolap_close(db);
    }
}

#[test]
fn test_clone_multi_thread() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER PRIMARY KEY)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        // Clone for use in another thread
        let mut db2: *mut StoolapDB = std::ptr::null_mut();
        stoolap_clone(db, &mut db2);

        // StoolapDB* can be sent to another thread
        let db2_ptr = db2 as usize; // smuggle the pointer as usize
        let handle = std::thread::spawn(move || {
            let thread_db = db2_ptr as *mut StoolapDB;
            let sql = CString::new("INSERT INTO t VALUES (1)").unwrap();
            let rc = stoolap_exec(thread_db, sql.as_ptr(), std::ptr::null_mut());
            assert_eq!(rc, STOOLAP_OK);
            stoolap_close(thread_db);
        });
        handle.join().unwrap();

        // Verify data from main thread
        let sql = cstr("SELECT id FROM t");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        stoolap_query(db, sql.as_ptr(), &mut rows);
        stoolap_rows_next(rows);
        assert_eq!(stoolap_rows_column_int64(rows, 0), 1);
        stoolap_rows_close(rows);

        stoolap_close(db);
    }
}

// =========================================================================
// File DSN
// =========================================================================

#[test]
fn test_file_dsn() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test_ffi_db");
    let dsn_str = format!("file://{}", db_path.display());

    unsafe {
        // Open with file DSN
        let dsn = cstr(&dsn_str);
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        let rc = stoolap_open(dsn.as_ptr(), &mut db);
        assert_eq!(rc, STOOLAP_OK);

        let sql = cstr("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());
        let sql = cstr("INSERT INTO t VALUES (1, 'persisted')");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        stoolap_close(db);

        // Re-open and verify data persisted
        let mut db2: *mut StoolapDB = std::ptr::null_mut();
        let rc = stoolap_open(dsn.as_ptr(), &mut db2);
        assert_eq!(rc, STOOLAP_OK);

        let sql = cstr("SELECT val FROM t WHERE id = 1");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        stoolap_query(db2, sql.as_ptr(), &mut rows);
        stoolap_rows_next(rows);
        assert_eq!(
            read_cstr(stoolap_rows_column_text(rows, 0, std::ptr::null_mut())),
            "persisted"
        );
        stoolap_rows_close(rows);

        stoolap_close(db2);
    }
}

#[test]
fn test_file_dsn_with_params() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test_ffi_params");
    let dsn_str = format!("file://{}?sync_mode=full&compression=on", db_path.display());

    unsafe {
        let dsn = cstr(&dsn_str);
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        let rc = stoolap_open(dsn.as_ptr(), &mut db);
        assert_eq!(rc, STOOLAP_OK);

        let sql = cstr("CREATE TABLE t (id INTEGER)");
        let rc = stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());
        assert_eq!(rc, STOOLAP_OK);

        let sql = cstr("INSERT INTO t VALUES (1)");
        let rc = stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());
        assert_eq!(rc, STOOLAP_OK);

        stoolap_close(db);
    }
}

// =========================================================================
// Edge cases
// =========================================================================

#[test]
fn test_finalize_null_is_noop() {
    unsafe {
        stoolap_stmt_finalize(std::ptr::null_mut());
    }
}

#[test]
fn test_rows_close_null_is_noop() {
    unsafe {
        stoolap_rows_close(std::ptr::null_mut());
    }
}

#[test]
fn test_multiple_result_sets() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());
        let sql = cstr("INSERT INTO t VALUES (1), (2), (3)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        // Two concurrent result sets
        let sql = cstr("SELECT id FROM t ORDER BY id");
        let mut rows1: *mut StoolapRows = std::ptr::null_mut();
        let mut rows2: *mut StoolapRows = std::ptr::null_mut();
        stoolap_query(db, sql.as_ptr(), &mut rows1);
        stoolap_query(db, sql.as_ptr(), &mut rows2);

        // Iterate first result set
        stoolap_rows_next(rows1);
        assert_eq!(stoolap_rows_column_int64(rows1, 0), 1);

        // Iterate second result set independently
        stoolap_rows_next(rows2);
        assert_eq!(stoolap_rows_column_int64(rows2, 0), 1);
        stoolap_rows_next(rows2);
        assert_eq!(stoolap_rows_column_int64(rows2, 0), 2);

        stoolap_rows_close(rows1);
        stoolap_rows_close(rows2);
        stoolap_close(db);
    }
}

// =========================================================================
// Regression: clone outlives original handle
// =========================================================================

#[test]
fn test_clone_outlives_original() {
    // Regression: closing the original handle before a clone would shut down
    // the shared engine, making the clone unusable.
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());
        let sql = cstr("INSERT INTO t VALUES (1, 'hello')");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        // Clone the handle
        let mut clone: *mut StoolapDB = std::ptr::null_mut();
        let rc = stoolap_clone(db, &mut clone);
        assert_eq!(rc, STOOLAP_OK);

        // Close the ORIGINAL first
        stoolap_close(db);

        // Clone must still work after original is closed
        let sql = cstr("SELECT val FROM t WHERE id = 1");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        let rc = stoolap_query(clone, sql.as_ptr(), &mut rows);
        assert_eq!(rc, STOOLAP_OK);

        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_ROW);
        assert_eq!(
            read_cstr(stoolap_rows_column_text(rows, 0, std::ptr::null_mut())),
            "hello"
        );
        stoolap_rows_close(rows);

        // Insert through the clone should also work
        let sql = cstr("INSERT INTO t VALUES (2, 'from_clone')");
        let mut affected: i64 = 0;
        let rc = stoolap_exec(clone, sql.as_ptr(), &mut affected);
        assert_eq!(rc, STOOLAP_OK);
        assert_eq!(affected, 1);

        stoolap_close(clone);
    }
}

#[test]
fn test_clone_of_clone_outlives_all() {
    // Regression: nested clones must also keep the engine alive.
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());
        let sql = cstr("INSERT INTO t VALUES (42)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        let mut clone1: *mut StoolapDB = std::ptr::null_mut();
        stoolap_clone(db, &mut clone1);

        let mut clone2: *mut StoolapDB = std::ptr::null_mut();
        stoolap_clone(clone1, &mut clone2);

        // Close original and first clone
        stoolap_close(db);
        stoolap_close(clone1);

        // Second-level clone must still work
        let sql = cstr("SELECT id FROM t");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        let rc = stoolap_query(clone2, sql.as_ptr(), &mut rows);
        assert_eq!(rc, STOOLAP_OK);
        stoolap_rows_next(rows);
        assert_eq!(stoolap_rows_column_int64(rows, 0), 42);
        stoolap_rows_close(rows);

        stoolap_close(clone2);
    }
}

// =========================================================================
// Regression: JSON parameter type
// =========================================================================

#[test]
fn test_json_parameter() {
    // Regression: STOOLAP_TYPE_JSON parameters were silently converted to NULL.
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER PRIMARY KEY, data JSON)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        let json_str = cstr(r#"{"key": "value", "n": 42}"#);
        let params = [
            StoolapValue {
                value_type: STOOLAP_TYPE_INTEGER,
                _padding: 0,
                v: StoolapValueData { integer: 1 },
            },
            StoolapValue {
                value_type: STOOLAP_TYPE_JSON,
                _padding: 0,
                v: StoolapValueData {
                    text: StoolapTextData {
                        ptr: json_str.as_ptr(),
                        len: r#"{"key": "value", "n": 42}"#.len() as i64,
                    },
                },
            },
        ];

        let sql = cstr("INSERT INTO t VALUES ($1, $2)");
        let mut affected: i64 = 0;
        let rc = stoolap_exec_params(db, sql.as_ptr(), params.as_ptr(), 2, &mut affected);
        assert_eq!(rc, STOOLAP_OK);
        assert_eq!(affected, 1);

        // Verify the JSON data is not NULL and is retrievable
        let sql = cstr("SELECT data FROM t WHERE id = 1");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        stoolap_query(db, sql.as_ptr(), &mut rows);
        stoolap_rows_next(rows);

        // Must not be NULL
        assert_eq!(stoolap_rows_column_is_null(rows, 0), 0);

        // The type should be JSON
        assert_eq!(stoolap_rows_column_type(rows, 0), STOOLAP_TYPE_JSON);

        // Text representation should contain the JSON content
        let text = read_cstr(stoolap_rows_column_text(rows, 0, std::ptr::null_mut()));
        assert!(text.contains("key"));
        assert!(text.contains("value"));

        stoolap_rows_close(rows);
        stoolap_close(db);
    }
}

// =========================================================================
// Regression: prepared statement column cache after DDL
// =========================================================================

#[test]
fn test_prepared_stmt_column_cache_after_ddl() {
    // Regression: prepared SELECT * cached column metadata on first query.
    // After ALTER TABLE ADD COLUMN, the cached count was stale, causing
    // text_cache to be undersized.
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());
        let sql = cstr("INSERT INTO t VALUES (1, 'Alice')");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        // Prepare SELECT *
        let sql = cstr("SELECT * FROM t WHERE id = $1");
        let mut stmt: *mut StoolapStmt = std::ptr::null_mut();
        stoolap_prepare(db, sql.as_ptr(), &mut stmt);

        // First query: populates cache with 2 columns
        let params = [StoolapValue {
            value_type: STOOLAP_TYPE_INTEGER,
            _padding: 0,
            v: StoolapValueData { integer: 1 },
        }];
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        stoolap_stmt_query(stmt, params.as_ptr(), 1, &mut rows);
        assert_eq!(stoolap_rows_column_count(rows), 2);
        stoolap_rows_next(rows);
        assert_eq!(
            read_cstr(stoolap_rows_column_text(rows, 1, std::ptr::null_mut())),
            "Alice"
        );
        stoolap_rows_close(rows);

        // DDL: add a column
        let sql = cstr("ALTER TABLE t ADD COLUMN age INTEGER");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        // Re-query with same prepared statement: must see 3 columns now
        let mut rows2: *mut StoolapRows = std::ptr::null_mut();
        stoolap_stmt_query(stmt, params.as_ptr(), 1, &mut rows2);
        assert_eq!(stoolap_rows_column_count(rows2), 3);
        stoolap_rows_next(rows2);
        // The new column should be NULL
        assert_eq!(stoolap_rows_column_is_null(rows2, 2), 1);
        stoolap_rows_close(rows2);

        stoolap_stmt_finalize(stmt);
        stoolap_close(db);
    }
}

#[test]
fn test_timestamp_column() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER, ts TIMESTAMP)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());
        let sql = cstr("INSERT INTO t VALUES (1, '2024-01-15T10:30:00Z')");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        let sql = cstr("SELECT ts FROM t WHERE id = 1");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        stoolap_query(db, sql.as_ptr(), &mut rows);

        stoolap_rows_next(rows);
        assert_eq!(stoolap_rows_column_type(rows, 0), STOOLAP_TYPE_TIMESTAMP);

        let nanos = stoolap_rows_column_timestamp(rows, 0);
        assert!(nanos > 0);

        // Also test text representation
        let text = read_cstr(stoolap_rows_column_text(rows, 0, std::ptr::null_mut()));
        assert!(text.contains("2024"));

        stoolap_rows_close(rows);
        stoolap_close(db);
    }
}

// =========================================================================
// Regression: registry leak when clone outlives original (registered DSN)
// =========================================================================

#[test]
fn test_clone_outlives_original_registered_dsn() {
    // Regression: stoolap_open() registers the engine in DATABASE_REGISTRY.
    // The clone's _engine_keepalive Arc inflates the strong_count, preventing
    // Database::drop() from removing the registry entry (it checks count == 2).
    // stoolap_close() must explicitly unregister to prevent the leak.
    unsafe {
        let dsn = cstr("memory://test_clone_registry_leak");
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        let rc = stoolap_open(dsn.as_ptr(), &mut db);
        assert_eq!(rc, STOOLAP_OK);

        let sql = cstr("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());
        let sql = cstr("INSERT INTO t VALUES (1, 'hello')");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        // Clone the handle
        let mut clone: *mut StoolapDB = std::ptr::null_mut();
        let rc = stoolap_clone(db, &mut clone);
        assert_eq!(rc, STOOLAP_OK);

        // Close the ORIGINAL first (this is the scenario that triggered the leak)
        stoolap_close(db);

        // Clone must still work — engine must NOT have been closed
        let sql = cstr("SELECT val FROM t WHERE id = 1");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        let rc = stoolap_query(clone, sql.as_ptr(), &mut rows);
        assert_eq!(rc, STOOLAP_OK);

        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_ROW);
        assert_eq!(
            read_cstr(stoolap_rows_column_text(rows, 0, std::ptr::null_mut())),
            "hello"
        );
        stoolap_rows_close(rows);

        // DML through the clone should also work
        let sql = cstr("INSERT INTO t VALUES (2, 'from_clone')");
        let mut affected: i64 = 0;
        let rc = stoolap_exec(clone, sql.as_ptr(), &mut affected);
        assert_eq!(rc, STOOLAP_OK);
        assert_eq!(affected, 1);

        stoolap_close(clone);
    }
}

// =========================================================================
// Regression: BLOB round-trip (Vector type)
// =========================================================================

#[test]
fn test_blob_round_trip_vector() {
    // Regression: BLOB output included the internal type-tag byte and matched
    // any Extension variant (including JSON). BLOB input created a Text value
    // instead of Vector. Both directions are now fixed.
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER PRIMARY KEY, v VECTOR)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        // Insert vector via BLOB parameter: packed little-endian f32 bytes
        let floats: [f32; 3] = [1.0, 2.0, 3.0];
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();

        let params = [
            StoolapValue {
                value_type: STOOLAP_TYPE_INTEGER,
                _padding: 0,
                v: StoolapValueData { integer: 1 },
            },
            StoolapValue {
                value_type: STOOLAP_TYPE_BLOB,
                _padding: 0,
                v: StoolapValueData {
                    blob: StoolapBlobData {
                        ptr: bytes.as_ptr(),
                        len: bytes.len() as i64,
                    },
                },
            },
        ];

        let sql = cstr("INSERT INTO t VALUES ($1, $2)");
        let mut affected: i64 = 0;
        let rc = stoolap_exec_params(db, sql.as_ptr(), params.as_ptr(), 2, &mut affected);
        assert_eq!(rc, STOOLAP_OK);
        assert_eq!(affected, 1);

        // Read it back via blob accessor
        let sql = cstr("SELECT v FROM t WHERE id = 1");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        stoolap_query(db, sql.as_ptr(), &mut rows);
        stoolap_rows_next(rows);

        assert_eq!(stoolap_rows_column_type(rows, 0), STOOLAP_TYPE_BLOB);

        let mut blob_len: i64 = 0;
        let blob_ptr = stoolap_rows_column_blob(rows, 0, &mut blob_len);
        assert!(!blob_ptr.is_null());
        // Must be exactly 12 bytes (3 x 4), with NO leading tag byte
        assert_eq!(blob_len, 12);

        let out_bytes = std::slice::from_raw_parts(blob_ptr, blob_len as usize);
        for (i, &expected) in floats.iter().enumerate() {
            let actual = f32::from_le_bytes(out_bytes[i * 4..i * 4 + 4].try_into().unwrap());
            assert!(
                (actual - expected).abs() < f32::EPSILON,
                "float[{}]: expected {}, got {}",
                i,
                expected,
                actual
            );
        }

        stoolap_rows_close(rows);
        stoolap_close(db);
    }
}

#[test]
fn test_blob_accessor_rejects_json() {
    // Regression: stoolap_rows_column_blob() used to match ANY Extension variant,
    // so JSON values would incorrectly produce a non-NULL blob pointer.
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER PRIMARY KEY, data JSON)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());
        let sql = cstr(r#"INSERT INTO t VALUES (1, '{"a":1}')"#);
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        let sql = cstr("SELECT data FROM t WHERE id = 1");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        stoolap_query(db, sql.as_ptr(), &mut rows);
        stoolap_rows_next(rows);

        assert_eq!(stoolap_rows_column_type(rows, 0), STOOLAP_TYPE_JSON);

        // blob accessor must return NULL for JSON values
        let mut blob_len: i64 = 0;
        let blob_ptr = stoolap_rows_column_blob(rows, 0, &mut blob_len);
        assert!(blob_ptr.is_null());
        assert_eq!(blob_len, 0);

        stoolap_rows_close(rows);
        stoolap_close(db);
    }
}

// =========================================================================
// Regression: prepared statement column cache after same-arity DDL
// =========================================================================

#[test]
fn test_prepared_stmt_column_cache_after_rename() {
    // Regression: prepared statement column name cache was only invalidated
    // when the column count changed. A RENAME COLUMN (same arity) left
    // stale cached names. Now we compare actual names, not just count.
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER PRIMARY KEY, old_name TEXT)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());
        let sql = cstr("INSERT INTO t VALUES (1, 'val')");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        // Prepare SELECT *
        let sql = cstr("SELECT * FROM t WHERE id = $1");
        let mut stmt: *mut StoolapStmt = std::ptr::null_mut();
        stoolap_prepare(db, sql.as_ptr(), &mut stmt);

        // First query: caches column names [id, old_name]
        let params = [StoolapValue {
            value_type: STOOLAP_TYPE_INTEGER,
            _padding: 0,
            v: StoolapValueData { integer: 1 },
        }];
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        stoolap_stmt_query(stmt, params.as_ptr(), 1, &mut rows);
        assert_eq!(stoolap_rows_column_count(rows), 2);
        assert_eq!(read_cstr(stoolap_rows_column_name(rows, 1)), "old_name");
        stoolap_rows_close(rows);

        // DDL: rename column (same arity)
        let sql = cstr("ALTER TABLE t RENAME COLUMN old_name TO new_name");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        // Re-query: must see the new column name
        let mut rows2: *mut StoolapRows = std::ptr::null_mut();
        stoolap_stmt_query(stmt, params.as_ptr(), 1, &mut rows2);
        assert_eq!(stoolap_rows_column_count(rows2), 2);
        assert_eq!(read_cstr(stoolap_rows_column_name(rows2, 1)), "new_name");
        stoolap_rows_close(rows2);

        stoolap_stmt_finalize(stmt);
        stoolap_close(db);
    }
}

// =========================================================================
// Regression: interior NUL in text values
// =========================================================================

#[test]
fn test_text_with_interior_nul_not_null() {
    // Regression: text values containing \0 bytes would be collapsed to NULL
    // because CString::new() rejects interior NULs. Now the full bytes are
    // preserved: out_len reports the real length and callers can read the
    // complete data via the length. C string consumers (strlen) naturally
    // see a truncated view at the first embedded NUL.
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        // Insert a text value that contains an interior NUL: "hello\0world"
        let text_with_nul = b"hello\0world";
        let params = [
            StoolapValue {
                value_type: STOOLAP_TYPE_INTEGER,
                _padding: 0,
                v: StoolapValueData { integer: 1 },
            },
            StoolapValue {
                value_type: STOOLAP_TYPE_TEXT,
                _padding: 0,
                v: StoolapValueData {
                    text: StoolapTextData {
                        ptr: text_with_nul.as_ptr() as *const c_char,
                        len: text_with_nul.len() as i64,
                    },
                },
            },
        ];

        let sql = cstr("INSERT INTO t VALUES ($1, $2)");
        stoolap_exec_params(db, sql.as_ptr(), params.as_ptr(), 2, std::ptr::null_mut());

        let sql = cstr("SELECT val FROM t WHERE id = 1");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        stoolap_query(db, sql.as_ptr(), &mut rows);
        stoolap_rows_next(rows);

        // Must NOT be NULL — the value exists
        assert_eq!(stoolap_rows_column_is_null(rows, 0), 0);

        let mut text_len: i64 = 0;
        let text_ptr = stoolap_rows_column_text(rows, 0, &mut text_len);
        assert!(!text_ptr.is_null());

        // C string view (via strlen) is truncated at the first NUL
        assert_eq!(read_cstr(text_ptr), "hello");

        // But out_len reports the full length including the embedded NUL
        assert_eq!(text_len, 11); // "hello\0world" = 11 bytes

        // Callers using the length can recover the full data
        let full_bytes = std::slice::from_raw_parts(text_ptr as *const u8, text_len as usize);
        assert_eq!(full_bytes, b"hello\0world");

        stoolap_rows_close(rows);
        stoolap_close(db);
    }
}

// =========================================================================
// Regression: shared DSN — closing one handle must not break others
// =========================================================================

#[test]
fn test_shared_dsn_close_first_handle() {
    // Regression: stoolap_close() unconditionally unregistered non-clone handles.
    // If two handles share a registered DSN, closing either one removed the
    // registry entry, so the next stoolap_open() with the same DSN would
    // create a second engine instead of sharing. Now we only unregister when
    // the last non-registry reference drops.
    unsafe {
        let dsn = cstr("memory://test_shared_dsn");

        // Open twice with the same DSN — should share the engine
        let mut db1: *mut StoolapDB = std::ptr::null_mut();
        let mut db2: *mut StoolapDB = std::ptr::null_mut();
        let rc = stoolap_open(dsn.as_ptr(), &mut db1);
        assert_eq!(rc, STOOLAP_OK);
        let rc = stoolap_open(dsn.as_ptr(), &mut db2);
        assert_eq!(rc, STOOLAP_OK);

        // Create table through first handle
        let sql = cstr("CREATE TABLE t (id INTEGER PRIMARY KEY)");
        stoolap_exec(db1, sql.as_ptr(), std::ptr::null_mut());
        let sql = cstr("INSERT INTO t VALUES (1)");
        stoolap_exec(db1, sql.as_ptr(), std::ptr::null_mut());

        // Close the first handle
        stoolap_close(db1);

        // Second handle must still work
        let sql = cstr("SELECT id FROM t");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        let rc = stoolap_query(db2, sql.as_ptr(), &mut rows);
        assert_eq!(rc, STOOLAP_OK);
        stoolap_rows_next(rows);
        assert_eq!(stoolap_rows_column_int64(rows, 0), 1);
        stoolap_rows_close(rows);

        // Open a third handle with the same DSN — must share the live engine
        let mut db3: *mut StoolapDB = std::ptr::null_mut();
        let rc = stoolap_open(dsn.as_ptr(), &mut db3);
        assert_eq!(rc, STOOLAP_OK);

        // db3 must see the same table (shared engine)
        let sql = cstr("SELECT COUNT(*) FROM t");
        let mut rows2: *mut StoolapRows = std::ptr::null_mut();
        let rc = stoolap_query(db3, sql.as_ptr(), &mut rows2);
        assert_eq!(rc, STOOLAP_OK);
        stoolap_rows_next(rows2);
        assert_eq!(stoolap_rows_column_int64(rows2, 0), 1);
        stoolap_rows_close(rows2);

        stoolap_close(db3);
        stoolap_close(db2);
    }
}

// =========================================================================
// Regression: BLOB with invalid byte length
// =========================================================================

#[test]
fn test_blob_rejects_non_multiple_of_4() {
    // Regression: BLOB input accepted any byte length and silently truncated
    // trailing bytes. A 10-byte payload was accepted as a 2-float vector.
    // Now len % 4 != 0 is rejected up front (produces NULL).
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER PRIMARY KEY, v VECTOR)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        // 10 bytes is not a valid packed-f32 length
        let bad_bytes: [u8; 10] = [0; 10];
        let params = [
            StoolapValue {
                value_type: STOOLAP_TYPE_INTEGER,
                _padding: 0,
                v: StoolapValueData { integer: 1 },
            },
            StoolapValue {
                value_type: STOOLAP_TYPE_BLOB,
                _padding: 0,
                v: StoolapValueData {
                    blob: StoolapBlobData {
                        ptr: bad_bytes.as_ptr(),
                        len: 10,
                    },
                },
            },
        ];

        let sql = cstr("INSERT INTO t VALUES ($1, $2)");
        stoolap_exec_params(db, sql.as_ptr(), params.as_ptr(), 2, std::ptr::null_mut());

        // The inserted value should be NULL (rejected blob)
        let sql = cstr("SELECT v FROM t WHERE id = 1");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        stoolap_query(db, sql.as_ptr(), &mut rows);
        stoolap_rows_next(rows);
        assert_eq!(stoolap_rows_column_is_null(rows, 0), 1);
        stoolap_rows_close(rows);

        stoolap_close(db);
    }
}

// =========================================================================
// Regression: JSON validation at FFI boundary
// =========================================================================

#[test]
fn test_json_parameter_rejects_invalid() {
    // Regression: STOOLAP_TYPE_JSON accepted any UTF-8 string without
    // validation. Malformed payloads like "not json" were tagged as JSON
    // and JSON_VALID() would return 1. Now the FFI input path validates
    // with serde_json and rejects invalid payloads as NULL.
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE t (id INTEGER PRIMARY KEY, data JSON)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        // Try inserting invalid JSON via STOOLAP_TYPE_JSON
        let bad_json = b"not json";
        let params = [
            StoolapValue {
                value_type: STOOLAP_TYPE_INTEGER,
                _padding: 0,
                v: StoolapValueData { integer: 1 },
            },
            StoolapValue {
                value_type: STOOLAP_TYPE_JSON,
                _padding: 0,
                v: StoolapValueData {
                    text: StoolapTextData {
                        ptr: bad_json.as_ptr() as *const c_char,
                        len: bad_json.len() as i64,
                    },
                },
            },
        ];

        let sql = cstr("INSERT INTO t VALUES ($1, $2)");
        stoolap_exec_params(db, sql.as_ptr(), params.as_ptr(), 2, std::ptr::null_mut());

        // The inserted value should be NULL (rejected invalid JSON)
        let sql = cstr("SELECT data FROM t WHERE id = 1");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        stoolap_query(db, sql.as_ptr(), &mut rows);
        stoolap_rows_next(rows);
        assert_eq!(stoolap_rows_column_is_null(rows, 0), 1);
        stoolap_rows_close(rows);

        // Also test empty string — not valid JSON
        let empty = b"";
        let params2 = [
            StoolapValue {
                value_type: STOOLAP_TYPE_INTEGER,
                _padding: 0,
                v: StoolapValueData { integer: 2 },
            },
            StoolapValue {
                value_type: STOOLAP_TYPE_JSON,
                _padding: 0,
                v: StoolapValueData {
                    text: StoolapTextData {
                        ptr: empty.as_ptr() as *const c_char,
                        len: 0,
                    },
                },
            },
        ];

        let sql = cstr("INSERT INTO t VALUES ($1, $2)");
        stoolap_exec_params(db, sql.as_ptr(), params2.as_ptr(), 2, std::ptr::null_mut());

        let sql = cstr("SELECT data FROM t WHERE id = 2");
        let mut rows2: *mut StoolapRows = std::ptr::null_mut();
        stoolap_query(db, sql.as_ptr(), &mut rows2);
        stoolap_rows_next(rows2);
        assert_eq!(stoolap_rows_column_is_null(rows2, 0), 1);
        stoolap_rows_close(rows2);

        stoolap_close(db);
    }
}

// =========================================================================
// Prepared statement lifetime: stmt outlives originating db handle
// =========================================================================

#[test]
fn test_prepared_stmt_outlives_db_handle() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        assert_eq!(stoolap_open_in_memory(&mut db), STOOLAP_OK);

        let create = cstr("CREATE TABLE t (id INTEGER, name TEXT)");
        stoolap_exec(db, create.as_ptr(), std::ptr::null_mut());

        let ins = cstr("INSERT INTO t VALUES (1, 'Alice'), (2, 'Bob')");
        stoolap_exec(db, ins.as_ptr(), std::ptr::null_mut());

        // Prepare a statement
        let mut stmt: *mut StoolapStmt = std::ptr::null_mut();
        let sql = cstr("SELECT name FROM t WHERE id = $1");
        assert_eq!(stoolap_prepare(db, sql.as_ptr(), &mut stmt), STOOLAP_OK);
        assert!(!stmt.is_null());

        // Close the db handle BEFORE using the statement
        stoolap_close(db);

        // The statement should still work because it keeps the engine alive
        let param = StoolapValue {
            value_type: STOOLAP_TYPE_INTEGER,
            _padding: 0,
            v: StoolapValueData { integer: 1 },
        };
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        assert_eq!(
            stoolap_stmt_query(stmt, &param, 1, &mut rows),
            STOOLAP_OK,
            "stmt_query should succeed after db close: {}",
            read_cstr(stoolap_stmt_errmsg(stmt))
        );
        assert_eq!(stoolap_rows_next(rows), STOOLAP_ROW);
        let mut len: i64 = 0;
        let text = stoolap_rows_column_text(rows, 0, &mut len);
        assert_eq!(read_cstr(text), "Alice");
        assert_eq!(stoolap_rows_next(rows), STOOLAP_DONE);
        stoolap_rows_close(rows);

        stoolap_stmt_finalize(stmt);
    }
}

#[test]
fn test_prepared_stmt_exec_outlives_db_handle() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        assert_eq!(stoolap_open_in_memory(&mut db), STOOLAP_OK);

        let create = cstr("CREATE TABLE t2 (id INTEGER, name TEXT)");
        stoolap_exec(db, create.as_ptr(), std::ptr::null_mut());

        // Prepare an INSERT statement
        let mut stmt: *mut StoolapStmt = std::ptr::null_mut();
        let sql = cstr("INSERT INTO t2 VALUES ($1, $2)");
        assert_eq!(stoolap_prepare(db, sql.as_ptr(), &mut stmt), STOOLAP_OK);

        // Close the db handle
        stoolap_close(db);

        // Execute the prepared statement after db close
        let params = [
            StoolapValue {
                value_type: STOOLAP_TYPE_INTEGER,
                _padding: 0,
                v: StoolapValueData { integer: 42 },
            },
            StoolapValue {
                value_type: STOOLAP_TYPE_TEXT,
                _padding: 0,
                v: StoolapValueData {
                    text: StoolapTextData {
                        ptr: c"test".as_ptr(),
                        len: 4,
                    },
                },
            },
        ];
        let mut affected: i64 = 0;
        assert_eq!(
            stoolap_stmt_exec(stmt, params.as_ptr(), 2, &mut affected),
            STOOLAP_OK,
            "stmt_exec should succeed after db close: {}",
            read_cstr(stoolap_stmt_errmsg(stmt))
        );
        assert_eq!(affected, 1);

        stoolap_stmt_finalize(stmt);
    }
}

#[test]
fn test_prepared_stmt_from_clone_outlives_both_handles() {
    unsafe {
        // Open original, insert data
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        assert_eq!(stoolap_open_in_memory(&mut db), STOOLAP_OK);

        let create = cstr("CREATE TABLE t3 (id INTEGER, name TEXT)");
        stoolap_exec(db, create.as_ptr(), std::ptr::null_mut());

        let ins = cstr("INSERT INTO t3 VALUES (1, 'Alice')");
        stoolap_exec(db, ins.as_ptr(), std::ptr::null_mut());

        // Clone the handle
        let mut clone: *mut StoolapDB = std::ptr::null_mut();
        assert_eq!(stoolap_clone(db, &mut clone), STOOLAP_OK);

        // Prepare a statement from the CLONE
        let mut stmt: *mut StoolapStmt = std::ptr::null_mut();
        let sql = cstr("SELECT name FROM t3 WHERE id = $1");
        assert_eq!(stoolap_prepare(clone, sql.as_ptr(), &mut stmt), STOOLAP_OK);

        // Close the clone first, then the original
        stoolap_close(clone);
        stoolap_close(db);

        // The statement should still work — it keeps the engine alive
        let param = StoolapValue {
            value_type: STOOLAP_TYPE_INTEGER,
            _padding: 0,
            v: StoolapValueData { integer: 1 },
        };
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        assert_eq!(
            stoolap_stmt_query(stmt, &param, 1, &mut rows),
            STOOLAP_OK,
            "stmt from clone should work after both handles closed: {}",
            read_cstr(stoolap_stmt_errmsg(stmt))
        );
        assert_eq!(stoolap_rows_next(rows), STOOLAP_ROW);
        let mut len: i64 = 0;
        let text = stoolap_rows_column_text(rows, 0, &mut len);
        assert_eq!(read_cstr(text), "Alice");
        assert_eq!(stoolap_rows_next(rows), STOOLAP_DONE);
        stoolap_rows_close(rows);

        stoolap_stmt_finalize(stmt);
    }
}

// =========================================================================
// Prepared statement registry cleanup after finalize
// =========================================================================

#[test]
fn test_prepared_stmt_finalize_cleans_registry() {
    unsafe {
        // Use a registered DSN (stoolap_open, not open_in_memory)
        let dsn = cstr("memory://stmt_registry_test");

        let mut db: *mut StoolapDB = std::ptr::null_mut();
        assert_eq!(stoolap_open(dsn.as_ptr(), &mut db), STOOLAP_OK);

        let create = cstr("CREATE TABLE reg (id INTEGER)");
        stoolap_exec(db, create.as_ptr(), std::ptr::null_mut());
        let ins = cstr("INSERT INTO reg VALUES (1)");
        stoolap_exec(db, ins.as_ptr(), std::ptr::null_mut());

        // Prepare a statement
        let mut stmt: *mut StoolapStmt = std::ptr::null_mut();
        let sql = cstr("SELECT id FROM reg WHERE id = $1");
        assert_eq!(stoolap_prepare(db, sql.as_ptr(), &mut stmt), STOOLAP_OK);

        // Close the db handle (stmt holds keepalive, so registry can't clean up yet)
        stoolap_close(db);

        // Finalize the statement — this should clean up the registry entry
        stoolap_stmt_finalize(stmt);

        // Re-open the same DSN. If the registry was cleaned up, we get a
        // fresh engine (no tables). If it leaked, we'd get the old engine
        // with table "reg" still present.
        let mut db2: *mut StoolapDB = std::ptr::null_mut();
        assert_eq!(stoolap_open(dsn.as_ptr(), &mut db2), STOOLAP_OK);

        // The old table should not exist in a fresh engine
        let q = cstr("SELECT id FROM reg");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        let rc = stoolap_query(db2, q.as_ptr(), &mut rows);
        assert_eq!(
            rc, STOOLAP_ERROR,
            "expected error querying dropped table, got OK (registry leaked)"
        );

        stoolap_close(db2);
    }
}

#[test]
fn test_prepared_stmt_from_clone_finalize_cleans_registry() {
    unsafe {
        // Use a registered DSN
        let dsn = cstr("memory://stmt_clone_registry_test");

        let mut db: *mut StoolapDB = std::ptr::null_mut();
        assert_eq!(stoolap_open(dsn.as_ptr(), &mut db), STOOLAP_OK);

        let create = cstr("CREATE TABLE creg (id INTEGER)");
        stoolap_exec(db, create.as_ptr(), std::ptr::null_mut());

        // Clone, prepare from clone
        let mut clone: *mut StoolapDB = std::ptr::null_mut();
        assert_eq!(stoolap_clone(db, &mut clone), STOOLAP_OK);

        let mut stmt: *mut StoolapStmt = std::ptr::null_mut();
        let sql = cstr("SELECT id FROM creg WHERE id = $1");
        assert_eq!(stoolap_prepare(clone, sql.as_ptr(), &mut stmt), STOOLAP_OK);

        // Close clone, then original
        stoolap_close(clone);
        stoolap_close(db);

        // Finalize — should clean up the registry
        stoolap_stmt_finalize(stmt);

        // Re-open: should get a fresh engine
        let mut db2: *mut StoolapDB = std::ptr::null_mut();
        assert_eq!(stoolap_open(dsn.as_ptr(), &mut db2), STOOLAP_OK);

        let q = cstr("SELECT id FROM creg");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        let rc = stoolap_query(db2, q.as_ptr(), &mut rows);
        assert_eq!(
            rc, STOOLAP_ERROR,
            "expected error querying dropped table, got OK (registry leaked)"
        );

        stoolap_close(db2);
    }
}

// =========================================================================
// Transaction lifetime: tx outlives originating db handle
// =========================================================================

#[test]
fn test_transaction_outlives_db_handle() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        assert_eq!(stoolap_open_in_memory(&mut db), STOOLAP_OK);

        let create = cstr("CREATE TABLE tx_life (id INTEGER, name TEXT)");
        stoolap_exec(db, create.as_ptr(), std::ptr::null_mut());

        // Begin a transaction
        let mut tx: *mut StoolapTx = std::ptr::null_mut();
        assert_eq!(stoolap_begin(db, &mut tx), STOOLAP_OK);

        // Insert inside the transaction
        let ins = cstr("INSERT INTO tx_life VALUES (1, 'Alice')");
        assert_eq!(
            stoolap_tx_exec(tx, ins.as_ptr(), std::ptr::null_mut()),
            STOOLAP_OK
        );

        // Close the db handle while tx is still open
        stoolap_close(db);

        // Query inside the transaction — should still work
        let q = cstr("SELECT name FROM tx_life WHERE id = 1");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        assert_eq!(
            stoolap_tx_query(tx, q.as_ptr(), &mut rows),
            STOOLAP_OK,
            "tx_query should succeed after db close: {}",
            read_cstr(stoolap_tx_errmsg(tx))
        );
        assert_eq!(stoolap_rows_next(rows), STOOLAP_ROW);
        let mut len: i64 = 0;
        let text = stoolap_rows_column_text(rows, 0, &mut len);
        assert_eq!(read_cstr(text), "Alice");
        stoolap_rows_close(rows);

        // Commit should also work
        assert_eq!(stoolap_tx_commit(tx), STOOLAP_OK);
    }
}

#[test]
fn test_transaction_from_clone_outlives_both_handles() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        assert_eq!(stoolap_open_in_memory(&mut db), STOOLAP_OK);

        let create = cstr("CREATE TABLE tx_clone (id INTEGER)");
        stoolap_exec(db, create.as_ptr(), std::ptr::null_mut());

        // Clone and begin tx from clone
        let mut clone: *mut StoolapDB = std::ptr::null_mut();
        assert_eq!(stoolap_clone(db, &mut clone), STOOLAP_OK);

        let mut tx: *mut StoolapTx = std::ptr::null_mut();
        assert_eq!(stoolap_begin(clone, &mut tx), STOOLAP_OK);

        let ins = cstr("INSERT INTO tx_clone VALUES (42)");
        assert_eq!(
            stoolap_tx_exec(tx, ins.as_ptr(), std::ptr::null_mut()),
            STOOLAP_OK
        );

        // Close clone, then original
        stoolap_close(clone);
        stoolap_close(db);

        // Tx should still work
        let q = cstr("SELECT id FROM tx_clone WHERE id = 42");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        assert_eq!(
            stoolap_tx_query(tx, q.as_ptr(), &mut rows),
            STOOLAP_OK,
            "tx from clone should work after both handles closed: {}",
            read_cstr(stoolap_tx_errmsg(tx))
        );
        assert_eq!(stoolap_rows_next(rows), STOOLAP_ROW);
        assert_eq!(stoolap_rows_column_int64(rows, 0), 42);
        stoolap_rows_close(rows);

        assert_eq!(stoolap_tx_commit(tx), STOOLAP_OK);
    }
}

#[test]
fn test_transaction_commit_cleans_registry() {
    unsafe {
        let dsn = cstr("memory://tx_registry_test");

        let mut db: *mut StoolapDB = std::ptr::null_mut();
        assert_eq!(stoolap_open(dsn.as_ptr(), &mut db), STOOLAP_OK);

        let create = cstr("CREATE TABLE txreg (id INTEGER)");
        stoolap_exec(db, create.as_ptr(), std::ptr::null_mut());

        let mut tx: *mut StoolapTx = std::ptr::null_mut();
        assert_eq!(stoolap_begin(db, &mut tx), STOOLAP_OK);

        // Close db while tx is open
        stoolap_close(db);

        // Commit consumes tx and should clean up the registry
        assert_eq!(stoolap_tx_commit(tx), STOOLAP_OK);

        // Re-open: should get a fresh engine (no old tables)
        let mut db2: *mut StoolapDB = std::ptr::null_mut();
        assert_eq!(stoolap_open(dsn.as_ptr(), &mut db2), STOOLAP_OK);

        let q = cstr("SELECT id FROM txreg");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        let rc = stoolap_query(db2, q.as_ptr(), &mut rows);
        assert_eq!(
            rc, STOOLAP_ERROR,
            "expected error querying dropped table, got OK (registry leaked)"
        );

        stoolap_close(db2);
    }
}

#[test]
fn test_transaction_rollback_cleans_registry() {
    unsafe {
        let dsn = cstr("memory://tx_rollback_registry_test");

        let mut db: *mut StoolapDB = std::ptr::null_mut();
        assert_eq!(stoolap_open(dsn.as_ptr(), &mut db), STOOLAP_OK);

        let create = cstr("CREATE TABLE txroll (id INTEGER)");
        stoolap_exec(db, create.as_ptr(), std::ptr::null_mut());

        let mut tx: *mut StoolapTx = std::ptr::null_mut();
        assert_eq!(stoolap_begin(db, &mut tx), STOOLAP_OK);

        stoolap_close(db);

        // Rollback should also clean up the registry
        assert_eq!(stoolap_tx_rollback(tx), STOOLAP_OK);

        let mut db2: *mut StoolapDB = std::ptr::null_mut();
        assert_eq!(stoolap_open(dsn.as_ptr(), &mut db2), STOOLAP_OK);

        let q = cstr("SELECT id FROM txroll");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        let rc = stoolap_query(db2, q.as_ptr(), &mut rows);
        assert_eq!(
            rc, STOOLAP_ERROR,
            "expected error querying dropped table, got OK (registry leaked)"
        );

        stoolap_close(db2);
    }
}

// =========================================================================
// Prepare validates SQL up front
// =========================================================================

#[test]
fn test_prepare_rejects_invalid_sql() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        assert_eq!(stoolap_open_in_memory(&mut db), STOOLAP_OK);

        let mut stmt: *mut StoolapStmt = std::ptr::null_mut();
        let sql = cstr("SELECT FROM");
        let rc = stoolap_prepare(db, sql.as_ptr(), &mut stmt);
        assert_eq!(rc, STOOLAP_ERROR, "invalid SQL should fail at prepare time");
        assert!(stmt.is_null());

        let err = read_cstr(stoolap_errmsg(db));
        assert!(!err.is_empty(), "error message should be set");

        stoolap_close(db);
    }
}

#[test]
fn test_prepare_accepts_valid_sql() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        assert_eq!(stoolap_open_in_memory(&mut db), STOOLAP_OK);

        let create = cstr("CREATE TABLE prep (id INTEGER)");
        stoolap_exec(db, create.as_ptr(), std::ptr::null_mut());

        let mut stmt: *mut StoolapStmt = std::ptr::null_mut();
        let sql = cstr("SELECT id FROM prep WHERE id = $1");
        assert_eq!(stoolap_prepare(db, sql.as_ptr(), &mut stmt), STOOLAP_OK);
        assert!(!stmt.is_null());

        stoolap_stmt_finalize(stmt);
        stoolap_close(db);
    }
}

// =========================================================================
// Empty vector round-trip
// =========================================================================

#[test]
fn test_empty_vector_round_trip() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        assert_eq!(stoolap_open_in_memory(&mut db), STOOLAP_OK);

        let create = cstr("CREATE TABLE ev (id INTEGER, v VECTOR)");
        stoolap_exec(db, create.as_ptr(), std::ptr::null_mut());

        // Insert an empty vector via BLOB parameter with len=0
        let sql = cstr("INSERT INTO ev VALUES ($1, $2)");
        let params = [
            StoolapValue {
                value_type: STOOLAP_TYPE_INTEGER,
                _padding: 0,
                v: StoolapValueData { integer: 1 },
            },
            StoolapValue {
                value_type: STOOLAP_TYPE_BLOB,
                _padding: 0,
                v: StoolapValueData {
                    blob: StoolapBlobData {
                        ptr: [0u8; 0].as_ptr(),
                        len: 0,
                    },
                },
            },
        ];
        assert_eq!(
            stoolap_exec_params(db, sql.as_ptr(), params.as_ptr(), 2, std::ptr::null_mut()),
            STOOLAP_OK
        );

        // Read it back
        let q = cstr("SELECT v FROM ev WHERE id = 1");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        assert_eq!(stoolap_query(db, q.as_ptr(), &mut rows), STOOLAP_OK);
        assert_eq!(stoolap_rows_next(rows), STOOLAP_ROW);

        // Should be BLOB type, not NULL
        assert_eq!(stoolap_rows_column_type(rows, 0), STOOLAP_TYPE_BLOB);
        assert_eq!(stoolap_rows_column_is_null(rows, 0), 0);

        // blob accessor should return non-NULL with len=0
        let mut blob_len: i64 = -1;
        let blob_ptr = stoolap_rows_column_blob(rows, 0, &mut blob_len);
        assert!(
            !blob_ptr.is_null(),
            "empty vector should return non-NULL pointer"
        );
        assert_eq!(blob_len, 0, "empty vector should have len=0");

        stoolap_rows_close(rows);
        stoolap_close(db);
    }
}

// =========================================================================
// Aggregate functions via FFI
// =========================================================================

/// Helper: open db, create a populated table, return db handle.
unsafe fn setup_aggregate_test_db() -> *mut StoolapDB {
    let mut db: *mut StoolapDB = std::ptr::null_mut();
    let rc = stoolap_open_in_memory(&mut db);
    assert_eq!(rc, STOOLAP_OK);

    let sql = cstr(
        "CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer TEXT NOT NULL,
            product TEXT,
            quantity INTEGER NOT NULL,
            price FLOAT NOT NULL,
            region TEXT
        )",
    );
    let rc = stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());
    assert_eq!(rc, STOOLAP_OK);

    let inserts = [
        "INSERT INTO orders VALUES (1, 'Alice', 'Widget', 10, 9.99, 'East')",
        "INSERT INTO orders VALUES (2, 'Bob', 'Gadget', 5, 24.50, 'West')",
        "INSERT INTO orders VALUES (3, 'Alice', 'Widget', 3, 9.99, 'East')",
        "INSERT INTO orders VALUES (4, 'Charlie', 'Gizmo', 7, 15.00, 'East')",
        "INSERT INTO orders VALUES (5, 'Bob', 'Widget', 12, 9.99, 'West')",
        "INSERT INTO orders VALUES (6, 'Alice', 'Gadget', 1, 24.50, 'North')",
        "INSERT INTO orders VALUES (7, 'Charlie', 'Widget', 20, 9.99, 'East')",
        "INSERT INTO orders VALUES (8, 'Bob', 'Gizmo', 2, 15.00, 'West')",
        "INSERT INTO orders VALUES (9, 'Alice', 'Gizmo', 8, 15.00, 'North')",
        "INSERT INTO orders VALUES (10, 'Charlie', 'Gadget', 4, 24.50, 'East')",
    ];
    for insert in &inserts {
        let sql = cstr(insert);
        let rc = stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());
        assert_eq!(rc, STOOLAP_OK, "Failed: {}", insert);
    }

    db
}

/// Helper: run a query that returns a single integer result.
unsafe fn query_single_int(db: *mut StoolapDB, sql_str: &str) -> i64 {
    let sql = cstr(sql_str);
    let mut rows: *mut StoolapRows = std::ptr::null_mut();
    let rc = stoolap_query(db, sql.as_ptr(), &mut rows);
    assert_eq!(
        rc,
        STOOLAP_OK,
        "Query failed: {} — {}",
        sql_str,
        read_cstr(stoolap_errmsg(db))
    );

    let rc = stoolap_rows_next(rows);
    assert_eq!(rc, STOOLAP_ROW, "No row returned for: {}", sql_str);
    let val = stoolap_rows_column_int64(rows, 0);
    stoolap_rows_close(rows);
    val
}

/// Helper: run a query that returns a single float result.
unsafe fn query_single_float(db: *mut StoolapDB, sql_str: &str) -> f64 {
    let sql = cstr(sql_str);
    let mut rows: *mut StoolapRows = std::ptr::null_mut();
    let rc = stoolap_query(db, sql.as_ptr(), &mut rows);
    assert_eq!(
        rc,
        STOOLAP_OK,
        "Query failed: {} — {}",
        sql_str,
        read_cstr(stoolap_errmsg(db))
    );

    let rc = stoolap_rows_next(rows);
    assert_eq!(rc, STOOLAP_ROW, "No row returned for: {}", sql_str);
    let val = stoolap_rows_column_double(rows, 0);
    stoolap_rows_close(rows);
    val
}

#[test]
fn test_ffi_count_star() {
    unsafe {
        let db = setup_aggregate_test_db();
        assert_eq!(query_single_int(db, "SELECT COUNT(*) FROM orders"), 10);
        stoolap_close(db);
    }
}

#[test]
fn test_ffi_count_with_where() {
    unsafe {
        let db = setup_aggregate_test_db();

        assert_eq!(
            query_single_int(db, "SELECT COUNT(*) FROM orders WHERE customer = 'Alice'"),
            4
        );
        assert_eq!(
            query_single_int(db, "SELECT COUNT(*) FROM orders WHERE price > 10.0"),
            6
        );
        // East rows: id 1(9.99), 3(9.99), 4(15.00), 7(9.99), 10(24.50) -> price < 20: 1,3,4,7 = 4
        assert_eq!(
            query_single_int(
                db,
                "SELECT COUNT(*) FROM orders WHERE region = 'East' AND price < 20.0"
            ),
            4
        );

        stoolap_close(db);
    }
}

#[test]
fn test_ffi_count_column_and_distinct() {
    unsafe {
        let db = setup_aggregate_test_db();

        assert_eq!(
            query_single_int(db, "SELECT COUNT(product) FROM orders"),
            10
        );
        assert_eq!(
            query_single_int(db, "SELECT COUNT(DISTINCT customer) FROM orders"),
            3
        );
        assert_eq!(
            query_single_int(db, "SELECT COUNT(DISTINCT product) FROM orders"),
            3
        );
        assert_eq!(
            query_single_int(db, "SELECT COUNT(DISTINCT region) FROM orders"),
            3
        );

        stoolap_close(db);
    }
}

#[test]
fn test_ffi_sum_avg_min_max() {
    unsafe {
        let db = setup_aggregate_test_db();

        assert_eq!(query_single_int(db, "SELECT SUM(quantity) FROM orders"), 72);

        let avg = query_single_float(db, "SELECT AVG(quantity) FROM orders");
        assert!((avg - 7.2).abs() < 0.01, "AVG expected 7.2, got {}", avg);

        assert_eq!(query_single_int(db, "SELECT MIN(quantity) FROM orders"), 1);
        assert_eq!(query_single_int(db, "SELECT MAX(quantity) FROM orders"), 20);

        let min_price = query_single_float(db, "SELECT MIN(price) FROM orders");
        assert!(
            (min_price - 9.99).abs() < 0.01,
            "MIN(price) expected 9.99, got {}",
            min_price
        );
        let max_price = query_single_float(db, "SELECT MAX(price) FROM orders");
        assert!(
            (max_price - 24.50).abs() < 0.01,
            "MAX(price) expected 24.50, got {}",
            max_price
        );

        stoolap_close(db);
    }
}

#[test]
fn test_ffi_group_by_count() {
    unsafe {
        let db = setup_aggregate_test_db();

        let sql = cstr("SELECT customer, COUNT(*) FROM orders GROUP BY customer ORDER BY customer");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        let rc = stoolap_query(db, sql.as_ptr(), &mut rows);
        assert_eq!(rc, STOOLAP_OK, "{}", read_cstr(stoolap_errmsg(db)));

        // Alice: 4
        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_ROW);
        assert_eq!(
            read_cstr(stoolap_rows_column_text(rows, 0, std::ptr::null_mut())),
            "Alice"
        );
        assert_eq!(stoolap_rows_column_int64(rows, 1), 4);

        // Bob: 3
        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_ROW);
        assert_eq!(
            read_cstr(stoolap_rows_column_text(rows, 0, std::ptr::null_mut())),
            "Bob"
        );
        assert_eq!(stoolap_rows_column_int64(rows, 1), 3);

        // Charlie: 3
        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_ROW);
        assert_eq!(
            read_cstr(stoolap_rows_column_text(rows, 0, std::ptr::null_mut())),
            "Charlie"
        );
        assert_eq!(stoolap_rows_column_int64(rows, 1), 3);

        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_DONE);

        stoolap_rows_close(rows);
        stoolap_close(db);
    }
}

#[test]
fn test_ffi_group_by_sum_avg() {
    unsafe {
        let db = setup_aggregate_test_db();

        let sql = cstr(
            "SELECT customer, SUM(quantity), AVG(price) FROM orders GROUP BY customer ORDER BY customer",
        );
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        let rc = stoolap_query(db, sql.as_ptr(), &mut rows);
        assert_eq!(rc, STOOLAP_OK, "{}", read_cstr(stoolap_errmsg(db)));

        // Alice: qty=10+3+1+8=22
        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_ROW);
        assert_eq!(
            read_cstr(stoolap_rows_column_text(rows, 0, std::ptr::null_mut())),
            "Alice"
        );
        assert_eq!(stoolap_rows_column_int64(rows, 1), 22);

        // Bob: qty=5+12+2=19
        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_ROW);
        assert_eq!(
            read_cstr(stoolap_rows_column_text(rows, 0, std::ptr::null_mut())),
            "Bob"
        );
        assert_eq!(stoolap_rows_column_int64(rows, 1), 19);

        // Charlie: qty=7+20+4=31
        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_ROW);
        assert_eq!(
            read_cstr(stoolap_rows_column_text(rows, 0, std::ptr::null_mut())),
            "Charlie"
        );
        assert_eq!(stoolap_rows_column_int64(rows, 1), 31);

        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_DONE);

        stoolap_rows_close(rows);
        stoolap_close(db);
    }
}

#[test]
fn test_ffi_having_clause() {
    unsafe {
        let db = setup_aggregate_test_db();

        let sql = cstr(
            "SELECT customer, COUNT(*) AS cnt FROM orders GROUP BY customer HAVING COUNT(*) >= 4 ORDER BY customer",
        );
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        let rc = stoolap_query(db, sql.as_ptr(), &mut rows);
        assert_eq!(rc, STOOLAP_OK, "{}", read_cstr(stoolap_errmsg(db)));

        // Only Alice has 4 orders
        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_ROW);
        assert_eq!(
            read_cstr(stoolap_rows_column_text(rows, 0, std::ptr::null_mut())),
            "Alice"
        );
        assert_eq!(stoolap_rows_column_int64(rows, 1), 4);

        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_DONE);

        stoolap_rows_close(rows);
        stoolap_close(db);
    }
}

#[test]
fn test_ffi_multiple_aggregates_in_select() {
    unsafe {
        let db = setup_aggregate_test_db();

        let sql = cstr(
            "SELECT COUNT(*), SUM(quantity), AVG(price), MIN(quantity), MAX(quantity) FROM orders",
        );
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        let rc = stoolap_query(db, sql.as_ptr(), &mut rows);
        assert_eq!(rc, STOOLAP_OK, "{}", read_cstr(stoolap_errmsg(db)));

        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_ROW);

        assert_eq!(stoolap_rows_column_int64(rows, 0), 10); // COUNT(*)
        assert_eq!(stoolap_rows_column_int64(rows, 1), 72); // SUM(quantity)
        let avg = stoolap_rows_column_double(rows, 2);
        assert!(
            (avg - 15.846).abs() < 0.01,
            "AVG(price) expected ~15.846, got {}",
            avg
        );
        assert_eq!(stoolap_rows_column_int64(rows, 3), 1); // MIN(quantity)
        assert_eq!(stoolap_rows_column_int64(rows, 4), 20); // MAX(quantity)

        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_DONE);

        stoolap_rows_close(rows);
        stoolap_close(db);
    }
}

#[test]
fn test_ffi_count_with_join() {
    unsafe {
        let db = setup_aggregate_test_db();

        let sql = cstr(
            "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, city TEXT, tier TEXT)",
        );
        let rc = stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());
        assert_eq!(rc, STOOLAP_OK, "{}", read_cstr(stoolap_errmsg(db)));

        for ins in &[
            "INSERT INTO customers VALUES (1, 'Alice', 'New York', 'Gold')",
            "INSERT INTO customers VALUES (2, 'Bob', 'Chicago', 'Silver')",
            "INSERT INTO customers VALUES (3, 'Charlie', 'Boston', 'Gold')",
        ] {
            let sql = cstr(ins);
            let rc = stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());
            assert_eq!(rc, STOOLAP_OK, "Failed: {}", ins);
        }

        // COUNT with JOIN
        let val = query_single_int(
            db,
            "SELECT COUNT(*) FROM orders o JOIN customers c ON o.customer = c.name WHERE c.tier = 'Gold'",
        );
        assert_eq!(val, 7); // Alice(4) + Charlie(3)

        // GROUP BY with JOIN
        let sql = cstr(
            "SELECT c.tier, COUNT(*), SUM(o.quantity) FROM orders o JOIN customers c ON o.customer = c.name GROUP BY c.tier ORDER BY c.tier",
        );
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        let rc = stoolap_query(db, sql.as_ptr(), &mut rows);
        assert_eq!(rc, STOOLAP_OK, "{}", read_cstr(stoolap_errmsg(db)));

        // Gold: 7 orders, qty=53
        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_ROW);
        assert_eq!(
            read_cstr(stoolap_rows_column_text(rows, 0, std::ptr::null_mut())),
            "Gold"
        );
        assert_eq!(stoolap_rows_column_int64(rows, 1), 7);
        assert_eq!(stoolap_rows_column_int64(rows, 2), 53);

        // Silver: 3 orders, qty=19
        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_ROW);
        assert_eq!(
            read_cstr(stoolap_rows_column_text(rows, 0, std::ptr::null_mut())),
            "Silver"
        );
        assert_eq!(stoolap_rows_column_int64(rows, 1), 3);
        assert_eq!(stoolap_rows_column_int64(rows, 2), 19);

        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_DONE);

        stoolap_rows_close(rows);
        stoolap_close(db);
    }
}

#[test]
fn test_ffi_count_in_subquery() {
    unsafe {
        let db = setup_aggregate_test_db();

        // Scalar subquery with AVG
        let sql = cstr(
            "SELECT customer FROM orders WHERE quantity > (SELECT AVG(quantity) FROM orders) GROUP BY customer ORDER BY customer",
        );
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        let rc = stoolap_query(db, sql.as_ptr(), &mut rows);
        assert_eq!(rc, STOOLAP_OK, "{}", read_cstr(stoolap_errmsg(db)));

        // AVG(quantity) = 7.2, so quantity > 7.2: ids 1(10), 5(12), 7(20), 9(8)
        // customers: Alice, Bob, Charlie
        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_ROW);
        assert_eq!(
            read_cstr(stoolap_rows_column_text(rows, 0, std::ptr::null_mut())),
            "Alice"
        );

        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_ROW);
        assert_eq!(
            read_cstr(stoolap_rows_column_text(rows, 0, std::ptr::null_mut())),
            "Bob"
        );

        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_ROW);
        assert_eq!(
            read_cstr(stoolap_rows_column_text(rows, 0, std::ptr::null_mut())),
            "Charlie"
        );

        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_DONE);

        stoolap_rows_close(rows);
        stoolap_close(db);
    }
}

#[test]
fn test_ffi_count_with_in_clause() {
    unsafe {
        let db = setup_aggregate_test_db();

        // COUNT with IN literal list — exercises IN-list fast path guard
        let val = query_single_int(
            db,
            "SELECT COUNT(*) FROM orders WHERE id IN (1, 3, 5, 7, 9)",
        );
        assert_eq!(val, 5);

        // COUNT with IN subquery — exercises IN-subquery fast path guard
        let val = query_single_int(
            db,
            "SELECT COUNT(*) FROM orders WHERE customer IN (SELECT customer FROM orders WHERE region = 'East')",
        );
        // East: Alice(1,3), Charlie(4,7,10) -> all orders by Alice(4) + Charlie(3) = 7
        assert_eq!(val, 7);

        stoolap_close(db);
    }
}

#[test]
fn test_ffi_prepared_stmt_with_aggregates() {
    unsafe {
        let db = setup_aggregate_test_db();

        let sql = cstr("SELECT COUNT(*), SUM(quantity) FROM orders WHERE customer = $1");
        let mut stmt: *mut StoolapStmt = std::ptr::null_mut();
        let rc = stoolap_prepare(db, sql.as_ptr(), &mut stmt);
        assert_eq!(rc, STOOLAP_OK, "{}", read_cstr(stoolap_errmsg(db)));

        // Alice
        let alice = cstr("Alice");
        let params = [StoolapValue {
            value_type: STOOLAP_TYPE_TEXT,
            _padding: 0,
            v: StoolapValueData {
                text: StoolapTextData {
                    ptr: alice.as_ptr(),
                    len: 5,
                },
            },
        }];
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        let rc = stoolap_stmt_query(stmt, params.as_ptr(), 1, &mut rows);
        assert_eq!(rc, STOOLAP_OK, "{}", read_cstr(stoolap_stmt_errmsg(stmt)));

        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_ROW);
        assert_eq!(stoolap_rows_column_int64(rows, 0), 4);
        assert_eq!(stoolap_rows_column_int64(rows, 1), 22);
        stoolap_rows_close(rows);

        // Bob
        let bob = cstr("Bob");
        let params = [StoolapValue {
            value_type: STOOLAP_TYPE_TEXT,
            _padding: 0,
            v: StoolapValueData {
                text: StoolapTextData {
                    ptr: bob.as_ptr(),
                    len: 3,
                },
            },
        }];
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        let rc = stoolap_stmt_query(stmt, params.as_ptr(), 1, &mut rows);
        assert_eq!(rc, STOOLAP_OK);

        let rc = stoolap_rows_next(rows);
        assert_eq!(rc, STOOLAP_ROW);
        assert_eq!(stoolap_rows_column_int64(rows, 0), 3);
        assert_eq!(stoolap_rows_column_int64(rows, 1), 19);
        stoolap_rows_close(rows);

        stoolap_stmt_finalize(stmt);
        stoolap_close(db);
    }
}

#[test]
fn test_ffi_count_after_mutations() {
    unsafe {
        let db = setup_aggregate_test_db();

        assert_eq!(query_single_int(db, "SELECT COUNT(*) FROM orders"), 10);

        // Delete Bob's orders
        let sql = cstr("DELETE FROM orders WHERE customer = 'Bob'");
        let mut affected: i64 = 0;
        stoolap_exec(db, sql.as_ptr(), &mut affected);
        assert_eq!(affected, 3);

        assert_eq!(query_single_int(db, "SELECT COUNT(*) FROM orders"), 7);
        assert_eq!(query_single_int(db, "SELECT SUM(quantity) FROM orders"), 53);

        // Update and recheck
        let sql = cstr("UPDATE orders SET quantity = 100 WHERE id = 1");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        assert_eq!(
            query_single_int(db, "SELECT SUM(quantity) FROM orders"),
            143 // 53 - 10 + 100
        );
        assert_eq!(
            query_single_int(db, "SELECT MAX(quantity) FROM orders"),
            100
        );

        stoolap_close(db);
    }
}

#[test]
fn test_ffi_count_empty_table() {
    unsafe {
        let mut db: *mut StoolapDB = std::ptr::null_mut();
        stoolap_open_in_memory(&mut db);

        let sql = cstr("CREATE TABLE empty_t (id INTEGER PRIMARY KEY, val TEXT)");
        stoolap_exec(db, sql.as_ptr(), std::ptr::null_mut());

        assert_eq!(query_single_int(db, "SELECT COUNT(*) FROM empty_t"), 0);
        assert_eq!(query_single_int(db, "SELECT COUNT(val) FROM empty_t"), 0);

        // SUM on empty table returns NULL
        let sql = cstr("SELECT SUM(id) FROM empty_t");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        stoolap_query(db, sql.as_ptr(), &mut rows);
        stoolap_rows_next(rows);
        assert_eq!(stoolap_rows_column_is_null(rows, 0), 1);
        stoolap_rows_close(rows);

        stoolap_close(db);
    }
}
