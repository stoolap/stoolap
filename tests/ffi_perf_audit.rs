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

//! Empirical verification of FFI hot-path bottlenecks.
//!
//! These are not assertions of behavior — they print measurements so a
//! reviewer can see the hot-path cost. Run with:
//!   cargo nextest run --features ffi --test ffi_perf_audit --no-capture

#![cfg(feature = "ffi")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use stoolap::ffi::*;

struct CountingAlloc;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static ENABLED: AtomicBool = AtomicBool::new(false);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ENABLED.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

fn cstr(s: &str) -> CString {
    CString::new(s).unwrap()
}

unsafe fn open() -> *mut StoolapDB {
    let mut db: *mut StoolapDB = std::ptr::null_mut();
    let dsn = cstr("memory://");
    assert_eq!(stoolap_open(dsn.as_ptr(), &mut db), STOOLAP_OK);
    db
}

unsafe fn exec(db: *mut StoolapDB, sql: &str) {
    let s = cstr(sql);
    let rc = stoolap_exec(db, s.as_ptr(), std::ptr::null_mut());
    assert_eq!(rc, STOOLAP_OK, "exec failed: {}", sql);
}

#[test]
fn audit_text_column_realloc_per_row() {
    unsafe {
        let db = open();
        exec(
            db,
            "CREATE TABLE t (id INTEGER PRIMARY KEY, s1 TEXT, s2 TEXT, s3 TEXT)",
        );
        // Insert 10_000 rows in one transaction.
        exec(db, "BEGIN");
        for i in 0..10_000 {
            let q =
                format!("INSERT INTO t VALUES ({i}, 'value-a-{i}', 'value-b-{i}', 'value-c-{i}')");
            exec(db, &q);
        }
        exec(db, "COMMIT");

        // Warm up plan + executor caches.
        for _ in 0..2 {
            run_text_scan(db);
        }

        // Measure.
        ALLOCS.store(0, Ordering::Relaxed);
        BYTES.store(0, Ordering::Relaxed);
        ENABLED.store(true, Ordering::Relaxed);
        let t0 = Instant::now();
        let rows_seen = run_text_scan(db);
        let elapsed = t0.elapsed();
        ENABLED.store(false, Ordering::Relaxed);
        let allocs = ALLOCS.load(Ordering::Relaxed);
        let bytes = BYTES.load(Ordering::Relaxed);

        eprintln!("---- TEXT COLUMN SCAN (3 text cols, 10k rows) ----");
        eprintln!("rows iterated:     {rows_seen}");
        eprintln!("elapsed:           {elapsed:?}");
        eprintln!("alloc count:       {allocs}");
        eprintln!("alloc bytes:       {bytes}");
        eprintln!("alloc per row:     {:.2}", allocs as f64 / rows_seen as f64);
        eprintln!(
            "alloc per col-read:{:.2}",
            allocs as f64 / (rows_seen as f64 * 3.0)
        );

        // Compare: int-only scan over same rowset.
        ALLOCS.store(0, Ordering::Relaxed);
        BYTES.store(0, Ordering::Relaxed);
        ENABLED.store(true, Ordering::Relaxed);
        let t0 = Instant::now();
        let int_rows = run_int_scan(db);
        let int_elapsed = t0.elapsed();
        ENABLED.store(false, Ordering::Relaxed);
        let int_allocs = ALLOCS.load(Ordering::Relaxed);
        let int_bytes = BYTES.load(Ordering::Relaxed);

        eprintln!("---- INT COLUMN SCAN (1 int col, 10k rows) ----");
        eprintln!("rows iterated:     {int_rows}");
        eprintln!("elapsed:           {int_elapsed:?}");
        eprintln!("alloc count:       {int_allocs}");
        eprintln!("alloc bytes:       {int_bytes}");
        eprintln!(
            "alloc per row:     {:.2}",
            int_allocs as f64 / int_rows as f64
        );

        stoolap_close(db);
    }
}

unsafe fn run_text_scan(db: *mut StoolapDB) -> usize {
    let sql = cstr("SELECT s1, s2, s3 FROM t");
    let mut rows: *mut StoolapRows = std::ptr::null_mut();
    assert_eq!(stoolap_query(db, sql.as_ptr(), &mut rows), STOOLAP_OK);
    let mut n = 0usize;
    loop {
        let rc = stoolap_rows_next(rows);
        if rc == STOOLAP_DONE {
            break;
        }
        assert_eq!(rc, STOOLAP_ROW);
        // Read each text column. This is the FFI hot path being audited.
        for col in 0..3 {
            let mut len: i64 = 0;
            let p: *const c_char = stoolap_rows_column_text(rows, col, &mut len);
            assert!(!p.is_null());
        }
        n += 1;
    }
    stoolap_rows_close(rows);
    n
}

unsafe fn run_int_scan(db: *mut StoolapDB) -> usize {
    let sql = cstr("SELECT id FROM t");
    let mut rows: *mut StoolapRows = std::ptr::null_mut();
    assert_eq!(stoolap_query(db, sql.as_ptr(), &mut rows), STOOLAP_OK);
    let mut n = 0usize;
    loop {
        let rc = stoolap_rows_next(rows);
        if rc == STOOLAP_DONE {
            break;
        }
        assert_eq!(rc, STOOLAP_ROW);
        let _ = stoolap_rows_column_int64(rows, 0);
        n += 1;
    }
    stoolap_rows_close(rows);
    n
}

#[test]
fn audit_repeated_query_cstring_cost() {
    // Verifies Fix #5: column-name CStrings are reused across calls when
    // the underlying Rows return the same CompactArc<Vec<String>>.
    unsafe {
        let db = open();
        exec(
            db,
            "CREATE TABLE wide (id INTEGER PRIMARY KEY, a TEXT, b TEXT, c TEXT, d TEXT, e TEXT)",
        );
        exec(db, "INSERT INTO wide VALUES (1, 'a', 'b', 'c', 'd', 'e')");

        // Warm up parser/plan caches.
        for _ in 0..2 {
            run_select_open_close(db);
        }

        let n_iters = 5_000;
        ALLOCS.store(0, Ordering::Relaxed);
        BYTES.store(0, Ordering::Relaxed);
        ENABLED.store(true, Ordering::Relaxed);
        let t0 = Instant::now();
        for _ in 0..n_iters {
            run_select_open_close(db);
        }
        let elapsed = t0.elapsed();
        ENABLED.store(false, Ordering::Relaxed);
        let allocs = ALLOCS.load(Ordering::Relaxed);
        let bytes = BYTES.load(Ordering::Relaxed);

        eprintln!("---- REPEATED ad-hoc SELECT (5 cols, {n_iters}x) ----");
        eprintln!("elapsed:           {elapsed:?}");
        eprintln!("alloc count:       {allocs}");
        eprintln!("alloc bytes:       {bytes}");
        eprintln!("alloc per query:   {:.2}", allocs as f64 / n_iters as f64);
        eprintln!(
            "us per query:      {:.2}",
            elapsed.as_micros() as f64 / n_iters as f64
        );

        stoolap_close(db);
    }
}

unsafe fn run_select_open_close(db: *mut StoolapDB) {
    let sql = cstr("SELECT a, b, c, d, e FROM wide WHERE id = 1");
    let mut rows: *mut StoolapRows = std::ptr::null_mut();
    assert_eq!(stoolap_query(db, sql.as_ptr(), &mut rows), STOOLAP_OK);
    while stoolap_rows_next(rows) == STOOLAP_ROW {}
    stoolap_rows_close(rows);
}

/// Regression: a hostile or accidental out-of-range column index passed
/// to `stoolap_rows_column_text` must not trigger a giant
/// `text_cache.resize_with(idx + 1, ...)` allocation. Returns NULL with
/// near-zero allocator activity.
#[test]
fn audit_column_text_oob_index_is_cheap() {
    unsafe {
        let db = open();
        exec(db, "CREATE TABLE t (id INTEGER PRIMARY KEY)");
        exec(db, "INSERT INTO t VALUES (1)");

        let sql = cstr("SELECT id FROM t");
        let mut rows: *mut StoolapRows = std::ptr::null_mut();
        assert_eq!(stoolap_query(db, sql.as_ptr(), &mut rows), STOOLAP_OK);
        assert_eq!(stoolap_rows_next(rows), STOOLAP_ROW);

        ALLOCS.store(0, Ordering::Relaxed);
        BYTES.store(0, Ordering::Relaxed);
        ENABLED.store(true, Ordering::Relaxed);

        // 1-column result, ask for column 1_000_000.
        let mut len: i64 = -1;
        let p = stoolap_rows_column_text(rows, 1_000_000, &mut len);
        assert!(p.is_null(), "OOB index must return NULL");

        // i32::MAX as well — pre-fix this would attempt to grow a
        // ~17-billion-byte Vec slot.
        let p2 = stoolap_rows_column_text(rows, i32::MAX, &mut len);
        assert!(p2.is_null(), "i32::MAX index must return NULL");

        ENABLED.store(false, Ordering::Relaxed);
        let bytes = BYTES.load(Ordering::Relaxed);
        let allocs = ALLOCS.load(Ordering::Relaxed);

        eprintln!("---- OOB column_text ----\nallocs: {allocs}\nbytes:  {bytes}");
        // A correct implementation should be byte-zero here. Cap at a
        // small budget to keep the assertion robust against future
        // unrelated-but-incidental allocations on the error path.
        assert!(
            bytes < 1024,
            "OOB column_text leaked {bytes} bytes — should be near zero"
        );

        stoolap_rows_close(rows);
        stoolap_close(db);
    }
}

#[test]
fn audit_json_param_validation_cost() {
    unsafe {
        let db = open();
        exec(db, "CREATE TABLE j (id INTEGER PRIMARY KEY, payload JSON)");

        let json_blob: String = format!(
            "{{\"a\":1,\"b\":2,\"c\":[1,2,3,4,5,6,7,8,9,10],\"d\":\"{}\"}}",
            "x".repeat(200)
        );
        let json_cstr = CString::new(json_blob.clone()).unwrap();

        let n_iters = 5_000;

        ALLOCS.store(0, Ordering::Relaxed);
        BYTES.store(0, Ordering::Relaxed);
        ENABLED.store(true, Ordering::Relaxed);
        let t0 = Instant::now();
        for i in 0..n_iters {
            let mut params: [StoolapValue; 2] = std::mem::zeroed();
            params[0].value_type = STOOLAP_TYPE_INTEGER;
            params[0].v.integer = i as i64;
            params[1].value_type = STOOLAP_TYPE_JSON;
            params[1].v.text = StoolapTextData {
                ptr: json_cstr.as_ptr(),
                len: json_blob.len() as i64,
            };
            let sql = cstr("INSERT INTO j VALUES ($1, $2)");
            let rc =
                stoolap_exec_params(db, sql.as_ptr(), params.as_ptr(), 2, std::ptr::null_mut());
            assert_eq!(rc, STOOLAP_OK);
        }
        let elapsed = t0.elapsed();
        ENABLED.store(false, Ordering::Relaxed);
        let allocs = ALLOCS.load(Ordering::Relaxed);
        let bytes = BYTES.load(Ordering::Relaxed);

        eprintln!("---- JSON PARAM VALIDATION ({n_iters} INSERTs) ----");
        eprintln!("elapsed:           {elapsed:?}");
        eprintln!("alloc count:       {allocs}");
        eprintln!("alloc bytes:       {bytes}");
        eprintln!("alloc per insert:  {:.2}", allocs as f64 / n_iters as f64);
        eprintln!(
            "us per insert:     {:.2}",
            elapsed.as_micros() as f64 / n_iters as f64
        );

        stoolap_close(db);
    }
}
