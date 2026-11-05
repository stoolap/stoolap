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

//! Decoding a sealed group of a wide JSON column keeps the decompressed
//! block as the column's bytes, copying none of them

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::Database;

struct ThreadCounting;

thread_local! {
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for ThreadCounting {
    // SAFETY: forwards to the system allocator; the counter is a const
    // thread-local Cell, which never allocates
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        BYTES.with(|b| b.set(b.get() + layout.size()));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        BYTES.with(|b| b.set(b.get() + new_size));
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: ThreadCounting = ThreadCounting;

const ROWS: usize = 2_000;
const WIDTH: usize = 2_000;

fn text(i: usize) -> String {
    (0..WIDTH)
        .map(|j| (b'a' + ((i * 7 + j) % 26) as u8) as char)
        .collect()
}

#[test]
fn a_wide_json_group_decodes_into_one_buffer() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/bytes_block_decode", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, doc JSON)", ())
        .unwrap();
    db.execute("BEGIN", ()).unwrap();
    for i in 0..ROWS {
        let doc = match i % 10 {
            3 => "NULL".to_string(),
            _ => format!("'{{\"k\":\"{}\"}}'", text(i)),
        };
        db.execute(&format!("INSERT INTO t VALUES ({i}, {doc})"), ())
            .unwrap();
    }
    db.execute("COMMIT", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    drop(db);
    let db = Database::open(&dsn).unwrap();
    // With no budget the read decodes the group again
    db.execute("PRAGMA GROUP_CACHE_MB = 0", ()).unwrap();
    let sql = "SELECT doc FROM t WHERE id = 1234";
    assert_eq!(db.query(sql, ()).unwrap().count(), 1);
    let before = BYTES.with(Cell::get);
    assert_eq!(db.query(sql, ()).unwrap().count(), 1);
    let allocated = BYTES.with(Cell::get) - before;
    // Nine rows in ten hold a document of WIDTH bytes and a little more
    let blob = ROWS / 10 * 9 * WIDTH;
    assert!(
        allocated < blob * 3 / 2,
        "{allocated} bytes allocated to read one row of a group of {blob} document bytes"
    );
    let doc = |id: usize| -> Option<String> {
        db.query_one(&format!("SELECT doc FROM t WHERE id = {id}"), ())
            .unwrap()
    };
    for id in [0, 1233, 1234, ROWS - 1] {
        let expected = (id % 10 != 3).then(|| format!("{{\"k\":\"{}\"}}", text(id)));
        assert_eq!(doc(id), expected, "row {id}");
    }
}
