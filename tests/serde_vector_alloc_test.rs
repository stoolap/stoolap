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

//! A VECTOR column deserialized into a `Vec<f32>` allocates the destination
//! and nothing else, and a TIMESTAMP into a `DateTime` allocates nothing,
//! counted in the bytes each deserialization allocates on its thread

// The mimalloc feature sets the library's own global allocator
#![cfg(not(feature = "mimalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use chrono::{DateTime, Utc};
use stoolap::Database;

struct ThreadCounting;

thread_local! {
    static ALLOCATED: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for ThreadCounting {
    // SAFETY: forwards to the system allocator; the counter is a const
    // thread-local Cell, which never allocates
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATED.with(|a| a.set(a.get() + layout.size()));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATED.with(|a| a.set(a.get() + new_size));
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: ThreadCounting = ThreadCounting;

const DIMENSIONS: usize = 1536;

#[test]
fn a_vector_read_into_a_vec_allocates_only_the_destination() {
    let db = Database::open_in_memory().unwrap();
    db.execute(
        &format!("CREATE TABLE embeddings (id INTEGER PRIMARY KEY, v VECTOR({DIMENSIONS}))"),
        (),
    )
    .unwrap();
    let literal = (0..DIMENSIONS)
        .map(|i| format!("{i}.5"))
        .collect::<Vec<_>>()
        .join(", ");
    db.execute(
        &format!("INSERT INTO embeddings VALUES (1, '[{literal}]')"),
        (),
    )
    .unwrap();
    let row = db
        .query("SELECT v FROM embeddings", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();

    // Once warm, so nothing initialized lazily is counted
    let (warm,): (Vec<f32>,) = row.deserialize().unwrap();
    assert_eq!(warm.len(), DIMENSIONS);

    let before = ALLOCATED.with(Cell::get);
    let (values,): (Vec<f32>,) = row.deserialize().unwrap();
    let allocated = ALLOCATED.with(Cell::get) - before;

    assert_eq!(values.len(), DIMENSIONS);
    assert_eq!(values[1], 1.5);
    assert_eq!(
        allocated,
        DIMENSIONS * std::mem::size_of::<f32>(),
        "only the destination buffer may be allocated"
    );
}

#[test]
fn a_timestamp_read_into_a_datetime_allocates_nothing() {
    let db = Database::open_in_memory().unwrap();
    db.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, at TIMESTAMP)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO events VALUES (1, '2026-09-26T10:30:00.123456789Z')",
        (),
    )
    .unwrap();
    let row = db
        .query("SELECT at FROM events", ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let (warm,): (DateTime<Utc>,) = row.deserialize().unwrap();

    let before = ALLOCATED.with(Cell::get);
    let (at,): (DateTime<Utc>,) = row.deserialize().unwrap();
    let allocated = ALLOCATED.with(Cell::get) - before;

    assert_eq!(at, warm);
    assert_eq!(at.to_rfc3339(), "2026-09-26T10:30:00.123456789+00:00");
    assert_eq!(allocated, 0, "the RFC 3339 text is written on the stack");

    // A String destination costs exactly its text
    let before = ALLOCATED.with(Cell::get);
    let (text,): (String,) = row.deserialize().unwrap();
    assert_eq!(ALLOCATED.with(Cell::get) - before, text.len());
}
