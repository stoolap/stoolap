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

//! Writing an INTEGER, a BOOLEAN or a TEXT value as text into a buffer
//! allocates nothing of its own, as `||` does for every row

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fmt::Write;

use stoolap::core::Value;

struct ThreadCounting;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for ThreadCounting {
    // SAFETY: forwards to the system allocator; the counter is a const
    // thread-local Cell, which never allocates
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|n| n.set(n.get() + 1));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.with(|n| n.set(n.get() + 1));
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: ThreadCounting = ThreadCounting;

#[test]
fn test_a_value_written_as_text_allocates_nothing_of_its_own() {
    let values = [
        Value::Integer(i64::MIN),
        Value::Integer(42),
        Value::Boolean(true),
        Value::text("a text longer than the inline limit"),
    ];
    let mut out = String::with_capacity(256);
    for value in &values {
        out.clear();
        let before = ALLOCATIONS.with(Cell::get);
        write!(out, "{}", value).unwrap();
        let allocated = ALLOCATIONS.with(Cell::get) - before;
        assert_eq!(
            allocated, 0,
            "writing {value:?} allocated {allocated} times"
        );
    }
}

#[test]
fn test_owning_a_formatted_text_costs_no_more_than_writing_it() {
    let timestamp = Value::text("2024-01-01 12:00:00.123456789")
        .into_coerce_to_type(stoolap::core::DataType::Timestamp);
    let values = [Value::Float(1e20), Value::Float(1.5), timestamp];
    let mut out = String::with_capacity(256);
    for value in &values {
        out.clear();
        let before = ALLOCATIONS.with(Cell::get);
        write!(out, "{}", value).unwrap();
        let written = ALLOCATIONS.with(Cell::get) - before;
        let before = ALLOCATIONS.with(Cell::get);
        let owned = value.as_string();
        let allocated = ALLOCATIONS.with(Cell::get) - before;
        assert_eq!(owned.as_deref(), Some(out.as_str()));
        assert!(
            allocated <= written,
            "owning {value:?} allocated {allocated} times, writing it {written}"
        );
    }
}
