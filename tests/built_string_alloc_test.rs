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

//! A pad call refused for its size allocates no more than its own copies of
//! the arguments: the pad's characters and the result come after the check

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::core::Value;
use stoolap::functions::scalar::{LpadFunction, RpadFunction};
use stoolap::functions::ScalarFunction;

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

#[test]
fn test_a_refused_pad_allocates_only_its_argument_copies() {
    let pad = "x".repeat(10 * 1024 * 1024);
    let functions: [(&str, Box<dyn ScalarFunction>); 2] = [
        ("LPAD", Box::new(LpadFunction)),
        ("RPAD", Box::new(RpadFunction)),
    ];
    for (name, function) in functions {
        let args = [
            Value::text("a"),
            Value::Integer(i64::MAX),
            Value::text(pad.as_str()),
        ];
        let before = BYTES.with(Cell::get);
        assert!(function.evaluate(&args).is_err(), "{name} is refused");
        let allocated = BYTES.with(Cell::get) - before;
        assert!(
            allocated < 2 * pad.len(),
            "{name} allocated {allocated} bytes for a {} byte pad",
            pad.len()
        );
    }
}
