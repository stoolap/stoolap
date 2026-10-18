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

//! A function that tries an INTEGER reading and falls back to FLOAT
//! allocates nothing for a number outside the INTEGER range

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::core::Value;
use stoolap::functions::scalar::{AbsFunction, ModFunction};
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
fn test_a_float_fallback_outside_the_integer_range_allocates_nothing() {
    let cases: [(&str, Box<dyn ScalarFunction>, Vec<Value>); 2] = [
        ("ABS", Box::new(AbsFunction), vec![Value::Float(1e30)]),
        (
            "MOD",
            Box::new(ModFunction),
            vec![Value::text("99999999999999999999"), Value::Integer(7)],
        ),
    ];
    for (name, function, args) in cases {
        let before = BYTES.with(Cell::get);
        let result = function.evaluate(&args);
        let allocated = BYTES.with(Cell::get) - before;
        assert!(
            matches!(result, Ok(Value::Float(_))),
            "{name} gives a FLOAT"
        );
        assert_eq!(allocated, 0, "{name} allocated {allocated} bytes");
    }
}
