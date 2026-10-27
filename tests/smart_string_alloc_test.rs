// Copyright 2026 Stoolap Contributors
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

//! A string past fifteen bytes takes one allocation, however it is built,
//! and a clone takes none

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::core::Value;
use stoolap::SmartString;

struct ThreadCounting;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for ThreadCounting {
    // SAFETY: forwards to the system allocator; the counter is a const
    // thread-local Cell, which never allocates
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|a| a.set(a.get() + 1));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.with(|a| a.set(a.get() + 1));
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static COUNTING: ThreadCounting = ThreadCounting;

fn allocations<T>(build: impl FnOnce() -> T) -> (usize, T) {
    let before = ALLOCATIONS.with(Cell::get);
    let built = build();
    (ALLOCATIONS.with(Cell::get) - before, built)
}

#[test]
fn a_string_past_fifteen_bytes_takes_one_allocation() {
    let fifteen = "abcdefghijklmno";
    let sixteen = "abcdefghijklmnop";
    assert_eq!(allocations(|| SmartString::new(fifteen)).0, 0);
    assert_eq!(allocations(|| SmartString::new(sixteen)).0, 1);
    assert_eq!(allocations(|| Value::text(sixteen)).0, 1);
    assert_eq!(
        allocations(|| SmartString::concat("abcdefgh", "ijklmnop")).0,
        1
    );
    let owned = sixteen.to_string();
    assert_eq!(
        allocations(|| SmartString::from_string(owned)).0,
        1,
        "an owned String is copied into one block"
    );
    let (_, long) = allocations(|| SmartString::new(sixteen));
    let (clone_allocations, mut clone) = allocations(|| long.clone());
    assert_eq!((clone_allocations, clone.as_str()), (0, sixteen));
    assert_eq!(
        (
            allocations(|| clone.make_ascii_uppercase()).0,
            clone.as_str()
        ),
        (1, "ABCDEFGHIJKLMNOP"),
        "a shared string is uppercased into one new block"
    );
    assert_eq!(long.as_str(), sixteen);
}
