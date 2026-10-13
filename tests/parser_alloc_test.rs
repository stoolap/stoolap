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

//! Parsing allocates for the tree it builds, not for operator precedence,
//! reserved keyword checks, clause names or a copy of the input

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::parser::Parser;

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

fn parse_allocations(sql: &str) -> usize {
    let before = ALLOCATIONS.with(Cell::get);
    let program = Parser::new(sql).parse_program().unwrap();
    let allocations = ALLOCATIONS.with(Cell::get) - before;
    drop(program);
    allocations
}

#[test]
fn test_parse_allocates_only_for_the_tree() {
    for (sql, most) in [
        ("SELECT * FROM users WHERE id = 42", 9),
        (
            "SELECT symbol, date_trunc('hour', ts) AS h, AVG(close), MAX(high), MIN(low), \
             SUM(volume) FROM candles WHERE ts >= 1700000000 AND ts < 1700086400 \
             AND symbol IN ('A','B','C') GROUP BY symbol, h HAVING COUNT(*) > 10 \
             ORDER BY h DESC LIMIT 100",
            45,
        ),
        (
            "SELECT o.id, c.name, SUM(i.qty * i.price) AS total FROM orders o \
             JOIN customers c ON o.customer_id = c.id LEFT JOIN items i ON i.order_id = o.id \
             WHERE o.status = 'open' AND c.region = $1 GROUP BY o.id, c.name \
             ORDER BY total DESC LIMIT 20",
            67,
        ),
    ] {
        parse_allocations(sql);
        let allocations = parse_allocations(sql);
        assert!(
            allocations <= most,
            "{allocations} allocations, at most {most} expected, parsing {sql}"
        );
    }
}
