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

//! A LIMIT over an EXISTS set takes the few members its rows need, not the
//! whole set, counted in the bytes the query allocates on its thread

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

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

#[test]
fn a_limit_over_a_large_exists_set_takes_a_few_members() {
    let db = Database::open("memory://exists_limit_alloc").unwrap();
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_orders_user ON orders(user_id)", ())
        .unwrap();
    let users = db.prepare("INSERT INTO users VALUES ($1, 0)").unwrap();
    let orders = db
        .prepare("INSERT INTO orders VALUES ($1, $1, 600)")
        .unwrap();
    db.execute("BEGIN", ()).unwrap();
    for id in 1..=50_000i64 {
        users.execute((id,)).unwrap();
        orders.execute((id,)).unwrap();
    }
    db.execute("COMMIT", ()).unwrap();
    let query = db
        .prepare(
            "SELECT * FROM users u WHERE EXISTS \
             (SELECT 1 FROM orders o WHERE o.user_id = u.id AND o.amount > 500) LIMIT 10",
        )
        .unwrap();
    // The first run builds and caches the set of 50,000 members
    assert_eq!(query.query(()).unwrap().count(), 10);
    let before = ALLOCATED.with(Cell::get);
    let rows = query.query(()).unwrap().count();
    let allocated = ALLOCATED.with(Cell::get) - before;
    assert_eq!(rows, 10);
    assert!(
        allocated < 256 * 1024,
        "a LIMIT 10 over the cached set allocated {allocated} bytes"
    );
}
