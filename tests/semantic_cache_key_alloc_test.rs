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

//! Looking a result up in the semantic cache allocates the same for a wide
//! table as for a narrow one

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::core::{Row, Value};
use stoolap::executor::semantic_cache::CacheStamp;
use stoolap::executor::{CacheLookupResult, SemanticCache};
use stoolap::parser::ast::{Expression, Statement};
use stoolap::parser::parse_sql;

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

const STAMP: CacheStamp = CacheStamp {
    publish: 1,
    schema: 1,
};

fn predicate() -> Expression {
    match parse_sql("SELECT * FROM t WHERE b = 1").unwrap().remove(0) {
        Statement::Select(select) => *select.where_clause.unwrap(),
        _ => unreachable!(),
    }
}

/// The allocations of an exact hit on `t`, cached with `width` columns
fn hit(width: usize) -> usize {
    let cache = SemanticCache::new();
    let columns: Vec<String> = (0..width).map(|i| format!("column_{i}")).collect();
    let row = Row::from_values(vec![Value::integer(1); width]);
    let predicate = predicate();
    cache.insert(
        "t",
        columns.clone(),
        vec![row],
        Some(predicate.clone()),
        STAMP,
    );
    let before = ALLOCATIONS.with(Cell::get);
    let found = cache.lookup("t", &columns, Some(&predicate), STAMP);
    let allocations = ALLOCATIONS.with(Cell::get) - before;
    assert!(matches!(found, CacheLookupResult::ExactHit(_)));
    allocations
}

#[test]
fn a_lookup_allocates_nothing_per_column() {
    let (narrow, wide) = (hit(2), hit(66));
    assert_eq!(
        wide, narrow,
        "{wide} allocations over 66 columns, {narrow} over 2"
    );
}

fn where_of(sql: &str) -> Expression {
    match parse_sql(sql).unwrap().remove(0) {
        Statement::Select(select) => *select.where_clause.unwrap(),
        _ => unreachable!(),
    }
}

/// The allocations of a lookup that matches none of `entries` ranges cached
/// on `t`, each asked whether it holds the new one
fn miss(entries: usize) -> usize {
    let cache = SemanticCache::new();
    let columns = vec!["a".to_string(), "b".to_string()];
    for i in 0..entries {
        let row = Row::from_values(vec![Value::integer(1); 2]);
        let predicate = where_of(&format!("SELECT * FROM t WHERE b > {i}"));
        cache.insert("t", columns.clone(), vec![row], Some(predicate), STAMP);
    }
    let predicate = where_of("SELECT * FROM t WHERE b < 0");
    let before = ALLOCATIONS.with(Cell::get);
    let found = cache.lookup("t", &columns, Some(&predicate), STAMP);
    let allocations = ALLOCATIONS.with(Cell::get) - before;
    assert!(matches!(found, CacheLookupResult::Miss));
    allocations
}

#[test]
fn a_miss_allocates_nothing_per_entry() {
    let (one, many) = (miss(1), miss(32));
    assert_eq!(
        many, one,
        "{many} allocations past 32 entries, {one} past one"
    );
}
