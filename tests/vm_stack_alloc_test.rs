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

//! A run of the expression VM allocates nothing for the values it stacks

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::core::{Row, Value};
use stoolap::executor::expression::{compile_expression, ExecuteContext, ExprVM};
use stoolap::parser::ast::Statement;

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

/// The allocations of ten runs of `predicate` over one row of 20 integer
/// columns c0 to c19, after a first run
fn runs(predicate: &str) -> usize {
    let columns: Vec<String> = (0..20).map(|i| format!("c{i}")).collect();
    let row = Row::from_values((0..20).map(Value::integer).collect());
    let sql = format!("SELECT * FROM t WHERE {predicate}");
    let Statement::Select(select) = stoolap::parser::parse_sql(&sql).unwrap().remove(0) else {
        panic!("not a SELECT: {sql}");
    };
    let program = compile_expression(select.where_clause.as_deref().unwrap(), &columns).unwrap();
    let mut vm = ExprVM::new();
    let ctx = ExecuteContext::new(&row);
    vm.execute(&program, &ctx).unwrap();
    let before = ALLOCATIONS.with(Cell::get);
    for _ in 0..10 {
        vm.execute(&program, &ctx).unwrap();
    }
    ALLOCATIONS.with(Cell::get) - before
}

#[test]
fn an_in_list_of_columns_runs_without_allocating() {
    let members: Vec<String> = (1..20).map(|i| format!("c{i}")).collect();
    let predicate = format!("c0 IN ({})", members.join(", "));
    assert_eq!(runs(&predicate), 0, "{predicate}");
}

#[test]
fn a_case_of_many_arms_runs_without_allocating() {
    let arms: Vec<String> = (1..=16)
        .map(|i| format!("WHEN c0 = {i} THEN {i}"))
        .collect();
    let predicate = format!("CASE {} ELSE 0 END = 0", arms.join(" "));
    assert_eq!(runs(&predicate), 0, "{predicate}");
}
