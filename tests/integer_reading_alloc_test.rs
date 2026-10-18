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

//! A path that reads a number outside the INTEGER range as NULL, false or
//! a FLOAT fallback allocates nothing; only a raised error builds a message

// The jemalloc feature, on by default, sets the library's own global allocator
#![cfg(not(feature = "jemalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::core::{DataType, Operator, Row, SchemaBuilder, Value};
use stoolap::functions::scalar::{AbsFunction, ModFunction};
use stoolap::functions::ScalarFunction;
use stoolap::{CastExpr, CompoundExpr, Expression};

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

fn bytes_for(f: impl FnOnce()) -> usize {
    let before = BYTES.with(Cell::get);
    f();
    BYTES.with(Cell::get) - before
}

#[test]
fn test_a_conversion_to_null_outside_the_integer_range_allocates_nothing() {
    let text = Value::text("99999999999999999999");
    let float = Value::Float(1e30);
    for value in [&text, &float] {
        let mut coerced = None;
        let allocated = bytes_for(|| coerced = Some(value.coerce_to_type(DataType::Integer)));
        assert_eq!(coerced, Some(Value::Null(DataType::Integer)), "{value:?}");
        assert_eq!(allocated, 0, "coerce_to_type of {value:?}");

        let owned = value.clone();
        let mut coerced = None;
        let allocated = bytes_for(|| coerced = Some(owned.into_coerce_to_type(DataType::Integer)));
        assert_eq!(coerced, Some(Value::Null(DataType::Integer)), "{value:?}");
        assert_eq!(allocated, 0, "into_coerce_to_type of {value:?}");
    }

    let schema = SchemaBuilder::new("t")
        .add_primary_key("id", DataType::Integer)
        .add("s", DataType::Text)
        .build();
    let mut expr = CompoundExpr::new(
        CastExpr::new("s", DataType::Integer),
        Operator::Gt,
        Value::Integer(0),
    );
    expr.prepare_for_schema(&schema);
    let row = Row::from_values(vec![Value::Integer(1), text.clone()]);
    let mut matched = true;
    let allocated = bytes_for(|| matched = expr.evaluate_fast(&row));
    assert!(!matched, "a value outside the range matches nothing");
    assert_eq!(allocated, 0, "CompoundExpr::evaluate_fast");
}
