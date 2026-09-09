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

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::core::DataType;
use stoolap::storage::volume::v5::column_block::{
    BorrowedTextDictionary, ColumnBlockRef, ColumnEncodePlan, ColumnExpectation, ColumnIdentity,
    ColumnInput, ColumnLimits, MAX_DECODED_BYTES,
};

thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
}
struct Counting;
fn count() {
    let _ = TRACK.try_with(|track| {
        if track.get() {
            CALLS.with(|calls| calls.set(calls.get() + 1));
        }
    });
}
// Test-only instrumentation forwards unchanged to the system allocator.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count();
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOC: Counting = Counting;
struct Stop;
impl Drop for Stop {
    fn drop(&mut self) {
        TRACK.with(|track| track.set(false));
    }
}

#[test]
fn borrowed_dictionary_proof_plan_encode_and_parse_allocate_nothing() {
    for (rows, unique) in [(10, 4), (4096, 4), (4096, 4096)] {
        let texts: Vec<String> = if unique == 4 {
            vec![String::new(), "alpha".into(), "s".repeat(16384), "é".into()]
        } else {
            (0..unique).map(|i| format!("{i:04}")).collect()
        };
        let mut blob = Vec::new();
        let mut offsets = vec![0];
        for text in &texts {
            blob.extend_from_slice(text.as_bytes());
            offsets.push(blob.len() as u32);
        }
        let nulls: Vec<_> = (0..rows).map(|i| i % 7 == 0).collect();
        let ids: Vec<_> = (0..rows)
            .map(|i| {
                if nulls[i] {
                    u32::MAX
                } else {
                    (i % unique) as u32
                }
            })
            .collect();
        let mut output = vec![0; MAX_DECODED_BYTES];
        let expect = ColumnExpectation {
            identity: ColumnIdentity {
                physical_column: 2,
                group: 3,
                row_start: 10,
                row_count: rows as u32,
                data_type: DataType::Text,
            },
            vector_dimensions: None,
        };
        CALLS.with(|calls| calls.set(0));
        TRACK.with(|track| track.set(true));
        let stop = Stop;
        let dictionary = BorrowedTextDictionary::new(&blob, &offsets).unwrap();
        for _ in 0..3 {
            let input = ColumnInput::BorrowedDictionaryText {
                ids: &ids,
                dictionary: &dictionary,
            };
            let plan =
                ColumnEncodePlan::new(expect, &nulls, input, ColumnLimits::default()).unwrap();
            let written = plan.encode_into(&mut output).unwrap();
            let view =
                ColumnBlockRef::parse(&output[..written], expect, ColumnLimits::default()).unwrap();
            std::hint::black_box(view.cell(rows - 1));
            assert_eq!(
                view.dictionary_lookup(&texts[unique - 1]).unwrap(),
                Some(unique as u32 - 1)
            );
            assert!(ColumnEncodePlan::new(
                expect,
                &nulls,
                input,
                ColumnLimits {
                    dictionary_entries: 0,
                    ..ColumnLimits::default()
                }
            )
            .is_err());
        }
        assert!(BorrowedTextDictionary::new(b"\xff", &[0, 1]).is_err());
        assert!(BorrowedTextDictionary::new("é".as_bytes(), &[0, 1, 2]).is_err());
        assert!(BorrowedTextDictionary::new(b"a", &[1, 1]).is_err());
        assert!(BorrowedTextDictionary::new(b"aa", &[0, 1, 2]).is_err());
        drop(stop);
        assert_eq!(CALLS.with(Cell::get), 0, "rows={rows}, unique={unique}");
    }
}
