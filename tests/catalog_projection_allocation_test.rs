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

#![cfg(not(feature = "mimalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use stoolap::core::{DataType, Schema, SchemaColumn, Value};
use stoolap::storage::catalog::{
    ColumnId, Incarnation, TableId, TableIdentity, TableSchemaHistory,
};

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
    static MAX_REQUEST: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

fn count(size: usize) {
    if COUNTING.try_with(Cell::get).unwrap_or(false) {
        let _ = CALLS.try_with(|calls| calls.set(calls.get() + 1));
        let _ = MAX_REQUEST.try_with(|maximum| maximum.set(maximum.get().max(size)));
    }
}

// SAFETY: Ownership, layouts and alignment are passed unchanged to System.
// The thread-local counters do not allocate or access allocation contents.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count(size);
        unsafe { System.realloc(pointer, layout, size) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn borrowed_projection_and_incarnation_sharing_do_not_allocate() {
    let identity = TableIdentity::new(TableId::new(1).unwrap(), Incarnation::FIRST);
    let first_schema = Schema::new(
        "t",
        vec![
            SchemaColumn::new(0, "id", DataType::Integer, false, true),
            SchemaColumn::new(1, "payload", DataType::Text, true, false),
        ],
    );
    let history = TableSchemaHistory::new(
        identity,
        0,
        first_schema.clone(),
        vec![ColumnId::new(1).unwrap(), ColumnId::new(2).unwrap()],
    )
    .unwrap();
    let mut next_schema = first_schema;
    let mut added = SchemaColumn::new(2, "introduced", DataType::Text, false, false);
    added.default_value = Some(Value::text("d".repeat(65_536)));
    next_schema.add_column(added).unwrap();
    let history = history
        .with_revision(
            identity,
            1,
            next_schema,
            vec![
                ColumnId::new(1).unwrap(),
                ColumnId::new(2).unwrap(),
                ColumnId::new(3).unwrap(),
            ],
        )
        .unwrap();
    let plan = history.projection(identity, 0).unwrap();
    let input = [Value::Integer(7), Value::text("p".repeat(65_536))];
    let default_pointer =
        plan.project(identity, 0, &input).unwrap().get(2).unwrap() as *const Value;

    struct StopCounting;
    impl Drop for StopCounting {
        fn drop(&mut self) {
            COUNTING.with(|counting| counting.set(false));
        }
    }
    CALLS.with(|calls| calls.set(0));
    COUNTING.with(|counting| counting.set(true));
    let stop = StopCounting;
    let snapshot = history.clone();
    let next_incarnation = snapshot.checked_next_incarnation().unwrap();
    assert_eq!(next_incarnation.column_high_water_mark(), 3);
    for _ in 0..1000 {
        let projected = plan.project(identity, 0, &input).unwrap();
        assert_eq!(projected.len(), 3);
        assert!(std::ptr::eq(projected.get(1).unwrap(), &input[1]));
        assert_eq!(projected.get(2).unwrap() as *const Value, default_pointer);
        assert_eq!(projected.iter().filter_map(Value::as_int64).sum::<i64>(), 7);
        assert_eq!(projected.iter().rev().count(), 3);
    }
    drop(stop);
    assert_eq!(CALLS.with(Cell::get), 0);
}

#[test]
fn catalog_streaming_encode_borrows_large_payload_without_allocating() {
    use std::io::{self, Write};
    use std::num::NonZeroU64;
    use stoolap::storage::catalog::codec::encode_into;
    use stoolap::storage::catalog::generation::*;

    let identity = TableIdentity::new(TableId::new(1).unwrap(), Incarnation::FIRST);
    let stamp = DdlStamp::new(1, 10).unwrap();
    let mut column = SchemaColumn::nullable(0, "payload", DataType::Text);
    column.default_value = Some(Value::text("p".repeat(1_048_576)));
    let history = TableSchemaHistory::new(
        identity,
        0,
        Schema::new("t", vec![column]),
        vec![ColumnId::new(1).unwrap()],
    )
    .unwrap();
    let table = CatalogTable {
        history,
        incarnations: vec![CatalogIncarnation {
            incarnation: Incarnation::FIRST,
            started_at: stamp,
            first_schema_version: 0,
            last_schema_version: 0,
            ended: None,
        }],
        names: vec![TableNameEvent {
            stamp,
            name: Some("t".into()),
        }],
        schema_events: vec![SchemaEvent {
            stamp,
            identity,
            version: 0,
        }],
        foreign_keys: vec![SchemaForeignKeys {
            schema_version: 0,
            bindings: vec![],
        }],
        indexes: vec![],
    };
    let catalog = CatalogGeneration::try_new(CatalogParts {
        generation: NonZeroU64::new(1).unwrap(),
        table_id_high_water_mark: 1,
        ddl_epoch_high_water_mark: 1,
        wal_observation_ceiling: 10,
        coverage: CatalogCoverage {
            through_lsn: 10,
            ddl_epoch_cut: 1,
            captured_mutations: vec![CatalogMutation {
                stamp,
                kind: DdlKind::CreateTable,
                effects: vec![CatalogEffect::Table {
                    identity,
                    schema_version: 0,
                }],
            }],
        },
        tables: vec![table],
        views: vec![],
    })
    .unwrap();
    let payload = catalog.tables()[0]
        .history
        .lookup(identity, 0)
        .unwrap()
        .schema()
        .columns[0]
        .default_value
        .as_ref()
        .unwrap()
        .as_str()
        .unwrap();
    struct BorrowedSink {
        pointer: *const u8,
        length: usize,
        bytes: usize,
        payload_writes: usize,
    }
    impl Write for BorrowedSink {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() == self.length {
                assert_eq!(bytes.as_ptr(), self.pointer);
                self.payload_writes += 1;
            }
            self.bytes += bytes.len();
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    struct StopCounting;
    impl Drop for StopCounting {
        fn drop(&mut self) {
            COUNTING.with(|counting| counting.set(false));
        }
    }
    let mut sink = BorrowedSink {
        pointer: payload.as_ptr(),
        length: payload.len(),
        bytes: 0,
        payload_writes: 0,
    };
    CALLS.with(|calls| calls.set(0));
    COUNTING.with(|counting| counting.set(true));
    let stop = StopCounting;
    let descriptor = encode_into(&catalog, &mut sink).unwrap();
    assert!(std::ptr::eq(descriptor.coverage(), catalog.coverage()));
    drop(stop);
    assert_eq!(CALLS.with(Cell::get), 0);
    assert_eq!(sink.payload_writes, 1);
    assert_eq!(descriptor.bytes(), sink.bytes as u64);
    assert!(sink.bytes > payload.len());
}

#[test]
fn catalog_enclosing_bytes_reject_large_counts_before_reserving() {
    use std::num::NonZeroU64;
    use stoolap::storage::catalog::codec::{decode_from, encode_into, CatalogDecodeLimits};
    use stoolap::storage::catalog::generation::*;
    let catalog = CatalogGeneration::try_new(CatalogParts {
        generation: NonZeroU64::new(1).unwrap(),
        table_id_high_water_mark: 0,
        ddl_epoch_high_water_mark: 0,
        wal_observation_ceiling: 0,
        coverage: CatalogCoverage::default(),
        tables: vec![],
        views: vec![],
    })
    .unwrap();
    let mut bytes = Vec::new();
    encode_into(&catalog, &mut bytes).unwrap();
    // Valid header/footer, but u32::MAX mutations cannot fit the enclosed body.
    bytes[80..84].copy_from_slice(&u32::MAX.to_le_bytes());
    let footer = bytes.len() - 4;
    let checksum = crc32fast::hash(&bytes[..footer]);
    bytes[footer..].copy_from_slice(&checksum.to_le_bytes());
    let limits = CatalogDecodeLimits {
        max_mutations: u32::MAX,
        max_records: u64::MAX,
        max_requested_metadata_bytes: u64::MAX,
        ..CatalogDecodeLimits::default()
    };
    struct StopCounting;
    impl Drop for StopCounting {
        fn drop(&mut self) {
            COUNTING.with(|counting| counting.set(false));
        }
    }
    MAX_REQUEST.with(|maximum| maximum.set(0));
    COUNTING.with(|counting| counting.set(true));
    let stop = StopCounting;
    let error = decode_from(&mut bytes.as_slice(), &limits).unwrap_err();
    drop(stop);
    assert!(error.to_string().contains("enclosing bytes"));
    assert!(MAX_REQUEST.with(Cell::get) < 1024);
}
