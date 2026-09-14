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

//! A volume's columns can become resident while a scan is under way: a
//! reader of every column promotes them, after which the scanner stops
//! refreshing its per-group cache. The rows past the cached group must
//! then come from the resident columns, not from the cache with an index
//! beyond its group.

use std::sync::Arc;
use stoolap::core::{DataType, Row, SchemaBuilder, Value};
use stoolap::storage::expression::{ComparisonExpr, Expression};
use stoolap::storage::traits::Scanner;
use stoolap::storage::volume::column::ROW_GROUP_SIZE;
use stoolap::storage::volume::scanner::VolumeScanner;
use stoolap::storage::volume::writer::{CompressedBlockStore, LazyColumns, VolumeBuilder};

#[test]
fn a_scan_survives_its_volume_becoming_resident_between_two_rows() {
    let schema = SchemaBuilder::new("t")
        .column("id", DataType::Integer, false, true)
        .column("v", DataType::Integer, false, false)
        .build();
    let rows = ROW_GROUP_SIZE + 4_464;
    let mut builder = VolumeBuilder::new(&schema);
    for id in 1..=rows as i64 {
        builder.add_row(
            id,
            &Row::from_values(vec![Value::Integer(id), Value::Integer(id % 7)]),
        );
    }
    let mut volume = builder.finish().unwrap();
    let store =
        CompressedBlockStore::compress_columns(&volume.columns, &[DataType::Integer; 2], rows)
            .unwrap();
    volume.columns = LazyColumns::deferred(store, vec![DataType::Integer; 2]);
    let volume = Arc::new(volume);
    assert!(!volume.columns.is_eager());

    let mut filter: Box<dyn Expression> = Box::new(ComparisonExpr::gte("v", Value::Integer(0)));
    filter.prepare_for_schema(&schema);
    let mut scanner = VolumeScanner::new(Arc::clone(&volume), vec![0, 1], None).unwrap();
    scanner.set_filter(filter);
    // The first row loads the cache of the first group
    assert!(scanner.next());
    assert_eq!(scanner.current_row_id(), 1);
    // Every column read once: the columns are resident from here on
    volume.columns.get(0).unwrap();
    volume.columns.get(1).unwrap();
    assert!(volume.columns.is_eager());
    let mut seen = 1;
    let mut last = 1;
    while scanner.next() {
        seen += 1;
        last = scanner.current_row_id();
        assert_eq!(scanner.row()[1], Value::Integer(last % 7));
    }
    assert!(scanner.err().is_none(), "{:?}", scanner.err());
    assert_eq!((seen, last), (rows, rows as i64));
}
