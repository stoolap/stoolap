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

//! A volume's row ids and its column payload are one list: the builder
//! refuses rows out of row id order rather than reorder ids alone.

use stoolap::core::{DataType, Row, SchemaBuilder, Value};
use stoolap::storage::volume::writer::VolumeBuilder;

fn builder_with(rows: &[(i64, &str)]) -> VolumeBuilder {
    let schema = SchemaBuilder::new("t")
        .column("id", DataType::Integer, false, true)
        .column("name", DataType::Text, false, false)
        .build();
    let mut builder = VolumeBuilder::new(&schema);
    for (id, name) in rows {
        builder.add_row(
            *id,
            &Row::from_values(vec![Value::Integer(*id), Value::text(*name)]),
        );
    }
    builder
}

#[test]
fn rows_added_in_row_id_order_keep_their_own_payload() {
    let volume = builder_with(&[(1, "a"), (2, "b"), (3, "c")])
        .finish()
        .unwrap();
    let ids = volume.row_ids().unwrap();
    assert_eq!(ids, &[1, 2, 3]);
    for (i, id) in ids.iter().enumerate() {
        let row = volume.get_row(i).unwrap();
        assert_eq!(
            row.get(0),
            Some(&Value::Integer(*id)),
            "row {i} is listed under id {id} but carries {row:?}"
        );
    }
}

#[test]
fn rows_added_out_of_row_id_order_are_refused() {
    let err = builder_with(&[(3, "c"), (1, "a"), (2, "b")])
        .finish()
        .err()
        .expect("rows out of row id order were accepted");
    assert!(err.to_string().contains("row id order"), "{err}");
}

#[test]
fn a_repeated_row_id_is_refused() {
    assert!(builder_with(&[(1, "a"), (1, "b")]).finish().is_err());
}
