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

use std::io::ErrorKind;
use std::panic::{catch_unwind, AssertUnwindSafe};
use stoolap::core::DataType;
use stoolap::storage::volume::column::{ColumnData, ROW_GROUP_SIZE};
use stoolap::storage::volume::writer::{CompressedBlockStore, LazyColumns};

fn store(block: Vec<u8>, decoded_len: usize, data_type: DataType) -> CompressedBlockStore {
    CompressedBlockStore::from_raw_blocks(
        vec![vec![block]],
        vec![vec![decoded_len]],
        vec![if data_type == DataType::Json { 6 } else { 1 }],
        vec![data_type],
        vec![data_type as u8],
        Vec::new(),
        Vec::new(),
        ROW_GROUP_SIZE,
        2,
    )
}

fn bytes_block(offset_count: u64) -> Vec<u8> {
    let mut raw = vec![0, 0];
    raw.extend_from_slice(&offset_count.to_le_bytes());
    for _ in 0..offset_count.min(3) {
        raw.extend_from_slice(&0u64.to_le_bytes());
        raw.extend_from_slice(&0u64.to_le_bytes());
    }
    raw.extend_from_slice(&0u64.to_le_bytes());
    raw
}

trait ColumnErrorKind {
    fn error_kind(self) -> Option<ErrorKind>;
}

impl ColumnErrorKind for ColumnData {
    fn error_kind(self) -> Option<ErrorKind> {
        None
    }
}

impl ColumnErrorKind for &ColumnData {
    fn error_kind(self) -> Option<ErrorKind> {
        None
    }
}

impl<T> ColumnErrorKind for std::io::Result<T> {
    fn error_kind(self) -> Option<ErrorKind> {
        self.err().map(|error| error.kind())
    }
}

impl ColumnErrorKind for stoolap::core::Row {
    fn error_kind(self) -> Option<ErrorKind> {
        None
    }
}

impl ColumnErrorKind for () {
    fn error_kind(self) -> Option<ErrorKind> {
        None
    }
}

impl<T> ColumnErrorKind for Vec<T> {
    fn error_kind(self) -> Option<ErrorKind> {
        None
    }
}

impl<T> ColumnErrorKind for Option<T> {
    fn error_kind(self) -> Option<ErrorKind> {
        None
    }
}

// Keep the guards executable when only production changes are reverted.
#[allow(dead_code)]
trait BaselineUnwrap: Sized {
    fn unwrap(self) -> Self {
        self
    }
}
impl<T> BaselineUnwrap for T {}

#[allow(dead_code)]
trait BaselineColumnAccess {
    fn get(&self, index: usize) -> std::io::Result<&ColumnData>;
}

impl BaselineColumnAccess for LazyColumns {
    fn get(&self, index: usize) -> std::io::Result<&ColumnData> {
        Ok(self.iter().nth(index).unwrap().unwrap())
    }
}

fn assert_io_failure<T: ColumnErrorKind>(read: impl FnOnce() -> T, kind: ErrorKind) {
    let outcome = catch_unwind(AssertUnwindSafe(|| read().error_kind()));
    assert!(outcome.is_ok(), "column failure panicked");
    assert_eq!(outcome.unwrap(), Some(kind));
}

fn assert_column_error(store: &CompressedBlockStore, col: usize, kind: ErrorKind) {
    for _ in 0..2 {
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            store.decompress_column(col).error_kind()
        }));
        assert!(outcome.is_ok(), "whole-column decode panicked");
        assert_eq!(outcome.unwrap(), Some(kind));
    }
}

fn two_group_store(
    first: Vec<u8>,
    second: Vec<u8>,
    lengths: Vec<usize>,
    data_type: DataType,
) -> CompressedBlockStore {
    let dictionary = data_type == DataType::Text;
    CompressedBlockStore::from_raw_blocks(
        vec![vec![first, second]],
        vec![lengths],
        vec![match data_type {
            DataType::Text => 5,
            DataType::Json => 6,
            _ => 1,
        }],
        vec![data_type],
        vec![data_type as u8],
        if dictionary {
            vec!["value".into()]
        } else {
            Vec::new()
        },
        if dictionary {
            vec![(0, 0, 1)]
        } else {
            Vec::new()
        },
        2,
        4,
    )
}

#[test]
fn whole_column_rejects_incorrect_lz4_output_length() {
    let store = store(lz4_flex::compress(&[0; 17]), 18, DataType::Integer);
    assert_column_error(&store, 0, ErrorKind::InvalidData);
}

#[test]
fn whole_column_rejects_late_short_lz4_output() {
    let store = two_group_store(
        lz4_flex::compress(&[0; 18]),
        lz4_flex::compress(&[0; 17]),
        vec![18, 18],
        DataType::Integer,
    );
    assert_column_error(&store, 0, ErrorKind::InvalidData);
}

#[test]
fn whole_column_rejects_late_trailing_payload() {
    for data_type in [DataType::Integer, DataType::Json] {
        let raw = if data_type == DataType::Json {
            bytes_block(2)
        } else {
            vec![0; 18]
        };
        let mut trailing = raw.clone();
        trailing.push(0);
        for compressed in [false, true] {
            let store = two_group_store(
                raw.clone(),
                if compressed {
                    lz4_flex::compress(&trailing)
                } else {
                    trailing.clone()
                },
                vec![raw.len(), trailing.len()],
                data_type,
            );
            assert_column_error(&store, 0, ErrorKind::InvalidData);
        }
    }
}

#[test]
fn whole_column_rejects_late_invalid_bytes_offsets() {
    for count in [1, 3, u64::MAX] {
        let raw = bytes_block(2);
        let bad = bytes_block(count);
        let lengths = vec![raw.len(), bad.len()];
        let store = two_group_store(raw, bad, lengths, DataType::Json);
        assert_column_error(&store, 0, ErrorKind::InvalidData);
    }
}

#[test]
fn whole_column_rejects_late_overflowing_blob_length() {
    let raw = bytes_block(2);
    let mut bad = raw.clone();
    let len_offset = bad.len() - 8;
    bad[len_offset..].copy_from_slice(&u64::MAX.to_le_bytes());
    let lengths = vec![raw.len(), bad.len()];
    let store = two_group_store(raw, bad, lengths, DataType::Json);
    assert_column_error(&store, 0, ErrorKind::InvalidData);
}

#[test]
fn whole_column_rejects_late_invalid_dictionary_ids() {
    let raw = vec![0; 10];
    let mut bad = raw.clone();
    bad[2..6].copy_from_slice(&1u32.to_le_bytes());
    let store = two_group_store(raw, bad, vec![10, 10], DataType::Text);
    assert_column_error(&store, 0, ErrorKind::InvalidData);
}

#[test]
fn whole_column_rejects_missing_group_lengths() {
    let store = two_group_store(vec![0; 18], vec![0; 18], vec![18], DataType::Integer);
    assert_column_error(&store, 0, ErrorKind::InvalidData);
}

#[test]
fn whole_column_rejects_inconsistent_row_geometry() {
    let store = CompressedBlockStore::from_raw_blocks(
        vec![vec![vec![0; 18]]],
        vec![vec![18]],
        vec![1],
        vec![DataType::Integer],
        vec![0],
        Vec::new(),
        Vec::new(),
        1,
        2,
    );
    assert_column_error(&store, 0, ErrorKind::InvalidData);
}

#[test]
fn whole_column_rejects_unrepresentable_decoded_length() {
    let store = store(vec![0xff], usize::MAX, DataType::Json);
    assert_column_error(&store, 0, ErrorKind::InvalidData);
}

#[test]
fn whole_column_rejects_out_of_range_indices() {
    let store = store(vec![0; 18], 18, DataType::Integer);
    for col in [1, usize::MAX] {
        assert_column_error(&store, col, ErrorKind::InvalidInput);
    }
}

#[test]
fn empty_whole_column_preserves_its_type() {
    let types = vec![DataType::Integer];
    let columns = LazyColumns::eager(
        vec![ColumnData::Int64 {
            values: Vec::new(),
            nulls: Vec::new(),
        }],
        types.clone(),
    );
    let store = CompressedBlockStore::compress_columns(&columns, &types, 0).unwrap();
    let decoded = LazyColumns::deferred(store, types);
    assert!(
        matches!(decoded.get(0).unwrap(), ColumnData::Int64 { values, nulls } if values.is_empty() && nulls.is_empty())
    );
}

fn assert_invalid_block(raw: Vec<u8>, data_type: DataType) {
    let store = store(raw.clone(), raw.len(), data_type);
    for _ in 0..2 {
        let outcome = catch_unwind(AssertUnwindSafe(|| store.group_column(0, 0)));
        assert!(outcome.is_ok(), "group access panicked");
        let error = outcome.unwrap().err().expect("invalid block was accepted");
        assert_eq!(error.kind(), ErrorKind::InvalidData);
    }
}

#[test]
fn bytes_group_rejects_missing_offsets() {
    assert_invalid_block(bytes_block(1), DataType::Json);
}

#[test]
fn bytes_group_rejects_extra_offsets() {
    assert_invalid_block(bytes_block(3), DataType::Json);
}

#[test]
fn bytes_group_rejects_oversized_offset_count_before_allocation() {
    assert_invalid_block(bytes_block(u64::MAX), DataType::Json);
}

#[test]
fn bytes_group_rejects_overflowing_blob_length() {
    let mut raw = bytes_block(2);
    let len_offset = raw.len() - 8;
    raw[len_offset..].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_invalid_block(raw, DataType::Json);
}

#[test]
fn bytes_group_rejects_trailing_payload() {
    let mut raw = bytes_block(2);
    raw.push(0);
    assert_invalid_block(raw, DataType::Json);
}

#[test]
fn integer_group_rejects_trailing_payload() {
    assert_invalid_block(vec![0; 19], DataType::Integer);
}

#[test]
fn group_rejects_incorrect_decoded_length() {
    let raw = vec![0; 18];
    let store = store(lz4_flex::compress(&raw), raw.len() + 1, DataType::Integer);
    let outcome = catch_unwind(AssertUnwindSafe(|| store.group_column(0, 0)));
    assert!(outcome.is_ok(), "group access panicked");
    let error = outcome
        .unwrap()
        .err()
        .expect("wrong decoded length accepted");
    assert_eq!(error.kind(), ErrorKind::InvalidData);
}

#[test]
fn bytes_group_rejects_incorrect_lz4_output_length() {
    let raw = bytes_block(2);
    let store = store(lz4_flex::compress(&raw), raw.len() + 1, DataType::Json);
    let error = store.group_column(0, 0).err().unwrap();
    assert_eq!(error.kind(), ErrorKind::InvalidData);
    assert!(error.to_string().contains("LZ4 decoded length"));
}

#[test]
fn group_rejects_unrepresentable_decoded_length() {
    for data_type in [DataType::Integer, DataType::Json] {
        let store = store(vec![0xff], usize::MAX, data_type);
        let outcome = catch_unwind(AssertUnwindSafe(|| store.group_column(0, 0)));
        assert!(outcome.is_ok(), "invalid decoded length panicked");
        assert_eq!(
            outcome.unwrap().err().unwrap().kind(),
            ErrorKind::InvalidData
        );
    }
}

#[test]
fn group_rejects_missing_metadata() {
    for missing in 0..3 {
        let store = CompressedBlockStore::from_raw_blocks(
            vec![vec![vec![0; 18]]],
            if missing == 0 {
                Vec::new()
            } else {
                vec![vec![18]]
            },
            if missing == 1 { Vec::new() } else { vec![1] },
            vec![DataType::Integer],
            if missing == 2 { Vec::new() } else { vec![0] },
            Vec::new(),
            Vec::new(),
            ROW_GROUP_SIZE,
            2,
        );
        let outcome = catch_unwind(AssertUnwindSafe(|| store.group_column(0, 0)));
        assert!(outcome.is_ok(), "missing metadata {missing} panicked");
        assert_eq!(
            outcome.unwrap().err().unwrap().kind(),
            ErrorKind::InvalidData
        );
    }
}

#[test]
fn group_rejects_inconsistent_row_geometry() {
    for (group_size, rows) in [(0, 2), (ROW_GROUP_SIZE, 0), (1, 2)] {
        let store = CompressedBlockStore::from_raw_blocks(
            vec![vec![vec![0; 18]]],
            vec![vec![18]],
            vec![1],
            vec![DataType::Integer],
            vec![0],
            Vec::new(),
            Vec::new(),
            group_size,
            rows,
        );
        let outcome = catch_unwind(AssertUnwindSafe(|| store.group_column(0, 0)));
        assert!(outcome.is_ok(), "invalid geometry panicked");
        assert_eq!(
            outcome.unwrap().err().unwrap().kind(),
            ErrorKind::InvalidData
        );
    }
}

#[test]
fn group_access_rejects_out_of_range_indices() {
    let store = store(vec![0; 18], 18, DataType::Integer);
    let cached = store.group_column(0, 0).unwrap();
    assert_eq!(cached.len(), 2);
    for (col, group) in [(1, 0), (0, 1), (usize::MAX, 0), (0, usize::MAX)] {
        let outcome = catch_unwind(AssertUnwindSafe(|| store.group_column(col, group)));
        assert!(
            outcome.is_ok(),
            "group access panicked for ({col}, {group})"
        );
        let error = outcome.unwrap().err().expect("invalid index accepted");
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            store.decompress_single_group(col, group)
        }));
        assert!(
            outcome.is_ok(),
            "direct decode panicked for ({col}, {group})"
        );
        assert_eq!(
            outcome.unwrap().err().unwrap().kind(),
            ErrorKind::InvalidInput
        );
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
fn group_cache_key_does_not_alias_large_indices() {
    let store = store(vec![0; 18], 18, DataType::Integer);
    let _cached = store.group_column(0, 0).unwrap();
    for (col, group) in [(1usize << 32, 0), (0, 1usize << 32)] {
        let outcome = catch_unwind(AssertUnwindSafe(|| store.group_column(col, group)));
        assert!(outcome.is_ok(), "large index panicked");
        assert_eq!(
            outcome.unwrap().err().unwrap().kind(),
            ErrorKind::InvalidInput
        );
    }
}

#[test]
fn valid_group_round_trip_preserves_all_column_types_and_nulls() {
    let types = vec![
        DataType::Integer,
        DataType::Float,
        DataType::Timestamp,
        DataType::Boolean,
        DataType::Text,
        DataType::Json,
    ];
    let columns = LazyColumns::eager(
        vec![
            ColumnData::Int64 {
                values: vec![-7, 0],
                nulls: vec![false, true],
            },
            ColumnData::Float64 {
                values: vec![1.5, 0.0],
                nulls: vec![false, true],
            },
            ColumnData::TimestampNanos {
                values: vec![100, 0],
                nulls: vec![false, true],
            },
            ColumnData::Boolean {
                values: vec![true, false],
                nulls: vec![false, true],
            },
            ColumnData::Dictionary {
                ids: vec![0, u32::MAX],
                dictionary: vec!["value".into()].into(),
                nulls: vec![false, true],
            },
            ColumnData::Bytes {
                data: vec![b'{', b'}'],
                offsets: vec![(0, 2), (2, 0)],
                ext_type: DataType::Json,
                nulls: vec![false, true],
            },
        ],
        types.clone(),
    );
    for compress in [false, true] {
        let store =
            CompressedBlockStore::compress_columns_opts(&columns, &types, 2, compress).unwrap();
        for ci in 0..types.len() {
            let group = store.group_column(ci, 0).unwrap();
            assert_eq!(group.len(), 2);
            for row in 0..2 {
                assert_eq!(
                    group.get_value(row),
                    columns.get(ci).unwrap().get_value(row)
                );
            }
        }
        let two_groups = CompressedBlockStore::from_raw_blocks(
            store
                .raw_blocks()
                .iter()
                .map(|blocks| vec![blocks[0].clone(); 2])
                .collect(),
            store
                .decompressed_lens()
                .iter()
                .map(|lengths| vec![lengths[0]; 2])
                .collect(),
            store.col_type_tags().to_vec(),
            types.clone(),
            store.col_ext_types().to_vec(),
            vec!["value".into()],
            vec![(4, 0, 1)],
            2,
            4,
        );
        for (store, rows) in [(store, 2), (two_groups, 4)] {
            let decoded = LazyColumns::deferred(store, types.clone());
            for ci in 0..types.len() {
                assert_eq!(decoded.get(ci).unwrap().len(), rows);
                for row in 0..rows {
                    assert_eq!(
                        decoded.get(ci).unwrap().get_value(row),
                        columns.get(ci).unwrap().get_value(row % 2)
                    );
                }
            }
        }
    }
}

#[test]
fn valid_final_partial_group_preserves_rows() {
    let rows = ROW_GROUP_SIZE + 3;
    let columns = LazyColumns::eager(
        vec![ColumnData::Int64 {
            values: (0..rows as i64).collect(),
            nulls: vec![false; rows],
        }],
        vec![DataType::Integer],
    );
    for compress in [false, true] {
        let store = CompressedBlockStore::compress_columns_opts(
            &columns,
            &[DataType::Integer],
            rows,
            compress,
        )
        .unwrap();
        assert_eq!(store.group_column(0, 0).unwrap().len(), ROW_GROUP_SIZE);
        let last = store.group_column(0, 1).unwrap();
        assert_eq!(last.len(), 3);
        for i in 0..3 {
            assert_eq!(last.get_i64(i), (ROW_GROUP_SIZE + i) as i64);
        }
        let decoded = LazyColumns::deferred(store, vec![DataType::Integer]);
        assert_eq!(decoded.get(0).unwrap().len(), rows);
        for i in [0, ROW_GROUP_SIZE - 1, ROW_GROUP_SIZE, rows - 1] {
            assert_eq!(decoded.get(0).unwrap().get_i64(i), i as i64);
        }
    }
}

#[test]
fn all_null_dictionary_group_does_not_require_a_dictionary_range() {
    use stoolap::core::Value;
    let mut raw = vec![1, 1];
    raw.extend_from_slice(&[0xff; 8]);
    let store = CompressedBlockStore::from_raw_blocks(
        vec![vec![raw]],
        vec![vec![10]],
        vec![5],
        vec![DataType::Text],
        vec![0],
        Vec::new(),
        Vec::new(),
        ROW_GROUP_SIZE,
        2,
    );
    let group = store.group_column(0, 0).unwrap();
    assert_eq!(group.len(), 2);
    assert_eq!(group.get_value(0), Value::Null(DataType::Text));
    assert_eq!(group.get_value(1), Value::Null(DataType::Text));
}

fn two_column_volume() -> stoolap::storage::volume::writer::FrozenVolume {
    use stoolap::core::{Row, Schema, SchemaColumn, Value};
    use stoolap::storage::volume::writer::VolumeBuilder;
    let schema = Schema::new(
        "group_access",
        vec![
            SchemaColumn::new(0, "a", DataType::Integer, false, false),
            SchemaColumn::new(1, "b", DataType::Integer, false, false),
        ],
    );
    let mut builder = VolumeBuilder::new(&schema);
    for id in [1, 2] {
        builder.add_row(
            id,
            &Row::from_values(vec![Value::Integer(id), Value::Integer(id)]),
        );
    }
    builder.finish()
}

fn malformed_columns() -> LazyColumns {
    let mut valid = vec![0, 0];
    valid.extend_from_slice(&1i64.to_le_bytes());
    valid.extend_from_slice(&2i64.to_le_bytes());
    LazyColumns::deferred(
        CompressedBlockStore::from_raw_blocks(
            vec![vec![valid], vec![vec![0; 19]]],
            vec![vec![18], vec![19]],
            vec![1; 2],
            vec![DataType::Integer; 2],
            vec![0; 2],
            Vec::new(),
            Vec::new(),
            ROW_GROUP_SIZE,
            2,
        ),
        vec![DataType::Integer; 2],
    )
}

#[test]
fn lazy_columns_cache_failure_and_never_promote_it_to_eager() {
    let mut columns = malformed_columns();
    assert_eq!(columns.get(0).unwrap().get_i64(1), 2);
    assert_io_failure(|| columns.get(1), ErrorKind::InvalidData);
    let good = two_column_volume();
    columns.attach_compressed_store(
        CompressedBlockStore::compress_columns(&good.columns, &[DataType::Integer; 2], 2).unwrap(),
    );
    assert_io_failure(|| columns.get(1), ErrorKind::InvalidData);
    assert_io_failure(|| columns.get_column_dictionary(1), ErrorKind::InvalidData);
    assert!(!columns.is_eager());
}

#[test]
fn lazy_column_iterator_yields_errors_and_reaches_the_end() {
    let columns = malformed_columns();
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        columns
            .iter()
            .map(|col| col.error_kind())
            .collect::<Vec<_>>()
    }));
    assert!(outcome.is_ok(), "column iterator panicked");
    assert_eq!(outcome.unwrap(), vec![None, Some(ErrorKind::InvalidData)]);
}

#[test]
fn take_columns_rejects_a_partial_decode() {
    let columns = malformed_columns();
    assert_eq!(columns.get(0).unwrap().get_i64(0), 1);
    assert_io_failure(|| columns.take_columns(), ErrorKind::InvalidData);
}

#[test]
fn metadata_only_columns_return_errors() {
    let columns = LazyColumns::metadata_only(vec![DataType::Integer]);
    assert_io_failure(|| columns.get(0), ErrorKind::InvalidData);
    assert_io_failure(|| columns.get_column_dictionary(0), ErrorKind::InvalidData);
    assert_io_failure(|| columns.take_columns(), ErrorKind::InvalidData);
}

#[test]
fn take_columns_decodes_every_uninitialized_column() {
    let volume = two_column_volume();
    let store = CompressedBlockStore::compress_columns(&volume.columns, &[DataType::Integer; 2], 2)
        .unwrap();
    let columns = LazyColumns::deferred(store, vec![DataType::Integer; 2]);
    let decoded = columns.take_columns().unwrap();
    assert_eq!(decoded.len(), 2);
    assert_eq!(decoded[1].get_i64(1), 2);
}

#[test]
fn lazy_column_access_rejects_invalid_indices() {
    let columns = two_column_volume().columns;
    assert_io_failure(|| columns.get(2), ErrorKind::InvalidInput);
    assert_io_failure(|| columns.get_column_dictionary(2), ErrorKind::InvalidInput);
}

fn assert_row_materializer_failure(which: usize) {
    use stoolap::storage::volume::writer::{ColSource, ColumnMapping};
    let mut volume = two_column_volume();
    volume.columns = malformed_columns();
    let mapping = ColumnMapping {
        sources: vec![ColSource::Volume(0), ColSource::Volume(1)],
        is_identity: true,
    };
    assert_io_failure(
        || match which {
            0 => volume.get_row(0),
            1 => volume.get_row_projected(0, &[1]),
            2 => volume.get_row_needed(0, &[false, true]),
            3 => volume.get_row_mapped(0, &mapping),
            4 => volume.get_row_mapped_projected(0, &mapping, &[1]),
            5 => volume.get_row_mapped_needed(0, &mapping, &[false, true]),
            _ => unreachable!(),
        },
        ErrorKind::InvalidData,
    );
}

#[test]
fn full_row_propagates_column_failure() {
    assert_row_materializer_failure(0);
}

#[test]
fn projected_row_propagates_column_failure() {
    assert_row_materializer_failure(1);
}

#[test]
fn needed_row_propagates_column_failure() {
    assert_row_materializer_failure(2);
}

#[test]
fn mapped_row_propagates_column_failure() {
    assert_row_materializer_failure(3);
}

#[test]
fn mapped_projection_propagates_column_failure() {
    assert_row_materializer_failure(4);
}

#[test]
fn mapped_needed_row_propagates_column_failure() {
    assert_row_materializer_failure(5);
}

#[test]
fn projections_leave_unrequested_corrupt_columns_untouched() {
    use stoolap::core::Value;
    use stoolap::storage::volume::writer::{ColSource, ColumnMapping};
    let mut volume = two_column_volume();
    volume.columns = malformed_columns();
    let mapping = ColumnMapping {
        sources: vec![ColSource::Volume(0), ColSource::Default(Value::Integer(9))],
        is_identity: false,
    };
    for row in [
        volume.get_row_projected(0, &[0]).unwrap(),
        volume.get_row_needed(0, &[true, false]).unwrap(),
        volume.get_row_mapped(0, &mapping).unwrap(),
        volume.get_row_mapped_projected(0, &mapping, &[0]).unwrap(),
        volume
            .get_row_mapped_needed(0, &mapping, &[true, false])
            .unwrap(),
    ] {
        assert_eq!(row.get(0), Some(&Value::Integer(1)));
    }
    assert!(!volume.columns.is_eager());
}

#[test]
fn unique_lookup_releases_its_guard_before_callbacks() {
    use stoolap::core::Value;
    let volume = two_column_volume();
    for _ in 0..2 {
        let mut hits = 0;
        volume
            .unique_lookup_all(&[1], &[&Value::Integer(1)], |_| {
                assert!(volume.unique_indices.try_write().is_some());
                hits += 1;
                false
            })
            .unwrap();
        assert_eq!(hits, 1);
    }
}

#[test]
fn cached_unique_miss_skips_columns_but_matching_hash_propagates_failure() {
    use stoolap::core::Value;
    let mut volume = two_column_volume();
    volume.prebuild_unique_index(&[1]).unwrap();
    volume.columns = malformed_columns();
    volume
        .unique_lookup_all(&[1], &[&Value::Integer(99)], |_| {
            panic!("missing hash invoked callback");
        })
        .unwrap();
    assert_io_failure(
        || {
            volume.unique_lookup_all(&[1], &[&Value::Integer(1)], |_| {
                panic!("failed column invoked callback");
            })
        },
        ErrorKind::InvalidData,
    );
}

#[test]
fn scanner_whole_column_failures_are_sticky_in_both_directions() {
    use std::sync::Arc;
    use stoolap::core::Value;
    use stoolap::storage::expression::ComparisonExpr;
    use stoolap::storage::traits::Scanner;
    use stoolap::storage::volume::scanner::VolumeScanner;
    for ascending in [true, false] {
        for mode in 0..3 {
            let mut volume = two_column_volume();
            volume.columns = LazyColumns::metadata_only(vec![DataType::Integer; 2]);
            let mut scanner = VolumeScanner::new(Arc::new(volume), vec![1], None);
            scanner.set_ordered_walk(ascending);
            if mode == 1 {
                scanner.set_stop_key(1, &Value::Integer(1), ascending);
            }
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                if mode == 2 {
                    scanner.set_filter(Box::new(ComparisonExpr::gte("b", Value::Integer(1))));
                }
                scanner.next()
            }));
            assert!(outcome.is_ok(), "whole-column scanner panicked");
            assert!(!outcome.unwrap());
            let error = scanner
                .err()
                .expect("failure became end of scan")
                .to_string();
            assert!(!scanner.next());
            assert_eq!(scanner.err().unwrap().to_string(), error);
        }
    }
}

fn malformed_warm_table() -> stoolap::storage::volume::table::SegmentedTable {
    use std::sync::Arc;
    use stoolap::core::SchemaBuilder;
    use stoolap::storage::mvcc::{MVCCTable, TransactionVersionStore, VersionStore};
    use stoolap::storage::traits::Table;
    use stoolap::storage::volume::manifest::{SegmentManager, SegmentMeta};
    use stoolap::storage::volume::table::SegmentedTable;
    let schema = SchemaBuilder::new("group_access")
        .column("a", DataType::Integer, false, true)
        .column("b", DataType::Integer, false, false)
        .build();
    let mut volume = two_column_volume();
    volume.columns = malformed_columns();
    let manager = Arc::new(SegmentManager::new("group_access", None));
    manager.register_segment(
        1,
        Arc::new(volume),
        SegmentMeta {
            segment_id: 1,
            file_path: Default::default(),
            row_count: 2,
            min_row_id: 1,
            max_row_id: 2,
            creation_lsn: 0,
            seal_seq: 0,
            schema_version: 0,
        },
        Some(&schema),
    );
    let store = Arc::new(VersionStore::new(schema.table_name.clone(), schema));
    let local = TransactionVersionStore::new(Arc::clone(&store), 1);
    let hot = Box::new(MVCCTable::new(1, store, local));
    hot.create_btree_index("b", false, Some("idx_b")).unwrap();
    SegmentedTable::new(hot, manager)
}

fn assert_warm_read_failure(
    read: impl FnOnce(&stoolap::storage::volume::table::SegmentedTable) -> stoolap::core::Result<()>,
) {
    let table = malformed_warm_table();
    table.segment_manager().add_tombstones(&[1], 1);
    let outcome = catch_unwind(AssertUnwindSafe(|| read(&table)));
    assert!(outcome.is_ok(), "warm-column read panicked");
    assert!(
        outcome.unwrap().is_err(),
        "corruption became a successful result"
    );
}

#[test]
fn warm_collection_propagates_column_failure() {
    use stoolap::storage::traits::Table;
    assert_warm_read_failure(|table| table.collect_all_rows(None).map(drop));
}

#[test]
fn warm_sum_propagates_column_failure() {
    use stoolap::storage::traits::Table;
    assert_warm_read_failure(|table| table.sum_column(1).map(drop));
}

#[test]
fn warm_partitions_propagate_column_failure() {
    use stoolap::storage::traits::Table;
    assert_warm_read_failure(|table| table.get_partition_values("b").map(drop));
}

#[test]
fn warm_distinct_propagates_column_failure() {
    use stoolap::storage::traits::Table;
    assert_warm_read_failure(|table| table.compute_distinct_values(1).map(drop));
}

#[test]
fn warm_filtered_aggregation_propagates_column_failure() {
    use stoolap::core::Value;
    use stoolap::storage::expression::ComparisonExpr;
    use stoolap::storage::mvcc::version_store::AggregateOp;
    use stoolap::storage::traits::Table;
    assert_warm_read_failure(|table| {
        table
            .compute_filtered_aggregates(
                &[(AggregateOp::Sum, 1)],
                &ComparisonExpr::gte("a", Value::Integer(0)),
            )
            .map(drop)
    });
}

#[test]
fn warm_grouped_aggregation_propagates_column_failure() {
    use stoolap::storage::mvcc::version_store::AggregateOp;
    use stoolap::storage::traits::Table;
    assert_warm_read_failure(|table| {
        table
            .compute_grouped_aggregates(&[0], &[(AggregateOp::Sum, 1)])
            .map(drop)
    });
}

#[test]
fn fully_hidden_warm_columns_are_not_decoded() {
    use stoolap::core::Value;
    use stoolap::storage::mvcc::version_store::AggregateOp;
    use stoolap::storage::traits::Table;
    let table = malformed_warm_table();
    table.segment_manager().add_tombstones(&[1, 2], 1);
    assert_eq!(table.sum_column(1).unwrap(), Some((0.0, 0)));
    assert_eq!(table.get_partition_count("b").unwrap(), Some(0));
    assert!(table.get_partition_values("b").unwrap().unwrap().is_empty());
    assert!(table
        .compute_distinct_values(1)
        .unwrap()
        .unwrap()
        .is_empty());
    assert!(table
        .collect_rows_grouped_by_partition("b")
        .unwrap()
        .unwrap()
        .is_empty());
    assert!(table
        .get_rows_for_partition_value("b", &Value::Integer(1))
        .unwrap()
        .unwrap()
        .is_empty());
    assert!(table
        .compute_grouped_aggregates(&[0], &[(AggregateOp::Sum, 1)])
        .unwrap()
        .unwrap()
        .is_empty());
    table.drop_index("idx_b").unwrap();
    table
        .create_btree_index("b", true, Some("unique_b"))
        .unwrap();
}

fn maintenance_config(path: &std::path::Path) -> stoolap::storage::config::Config {
    let mut config = stoolap::storage::config::Config::with_path(path.to_str().unwrap());
    config.cleanup.enabled = false;
    config.persistence.checkpoint_on_close = false;
    config.persistence.checkpoint_interval = 3600;
    config.persistence.target_volume_rows = 32_768;
    config
}

fn volume_files(path: &std::path::Path) -> std::collections::BTreeSet<std::path::PathBuf> {
    std::fs::read_dir(path)
        .into_iter()
        .flatten()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "vol"))
        .collect()
}

#[test]
fn late_compaction_failure_removes_output_and_preserves_original_volumes() {
    use std::sync::Arc;
    use stoolap::core::{Row, SchemaBuilder, Value};
    use stoolap::storage::mvcc::MVCCEngine;
    use stoolap::storage::traits::Engine;
    use stoolap::storage::volume::io::{read_volume_from_disk, write_volume_to_disk};
    use stoolap::storage::volume::manifest::{SegmentManager, SegmentMeta, TableManifest};
    use stoolap::storage::volume::writer::VolumeBuilder;
    let dir = tempfile::tempdir().unwrap();
    let schema = SchemaBuilder::new("group_access")
        .column("a", DataType::Integer, false, true)
        .column("b", DataType::Integer, false, false)
        .build();
    let engine = MVCCEngine::new(maintenance_config(dir.path()));
    engine.open_engine().unwrap();
    engine.create_table(schema.clone()).unwrap();
    engine.close_engine().unwrap();
    drop(engine);
    let volume_dir = dir.path().join("volumes");
    let table_dir = volume_dir.join("group_access");
    let manager = SegmentManager::new("group_access", Some(volume_dir.clone()));
    let mut good_last_file = Vec::new();
    let mut last_path = std::path::PathBuf::new();
    for (volume_id, start, end) in [(1, 1, 65_536), (2, 65_537, 65_537)] {
        let mut builder = VolumeBuilder::new(&schema);
        for row_id in start..=end {
            builder.add_row(row_id, &Row::from_values(vec![Value::Integer(row_id); 2]));
        }
        let volume = builder.finish();
        let path = write_volume_to_disk(&volume_dir, "group_access", volume_id, &volume).unwrap();
        if volume_id == 2 {
            good_last_file = std::fs::read(&path).unwrap();
            let meta_len = u32::from_le_bytes(good_last_file[16..20].try_into().unwrap()) as usize;
            let mut bytes = good_last_file[..20 + meta_len].to_vec();
            for size in [9u64, 10] {
                bytes.extend_from_slice(&size.to_le_bytes());
                bytes.extend_from_slice(&size.to_le_bytes());
            }
            bytes.extend_from_slice(&[0; 19]);
            bytes.extend_from_slice(&crc32fast::hash(&bytes).to_le_bytes());
            std::fs::write(&path, bytes).unwrap();
            last_path = path.clone();
        }
        manager.register_segment(
            volume_id,
            Arc::new(read_volume_from_disk(&path).unwrap()),
            SegmentMeta {
                segment_id: volume_id,
                file_path: path.file_name().unwrap().into(),
                row_count: volume.meta.row_count,
                min_row_id: start,
                max_row_id: end,
                creation_lsn: 0,
                seal_seq: 0,
                schema_version: 0,
            },
            Some(&schema),
        );
    }
    manager.persist().unwrap();
    let original_files = volume_files(&table_dir);
    let manifest_path = table_dir.join("manifest.bin");
    let engine = MVCCEngine::new(maintenance_config(dir.path()));
    engine.open_engine().unwrap();
    for _ in 0..2 {
        let outcome = catch_unwind(AssertUnwindSafe(|| engine.force_checkpoint_cycle()));
        assert!(outcome.is_ok(), "compaction panicked after writing a chunk");
        assert!(
            matches!(outcome.unwrap(), Err(stoolap::core::Error::Io { message })
            if message == "invalid column block length")
        );
        assert_eq!(
            engine
                .volume_stats()
                .into_iter()
                .map(|entry| (entry.1, entry.3))
                .collect::<std::collections::BTreeSet<_>>(),
            [(1, 65_536), (2, 1)].into()
        );
        assert_eq!(volume_files(&table_dir), original_files);
        let manifest = TableManifest::read_from_disk(&manifest_path).unwrap();
        assert_eq!(
            manifest
                .segments
                .iter()
                .map(|seg| seg.segment_id)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }
    engine.close_engine().unwrap();
    drop(engine);
    std::fs::write(&last_path, good_last_file).unwrap();
    let engine = MVCCEngine::new(maintenance_config(dir.path()));
    engine.open_engine().unwrap();
    engine.force_checkpoint_cycle().unwrap();
    let mut tx = engine.begin_transaction().unwrap();
    assert_eq!(
        tx.get_table("group_access").unwrap().row_count().unwrap(),
        65_537
    );
    tx.rollback().unwrap();
    engine.close_engine().unwrap();
}

fn hot_maintenance_engine(path: &std::path::Path) -> stoolap::storage::mvcc::MVCCEngine {
    use stoolap::core::{Row, SchemaBuilder, Value};
    use stoolap::storage::mvcc::MVCCEngine;
    use stoolap::storage::traits::Engine;
    let engine = MVCCEngine::new(maintenance_config(path));
    engine.open_engine().unwrap();
    engine
        .create_table(
            SchemaBuilder::new("group_access")
                .column("a", DataType::Integer, false, true)
                .column("b", DataType::Integer, false, false)
                .build(),
        )
        .unwrap();
    let mut tx = engine.begin_transaction().unwrap();
    tx.get_table("group_access")
        .unwrap()
        .insert_batch(vec![
            Row::from_values(vec![Value::Integer(1); 2]),
            Row::from_values(vec![Value::Integer(2); 2]),
        ])
        .unwrap();
    tx.commit().unwrap();
    engine
}

#[test]
fn seal_prebuild_schema_mismatch_removes_output_and_keeps_hot_rows() {
    use std::sync::Arc;
    use stoolap::common::CompactArc;
    use stoolap::core::SchemaBuilder;
    use stoolap::storage::index::HashIndex;
    use stoolap::storage::traits::Engine;
    let dir = tempfile::tempdir().unwrap();
    let engine = hot_maintenance_engine(dir.path());
    let store = engine.get_version_store("group_access").unwrap();
    let original_schema = store.schema();
    *store.schema_mut() = CompactArc::new(
        SchemaBuilder::new("group_access")
            .column("a", DataType::Integer, false, true)
            .column("b", DataType::Integer, false, false)
            .column("extra", DataType::Integer, false, false)
            .build(),
    );
    store.add_index(
        "extra_unique".into(),
        Arc::new(HashIndex::new(
            "extra_unique".into(),
            "group_access".into(),
            vec!["extra".into()],
            vec![2],
            vec![DataType::Integer],
            true,
            0,
        )),
    );
    let outcome = catch_unwind(AssertUnwindSafe(|| engine.force_checkpoint_cycle()));
    assert!(outcome.is_ok(), "seal prebuild panicked on schema mismatch");
    assert!(
        matches!(outcome.unwrap(), Err(stoolap::core::Error::Io { message })
        if message == "column index out of range")
    );
    assert!(engine.volume_stats().is_empty());
    assert_eq!(store.committed_row_count(), 2);
    let table_dir = dir.path().join("volumes/group_access");
    assert!(volume_files(&table_dir).is_empty());
    store.remove_index("extra_unique");
    *store.schema_mut() = original_schema;
    engine.force_checkpoint_cycle().unwrap();
    assert_eq!(store.committed_row_count(), 0);
    assert_eq!(volume_files(&table_dir).len(), 1);
    engine.close_engine().unwrap();
}

#[test]
fn seal_write_failure_is_reported_and_can_be_retried() {
    use stoolap::storage::traits::Engine;
    let dir = tempfile::tempdir().unwrap();
    let engine = hot_maintenance_engine(dir.path());
    let table_dir = dir.path().join("volumes/group_access");
    std::fs::create_dir_all(table_dir.parent().unwrap()).unwrap();
    std::fs::write(&table_dir, b"blocked directory").unwrap();
    assert!(
        engine.force_checkpoint_cycle().is_err(),
        "seal write failure was swallowed"
    );
    assert_eq!(
        engine
            .get_version_store("group_access")
            .unwrap()
            .committed_row_count(),
        2
    );
    std::fs::remove_file(&table_dir).unwrap();
    engine.force_checkpoint_cycle().unwrap();
    assert_eq!(volume_files(&table_dir).len(), 1);
    engine.close_engine().unwrap();
}

#[test]
fn scanner_reports_a_missing_group_for_full_and_partial_projections() {
    use std::sync::Arc;
    use stoolap::storage::traits::Scanner;
    use stoolap::storage::volume::scanner::VolumeScanner;
    for projection in [vec![], vec![1]] {
        let mut volume = two_column_volume();
        let store = CompressedBlockStore::from_raw_blocks(
            vec![vec![vec![0; 18]], vec![]],
            vec![vec![18], vec![]],
            vec![1, 1],
            vec![DataType::Integer; 2],
            vec![0; 2],
            Vec::new(),
            Vec::new(),
            ROW_GROUP_SIZE,
            2,
        );
        volume.columns = LazyColumns::deferred(store, vec![DataType::Integer; 2]);
        let mut scanner = VolumeScanner::new(Arc::new(volume), projection, None);
        let outcome = catch_unwind(AssertUnwindSafe(|| scanner.next()));
        assert!(outcome.is_ok(), "scanner panicked on missing group");
        assert!(!outcome.unwrap());
        assert!(
            scanner.err().is_some(),
            "missing group became an empty result"
        );
    }
}

#[test]
fn scanner_reports_structural_payload_corruption_after_reopen() {
    use std::sync::Arc;
    use stoolap::storage::traits::Scanner;
    use stoolap::storage::volume::io::{read_volume_from_disk, write_volume_to_disk};
    use stoolap::storage::volume::scanner::VolumeScanner;
    let dir = tempfile::tempdir().unwrap();
    let path = write_volume_to_disk(dir.path(), "group_access", 1, &two_column_volume()).unwrap();
    let original = std::fs::read(&path).unwrap();
    let meta_len = u32::from_le_bytes(original[16..20].try_into().unwrap()) as usize;
    let index_offset = 20 + meta_len;
    let mut bytes = original[..index_offset].to_vec();
    for size in [18u64, 19] {
        bytes.extend_from_slice(&size.to_le_bytes());
        bytes.extend_from_slice(&size.to_le_bytes());
    }
    bytes.extend_from_slice(&[0; 37]);
    bytes.extend_from_slice(&crc32fast::hash(&bytes).to_le_bytes());
    std::fs::write(&path, bytes).unwrap();
    let volume = Arc::new(read_volume_from_disk(&path).unwrap());
    let mut scanner = VolumeScanner::new(volume, vec![1], None);
    assert!(!scanner.next(), "malformed block yielded a row");
    assert!(
        scanner.err().is_some(),
        "malformed block became an empty result"
    );
}
