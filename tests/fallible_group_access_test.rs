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
        let store = CompressedBlockStore::compress_columns_opts(&columns, &types, 2, compress);
        for ci in 0..types.len() {
            let group = store.group_column(ci, 0).unwrap();
            assert_eq!(group.len(), 2);
            for row in 0..2 {
                assert_eq!(group.get_value(row), columns[ci].get_value(row));
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
        );
        assert_eq!(store.group_column(0, 0).unwrap().len(), ROW_GROUP_SIZE);
        let last = store.group_column(0, 1).unwrap();
        assert_eq!(last.len(), 3);
        for i in 0..3 {
            assert_eq!(last.get_i64(i), (ROW_GROUP_SIZE + i) as i64);
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
