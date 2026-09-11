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

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use stoolap::core::{DataType, Row, SchemaBuilder, Value};
use stoolap::storage::volume::io::{read_volume_from_disk, write_volume_to_disk};
use stoolap::storage::volume::writer::VolumeBuilder;

fn fixture() -> (tempfile::TempDir, std::path::PathBuf, Vec<u8>) {
    let dir = tempfile::tempdir().unwrap();
    let schema = SchemaBuilder::new("metadata")
        .column("v", DataType::Text, true, false)
        .build();
    let mut builder = VolumeBuilder::new(&schema);
    builder.add_row(1, &Row::from_values(vec![Value::Null(DataType::Text)]));
    let path = write_volume_to_disk(dir.path(), "metadata", 1, &builder.finish()).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    (dir, path, bytes)
}

fn replace_metadata(bytes: &mut Vec<u8>, edit: impl FnOnce(&mut Vec<u8>)) {
    let len = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
    let mut metadata = lz4_flex::decompress_size_prepended(&bytes[20..20 + len]).unwrap();
    edit(&mut metadata);
    let compressed = lz4_flex::compress_prepend_size(&metadata);
    bytes[16..20].copy_from_slice(&(compressed.len() as u32).to_le_bytes());
    bytes.splice(20..20 + len, compressed);
}

fn assert_rejected(path: &Path, mut bytes: Vec<u8>) {
    bytes.truncate(bytes.len() - 4);
    bytes.extend_from_slice(&crc32fast::hash(&bytes).to_le_bytes());
    std::fs::write(path, bytes).unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| read_volume_from_disk(path)));
    assert!(result.is_ok(), "malformed file panicked while loading");
    assert!(result.unwrap().is_err(), "malformed file was accepted");
}

#[test]
fn metadata_rejects_short_lz4_output() {
    let (_dir, path, mut bytes) = fixture();
    let size = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
    bytes[20..24].copy_from_slice(&(size + 1).to_le_bytes());
    assert_rejected(&path, bytes);
}

#[test]
fn metadata_rejects_impossible_lz4_expansion() {
    let (_dir, path, mut bytes) = fixture();
    let compressed_len = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) - 4;
    bytes[20..24].copy_from_slice(&(compressed_len * 255 + 1).to_le_bytes());
    assert_rejected(&path, bytes);
}

#[test]
fn metadata_rejects_trailing_decoded_bytes() {
    let (_dir, path, mut bytes) = fixture();
    replace_metadata(&mut bytes, |metadata| metadata.push(0));
    assert_rejected(&path, bytes);
}

#[test]
fn metadata_rejects_overflowing_row_count() {
    let (_dir, path, mut bytes) = fixture();
    replace_metadata(&mut bytes, |metadata| {
        metadata[..8].copy_from_slice(&u64::MAX.to_le_bytes());
    });
    assert_rejected(&path, bytes);
}

#[test]
fn metadata_rejects_dictionary_count_past_shared_dictionary() {
    let (_dir, path, mut bytes) = fixture();
    replace_metadata(&mut bytes, |metadata| {
        metadata[14..18].copy_from_slice(&1u32.to_le_bytes());
    });
    assert_rejected(&path, bytes);
}

#[test]
fn metadata_rejects_unassigned_shared_dictionary_entries() {
    let (_dir, path, mut bytes) = fixture();
    replace_metadata(&mut bytes, |metadata| {
        metadata[18..22].copy_from_slice(&1u32.to_le_bytes());
        metadata.splice(22..22, 0u32.to_le_bytes());
    });
    assert_rejected(&path, bytes);
}

#[test]
fn metadata_rejects_empty_bloom_backing() {
    let (_dir, path, mut bytes) = fixture();
    replace_metadata(&mut bytes, |metadata| {
        // One null TEXT row puts the first bloom after its 12-byte zone map.
        metadata[54..58].copy_from_slice(&0u32.to_le_bytes());
        metadata.drain(58..66);
    });
    assert_rejected(&path, bytes);
}

#[test]
fn metadata_rejects_incomplete_bloom_word() {
    let (_dir, path, mut bytes) = fixture();
    replace_metadata(&mut bytes, |metadata| {
        metadata[54..58].copy_from_slice(&7u32.to_le_bytes());
        metadata.remove(65);
    });
    assert_rejected(&path, bytes);
}

#[test]
fn metadata_rejects_bloom_bit_count_outside_backing() {
    for bits in [0u64, 65] {
        let (_dir, path, mut bytes) = fixture();
        replace_metadata(&mut bytes, |metadata| {
            metadata[46..54].copy_from_slice(&bits.to_le_bytes());
        });
        assert_rejected(&path, bytes);
    }
}

#[test]
fn metadata_rejects_missing_column_statistics() {
    let (_dir, path, mut bytes) = fixture();
    replace_metadata(&mut bytes, |metadata| {
        metadata[82..86].copy_from_slice(&0u32.to_le_bytes());
        metadata.drain(86..130);
    });
    assert_rejected(&path, bytes);
}

#[test]
fn volume_rejects_oversized_compressed_block() {
    let (_dir, path, mut bytes) = fixture();
    let meta_len = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
    bytes[20 + meta_len..28 + meta_len].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_rejected(&path, bytes);
}

#[test]
fn volume_rejects_trailing_file_bytes() {
    let (_dir, path, mut bytes) = fixture();
    bytes.extend_from_slice(&[0; 4]);
    assert_rejected(&path, bytes);
}

#[test]
fn volume_rejects_missing_block_group() {
    let (_dir, path, mut bytes) = fixture();
    bytes[12..16].copy_from_slice(&0u32.to_le_bytes());
    let meta_len = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
    bytes.truncate(24 + meta_len);
    assert_rejected(&path, bytes);
}

#[test]
fn metadata_preserves_null_dictionary_and_implicit_group() {
    let (_dir, path, _bytes) = fixture();
    let volume = read_volume_from_disk(&path).unwrap();
    assert_eq!(volume.row_ids().unwrap(), &[1]);
    assert!(volume.meta.row_groups.is_empty());
    assert_eq!(
        volume.get_row(0).unwrap().get(0),
        Some(&Value::Null(DataType::Text))
    );
}
