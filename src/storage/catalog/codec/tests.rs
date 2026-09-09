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

use super::super::TableSchemaHistory;
use super::*;

fn full_catalog() -> CatalogGeneration {
    CatalogGeneration::try_new(super::super::generation::tests::parts()).unwrap()
}

fn empty_catalog() -> CatalogGeneration {
    CatalogGeneration::try_new(CatalogParts {
        generation: NonZeroU64::new(u64::MAX).unwrap(),
        table_id_high_water_mark: u64::MAX,
        ddl_epoch_high_water_mark: u64::MAX,
        wal_observation_ceiling: u64::MAX,
        coverage: CatalogCoverage::default(),
        tables: vec![],
        views: vec![],
    })
    .unwrap()
}

fn encode(catalog: &CatalogGeneration) -> Vec<u8> {
    let mut bytes = Vec::new();
    let descriptor = encode_into(catalog, &mut bytes).unwrap();
    assert_eq!(descriptor.bytes(), bytes.len() as u64);
    assert_eq!(descriptor.generation(), catalog.generation());
    assert!(std::ptr::eq(descriptor.coverage(), catalog.coverage()));
    assert_eq!(
        descriptor.checksum(),
        crc32fast::hash(&bytes[..bytes.len() - 4])
    );
    bytes
}

fn decode(bytes: &[u8]) -> Result<CatalogGeneration> {
    decode_from(&mut &*bytes, &CatalogDecodeLimits::default())
}

fn checksum(bytes: &mut [u8]) {
    let header_crc = crc32fast::hash(&bytes[..36]);
    bytes[36..40].copy_from_slice(&header_crc.to_le_bytes());
    let footer_at = bytes.len() - 4;
    let footer_crc = crc32fast::hash(&bytes[..footer_at]);
    bytes[footer_at..].copy_from_slice(&footer_crc.to_le_bytes());
}

fn wire_value(value: &Value) -> Vec<u8> {
    let mut bytes = Vec::new();
    Encoder(&mut bytes).record(|w| w.value(value)).unwrap();
    bytes
}

fn read_value(bytes: &[u8]) -> Result<Value> {
    let mut input = bytes;
    let limits = CatalogDecodeLimits::default();
    let mut decoder = Decoder {
        input: &mut input,
        limits: &limits,
        hash: Hasher::new(),
        position: 0,
        end: bytes.len() as u64,
        records: 0,
        payload_bytes: 0,
        metadata_bytes: 0,
    };
    let value = decoder.record(Decoder::value)?;
    if decoder.remaining() != 0 {
        return Err(invalid("test value trailing bytes"));
    }
    Ok(value)
}

fn value_catalog(values: Vec<Value>) -> CatalogGeneration {
    let identity = TableIdentity::new(TableId::new(17).unwrap(), Incarnation::FIRST);
    let stamp = DdlStamp::new(1, 900).unwrap();
    let columns: Vec<_> = values
        .into_iter()
        .enumerate()
        .map(|(position, value)| {
            let kind = value.data_type();
            let mut column = SchemaColumn::with_default_value(
                position,
                format!("Στήληİ{position}"),
                kind,
                true,
                false,
                false,
                Some(String::new()),
                Some(value),
                None,
            );
            if kind == DataType::Vector {
                column.vector_dimensions = 0;
            }
            column
        })
        .collect();
    let ids = (1..=columns.len())
        .map(|id| ColumnId::new(id as u64).unwrap())
        .collect();
    let schema = Schema::with_timestamps_and_foreign_keys(
        "Πίνακαςİ",
        columns,
        vec![],
        DateTime::from_timestamp(-30_000_000_000, 123_456_789).unwrap(),
        DateTime::from_timestamp(30_000_000_000, 987_654_321).unwrap(),
    );
    let table = CatalogTable {
        history: TableSchemaHistory::new(identity, 0, schema, ids)
            .unwrap()
            .with_column_high_water_mark(100)
            .unwrap(),
        incarnations: vec![CatalogIncarnation {
            incarnation: Incarnation::FIRST,
            started_at: stamp,
            first_schema_version: 0,
            last_schema_version: 0,
            ended: None,
        }],
        names: vec![TableNameEvent {
            stamp,
            name: Some("Πίνακαςİ".into()),
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
    CatalogGeneration::try_new(CatalogParts {
        generation: NonZeroU64::new(7).unwrap(),
        table_id_high_water_mark: 31,
        ddl_epoch_high_water_mark: 8,
        wal_observation_ceiling: 1000,
        coverage: CatalogCoverage {
            through_lsn: 1000,
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
    .unwrap()
}

#[test]
fn full_history_roundtrip_is_canonical_and_preserves_exact_identities() {
    for catalog in [empty_catalog(), full_catalog()] {
        let bytes = encode(&catalog);
        let recovered = decode(&bytes).unwrap();
        assert_eq!(encode(&recovered), bytes);
        assert_eq!(encode(&catalog), bytes);
        assert_eq!(
            recovered.parts().table_id_high_water_mark,
            catalog.parts().table_id_high_water_mark
        );
        assert_eq!(
            recovered.parts().ddl_epoch_high_water_mark,
            catalog.parts().ddl_epoch_high_water_mark
        );
        for (original, recovered) in catalog.tables().iter().zip(recovered.tables()) {
            assert_eq!(original.history.identity(), recovered.history.identity());
            assert_eq!(
                original.history.column_high_water_mark(),
                recovered.history.column_high_water_mark()
            );
            for (before, after) in original
                .history
                .revisions()
                .zip(recovered.history.revisions())
            {
                assert_eq!(before.schema(), after.schema());
                assert_eq!(before.column_ids(), after.column_ids());
                assert_eq!(before.version(), after.version());
            }
            for (before, after) in original.foreign_keys.iter().zip(&recovered.foreign_keys) {
                assert_eq!(before.bindings, after.bindings); // Includes defined_at.
            }
            for incarnation in &original.incarnations {
                let identity = TableIdentity::new(original.table_id(), incarnation.incarnation);
                let version = incarnation.first_schema_version;
                assert_eq!(
                    catalog.resolve_schema(identity, version).unwrap().schema(),
                    recovered.history.lookup_version(version).unwrap().schema()
                );
            }
        }
    }
    let decoded = decode(&encode(&full_catalog())).unwrap();
    let old = TableIdentity::new(TableId::new(1).unwrap(), Incarnation::FIRST);
    assert!(decoded.resolve_schema(old, 0).is_ok());
    let wrong = TableIdentity::new(old.table_id, Incarnation::new(3).unwrap());
    assert!(decoded.resolve_schema(wrong, 0).is_err());
}

#[test]
fn values_preserve_bits_typed_nulls_unicode_and_wide_timestamps() {
    let mut values: Vec<_> = (0..=7)
        .map(|tag| Value::Null(DataType::from_u8(tag).unwrap()))
        .collect();
    values.extend([
        Value::Boolean(false),
        Value::Boolean(true),
        Value::Integer(i64::MIN),
        Value::Integer(i64::MAX),
        Value::Float(f64::from_bits(0x7ff8_0000_0000_1234)),
        Value::Float(f64::from_bits(0xfff8_0000_0000_5678)),
        Value::Float(-0.0),
        Value::Float(f64::INFINITY),
        Value::text(""),
        Value::text("İ Σ ΟΣ 🦀\0"),
        Value::Timestamp(DateTime::from_timestamp(-30_000_000_000, 123_456_789).unwrap()),
        Value::Timestamp(DateTime::from_timestamp(30_000_000_000, 987_654_321).unwrap()),
        Value::Timestamp(DateTime::from_timestamp(59, 1_500_000_000).unwrap()),
        Value::json(""),
        Value::json("{\"日本\":\"🦀\"}"),
        Value::vector(vec![]),
        Value::vector(vec![f32::from_bits(0x7fc0_0123), -0.0, f32::NEG_INFINITY]),
    ]);
    for value in &values {
        let bytes = wire_value(value);
        assert_eq!(wire_value(&read_value(&bytes).unwrap()), bytes);
    }
    let catalog = value_catalog(values);
    let bytes = encode(&catalog);
    let recovered = decode(&bytes).unwrap();
    assert_eq!(encode(&recovered), bytes);
    let before = catalog.tables()[0]
        .history
        .revisions()
        .next()
        .unwrap()
        .schema();
    let after = recovered.tables()[0]
        .history
        .revisions()
        .next()
        .unwrap()
        .schema();
    assert_eq!(before.created_at, after.created_at);
    assert_eq!(before.updated_at, after.updated_at);
    for column in &after.columns {
        assert_eq!(column.default_expr.as_deref(), Some(""));
        assert_eq!(column.check_expr, None);
    }
}

#[test]
fn every_truncation_and_single_byte_corruption_fails() {
    let bytes = encode(&full_catalog());
    for end in 0..bytes.len() {
        assert!(decode(&bytes[..end]).is_err(), "truncation at {end}");
    }
    for position in 0..bytes.len() {
        let mut corrupt = bytes.clone();
        corrupt[position] ^= 0x80;
        assert!(decode(&corrupt).is_err(), "corruption at {position}");
    }
    let mut extra = bytes;
    extra.push(0);
    assert!(decode(&extra).is_err());
}

#[test]
fn recomputed_header_and_body_corruptions_do_not_bypass_validation() {
    let bytes = encode(&full_catalog());
    for position in [0, 4, 6, 8, 28] {
        let mut corrupt = bytes.clone();
        corrupt[position] ^= 0x80;
        checksum(&mut corrupt);
        assert!(
            decode(&corrupt).is_err(),
            "unsupported header at {position}"
        );
    }
    for (position, value) in [(12, u64::MAX), (20, 0), (40, 0)] {
        let mut corrupt = bytes.clone();
        corrupt[position..position + 8].copy_from_slice(&value.to_le_bytes());
        checksum(&mut corrupt);
        assert!(decode(&corrupt).is_err(), "invalid scalar at {position}");
    }
    // The first mutation's record length is at header+five u64s+count.
    let first_record = HEADER_BYTES + 40 + 4;
    for length in [0, 1, u64::MAX] {
        let mut corrupt = bytes.clone();
        corrupt[first_record..first_record + 8].copy_from_slice(&length.to_le_bytes());
        checksum(&mut corrupt);
        assert!(decode(&corrupt).is_err());
    }
    // Valid checksums do not authorize an unknown DDL discriminant.
    let mut corrupt = bytes.clone();
    corrupt[first_record + 8 + 16] = 255;
    checksum(&mut corrupt);
    assert!(decode(&corrupt).is_err());
    // Inflating the first mutation record by one keeps all body bytes valid,
    // but must fail its exact record-end check before the next list item.
    let mut corrupt = bytes;
    let length = u64::from_le_bytes(corrupt[first_record..first_record + 8].try_into().unwrap());
    corrupt[first_record..first_record + 8].copy_from_slice(&(length + 1).to_le_bytes());
    checksum(&mut corrupt);
    assert!(decode(&corrupt).is_err());
}

#[test]
fn strict_values_reject_noncanonical_tags_lengths_and_trailing_bytes() {
    fn record(body: &[u8]) -> Vec<u8> {
        let mut bytes = (body.len() as u64).to_le_bytes().to_vec();
        bytes.extend(body);
        bytes
    }
    for body in [
        vec![0, 255],
        vec![1, 2],
        vec![1, 255],
        vec![5],
        vec![9],
        vec![255],
        vec![4, 1, 0, 0, 0, 255],
        vec![6, 1, 0, 0, 0, 255],
        vec![10, 255, 255, 255, 255],
        vec![10, 1, 0, 0, 0, 0],
        vec![0, 1, 0],
    ] {
        assert!(read_value(&record(&body)).is_err(), "{body:?}");
    }
    let mut timestamp = vec![8];
    timestamp.extend(i64::MAX.to_le_bytes());
    timestamp.extend(0u32.to_le_bytes());
    assert!(read_value(&record(&timestamp)).is_err());
    let mut timestamp = vec![8];
    timestamp.extend(0i64.to_le_bytes());
    timestamp.extend(u32::MAX.to_le_bytes());
    assert!(read_value(&record(&timestamp)).is_err());
    let mut oversized = vec![4];
    oversized.extend(u32::MAX.to_le_bytes());
    assert!(read_value(&record(&oversized)).is_err());
    for value in [
        Value::Extension(CompactArc::from(Vec::<u8>::new())),
        Value::Extension(CompactArc::from(vec![255])),
        Value::Extension(CompactArc::from(vec![6, 255])),
        Value::Extension(CompactArc::from(vec![7, 0])),
    ] {
        let mut bytes = Vec::new();
        assert!(Encoder(&mut bytes).value(&value).is_err());
    }
}

#[test]
fn quotas_reject_before_advertised_allocation_and_include_cached_names() {
    let bytes = encode(&full_catalog());
    let defaults = CatalogDecodeLimits::default();
    let tests = [
        CatalogDecodeLimits {
            max_wire_bytes: bytes.len() as u64 - 1,
            ..defaults.clone()
        },
        CatalogDecodeLimits {
            max_records: 0,
            ..defaults.clone()
        },
        CatalogDecodeLimits {
            max_tables: 0,
            ..defaults.clone()
        },
        CatalogDecodeLimits {
            max_revisions_per_table: 0,
            ..defaults.clone()
        },
        CatalogDecodeLimits {
            max_columns_per_revision: 0,
            ..defaults.clone()
        },
        CatalogDecodeLimits {
            max_incarnations_per_table: 0,
            ..defaults.clone()
        },
        CatalogDecodeLimits {
            max_events_per_table: 0,
            ..defaults.clone()
        },
        CatalogDecodeLimits {
            max_indexes_per_table: 0,
            ..defaults.clone()
        },
        CatalogDecodeLimits {
            max_foreign_keys_per_revision: 0,
            ..defaults.clone()
        },
        CatalogDecodeLimits {
            max_mutations: 0,
            ..defaults.clone()
        },
        CatalogDecodeLimits {
            max_effects_per_mutation: 0,
            ..defaults.clone()
        },
        CatalogDecodeLimits {
            max_views: 0,
            ..defaults.clone()
        },
        CatalogDecodeLimits {
            max_string_bytes: 0,
            ..defaults.clone()
        },
        CatalogDecodeLimits {
            max_total_payload_bytes: 0,
            ..defaults.clone()
        },
        CatalogDecodeLimits {
            max_requested_metadata_bytes: 0,
            ..defaults.clone()
        },
    ];
    for limits in tests {
        assert!(
            decode_from(&mut bytes.as_slice(), &limits).is_err(),
            "{limits:?}"
        );
    }
    let value_bytes = encode(&value_catalog(vec![Value::text("more than four bytes")]));
    assert!(decode_from(
        &mut value_bytes.as_slice(),
        &CatalogDecodeLimits {
            max_value_bytes: 4,
            ..defaults.clone()
        }
    )
    .is_err());
    let mut count_attack = encode(&empty_catalog());
    count_attack[80..84].copy_from_slice(&u32::MAX.to_le_bytes());
    checksum(&mut count_attack);
    assert!(decode_from(
        &mut count_attack.as_slice(),
        &CatalogDecodeLimits {
            max_mutations: u32::MAX,
            max_records: u64::MAX,
            max_requested_metadata_bytes: u64::MAX,
            ..defaults
        }
    )
    .is_err());
}

struct FragmentedReader<'a> {
    bytes: &'a [u8],
    chunk: usize,
}
impl Read for FragmentedReader<'_> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let length = output.len().min(self.chunk).min(self.bytes.len());
        output[..length].copy_from_slice(&self.bytes[..length]);
        self.bytes = &self.bytes[length..];
        Ok(length)
    }
}
struct FailingWriter {
    remaining: usize,
    written: usize,
}
impl Write for FailingWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            return Err(io::Error::other("catalog test write failure"));
        }
        let length = self.remaining.min(bytes.len());
        self.remaining -= length;
        self.written += length;
        Ok(length)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn fragmented_reads_and_writes_preserve_io_errors_and_no_success_descriptor() {
    let catalog = full_catalog();
    let bytes = encode(&catalog);
    for chunk in [1, 2, 3, 7, 64] {
        let recovered = decode_from(
            &mut FragmentedReader {
                bytes: &bytes,
                chunk,
            },
            &CatalogDecodeLimits::default(),
        )
        .unwrap();
        assert_eq!(encode(&recovered), bytes);
    }
    for fail_at in 0..bytes.len() {
        let mut writer = FailingWriter {
            remaining: fail_at,
            written: 0,
        };
        let error = encode_into(&catalog, &mut writer).unwrap_err();
        assert!(matches!(error, Error::Io { .. }));
        assert_eq!(writer.written, fail_at);
    }
    struct FailingReader<'a> {
        bytes: &'a [u8],
        remaining: usize,
    }
    impl Read for FailingReader<'_> {
        fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
            if self.remaining == 0 {
                return Err(io::Error::other("catalog test read failure"));
            }
            let length = self.remaining.min(output.len()).min(self.bytes.len());
            output[..length].copy_from_slice(&self.bytes[..length]);
            self.bytes = &self.bytes[length..];
            self.remaining -= length;
            Ok(length)
        }
    }
    for fail_at in 0..bytes.len() {
        let error = decode_from(
            &mut FailingReader {
                bytes: &bytes,
                remaining: fail_at,
            },
            &CatalogDecodeLimits::default(),
        )
        .unwrap_err();
        assert!(matches!(error, Error::Io { .. }));
        assert!(error.to_string().contains("catalog test read failure"));
    }
    struct OneByteWriter(Vec<u8>);
    impl Write for OneByteWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.is_empty() {
                return Ok(0);
            }
            self.0.push(bytes[0]);
            Ok(1)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut writer = OneByteWriter(Vec::new());
    encode_into(&catalog, &mut writer).unwrap();
    assert_eq!(writer.0, bytes);
    struct ErrorAtEof<'a>(&'a [u8]);
    impl Read for ErrorAtEof<'_> {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
            if self.0.is_empty() {
                return Err(io::Error::other("catalog test EOF error"));
            }
            self.0.read(bytes)
        }
    }
    assert!(matches!(
        decode_from(&mut ErrorAtEof(&bytes), &CatalogDecodeLimits::default()),
        Err(Error::Io { .. })
    ));
}

#[test]
fn counting_pass_rejects_invalid_values_before_any_output() {
    let catalog = value_catalog(vec![Value::Extension(CompactArc::from(Vec::<u8>::new()))]);
    let mut writer = FailingWriter {
        remaining: usize::MAX,
        written: 0,
    };
    assert!(encode_into(&catalog, &mut writer).is_err());
    assert_eq!(writer.written, 0);
}

#[test]
fn history_builder_preserves_origins_without_reusing_removed_ids() {
    let identity = TableIdentity::new(TableId::new(1).unwrap(), Incarnation::FIRST);
    let mut builder = HistoryBuilder::new(identity);
    for version in 0..128 {
        let column = SchemaColumn::with_default_value(
            0,
            "c",
            DataType::Integer,
            false,
            false,
            false,
            None,
            Some(Value::Integer(version as i64)),
            None,
        );
        builder
            .push(
                version,
                Schema::new("t", vec![column]),
                vec![ColumnId::new(version + 1).unwrap()],
            )
            .unwrap();
    }
    let history = builder.finish(127, 500).unwrap();
    assert_eq!(history.revisions().len(), 128);
    assert_eq!(history.column_high_water_mark(), 500);
    let plan = history.projection(identity, 0).unwrap();
    let row = [Value::Integer(1)];
    assert_eq!(
        plan.project(identity, 0, &row).unwrap().get(0),
        Some(&Value::Integer(127))
    );
    let mut builder = HistoryBuilder::new(identity);
    let schema = || Schema::new("t", vec![SchemaColumn::nullable(0, "c", DataType::Integer)]);
    builder
        .push(0, schema(), vec![ColumnId::new(1).unwrap()])
        .unwrap();
    builder
        .push(1, schema(), vec![ColumnId::new(2).unwrap()])
        .unwrap();
    assert!(builder
        .push(2, schema(), vec![ColumnId::new(1).unwrap()])
        .is_err());
    let mut builder = HistoryBuilder::new(identity);
    builder
        .push(0, schema(), vec![ColumnId::new(9).unwrap()])
        .unwrap();
    assert!(builder.finish(0, 8).is_err());
    assert!(HistoryBuilder::new(identity).finish(0, 0).is_err());
}

#[test]
fn minimum_column_records_roundtrip_with_short_names() {
    let mut parts = value_catalog(vec![Value::Integer(1)]).parts().clone();
    let previous = &parts.tables[0].history;
    let name = previous
        .revisions()
        .next()
        .unwrap()
        .schema()
        .table_name
        .clone();
    let columns = ["", "a", "b", "c", "d", "e"]
        .into_iter()
        .enumerate()
        .map(|(position, name)| SchemaColumn::nullable(position, name, DataType::Integer))
        .collect();
    parts.tables[0].history = TableSchemaHistory::new(
        previous.identity(),
        0,
        Schema::new(name, columns),
        (1..=6).map(|id| ColumnId::new(id).unwrap()).collect(),
    )
    .unwrap();
    let catalog = CatalogGeneration::try_new(parts).unwrap();
    let bytes = encode(&catalog);
    assert_eq!(encode(&decode(&bytes).unwrap()), bytes);
}

#[test]
fn column_flags_and_optional_empty_sql_roundtrip_strictly() {
    for flags in 0..8 {
        for type_tag in 0..8 {
            let kind = DataType::from_u8(type_tag).unwrap();
            let column = SchemaColumn::with_default_value(
                0,
                "İ",
                kind,
                flags & 1 != 0,
                flags & 2 != 0,
                flags & 4 != 0,
                Some(String::new()),
                Some(Value::Null(kind)),
                Some(String::new()),
            )
            .with_vector_dimensions(if kind == DataType::Vector { 16 } else { 0 });
            let id = ColumnId::new(27).unwrap();
            let mut bytes = Vec::new();
            Encoder(&mut bytes)
                .record(|w| w.column(&column, id))
                .unwrap();
            let mut input = bytes.as_slice();
            let limits = CatalogDecodeLimits::default();
            let mut decoder = Decoder {
                input: &mut input,
                limits: &limits,
                hash: Hasher::new(),
                position: 0,
                end: bytes.len() as u64,
                records: 0,
                payload_bytes: 0,
                metadata_bytes: 0,
            };
            let (recovered, recovered_id) = decoder.record(|d| d.column(0)).unwrap();
            assert_eq!(column, recovered);
            assert_eq!(id, recovered_id);
            for offset in 0..3 {
                let mut corrupt = bytes.clone();
                corrupt[8 + 8 + 4 + "İ".len() + 1 + offset] = 2;
                let mut input = corrupt.as_slice();
                let mut decoder = Decoder {
                    input: &mut input,
                    limits: &limits,
                    hash: Hasher::new(),
                    position: 0,
                    end: corrupt.len() as u64,
                    records: 0,
                    payload_bytes: 0,
                    metadata_bytes: 0,
                };
                assert!(decoder.record(|d| d.column(0)).is_err());
            }
        }
    }
}
