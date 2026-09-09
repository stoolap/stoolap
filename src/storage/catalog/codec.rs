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

//! Strict STCG version 1 streaming metadata codec. Encoding is not installation
//! or a durable coverage acknowledgment. Decoding publishes only after exact
//! record boundaries, both checksums, EOF, and semantic validation succeed.
//!
//! Header (40 bytes, little endian): magic[4], version:u16, features:u16,
//! header_bytes:u32, body_bytes:u64, generation:u64, reserved:u64, crc32:u32.
//! Header CRC covers its first 36 bytes. The final u32 covers header and body.
//! The body contains allocation/coverage fields, mutations, tables, then views.
//! Lists have u32 counts; metadata list items and Values have u64 byte lengths.
//! Strings use u32 UTF-8 byte lengths; options and booleans use strict 0/1 tags.

use std::io::{self, Read, Write};
use std::num::NonZeroU64;

use chrono::{DateTime, Utc};
use crc32fast::Hasher;

use crate::common::CompactArc;
use crate::core::{
    DataType, Error, ForeignKeyAction, ForeignKeyConstraint, IndexType, Result, Schema,
    SchemaColumn, Value,
};

use super::generation::*;
use super::history::HistoryBuilder;
use super::{ColumnId, Incarnation, SchemaRevision, TableId, TableIdentity};

const HEADER_BYTES: usize = 40;
const VERSION: u16 = 1;

fn invalid(message: &'static str) -> Error {
    Error::internal(message)
}

/// Structural and requested-metadata quotas, not an allocator/RSS hard budget.
/// String allowance includes the original payload; requested metadata also
/// conservatively includes known cached name copies and Unicode expansion.
/// Collection quotas bound map/Arc overhead, whose allocator rounding is not
/// inferred from wire length. Private allocations can exist before footer CRC.
#[derive(Clone, Debug)]
pub struct CatalogDecodeLimits {
    pub max_wire_bytes: u64,
    pub max_records: u64,
    pub max_tables: u32,
    pub max_revisions_per_table: u32,
    pub max_columns_per_revision: u32,
    pub max_incarnations_per_table: u32,
    pub max_events_per_table: u32,
    pub max_indexes_per_table: u32,
    pub max_foreign_keys_per_revision: u32,
    pub max_mutations: u32,
    pub max_effects_per_mutation: u32,
    pub max_views: u32,
    pub max_string_bytes: u32,
    /// Complete encoded Value bytes, excluding its u64 record-length prefix.
    pub max_value_bytes: u32,
    pub max_total_payload_bytes: u64,
    pub max_requested_metadata_bytes: u64,
}

impl Default for CatalogDecodeLimits {
    fn default() -> Self {
        Self {
            max_wire_bytes: 256 * 1024 * 1024,
            max_records: 1_000_000,
            max_tables: 16_384,
            max_revisions_per_table: 65_536,
            max_columns_per_revision: 4096,
            max_incarnations_per_table: 65_536,
            max_events_per_table: 131_072,
            max_indexes_per_table: 65_536,
            max_foreign_keys_per_revision: 4096,
            max_mutations: 262_144,
            max_effects_per_mutation: 65_536,
            max_views: 65_536,
            max_string_bytes: 16 * 1024 * 1024,
            max_value_bytes: 16 * 1024 * 1024,
            max_total_payload_bytes: 128 * 1024 * 1024,
            max_requested_metadata_bytes: 512 * 1024 * 1024,
        }
    }
}

/// Successful encoding metadata, deliberately not a durability receipt. Keeps
/// the same immutable coverage declaration without cloning its mutation list.
#[derive(Clone, Debug)]
pub struct EncodedCatalogDescriptor {
    catalog: CatalogGeneration,
    bytes: u64,
    checksum: u32,
}

impl EncodedCatalogDescriptor {
    pub fn generation(&self) -> NonZeroU64 {
        self.catalog.generation()
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn checksum(&self) -> u32 {
        self.checksum
    }
    pub fn coverage(&self) -> &CatalogCoverage {
        self.catalog.coverage()
    }
}

#[derive(Default)]
struct Counter(u64);
impl Write for Counter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| io::Error::other("catalog encoded size overflow"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct ChecksumWriter<'a> {
    output: &'a mut dyn Write,
    hash: Hasher,
    bytes: u64,
}
impl Write for ChecksumWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let written = self.output.write(bytes)?;
        self.bytes = self
            .bytes
            .checked_add(written as u64)
            .ok_or_else(|| io::Error::other("catalog encoded size overflow"))?;
        self.hash.update(&bytes[..written]);
        Ok(written)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.output.flush()
    }
}

struct Encoder<'a>(&'a mut dyn Write);
impl Encoder<'_> {
    fn raw(&mut self, bytes: &[u8]) -> Result<()> {
        self.0.write_all(bytes)?;
        Ok(())
    }
    fn u8(&mut self, value: u8) -> Result<()> {
        self.raw(&[value])
    }
    fn u16(&mut self, value: u16) -> Result<()> {
        self.raw(&value.to_le_bytes())
    }
    fn u32(&mut self, value: u32) -> Result<()> {
        self.raw(&value.to_le_bytes())
    }
    fn u64(&mut self, value: u64) -> Result<()> {
        self.raw(&value.to_le_bytes())
    }
    fn count(&mut self, value: usize) -> Result<()> {
        self.u32(u32::try_from(value).map_err(|_| invalid("catalog count exceeds u32"))?)
    }
    fn string(&mut self, value: &str) -> Result<()> {
        self.count(value.len())?;
        self.raw(value.as_bytes())
    }
    fn option<T>(
        &mut self,
        value: Option<T>,
        f: impl FnOnce(&mut Self, T) -> Result<()>,
    ) -> Result<()> {
        self.u8(u8::from(value.is_some()))?;
        if let Some(value) = value {
            f(self, value)?;
        }
        Ok(())
    }
    fn record(&mut self, f: impl Fn(&mut Encoder<'_>) -> Result<()>) -> Result<()> {
        let mut size = Counter::default();
        f(&mut Encoder(&mut size))?;
        self.u64(size.0)?;
        f(self)
    }
    fn list<T>(
        &mut self,
        values: &[T],
        f: impl Fn(&mut Encoder<'_>, &T) -> Result<()>,
    ) -> Result<()> {
        self.count(values.len())?;
        for value in values {
            self.record(|writer| f(writer, value))?;
        }
        Ok(())
    }
    fn stamp(&mut self, value: DdlStamp) -> Result<()> {
        self.u64(value.epoch())?;
        self.u64(value.source_lsn())
    }
    fn identity(&mut self, value: TableIdentity) -> Result<()> {
        self.u64(value.table_id.get())?;
        self.u64(value.incarnation.get())
    }
    fn timestamp(&mut self, value: DateTime<Utc>) -> Result<()> {
        self.raw(&value.timestamp().to_le_bytes())?;
        self.u32(value.timestamp_subsec_nanos())
    }
    fn value(&mut self, value: &Value) -> Result<()> {
        match value {
            Value::Null(kind) => {
                self.u8(0)?;
                self.u8(kind.as_u8())
            }
            Value::Boolean(value) => {
                self.u8(1)?;
                self.u8(u8::from(*value))
            }
            Value::Integer(value) => {
                self.u8(2)?;
                self.raw(&value.to_le_bytes())
            }
            Value::Float(value) => {
                self.u8(3)?;
                self.u64(value.to_bits())
            }
            Value::Text(value) => {
                self.u8(4)?;
                self.string(value.as_str())
            }
            Value::Timestamp(value) => {
                self.u8(8)?;
                self.timestamp(*value)
            }
            Value::Extension(bytes) => match bytes.first().copied() {
                Some(tag) if tag == DataType::Json.as_u8() => {
                    let text = std::str::from_utf8(&bytes[1..])
                        .map_err(|_| invalid("invalid UTF-8 catalog JSON"))?;
                    self.u8(6)?;
                    self.string(text)
                }
                Some(tag) if tag == DataType::Vector.as_u8() => {
                    if (bytes.len() - 1) % 4 != 0 {
                        return Err(invalid("invalid catalog vector bytes"));
                    }
                    u32::try_from(bytes.len() - 1)
                        .map_err(|_| invalid("catalog vector payload exceeds u32"))?;
                    self.u8(10)?;
                    self.count((bytes.len() - 1) / 4)?;
                    self.raw(&bytes[1..])
                }
                _ => Err(invalid("invalid catalog Value extension tag")),
            },
        }
    }
    fn column(&mut self, column: &SchemaColumn, id: ColumnId) -> Result<()> {
        self.u64(id.get())?;
        self.string(&column.name)?;
        self.u8(column.data_type.as_u8())?;
        self.u8(u8::from(column.nullable))?;
        self.u8(u8::from(column.primary_key))?;
        self.u8(u8::from(column.auto_increment))?;
        self.u16(column.vector_dimensions)?;
        self.option(column.default_expr.as_deref(), |w, value| w.string(value))?;
        self.option(column.default_value.as_ref(), |w, value| {
            w.record(|w| w.value(value))
        })?;
        self.option(column.check_expr.as_deref(), |w, value| w.string(value))
    }
    fn revision(&mut self, revision: &SchemaRevision) -> Result<()> {
        let schema = revision.schema();
        self.u64(revision.version())?;
        self.string(&schema.table_name)?;
        self.timestamp(schema.created_at)?;
        self.timestamp(schema.updated_at)?;
        self.count(schema.columns.len())?;
        for (column, &id) in schema.columns.iter().zip(revision.column_ids()) {
            self.record(|w| w.column(column, id))?;
        }
        self.list(&schema.foreign_keys, |w, fk| {
            w.count(fk.column_index)?;
            w.string(&fk.column_name)?;
            w.string(&fk.referenced_table)?;
            w.string(&fk.referenced_column)?;
            w.u8(fk.on_delete.as_u8())?;
            w.u8(fk.on_update.as_u8())
        })
    }
    fn table(&mut self, table: &CatalogTable) -> Result<()> {
        let history = &table.history;
        self.identity(history.identity())?;
        self.u64(history.current_version())?;
        self.u64(history.column_high_water_mark())?;
        self.count(history.revisions().len())?;
        for revision in history.revisions() {
            self.record(|w| w.revision(revision))?;
        }
        self.list(&table.incarnations, |w, item| {
            w.u64(item.incarnation.get())?;
            w.stamp(item.started_at)?;
            w.u64(item.first_schema_version)?;
            w.u64(item.last_schema_version)?;
            w.option(item.ended, |w, end| {
                w.stamp(end.stamp)?;
                w.u8(match end.kind {
                    IncarnationEndKind::Truncate => 0,
                    IncarnationEndKind::Drop => 1,
                })
            })
        })?;
        self.list(&table.names, |w, item| {
            w.stamp(item.stamp)?;
            w.option(item.name.as_deref(), |w, value| w.string(value))
        })?;
        self.list(&table.schema_events, |w, item| {
            w.stamp(item.stamp)?;
            w.identity(item.identity)?;
            w.u64(item.version)
        })?;
        self.list(&table.foreign_keys, |w, item| {
            w.u64(item.schema_version)?;
            w.list(&item.bindings, |w, fk| {
                w.stamp(fk.defined_at)?;
                w.u64(fk.local_column.get())?;
                w.u64(fk.parent_table.get())?;
                w.u64(fk.parent_column.get())?;
                w.u64(fk.parent_schema_version)
            })
        })?;
        self.list(&table.indexes, |w, index| {
            w.string(&index.name)?;
            w.stamp(index.defined_at)?;
            w.option(index.dropped_at, |w, value| w.stamp(value))?;
            w.u64(index.schema_version)?;
            w.count(index.columns.len())?;
            for id in &index.columns {
                w.u64(id.get())?;
            }
            w.u8(match index.index_type {
                IndexType::BTree => 0,
                IndexType::Hash => 1,
                IndexType::Bitmap => 2,
                IndexType::MultiColumn => 3,
                IndexType::Hnsw => 5,
                IndexType::PrimaryKey => return Err(invalid("implicit primary index in catalog")),
            })?;
            w.u8(u8::from(index.is_unique))?;
            w.option(index.hnsw.m, |w, value| w.u16(value))?;
            w.option(index.hnsw.ef_construction, |w, value| w.u16(value))?;
            w.option(index.hnsw.ef_search, |w, value| w.u16(value))?;
            w.option(index.hnsw.distance_metric, |w, value| w.u8(value))
        })
    }
    fn body(&mut self, catalog: &CatalogGeneration) -> Result<()> {
        let parts = catalog.parts();
        self.u64(parts.table_id_high_water_mark)?;
        self.u64(parts.ddl_epoch_high_water_mark)?;
        self.u64(parts.wal_observation_ceiling)?;
        self.u64(parts.coverage.through_lsn)?;
        self.u64(parts.coverage.ddl_epoch_cut)?;
        self.list(&parts.coverage.captured_mutations, |w, mutation| {
            w.stamp(mutation.stamp)?;
            w.u8(match mutation.kind {
                DdlKind::CreateTable => 0,
                DdlKind::AlterTable => 1,
                DdlKind::RenameTable => 2,
                DdlKind::TruncateTable => 3,
                DdlKind::DropTable => 4,
                DdlKind::CreateIndex => 5,
                DdlKind::DropIndex => 6,
                DdlKind::CreateView => 7,
                DdlKind::DropView => 8,
            })?;
            w.list(&mutation.effects, |w, effect| match effect {
                CatalogEffect::Table {
                    identity,
                    schema_version,
                } => {
                    w.u8(0)?;
                    w.identity(*identity)?;
                    w.u64(*schema_version)
                }
                CatalogEffect::Index {
                    identity,
                    name,
                    definition_epoch,
                } => {
                    w.u8(1)?;
                    w.identity(*identity)?;
                    w.string(name)?;
                    w.u64(*definition_epoch)
                }
                CatalogEffect::View {
                    name,
                    definition_epoch,
                } => {
                    w.u8(2)?;
                    w.string(name)?;
                    w.u64(*definition_epoch)
                }
            })
        })?;
        self.list(&parts.tables, |w, table| w.table(table))?;
        self.list(&parts.views, |w, view| {
            w.string(&view.original_name)?;
            w.string(&view.query)?;
            w.stamp(view.defined_at)?;
            w.option(view.dropped_at, |w, value| w.stamp(value))
        })
    }
}

pub fn encode_into<W: Write>(
    catalog: &CatalogGeneration,
    output: &mut W,
) -> Result<EncodedCatalogDescriptor> {
    // CatalogGeneration is immutable and already semantically validated. The
    // counting pass additionally validates each encoded Value and checked size.
    let mut size = Counter::default();
    Encoder(&mut size).body(catalog)?;
    let bytes = size
        .0
        .checked_add(HEADER_BYTES as u64 + 4)
        .ok_or_else(|| invalid("catalog encoded size overflow"))?;
    let mut header = [0u8; HEADER_BYTES];
    header[..4].copy_from_slice(b"STCG");
    header[4..6].copy_from_slice(&VERSION.to_le_bytes());
    header[8..12].copy_from_slice(&(HEADER_BYTES as u32).to_le_bytes());
    header[12..20].copy_from_slice(&size.0.to_le_bytes());
    header[20..28].copy_from_slice(&catalog.generation().get().to_le_bytes());
    let header_crc = crc32fast::hash(&header[..36]);
    header[36..].copy_from_slice(&header_crc.to_le_bytes());
    let mut writer = ChecksumWriter {
        output,
        hash: Hasher::new(),
        bytes: 0,
    };
    writer.write_all(&header)?;
    Encoder(&mut writer).body(catalog)?;
    if writer.bytes != bytes - 4 {
        return Err(invalid("catalog counting pass disagrees with output"));
    }
    let checksum = writer.hash.finalize();
    writer.output.write_all(&checksum.to_le_bytes())?;
    Ok(EncodedCatalogDescriptor {
        catalog: catalog.clone(),
        bytes,
        checksum,
    })
}

struct Decoder<'a> {
    input: &'a mut dyn Read,
    limits: &'a CatalogDecodeLimits,
    hash: Hasher,
    position: u64,
    end: u64,
    records: u64,
    payload_bytes: u64,
    metadata_bytes: u64,
}

impl Decoder<'_> {
    fn remaining(&self) -> u64 {
        self.end - self.position
    }
    fn read(&mut self, bytes: &mut [u8]) -> Result<()> {
        if bytes.len() as u64 > self.remaining() {
            return Err(invalid("catalog field exceeds record boundary"));
        }
        self.input.read_exact(bytes)?;
        self.hash.update(bytes);
        self.position += bytes.len() as u64;
        Ok(())
    }
    fn fixed<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut bytes = [0; N];
        self.read(&mut bytes)?;
        Ok(bytes)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.fixed::<1>()?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.fixed()?))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.fixed()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.fixed()?))
    }
    fn boolean(&mut self) -> Result<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid("invalid catalog boolean/presence tag")),
        }
    }
    fn option<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<Option<T>> {
        if self.boolean()? {
            f(self).map(Some)
        } else {
            Ok(None)
        }
    }
    fn charge_metadata(&mut self, bytes: u64) -> Result<()> {
        self.metadata_bytes = self
            .metadata_bytes
            .checked_add(bytes)
            .filter(|&total| total <= self.limits.max_requested_metadata_bytes)
            .ok_or_else(|| invalid("catalog requested metadata limit exceeded"))?;
        Ok(())
    }
    fn count<T>(&mut self, limit: u32, minimum_wire_bytes: u64) -> Result<usize> {
        let count = self.u32()?;
        if count > limit
            || u64::from(count)
                .checked_mul(minimum_wire_bytes)
                .is_none_or(|bytes| bytes > self.remaining())
        {
            return Err(invalid("catalog count exceeds limit or enclosing bytes"));
        }
        self.records = self
            .records
            .checked_add(u64::from(count))
            .filter(|&total| total <= self.limits.max_records)
            .ok_or_else(|| invalid("catalog total record limit exceeded"))?;
        let count = usize::try_from(count).map_err(|_| invalid("catalog count exceeds usize"))?;
        self.charge_metadata(
            (count as u64)
                .checked_mul(std::mem::size_of::<T>() as u64)
                .ok_or_else(|| invalid("catalog metadata size overflow"))?,
        )?;
        Ok(count)
    }
    fn reserve<T>(&self, count: usize) -> Result<Vec<T>> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| invalid("catalog allocation failed"))?;
        Ok(values)
    }
    fn record<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        let length = self.u64()?;
        if length > self.remaining() {
            return Err(invalid("catalog record exceeds enclosing bytes"));
        }
        let previous_end = self.end;
        self.end = self.position + length;
        let result = f(self).and_then(|value| {
            if self.position != self.end {
                Err(invalid("trailing bytes in catalog record"))
            } else {
                Ok(value)
            }
        });
        self.end = previous_end;
        result
    }
    fn list<T>(
        &mut self,
        limit: u32,
        minimum: u64,
        mut f: impl FnMut(&mut Self) -> Result<T>,
    ) -> Result<Vec<T>> {
        let count = self.count::<T>(limit, minimum)?;
        let mut values = self.reserve(count)?;
        for _ in 0..count {
            values.push(self.record(&mut f)?);
        }
        Ok(values)
    }
    fn bytes(
        &mut self,
        length: u32,
        limit: u32,
        copies: u64,
        prefix: Option<u8>,
    ) -> Result<Vec<u8>> {
        if length > limit || u64::from(length) > self.remaining() {
            return Err(invalid("catalog payload exceeds limit or enclosing bytes"));
        }
        self.payload_bytes = self
            .payload_bytes
            .checked_add(u64::from(length))
            .filter(|&total| total <= self.limits.max_total_payload_bytes)
            .ok_or_else(|| invalid("catalog total payload limit exceeded"))?;
        self.charge_metadata(
            u64::from(length)
                .checked_mul(copies)
                .and_then(|n| n.checked_add(u64::from(prefix.is_some()) * copies))
                .ok_or_else(|| invalid("catalog payload accounting overflow"))?,
        )?;
        let length =
            usize::try_from(length).map_err(|_| invalid("catalog length exceeds usize"))?;
        let total = length
            .checked_add(usize::from(prefix.is_some()))
            .ok_or_else(|| invalid("catalog payload allocation overflow"))?;
        let mut bytes = self.reserve(total)?;
        bytes.resize(total, 0);
        if let Some(tag) = prefix {
            bytes[0] = tag;
        }
        self.read(&mut bytes[usize::from(prefix.is_some())..])?;
        Ok(bytes)
    }
    fn string_with_limit(&mut self, copies: u64, limit: u32) -> Result<String> {
        let length = self.u32()?;
        let bytes = self.bytes(
            length,
            limit.min(self.limits.max_string_bytes),
            copies,
            None,
        )?;
        String::from_utf8(bytes).map_err(|_| invalid("invalid UTF-8 catalog string"))
    }
    fn string(&mut self) -> Result<String> {
        self.string_with_limit(1, self.limits.max_string_bytes)
    }
    fn name(&mut self, copies: u64) -> Result<String> {
        self.string_with_limit(copies, self.limits.max_string_bytes)
    }
    fn stamp(&mut self) -> Result<DdlStamp> {
        DdlStamp::new(self.u64()?, self.u64()?)
    }
    fn identity(&mut self) -> Result<TableIdentity> {
        Ok(TableIdentity::new(
            TableId::new(self.u64()?)?,
            Incarnation::new(self.u64()?)?,
        ))
    }
    fn timestamp(&mut self) -> Result<DateTime<Utc>> {
        let seconds = i64::from_le_bytes(self.fixed()?);
        let nanos = self.u32()?;
        DateTime::from_timestamp(seconds, nanos).ok_or_else(|| invalid("invalid catalog timestamp"))
    }
    fn data_type(&mut self) -> Result<DataType> {
        DataType::from_u8(self.u8()?).ok_or_else(|| invalid("unknown catalog data type"))
    }
    fn action(&mut self) -> Result<ForeignKeyAction> {
        ForeignKeyAction::from_u8(self.u8()?)
            .ok_or_else(|| invalid("unknown catalog foreign key action"))
    }
    fn value(&mut self) -> Result<Value> {
        if self.remaining() > u64::from(self.limits.max_value_bytes) {
            return Err(invalid("catalog Value record exceeds limit"));
        }
        match self.u8()? {
            0 => Ok(Value::Null(self.data_type()?)),
            1 => Ok(Value::Boolean(self.boolean()?)),
            2 => Ok(Value::Integer(i64::from_le_bytes(self.fixed()?))),
            3 => Ok(Value::Float(f64::from_bits(self.u64()?))),
            4 => Ok(Value::text(
                self.string_with_limit(2, self.limits.max_value_bytes)?,
            )),
            8 => Ok(Value::Timestamp(self.timestamp()?)),
            tag @ (6 | 10) => {
                let length = self.u32()?;
                let (length, kind) = if tag == 10 {
                    (
                        length
                            .checked_mul(4)
                            .ok_or_else(|| invalid("catalog vector byte size overflow"))?,
                        DataType::Vector,
                    )
                } else {
                    (length, DataType::Json)
                };
                // Vec and final CompactArc may coexist during conversion.
                let bytes =
                    self.bytes(length, self.limits.max_value_bytes, 2, Some(kind.as_u8()))?;
                if kind == DataType::Json && std::str::from_utf8(&bytes[1..]).is_err() {
                    return Err(invalid("invalid UTF-8 catalog JSON"));
                }
                Ok(Value::Extension(CompactArc::from(bytes)))
            }
            _ => Err(invalid("unknown or legacy catalog Value tag")),
        }
    }
    fn column(&mut self, position: usize) -> Result<(SchemaColumn, ColumnId)> {
        let id = ColumnId::new(self.u64()?)?;
        // Original + cached original + three lowercase copies, allowing 3x
        // UTF-8 expansion for each. This is conservative, not exact capacity.
        let name = self.name(11)?;
        let data_type = self.data_type()?;
        let nullable = self.boolean()?;
        let primary_key = self.boolean()?;
        let auto_increment = self.boolean()?;
        let dimensions = self.u16()?;
        let default_expr = self.option(Self::string)?;
        let default_value = self.option(|d| d.record(Self::value))?;
        let check_expr = self.option(Self::string)?;
        let column = SchemaColumn::with_default_value(
            position,
            name,
            data_type,
            nullable,
            primary_key,
            auto_increment,
            default_expr,
            default_value,
            check_expr,
        )
        .with_vector_dimensions(dimensions);
        Ok((column, id))
    }
    fn revision(&mut self, builder: &mut HistoryBuilder) -> Result<()> {
        let version = self.u64()?;
        let table_name = self.name(4)?;
        let created = self.timestamp()?;
        let updated = self.timestamp()?;
        // Include the parallel identity vector and eagerly rebuilt Schema cache
        // slots in the structural quota before any column reservation.
        let count = self.count::<(SchemaColumn, ColumnId, [usize; 12])>(
            self.limits.max_columns_per_revision,
            29,
        )?;
        let mut columns = self.reserve(count)?;
        let mut ids = self.reserve(count)?;
        for position in 0..count {
            let (column, id) = self.record(|d| d.column(position))?;
            columns.push(column);
            ids.push(id);
        }
        let foreign_keys = self.list(self.limits.max_foreign_keys_per_revision, 26, |d| {
            let column_index = usize::try_from(d.u32()?)
                .map_err(|_| invalid("catalog FK position exceeds usize"))?;
            Ok(ForeignKeyConstraint {
                column_index,
                column_name: d.string()?,
                referenced_table: d.string()?,
                referenced_column: d.string()?,
                on_delete: d.action()?,
                on_update: d.action()?,
            })
        })?;
        builder.push(
            version,
            Schema::with_timestamps_and_foreign_keys(
                table_name,
                columns,
                foreign_keys,
                created,
                updated,
            ),
            ids,
        )
    }
    fn table(&mut self) -> Result<CatalogTable> {
        let identity = self.identity()?;
        let current_version = self.u64()?;
        let high_water_mark = self.u64()?;
        let revisions =
            self.count::<(SchemaRevision, [usize; 12])>(self.limits.max_revisions_per_table, 52)?;
        let mut builder = HistoryBuilder::new(identity);
        for _ in 0..revisions {
            self.record(|d| d.revision(&mut builder))?;
        }
        let history = builder.finish(current_version, high_water_mark)?;
        let incarnations = self.list(self.limits.max_incarnations_per_table, 49, |d| {
            Ok(CatalogIncarnation {
                incarnation: Incarnation::new(d.u64()?)?,
                started_at: d.stamp()?,
                first_schema_version: d.u64()?,
                last_schema_version: d.u64()?,
                ended: d.option(|d| {
                    let stamp = d.stamp()?;
                    let kind = match d.u8()? {
                        0 => IncarnationEndKind::Truncate,
                        1 => IncarnationEndKind::Drop,
                        _ => return Err(invalid("unknown catalog incarnation end kind")),
                    };
                    Ok(IncarnationEnd { stamp, kind })
                })?,
            })
        })?;
        let names = self.list(self.limits.max_events_per_table, 25, |d| {
            Ok(TableNameEvent {
                stamp: d.stamp()?,
                name: d.option(Self::string)?,
            })
        })?;
        let schema_events = self.list(self.limits.max_events_per_table, 48, |d| {
            Ok(SchemaEvent {
                stamp: d.stamp()?,
                identity: d.identity()?,
                version: d.u64()?,
            })
        })?;
        let foreign_keys = self.list(self.limits.max_revisions_per_table, 20, |d| {
            Ok(SchemaForeignKeys {
                schema_version: d.u64()?,
                bindings: d.list(d.limits.max_foreign_keys_per_revision, 56, |d| {
                    Ok(ForeignKeyBinding {
                        defined_at: d.stamp()?,
                        local_column: ColumnId::new(d.u64()?)?,
                        parent_table: TableId::new(d.u64()?)?,
                        parent_column: ColumnId::new(d.u64()?)?,
                        parent_schema_version: d.u64()?,
                    })
                })?,
            })
        })?;
        let indexes = self.list(self.limits.max_indexes_per_table, 47, |d| {
            let name = d.string()?;
            let defined_at = d.stamp()?;
            let dropped_at = d.option(Self::stamp)?;
            let schema_version = d.u64()?;
            let count = d.count::<ColumnId>(d.limits.max_columns_per_revision, 8)?;
            let mut columns = d.reserve(count)?;
            for _ in 0..count {
                columns.push(ColumnId::new(d.u64()?)?);
            }
            let index_type = match d.u8()? {
                0 => IndexType::BTree,
                1 => IndexType::Hash,
                2 => IndexType::Bitmap,
                3 => IndexType::MultiColumn,
                5 => IndexType::Hnsw,
                _ => return Err(invalid("unknown or implicit catalog index type")),
            };
            let is_unique = d.boolean()?;
            let hnsw = CatalogHnswOptions {
                m: d.option(Self::u16)?,
                ef_construction: d.option(Self::u16)?,
                ef_search: d.option(Self::u16)?,
                distance_metric: d.option(Self::u8)?,
            };
            Ok(CatalogIndex {
                name,
                defined_at,
                dropped_at,
                schema_version,
                columns,
                index_type,
                is_unique,
                hnsw,
            })
        })?;
        Ok(CatalogTable {
            history,
            incarnations,
            names,
            schema_events,
            foreign_keys,
            indexes,
        })
    }
    fn body(&mut self, generation: NonZeroU64) -> Result<CatalogParts> {
        let table_id_high_water_mark = self.u64()?;
        let ddl_epoch_high_water_mark = self.u64()?;
        let wal_observation_ceiling = self.u64()?;
        let through_lsn = self.u64()?;
        let ddl_epoch_cut = self.u64()?;
        let captured_mutations = self.list(self.limits.max_mutations, 29, |d| {
            let stamp = d.stamp()?;
            let kind = match d.u8()? {
                0 => DdlKind::CreateTable,
                1 => DdlKind::AlterTable,
                2 => DdlKind::RenameTable,
                3 => DdlKind::TruncateTable,
                4 => DdlKind::DropTable,
                5 => DdlKind::CreateIndex,
                6 => DdlKind::DropIndex,
                7 => DdlKind::CreateView,
                8 => DdlKind::DropView,
                _ => return Err(invalid("unknown catalog DDL kind")),
            };
            let effects = d.list(d.limits.max_effects_per_mutation, 21, |d| match d.u8()? {
                0 => Ok(CatalogEffect::Table {
                    identity: d.identity()?,
                    schema_version: d.u64()?,
                }),
                1 => Ok(CatalogEffect::Index {
                    identity: d.identity()?,
                    name: d.string()?,
                    definition_epoch: d.u64()?,
                }),
                2 => Ok(CatalogEffect::View {
                    name: d.string()?,
                    definition_epoch: d.u64()?,
                }),
                _ => Err(invalid("unknown catalog effect kind")),
            })?;
            Ok(CatalogMutation {
                stamp,
                kind,
                effects,
            })
        })?;
        let tables = self.list(self.limits.max_tables, 64, Self::table)?;
        let views = self.list(self.limits.max_views, 33, |d| {
            Ok(CatalogView {
                original_name: d.string()?,
                query: d.string()?,
                defined_at: d.stamp()?,
                dropped_at: d.option(Self::stamp)?,
            })
        })?;
        Ok(CatalogParts {
            generation,
            table_id_high_water_mark,
            ddl_epoch_high_water_mark,
            wal_observation_ceiling,
            coverage: CatalogCoverage {
                through_lsn,
                ddl_epoch_cut,
                captured_mutations,
            },
            tables,
            views,
        })
    }
}

pub fn decode_from<R: Read>(
    input: &mut R,
    limits: &CatalogDecodeLimits,
) -> Result<CatalogGeneration> {
    if limits.max_wire_bytes < HEADER_BYTES as u64 + 4 {
        return Err(invalid("catalog wire limit is below header size"));
    }
    let mut header = [0u8; HEADER_BYTES];
    input.read_exact(&mut header)?;
    if &header[..4] != b"STCG"
        || u16::from_le_bytes(header[4..6].try_into().unwrap()) != VERSION
        || header[6..8] != [0; 2]
        || u32::from_le_bytes(header[8..12].try_into().unwrap()) != HEADER_BYTES as u32
        || header[28..36] != [0; 8]
    {
        return Err(invalid("unsupported catalog header"));
    }
    if crc32fast::hash(&header[..36]) != u32::from_le_bytes(header[36..40].try_into().unwrap()) {
        return Err(invalid("catalog header checksum mismatch"));
    }
    let body_bytes = u64::from_le_bytes(header[12..20].try_into().unwrap());
    if body_bytes
        .checked_add(HEADER_BYTES as u64 + 4)
        .is_none_or(|total| total > limits.max_wire_bytes)
    {
        return Err(invalid("catalog wire length exceeds limit"));
    }
    let generation = NonZeroU64::new(u64::from_le_bytes(header[20..28].try_into().unwrap()))
        .ok_or_else(|| invalid("zero catalog generation"))?;
    let mut hash = Hasher::new();
    hash.update(&header);
    let mut decoder = Decoder {
        input,
        limits,
        hash,
        position: 0,
        end: body_bytes,
        records: 0,
        payload_bytes: 0,
        metadata_bytes: 0,
    };
    let parts = decoder.body(generation)?;
    if decoder.remaining() != 0 {
        return Err(invalid("trailing bytes in catalog body"));
    }
    let checksum = decoder.hash.finalize();
    let mut footer = [0; 4];
    decoder.input.read_exact(&mut footer)?;
    if checksum != u32::from_le_bytes(footer) {
        return Err(invalid("catalog checksum mismatch"));
    }
    let mut extra = [0; 1];
    loop {
        match decoder.input.read(&mut extra) {
            Ok(0) => break,
            Ok(_) => return Err(invalid("trailing bytes after catalog footer")),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        }
    }
    CatalogGeneration::try_new(parts)
}

#[cfg(test)]
mod tests;
