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

//! TiKV secondary index implementation
//!
//! Stores index entries as TiKV key-value pairs with order-preserving encoding.
//! Key format: i{table_id}{index_id}{encoded_values}{row_id} → empty value
//! For unique indexes: i{table_id}{index_id}{encoded_values} → row_id (8 bytes)

use parking_lot::Mutex;
use std::sync::Arc;

use crate::common::I64Map;
use crate::core::{DataType, Error, IndexEntry, IndexType, Operator, Result, RowIdVec, Value};
use crate::storage::expression::Expression;
use crate::storage::traits::Index;

use super::encoding;
use super::error::from_tikv_error;

/// Index metadata stored in memory (serialized as JSON to TiKV)
#[derive(Clone, Debug)]
pub struct IndexMetadata {
    pub name: String,
    pub table_name: String,
    pub table_id: u64,
    pub index_id: u64,
    pub column_names: Vec<String>,
    pub column_ids: Vec<i32>,
    pub data_types: Vec<DataType>,
    pub is_unique: bool,
}

impl IndexMetadata {
    /// Serialize to JSON bytes for TiKV storage
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let data_type_strs: Vec<String> = self
            .data_types
            .iter()
            .map(|dt| format!("{:?}", dt))
            .collect();
        let json = serde_json::json!({
            "name": self.name,
            "table_name": self.table_name,
            "table_id": self.table_id,
            "index_id": self.index_id,
            "column_names": self.column_names,
            "column_ids": self.column_ids,
            "data_types": data_type_strs,
            "is_unique": self.is_unique,
        });
        serde_json::to_vec(&json)
            .map_err(|e| Error::internal(format!("serialize index meta: {}", e)))
    }

    /// Deserialize from JSON bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let json: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| Error::internal(format!("deserialize index meta: {}", e)))?;
        let data_type_strs: Vec<String> = json["data_types"]
            .as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .map(|v| v.as_str().unwrap_or("Text").to_string())
            .collect();
        let data_types: Vec<DataType> = data_type_strs
            .iter()
            .map(|s| match s.as_str() {
                "Integer" => DataType::Integer,
                "Float" => DataType::Float,
                "Boolean" => DataType::Boolean,
                "Timestamp" => DataType::Timestamp,
                "Json" => DataType::Json,
                _ => DataType::Text,
            })
            .collect();
        Ok(IndexMetadata {
            name: json["name"].as_str().unwrap_or("").to_string(),
            table_name: json["table_name"].as_str().unwrap_or("").to_string(),
            table_id: json["table_id"].as_u64().unwrap_or(0),
            index_id: json["index_id"].as_u64().unwrap_or(0),
            column_names: json["column_names"]
                .as_array()
                .unwrap_or(&Vec::new())
                .iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect(),
            column_ids: json["column_ids"]
                .as_array()
                .unwrap_or(&Vec::new())
                .iter()
                .map(|v| v.as_i64().unwrap_or(0) as i32)
                .collect(),
            data_types,
            is_unique: json["is_unique"].as_bool().unwrap_or(false),
        })
    }
}

/// TiKV secondary index
pub struct TiKVIndex {
    meta: IndexMetadata,
    txn: Arc<Mutex<Option<tikv_client::Transaction>>>,
    runtime: tokio::runtime::Handle,
}

impl TiKVIndex {
    pub fn new(
        meta: IndexMetadata,
        txn: Arc<Mutex<Option<tikv_client::Transaction>>>,
        runtime: tokio::runtime::Handle,
    ) -> Self {
        Self { meta, txn, runtime }
    }

    /// Helper to run an operation on the transaction
    fn with_txn<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut tikv_client::Transaction) -> Result<R>,
    {
        let mut guard = self.txn.lock();
        match guard.as_mut() {
            Some(txn) => f(txn),
            None => Err(Error::internal(
                "Transaction already committed or rolled back",
            )),
        }
    }

    /// Make an index key with value prefix for scanning a specific value
    fn make_value_prefix(&self, values: &[Value]) -> Vec<u8> {
        let mut key = encoding::make_index_prefix(self.meta.table_id, self.meta.index_id);
        for v in values {
            key.extend_from_slice(&encoding::encode_value(v));
        }
        key
    }

    /// Scan all index entries with a given value prefix, returning (row_id, ref_id) pairs
    fn scan_value_prefix(&self, prefix: &[u8]) -> Result<Vec<IndexEntry>> {
        let end = encoding::prefix_end_key(prefix);
        let prefix_clone = prefix.to_vec();

        self.with_txn(|txn| {
            let pairs = self.runtime.block_on(async {
                txn.scan(prefix_clone..end, u32::MAX)
                    .await
                    .map_err(from_tikv_error)
            })?;

            let mut entries = Vec::new();
            for pair in pairs {
                let key: Vec<u8> = pair.0.into();
                if self.meta.is_unique {
                    // Unique index: value is row_id
                    let value: Vec<u8> = pair.1;
                    if value.len() >= 8 {
                        let row_id = encoding::decode_i64(&value);
                        entries.push(IndexEntry::new(row_id, row_id));
                    }
                } else {
                    // Non-unique index: row_id is at the end of the key
                    if key.len() >= 8 {
                        let row_id = encoding::decode_i64(&key[key.len() - 8..]);
                        entries.push(IndexEntry::new(row_id, row_id));
                    }
                }
            }
            Ok(entries)
        })
    }
}

impl Index for TiKVIndex {
    fn name(&self) -> &str {
        &self.meta.name
    }

    fn table_name(&self) -> &str {
        &self.meta.table_name
    }

    fn build(&mut self) -> Result<()> {
        // Index is built by scanning existing rows and inserting entries
        // This is handled at a higher level (CREATE INDEX)
        Ok(())
    }

    fn add(&self, values: &[Value], row_id: i64, _ref_id: i64) -> Result<()> {
        if self.meta.is_unique {
            // Check for existing entry with same values
            let prefix = self.make_value_prefix(values);
            let existing = self.scan_value_prefix(&prefix)?;
            if let Some(entry) = existing.first() {
                if entry.row_id != row_id {
                    return Err(Error::unique_constraint(
                        &self.meta.name,
                        self.meta.column_names.join(","),
                        format!("{:?}", values),
                    ));
                }
            }
            // Store: key = prefix (without row_id), value = row_id bytes
            self.with_txn(|txn| {
                self.runtime.block_on(async {
                    txn.put(prefix, encoding::encode_i64(row_id).to_vec())
                        .await
                        .map_err(from_tikv_error)
                })
            })
        } else {
            // Non-unique: key includes row_id for uniqueness
            let key =
                encoding::make_index_key(self.meta.table_id, self.meta.index_id, values, row_id);
            self.with_txn(|txn| {
                self.runtime
                    .block_on(async { txn.put(key, vec![]).await.map_err(from_tikv_error) })
            })
        }
    }

    fn add_batch(&self, entries: &I64Map<Vec<Value>>) -> Result<()> {
        for row_id in entries.keys() {
            if let Some(values) = entries.get(row_id) {
                self.add(values, row_id, row_id)?;
            }
        }
        Ok(())
    }

    fn remove(&self, values: &[Value], row_id: i64, _ref_id: i64) -> Result<()> {
        if self.meta.is_unique {
            let prefix = self.make_value_prefix(values);
            self.with_txn(|txn| {
                self.runtime
                    .block_on(async { txn.delete(prefix).await.map_err(from_tikv_error) })
            })
        } else {
            let key =
                encoding::make_index_key(self.meta.table_id, self.meta.index_id, values, row_id);
            self.with_txn(|txn| {
                self.runtime
                    .block_on(async { txn.delete(key).await.map_err(from_tikv_error) })
            })
        }
    }

    fn remove_batch(&self, entries: &I64Map<Vec<Value>>) -> Result<()> {
        for row_id in entries.keys() {
            if let Some(values) = entries.get(row_id) {
                self.remove(values, row_id, row_id)?;
            }
        }
        Ok(())
    }

    fn column_ids(&self) -> &[i32] {
        &self.meta.column_ids
    }

    fn column_names(&self) -> &[String] {
        &self.meta.column_names
    }

    fn data_types(&self) -> &[DataType] {
        &self.meta.data_types
    }

    fn index_type(&self) -> IndexType {
        IndexType::BTree
    }

    fn is_unique(&self) -> bool {
        self.meta.is_unique
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn find(&self, values: &[Value]) -> Result<Vec<IndexEntry>> {
        let prefix = self.make_value_prefix(values);
        self.scan_value_prefix(&prefix)
    }

    fn find_range(
        &self,
        min: &[Value],
        max: &[Value],
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Result<Vec<IndexEntry>> {
        let min_prefix = self.make_value_prefix(min);
        let max_prefix = self.make_value_prefix(max);
        let max_end = encoding::prefix_end_key(&max_prefix);

        // Scan the range between min and max
        let start = min_prefix.clone();
        let end = if max_inclusive {
            max_end
        } else {
            max_prefix.clone()
        };

        self.with_txn(|txn| {
            let pairs = self.runtime.block_on(async {
                txn.scan(start..end, u32::MAX)
                    .await
                    .map_err(from_tikv_error)
            })?;

            let mut entries = Vec::new();
            for pair in pairs {
                let key: Vec<u8> = pair.0.into();
                let row_id = if self.meta.is_unique {
                    let value: Vec<u8> = pair.1;
                    if value.len() >= 8 {
                        encoding::decode_i64(&value)
                    } else {
                        continue;
                    }
                } else if key.len() >= 8 {
                    encoding::decode_i64(&key[key.len() - 8..])
                } else {
                    continue;
                };

                // Check inclusiveness for min boundary
                if !min_inclusive && key.starts_with(&min_prefix) {
                    // Need to check if this is exactly the min value
                    // For non-unique, the key has row_id appended, so prefix match is the value
                    continue;
                }

                entries.push(IndexEntry::new(row_id, row_id));
            }
            Ok(entries)
        })
    }

    fn find_with_operator(&self, op: Operator, values: &[Value]) -> Result<Vec<IndexEntry>> {
        match op {
            Operator::Eq => self.find(values),
            Operator::Lt | Operator::Lte | Operator::Gt | Operator::Gte => {
                let index_prefix =
                    encoding::make_index_prefix(self.meta.table_id, self.meta.index_id);
                let index_end = encoding::prefix_end_key(&index_prefix);
                let value_prefix = self.make_value_prefix(values);

                let (start, end) = match op {
                    Operator::Lt => (index_prefix, value_prefix),
                    Operator::Lte => (index_prefix, encoding::prefix_end_key(&value_prefix)),
                    Operator::Gt => (encoding::prefix_end_key(&value_prefix), index_end),
                    Operator::Gte => (value_prefix, index_end),
                    _ => unreachable!(),
                };

                self.with_txn(|txn| {
                    let pairs = self.runtime.block_on(async {
                        txn.scan(start..end, u32::MAX)
                            .await
                            .map_err(from_tikv_error)
                    })?;

                    let mut entries = Vec::new();
                    for pair in pairs {
                        let key: Vec<u8> = pair.0.into();
                        let row_id = if self.meta.is_unique {
                            let value: Vec<u8> = pair.1;
                            if value.len() >= 8 {
                                encoding::decode_i64(&value)
                            } else {
                                continue;
                            }
                        } else if key.len() >= 8 {
                            encoding::decode_i64(&key[key.len() - 8..])
                        } else {
                            continue;
                        };
                        entries.push(IndexEntry::new(row_id, row_id));
                    }
                    Ok(entries)
                })
            }
            _ => {
                // For other operators, fall back to full scan
                Ok(Vec::new())
            }
        }
    }

    fn get_filtered_row_ids(&self, _expr: &dyn Expression) -> RowIdVec {
        // Expression-based filtering requires evaluating arbitrary expressions
        // against index values — fall back to empty (executor will do full scan)
        RowIdVec::new()
    }

    fn get_min_value(&self) -> Option<Value> {
        let prefix = encoding::make_index_prefix(self.meta.table_id, self.meta.index_id);
        let end = encoding::prefix_end_key(&prefix);

        self.with_txn(|txn| {
            let pairs = self
                .runtime
                .block_on(async { txn.scan(prefix..end, 1).await.map_err(from_tikv_error) })?;

            if let Some(pair) = pairs.into_iter().next() {
                let key: Vec<u8> = pair.0.into();
                // Skip index prefix (1 + 8 + 8 = 17 bytes) to get to value bytes
                let value_start = 17;
                if key.len() > value_start {
                    return Ok(Some(encoding::decode_value(&key[value_start..])));
                }
            }
            Ok(None)
        })
        .ok()
        .flatten()
    }

    fn get_max_value(&self) -> Option<Value> {
        let prefix = encoding::make_index_prefix(self.meta.table_id, self.meta.index_id);
        let end = encoding::prefix_end_key(&prefix);

        self.with_txn(|txn| {
            let pairs = self.runtime.block_on(async {
                txn.scan_reverse(prefix..end, 1)
                    .await
                    .map_err(from_tikv_error)
            })?;

            if let Some(pair) = pairs.into_iter().next() {
                let key: Vec<u8> = pair.0.into();
                let value_start = 17;
                if key.len() > value_start {
                    return Ok(Some(encoding::decode_value(&key[value_start..])));
                }
            }
            Ok(None)
        })
        .ok()
        .flatten()
    }

    fn get_row_ids_ordered(
        &self,
        ascending: bool,
        limit: usize,
        offset: usize,
    ) -> Option<Vec<i64>> {
        let prefix = encoding::make_index_prefix(self.meta.table_id, self.meta.index_id);
        let end = encoding::prefix_end_key(&prefix);
        let total = limit + offset;

        let result = self.with_txn(|txn| {
            // Collect KvPairs into Vec<(Vec<u8>, Vec<u8>)> regardless of scan direction
            let kv_pairs: Vec<(Vec<u8>, Vec<u8>)> = if ascending {
                let pairs = self.runtime.block_on(async {
                    txn.scan(prefix..end, total as u32)
                        .await
                        .map_err(from_tikv_error)
                })?;
                pairs.into_iter().map(|p| (p.0.into(), p.1)).collect()
            } else {
                let pairs = self.runtime.block_on(async {
                    txn.scan_reverse(prefix..end, total as u32)
                        .await
                        .map_err(from_tikv_error)
                })?;
                pairs.into_iter().map(|p| (p.0.into(), p.1)).collect()
            };

            let mut row_ids = Vec::new();
            for (i, (key, value)) in kv_pairs.into_iter().enumerate() {
                if i < offset {
                    continue;
                }
                let row_id = if self.meta.is_unique {
                    if value.len() >= 8 {
                        encoding::decode_i64(&value)
                    } else {
                        continue;
                    }
                } else if key.len() >= 8 {
                    encoding::decode_i64(&key[key.len() - 8..])
                } else {
                    continue;
                };
                row_ids.push(row_id);
            }
            Ok(row_ids)
        });

        result.ok()
    }

    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
