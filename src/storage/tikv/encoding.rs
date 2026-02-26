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

//! Key encoding/decoding for TiKV storage backend.
//!
//! Uses order-preserving (memcomparable) encoding so that TiKV's sorted key
//! iteration produces correctly ordered results for range queries.
//!
//! Also provides custom serialization for Value and Schema types since they
//! do not implement serde traits.

use chrono::{DateTime, Utc};

use crate::common::{CompactArc, SmartString};
use crate::core::types::{DataType, ForeignKeyAction};
use crate::core::{Error, ForeignKeyConstraint, Result, Row, Schema, SchemaColumn, Value};

// Key prefixes
pub const META_SCHEMA_PREFIX: &[u8] = b"m_s_";
pub const META_TABLE_ID_PREFIX: &[u8] = b"m_t_";
pub const META_NEXT_TABLE_ID: &[u8] = b"m_n";
pub const META_NEXT_ROW_ID_PREFIX: &[u8] = b"m_r_";
pub const META_INDEX_PREFIX: &[u8] = b"m_i_";
pub const META_VIEW_PREFIX: &[u8] = b"m_v_";
pub const DATA_PREFIX: u8 = b'd';
pub const INDEX_PREFIX: u8 = b'i';

/// Encode i64 in order-preserving format (flip sign bit + big-endian)
#[inline]
pub fn encode_i64(v: i64) -> [u8; 8] {
    // XOR with sign bit to make negative values sort before positive
    (v ^ (1i64 << 63)).to_be_bytes()
}

/// Decode i64 from order-preserving format
#[inline]
pub fn decode_i64(bytes: &[u8]) -> i64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[..8]);
    i64::from_be_bytes(buf) ^ (1i64 << 63)
}

/// Encode u64 in order-preserving format (big-endian)
#[inline]
pub fn encode_u64(v: u64) -> [u8; 8] {
    v.to_be_bytes()
}

/// Decode u64 from big-endian bytes
#[inline]
pub fn decode_u64(bytes: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[..8]);
    u64::from_be_bytes(buf)
}

/// Encode f64 in order-preserving format
#[inline]
pub fn encode_f64(v: f64) -> [u8; 8] {
    let bits = v.to_bits();
    // If positive (sign bit 0), flip sign bit
    // If negative (sign bit 1), flip all bits
    let encoded = if bits & (1u64 << 63) == 0 {
        bits ^ (1u64 << 63)
    } else {
        !bits
    };
    encoded.to_be_bytes()
}

/// Decode f64 from order-preserving format
#[inline]
pub fn decode_f64(bytes: &[u8]) -> f64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[..8]);
    let encoded = u64::from_be_bytes(buf);
    let bits = if encoded & (1u64 << 63) != 0 {
        encoded ^ (1u64 << 63)
    } else {
        !encoded
    };
    f64::from_bits(bits)
}

/// Encode a Value for use in index keys (order-preserving)
pub fn encode_value(v: &Value) -> Vec<u8> {
    match v {
        Value::Null(_) => vec![0x00], // NULL sorts first
        Value::Boolean(b) => vec![0x01, if *b { 1 } else { 0 }],
        Value::Integer(i) => {
            let mut buf = vec![0x02];
            buf.extend_from_slice(&encode_i64(*i));
            buf
        }
        Value::Float(f) => {
            let mut buf = vec![0x03];
            buf.extend_from_slice(&encode_f64(*f));
            buf
        }
        Value::Text(s) => {
            let mut buf = vec![0x04];
            // Null-terminated encoding with escape: 0x00 → 0x00 0xFF, end → 0x00 0x00
            for &b in s.as_str().as_bytes() {
                if b == 0x00 {
                    buf.push(0x00);
                    buf.push(0xFF);
                } else {
                    buf.push(b);
                }
            }
            buf.push(0x00);
            buf.push(0x00);
            buf
        }
        Value::Timestamp(ts) => {
            let mut buf = vec![0x05];
            buf.extend_from_slice(&encode_i64(ts.timestamp_millis()));
            buf
        }
        Value::Extension(ext) => {
            let mut buf = vec![0x06];
            for &b in ext.as_ref() {
                if b == 0x00 {
                    buf.push(0x00);
                    buf.push(0xFF);
                } else {
                    buf.push(b);
                }
            }
            buf.push(0x00);
            buf.push(0x00);
            buf
        }
    }
}

/// Decode a Value from order-preserving encoded bytes.
/// Returns the decoded value and the number of bytes consumed.
pub fn decode_value(bytes: &[u8]) -> Value {
    decode_value_with_len(bytes).0
}

/// Decode a Value and return (value, bytes_consumed)
pub fn decode_value_with_len(bytes: &[u8]) -> (Value, usize) {
    if bytes.is_empty() {
        return (Value::Null(DataType::Text), 0);
    }
    match bytes[0] {
        0x00 => (Value::Null(DataType::Text), 1),
        0x01 => {
            let b = bytes.get(1).copied().unwrap_or(0) != 0;
            (Value::Boolean(b), 2)
        }
        0x02 => {
            let i = decode_i64(&bytes[1..9]);
            (Value::Integer(i), 9)
        }
        0x03 => {
            let f = decode_f64(&bytes[1..9]);
            (Value::Float(f), 9)
        }
        0x04 => {
            // Null-terminated string with escape: 0x00 0xFF = literal 0x00, 0x00 0x00 = end
            let mut s = Vec::new();
            let mut i = 1;
            while i < bytes.len() {
                if bytes[i] == 0x00 {
                    if i + 1 < bytes.len() && bytes[i + 1] == 0xFF {
                        s.push(0x00);
                        i += 2;
                    } else {
                        // End marker (0x00 0x00)
                        i += 2;
                        break;
                    }
                } else {
                    s.push(bytes[i]);
                    i += 1;
                }
            }
            (
                Value::Text(String::from_utf8_lossy(&s).to_string().into()),
                i,
            )
        }
        0x05 => {
            let millis = decode_i64(&bytes[1..9]);
            use chrono::{DateTime, Utc};
            let ts = DateTime::<Utc>::from_timestamp_millis(millis)
                .unwrap_or_default();
            (Value::Timestamp(ts), 9)
        }
        0x06 => {
            // Extension: tag byte + payload with null-terminated encoding
            let mut s = Vec::new();
            let mut i = 1;
            while i < bytes.len() {
                if bytes[i] == 0x00 {
                    if i + 1 < bytes.len() && bytes[i + 1] == 0xFF {
                        s.push(0x00);
                        i += 2;
                    } else {
                        i += 2;
                        break;
                    }
                } else {
                    s.push(bytes[i]);
                    i += 1;
                }
            }
            (
                Value::Extension(CompactArc::from(s)),
                i,
            )
        }
        _ => (Value::Null(DataType::Text), 1),
    }
}

/// Make a data key: d{table_id}{row_id}
pub fn make_data_key(table_id: u64, row_id: i64) -> Vec<u8> {
    let mut key = Vec::with_capacity(17);
    key.push(DATA_PREFIX);
    key.extend_from_slice(&encode_u64(table_id));
    key.extend_from_slice(&encode_i64(row_id));
    key
}

/// Make a data key prefix for scanning all rows of a table: d{table_id}
pub fn make_data_prefix(table_id: u64) -> Vec<u8> {
    let mut key = Vec::with_capacity(9);
    key.push(DATA_PREFIX);
    key.extend_from_slice(&encode_u64(table_id));
    key
}

/// Extract row_id from a data key
pub fn extract_row_id_from_data_key(key: &[u8]) -> i64 {
    // key = d(1) + table_id(8) + row_id(8) = 17 bytes
    decode_i64(&key[9..17])
}

/// Make an index key for non-unique index: i{table_id}{index_id}{encoded_values}{row_id}
pub fn make_index_key(table_id: u64, index_id: u64, values: &[Value], row_id: i64) -> Vec<u8> {
    let mut key = Vec::with_capacity(32);
    key.push(INDEX_PREFIX);
    key.extend_from_slice(&encode_u64(table_id));
    key.extend_from_slice(&encode_u64(index_id));
    for v in values {
        key.extend_from_slice(&encode_value(v));
    }
    key.extend_from_slice(&encode_i64(row_id));
    key
}

/// Make an index key prefix for scanning: i{table_id}{index_id}
pub fn make_index_prefix(table_id: u64, index_id: u64) -> Vec<u8> {
    let mut key = Vec::with_capacity(17);
    key.push(INDEX_PREFIX);
    key.extend_from_slice(&encode_u64(table_id));
    key.extend_from_slice(&encode_u64(index_id));
    key
}

/// Make a meta key for schema: m_s_{table_name}
pub fn make_schema_key(table_name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(META_SCHEMA_PREFIX.len() + table_name.len());
    key.extend_from_slice(META_SCHEMA_PREFIX);
    key.extend_from_slice(table_name.as_bytes());
    key
}

/// Make a meta key for table ID: m_t_{table_name}
pub fn make_table_id_key(table_name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(META_TABLE_ID_PREFIX.len() + table_name.len());
    key.extend_from_slice(META_TABLE_ID_PREFIX);
    key.extend_from_slice(table_name.as_bytes());
    key
}

/// Make a meta key for next row ID: m_r_{table_name}
pub fn make_next_row_id_key(table_name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(META_NEXT_ROW_ID_PREFIX.len() + table_name.len());
    key.extend_from_slice(META_NEXT_ROW_ID_PREFIX);
    key.extend_from_slice(table_name.as_bytes());
    key
}

/// Make a meta key for view: m_v_{view_name}
pub fn make_view_key(view_name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(META_VIEW_PREFIX.len() + view_name.len());
    key.extend_from_slice(META_VIEW_PREFIX);
    key.extend_from_slice(view_name.as_bytes());
    key
}

/// Make a meta key for index: m_i_{table_name}_{index_name}
pub fn make_index_meta_key(table_name: &str, index_name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(META_INDEX_PREFIX.len() + table_name.len() + 1 + index_name.len());
    key.extend_from_slice(META_INDEX_PREFIX);
    key.extend_from_slice(table_name.as_bytes());
    key.push(b'_');
    key.extend_from_slice(index_name.as_bytes());
    key
}

/// Compute the exclusive end key for a prefix scan (increment last byte)
pub fn prefix_end_key(prefix: &[u8]) -> Vec<u8> {
    let mut end = prefix.to_vec();
    // Find the last byte that can be incremented
    while let Some(last) = end.last_mut() {
        if *last < 0xFF {
            *last += 1;
            return end;
        }
        end.pop();
    }
    // All bytes were 0xFF — return empty (scan to end)
    end
}

// ============================================================================
// Custom Value serialization (Value does not implement serde traits)
// ============================================================================

/// Serialize a Value to a JSON-compatible representation using serde_json::Value
fn value_to_json(v: &Value) -> serde_json::Value {
    match v {
        Value::Null(dt) => {
            serde_json::json!({"t": "null", "dt": datatype_to_str(dt)})
        }
        Value::Integer(i) => serde_json::json!({"t": "int", "v": *i}),
        Value::Float(f) => {
            // JSON doesn't support infinity/NaN, encode as string for those
            if f.is_finite() {
                serde_json::json!({"t": "float", "v": *f})
            } else {
                serde_json::json!({"t": "float", "s": format!("{}", f)})
            }
        }
        Value::Text(s) => serde_json::json!({"t": "text", "v": s.as_str()}),
        Value::Boolean(b) => serde_json::json!({"t": "bool", "v": *b}),
        Value::Timestamp(ts) => {
            serde_json::json!({"t": "ts", "v": ts.to_rfc3339()})
        }
        Value::Extension(ext) => {
            let tag = ext.first().copied().unwrap_or(0);
            if tag == DataType::Json as u8 {
                let s = std::str::from_utf8(&ext[1..]).unwrap_or("");
                serde_json::json!({"t": "json", "v": s})
            } else if tag == DataType::Vector as u8 {
                let v = v.as_vector_f32().unwrap_or_default();
                serde_json::json!({"t": "vec", "v": v})
            } else {
                serde_json::json!({"t": "ext", "tag": tag, "v": format!("{:?}", &ext[1..])})
            }
        }
    }
}

/// Deserialize a Value from a JSON representation
fn json_to_value(j: &serde_json::Value) -> Result<Value> {
    let t = j["t"]
        .as_str()
        .ok_or_else(|| Error::internal("missing type tag in serialized value"))?;
    match t {
        "null" => {
            let dt = j["dt"]
                .as_str()
                .map(str_to_datatype)
                .unwrap_or(DataType::Null);
            Ok(Value::Null(dt))
        }
        "int" => {
            let v = j["v"]
                .as_i64()
                .ok_or_else(|| Error::internal("bad int value"))?;
            Ok(Value::Integer(v))
        }
        "float" => {
            // Check for string-encoded special values (infinity, NaN)
            if let Some(s) = j["s"].as_str() {
                let v: f64 = s
                    .parse()
                    .map_err(|e| Error::internal(format!("parse float: {e}")))?;
                Ok(Value::Float(v))
            } else {
                let v = j["v"]
                    .as_f64()
                    .ok_or_else(|| Error::internal("bad float value"))?;
                Ok(Value::Float(v))
            }
        }
        "text" => {
            let v = j["v"]
                .as_str()
                .ok_or_else(|| Error::internal("bad text value"))?;
            Ok(Value::Text(SmartString::from(v)))
        }
        "bool" => {
            let v = j["v"]
                .as_bool()
                .ok_or_else(|| Error::internal("bad bool value"))?;
            Ok(Value::Boolean(v))
        }
        "ts" => {
            let s = j["v"]
                .as_str()
                .ok_or_else(|| Error::internal("bad timestamp value"))?;
            let ts = DateTime::parse_from_rfc3339(s)
                .map_err(|e| Error::internal(format!("parse timestamp: {e}")))?
                .with_timezone(&Utc);
            Ok(Value::Timestamp(ts))
        }
        "json" => {
            let v = j["v"]
                .as_str()
                .ok_or_else(|| Error::internal("bad json value"))?;
            Ok(Value::json(v))
        }
        "vec" => {
            let arr = j["v"]
                .as_array()
                .ok_or_else(|| Error::internal("bad vector value"))?;
            let mut v = Vec::with_capacity(arr.len());
            for val in arr {
                v.push(val.as_f64().ok_or_else(|| Error::internal("bad vector element"))? as f32);
            }
            Ok(Value::vector(v))
        }
        "ext" => {
            let tag = j["tag"].as_u64().unwrap_or(0) as u8;
            // For now, don't support deserializing generic extensions
            let mut bytes = Vec::with_capacity(1);
            bytes.push(tag);
            Ok(Value::Extension(CompactArc::from(bytes)))
        }
        other => Err(Error::internal(format!("unknown value type tag: {other}"))),
    }
}

fn datatype_to_str(dt: &DataType) -> &'static str {
    match dt {
        DataType::Integer => "int",
        DataType::Float => "float",
        DataType::Boolean => "bool",
        DataType::Text => "text",
        DataType::Timestamp => "ts",
        DataType::Json => "json",
        DataType::Vector => "vec",
        DataType::Null => "null",
    }
}

fn str_to_datatype(s: &str) -> DataType {
    match s {
        "int" => DataType::Integer,
        "float" => DataType::Float,
        "bool" => DataType::Boolean,
        "text" => DataType::Text,
        "ts" => DataType::Timestamp,
        "json" => DataType::Json,
        "vec" => DataType::Vector,
        _ => DataType::Null,
    }
}

/// Serialize a row (Vec<Value>) to bytes
pub fn serialize_row(row: &[Value]) -> Result<Vec<u8>> {
    let json_values: Vec<serde_json::Value> = row.iter().map(value_to_json).collect();
    serde_json::to_vec(&json_values).map_err(|e| Error::internal(format!("serialize row: {e}")))
}

/// Deserialize a row from bytes
pub fn deserialize_row(bytes: &[u8]) -> Result<Vec<Value>> {
    let json_values: Vec<serde_json::Value> =
        serde_json::from_slice(bytes).map_err(|e| Error::internal(format!("deserialize row: {e}")))?;
    json_values.iter().map(json_to_value).collect()
}

/// Convert a Row to a Vec<Value>
pub fn row_to_values(row: &Row) -> Vec<Value> {
    (0..row.len())
        .map(|i| row.get(i).cloned().unwrap_or_else(Value::null_unknown))
        .collect()
}

/// Convert a Vec<Value> to a Row
pub fn values_to_row(values: Vec<Value>) -> Row {
    Row::from_values(values)
}

// ============================================================================
// Custom Schema serialization
// ============================================================================

fn fk_action_to_str(a: &ForeignKeyAction) -> &'static str {
    match a {
        ForeignKeyAction::NoAction => "no_action",
        ForeignKeyAction::Cascade => "cascade",
        ForeignKeyAction::SetNull => "set_null",
        ForeignKeyAction::Restrict => "restrict",
    }
}

fn str_to_fk_action(s: &str) -> ForeignKeyAction {
    match s {
        "cascade" => ForeignKeyAction::Cascade,
        "set_null" => ForeignKeyAction::SetNull,
        "restrict" => ForeignKeyAction::Restrict,
        _ => ForeignKeyAction::NoAction,
    }
}

/// Serialize a Schema to bytes
pub fn serialize_schema(schema: &Schema) -> Result<Vec<u8>> {
    let columns: Vec<serde_json::Value> = schema
        .columns
        .iter()
        .map(|col| {
            let mut obj = serde_json::json!({
                "id": col.id,
                "name": col.name,
                "data_type": datatype_to_str(&col.data_type),
                "nullable": col.nullable,
                "primary_key": col.primary_key,
                "auto_increment": col.auto_increment,
            });
            if let Some(ref de) = col.default_expr {
                obj["default_expr"] = serde_json::Value::String(de.clone());
            }
            if let Some(ref ce) = col.check_expr {
                obj["check_expr"] = serde_json::Value::String(ce.clone());
            }
            obj
        })
        .collect();

    let fks: Vec<serde_json::Value> = schema
        .foreign_keys
        .iter()
        .map(|fk| {
            serde_json::json!({
                "column_index": fk.column_index,
                "column_name": fk.column_name,
                "referenced_table": fk.referenced_table,
                "referenced_column": fk.referenced_column,
                "on_delete": fk_action_to_str(&fk.on_delete),
                "on_update": fk_action_to_str(&fk.on_update),
            })
        })
        .collect();

    let obj = serde_json::json!({
        "table_name": schema.table_name,
        "columns": columns,
        "foreign_keys": fks,
        "created_at": schema.created_at.to_rfc3339(),
        "updated_at": schema.updated_at.to_rfc3339(),
    });

    serde_json::to_vec(&obj).map_err(|e| Error::internal(format!("serialize schema: {e}")))
}

/// Deserialize a Schema from bytes
pub fn deserialize_schema(bytes: &[u8]) -> Result<Schema> {
    let obj: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| Error::internal(format!("deserialize schema: {e}")))?;

    let table_name = obj["table_name"]
        .as_str()
        .ok_or_else(|| Error::internal("missing table_name"))?
        .to_string();

    let columns: Vec<SchemaColumn> = obj["columns"]
        .as_array()
        .ok_or_else(|| Error::internal("missing columns"))?
        .iter()
        .map(|col| {
            let id = col["id"].as_u64().unwrap_or(0) as usize;
            let name = col["name"].as_str().unwrap_or("").to_string();
            let data_type = str_to_datatype(col["data_type"].as_str().unwrap_or("null"));
            let nullable = col["nullable"].as_bool().unwrap_or(false);
            let primary_key = col["primary_key"].as_bool().unwrap_or(false);
            let auto_increment = col["auto_increment"].as_bool().unwrap_or(false);
            let default_expr = col["default_expr"].as_str().map(|s| s.to_string());
            let check_expr = col["check_expr"].as_str().map(|s| s.to_string());

            SchemaColumn::with_constraints(
                id,
                name,
                data_type,
                nullable,
                primary_key,
                auto_increment,
                default_expr,
                check_expr,
            )
        })
        .collect();

    let foreign_keys: Vec<ForeignKeyConstraint> = obj["foreign_keys"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .map(|fk| ForeignKeyConstraint {
                    column_index: fk["column_index"].as_u64().unwrap_or(0) as usize,
                    column_name: fk["column_name"].as_str().unwrap_or("").to_string(),
                    referenced_table: fk["referenced_table"].as_str().unwrap_or("").to_string(),
                    referenced_column: fk["referenced_column"].as_str().unwrap_or("").to_string(),
                    on_delete: str_to_fk_action(fk["on_delete"].as_str().unwrap_or("no_action")),
                    on_update: str_to_fk_action(fk["on_update"].as_str().unwrap_or("no_action")),
                })
                .collect()
        })
        .unwrap_or_default();

    let created_at = obj["created_at"]
        .as_str()
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(Utc::now);

    let updated_at = obj["updated_at"]
        .as_str()
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(Utc::now);

    Ok(Schema::with_timestamps_and_foreign_keys(
        table_name, columns, foreign_keys, created_at, updated_at,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i64_encoding_order() {
        let values = [i64::MIN, -1000, -1, 0, 1, 1000, i64::MAX];
        let encoded: Vec<_> = values.iter().map(|v| encode_i64(*v)).collect();
        for i in 0..encoded.len() - 1 {
            assert!(
                encoded[i] < encoded[i + 1],
                "Order violated: {:?} >= {:?} for {} vs {}",
                encoded[i],
                encoded[i + 1],
                values[i],
                values[i + 1]
            );
        }
        for &v in &values {
            assert_eq!(decode_i64(&encode_i64(v)), v);
        }
    }

    #[test]
    fn test_f64_encoding_order() {
        let values = [f64::NEG_INFINITY, -1.0, -0.0, 0.0, 1.0, f64::INFINITY];
        let encoded: Vec<_> = values.iter().map(|v| encode_f64(*v)).collect();
        for i in 0..encoded.len() - 1 {
            assert!(
                encoded[i] <= encoded[i + 1],
                "Order violated for {} vs {}",
                values[i],
                values[i + 1]
            );
        }
        for &v in &values {
            assert_eq!(decode_f64(&encode_f64(v)), v);
        }
    }

    #[test]
    fn test_value_encoding_type_order() {
        // NULL < Boolean < Integer < Float < Text
        let null_enc = encode_value(&Value::Null(DataType::Null));
        let bool_enc = encode_value(&Value::Boolean(false));
        let int_enc = encode_value(&Value::Integer(0));
        let float_enc = encode_value(&Value::Float(0.0));
        let str_enc = encode_value(&Value::Text(SmartString::from("")));

        assert!(null_enc < bool_enc);
        assert!(bool_enc < int_enc);
        assert!(int_enc < float_enc);
        assert!(float_enc < str_enc);
    }

    #[test]
    fn test_prefix_end_key() {
        assert_eq!(prefix_end_key(b"abc"), b"abd");
        assert_eq!(prefix_end_key(b"ab\xff"), b"ac");
        assert_eq!(prefix_end_key(b"\xff\xff"), b"");
    }

    #[test]
    fn test_row_serialization_roundtrip() {
        let values = vec![
            Value::Integer(42),
            Value::Text(SmartString::from("hello")),
            Value::Null(DataType::Text),
            Value::Boolean(true),
            Value::Float(3.14),
        ];
        let bytes = serialize_row(&values).unwrap();
        let decoded = deserialize_row(&bytes).unwrap();
        assert_eq!(decoded.len(), values.len());
        assert_eq!(decoded[0], Value::Integer(42));
        assert!(matches!(&decoded[1], Value::Text(s) if s.as_str() == "hello"));
        assert!(matches!(&decoded[2], Value::Null(DataType::Text)));
        assert_eq!(decoded[3], Value::Boolean(true));
        assert_eq!(decoded[4], Value::Float(3.14));
    }

    #[test]
    fn test_row_serialization_empty() {
        let values: Vec<Value> = vec![];
        let bytes = serialize_row(&values).unwrap();
        let decoded = deserialize_row(&bytes).unwrap();
        assert_eq!(decoded.len(), 0);
    }

    #[test]
    fn test_row_serialization_timestamp() {
        use chrono::Utc;
        let now = Utc::now();
        let values = vec![Value::Timestamp(now)];
        let bytes = serialize_row(&values).unwrap();
        let decoded = deserialize_row(&bytes).unwrap();
        assert_eq!(decoded.len(), 1);
        if let Value::Timestamp(ts) = &decoded[0] {
            // RFC3339 roundtrip may lose sub-nanosecond precision
            assert_eq!(ts.timestamp_millis(), now.timestamp_millis());
        } else {
            panic!("Expected Timestamp, got {:?}", decoded[0]);
        }
    }

    #[test]
    fn test_row_serialization_json() {
        let values = vec![Value::json(r#"{"key": "value"}"#)];
        let bytes = serialize_row(&values).unwrap();
        let decoded = deserialize_row(&bytes).unwrap();
        assert_eq!(decoded.len(), 1);
        if let Some(j) = decoded[0].as_json() {
            assert_eq!(j, r#"{"key": "value"}"#);
        } else {
            panic!("Expected Json, got {:?}", decoded[0]);
        }
    }

    #[test]
    fn test_row_serialization_all_null_types() {
        let values = vec![
            Value::Null(DataType::Null),
            Value::Null(DataType::Integer),
            Value::Null(DataType::Float),
            Value::Null(DataType::Text),
            Value::Null(DataType::Boolean),
            Value::Null(DataType::Timestamp),
            Value::Null(DataType::Json),
        ];
        let bytes = serialize_row(&values).unwrap();
        let decoded = deserialize_row(&bytes).unwrap();
        assert_eq!(decoded.len(), 7);
        assert!(matches!(&decoded[0], Value::Null(DataType::Null)));
        assert!(matches!(&decoded[1], Value::Null(DataType::Integer)));
        assert!(matches!(&decoded[2], Value::Null(DataType::Float)));
        assert!(matches!(&decoded[3], Value::Null(DataType::Text)));
        assert!(matches!(&decoded[4], Value::Null(DataType::Boolean)));
        assert!(matches!(&decoded[5], Value::Null(DataType::Timestamp)));
        assert!(matches!(&decoded[6], Value::Null(DataType::Json)));
    }

    #[test]
    fn test_row_to_values_and_back() {
        let original = vec![
            Value::Integer(100),
            Value::Text(SmartString::from("test")),
            Value::Boolean(false),
        ];
        let row = values_to_row(original.clone());
        let roundtrip = row_to_values(&row);
        assert_eq!(roundtrip.len(), 3);
        assert_eq!(roundtrip[0], Value::Integer(100));
        assert!(matches!(&roundtrip[1], Value::Text(s) if s.as_str() == "test"));
        assert_eq!(roundtrip[2], Value::Boolean(false));
    }

    #[test]
    fn test_schema_serialization_roundtrip() {
        let columns = vec![
            SchemaColumn::new(0, "id", DataType::Integer, false, true),
            SchemaColumn::with_constraints(
                1,
                "name",
                DataType::Text,
                true,
                false,
                false,
                Some("'unknown'".to_string()),
                None,
            ),
            SchemaColumn::with_constraints(
                2,
                "active",
                DataType::Boolean,
                false,
                false,
                false,
                None,
                Some("active = TRUE OR active = FALSE".to_string()),
            ),
        ];
        let schema = Schema::new("test_table", columns);

        let bytes = serialize_schema(&schema).unwrap();
        let decoded = deserialize_schema(&bytes).unwrap();

        assert_eq!(decoded.table_name, "test_table");
        assert_eq!(decoded.columns.len(), 3);
        assert_eq!(decoded.columns[0].name, "id");
        assert_eq!(decoded.columns[0].data_type, DataType::Integer);
        assert!(decoded.columns[0].primary_key);
        assert!(!decoded.columns[0].nullable);

        assert_eq!(decoded.columns[1].name, "name");
        assert_eq!(decoded.columns[1].data_type, DataType::Text);
        assert!(decoded.columns[1].nullable);
        assert_eq!(
            decoded.columns[1].default_expr.as_deref(),
            Some("'unknown'")
        );

        assert_eq!(decoded.columns[2].name, "active");
        assert_eq!(
            decoded.columns[2].check_expr.as_deref(),
            Some("active = TRUE OR active = FALSE")
        );
    }

    #[test]
    fn test_schema_with_foreign_keys_roundtrip() {
        let columns = vec![
            SchemaColumn::new(0, "id", DataType::Integer, false, true),
            SchemaColumn::new(1, "user_id", DataType::Integer, false, false),
        ];
        let fks = vec![ForeignKeyConstraint {
            column_index: 1,
            column_name: "user_id".to_string(),
            referenced_table: "users".to_string(),
            referenced_column: "id".to_string(),
            on_delete: ForeignKeyAction::Cascade,
            on_update: ForeignKeyAction::Restrict,
        }];
        let schema = Schema::with_foreign_keys("orders", columns, fks);

        let bytes = serialize_schema(&schema).unwrap();
        let decoded = deserialize_schema(&bytes).unwrap();

        assert_eq!(decoded.table_name, "orders");
        assert_eq!(decoded.foreign_keys.len(), 1);
        let fk = &decoded.foreign_keys[0];
        assert_eq!(fk.column_index, 1);
        assert_eq!(fk.column_name, "user_id");
        assert_eq!(fk.referenced_table, "users");
        assert_eq!(fk.referenced_column, "id");
        assert_eq!(fk.on_delete, ForeignKeyAction::Cascade);
        assert_eq!(fk.on_update, ForeignKeyAction::Restrict);
    }

    #[test]
    fn test_data_key_roundtrip() {
        let table_id = 42u64;
        let row_id = -100i64;
        let key = make_data_key(table_id, row_id);
        assert_eq!(key.len(), 17);
        assert_eq!(key[0], DATA_PREFIX);
        let extracted = extract_row_id_from_data_key(&key);
        assert_eq!(extracted, row_id);
    }

    #[test]
    fn test_data_keys_sort_by_row_id() {
        let table_id = 1u64;
        let key_neg = make_data_key(table_id, -5);
        let key_zero = make_data_key(table_id, 0);
        let key_pos = make_data_key(table_id, 5);
        assert!(key_neg < key_zero);
        assert!(key_zero < key_pos);
    }

    #[test]
    fn test_data_keys_sort_by_table_id() {
        let key_t1 = make_data_key(1, 0);
        let key_t2 = make_data_key(2, 0);
        assert!(key_t1 < key_t2);
    }

    #[test]
    fn test_text_value_encoding_order() {
        let a = encode_value(&Value::Text(SmartString::from("aaa")));
        let b = encode_value(&Value::Text(SmartString::from("bbb")));
        let c = encode_value(&Value::Text(SmartString::from("ccc")));
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn test_integer_value_encoding_order() {
        let neg = encode_value(&Value::Integer(-100));
        let zero = encode_value(&Value::Integer(0));
        let pos = encode_value(&Value::Integer(100));
        assert!(neg < zero);
        assert!(zero < pos);
    }

    #[test]
    fn test_large_text_serialization() {
        let large_text = "x".repeat(10000);
        let values = vec![Value::Text(SmartString::from(large_text.as_str()))];
        let bytes = serialize_row(&values).unwrap();
        let decoded = deserialize_row(&bytes).unwrap();
        if let Value::Text(s) = &decoded[0] {
            assert_eq!(s.as_str(), large_text);
        } else {
            panic!("Expected Text");
        }
    }

    #[test]
    fn test_special_characters_in_text() {
        let special = "hello\0world\nfoo\tbar\"baz\\qux";
        let values = vec![Value::Text(SmartString::from(special))];
        let bytes = serialize_row(&values).unwrap();
        let decoded = deserialize_row(&bytes).unwrap();
        if let Value::Text(s) = &decoded[0] {
            assert_eq!(s.as_str(), special);
        } else {
            panic!("Expected Text");
        }
    }

    #[test]
    fn test_extreme_numeric_values() {
        let values = vec![
            Value::Integer(i64::MIN),
            Value::Integer(i64::MAX),
            Value::Float(f64::MIN),
            Value::Float(f64::MAX),
            Value::Float(f64::EPSILON),
            Value::Float(f64::NEG_INFINITY),
            Value::Float(f64::INFINITY),
        ];
        let bytes = serialize_row(&values).unwrap();
        let decoded = deserialize_row(&bytes).unwrap();
        assert_eq!(decoded[0], Value::Integer(i64::MIN));
        assert_eq!(decoded[1], Value::Integer(i64::MAX));
        assert_eq!(decoded[2], Value::Float(f64::MIN));
        assert_eq!(decoded[3], Value::Float(f64::MAX));
        assert_eq!(decoded[4], Value::Float(f64::EPSILON));
        assert_eq!(decoded[5], Value::Float(f64::NEG_INFINITY));
        assert_eq!(decoded[6], Value::Float(f64::INFINITY));
    }
}
