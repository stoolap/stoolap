# 12. Type System, Complex Data Types & Column Layouts

## Subsystem Architecture Overview

Stoolap’s type system is designed for high-throughput vectorized operations and columnar storage:

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Supported Core Data Types (src/core/types.rs & src/core/value.rs)          │
├─────────────────┬──────────────────────────┬───────────────────────────────┤
│ SQL Type        │ In-Memory Value          │ Cold Column Physical Array    │
├─────────────────┼──────────────────────────┼───────────────────────────────┤
│ INTEGER / BIGINT│ Value::Integer(i64)      │ TypedColumn::I64(Vec<i64>)    │
│ FLOAT / REAL    │ Value::Float(f64)        │ TypedColumn::F64(Vec<f64>)    │
│ BOOLEAN         │ Value::Boolean(bool)     │ TypedColumn::Bool(BitVec)     │
│ TEXT / VARCHAR  │ Value::Text(SmartString) │ TypedColumn::Dictionary/String│
│ TIMESTAMP       │ Value::Timestamp(i64)    │ TypedColumn::I64(Vec<i64>)    │
│ JSON            │ Value::Json(CompactStr)  │ TypedColumn::String           │
│ VECTOR (Float32)│ Value::Vector(Vec<f32>)  │ TypedColumn::Vector           │
└─────────────────┴──────────────────────────┴───────────────────────────────┘
```

### Key Source References
- [`src/core/types.rs`](file:///home/irshad/stoolap/src/core/types.rs): Defines `DataType` enum.
- [`src/core/value.rs`](file:///home/irshad/stoolap/src/core/value.rs): Defines `Value` runtime enum and type coercion methods.
- [`src/storage/volume/column.rs`](file:///home/irshad/stoolap/src/storage/volume/column.rs): Implements typed columnar builders and on-disk array packing.

---

## Known Limitations Breakdown

1. **No Native BLOB / BINARY Column Type**: Raw arbitrary byte payloads cannot be stored directly as a distinct binary data type. Applications must base64-encode binary data as `TEXT` or store it inside `JSON` strings.
2. **No Native ARRAY Column Type**: First-class array columns (e.g. `INTEGER[]`, `TEXT[]`) are not supported. Applications must use JSON arrays (`Value::Json`) as an alternative.
3. **No Native ENUM Type**: Static enumerated types (`CREATE TYPE mood AS ENUM (...)`) are not supported. Applications must use `TEXT` columns with `CHECK (col IN ('val1', 'val2'))` constraints.
4. **No Stored INTERVAL Column Type**: `INTERVAL` expressions (e.g. `NOW() - INTERVAL '7 days'`) are supported in runtime arithmetic, but columns cannot be declared with type `INTERVAL` for persistent storage.

---

## Architectural Root Causes

### 1. Vectorized Column Alignment & Simplicity
Stoolap’s cold frozen volume layout ([`src/storage/volume/column.rs`](file:///home/irshad/stoolap/src/storage/volume/column.rs)) relies on flat, contiguous memory buffers for fast SIMD processing. Supporting nested data structures like `ARRAY<T>` requires **List Array** representations with separate offset arrays and child data arrays (similar to Apache Arrow), which adds complexity to the column scanner and expression compiler.

### 2. In-Memory Enum Footprint vs Text Dictionary Encoding
Because cold storage string columns automatically apply **Dictionary Encoding** (mapping repetitive string values to 8-bit or 16-bit integer IDs), the on-disk storage efficiency of `TEXT` is often comparable to native `ENUM` types. Consequently, a dedicated `DataType::Enum` was omitted in initial releases.

### 3. Binary vs String Sanitization
`Value::Text` enforces UTF-8 validity guarantees. Without a separate `Value::Blob(Vec<u8>)` variant, storing non-UTF8 arbitrary binary sequences (images, serialized protobufs, audio files) in string containers fails UTF-8 validation checks.

---

## Performance & System Impact

- **Base64 Encoding Overhead**: Storing binary assets in `TEXT` introduces a ~33% storage size amplification and CPU serialization overhead.
- **Array Querying Limitations**: Using JSON arrays instead of native typed arrays prevents the query optimizer from leveraging SIMD-vectorized array containment operations (`array_contains`, `&&`, `@>`).

---

## Proposed Engineering Roadmap

### Phase 1: Native BLOB / BYTES Data Type
- Add `DataType::Blob` and `Value::Blob(Vec<u8>)` (or `CompactBytes`).
- Implement `TypedColumn::Bytes` in [`src/storage/volume/column.rs`](file:///home/irshad/stoolap/src/storage/volume/column.rs) storing contiguous byte buffers with 32-bit offset vectors.
- Add hex/base64 scalar conversion functions (`to_hex()`, `from_hex()`, `to_base64()`, `from_base64()`).

### Phase 2: Native Typed ARRAY Type (Arrow List Layout)
- Introduce `DataType::Array(Box<DataType>)` and `Value::Array(Vec<Value>)`.
- Implement Arrow-compatible offset + values columnar encoding for cold storage.
- Add array manipulation and unnesting functions (`UNNEST()`, `ARRAY_LENGTH()`, `ARRAY_APPEND()`).

### Phase 3: Native ENUM Type
- Support `CREATE TYPE name AS ENUM ('a', 'b', 'c')`.
- Store enum values internally as compact 8-bit integers (`u8`) referencing catalog string tables.

### Phase 4: Persistent INTERVAL Type
- Add `DataType::Interval` and `Value::Interval { months: i32, days: i32, microseconds: i64 }` for persistent temporal duration storage.
