# 03. Date, Time & Timezone Architecture

## Subsystem Architecture Overview

Stoolap manages temporal data using microsecond/nanosecond integer timestamps normalized to Coordinated Universal Time (UTC):

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Client Timestamp: '2026-04-20 14:30:00+05:30'                             │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────────┐
       │   Parser & Cast System (src/functions/scalar/datetime.rs)  │
       │   - Parses ISO 8601 string with chrono                     │
       │   - Normalizes and converts to UTC Unix timestamp          │
       └────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────────┐
       │   Storage Value: Value::Timestamp(i64)                     │
       │   - Hot MVCC Row & Cold Column Store (i64 flat array)      │
       └────────────────────────────────────────────────────────────┘
```

### Key Source References
- [`src/core/types.rs`](file:///home/irshad/stoolap/src/core/types.rs): Defines `DataType::Timestamp`.
- [`src/core/value.rs`](file:///home/irshad/stoolap/src/core/value.rs): Defines `Value::Timestamp(i64)` storing epoch microseconds/nanoseconds.
- [`src/functions/scalar/datetime.rs`](file:///home/irshad/stoolap/src/functions/scalar/datetime.rs): Implements functions such as `now()`, `date_add()`, `date_trunc()`, `extract()`, `strftime()`.
- [`src/storage/volume/column.rs`](file:///home/irshad/stoolap/src/storage/volume/column.rs): Columnar storage for timestamps packed as contiguous 64-bit integers.

---

## Known Limitations Breakdown

1. **UTC Normalization Only**: Timestamps are converted and stored strictly in UTC. The original client timezone offset or location name is discarded upon ingestion.
2. **No Timezone Conversion Functions**: SQL functions for converting between arbitrary timezones (e.g. `CONVERT_TZ(ts, from_tz, to_tz)`, `ts AT TIME ZONE 'America/New_York'`) are not implemented.

---

## Architectural Root Causes

### 1. Fixed 64-bit Physical Storage Representation
To maximize columnar analytical scan throughput in frozen volumes ([`src/storage/volume/column.rs`](file:///home/irshad/stoolap/src/storage/volume/column.rs)), `Value::Timestamp` is encoded as a compact 8-byte scalar `i64`.
Storing explicit timezone offsets or Olson timezone identifiers (e.g. `Europe/London`) requires either:
- A compound 12-to-16 byte structure (`i64` timestamp + `i16` offset or string dictionary ID).
- Additional metadata columns, which would alter physical column array packing and SIMD vectorized operations.

### 2. Embedded Binary Size vs IANA Timezone Database
Stoolap is designed as a zero-dependency lightweight embedded engine. Embedding the full IANA Time Zone Database (`tzdb` via `chrono-tz`) increases the compiled binary size and complicates cross-compilation for WebAssembly (`wasm32-unknown-unknown`). Consequently, timezone conversion logic was omitted in early iterations.

---

## Performance & System Impact

- **Display & Regional Analytics**: Applications operating across multiple regional jurisdictions must perform timezone transformations in their application layer or via manual arithmetic (`INTERVAL` offsets).
- **Daylight Saving Time (DST) Anomalies**: Grouping data by local calendar days (e.g. `date_trunc('day', ts)`) without proper timezone and DST awareness can produce bucket alignment errors near clock-change transitions.

---

## Proposed Engineering Roadmap

### Phase 1: Implement Timezone Scalar Functions via Static Offsets
- Introduce `convert_tz(ts, from_offset, to_offset)` supporting fixed numerical and ISO offsets (e.g. `'+05:30'`, `'-08:00'`).
- Support standard timezone conversion syntax: `expr AT TIME ZONE offset`.

### Phase 2: Feature-Gated IANA Timezone Database
- Add an optional Cargo feature `features = ["timezone-db"]` pulling `chrono-tz`.
- Enable string-based named zone lookups (e.g. `'America/Los_Angeles'`, `'UTC'`, `'Asia/Tokyo'`) with full DST transition tables.

### Phase 3: First-Class `TIMESTAMPTZ` Column Type
- Add `DataType::TimestampTz` and `Value::TimestampTz(i64, i16)` (storing timestamp + minute offset).
- Implement specialized columnar arrays for `TimestampTz` in cold frozen volumes with delta-encoded offsets.
