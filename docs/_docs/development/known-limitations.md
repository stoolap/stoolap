---
layout: doc
title: Known Limitations
category: Development
order: 2
---

# Known Limitations

This page consolidates all known limitations across Stoolap. Each limitation is categorized by feature area and links to the relevant documentation for more details.

## JSON

- **No modification functions**: JSON_SET, JSON_INSERT, JSON_REPLACE, and JSON_REMOVE are not yet supported. JSON values can be read and queried but not modified in place.
- **No path query functions**: JSON_CONTAINS and JSON_CONTAINS_PATH are not yet available.
- **No JSON property indexing**: Indexes cannot be created on values within JSON documents.

See [JSON Support]({{ '/docs/data-types/json-support/' | relative_url }}) for supported JSON features.

## Foreign Keys

- **Single-column only**: Composite foreign keys (referencing multiple columns) are not yet supported.
- **Self-referencing insertion order**: Self-referencing foreign keys require careful ordering of inserts.

See [Foreign Keys]({{ '/docs/sql-features/foreign-keys/' | relative_url }}) for full FK documentation.

## Date and Time

- **UTC only**: Timestamps are normalized to UTC internally. There are no explicit functions for time zone conversion.

See [Date and Time Handling]({{ '/docs/data-types/date-and-time/' | relative_url }}) for supported date/time features.

## Temporal Queries (AS OF)

- **No subqueries with AS OF**: AS OF clauses cannot be combined with subqueries.
- **System clock dependency**: Timestamp resolution depends on the system clock precision.
- **VACUUM removes history**: Running VACUUM permanently removes all historical row versions not needed by currently active transactions. After a VACUUM, AS OF queries referencing timestamps before the VACUUM will return no results.

See [Temporal Queries]({{ '/docs/sql-features/temporal-queries/' | relative_url }}) for AS OF usage.

## Views

- **Read-only**: INSERT, UPDATE, and DELETE on views are not supported.
- **Shared namespace**: View names and table names share the same namespace and cannot conflict.
- **Nesting limit**: Maximum nesting depth of 32 levels prevents infinite recursion from circular definitions.

See [Views]({{ '/docs/sql-features/views/' | relative_url }}) for view documentation.

## ALTER TABLE

- **Blocking**: ALTER TABLE operations may temporarily block concurrent writes.
- **No composite PK changes**: Composite primary key modifications are not supported.

See [ALTER TABLE]({{ '/docs/sql-features/alter-table/' | relative_url }}) for full syntax.

## Upsert (ON CONFLICT / ON DUPLICATE KEY)

- **No MySQL VALUES() syntax**: Stoolap uses `EXCLUDED.column` (PostgreSQL-style) instead of MySQL's `VALUES(column)` to reference incoming insert values.
- **No WHERE on conflict action**: PostgreSQL's `ON CONFLICT ... DO UPDATE SET ... WHERE ...` conditional update is not yet supported.

See [Upsert]({{ '/docs/sql-features/on-duplicate-key-update/' | relative_url }}) for full upsert documentation.

## WebAssembly (WASM)

| Feature | Status |
|---------|--------|
| File persistence | Not available (in-memory only, data lost on page reload) |
| Background threads | Not available (no parallel execution, no automatic cleanup) |
| Cleanup | Manual only (use `VACUUM` or `PRAGMA vacuum`) |
| WAL / Snapshots | Not available (no crash recovery needed) |

See [WebAssembly]({{ '/docs/drivers/wasm/' | relative_url }}) for WASM usage.

## Cold Segments (Frozen Volumes)

- **AS OF on cold rows**: Historical point-in-time queries (AS OF TRANSACTION) are not supported on tables with cold segments because cold rows lack version chains. AS OF CURRENT queries work correctly.
- **Compaction memory**: Compaction materializes the full cold dataset in memory before rewriting. For tables with millions of cold rows, this causes a temporary memory spike. Similarly, parallel GROUP BY on 4+ volumes materializes one group map per volume simultaneously before merging.
- **Skip-set cloning**: Each scan builds per-volume skip sets by cloning cumulative row_id sets. For tables with many volumes, this is O(N*V). Compaction keeps volume counts low (default threshold: 4).
- **WAL growth under continuous writes**: WAL truncation requires all hot buffers to be empty. Under continuous writes, new rows may arrive between the seal pass and the truncation check. The checkpoint cycle force-seals small tables below the normal threshold, but truly continuous writes can delay truncation until a quiet period or clean close.
- **Snapshot transactions limit seal throughput**: Active snapshot isolation transactions use cutoff-filtered seal: only rows committed before the earliest snapshot's begin_seq are sealed. Rows committed after remain in the hot buffer. Under long-running snapshots with high write throughput, the hot buffer grows proportionally to the writes since the snapshot started. Compaction similarly filters: only volumes sealed before the snapshot and tombstones committed before the snapshot are physically applied.

- **Multi-column DISTINCT on large tables**: `SELECT DISTINCT col1, col2 FROM t` on tables with cold volumes does not use dictionary extraction and falls through to a full row scan. Single-column DISTINCT uses dictionary metadata.
- **Window functions + LIMIT on large tables**: Window functions materialize all rows before LIMIT is applied. `ROW_NUMBER() OVER (...) LIMIT 10` on a large table processes every row. Workaround: use PARTITION BY to enable the streaming window path.

See [Persistence]({{ '/docs/architecture/persistence/' | relative_url }}) for full details.

## Cold Segments (Accepted Tradeoffs)

These are deliberate design decisions, not bugs:

- **Binary search only for Integer/Timestamp columns**: Float, Text, and Boolean columns in cold segments fall back to linear scan with zone map pruning. Binary search is only available on sorted i64-based columns.
- **ALTER TABLE only modifies hot schema**: Cold volumes retain their original schema. Column additions, drops, and renames are normalized at scan time. DROP COLUMN does not reclaim cold storage space until the next compaction cycle.

## Transactions

- **No primary key updates**: UPDATE on primary key columns is rejected with an error. The engine uses row_id == pk_value as a core invariant. Use DELETE + INSERT to change a row's primary key value.

## General SQL

- **No stored procedures or triggers**: Only built-in functions and SQL statements are supported.
- **No user-defined functions**: Custom functions cannot be registered through SQL.
- **No GRANT/REVOKE**: There is no access control or permission system. Stoolap is an embedded database, so access control is managed at the application level.
- **No full-text search**: Only pattern matching (LIKE, ILIKE, GLOB, REGEXP) is available.
- **No materialized views**: Views are always computed on demand.
- **No event-based notifications**: There is no LISTEN/NOTIFY mechanism.

## Data Types

- **No BLOB/BINARY type**: Binary data is not directly supported as a column type.
- **No ARRAY type**: Array columns are not supported. Use JSON arrays as an alternative.
- **No ENUM type**: Enumerated types are not available. Use TEXT with CHECK constraints as an alternative.
- **No INTERVAL type as column**: INTERVAL is supported in expressions (e.g., `NOW() - INTERVAL '1 day'`) but not as a stored column type.
