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
- **No recursive CASCADE**: ON UPDATE CASCADE does not recursively cascade to grandchild tables when the child's FK column is also its PK.
- **Self-referencing insertion order**: Self-referencing foreign keys require careful ordering of inserts.

See [Foreign Keys]({{ '/docs/sql-features/foreign-keys/' | relative_url }}) for full FK documentation.

## Date and Time

- **UTC only**: Timestamps are normalized to UTC internally. There are no explicit functions for time zone conversion.
- **Approximate intervals**: INTERVAL calculations for months and years use approximations (30 days per month, 365 days per year) rather than calendar-aware calculations.

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
- **No existing data validation**: MODIFY COLUMN can change nullability (add or remove NOT NULL), but does not validate that existing data satisfies the new constraint.
- **No composite PK changes**: Composite primary key modifications are not supported.

See [ALTER TABLE]({{ '/docs/sql-features/alter-table/' | relative_url }}) for full syntax.

## ON DUPLICATE KEY UPDATE

- **No inserted value reference**: Unlike MySQL's `VALUES()` function, Stoolap does not provide special syntax to reference values from the failed insert.

See [ON DUPLICATE KEY UPDATE]({{ '/docs/sql-features/on-duplicate-key-update/' | relative_url }}) for upsert documentation.

## WebAssembly (WASM)

| Feature | Status |
|---------|--------|
| File persistence | Not available (in-memory only, data lost on page reload) |
| Background threads | Not available (no parallel execution, no automatic cleanup) |
| Cleanup | Manual only (use `VACUUM` or `PRAGMA vacuum`) |
| WAL / Snapshots | Not available (no crash recovery needed) |

See [WebAssembly]({{ '/docs/drivers/wasm/' | relative_url }}) for WASM usage.

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
