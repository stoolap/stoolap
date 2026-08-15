# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **`REPLACE INTO` & `INSERT OR REPLACE` Syntax**: Added support for MySQL/SQLite-style `REPLACE INTO table (cols) VALUES (...)` and `INSERT OR REPLACE INTO table ...` for seamless row overwrite on primary key or unique conflict.
- **String Dialect & Phonetic Functions**: Added `FIELD(target, val1, ...)`, `FIND_IN_SET(str, strlist)`, `ELT(n, str1, ...)`, `SOUNDEX(str)`, and `QUOTE(str)` for MySQL / PostgreSQL string compatibility.
- **Vector Arithmetic & Slice Functions**: Added `VEC_ADD(v1, v2)`, `VEC_SUB(v1, v2)`, `VEC_MUL(v1, v2)` (Hadamard product), `VEC_SLICE(vec, start, len)`, and `VEC_CONCAT(v1, v2)` for advanced in-database vector algebra and manipulation.
- **Extended Spatial GIS Properties**: Added `ST_LENGTH(geom)`, `ST_PERIMETER(poly)`, `ST_NUMPOINTS(geom)`, `ST_SRID(geom)`, and `ST_SETSRID(geom, srid)`.
- **High-Precision Clock & Month-End Functions**: Added `TIMEOFDAY()`, `CLOCK_TIMESTAMP()`, `STATEMENT_TIMESTAMP()`, and `LAST_DAY(date)`.
- **IPv6 Hex Codec & JSON Validator**: Added `INET6_ATON(ipv6_str)`, `INET6_NTOA(hex_str)`, and `IS_VALID_JSON(str)`.
- **DDL Schema Evolution (`IF NOT EXISTS` / `IF EXISTS`)**: Added idempotent schema migration support: `ALTER TABLE t ADD COLUMN IF NOT EXISTS c <type>`, `ALTER TABLE t DROP COLUMN IF EXISTS c`, and `ALTER TABLE t RENAME COLUMN IF EXISTS c TO c2`.
- **SQLite `INSERT OR IGNORE` Syntax**: Added support for `INSERT OR IGNORE INTO table ...` for conflict-safe upsert without explicit `ON CONFLICT` clauses.
- **IP Address & Network Functions**: Added `INET_ATON`, `INET_NTOA`, `IS_IPV4`, and `IS_IPV6` for network address conversions, validation, and binary arithmetic.
- **Date/Time Constructors & Difference Functions**: Added `MAKE_DATE(year, month, day)`, `MAKE_TIME(hour, minute, second)`, `MAKE_TIMESTAMP(year, month, day, hour, minute, second)`, and PostgreSQL-compatible `AGE(ts1, [ts2])`.
- **Extended Vector AI Metric Functions**: Added `COSINE_SIMILARITY`, `MANHATTAN_DISTANCE` (L1 distance), `CHEBYSHEV_DISTANCE` ($L_\infty$ distance), and `HAMMING_DISTANCE` (dimension mismatch count).
- **Extended Geospatial (GIS) Functions**: Added `ST_INTERSECTS` (geometry intersection testing) and `ST_ENVELOPE` (bounding box polygon computation).
- **OGC Geospatial (GIS) Functions**: Implemented standard spatial SQL functions: `ST_POINT`, `ST_MAKEPOINT`, `ST_X`, `ST_Y`, `ST_DISTANCE` (planar 2D), `ST_DISTANCE_SPHERE` (Haversine geodesic in meters), `ST_DWITHIN`, `ST_ASTEXT`, `ST_GEOMFROMTEXT`, `ST_CONTAINS` (ray-casting point-in-polygon), `ST_AREA` (Shoelace formula), and `ST_CENTROID`.
- **Regular Expression & Text Functions**: Added `REGEXP_LIKE`, `REGEXP_REPLACE`, `REGEXP_SUBSTR`, and MySQL-compatible `SUBSTRING_INDEX` delimiter substring extraction.
- **Hexadecimal Codec & UUID Generator**: Added `HEX`, `UNHEX`, and RFC 4122 v4 `GEN_RANDOM_UUID()` and `UUID()`.
- **Vector Math & Normalization Functions**: Added `VEC_NORM`, `VECTOR_NORM`, `VEC_NORMALIZE` (unit vector), `COSINE_DISTANCE`, `L2_DISTANCE`, and `INNER_PRODUCT`.
- **JSON Mutation & Query Functions**: Added `JSON_SET`, `JSON_INSERT`, `JSON_REPLACE`, `JSON_REMOVE`, `JSON_CONTAINS`, `JSON_CONTAINS_PATH`, `JSON_QUOTE`, `JSON_UNQUOTE`, `ARRAY_LENGTH`, and `ARRAY_CONTAINS` scalar functions for rich JSON document manipulation and indexing.
- **Timezone Conversion Function (`CONVERT_TZ`)**: Added `CONVERT_TZ(dt, from_tz, to_tz)` with support for UTC offsets, `UTC`, `GMT`, and ISO string timestamps.
- **Conditional Upsert (`ON CONFLICT ... WHERE ...`)**: Added support for `WHERE <condition>` predicates in PostgreSQL `ON CONFLICT ... DO UPDATE SET ... WHERE ...` and MySQL `ON DUPLICATE KEY UPDATE ... WHERE ...`.
- **MySQL `VALUES(col)` Upsert Syntax**: Added support for MySQL `VALUES(col)` syntax in `ON DUPLICATE KEY UPDATE` expressions as an alias to `EXCLUDED.col`.
- **Architectural Limitation Analysis Series**: Added comprehensive technical documentation and implementation roadmaps in `docs/limitations/` covering all 12 core database subsystems.
- **COPY FROM Statement**: Support for high-performance bulk loading from CSV and JSON files (`COPY table_name FROM '/path/to/file.csv' WITH (FORMAT 'csv', HEADER true)`).
- **Prepared Statements Named Parameter Support**: Full support for named parameters (`:name`, `@name`, `$name`) across the native Rust API and C/FFI interface.
- **HNSW Index Parameters in SHOW**: Expose vector search parameters (`m`, `ef_construction`, `metric`) in `SHOW INDEXES` output.
- **PostgreSQL-style DISTINCT ON**: Extended DISTINCT syntax allowing deduplication based on specific expression sets (`SELECT DISTINCT ON (department) * FROM employees ORDER BY department, salary DESC`).
- **Comprehensive Driver Support**: Added documentation, native bindings, and connection abstractions for Python, Go (purego), Node.js, Java, C#, PHP, Ruby, and Swift.

### Fixed
- **Parser Robustness**:
  - Allow non-reserved SQL keywords as valid table names in `CREATE TABLE` and `SELECT` queries.
  - Fix double-dash (`--`) line comment parsing in multi-line SQL strings.
- **Query & Expression Engine**:
  - Resolve qualified column name ambiguity in `DISTINCT ON` and `ORDER BY` when joining multiple tables with identical column names.
  - Fix alias shadowing and pattern cache collisions in `LIKE` / `GLOB` / `REGEXP` evaluation.
  - Ensure correct error propagation on invalid regular expressions.
- **Clippy**: Addressed Rust 1.95 clippy lints (`collapsible_match`, `sort_by_key`, etc.).

---

## [0.4.0] - 2026-04-20

### Added
- **Immutable Volume Storage Engine (Hot/Cold Tiering)**:
  - Hybrid storage architecture splitting table data into a Hot MVCC Buffer (in-memory lock-free version store with Write-Ahead Logging) and Cold Frozen Volumes (immutable, column-major on-disk files).
  - Column-major storage layout with typed column arrays (`i64`, `f64`, `bool`, `string`, `timestamp`, `bytes`), null bitmasks, and CRC32 integrity validation.
  - Zone Maps (min/max metadata per column block) and per-volume Bloom Filters for query pruning.
  - Dictionary Encoding for low-cardinality string columns to reduce on-disk storage footprint and accelerate scalar filtering.
  - Automatic and manual background compaction cycle merging cold volumes, applying versioned tombstones, and rewriting compacted frozen segments.
  - `SegmentManager` and atomic manifest swap (`manifest.json`) guaranteeing crash-consistent atomic volume switches.
  - Transparent `MergingScanner` combining hot active rows and cold column blocks under a unified iterator interface.
- **PostgreSQL Upsert Syntax**: Support for `ON CONFLICT (col) DO UPDATE SET col = EXCLUDED.col` alongside MySQL `ON DUPLICATE KEY UPDATE`.
- **Constant Folding Optimizer**: Rule-based compile-time evaluation of constant expressions before query plan execution.

### Changed
- **Persistence Layer Replacement**: Replaced full-database snapshot persistence mechanism with modular immutable frozen volumes and append-only WAL.
- **Build Configuration**: Enabled `panic = "abort"` in release profile to optimize binary size and execution speed.

### Fixed
- MVCC snapshot isolation visibility race conditions during concurrent WAL truncation checkpoints.

---

## [0.3.7] - 2026-03-15

### Added
- Collation support for `COLLATE NOCASE` in string comparisons, sorting, and indexing.
- Enhanced table-level statistics collection for query plan cost estimations.

### Fixed
- Parameter resolution in subquery expressions and correlated subquery scoping.

---

## [0.3.5] - 2026-02-28

### Added
- Table-valued functions (`generate_series`, `json_each`).
- Extended window function capabilities including `NTILE`, `PERCENT_RANK`, and `CUME_DIST`.

---

## [0.3.4] - 2026-02-10

### Added
- **FFI Bulk Fetch API**: `stoolap_rows_fetch_all` high-throughput binary row fetching interface for foreign function interfaces.
- **Comprehensive FFI Test Suite**: Added 14 FFI test suites verifying aggregates, joins, subqueries, and prepared statements.

### Fixed
- Fixed potential undefined behavior in FFI buffer deallocation using `into_boxed_slice()`.
- Parser rejection of bare expressions in `prepare()` (e.g. catching typos like `SELECTX`).
- Parser handling of consecutive semicolons (`SELECT 1;;`).

---

## [0.3.2] - 2026-01-18

### Added
- `execute_named()` and `query_named()` methods added to the Transaction API.

### Fixed
- Fixed `UPDATE SET` expressions failing to resolve positional and named parameters inside transactions.
- Replaced deep parameter cloning with `Arc` reference sharing in `UPDATE` execution closures.
- Eliminated code duplication across query execution pipelines via `execute_sql_with_ctx`.

---

## [0.3.1] - 2026-01-05

### Added
- **Cached Plan API**: `Database::cached_plan()`, `execute_plan()`, and `query_plan()` enabling zero-overhead execution of pre-parsed, pre-optimized SQL plans.
- `Executor::get_or_create_plan()` and `execute_with_cached_plan()` execution backend.
- Re-exported `CachedPlanRef` and `ParamVec` in root `stoolap` crate exports.
- Node.js driver documentation and integration guide.

---

## [0.3.0] - 2025-12-15

### Added
- **SQL Schema Evolution**: `ALTER TABLE` statement support for adding, renaming, and modifying columns.
- **DML Returning**: `RETURNING` clause support for `INSERT`, `UPDATE`, and `DELETE`.
- **Set Operations**: Full support for `UNION`, `UNION ALL`, `INTERSECT`, and `EXCEPT`.
- **Views**: `CREATE VIEW` and `DROP VIEW` with query expansion up to 32 nesting levels.
- **Transaction Isolation Levels**: Explicit transaction control with `BEGIN`, `COMMIT`, `ROLLBACK`, supporting Read Committed and Snapshot Isolation.
- **Auto-Increment**: Primary key column `AUTOINCREMENT` sequence generation.
- **Inspection Commands**: `DESCRIBE <table>`, `SHOW TABLES`, and `SHOW INDEXES`.
- **JSON Operators**: Fast path extraction operators `->` (extract JSON object) and `->>` (extract JSON string/scalar).
- **Relational Joins**: `NATURAL JOIN` and `USING (col)` clause support.

---

## [0.2.4] - 2025-11-20

### Fixed
- Corrected import path for `ConcurrentI64Map` across internal index and storage modules.

---

## [0.2.1] - 2025-11-05

### Added
- **Drop Lifecycle Hooks**: Clean cache evictions on `Database::drop` to prevent memory leaks across thread-local pools.
- **Weak References in Statement**: `Weak` database references inside `Statement` to break circular reference cycles.
- **Version Store Arena Slot Reuse**: Slot reuse in version store arena to prevent unbounded memory allocation during long-lived workloads.

### Performance
- **AST Memory Optimization**:
  - Migrated AST string fields to `CompactString` (inline allocation for strings $\le 24$ bytes).
  - Boxed large enum variants (`TableSource`, `SubquerySource`, `ValuesSource`, `CteReference`, `AlterTable`, `Set`) to drastically reduce AST node enum memory footprints.
  - Pre-computed lowercase identifiers (`value_lower`) to eliminate repeated runtime `to_lowercase()` allocations.

---

## [0.2.0] - 2025-10-15

### Added
- Standardized database error types and structured error codes.
- Constant-based versioning and test suite refactoring.

---

## [0.1.0] - 2025-09-01

### Added
- Initial release of Stoolap: In-memory and file-backed embedded SQL database engine in Rust.
- Multi-Version Concurrency Control (MVCC) with lock-free reads.
- Time-travel temporal queries (`AS OF TRANSACTION`, `AS OF TIMESTAMP`).
- B-Tree and Hash indexing.
- Vector search with HNSW index integration.
- Full ACID compliance via Write-Ahead Log (WAL).
