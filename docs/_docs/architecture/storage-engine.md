---
layout: doc
title: Storage Engine
category: Architecture
order: 2
---

# Storage Engine

This document provides a detailed overview of Stoolap's storage engine, including its design principles, components, and how data is stored and retrieved.

## Storage Engine Design

Stoolap's storage engine is designed with the following principles:

- **Memory-optimized** - Prioritizes in-memory performance with optional persistence
- **MVCC-based** - Uses multi-version concurrency control for transaction isolation
- **Version-organized** - Tracks different versions of rows for transaction isolation
- **Type-specialized** - Uses different strategies for different data types
- **Index-accelerated** - Multiple index types to optimize different query patterns

## Storage Components

### Table Structure

Tables in Stoolap are composed of:

- **Metadata** - Schema information, column definitions, and indexes
- **Hot Buffer** - In-memory row storage for recent writes (row-major, MVCC-managed)
- **Cold Segments (Frozen Volumes)** - Column-major storage for sealed historical data. Stores timestamps at nanosecond precision. Includes zone maps, bloom filters, dictionary encoding, and CRC32 integrity checks
- **Tombstones** - Manifest-tracked markers for deleted cold rows, with per-transaction deferred application for isolation
- **Version Store** - Tracks row versions for MVCC
- **Indexes** - B-tree, Hash, Bitmap, HNSW, and multi-column indexes (hot data only, cold uses zone maps and bloom filters)
- **Transaction Manager** - Manages transaction state and visibility

In memory mode, all data lives in the hot buffer. In persistence mode, tables automatically seal hot rows into cold segments when they accumulate enough data (100K rows for first seal, 10K incremental). The query engine merges hot and cold sources transparently. See [Persistence]({{ '/docs/architecture/persistence/' | relative_url }}) for details.

### Data Types

Stoolap supports a variety of data types, each with optimized storage:

- **INTEGER** - 64-bit signed integers
- **FLOAT** - 64-bit floating-point numbers
- **TEXT** - Variable-length UTF-8 strings
- **BOOLEAN** - Boolean values (true/false)
- **TIMESTAMP** - Date and time values
- **JSON** - JSON documents
- **VECTOR** - Fixed-dimension floating-point vectors for similarity search
- **NULL** - Null values supported for all types

### Version Management

Stoolap tracks different versions of data for transaction isolation:

- Each change creates a new version rather than overwriting
- Versions are associated with transaction IDs
- Visibility rules determine which versions each transaction can see
- Old versions are garbage collected when no longer needed

## Data Storage Format

### In-Memory Format

In memory, data is stored with these characteristics:

- **Row-based primary storage** - Records are stored as coherent rows
- **Version chains** - Linked versions for MVCC
- **Type-specific indexes** - B-tree, Hash, Bitmap based on column type
- **Efficient structures** - Optimized for different data types

### On-Disk Format

When persistence is enabled, data is stored on disk with:

- **Binary serialization** - Compact binary format for storage
- **WAL files** - Write-ahead log for durability
- **Volume files** - Immutable columnar cold segments with manifests
- **Snapshot files** - Optional backup files (via PRAGMA SNAPSHOT)

## MVCC Implementation

The storage engine uses MVCC to provide transaction isolation:

- **Full Version Chains** - Version history per row linked via pointers
- **Transaction IDs** - Each version is associated with a transaction ID
- **Visibility Rules** - Traverse version chains to find visible versions
- **Lock-Free Reads** - Readers never block writers
- **Automatic Cleanup** - Old versions garbage collected when no longer needed

For more details, see the [MVCC Implementation]({% link _docs/architecture/mvcc-implementation.md %}) and [Transaction Isolation]({% link _docs/architecture/transaction-isolation.md %}) documentation.

## Data Access Paths

### Point Lookups

For point queries (e.g., `WHERE id = 5`):

1. Use primary key or index to locate the row
2. Apply visibility rules based on transaction
3. Return the visible version

### Range Scans

For range queries (e.g., `WHERE price > 100`):

1. Use B-tree index if available for the column
2. Scan matching index entries
3. Apply visibility rules to each row
4. Return visible results

### Full Table Scans

For queries without applicable indexes:

1. Scan all rows in the table
2. Apply WHERE clause filters
3. Apply visibility rules
4. For large tables, parallelize the scan

## Data Modification

### Insert Operations

When data is inserted:

1. Values are validated against column types
2. A new row version is created with the current transaction ID
3. The row is added to the primary row storage
4. Indexes are updated
5. The operation is recorded in the WAL (if enabled)

### Update Operations

When data is updated:

1. The existing row is located via indexes or scan
2. A new version is created with updated values
3. The new version links to the previous version
4. Indexes are updated to reflect the changes
5. The operation is recorded in the WAL (if enabled)

### Delete Operations

When data is deleted:

1. The existing row is located
2. A deletion marker version is created
3. Indexes are updated to reflect the deletion
4. The operation is recorded in the WAL (if enabled)

## Persistence and Recovery

When persistence is enabled:

### Write-Ahead Logging (WAL)

1. All modifications are recorded in the WAL before being applied
2. WAL entries include transaction ID, operation type, and data
3. WAL is flushed to disk for durability
4. This ensures recovery in case of crashes

### Checkpoint Cycle

The background thread periodically seals hot rows into cold volumes:
1. Hot buffer rows are written to immutable columnar `.vol` files
2. Manifests (volume list, tombstones, checkpoint LSN) are persisted
3. Compaction merges the smallest volumes when count exceeds threshold
4. WAL is truncated when all hot data is sealed

### Recovery Process

After a crash, recovery proceeds as follows:

1. Manifests and cold volumes are loaded from `volumes/`
2. WAL entries after the checkpoint LSN are replayed (idempotent for sealed rows)
3. Index definitions are restored and indexes rebuilt
4. Incomplete transactions are rolled back

## Implementation Details

Core storage engine components in the Rust codebase:

```
src/storage/
├── mod.rs              # Storage module entry point
├── traits/             # Storage interfaces
│   ├── engine.rs       # Engine trait
│   ├── table.rs        # Table trait
│   └── transaction.rs  # Transaction trait
├── index/              # Index implementations
│   ├── btree.rs        # B-tree index
│   ├── hash.rs         # Hash index
│   ├── bitmap.rs       # Bitmap index
│   ├── multi_column.rs # Multi-column index
│   └── hnsw.rs         # HNSW vector index
├── volume/             # Immutable volume storage
│   ├── manifest.rs     # Segment manager, tombstones, manifest I/O
│   ├── table.rs        # Volume-backed table operations
│   ├── writer.rs       # Volume builder (hot to columnar)
│   ├── scanner.rs      # Volume scanner with skip sets
│   ├── column.rs       # Columnar data, bloom filters, dict encoding
│   ├── format.rs       # Binary format reader
│   └── io.rs           # Atomic volume file I/O
└── mvcc/               # MVCC implementation
    ├── engine.rs       # MVCC storage engine (checkpoint, seal, compact)
    ├── table.rs        # Table with row storage
    ├── transaction.rs  # Transaction management
    ├── version_store.rs # Version tracking
    ├── persistence.rs  # WAL and persistence manager
    └── snapshot.rs     # Backup snapshot reader/writer
```

## Architecture Comparison

Stoolap's storage engine combines ideas from several database architectures into a design tailored for an embedded, single-node MVCC database.

### Delta-Main Architecture

The core design follows the **delta-main** pattern used by systems like SAP HANA and SingleStore (MemSQL):

- **Delta store (hot buffer):** Row-oriented, MVCC-managed B-tree for writes. Supports full transaction isolation with version chains, visibility checks, and lock-free reads.
- **Main store (cold volumes):** Columnar, immutable `.vol` files optimized for analytical reads. Zone maps, bloom filters, dictionary encoding, and CRC32 checksums.

Most databases pick one format. PostgreSQL is row-oriented everywhere. DuckDB is columnar everywhere. Stoolap uses **row-oriented for writes** and **columnar for reads**, which gives good performance on both OLTP and analytical workloads without the complexity of a distributed system.

### Similarities to Existing Systems

| Concept | Stoolap | Similar To |
|---|---|---|
| Hot buffer to immutable cold files | Hot buffer seal to `.vol` files | LSM-tree memtable to SST (RocksDB) |
| Columnar volumes with zone maps, bloom filters | FrozenVolume format | Apache Parquet, Delta Lake data files |
| Manifest tracking segments and tombstones | manifest.bin per table | Delta Lake transaction log, Iceberg manifest lists |
| Newest-source-wins dedup via skip sets | Hot shadows cold by row_id | Iceberg position delete files, merge-on-read |
| Compaction merging multiple volumes | compact_volumes() | LSM compaction (universal style) |
| WAL with checkpoint-based truncation | WAL truncation after seal | SQLite WAL, PostgreSQL checkpoint |
| Delta store plus columnar main store | Hot MVCC plus cold volumes | SAP HANA, Apache Kudu, SingleStore |

### Key Differences

**Row-level skip-set dedup without coordination.** Most LSM engines use sequence numbers or tombstone markers that require merge iterators with multi-way comparison. Stoolap uses a single rule: "for any row_id, the newest source wins." At scan time, a FxHashSet skip set is built from hot row_ids and passed to cold scanners. No bloom filter probes across levels, no merge cursors, no coordination between layers.

**No levels or compaction tiers.** LSM databases (RocksDB, LevelDB) use multi-level compaction (L0, L1, L2 and so on) with size ratios between levels. Stoolap compacts all volumes for a table into one when the count exceeds a threshold (default 4). This is closer to universal compaction but even simpler. It works because the engine targets embedded, single-node workloads with bounded data sizes.

**Versioned tombstones for snapshot isolation.** Cold tombstones carry a commit_seq that is checked against the reading transaction's begin_seq. Most systems either use merge-on-read where deletes are separate files (Iceberg) or copy-on-write where deletes rewrite data files (Delta Lake). Stoolap inlines tombstone visibility into the scan path, avoiding both file rewrites and separate delete file management.

**Seal-overlap window with atomic counter.** During seal, rows briefly exist in both hot and cold. A `seal_overlap` atomic counter tracks how many rows are in transition, avoiding systematic double-counting in `row_count()` and `fast_row_count()` without expensive dedup. The implementation still allows small transient drift during the brief seal window. Most systems use heavier coordination mechanisms like barriers or epoch-based reclamation.

**Hot-only secondary indexes.** B-tree, Hash, and Bitmap indexes cover only hot buffer rows. Cold data is accessed through volume metadata: zone maps for range pruning, bloom filters for point lookups, dictionary pre-filters for equality checks. This avoids the cost of maintaining indexes across millions of immutable cold rows. The one exception is HNSW vector indexes, which must include cold data because vector similarity search has no equivalent to zone map pruning.

## Performance Characteristics

### Read Performance

- **Point Lookups** - O(1) with hash index, O(log n) with B-tree
- **Range Scans** - O(log n + k) with B-tree index
- **Full Scans** - Parallelized for large tables

### Write Performance

- **Inserts** - O(log n) per index
- **Updates** - O(log n) per index plus version creation
- **Deletes** - O(log n) per index for marker creation

### Concurrency

- **High Read Concurrency** - MVCC enables many concurrent readers
- **Write Concurrency** - Multiple writers with conflict detection
- **No Read Locks** - Readers never block on writes
