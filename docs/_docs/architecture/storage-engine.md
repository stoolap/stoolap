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
- **Row Data** - The primary data storage, organized by row
- **Version Store** - Tracks row versions for MVCC
- **Indexes** - B-tree, Hash, Bitmap, and multi-column indexes
- **Transaction Manager** - Manages transaction state and visibility

### Data Types

Stoolap supports a variety of data types, each with optimized storage:

- **INTEGER** - 64-bit signed integers
- **FLOAT** - 64-bit floating-point numbers
- **TEXT** - Variable-length UTF-8 strings
- **BOOLEAN** - Boolean values (true/false)
- **TIMESTAMP** - Date and time values
- **JSON** - JSON documents
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
- **Snapshot files** - Point-in-time table snapshots
- **Metadata files** - Schema and index information

## MVCC Implementation

The storage engine uses MVCC to provide transaction isolation:

- **Full Version Chains** - Version history per row linked via pointers
- **Transaction IDs** - Each version is associated with a transaction ID
- **Visibility Rules** - Traverse version chains to find visible versions
- **Lock-Free Reads** - Readers never block writers
- **Automatic Cleanup** - Old versions garbage collected when no longer needed

For more details, see the [MVCC Implementation](mvcc-implementation) and [Transaction Isolation](transaction-isolation) documentation.

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

### Snapshots

1. Periodically, consistent snapshots of tables are created
2. Snapshots contain the latest version of each row
3. Snapshots accelerate recovery compared to replaying the entire WAL

### Recovery Process

After a crash, recovery proceeds as follows:

1. The latest valid snapshot is loaded for each table
2. WAL entries after the snapshot are replayed
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
└── mvcc/               # MVCC implementation
    ├── engine.rs       # MVCC storage engine
    ├── table.rs        # Table with row storage
    ├── transaction.rs  # Transaction management
    ├── version_store.rs # Version tracking
    ├── btree_index.rs  # B-tree index
    ├── hash_index.rs   # Hash index
    ├── bitmap_index.rs # Bitmap index
    ├── multi_column_index.rs # Multi-column index
    └── persistence.rs  # WAL and snapshots
```

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
