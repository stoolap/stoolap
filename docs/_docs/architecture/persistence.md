---
layout: doc
title: Persistence
category: Architecture
order: 7
---

# Persistence

Stoolap provides durable storage through a combination of Write-Ahead Logging (WAL) and immutable cold volumes. This architecture ensures data durability while maintaining high performance, even for tables with millions of rows.

## Overview

Stoolap's persistence layer consists of three components:

1. **Write-Ahead Log (WAL)**: Records all changes before they are applied to memory
2. **Cold Volumes**: Immutable columnar storage for sealed historical data
3. **Backup Snapshots**: Optional point-in-time backup files for disaster recovery

This approach provides:
- **Durability**: Changes are persisted before acknowledgment
- **Fast Recovery**: Volumes load in milliseconds instead of seconds
- **Crash Safety**: WAL ensures no committed transactions are lost
- **One Invariant**: For any row_id, the newest source wins. Hot overrides cold. Newer volumes override older volumes.

## Enabling Persistence

To enable persistence, use a `file://` connection string:

```rust
use stoolap::Database;

// In-memory only (no persistence)
let db = Database::open("memory://")?;

// With disk persistence
let db = Database::open("file:///path/to/database")?;
```

Command line:
```bash
# In-memory
stoolap

# With persistence
stoolap --db "file:///path/to/database"
```

## Write-Ahead Log (WAL)

### How WAL Works

1. When a transaction commits, changes are first written to the WAL file
2. The WAL is synced to disk (based on sync_mode)
3. Changes are then applied to the in-memory structures
4. Transaction is acknowledged to the client

This sequence ensures that committed transactions survive crashes.

### WAL Operations

The WAL records these operations:
- **INSERT**: New row insertions
- **UPDATE**: Row modifications
- **DELETE**: Row deletions
- **TRUNCATE**: Bulk row removal
- **CREATE TABLE**: Table creation (DDL)
- **DROP TABLE**: Table deletion (DDL)
- **CREATE INDEX**: Index creation (DDL)
- **DROP INDEX**: Index deletion (DDL)
- **ALTER TABLE**: Schema modifications (ADD/DROP/RENAME/MODIFY COLUMN, RENAME TABLE)
- **CREATE VIEW / DROP VIEW**: View management (DDL)

### WAL Configuration

Configure WAL behavior using PRAGMA:

```sql
-- Sync mode: 0=None, 1=Normal (default), 2=Full
PRAGMA sync_mode = 1;

-- Buffer size in bytes before automatic WAL flush
PRAGMA wal_flush_trigger = 32768;
```

| Sync Mode | Value | Behavior | Max Data Loss |
|-----------|-------|----------|---------------|
| None | 0 | No fsync. WAL is written to the OS buffer cache only. Data becomes durable at the next checkpoint cycle (every 60s by default) when it is sealed into fsynced volume files. | ~60s (checkpoint interval) |
| Normal | 1 | Fsync WAL at most once per second and on DDL operations. Similar to SQLite WAL mode with `synchronous=NORMAL`. | ~1s |
| Full | 2 | Fsync on every WAL write. Maximum durability at the cost of write throughput. | 0 |

### WAL Files

WAL files are stored in the database directory:
```
/path/to/database/
  wal/
    wal_000001.log
    wal_000002.log
    ...
```

Old WAL files are automatically cleaned up after successful checkpoints when all hot data has been sealed into volumes.

## Cold Volumes (Immutable Segments)

Cold volumes are the primary persistence mechanism. Each table is logically: **Hot Delta + Cold Immutable Segments**. Hot data lives in the MVCC B-tree (WAL-backed, mutable). Cold data lives in frozen volumes (columnar, zone maps, bloom filters, dictionary encoding, CRC32 integrity).

### How Volumes Work

The background checkpoint thread periodically seals hot buffer rows into cold segments:

1. **Seal**: When a table accumulates enough committed rows in hot (100K for first seal, 10K incremental), the rows are written to a columnar `.vol` file in `volumes/<table>/`
2. **Remove from hot**: Sealed rows are removed from the B-tree and hot indexes
3. **Register segment**: The new cold segment is registered in the segment manager
4. **Persist manifest**: The manifest (volume list, tombstones, checkpoint LSN) is written to disk

On clean shutdown, a force checkpoint seals ALL remaining hot rows regardless of threshold.

### Per-Volume Skip Sets (Deduplication)

For any row_id, the newest source wins:

1. Hot buffer rows override all cold volumes
2. Newer cold volumes override older cold volumes
3. Tombstones (manifest-tracked) mark cold rows deleted by DML

During scans, a cumulative skip set is built:
1. Start with hot row_ids + committed tombstones
2. For each volume (newest first): scanner gets the skip set, then the volume's row_ids are added for older volumes
3. Results are merged with a chain scanner

No separate deduplication step is needed.

### Tombstones

When a cold row is deleted or updated, a versioned tombstone is recorded as a `(row_id, commit_seq)` pair:

- **Per-transaction pending**: Tracked on the segment manager, applied at commit with the transaction's commit sequence
- **Committed tombstones**: Stored in the manifest (V4 format), persisted atomically
- **Snapshot isolation**: A snapshot transaction at `begin_seq=N` only sees tombstones with `commit_seq <= N`. Newer tombstones are invisible, so the original cold row remains visible to the older snapshot. Auto-commit transactions see all tombstones.
- **Cleared after compaction**: Rows are physically removed from the merged volume
- **Cleared after seal**: When sealed row_ids overlap with existing tombstones, those tombstones are removed (the new volume is authoritative)

### Compaction

When a table accumulates more than `compact_threshold` cold volumes, compaction runs:

1. Sort volumes by row count, select the smallest cluster (bounded I/O per cycle)
2. Iterate selected volumes newest-first, collecting live rows
3. Tombstones filter pure-DELETE rows (no newer version exists)
4. The `seen` set handles UPDATE dedup (newest version wins by row_id)
5. Write a single clean merged volume
6. Atomically swap old segments for the new one (no visibility gap for queries)
7. Clear tombstones only for row_ids in the merged volumes
8. Skipped entirely when snapshot isolation transactions are active

### What Volumes Optimize

Frozen volumes store data column by column:

- **Startup**: Loading cold data from volumes takes milliseconds instead of seconds
- **Aggregation without WHERE**: `SELECT MAX(time)` answers from pre-computed statistics in microseconds
- **COUNT(\*)**: Returns instantly from stored metadata
- **SUM, MIN, MAX, AVG**: Same instant path when no WHERE clause is present

For queries with WHERE clauses, volumes use zone maps (per-column min/max) to skip data that cannot match, bloom filters for equality predicates on text columns, and binary search on sorted columns to jump directly to matching ranges.

### Volume File Layout

```
/path/to/database/
  volumes/
    table_name/
      vol_00064d50e5946141.vol    # Sealed cold segment
      manifest.bin                # Segment metadata (volumes, tombstones, checkpoint LSN)
```

Volume files include a trailing CRC32 checksum for corruption detection. Bloom filters are serialized alongside the volume data (no rebuild on load).

### Checkpoint Cycle

`PRAGMA CHECKPOINT` (and the periodic background cycle) executes:

1. **Seal**: Move hot buffer rows into new immutable `.vol` files (per-table seal fence ensures DML consistency)
2. **Fence**: Brief exclusive check that all hot buffers are empty, advance WAL checkpoint LSN
3. **Re-record DDL**: Write DDL entries after checkpoint LSN so they survive WAL truncation
4. **Persist manifests**: Write `manifest.bin` per table (volume list, tombstones, checkpoint LSN) atomically via fsync-before-rename
5. **WAL truncate**: Remove WAL entries before checkpoint LSN (only when all manifests are persisted)
6. **Compact** (background thread): Merge cold volumes when count exceeds `compact_threshold` using adaptive 4-phase strategy

### Constraints and DML

Volume-backed tables enforce all constraints:

- **Primary key**: Checked against cold segments using zone maps and binary search
- **Unique indexes**: Checked against cold segments with zone map pruning, bloom filters, and per-volume hash indexes (built lazily on first lookup per column set, never invalidated since volumes are immutable)
- **UPDATE and DELETE**: A tombstone is created for the cold row (deferred to commit). The new version goes to hot. The skip set handles dedup during scans
- **TRUNCATE and DROP TABLE**: Cold segments and tombstones are cleared both in memory and on disk

### Schema Changes

ALTER TABLE operations work on tables with frozen volumes. When the schema changes (ADD COLUMN, DROP COLUMN, etc.), volume data is normalized on read:

- **ADD COLUMN**: New column returns NULL or the column's DEFAULT value for volume rows. When rows are sealed or compacted, the default is materialized
- **DROP COLUMN**: Dropped column is skipped during projection
- **RENAME COLUMN**: Column matching is by name, so the renamed column maps correctly
- **MODIFY COLUMN**: Type coercion is applied on read

No volume rebuild is needed. The next seal or compact cycle produces new segments with the updated schema.

### Memory Model

| Mode | Hot buffer | Cold segments |
|------|-----------|---------------|
| `memory://` | All data in arena | None (no persistence) |
| `file://` (small tables) | All data in arena | None (below seal threshold) |
| `file://` (large tables) | Recent rows only | Sealed historical data in columnar volumes |

For large tables, the hot buffer contains only rows added or modified since the last seal. Historical data stays in cold segments and is merged during scans.

## Backup Snapshots

### How Snapshots Work

`PRAGMA SNAPSHOT` creates a full backup of all tables to `.bin` files. This is separate from the checkpoint cycle and is intended for manual backup and disaster recovery.

Snapshots capture:
1. All table schemas
2. All committed data (both hot buffer and cold volume rows)
3. Point-in-time consistency via commit sequence cutoff
4. Index and view definitions in `ddl.bin` (BTree, Hash, Bitmap, HNSW indexes and all views)

The `ddl.bin` file is critical for restore: snapshot `.bin` files contain only row data and schema, not index or view definitions. Without `ddl.bin`, indexes and views must be recreated manually after restore.

### Snapshot Configuration

```sql
-- Number of backup snapshots to retain per table (default: 3)
PRAGMA keep_snapshots = 3;

-- Manually create a backup snapshot
PRAGMA SNAPSHOT;

-- Restore from latest backup (engine-level, indexes and views preserved)
PRAGMA RESTORE;

-- Restore from a specific backup
PRAGMA RESTORE = '20260315-100000.000';
```

Backup and restore are also available from the CLI for recovery scenarios when the database cannot open:

```bash
# Create backup
stoolap -d "file:///path/to/db" --snapshot

# Restore (filesystem-level, works with corrupted volumes)
stoolap -d "file:///path/to/db" --restore

# Force snapshot recovery when volumes are corrupted
stoolap -d "file:///path/to/db" --reset-volumes --restore
```

### Snapshot Files

Snapshots are stored as binary files, organized per table:
```
/path/to/database/
  snapshots/
    snapshot_meta.bin               # Global snapshot metadata
    ddl.bin                        # Index and view definitions (CRC32 protected)
    table_name/
      snapshot-20240101-120000.000.bin
      snapshot-20240101-130000.000.bin
    other_table/
      snapshot-20240101-120000.000.bin
```

Each table has its own subdirectory. Filenames include the timestamp of creation. Old snapshots beyond `keep_snapshots` are automatically cleaned up.

### Lock File

The database lock file (`db.lock`) uses OS-level file locking (flock) to prevent concurrent access. The lock file is **not** deleted on shutdown because flock protects the inode, not the pathname. Deleting the file while the lock is held would allow a race where another process creates a new inode and acquires its own lock. The stale file on disk is harmless and is re-locked on the next open.

## Recovery Process

When opening a database, Stoolap performs recovery automatically:

1. **Load Manifests + Volumes**: For each table, load manifest.bin and cold volumes from `volumes/<table>/`
2. **Migrate Legacy Snapshots** (v0.3.7 only): If `snapshots/` exists but `volumes/` does not, automatic migration runs (see below)
3. **Replay WAL**: Apply operations logged after the checkpoint LSN
   - INSERT with row_id already in a volume: skip (idempotent)
   - INSERT with tombstoned row_id: apply to hot (post-seal UPDATE)
   - UPDATE/DELETE: apply to hot buffer (creates shadow)
   - Only committed transactions are applied
4. **Rebuild Indexes**: Populate hot indexes from recovered data
5. **Sync Auto-increment**: Ensure counters account for cold segment max row IDs
6. **Post-recovery Seal**: If hot buffer has rows from WAL replay, seal them immediately to cold volumes. This prevents query slowness from a large hot buffer. Manifests are persisted so the sealed data survives another crash.
7. **Ready for Operations**: Database is now available

Recovery is transparent to the application.

### Migration from v0.3.7

Databases created with Stoolap v0.3.7 or earlier used a different persistence format based on snapshot `.bin` files. When opening such a database for the first time with a newer version:

1. The engine detects the `snapshots/` directory and the absence of `volumes/`
2. All legacy snapshot data is loaded into the hot buffer
3. The data is immediately sealed into immutable cold volumes in `volumes/<table>/`
4. The old `snapshots/` directory and `snapshot_meta.bin` are removed
5. On subsequent opens, only the volume-based format is used

Migration is automatic and one-time. You will see `[migration]` messages on stderr during this process. No user intervention is required.

### Crash Safety

| Scenario | Recovery | Data loss? |
|----------|----------|------------|
| Crash during DML | WAL replay | None |
| Crash during checkpoint (before manifest) | Old manifest, WAL replays all | None |
| Crash during manifest write | Atomic rename: old or new | None |
| Crash after manifest, before WAL truncation | Volumes + redundant WAL, idempotent replay | None |
| Crash during compaction (before manifest) | Old volumes intact | None |
| Crash after compaction manifest | New volume, old files orphaned, cleaned up | None |

One atomic state transition: the manifest write. Everything before is preparation (discardable). Everything after is cleanup (repeatable).

### Recovery Example

```rust
// Opening automatically triggers recovery if needed
let db = Database::open("file:///path/to/database")?;

// Database is ready with all committed data restored
let results = db.query("SELECT * FROM users", ())?;
```

## Configuration Options

### Connection String Parameters

```
file:///path/to/database?sync_mode=2&checkpoint_interval=60&compact_threshold=4&keep_snapshots=3
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| sync_mode | WAL sync mode (0, 1, 2) | 1 |
| checkpoint_interval | Seconds between checkpoints | 60 |
| compact_threshold | Volume count before compaction | 4 |
| keep_snapshots | Backup snapshots to retain per table | 3 |
| checkpoint_on_close | Seal all hot rows on clean shutdown | on |

Legacy parameter names are accepted for backward compatibility:
- `snapshot_interval` maps to `checkpoint_interval`

### PRAGMA Commands

```sql
-- Read current settings
PRAGMA sync_mode;
PRAGMA checkpoint_interval;
PRAGMA compact_threshold;
PRAGMA keep_snapshots;
PRAGMA wal_flush_trigger;

-- Modify settings
PRAGMA sync_mode = 2;
PRAGMA checkpoint_interval = 60;
PRAGMA compact_threshold = 4;
PRAGMA keep_snapshots = 5;
PRAGMA wal_flush_trigger = 10000;

-- Create a backup snapshot
PRAGMA snapshot;

-- Run checkpoint cycle manually
PRAGMA checkpoint;
```

## Best Practices

### Durability vs Performance

Choose sync_mode based on your requirements:

| Use Case | Recommended sync_mode | Write Throughput |
|----------|----------------------|-----------------|
| Development/Testing | 0 (None) | Highest. No fsync overhead. Data durable at checkpoint. |
| General Use (recommended) | 1 (Normal) | High. Fsync at most once per second. ~1s durability gap. |
| Financial/Critical Data | 2 (Full) | Lower. Fsync per WAL write. Zero data loss on crash. |

### Checkpoint Frequency

- **Frequent checkpoints** (low interval): Faster recovery, more disk I/O
- **Infrequent checkpoints** (high interval): Slower recovery, less disk I/O

For databases with high write rates, consider shorter intervals.

### Disk Space

Monitor disk usage:
- WAL files grow until the next checkpoint
- Old volumes are compacted when threshold is exceeded
- Old backup snapshots are retained based on `keep_snapshots`

### Backup Strategy

For backups:
1. Create a manual snapshot: `PRAGMA snapshot;`
2. Copy the entire database directory while the database is idle
3. For hot backups, use filesystem snapshots (ZFS, LVM)

## Directory Structure

A persistent database creates this directory structure:

```
/path/to/database/
  db.lock                              # Lock file for single-writer
  wal/
    wal_NNNNNN.log                     # WAL segment files
    checkpoint.meta                    # WAL truncation metadata
  volumes/
    table_name/
      vol_XXXXXXXXXXXXXXXX.vol         # Sealed cold segment
      manifest.bin                     # Segment metadata (volumes, tombstones, checkpoint LSN)
  snapshots/                           # Created by PRAGMA SNAPSHOT (backup)
    snapshot_meta.bin                   # Global snapshot metadata
    table_name/
      snapshot-TIMESTAMP.bin           # Per-table backup snapshot files
```

## Error Handling

### Corrupt WAL

If WAL corruption is detected during recovery:
- Stoolap attempts to recover up to the last valid entry
- Corrupted entries at the end are discarded
- A warning is logged

### Disk Full

If disk becomes full:
- WAL writes will fail
- Transactions will be rolled back
- Free disk space before continuing

### Lock Contention

Only one process can open a database directory:
- A lock file (`db.lock`) prevents concurrent access
- If a previous process crashed, the lock is automatically released on open

## Example: Complete Configuration

```rust
use stoolap::Database;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open with custom persistence settings
    let db = Database::open(
        "file:///var/lib/myapp/data?sync_mode=2&checkpoint_interval=120&keep_snapshots=5"
    )?;

    // Fine-tune at runtime
    db.execute("PRAGMA wal_flush_trigger = 5000", ())?;
    db.execute("PRAGMA compact_threshold = 8", ())?;

    // Your application logic...
    db.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTO_INCREMENT, data JSON)", ())?;

    // Force a backup snapshot before maintenance
    db.execute("PRAGMA snapshot", ())?;

    Ok(())
}
```
