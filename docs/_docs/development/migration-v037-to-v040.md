---
layout: doc
title: Migration Guide v0.3.7 to v0.4.0
category: Development
order: 1
---

# Migration Guide: v0.3.7 to v0.4.0

This guide covers everything you need to know when upgrading from Stoolap v0.3.7 (snapshot-based storage) to v0.4.0 (volume-based storage). The migration is automatic and requires no manual steps, but understanding the changes will help you operate your database effectively.

## Architecture Change Summary

v0.4.0 replaces the snapshot-based persistence engine with an immutable volume-based design inspired by Apache Iceberg and Delta Lake. This is the most significant architectural change in the project's history.

| Aspect | v0.3.7 (Snapshots) | v0.4.0 (Volumes) |
|--------|-------------------|-------------------|
| Storage model | All rows in hot buffer, periodic snapshot to `.bin` | Hot/cold split: mutable hot buffer + immutable cold `.vol` files |
| Persistence format | Monolithic row-serialized `.bin` per table | Columnar `.vol` files with zone maps, bloom filters, dictionary encoding, LZ4 compression |
| Persistence location | `snapshots/<table>/` | `volumes/<table>/` |
| Metadata | `snapshot_meta.bin` | `manifest.bin` per table (versioned, V4 format) |
| Startup speed | Proportional to data size (deserialize all rows into memory) | Near-instant (cold data stays on disk, only WAL replay for recent writes) |
| Memory usage | All data in memory at all times | Only hot buffer in memory, cold data read on demand |
| Query on persisted data | Must be in memory first | Zone map pruning, bloom filters, row-group skipping, column projection directly from disk |
| Snapshot isolation | Hot buffer only | Full support including cold rows via versioned tombstones |
| Concurrent writes to persisted rows | N/A (all rows in memory) | Per-table seal fence + row-level claim prevents lost updates |
| Compaction | Not applicable | Adaptive 4-phase merge (convergence, opportunistic, incremental dedup, epoch staleness) |
| DELETE performance | Row removed from memory, snapshot rewrites all | Tombstone (no rewrite until compaction) |
| Compression | WAL-only (optional) | LZ4 for both WAL and cold volumes (independent `wal_compression` and `volume_compression` flags) |

## What Happens Automatically

### On First Open

When you open a v0.3.7 database with v0.4.0 for the first time, the engine performs a one-time migration:

1. Detects `snapshots/` directory exists and `volumes/` does not
2. Loads all snapshot `.bin` files into the hot buffer
3. Replays WAL entries written after the last snapshot
4. Seals all hot data into immutable `.vol` files under `volumes/<table>/`
5. Creates `manifest.bin` for each table
6. Removes the `snapshots/` directory entirely

You will see these messages on stderr:

```
[migration] Converting legacy snapshots to volumes...
[migration] Legacy snapshot migration complete.
```

### Data Safety

- **Zero data loss**: All rows from snapshots and WAL are preserved
- **Atomic**: If migration fails partway, the original `snapshots/` directory is untouched
- **Idempotent**: If interrupted, the next open retries from scratch
- **Indexes**: All indexes (BTree, Hash, Bitmap, HNSW) are rebuilt during recovery
- **Schema**: All column types, constraints, NOT NULL, PRIMARY KEY, UNIQUE survive
- **All data types**: INTEGER, FLOAT, TEXT, BOOLEAN, TIMESTAMP, JSON, VECTOR all migrate correctly

### Migration Time

Migration time depends on data size:

| Data size | Approximate time |
|-----------|-----------------|
| < 1 MB | < 1 second |
| 1-100 MB | 1-5 seconds |
| 100 MB - 1 GB | 5-30 seconds |
| > 1 GB | Plan for minutes |

The migration is I/O bound. SSD storage significantly reduces migration time.

## Configuration Changes

### Renamed Parameters

| v0.3.7 Parameter | v0.4.0 Parameter | Notes |
|------------------|------------------|-------|
| `snapshot_interval` | `checkpoint_interval` | Backward compatible: old name still accepted in DSN |
| `snapshot_compression` | `compression` | Backward compatible: old name still accepted. Now sets both `wal_compression` and `volume_compression` |
| `keep_snapshots` | `keep_snapshots` | Unchanged. Now controls backup snapshot retention |

### New Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sync_mode` | `normal` | `none` (no fsync, data durable at checkpoint), `normal` (fsync every 1 second), `full` (fsync every write) |
| `checkpoint_interval` | 60 | Seconds between checkpoint cycles (seal + compact + WAL truncate) |
| `compact_threshold` | 4 | Number of cold volumes before compaction merges them |
| `checkpoint_on_close` | on | Seal all hot rows on clean shutdown for fast startup |
| `wal_compression` | on | LZ4 compression for WAL entries |
| `volume_compression` | on | LZ4 compression for cold volume files |
| `compression` | on | Shorthand: sets both `wal_compression` and `volume_compression` |

### Removed Parameters

| v0.3.7 Parameter | Reason |
|------------------|--------|
| `snapshot_compression` | Replaced by `compression` (sets both WAL and volume). Use `wal_compression` or `volume_compression` for independent control |

### DSN Example

v0.3.7:
```
file:///data/mydb?sync=normal&snapshot_interval=300&keep_snapshots=5
```

v0.4.0:
```
file:///data/mydb?sync_mode=normal&checkpoint_interval=60&compact_threshold=4
```

The old DSN still works (backward compatible), but the new parameter names are recommended.

### Sync Mode Behavior

The `sync_mode` parameter controls durability guarantees:

| Mode | Behavior | Use case |
|------|----------|----------|
| `none` | No fsync calls. Data is durable only after checkpoint writes volumes to disk. | Maximum throughput, acceptable data loss window |
| `normal` | Fsync every 1 second (batched). DDL operations fsync immediately. | Balanced (comparable to SQLite WAL + synchronous=NORMAL) |
| `full` | Fsync on every write operation. | Maximum durability, lower throughput |

### CLI Flag Changes

| v0.3.7 Flag | v0.4.0 Flag |
|-------------|-------------|
| `--snapshot-interval` | `--checkpoint-interval` |
| `--keep-snapshots` | `--keep-snapshots` (unchanged) |
| N/A | `--compact-threshold` (new) |
| N/A | `--no-checkpoint-on-close` (new) |
| N/A | `--snapshot` (new, create backup and exit) |
| N/A | `--restore [TIMESTAMP]` (new, filesystem-level restore) |
| N/A | `--reset-volumes` (new, delete volumes/ for recovery) |

### PRAGMA Changes

| v0.3.7 PRAGMA | v0.4.0 PRAGMA | Notes |
|---------------|---------------|-------|
| `PRAGMA snapshot_interval` | `PRAGMA checkpoint_interval` | Old name still accepted |
| `PRAGMA SNAPSHOT` | `PRAGMA SNAPSHOT` | Now creates backup `.bin` files (not primary storage) |
| N/A | `PRAGMA CHECKPOINT` | Runs checkpoint cycle: seal + compact + WAL truncate |
| N/A | `PRAGMA RESTORE` | Restore from a backup snapshot (latest or by timestamp) |
| N/A | `PRAGMA compact_threshold` | Set/read compaction threshold |
| N/A | `PRAGMA checkpoint_on_close` | Set/read close behavior |

**Important**: `PRAGMA SNAPSHOT` changed meaning. In v0.3.7 it was the primary persistence mechanism. In v0.4.0 it creates optional backup files for disaster recovery. It saves table data as `.bin` files and index/view definitions as `ddl.bin` (with CRC32 integrity check). Primary persistence is handled automatically by the checkpoint cycle. Use `PRAGMA RESTORE` to recover from a backup snapshot if needed.

```sql
-- Create a backup snapshot
PRAGMA SNAPSHOT;

-- Restore from the latest backup
PRAGMA RESTORE;

-- Restore from a specific backup (by timestamp)
PRAGMA RESTORE = '20260315-100000.000';
```

## On-Disk Layout Changes

### v0.3.7 Layout

```
mydb/
  db.lock
  wal/
    wal-0001.bin
    checkpoint.meta
  snapshots/
    snapshot_meta.bin
    users/
      snapshot-20240101-120000.000.bin
    orders/
      snapshot-20240101-120000.000.bin
```

### v0.4.0 Layout (After Migration)

```
mydb/
  db.lock
  wal/
    wal-0001.bin
    checkpoint.meta
  volumes/
    users/
      manifest.bin
      vol-00001.vol          (columnar, LZ4 compressed)
    orders/
      manifest.bin
      vol-00001.vol
  snapshots/                 (only if PRAGMA SNAPSHOT used, optional)
    snapshot_meta.bin
    ddl.bin                  (index + view definitions, CRC32 protected)
    users/
      snapshot-20240315-100000.000.bin
```

Key differences:
- `volumes/` replaces `snapshots/` as primary storage
- Each table has a `manifest.bin` tracking its volumes and tombstones
- `.vol` files are columnar (not row-serialized `.bin`), with per-column zone maps, bloom filters, and dictionary encoding
- `.vol` files use LZ4 compression by default (STVZ magic header when compressed, STVL when uncompressed)
- Row-group zone maps (64K-row groups) enable sub-volume pruning
- `snapshots/` only appears if you explicitly run `PRAGMA SNAPSHOT` for backups

## Behavioral Changes

### Checkpoint Cycle (New)

v0.3.7 had a snapshot cycle that periodically serialized the entire hot buffer into `.bin` files. v0.4.0 replaces this with a checkpoint cycle:

1. **Seal**: Move hot buffer rows into a new immutable `.vol` file (per-table seal fence ensures DML consistency)
2. **Persist manifests**: Write `manifest.bin` atomically (fsync tmp, rename, fsync directory)
3. **WAL Truncate**: Remove WAL entries before the checkpoint LSN
4. **Compact** (background thread): If volume count exceeds `compact_threshold`, merge volumes using adaptive 4-phase strategy

The checkpoint cycle runs automatically every `checkpoint_interval` seconds (default: 60) and on clean shutdown.

### Seal Thresholds

Not every checkpoint creates a new volume. Rows are only sealed when:
- First seal: 100,000+ rows in hot buffer
- Incremental seals: 10,000+ new rows since last seal
- On close: ALL remaining hot rows are sealed (regardless of count)

This prevents creating many tiny volume files.

### Seal Fence (DML Safety)

v0.4.0 uses a per-table `RwLock` (seal fence) to coordinate between DML operations and the seal process:
- **DML** (INSERT, UPDATE, DELETE) acquires a shared read lock
- **Seal** acquires an exclusive write lock

This ensures no DML can see a partially-sealed state. Additionally, at commit time, a generation counter detects whether a seal happened during the transaction's lifetime. If so, pending rows are revalidated against cold segments to catch constraint violations that the original INSERT-time check may have missed.

### Adaptive Compaction

v0.4.0 uses a 4-phase compaction strategy instead of fixed-count merging:

1. **Convergence**: Merge enough volumes to reach `compact_threshold` in one cycle
2. **Opportunistic**: Keep including volumes as long as each is no larger than the running average
3. **Incremental dedup**: If selected rows exceed 25% of the largest volume, include it too (deduplicates overlapping rows)
4. **Epoch staleness**: Force-include all volumes if any hasn't been rewritten for `compact_threshold * 2` compaction epochs

This eliminates the problem of large base volumes accumulating stale tombstones or outdated format versions.

### Snapshot Isolation on Cold Data

v0.3.7 kept all data in the hot buffer, so snapshot isolation only applied to in-memory rows. v0.4.0 extends snapshot isolation to cold (volume-backed) rows using versioned tombstones:

```sql
BEGIN ISOLATION LEVEL SNAPSHOT;
-- This transaction sees a frozen point-in-time view,
-- even for rows stored in cold volumes.
-- Concurrent DELETEs and UPDATEs are invisible to this snapshot.
COMMIT;
```

Each tombstone stores a `commit_seq`. A snapshot transaction at `begin_seq=N` only sees tombstones with `commit_seq <= N`. This means concurrent modifications to cold rows are invisible to running snapshot transactions.

### Cold-Row Write Conflict Detection

v0.3.7 had no cold storage, so all writes targeted the hot buffer which already had MVCC conflict detection. v0.4.0 introduces cold rows that live in volumes. When two transactions concurrently modify the same cold row, `try_claim_row()` uses the hot buffer's `uncommitted_writes` map to prevent lost updates:

```
Txn A: UPDATE WHERE id=5 (cold row) -> claims row 5
Txn B: UPDATE WHERE id=5 (cold row) -> conflict error (row already claimed)
```

Only one transaction can modify a given cold row at a time. The loser receives a write conflict error and must retry.

### Query Performance During Checkpoint

v0.3.7 snapshot creation serialized the entire hot buffer under a lock, which could block reads. v0.4.0's checkpoint cycle (seal + compact) is designed to minimize query impact:

- `COUNT(*)` uses O(1) formula during seal overlap (no scanning needed)
- Aggregation pushdowns (`SUM`, `MIN`, `MAX`) use pre-computed cold volume statistics
- Cold data scanning is lazy via `MergingScanner` (streams rows from volumes without loading all into memory)
- Column pruning: only columns referenced by filters and projections are materialized from cold storage

### Aggregation Pushdown

v0.4.0 pushes `COUNT(*)`, `SUM()`, `MIN()`, `MAX()` directly to cold volume statistics when possible. This makes aggregations on large cold tables near-instant. The optimization is disabled for snapshot isolation transactions (which need tombstone filtering).

## Pre-Migration Checklist

Before upgrading:

1. **Backup your database directory** including `snapshots/` and `wal/`
2. **Ensure clean shutdown** of v0.3.7 (wait for any in-progress snapshots to complete)
3. **Check disk space**: Migration temporarily needs ~2x the snapshot data size (old snapshots + new volumes exist briefly during migration)
4. **Plan for downtime**: The database is unavailable during migration (proportional to data size)
5. **Test first**: Copy your database to a test location and open with v0.4.0 to verify

## Post-Migration Verification

After migration, verify your data:

```sql
-- Check all tables exist
SHOW TABLES;

-- Verify row counts
SELECT COUNT(*) FROM your_table;

-- Spot-check values
SELECT * FROM your_table WHERE id = 1;

-- Verify indexes work
SELECT * FROM your_table WHERE indexed_column = 'value';

-- Test writes
INSERT INTO your_table VALUES (...);
UPDATE your_table SET col = 'new' WHERE id = 1;
DELETE FROM your_table WHERE id = 2;
```

## Rollback Plan

If you need to revert to v0.3.7:

1. **Before migration**: Simply use the v0.3.7 binary with your original database
2. **After migration**: The `snapshots/` directory has been removed. You must restore from your pre-migration backup

There is no automatic downgrade path from v0.4.0 to v0.3.7. Always backup before upgrading.

## Disaster Recovery

If your v0.4.0 database becomes corrupted (e.g., disk failure, interrupted checkpoint), use the CLI restore:

```bash
# Restore from backup snapshot (removes corrupted volumes and WAL)
stoolap -d "file:///path/to/db" --restore

# Or reset volumes first, then open normally
stoolap -d "file:///path/to/db" --reset-volumes
```

The `--restore` flag works at the filesystem level without opening the database engine. It removes `volumes/`, `wal/`, and `db.lock`, then opens the database which rebuilds from the backup snapshot files.

To ensure you always have backup snapshots available:

```bash
# Create regular backups
stoolap -d "file:///path/to/db" --snapshot

# Or from SQL
PRAGMA SNAPSHOT;
```

## Driver Compatibility

All official drivers (Go, Python, Node.js, WASM) are compatible with v0.4.0. The migration is transparent to drivers since it happens at the storage layer. No driver code changes are needed.

If you use the C FFI (`include/stoolap.h`), the API is unchanged. The header file has been updated with new PRAGMA documentation but no function signature changes.

## Troubleshooting

### Migration appears stuck

For very large databases (>1GB), migration can take minutes. Check stderr for progress messages. If no output appears for >5 minutes, the process may be hung. Kill it safely and retry. The migration is idempotent.

### "db.lock" prevents opening

If a previous process crashed during migration, the lock file may be stale. On macOS/Linux, the flock is released when the process exits. Simply retry opening. If the file persists, it is safe to delete `db.lock` when no other process is accessing the database.

### Missing data after migration

If row counts differ, check:
1. Were there uncommitted transactions when v0.3.7 shut down? Only committed data is migrated.
2. Was the WAL intact? WAL corruption may cause loss of post-snapshot data.
3. Did you shut down v0.3.7 cleanly? A dirty shutdown with WAL corruption can lose recent writes.

### Performance regression after migration

If queries are slower after migration:
1. Run `PRAGMA CHECKPOINT` to ensure all data is in cold volumes (not hot buffer)
2. Check that indexes are created. Migration preserves index definitions but rebuilds them. Large tables may need time for index population.
3. v0.4.0 uses lazy scanning. Some query patterns may behave differently. File an issue if you observe consistent regressions.
