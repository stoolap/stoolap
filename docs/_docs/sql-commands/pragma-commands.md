---
layout: doc
title: PRAGMA Commands
category: SQL Commands
order: 4
---

# PRAGMA Commands

This document describes the PRAGMA commands available in Stoolap based on implementations and test cases.

## Overview

Stoolap provides PRAGMA commands for configuring and inspecting the database engine. These commands primarily focus on persistence settings and storage behavior.

## Syntax

The basic syntax for PRAGMA commands is:

```sql
PRAGMA [pragma_name] = [value];
```

or to retrieve the current value:

```sql
PRAGMA [pragma_name];
```

## Available PRAGMA Commands

Stoolap currently supports the following PRAGMA commands:

### Checkpoint and WAL Configuration

| PRAGMA | Description | Default |
|--------|-------------|---------|
| `checkpoint_interval` | Seconds between automatic checkpoints | 60 |
| `compact_threshold` | Sub-target volumes per table before merging | 4 |
| `keep_snapshots` | Backup snapshots to retain per table | 3 |
| `target_volume_rows` | Target rows per cold volume (compaction split boundary) | 1048576 |
| `snapshot` | Create a full backup snapshot | - |
| `restore` | Restore database from a backup snapshot | - |
| `checkpoint` | Run checkpoint cycle (seal + compact + WAL truncate) | - |
| `vacuum` | Manual cleanup of deleted rows and index compaction | - |
| `sync_mode` | Read current WAL sync mode (read-only, set via DSN) | 1 |
| `wal_flush_trigger` | Read current WAL flush trigger (read-only, set via DSN) | 32768 |
| `volume_stats` | Show per-volume storage statistics | - |

### Memory and Hot Store

| PRAGMA | Description | Default |
|--------|-------------|---------|
| `hot_max_rows` | Committed hot rows per table that request an early seal (0 = off) | 262144 |
| `hot_max_bytes` | Hot row bytes per table above which commits wait for a seal (0 = off) | 0 |
| `group_cache_mb` | Budget of the decoded row group cache in MB (0 = bypass) | 64 |
| `group_cache_stats` | Show the decoded row group cache's budget, size and hit counts | - |
| `memory_stats` | Show per-table hot and cold memory figures | - |

#### checkpoint_interval

Controls how often the background checkpoint cycle runs (in seconds). The checkpoint seals hot buffer rows into immutable cold volumes, persists manifests, and truncates the WAL. Default: 60.

```sql
PRAGMA checkpoint_interval = 60;
PRAGMA checkpoint_interval;       -- read current value

-- Legacy alias (backward compatible)
PRAGMA snapshot_interval = 60;
```

#### sync_mode (read-only)

Returns the current WAL synchronization mode. This setting can only be configured via the connection string and takes effect at database open time.

```sql
PRAGMA sync_mode;               -- read current value (0=none, 1=normal, 2=full)
```

To change sync_mode, set it in the connection string:

```
file:///path/to/db?sync_mode=none
file:///path/to/db?sync_mode=normal
file:///path/to/db?sync_mode=full
```

#### compact_threshold

Controls how many sub-target volumes (smaller than `target_volume_rows`) accumulate per table before compaction merges them. At-target volumes are not counted and are never rewritten unless they have tombstoned rows. Default: 4.

```sql
PRAGMA compact_threshold = 4;
PRAGMA compact_threshold;          -- read current value
```

Note: `compact_threshold` and `keep_snapshots` are separate settings. `compact_threshold` controls cold volume compaction. `keep_snapshots` controls backup snapshot file retention.

#### wal_flush_trigger (read-only)

Returns the current WAL flush trigger (buffer size in bytes before flush). This setting can only be configured via the connection string.

```sql
PRAGMA wal_flush_trigger;       -- read current value
```

To change wal_flush_trigger, set it in the connection string:

```
file:///path/to/db?wal_flush_trigger=65536
```

#### keep_snapshots

Controls how many backup snapshots are retained per table. Older snapshots beyond this count are automatically deleted after each `PRAGMA snapshot`. Default: 3.

```sql
PRAGMA keep_snapshots = 5;
PRAGMA keep_snapshots;              -- read current value
```

#### target_volume_rows

Controls the target number of rows per cold volume. Compaction splits output into volumes of approximately this size (rounded down to the nearest row-group boundary of 64K rows). Values below 65,536 are rejected. Default: 1,048,576 (1M rows, 16 row groups).

```sql
PRAGMA target_volume_rows = 1048576;
PRAGMA target_volume_rows;          -- read current value
```

#### hot_max_rows

A commit that takes a table's committed hot rows past this value asks for a seal right away. The checkpoint thread runs the seal cycle on its next tick (within 100 ms) instead of waiting for `checkpoint_interval`, and the seal's own row thresholds are bounded by the same value. The trigger keeps the hot store of a bulk load bounded at the cost of running the seal more often during the load. 0 turns it off. Default: 262,144 (four row groups).

```sql
PRAGMA hot_max_rows = 262144;
PRAGMA hot_max_rows = 0;            -- seal only at the checkpoint interval
PRAGMA hot_max_rows;                -- read current value
```

#### hot_max_bytes

When a table holds at least this many hot row bytes, a commit that writes to it requests a seal and waits, before it takes the seal fence, until the seal brings the table back under the limit. The wait is bounded at 10 seconds and counted (see `memory_stats`). A transaction that updates or deletes existing hot rows does not wait, since the seal skips rows it has claimed. In-memory databases never wait because nothing can seal. 0 turns the wait off. Default: 0.

```sql
PRAGMA hot_max_bytes = 268435456;   -- 256 MB per table
PRAGMA hot_max_bytes;               -- read current value
```

#### group_cache_mb

Sets the budget of the decoded row group cache shared by all cold volumes in the process. Queries over warm volumes decode the row groups they need from the in-memory compressed blocks; the cache keeps decoded groups so repeated queries do not decode them again. When a workload's groups exceed the budget, every query decodes again: a `SELECT` that scans a few million warm rows with several projected columns needs a few hundred MB. Raising the budget trades memory for repeat query time; 0 bypasses the cache. Default: 64.

```sql
PRAGMA group_cache_mb = 512;
PRAGMA group_cache_mb;              -- read current value in MB
```

#### group_cache_stats

Returns one row with the cache's budget, current size, entry count and hit and miss counters since the process started.

```sql
PRAGMA group_cache_stats;
-- budget_bytes | bytes | entries | hits | misses
```

#### memory_stats

Returns one row per table and a final `*` row with the totals.

| Column | Meaning |
|--------|---------|
| `table_name` | Table, or `*` for the totals |
| `hot_rows` | Committed rows in the hot store |
| `hot_bytes` | Bytes of the rows the hot arena holds, heap text and extension payloads included |
| `arena_slots` | Arena slots in use, deleted and cleared ones included |
| `arena_capacity_bytes` | Bytes the arena's slot vectors reserve |
| `chain_entries` | Previous row versions kept alive by version chains |
| `volume_bytes` | Memory of the table's loaded cold volumes |
| `admission_waits` | On the `*` row: commits that waited for a seal under `hot_max_bytes` |

```sql
PRAGMA memory_stats;
```

### Manual Snapshot and Checkpoint Control

#### snapshot

Creates a full backup snapshot of all tables. Snapshot files (.bin) are stored in the `snapshots/` directory with per-timestamp `ddl-{timestamp}.bin` and `manifest-{timestamp}.json` files. The `keep_snapshots` setting limits how many snapshot files are retained per table.

```sql
-- Create a full backup snapshot
PRAGMA snapshot;
```

The snapshot captures a consistent point-in-time view of all tables. This is useful for:
- Creating consistent backup points before critical operations
- Manual full-database backup for disaster recovery
- Ensuring data is persisted before shutting down

Note: This PRAGMA cannot run inside an explicit transaction.

#### checkpoint

Runs the checkpoint cycle, which is the core persistence mechanism for the hot/cold volume architecture:

```sql
-- Run the checkpoint cycle
PRAGMA checkpoint;
```

The checkpoint cycle performs these steps in order:

1. **Seal**: Move eligible hot buffer rows to immutable cold volumes (.vol files)
2. **Persist**: Write manifests (volume list, tombstones, checkpoint LSN) to disk
3. **WAL truncate**: Remove WAL entries before checkpoint LSN (only when all hot data is sealed)
4. **Compact**: Merge sub-target, oversized, and tombstoned volumes into target-sized outputs

The background thread runs this cycle automatically every `checkpoint_interval` seconds. On clean shutdown, a force checkpoint seals ALL remaining hot rows regardless of threshold.

This command is useful for:
- Ensuring data is persisted to volumes before critical operations
- Reclaiming memory by moving hot data to columnar cold segments
- Manual control over checkpoint timing instead of relying on `checkpoint_interval`

Note: This PRAGMA cannot run inside an explicit transaction.

#### restore

Restores the database state from backup snapshots created by `PRAGMA snapshot`. This is a destructive operation that replaces all current data with the snapshot data.

Without a timestamp, restore uses the latest `manifest-*.json` to filter which tables are eligible (preventing dropped tables from being resurrected), then picks the newest snapshot file per eligible table. If no manifest exists (older snapshots), all table directories are included. Index and view definitions are loaded from the `ddl-{timestamp}.bin` matching the oldest selected snapshot; if that file is missing, current in-memory definitions are preserved as a fallback.

With a timestamp, restore selects the exact snapshot file per table matching that timestamp and loads the corresponding `ddl-{timestamp}.bin` for index/view definitions. If the DDL file is missing, the restore fails with an error.

```sql
-- Restore from a specific snapshot by timestamp (recommended)
PRAGMA restore = '20260315-120000.000';

-- Restore from the latest backup snapshot
PRAGMA restore;
```

The restore operation:

1. **Validates** all snapshot files before making any changes
2. **Reads** `ddl-{timestamp}.bin` for index and view definitions
3. **Truncates** WAL to prevent post-snapshot entries from overwriting restored data
4. **Clears** all current data (hot buffer, cold volumes, in-memory state)
5. **Loads** snapshot data for each table
6. **Recreates** indexes and views from DDL metadata
7. **Syncs** auto-increment counters with restored data
8. **Re-records** DDL to WAL for crash safety
9. **Checkpoints** the restored data into volumes for immediate durability

If the database cannot open due to corrupted volumes or manifests, use the CLI with `--reset-volumes --restore` to clean up bad on-disk state before restoring.

This command is useful for:
- Rolling back to a known good state after accidental data corruption
- Point-in-time recovery from a backup
- Testing with a consistent dataset

The timestamp format matches the snapshot filename: `YYYYMMDD-HHMMSS.fff` (e.g. `20260315-120000.000`). You can find available timestamps by listing the snapshot files in the `snapshots/<table>/` directory.

Important notes:
- This PRAGMA cannot run inside an explicit transaction
- Indexes and views are automatically preserved via `ddl-{timestamp}.bin` saved by `PRAGMA SNAPSHOT`
- Tables created after the snapshot will not exist after restore
- Backup snapshots in `snapshots/` are preserved (not deleted) for future restores

For recovery from corrupted databases where `Database::open()` fails, use the CLI with `--reset-volumes` to clean up bad on-disk state before restoring:

```bash
stoolap -d "file:///path/to/db" --reset-volumes --restore
```

#### dedup_segments

Previously used to fix ghost duplicate rows across cold segments. Deduplication is now handled automatically during the seal/compact cycle, so this pragma is a no-op.

```sql
PRAGMA dedup_segments;
```

### Maintenance

#### vacuum

Performs manual cleanup of deleted rows, old version chains, stale transaction metadata, and triggers index compaction (e.g., HNSW graph rebuild when tombstone ratio exceeds 20%).

```sql
PRAGMA vacuum;
```

**Note:** This PRAGMA cannot run inside an explicit transaction.

## Connection String Parameters

All PRAGMA values can also be set via the connection string:

```
file:///path/to/db?checkpoint_interval=60&compact_threshold=4&keep_snapshots=3&sync_mode=normal&hot_max_rows=262144
```

`group_cache_mb` has no connection string form; the cache is process-wide and is set with the PRAGMA.

Legacy parameter names are accepted for backward compatibility:
- `snapshot_interval` maps to `checkpoint_interval`
- `snapshot_compression` maps to `compression` (sets both `wal_compression` and `volume_compression`)
