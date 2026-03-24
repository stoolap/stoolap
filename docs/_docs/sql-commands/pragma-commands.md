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
| `sync_mode` | WAL sync mode (0=None, 1=Normal, 2=Full) | 1 |
| `compact_threshold` | Volume count before compaction triggers | 4 |
| `keep_snapshots` | Backup snapshots to retain per table | 3 |
| `wal_flush_trigger` | Buffer size in bytes before WAL flush | 32768 |
| `checkpoint_on_close` | Seal all hot rows on clean shutdown | on |
| `snapshot` | Create a full backup snapshot | - |
| `restore` | Restore database from a backup snapshot | - |
| `checkpoint` | Run checkpoint cycle (seal + compact + WAL truncate) | - |
| `vacuum` | Manual cleanup of deleted rows and index compaction | - |

#### checkpoint_interval

Controls how often the background checkpoint cycle runs (in seconds). The checkpoint seals hot buffer rows into immutable cold volumes, persists manifests, and truncates the WAL. Default: 60.

```sql
PRAGMA checkpoint_interval = 60;
PRAGMA checkpoint_interval;       -- read current value

-- Legacy alias (backward compatible)
PRAGMA snapshot_interval = 60;
```

#### sync_mode

Controls the synchronization mode for the Write-Ahead Log (WAL). Default: 1 (Normal).

```sql
PRAGMA sync_mode = 1;
PRAGMA sync_mode;               -- read current value
```

Supported values:
- 0: No sync (fastest, but risks data loss on power failure)
- 1: Normal sync (balances performance and durability)
- 2: Full sync (maximum durability, slowest performance)

#### compact_threshold

Controls how many cold volumes accumulate per table before compaction triggers. Compaction merges the smallest volumes first, bounded per cycle. Default: 4.

```sql
PRAGMA compact_threshold = 4;
PRAGMA compact_threshold;          -- read current value
```

Note: `compact_threshold` and `keep_snapshots` are separate settings. `compact_threshold` controls cold volume compaction. `keep_snapshots` controls backup snapshot file retention.

#### wal_flush_trigger

Controls the buffer size in bytes before the WAL is flushed to disk. Default: 32768.

```sql
PRAGMA wal_flush_trigger = 1000;
PRAGMA wal_flush_trigger;       -- read current value
```

#### keep_snapshots

Controls how many backup snapshots are retained per table. Older snapshots beyond this count are automatically deleted after each `PRAGMA snapshot`. Default: 3.

```sql
PRAGMA keep_snapshots = 5;
PRAGMA keep_snapshots;              -- read current value
```

#### checkpoint_on_close

Controls whether all hot rows are sealed to cold volumes on clean shutdown. When enabled (default), the engine runs a force checkpoint during `close()`, ensuring fast startup because no WAL replay is needed. Set to `off` only for crash simulation in tests.

```sql
PRAGMA checkpoint_on_close = on;
PRAGMA checkpoint_on_close;         -- read current value
```

### Manual Snapshot and Checkpoint Control

#### snapshot

Creates a full backup snapshot of all tables. Snapshot files (.bin) are stored in the `snapshots/` directory alongside a `ddl.bin` file containing index and view definitions. The `keep_snapshots` setting limits how many snapshot files are retained per table.

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
3. **Compact**: Merge the smallest volumes when count exceeds `compact_threshold`
4. **WAL truncate**: Remove WAL entries before checkpoint LSN (only when all hot data is sealed)

The background thread runs this cycle automatically every `checkpoint_interval` seconds. On clean shutdown, a force checkpoint seals ALL remaining hot rows regardless of threshold.

This command is useful for:
- Ensuring data is persisted to volumes before critical operations
- Reclaiming memory by moving hot data to columnar cold segments
- Manual control over checkpoint timing instead of relying on `checkpoint_interval`

Note: This PRAGMA cannot run inside an explicit transaction.

#### restore

Restores the database state from backup snapshots created by `PRAGMA snapshot`. This is a destructive operation that replaces all current data with the snapshot data.

```sql
-- Restore from the latest backup snapshot
PRAGMA restore;

-- Restore from a specific snapshot by timestamp
PRAGMA restore = '20260315-120000.000';
```

The restore operation:

1. **Validates** all snapshot files before making any changes
2. **Reads** `ddl.bin` for index and view definitions (if available, falls back to saving current in-memory definitions)
3. **Truncates** WAL to prevent post-snapshot entries from overwriting restored data
4. **Clears** all current data (hot buffer, cold volumes, in-memory state)
5. **Loads** snapshot data for each table
6. **Recreates** indexes and views from `ddl.bin` or fallback
7. **Syncs** auto-increment counters with restored data
8. **Re-records** DDL to WAL for crash safety
9. **Checkpoints** the restored data into volumes for immediate durability

This command is useful for:
- Rolling back to a known good state after accidental data corruption
- Point-in-time recovery from a backup
- Testing with a consistent dataset

The timestamp format matches the snapshot filename: `YYYYMMDD-HHMMSS.fff` (e.g. `20260315-120000.000`). You can find available timestamps by listing the snapshot files in the `snapshots/<table>/` directory.

Important notes:
- This PRAGMA cannot run inside an explicit transaction
- Indexes and views are automatically preserved via `ddl.bin` saved by `PRAGMA SNAPSHOT`
- If no `ddl.bin` exists (snapshots from older versions), indexes and views are saved from the current in-memory state before the destructive step
- Tables created after the snapshot will not exist after restore
- Backup snapshots in `snapshots/` are preserved (not deleted) for future restores

For recovery from corrupted databases where `Database::open()` fails, use the CLI `--restore` flag instead, which works at the filesystem level:

```bash
stoolap -d "file:///path/to/db" --restore
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

## Connection String Parameters

All PRAGMA values can also be set via the connection string:

```
file:///path/to/db?checkpoint_interval=60&compact_threshold=4&keep_snapshots=3&sync_mode=1
```

Legacy parameter names are accepted for backward compatibility:
- `snapshot_interval` maps to `checkpoint_interval`
- `snapshot_compression` maps to `wal_compression`
