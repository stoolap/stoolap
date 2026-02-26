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

### Snapshot and WAL Configuration

| PRAGMA | Description | Default |
|--------|-------------|---------|
| `snapshot_interval` | Seconds between automatic snapshots | 300 |
| `sync_mode` | WAL sync mode (0=None, 1=Normal, 2=Full) | 1 |
| `keep_snapshots` | Number of snapshots to retain per table | 5 |
| `wal_flush_trigger` | Buffer size in bytes before WAL flush | 32768 |
| `snapshot` | Manually create a snapshot (no value) | - |
| `checkpoint` | Alias for `snapshot` (SQLite-compatible) | - |
| `vacuum` | Manual cleanup of deleted rows and index compaction | - |

#### snapshot_interval

Controls how often the database creates snapshots of the data (in seconds). Default: 300.

```sql
PRAGMA snapshot_interval = 60;
PRAGMA snapshot_interval;       -- read current value
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

#### keep_snapshots

Controls how many snapshots to retain for each table. Default: 5.

```sql
PRAGMA keep_snapshots = 5;
PRAGMA keep_snapshots;          -- read current value
```

#### wal_flush_trigger

Controls the number of operations before the WAL is flushed to disk. Default: 32768.

```sql
PRAGMA wal_flush_trigger = 1000;
PRAGMA wal_flush_trigger;       -- read current value
```

### Manual Snapshot Control

#### snapshot

Creates an immediate snapshot of all tables in the database:

```sql
-- Create a snapshot immediately
PRAGMA snapshot;

-- SQLite-compatible alias
PRAGMA checkpoint;
```

This command is useful for:
- Creating consistent backup points
- Ensuring data is persisted before critical operations
- Manual control over snapshot timing instead of relying on `snapshot_interval`

Note: This PRAGMA does not accept any values.

### Maintenance

#### vacuum

Performs manual cleanup of deleted rows, old version chains, stale transaction metadata, and triggers index compaction (e.g., HNSW graph rebuild when tombstone ratio exceeds 20%).

```sql
-- Vacuum all tables
PRAGMA vacuum;

-- Also available as a standalone SQL command
VACUUM;

-- Vacuum a specific table
VACUUM table_name;
```

Returns a result row with three columns:

| Column | Description |
|--------|-------------|
| `deleted_rows_cleaned` | Number of tombstoned rows reclaimed |
| `old_versions_cleaned` | Number of old version chains pruned |
| `transactions_cleaned` | Number of stale transaction entries removed |

This is especially useful on WASM where the background cleanup thread is unavailable, but can be called on any platform for on-demand maintenance.

**Warning:** VACUUM uses zero retention, meaning all historical row versions not needed by currently active transactions are permanently removed. This destroys AS OF TIMESTAMP history. Temporal queries referencing timestamps before the VACUUM will no longer return results. If you rely on time-travel queries, consider using the background cleanup (which preserves a 5-minute retention window) instead of VACUUM.

Note: This PRAGMA does not accept any values.

## Examples

### Basic PRAGMA Usage

```sql
-- Set snapshot interval to 60 seconds
PRAGMA snapshot_interval = 60;

-- Verify the setting
PRAGMA snapshot_interval;
```

### Multiple PRAGMA Commands

```sql
-- Set sync mode to full
PRAGMA sync_mode = 2;

-- Keep 10 snapshots per table
PRAGMA keep_snapshots = 10;

-- Set WAL flush trigger to 1000 operations
PRAGMA wal_flush_trigger = 1000;
```

### Manual Snapshot Example

```sql
-- Insert some data
INSERT INTO users (id, name) VALUES (1, 'John');

-- Create a snapshot immediately to ensure data is persisted
PRAGMA snapshot;

-- Continue with more operations
UPDATE users SET name = 'Jane' WHERE id = 1;

-- Create another snapshot after the update
PRAGMA snapshot;
```

### VACUUM Example

```sql
-- Delete some data
DELETE FROM orders WHERE status = 'cancelled';

-- Run vacuum to reclaim storage and compact indexes
VACUUM;

-- Or vacuum a specific table
VACUUM orders;

-- Or use PRAGMA syntax
PRAGMA vacuum;
```

## PRAGMA Persistence

PRAGMA settings affect the current engine instance in memory. They are not saved to disk, so they reset when the database is closed and reopened. To apply custom settings consistently, execute PRAGMA commands after opening the connection.

## Best Practices

1. **Tune Snapshot Interval**: Adjust `snapshot_interval` based on your workload. Lower values provide better durability but more I/O overhead.

2. **Choose Appropriate Sync Mode**: 
   - Use `sync_mode = 2` for critical data where durability is paramount
   - Use `sync_mode = 1` for most applications (good balance)
   - Use `sync_mode = 0` only for non-critical data or testing

3. **Manage Snapshots**: Set `keep_snapshots` based on your backup needs and disk space constraints.

4. **Apply PRAGMA at Startup**: Run important PRAGMA commands right after opening the database connection.

5. **Configure Cleanup via DSN**: Set cleanup retention values in the connection string for consistent behavior across connections:
   ```
   file:///data/mydb?cleanup_interval=30&deleted_row_retention=60&transaction_retention=1800
   ```

6. **Use VACUUM on WASM**: The background cleanup thread is unavailable on WASM. Use `VACUUM` or `PRAGMA vacuum` for manual cleanup instead.

## Cleanup Configuration (DSN Options)

Background cleanup settings can be configured via connection string query parameters:

| Option | Default | Description |
|--------|---------|-------------|
| `cleanup` | on | Enable/disable the background cleanup thread |
| `cleanup_interval` | 60 | Seconds between automatic cleanup runs |
| `deleted_row_retention` | 300 | Seconds before deleted rows are permanently removed |
| `transaction_retention` | 3600 | Seconds before stale transaction metadata is removed |

See [Connection Strings]({% link _docs/getting-started/connection-strings.md %}) for the full list of DSN options.

## Implementation Details

PRAGMA commands are handled directly by the storage engine and affect the persistence behavior of the database. They do not require transactions and take effect immediately after being set.
