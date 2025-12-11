---
layout: doc
title: Persistence
category: Architecture
order: 7
---

# Persistence

Stoolap provides durable storage through a combination of Write-Ahead Logging (WAL) and periodic snapshots. This architecture ensures data durability while maintaining high performance.

## Overview

Stoolap's persistence layer consists of two main components:

1. **Write-Ahead Log (WAL)**: Records all changes before they're applied to memory
2. **Snapshots**: Periodic full copies of the database state

This dual approach provides:
- **Durability**: Changes are persisted before acknowledgment
- **Fast Recovery**: Snapshots reduce recovery time
- **Crash Safety**: WAL ensures no committed transactions are lost

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
- **CREATE TABLE**: Table creation (DDL)
- **DROP TABLE**: Table deletion (DDL)
- **CREATE INDEX**: Index creation (DDL)
- **DROP INDEX**: Index deletion (DDL)

### WAL Configuration

Configure WAL behavior using PRAGMA:

```sql
-- Sync mode: 0=None, 1=Normal (default), 2=Full
PRAGMA sync_mode = 1;

-- Number of operations before automatic WAL flush
PRAGMA wal_flush_trigger = 32768;
```

| Sync Mode | Value | Behavior |
|-----------|-------|----------|
| None | 0 | No sync (fastest, but data may be lost on crash) |
| Normal | 1 | Sync on commit (balanced performance and durability) |
| Full | 2 | Sync every operation (slowest, maximum durability) |

### WAL Files

WAL files are stored in the database directory:
```
/path/to/database/
  wal/
    wal_000001.log
    wal_000002.log
    ...
```

Old WAL files are automatically cleaned up after successful snapshots.

## Snapshots

### How Snapshots Work

Snapshots capture the complete database state at a point in time:
1. All table data and schema
2. All index definitions
3. Current transaction state

After a snapshot is created, older WAL entries can be safely deleted.

### Snapshot Configuration

```sql
-- Interval between automatic snapshots (in seconds, default: 300)
PRAGMA snapshot_interval = 300;

-- Number of snapshots to retain (default: 5)
PRAGMA keep_snapshots = 5;

-- Manually create a snapshot
PRAGMA create_snapshot;
```

### Snapshot Files

Snapshots are stored as binary files:
```
/path/to/database/
  snapshots/
    snapshot_1704067200.bin
    snapshot_1704067500.bin
    ...
```

The filename includes the timestamp of creation.

## Recovery Process

When opening a database, Stoolap performs recovery automatically:

1. **Load Latest Snapshot**: Restore base state from most recent snapshot
2. **Replay WAL**: Apply any operations logged after the snapshot
3. **Verify Consistency**: Ensure all data structures are valid
4. **Ready for Operations**: Database is now available

Recovery is transparent to the application.

### Recovery Example

```rust
// Opening automatically triggers recovery if needed
let db = Database::open("file:///path/to/database")?;

// Database is ready with all committed data restored
let results = db.query("SELECT * FROM users")?;
```

## Configuration Options

### Connection String Parameters

```
file:///path/to/database?sync_mode=2&snapshot_interval=60&keep_snapshots=3
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| sync_mode | WAL sync mode (0, 1, 2) | 1 |
| snapshot_interval | Seconds between snapshots | 300 |
| keep_snapshots | Number of snapshots to keep | 5 |

### PRAGMA Commands

```sql
-- Read current settings
PRAGMA sync_mode;
PRAGMA snapshot_interval;
PRAGMA keep_snapshots;
PRAGMA wal_flush_trigger;

-- Modify settings
PRAGMA sync_mode = 2;
PRAGMA snapshot_interval = 60;
PRAGMA keep_snapshots = 3;
PRAGMA wal_flush_trigger = 10000;

-- Manually trigger snapshot
PRAGMA create_snapshot;
```

## Best Practices

### Durability vs Performance

Choose sync_mode based on your requirements:

| Use Case | Recommended sync_mode |
|----------|----------------------|
| Development/Testing | 0 (None) |
| General Use | 1 (Normal) |
| Financial/Critical Data | 2 (Full) |

### Snapshot Frequency

- **Frequent snapshots** (low interval): Faster recovery, more disk I/O
- **Infrequent snapshots** (high interval): Slower recovery, less disk I/O

For databases with high write rates, consider shorter intervals.

### Disk Space

Monitor disk usage:
- WAL files grow until the next snapshot
- Old snapshots are retained based on `keep_snapshots`
- Plan for peak WAL size between snapshots

### Backup Strategy

For backups:
1. Create a manual snapshot: `PRAGMA create_snapshot;`
2. Copy the entire database directory while the database is idle
3. For hot backups, use filesystem snapshots (ZFS, LVM)

## Directory Structure

A persistent database creates this directory structure:

```
/path/to/database/
  db.lock              # Lock file for single-writer
  wal/
    wal_NNNNNN.log     # WAL segment files
  snapshots/
    snapshot_TIMESTAMP.bin  # Snapshot files
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
        "file:///var/lib/myapp/data?sync_mode=2&snapshot_interval=120"
    )?;

    // Fine-tune at runtime
    db.execute("PRAGMA wal_flush_trigger = 5000")?;
    db.execute("PRAGMA keep_snapshots = 7")?;

    // Your application logic...
    db.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTO_INCREMENT, data JSON)")?;

    // Force a snapshot before maintenance
    db.execute("PRAGMA create_snapshot")?;

    Ok(())
}
```
