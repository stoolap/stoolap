---
layout: doc
title: Connection String Reference
category: Getting Started
order: 3
---

# Connection String Reference

This document provides information about Stoolap connection string formats and their usage.

## Connection String Basics

Stoolap connection strings follow a URL-like format:

```
scheme://[path][?options]
```

Where:
- `scheme` specifies the storage mode (memory or file)
- `path` provides location information for persistent storage
- `options` are optional query parameters for configuration

## Storage Modes

Stoolap supports two storage modes:

### In-Memory Mode (memory://)

```
memory://
```

All data stored in RAM:
- Maximum performance for temporary data
- Full MVCC transaction isolation
- Data is lost when the process terminates
- Ideal for testing and ephemeral workloads

### Persistent Mode (file://)

```
file:///path/to/data
```

Data persisted to disk:
- WAL (Write-Ahead Logging) for crash recovery
- Immutable cold volumes for fast startup on large tables
- Optional backup snapshots for disaster recovery
- Same MVCC features as memory mode
- Survives process restarts

Examples:
```
file:///data/mydb
file:///Users/username/stoolap/data
file:///tmp/test_db
```

## Configuration Options

Configuration can be set via query parameters in the connection string:

```
file:///path/to/data?sync_mode=normal&checkpoint_interval=60
```

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `sync` / `sync_mode` | none, normal, full (or 0, 1, 2) | normal | WAL synchronization mode. none=no fsync (data durable at checkpoint), normal=fsync every 1s, full=fsync every write |
| `checkpoint_interval` | Integer (seconds) | 60 | Time between automatic checkpoint cycles |
| `compact_threshold` | Integer | 4 | Volume count before compaction triggers |
| `keep_snapshots` | Integer | 3 | Backup snapshots to retain per table |
| `wal_flush_trigger` | Integer (bytes) | 32768 | Size in bytes before WAL flush |
| `wal_buffer_size` | Integer (bytes) | 65536 | WAL write buffer size |
| `wal_max_size` | Integer (bytes) | 67108864 | Maximum WAL file size (64MB) |
| `wal_compression` | on/off | on | Enable LZ4 compression for WAL entries |
| `volume_compression` | on/off | on | Enable LZ4 compression for cold volume files |
| `compression` | on/off | on | Enable LZ4 compression for both WAL and volumes |
| `compression_threshold` | Integer (bytes) | 64 | Minimum size for WAL compression |
| `checkpoint_on_close` | on/off | on | Seal all hot rows to volumes on clean shutdown |
| `commit_batch_size` | Integer | 100 | Commits to batch before sync |
| `sync_interval_ms` | Integer (ms) | 1000 | Minimum time between syncs in normal mode |

Legacy parameter names are accepted for backward compatibility:
- `snapshot_interval` maps to `checkpoint_interval`
- `snapshot_compression` maps to `compression` (sets both WAL and volume)

### Cleanup Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `cleanup` | on/off | on | Enable/disable background cleanup thread |
| `cleanup_interval` | Integer (seconds) | 60 | Interval between automatic cleanup runs |
| `deleted_row_retention` | Integer (seconds) | 300 | How long deleted rows are kept before permanent removal |
| `transaction_retention` | Integer (seconds) | 3600 | How long stale transaction metadata is kept |

The background cleanup thread periodically removes deleted rows, old version chains, and stale transaction metadata. On WASM where background threads are unavailable, use the `VACUUM` command for manual cleanup.

Example:
```
file:///data/mydb?cleanup_interval=30&deleted_row_retention=60
file:///data/mydb?cleanup=off
```

### Sync Mode Details

| Mode | Value | Description |
|------|-------|-------------|
| none | 0 | No fsync. Data is durable only after checkpoint writes volumes to disk. |
| normal | 1 | Fsync every 1 second (configurable via `sync_interval_ms`). DDL fsyncs immediately. Comparable to SQLite WAL + synchronous=NORMAL. |
| full | 2 | Fsync on every write operation. Maximum durability. |

## Usage Examples

### Rust API

```rust
use stoolap::Database;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // In-memory database
    let db = Database::open("memory://")?;

    // Or create a unique in-memory instance
    let db = Database::open_in_memory()?;

    // Persistent database
    let db = Database::open("file:///data/mydb")?;

    // With configuration options
    let db = Database::open("file:///data/mydb?sync_mode=full&checkpoint_interval=60")?;

    // Execute SQL
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", ())?;
    db.execute("INSERT INTO users VALUES ($1, $2)", (1, "Alice"))?;

    // Query data
    for row in db.query("SELECT * FROM users WHERE id = $1", (1,))? {
        let row = row?;
        let name: String = row.get_by_name("name")?;
        println!("Name: {}", name);
    }

    Ok(())
}
```

### Command Line

```bash
# In-memory database (default)
stoolap

# Persistent database
stoolap --db "file:///data/mydb"

# With configuration via DSN
stoolap --db "file:///data/mydb?sync_mode=full"

# With configuration via CLI flags
stoolap --db "file:///data/mydb" --sync full --checkpoint-interval 30

# Execute a query directly
stoolap --db "file:///data/mydb" -e "SELECT * FROM users"

# Execute from a SQL file
stoolap --db "file:///data/mydb" -f script.sql
```

### Backup and Restore

```bash
# Create a backup snapshot
stoolap --db "file:///data/mydb" --snapshot

# Restore from latest backup (filesystem-level, works with corrupted data)
stoolap --db "file:///data/mydb" --restore

# Restore from a specific backup by timestamp
stoolap --db "file:///data/mydb" --restore "20260315-100000.000"

# Recovery from corrupted volumes (removes volumes/ before restore)
stoolap --db "file:///data/mydb" --reset-volumes --restore
```

The `--restore` command works at the filesystem level without opening the database engine. It removes corrupted `volumes/` and `wal/` directories, then opens the database which rebuilds from the backup snapshot files in `snapshots/`.

## PRAGMA Configuration

You can also configure settings after connection using PRAGMA commands:

```sql
-- Set configuration values
PRAGMA sync_mode = 2;
PRAGMA checkpoint_interval = 60;
PRAGMA compact_threshold = 4;
PRAGMA keep_snapshots = 5;
PRAGMA wal_flush_trigger = 500;

-- Read current values
PRAGMA sync_mode;
PRAGMA checkpoint_interval;
PRAGMA keep_snapshots;

-- Create a backup snapshot
PRAGMA SNAPSHOT;

-- Restore from a backup snapshot
PRAGMA RESTORE;
PRAGMA RESTORE = '20260315-100000.000';

-- Run checkpoint cycle manually
PRAGMA CHECKPOINT;

-- Manual cleanup (works on all platforms including WASM)
VACUUM;
VACUUM table_name;
PRAGMA vacuum;
```

See the [PRAGMA Commands]({% link _docs/sql-commands/pragma-commands.md %}) documentation for details.

## Best Practices

1. **Development**: Use `memory://` for fast iteration and testing
2. **Production**: Use `file://` with appropriate `sync_mode`
3. **Critical data**: Set `sync_mode=full` for maximum durability
4. **High throughput**: Use `sync_mode=normal` with the default checkpoint cycle
5. **Backup**: Use `PRAGMA SNAPSHOT` or `--snapshot` for backups, `keep_snapshots` to control retention
6. **Recovery**: Use `--restore` from CLI to recover from corrupted state
