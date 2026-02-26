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
- Periodic snapshots for fast recovery
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
file:///path/to/data?sync_mode=normal&snapshot_interval=60
```

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `sync` / `sync_mode` | none, normal, full (or 0, 1, 2) | normal | WAL synchronization mode |
| `snapshot_interval` | Integer (seconds) | 300 | Time between automatic snapshots |
| `keep_snapshots` | Integer | 5 | Number of snapshots to retain |
| `wal_flush_trigger` | Integer (bytes) | 32768 | Size in bytes before WAL flush |
| `wal_buffer_size` | Integer (bytes) | 65536 | WAL write buffer size |
| `wal_max_size` | Integer (bytes) | 67108864 | Maximum WAL file size (64MB) |
| `commit_batch_size` | Integer | 100 | Operations batched per commit flush |
| `sync_interval_ms` | Integer (ms) | 10 | Background sync interval |
| `wal_compression` | on/off | on | Enable WAL compression |
| `snapshot_compression` | on/off | on | Enable snapshot compression |
| `compression` | on/off | on | Enable both WAL and snapshot compression |
| `compression_threshold` | Integer (bytes) | 64 | Minimum size for compression |

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
| none | 0 | No sync (fastest, risk of data loss) |
| normal | 1 | Sync on flush (balanced) |
| full | 2 | Sync every write (maximum durability) |

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
    let db = Database::open("file:///data/mydb?sync_mode=full&snapshot_interval=60")?;

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

# With configuration
stoolap --db "file:///data/mydb?sync_mode=full"

# Execute a query directly
stoolap --db "file:///data/mydb" -e "SELECT * FROM users"
```

## PRAGMA Configuration

You can also configure settings after connection using PRAGMA commands:

```sql
-- Set configuration values
PRAGMA sync_mode = 2;
PRAGMA snapshot_interval = 60;
PRAGMA keep_snapshots = 5;
PRAGMA wal_flush_trigger = 500;

-- Read current values
PRAGMA sync_mode;
PRAGMA snapshot_interval;

-- Create a snapshot manually
PRAGMA snapshot;

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
4. **High throughput**: Use `sync_mode=normal` with periodic `PRAGMA snapshot`
5. **Limited disk space**: Reduce `keep_snapshots` to retain fewer snapshots
