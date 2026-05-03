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
| `sync` / `sync_mode` | none/off, normal, full (or 0, 1, 2) | normal | WAL synchronization mode. none/off=no fsync (data durable at checkpoint), normal=fsync every 1s, full=fsync every write |
| `checkpoint_interval` | Integer (seconds) | 60 | Time between automatic checkpoint cycles |
| `compact_threshold` | Integer (min 2) | 4 | Sub-target volumes per table before merging |
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
| `sync_interval_ms` / `sync_interval` | Integer (ms) | 1000 | Minimum time between syncs in normal mode |
| `target_volume_rows` | Integer | 1048576 | Target rows per cold volume (min: 65536). Controls compaction split boundary. |
| `read_only` / `readonly` | true/false (or 1/0, yes/no, on/off) | false | Open in read-only mode. Must be passed to `Database::open_read_only(dsn)` (returns `ReadOnlyDatabase`); `Database::open(dsn)` REJECTS this flag with a clear error pointing to the right entry point. |
| `mode` | ro / rw | rw | SQLite-style alias for `read_only`. `mode=ro` is equivalent to `read_only=true`. Same routing rule applies. |
| `auto_refresh` | on/off (or true/false, 1/0, yes/no) | on | Read-only handles only. `on` (default): every query polls the manifest epoch (~1µs) and reloads if the writer advanced. `off` is the master switch for "no implicit refresh on this handle" — both the per-query path AND the background ticker (if any) pause until you re-enable it (or call `refresh()` explicitly). |
| `refresh_interval` | Duration (`Nms` / `Ns` / `Nm`, or `0`) | `0` | Read-only handles only. Spawns a background thread that calls `refresh()` every interval to advance the per-handle WAL pin while the handle is idle (otherwise the writer can't truncate WAL past it). Minimum 100ms. `0` disables. Pauses while `auto_refresh=off` or a `BEGIN` is active. |

Legacy parameter names are accepted for backward compatibility:
- `snapshot_interval` maps to `checkpoint_interval`
- `snapshot_compression` maps to `compression` (sets both WAL and volume)

### Read-only mode

Read-only access is enforced at the type system. There is one entry point — `Database::open_read_only(dsn)` — and one return type — `ReadOnlyDatabase` — which exposes only read methods (`query`, `query_named`, `cached_plan`, `query_plan`, `query_named_plan`, `table_exists`, `refresh`, `read_engine`, `set_auto_refresh`, `set_refresh_interval`, `try_clone`). `execute` and `begin` are not on `ReadOnlyDatabase` at all, so write SQL is a *compile-time* error rather than a runtime `ReadOnlyViolation`.

`Database::open(dsn)` REJECTS `?read_only=true` / `?readonly=true` / `?mode=ro` with `Error::InvalidArgument` containing the message *"read-only DSN flag passed to Database::open. Read-only handles must be opened via Database::open_read_only(dsn)..."*. The DSN string itself can be passed unchanged to `open_read_only`; the flag is accepted there as a redundant no-op.

Read-only opens:

- The engine acquires a *shared* file lock (`LOCK_SH`), so multiple processes can open the same database for reading at the same time. A writable open is rejected while any reader is active, and vice versa.
- The background cleanup thread is not started (read-only opens never modify on-disk state).
- Cross-process visibility uses lease files at `<db>/readers/<pid>.lease` plus an mmap-backed shm header at `<db>/db.shm`. The reader picks up writer checkpoints via the manifest-epoch poll and sub-checkpoint commits via WAL tailing. Writer reincarnation and post-attach DDL surface as typed must-reopen errors (`Error::SwmrWriterReincarnated`, `Error::SwmrPendingDdl`).
- Read-only opens against a non-stoolap directory or a missing path are refused; an empty database is never created on disk.
- Read-only opens work against directories on read-only mounts (the kernel-level read-only flag) AND chmod-read-only directories. Chmod-RO opens hold a long-lived `LOCK_SH` on `db.lock` so a privileged writer (different uid) cannot acquire `LOCK_EX` and reclaim WAL/volumes under the reader.

```
file:///data/mydb?read_only=true       # passed to open_read_only — accepted
file:///data/mydb?mode=ro              # same
file:///data/mydb?read_only=true&sync_mode=normal
file:///data/mydb                      # equally valid for open_read_only — flag is redundant

# Long-lived idle reader: keep WAL pin advancing every 30s so the
# writer can truncate WAL even when this handle issues no queries.
file:///data/mydb?refresh_interval=30s

# Ad-hoc stable multi-query block from the start: snapshot frozen
# at open until the caller flips set_auto_refresh(true).
file:///data/mydb?auto_refresh=off
```

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

    // Read-only handles MUST come from open_read_only — Database::open
    // rejects ?read_only=true with a clear error pointing here.
    let ro = Database::open_read_only("file:///data/mydb")?;
    // The DSN flag is also accepted (redundant) so existing driver
    // DSN strings keep working unchanged:
    let ro = Database::open_read_only("file:///data/mydb?read_only=true")?;
    // Or wrap an existing writable Database — in-process typed read surface,
    // shares the engine. No SWMR coordination concerns (same engine).
    let ro = db.as_read_only();

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

# Open read-only (CLI dispatches to Database::open_read_only internally).
# The --read-only flag is required; the DSN flag alone (without
# --read-only) reaches Database::open and is rejected with a clear
# migration error.
stoolap --db "file:///data/mydb" --read-only
stoolap --db "file:///data/mydb?read_only=true" --read-only
stoolap --db "file:///data/mydb?mode=ro" --read-only

# Execute a query directly
stoolap --db "file:///data/mydb" -e "SELECT * FROM users"

# Execute from a SQL file
stoolap --db "file:///data/mydb" -f script.sql
```

### Backup and Restore

```bash
# Create a backup snapshot
stoolap --db "file:///data/mydb" --snapshot

# Restore from a specific backup by timestamp (recommended)
stoolap --db "file:///data/mydb" --restore "20260315-100000.000"

# Restore from latest backup
stoolap --db "file:///data/mydb" --restore

# Recovery from corrupted volumes/manifests (cleans up first)
stoolap --db "file:///data/mydb" --reset-volumes --restore
```

The `--restore` command opens the database and replaces current data with snapshot data. If volumes or manifests are corrupted, use `--reset-volumes --restore` which removes bad on-disk state before opening.

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
6. **Recovery**: Use `--reset-volumes --restore` from CLI to recover from corrupted state
