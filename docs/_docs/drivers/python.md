---
layout: doc
title: Python Driver
category: Drivers
order: 2
---

# Python Driver

High-performance Python driver for Stoolap. Built with [PyO3](https://pyo3.rs) for native performance with both sync and async APIs. All operations release the GIL for true concurrency.

## Installation

```bash
pip install stoolap-python
```

Requires Python >= 3.9. Supported versions: 3.9, 3.10, 3.11, 3.12, 3.13.

## Quick Start

```python
from stoolap import Database

db = Database.open(':memory:')

db.exec("""
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT
    )
""")

# Insert with positional parameters ($1, $2, ...)
db.execute(
    'INSERT INTO users (id, name, email) VALUES ($1, $2, $3)',
    [1, 'Alice', 'alice@example.com']
)

# Insert with named parameters (:key)
db.execute(
    'INSERT INTO users (id, name, email) VALUES (:id, :name, :email)',
    {'id': 2, 'name': 'Bob', 'email': 'bob@example.com'}
)

# Query rows as dicts
users = db.query('SELECT * FROM users ORDER BY id')
# [{'id': 1, 'name': 'Alice', 'email': 'alice@example.com'}, ...]

# Query single row
user = db.query_one('SELECT * FROM users WHERE id = $1', [1])
# {'id': 1, 'name': 'Alice', 'email': 'alice@example.com'}

# Query in raw columnar format (faster)
raw = db.query_raw('SELECT id, name FROM users ORDER BY id')
# {'columns': ['id', 'name'], 'rows': [[1, 'Alice'], [2, 'Bob']]}

db.close()
```

## Opening a Database

```python
# In-memory
db = Database.open(':memory:')
db = Database.open('')
db = Database.open('memory://')

# File-based (data persists across restarts)
db = Database.open('./mydata')
db = Database.open('file:///absolute/path/to/db')
```

## Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `execute(sql, params?)` | `int` | Execute DML statement, return rows affected |
| `exec(sql)` | `None` | Execute one or more statements (no parameters) |
| `query(sql, params?)` | `list[dict]` | Query rows as dicts |
| `query_one(sql, params?)` | `dict \| None` | Query single row |
| `query_raw(sql, params?)` | `dict` | Query in columnar format |
| `execute_batch(sql, params_list)` | `int` | Execute with multiple param sets |
| `prepare(sql)` | `PreparedStatement` | Create a prepared statement |
| `begin()` | `Transaction` | Begin a transaction |
| `close()` | `None` | Close the database |

## Persistence

File-based databases persist data to disk using WAL (Write-Ahead Logging) and periodic snapshots. Data survives process restarts.

```python
db = Database.open('./mydata')

db.exec('CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT)')
db.execute('INSERT INTO kv VALUES ($1, $2)', ['hello', 'world'])
db.close()

# Reopen: data is still there
db2 = Database.open('./mydata')
row = db2.query_one('SELECT * FROM kv WHERE key = $1', ['hello'])
# {'key': 'hello', 'value': 'world'}
db2.close()
```

### Configuration

Pass configuration as query parameters in the path:

```python
# Maximum durability (fsync on every WAL write)
db = Database.open('./mydata?sync=full')

# High throughput (no fsync, larger buffers)
db = Database.open('./mydata?sync=none&wal_buffer_size=131072')

# Custom snapshot interval with compression
db = Database.open('./mydata?snapshot_interval=60&compression=on')

# Multiple options
db = Database.open(
    './mydata?sync=full&snapshot_interval=120&keep_snapshots=10&wal_max_size=134217728'
)
```

### Sync Modes

Controls the durability vs. performance trade-off:

| Mode | Value | Description |
|------|-------|-------------|
| `none` | `sync=none` | No fsync. Fastest, but data may be lost on crash |
| `normal` | `sync=normal` | Fsync on commit batches. Good balance (default) |
| `full` | `sync=full` | Fsync on every WAL write. Slowest, maximum durability |

### All Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sync` | `normal` | Sync mode: `none`, `normal`, or `full` |
| `snapshot_interval` | `300` | Seconds between automatic snapshots (5 min) |
| `keep_snapshots` | `5` | Number of snapshot files to retain |
| `wal_flush_trigger` | `32768` | WAL flush trigger size in bytes (32 KB) |
| `wal_buffer_size` | `65536` | WAL buffer size in bytes (64 KB) |
| `wal_max_size` | `67108864` | Max WAL file size before rotation (64 MB) |
| `commit_batch_size` | `100` | Commits to batch before syncing (normal mode) |
| `sync_interval_ms` | `10` | Minimum ms between syncs (normal mode) |
| `wal_compression` | `on` | LZ4 compression for WAL entries |
| `snapshot_compression` | `on` | LZ4 compression for snapshots |
| `compression` | -- | Set both `wal_compression` and `snapshot_compression` |
| `compression_threshold` | `64` | Minimum bytes before compressing an entry |

## Raw Query Format

`query_raw` returns `{"columns": [...], "rows": [[...], ...]}` instead of a list of dicts. Faster when you don't need named keys.

```python
raw = db.query_raw('SELECT id, name, email FROM users ORDER BY id')
print(raw['columns'])  # ['id', 'name', 'email']
print(raw['rows'])     # [[1, 'Alice', 'alice@example.com'], [2, 'Bob', 'bob@example.com']]
```

## Batch Execution

Execute the same SQL with multiple parameter sets in a single call. Automatically wraps in a transaction.

```python
changes = db.execute_batch(
    'INSERT INTO users VALUES ($1, $2, $3)',
    [
        [1, 'Alice', 'alice@example.com'],
        [2, 'Bob', 'bob@example.com'],
        [3, 'Charlie', 'charlie@example.com'],
    ]
)
print(changes)  # 3
```

## Prepared Statements

Prepared statements parse SQL once and reuse the cached execution plan on every call. No parsing or cache lookup overhead per execution.

```python
insert = db.prepare('INSERT INTO users VALUES ($1, $2, $3)')
insert.execute([1, 'Alice', 'alice@example.com'])
insert.execute([2, 'Bob', 'bob@example.com'])

lookup = db.prepare('SELECT * FROM users WHERE id = $1')
user = lookup.query_one([1])
# {'id': 1, 'name': 'Alice', 'email': 'alice@example.com'}
```

### Methods

All methods mirror `Database` but without the `sql` parameter (it's bound at prepare time).

| Method | Returns | Description |
|--------|---------|-------------|
| `execute(params?)` | `int` | Execute DML statement |
| `query(params?)` | `list[dict]` | Query rows as dicts |
| `query_one(params?)` | `dict \| None` | Query single row |
| `query_raw(params?)` | `dict` | Query in columnar format |
| `execute_batch(params_list)` | `int` | Execute with multiple param sets |

Property: `sql` returns the SQL text of this prepared statement.

### Batch with Prepared Statement

```python
insert = db.prepare('INSERT INTO users VALUES ($1, $2, $3)')
changes = insert.execute_batch([
    [1, 'Alice', 'alice@example.com'],
    [2, 'Bob', 'bob@example.com'],
    [3, 'Charlie', 'charlie@example.com'],
])
print(changes)  # 3
```

## Transactions

### Using Context Manager

```python
with db.begin() as tx:
    tx.execute('INSERT INTO users VALUES ($1, $2, $3)', [1, 'Alice', 'alice@example.com'])
    tx.execute('INSERT INTO users VALUES ($1, $2, $3)', [2, 'Bob', 'bob@example.com'])

    # Read within the transaction (sees uncommitted changes)
    rows = tx.query('SELECT * FROM users')
    one = tx.query_one('SELECT * FROM users WHERE id = $1', [1])
    raw = tx.query_raw('SELECT id, name FROM users')

    # Auto-commits on clean exit, auto-rollbacks on exception
```

### Manual Control

```python
tx = db.begin()
try:
    tx.execute('INSERT INTO users VALUES ($1, $2, $3)', [1, 'Alice', 'alice@example.com'])
    tx.execute('INSERT INTO users VALUES ($1, $2, $3)', [2, 'Bob', 'bob@example.com'])
    tx.commit()
except:
    tx.rollback()
    raise
```

### Transaction Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `execute(sql, params?)` | `int` | Execute DML statement |
| `query(sql, params?)` | `list[dict]` | Query rows as dicts |
| `query_one(sql, params?)` | `dict \| None` | Query single row |
| `query_raw(sql, params?)` | `dict` | Query in columnar format |
| `execute_batch(sql, params_list)` | `int` | Execute with multiple param sets |
| `commit()` | `None` | Commit the transaction |
| `rollback()` | `None` | Rollback the transaction |

### Batch in Transaction

```python
with db.begin() as tx:
    changes = tx.execute_batch(
        'INSERT INTO users VALUES ($1, $2, $3)',
        [
            [1, 'Alice', 'alice@example.com'],
            [2, 'Bob', 'bob@example.com'],
        ]
    )
    print(changes)  # 2
```

## Async API

Async wrappers use `asyncio.to_thread()` for non-blocking operations. All methods release the GIL.

```python
import asyncio
from stoolap import AsyncDatabase

async def main():
    db = await AsyncDatabase.open(':memory:')

    await db.exec("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT
        )
    """)

    await db.execute(
        'INSERT INTO users VALUES ($1, $2, $3)',
        [1, 'Alice', 'alice@example.com']
    )

    users = await db.query('SELECT * FROM users')
    print(users)

    await db.close()

asyncio.run(main())
```

### Async Transaction

```python
async with await db.begin() as tx:
    await tx.execute('INSERT INTO users VALUES ($1, $2, $3)', [1, 'Alice', 'alice@example.com'])
    await tx.execute('INSERT INTO users VALUES ($1, $2, $3)', [2, 'Bob', 'bob@example.com'])
    # Auto-commits on clean exit, auto-rollbacks on exception
```

### Async Prepared Statement

```python
stmt = db.prepare('SELECT * FROM users WHERE id = $1')
user = await stmt.query_one([1])
rows = await stmt.query([1])
raw = await stmt.query_raw([1])
```

### Async Methods

`AsyncDatabase`, `AsyncTransaction`, and `AsyncPreparedStatement` mirror all sync methods as coroutines with the same names, parameters, and return types. Just `await` the call. The only exception is `prepare()`, which is synchronous (no `await` needed).

## Parameters

Both positional and named parameters are supported across all methods:

```python
# Positional ($1, $2, ...)
db.query('SELECT * FROM users WHERE id = $1 AND name = $2', [1, 'Alice'])

# Named (:key)
db.query(
    'SELECT * FROM users WHERE id = :id AND name = :name',
    {'id': 1, 'name': 'Alice'}
)
```

Named parameter keys can include an optional prefix:

```python
# All equivalent
db.query('SELECT * FROM users WHERE id = :id', {'id': 1})
db.query('SELECT * FROM users WHERE id = :id', {':id': 1})
db.query('SELECT * FROM users WHERE id = :id', {'@id': 1})
db.query('SELECT * FROM users WHERE id = :id', {'$id': 1})
```

## Error Handling

All methods raise `StoolapError` on errors (invalid SQL, constraint violations, etc.):

```python
from stoolap import Database, StoolapError

try:
    db.execute('INSERT INTO users VALUES ($1, $2)', [1, None])  # NOT NULL violation
except StoolapError as e:
    print(f'Database error: {e}')

# Invalid SQL raises at prepare time
try:
    db.prepare('INVALID SQL HERE')
except StoolapError as e:
    print(f'Parse error: {e}')
```

`StoolapError` inherits from `RuntimeError`.

## Type Mapping

| Python | Stoolap | Notes |
|--------|---------|-------|
| `int` | `INTEGER` | 64-bit signed |
| `float` | `FLOAT` | 64-bit double |
| `str` | `TEXT` | UTF-8 encoded |
| `bool` | `BOOLEAN` | Checked before `int` (bool is a subclass of int in Python) |
| `None` | `NULL` | Any type |
| `datetime.datetime` | `TIMESTAMP` | Timezone-aware converted to UTC; naive treated as UTC |
| `dict` | `JSON` | Serialized via `json.dumps()` |
| `list` | `JSON` | Serialized via `json.dumps()` |

## Building from Source

Requires:
- [Rust](https://rustup.rs) (stable)
- [Python](https://python.org) >= 3.9
- [maturin](https://www.maturin.rs) (`pip install maturin`)

```bash
git clone https://github.com/stoolap/stoolap-python.git
cd stoolap-python
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest pytest-asyncio
maturin develop --release
pytest
```
