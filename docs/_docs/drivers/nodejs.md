---
layout: doc
title: Node.js Driver
category: Drivers
order: 1
icon: nodejs
---

# Node.js Driver

High-performance Node.js driver for Stoolap. Built with a native N-API C addon for minimal overhead. Provides both async and sync APIs.

## Installation

```bash
npm install @stoolap/node
```

The stoolap engine shared library is pre-built for:
- macOS (x64, ARM64)
- Linux (x64, ARM64 GNU)
- Windows (x64 MSVC)

A C compiler is required to build the thin N-API addon on install (compiled automatically via `node-gyp`):
- **macOS**: `xcode-select --install`
- **Linux**: `sudo apt-get install build-essential` (or equivalent)
- **Windows**: [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with "Desktop development with C++"

## Quick Start

```js
// ESM
import { Database } from '@stoolap/node';

// CommonJS
const { Database } = require('@stoolap/node');
```

```js
const db = await Database.open(':memory:');

await db.exec(`
  CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT
  )
`);

// Insert with positional parameters ($1, $2, ...)
await db.execute(
  'INSERT INTO users (id, name, email) VALUES ($1, $2, $3)',
  [1, 'Alice', 'alice@example.com']
);

// Insert with named parameters (:key)
await db.execute(
  'INSERT INTO users (id, name, email) VALUES (:id, :name, :email)',
  { id: 2, name: 'Bob', email: 'bob@example.com' }
);

// Query rows as objects
const users = await db.query('SELECT * FROM users ORDER BY id');
// [{ id: 1, name: 'Alice', email: 'alice@example.com' }, ...]

// Query single row
const user = await db.queryOne('SELECT * FROM users WHERE id = $1', [1]);
// { id: 1, name: 'Alice', email: 'alice@example.com' }

// Query in raw columnar format (faster, no per-row object creation)
const raw = await db.queryRaw('SELECT id, name FROM users ORDER BY id');
// { columns: ['id', 'name'], rows: [[1, 'Alice'], [2, 'Bob']] }

await db.close();
```

## Opening a Database

```js
// In-memory
const db = await Database.open(':memory:');
const db = await Database.open('');
const db = await Database.open('memory://');

// File-based (data persists across restarts)
const db = await Database.open('./mydata');
const db = await Database.open('file:///absolute/path/to/db');
```

## Async Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `Database.open(path)` | `Promise<Database>` | Open a database |
| `clone()` | `Database` | Clone handle (shared engine, own state) |
| `execute(sql, params?)` | `Promise<RunResult>` | Execute DML statement |
| `exec(sql)` | `Promise<void>` | Execute a DDL statement |
| `query(sql, params?)` | `Promise<Object[]>` | Query rows as objects |
| `queryOne(sql, params?)` | `Promise<Object \| null>` | Query single row |
| `queryRaw(sql, params?)` | `Promise<{columns, rows}>` | Query in columnar format |
| `begin()` | `Promise<Transaction>` | Begin a transaction |
| `close()` | `Promise<void>` | Close the database |

## Sync Methods

Sync methods run on the main thread. Faster for simple operations but blocks the event loop.

| Method | Returns | Description |
|--------|---------|-------------|
| `Database.openSync(path)` | `Database` | Open a database |
| `clone()` | `Database` | Clone handle (shared engine, own state) |
| `executeSync(sql, params?)` | `RunResult` | Execute DML statement |
| `execSync(sql)` | `void` | Execute a DDL statement |
| `querySync(sql, params?)` | `Object[]` | Query rows as objects |
| `queryOneSync(sql, params?)` | `Object \| null` | Query single row |
| `queryRawSync(sql, params?)` | `{columns, rows}` | Query in columnar format |
| `executeBatchSync(sql, paramsArray)` | `RunResult` | Execute with multiple param sets |
| `beginSync()` | `Transaction` | Begin a transaction |
| `prepare(sql)` | `PreparedStatement` | Create a prepared statement |
| `closeSync()` | `void` | Close the database |

`RunResult` is `{ changes: number }`. It can be imported as a type:

```ts
import { Database, RunResult } from '@stoolap/node';
```

## Persistence

File-based databases persist data to disk using WAL (Write-Ahead Logging) and immutable cold volumes. A background checkpoint cycle seals hot rows into columnar volume files, compacts them, and truncates the WAL. Data survives process restarts.

```js
const db = await Database.open('./mydata');

await db.exec('CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT)');
await db.execute('INSERT INTO kv VALUES ($1, $2)', ['hello', 'world']);
await db.close();

// Reopen: data is still there
const db2 = await Database.open('./mydata');
const row = await db2.queryOne('SELECT * FROM kv WHERE key = $1', ['hello']);
// { key: 'hello', value: 'world' }
await db2.close();
```

### Configuration

Pass configuration as query parameters in the path:

```js
// Maximum durability: fsync on every WAL write
const db = await Database.open('./mydata?sync=full');

// High throughput: no fsync, larger buffers
const db = await Database.open('./mydata?sync=none&wal_buffer_size=131072');

// Custom checkpoint interval with compression
const db = await Database.open('./mydata?checkpoint_interval=60&compression=on');

// Multiple options
const db = await Database.open(
  './mydata?sync=full&checkpoint_interval=120&compact_threshold=8&wal_max_size=134217728'
);
```

### Sync Modes

Controls the durability vs. performance trade-off:

| Mode | Value | Description |
|------|-------|-------------|
| `none` | `sync=none` | No fsync. Fastest, data may be lost on crash |
| `normal` | `sync=normal` | Fsync on commit batches. Good balance (default) |
| `full` | `sync=full` | Fsync on every WAL write. Slowest, maximum durability |

### All Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sync` | `normal` | Sync mode: `none`, `normal`, or `full` |
| `checkpoint_interval` | `60` | Seconds between automatic checkpoint cycles |
| `compact_threshold` | `4` | Sub-target volumes per table before merging |
| `keep_snapshots` | `3` | Backup snapshots to retain per table |
| `wal_flush_trigger` | `32768` | WAL flush trigger size in bytes (32 KB) |
| `wal_buffer_size` | `65536` | WAL buffer size in bytes (64 KB) |
| `wal_max_size` | `67108864` | Max WAL file size before rotation (64 MB) |
| `commit_batch_size` | `100` | Commits to batch before syncing (normal mode) |
| `sync_interval_ms` | `1000` | Minimum ms between syncs (normal mode) |
| `wal_compression` | `on` | LZ4 compression for WAL entries |
| `volume_compression` | `on` | LZ4 compression for cold volume files |
| `compression` | | Alias that sets both `wal_compression` and `volume_compression` |
| `compression_threshold` | `64` | Minimum bytes before compressing an entry |
| `checkpoint_on_close` | `on` | Seal all hot rows to volumes on clean shutdown |
| `target_volume_rows` | `1048576` | Target rows per cold volume (min 65536) |

## Cloning

`clone()` creates a new `Database` handle that shares the same underlying engine (data, indexes, transactions) but has its own executor and error state. Useful for concurrent access patterns such as worker threads.

```js
const db = await Database.open('./mydata');
const db2 = db.clone();

// Both see the same data
await db.execute('INSERT INTO users VALUES ($1, $2)', [1, 'Alice']);
const row = db2.queryOneSync('SELECT * FROM users WHERE id = $1', [1]);
// { id: 1, name: 'Alice' }

// Each clone must be closed independently
await db2.close();
await db.close();
```

## Raw Query Format

`queryRaw` / `queryRawSync` return `{ columns: string[], rows: any[][] }` instead of an array of objects. Faster when you don't need named keys.

```js
const raw = db.queryRawSync('SELECT id, name, email FROM users ORDER BY id');
console.log(raw.columns); // ['id', 'name', 'email']
console.log(raw.rows);    // [[1, 'Alice', 'alice@example.com'], [2, 'Bob', 'bob@example.com']]
```

## Batch Execution

Execute the same SQL with multiple parameter sets in a single call. Automatically wraps in a transaction.

```js
const result = db.executeBatchSync(
  'INSERT INTO users VALUES ($1, $2, $3)',
  [
    [1, 'Alice', 'alice@example.com'],
    [2, 'Bob', 'bob@example.com'],
    [3, 'Charlie', 'charlie@example.com'],
  ]
);
console.log(result.changes); // 3
```

## Prepared Statements

Prepared statements parse SQL once and reuse the cached execution plan on every call. No parsing or cache lookup overhead per execution.

```js
const insert = db.prepare('INSERT INTO users VALUES ($1, $2, $3)');
insert.executeSync([1, 'Alice', 'alice@example.com']);
insert.executeSync([2, 'Bob', 'bob@example.com']);

const lookup = db.prepare('SELECT * FROM users WHERE id = $1');
const user = lookup.queryOneSync([1]);
// { id: 1, name: 'Alice', email: 'alice@example.com' }
```

### Methods

All methods mirror `Database` but without the `sql` parameter (it's bound at prepare time).

| Async | Sync | Description |
|-------|------|-------------|
| `execute(params?)` | `executeSync(params?)` | Execute DML statement |
| `query(params?)` | `querySync(params?)` | Query rows as objects |
| `queryOne(params?)` | `queryOneSync(params?)` | Query single row |
| `queryRaw(params?)` | `queryRawSync(params?)` | Query in columnar format |
| | `executeBatchSync(paramsArray)` | Execute with multiple param sets |
| | `finalize()` | Release the prepared statement |

Property: `sql` returns the SQL text of this prepared statement.

### Async Usage

```js
const stmt = db.prepare('SELECT * FROM users WHERE id = $1');

const rows = await stmt.query([1]);
const one = await stmt.queryOne([1]);
const raw = await stmt.queryRaw([1]);
const result = await stmt.execute([1]); // for DML
```

### Sync Usage

```js
const stmt = db.prepare('SELECT * FROM users WHERE id = $1');

const rows = stmt.querySync([1]);
const one = stmt.queryOneSync([1]);
const raw = stmt.queryRawSync([1]);
const result = stmt.executeSync([1]); // for DML
```

### Batch with Prepared Statement

```js
const insert = db.prepare('INSERT INTO users VALUES ($1, $2, $3)');
const result = insert.executeBatchSync([
  [1, 'Alice', 'alice@example.com'],
  [2, 'Bob', 'bob@example.com'],
  [3, 'Charlie', 'charlie@example.com'],
]);
console.log(result.changes); // 3
```

## Transactions

### Async Transaction

```js
const tx = await db.begin();
try {
  await tx.execute('INSERT INTO users VALUES ($1, $2, $3)', [1, 'Alice', 'alice@example.com']);
  await tx.execute('INSERT INTO users VALUES ($1, $2, $3)', [2, 'Bob', 'bob@example.com']);

  // Read within the transaction (sees uncommitted changes)
  const rows = await tx.query('SELECT * FROM users');
  const one = await tx.queryOne('SELECT * FROM users WHERE id = $1', [1]);
  const raw = await tx.queryRaw('SELECT id, name FROM users');

  await tx.commit();
} catch (e) {
  await tx.rollback();
  throw e;
}
```

### Sync Transaction

```js
const tx = db.beginSync();
try {
  tx.executeSync('INSERT INTO users VALUES ($1, $2, $3)', [1, 'Alice', 'alice@example.com']);
  tx.executeSync('INSERT INTO users VALUES ($1, $2, $3)', [2, 'Bob', 'bob@example.com']);

  const rows = tx.querySync('SELECT * FROM users');
  const one = tx.queryOneSync('SELECT * FROM users WHERE id = $1', [1]);
  const raw = tx.queryRawSync('SELECT id, name FROM users');

  tx.commitSync();
} catch (e) {
  tx.rollbackSync();
  throw e;
}
```

### Transaction Methods

| Async | Sync | Description |
|-------|------|-------------|
| `execute(sql, params?)` | `executeSync(sql, params?)` | Execute DML statement |
| `query(sql, params?)` | `querySync(sql, params?)` | Query rows as objects |
| `queryOne(sql, params?)` | `queryOneSync(sql, params?)` | Query single row |
| `queryRaw(sql, params?)` | `queryRawSync(sql, params?)` | Query in columnar format |
| `commit()` | `commitSync()` | Commit the transaction |
| `rollback()` | `rollbackSync()` | Rollback the transaction |
| | `executeBatchSync(sql, paramsArray)` | Execute with multiple param sets |

### Batch in Transaction

```js
const tx = db.beginSync();
const result = tx.executeBatchSync(
  'INSERT INTO users VALUES ($1, $2, $3)',
  [
    [1, 'Alice', 'alice@example.com'],
    [2, 'Bob', 'bob@example.com'],
  ]
);
tx.commitSync();
console.log(result.changes); // 2
```

## Parameters

Both positional and named parameters are supported across all methods:

```js
// Positional ($1, $2, ...)
db.querySync('SELECT * FROM users WHERE id = $1 AND name = $2', [1, 'Alice']);

// Named (:key)
db.querySync(
  'SELECT * FROM users WHERE id = :id AND name = :name',
  { id: 1, name: 'Alice' }
);
```

## Error Handling

All methods throw on errors (invalid SQL, constraint violations, etc.):

```js
// Async
try {
  await db.execute('INSERT INTO users VALUES ($1, $2)', [1, null]); // NOT NULL violation
} catch (err) {
  console.error(err.message);
}

// Sync
try {
  db.executeSync('SELECTX * FROM users'); // syntax error
} catch (err) {
  console.error(err.message);
}

// Invalid SQL raises at prepare time
try {
  db.prepare('INVALID SQL HERE');
} catch (err) {
  console.error(err.message);
}
```

## Type Mapping

| JavaScript | Stoolap | Notes |
|-----------|---------|-------|
| `number` (integer) | `INTEGER` | |
| `number` (float) | `FLOAT` | |
| `string` | `TEXT` | |
| `boolean` | `BOOLEAN` | |
| `null` / `undefined` | `NULL` | |
| `BigInt` | `INTEGER` | |
| `Date` | `TIMESTAMP` | |
| `Float32Array` | `VECTOR(N)` | Returned as `Float32Array` |
| `Buffer` | `TEXT` (UTF-8) | |
| `Object` / `Array` | `JSON` (stringified) | |

## Vector Support

Stoolap supports native vector storage and similarity search. Vectors are returned as `Float32Array` and can be passed as `Float32Array` bind parameters.

```js
// Create a table with a vector column
await db.exec('CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(3))');

// Insert vectors via SQL string literals
await db.execute("INSERT INTO embeddings VALUES (1, '[0.1, 0.2, 0.3]')");

// Query: vectors are returned as Float32Array
const row = await db.queryOne('SELECT vec FROM embeddings WHERE id = 1');
console.log(row.vec);              // Float32Array(3) [0.1, 0.2, 0.3]
console.log(row.vec instanceof Float32Array); // true

// k-NN search with distance functions
const nearest = await db.query(`
  SELECT id, VEC_DISTANCE_L2(vec, '[0.15, 0.25, 0.35]') AS dist
  FROM embeddings ORDER BY dist LIMIT 5
`);

// HNSW index for fast approximate nearest neighbor search
await db.exec('CREATE INDEX idx ON embeddings(vec) USING HNSW');
```

Available distance functions: `VEC_DISTANCE_L2`, `VEC_DISTANCE_COSINE`, `VEC_DISTANCE_IP`.

See [Vector Search]({% link _docs/data-types/vector-search.md %}) for full details on HNSW indexes, distance metrics, and configuration.

## Building from Source

Requires:
- [Node.js](https://nodejs.org) >= 18
- C compiler (gcc, clang, or MSVC)
- [node-gyp](https://github.com/nodejs/node-gyp) and its prerequisites

The stoolap shared library (`libstoolap.dylib` / `libstoolap.so` / `stoolap.dll`) must be available, either via a platform package or built from the [Stoolap](https://github.com/stoolap/stoolap) repository.

```bash
git clone https://github.com/stoolap/stoolap-node.git
cd stoolap-node
npm install
npm test
```
