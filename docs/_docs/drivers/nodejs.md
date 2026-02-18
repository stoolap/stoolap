---
layout: doc
title: Node.js Driver
category: Drivers
order: 1
---

# Node.js Driver

High-performance Node.js driver for Stoolap. Built with [NAPI-RS](https://napi.rs) for native performance with both async and sync APIs.

## Installation

```bash
npm install @stoolap/node
```

Pre-built binaries are available for:
- macOS (x64, ARM64)
- Linux (x64, ARM64 GNU)
- Windows (x64 MSVC)

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

// Query in raw columnar format (faster)
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
| `execute(sql, params?)` | `Promise<RunResult>` | Execute DML statement |
| `exec(sql)` | `Promise<void>` | Execute one or more statements |
| `query(sql, params?)` | `Promise<Object[]>` | Query rows as objects |
| `queryOne(sql, params?)` | `Promise<Object \| null>` | Query single row |
| `queryRaw(sql, params?)` | `Promise<{columns, rows}>` | Query in columnar format |
| `begin()` | `Promise<Transaction>` | Begin a transaction |
| `close()` | `Promise<void>` | Close the database |

## Sync Methods

Sync methods run on the main thread -- faster for simple operations but block the event loop.

| Method | Returns | Description |
|--------|---------|-------------|
| `executeSync(sql, params?)` | `RunResult` | Execute DML statement |
| `execSync(sql)` | `void` | Execute one or more statements |
| `querySync(sql, params?)` | `Object[]` | Query rows as objects |
| `queryOneSync(sql, params?)` | `Object \| null` | Query single row |
| `queryRawSync(sql, params?)` | `{columns, rows}` | Query in columnar format |
| `executeBatchSync(sql, paramsArray)` | `RunResult` | Execute with multiple param sets |
| `beginSync()` | `Transaction` | Begin a transaction |
| `prepare(sql)` | `PreparedStatement` | Create a prepared statement |

`RunResult` is `{ changes: number }`:

```ts
import { Database, RunResult } from '@stoolap/node';
```

## Persistence

File-based databases persist data to disk using WAL (Write-Ahead Logging) and periodic snapshots. Data survives process restarts.

```js
const db = await Database.open('./mydata');

await db.exec('CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT)');
await db.execute('INSERT INTO kv VALUES ($1, $2)', ['hello', 'world']);
await db.close();

// Reopen -- data is still there
const db2 = await Database.open('./mydata');
const row = await db2.queryOne('SELECT * FROM kv WHERE key = $1', ['hello']);
// { key: 'hello', value: 'world' }
await db2.close();
```

### Configuration

Pass configuration as query parameters in the path:

```js
// Maximum durability -- fsync on every WAL write
const db = await Database.open('./mydata?sync=full');

// High throughput -- no fsync, larger buffers
const db = await Database.open('./mydata?sync=none&wal_buffer_size=131072');

// Custom snapshot interval with compression
const db = await Database.open('./mydata?snapshot_interval=60&compression=on');

// Multiple options
const db = await Database.open(
  './mydata?sync=full&snapshot_interval=120&keep_snapshots=10&wal_max_size=134217728'
);
```

### Sync Modes

Controls the durability vs. performance trade-off:

| Mode | Value | Description |
|------|-------|-------------|
| `none` | `sync=none` | No fsync -- fastest, data may be lost on crash |
| `normal` | `sync=normal` | Fsync on commit batches -- good balance (default) |
| `full` | `sync=full` | Fsync on every WAL write -- slowest, maximum durability |

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

Prepared statements parse SQL once and reuse the cached execution plan on every call -- no parsing or cache lookup overhead per execution.

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

Property: `sql` -- returns the SQL text of this prepared statement.

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

// Prepared statement -- errors at prepare time for invalid SQL
try {
  db.prepare('INVALID SQL HERE');
} catch (err) {
  console.error(err.message);
}
```

## Type Mapping

| JavaScript | Stoolap |
|-----------|---------|
| `number` (integer) | `INTEGER` |
| `number` (float) | `FLOAT` |
| `string` | `TEXT` |
| `boolean` | `BOOLEAN` |
| `null` / `undefined` | `NULL` |
| `BigInt` | `INTEGER` |
| `Date` | `TIMESTAMP` |
| `Buffer` | `TEXT` (UTF-8) |
| `Object` / `Array` | `JSON` (stringified) |

## Building from Source

Requires:
- [Rust](https://rustup.rs) (stable)
- [Node.js](https://nodejs.org) >= 22 (with development headers)
- C++ compiler (for V8 helpers)

```bash
git clone https://github.com/stoolap/stoolap-node.git
cd stoolap-node
npm install
npm run build
npm test
```
