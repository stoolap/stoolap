---
layout: doc
title: WebAssembly (WASM)
category: Drivers
order: 3
---

# WebAssembly (WASM)

Run Stoolap entirely in the browser or any WebAssembly runtime. The full SQL engine compiles to a single `.wasm` module with no server, no network requests, and no native dependencies.

**Try it now**: [Playground]({{ site.baseurl }}/playground)

## Quick Start

### Browser (ES Module)

Copy `stoolap.js`, `stoolap_bg.wasm`, and `stoolap.d.ts` to your static assets directory. `stoolap.js` and `stoolap_bg.wasm` must be served from the same path because the init function resolves the `.wasm` binary relative to `stoolap.js` via `import.meta.url`.

```html
<script type="module">
  // Import the WASM module and the init function
  const wasm = await import('./assets/wasm/stoolap.js');

  // Initialize the WASM binary (fetches and compiles stoolap_bg.wasm)
  await wasm.default();

  // Create a database instance
  const db = new wasm.StoolapDB();

  db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)");
  db.execute("INSERT INTO users VALUES (1, 'Alice')");

  const result = JSON.parse(db.execute("SELECT * FROM users"));
  console.log(result.columns); // ["id", "name"]
  console.log(result.rows);    // [[1, "Alice"]]
</script>
```

### Dynamic Import (Recommended for SPAs)

Load the WASM module lazily to avoid blocking page load:

```javascript
let db = null;

async function initDatabase() {
  const wasm = await import('/assets/wasm/stoolap.js');
  await wasm.default();
  db = new wasm.StoolapDB();
}

// Call once at startup
await initDatabase();

// Then use db.execute() anywhere
const result = JSON.parse(db.execute("SELECT 1 + 1 AS answer"));
console.log(result.rows[0][0]); // 2
```

## API Reference

### `StoolapDB` Class

```typescript
class StoolapDB {
  constructor();
  execute(sql: string): string;
  execute_batch(sql: string): string;
  version(): string;
  free(): void;
}
```

#### `new StoolapDB()`

Creates a new in-memory database instance. Each instance is fully independent. The WASM module must be initialized first by calling the default export.

```javascript
const wasm = await import('./stoolap.js');
await wasm.default();
const db = new wasm.StoolapDB();
```

#### `execute(sql: string): string`

Executes a single SQL statement and returns a JSON string. Transaction commands (`BEGIN`, `COMMIT`, `ROLLBACK`) are handled automatically.

```javascript
const result = JSON.parse(db.execute("SELECT * FROM users WHERE id = 1"));
```

#### `execute_batch(sql: string): string`

Executes multiple semicolon-separated SQL statements. Handles quoted strings and comments correctly. Returns the result of the last statement, or stops on the first error.

```javascript
db.execute_batch(`
  CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT);
  INSERT INTO t VALUES (1, 'hello');
  INSERT INTO t VALUES (2, 'world');
`);
```

#### `version(): string`

Returns the Stoolap version string.

```javascript
console.log(db.version()); // "0.3.3"
```

#### `free()`

Releases the WASM memory associated with this database instance. The instance must not be used after calling `free()`. Also available as `Symbol.dispose` for use with `using` declarations.

### Response Format

`execute()` and `execute_batch()` return JSON strings with one of three shapes:

**Query result** (SELECT, SHOW, DESCRIBE, EXPLAIN, PRAGMA, VACUUM):

```json
{
  "type": "rows",
  "columns": ["id", "name", "email"],
  "rows": [[1, "Alice", "alice@example.com"], [2, "Bob", "bob@example.com"]],
  "count": 2
}
```

**Write result** (INSERT, UPDATE, DELETE, CREATE, DROP):

```json
{
  "type": "affected",
  "affected": 3
}
```

**Error**:

```json
{
  "type": "error",
  "message": "table 'users' does not exist"
}
```

### Value Types

| SQL Type | JSON Representation |
|----------|-------------------|
| NULL | `null` |
| BOOLEAN | `true` / `false` |
| INTEGER | number |
| FLOAT | number (`NaN`/`Infinity` become `null`) |
| TEXT | string |
| TIMESTAMP | ISO 8601 string (e.g., `"2024-01-15T10:30:00Z"`) |
| JSON | string (JSON text) |
| VECTOR | string (e.g., `"[0.1, 0.2, 0.3]"`) |

## Transactions

```javascript
db.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance FLOAT)");
db.execute("INSERT INTO accounts VALUES (1, 1000.0)");
db.execute("INSERT INTO accounts VALUES (2, 500.0)");

db.execute("BEGIN");
db.execute("UPDATE accounts SET balance = balance - 100 WHERE id = 1");
db.execute("UPDATE accounts SET balance = balance + 100 WHERE id = 2");
db.execute("COMMIT");

// Or rollback on error
db.execute("BEGIN");
db.execute("UPDATE accounts SET balance = balance - 9999 WHERE id = 1");
db.execute("ROLLBACK");
```

Reads within a transaction see uncommitted changes. Writes are atomic on `COMMIT`.

## What Works in WASM

The WASM build includes the full SQL engine:

- All SQL features: joins, subqueries, CTEs (including recursive), window functions, aggregations (ROLLUP, CUBE, GROUPING SETS), set operations
- All data types: INTEGER, FLOAT, TEXT, BOOLEAN, TIMESTAMP, JSON, VECTOR
- All 117+ built-in functions (string, math, date/time, JSON, vector, aggregate, window)
- MVCC transactions with snapshot isolation
- Cost-based query optimizer with EXPLAIN/EXPLAIN ANALYZE
- All index types: B-tree, Hash, Bitmap, multi-column, HNSW
- Semantic query caching
- AS OF time-travel queries

## WASM Limitations

| Feature | Status | Notes |
|---------|--------|-------|
| File persistence | Not available | In-memory only; data is lost on page reload |
| Background threads | Not available | No parallel query execution; no automatic cleanup |
| Cleanup | Manual only | Use `VACUUM` or `PRAGMA vacuum` for maintenance |
| WAL / Snapshots | Not available | No crash recovery needed (in-memory) |

Since the background cleanup thread is unavailable in WASM, run `VACUUM` periodically in long-running sessions to reclaim memory from deleted rows:

```javascript
// After bulk deletes or updates, reclaim space
const result = JSON.parse(db.execute("VACUUM"));
console.log(result.columns); // ["deleted_rows_cleaned", "old_versions_cleaned", "transactions_cleaned"]
console.log(result.rows[0]); // [42, 15, 3]
```

## Building from Source

### Prerequisites

- [Rust](https://rustup.rs/) (stable)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)

### Build

```bash
wasm-pack build --target web --out-dir pkg -- --no-default-features --features wasm
```

The `--no-default-features --features wasm` flags disable native-only features (file I/O, parallel execution) and enable the WASM bindings.

This produces:

| File | Description |
|------|-------------|
| `stoolap_bg.wasm` | Binary WASM module (~4 MB, ~1 MB with gzip/brotli) |
| `stoolap.js` | JavaScript bindings (ES module) |
| `stoolap.d.ts` | TypeScript type definitions |
| `package.json` | npm package metadata |

### Serving

Copy the output files to your web server or static assets directory. Both `stoolap.js` and `stoolap_bg.wasm` must be served from the same path:

```
your-app/
  assets/
    wasm/
      stoolap.js
      stoolap_bg.wasm
      stoolap.d.ts
```

### Build Targets

| Target | Use case |
|--------|----------|
| `--target web` | Direct `<script type="module">` or dynamic `import()` |
| `--target bundler` | Vite, Webpack, Rollup (bundler must support WASM) |
| `--target nodejs` | Node.js (the native driver is recommended instead) |

## Examples

### Create a Searchable Table

```javascript
const wasm = await import('/assets/wasm/stoolap.js');
await wasm.default();
const db = new wasm.StoolapDB();

db.execute_batch(`
  CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    price FLOAT
  );
  CREATE INDEX idx_category ON products(category);
  INSERT INTO products VALUES (1, 'Laptop', 'Electronics', 999.99);
  INSERT INTO products VALUES (2, 'Mouse', 'Electronics', 29.99);
  INSERT INTO products VALUES (3, 'Desk', 'Furniture', 249.99);
`);

const result = JSON.parse(db.execute(
  "SELECT name, price FROM products WHERE category = 'Electronics' ORDER BY price"
));
// result.rows = [["Mouse", 29.99], ["Laptop", 999.99]]
```

### Analytics Query

```javascript
const result = JSON.parse(db.execute(`
  SELECT
    category,
    COUNT(*) AS cnt,
    ROUND(AVG(price), 2) AS avg_price
  FROM products
  GROUP BY category
  ORDER BY avg_price DESC
`));
console.table(result.rows);
```

### Vector Search in the Browser

```javascript
db.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, title TEXT, emb VECTOR(3))");
db.execute("CREATE INDEX idx_emb ON docs(emb) USING HNSW");

db.execute("INSERT INTO docs VALUES (1, 'Rust programming', '[1.0, 0.0, 0.0]')");
db.execute("INSERT INTO docs VALUES (2, 'Web development', '[0.0, 1.0, 0.0]')");
db.execute("INSERT INTO docs VALUES (3, 'Database internals', '[0.7, 0.0, 0.7]')");

const result = JSON.parse(db.execute(`
  SELECT title, VEC_DISTANCE_L2(emb, '[1.0, 0.0, 0.0]') AS dist
  FROM docs
  ORDER BY dist
  LIMIT 2
`));
// Nearest to [1,0,0]: "Rust programming" (0.0), "Database internals" (0.58)
```

### Helper Wrapper

For convenience, wrap the JSON parsing:

```javascript
function query(db, sql) {
  const result = JSON.parse(db.execute(sql));
  if (result.type === 'error') throw new Error(result.message);
  return result;
}

function exec(db, sql) {
  const result = JSON.parse(db.execute(sql));
  if (result.type === 'error') throw new Error(result.message);
  return result.affected ?? result.count;
}

// Usage
const { columns, rows } = query(db, "SELECT * FROM users");
exec(db, "DELETE FROM users WHERE id = 99");
```
