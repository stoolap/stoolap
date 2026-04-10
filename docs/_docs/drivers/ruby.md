---
layout: doc
title: Ruby Driver
category: Drivers
order: 7
icon: ruby
---

# Ruby Driver

High-performance Ruby driver for Stoolap. Built with [Magnus](https://github.com/matsadler/magnus) + [rb-sys](https://github.com/oxidize-rb/rb-sys) for direct Rust bindings with no FFI overhead, so every call goes from Ruby straight into Stoolap's Rust engine.

## Installation

```bash
gem install stoolap
```

Or add it to your `Gemfile`:

```ruby
gem "stoolap"
```

Requires Ruby `>= 3.3` and a stable [Rust toolchain](https://rustup.rs) at install time. The native extension is compiled from source on your machine via rake-compiler + rb-sys.

## Quick Start

```ruby
require "stoolap"

db = Stoolap::Database.open(":memory:")

db.exec(<<~SQL)
  CREATE TABLE users (
    id    INTEGER PRIMARY KEY,
    name  TEXT NOT NULL,
    email TEXT
  )
SQL

# Insert with positional parameters ($1, $2, ...)
db.execute(
  "INSERT INTO users (id, name, email) VALUES ($1, $2, $3)",
  [1, "Alice", "alice@example.com"]
)

# Insert with named parameters (:key)
db.execute(
  "INSERT INTO users (id, name, email) VALUES (:id, :name, :email)",
  { id: 2, name: "Bob", email: "bob@example.com" }
)

# Query rows as an Array of Hashes (String keys)
users = db.query("SELECT * FROM users ORDER BY id")
# => [{"id" => 1, "name" => "Alice", "email" => "alice@example.com"}, ...]

# Query a single row
user = db.query_one("SELECT * FROM users WHERE id = $1", [1])
# => {"id" => 1, "name" => "Alice", "email" => "alice@example.com"}

# Query in raw columnar format (no per-row Hash allocation)
raw = db.query_raw("SELECT id, name FROM users ORDER BY id")
# => {"columns" => ["id", "name"], "rows" => [[1, "Alice"], [2, "Bob"]]}

db.close
```

### Block form auto-closes

```ruby
Stoolap::Database.open(":memory:") do |db|
  db.exec("CREATE TABLE t (id INTEGER PRIMARY KEY)")
  db.execute("INSERT INTO t VALUES ($1)", [1])
end
# db.close already ran, even if the block raised
```

## Opening a Database

```ruby
# In-memory
Stoolap::Database.open(":memory:")
Stoolap::Database.open("")
Stoolap::Database.open("memory://")

# File-based (data persists across restarts)
Stoolap::Database.open("./mydata")
Stoolap::Database.open("file:///absolute/path/to/db")
```

Opening the same DSN twice in one process returns the same engine. Closing one handle closes the engine for all handles, so open a database once per process and pass the instance around.

## Methods

| Method | Returns | Description |
|---|---|---|
| `execute(sql, params = nil)` | `Integer` | Execute DML, return rows affected |
| `exec(sql)` | `nil` | Execute one or more statements (no parameters) |
| `query(sql, params = nil)` | `Array<Hash>` | All rows as Array of Hashes |
| `query_one(sql, params = nil)` | `Hash, nil` | First row as a Hash, or `nil` |
| `query_raw(sql, params = nil)` | `Hash` | `{"columns" => [...], "rows" => [[...], ...]}` |
| `execute_batch(sql, params_list)` | `Integer` | Same SQL, many param sets, auto-tx |
| `prepare(sql)` | `PreparedStatement` | Cache a parsed + planned statement |
| `begin_transaction` | `Transaction` | Start a manual transaction |
| `transaction { \|tx\| ... }` | block return | Auto-commit on clean exit, rollback on raise |
| `close` | `nil` | Close the database |

## Persistence

File-based databases persist data via Write-Ahead Logging and immutable columnar cold volumes. A background checkpoint cycle seals hot rows into volume files, compacts them, and truncates the WAL. Data survives process restarts.

```ruby
Stoolap::Database.open("./mydata") do |db|
  db.exec("CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT)")
  db.execute("INSERT INTO kv VALUES ($1, $2)", ["hello", "world"])
end

# Reopen: data is still there
Stoolap::Database.open("./mydata") do |db|
  db.query_one("SELECT value FROM kv WHERE key = $1", ["hello"])
  # => {"value" => "world"}
end
```

### Configuration

Pass configuration as query parameters in the path:

```ruby
# Maximum durability (fsync on every WAL write)
Stoolap::Database.open("./mydata?sync=full")

# High throughput (no fsync, larger buffers)
Stoolap::Database.open("./mydata?sync=none&wal_buffer_size=131072")

# Custom checkpoint interval with compression
Stoolap::Database.open("./mydata?checkpoint_interval=60&compression=on")

# Multiple options
Stoolap::Database.open(
  "./mydata?sync=full&checkpoint_interval=120&compact_threshold=8&wal_max_size=134217728"
)
```

### Sync Modes

Controls the durability vs. performance trade-off:

| Mode | Value | Description |
|---|---|---|
| `none` | `sync=none` | No fsync. Fastest, data may be lost on crash |
| `normal` | `sync=normal` | Fsync on commit batches. Good balance (default) |
| `full` | `sync=full` | Fsync on every WAL write. Slowest, maximum durability |

### All Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `sync` | `normal` | Sync mode: `none`, `normal`, or `full` |
| `checkpoint_interval` | `60` | Seconds between automatic checkpoint cycles |
| `compact_threshold` | `4` | Sub-target volumes per table before merging |
| `keep_snapshots` | `3` | Backup snapshots retained per table |
| `wal_flush_trigger` | `32768` | WAL flush trigger size in bytes (32 KB) |
| `wal_buffer_size` | `65536` | WAL buffer size in bytes (64 KB) |
| `wal_max_size` | `67108864` | Max WAL file size before rotation (64 MB) |
| `commit_batch_size` | `100` | Commits to batch before syncing (normal mode) |
| `sync_interval_ms` | `1000` | Minimum ms between syncs (normal mode) |
| `wal_compression` | `on` | LZ4 compression for WAL entries |
| `volume_compression` | `on` | LZ4 compression for cold volume files |
| `compression` | -- | Alias that sets both `wal_compression` and `volume_compression` |
| `compression_threshold` | `64` | Minimum bytes before compressing an entry |
| `checkpoint_on_close` | `on` | Seal all hot rows to volumes on clean shutdown |
| `target_volume_rows` | `1048576` | Target rows per cold volume (min 65536) |

## Raw Query Format

`query_raw` returns `{"columns" => [...], "rows" => [[...], ...]}` instead of an Array of Hashes. Faster when you do not need named access to each row.

```ruby
raw = db.query_raw("SELECT id, name, email FROM users ORDER BY id")
raw["columns"]  # => ["id", "name", "email"]
raw["rows"]     # => [[1, "Alice", "alice@example.com"], [2, "Bob", "bob@example.com"]]
```

## Batch Execution

Execute the same SQL with multiple parameter sets in a single call. Automatically wraps in a transaction.

```ruby
changes = db.execute_batch(
  "INSERT INTO users VALUES ($1, $2, $3)",
  [
    [1, "Alice",   "alice@example.com"],
    [2, "Bob",     "bob@example.com"],
    [3, "Charlie", "charlie@example.com"]
  ]
)
# changes => 3
```

`execute_batch` only supports positional parameters (Arrays). Hash parameter sets raise `Stoolap::Error`.

## Prepared Statements

Prepared statements parse SQL once, cache the execution plan, and also cache the column-key `String`s on the first query so repeated calls do not re-allocate keys.

```ruby
insert = db.prepare("INSERT INTO users VALUES ($1, $2, $3)")
insert.execute([1, "Alice", "alice@example.com"])
insert.execute([2, "Bob",   "bob@example.com"])

lookup = db.prepare("SELECT * FROM users WHERE id = $1")
user = lookup.query_one([1])
# => {"id" => 1, "name" => "Alice", "email" => "alice@example.com"}
```

### Methods

All methods mirror `Database` but without the `sql` parameter (it is bound at prepare time).

| Method | Returns | Description |
|---|---|---|
| `execute(params = nil)` | `Integer` | Execute DML |
| `query(params = nil)` | `Array<Hash>` | All rows |
| `query_one(params = nil)` | `Hash, nil` | First row or nil |
| `query_raw(params = nil)` | `Hash` | Columnar format |
| `execute_batch(params_list)` | `Integer` | Many param sets, auto-tx |
| `sql` | `String` | The SQL text this statement was built from |

### Batch with Prepared Statement

```ruby
insert = db.prepare("INSERT INTO users VALUES ($1, $2, $3)")
changes = insert.execute_batch([
  [1, "Alice",   "alice@example.com"],
  [2, "Bob",     "bob@example.com"],
  [3, "Charlie", "charlie@example.com"]
])
# changes => 3
```

## Transactions

### Block form (recommended)

```ruby
db.transaction do |tx|
  tx.execute("INSERT INTO users VALUES ($1, $2, $3)", [1, "Alice", "alice@example.com"])
  tx.execute("INSERT INTO users VALUES ($1, $2, $3)", [2, "Bob",   "bob@example.com"])

  # Reads within the tx see its own uncommitted writes
  rows = tx.query("SELECT * FROM users")
  one  = tx.query_one("SELECT * FROM users WHERE id = $1", [1])
  raw  = tx.query_raw("SELECT id, name FROM users")
end
# commit on clean exit, rollback on any raise
```

The block form returns whatever the block returns:

```ruby
count = db.transaction do |tx|
  tx.query_one("SELECT COUNT(*) AS c FROM users")["c"]
end
```

### Manual control

```ruby
tx = db.begin_transaction
begin
  tx.execute("INSERT INTO users VALUES ($1, $2, $3)", [1, "Alice", "alice@example.com"])
  tx.execute("INSERT INTO users VALUES ($1, $2, $3)", [2, "Bob",   "bob@example.com"])
  tx.commit
rescue StandardError
  tx.rollback
  raise
end
```

Or `Transaction#with_rollback` for the same semantics on an existing handle:

```ruby
tx = db.begin_transaction
tx.with_rollback do |t|
  t.execute("INSERT INTO users VALUES ($1, $2, $3)", [1, "Alice", "alice@example.com"])
end
# commit on success, rollback on raise
```

### Transaction Methods

| Method | Returns | Description |
|---|---|---|
| `execute(sql, params = nil)` | `Integer` | Execute DML |
| `query(sql, params = nil)` | `Array<Hash>` | All rows |
| `query_one(sql, params = nil)` | `Hash, nil` | First row or nil |
| `query_raw(sql, params = nil)` | `Hash` | Columnar format |
| `execute_batch(sql, params_list)` | `Integer` | Many param sets |
| `execute_prepared(stmt, params = nil)` | `Integer` | Run a `PreparedStatement` inside this tx |
| `query_prepared(stmt, params = nil)` | `Array<Hash>` | Query a `PreparedStatement` inside this tx |
| `query_one_prepared(stmt, params = nil)` | `Hash, nil` | First row from a prepared query |
| `query_raw_prepared(stmt, params = nil)` | `Hash` | Columnar result from a prepared query |
| `commit` | `nil` | Commit the transaction |
| `rollback` | `nil` | Roll back the transaction |
| `with_rollback { \|tx\| ... }` | block return | Commit on clean exit, rollback on raise |

Calling `execute` / `query` / `commit` / `rollback` on a committed or rolled-back transaction raises `Stoolap::Error`. DDL statements (`CREATE TABLE`, etc.) are not allowed inside explicit transactions; run them outside.

### Batch in Transaction

```ruby
db.transaction do |tx|
  changes = tx.execute_batch(
    "INSERT INTO users VALUES ($1, $2, $3)",
    [
      [1, "Alice", "alice@example.com"],
      [2, "Bob",   "bob@example.com"]
    ]
  )
  # changes => 2
end
```

## Parameters

Both positional and named parameters are supported across all methods:

```ruby
# Positional ($1, $2, ...)
db.query("SELECT * FROM users WHERE id = $1 AND name = $2", [1, "Alice"])

# Named (:key) with Symbol keys
db.query(
  "SELECT * FROM users WHERE id = :id AND name = :name",
  { id: 1, name: "Alice" }
)

# Named with String keys
db.query(
  "SELECT * FROM users WHERE id = :id",
  { "id" => 1 }
)
```

Named-parameter keys can include an optional prefix sigil. The driver strips it before binding:

```ruby
# All equivalent
db.query("SELECT * FROM users WHERE id = :id", { id: 1 })
db.query("SELECT * FROM users WHERE id = :id", { ":id" => 1 })
db.query("SELECT * FROM users WHERE id = :id", { "@id" => 1 })
db.query("SELECT * FROM users WHERE id = :id", { "$id" => 1 })
```

## Error Handling

All database errors raise `Stoolap::Error` (a subclass of `StandardError`):

```ruby
require "stoolap"

db = Stoolap::Database.open(":memory:")
db.exec("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")

begin
  db.execute("INSERT INTO users VALUES ($1, $2)", [1, nil])  # NOT NULL violation
rescue Stoolap::Error => e
  warn "database error: #{e.message}"
end

# Invalid SQL raises at prepare time
begin
  db.prepare("INVALID SQL HERE")
rescue Stoolap::Error => e
  warn "parse error: #{e.message}"
end
```

Invalid parameter types raise `TypeError`:

```ruby
begin
  db.execute("INSERT INTO t VALUES ($1)", [Object.new])
rescue TypeError => e
  warn e.message
end
```

## Type Mapping

| Ruby | Stoolap | Notes |
|---|---|---|
| `Integer` | `INTEGER` | 64-bit signed (full `i64::MIN..=i64::MAX`) |
| `Float` | `FLOAT` | 64-bit double |
| `String` | `TEXT` | UTF-8, returned as UTF-8-encoded `String` |
| `true` / `false` | `BOOLEAN` | |
| `nil` | `NULL` | Any column type |
| `Time` | `TIMESTAMP` | Nanosecond precision; timezone-aware converted to UTC, naive treated as UTC |
| `Symbol` | `TEXT` | Converted to its string name |
| `Hash` | `JSON` | Serialized via `JSON.dump` (requires `require "json"`) |
| `Array` | `JSON` | Serialized via `JSON.dump` |
| `Stoolap::Vector` | `VECTOR(N)` | See below |

Any other type passed as a parameter raises `TypeError`.

## Vector Similarity Search

Stoolap has native `VECTOR(N)` columns and HNSW indexes. Wrap a Ruby numeric array in `Stoolap::Vector` to store it as a native vector (not a JSON-encoded array):

```ruby
require "stoolap"

db = Stoolap::Database.open(":memory:")

db.exec(<<~SQL)
  CREATE TABLE documents (
    id        INTEGER PRIMARY KEY,
    title     TEXT,
    embedding VECTOR(3)
  );
  CREATE INDEX idx_emb ON documents(embedding) USING HNSW WITH (metric = 'cosine');
SQL

db.execute(
  "INSERT INTO documents VALUES ($1, $2, $3)",
  [1, "Hello world",   Stoolap::Vector.new([0.1, 0.2, 0.3])]
)
db.execute(
  "INSERT INTO documents VALUES ($1, $2, $3)",
  [2, "Goodbye world", Stoolap::Vector.new([0.9, 0.1, 0.0])]
)

# k-NN search: 5 nearest neighbours by cosine distance
results = db.query(<<~SQL)
  SELECT id, title,
         VEC_DISTANCE_COSINE(embedding, '[0.1, 0.2, 0.3]') AS dist
  FROM documents
  ORDER BY dist
  LIMIT 5
SQL

# Read vectors back as Array<Float>
row = db.query_one("SELECT embedding FROM documents WHERE id = 1")
row["embedding"]  # => [0.1, 0.2, 0.3]
```

### Vector class

| Method | Returns | Description |
|---|---|---|
| `Stoolap::Vector.new(array)` | `Vector` | Build from an Array of numbers. Raises `TypeError` on non-numeric |
| `#to_a` | `Array<Float>` | Copy to a plain Array |
| `#length` / `#size` | `Integer` | Dimension count |
| `#inspect` / `#to_s` | `String` | e.g. `#<Stoolap::Vector [0.1, 0.2, 0.3]>` |

### Distance Functions

| Function | Description |
|---|---|
| `VEC_DISTANCE_L2(a, b)` | Euclidean distance |
| `VEC_DISTANCE_COSINE(a, b)` | Cosine distance (1 minus cosine similarity) |
| `VEC_DISTANCE_IP(a, b)` | Negative inner product |

### Vector Utilities

| Function | Description |
|---|---|
| `VEC_DIMS(v)` | Number of dimensions |
| `VEC_NORM(v)` | L2 norm (magnitude) |
| `VEC_TO_TEXT(v)` | Convert to string `[1.0, 2.0, 3.0]` |

### HNSW Index Options

```sql
CREATE INDEX idx ON documents(embedding) USING HNSW WITH (metric = 'cosine');
```

Supported metrics: `l2` (default), `cosine`, `ip` (inner product).

## Features

Stoolap is a full-featured embedded SQL database:

- **MVCC transactions** with snapshot isolation.
- **Cost-based query optimizer** with adaptive execution.
- **Parallel execution** for filter, join, sort, and distinct operators.
- **JOINs**: `INNER`, `LEFT`, `RIGHT`, `FULL OUTER`, `CROSS`, `NATURAL`.
- **Subqueries**: scalar, `EXISTS`, `IN`, `NOT IN`, `ANY`/`ALL`, correlated.
- **Window functions**: `ROW_NUMBER`, `RANK`, `DENSE_RANK`, `LAG`, `LEAD`, `NTILE`, plus frame specs.
- **CTEs**: `WITH` and `WITH RECURSIVE`.
- **Aggregations**: `GROUP BY`, `HAVING`, `ROLLUP`, `CUBE`, `GROUPING SETS`.
- **Vector similarity search** with HNSW indexes over `l2`, `cosine`, `ip`.
- **Indexes**: B-tree, hash, bitmap (auto-selected), HNSW, multi-column composite.
- **131 built-in functions**: string, math, date/time, JSON, vector, aggregate.
- **Immutable volume storage** with columnar format, zone maps, bloom filters, LZ4 compression.
- **WAL + checkpoint cycles** for crash recovery.
- **Aggregation pushdown** to cold volume statistics (`COUNT`, `SUM`, `MIN`, `MAX`).
- **Semantic query caching** with predicate subsumption.

## Building from Source

Requires:
- [Rust](https://rustup.rs) (stable)
- Ruby `>= 3.3`
- A C toolchain for the Ruby header files

```bash
git clone https://github.com/stoolap/stoolap-ruby.git
cd stoolap-ruby
bundle install
bundle exec rake compile
bundle exec rake test
```

The `compile` task invokes `rb_sys/mkmf`, which builds the Rust extension into `lib/stoolap/stoolap.<ext>` so the installed gem can `require` it.
