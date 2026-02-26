<div align="center">
  <img src="logo.svg" alt="Stoolap Logo" width="360">

  <h3>A Modern Embedded SQL Database in Pure Rust</h3>

  <p>
    <a href="https://stoolap.io/docs">Documentation</a> •
    <a href="https://stoolap.io/playground">Playground</a> •
    <a href="https://github.com/stoolap/stoolap/releases">Releases</a> •
    <a href="BENCHMARKS.md">Benchmarks</a>
  </p>

  <p>
    <a href="https://github.com/stoolap/stoolap/actions/workflows/ci.yml"><img src="https://github.com/stoolap/stoolap/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
    <a href="https://codecov.io/gh/stoolap/stoolap"><img src="https://codecov.io/gh/stoolap/stoolap/branch/main/graph/badge.svg" alt="codecov"></a>
    <a href="https://crates.io/crates/stoolap"><img src="https://img.shields.io/crates/v/stoolap.svg" alt="Crates.io"></a>
    <a href="https://github.com/stoolap/stoolap/releases"><img src="https://img.shields.io/github/v/release/stoolap/stoolap" alt="GitHub release"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  </p>
</div>

---

Stoolap is a feature-rich embedded SQL database built in pure Rust.
It targets low-latency transactional workloads and real-time analytical queries, with modern SQL features and no external server process.

## Why Stoolap?

Stoolap is designed around practical embedded database needs:

- **ACID + MVCC**: concurrent reads and writes with transaction isolation
- **Cost-based optimization**: statistics-aware planning with adaptive execution
- **Rich SQL surface**: joins, subqueries, CTEs, window functions, advanced aggregations
- **Multiple index types**: B-tree, Hash, Bitmap, multi-column, and HNSW for vectors
- **Pure Rust runtime**: memory-safe implementation, no C/C++ dependency chain

### Feature Snapshot

| Feature | Stoolap | SQLite | DuckDB | PostgreSQL |
|---------|:-------:|:------:|:------:|:----------:|
| AS OF Time-Travel Queries | ✅ | ❌ | ❌ | ❌* |
| MVCC Transactions | ✅ | ❌ | ✅ | ✅ |
| Cost-Based Optimizer | ✅ | ❌ | ✅ | ✅ |
| Adaptive Query Execution | ✅ | ❌ | ❌ | ❌ |
| Semantic Query Caching | ✅ | ❌ | ❌ | ❌ |
| Parallel Query Execution | ✅ | ❌ | ✅ | ✅ |
| Native Vector / HNSW Search | ✅ | ❌ | ❌ | ❌ |
| Pure Rust (Memory Safe) | ✅ | ❌ | ❌ | ❌ |

*PostgreSQL typically needs extensions for temporal query workflows.

## Quick Start

### Installation

```toml
[dependencies]
stoolap = "0.3"
```

Build from source:

```bash
git clone https://github.com/stoolap/stoolap.git
cd stoolap
cargo build --release
```

### Rust API

```rust
use stoolap::api::Database;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = Database::open_in_memory()?;

    db.execute(
        "CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )",
        (),
    )?;

    db.execute(
        "INSERT INTO users (id, name, email) VALUES ($1, $2, $3)",
        (1, "Alice", "alice@example.com"),
    )?;

    for row in db.query("SELECT id, name, email FROM users WHERE id = $1", (1,))? {
        let row = row?;
        println!(
            "id={} name={} email={}",
            row.get::<i64>(0)?,
            row.get::<String>(1)?,
            row.get::<String>(2)?
        );
    }

    Ok(())
}
```

### CLI

```bash
# Interactive REPL
./stoolap

# Execute a single query
./stoolap -e "SELECT version()"

# Persistent database
./stoolap --db "file://./mydb"
```

## Core SQL Capabilities

### Transactions and Time-Travel

```sql
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;

SELECT * FROM accounts AS OF TIMESTAMP '2024-01-15 10:30:00';
SELECT * FROM inventory AS OF TRANSACTION 1234;
```

### Cost-Based Query Optimizer

```sql
ANALYZE orders;
ANALYZE customers;

EXPLAIN SELECT * FROM orders WHERE customer_id = 100;

EXPLAIN ANALYZE
SELECT o.*, c.name
FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE c.country = 'US';
```

### Indexing

```sql
-- Auto-selected by data type
CREATE INDEX idx_created_at ON orders(created_at);   -- B-tree
CREATE INDEX idx_email ON users(email);              -- Hash
CREATE INDEX idx_active ON users(is_active) USING BITMAP;

-- Multi-column
CREATE INDEX idx_lookup ON events(user_id, event_type);
```

### Advanced SQL

```sql
WITH ranked AS (
    SELECT
        customer_id,
        amount,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY amount DESC) AS rn
    FROM orders
)
SELECT * FROM ranked WHERE rn = 1;
```

## Vector and Semantic Search

Stoolap supports native vectors via `VECTOR(N)` and approximate nearest-neighbor search with HNSW.

```sql
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY,
    content TEXT,
    embedding VECTOR(384)
);

CREATE INDEX idx_emb ON embeddings(embedding)
USING HNSW WITH (metric = 'cosine', m = 32, ef_construction = 400, ef_search = 128);

SELECT id, content,
       VEC_DISTANCE_COSINE(embedding, '[0.1, 0.2, ...]') AS dist
FROM embeddings
ORDER BY dist
LIMIT 10;
```

For built-in semantic text embeddings, enable the `semantic` feature:

```toml
[dependencies]
stoolap = { version = "0.3", features = ["semantic"] }
```

```sql
SELECT EMBED('How to reset my password');
```

See [Vector Search](https://stoolap.io/docs/data-types/vector-search/) and [Semantic Search](https://stoolap.io/docs/data-types/semantic-search/) docs for full workflows.

## Storage and Durability

- Write-Ahead Logging (WAL)
- Periodic snapshots
- Crash recovery and index persistence
- Configurable sync and compression behavior

## Performance

Detailed benchmark results are in [BENCHMARKS.md](BENCHMARKS.md).

Benchmark figures are point-in-time and workload-dependent. Validate on your own hardware, data distribution, and query patterns.

## Documentation

- Installation: https://stoolap.io/docs/getting-started/installation/
- SQL commands: https://stoolap.io/docs/sql-commands/sql-commands/
- Data types: https://stoolap.io/docs/data-types/data-types/
- Functions: https://stoolap.io/docs/functions/sql-functions-reference/
- Architecture: https://stoolap.io/docs/architecture/architecture/
- Drivers: [Node.js](https://stoolap.io/docs/drivers/nodejs/) | [Python](https://stoolap.io/docs/drivers/python/) | [WASM](https://stoolap.io/docs/drivers/wasm/)

## Development

```bash
cargo build
cargo nextest run
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
