---
layout: doc
title: MCP Server
category: Drivers
order: 12
icon: mcp
---

# MCP Server

MCP ([Model Context Protocol](https://modelcontextprotocol.io)) server for Stoolap. Lets AI assistants query, manage, and analyze Stoolap databases with full access to all SQL features.

Works with any MCP-compatible AI client: Claude Desktop, Claude Code, Cursor, Windsurf, Cline, and others.

The server provides 30 tools, 2 resources, and 1 prompt. On connection, it sends built-in instructions so the AI can write correct Stoolap SQL from the first query. Every tool carries MCP annotations (`readOnlyHint`, `destructiveHint`), so clients can auto-approve the read-only ones.

This page describes `@stoolap/mcp` 0.4.1 or later, which targets the Stoolap 0.4.x engine (volume-based storage) through [`@stoolap/node`](https://github.com/stoolap/stoolap-node). Pin the version in your client configuration (`@stoolap/mcp@^0.4.1`) if you rely on the read-only guarantees below; 0.4.0 does not block COPY ... FROM in read-only mode.

## Installation

The MCP server is published as `@stoolap/mcp` on npm. No manual installation is needed when using `npx`.

Requirements:
- Node.js >= 20
- The `@stoolap/node` package (installed automatically as a dependency) with prebuilt engine libraries for Linux (x64, arm64), macOS (x64, arm64) and Windows (x64). A C compiler is needed for its small N-API addon.

The first `npx` run compiles that addon, which can take a while. MCP clients with a short startup timeout may report a failed connection on that first run. Retry once the install has finished, or install the package globally beforehand:

```bash
npm install -g @stoolap/mcp
```

## Quick Start

### Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "stoolap": {
      "command": "npx",
      "args": ["-y", "@stoolap/mcp", "--path", "./mydata"]
    }
  }
}
```

### Claude Code

```bash
claude mcp add stoolap -- npx -y @stoolap/mcp --path ./mydata
```

### Cursor

Add to `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "stoolap": {
      "command": "npx",
      "args": ["-y", "@stoolap/mcp", "--path", "./mydata"]
    }
  }
}
```

### In-memory (no persistence)

Omit the `--path` flag to use an in-memory database:

```json
{
  "mcpServers": {
    "stoolap": {
      "command": "npx",
      "args": ["-y", "@stoolap/mcp"]
    }
  }
}
```

### Read-only mode

Add `--read-only` to reject every statement that writes data, schema or engine state. Read-only transactions (begin, query, commit) are still allowed for consistent reads.

```json
{
  "mcpServers": {
    "stoolap": {
      "command": "npx",
      "args": ["-y", "@stoolap/mcp", "--path", "./mydata", "--read-only"]
    }
  }
}
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--path <path>` | `:memory:` | Database path or DSN. Engine options go in the query string, e.g. `./mydata?sync_mode=full&checkpoint_interval=30`. |
| `--read-only` | `false` | Reject every statement that writes data, schema or engine state. |
| `--version` | | Print the server version and exit. |

## Tools

The server exposes 30 tools organized into five categories.

### Query and Analysis (4 tools)

| Tool | Description |
|------|-------------|
| `query` | Run SELECT, SHOW, DESCRIBE, EXPLAIN, VALUES and WITH ... SELECT. Returns rows as JSON. Runs inside the active transaction if one is open. |
| `execute` | Run INSERT, UPDATE, DELETE, COPY ... FROM, DDL, SET, ANALYZE, VACUUM with parameter binding. Supports upsert (ON CONFLICT / ON DUPLICATE KEY UPDATE) and RETURNING. Returns rows for RETURNING, otherwise the affected row count. |
| `execute_batch` | Execute the same SQL with multiple parameter sets in a single atomic transaction. All rows succeed or all are rolled back. |
| `explain` | Show the query plan. `analyze=true` runs the statement and reports actual row counts and timings (refused for write statements). |

### Transaction Control (9 tools)

| Tool | Description |
|------|-------------|
| `begin_transaction` | Begin a transaction with optional isolation level (`read_committed` or `snapshot`). One active transaction at a time. |
| `transaction_execute` | Execute INSERT, UPDATE or DELETE inside the active transaction. DDL, TRUNCATE and COPY are refused. |
| `transaction_query` | Run a read-only statement inside the active transaction. Sees uncommitted changes. |
| `transaction_execute_batch` | Execute the same SQL with multiple parameter sets inside the active transaction. |
| `commit_transaction` | Commit the active transaction. |
| `rollback_transaction` | Rollback the active transaction. |
| `savepoint` | Create a named savepoint. |
| `rollback_to_savepoint` | Undo changes made after a savepoint without ending the transaction. |
| `release_savepoint` | Remove a savepoint, keeping its changes. |

### Schema Inspection (7 tools)

| Tool | Description |
|------|-------------|
| `list_tables` | List all tables |
| `list_views` | List all views |
| `describe_table` | Columns, types, nullability, keys, defaults and extras |
| `show_create_table` | Full CREATE TABLE DDL including constraints and foreign keys |
| `show_create_view` | Full CREATE VIEW DDL |
| `show_indexes` | Indexes of a table: name, type, columns, uniqueness, options |
| `get_schema` | The complete schema: every table with columns, indexes and DDL, plus every view |

### Schema Modification (5 tools)

| Tool | Description |
|------|-------------|
| `create_table` | INTEGER, FLOAT, TEXT, BOOLEAN, TIMESTAMP, JSON, VECTOR(N) columns; PRIMARY KEY (including composite), NOT NULL, UNIQUE, DEFAULT, CHECK, AUTO_INCREMENT, single-column foreign keys; IF NOT EXISTS; CREATE TABLE AS SELECT |
| `create_index` | BTREE, HASH, BITMAP or HNSW indexes, UNIQUE and composite. HNSW options: m, ef_construction, ef_search, metric |
| `create_view` | Read-only view that persists across restarts |
| `alter_table` | ADD COLUMN, DROP COLUMN, RENAME COLUMN, MODIFY COLUMN, RENAME TO |
| `drop` | DROP TABLE / VIEW / INDEX ... ON table (supports IF EXISTS) |

### Database Administration (5 tools)

| Tool | Description |
|------|-------------|
| `analyze_table` | Collect optimizer statistics for a table |
| `vacuum` | Remove deleted rows and old MVCC versions, compact indexes (discards time-travel history) |
| `pragma` | Read or set `checkpoint_interval`, `compact_threshold`, `target_volume_rows`, `keep_snapshots`; read `sync_mode`, `wal_flush_trigger`, `volume_stats`; run `snapshot`, `checkpoint`, `vacuum`, `restore` |
| `version` | Engine and server version |
| `list_functions` | All built-in SQL functions with signatures, grouped by category |

## Resources

| URI | Description |
|-----|-------------|
| `stoolap://schema` | Full database schema with all tables, views, columns, indexes, and DDL statements (JSON) |
| `stoolap://sql-reference` | Live database schema plus the complete Stoolap SQL reference (Markdown) |

## Prompts

| Prompt | Description |
|--------|-------------|
| `sql-assistant` | Same content as `stoolap://sql-reference` delivered as an MCP prompt. Use whichever your client supports. |

## Auto-injected Instructions

The server sends [MCP instructions](https://modelcontextprotocol.io/specification/2025-03-26/server/utilities/instructions) during the connection handshake, so any AI client receives a compact Stoolap SQL reference on connect: data types, tool routing, upsert syntax, index and vector rules, transaction rules, and the known limitations of the 0.4.x engine, without the user needing to configure anything.

For the full reference with the live schema, attach the `sql-assistant` prompt or read `stoolap://sql-reference`.

## Parameter Binding

All query and execute tools support parameter binding:

```
-- Positional ($1, $2, ... or ?)
params: [1, "Alice", 30]

-- Named (:key)
params: {"id": 1, "name": "Alice"}
```

Parameter types supported: `string`, `number`, `boolean`, `null`.

## Usage Examples

### Creating a Table and Inserting Data

Use the `create_table` tool:

```sql
CREATE TABLE users (
  id INTEGER PRIMARY KEY AUTO_INCREMENT,
  name TEXT NOT NULL,
  email TEXT UNIQUE,
  created_at TIMESTAMP DEFAULT NOW()
)
```

Then insert with `execute`:

```sql
INSERT INTO users (name, email) VALUES ($1, $2)
```
```json
params: ["Alice", "alice@example.com"]
```

### Upsert

Use `execute` with ON CONFLICT. Refer to the incoming row as `EXCLUDED`:

```sql
INSERT INTO users (id, name, email) VALUES ($1, $2, $3)
ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, email = EXCLUDED.email
```

### Bulk Insert

Use `execute_batch` with a single SQL and multiple parameter sets:

```sql
INSERT INTO users (name, email) VALUES ($1, $2)
```
```json
params_array: [
  ["Alice", "alice@example.com"],
  ["Bob", "bob@example.com"],
  ["Charlie", "charlie@example.com"]
]
```

All rows are inserted atomically in a single transaction.

To load a CSV file from the server host, use `execute`:

```sql
COPY users (name, email) FROM '/data/users.csv' WITH (FORMAT CSV, HEADER true)
```

### Querying with Aggregates

Use the `query` tool:

```sql
SELECT category, COUNT(*) as count, AVG(price) as avg_price
FROM products
GROUP BY category
ORDER BY count DESC
```

### Transactions

Begin a transaction with `begin_transaction`, then use `transaction_execute` and `transaction_query`:

```
1. begin_transaction(isolation: "snapshot")
2. transaction_execute("INSERT INTO orders VALUES ($1, $2, $3)", [1, 42, 99.99])
3. transaction_query("SELECT SUM(amount) FROM orders WHERE user_id = $1", [42])
4. commit_transaction()
```

Use savepoints for partial rollback:

```
1. begin_transaction()
2. transaction_execute("INSERT INTO log VALUES ($1, $2)", [1, "step1"])
3. savepoint("before_risky")
4. transaction_execute("INSERT INTO log VALUES ($1, $2)", [2, "risky"])
5. rollback_to_savepoint("before_risky")
6. commit_transaction()   -- row 2 is gone, row 1 is committed
```

### Vector Search

Create a table with a vector column using `create_table`:

```sql
CREATE TABLE docs (
  id INTEGER PRIMARY KEY,
  title TEXT,
  embedding VECTOR(384)
)
```

Create an HNSW index for fast similarity search using `create_index`:

```sql
CREATE INDEX idx_emb ON docs(embedding) USING HNSW WITH (metric = 'cosine')
```

Query nearest neighbors with `query`. The distance function must match the index metric:

```sql
SELECT id, title, VEC_DISTANCE_COSINE(embedding, '[0.1, 0.2, ...]') AS dist
FROM docs
ORDER BY dist
LIMIT 10
```

### Inspecting the Database

Use `get_schema` (no parameters) to get the complete schema before writing queries. Use `describe_table` for a single table's columns, and `show_indexes` for its indexes.

Use `explain` with `analyze=true` to see actual execution stats:

```sql
SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name
```

## SQL Coverage

The MCP server exposes the full Stoolap SQL surface:

- **7 data types**: INTEGER, FLOAT, TEXT, BOOLEAN, TIMESTAMP, JSON, VECTOR(N)
- **Joins**: INNER, LEFT, RIGHT, FULL OUTER, CROSS, NATURAL, self-joins, multi-table
- **Subqueries**: scalar, IN/NOT IN, EXISTS/NOT EXISTS, ANY/SOME/ALL, correlated, derived tables
- **CTEs**: WITH, WITH RECURSIVE, multiple CTEs, column aliases, WITH before INSERT/UPDATE/DELETE
- **Window functions**: ROW_NUMBER, RANK, DENSE_RANK, NTILE, LEAD, LAG, FIRST_VALUE, LAST_VALUE, NTH_VALUE, PERCENT_RANK, CUME_DIST, every aggregate with OVER, named windows
- **Aggregates**: 17 functions with DISTINCT and FILTER; GROUP BY ROLLUP, CUBE, GROUPING SETS; DISTINCT ON
- **Scalar functions**: 98 functions across string, math, date/time, JSON, hash, conditional, type and vector categories
- **Operators**: arithmetic, comparison, logical, bitwise, LIKE/ILIKE/GLOB/REGEXP, JSON (->/->>), vector (<=>), BETWEEN, IN, IS [NOT] DISTINCT FROM, INTERVAL
- **Upsert**: ON CONFLICT DO UPDATE / DO NOTHING with EXCLUDED, ON DUPLICATE KEY UPDATE
- **Bulk load**: COPY table FROM 'file.csv' WITH (FORMAT CSV, HEADER true)
- **Transactions**: READ COMMITTED and SNAPSHOT isolation, savepoints
- **Temporal queries**: AS OF TIMESTAMP, AS OF TRANSACTION
- **Indexes**: BTree, Hash, Bitmap, HNSW (vector), unique, composite
- **Vector search**: k-NN with L2, cosine and inner product distances, HNSW indexing
- **Set operations**: UNION [ALL], INTERSECT [ALL], EXCEPT [ALL]
- **EXPLAIN / EXPLAIN ANALYZE** for query plan inspection

## Safety

The server includes several safety measures:

- **Single statement per call**: the engine executes every statement of a multi-statement string but reports only the last one, so semicolon-separated batches are rejected.
- **Tool routing**: `query` accepts only read statements, `execute` is blocked while a transaction is open, and transaction control statements (BEGIN, COMMIT, ROLLBACK, SAVEPOINT) are only reachable through the transaction tools, so the server always knows the connection's transaction state.
- **Read-only mode**: `--read-only` rejects every write, including COPY, DDL, SET, ANALYZE, VACUUM and PRAGMA actions.
- **COPY ... FROM reads files on the host** with the server process's permissions, so an assistant can load any readable file into a table. Run with `--read-only` when that is not acceptable. Server 0.4.0 does not apply the read-only check to COPY; use 0.4.1 or later.
- **DDL outside transactions**: only CREATE TABLE is rolled back reliably by the engine, so DDL, TRUNCATE and COPY are refused inside a transaction.
- **EXPLAIN ANALYZE** is refused for write statements because it executes them.
- **Injection guards**: table and view names are double-quoted, savepoint and pragma names must be bare identifiers, pragma values are validated per pragma.
- **Clean shutdown**: the database is closed cleanly (open transaction rolled back, checkpoint on close) when the client disconnects or the process receives SIGINT/SIGTERM.

## Building from Source

```bash
git clone https://github.com/stoolap/stoolap-mcp.git
cd stoolap-mcp
npm install
npm run build
node build/index.js --path ./mydata
```
