---
layout: doc
title: MCP Server
category: Drivers
order: 7
icon: mcp
---

# MCP Server

MCP ([Model Context Protocol](https://modelcontextprotocol.io)) server for Stoolap. Lets AI assistants query, manage, and analyze Stoolap databases with full access to all SQL features.

Works with any MCP-compatible AI client: Claude Desktop, Claude Code, Cursor, Windsurf, Cline, and others.

The server provides 30 tools, 2 resources, and 1 prompt. On connection, it sends built-in instructions so the AI can write correct Stoolap SQL from the first query.

## Installation

The MCP server is published as `@stoolap/mcp` on npm. No manual installation is needed when using `npx`.

Requirements:
- Node.js >= 18
- The `@stoolap/node` package (installed automatically as a dependency)

Prebuilt native binaries are bundled for Linux (x64, arm64) and macOS (x64, arm64).

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

Add `--read-only` to disable all write operations. Read-only transactions (begin, query, commit) are still allowed for consistent reads.

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
| `--path <path>` | `:memory:` | Database path. Use `:memory:` for in-memory or a file path for persistence. |
| `--read-only` | `false` | Disable write operations |

## Tools

The server exposes 30 tools organized into five categories.

### Query and Analysis (4 tools)

| Tool | Description |
|------|-------------|
| `query` | Run SELECT, SHOW, DESCRIBE, EXPLAIN queries. Returns results as JSON. |
| `execute` | Run INSERT, UPDATE, DELETE with parameter binding. Supports RETURNING clause and upsert (ON DUPLICATE KEY UPDATE). Returns affected row count. |
| `execute_batch` | Execute the same SQL with multiple parameter sets in a single atomic transaction. All rows succeed or all are rolled back. |
| `explain` | Show query execution plan. Set `analyze=true` to run the query and show actual runtime stats. |

### Transaction Control (9 tools)

| Tool | Description |
|------|-------------|
| `begin_transaction` | Begin a new transaction with optional isolation level (`read_committed` or `snapshot`). Only one active transaction at a time. |
| `transaction_execute` | Execute a DML statement within the active transaction. Sees uncommitted changes. |
| `transaction_query` | Run a SELECT query within the active transaction. Sees uncommitted changes. Full SQL feature support. |
| `transaction_execute_batch` | Execute the same SQL with multiple parameter sets within the active transaction. |
| `commit_transaction` | Commit the active transaction. All changes become permanent. |
| `rollback_transaction` | Rollback the active transaction. All changes are discarded. |
| `savepoint` | Create a named savepoint within the active transaction. |
| `rollback_to_savepoint` | Rollback to a savepoint, undoing changes after it without aborting the transaction. |
| `release_savepoint` | Release a savepoint. Changes are kept. |

### Schema Inspection (7 tools)

| Tool | Description |
|------|-------------|
| `list_tables` | List all tables |
| `list_views` | List all views |
| `describe_table` | Show columns, types, nullability, keys, defaults, and extras |
| `show_create_table` | Get the full CREATE TABLE DDL including all constraints |
| `show_create_view` | Get the full CREATE VIEW DDL |
| `show_indexes` | Show all indexes on a table (type, columns, uniqueness) |
| `get_schema` | Get the complete database schema: all tables with columns, indexes, DDL, plus all views |

### Schema Modification (5 tools)

| Tool | Description |
|------|-------------|
| `create_table` | Create a table with all column types and constraints. Supports IF NOT EXISTS and CREATE TABLE AS SELECT. |
| `create_index` | Create BTREE, HASH, BITMAP, or HNSW indexes. Supports UNIQUE and composite. |
| `create_view` | Create a read-only view (persists across restarts) |
| `alter_table` | ADD COLUMN, DROP COLUMN, RENAME COLUMN, MODIFY COLUMN, RENAME TO |
| `drop` | Drop a table, view, or index (supports IF EXISTS) |

### Database Administration (5 tools)

| Tool | Description |
|------|-------------|
| `analyze_table` | Collect optimizer statistics for better query plans |
| `vacuum` | Clean up deleted rows, old MVCC versions, and compact indexes |
| `pragma` | Get/set database config: sync_mode, checkpoint_interval, compact_threshold, keep_snapshots, wal_flush_trigger |
| `version` | Get the Stoolap engine version |
| `list_functions` | List all 130+ built-in SQL functions with signatures, grouped by category |

## Resources

| URI | Description |
|-----|-------------|
| `stoolap://schema` | Full database schema with all tables, views, columns, indexes, and DDL statements |
| `stoolap://sql-reference` | Live database schema plus complete Stoolap SQL reference: data types, 130+ functions with signatures, operators, joins, indexes, window functions, CTEs, transactions, temporal queries, vector search, and known limitations |

## Prompts

| Prompt | Description |
|--------|-------------|
| `sql-assistant` | Same content as `stoolap://sql-reference` delivered as an MCP prompt. Use whichever your client supports. |

## Auto-injected Instructions

The server provides built-in [MCP instructions](https://modelcontextprotocol.io/specification/2025-03-26/server/utilities/instructions) that are automatically sent to the AI during the connection handshake. Any AI client receives a comprehensive Stoolap SQL reference on connect, covering data types, supported syntax, all operator categories, index types, vector search, transaction isolation levels, and known limitations, without the user needing to configure anything.

For deeper reference (live schema and full function signatures), attach the `sql-assistant` prompt.

## Parameter Binding

All query and execute tools support parameter binding:

```
-- Positional ($1, $2, ...)
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

Query nearest neighbors with `query`:

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
- **CTEs**: WITH, WITH RECURSIVE, multiple CTEs, column aliases
- **Window functions**: ROW_NUMBER, RANK, DENSE_RANK, NTILE, LEAD, LAG, FIRST_VALUE, LAST_VALUE, NTH_VALUE, PERCENT_RANK, CUME_DIST (plus all aggregates with OVER)
- **GROUP BY extensions**: ROLLUP, CUBE, GROUPING SETS, GROUPING()
- **Aggregates**: COUNT, SUM, AVG, MIN, MAX, MEDIAN, STRING_AGG, ARRAY_AGG, STDDEV, VARIANCE, and more
- **100+ scalar functions**: string, math, date/time, JSON, hash, conditional, vector, type conversion
- **Operators**: arithmetic, comparison, logical, bitwise, LIKE/ILIKE/GLOB/REGEXP, JSON (->/->>), vector (<=>), BETWEEN, IN, IS [NOT] DISTINCT FROM, INTERVAL
- **Transactions**: BEGIN with isolation levels (READ COMMITTED, SNAPSHOT), COMMIT, ROLLBACK, SAVEPOINT
- **Temporal queries**: AS OF TIMESTAMP, AS OF TRANSACTION
- **Index types**: BTree, Hash, Bitmap, HNSW (vector), Unique, Composite
- **Vector search**: k-NN with L2, cosine, inner product distances and HNSW indexing
- **EXPLAIN / EXPLAIN ANALYZE** for query plan inspection
- **DISTINCT ON**: Per-group deduplication (PostgreSQL-inspired, no leading ORDER BY restriction)
- **Set operations**: UNION [ALL], INTERSECT [ALL], EXCEPT [ALL]

## Safety

The server includes several safety measures:

- **Single-statement enforcement**: Multi-statement SQL (semicolons outside strings/comments) is rejected. Each tool call runs one statement.
- **Read-only mode**: `--read-only` disables all write operations at the server level.
- **Tool routing**: `query` only accepts read-only statements (SELECT, SHOW, DESCRIBE, EXPLAIN). `execute` only accepts write statements. Misrouted statements are rejected with a descriptive error.
- **Transaction isolation**: DDL (CREATE/ALTER/DROP) is rejected inside transactions because it is auto-committed and cannot be rolled back.
- **SQL injection prevention**: Table/view names are quoted with double-quote escaping. Savepoint and pragma names are validated as bare identifiers. PRAGMA values are validated as numeric.

## Building from Source

```bash
git clone https://github.com/stoolap/stoolap-mcp.git
cd stoolap-mcp
npm install
npm run build
node build/index.js --path ./mydata
```
