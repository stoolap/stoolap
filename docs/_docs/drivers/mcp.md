---
layout: doc
title: MCP Server
category: Drivers
order: 12
icon: mcp
---

# MCP Server

MCP ([Model Context Protocol](https://modelcontextprotocol.io)) server for Stoolap. It lets an AI assistant inspect a schema, run queries, write data inside transactions, read query plans and check the health of a database file.

It works with any MCP client that speaks stdio: Claude Desktop, Claude Code, Cursor, Windsurf, Cline and others.

The server provides 14 tools, 2 resources and 1 prompt. On connection it sends built-in instructions, so the assistant writes correct Stoolap SQL from the first query. Every tool carries MCP annotations (`readOnlyHint`, `destructiveHint`), so a client can approve the read-only ones automatically.

This page describes `@stoolap/mcp` 0.5.0, which runs the Stoolap 0.5 engine through [`@stoolap/node`](https://github.com/stoolap/stoolap-node) 0.5. Server 0.4.x has a read-only bypass (see [Upgrading from 0.4](#upgrading-from-04)); use 0.5.0 or later if you rely on `--read-only`.

## Installation

The server is published as `@stoolap/mcp` on npm and in the [MCP Registry](https://registry.modelcontextprotocol.io/) as `io.github.stoolap/mcp`. No manual installation is needed when using `npx`.

Requirements:
- Node.js 20 or later
- `@stoolap/node` 0.5, installed automatically, with prebuilt engine libraries for Linux (x64, arm64), macOS (x64, arm64) and Windows (x64). Its small N-API addon needs a C compiler at install time.

The first `npx` run compiles that addon, which can take a while. A client with a short startup timeout may report a failed connection on that first run. Retry once the install has finished, or install the package globally first:

```bash
npm install -g @stoolap/mcp
```

## Quick Start

### Claude Code

```bash
claude mcp add stoolap -- npx -y @stoolap/mcp --path ./mydata
```

### Claude Desktop and other JSON-configured clients

On macOS the Claude Desktop file is `~/Library/Application Support/Claude/claude_desktop_config.json`. Cursor reads `.cursor/mcp.json` in the project root.

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

Leave out `--path` for an in-memory database that disappears when the server stops.

### Read-only mode

Add `--read-only` to let the assistant read but never write. Reads and read-only transactions still work.

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

### Docker

```bash
git clone https://github.com/stoolap/stoolap-mcp.git
cd stoolap-mcp
docker build -t stoolap-mcp .
docker run -i --rm -v "$PWD/data:/data" stoolap-mcp --path /data/mydb
```

The server talks over stdio, so the container must run with `-i`.

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--path <path>` | `:memory:` | Database path or DSN. Engine options go in the query string, e.g. `./mydata?sync_mode=full&checkpoint_interval=30`. See the [Connection String Reference]({{ '/docs/getting-started/connection-strings/' | relative_url }}). |
| `--read-only` | off | Refuse every statement that writes data, schema or engine settings. |
| `--max-rows <n>` | `1000` | The most rows one call returns. Longer results come back cut, with `truncated: true`. A call can ask for fewer with `max_rows`, never for more. `0` removes the limit. |
| `--query-timeout <ms>` | `30000` | Stop any statement that runs longer, with the error `query cancelled`. `0` turns it off. Without the flag, a `query_timeout` in the DSN is used instead of the default. |
| `--version` | | Print the server version and exit. |

## Tools

### SQL (4 tools)

| Tool | Description |
|------|-------------|
| `query` | Runs one read-only statement: SELECT, WITH ... SELECT, SHOW, DESCRIBE, EXPLAIN. |
| `execute` | Runs one statement that writes: INSERT, UPDATE, DELETE, COPY, CREATE, ALTER, DROP, TRUNCATE, ANALYZE, VACUUM, SET. Returns `{"affected_rows": n}`, or rows when the statement has RETURNING. |
| `execute_batch` | Runs one INSERT, UPDATE or DELETE once per parameter set, all or nothing: when one set fails, none of them stays applied, inside a transaction too. |
| `explain` | Prints the plan of a statement (given without EXPLAIN), one step per line. With `analyze: true` it runs the statement and shows actual time, rows and the paths taken; this is refused for anything but reads. |

`query` and RETURNING return rows in one shape:

```json
{"columns": ["id", "name"], "rows": [[1, "alice"], [2, "bob"]], "row_count": 2, "truncated": false}
```

Columns keep their order and duplicate names (`SELECT a.id, b.id`), and an empty result still lists its columns. VECTOR values are arrays of numbers. The FLOAT values NaN, Infinity and -Infinity are the strings `"NaN"`, `"Infinity"` and `"-Infinity"`, because JSON has no way to write them.

### Transactions (4 tools)

| Tool | Description |
|------|-------------|
| `begin_transaction` | Opens a transaction, `read_committed` (default) or `snapshot`. |
| `commit_transaction` | Commits it. |
| `rollback_transaction` | Rolls it back. |
| `savepoint` | `create`, `rollback_to` or `release` a named savepoint. |

While a transaction is open, `query`, `execute` and `execute_batch` run inside it and see its uncommitted rows. There is one transaction at a time, on the server's single connection.

### Schema (2 tools)

| Tool | Description |
|------|-------------|
| `list_tables` | The names of all tables and views. |
| `get_schema` | CREATE TABLE for every table (or the one table or view named) with constraints and foreign keys, a line per secondary index, and CREATE VIEW for every view. |

Both leave out the `_sys_table_stats` and `_sys_column_stats` tables in which ANALYZE keeps optimizer statistics.

### Engine (4 tools)

| Tool | Description |
|------|-------------|
| `pragma` | Reads or changes an engine setting, reads a report, or runs a maintenance action (see below). |
| `verify` | Reads every column volume of a file database, checks each block's checksum, and reports each segment as `ok`, `damaged`, `unreadable` or `failed`. It changes nothing. |
| `list_functions` | Every built-in SQL function with its signature and semantics, by category. |
| `version` | Engine and server versions, the read-only flag, row limit and query timeout in effect, and the isolation level of the open transaction (`null` when none is open). |

`pragma` accepts these names:

- **Settings** (integer; omit the value to read): `checkpoint_interval`, `compact_threshold`, `target_volume_rows`, `keep_snapshots`, `hot_max_rows`, `hot_max_bytes`, `group_cache_mb`, `index_cache_mb`, `index_build_mb`, `query_timeout` (milliseconds, this connection only).
- **Reports**: `sync_mode`, `wal_flush_trigger`, `volume_stats`, `memory_stats`, `group_cache_stats`, `index_stats`, `index_read_stats`. `PRAGMA VERIFY` is the `verify` tool.
- **Actions**: `checkpoint`, `snapshot`, `vacuum`, `index_backfill` (value: the most volumes to build), `restore` (value: a snapshot timestamp such as `20260315-120000.000`, the latest when omitted; it replaces all data).

See [PRAGMA]({{ '/docs/sql-commands/pragma-commands/' | relative_url }}) for what each one does.

## Resources

| URI | Description |
|-----|-------------|
| `stoolap://schema` | The live schema, as returned by `get_schema`. |
| `stoolap://sql-reference` | The live schema plus the complete Stoolap 0.5 SQL reference and every function. |

## Prompts

| Prompt | Description |
|--------|-------------|
| `sql-assistant` | The same content as `stoolap://sql-reference`, for clients that take prompts rather than resources. |

## Auto-injected Instructions

The server sends [MCP instructions](https://modelcontextprotocol.io/specification/2025-06-18/server/utilities/instructions) when a client connects: which tool to use for what, the result shape, the SQL rules most likely to surprise (comparison of TEXT with numbers, overflow errors, strict name resolution, DDL in transactions) and what Stoolap does not support.

For the full reference with the live schema, attach the `sql-assistant` prompt or read `stoolap://sql-reference`.

## Parameter Binding

`query`, `execute`, `execute_batch` and `explain` take parameters:

```
-- Positional ($1, $2, ... or ?)
params: [1, "Alice", 30]

-- Named (:key)
params: {"id": 1, "name": "Alice"}
```

Parameter values are strings, numbers, booleans or null. JSON and VECTOR values are passed as text.

Two limits come from the Node.js driver, and the instructions tell the assistant about both:

- INTEGER values travel as JavaScript numbers, which are exact only up to 2^53. A larger parameter is refused with a hint to pass it as a string, which the engine converts exactly; read large values with `CAST(x AS TEXT)`.
- TIMESTAMP values come back as ISO 8601 strings to the millisecond. `CAST(ts AS TEXT)` gives all nine digits.

## Usage Examples

### Creating a Table and Inserting Data

Create the table with `execute`:

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
INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id
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

Use `execute_batch` with one statement and several parameter sets:

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

Either every row is inserted or none is.

To load a CSV file from the server host, use `execute`:

```sql
COPY users (name, email) FROM '/data/users.csv' WITH (FORMAT CSV, HEADER true)
```

### Transactions

Open a transaction with `begin_transaction`; `query` and `execute` then run inside it:

```
1. begin_transaction(isolation: "snapshot")
2. execute("INSERT INTO orders VALUES ($1, $2, $3)", [1, 42, 99.99])
3. query("SELECT SUM(amount) FROM orders WHERE user_id = $1", [42])
4. commit_transaction()
```

Use savepoints for a partial rollback:

```
1. begin_transaction()
2. execute("INSERT INTO log VALUES ($1, $2)", [1, "step1"])
3. savepoint(action: "create", name: "before_risky")
4. execute("INSERT INTO log VALUES ($1, $2)", [2, "risky"])
5. savepoint(action: "rollback_to", name: "before_risky")
6. commit_transaction()   -- row 2 is gone, row 1 is committed
```

### Vector Search

Create a table with a vector column and an HNSW index with `execute`:

```sql
CREATE TABLE docs (
  id INTEGER PRIMARY KEY,
  title TEXT,
  embedding VECTOR(384)
)
```

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

Call `get_schema` with no arguments for the complete schema before writing queries, or with a table name for one table. `query` with `SHOW INDEXES FROM users` or `DESCRIBE users` gives the same facts as rows.

Use `explain` with `analyze: true` to see what a query actually did:

```sql
SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name
```

Call `verify` on a file database to check every block of its column volumes.

## Safety

- **Read-only mode.** The Stoolap engine has no read-only open mode, so the server enforces it: in `--read-only` every statement that writes is refused before it reaches the engine, including COPY, DDL, SET, ANALYZE, VACUUM, EXPLAIN ANALYZE of anything but a read, and every pragma action and setting, `query_timeout` included.
- **SQL is read the way the engine reads it.** The checks tokenize SQL like the engine lexer: string escapes, quoted identifiers, `--`, `#` and `/* */` comments, and every Unicode whitespace character. They judge the statement that is actually sent, after EXPLAIN [ANALYZE], and find the main verb of a WITH at its own depth. A NUL character or a byte order mark is refused.
- **One statement per call.** The engine runs every statement of a multi-statement string and reports only the last, so a second statement is refused.
- **Tool routing.** `query` takes only reads and `execute` only writes. Transaction control statements (BEGIN, COMMIT, ROLLBACK, SAVEPOINT, RELEASE) are refused in SQL and run only through the transaction tools, so the server always knows whether a transaction is open.
- **DDL inside a transaction.** The engine rolls back a plain CREATE TABLE made in a transaction, but CREATE TABLE ... AS SELECT, ALTER TABLE, CREATE INDEX and CREATE VIEW persist after ROLLBACK, and DROP TABLE and TRUNCATE lose rows. The server refuses all of those but the plain CREATE TABLE while a transaction is open.
- **Isolation.** `begin_transaction` always names its level, so a `SET isolation_level` run earlier cannot change what it opens.
- **Limits.** The row limit keeps one call from returning millions of rows, and the query timeout keeps it from running forever. A cancelled statement changes nothing and the connection stays usable. The driver has no raw cursor yet, so the server still reads the whole result before cutting it; the timeout bounds that work.
- **One call at a time.** Clients may send requests concurrently; the server runs them one after another, so statements of one transaction never interleave.
- **COPY ... FROM reads files on the host** with the server process's permissions, so an assistant can load any file the process can read. Run with `--read-only` when that is not acceptable.
- **Names and values.** Table names are quoted, savepoint names must be plain identifiers, and pragma names and values are checked before they reach SQL.
- **Clean shutdown.** When the client disconnects or the process gets SIGINT, SIGTERM or SIGHUP, an open transaction is rolled back and the database is closed cleanly.

## Upgrading from 0.4

- The 30 tools became 14. `transaction_query`, `transaction_execute` and `transaction_execute_batch` are gone: `query`, `execute` and `execute_batch` run inside the open transaction. `create_table`, `create_index`, `create_view`, `alter_table`, `drop`, `analyze_table` and `vacuum` are statements for `execute`. `list_views` is part of `list_tables`; `describe_table`, `show_create_table`, `show_create_view` and `show_indexes` are covered by `get_schema` and by `query` with SHOW and DESCRIBE. `savepoint`, `rollback_to_savepoint` and `release_savepoint` are one `savepoint` tool with an `action`.
- Rows come back as `{columns, rows, row_count, truncated}` instead of an array of objects, and writes as `{"affected_rows": n}`.
- `explain` takes the statement without the EXPLAIN keyword.
- `--read-only` was not airtight in 0.4: a `#` line comment could hide a second statement from the check. Upgrade if you rely on it.
- Results stop at 1,000 rows and statements at 30 seconds by default.
- `stoolap://schema` is DDL text instead of JSON.
- Stoolap 0.5 opens a 0.4 database without a manual step and moves its format forward as it runs; 0.4 cannot open it afterwards. Back up the directory first; the [migration guide]({{ '/docs/development/migration-v042-to-v050/' | relative_url }}) lists every answer that changed.

## Building from Source

```bash
git clone https://github.com/stoolap/stoolap-mcp.git
cd stoolap-mcp
npm install
npm test          # builds, then runs the unit and end-to-end tests
node build/index.js --path ./mydata
```
