# 11. SQL Capabilities, Security & Extensibility

## Subsystem Architecture Overview

Stoolap is engineered as a lean, embedded HTAP database engine. Its query planning and function resolution pipelines prioritize zero-cost abstractions and predictable in-process memory footprints:

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Client Query / Extensibility Hook                                         │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
┌───────────────────────────┐                   ┌───────────────────────────┐
│ Function Registry         │                   │ Execution Pipeline        │
│ (src/functions/registry.rs)│                   │ (src/executor/planner.rs) │
│ - Static function lookup  │                   │ - Relational operators    │
│ - Built-in scalar & vector│                   │ - Volatile/Pure rules     │
└───────────────────────────┘                   └───────────────────────────┘
```

### Key Source References
- [`src/functions/registry.rs`](file:///home/irshad/stoolap/src/functions/registry.rs): Global static registry storing built-in scalar, aggregate, window, and table-valued functions.
- [`src/executor/planner.rs`](file:///home/irshad/stoolap/src/executor/planner.rs): Compiles AST into relational operator execution trees.
- [`src/api/database.rs`](file:///home/irshad/stoolap/src/api/database.rs): Public embedded API entry point.

---

## Known Limitations Breakdown

1. **No Stored Procedures or Triggers**: Procedural SQL blocks (`CREATE PROCEDURE`, `CREATE TRIGGER`, control flow statements) are not supported.
2. **No User-Defined Functions (UDFs)**: SQL statements cannot dynamically register custom functions at runtime (`CREATE FUNCTION ...`).
3. **No Role-Based Access Control (`GRANT` / `REVOKE`)**: The engine lacks user accounts, roles, and object-level permissions. Security boundaries must be enforced by the host application.
4. **No Inverted Full-Text Search (FTS) Index**: Text search is limited to pattern matching operators (`LIKE`, `ILIKE`, `GLOB`, `REGEXP`). Inverted BM25 term indices and tokenizers are not built-in.
5. **No Materialized Views**: View definitions are expanded and recomputed dynamically on every query. Automatic incremental or periodic persistence of view datasets is not supported.
6. **No Asynchronous Event Notifications**: PostgreSQL-style `LISTEN` and `NOTIFY` pub-sub channels are not available.

---

## Architectural Root Causes

### 1. Embedded Process Design Philosophy
Unlike client-server database engines (e.g. PostgreSQL, MySQL) that manage multi-tenant network authentication, connection pools, and long-running background daemons, Stoolap is embedded directly into host applications (similar to SQLite or DuckDB). Consequently, authorization and session governance are typically managed by the host process.

### 2. Missing Procedural Bytecode Virtual Machine
Stoolap’s expression VM ([`src/executor/expression/vm.rs`](file:///home/irshad/stoolap/src/executor/expression/vm.rs)) evaluates pure scalar expressions without support for procedural control-flow instructions (`LOOP`, `WHILE`, `IF...ELSE`, exception handlers, or mutable procedural variables).

### 3. Separation of Analytical Vector Search vs Inverted FTS Indices
Stoolap prioritized semantic and vector search capabilities via HNSW index algorithms ([`src/storage/index/hnsw.rs`](file:///home/irshad/stoolap/src/storage/index/hnsw.rs)) over traditional inverted index text search engines.

---

## Performance & System Impact

- **Full-Text Scan Overhead**: Executing keyword searches across millions of documents via `LIKE '%keyword%'` or `REGEXP` forces linear table scans and CPU-bound regex evaluations without index acceleration.
- **Complex Aggregation Latency in Views**: Highly complex analytical views containing heavy joins and multi-level aggregations cannot cache their results, requiring repetitive computation on every read.

---

## Proposed Engineering Roadmap

### Phase 1: Native Rust & WASM UDF Registration API
- Add an API to `Database` allowing host applications to register custom Rust closures or WebAssembly scalar functions:
  ```rust
  db.register_scalar_function("custom_hash", |args| {
      let val = args[0].as_str()?;
      Ok(Value::Text(custom_algo(val)))
  })?;
  ```

### Phase 2: Inverted Full-Text Search (FTS) Index Extension
- Add a specialized index type `IndexType::FTS` using an inverted index with BM25 ranking and standard tokenizers (unicode word boundary segmentation, porter stemming).
- Support `MATCH` syntax (e.g. `WHERE content MATCH 'database AND performance'`).

### Phase 3: Materialized Views with Incremental Refresh
- Introduce `CREATE MATERIALIZED VIEW name AS SELECT ...` backed by dedicated cold storage frozen volumes.
- Support `REFRESH MATERIALIZED VIEW` commands and change-data-capture (CDC) incremental updates based on WAL commit sequences.

### Phase 4: In-Process Event Notification Bus
- Implement an in-memory broadcast channel (`tokio::sync::broadcast` or `crossbeam_channel`) accessible via the Rust API for subscribing to table mutation events.
