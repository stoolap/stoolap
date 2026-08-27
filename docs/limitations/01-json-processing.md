# 01. JSON Processing, Path Querying & Indexing

## Subsystem Architecture Overview

Stoolap provides native JSON handling using the `serde_json` ecosystem integrated directly into its core type and expression evaluation systems:

```
┌────────────────────────────────────────────────────────────────────────┐
│ SQL Query: SELECT payload->'user'->>'id' FROM events WHERE payload->'status' = 'active' │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │   Parser AST (BinaryOp::JsonExtract / JsonExtractText) │
       └────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │   Expression VM Bytecode (OpCode::JsonExtract)         │
       │   (src/executor/expression/ops.rs)                     │
       └────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │   Core Value Representation: Value::Json(CompactString)│
       │   (src/core/value.rs & serde_json::Value parsing)      │
       └────────────────────────────────────────────────────────┘
```

### Key Source References
- [`src/core/value.rs`](file:///home/irshad/stoolap/src/core/value.rs): Defines `Value::Json(CompactString)` and JSON literal conversions.
- [`src/executor/expression/ops.rs`](file:///home/irshad/stoolap/src/executor/expression/ops.rs): Evaluates extraction operators (`->` and `->>`).
- [`src/functions/scalar/utility.rs`](file:///home/irshad/stoolap/src/functions/scalar/utility.rs): Implements basic JSON validation and table-valued `json_each`.
- [`src/storage/index/`](file:///home/irshad/stoolap/src/storage/index/): Physical column index managers (BTree, Hash, Bitmap, HNSW).

---

## Known Limitations Breakdown

According to the Stoolap specification, the following limitations exist in the JSON subsystem:

1. **No in-place JSON modification functions**: `JSON_SET`, `JSON_INSERT`, `JSON_REPLACE`, and `JSON_REMOVE` are not yet supported. JSON values can be read and extracted, but cannot be modified in place.
2. **No JSON path query functions**: `JSON_CONTAINS` and `JSON_CONTAINS_PATH` are not available.
3. **No JSON property indexing**: Indexes cannot be created on individual properties or paths within JSON documents.

---

## Architectural Root Causes

### 1. Representation & In-Memory Deserialization Cost
In [`src/core/value.rs`](file:///home/irshad/stoolap/src/core/value.rs), JSON is stored as `Value::Json(CompactString)` or raw strings. Every extraction (`->` or `->>`) parses the JSON text on demand into transient `serde_json::Value` trees.
Because there is no structured binary JSON container (such as JSONB or a parsed DOM arena), implementing `JSON_SET` or `JSON_REMOVE` requires:
- Parsing the entire text payload into a `serde_json::Value`.
- Mutating the tree in-place.
- Reserializing back to string format.
Without partial-tree mutators, chained updates cause significant CPU and memory allocation churn.

### 2. Missing JSONPath Bytecode Operators
The Expression VM in [`src/executor/expression/vm.rs`](file:///home/irshad/stoolap/src/executor/expression/vm.rs) is optimized for flat register-based scalar operations. It lacks dedicated opcodes for compiling and evaluating compiled JSONPath expressions (e.g. `$.users[*].id`), meaning any advanced query function (`JSON_CONTAINS`) would require custom runtime parsing per row.

### 3. Physical Index vs Expression Index Architecture
The indexing subsystem ([`src/storage/index/`](file:///home/irshad/stoolap/src/storage/index/)) is strictly physical: indexes are mapped to static column IDs (`usize`) in the table schema ([`src/core/schema.rs`](file:///home/irshad/stoolap/src/core/schema.rs)). There is currently no abstraction for **Generated/Virtual Columns** or **Expression Indexes** (e.g. `CREATE INDEX idx ON t ((payload->>'user_id'))`).

---

## Performance & System Impact

- **Analytical Query Overhead**: Queries filtering on nested JSON fields in large cold volumes (e.g., millions of rows) cannot utilize Zone Maps or Index scans. They fall back to full linear table scans where every row performs dynamic string parsing.
- **Write Amplification**: Updating a single property in a large JSON document requires a complete row rewrite in MVCC hot storage, inflating the Write-Ahead Log (WAL) and version store arena.

---

## Proposed Engineering Roadmap

### Phase 1: Implement JSON Scalar Mutation & Inspection Functions
- Add scalar functions to [`src/functions/scalar/utility.rs`](file:///home/irshad/stoolap/src/functions/scalar/utility.rs):
  - `json_set(json, path, value)`: Updates or inserts if path does not exist.
  - `json_insert(json, path, value)`: Inserts value only if path is missing.
  - `json_replace(json, path, value)`: Replaces value only if path exists.
  - `json_remove(json, path)`: Deletes key/index at target path.
  - `json_contains(target, candidate[, path])`: Verifies containment of sub-objects/arrays.
  - `json_contains_path(json, 'one'|'all', path1, path2...)`: Validates path existence.

### Phase 2: Expression / Functional Indexing
- Extend `IndexDef` in [`src/core/schema.rs`](file:///home/irshad/stoolap/src/core/schema.rs) to support expression-based key extractors.
- Update the write pipeline in [`src/storage/mvcc/table.rs`](file:///home/irshad/stoolap/src/storage/mvcc/table.rs) to evaluate the JSON extraction expression during row insertion and populate secondary B-Tree/Hash indexes.
- Update query optimizer index selection in [`src/executor/index_optimizer.rs`](file:///home/irshad/stoolap/src/executor/index_optimizer.rs) to recognize `payload->>'key' = ?` predicates matching registered functional indexes.

### Phase 3: Binary JSON Format (JSONB)
- Introduce a compact binary JSON format (similar to PostgreSQL JSONB or SQLite JSONB) stored directly in columnar cold volumes to eliminate serialization overhead and enable instant field seeking via byte offset headers.
