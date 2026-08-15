# 02. Foreign Key Integrity & Multi-Column Constraints

## Subsystem Architecture Overview

Stoolap enforces relational integrity using foreign key constraint descriptors attached to table schemas:

```
┌───────────────────────────────────────────────────────────────────────────┐
│ DML Execution: INSERT INTO orders (order_id, user_id) VALUES (101, 42)    │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────────┐
       │   ForeignKeyChecker (src/executor/foreign_key.rs)          │
       │   - Resolves target table metadata                         │
       │   - Verifies target primary key index contains 42          │
       └────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────────┐
       │   Hot MVCC Insert (src/storage/mvcc/table.rs)              │
       │   - Acquires row_id, updates version arena & WAL           │
       └────────────────────────────────────────────────────────────┘
```

### Key Source References
- [`src/core/schema.rs`](file:///home/irshad/stoolap/src/core/schema.rs): Defines `ForeignKeyConstraint` storing `column: String`, `referenced_table: String`, and `referenced_column: String`.
- [`src/executor/foreign_key.rs`](file:///home/irshad/stoolap/src/executor/foreign_key.rs): Implements `ForeignKeyChecker`, coordinating parent existence checks during `INSERT`/`UPDATE` and child restriction/cascades on `DELETE`.
- [`src/executor/dml.rs`](file:///home/irshad/stoolap/src/executor/dml.rs): Invokes constraint checks immediately prior to row version allocation.

---

## Known Limitations Breakdown

1. **Single-Column Only**: Foreign keys cannot reference composite keys (multiple columns). For example, `FOREIGN KEY (tenant_id, user_id) REFERENCES users (tenant_id, id)` is rejected at parse/schema creation time.
2. **Self-Referencing Insertion Order**: Tables with self-referencing foreign keys (e.g. `parent_id` referencing `id` in the same table) require sequential, topologically ordered inserts. Batch inserts inserting child rows before their parent rows in the same transaction fail immediately.

---

## Architectural Root Causes

### 1. Schema & Index Modeling Limitations
The `ForeignKeyConstraint` struct in [`src/core/schema.rs`](file:///home/irshad/stoolap/src/core/schema.rs) is structurally modeled as 1-to-1 column mappings:
```rust
pub struct ForeignKeyConstraint {
    pub name: String,
    pub column: String,                // Single column string
    pub referenced_table: String,
    pub referenced_column: String,     // Single referenced column string
    pub on_delete: ForeignKeyAction,
    pub on_update: ForeignKeyAction,
}
```
Composite foreign key validation requires matching a composite tuple `(v1, v2, ...)` against a multi-column composite index (`MultiColumnIndex` in [`src/storage/index/multi_column.rs`](file:///home/irshad/stoolap/src/storage/index/multi_column.rs)). Currently, `ForeignKeyChecker` relies purely on single-value lookups.

### 2. Immediate vs Deferred Constraint Evaluation in MVCC
Stoolap checks foreign key constraints **eagerly and row-by-row** during statement execution. When inserting multiple rows in a batch statement (`INSERT INTO categories (id, parent_id) VALUES (2, 1), (1, NULL)`):
- Row `(2, 1)` is checked against the database state before Row `(1, NULL)` has been committed to the hot MVCC table.
- Because Stoolap lacks **Deferred Constraint Validation** (evaluating integrity at transaction `COMMIT` time or post-statement), the entire batch fails.

---

## Performance & System Impact

- **Schema Normalization Hurdles**: Multi-tenant architectures and data warehouse star/snowflake schemas that rely on composite surrogate keys `(tenant_id, entity_id)` cannot enforce database-level referential integrity.
- **Bulk Migration Bottlenecks**: Loading hierarchical datasets (org charts, tree hierarchies) requires developers to pre-sort data topologically in the application layer before issuing chunked inserts.

---

## Proposed Engineering Roadmap

### Phase 1: Composite Foreign Key Schema & Executor Support
- Refactor `ForeignKeyConstraint` in [`src/core/schema.rs`](file:///home/irshad/stoolap/src/core/schema.rs) to accept `Vec<String>` for `columns` and `referenced_columns`.
- Update [`src/parser/`](file:///home/irshad/stoolap/src/parser/) to parse `FOREIGN KEY (col1, col2) REFERENCES parent (col1, col2)`.
- Update `ForeignKeyChecker` in [`src/executor/foreign_key.rs`](file:///home/irshad/stoolap/src/executor/foreign_key.rs) to perform composite key tuple searches against `MultiColumnIndex`.

### Phase 2: Deferred Constraint Checking for Transactions & Batches
- Add a deferred validation queue inside `ExecutionContext` ([`src/executor/context.rs`](file:///home/irshad/stoolap/src/executor/context.rs)).
- Support `SET CONSTRAINTS ALL DEFERRED` or enable statement-level deferral: evaluate all pending foreign key references against the active transaction's private write-set before finalization.
- Implement an in-memory topological cycle detector for self-referential graph batches.
