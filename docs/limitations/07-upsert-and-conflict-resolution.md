# 07. Upsert Execution (`ON CONFLICT` / `ON DUPLICATE KEY`)

## Subsystem Architecture Overview

Stoolap handles upserts (insert-or-update operations) across both standard DML execution pipelines and optimized fast paths:

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Query: INSERT INTO metrics (id, count) VALUES (1, 10)                      │
│        ON CONFLICT (id) DO UPDATE SET count = metrics.count + EXCLUDED.count│
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────────┐
       │   Conflict Detection (src/executor/dml.rs)                 │
       │   - Attempts PK/Unique index lookup                        │
       │   - Detects collision with existing row_id                 │
       └────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────────┐
       │   Expression Evaluation (src/executor/expression/ops.rs)   │
       │   - Binds target row values to 'table_name.col'            │
       │   - Binds incoming row values to 'EXCLUDED.col'            │
       │   - Computes updated row vector                            │
       └────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────────┐
       │   Hot MVCC Update (src/storage/mvcc/table.rs)              │
       │   - Commits new row version in Arena                       │
       └────────────────────────────────────────────────────────────┘
```

### Key Source References
- [`src/executor/dml.rs`](file:///home/irshad/stoolap/src/executor/dml.rs): Coordinates conflict detection, `EXCLUDED` scope binding, and update execution.
- [`src/executor/dml_fast_path.rs`](file:///home/irshad/stoolap/src/executor/dml_fast_path.rs): Fast-path single-row upsert executor.
- [`src/parser/`](file:///home/irshad/stoolap/src/parser/): Parses `ON CONFLICT (...) DO UPDATE` and `ON DUPLICATE KEY UPDATE` clauses.

---

## Known Limitations Breakdown

1. **No MySQL `VALUES(column)` Reference Syntax**: In upsert update expressions, referencing the proposed incoming values requires PostgreSQL-style `EXCLUDED.column`. MySQL's legacy `VALUES(col)` expression syntax is not supported in `ON DUPLICATE KEY UPDATE`.
2. **No `WHERE` Filter on Conflict Action**: PostgreSQL-style conditional updates (`ON CONFLICT (id) DO UPDATE SET ... WHERE condition`) are not yet implemented. All conflicting rows unconditionally execute the `DO UPDATE` expression.

---

## Architectural Root Causes

### 1. Unified Expression Scope Resolution
Stoolap unifies `ON CONFLICT` and `ON DUPLICATE KEY UPDATE` into a single AST representation. To avoid ambiguity with the SQL `VALUES (...)` table constructor keyword, the expression binder treats `EXCLUDED` as a virtual table alias in the symbol table. MySQL's `VALUES(col)` is treated as a function call syntax, which collides with standard SQL scalar function resolution.

### 2. DML Fast Path Architecture
The DML pipeline in [`src/executor/dml_fast_path.rs`](file:///home/irshad/stoolap/src/executor/dml_fast_path.rs) is optimized for microsecond-latency transactional point writes:
```rust
// Current execution flow in dml.rs:
if let Some(conflict_row) = index.get(&key) {
    let updated_row = evaluate_update_assignments(conflict_row, excluded_row)?;
    table.update(conflict_row.id, updated_row)?;
}
```
Inserting a conditional `WHERE` clause requires compiling and executing a filter predicate against `(conflict_row, excluded_row)` before applying updates, which was deferred to keep the fast path minimal.

---

## Performance & System Impact

- **Idempotent Ingestion Filtering**: Workloads that want to skip updates when incoming data is stale (e.g. `DO UPDATE SET val = EXCLUDED.val WHERE EXCLUDED.version > metrics.version`) must execute two queries (a preliminary `SELECT` followed by conditional `INSERT`/`UPDATE`) or implement complicated `CASE WHEN` logic in the `SET` clause.
- **Migration Compatibility**: Tooling and ORMs (e.g. TypeORM, Sequelize, Prisma) configured for MySQL upsert syntax may require dialect adjustments to generate `EXCLUDED.` identifiers.

---

## Proposed Engineering Roadmap

### Phase 1: Conditional `WHERE` Predicate Support
- Extend `OnConflictClause` AST in [`src/parser/`](file:///home/irshad/stoolap/src/parser/) to include `pub where_clause: Option<Expression>`.
- In [`src/executor/dml.rs`](file:///home/irshad/stoolap/src/executor/dml.rs), evaluate `where_clause` using the combined `(current_row, excluded_row)` environment. If the predicate evaluates to `false` or `null`, skip the update cleanly.

### Phase 2: MySQL `VALUES(col)` Expression Alias
- Update the parser and expression compiler to recognize `VALUES(ident)` within `ON DUPLICATE KEY UPDATE` contexts, rewriting it automatically to `EXCLUDED.ident`.
