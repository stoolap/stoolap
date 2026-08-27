# 05. Views & Virtual Tables

## Subsystem Architecture Overview

Stoolap implements SQL Views as parameterized query definitions expanded inline during query compilation:

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Query: SELECT * FROM active_users_view WHERE score > 50                    │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────────┐
       │   Schema Catalog Resolution (src/storage/mvcc/registry.rs) │
       │   - Identifies 'active_users_view' as ViewDefinition       │
       │   - Extracts stored SQL definition string                  │
       └────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────────┐
       │   AST Inline Expansion & Substitution (src/executor/)      │
       │   - Parses view SQL into SubquerySource                    │
       │   - Increments recursion depth counter (depth <= 32)       │
       │   - Merges outer WHERE predicates into subquery plan       │
       └────────────────────────────────────────────────────────────┘
```

### Key Source References
- [`src/executor/ddl.rs`](file:///home/irshad/stoolap/src/executor/ddl.rs): Handles `CREATE VIEW` and `DROP VIEW` execution.
- [`src/storage/mvcc/registry.rs`](file:///home/irshad/stoolap/src/storage/mvcc/registry.rs): Stores view definitions alongside table schemas in the database catalog.
- [`src/executor/query.rs`](file:///home/irshad/stoolap/src/executor/query.rs): Resolves table sources and expands views into query plan subtrees.

---

## Known Limitations Breakdown

1. **Read-Only Views**: Views cannot be targeted by `INSERT`, `UPDATE`, or `DELETE` statements. Modifying data through views is strictly rejected.
2. **Shared Namespace**: View names and base table names share the same identifier namespace. You cannot create a view with the same name as an existing table.
3. **Nesting Depth Limit (32 Levels)**: Query expansion imposes a hardcoded ceiling of 32 view nesting levels to guard against infinite recursion and stack overflow from mutually referencing definitions.

---

## Architectural Root Causes

### 1. Absence of Reverse DML Projection & `INSTEAD OF` Triggers
Updatable views require bidirectional AST analysis:
- The engine must determine whether a view is **Key-Preserving** (i.e. every row in the view maps uniquely to exactly one row in a single base table without aggregations, `DISTINCT`, or group operations).
- In the absence of an `INSTEAD OF` trigger system or reverse DML mapping compiler, Stoolap restricts DML operators ([`src/executor/dml.rs`](file:///home/irshad/stoolap/src/executor/dml.rs)) strictly to physical table descriptors (`TableRef`).

### 2. Single Unified Schema Catalog
In [`src/storage/mvcc/registry.rs`](file:///home/irshad/stoolap/src/storage/mvcc/registry.rs), table descriptors and view definitions are indexed in a single `DashMap<String, CatalogEntry>`. This simplified lookup model avoids duplicate name collisions and eliminates the need for complex type disambiguation during planning, but enforces namespace exclusivity.

### 3. Iterative Expansion without Cycle Detection Graphs
View expansion occurs during top-down query AST traversal. Rather than constructing a directed acyclic graph (DAG) and running topological cycle detection during `CREATE VIEW`, the engine uses a runtime call counter (`depth > 32`).

---

## Performance & System Impact

- **Encapsulation Constraints**: Applications using views for row-level security or multi-tenant filtering cannot issue simple updates (e.g. `UPDATE tenant_users SET status = 'inactive' WHERE id = 5`); they must expose and target the underlying physical tables.
- **Deep Hierarchy Overhead**: Highly nested views (e.g. 15–25 levels in complex enterprise schemas) produce large expanded AST trees with redundant subquery projections, which increases query planning latency.

---

## Proposed Engineering Roadmap

### Phase 1: Simple Updatable Views (Key-Preserving)
- Implement AST analysis in [`src/executor/planner.rs`](file:///home/irshad/stoolap/src/executor/planner.rs) to detect simple single-table 1:1 projection views.
- Route `INSERT`, `UPDATE`, `DELETE` statements targeting updatable views directly to the underlying physical table while applying view `WHERE` constraints (support for `WITH CHECK OPTION`).

### Phase 2: `INSTEAD OF` Triggers
- Introduce trigger execution hooks allowing custom procedural logic or SQL blocks to execute in place of standard DML operations on views.

### Phase 3: DAG Cycle Validation at Schema Definition Time
- Validate view dependencies at `CREATE VIEW` time by traversing the catalog dependency graph, providing immediate cycle error diagnostics.
