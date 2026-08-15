# 10. Transaction Model & Primary Key Mutability

## Subsystem Architecture Overview

Stoolap’s MVCC storage engine tightly couples logical primary keys with physical internal row identifiers (`row_id`):

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Query: UPDATE users SET id = 100 WHERE id = 10;                            │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────────┐
       │   DML Executor Validation (src/executor/dml.rs)            │
       │   - Checks assignment target columns against PK columns    │
       │   - Identifies PK mutation attempt                          │
       │   - REJECTS WITH ERROR: "Cannot update primary key column" │
       └────────────────────────────────────────────────────────────┘
```

### Key Source References
- [`src/storage/index/pk.rs`](file:///home/irshad/stoolap/src/storage/index/pk.rs): Implements primary key index structures mapping `pk_value` $\leftrightarrow$ `row_id`.
- [`src/storage/mvcc/table.rs`](file:///home/irshad/stoolap/src/storage/mvcc/table.rs): `Table` implementation enforcing immutable `row_id` identity across version arena slots.
- [`src/executor/dml.rs`](file:///home/irshad/stoolap/src/executor/dml.rs): Rejects `UPDATE` statements containing primary key assignments.

---

## Known Limitations Breakdown

1. **No Primary Key Updates**: Executing an `UPDATE` statement targeting primary key columns is explicitly prohibited and returns a runtime error.
2. **Mandatory Workaround**: To modify a primary key, the application must execute an explicit `DELETE` of the existing row followed by an `INSERT` of the new row with updated primary key values.

---

## Architectural Root Causes

### 1. The `row_id == pk_value` Invariant
In Stoolap's storage architecture:
- **Integer Primary Keys**: For single-column integer primary keys, `row_id` is identically set to the primary key's numerical value.
- **Secondary Index Design**: All secondary B-Tree, Hash, and HNSW vector indices store pointers to `row_id`.
- **MVCC Version Store**: The version store arena ([`src/storage/mvcc/version_store.rs`](file:///home/irshad/stoolap/src/storage/mvcc/version_store.rs)) chains row deltas using fixed `row_id` arena slots.
- **Cold Volume Tombstones**: Deletions and updates in cold storage are tracked using bitmap offsets directly corresponding to `row_id`.

If a row's `row_id` were mutated in-place during an `UPDATE`:
- The previous `row_id` version chain in the MVCC arena would be orphaned or severed.
- All secondary index pointers for that row would become invalid or point to stale memory slots.
- Cold volume tombstones referencing the previous `row_id` would no longer reflect the updated entity.

---

## Performance & System Impact

- **Developer Friction & Transaction Scope**: Applications performing primary key updates (e.g. migrating customer ID schemes or reordering sequential keys) must wrap `DELETE` + `INSERT` operations in an explicit multi-statement transaction (`BEGIN ... COMMIT`) to guarantee atomicity.
- **Foreign Key Cascade Overhead**: Without automated primary key rewrite support, foreign keys configured with `ON UPDATE CASCADE` cannot automatically propagate primary key modifications down child tables.

---

## Proposed Engineering Roadmap

### Phase 1: Transparent DML Decomposition in Executor
- Update [`src/executor/dml.rs`](file:///home/irshad/stoolap/src/executor/dml.rs) to detect primary key updates and automatically decompose them into an atomic sequence of internal operations:
  1. Retrieve the existing row state and verify unique constraints for the new key.
  2. Execute foreign key `ON UPDATE CASCADE` triggers on dependent tables.
  3. Mark the old `row_id` as deleted (tombstoned in MVCC / cold manifest).
  4. Insert a new row with the updated primary key and allocate a new `row_id`.
- Execute this decomposition within the same atomic transaction commit sequence.

### Phase 2: Decoupled Surrogate `row_id` Architecture
- Transition from natural `row_id == pk` binding to a decoupled 64-bit monotonic surrogate `row_id` system.
- Maintain a primary index mapping `PK_Value -> Surrogate_Row_ID`, enabling primary key updates to simply re-key the index without altering the physical `row_id` identity in the version arena.
