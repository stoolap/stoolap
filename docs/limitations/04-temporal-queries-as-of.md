# 04. Temporal Queries (`AS OF`) & History Pruning

## Subsystem Architecture Overview

Stoolap implements temporal "time-travel" queries using its Multi-Version Concurrency Control (MVCC) version chain architecture:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Query: SELECT * FROM balances AS OF TIMESTAMP '2026-04-20 10:00:00' WHERE user_id = 42 │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────────┐
       │   Snapshot Resolver (src/storage/mvcc/snapshot.rs)         │
       │   - Maps physical timestamp to highest commit_seq <= ts    │
       │   - Builds ReadSnapshot { read_seq: 1054 }                 │
       └────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────────┐
       │   MVCC Version Chain Traversal (src/storage/mvcc/)         │
       │   - Traverses row version pointers backward in Arena       │
       │   - Selects version where created_seq <= 1054 < expired_seq│
       └────────────────────────────────────────────────────────────┘
```

### Key Source References
- [`src/storage/mvcc/version_store.rs`](file:///home/irshad/stoolap/src/storage/mvcc/version_store.rs): Maintains backward delta chains for updated and deleted rows.
- [`src/storage/mvcc/snapshot.rs`](file:///home/irshad/stoolap/src/storage/mvcc/snapshot.rs): Creates temporal snapshots based on transaction IDs or timestamps.
- [`src/storage/mvcc/arena.rs`](file:///home/irshad/stoolap/src/storage/mvcc/arena.rs): Memory arena allocating version structs.
- [`src/executor/subquery.rs`](file:///home/irshad/stoolap/src/executor/subquery.rs): Evaluates subqueries and scalar CTEs.

---

## Known Limitations Breakdown

1. **No Subqueries with `AS OF`**: `AS OF` clauses cannot be combined with subqueries (e.g. `SELECT * FROM (SELECT * FROM t) AS OF '2026-04-20 10:00:00'` or subqueries inside an `AS OF` query's WHERE clause).
2. **System Clock Dependency**: Resolution and ordering of `AS OF TIMESTAMP` queries rely on system clock precision and monotonicity. Clock skew or NTP backward adjustments can cause temporal inconsistencies.
3. **`VACUUM` History Purge**: Executing `VACUUM` permanently reclaims all historical row versions not pinned by currently active transactions. After a `VACUUM`, `AS OF` queries targeting timestamps prior to the vacuum operation fail to find historical versions.

---

## Architectural Root Causes

### 1. Snapshot Context Scoping in Query Planner
The query planner ([`src/executor/planner.rs`](file:///home/irshad/stoolap/src/executor/planner.rs)) and subquery evaluator ([`src/executor/subquery.rs`](file:///home/irshad/stoolap/src/executor/subquery.rs)) instantiate distinct execution contexts. The temporal snapshot modifier is attached specifically to individual table scan operators rather than propagating through recursive AST expression subtrees. When a subquery or derived table is encountered, the parent temporal snapshot context is not automatically inherited by child plan nodes.

### 2. Physical Clock vs Hybrid Logical Clock (HLC)
Stoolap records timestamps using `chrono::Utc::now()`. It lacks a **Hybrid Logical Clock (HLC)** or centralized Lamport timestamp coordinator. If the host system clock steps backward or fluctuates, multiple transactions might share identical physical timestamps despite having distinct commit sequence numbers (`commit_seq`).

### 3. Eager In-Memory Garbage Collection
To prevent unbounded memory growth in memory-constrained environments, the version arena ([`src/storage/mvcc/arena.rs`](file:///home/irshad/stoolap/src/storage/mvcc/arena.rs)) reclaims dead versions when `VACUUM` is triggered. Because version deltas reside in hot memory and are not archived to cold volumes during compaction, running `VACUUM` severs older version links.

---

## Performance & System Impact

- **Auditing & Compliance Fragility**: Point-in-time auditing relying on `AS OF` can be inadvertently invalidated if administrative maintenance jobs trigger `VACUUM` or `PRAGMA vacuum`.
- **Complex Analytical Queries**: Financial queries requiring temporal joins between a historical balance and a subquery calculation cannot be expressed in a single SQL statement.

---

## Proposed Engineering Roadmap

### Phase 1: Propagate Temporal Snapshot Across Query Plan Tree
- Modify `ExecutionContext` in [`src/executor/context.rs`](file:///home/irshad/stoolap/src/executor/context.rs) to maintain an active `temporal_snapshot: Option<ReadSnapshot>`.
- Ensure all nested subqueries, derived tables, and CTE scans automatically inherit the parent query's temporal snapshot unless overridden.

### Phase 2: Hybrid Logical Clock (HLC) Integration
- Replace raw wall-clock calls with a monotonic Hybrid Logical Clock coordinator.
- Guarantee strict causal ordering of `commit_seq` and physical timestamp mappings even during NTP adjustments.

### Phase 3: Configurable History Retention Policies
- Introduce retention window configurations (e.g. `PRAGMA time_travel_retention = '7 days'`).
- Update `VACUUM` to respect the retention cutoff timestamp, pruning only version deltas older than `now() - retention_window`.
