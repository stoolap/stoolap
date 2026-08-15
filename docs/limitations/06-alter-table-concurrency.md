# 06. ALTER TABLE, Concurrent Writes & Schema Adaptation

## Subsystem Architecture Overview

Stoolap manages schema changes across a hybrid storage model containing mutable hot MVCC memory buffers and immutable cold disk volumes:

```
┌────────────────────────────────────────────────────────────────────────────┐
│ DDL Command: ALTER TABLE users ADD COLUMN bio TEXT DEFAULT ''              │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────────┐
       │   Schema Lock Acquisition (src/executor/ddl.rs)            │
       │   - Acquires exclusive write lock on TableCatalogEntry     │
       │   - Blocks incoming DML write transactions                 │
       └────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────────┐
       │   Schema Version Update (src/storage/mvcc/table.rs)        │
       │   - Updates in-memory Schema definition                    │
       │   - Hot Buffer immediately reflects new column layout      │
       └────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────────┐
       │   Cold Volume Normalization (src/storage/volume/scanner.rs)│
       │   - On-disk Frozen Volumes remain unchanged                │
       │   - MergingScanner injects default values on-the-fly       │
       └────────────────────────────────────────────────────────────┘
```

### Key Source References
- [`src/executor/ddl.rs`](file:///home/irshad/stoolap/src/executor/ddl.rs): Implements `ALTER TABLE` operations (Add, Drop, Rename Column).
- [`src/storage/volume/scanner.rs`](file:///home/irshad/stoolap/src/storage/volume/scanner.rs): Implements `MergingScanner` schema projection and normalization between cold volume physical layouts and current logical schemas.
- [`src/storage/volume/table.rs`](file:///home/irshad/stoolap/src/storage/volume/table.rs): Coordinates hot/cold table representation.

---

## Known Limitations Breakdown

1. **Write Blocking during DDL**: `ALTER TABLE` operations acquire an exclusive table-level write lock, temporarily blocking concurrent `INSERT`, `UPDATE`, and `DELETE` transactions.
2. **No Composite Primary Key Modifications**: Modifying existing composite primary keys via `ALTER TABLE` is unsupported.
3. **Cold Volume Schema Divergence & Space Lag**: `ALTER TABLE` only mutates the active hot schema. Cold frozen volumes retain their historical on-disk schema. While added and renamed columns are adapted on-the-fly during scan time, `DROP COLUMN` does not immediately reclaim physical disk space until a full background compaction cycle completes.

---

## Architectural Root Causes

### 1. Table-Level Exclusive Locking
To prevent schema race conditions where an active write inserts a row conforming to Schema Version $N$ while DDL transitions the catalog to Version $N+1$, the DDL executor acquires an exclusive `parking_lot::RwLock` write guard over the table instance. In high-throughput write workloads, this can cause a transient spike in transaction wait queues.

### 2. Invariant: `row_id` Direct Mapping to Primary Key
Stoolap uses `row_id` as the fundamental lookup coordinate in the hot MVCC index ([`src/storage/index/pk.rs`](file:///home/irshad/stoolap/src/storage/index/pk.rs)) and volume tombstone bitmasks. Modifying composite primary key definitions requires recalculating and restructuring primary key indices and tombstone maps across both hot memory and immutable cold disk files.

### 3. Immutable Frozen Volumes Design
Cold volumes ([`src/storage/volume/`](file:///home/irshad/stoolap/src/storage/volume/)) are strictly append-only and immutable once sealed. Changing on-disk column offsets immediately upon `ALTER TABLE` would require rewriting gigabytes of cold data synchronously. Instead, Stoolap delegates reconciliation to the scanner layer:

```rust
// In src/storage/volume/scanner.rs:
// If table schema contains columns not present in cold volume,
// supply default Value on-the-fly during iteration.
```

Physical space reclamation for dropped columns is therefore deferred to the compaction engine.

---

## Performance & System Impact

- **Scan Latency Jitter**: Reading cold volumes that have undergone multiple schema migrations incurs slight CPU overhead due to on-the-fly index remap tables and null/default value synthesis.
- **Disk Space Amplification**: Dropping large text or vector columns from tables with multiple gigabytes in cold storage will not reduce disk usage until compaction is triggered.

---

## Proposed Engineering Roadmap

### Phase 1: Lock-Free Online Schema Change (OSC)
- Implement schema version tagging on write transactions.
- Use atomic catalog swaps (`ArcSwap<Schema>`) so readers and writers continue concurrently without global table locks.

### Phase 2: Explicit Volume Schema Descriptors
- Store a lightweight `SchemaHeader` in each volume file header.
- Allow `MergingScanner` to use zero-cost vectorized projection masks instead of per-row column index remapping.

### Phase 3: Targeted Compaction on Dropped Columns
- Enhance the background compactor ([`src/storage/volume/manifest.rs`](file:///home/irshad/stoolap/src/storage/volume/manifest.rs)) to prioritize rewriting volumes with large dropped column overhead.
