# 09. Cold Storage Engine (Frozen Volumes), Compaction & Analytics

## Subsystem Architecture Overview

Stoolap implements a two-tier hybrid storage architecture:

```
                                 ┌─────────────────────────┐
                                 │   Write Path (DML)      │
                                 └───────────┬─────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             ┌───────────────────┐                       ┌───────────────────┐
             │ Write-Ahead Log   │                       │ Hot MVCC Buffer   │
             │ (Append-Only WAL) │                       │ (In-Memory Arena) │
             └───────────────────┘                       └─────────┬─────────┘
                                                                   │ (Seal Threshold Reached)
                                                                   ▼
                                                         ┌───────────────────┐
                                                         │ VolumeBuilder     │
                                                         └─────────┬─────────┘
                                                                   │ (Flush to Disk)
                                                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Cold Storage Tier (src/storage/volume/)                                                │
│                                                                                        │
│  ┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐     │
│  │ Frozen Volume 0       │   │ Frozen Volume 1       │   │ Frozen Volume 2       │     │
│  │ - Column Arrays (i64) │   │ - Column Arrays (i64) │   │ - Column Arrays (i64) │     │
│  │ - Dictionary (Text)   │   │ - Dictionary (Text)   │   │ - Dictionary (Text)   │     │
│  │ - Zone Maps (Min/Max) │   │ - Zone Maps (Min/Max) │   │ - Zone Maps (Min/Max) │     │
│  │ - Bloom Filters       │   │ - Bloom Filters       │   │ - Bloom Filters       │     │
│  └───────────────────────┘   └───────────────────────┘   └───────────────────────┘     │
│                                          ▲                                             │
│                                          │                                             │
│                       ┌──────────────────┴──────────────────┐                          │
│                       │ SegmentManager & manifest.json      │                          │
│                       │ (Versioned Tombstone Bitmaps)       │                          │
│                       └─────────────────────────────────────┘                          │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Key Source References
- [`src/storage/volume/table.rs`](file:///home/irshad/stoolap/src/storage/volume/table.rs): `VolumeTable` managing hot buffer coordination, seal passes, and scanner generation.
- [`src/storage/volume/column.rs`](file:///home/irshad/stoolap/src/storage/volume/column.rs): Column-major typed storage layout (`i64`, `f64`, `bool`, `string`, `timestamp`, `bytes`).
- [`src/storage/volume/writer.rs`](file:///home/irshad/stoolap/src/storage/volume/writer.rs): `VolumeBuilder` creating immutable frozen volume files.
- [`src/storage/volume/manifest.rs`](file:///home/irshad/stoolap/src/storage/volume/manifest.rs): `SegmentManager` handling atomic manifest swaps and compaction.
- [`src/storage/volume/scanner.rs`](file:///home/irshad/stoolap/src/storage/volume/scanner.rs): `MergingScanner` combining hot and cold sources.

---

## Known Limitations Breakdown

### 1. `AS OF` Point-in-Time Queries on Cold Rows
- Historical time-travel queries (`AS OF TRANSACTION` / `AS OF TIMESTAMP`) targeting cold volumes are not supported because frozen volumes store **only the latest sealed row states** without backward version delta chains.
- `AS OF CURRENT` works transparently across both hot and cold storage.

### 2. Compaction Memory Spikes
- The volume compactor ([`src/storage/volume/manifest.rs`](file:///home/irshad/stoolap/src/storage/volume/manifest.rs)) materializes the entire cold dataset in memory before sorting, applying tombstones, and rewriting a single consolidated volume.
- Similarly, parallel `GROUP BY` across 4+ volumes materializes one hash aggregate map per volume concurrently before reduction.

### 3. Skip-Set Cloning Overhead
- When scanning multiple cold volumes, the scanner builds per-volume skip sets (filtering rows deleted or superseded in hot storage) by cloning cumulative `row_id` hash sets. With $V$ volumes and $N$ deleted rows, this is $O(N \cdot V)$. Compaction maintains low volume counts (default threshold: 4 volumes).

### 4. Continuous Writes Delaying WAL Truncation
- WAL truncation requires all hot buffers to be fully sealed and empty. Under high-throughput continuous write traffic, newly arriving rows enter the hot buffer between the seal pass and the truncation check, postponing log truncation until a quiescent period or database shutdown.

### 5. Snapshot Transactions Throttling Seal Throughput
- Active long-running snapshot isolation transactions enforce a **cutoff-filtered seal**: only rows committed before the earliest snapshot's `begin_seq` can be frozen. Newer rows remain in hot memory, causing the hot buffer to expand proportionally to the write volume during the transaction's lifetime.

### 6. Multi-Column DISTINCT Scan Fallback
- `SELECT DISTINCT col1, col2 FROM table` on cold volumes cannot utilize per-column dictionary metadata and falls back to a full row scan and hash deduplication. Single-column `SELECT DISTINCT col1` directly extracts dictionary keys in $O(\text{unique})$ time.

### 7. Window Functions + LIMIT Full Materialization
- `ROW_NUMBER() OVER (...) LIMIT 10` on cold volumes materializes and sorts the entire partition/dataset before applying `LIMIT`. (Workaround: Use explicit `PARTITION BY` clauses to activate streaming window buffers).

### 8. Accepted Tradeoff: Binary Search Restricted to Integer/Timestamp
- Cold volume binary search is only implemented for sorted `Integer` and `Timestamp` columns (`i64`). Float, Text, and Boolean columns utilize linear scans accelerated by Zone Map (min/max) pruning.

---

## Architectural Root Causes

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Architectural Root Cause Analysis                                          │
├────────────────────────────────┬───────────────────────────────────────────┤
│ Problem                        │ Architectural Driver                      │
├────────────────────────────────┼───────────────────────────────────────────┤
│ Lack of AS OF on Cold Rows     │ Storing version deltas on disk would      │
│                                │ convert column-major format into complex  │
│                                │ append trees, degrading OLAP SIMD speed.  │
├────────────────────────────────┼───────────────────────────────────────────┤
│ Compaction Memory Spikes       │ Non-streaming compaction simplifies       │
│                                │ dictionary unification and sorting.       │
├────────────────────────────────┼───────────────────────────────────────────┤
│ WAL Truncation Delay           │ Monolithic WAL journal shared across all  │
│                                │ tables rather than per-table segment logs.│
├────────────────────────────────┼───────────────────────────────────────────┤
│ Multi-Column DISTINCT Fallback │ Dictionaries are column-isolated; no      │
│                                │ cross-column joint dictionary encoding.   │
└────────────────────────────────┴───────────────────────────────────────────┘
```

---

## Performance & System Impact

- **Memory Pressure in High-Volume Ingestion**: Ingestion spikes combined with long-running analytical queries can cause substantial RAM consumption due to unsealed hot buffers and in-memory compaction buffers.
- **OLAP Query Latency**: Analytical queries combining multi-column distinct calculations or unbounded window functions on large tables experience higher latency and memory allocations.

---

## Proposed Engineering Roadmap

### Phase 1: Streaming Multi-Way Merge Compactor
- Refactor the compactor in [`src/storage/volume/manifest.rs`](file:///home/irshad/stoolap/src/storage/volume/manifest.rs) into a **Streaming K-Way Merge Iterator**.
- Process rows in fixed chunks (e.g. 64K rows at a time) using a bounded priority queue, bounding compaction memory usage to $O(K \times \text{chunk\_size})$.

### Phase 2: Roaring Bitmap Skip-Sets
- Replace `HashSet<i64>` skip sets with compressed **Roaring Bitmaps** (`roaring::RoaringBitmap`).
- Enable zero-copy bitmap intersection and $O(1)$ cloning across scanner iterators.

### Phase 3: Segmented Segment-Level WAL Architecture
- Decouple the monolithic WAL into table-scoped or segment-scoped log segments.
- Enable independent log segment recycling immediately when individual table hot buffers are sealed to disk.

### Phase 4: Streaming Top-N Window Operator
- Implement an early-stopping bounded binary heap operator for `WindowFunction + LIMIT` queries to eliminate full-table materialization.
