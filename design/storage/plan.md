# Stoolap storage plan: HTAP lifecycle, hot store, cold layout

Single authoritative plan, revision 3. Replaces `hot-store-redesign.md` (archived) and the interim HTAP document.
Base commit `b227acbd`. Source line references describe that commit. Implementation starts at `9cf15ea4`.

The concrete protocols in [contracts.md](contracts.md) refine this accepted plan; [progress.md](progress.md) records implementation, review and PR status. The contracts govern implementation where they explicitly refine a mechanism below.

Revision 3 answers two review rounds. The first found eleven correctness gaps in the merge of the two earlier plans; the second found three more, plus two rollout dependencies. Its central change: **section 4 holds eight contracts (K1 to K8) that must be closed as design work before any core implementation starts.** They are not patches to be applied later; several of them decide the shape of the code.

## 0. Thesis and acceptance rule

Stoolap runs a **row-based transaction engine** and a **column-based analytic engine over the same data**. The point of this plan is that the two complete each other instead of competing for memory, latency and correctness.

Acceptance rule: **improving the columnar side must not cost the row side its latency.** A change that speeds up analytics while lengthening the tail of INSERT, UPDATE or point lookup has failed, whatever the analytic numbers say.

| Layer | Representation | Primary duty |
|---|---|---|
| Hot | Mutable rows, MVCC versions, secondary indexes | INSERT, UPDATE, point lookup, recent data |
| Warm | Immutable column blocks held in memory | Repeated analytic queries |
| Cold | Immutable column blocks on disk | A large history at low memory cost |

Row versus column is a **representation** decision made at seal. Warm versus cold is a **residency** decision made by access. Today both are tied to the volume and to a checkpoint counter, which is why neither holds.

## 1. Status: what is already shipped

| PR | Commit | What it changed | Measured |
|---|---|---|---|
| #107 | 7a723370 | Decoded row-group columns in a byte-budgeted process cache (`DECODED_GROUPS`, 64 MB default, `PRAGMA GROUP_CACHE_MB`) | warm repeats stop re-decoding |
| #108 | 572e28fb | Top-k scan reads an added column's DEFAULT from older volumes | correctness |
| #109 | a8d93d4d | A series' latest rows by an ordered walk of the multi-column index (`walk_prefix_ordered`, per-group `walk_orders`); publish protocol (`PublishGuard`, `PublishHold`, `IndexUndo`) | pair latest 36 ms to 0.01 ms; 174-pair fan-out 2.1 s to 27 ms |
| #110 | 15e6547f | Seal removes hot index entries by row id (`Index::remove_batch_ids`, shared `subtract_sorted`); the index-cleanup CowBTree snapshot is gone | 2M-row seal 4.1 s to 2.5 s |
| #111 | 9856c93c | `hot_max_rows` (262,144) and `hot_max_bytes` **seal triggers with an admission wait**, exact arena byte counter, `PRAGMA MEMORY_STATS`, seal request raised only after the commit is visible | bulk load: final checkpoint 3.3 s to 0.8 s |
| #112 | b227acbd | Dictionary equality over a row group's raw ids in one pass (`ColumnData::dict_matching_offsets`), selectivity cap kept per group | 7-day pair aggregation 15 to 11 ms; full history 104 to 93 ms; COUNT 68 to 61 ms at 4M rows |
| #113 | 060681d9 | Documentation for the hot limits, the group cache and `MEMORY_STATS` | commit present locally |

**What #111 is and is not.** It shipped seal triggers and a commit-time admission wait measured on committed arena bytes. It is **not** a hard bound over all retained allocations: uncommitted writes, previous versions, chunk capacity, index ownership and residual data are not yet inside one admission decision. `extract_for_seal` still uses `ExtractionSnapshot`. Calling the hot store "bounded" today would be too strong.

These PRs did not deliver: bounded seal memory, block-level residency, proportional cold reads, or an account covering the whole engine. That is the rest of this plan.

## 2. The engine today, verified

**Row side.** Per table a `VersionStore`: MVCC version chains, a row arena, hot-only secondary indexes. Writers commit under a shared seal fence.

**Column side.** The checkpoint cycle seals hot rows into immutable volumes: about 1,048,576 rows, fixed 65,536-row groups (`ROW_GROUP_SIZE`, column.rs:108), one LZ4 block per column per group, zone maps per group (column.rs:114) and per volume, blooms at 10 bits per row per column, a volume-wide TEXT dictionary. Physical order is row id order; `VolumeBuilder::finish` (writer.rs:1439) enforces it and silently sorts `row_ids` alone if the input is unsorted (writer.rs:1530), which would detach identity from payload.

**Where the engines meet.**
- Scans merge hot and cold under one snapshot, and there are **two different skip rules in the code today**. `scan` derives its skip set from the hot rows it actually returned, which are filtered by the predicate and by visibility (volume/table.rs:1935-1947); this is why a snapshot that began before a cold row was updated still reads the old cold value, verified by probe on scan, COUNT, GROUP BY and a filtered read. `collect_all_rows` does the same (volume/table.rs:1968-1980). The raw-membership rule (`collect_hot_row_ids_into`, version_store.rs:1871) is used by the specialised paths instead, about eighteen sites on this commit, among them the cold unique validation (volume/table.rs:277), the limited collect (2117), the partition scans (3223, 3293, 3480), the ordered index merge (3731) and the aggregate pushdowns, and most of those bail out under snapshot isolation. Neither rule is the invariant: results are wrong once a hot copy can disappear asynchronously (a committed hot DELETE returns no row), membership is wrong for snapshots. K4 defines what replaces both.
- Ordered scans visit volumes and groups in the order of the queried column's bound and stop when the remaining bound cannot improve the heap (`cannot_improve`, volume/table.rs:4120). Ingest order is never assumed to be value order.
- Partial aggregation across both engines exists: typed accumulators with `merge_accum` (volume/table.rs:4717, 5548; version_store.rs:6070), combined per table (volume/table.rs:4610, 5492).
- Cold constraint checks are on the INSERT path: `check_segment_constraints` (volume/table.rs:557), per UNIQUE index a per-volume lookup (manifest.rs:1115) over zone maps, blooms and a per-volume sorted hash index built over the whole volume on first use (writer.rs:1804), prebuilt at seal and compaction to avoid stalling the first INSERT (engine.rs:5693, 5981).

**Facts that constrain the design.**
- `remove_tombstones_for_rows` (manifest.rs:1723) removes tombstones **unconditionally by row id**. Today the long fence makes that safe.
- Version chains keep at most `max_version_history` entries; past the limit the whole previous chain is dropped (version_store.rs:802).
- Point lookup acquires and drops the arena read guard **before** the B-tree fallback, deliberately, because the commit path holds `versions.write` then `arena.write` (version_store.rs:1218-1232).
- Commit skips an index entirely when none of its columns changed, on two paths (version_store.rs:7213, 7485).
- WAL recovery skips any entry larger than 64 MiB (wal_manager.rs:1777, 1914).
- WAL replay resolves a table **by name** and skips a missing store (engine.rs:1776); RENAME moves the same store to the new name (engine.rs:3741).
- WAL truncation already requires that all manifests persisted successfully (engine.rs:5183).
- `PRAGMA HOT_MAX_ROWS` accepts 0 (trigger off) and any positive value, including values far above 262,144 (query.rs:9786).

## 3. What is wrong, measured

Workload: one-minute candles, (id AUTO_INCREMENT PK, time TIMESTAMP, exchange TEXT, symbol TEXT, open, high, low, close, volume FLOAT), UNIQUE(exchange, symbol, time), 174 pairs, 15.7M rows in 15 volumes.

- **Placement.** In row id order one key's rows are spread through every group of every volume. On a 4M-row probe: per-pair aggregation of the whole history 82 to 104 ms, per-pair COUNT 61 to 68 ms, 83% in LZ4 group decoding. Live, at 15.7M rows, about 1 s per pair.
- **Residency.** The decoded working set is 153 MB against a 64 MB budget, so nothing survives between queries; at 512 MB the same COUNT is 6.4 ms and the aggregation 31 ms on repeat. Whole-column access escapes the budget entirely.
- **Hot to columnar transition.** A seal materialises the whole decoded volume, then all compressed blocks, then a whole-file buffer (io.rs:142-155). At 73 bytes per row that is about 73 MiB for a 1M-row volume against a 64 MiB default budget.
- **Writer stalls.** Removal of a sealed table's rows holds the per-table write fence for its whole duration (about 2.5 s at 2.175M rows after #110).
- **Two-layer execution falls back to rows where it matters most**: under snapshot isolation (volume/table.rs:4616, 5497), during a seal overlap (4621, 5512), for multi-column GROUP BY (5501), for filters that are not conjunctive-simple (4628).
- **Memory has no owner.** Arena capacity is never returned after a seal; volume tiers are counted separately from the group cache; nothing accounts for maintenance.

## 4. Contracts to close before core implementation

These are design deliverables. Each states the rule, the code it must hold against, and the test that proves it. **Phase 0 of the roadmap is closing them.**

### K1. Chunk addressing is fixed; the hot limit stays a policy

Physical chunk capacity is a **compile-time constant** (`ARENA_CHUNK_ROWS`, 262,144 = 1 << 18) and the address encoding `chunk << 18 | offset` is derived from that constant alone. `hot_max_rows` remains a runtime policy threshold: 0 disables the trigger and any positive value is legal, including values far above the chunk size (query.rs:9786). Tying capacity to the pragma would let `(chunk 0, offset 262144)` and `(chunk 1, offset 0)` collapse onto the same address.

Chunk ids are monotonic and never reused; dropping an empty chunk does not shift the ids of later chunks. Any handle that outlives its chunk resolves to "gone", never to a different chunk.
*Tests:* `hot_max_rows` of 0, of 1,000, of 1,048,576, and a change at runtime while chunks exist; a stale arena index after its chunk was dropped.

### K2. The seal captures a fixed eligibility bound at build start

At build start the sealer captures a finite bound: the minimum snapshot begin sequence when one exists, otherwise the registry's current sequence at that moment. **Every batch of that build uses the same bound.** The absence of a snapshot must never mean "everything committed at any later time", because a frozen chunk stays mutable: a snapshot opened mid-build, followed by an UPDATE, would otherwise put the new value into a later batch while the remover deletes the chain the old snapshot needs.

The registry must retain what the bound needs for the build's duration.
*Test:* open a snapshot and commit an UPDATE between two batches of one build; the old snapshot must still read the old value after the remover has run.

### K3. Tombstone transfer is atomic with the row transfer

Visibility change and removal of a superseded tombstone happen **in the same critical section as the row transfer**, per sub-batch. Only the disk serialization is deferred to the end of the pass.

Removal must be **sequence-matched**: remove only a tombstone whose commit sequence equals the one recorded when the row was transferred. Today's `remove_tombstones_for_rows` (manifest.rs:1723) removes unconditionally by row id, which is safe only under the long fence. With a per-sub-batch fence, a DELETE committed between the transfer and the end of the pass would have its brand-new tombstone deleted by the pass-end cleanup, and the row would come back.
*Test:* commit a cold DELETE for a row between two remover sub-batches; the DELETE must remain effective; before that DELETE commits, transfer alone must never make the row temporarily disappear.

### K4. Shadowing is defined by snapshot-authoritative hot state

A cold row is skipped when, **for the statement's snapshot**, the hot store holds an authoritative version or a delete marker for that row id. Not raw membership, and not the rows the query returned.

- Raw membership is wrong for snapshots. A row sits cold; snapshot S begins; another transaction updates it and commits. Hot now holds only a version newer than S, which S cannot see, and the first hot version of a cold row is created with no previous chain (version_store.rs:899). Skipping cold because the id appears in the hot B-tree would make the row vanish for S. The existing tests `test_snapshot_isolation_cold_update` and `test_snapshot_isolation_cold_delete` cover exactly this, and today they pass because `scan` uses the result-derived rule.
- Result-derived skipping is wrong in the other direction. It is predicate-dependent, and once the remover can delete a hot copy asynchronously, a committed hot DELETE returns no row and the stale cold copy would be admitted.

The statement therefore captures three things together: the hot versions visible to its snapshot, the hot delete markers visible to its snapshot, and the cold generation, plus its own transaction-local overlay. Raw membership may still be tracked, but only as an input to the transfer window of K3, never as the visibility decision.

*Tests:* a cold UPDATE and a cold DELETE committed after the snapshot began, checked on scan, COUNT, SUM and GROUP BY; a visible hot UPDATE whose new value no longer matches the predicate must not resurrect the old cold match; and the same query while a seal registers a volume and the remover drains it, against a quiesced engine.

### K5. Supersession evidence survives until the remover passes

The remover stamps a skipped row's tombstone with the sequence of the **first** version that superseded the built one, which it reads from the chain. But chains keep at most `max_version_history` entries and drop the whole previous chain past the limit (version_store.rs:802), and GC prunes as well. A slow remover can therefore find the evidence gone.

Contract: the evidence that must survive is **a bounded piece of transition metadata (the row id and the first superseding sequence), not the payload history**. Pinning whole chains instead is not bounded by the chunk size: one pinned row updated M times with a wide payload retains M previous versions (version_store.rs:859), and every update walks the chain to count its depth (version_store.rs:799), so blanket pinning grows both memory and update latency with M.

The Phase 0 deliverable states: the transition record and where it is written (at the moment of supersession, for row ids in an outstanding remover set, which is a bounded set); its **byte cap**; the **bounded per-update cost** of maintaining it; and the behaviour when the cap is reached, which must be an explicit safe fallback with its correctness argument, not silent truncation. Protection must be gapless from capture to the remover's pass.

*Tests:* between build and removal, apply far more updates than the history limit to one wide row, plus a DELETE and reinsert, with a snapshot bracketing them; measure peak retained bytes, UPDATE p99 and the remover's fence time, not only correctness.

### K6. The WAL mark follows durable coverage, not chunk liveness

The mark may only advance past an LSN when every seal, remover pass and tombstone publication that covers it has a **durable manifest**. A chunk whose last row was removed but whose manifest is not yet fsynced must still hold its WAL retention pin, otherwise a crash after truncation leaves a volume nothing references, and orphan cleanup may delete it. The existing checkpoint already treats manifest persistence as a precondition (engine.rs:5183) and that ordering must be kept.

`first_lsn` is the earliest **required DML LSN**, not the commit marker, and it propagates to every chunk a batch touched.
*Test:* crash between the last hot removal and the manifest fsync; the WAL must still contain the covering records.

### K7. Catalog identity survives DDL

Each table carries a stable id and an incarnation. The catalog retains name and schema history, and replay resolves a table **by id**, not by name. Without this: CREATE `t` at LSN 1, an unsealed INSERT at 10, RENAME `t` to `u` at 20, catalog persisted for `u`, WAL truncated below 10; on restart the retained record still says `t`, replay looks up by name and skips it (engine.rs:1776), and the row is lost. RENAME moves the same store (engine.rs:3741), so the hot rows genuinely belong to the retained WAL.
*Tests:* RENAME, DROP and re-CREATE, and replay of DML written under an older schema version.

### K8. A hard admission limit ships with a memory-pressure seal

Admission that blocks writers is only safe if something can relieve it. Today the background sealer waits for 100,000 rows before the first volume (`SEAL_ROW_THRESHOLD`, engine.rs:5800) and filters candidates by that threshold both when choosing tables (engine.rs:5874) and again after extraction (engine.rs:5920), while `hot_max_bytes` defaults to 0 (config.rs:142). With wide rows a 16 MiB budget fills long before the row threshold: admission stops the writers, the sealer keeps skipping the table as below threshold, and the rows needed to reach the threshold can never arrive.

Contract: **memory pressure seals a partially filled chunk regardless of the row threshold**; the scratch the seal and remover need is reserved before admission blocks; and when there is genuinely nothing sealable the engine waits or returns a resource error, and never exceeds the budget to make progress. Roadmap consequence: step 2 introduces accounting only, and the hard guarantee is enabled together with the bounded writer of steps 4 and 5.

*Test:* wide rows that fill the budget far below the row threshold, at default row limits, at 16 and 64 MiB; the engine must either seal under pressure and continue, or report the defined resource behaviour. Exceeding the budget to make progress is not a pass.

## 5. Design

### 5.1 Hot store: chunked arena and freeze

`RowArena` becomes a list of chunks of `ARENA_CHUNK_ROWS` slots (K1). Rows append to the active chunk; `update_at` and `mark_deleted` keep working in place on any chunk, because a frozen chunk still holds live hot rows. The free list exists only for the active chunk. The probe becomes `row_id - chunk.base_row_id` after a binary search over chunk bases, still validated by `meta.row_id`.

Freeze closes the active chunk: record its `first_lsn` (the earliest required DML LSN of any commit that wrote into it, K6) and open a new chunk. No quiescing; a commit's rows may span two chunks, because the B-tree and the indexes stay one per table.

Triggers stay as shipped in #111, plus the memory-pressure trigger of K8: when the budget is the binding constraint, a partially filled chunk is frozen and sealed regardless of the row threshold. A chunk that becomes empty is dropped whole and its capacity goes with it; chunk ids do not shift (K1).

**Residual rows count toward the account.** Rows above the seal bound stay in their chunk and are sealed by a later pass. Being exempt from the sealer's queue is not the same as being exempt from the memory budget: under a long snapshot with continuous INSERT, exempting them would let chunks accumulate without bound. **All retained hot bytes, residual chunks included, are measured**, together with uncommitted writes, previous versions, chunk capacity and index ownership. They are measured from step 2 and enforced from steps 4 and 5, when the memory-pressure seal of K8 can relieve the limit.

### 5.2 Seal: one bounded streaming pipeline

1. **Capture the eligibility bound** (K2) and the table incarnation (bumped by TRUNCATE, version_store.rs:2136, and by DROP or re-CREATE). The build is abandoned if the incarnation changes.
2. **Read the chunk in bounded batches.** Under the arena read guard copy a small reusable buffer of `(row_id, txn_id, CompactArc<[Value]>)`, release the guard, process, repeat. The guard is never held across a transposition and never across I/O or an allocation wait (K1 of the archive's rule, and 5.12 below). Include a slot only if `txn_id != 0`, `deleted_at == 0`, and it is committed at or below the captured bound. Batches are capped by rows **and** bytes.
3. **Order the rows.** Arena order is **not** row id order: an explicit PK can insert 100 then 10, and free-list slot reuse breaks physical order regardless. So the sealer sorts a `Vec<u32>` of offsets whenever the ids are not ascending, and only verifies when they are. Falling through to the builder's `row_ids`-only sort (writer.rs:1530) would detach identity from payload and must be removed. For a clustered table the sort key is the clustering key; the sort spills to disk when it does not fit (5.6, K-peak below).
4. **Emit group by group, column by column.** Gather one column of one group into a reused typed buffer, compress, write, release. Directory entries, zone maps and dictionaries accumulate during emission. `VolumeBuilder` gains `add_row_values(row_id, &[Value])` filling missing columns with schema DEFAULTs (what engine.rs:5942-5955 does today).
5. **Stream to the file** through the new format envelope: no store holding every compressed block, no whole-file buffer. Offsets are recorded as blocks are written; the directory is written last with its own checksum. **This requires the format envelope, so it ships with it** (roadmap step 4), not before.
6. **Remover records** `(row_id, txn_id)`, 16 bytes per row, are part of the maintenance budget and **spill to disk** past a threshold. At 1M rows they are 16 MiB, which alone would consume a 16 MiB test budget.

Peak memory formula, to be stated and measured, not assumed: one group column, plus one compressed block, plus the batch key array, plus the remover records (spilled past the threshold), plus pinned payloads, plus dictionary and directory accumulation. A fan-in cap and a policy for a single oversize value belong in the same formula.

### 5.3 Publication: microseconds, no I/O

Under `acquire_seal_write` and nothing else: check the table incarnation (a TRUNCATE between build and registration cancels the seal and discards the file), `set_seal_overlap(included_rows)`, swap in the visibility bitmap **prepared outside the fence against a recorded manifest generation**, register the volume, mark the chunk, bump auto-increment. If the generation moved, leave the fence, re-prepare, retry with bounded retries. **No I/O and no page fault under the fence.**

### 5.4 Removal: a fence held per sub-batch, with atomic tombstone transfer

A remover thread drains a chunk's records in 2,000-row sub-batches. Per sub-batch, under the write fence and `versions.write()`:

- remove the row only if its head `txn_id` still equals the recorded one and it is not claimed;
- for a row whose head changed, write a tombstone stamped with the sequence of the first version that superseded the built one, taken from **K5's byte-capped transition record**, not from a pinned version chain;
- for a claimed or unresolved publishing row, defer its transfer record until commit or abort, as specified in contracts.md; no guessed-sequence tombstone;
- **in the same critical section**, remove the superseded tombstone of any row it transferred, matched by sequence (K3); only the disk serialization is deferred;
- index entries by row id (#110), HNSW skipped.

`seal_overlap` and `committed_row_count` are decremented per sub-batch; the manifest is persisted after the pass, and the WAL retention pin is held until it is durable (K6).

**Readers while a chunk's overlap is non-zero** apply K4's rule: a cold row is skipped when the statement's snapshot sees an authoritative hot version or a hot delete marker for that id, and a row whose hot copy the remover has already transferred is authoritative in cold from the moment of the transfer (K3). A result-derived skip set is not sufficient here, because a committed hot DELETE returns no row and its cold copy would be admitted; raw membership is not sufficient either, because it hides the cold row from an older snapshot. Point lookups become hot-first in the two table methods and in the engine's fetchers and row counter; the ordered index merge dedupes equal adjacent ids before OFFSET and LIMIT.

**Transitional versus target behaviour.** While the remover is being introduced, the aggregate pushdowns keep bailing out when `seal_overlap > 0`. The target, after K4 is closed, is that they do not need to. The plan states both, and the tests for step 3 must be rerun after step 5 lands.

### 5.5 WAL and recovery

Replay is unchanged except for idempotence and identity. Before the mark moves, the checkpoint cycle persists a **catalog** per table (schema with `schema_version`, indexes, views: what `rerecord_ddl_to_wal` re-records today, engine.rs:5298), carrying the **stable table id, incarnation and name history** (K7). `CreateTable` replay becomes a no-op for a table already present (it currently inserts a fresh store, engine.rs:1614); `AlterTable` replay skips records older than the catalog's `schema_version`; every DDL bumps a DDL epoch and records its LSN, and the mark never passes the LSN of a DDL newer than the persisted catalog.

`checkpoint_lsn` is the minimum over tables of the earliest LSN still required by **durable coverage** (K6), capped by the catalog's DDL coverage, computed under the engine-global seal fence in microseconds. WAL cleanup deletes through `up_to_lsn` inclusive, so the mark passed is that LSN minus one. Rotate the WAL at each freeze so truncation frees files. A crash between volume write and manifest persist leaves an orphan `.vol`, removed at open by the stale-file cleanup extended to `.vol`.

**WAL batch records are byte-bounded.** One record per table per commit is the goal, but recovery skips any entry above 64 MiB (wal_manager.rs:1777, 1914), so a large transaction must be split into **multiple frames under the cap, sharing one `txn_id`, with a single final commit marker**. Raising the cap is not a fix: the reader allocates the whole payload.

### 5.6 Cold layout: clustering is explicit, with a window

A table is clustered only when the operator declares it, with an equality prefix and an ordering suffix. No automatic derivation in version 1: column type is not evidence of cardinality. Unclustered tables keep row id order.

A clustered table declares a **window key** (default one day). Compaction keeps outputs window-bounded so volume-level time pruning survives. **Window and contiguity interact**: with manifest order `[D1a, D2a, D1b]`, a single-window merge of `D1a` and `D1b` is not contiguous. The resolution is that the merge takes a **contiguous input run that may cross windows** and splits its output atomically by window, rather than choosing non-contiguous inputs.

Groups become directory entries: start, end, per column zone maps, first and last clustering key, per column local-order flags, run boundaries. Caps are 4,096 rows and 1 MiB decoded. Runs shorter than 256 rows are packed together.

**Dictionaries are group-local, and local codes are not comparable across groups or with hot rows.** Any path that groups, joins or merges on a dictionary column uses a shared typed key or a query-local re-encoding, never raw local codes; a spill of such keys is ordered by value, not by code.

**The directory itself is paged.** Resident per volume: window bounds, row id extrema, group count, sparse fences. Zone maps, run boundaries and lower directory pages are budget-charged. **Fences prove absence, never presence.**

### 5.7 Identity, and the cold constraint path

`meta.row_ids` in physical order remains the identity used by tombstones and the positional visibility bitmap.

- **Locator**: per volume, two parallel arrays, ascending row ids and their `u32` physical positions, written to the file, paged, selected by sparse fences.
- **Explicit extrema** in metadata, replacing derivation from array endpoints (volume/table.rs:1120, 1173; engine.rs:3100, 3424).
- **Batched lookups**: `collect_rows_by_ids` (volume/table.rs:2011), `fetch_rows_by_ids` (engine.rs:6544) and WAL replay sort their ids once and walk each locator page once.
- **Cold UNIQUE check (C1).** For a cluster-aligned constraint: window and volume pruning, fence selects candidate groups, key column blocks are read and matched exactly, then visibility. For other constraints the sidecar stays but becomes paged and persisted, built at seal, so **no INSERT triggers an O(volume) build**. Because this path is inside a statement that holds the shared seal fence, its page misses need a preflight: the DML and commit paths pin what they will need and validate the generation, or take an equivalent protocol. A page fault inside the fence is not acceptable.

### 5.8 Resource management: one account, four ledgers

| Ledger | Holds | Rule |
|---|---|---|
| Hot transaction state | version chains, arena chunks (residual included), hot indexes, uncommitted writes | not evictable; all retained hot bytes participate in admission |
| Analytic working memory | accumulators, group tables, sort buffers | charged per statement, released at end, per-statement cap |
| Block cache | decoded blocks, compressed bytes in flight, dictionaries, locator and directory pages, blooms, constraint index pages, visibility generations | evictable, two queues plus idle sweep |
| Maintenance | seal and compaction scratch, remover records, emit bitmaps | capped, spillable, with a reserved progress capacity for seal that compaction may not take |

Cross-cutting rules: charge for the allocation's lifetime, not the cache entry's; two queues so one large scan cannot evict the reused working set (replacing the O(n) victim search, group_cache.rs:161); an idle sweep independent of checkpoints; single flight per block; no automatic pairing of compressed and decoded copies; reservations that never wait while holding a publication lock; progress capacity for short transactions and seal; a non-evictable floor with the minimum budget stated as a formula.

The per-volume tier machine goes away: `evict_idle_volumes` (manifest.rs:1389), `to_warm`, `to_cold`, `is_cold` (writer.rs:1972), `ensure_volume` (manifest.rs:2628), `should_use_group_cache` (writer.rs:944), and the `LazyColumns` whole-column slots with their eager promotion (writer.rs:944-1023).

### 5.9 Two-layer execution under one snapshot

The mechanism exists (`merge_accum`, typed accumulators on both sides). The work is (a) closing K4, then (b) removing the bail-outs in order: snapshot isolation (volume/table.rs:4616, 5497), seal overlap (4621, 5512), multi-column GROUP BY (5501). Filters that are not conjunctive-simple stay on the generic path.

The typed hot path must also support snapshot isolation and the transaction-local overlay, not only autocommit reads. Composite group keys over dictionary columns obey 5.6's identity rule.

### 5.10 Compaction: identity first, then order

**Contiguity.** The input set of a merge must be a contiguous run in manifest order, because the output is inserted at the oldest input's position (manifest.rs:2501-2514) and a volume left in between would win over rows a newer input had won. Combined with windows, see 5.6.

**Key-changing updates.** Two phases: an **identity phase**, a k-way merge over the inputs' row-id sorted locators deciding the winner per row id by manifest precedence under today's snapshot-safety gates, producing an emit bitmap per input (1 bit per input row, budgeted and spillable); then an **emission phase** in clustering order emitting only set bits and applying tombstones with their sequences.

Precedence must be made consistent: dedup by segment id (engine.rs:5540) and replacement by manifest position (manifest.rs:2515) disagree today; manifest order is authoritative. Tombstone removal keeps its stronger requirement over complete overlapping components. Memory follows 5.2's discipline, replacing the 24 bytes per live row plus whole-table hash set at engine.rs:5550-5590.

### 5.11 Ordered scans keep their correctness rule

Volumes and groups in bound order, stopping only on `cannot_improve` (volume/table.rs:4120). Global order and group-local order are separate flags.

### 5.12 The data path stays columnar, and the allocation program

No `Row` or `Value` materialisation in scan, gather, seal or compaction; typed buffers reused per worker; `get_row` only where a caller needs a row.

Write path, four items:

1. **Index entries hold the row's `CompactArc<[Value]>`** and compare by projection. Conditions that must be carried over, not dropped: both commit paths that skip an index when its columns did not change (version_store.rs:7213, 7485) must perform the unchanged-key Arc swap, or a wide UPDATE leaves the old full row alive inside the index; `IndexUndo` must own the old Arc; NULL exemption, intra-batch duplicate detection, removals before additions on key swaps and cross-index rollback stay; the lazy `sorted_values` still builds owned keys, so the "zero allocation" claim is amortized and conditional, and retained payload counts against the ledger.
2. **WAL batch records**, byte-bounded per 5.5.
3. **A real `insert_batch`** for the executor's VALUES and INSERT..SELECT loops, hoisting the transaction store lock and index list per batch; conflict paths stay per row.
4. **Arena guard per bounded batch, not per statement.** The guard blocks every commit that needs an arena write, so a long analytic statement must not hold it. The pattern is: take the guard, copy a bounded batch of `(row_id, meta, Arc)` pins, release the guard, then process. The existing lock order is `versions.write` then `arena.write`, and the point lookup deliberately drops the arena guard before its B-tree fallback (version_store.rs:1218-1232); a visitor that falls back while holding the guard would invert it. No cold I/O, no allocation wait and no callback inside the guard.

`SeriesRun` stays deferred until its contracts are written.

## 6. Costs

Per row, candle schema, 9 columns:

| Structure | Today | This plan | Ledger |
|---|---|---|---|
| Decoded columns | 73 B | 73 B | block cache, per group |
| Compressed bytes | resident per volume when warm | transient during decode | block cache |
| Physical row ids | 8 B always resident | 8 B, paged | block cache |
| Locator | none | 12 B, paged | block cache |
| Source DML LSN | none per row | 8 B before compression, paged | block cache |
| Bloom filters | 1.25 B per column per row, always | paged, consulted only when fences do not decide | block cache |
| UNIQUE sidecar | 16 B per entry, prebuilt, never released | absent for cluster-aligned constraints, paged otherwise | block cache |
| Directory | 9 zone maps per 64K group, resident | root and fences resident, rest paged | floor plus block cache |
| Dictionaries | one per volume | one per group, charged once | block cache |
| Hot arena | capacity never returned after a seal | chunk dropped whole when empty | hot |
| Remover records | none (long fence instead) | 16 B per row, spillable | maintenance |
| Compaction emit bitmaps | none | 1 bit per input row, spillable | maintenance |
| Seal scratch | whole decoded volume, all compressed blocks, whole-file buffer | one group column, one compressed block, one batch key array | maintenance |
| Compaction scratch | 24 B per live row plus a whole-table hash set | one locator page per input, one output group | maintenance |

`(u64, u32)` occupies 16 bytes, not 12; the comment at writer.rs:1132 is wrong.

## 7. Roadmap

Done: #107 to #113 (section 1).

| # | Step | Depends on | Delivers |
|---|---|---|---|
| 0 | **Contracts K1 to K8** written as design, with their failure scenarios and test plans. | none | the shape of everything below |
| 1 | **Fallible access and statement atomicity.** Group views (`group(col, gi) -> Result<Arc<ColumnData>>`) and identity accessors over the current format; fallible signatures at `is_row_id_in_volume` (manifest.rs:2097), `row_exists` (1894), `find_segment_row_in` (volume/table.rs:1096), the aggregate row-count closures, `get_authoritative_value` (manifest.rs:968); statement rollback. No format change. | 0 | precondition for 3, 4, 5 |
| 2 | **Chunked arena** with fixed physical capacity (K1), chunk dropped when empty, `first_lsn` per chunk, and **accounting only**: all retained hot bytes including residual chunks are measured and reported, but the hard limit is not enforced yet, because nothing can relieve it until the bounded writer exists (K8). | 0 | memory returns after a seal; the batch unit for step 4 |
| 3 | **Two-layer execution**: K4's coherent snapshot, then the bail-outs in order. | 0 (K4), 1 | the analytic half of the thesis; rerun after step 5 |
| 4 | **Format envelope and streaming seal together.** Independently checksummed metadata, directory and blocks, streaming writer, bounded batches, offset sort with spill, remover records with spill. The envelope cannot come later: "directory last, independent checksum" is not the V4 layout. | 1, 2 | C2 |
| 5 | **Remover and durable WAL mark together.** Per-sub-batch fence, atomic tombstone transfer (K3), the byte-capped transition record (K5), K4's snapshot-authoritative skip rule in the overlap window, hot-first point lookups; catalog identity (K7), durable coverage mark (K6), byte-bounded WAL frames. With step 4 this enables the hard limit **for hot and maintenance memory**, together with the memory-pressure seal (K8). The old drain-then-truncate path stays until both land. | 4 | the writer stall fix, and C1 |
| 6 | **Paged reads, the four ledgers, and the DML preflight.** This is where the budget becomes an **engine-wide** guarantee, because cold allocations enter the account only here. Lazy pages, the account of 5.8, deletion of the tier machine and whole-column slots, bounded V4 metadata parsing (streaming, spooled). The preflight belongs here, not with clustering: an unclustered INSERT already runs its cold constraint check under the shared seal fence (volume/table.rs:1362-1366) and commit revalidates the same way (engine.rs:7635), so the first lazy cold page must not be able to fault inside that fence. | 4, 5 | C4, residency by access |
| 7 | **Clustering**: specification and window key, cluster-order emission, variable groups with fences and runs, group-local dictionaries with the identity rule, locator, cold constraint path. **The old compactor's compatibility path lands here, before the first clustered volume exists**: it keeps row id and physical position together and sorts those references (engine.rs:5586, 5665), so an unclustered-output fallback is possible given the layout metadata and the budget, but it must be ready before any new-layout volume can appear. | 6 | proportional cold reads, C5 |
| 8 | **Compaction and migration**: identity-first protocol, contiguous input runs with window-split outputs, consistent precedence, conversion driven by format and layout with a progress marker and a byte cap. | 7 | keeps 7 correct over time |
| 9 | **Allocation program.** Item 2 (WAL frames) depends on step 5's recovery format; item 1 (Arc index) depends on the publish and undo conditions of #109 and on 5.12's ownership rules; item 4 (batch reader) depends on the lock-order protocol. Not a single independent package. | 2, 5 | hot-path allocations and locks |

Total remaining: about 8,000 to 10,000 production lines plus tests.

## 8. Legacy data

A V4 file has one metadata blob and one CRC over the whole file (io.rs:189, 314). Decompressing that blob whole is not bounded: for a 1M-row candle volume the row ids and blooms alone are about 19 MiB. So the compatibility path **parses the metadata as a stream with pageable or spooled backing**, verifies the CRC in one pass without retaining blocks, and keeps only the directory root.

Conversion is triggered by format or layout, with a persisted progress marker per table and a per-cycle byte cap; today's compaction skips clean at-target volumes (engine.rs:5455-5495), so the measured history would otherwise never convert.

## 9. Acceptance

**Primary benchmark: the mixed workload.** Ingest, analytic queries and compaction at the same time, reporting INSERT and UPDATE p50/p95/p99 against both idle (C1); aggregation latency while ingest continues (C3); peak resident bytes and allocation per ledger (C2, C4); bytes copied and blocks read and decoded; memory returned after idle and after a seal; short-transaction commit latency while a scan and a compaction hold memory (C4); an unclustered table's PK point lookup and PK-ordered scan unchanged (C5). All at **16 MiB and 64 MiB budgets**, with the remover records, compaction bitmaps and allocator capacity inside the measurement.

Comparisons must hold data, budget, arrival rate and cache state constant, and separate cold first touch from warm repeats. "Normal latency band" needs a stated threshold and a noise figure.

**Failpoint and differential tests each implementation PR must carry:**

- a cold UPDATE and a cold DELETE committed after a snapshot began, checked on scan, COUNT, SUM and GROUP BY, plus a visible hot UPDATE that no longer matches the predicate (K4);
- wide rows filling the budget far below the row threshold, at 16 and 64 MiB: pressure seal or a defined resource error, never overshoot (K8);
- a snapshot opened and an UPDATE committed between two build batches (K2);
- a cold DELETE committed between two remover sub-batches, and the transfer window itself (K3);
- explicit PK order and free-list reuse, then seal and reopen, comparing values and identities (5.2 step 3);
- a long snapshot with continuous ingest: residual bytes stay inside the limit and backpressure is reported (5.1);
- a crash between the last hot removal and the manifest fsync (K6);
- catalog persist, RENAME, retained WAL under the old name; DROP and re-CREATE; DML replayed under an older schema (K7);
- one transaction writing to several tables past 64 MiB, with crash and restart (5.5);
- far more updates than the history limit to one wide row between build and removal, plus DELETE and reinsert, measuring retained bytes, UPDATE p99 and remover fence time (K5);
- an analytic scan with an arena and B-tree fallback concurrent with commits: no long guard, no lock inversion (5.12);
- groups whose local dictionary codes collide on different values, in a hot and cold composite GROUP BY (5.6);
- interleaved window backfill: compaction progresses and results are preserved (5.6, 5.10);
- V4 open and conversion inside the 16 MiB budget (section 8).

Per-query expectations, as estimates: per-pair aggregation of the whole history 15 to 35 ms at 4M rows and 60 to 200 ms at 15.7M, against 82 to 104 ms and about 1 s today; per-pair COUNT under 1 ms at 4M rows; latest N of one pair 1 to 5 ms cold. The 31 ms measured at a 512 MB budget today is a full-scan number and is not the mathematical floor for a clustered plan, which also reduces the rows scanned and filtered. The all-keys recent-time query on cold clustered data gets worse, 5 to 30 ms.

## 10. Rejected

- Raising the cache budget alone: helps repeats (60 ms to 6.4 ms) but bounds nothing and does not reduce first-touch decode.
- Generations inside `VersionStore`: 5,500 to 6,500 lines, a per-read k-way cost and many boundary races.
- Removal fully off the fence: DML classifies hot versus cold in one step and acts in another under the statement fence, so lock-free removal loses UPDATEs and creates phantom DELETEs.
- A tombstone-at-commit hook: it races the build.
- A compact row layout first: indexes, not values, dominate the roughly 700 B per hot row.
- Cold secondary indexes or posting lists: they locate rows but still decode every group those rows touch.
- Smaller fixed groups without clustering: more directory, worse compression, still proportional to the table.
- One volume per key; time bucket as the leading clustering column; mmap as the residency mechanism; query-driven automatic reclustering; automatic clustering-key derivation in version 1; rewriting all volumes at upgrade.
- Raising the 64 MiB WAL entry cap instead of framing: the reader allocates the whole payload.

## 11. Risks and open questions

- **Cold constraint lookup** is the sharpest hot-path risk: a paged key-block miss inside an INSERT puts disk latency on the row path, inside the statement's shared fence. Its preflight is a design item, not an implementation detail.
- **Supersession evidence** (K5) is bounded only if it is transition metadata with a byte cap; pinning chains is bounded by updates, not by rows, and also lengthens every update.
- **Hard admission without a pressure seal deadlocks writers** (K8); the two must be enabled together.
- **PK locality on clustered tables** must be measured on a clustered table, not only on an unclustered one.
- **All-keys recent-time queries** on cold clustered data get worse.
- **Group caps** (4,096 rows, 1 MiB) need a sweep; small groups compress worse and multiply directory pages.
- **Group-local dictionaries repeat** across groups and cannot be compared across them; the crossover against a volume-wide dictionary is unmeasured.
- **Locator paging** is 12 B per row on disk; a random-id workload pages it heavily.
- **Whether a compressed warm copy pays at all** is unmeasured.
- **Conversion amplification** needs its byte cap tuned against real ingest.
- **The remover's lag** keeps `seal_overlap` non-zero for about 0.3 s per chunk, during which the aggregate pushdowns take the scan path until K4 lands.
- Not verified by reading alone: how many paths outside the volume layer assume `row_id == pk` (project history says about 50), and whether every executor path ending in `fetch_rows_by_ids` accepts a fallible result.
