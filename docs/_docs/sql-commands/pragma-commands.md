---
layout: doc
title: PRAGMA Commands
category: SQL Commands
order: 4
---

# PRAGMA Commands

This document describes the PRAGMA commands available in Stoolap based on implementations and test cases.

## Overview

Stoolap provides PRAGMA commands for configuring and inspecting the database engine. These commands primarily focus on persistence settings and storage behavior.

## Syntax

The basic syntax for PRAGMA commands is:

```sql
PRAGMA [pragma_name] = [value];
```

or to retrieve the current value:

```sql
PRAGMA [pragma_name];
```

## Available PRAGMA Commands

Stoolap currently supports the following PRAGMA commands:

### Checkpoint and WAL Configuration

| PRAGMA | Description | Default |
|--------|-------------|---------|
| `checkpoint_interval` | Seconds between automatic checkpoints | 60 |
| `compact_threshold` | Sub-target volumes per table before merging | 4 |
| `keep_snapshots` | Backup snapshots to retain per table | 3 |
| `target_volume_rows` | Target rows per cold volume (compaction split boundary) | 1048576 |
| `snapshot` | Create a full backup snapshot | - |
| `restore` | Restore database from a backup snapshot | - |
| `checkpoint` | Run checkpoint cycle (seal + compact + WAL truncate) | - |
| `vacuum` | Manual cleanup of deleted rows and index compaction | - |
| `sync_mode` | Read current WAL sync mode (read-only, set via DSN) | 1 |
| `wal_flush_trigger` | Read current WAL flush trigger (read-only, set via DSN) | 32768 |
| `volume_stats` | Show per-volume storage statistics | - |

### Memory and Hot Store

| PRAGMA | Description | Default |
|--------|-------------|---------|
| `hot_max_rows` | Committed hot rows per table that request an early seal (0 = off) | 262144 |
| `hot_max_bytes` | Hot row bytes per table above which commits wait for a seal (0 = off) | 0 |
| `group_cache_mb` | Budget of the decoded row group cache in MB (0 = bypass) | 64 |
| `group_cache_stats` | Show the decoded row group cache's budget, size and hit counts | - |
| `memory_stats` | Show per-table hot and cold memory figures | - |

#### checkpoint_interval

Controls how often the background checkpoint cycle runs (in seconds). The checkpoint seals hot buffer rows into immutable cold volumes, persists manifests, and truncates the WAL. Default: 60.

```sql
PRAGMA checkpoint_interval = 60;
PRAGMA checkpoint_interval;       -- read current value

-- Legacy alias (backward compatible)
PRAGMA snapshot_interval = 60;
```

#### sync_mode (read-only)

Returns the current WAL synchronization mode. This setting can only be configured via the connection string and takes effect at database open time.

```sql
PRAGMA sync_mode;               -- read current value (0=none, 1=normal, 2=full)
```

To change sync_mode, set it in the connection string:

```
file:///path/to/db?sync_mode=none
file:///path/to/db?sync_mode=normal
file:///path/to/db?sync_mode=full
```

#### compact_threshold

Controls how many sub-target volumes (smaller than `target_volume_rows`) accumulate per table before compaction merges them. At-target volumes are not counted and are never rewritten unless they have tombstoned rows. Default: 4.

```sql
PRAGMA compact_threshold = 4;
PRAGMA compact_threshold;          -- read current value
```

Note: `compact_threshold` and `keep_snapshots` are separate settings. `compact_threshold` controls cold volume compaction. `keep_snapshots` controls backup snapshot file retention.

#### wal_flush_trigger (read-only)

Returns the current WAL flush trigger (buffer size in bytes before flush). This setting can only be configured via the connection string.

```sql
PRAGMA wal_flush_trigger;       -- read current value
```

To change wal_flush_trigger, set it in the connection string:

```
file:///path/to/db?wal_flush_trigger=65536
```

#### keep_snapshots

Controls how many backup snapshots are retained per table. Older snapshots beyond this count are automatically deleted after each `PRAGMA snapshot`. Default: 3.

```sql
PRAGMA keep_snapshots = 5;
PRAGMA keep_snapshots;              -- read current value
```

#### target_volume_rows

Controls the target number of rows per cold volume. Compaction splits output into volumes of approximately this size (rounded down to the nearest row-group boundary of 64K rows). Values below 65,536 are rejected. Default: 1,048,576 (1M rows, 16 row groups).

```sql
PRAGMA target_volume_rows = 1048576;
PRAGMA target_volume_rows;          -- read current value
```

#### hot_max_rows

A commit that takes a table's committed hot rows past this value asks for a seal right away. The checkpoint thread runs the seal cycle on its next tick (within 100 ms) instead of waiting for `checkpoint_interval`, and the seal's own row thresholds are bounded by the same value. The trigger keeps the hot store of a bulk load bounded at the cost of running the seal more often during the load. 0 turns it off. Default: 262,144 (four row groups).

```sql
PRAGMA hot_max_rows = 262144;
PRAGMA hot_max_rows = 0;            -- seal only at the checkpoint interval
PRAGMA hot_max_rows;                -- read current value
```

#### hot_max_bytes

When a table holds at least this many hot row bytes, a commit that writes to it requests a seal and waits, before it takes the seal fence, until the seal brings the table back under the limit. The wait is bounded at 10 seconds and counted (see `memory_stats`). A transaction that updates or deletes existing hot rows does not wait, since the seal skips rows it has claimed. In-memory databases never wait because nothing can seal. 0 turns the wait off. Default: 0.

```sql
PRAGMA hot_max_bytes = 268435456;   -- 256 MB per table
PRAGMA hot_max_bytes;               -- read current value
```

#### group_cache_mb

Sets the budget of the decoded row group cache shared by all cold volumes in the process. Queries over warm volumes decode the row groups they need from the in-memory compressed blocks; the cache keeps decoded groups so repeated queries do not decode them again. When a workload's groups exceed the budget, every query decodes again: a `SELECT` that scans a few million warm rows with several projected columns needs a few hundred MB. Raising the budget trades memory for repeat query time; 0 bypasses the cache. Default: 64.

```sql
PRAGMA group_cache_mb = 512;
PRAGMA group_cache_mb;              -- read current value in MB
```

#### group_cache_stats

Returns one row with the cache's budget, current size, entry count and hit and miss counters since the process started.

```sql
PRAGMA group_cache_stats;
-- budget_bytes | bytes | entries | hits | misses
```

#### memory_stats

Returns one row per table and a final `*` row with the totals.

| Column | Meaning |
|--------|---------|
| `table_name` | Table, or `*` for the totals |
| `hot_rows` | Committed rows in the hot store |
| `hot_bytes` | Bytes of the rows the hot arena holds, heap text and extension payloads included |
| `arena_slots` | Arena slots in use, deleted and cleared ones included |
| `arena_capacity_bytes` | Capacity of arena slots, directories, free lists and prepared or detached buffers |
| `chain_entries` | Previous row versions kept alive by version chains |
| `volume_bytes` | Memory of the table's loaded cold volumes |
| `admission_waits` | On the `*` row: commits that waited for a seal under `hot_max_bytes` |
| `version_payload_bytes` | Payload bound for current and previous hot versions, including shared children |
| `pinned_version_payload_bytes` | Whole-root payload bounds retained by readers and pending TRUNCATE destruction |
| `retired_arena_payload_bytes` | Arena payloads removed from live slots but still awaiting destruction |
| `version_tree_bytes` | Requested tree-node capacity and previous-version link allocations in the current hot tree |
| `pinned_version_tree_bytes` | Whole-tree node and link bounds retained by readers and pending TRUNCATE destruction |
| `transaction_version_bytes` | Local and original row payloads and spilled version-history capacity retained by transactions |
| `transaction_undo_bytes` | Index-undo vector capacity and key payloads retained until release or reversal |
| `transaction_map_bytes` | Process-wide capacity of active and pooled transaction maps and their pool vectors, shown only in `*` |
| `index_requested_bytes` | Requested built-in index allocations retained by the table or outstanding index owners |
| `index_estimated_bytes` | Conservative bounds for opaque hash bucket/control storage, B-tree nodes and Roaring bitmap allocations |
| `index_scratch_bytes` | Process-wide retained HNSW query and parallel-build scratch capacity, shown only in `*` |
| `row_claim_bytes` | Row-claim map allocation, including capacity retained after claims are released |
| `row_pool_bytes` | Process-wide cached row and row-ID vector capacities, shown only in `*` |
| `exported_payload_bytes` | Process-wide payload ownership in SQL results, caches, compiled expressions and retained index cleanup, shown only in `*` |
| `transaction_registry_bytes` | Process-wide requested capacity of transaction-state, snapshot-sequence and isolation-override maps, shown only in `*` |
| `hot_metadata_bytes` | Process-wide retained hot catalog, schema, transaction, result-name and accounting metadata bounds, shown only in `*` |

The payload subtotals count shared allocations per owner. Canonical owned tombstone vectors use their current count times the largest capacity admitted while any owned tombstone remains; the maximum resets when that count reaches zero. Captured roots keep their recorded bound until release. These are requested-byte bounds, excluding allocator rounding. Reported counters saturate at their platform and API limits.

Tree capacity includes each node's allocation header and fixed key, value or child-pointer capacity, plus each previous-version link and its Arc header. Each captured root counts its full reachable tree and history, so shared allocations can be counted more than once. Temporary copy and merge buffers are outside this tree subtotal.

Transaction versions include every retained savepoint version and the original row used for conflict detection. Rollback releases this charge when it discards those objects; releasing a row claim alone does not release memory. Index undo is counted separately and remains charged during reversal, including after its transaction store is released.

Transaction map capacity stays charged as maps move between active transactions and the shared pools. The process-wide subtotal includes maps used by other open database instances; do not sum it across instances. Temporary old/new arrays during a map rehash are outside this retained-capacity subtotal.

Transaction registry maps include their minimum empty allocation and follow actual capacity after growth and garbage collection. Their charge survives until the last registry owner releases the maps. Updates within existing capacity perform no counter update. The separate committed-transaction cache is a fixed inline thread-local array of 65,536 `i64` entries, 512 KiB per participating thread; it is not heap capacity and is outside this subtotal.

Shared transaction-store cache metadata includes the actual outer map allocation, spilled table-list capacity, heap table names and the cache's Arc/RwLock allocation. Prepared/publication buffers, copied table names for commit/rollback checks and the hot-admission store list keep independent charges until destruction. A retained transaction store is charged separately until its final owner dies, using an envelope of `size_of::<RwLock<TransactionVersionStore>>()` plus two reference-count words and layout padding. The envelope also applies to stack-created transaction stores and therefore can overcount. Application-owned weak references that keep an Arc allocation after the contained object dies are outside engine ownership. Allocator rounding is excluded.

Hot metadata includes schemas and their names, column/FK vectors, default-value children and cached names/index maps. Actual vector and string capacities are counted; hash maps use the Swiss-table bound below. Shared schema caches can be counted more than once. DDL mutations and engine adoption refresh the charge. `VersionStore::schema_mut()` returns a guard that mutates `Schema` directly and refreshes before unlocking. Direct public-field edits made by Rust callers outside engine adoption are outside this estimate. Savepoint and DDL vectors retain their capacity and names through rollback and cleanup; dropped schemas keep their own charges.

Catalog table/store/index/view maps retain bucket high-water bounds and actual stored key capacities. View definitions, reverse-FK lists and installed hot zone maps own their charges independently of catalog membership. Stale zone maps still retain their segment capacities and minimum/maximum value children. Recovery timestamp strings and configuration paths are included. Result column names, compiled COUNT column names and the EXISTS name cache keep separate charges, including alias changes, retained cache capacity and statement-local exports after cache replacement.

Fixed envelopes cover engines, allocated transaction callbacks, stores, boxed hot-table handles, transaction registries and accounting objects. Shared synchronization envelopes are counted through their engine-created owners; directly detached public Rust synchronization guards are outside this estimate. Each engine-held Weak counts an additional allocation envelope until that Weak is released, covering backing memory after the last strong owner dies. This deliberately overcounts while strong owners remain. Registry vector capacities and spilled owner/import SmallVec capacities remain charged. Arc envelopes are also allowed for inline map roots and boxed or stack-created objects, so these are conservative bounds rather than exact allocator totals. No universal row, string or Arc header is enlarged.

Row-claim capacity follows insertion growth and removal-driven shrinking. TRUNCATE retains the empty map's allocation; destruction releases its charge. The row-vector pool subtotal includes each outer pool vector and every cached backing buffer, using their actual capacities and element sizes. Cached buffers contain no row payloads. Checkout transfers capacity out of this pool subtotal; clearing a pool retains its outer vector allocation. Thread-local destruction releases the remaining charge. Do not sum this process-wide subtotal across instances; active result-buffer capacity is outside it.

Primary-key index accounting includes names, bitset capacity and overflow-set capacity. Hash indexes add column metadata, actual reverse-map allocation, collision/key/row-ID vector capacities and each stored key's Arc allocation and heap children. Shared children are counted for each key owner. Clear retains and reports bitset and hash-table capacity; failed writes still report allocations they retained before returning the error.

Hash bucket storage is estimated separately. Let `h` be the largest reported map capacity observed during the retained allocation's lifetime, including construction, reservation and insertion. Its bound is zero when `h` is zero, otherwise `2 * h * (size_of(bucket) + 1) + 2 * max(align_of(bucket), 16)`, where a bucket contains the hash and collision-vector handle. This assumes the standard-library Swiss-table layout with at most twice that capacity in buckets and at most 16 control bytes per group. The high-water mark survives deletion and clear, since those operations retain the allocation.

B-tree requested bytes include names, reverse-map allocation, row-ID vector capacity and each sorted, reverse-map and cached minimum/maximum key owner. Invalidating a cache retains its charge until refresh or clear replaces its values. Shared key allocations can therefore be counted more than once. The node estimate is zero before the first insertion and after clear; otherwise it uses `(1 + unique_keys / 5) * 16 * (size_of(key_handle) + size_of(row_id_vector) + size_of(pointer))`, with integer division. This bound assumes the standard-library B-tree geometry of at most 11 keys and 12 child pointers per node, and at least five keys per non-root node. On a 64-bit build the per-node allowance is 512 bytes. It includes a possible empty root after deletion.

Bitmap requested bytes include metadata, reverse-map allocation and each bitmap-key and reverse-map key owner. The outer hash map uses the hash-capacity bound above, with `(CompactArc<Value>, BitmapRows)` as its bucket type. Each indexed value additionally tracks its current 16-bit container count `C`, current 32-bit submap count `S`, historical maximum container count `H`, and payload allowance `W`. The directory bound is `64 * S * max(4, 2 * min(H, 65536))`; `H` survives partial deletion. Submap B-tree nodes use the B-tree formula above with `S` entries, a `u32` key and a `RoaringBitmap` value.

For the bitmap index's point-insert/remove paths, a new container adds 8 bytes to `W`, a successful insertion into an existing container adds 4, and container destruction subtracts 8. Duplicate inserts and removals within surviving containers leave it unchanged. The allowance is capped at `8192 * C`, and reaches zero when the last container is destroyed. This bounds the retained arrays and bitsets using Roaring 0.11.3's array growth and conversion behavior. Fresh singleton containers each contribute 8 payload bytes; repeated insert/delete churn can overcount up to 8192 bytes per surviving container. The five tracking scalars add 40 bytes per distinct indexed value on a 64-bit build, included in the outer hash estimate. The bound covers completed mutations; transient allocation overlap is excluded.

Hash, B-tree and bitmap slice/ID batch mutations publish their aggregate accounting change once per batch. The legacy map-based batch methods still call scalar mutations for each row.

Composite indexes count metadata, reverse-map allocation, stored key/value and row-ID vector capacities, and shared children once per retained owner. This includes lazily built range trees, prefix maps and ordered prefix groups. Hash maps use the capacity bound above with their actual bucket types. Range trees and ordered groups use the B-tree node formula with their actual key/value types; each retained empty group includes a possible root. Clear releases those nodes and nested payloads while retaining hash capacity. Batch accounting follows each existing locked mutation scope; small ordered-group updates and legacy map batches still use scalar updates.

HNSW requested bytes include metadata, packed vector capacity, node and row-ID arrays, reverse-map allocation, deleted-bit capacity, every node's layer and neighbor capacities, embedded build scratch, and UNIQUE collision-vector capacities. Its UNIQUE hash map uses the hash bound above. Deletion retains the graph and its capacity; cleanup rebuilds it, and clear releases it. Three cached graph counters leave the per-node layout unchanged. HNSW search scratch owned by a thread is counted separately, since it survives index destruction and can serve multiple indexes. Its subtotal is the actual visited-bitset and candidate/result-heap capacity, released when the thread-local owner dies. Do not sum this process-wide subtotal across database instances.

Removing an index or table does not release its charge while another index owner retains it. Repeated attachment to one table counts once; an index explicitly shared between different stores is counted for each owner. Requested bytes include the concrete index object and its Arc envelope. Accounting handles and registry metadata are counted in `hot_metadata_bytes`; construction before attachment is outside the registered index subtotals. A custom Rust `Index` returning no memory account is unreported, not zero bytes. Temporary growth/rehash overlap, HNSW rebuild overlap, returned search/serialization buffers and mutation scratch maps are excluded.

Composite cleanup vectors and their old key pins, including the values collected for ID-only removal, are counted in `exported_payload_bytes` until their local owners are destroyed. This includes their actual vector capacities and shared value allocations, and applies to maintenance calls as well as SQL.

SQL readers charge exported rows and shared value children before releasing their protecting storage guard or captured root. Export ownership is created on the first nonzero export; statements that export no payload allocate no retained-payload scope. A statement retains its cumulative export charge until its result and nested result owners are destroyed, including after UPDATE, TRUNCATE or DROP TABLE. Closing a result does not necessarily destroy its buffers. Repeated reads and projections can overcount shared children. Result-buffer and analytic-operator capacities are outside this subtotal.

Deferred readers retain their export owner with the schema-default mapping or expression result. The owner is created lazily on the first nonzero export and follows the result across threads. Interleaved results keep separate owners unless they share a containing SQL statement. Closing a reader preserves its charge while its values remain accessible.

Each thread can retain one empty scope for reuse after a materialized or deferred result releases its last owner. Recycling clears exported payload charges and import storage; the idle scope object stays counted in `hot_metadata_bytes` until reuse or thread destruction.

Row exports charge their source owner's largest recorded row-payload bound, including heap value children. Committed readers capture the version-tree bound with their root; arena and transaction-local readers use their respective owner's bound. Batches multiply this bound by the returned row count once. The arena-probing callback reader charges each captured batch before releasing its source guards, including prefetched rows skipped by early exit. This avoids walking exported rows' columns and covers owned-to-shared conversion where needed. It can overcount after a large row or old version is removed, and resets when the owner's payloads become empty. The arena keeps its bound independently while sealed rows await cleanup. Scalar value exports use their returned payload sizes, available in constant time.

Semantic, scalar-subquery, IN-subquery, semi-join and batch-aggregate caches retain independent payload charges. A cache hit adds its owner's bound to the receiving statement once per cache-owner identity; eviction cannot release that statement's charge. IN-subquery entries allow two copies of value children for the value vector and its lazy set. Compiled expression programs and dynamic pattern keys retain their own shared-child charges. Reusable expression stacks and argument buffers release values after evaluation succeeds or returns an error while retaining buffer capacity.

Schema defaults appended to older hot rows or read through cold column mappings receive export charges. Engine-built mappings retain their own default-child charges through cloning and cache replacement. Mapped reads charge only selected defaults, and parallel row/group scans retain the statement's scope for each volume task. Rust callers construct mappings with `ColumnMapping::new`; later direct edits to its public `sources` are outside this retained-owner accounting.

Published volume dictionaries, zone maps and statistics become cold-owned state. Their lifetime accounting belongs to the cold ledger; hot extraction and export owners keep their independent charges until destruction. The current `volume_bytes` remains an estimate of live registered volumes: it does not bound retained capacities or generations pinned after replacement. Bounded construction and maintenance accounting, followed by complete cold-ledger coverage, are separate stages.

Rows and values handed to application code through the Rust storage API or result iteration transfer to application ownership. Internal SQL materialization stays within the statement's charge. These counters are process-wide, including payloads retained by other database instances, so do not sum them across instances. They are conservative requested-byte subtotals, not process RSS or an enforced engine-wide budget.

The `*` row includes version, arena and registered index ownership retained after DROP TABLE, including pending destruction. Table rows and logical row counts describe currently registered tables, so retained byte totals can exceed their sum. Returned bitmap/set-operation buffers and other unlisted metadata or pools are outside these subtotals. They do not change the existing hot-size trigger.

```sql
PRAGMA memory_stats;
```

### Manual Snapshot and Checkpoint Control

#### snapshot

Creates a full backup snapshot of all tables. Snapshot files (.bin) are stored in the `snapshots/` directory with per-timestamp `ddl-{timestamp}.bin` and `manifest-{timestamp}.json` files. The `keep_snapshots` setting limits how many snapshot files are retained per table.

```sql
-- Create a full backup snapshot
PRAGMA snapshot;
```

The snapshot captures a consistent point-in-time view of all tables. This is useful for:
- Creating consistent backup points before critical operations
- Manual full-database backup for disaster recovery
- Ensuring data is persisted before shutting down

Note: This PRAGMA cannot run inside an explicit transaction.

#### checkpoint

Runs the checkpoint cycle, which is the core persistence mechanism for the hot/cold volume architecture:

```sql
-- Run the checkpoint cycle
PRAGMA checkpoint;
```

The checkpoint cycle performs these steps in order:

1. **Seal**: Move eligible hot buffer rows to immutable cold volumes (.vol files)
2. **Persist**: Write manifests (volume list, tombstones, checkpoint LSN) to disk
3. **WAL truncate**: Remove WAL entries before checkpoint LSN (only when all hot data is sealed)
4. **Compact**: Merge sub-target, oversized, and tombstoned volumes into target-sized outputs

The background thread runs this cycle automatically every `checkpoint_interval` seconds. On clean shutdown, a force checkpoint seals ALL remaining hot rows regardless of threshold.

This command is useful for:
- Ensuring data is persisted to volumes before critical operations
- Reclaiming memory by moving hot data to columnar cold segments
- Manual control over checkpoint timing instead of relying on `checkpoint_interval`

Note: This PRAGMA cannot run inside an explicit transaction.

#### restore

Restores the database state from backup snapshots created by `PRAGMA snapshot`. This is a destructive operation that replaces all current data with the snapshot data.

Without a timestamp, restore uses the latest `manifest-*.json` to filter which tables are eligible (preventing dropped tables from being resurrected), then picks the newest snapshot file per eligible table. If no manifest exists (older snapshots), all table directories are included. Index and view definitions are loaded from the `ddl-{timestamp}.bin` matching the oldest selected snapshot; if that file is missing, current in-memory definitions are preserved as a fallback.

With a timestamp, restore selects the exact snapshot file per table matching that timestamp and loads the corresponding `ddl-{timestamp}.bin` for index/view definitions. If the DDL file is missing, the restore fails with an error.

```sql
-- Restore from a specific snapshot by timestamp (recommended)
PRAGMA restore = '20260315-120000.000';

-- Restore from the latest backup snapshot
PRAGMA restore;
```

The restore operation:

1. **Validates** all snapshot files before making any changes
2. **Reads** `ddl-{timestamp}.bin` for index and view definitions
3. **Truncates** WAL to prevent post-snapshot entries from overwriting restored data
4. **Clears** all current data (hot buffer, cold volumes, in-memory state)
5. **Loads** snapshot data for each table
6. **Recreates** indexes and views from DDL metadata
7. **Syncs** auto-increment counters with restored data
8. **Re-records** DDL to WAL for crash safety
9. **Checkpoints** the restored data into volumes for immediate durability

If the database cannot open due to corrupted volumes or manifests, use the CLI with `--reset-volumes --restore` to clean up bad on-disk state before restoring.

This command is useful for:
- Rolling back to a known good state after accidental data corruption
- Point-in-time recovery from a backup
- Testing with a consistent dataset

The timestamp format matches the snapshot filename: `YYYYMMDD-HHMMSS.fff` (e.g. `20260315-120000.000`). You can find available timestamps by listing the snapshot files in the `snapshots/<table>/` directory.

Important notes:
- This PRAGMA cannot run inside an explicit transaction
- Indexes and views are automatically preserved via `ddl-{timestamp}.bin` saved by `PRAGMA SNAPSHOT`
- Tables created after the snapshot will not exist after restore
- Backup snapshots in `snapshots/` are preserved (not deleted) for future restores

For recovery from corrupted databases where `Database::open()` fails, use the CLI with `--reset-volumes` to clean up bad on-disk state before restoring:

```bash
stoolap -d "file:///path/to/db" --reset-volumes --restore
```

#### dedup_segments

Previously used to fix ghost duplicate rows across cold segments. Deduplication is now handled automatically during the seal/compact cycle, so this pragma is a no-op.

```sql
PRAGMA dedup_segments;
```

### Maintenance

#### vacuum

Performs manual cleanup of deleted rows, old version chains, stale transaction metadata, and triggers index compaction (e.g., HNSW graph rebuild when tombstone ratio exceeds 20%).

```sql
PRAGMA vacuum;
```

**Note:** This PRAGMA cannot run inside an explicit transaction.

## Connection String Parameters

All PRAGMA values can also be set via the connection string:

```
file:///path/to/db?checkpoint_interval=60&compact_threshold=4&keep_snapshots=3&sync_mode=normal&hot_max_rows=262144
```

`group_cache_mb` has no connection string form; the cache is process-wide and is set with the PRAGMA.

Legacy parameter names are accepted for backward compatibility:
- `snapshot_interval` maps to `checkpoint_interval`
- `snapshot_compression` maps to `compression` (sets both `wal_compression` and `volume_compression`)
