# Storage lifecycle contracts

Status: Phase 0 design under review. Baseline: `9cf15ea4`.
The accepted scope and phase boundaries are in [plan.md](plan.md).
This document selects the protocols used to implement K1–K8. It does not
declare an unimplemented budget or benchmark target achieved.

## Shared vocabulary and lock rules

`TableId` is a persisted, never-reused u64. `Incarnation` increases on
TRUNCATE and identifies a particular lifetime of the table's rows.
`BuildId` identifies one capture/publication/removal pass. Row identity is
`(TableId, Incarnation, row_id)`; a slot address is only an acceleration.
Commit sequence orders visibility. WAL LSN orders recovery. Neither may
be substituted for the other.

The existing engine-wide seal fence coordinates catalog/checkpoint cuts.
The per-table seal fence coordinates cold publication and hot transfer.
Existing writers take its shared side; a transfer takes its exclusive side.
Table fences, when more than one is required, are acquired by TableId.
Within a table the order is claims, versions, arena. Registry inspection
does not call back into a table while the registry mutex is held. A lookup
releases its arena guard before a B-tree fallback.

No page read, decompression, file operation, user callback, or waiting for
memory is permitted under a publication/transfer fence or arena guard.
Buffers and map capacity are reserved beforehand. A stale generation
causes release, re-preparation and retry, never loading pages under a fence.
An in-flight commit's long lifetime is represented by row claims, a counted
table publication lease and WAL retention ownership, not by holding a fence
across its WAL commit-marker write. Acquiring a publication lease and a DDL
incarnation change are serialized by the short table fence. Destructive DDL
waits outside that fence or returns a conflict until publication leases end.
Three optimistic retries are allowed before returning a retryable conflict;
maintenance yields and requeues instead of monopolizing a writer lock.

## K1: chunk identity and allocation

`ARENA_CHUNK_SHIFT = 18`, `ARENA_CHUNK_ROWS = 1 << 18` and
`ARENA_CHUNK_MASK = ARENA_CHUNK_ROWS - 1` define address encoding.
`ArenaId` packs `(chunk_id << 18) | slot` with checked arithmetic; the
existing nonzero representation adds one with overflow detection.
Chunk IDs never move or wrap. An exhausted address space returns an error.

The directory is a sorted vector of `(chunk_id, Chunk)` entries with binary
lookup. Empty chunks are physically removed; a missing ID resolves to gone.
Each chunk owns parallel metadata and payload vectors. Capacity grows on
demand, capped at 262,144 slots; creating a small table does not allocate a
full chunk. Retained capacity, including spare capacity, is measured.
Only the active chunk reuses free slots. Freeze retires its free list and
opens an empty, initially unallocated active chunk. Frozen slots remain
mutable until transferred. An in-progress capture prevents reuse of its
identity even across DELETE/reinsert.

A row-id arithmetic probe is speculative and must validate `meta.row_id`.
Arbitrary explicit PK order, gaps and reuse always retain a B-tree fallback.
Metadata carries the latest row mutation's required DML LSN; the chunk's
first LSN is the minimum still needed by its remaining rows and pending
durability receipts. A commit touching two chunks updates both pins.

Allocation ownership follows the last owner. The arena counts structural
capacity separately from shared payload bytes. The payload's charge is
released with its final Arc owner, including an index, read view or undo
record. Phase 2 reports accounting and does not enforce a new hard limit.

## K2: registry leases and fixed seal eligibility

Introduce an RAII registry read lease. Under the registry mutex, capture
the sequence and register the lease before GC can advance. Sequence
allocation and the transition to Committing take place under that same
mutex; incrementing the sequence before acquiring it leaves a capture gap.
Commit sequence mappings are retained whenever a transaction snapshot,
statement read lease or build lease could need them, even in ReadCommitted.

A read epoch's retention horizon is the minimum of its cutoff and one less
than its smallest excluded committing sequence (if any). Its exclusions
remain part of that horizon even after those transactions complete. A build
chooses the minimum of every live epoch's retention horizon and the current
safe completed-commit cutoff (the first in-flight commit sequence minus one,
or current sequence if none). This may conservatively leave rows hot.
It never changes between batches. Lease registration and bound selection
are atomic with respect to begin/commit/GC. A slot is eligible only if its
recorded transaction is committed and its sequence is at or below this bound.

A build is `Capturing -> Written -> Published -> Draining -> Durable`.
Before publication it may be cancelled and its scratch files removed.
After publication it must finish transfer or preserve enough hot/WAL state
to retry; cancellation cannot silently release a durability pin. An
incarnation change cancels work for the old incarnation at every transition.

## K3: atomic transfer and versioned tombstones

A transfer batch is capped at 2,000 records and also stops at a measured
fence-time budget. It preloads all records and reserves tombstone/index
capacity before taking the table's exclusive fence and versions write lock.

For each captured identity:

1. Revalidate incarnation and capture token. If the current head equals the
   captured transaction and no transaction claims the row, remove that hot
   copy and its hot index entries.
2. If a committed version superseded it, install the first superseding
   commit sequence from K5 before relinquishing the hot shadow.
3. If a claimant or unresolved publication is present, leave the row and
   capture record pending. Do not invent a current-sequence tombstone;
   resolve the claimant's commit/abort and retry a later batch.
4. Clear a superseded tombstone only by matching the captured sequence in
   this same critical section. Never clear tombstones solely by row id.

Update overlap counters for finalized records; update live-row counters
only for actual live hot removals. A DELETE already changed the live count.
Bitmap/tombstone/cold-generation changes are published together. Serialization
and fsync occur after releasing the fence, with K6 retention still held.

Readers use K4 throughout overlap. HNSW's existing lifetime rules remain;
row-based hot index removal does not remove its cold index entries.

## K4: coherent read views without long reader guards

`ReadEpoch` contains a cutoff and the in-flight committing sequences at
capture. A statement in ReadCommitted creates one at statement start;
SnapshotIsolation keeps its transaction-start epoch across statements.
All state is captured under the registry mutex. A commit that was in flight
cannot become visible halfway through a query merely by completing later.
Its sequence remains excluded from that epoch. This exclusion also applies
to versioned cold tombstones. Own uncommitted changes form a separate overlay.

For each table, briefly take its exclusive transfer fence, clone the existing
CowBTree root and immutable cold generation, then release. This reuses the
existing B-tree snapshot mechanism; it does not add hot store generations
or copy all rows. Concurrent commits may exist in a cloned root, but the
fixed epoch determines visibility. The read lease protects required history
until capture completes. Normal history GC keeps the version needed by the
oldest read lease's retention horizon and newer versions; max history must not override a live
reader. Retained history and COW nodes participate in the hot budget.

The captured table view owns the selected hot root, cold generation and
epoch. It produces `HotState::Value`, `HotState::Deleted`, or
`HotState::NoVisibleVersion` for an identity. Only Value and Deleted shadow
cold. Apply the query predicate after this decision. Apply the transaction's
own overlay before committed state. A post-snapshot cold UPDATE with no
visible hot predecessor must continue to read the old cold value.

Typed aggregation streams the captured hot view into the same accumulators
as cold groups. It does not allocate `Row` objects or hold an arena guard
for the statement. A query may use a budgeted/spilled authority set or probe
the captured tree; it may not materialize an unbounded full-hot hash set.
All cold access is fallible. Errors discard the aggregate rather than
returning a plausible partial answer. Composite dictionary keys compare
typed values or query-local canonical IDs, never unrelated local codes.

## K5: bounded supersession witnesses

The capture owns a preallocated witness table, keyed by stable arena slot
plus BuildId, for the subset of rows actually selected by the build. A chunk
has at most one active capture. The cap limits selected rows as well as
bytes; the next pass handles residual rows. Do not allocate a full-chunk
witness table when only a small batch fits the maintenance reservation.

A witness contains the captured transaction, the first committed superseding
sequence (initially absent), and at most one pending superseding transaction
and sequence. Capacity and hash load-factor slack are charged before capture.
While holding the arena guard, install each witness before releasing the
captured `(row_id, txn_id, Arc)` payload. There is no unprotected interval.

Updating a watched slot performs a bounded lookup in this already allocated
table. Before replacing a pending superseder, resolve it through the registry:
committed installs the first sequence once, aborted is discarded, and active
or committing remains pending. Row claims prevent a second publisher from
overwriting an unresolved transaction. The current publisher then records
its pending sequence if no first committed sequence is known. Move claim
release out of the existing per-table commit: publication claims remain owned
by the transaction's PublishHold until its terminal registry outcome and,
on abort, completion of version/index undo. A failed WAL commit marker cannot
release a slot for another publisher while leaving a pending witness behind.
This changes the present early-release path; an observer publish counter
alone is not a per-row exclusion gate. Repeated
updates after a first committed superseder require no history walk.

The remover resolves pending witnesses by the same rule. It defers a pending
transaction and never stamps an aborted one. Witnesses store no payload
history. Ordinary MVCC retention still serves reader epochs under K4; K5
does not turn off pruning. The witness is removed after its transfer record
is finalized or its unpublished build is cancelled.

If reservation cannot cover a new capture, shrink the selected subset down
to a group/batch or yield before capture. An active witness never needs to
grow: one pending and one committed stamp suffice. This is the cap fallback;
discarding evidence or scanning an arbitrarily long history is forbidden.
Witness lookup has expected constant cost with a fixed maximum load factor;
adversarial collision costs are included in performance validation.

## K6: durable receipts and replay identity

Every committed mutation retains its required DML LSN before the commit
marker. A pending transaction retains its first DML LSN until abort or all
affected data is durably covered. Removing the final live row from a chunk
moves its retention ownership to a pending manifest receipt; it does not
release it. Tombstone-only changes and DDL have their own receipts.

Publication prepares a new immutable manifest generation. Durability order:
finish/checksum data files, fsync files, rename and sync their directories;
write/fsync the manifest generation, atomically install it and sync its
directory; only then acknowledge its receipt. A per-table durability mutex,
separate from the transfer fence, serializes installation and acknowledgment.
An older generation may never replace an installed newer generation; stale
prepared files are discarded. Catalog installation uses the same monotonic
rule. Concurrent manifest changes do not get acknowledged by an older
generation's fsync; a receipt lists the exact covered generation and mutations.

Capture the minimum required LSN across live hot rows, pending commits,
unacknowledged receipts and catalog DDL coverage under the global fence.
Pass `minimum - 1` to inclusive WAL truncation, with zero/empty cases checked.
The cut also records a current WAL LSN ceiling, so truncation never includes
subsequent appends. An append publishes its retention ownership before its
LSN becomes visible to this cut. The actual file I/O and rotation run outside
this fence. A freeze requests
rotation; it does not fsync while holding the arena lock.

V5 stores a paged source-DML-LSN column (8 bytes per row before compression)
and tombstones retain their source LSN. Recovery compares source LSNs to
make partially covered transactions idempotent: an older/equal mutation
cannot overwrite a newer durable row or deletion. This metadata cost is
additional to the locator and is included in disk and cache measurements.
V4 inputs have a distinct LegacyBase identity established by the upgrade
barrier below. Conversion never assigns its current LSN to an older V4 row:
that would suppress a newer unsealed update during replay. Their checkpoint
horizon is established only by the complete upgrade generation.

Transaction publication uses an all-table prepare/apply/finalize protocol.
Prepare validates every touched table and reserves complete undo before
publishing the first. The undo owns previous version heads, arena payload/meta,
counter deltas, index entries and any changed committed cold tombstones.
It restores only this transaction's claimed identities, preserving unrelated
concurrent writes. Keep local state and publication leases until the outcome.
Any failure before a durable COMMIT restores every touched table while claims
remain held, marks the transaction aborted, then releases claims and leases.
In particular, replace the baseline `any_committed` branch that completes a
partially failed transaction. Cold tombstones are undone with the same outcome.
After the COMMIT marker is durable, only pre-reserved infallible finalization
remains before registry visibility. No later read or allocation can turn an
acknowledged commit into a partial failure. Phase 1 supplies complete undo
before adding fallible commit-time reads; Phase 5 extends it to witnesses,
source LSNs and the new WAL framing. A commit-marker write/fsync error with
indeterminate durability is not a proven abort. Fence the database from
further reads/writes and require recovery to resolve the outcome; do not
report a definite rollback and continue against a possibly committed WAL.
Ordinary undo applies when the marker is definitely unsubmitted/invalid or
durably removed by the WAL failure protocol. Failpoints cover both cases.

WAL batch frames share a transaction ID and one final commit marker. Each
decoded and encoded frame is capped at 1 MiB, below the legacy 64 MiB cap.
A single larger row is fragmented with row/fragment identity and validated
lengths; it is never silently omitted. Recovery's status pass finds complete
transactions; redo consumes their frames in LSN order, bounded by reserved
buffers or a spill for a large row. Missing fragments or checksums return
corruption, not partial commit. The transaction outcome governs every table.

## K7: catalog identity and schema history

Persist a checksummed catalog generation containing the next TableId,
table IDs/incarnations, current names, schema versions, indexed columns,
views and the DDL coverage LSN. DML frames carry TableId, incarnation,
schema version and row identity. DDL carries the same stable identity.
Rename changes name history; it never assigns a new TableId. Drop leaves
a catalog tombstone, and recreating the name allocates a new ID.

Recovery loads the last complete catalog generation, then replays retained
DDL/DML by identity and source LSN. Already covered DDL is idempotent.
Column identity is stable across rename/drop/re-add; replay projects older
schema versions through catalog history and restores recorded defaults.
History can be retired only after the WAL horizon and all referenced volume
schema versions no longer need it. Old name-based WAL is consumed by the
upgrade barrier below; upgrade cannot guess identities across an ambiguous
drop/recreate boundary.

Build, cold generation, locator, cache and compaction keys include TableId,
incarnation and file identity. A stale result from before TRUNCATE cannot
register against a new table with the same name.

### Legacy bootstrap and Phase 4 enablement

Choose a one-time quiescent upgrade checkpoint before accepting new writes
in the new durability mode. First finish legacy recovery using the existing
catalog/WAL semantics, including retained DDL, and reject ambiguous or corrupt
history instead of assigning invented identities. Choose its replay ceiling
G, with no active transactions. Allocate stable table/column identities for
the recovered catalog and stage a new upgrade generation E in a separate
directory that the old manifest root does not reference.

Seal all remaining recovered hot state through the bounded writer, including
updates newer than existing cold V4 rows. Normalize pending deletions into
the staged manifests. Recovery may spill/pressure-seal staged batches rather
than keeping all replayed rows resident. Retain original WAL through G and
the old committed manifest/catalog generation throughout this process.
Unchanged V4 files remain referenced; this does not rewrite all cold payloads.

Fsync every newly required volume, staged manifest and catalog, then install
one checksummed upgrade root naming the exact complete generations and G;
fsync its directory. Only this complete root activates E and permits WAL
truncation through G. Before that point a crash resumes legacy recovery and
discards incomplete E files; after it, recovery loads E and ignores old WAL
at or below G even if truncation had not completed. No mixed set of staged
and legacy manifests is accepted as a complete upgrade checkpoint.

Rows inherited from unchanged V4 files have `LegacyBase(E,G)` source identity.
Their ordering is below any post-barrier mutation with LSN greater than G;
conversion preserves this identity rather than stamping the conversion time.
Recovered updates sealed into E take precedence over old cold copies in E's
manifest. The all-table checkpoint proves the replay floor for both, including
deletions. New DML is exclusively identity-based and uses monotonically higher
LSNs. Retained legacy rename/drop/recreate history is resolved before E is
activated, never by applying old names to the new catalog.

Phase 4 implements and tests the V5 writer/reader, but production V5 emission
remains gated until Phase 5 supplies this durable identity/bootstrap protocol.
There are no zero/fabricated TableIds that a later phase reinterprets. The
old production path remains active until the joint switch. Reverse-order
generation completion and crashes at every upgrade publication boundary are
required tests, including V4 cold row -> newer hot UPDATE -> conversion ->
crash/reopen.

## K8: allocation ownership, reservation and progress

The engine account has hot, query, cache and maintenance ledgers. A shared
allocation owns one charge through its lifetime; moving an Arc between
cache and reader does not free it. Structural capacity, overflow strings,
vectors, index keys, COW nodes, tombstones and registry records are included.
Reallocation reserves the new allocation before releasing the old charge,
including the transient overlap. Counters distinguish requested/retained
allocation bytes from process RSS and allocator fragmentation.

The minimum supported budget is computed from resident catalog/roots and
active structural capacity plus a seal progress reserve and one short-write
reserve. Chunk maximum capacity is not an up-front minimum. A configuration
below the computed floor returns a resource error instead of oversubscribing.

Use default group caps of 4,096 rows and 1 MiB decoded bytes; for a small
budget adapt capture batches and output groups downward. A worker's bound is
`pinned input batch + sort keys + one column buffer + compression output +
directory page + witness table + remover page + merge input pages`.
All terms are reserved by capacity. Merge fan-in defaults to 8 and shrinks
to the reservation; additional passes spill runs. Oversize values use a
reserved single-value path with an explicit maximum, or return a resource
error before mutating a statement. They do not bypass the account.

Admission first evicts unpinned cache entries and requests pressure seal of
eligible, partially filled chunks regardless of row count. It waits only
outside locks. Compaction cannot borrow the seal progress reserve. A writer
holding claims needed by the remover must not wait for that remover: it uses
its pre-reserved commit capacity or fails and releases its own changes.
A long snapshot may legitimately prevent reclamation; a configurable
bounded wait then returns a resource error. Never exceed the budget to
break a wait. An allocation failure must leave existing transactions usable.

Phase 2 enables accounting only. Phases 4 and 5 jointly enable a hard hot
plus maintenance limit and pressure seal. Phase 6 includes query and cold
allocations and enables the engine-wide claim. Step 6 therefore depends on
step 5 as well as step 4. Before this point MEMORY_STATS explicitly reports
the enforced scope. The plan's 16/64 MiB acceptance applies to the full
account only after Phase 6.

## Fallible access and statement atomicity (Phase 1)

Every disk-backed group, identity, existence, constraint and aggregate read
returns Result. Unsupported optimization (`Ok(None)`) differs from I/O or
corruption (`Err`). Existing Result APIs are reused. Checked offsets and
lengths precede allocations; malformed input never becomes NULL/false/zero.

Before DML, create a statement checkpoint of local version state, acquired
claims/write set, pending cold tombstones and relevant per-transaction
metadata. Restore only changes since that checkpoint on failure; earlier
successful statements survive. Timestamp-only rollback is insufficient when
a cold row is claimed and its next key read fails before a local version or
tombstone is added. Compiled INSERT paths use the same guard as dispatched
statements. Savepoints and outer transaction rollback keep their semantics.

## Streaming format and residency interfaces

Phase 4 introduces a V5 file writer and reader together. The writer returns
a completed file-backed volume handle (root, file identity and directory
locator), never a store containing every compressed block. New seal output
must not immediately pass through the old whole-file/whole-column reader.

The format uses a 64-byte header, independently checksummed blocks and metadata
pages, a paged directory written last, a bounded root and a 64-byte footer.
The header identifies `STV5`, version, required feature flags, TableId,
incarnation and volume ID, with reserved bytes and a header checksum.
The footer identifies version and size, exact file length, root offset,
stored/decoded root lengths, root codec/checksum and footer checksum.
Every page descriptor carries an explicit Raw or Lz4Block codec, u64 offset,
stored/decoded lengths and a stored-byte checksum. Equal lengths do not imply
a codec. Numeric fields use little endian and checked arithmetic.

Directory keys identify section, column and group/page. Sections include
column blocks, row IDs, source LSNs, locator arrays, dictionaries, group/run
metadata, blooms and constraint pages. A bounded-fan-out directory tree keeps
the resident root bounded even when volume/group counts grow. The root summary
stores counts, explicit row-ID extrema, window bounds and the actual layout
kind; group ranges are explicit from the first V5 version.

Reader validates header/footer/root checksums and file bounds before following
page pointers. Stored-byte checksums precede decompression, whose returned
length must exactly match decoded length. It rejects invalid tags, unknown
required features, duplicate entries, cycles, excessive directory depth,
overlapping/gapped group ranges and inconsistent counts. Decoded fixed-width
arrays have exact lengths; variable offsets, dictionary IDs, UTF-8 and null
flags are checked before indexing. Payload parsers consume their entire slice.
Unknown optional pruning metadata disables pruning instead of proving absence.
Read buffers are reserved before allocation or decompression. Parallel file
reads use positioned I/O or a short seek/read mutex; a cloned File does not
provide an independent seek cursor.

Capture sorts immutable captured row references, never mutable arena offsets
read again later. When captured payloads do not fit, sorted runs own their
serialized payloads on disk. A merge carries row identity, version LSN and
payload position together. Directory/dictionary accumulation is paged or
spilled; neither may grow silently with volume size.

Phase 6 replaces tier promotion with group/page sources, single-flight loads
and probation/protected cache queues. A scan enters probation; repeated
access promotes. Pins preserve charges after eviction. Idle sweep releases
unpinned pages without requiring a checkpoint. Compressed and decoded copies
are not retained together by default.

Legacy V4 has one raw LZ4 metadata block, not an LZ4 frame. Its compatibility
decoder uses a bounded 64 KiB history ring and streams validated metadata to
pageable/spooled backing; a whole-output block decoder is not sufficient.
The whole-file CRC is streamed once. The same bounded raw-LZ4 decoder also
handles V4 column blocks; raw blocks stream directly to validated spool
backing. The old 65,536-row physical group is not an admission atom. Expose
byte-capped logical subgroups/ranges through the fallible group-view interface
without ever allocating the old whole decoded block. The spool records
checked row/value offsets; reusable buffers gather only the requested range.
All requested columns agree on the physical row range. Old zone maps may
serve as conservative bounds for subgroups, never as stronger new facts.
A valid group of 65,536 Bytes/Vector values of 512 bytes exceeds 32 MiB even
though no value is large; it must read and convert within 16 MiB. This is a
required compatibility test, not an oversized-single-value exception.
V4 dictionary IDs still refer to the volume
dictionary: fetch only referenced dictionary pages and re-encode a decoded
group's IDs to a group-local dictionary, rather than pinning the entire old
dictionary. Conversion is capped by bytes and tracked by format/layout
progress, including clean target-sized volumes. The raw-LZ4 adapter is tested
differentially against lz4_flex on valid blocks and with truncated tokens,
invalid/zero offsets, overlapping matches and decoded-length mismatches.

Cold DML preflight prepares bounded lookup proofs against a recorded cold
generation outside the fence. Under the fence it validates the generation
and required hot conflicts. On a changed generation it releases and retries.
It does not pin all candidate key blocks for a whole unbounded statement;
statement batches carry already checked candidates and budgeted proofs.

## Clustering and compaction enablement

Clustering is an explicit table declaration with equality keys, ordering
suffix and window key. Unclustered output remains row-id ordered. V5 records
layout version, physical row identity, explicit row-id extrema and a paged
row-id/physical-position locator. All physical identity consumers use these
accessors before clustering can be enabled.

Phase 7's old-compactor fallback reads identity and position together and
labels its output with the layout actually emitted. It obeys the same memory
budget and precedence rules. No first clustered file is written until this
fallback and mixed-format reopen tests pass.

Phase 8 selects contiguous manifest input runs. Identity merge chooses the
newest authoritative row per identity; budgeted/spilled emit bitmaps feed
the clustering-order emission. Cross-window inputs produce window-bounded
outputs installed atomically at the input position. Tombstones are retired
only for completely covered overlap components, safe snapshot horizons and
a durable WAL replay floor past the deletion's source LSN. A durable deletion
witness does not itself pin WAL after its receipt is acknowledged, but it
remains available to redo until that floor advances. Otherwise retained WAL
for another table could replay an old INSERT after compaction erased the
only evidence of its later DELETE. Equivalent persisted coverage fences may
replace individual witnesses only if recovery consults them before redo.
Files remain leased by readers after manifest replacement; deletion waits
for the last file lease, independently of whether any decoded page is cached.

## Validation gates

Each phase has an independent review, correction and re-review before its PR
is considered ready. A dependency is not complete merely because its APIs
compile. Production checks include formatting, relevant unit/integration
tests, failpoint tests where applicable, default-feature compilation and a
no-default-feature build. Broad suite runs follow integration milestones.

The lifecycle probe records input size, value width, repetitions, build,
allocator and cache state with p50/p95/p99 plus allocator live/peak bytes and
allocation calls. Reopening a file is not claimed to flush the OS cache.
Compare five alternating baseline/candidate runs; report medians and spread.
The initial writer acceptance threshold is no regression exceeding both
5% and twice the measured run-to-run noise at equal offered load. A larger
regression blocks readiness pending diagnosis. Benchmark instrumentation is
kept identical across compared builds, and its overhead is reported.

Required adversarial schedules: snapshot during build; an epoch excluding an
in-flight commit followed by that commit completing and a later seal;
snapshot before cold
UPDATE/DELETE; predicate-changing hot UPDATE; DELETE between transfer batches;
claim then cold read error; commit/abort while a witness is pending; repeated
wide updates while remover stalls; arbitrary PK order and slot reuse; stale
incarnation/handle; pressure below seal row threshold; long reader retaining
history; manifest fsync failure before WAL truncation; rename/drop/recreate
with retained WAL; multi-table fragmented transaction and torn frame; colliding
group-local dictionary codes; interleaved windows; bounded V4 conversion.
