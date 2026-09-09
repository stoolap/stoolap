# Storage implementation progress

Baseline: `9cf15ea4`. Each phase is a separate dependent PR. The previous
phase is the next PR's base until the stack is merged. This keeps each diff
independently reviewable without claiming unmerged work is shipped.

| Phase | Deliverable | Status | Review | PR |
|---|---|---|---|---|
| 0 | K1–K8 concrete protocols and validation gates | Design complete | Passed after corrections | [#115](https://github.com/stoolap/stoolap/pull/115) |
| 1 | Fallible access and complete statement rollback | Implemented; local gates and CI passed | Passed after corrections | [#116](https://github.com/stoolap/stoolap/pull/116) |
| 2 | Chunked arena and retained-hot accounting | Implemented; local gates and CI passed | Passed after corrections | [#117](https://github.com/stoolap/stoolap/pull/117) |
| 3 | Coherent two-layer execution | Implemented; local gates passed; final CI running | Passed after corrections | [#118](https://github.com/stoolap/stoolap/pull/118) (draft) |
| 4 | V5 envelope and bounded streaming seal | Pending | Pending | — |
| 5 | Remover, durable WAL/catalog and pressure seal | Pending | Pending | — |
| 6 | Paged reads, four ledgers and DML preflight | Pending | Pending | — |
| 7 | Explicit clustering and compatibility fallback | Pending | Pending | — |
| 8 | Identity-first compaction and bounded migration | Pending | Pending | — |
| 9 | Index ownership, write batching and allocation reduction | Pending | Pending | — |

Backup and restore are outside this sequence. The temporary lifecycle probe
in examples is used for local measurements; it is not an engine feature.

## Phase 0

The normative protocols are in [contracts.md](contracts.md). They refine the
accepted [plan](plan.md) with read epochs, stable source LSNs, bounded witness
records, pressure reservations and a file-backed writer result. Claimed rows
are deferred by the remover until commit/abort instead of guessing a tombstone
sequence. All implementation phases remain pending until their tests,
measurements and review are recorded here.

Independent review covered MVCC/publication/recovery, format/compatibility,
and memory/progress. Re-review found no remaining blockers after correcting:

- permanent in-flight exclusions in seal/GC horizons;
- claim ownership through terminal transaction outcome;
- all-table publication undo and indeterminate COMMIT-I/O handling;
- monotonic durable manifest/catalog installation;
- deletion witness retirement gated by the durable WAL replay floor;
- a complete legacy upgrade generation and Phase 4 identity gating;
- streaming/subdivision of large V4 data blocks, not only metadata.

The baseline lifecycle probe passed row/checksum and allocator-balance checks
with 20,000 rows/64-byte payloads and a 4,096-row/4,096-byte variant including
concurrent ingest and analytics. These are baseline observations, not achieved
performance gains or proof of the future budget. CI also runs for dependent
storage PR branches so each phase receives its own checks.

## Phase 1

Required cold reads now return errors through point access, constraints,
optimized aggregates, parallel workers, windows and correlated expressions.
A failed statement rolls back only its own writes and claims. All-table
publication retains row/index/counter/tombstone undo and old/new UNIQUE-key
reservations until the outcome is known. Indeterminate durable COMMIT errors
fence the engine until recovery. Index builders validate detached state before
publishing it; first-seal generation tracking includes generation zero.

Independent review reproduced and drove fixes for cold-key reservations,
VACUUM racing publication undo, first-seal generation loss, partial index
installation, and correlated-query error swallowing. Regression tests retain
those cases. The default test targets completed across the full-suite run and
its continuation; the corruption fixture was corrected to damage the encoded
block while retaining a valid directory length, then passed. Final default
unit tests: 2,050 passed. Serial feature-enabled volume tests: 71 passed after the final filter change.
Clippy (all targets, test-failpoints and FFI), no-default compilation and the
five enabled doctests passed. Failpoint unit tests are feature-gated and run
serially, including coverage, so injections cannot hit unrelated unit tests.

The first five-pair lifecycle comparison caught two avoidable regressions:
repeated lazy-column access in the aggregate row loop, and oversized pooled
transaction maps surviving terminal publication because clear bypassed the
existing drain/shrink policy. Aggregate inputs now bind once on the first
matching row; terminal map drainage restores the existing pool policy after
undo is no longer needed. A capacity regression covers commit and abort.
A profiled dictionary filter now tests eight leading IDs at a time and uses
an exact full-mask path for dense matches. Checked windows cover both IDs and
NULL flags before any output is appended. Independent review and exhaustive
mask/offset/partial-block tests passed.

Final five-pair measurements passed the acceptance gate. Warm aggregate p50
changed by +0.59% on narrow rows and +2.73% on wide rows. INSERT allocations
fell 15.9%; UPDATE allocations fell 11.75–12.10%. The short narrow INSERT p50
was +6.07% (124 ns), below twice its run range; a longer, otherwise matched
10,000-operation comparison measured +4.15% p50 and +4.19% total time. Wide
INSERT p50 was unchanged. Retained and peak requested heap bytes stayed near
baseline. No hard memory bound or competitor claim follows from this probe.
The exact fixture, source digest, medians, ranges and limitations are recorded
in [measurements](measurements/README.md). Clippy with FFI and failpoints,
no-default compilation and final targeted tests passed.

## Phase 2

The current accounting scope covers owned hot row payloads, arena and COW
capacity, retained previous versions, hot index storage, transaction histories,
publication undo and key reservations, pending cold tombstones, registry records
and engine-owned idle/leased transaction and index scratch buffers. Charges
follow shared allocation owners and remain after a table or engine handle drops
while a reader still holds the allocation. Requested capacity and conservative
structural bounds are reported separately; neither is a process RSS estimate.

Resident schema/catalog containers (including the index directory), decoded cold
columns and cold metadata, query results/scratch and maintenance buffers are not
part of this Phase 2 scope. Their complete lifetime accounting and reservation
belong to the later ledger stages; the computed minimum budget must include
resident catalog and root capacity before the full-engine limit is enabled.
MEMORY_STATS names these exclusions and reports hard_limit_enforced=false.
Phase 2 does not claim a hard limit or a complete inventory of engine memory.

The arena uses fixed 262,144-slot chunks and monotonic, non-reused chunk
identities. Runtime seal thresholds do not affect address decoding. Empty chunks
release their row and metadata buffers, while LSN receipts can outlive a chunk
until the publication that needs its WAL records reaches a terminal outcome.
Source LSN minima survive update, delete and rollback instead of being inferred
from the newest commit marker.

Payload accounts follow CompactArc and SmartString allocation ownership through
cloning, mutable detachment and final destruction. Row ingress certifies the
complete payload graph once; an allocation belonging to another engine is copied
into the destination account. COW roots keep their structural and payload charges
while readers retain them. Index accounting covers hash buckets, conservative
B-tree nodes, bitmap storage and HNSW buffers. Transaction history, publication
undo and UNIQUE-key claims keep their charge until their actual backing is freed.

Independent review required explicit replacement buffers for arena and receipt
growth: a moving allocator can keep both old and new buffers live during growth.
Unchanged allocator probes now report 459,592 actual peak bytes versus 459,608
accounted bytes for the arena, and 93,604 versus 93,620 for 1,024 same-chunk pins.
The 16-byte difference is the account object's existing conservative allowance.
Retained totals are unchanged; the fix accounts the temporary overlap. Single-pin
receipts remain inline, and partial promotion failure preserves the original pin.

The integrated full-suite run passed 5,444 tests and exposed two transparent
index-wrapper fixture failures; forwarding the underlying allocation account
fixed both, and all 11 tests in that target passed. The resulting revision also
passes the complete Linux, macOS and Windows CI test jobs. Final counter/inline
owner changes pass all 2,136 serial failpoint-enabled library tests, all-target/
all-feature Clippy, Rust 1.88 all-feature and no-default compilation. All final remote CI checks pass, including Linux, macOS, Windows, coverage,
MSRV, lint, feature and release checks; PR #117 is ready for review.

Five alternating pairs for each lifecycle case pass the predeclared latency
and noise rule. INSERT allocation calls fall 2.69–2.70%, while warm aggregate
p50 changes by +0.29% narrow and -0.71% wide. The longer INSERT median retains
a +6.05% cost (126 ns, below its 166 ns twice-range threshold), reduced from
the prior +12.19% failing attempt by eliminating account-reference churn for
inline transaction buffers. The report preserves that failed attempt and the
residual cost. Hot point lookup and short INSERT percentiles also rise within
the measured noise gate; this phase does not claim every hot operation is faster.

Concurrent wide writer p50/p99 improve 25.41%/20.73%. Short INSERT peak heap
increases 14.51–15.94%; checkpoint releases more capacity, leaving total requested
heap 3.07% lower on narrow rows and 0.32% lower on wide rows. Exact ranges,
allocation counts, source/binary/fixture identities and portable moving-allocator
proofs are in [measurements](measurements/README.md).

## Phase 3 (in progress)

Statements bind one fixed read epoch to a captured hot root, cold generation
and private-write overlay. Visible hot values and delete markers establish
authority before predicate evaluation. A newer invisible hot version leaves
an older visible cold value available. Registry leases preserve cutoff and
in-flight exclusions together; history pruning respects the oldest live lease.

Typed aggregation and ordered traversal consume that view under snapshot
isolation and seal overlap. Composite dictionary keys compare typed values,
not unrelated local codes. Result ownership releases its epoch on exhaustion,
close and error. Index construction reads canonical committed state and maps
older cold columns through current schema defaults. DML target capture and
fallible RETURNING projection preserve statement rollback boundaries.

Result-cache entries retain the original visibility proof. Commit and immediate
logical mutations invalidate it; an old result cannot acquire a later proof
when insertion into the cache is delayed. Public committed mutation paths and
catalog changes participate in the same invalidation protocol. Independent
review reproduced the raw-version-store escape and verified its correction.

Cold generations retain file identities without a persistent descriptor for
every volume. Positioned reads acquire a short verified lease; rename and
replacement keep aliases tied to the original identity. Destructive DDL prepares
fresh native indexes and accounted hot/cold maps before publication. Detached
storage owners are destroyed after the transfer fence. A paired weak cache
handle retains the cache allocation's charge through the final weak owner.

Independent source review and re-review passed after visibility, cache-proof,
file-identity and allocation-lifetime corrections. Integrated library tests pass
2,247/2,247 after the final parent merge, and 21 SQL/file/atomicity targets pass 166/166. The complete integration run passes 5,619 tests across 284 harnesses, with
63 existing ignored doctests and no failures. All-target/all-feature Clippy
passes after three test-only slice-reference simplifications. Final-parent Rust 1.88 all-feature/no-default and Windows compilation pass.
A subsequent bounded dictionary-selection optimization passes 12 focused
aggregation regressions and all-target/all-feature Clippy after independent
review. Independent volume failpoint tests pass 115/115, final Rust 1.88
all-feature compilation passes, and the raw measurement summaries match an
independent recalculation. All three fresh five-pair lifecycle comparisons pass. Warm aggregate
p50 improves 27.39–29.74%; UPDATE and point lookup also improve. INSERT has
residual costs (+11.11% short narrow, +3.80% long p50), retained with the measured
noise ranges rather than hidden by the aggregate improvements. The failed
pre-optimization warm-query comparison remains in the report. Exact source,
raw measurements and memory tradeoffs are in [measurements](measurements/README.md).
Remote CI then exposed a consumed pressure-seal request while an older reader
prevented progress. Releasing the reader did not rearm the request. A bounded
100 ms retry fixes the actual admission loop without relaxing fixed cutoffs,
the existing deadline or test limits. Its deterministic regression failed
before the correction and passes in default/failpoint configurations afterward;
70 unchanged hot-limit test executions and the fixed-build regression pass.
Independent source review and all-target/all-feature Clippy pass.

Independent recalculation of all raw summaries and final source/binary/fixture
identities passes. All three final-source five-pair comparisons pass again. Warm aggregate p50
improves 24.65–29.74%, with 41.94% fewer allocation calls. INSERT p50 changes
+7.24% narrow, -3.92% wide and +1.90% long. Concurrent writer and cold-start
residual costs remain in the report with their noise ranges; both earlier
comparisons are preserved. The remote CI rerun remains the readiness gate.
