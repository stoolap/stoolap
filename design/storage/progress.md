# Storage implementation progress

Baseline: `9cf15ea4`. Each phase is a separate dependent PR. The previous
phase is the next PR's base until the stack is merged. This keeps each diff
independently reviewable without claiming unmerged work is shipped.

| Phase | Deliverable | Status | Review | PR |
|---|---|---|---|---|
| 0 | K1–K8 concrete protocols and validation gates | Design complete | Passed after corrections | [#115](https://github.com/stoolap/stoolap/pull/115) |
| 1 | Fallible access and complete statement rollback | Implemented; local gates passed | Passed after corrections | [#116](https://github.com/stoolap/stoolap/pull/116) |
| 2 | Chunked arena and retained-hot accounting | Pending | Pending | — |
| 3 | Coherent two-layer execution | Pending | Pending | — |
| 4 | V5 envelope and bounded streaming seal | Pending | Pending | — |
| 5 | Remover, durable WAL/catalog and pressure seal | Catalog foundations implemented; lifecycle integration pending | Foundations passed independent review | — |
| 6 | Paged reads, four ledgers and DML preflight | Bounded legacy decompression and column subdivision implemented; reader and admission integration pending | Decoder and adapter passed independent review | — |
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

## Phase 5 catalog foundations

Stable table/incarnation/column identities and immutable schema history now
retain introduction defaults and historical DDL/FK/index bindings. Catalog
validation checks exact revision lifetimes, name reuse and covered DDL effects;
it does not establish durable installation or WAL coverage by itself.

The STCG codec streams immutable generations without a whole-catalog buffer.
Decoding checks explicit wire lengths, counts, nested record boundaries, CRCs,
strict tags and EOF before publishing the reconstructed model. Requested
metadata and payload quotas precede reservations; one mutable history builder
avoids repeatedly cloning the growing history. All Value variants retain their
representation, including signed zero, NaN bits and wide/leap timestamps.

Independent review and root validation passed 33 catalog tests. Three allocation
tests passed, including zero allocation calls while encoding a borrowed 1 MiB
default. The writer receives that original payload pointer. Feature-enabled
clippy passes. Decode quotas are conservative structural limits, not the final
retained-allocation ledger or an RSS guarantee.

This stage remains incomplete: engine DDL/WAL identity wiring, complete legacy
bootstrap, durable installation/receipts, witness/remover coordination and
memory-pressure activation are still required. The codec descriptor cannot
acknowledge durability, and production lifecycle behavior is not enabled.

## Phase 6 legacy decoder foundation

The inactive raw LZ4 decoder streams a block from a caller-owned reader into a
caller-owned sink using a 64 KiB history ring and bounded input/output scratch.
Literal copies and overlapping matches use bulk spans. Declared stored/decoded
lengths bound every operation, and an error or unwinding callback leaves the
decoder aborted. Short and interrupted I/O, invalid reported byte counts and
following-block boundaries are covered. No whole-block allocation is required.

Root and independent review passed after correcting byte-count validation.
Seven unit tests and two allocation tests pass. A real file-to-file fixture
larger than 32 MiB used 80 KiB caller scratch and observed zero allocation calls
during decoding; every output byte was verified. This raw fixture does not yet
exercise actual V4 column conversion. Rust 1.88 all-feature compilation and
no-default/failpoint clippy pass.

The decoded-spool adapter now validates all six V4 column encodings and gathers
byte-capped physical row ranges into caller-owned buffers. Noncanonical valid
Bytes offsets and NULL dictionary IDs preserve existing decoder behavior.
Preflight lengths remain u64 until compared with actual buffer capacity;
capacity errors leave output unchanged. Partial I/O and panics abort the handle.
Borrowed row metadata occupies two machine words, including distinct NULL and
non-NULL empty values, without an extra flag word.

Nine adapter units and the allocation test pass in author and independent runs.
The actual V4 Vector fixture has 65,536 rows and 34,668,560 decoded bytes. Raw
LZ4 decoding, spool validation, subdivision and every value/NULL check use zero
allocation calls in the measured interval. Decoder/adapter buffers total 152 KiB;
the verification cell adds 512 bytes. Fixture construction, file backing and OS
memory are outside that working-buffer statement. Rust 1.88 all-feature
compilation and focused no-default/failpoint clippy pass.

Paged V4 metadata/dictionaries, oversized-cell streaming, outer checksum
validation, V5 reader activation, managed file ownership, the four retained-memory
ledgers and bounded DML admission remain required. Scratch reservations and
temporary file lifetime belong to that later coordinator; these modules neither
install files nor establish an engine-wide memory bound.
