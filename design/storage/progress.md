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
| 4 | V5 envelope and bounded streaming seal | Format foundations implemented; seal integration pending | Foundations passed independent review | — |
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

## Phase 4 foundations

The V5 envelope, bounded directory codecs, positioned page I/O, directory
lookup/walker, column payloads, explicit group ranges and row/source identity
pages are implemented. Production emission remains disabled. Payloads retain
borrowed typed access, including a lossless timestamp path: ordinary timestamps
use eight-byte nanoseconds; wide dates and leap seconds use twelve-byte
seconds/subseconds. Group-local text dictionaries and plain text are supported.

Directory construction accepts sorted descriptors with fixed caller scratch.
An external descriptor sorter now supplies that order without retaining every
descriptor: caller-sized in-place runs, two scratch streams, fixed read/write
batches and deterministic adjacent merge passes. Logical spool lengths exclude
stale tails; each record has a checksum. File ownership, reservations and
durability remain the caller's responsibility. No module opens or deletes files.

Independent review and re-review passed after fixing UTF-8 rescans, exact deep
directory validation, timestamp range/leap preservation and mixed legacy/new
source conversion. Current V5 tests pass 86/86. Allocation integrations pass
6/6 for shared envelope/directory/identity operations, 3/3 for columns and
compression, 1/1 for external descriptor sorting and 1/1 for complete payload
emission, all with zero allocation calls inside the
measured codec operations. Caller-owned scratch and test backing storage are
outside those allocation meters; these results do not prove an engine budget.
All-target clippy with test failpoints and Rust 1.88 all-feature compilation pass.

Logical volume shape is now checked before a directory exists. The staged
writer's checkpoint assertion is separate from installed-file read evidence;
encoding does not authorize legacy decoding or acknowledge durability. This
separation passed independent review, wire regressions and unchanged allocation
gates. Completed roots still require their actual directory and exact counts.

The payload producer now emits each RowId-layout group, its identity/source
pages and one column at a time directly into the file sink and descriptor
sorter. Completion checks exact group/column coverage and the sorted descriptor
sequence before writing the directory, root and footer. I/O errors or unwinding
callbacks poison the producer; pure input/scratch failures remain retryable
before any output is written. The descriptor spool belongs exclusively to that
build; its records alone do not authenticate a substituted same-shaped file.

Page compression reuses a caller-owned LZ4 table and exposes the output scratch
reservation in advance. Incompressible input is borrowed as Raw bytes. The
producer can encode a column larger than the stored-page cap when its compressed
representation fits. Independent review and fresh tests cover both Raw and LZ4,
empty volumes, zero-column groups, 65-group directory boundaries, fault and
panic handling, and zero allocations during the complete emission loops.

The stage remains incomplete: bounded payload capture/spill, actual streaming
seal integration, reservation ownership, durable identity activation and
replacement of the eager V4 readback path are still required. The actual cold
reader and engine-wide admission guarantee remain later integration gates.
