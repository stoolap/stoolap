# Storage implementation progress

Baseline: `9cf15ea4`. Each phase is a separate dependent PR. The previous
phase is the next PR's base until the stack is merged. This keeps each diff
independently reviewable without claiming unmerged work is shipped.

| Phase | Deliverable | Status | Review | PR |
|---|---|---|---|---|
| 0 | K1–K8 concrete protocols and validation gates | Design complete | Passed after corrections | Opening |
| 1 | Fallible access and complete statement rollback | Pending | Pending | — |
| 2 | Chunked arena and retained-hot accounting | Pending | Pending | — |
| 3 | Coherent two-layer execution | Pending | Pending | — |
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
