# Storage implementation progress

The [plan](plan.md) and [contracts](contracts.md) define the accepted design.
Each implementation phase has a separate dependent pull request. A component
with no engine caller does not count as an implemented phase.

| Phase | Deliverable | Status | PR |
|---|---|---|---|
| 0 | Lifecycle contracts and validation rules | Design accepted | [#115](https://github.com/stoolap/stoolap/pull/115) |
| 1 | Fallible cold access and statement rollback | Cleanup verified; SQL statement rollback proven against parent | [#116](https://github.com/stoolap/stoolap/pull/116) |
| 2 | Releasable chunks and retained hot accounting | Cleanup verified; SQL arena release proven against parent; performance acceptance pending | [#117](https://github.com/stoolap/stoolap/pull/117) |
| 3 | Coherent hot/cold reads | Cleanup verified; cache freshness proven against parent; lifecycle and performance acceptance incomplete | [#118](https://github.com/stoolap/stoolap/pull/118) |
| 4 | Bounded streaming seal | Incomplete; unintegrated V5 prototype withdrawn | [#119](https://github.com/stoolap/stoolap/pull/119) |
| 5 | Durable lifecycle and pressure seal | Incomplete; unintegrated catalog and WAL prototypes withdrawn | [#120](https://github.com/stoolap/stoolap/pull/120) |
| 6 | Paged residency and engine memory budgets | Frozen until phases 2 and 3 are clean; existing adapters are component work only | [#121](https://github.com/stoolap/stoolap/pull/121) |
| 7 | Explicit clustering | Not started | |
| 8 | Compaction and migration | Not started | |
| 9 | Index ownership and write batching | Not started | |

## Review requirements

Each claimed engine behavior needs a SQL or public-API regression that fails
on the phase's parent. Format, strict Clippy, touched test targets, the default
nextest suite and storage tests with `test-filedb` are recorded in each PR.
Helper allocation tests establish only their own scope. Hot accounting does
not enforce a hard engine memory limit or measure process RSS.

Raw measurements, temporary probes and withdrawn prototypes are kept outside
the repository. PR descriptions contain the relevant measurements, profiles,
ranges and limitations. Historical alternating-run timings require reproduction
under the current measurement protocol before supporting performance claims.

## Remaining Phase 3 aggregate migration

Before #118 leaves draft, the final read-epoch binding cleanup removes the
legacy `SegmentedTable::compute_filtered_aggregates` and
`SegmentedTable::compute_grouped_aggregates` bodies. Captured views remain the
SQL path; unbound `SegmentedTable` handles decline these optional pushdowns
with `Ok(None)`. The existing hot-only `MVCCTable` fallback is outside this step.
The two direct error fixtures must use epoch-bound tables, including a fresh
binding after registering another segment. Their obsolete comparison helpers
are removed with the legacy bodies.

Backup and restore remain outside this sequence.
