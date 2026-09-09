# Storage lifecycle measurements

[Phase 1 results](phase-01.json) record five alternating baseline/candidate
pairs for each case. Baseline production source is `9cf15ea4`. The candidate
production source digest covers sorted `src/**/*.rs`, Cargo.toml and Cargo.lock
as `path + NUL + content + NUL`, excluding design-only commit changes.
The fixture SHA-256 is recorded separately.

The [exact temporary example](lifecycle_probe.rs.txt) is preserved for
reproduction. Copy it to `examples/storage_lifecycle_probe.rs` in both the
baseline and candidate checkouts and build with the same release settings:

```sh
RUSTC_WRAPPER= cargo build --release --example storage_lifecycle_probe
./target/release/examples/storage_lifecycle_probe --iterations 1000 --seed 42
./target/release/examples/storage_lifecycle_probe --iterations 1000 --seed 42 --rows 4096 --payload-size 4096 --concurrent
./target/release/examples/storage_lifecycle_probe --iterations 10000 --seed 42
```

Run the binaries serially in AB/BA order for five pairs, with other builds and
benchmarks stopped. Every run verifies rows, checksums and allocator balance.
The narrow fixture has 20,000 initial rows with a 64-byte text payload; the
wide fixture has 4,096 initial rows with a 4,096-byte text payload. Warm scans
have 1,000 untimed repetitions and at least 10,000 measured repetitions.
The longer DML case increases the DML operations to 10,000 in both binaries;
its later scans therefore see a larger, still matched fixture.

The allocator wraps System with an identical metering lock in both builds.
Times include that instrumentation cost; they are not uninstrumented maximum
throughput claims. Requested bytes and capacities exclude allocator metadata,
fragmentation and RSS. In-place realloc and allocator-internal transient
old/new overlap are not separately observable. First touch after reopen does
not flush the OS page cache. Values are p50/p99 latency or whole-phase time,
with medians and full min/max ranges across runs. This report takes measured
run-to-run noise as the larger within-binary max-minus-min range. A regression
must exceed both 5% and twice that noise to block the gate.

The short narrow INSERT result (+6.07% p50, 124 ns) is retained in the report,
even though its total time changes only +1.63% and its 84 ns run range keeps
it below the combined threshold. The longer comparison measures +4.15% p50
and +4.19% total time. Wide INSERT p50 is unchanged; UPDATE p50 is +0.17% narrow
and -1.36% wide. INSERT allocation calls fall 15.9%, UPDATE 11.75–12.10%.
During concurrent wide tests, writer p50/p99 changed by -10.36%/-7.77% and
reader p50/p99 by -0.97%/-2.03%; the report preserves both workers' operation
counts and all measured latency percentiles.
Warm aggregate p50 is +0.59% narrow and +2.73% wide. First-touch aggregate p50
is lower in both fixtures, but reopen/page-cache observations are deliberately
reported separately from warm-query latency.

Phase 1 does not enforce an engine memory budget. Later phases must repeat
these checks and add their own retention, pressure and durability schedules.

## Phase 2

[Phase 2 results](phase-02.json) compare the final Phase 1 production source
with chunked hot storage and allocation-lifetime accounting. The source digest
also includes the pinned vendor tree. The same fixture, allocator, parameters
and five alternating pairs are used. Every latency gate passes under the
predeclared 5%-and-twice-range rule. This includes a residual latency cost;
it does not establish that every hot operation became faster.

| Metric | Narrow | Wide | 10,000-operation case |
|---|---:|---:|---:|
| Warm aggregate p50 | +0.29% | -0.71% | +0.22% |
| INSERT p50 | +7.86% | +7.07% | +6.05% |
| INSERT p99 | +8.78% | +14.29% | +9.22% |
| INSERT allocation calls | -2.70% | -2.69% | -2.70% |
| UPDATE p50 | +0.16% | +1.37% | -0.72% |
| Hot PK p50 | +6.42% | +10.63% | +6.64% |
| Extra heap peak during checkpoint | -15.56% | 0.00% | -5.51% |

The long INSERT median is 2,083 ns baseline versus 2,209 ns candidate: a
126 ns difference, below twice the larger 83 ns run range. Its total time
is +6.66%; full ranges and every writer sample are retained in the report.
The earlier counter-only attempt produced a stable +12.19% long INSERT p50
and failed readiness. Inline transaction buffers now acquire an allocation
account only when they spill. Their previous zero-byte owners performed
unnecessary account-reference increments/decrements. A simpler counter layout
also removes a redundant retained-byte atomic while keeping the single total
authoritative. Both earlier comparisons, including the failed one, are preserved.

Concurrent wide ingest/aggregation completes 12.56% sooner. Writer p50/p99
change by -25.41%/-20.73%, and reader p50/p99 by +1.47%/-33.79%. These are
instrumented local comparisons, not maximum throughput or competitor claims.

Memory tradeoffs are explicit: extra heap peak during short INSERT phases is
14.51% higher on narrow rows and 15.94% higher on wide rows; the long case is
0.53% lower. More storage is released at checkpoint. Total requested heap still
live immediately afterward falls from 39,863,320 to 38,639,896 bytes on narrow
rows (-3.07%), from 95,375,833 to 95,066,073 on wide rows (-0.32%), and from
43,721,723 to 42,426,299 in the long case (-2.96%). The percentage change of a
negative retained delta describes additional released bytes, not the percentage
reduction of the engine's total memory. The hot account excludes cold/query/
maintenance memory and does not enforce a hard limit in this phase.

Separate [arena](arena_growth_probe.rs.txt) and
[receipt](receipt_growth_probe.rs.txt) probes force reallocations to move while
both old and new buffers remain live. They preserve the original independent
proof's operations and assertions, using the actual public engine types instead
of absolute source includes. Copy them to examples as arena_growth_probe.rs
and receipt_growth_probe.rs, then run with the same release profile. The arena
peak is 459,592 actual versus 459,608 accounted bytes; 1,024 receipt pins peak
at 93,604 versus 93,620. The 16-byte excess is the conservative account-object
allowance. The test-owned payload and receipt-handle vector are allocated before
the measured interval so these probes isolate structural arena/receipt capacity.
They do not time allocations or claim to measure RSS.
