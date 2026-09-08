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
