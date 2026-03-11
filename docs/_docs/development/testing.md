---
layout: doc
title: Testing
category: Development
order: 1
---

# Testing

Stoolap has a comprehensive test suite covering all major subsystems, from SQL parsing to crash recovery. This page documents the test infrastructure, how to run tests, and the CI/CD pipeline.

## Test Suite Overview

| Metric | Value |
|--------|-------|
| **Rust Test Files** | 160 |
| **SQLLogicTest Files** | 30 |
| **Total Test Cases** | ~4,100 |
| **Benchmark Suites** | 6 |
| **CI Platforms** | Linux, macOS, Windows |

## Running Tests

### Quick Local Testing

```bash
# Run all tests (debug mode)
cargo nextest run

# Run a specific test file (faster, compiles only that target)
cargo nextest run --test subquery_advanced_test

# Run library unit tests only
cargo nextest run --lib

# Lint check
cargo clippy --all-targets --all-features -- -D warnings

# Format check
cargo fmt --all -- --check
```

**Important**: Always use `cargo nextest run --test <name>` instead of keyword filtering (`cargo nextest run <keyword>`). The `--test` flag compiles only the specified test binary, which is significantly faster.

**Never** use `cargo test --release` due to disk space constraints. Always test in debug mode.

### Feature-Gated Tests

Some tests require feature flags to run:

```bash
# Stress tests (crash soak, metamorphic, concurrency)
cargo nextest run --features stress-tests --test crash_soak_test
cargo nextest run --features stress-tests --test metamorphic_test
cargo nextest run --features stress-tests --test concurrency_history_test

# Differential oracle (compares against SQLite)
cargo nextest run --features sqlite --test differential_oracle_test

# I/O fault injection (must run single-threaded)
cargo nextest run --features test-failpoints --test failpoint_io_test -- --test-threads=1

# C FFI layer tests
cargo nextest run --features ffi --test ffi_test
```

### Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run a specific benchmark
cargo bench --bench select_by_id
```

Available benchmarks: `select_by_id`, `select_complex`, `update_by_id`, `update_complex`, `delete_by_id`, `delete_complex`.

A C FFI benchmark is also available for comparing overhead:

```bash
cargo build --release --features ffi
cc -O2 -o benchmark_ffi examples/benchmark_ffi.c -I include -L target/release -lstoolap
DYLD_LIBRARY_PATH=target/release ./benchmark_ffi   # macOS
LD_LIBRARY_PATH=target/release ./benchmark_ffi      # Linux
```

## Test Categories

### Regression Tests (10 files)

Files: `bugs_regression_test.rs` through `bugs9_regression_test.rs`, `bug_aggregate_in_subquery_test.rs`, `bug_unique_index_test.rs`

Catch and prevent regression of previously fixed bugs. Each bug fix should include a corresponding regression test.

### Persistence and Durability (8 files)

Files: `durability_test.rs` (largest at ~9,170 lines), `persistence_advanced_test.rs`, `persistence_comprehensive_test.rs`, `persistence_test.rs`, `persistence_debug_test.rs`, `persistence_trace_test.rs`, `persistence_wal_dump_test.rs`, `persistence_wal_dump2_test.rs`

Verify WAL, snapshots, crash recovery, and ACID compliance.

### Snapshot and Recovery (4 files)

Files: `snapshot_system_test.rs`, `snapshot_recovery_test.rs`, `wal_visibility_test.rs`, `wal_path_test.rs`

Snapshot creation, loading, recovery after crashes.

### MVCC and Transactions (5 files)

Files: `transaction_test.rs`, `mvcc_isolation_sql_test.rs`, `dirty_read_test.rs`, `isolation_level_test.rs`, `concurrency_history_test.rs`

Snapshot isolation, transaction visibility, and consistency.

### Index Tests (8 files)

Files: `index_operations_test.rs`, `index_optimizer_test.rs`, `index_optimizer_coverage_test.rs`, `btree_index_sql_test.rs`, `hash_index_sql_test.rs`, `bitmap_index_sql_test.rs`, `unique_index_test.rs`, `duplicate_index_test.rs`

B-tree, Hash, Bitmap indexes, index selection, and multi-column indexes.

### Join Tests (4 files)

Files: `join_simple_test.rs`, `join_comprehensive_test.rs`, `join_optimizer_test.rs`, `hash_join_test.rs`

Hash Join, Merge Join, Nested Loop, Index Nested Loop, and join order optimization.

### Subquery Tests (7 files)

Files: `correlated_subquery_test.rs`, `subquery_coverage_test.rs`, `subquery_advanced_test.rs`, `scalar_subquery_test.rs`, `exists_subquery_test.rs`, `subquery_update_test.rs`, `subquery_delete_test.rs`

Scalar subqueries, EXISTS, IN, correlated references.

### Window Function Tests (3 files)

Files: `window_function_test.rs`, `window_advanced_test.rs`, `window_coverage_test.rs`

ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, NTILE, and more.

### CTE Tests (9 files)

Files: `cte_test.rs`, `cte_advanced_test.rs`, `cte_coverage_test.rs`, `cte_exists_totals_test.rs`, `cte_filtering_test.rs`, `cte_simple_alias_test.rs`, `cte_expression_alias_test.rs`, `cte_subquery_test.rs`, `recursive_cte_test.rs`

WITH clauses, recursive CTEs, and expression handling.

### Aggregation Tests (7 files)

Files: `aggregate_functions_test.rs`, `aggregation_test.rs`, `aggregation_advanced_test.rs`, `aggregate_order_test.rs`, `count_aggregate_test.rs`, `first_last_aggregate_test.rs`, `count_with_index_test.rs`

GROUP BY, HAVING, ROLLUP, CUBE, GROUPING SETS.

### Function Tests (15+ files)

Scalar functions, date/time functions, JSON functions, vector functions, hash functions, pattern matching, and collation functions.

### Query Optimization Tests (5 files)

Files: `query_executor_test.rs`, `query_optimization_paths_test.rs`, `cardinality_feedback_test.rs`, `adaptive_execution_test.rs`, `expression_pushdown_test.rs`, `filter_pushdown_test.rs`

Cost estimation, join algorithm selection, pushdown optimization, and adaptive execution.

### DML Tests (8 files)

INSERT, UPDATE, DELETE operations including fast-path optimizations, foreign key enforcement, and ON DUPLICATE KEY UPDATE.

## SQLLogicTest Suite

In addition to Rust integration tests, Stoolap uses the [SQLLogicTest](https://www.sqlite.org/sqllogictest/doc/trunk/about.wiki) format with 30 `.slt` files:

```
tests/slt/
  basic/        - SELECT, INSERT, types, expressions, constraints
  aggregate/    - GROUP BY, DISTINCT, ROLLUP/CUBE
  join/         - INNER, OUTER, CROSS, SELF joins
  subquery/     - EXISTS, IN, correlated, derived tables, scalar
  advanced/     - CTEs, recursive CTEs, set operations, views, windows
  functions/    - String, math, date, JSON, pattern matching
  index/        - Index operations
  transaction/  - Basic transaction behavior
```

SQLLogicTest files provide a database-agnostic specification format, making it possible to compare behavior across database engines.

## CI/CD Pipeline

### Primary CI (Every Push)

| Job | Description |
|-----|-------------|
| **Lint** | `cargo fmt --check` + `cargo clippy -D warnings` |
| **Test** | Full test suite on Linux, macOS, Windows |
| **Feature-Gated** | Differential oracle (SQLite), failpoint I/O tests |
| **Coverage** | `cargo llvm-cov` uploaded to Codecov |
| **License** | Verifies Apache 2.0 headers in all `.rs` files |
| **Build** | Cross-platform binaries (Linux x86/ARM64, macOS x86/ARM64, Windows) |

### Nightly CI (Scheduled)

| Job | Duration | Description |
|-----|----------|-------------|
| **Stress Tests** | ~30 min | Crash soak, metamorphic testing, concurrency stress |
| **ThreadSanitizer** | ~20 min | Data race detection on MVCC and parallel execution |
| **AddressSanitizer** | ~20 min | Memory bug detection on persistence and recovery |
| **Mutation Testing** | ~100 min | 96 daily shards covering executor, functions, and MVCC |
| **Miri** | ~100 min | Undefined behavior detection on core types |

Mutation testing and Miri alternate daily (Mon/Wed/Fri/Sun = mutation, Tue/Thu/Sat = Miri) to keep total wall-time bounded.

## Property-Based Testing

Stoolap uses [proptest](https://github.com/proptest-rs/proptest) for metamorphic testing:

- **`metamorphic_test.rs`** tests query equivalences (e.g., a query with `WHERE a > 100` should always return a subset of `WHERE a > 50`)
- Deterministic seed-based database generation
- Feature-gated under `stress-tests`

## Differential Oracle Testing

**`differential_oracle_test.rs`** runs queries against both Stoolap and SQLite, comparing results to detect behavioral divergences. Requires the `sqlite` feature flag.

## Writing New Tests

1. Create a test file in `tests/` following the naming pattern: `<feature>_test.rs`
2. Use `stoolap::Database` for the public API
3. Use `memory://` connection strings for in-memory test databases
4. Each test function should be independent (create its own database)
5. For bug fixes, add a regression test in a `bug_*_test.rs` file

Example:

```rust
use stoolap::Database;

#[test]
fn test_my_feature() {
    let db = Database::open("memory://test_my_feature")
        .expect("Failed to create database");
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)", ())
        .expect("create");
    db.execute("INSERT INTO t VALUES (1, 'hello')", ())
        .expect("insert");

    let result: String = db
        .query_one("SELECT v FROM t WHERE id = 1", ())
        .expect("query");
    assert_eq!(result, "hello");
}
```
