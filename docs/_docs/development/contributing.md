---
layout: doc
title: Contributing
category: Development
order: 4
---

# Contributing

Thank you for your interest in contributing to Stoolap! This guide covers the development workflow, coding standards, and how to submit your changes.

## License

Stoolap is licensed under the [Apache License, Version 2.0](https://github.com/stoolap/stoolap/blob/main/LICENSE). By contributing, you agree that your contributions will be licensed under the same license.

The Apache License 2.0 includes an express grant of patent rights from contributors to users. Be aware of this when contributing patented intellectual property.

## Getting Started

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR-USERNAME/stoolap.git
cd stoolap

# Add upstream remote
git remote add upstream https://github.com/stoolap/stoolap.git

# Create a branch for your work
git checkout -b feature-or-bugfix-name
```

### Build and Verify

```bash
# Build
cargo build

# Run tests
cargo nextest run

# Lint (must pass with zero warnings)
cargo clippy --all-targets --all-features -- -D warnings

# Format
cargo fmt
```

See [Building from Source]({{ '/docs/development/building/' | relative_url }}) for prerequisites and feature flags.

## Development Rules

### No TODOs, No Stubs, No Placeholders

- **Never** write `// TODO` comments
- **Never** write `unimplemented!()` or `todo!()`
- Each file must be fully functional before creation

### Performance First

- Minimize memory usage and allocations
- Avoid unnecessary `.clone()` and `.format()` calls
- Use the correct HashMap type for each key type (see [HashMap Guide](#hashmap-selection) below)
- Pre-allocate with `.with_capacity()` when the size is known
- Prefer tuples over `Vec` for fixed-size composite keys (30% faster)

### Quality Over Speed

- No shortcuts or partial implementations
- Handle all edge cases
- Preserve all error handling
- Avoid `unwrap()` in library code, use proper error handling
- Avoid `unsafe` code unless absolutely necessary (and document why)

## Coding Standards

### Formatting and Linting

Every commit must pass:

```bash
# Format check
cargo fmt --all -- --check

# Lint check (zero warnings)
cargo clippy --all-targets --all-features -- -D warnings
```

### License Headers

All `.rs` source files must include the Apache 2.0 license header. CI will reject files without it.

```rust
// Copyright 2025 Stoolap Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
```

### General Conventions

- Follow Rust standard conventions and idioms
- Keep functions small and focused
- Document public APIs with doc comments (`///`)
- Write tests alongside implementation
- Prefer editing existing files over creating new ones

## Testing Requirements

Every change must pass the full test suite:

```bash
# Run all tests (use --test for specific files, much faster)
cargo nextest run

# Run a specific test file
cargo nextest run --test my_feature_test

# Never use --release (disk space constraints)
# Never use keyword filtering (compiles all targets)
```

For bug fixes, add a regression test in a `bug_*_test.rs` file to prevent the issue from recurring.

See [Testing]({{ '/docs/development/testing/' | relative_url }}) for the full testing guide.

## HashMap Selection

Choosing the right hasher is critical for performance. Use this quick reference:

| Key Type | Use This |
|----------|----------|
| `i64` | `I64Map` (custom, fastest for row IDs and transaction IDs) |
| `u64`, `usize` | `FxHashMap` |
| `String`, `SmartString` | `AHashMap` (required for `Borrow<str>` compatibility) |
| `Value` | `ValueMap` / `ValueSet` (AHash for HashDoS resistance) |
| `Vec<Value>` | `AHashMap` (user-controlled keys need HashDoS resistance) |
| `(Value, Value)` | `AHashMap` (tuples are 30% faster than Vec for fixed columns) |

```rust
use crate::common::I64Map;           // i64 keys
use rustc_hash::FxHashMap;           // u64, usize keys
use ahash::AHashMap;                 // String, SmartString keys
use crate::core::{ValueMap, ValueSet}; // Value keys
```

**Caveat**: `I64Map` uses `i64::MIN` as a sentinel value. It cannot be used as a key.

## Project Structure

```
src/
  api/           Public Database API
  core/          Core types (Value, Row, Schema, Error)
  executor/      Query execution engine
    query.rs     Main query executor
    planner.rs   Cost-based query planner
    expression/  Expression VM and compiled evaluators
    operators/   Volcano join operators (hash, merge, nested loop)
    subquery.rs  Subquery execution (EXISTS, IN, scalar)
  functions/     Built-in functions (scalar/, aggregate/, window/)
  optimizer/     Cost-based optimizer (cost, join DP, bloom, feedback)
  parser/        SQL parser (lexer, AST, parser)
  storage/       Storage engine (mvcc/, index/, statistics)
  bin/           CLI binary
```

## Pull Request Process

1. Ensure all tests pass (`cargo nextest run`)
2. Ensure clippy passes with zero warnings
3. Ensure code is formatted (`cargo fmt`)
4. Write clear commit messages that explain the problem and approach
5. Push your branch and open a pull request against `main`
6. Update documentation if your change affects user-facing behavior
7. Maintainers will review and may request changes
8. Once approved, your PR will be merged

## Expression Compilation Pattern

When working on the executor, always pre-compile expressions:

```rust
// Correct: Pre-compile once, reuse for all rows
let filter = RowFilter::new(&where_expr, &columns)?;
for row in rows {
    if filter.matches(&row) { ... }
}

// Wrong: Evaluating expression AST directly per row (slow)
for row in rows {
    if evaluate_expression(&where_expr, &row)? { ... }
}
```

Key components to use:

| Task | Component |
|------|-----------|
| WHERE filtering | `RowFilter` + `FilteredResult` |
| SELECT projection | `ExprMappedResult` or `StreamingProjectionResult` |
| JOINs | `JoinExecutor` (auto-selects algorithm via `QueryPlanner`) |
| Subqueries | `subquery.rs` semi-join optimization |
