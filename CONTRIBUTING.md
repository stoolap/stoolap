# Contributing to Stoolap

Thank you for your interest in contributing to Stoolap! For the full contributing guide, see our [documentation](https://stoolap.github.io/stoolap/docs/development/contributing/).

## Quick Start

```bash
# Fork and clone
git clone https://github.com/YOUR-USERNAME/stoolap.git
cd stoolap

# Build
cargo build

# Run tests
cargo nextest run

# Lint (must pass with zero warnings)
cargo clippy --all-targets --all-features -- -D warnings

# Format
cargo fmt
```

## Development Rules

- **No TODOs**: Never write `// TODO`, `unimplemented!()`, or `todo!()`
- **Performance first**: Minimize allocations, avoid unnecessary `.clone()` and `.format()`
- **Test everything**: `cargo nextest run` must pass before submitting
- **Zero warnings**: `cargo clippy --all-targets --all-features -- -D warnings`
- **Format code**: Always run `cargo fmt`

## Testing

```bash
# Run all tests (debug mode, never use --release)
cargo nextest run

# Run a specific test file (faster, compiles only that target)
cargo nextest run --test my_feature_test
```

**Important**: Use `cargo nextest run --test <name>` instead of keyword filtering. The `--test` flag compiles only the specified test binary, which is significantly faster.

## Pull Request Process

1. Ensure all tests pass
2. Ensure clippy passes with zero warnings
3. Ensure code is formatted (`cargo fmt`)
4. Write clear commit messages
5. Open a pull request against `main`

## License

Stoolap is licensed under the [Apache License, Version 2.0](LICENSE). By contributing, you agree that your contributions will be licensed under the same license.

### License Headers

All `.rs` source files must include the Apache 2.0 header:

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
