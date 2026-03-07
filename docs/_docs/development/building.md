---
layout: doc
title: Building from Source
category: Development
order: 3
---

# Building from Source

This page covers how to build Stoolap from source, including prerequisites, feature flags, cross-compilation, and WebAssembly builds.

## Prerequisites

- **Rust toolchain**: Latest stable (edition 2021, Rust 1.56+)
- **Git**: For version embedding at build time

Install Rust via [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Quick Build

```bash
# Clone the repository
git clone https://github.com/stoolap/stoolap.git
cd stoolap

# Debug build (fast compile, slower runtime)
cargo build

# Release build (optimized, with LTO)
cargo build --release

# Build and install the CLI
cargo install --path . --features cli
```

## Feature Flags

Stoolap uses Cargo features to enable optional functionality. The default features are `cli` and `parallel`.

### Core Features

| Feature | Default | Description |
|---------|:-------:|-------------|
| `cli` | Yes | Command-line interface with interactive REPL, requires `clap`, `rustyline`, `comfy-table`, `dirs` |
| `parallel` | Yes | Multi-threaded query execution via Rayon (filter, join, sort, distinct) |
| `wasm` | No | WebAssembly build mode. Excludes parallel execution and file persistence |
| `ffi` | No | C FFI layer for language bindings. Produces `libstoolap.{so,dylib,dll}` with `include/stoolap.h` |
| `mimalloc` | No | Use Microsoft's mimalloc allocator instead of the system allocator |

### Testing and Benchmarking Features

| Feature | Description |
|---------|-------------|
| `sqlite` | Enable SQLite comparison (differential oracle tests and benchmarks) |
| `duckdb` | Enable DuckDB comparison benchmarks |
| `stress-tests` | Enable stress tests: crash soak, metamorphic, concurrency |
| `test-failpoints` | Enable I/O fault injection testing |
| `dhat-heap` | Enable heap allocation profiling via dhat |
| `ann-benchmark` | Enable ANN (vector search) benchmarks with dataset downloading |

### Advanced Features

| Feature | Description |
|---------|-------------|
| `semantic` | Semantic search via HuggingFace embeddings. Requires `candle-core`, `candle-nn`, `candle-transformers`, `tokenizers`, `hf-hub` |

### Build Examples

```bash
# Default build (CLI + parallel)
cargo build --release

# Library only (no CLI)
cargo build --release --no-default-features --features parallel

# With mimalloc allocator
cargo build --release --features mimalloc

# With SQLite comparison support
cargo build --release --features sqlite

# C FFI shared library (libstoolap.so / .dylib / .dll)
cargo build --profile release-ffi --features ffi

# Minimal build (no CLI, no parallel)
cargo build --release --no-default-features
```

## Release Profile

The release profile is configured for maximum performance:

```toml
[profile.release]
lto = true           # Full Link-Time Optimization
codegen-units = 1    # Single codegen unit for better optimization
panic = "abort"      # Abort on panic (smaller binaries)
opt-level = 3        # Maximum optimization
debug = true         # Debug symbols for profiling
```

## WebAssembly Build

Stoolap can be compiled to WebAssembly for browser and Node.js usage.

### Prerequisites

```bash
# Install wasm-pack
cargo install wasm-pack

# Or via npm
npm install -g wasm-pack
```

### Building

```bash
# Build for browser
wasm-pack build --target web --no-default-features --features wasm

# Build for Node.js
wasm-pack build --target nodejs --no-default-features --features wasm

# Build for bundlers (webpack, etc.)
wasm-pack build --target bundler --no-default-features --features wasm
```

### WASM-Specific Notes

- The `wasm` feature disables `parallel` and file persistence
- All data is in-memory only (lost on page reload)
- Uses `web-time` for timestamp support instead of `std::time`
- Random number generation uses the browser's crypto API via `getrandom`
- Published as the `stoolap` npm package with TypeScript definitions

See [WebAssembly]({{ '/docs/drivers/wasm/' | relative_url }}) for usage documentation.

## Cross-Compilation

Stoolap CI builds binaries for five targets:

| Target | Binary Name | Notes |
|--------|-------------|-------|
| `x86_64-unknown-linux-gnu` | `stoolap-linux-amd64` | Standard Linux |
| `aarch64-unknown-linux-gnu` | `stoolap-linux-arm64` | Requires `gcc-aarch64-linux-gnu` |
| `x86_64-apple-darwin` | `stoolap-darwin-amd64` | Intel Mac |
| `aarch64-apple-darwin` | `stoolap-darwin-arm64` | Apple Silicon |
| `x86_64-pc-windows-msvc` | `stoolap-windows-amd64.exe` | Windows |

To cross-compile locally:

```bash
# Add the target
rustup target add aarch64-unknown-linux-gnu

# Build for the target
cargo build --release --target aarch64-unknown-linux-gnu
```

## Platform-Specific Dependencies

Stoolap uses conditional compilation for platform-specific file locking:

- **Unix**: `libc` for `flock()`-based file locking
- **Windows**: `windows-sys` for Windows file locking APIs
- **WebAssembly**: No file locking (in-memory only)

## Build Script

The `build.rs` script embeds the git commit hash at compile time as `STOOLAP_GIT_COMMIT`. This can be overridden by setting the `STOOLAP_GIT_COMMIT` environment variable before building.

## Binary Targets

```bash
# CLI binary (requires 'cli' feature)
cargo run --features cli -- --help

# As a library (default in your Cargo.toml)
# [dependencies]
# stoolap = "0.3"
```

## CI Profiles

For faster CI builds, a dedicated profile is available:

```toml
[profile.ci]
inherits = "release"
lto = "thin"          # Faster than full LTO
codegen-units = 16    # Parallel codegen
debug = false         # No debug symbols
```

Use it with: `cargo build --profile ci`
