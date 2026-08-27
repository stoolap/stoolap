# 08. WebAssembly (WASM) Architecture & In-Memory Constraints

## Subsystem Architecture Overview

Stoolap compiles to WebAssembly (`wasm32-unknown-unknown`), enabling in-browser embedded SQL execution via WebAssembly JavaScript bindings:

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Browser JavaScript: const db = new StoolapDatabase(); db.execute(...)     │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────────┐
       │   WASM FFI & JS Glue (src/wasm.rs)                         │
       │   - Exposes synchronous C/JS bridge                        │
       │   - Serializes query results to JSON / packed buffers      │
       └────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────────┐
       │   In-Memory Storage Engine (src/storage/mvcc/engine.rs)    │
       │   - Lock-free MVCC version arena in WASM linear memory     │
       │   - Single-threaded execution loop                         │
       └────────────────────────────────────────────────────────────┘
```

### Key Source References
- [`src/wasm.rs`](file:///home/irshad/stoolap/src/wasm.rs): Defines WebAssembly exported interfaces (`WasmDatabase`, `execute_query`, `fetch_rows`).
- [`src/storage/mvcc/engine.rs`](file:///home/irshad/stoolap/src/storage/mvcc/engine.rs): Instantiates the engine in pure in-memory mode when compiled under `target_arch = "wasm32"`.

---

## Known Limitations Breakdown

| Feature | WASM Status | Native Native (x86/ARM) Status |
|---|---|---|
| **File Persistence** | ❌ Not available (In-memory only; data lost on page reload) | ✅ Full frozen volume + WAL on disk |
| **Background Threads** | ❌ Not available (Single-threaded; no background flusher) | ✅ Multi-threaded background compaction & WAL flush |
| **Garbage Cleanup** | ⚠️ Manual only (requires explicit `VACUUM` queries) | ✅ Automatic periodic background vacuum |
| **WAL & Crash Recovery** | ❌ Disabled (Ephemeral memory lifecycle) | ✅ Full ACID WAL replay on restart |

---

## Architectural Root Causes

### 1. `wasm32-unknown-unknown` Standard Library Constraints
The standard Rust WebAssembly target (`wasm32-unknown-unknown`) does not provide POSIX filesystem APIs (`std::fs::File`) or OS threading primitives (`std::thread::spawn`). 
In Stoolap:
- Native persistence relies on synchronous file system calls ([`src/storage/volume/io.rs`](file:///home/irshad/stoolap/src/storage/volume/io.rs)).
- Native background compaction relies on dedicated worker threads ([`src/storage/volume/manifest.rs`](file:///home/irshad/stoolap/src/storage/volume/manifest.rs)).
When compiled to WASM, all file I/O and background thread spawns are compiled out via conditional compilation (`#[cfg(not(target_arch = "wasm32"))]`).

### 2. Lack of Virtual File System (VFS) Abstraction
Unlike SQLite (which abstracts all I/O behind an OS/VFS interface that can be implemented for browser **Origin Private File System (OPFS)** or **IndexedDB**), Stoolap's storage layer historically targeted direct POSIX/Win32 file descriptors.

---

## Performance & System Impact

- **Memory Ceiling**: In 32-bit WASM, the browser runtime caps total linear memory at 2GB or 4GB. Without disk paging or background garbage collection, intensive OLAP queries or continuous ingestion will exhaust available memory.
- **Application Complexity**: Web applications requiring data persistence between browser sessions must manually serialize database dumps to IndexedDB using custom export functions.

---

## Proposed Engineering Roadmap

### Phase 1: Virtual File System (VFS) Abstraction
- Introduce a storage trait `StorageVfs` abstracting file operations:
  ```rust
  pub trait StorageVfs: Send + Sync {
      fn open(&self, path: &str, opts: OpenOptions) -> Result<Box<dyn VfsFile>>;
      fn delete(&self, path: &str) -> Result<()>;
      fn exists(&self, path: &str) -> Result<bool>;
  }
  ```
- Implement `NativeVfs` for native operating systems and `MemoryVfs` / `OpfsVfs` for WebAssembly.

### Phase 2: Origin Private File System (OPFS) Driver
- Leverage the browser OPFS synchronous access handle (`FileSystemSyncAccessHandle`) inside Web Workers to provide high-speed persistent frozen volumes and WAL in the browser.

### Phase 3: Web Worker Threading & WebAssembly Threads
- Utilize `wasm32-unknown-emscripten` or `wasm-bindgen-rayon` with `SharedArrayBuffer` to enable background compaction workers and parallel query execution in modern browsers.
