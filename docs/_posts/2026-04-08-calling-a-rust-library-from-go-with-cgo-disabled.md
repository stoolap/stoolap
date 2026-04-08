---
layout: post
title: "Calling a Rust library from Go with CGO_ENABLED=0"
author: Semih Alev
date: 2026-04-08
category: engineering
---

Stoolap is a database engine written in Rust. We ship it as a shared library so non-Rust users can call it. When I wrote the Go driver for it, I wanted the driver to work with `CGO_ENABLED=0`. This post is about what that took, what I borrowed from other projects, and what I had to be careful about.

Up front: this is not an argument that you should avoid cgo in general. Cgo is a reasonable choice for most projects, and what I did is only worth it for specific reasons that I will lay out. If you take nothing else from this post, please take that.

## Why not cgo

The obvious way to call a C-ABI library from Go is cgo. It works, it is well-understood, and most projects should use it.

The reason I did not want it for this driver is a collection of small papercuts that add up for a library that other people will depend on:

- Cross-compilation stops being `GOOS=linux GOARCH=arm64 go build`. You need a C toolchain for the target, or a workaround like Zig as a cross-compiler. Most Go developers have been bitten by this at least once.
- Distroless and scratch container images become more work. You need to pin the glibc version, or statically link, or switch to musl and accept the behavioral differences. None of this is a blocker; it is friction for every user.
- A binary built with `CGO_ENABLED=1` loses some portability. It is no longer something you can just drop onto any Linux box with a compatible kernel.
- There is a real per-call cost on each foreign call, because `runtime.cgocall` transitions the goroutine through `entersyscall` and `exitsyscall` before handing off to the foreign function.

None of this matters if you are building a CLI that calls OpenSSL twice at startup. It matters more if you are building a database driver that a user will call millions of times in a hot path.

## Why not WASM

The other path is to compile the Rust library to WebAssembly and run it with wazero or similar. We actually ship a WASM driver too, at `github.com/stoolap/stoolap-go/wasm`. If you need a database with no shared library dependency and no platform-specific binaries, that is the right choice.

What the WASM driver is not the right choice for is a file-based database under sustained write load, and the reason is background work. Stoolap's engine runs a checkpoint cycle on a background thread (seal hot rows to cold volumes, persist manifests, truncate the WAL), a compaction thread that merges cold volumes, and parallel query execution for large scans and joins. In the WASM module none of that runs in the background, because classical WASM is single-threaded and all operations on the host side are serialised through a mutex. The WASM driver's own documentation is explicit about this: if you want file-based persistence in WASM, you have to call `PRAGMA checkpoint` yourself on a timer, or the WAL grows indefinitely and the hot buffer eats your memory.

WASM threads are a proposal, and wazero has partial support, but Rust's `std::thread` on `wasm32` targets works only in specific toolchain configurations that are not what most people use. Shipping a thread-capable WASM build would mean rewriting the engine's checkpoint and compaction as cooperative async driven from the host, which is a structural change to the engine itself to satisfy one deployment target.

There is also a fetch-heavy throughput angle. WASM still pays for the ahead-of-time-vs-native translation, and for result sets larger than a few thousand rows the marshalling cost between the WASM linear memory and the Go side adds up faster than the native FFI path. A `FetchAll` helper mitigates it, but the native driver still wins on large scans because it skips the linear-memory copy entirely.

None of this is meant to say WASM is bad. It is not. We ship a WASM driver because it is the right answer for a non-trivial set of use cases: sandboxed execution, platforms without native builds, environments where "one binary, no deps" is the hard constraint. The native FFI driver exists for the other set: always-on Go backends that want automatic background maintenance without a C toolchain.

## What the driver actually does

The driver uses three ingredients, none of which are entirely new on their own. The interesting part is the combination.

### 1. Load the library without libc

On Linux we use Go's `//go:cgo_import_dynamic` directive to pull `dlopen`, `dlsym`, and `dlerror` symbols in from `libdl.so.2` at binary-link time, without compiling any C code. The directive tells the Go toolchain to add entries to the binary's dynamic symbol table, and the OS loader resolves them at program start. It looks like this:

```go
//go:cgo_import_dynamic stoolap_dlopen_sym dlopen "libdl.so.2"
//go:cgo_import_dynamic stoolap_dlsym_sym dlsym "libdl.so.2"
//go:cgo_import_dynamic stoolap_dlerror_sym dlerror "libdl.so.2"
```

On macOS the equivalent pulls from `libSystem.B.dylib`. On Windows we call `LoadLibraryW` and `GetProcAddress` through the standard no-cgo Windows syscall path.

This part is ordinary Go toolchain work. The directive is documented as internal but has been stable for years. `ebitengine/purego` uses the same approach, and so do several other no-cgo FFI projects.

### 2. Dispatch via runtime.asmcgocall

Once we have the foreign function pointer, we need to actually call it. The straightforward way would be to use something like `purego.SyscallN`, which packs arguments and executes the foreign call on the caller's goroutine stack.

This driver does it slightly differently. It linknames `runtime.asmcgocall` directly and enters a hand-written Go assembly trampoline through it:

```go
//go:linkname runtime_asmcgocall runtime.asmcgocall
func runtime_asmcgocall(fn, arg unsafe.Pointer) int32
```

`runtime.asmcgocall` is the Go runtime's internal primitive for "switch to g0's stack and call this C function". It is what `runtime.cgocall` uses under the hood, except `runtime.cgocall` wraps it by calling `entersyscall` before and `exitsyscall` after.

By going to `asmcgocall` directly we skip the `entersyscall` and `exitsyscall` pair. We pay for that choice in specific ways that I will describe in the trade-offs section. The reason we do it is that for short, bounded FFI calls, which is what a SQL driver makes, the transition cost is a meaningful fraction of the total call time, and the downsides of skipping it are acceptable for this workload.

The trampoline is a small piece of Go assembly that reads an `abiCallFrame` struct, marshals arguments into the target ABI's registers (System V AMD64 on Linux and macOS, AAPCS64 on ARM64, Microsoft x64 on Windows), calls the function, and writes the return value back into the frame:

```
TEXT abiCallTrampoline(SB), NOSPLIT|NOFRAME, $0
    SUBQ $40, SP
    MOVQ BX, 8(SP)         // save callee-saved BX
    MOVQ R12, 16(SP)       // save callee-saved R12
    MOVQ DI, R12           // save frame ptr in callee-saved R12

    MOVQ 0(R12), R10       // fn
    MOVQ 8(R12), DI        // a1 -> RDI
    MOVQ 16(R12), SI       // a2 -> RSI
    MOVQ 24(R12), DX       // a3 -> RDX
    MOVQ 32(R12), CX       // a4 -> RCX
    MOVQ 40(R12), R8       // a5 -> R8

    XORL AX, AX            // no vector registers used
    CALL R10

    MOVQ AX, 48(R12)       // r1 = return value
    MOVQ 16(SP), R12
    MOVQ 8(SP), BX
    ADDQ $40, SP
    RET
```

Because `asmcgocall` switches to g0's stack before calling, the goroutine's own stack stays in a clean state the garbage collector can scan without any special handling. That is the one GC-related property this approach keeps. More on what it does not keep further down.

### 3. A minimal fake-cgo runtime on Linux

This is the part that surprised me, and it is why I almost called this post "the glibc TLS trap".

When you compile a Go program with `CGO_ENABLED=1`, the Go runtime pulls in a small C shim, `runtime/cgo`, that does the following on startup:

1. Tells the Go runtime "cgo is present" by setting `runtime.iscgo = true`.
2. Initialises a `setg` function the runtime uses to manage the goroutine pointer in thread-local storage.
3. Sets up `pthread_create` hooks so any OS threads the Go scheduler spawns get a correctly configured glibc TLS block before they run user code.

None of this matters for most Go programs. But it matters a lot if you are going to call into a glibc-linked shared library, because glibc uses the `%fs` segment register on x86_64 for its own pthread TLS block. Without cgo init, the Go scheduler uses `%fs` for its own g pointer instead. When you then call `dlopen` or `malloc` or anything else that touches glibc TLS, glibc reads `%fs`, finds Go's g pointer, interprets it as a pthread struct, and crashes. The crash happens before the library is even loaded.

This is a known footgun. The `ebitengine/purego` project solved the same problem with its own internal `fakecgo` package, and I borrowed the idea directly. What the stoolap-go driver ships is a small Go package also called `fakecgo` that provides the symbols the Go runtime normally gets from `runtime/cgo`: `x_cgo_init`, `x_cgo_notify_runtime_init_done`, `_cgo_sys_thread_start`, and friends. It is a Go reimplementation of what a few hundred lines of C code normally do at process startup on Linux.

The shim is Linux-only. On macOS, pthread does not use `%fs`, so the problem does not arise. On Windows, the native loader handles TLS setup itself.

On Linux amd64, the import is one line:

```go
//go:build linux && amd64 && !cgo
package stoolap

import _ "github.com/stoolap/stoolap-go/internal/fakecgo"
```

The `!cgo` build tag means this is a no-op for anyone who does build with cgo. If you set `CGO_ENABLED=1`, Go's normal `runtime/cgo` takes over and the shim is not compiled in.

## The trade-offs

Skipping `entersyscall` is not free, and I do not want to wave it away.

**Scheduler accounting.** In a normal cgo call, `entersyscall` tells the runtime "this M is blocked in foreign code, feel free to run other goroutines on a different M". Without it, the P stays pinned to the calling goroutine for the duration of the FFI call. No other goroutine can be scheduled on that P until the call returns. For microsecond-scale SQL calls this is invisible. For a hypothetical long-running C call it would starve sibling goroutines.

**GC stop-the-world latency.** A goroutine in `_Gsyscall` state counts as a safe point, so STW can proceed without waiting on it. Our goroutines are in `_Grunning` during the FFI call. Go's async preemption delivers a SIGURG to the thread, but async preemption rewinds the goroutine to a Go safe point, and a goroutine that is currently executing C code on g0's stack is not at a Go safe point. So STW waits until the C call returns. For sub-microsecond FFI calls, this is noise. For a multi-millisecond call it would be a real stall.

**GC stack scanning is correct.** This is the property `asmcgocall` preserves. The foreign call runs on g0's stack, not the calling goroutine's stack, so the goroutine stack stays in a state the GC can scan without help. Arguments we pass to the C code live in a stack-local `abiCallFrame` struct of `uintptr` fields, which the GC treats as non-pointer. Because the call is synchronous and the C code does not retain pointers past return, there is no hazard.

**Runtime internals dependency.** `runtime.asmcgocall` is not public API. It is an internal primitive the Go runtime team can change. The driver builds against Go 1.24 and 1.25 today. If a future Go release changes the calling convention of `asmcgocall`, the driver will break in a way that tests should catch immediately. If you ship something like this, you should have CI that runs against Go tip on a cron and warns you early.

**The whole approach is only defensible for short, bounded calls.** The FFI surface of stoolap-go is exec, query, fetch, bind parameter, and next row. None of these are blocking calls in the "open a network connection" sense. If the driver ever grows a truly long-running foreign call, it should route that call through `runtime.cgocall` or wrap it with a manual `entersyscall`/`exitsyscall` pair. There is a comment in the ABI layer that says "do not add long-running calls here" as a guardrail for future contributors.

## Measurements

I am not going to put a ratio in this post, because numbers without methodology invite arguments that do not matter. The benchmark code is in `example/benchmark/main.go` and `bench_test.go` in the driver repo. It uses:

- A 10,000-row fixture with mixed scalar, index, and aggregate queries.
- Warmup of 10 iterations per benchmark to exclude first-call loader and cache effects.
- The same query shapes against `mattn/go-sqlite3` (cgo path) and the stoolap-go driver (no-cgo path) on the same machine.

You can reproduce them yourself with `go test -run NONE -bench . ./...` in the repo.

What I will say about the shape of the numbers, without quoting specifics:

1. The per-call FFI dispatch cost in isolation is in the single-digit nanoseconds on modern x86_64 and ARM64. That matters for the argument about whether skipping `entersyscall` is worth the engineering effort, and not much else.
2. For end-to-end SQL workloads, the FFI layer is not what dominates. What dominates is the database engine itself. Do not expect the no-cgo trick to make your queries suddenly much faster; the win is in the deployment story, not the hot path.
3. For fetch-heavy queries, the much bigger win is a `FetchAll` path that returns all rows in a single FFI crossing instead of one per row. That is basic engineering that any FFI binding should do, regardless of how the dispatch is implemented.

If any of these claims sound fishy to you, run the benchmarks yourself. That is why they are in the repo.

## Why not just use purego

A fair question. `ebitengine/purego` is the canonical no-cgo FFI library for Go, and a large part of the work in this post builds on ideas from it.

The driver does not depend on the purego library for two reasons:

1. We wanted the dispatch path to be hand-written for the exact FFI surface of stoolap, using a small set of fixed trampolines (integer-only, pointer-plus-length-out, float-return). purego's more general `SyscallN` is necessarily more flexible, and more flexible costs something.
2. We wanted to use `runtime.asmcgocall` directly rather than purego's own dispatch. The trampoline runs on g0's stack, which is the property that lets us keep GC stack scanning correct without adding special handling in the driver.

Neither of these is a criticism of purego. If I were writing a generic FFI binding that loaded an arbitrary C library at runtime, I would use purego. I was writing one specific driver for one specific library, and the specialisation paid for itself in code that is easier to reason about.

## When to do this in your own project

For most Go projects calling a C or Rust library, use cgo. It is mature, the documentation is good, and the tooling builds it for you. Do not pick this approach because it sounds clever; pick it only if you have a concrete reason.

The reasons the stoolap-go driver took this path were specific:

1. The driver had to work with `CGO_ENABLED=0` so that users could keep their existing cross-compilation setup, distroless containers, and static binaries without changing anything.
2. We were already going to need a fake-cgo runtime to solve the glibc TLS problem, even if we had used `purego.SyscallN` for dispatch, so the incremental cost of going all the way to `asmcgocall` was small.
3. The FFI surface is short, bounded calls where the scheduler trade-off does not matter.
4. We have the engineering budget to track Go runtime changes and run CI against the Go versions we support.

If any of those four things were not true, the driver would use cgo and this post would not exist.

## Credits

This approach is built on a lot of prior work:

- The `ebitengine/purego` project is the canonical no-cgo FFI library for Go. Its internal `fakecgo` package solved the glibc TLS problem before we did, and reading its code is how I understood the shape of the solution.
- The Go runtime team's `//go:cgo_import_dynamic` directive and `runtime.asmcgocall` primitive are what make any of this possible at all. Neither is documented as public API, and they have stayed stable for long enough to build on.
- The Rust community for keeping the C ABI of `cdylib` predictable and boring.

## Source

The driver is at [github.com/stoolap/stoolap-go](https://github.com/stoolap/stoolap-go), Apache-2.0. The ABI layer is in `abi_*.go` and `abi_*.s`. The Linux TLS shim is in `internal/fakecgo`. The benchmarks are in `example/benchmark/` and `bench_test.go`.

If you find a bug in the scheduler trade-off reasoning, or a case where the trampolines could be tighter, open an issue or send a PR. I would rather hear about a problem than not.
