---
layout: doc
title: Architecture Overview
category: Architecture
order: 1
---

# Stoolap Architecture

This document provides a high-level overview of Stoolap's architecture, including its major components and how they interact.

## System Overview

Stoolap is a high-performance embedded SQL database written in pure Rust. Its architecture prioritizes:

- Memory-first design with optional disk persistence
- Full ACID transactions with MVCC
- Cost-based query optimizer with adaptive execution
- Multiple index types (B-tree, Hash, Bitmap)
- Parallel query execution via Rayon
- Zero unsafe code

## Core Components

Stoolap's architecture consists of the following major components:

### Client Interface

- **Command-Line Interface** - Interactive CLI for database operations
- **Library API** - Direct Rust API for embedded use

### Query Processing Pipeline

1. **Parser** - Converts SQL text into an abstract syntax tree (AST)
   - Lexical analyzer (lexer.rs)
   - Syntax parser (parser.rs)
   - AST builder (ast.rs)

2. **Planner/Optimizer** - Converts AST into an optimized execution plan
   - Cost-based planning with I/O and CPU costs
   - Statistics-based optimization
   - Join order optimization (dynamic programming)
   - Adaptive query execution

3. **Executor** - Executes the plan and produces results
   - Query executor (query.rs)
   - DDL executor (ddl.rs)
   - Semantic query cache (semantic_cache.rs)

### Storage Engine

- **MVCC Engine** - Multi-version concurrency control for transaction isolation
  - Transaction management (transaction.rs)
  - Version store (version_store.rs)
  - Visibility rules

- **Table Management** - Table creation and schema handling
  - Schema validation
  - Table metadata management
  - Column type management

- **Index System** - Multiple index types for different query patterns
  - B-tree indexes (btree_index.rs) - Range queries, sorting
  - Hash indexes (hash_index.rs) - O(1) equality lookups
  - Bitmap indexes (bitmap_index.rs) - Low-cardinality columns
  - Multi-column indexes (multi_column_index.rs)

- **Persistence Layer** - Optional disk storage
  - Write-ahead logging (WAL)
  - Snapshot management
  - Crash recovery

### Function System

- **Function Registry** - Central registry for 101+ SQL functions
  - Scalar functions (scalar/)
  - Aggregate functions (aggregate/)
  - Window functions (window/)

### Parallel Execution

- **Rayon Integration** - Work-stealing parallelism
- **Parallel Filter** - Multi-threaded WHERE evaluation
- **Parallel Join** - Concurrent hash build and probe
- **Parallel Sort** - Multi-threaded ORDER BY

## Request Flow

When a query is executed, it flows through the system as follows:

1. **Query Submission**
   - SQL text is submitted via CLI or library API

2. **Parsing and Validation**
   - SQL is parsed into an AST
   - Syntax and semantic validation is performed
   - Query is prepared for execution

3. **Planning and Optimization**
   - Execution plan is generated with cost estimates
   - Statistics are used to optimize the plan
   - Indexes are selected based on query patterns
   - Join ordering is optimized using dynamic programming

4. **Execution**
   - For read queries:
     - Appropriate isolation level is applied
     - Storage engine provides data with visibility rules
     - Filters and projections are applied (with parallelism for large datasets)
     - Results are processed (joins, aggregations, sorting)
     - Final result set is returned

   - For write queries:
     - Transaction is started if not already active
     - Write operations are applied with MVCC rules
     - Indexes are updated atomically
     - Changes are committed or rolled back

5. **Result Handling**
   - Results are formatted and returned to the client
   - Memory is released
   - Transaction state is updated

## Physical Architecture

### In-Memory Mode

In memory-only mode, Stoolap operates entirely in RAM:

- All data structures reside in memory
- No disk I/O for data access
- Highest performance but no durability
- Use with `Database::open("memory://")`

### Persistent Mode

In persistent mode, Stoolap uses disk storage with memory caching:

- Data is stored on disk with WAL for durability
- Write-ahead logging ensures crash recovery
- Periodic snapshots for faster recovery
- Use with `Database::open("file:///path/to/db")`

## Concurrency Model

Stoolap uses MVCC for concurrency:

- **True MVCC** - Full version chains with history
- **Optimistic Concurrency Control** - Transactions validate at commit time
- **Lock-Free Reads** - Readers never block writers
- **Concurrent Writers** - Multiple transactions can commit simultaneously
- **Two Isolation Levels** - READ COMMITTED (default) and SNAPSHOT

## Implementation Details

The core implementation is organized as follows:

```
src/
├── api/           # Public Database API
├── core/          # Core types (Value, Row, Schema, Error)
├── executor/      # Query execution engine
│   ├── query.rs   # Main query executor
│   ├── planner.rs # Query planner with cost estimation
│   └── ...
├── functions/     # 101+ built-in functions
│   ├── scalar/    # String, math, date, JSON functions
│   ├── aggregate/ # COUNT, SUM, AVG, etc.
│   └── window/    # ROW_NUMBER, RANK, LAG, etc.
├── optimizer/     # Cost-based query optimizer
│   ├── cost.rs    # Cost estimator
│   ├── join.rs    # Join optimization
│   └── ...
├── parser/        # SQL parser (lexer, AST, parser)
├── storage/       # Storage engine
│   └── mvcc/      # MVCC implementation with indexes
└── bin/           # CLI binary
```

## Architectural Principles

Stoolap's architecture is guided by the following principles:

1. **Performance First** - Optimize for speed and memory efficiency
2. **Memory Safety** - Pure Rust with zero unsafe code
3. **Modularity** - Clean component interfaces for extensibility
4. **Simplicity** - Favor simple solutions over complex ones
5. **Data Integrity** - Ensure consistent and correct results