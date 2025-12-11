---
title: Documentation
layout: doc
---

# Stoolap Documentation

Welcome to the Stoolap Documentation! This is your comprehensive guide to using and understanding Stoolap, a high-performance embedded SQL database written in pure Rust.

## What is Stoolap?

Stoolap is a modern embedded SQL database that provides full ACID transactions with MVCC, a sophisticated cost-based query optimizer, and features that rival established databases like PostgreSQL and DuckDB. Built entirely in Rust with zero unsafe code, Stoolap features:

- **Multiple Index Types**: B-tree, Hash, and Bitmap indexes with automatic type selection
- **Multi-Column Indexes**: Composite indexes for complex query patterns
- **Parallel Query Execution**: Automatic parallelization using Rayon for large datasets
- **Cost-Based Optimizer**: PostgreSQL-style optimizer with adaptive execution and cardinality feedback
- **Semantic Query Caching**: Intelligent result caching with predicate subsumption
- **Disk Persistence**: WAL and snapshots with crash recovery
- **Rich SQL Support**: Window functions, CTEs (including recursive), subqueries, ROLLUP/CUBE, and 101+ built-in functions

## Key Documentation Sections

### Getting Started
* [Installation Guide](getting-started/installation) - How to install and set up Stoolap
* [Quick Start Tutorial](getting-started/quickstart) - A step-by-step guide to your first Stoolap database
* [Connection String Reference](getting-started/connection-strings) - Understanding and using Stoolap connection strings
* [API Reference](getting-started/api-reference) - Complete API documentation for the Stoolap package
* [Subqueries Quick Start](getting-started/subqueries-quickstart) - Learn to use subqueries in SQL statements

### SQL Reference
* [Data Types](data-types/data-types) - All supported data types and their behaviors
* [SQL Commands](sql-commands/sql-commands) - SQL syntax and command reference
* [PRAGMA Commands](sql-commands/pragma-commands) - Database configuration options
* [Aggregate Functions](functions/aggregate-functions) - Functions for data summarization
* [Scalar Functions](functions/scalar-functions) - Built-in scalar functions (101+)
* [Window Functions](sql-features/window-functions) - Row-based analytical functions
* [Temporal Queries (AS OF)](sql-features/temporal-queries) - Time travel queries with AS OF syntax
* [Subqueries](sql-features/subqueries) - Using nested queries in SQL statements
* [Common Table Expressions (CTEs)](sql-features/common-table-expressions) - WITH clause for readable complex queries
* [ROLLUP and CUBE](sql-features/rollup-cube) - Multi-dimensional aggregation
* [EXPLAIN](sql-features/explain) - Query plan analysis
* [Savepoints](sql-features/savepoints) - Transaction savepoints

### SQL Features
* [CAST Operations](sql-features/cast-operations) - Type conversion capabilities
* [NULL Handling](sql-features/null-handling) - Working with NULL values
* [JOIN Operations](sql-features/join-operations) - Combining data from multiple tables
* [DISTINCT Operations](sql-features/distinct-operations) - Working with unique values
* [Parameter Binding](sql-features/parameter-binding) - Using parameters in SQL
* [ON DUPLICATE KEY UPDATE](sql-features/on-duplicate-key-update) - Handling duplicate key conflicts

### Architecture
* [Storage Engine](architecture/storage-engine) - How data is stored and retrieved
* [Storage Architecture](architecture/hybrid-storage) - Row-based storage with multiple index types
* [MVCC Implementation](architecture/mvcc-implementation) - Multi-Version Concurrency Control details
* [Transaction Isolation](architecture/transaction-isolation) - Transaction isolation levels
* [Indexing](architecture/indexing) - How indexes work and when to use them (B-tree, Hash, Bitmap)
* [Unique Indexes](architecture/unique-indexes) - Enforcing data uniqueness
* [Expression Pushdown](architecture/expression-pushdown) - Filtering optimization
* [Persistence](architecture/persistence) - WAL, snapshots, and crash recovery

### Performance
* [Query Optimizer](performance/query-optimizer) - Cost-based optimization and adaptive execution
* [Parallel Execution](performance/parallel-execution) - Multi-threaded query processing
* [Semantic Query Cache](performance/semantic-cache) - Intelligent result caching
* [Query Execution](performance/query-execution) - Execution strategies and optimization

### Advanced Features
* [JSON Support](data-types/json-support) - Working with JSON data
* [Date/Time Handling](data-types/date-and-time) - Working with temporal data

## Need Help?

If you can't find what you're looking for in the documentation, you can:
* [Open an issue](https://github.com/stoolap/stoolap/issues) on GitHub
* [Join the discussions](https://github.com/stoolap/stoolap/discussions) to ask questions

---

This documentation is under active development. Contributions are welcome!
