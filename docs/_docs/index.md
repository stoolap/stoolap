---
title: Documentation
layout: doc
---

# Stoolap Documentation

Welcome to the Stoolap Documentation! This is your comprehensive guide to using and understanding Stoolap, a high-performance HTAP database written in pure Go.

## What is Stoolap?

Stoolap is a Hybrid Transactional/Analytical Processing (HTAP) database engine that combines transactional (OLTP) and analytical (OLAP) capabilities in a single system. Built entirely in Go with zero external dependencies, Stoolap features a row-based version store for efficient OLTP operations and columnar indexing for optimized analytical queries, providing a unified solution for diverse workloads.

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
* [Aggregate Functions](functions/aggregate-functions) - Functions for data summarization
* [Scalar Functions](functions/scalar-functions) - Built-in scalar functions
* [Window Functions](sql-features/window-functions) - Row-based analytical functions
* [Temporal Queries (AS OF)](sql-features/temporal-queries) - Time travel queries with AS OF syntax
* [Subqueries](sql-features/subqueries) - Using nested queries in SQL statements
* [Common Table Expressions (CTEs)](sql-features/common-table-expressions) - WITH clause for readable complex queries
* [CAST Operations](sql-features/cast-operations) - Type conversion capabilities
* [NULL Handling](sql-features/null-handling) - Working with NULL values
* [JOIN Operations](sql-features/join-operations) - Combining data from multiple tables
* [DISTINCT Operations](sql-features/distinct-operations) - Working with unique values
* [Parameter Binding](sql-features/parameter-binding) - Using parameters in SQL

### Architecture
* [Storage Engine](architecture/storage-engine) - How data is stored and retrieved
* [Hybrid Storage with Columnar Indexing](architecture/hybrid-storage) - Row-based storage with columnar indexing
* [MVCC Implementation](architecture/mvcc-implementation) - Multi-Version Concurrency Control details
* [Transaction Isolation](architecture/transaction-isolation) - Transaction isolation levels
* [Indexing](architecture/indexing-in-stoolap) - How indexes work and when to use them
* [Unique Indexes](architecture/unique-indexes) - Enforcing data uniqueness
* [Expression Pushdown](architecture/expression-pushdown) - Filtering optimization

### Advanced Features
* [Vectorized Execution](performance/vectorized-execution) - SIMD-accelerated query processing
* [JSON Support](data-types/json-support) - Working with JSON data
* [PRAGMA Commands](sql-commands/pragma-commands) - Database configuration options
* [Date/Time Handling](data-types/date-and-time) - Working with temporal data
* [ON DUPLICATE KEY UPDATE](sql-features/on-duplicate-key-update) - Handling duplicate key conflicts

## Need Help?

If you can't find what you're looking for in the documentation, you can:
* [Open an issue](https://github.com/stoolap/stoolap/issues) on GitHub
* [Join the discussions](https://github.com/stoolap/stoolap/discussions) to ask questions

---

This documentation is under active development. Contributions are welcome!