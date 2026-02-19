---
layout: post
title: "Introducing @stoolap/node: A Native Node.js Driver That's Surprisingly Fast"
author: Semih Alev
date: 2026-02-19
---

I've been working on Stoolap for a while now -- an embedded SQL database written in pure Rust. It started as a Go project, grew into something much bigger, and recently hit a point where I thought: okay, this thing is fast, but how do people actually *use* it outside of Rust?

The answer, for a lot of developers, is Node.js. So I built **[@stoolap/node](https://www.npmjs.com/package/@stoolap/node)** -- a native driver powered by [NAPI-RS](https://napi.rs) that gives you direct access to Stoolap from JavaScript and TypeScript.

No HTTP server in between. No serialization overhead. Just your Node.js process talking directly to the database engine through native bindings.

## Why Not Just Use SQLite?

Look, SQLite is great. I use it myself. It's battle-tested, well-documented, and everywhere. But there are things it doesn't do well -- or doesn't do at all.

Stoolap has MVCC transactions, a cost-based query optimizer, parallel execution, semantic query caching, and temporal queries with `AS OF`. These aren't checkbox features; they actually show up in real workloads.

But the question I kept getting was: **is it actually faster?**

Fair question. So I ran the benchmarks.

## The Benchmark

I wrote a comprehensive benchmark suite that runs 53 identical SQL operations against both `@stoolap/node` and `better-sqlite3` (the gold standard for SQLite in Node.js). Same data, same queries, same machine.

The setup: 10,000 rows, a mix of point queries, joins, aggregations, subqueries, and analytical operations. Everything runs in-memory to keep it fair.

Here's the summary:

```
Stoolap wins:  47 / 53 tests
SQLite wins:    6 / 53 tests
```

I wasn't expecting that ratio, honestly. Let me break down where the biggest gaps are.

## Where Stoolap Pulls Ahead

The differences aren't small. Some of these numbers surprised me:

| Operation | Stoolap | SQLite | How Much Faster |
|-----------|---------|--------|-----------------|
| COUNT DISTINCT | 0.003 ms | 0.41 ms | **138x** |
| DELETE (complex WHERE) | 0.02 ms | 2.44 ms | **122x** |
| Compare with subquery | 0.04 ms | 2.56 ms | **64x** |
| NOT EXISTS subquery | 0.17 ms | 9.70 ms | **57x** |
| Aggregation (GROUP BY) | 0.32 ms | 7.68 ms | **24x** |
| Scalar subquery | 0.08 ms | 1.68 ms | **21x** |
| DISTINCT + ORDER BY | 0.04 ms | 0.56 ms | **14x** |
| NOT IN subquery | 0.61 ms | 8.02 ms | **13x** |
| Window PARTITION BY | 0.06 ms | 0.43 ms | **7x** |
| IN subquery | 0.69 ms | 4.67 ms | **7x** |

The `COUNT DISTINCT` result at 138x faster is probably the most dramatic. Stoolap maintains internal data structures that make distinct counting nearly free, while SQLite has to scan and deduplicate every time.

The subquery performance (EXISTS, NOT EXISTS, IN, NOT IN) comes from Stoolap's semi-join optimization -- it builds a HashSet from the subquery result and probes it, rather than running correlated subqueries row by row.

## Where SQLite Still Wins

Let's be honest about where SQLite is faster:

| Operation | SQLite | Stoolap | SQLite's Edge |
|-----------|--------|---------|---------------|
| SELECT by ID | 0.001 ms | 0.002 ms | 1.57x |
| UPDATE by ID | 0.003 ms | 0.004 ms | 1.39x |
| Batch INSERT (100 rows) | 0.39 ms | 0.53 ms | 1.35x |
| INSERT single row | 0.008 ms | 0.009 ms | 1.13x |
| INNER JOIN | 0.10 ms | 0.11 ms | 1.13x |
| Self JOIN | 0.11 ms | 0.11 ms | 1.02x |

These are all small margins -- mostly in the 1.0x to 1.6x range. SQLite's single-row operations benefit from decades of optimization on that specific path. The B-tree page cache is incredibly well-tuned for point lookups.

But notice the pattern: SQLite's wins are on simple, single-row operations where both databases are already sub-millisecond. Stoolap's wins are on the analytical and complex queries where the difference is 10x to 100x+.

## What Makes It Fast

Three things, mainly:

**MVCC without locks.** Stoolap uses multi-version concurrency control, which means readers never block writers. In the Node.js driver, this means your async queries don't stall behind pending writes.

**Cost-based optimizer.** Instead of always doing a sequential scan or always using an index, Stoolap estimates the cost of different execution strategies and picks the cheapest one. For queries with multiple conditions or joins, this makes a huge difference.

**Parallel execution.** Queries over large datasets automatically parallelize using Rayon's work-stealing scheduler. Filters, hash joins, sorts, and distinct operations all scale across cores.

## Using the Driver

The API should feel familiar if you've used `better-sqlite3` or any other embedded database driver:

```js
import { Database } from '@stoolap/node';

const db = await Database.open(':memory:');

await db.exec(`
  CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    price FLOAT,
    category TEXT
  )
`);

// Positional parameters
await db.execute(
  'INSERT INTO products VALUES ($1, $2, $3, $4)',
  [1, 'Laptop', 999.99, 'Electronics']
);

// Named parameters
await db.execute(
  'INSERT INTO products VALUES (:id, :name, :price, :cat)',
  { id: 2, name: 'Book', price: 12.99, cat: 'Media' }
);

// Query returns plain objects
const products = await db.query(
  'SELECT * FROM products WHERE price > $1',
  [10]
);
// [{ id: 1, name: 'Laptop', ... }, { id: 2, name: 'Book', ... }]
```

Both async and sync APIs are available. Async runs on the libuv thread pool so it won't block your event loop. Sync is slightly faster for simple operations if you're in a context where blocking is fine (scripts, CLI tools, tests).

For hot paths, prepared statements skip parsing entirely:

```js
const lookup = db.prepare('SELECT * FROM products WHERE id = $1');

// Reuse without re-parsing
const p1 = lookup.queryOneSync([1]);
const p2 = lookup.queryOneSync([2]);
```

Transactions work the way you'd expect:

```js
const tx = await db.begin();
try {
  await tx.execute('INSERT INTO products VALUES ($1, $2, $3, $4)', [3, 'Phone', 699, 'Electronics']);
  await tx.execute('UPDATE products SET price = $1 WHERE id = $2', [899, 1]);
  await tx.commit();
} catch (e) {
  await tx.rollback();
  throw e;
}
```

## File-Based Persistence

In-memory is great for benchmarks, but real applications need persistence. Stoolap uses WAL (Write-Ahead Logging) with configurable durability:

```js
// Maximum durability -- fsync on every write
const db = await Database.open('./mydata?sync=full');

// Balanced (default) -- fsync on commit batches
const db = await Database.open('./mydata');

// Maximum throughput -- no fsync
const db = await Database.open('./mydata?sync=none');
```

Data survives process restarts. Snapshots run periodically in the background so WAL doesn't grow forever.

## Getting Started

```bash
npm install @stoolap/node
```

Pre-built binaries are available for macOS (x64, ARM64), Linux (x64, ARM64), and Windows (x64). No Rust toolchain required.

If you want to build from source:

```bash
git clone https://github.com/stoolap/stoolap-node.git
cd stoolap-node
npm install && npm run build
npm test
```

The full API documentation is in the [driver docs](/docs/drivers/nodejs/).

## What's Next

The Node.js driver is at v0.3.1 right now. It covers the full Stoolap API -- databases, transactions, prepared statements, batch operations, and all the query methods.

I'm planning to add connection pooling helpers and streaming query support in upcoming releases. If you run into issues or have feature requests, [open an issue on GitHub](https://github.com/stoolap/stoolap-node/issues).

Give it a try and let me know what you think.
