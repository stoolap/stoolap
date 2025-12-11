---
title: Semantic Query Cache
category: Performance
order: 4
---

# Semantic Query Cache

Stoolap implements semantic query caching, an intelligent result caching system that can answer new queries by filtering cached results from previous queries. This goes beyond simple query result caching by understanding the relationship between queries.

## Overview

Traditional query caches require exact query matches. Semantic caching understands when a new query is a subset of a cached query and can filter the cached results instead of re-executing against storage.

## How It Works

### Predicate Subsumption

When Query A caches results, and Query B has a stricter predicate that is a subset of Query A's predicate, Stoolap can answer Query B by filtering the cached results.

### Example

```sql
-- Query 1: First execution scans the table
SELECT * FROM orders WHERE amount > 100;
-- Result: 5,000 rows cached

-- Query 2: Semantic cache hit!
-- The predicate (amount > 500) is stricter than (amount > 100)
SELECT * FROM orders WHERE amount > 500;
-- Filters cached 5,000 rows to return 1,000 rows
-- No storage access needed!
```

## Supported Subsumption Patterns

| Pattern | Cached Query | New Query | Result |
|---------|--------------|-----------|--------|
| **Range Tightening** | `col > 100` | `col > 200` | Cache hit |
| **Range Tightening** | `col < 100` | `col < 50` | Cache hit |
| **AND Strengthening** | `col = 'A'` | `col = 'A' AND x > 5` | Cache hit |
| **IN List Subset** | `id IN (1,2,3,4,5)` | `id IN (2,3)` | Cache hit |
| **BETWEEN Narrowing** | `BETWEEN 30 AND 70` | `BETWEEN 40 AND 60` | Cache hit |
| **Equality to Range** | `col > 0` | `col = 50` | Cache hit |

## Cache Behavior

### Cache Keys

Queries are cached per table with these components:
- Table name
- Selected columns
- Predicate structure

### Automatic Invalidation

The cache is automatically invalidated when data changes:
- INSERT operations
- UPDATE operations
- DELETE operations
- TRUNCATE operations

```sql
-- Query caches results
SELECT * FROM products WHERE price > 100;

-- INSERT invalidates the cache for 'products' table
INSERT INTO products VALUES (101, 'New Product', 150.00);

-- Next query must re-execute (cache miss)
SELECT * FROM products WHERE price > 100;
```

### Cache Limits

| Setting | Default | Description |
|---------|---------|-------------|
| Entries per table | 64 | Maximum cached queries per table |
| TTL | 5 minutes | Time-to-live for cached results |
| Eviction | LRU | Least-recently-used eviction policy |

## When Semantic Caching Helps

Best scenarios for semantic caching:
- **Drill-down queries**: Starting with broad filters, then narrowing
- **Dashboard queries**: Similar queries with different filter values
- **Range queries**: Successive queries on ranges
- **Exploratory analysis**: Progressively filtering data

## When Caching Doesn't Apply

Cache misses occur when:
- Predicate is not a strict subset
- Query has different columns selected
- Table was modified since caching
- Cache entry expired (TTL)
- Cache is full and entry was evicted

## Example Use Case

Consider an analytics dashboard showing orders:

```sql
-- Initial dashboard load: scan full table
SELECT * FROM orders WHERE order_date >= '2024-01-01';
-- Caches 50,000 orders

-- User filters to specific status: cache hit!
SELECT * FROM orders WHERE order_date >= '2024-01-01' AND status = 'pending';
-- Filters cached results, returns 5,000 orders

-- User narrows date range: cache hit!
SELECT * FROM orders WHERE order_date >= '2024-06-01' AND status = 'pending';
-- Filters cached results, returns 1,000 orders

-- User changes to different status: cache hit!
SELECT * FROM orders WHERE order_date >= '2024-01-01' AND status = 'shipped';
-- Filters cached results from first query
```

## Performance Impact

Semantic caching provides significant speedups when:
- Base query is expensive (large table scan)
- Follow-up queries filter the same data
- Data changes infrequently between queries

Typical speedups:
- Cache hit with filtering: **10-100x** faster than storage access
- Particularly effective for large result sets being filtered down

## Best Practices

1. **Order queries from broad to narrow**: Cache broad results first
2. **Use consistent column selections**: Same columns enable cache reuse
3. **Be aware of cache invalidation**: Writes invalidate relevant caches
4. **Consider query patterns**: Design queries to maximize cache hits

## Monitoring

Currently, cache statistics are not exposed through SQL. Cache behavior is internal to the query executor.
