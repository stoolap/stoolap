---
layout: doc
title: Indexing
category: Architecture
order: 3
---

# Indexing in Stoolap

This document explains Stoolap's indexing system, including the types of indexes available, when to use each type, and best practices for index management.

## Index Types

Stoolap supports five primary index types, each optimized for different query patterns:

### 1. B-tree Indexes

B-tree indexes are the default for numeric and timestamp columns:

- **Design**: Balanced tree structure with sorted values
- **Strengths**: Range queries, equality lookups, sorting, prefix matching
- **Default For**: `INTEGER`, `FLOAT`, `TIMESTAMP` columns
- **Use Cases**: Price ranges, date ranges, numeric comparisons

```sql
-- Auto-selected for INTEGER column
CREATE INDEX idx_price ON products(price);

-- Explicitly specify B-tree
CREATE INDEX idx_date ON orders(order_date) USING BTREE;
```

**Supported Operations:**
- Equality: `WHERE price = 100`
- Range: `WHERE price > 100`, `WHERE price BETWEEN 50 AND 200`
- IN clause: `WHERE id IN (1, 2, 3)`
- Sorting: `ORDER BY price` (can use index for sorted access)

### 2. Hash Indexes

Hash indexes provide O(1) equality lookups:

- **Design**: Hash table mapping values to row IDs
- **Strengths**: Fast equality lookups, O(1) average case
- **Default For**: `TEXT`, `JSON` columns
- **Use Cases**: Email lookups, username searches, exact string matches

```sql
-- Auto-selected for TEXT column
CREATE INDEX idx_email ON users(email);

-- Explicitly specify Hash
CREATE INDEX idx_status ON orders(status) USING HASH;
```

**Supported Operations:**
- Equality: `WHERE email = 'alice@example.com'`
- IN clause: `WHERE status IN ('pending', 'shipped')`

**Not Supported:**
- Range queries: `WHERE name > 'A'` (will not use hash index)
- Sorting: Cannot provide sorted access

### 3. Bitmap Indexes

Bitmap indexes are optimized for low-cardinality columns:

- **Design**: Bitmap per unique value using RoaringTreemap
- **Strengths**: Fast boolean operations, low memory for low cardinality
- **Default For**: `BOOLEAN` columns
- **Use Cases**: Status flags, boolean fields, enum-like columns

```sql
-- Auto-selected for BOOLEAN column
CREATE INDEX idx_active ON users(active);

-- Explicitly specify Bitmap
CREATE INDEX idx_verified ON users(verified) USING BITMAP;
```

**Supported Operations:**
- Equality: `WHERE active = true`
- Fast AND/OR combinations with other bitmap indexes

**Not Supported:**
- Range queries
- IN clause with many values

### 4. HNSW Indexes

HNSW (Hierarchical Navigable Small World) indexes provide approximate nearest neighbor search for vector data:

- **Design**: Multi-layer navigable small world graph with skip-list structure
- **Strengths**: O(log N) approximate nearest neighbor search with high recall
- **Default For**: `VECTOR` columns
- **Use Cases**: Similarity search, semantic search, recommendation systems

```sql
-- Create HNSW index with default parameters
CREATE INDEX idx_emb ON embeddings(embedding) USING HNSW;

-- Create HNSW index with custom parameters
CREATE INDEX idx_emb ON embeddings(embedding) USING HNSW
WITH (m = 32, ef_construction = 400, ef_search = 128, metric = 'cosine');
```

**Supported Operations:**
- k-nearest neighbor search: `ORDER BY VEC_DISTANCE_*(col, query) LIMIT k`
- Multiple distance metrics: L2, cosine, inner product

**HNSW Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `m` | Max connections per node | 16 |
| `ef_construction` | Search width during build | 200 |
| `ef_search` | Search width during queries | 64 |
| `metric` | Distance metric (`l2`, `cosine`, `ip`) | `l2` |

See [Vector Search]({% link _docs/data-types/vector-search.md %}) for detailed documentation.

## Automatic Index Type Selection

When you create an index without specifying a type, Stoolap automatically selects the optimal type based on the column's data type:

| Data Type | Default Index | Reason |
|-----------|---------------|--------|
| `INTEGER` | B-tree | Range queries common on numbers |
| `FLOAT` | B-tree | Range queries common on decimals |
| `TIMESTAMP` | B-tree | Date range queries common |
| `TEXT` | Hash | O(1) equality lookups for strings |
| `JSON` | Hash | O(1) equality lookups for JSON |
| `BOOLEAN` | Bitmap | Only two values, perfect for bitmap |
| `VECTOR` | HNSW | Nearest neighbor search |

## The USING Clause

Override the default index type with the `USING` clause:

```sql
-- Force B-tree on a text column (for prefix queries)
CREATE INDEX idx_name_btree ON users(name) USING BTREE;

-- Force Hash on an integer column (pure equality lookups)
CREATE INDEX idx_id_hash ON orders(user_id) USING HASH;

-- Force Bitmap on a low-cardinality text column
CREATE INDEX idx_status_bitmap ON orders(status) USING BITMAP;

-- HNSW for vector similarity search
CREATE INDEX idx_emb ON embeddings(embedding) USING HNSW;

-- HNSW with custom parameters via WITH clause
CREATE INDEX idx_emb ON embeddings(embedding) USING HNSW
WITH (m = 32, ef_construction = 400, metric = 'cosine');
```

## Multi-Column Indexes

Stoolap supports composite indexes on multiple columns:

```sql
-- Create a multi-column index
CREATE INDEX idx_cust_date ON orders(customer_id, order_date);

-- Create a unique multi-column index
CREATE UNIQUE INDEX idx_unique_cust_date ON orders(customer_id, order_date);
```

### Features

- **Hash-Based**: Efficient equality lookups on all indexed columns
- **Lazy Build**: Index is built on first query access for fast table loads
- **Unique Constraints**: Enforces uniqueness across the combination of columns
- **NULL Handling**: Multiple NULL values allowed (SQL standard behavior)
- **Full Persistence**: WAL and snapshot support for durability

### Usage

Multi-column indexes are used when queries filter on all indexed columns:

```sql
-- Uses idx_cust_date efficiently
SELECT * FROM orders WHERE customer_id = 100 AND order_date = '2024-01-15';

-- Partial match - may or may not use multi-column index
SELECT * FROM orders WHERE customer_id = 100;
```

## Index Intersection

When multiple indexes exist on different columns, Stoolap can combine them:

```sql
-- If idx_category (Hash) and idx_price (B-tree) exist:
SELECT * FROM products WHERE category = 'Electronics' AND price > 500;
-- Both indexes used, results intersected
```

The query executor:
1. Looks up row IDs from each applicable index
2. Intersects the results for AND conditions
3. Unions the results for OR conditions

## Creating Indexes

### Basic Syntax

```sql
-- Standard index (type auto-selected)
CREATE INDEX index_name ON table_name(column_name);

-- Explicit index type
CREATE INDEX index_name ON table_name(column_name) USING BTREE;
CREATE INDEX index_name ON table_name(column_name) USING HASH;
CREATE INDEX index_name ON table_name(column_name) USING BITMAP;
CREATE INDEX index_name ON table_name(column_name) USING HNSW;

-- HNSW with parameters
CREATE INDEX index_name ON table_name(column_name) USING HNSW
WITH (m = 32, ef_construction = 400, ef_search = 128, metric = 'cosine');

-- Multi-column index
CREATE INDEX index_name ON table_name(col1, col2, col3);

-- Unique index
CREATE UNIQUE INDEX index_name ON table_name(column_name);
```

### Dropping Indexes

```sql
DROP INDEX index_name ON table_name;
```

## Index and MVCC

Stoolap's indexes are integrated with the MVCC system:

- Indexes are updated during transaction commit
- For UPDATE: old values removed, new values added
- For DELETE: values removed from index
- For INSERT: values added to index
- All index updates are transactional

## Persistence

All indexes are fully persisted:

- Index metadata stored in WAL (type, columns, unique flag, HNSW parameters)
- Index data rebuilt from table data on recovery
- HNSW graph structure is serialized to binary files during snapshots for fast recovery
- Snapshots capture index definitions
- Recovery restores all indexes automatically

## Query Optimizer Integration

The cost-based optimizer considers indexes when planning queries:

- Estimates selectivity based on index statistics
- Chooses between index scan and sequential scan
- Considers index type capabilities (range vs equality)
- Uses ANALYZE statistics for better estimates

```sql
-- View query plan including index usage
EXPLAIN SELECT * FROM orders WHERE amount > 100;

-- Collect statistics for optimizer
ANALYZE orders;
```

## Best Practices

### When to Create Indexes

1. **Primary key columns** - Always indexed automatically
2. **Foreign key columns** - Improves join performance
3. **Columns in WHERE clauses** - Speeds up filtering
4. **Columns in JOIN conditions** - Accelerates joins
5. **Columns in ORDER BY** - B-tree can avoid sorting

### Index Selection Guidelines

| Query Pattern | Recommended Index |
|--------------|-------------------|
| `WHERE id = value` | B-tree or Hash |
| `WHERE price > 100` | B-tree |
| `WHERE email = value` | Hash (default for TEXT) |
| `WHERE active = true` | Bitmap (default for BOOLEAN) |
| `WHERE cat = x AND brand = y` | Multi-column |
| `ORDER BY date` | B-tree |
| `ORDER BY VEC_DISTANCE_*(col, q) LIMIT k` | HNSW |

### Common Mistakes

1. **Over-indexing** - Too many indexes slow down writes
2. **Wrong index type** - Using B-tree for pure equality on text
3. **Missing multi-column** - Creating separate indexes instead of composite
4. **Indexing low-selectivity columns** - Limited benefit, wastes space

## Performance Characteristics

| Index Type | Equality | Range | k-NN | Space | Write Cost |
|------------|:--------:|:-----:|:----:|:-----:|:----------:|
| B-tree | O(log n) | O(log n + k) | N/A | Medium | Medium |
| Hash | O(1) avg | N/A | N/A | Medium | Low |
| Bitmap | O(1) | N/A | N/A | Low* | Low |
| HNSW | N/A | N/A | O(log n) | High | High |

*For low cardinality columns

## Implementation Notes

Stoolap's indexes are implemented in:

- `src/storage/index/btree.rs` - B-tree index implementation
- `src/storage/index/hash.rs` - Hash index implementation
- `src/storage/index/bitmap.rs` - Bitmap index implementation
- `src/storage/index/multi_column.rs` - Multi-column index
- `src/storage/index/hnsw.rs` - HNSW vector index implementation
- `src/storage/index/pk.rs` - Primary key index (auto-created, hybrid bitset + overflow design)
- `src/storage/traits/index_trait.rs` - Common index trait
