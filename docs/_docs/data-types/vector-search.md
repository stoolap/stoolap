---
layout: doc
title: Vector Search
category: Data Types
order: 4
---

# Vector Search

Stoolap supports native vector storage and similarity search, enabling AI/ML use cases such as semantic search, recommendation systems, and retrieval-augmented generation (RAG) directly within the database.

> **Built-in semantic search:** With the `semantic` feature flag, Stoolap provides the `EMBED()` function for automatic text-to-vector conversion using a built-in sentence-transformer model. See [Semantic Search]({% link _docs/data-types/semantic-search.md %}) for details.

## VECTOR Data Type

The `VECTOR(N)` type stores fixed-dimension floating-point vectors, where `N` is the number of dimensions. Vectors are stored as packed little-endian f32 arrays for compact storage and zero-copy distance computation.

```sql
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY,
    content TEXT,
    embedding VECTOR(384)
);
```

### Inserting Vectors

Vectors are inserted as bracket-delimited, comma-separated float values:

```sql
INSERT INTO embeddings (id, content, embedding)
VALUES (1, 'Hello world', '[0.1, 0.2, 0.3, ...]');
```

Dimension count is validated on insert. Providing a vector with the wrong number of dimensions will return an error.

### NULL Handling

VECTOR columns accept NULL values like any other column:

```sql
INSERT INTO embeddings (id, content, embedding) VALUES (2, 'No vector', NULL);
```

## Distance Functions

Stoolap provides three distance metrics for comparing vectors:

| Function | Description | Range | Best For |
|----------|-------------|-------|----------|
| `VEC_DISTANCE_L2(a, b)` | Euclidean distance | 0 to infinity | General-purpose similarity |
| `VEC_DISTANCE_COSINE(a, b)` | Cosine distance (1 - cosine similarity) | 0 to 2 | Text embeddings, normalized vectors |
| `VEC_DISTANCE_IP(a, b)` | Negative inner product distance (-dot product) | varies | Maximum inner product search |

### Examples

```sql
-- Euclidean distance
SELECT VEC_DISTANCE_L2(embedding, '[0.1, 0.2, 0.3]') AS dist
FROM embeddings;

-- Cosine distance
SELECT VEC_DISTANCE_COSINE(embedding, '[0.1, 0.2, 0.3]') AS dist
FROM embeddings;

-- Inner product distance
SELECT VEC_DISTANCE_IP(embedding, '[0.1, 0.2, 0.3]') AS dist
FROM embeddings;
```

### Distance Operator

The `<=>` operator is a shorthand for L2 (Euclidean) distance:

```sql
-- These are equivalent
SELECT embedding <=> '[0.1, 0.2, 0.3]' AS dist FROM embeddings;
SELECT VEC_DISTANCE_L2(embedding, '[0.1, 0.2, 0.3]') AS dist FROM embeddings;
```

For cosine or inner product distance, use the explicit `VEC_DISTANCE_COSINE` or `VEC_DISTANCE_IP` functions.

## k-Nearest Neighbor (k-NN) Search

To find the k most similar vectors, use `ORDER BY distance LIMIT k`:

```sql
-- Find 10 nearest neighbors by Euclidean distance
SELECT id, content, VEC_DISTANCE_L2(embedding, '[0.1, 0.2, ...]') AS dist
FROM embeddings
ORDER BY dist
LIMIT 10;

-- Find 5 nearest neighbors by cosine distance
SELECT id, content, VEC_DISTANCE_COSINE(embedding, '[0.1, 0.2, ...]') AS dist
FROM embeddings
ORDER BY dist
LIMIT 5;
```

The query optimizer detects this `ORDER BY distance LIMIT k` pattern and automatically uses an HNSW index (if available) or a parallel brute-force scan with a min-heap for efficient top-k selection.

## HNSW Indexes

HNSW (Hierarchical Navigable Small World) indexes provide approximate nearest neighbor search in O(log N) time instead of O(N) brute-force scanning.

### Creating an HNSW Index

```sql
-- Basic HNSW index (defaults auto-selected from dimensions: metric=l2)
CREATE INDEX idx_emb ON embeddings(embedding) USING HNSW;

-- With custom parameters
CREATE INDEX idx_emb ON embeddings(embedding) USING HNSW
WITH (m = 32, ef_construction = 400, ef_search = 128, metric = 'cosine');
```

### HNSW Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `m` | Max connections per node. Higher values improve recall but use more memory. Auto-selected from dimensions (16-48). | auto | 4-64 |
| `ef_construction` | Search width during index build. Higher values improve quality but slow construction. Auto-selected from M (128-256). | auto | 50-1000 |
| `ef_search` | Search width during queries. Higher values improve recall but slow queries. Auto-selected from M (128-256). | auto | 10-1000 |
| `metric` | Distance metric: `l2`, `cosine`, or `ip`. Must match the distance function used in queries. | `l2` | - |

### Metric Matching

The optimizer automatically matches your distance function to the HNSW index metric. An HNSW index built with `metric = 'cosine'` is only used for `VEC_DISTANCE_COSINE` queries:

```sql
-- This index is used for cosine queries
CREATE INDEX idx_cosine ON embeddings(embedding) USING HNSW WITH (metric = 'cosine');

-- Uses the HNSW index (metric matches)
SELECT * FROM embeddings
ORDER BY VEC_DISTANCE_COSINE(embedding, '[...]')
LIMIT 10;

-- Falls back to brute-force (metric does not match)
SELECT * FROM embeddings
ORDER BY VEC_DISTANCE_L2(embedding, '[...]')
LIMIT 10;
```

You can create multiple HNSW indexes with different metrics on the same column if you need to query with different distance functions.

### How HNSW Works

HNSW builds a multi-layer graph where:
- The bottom layer contains all vectors
- Upper layers contain progressively fewer vectors (skip-list structure)
- Search starts at the top layer and navigates down, narrowing candidates at each level

In practice, this yields sub-linear average-case search with tunable recall/latency tradeoffs, especially for larger datasets.

## Filtering with WHERE Clauses

You can combine vector search with standard SQL filters:

```sql
-- Vector search with a filter
SELECT id, content, VEC_DISTANCE_L2(embedding, '[0.1, 0.2, ...]') AS dist
FROM embeddings
WHERE content LIKE '%science%'
ORDER BY dist
LIMIT 10;
```

When an HNSW index is available, Stoolap first retrieves candidates from the index, then applies the WHERE filter. If no matching results are found after HNSW filtering, the optimizer falls back to a brute-force scan to ensure correct results.

## Utility Functions

| Function | Description | Example |
|----------|-------------|---------|
| `VEC_DIMS(v)` | Returns the number of dimensions | `SELECT VEC_DIMS(embedding) FROM t` |
| `VEC_NORM(v)` | Returns the L2 norm (magnitude) | `SELECT VEC_NORM(embedding) FROM t` |
| `VEC_TO_TEXT(v)` | Converts a vector to its text representation | `SELECT VEC_TO_TEXT(embedding) FROM t` |

```sql
-- Check dimensions
SELECT VEC_DIMS(embedding) FROM embeddings WHERE id = 1;
-- Returns: 384

-- Compute vector magnitude
SELECT VEC_NORM(embedding) FROM embeddings WHERE id = 1;
-- Returns: 1.0 (for normalized vectors)

-- Display vector as text
SELECT VEC_TO_TEXT(embedding) FROM embeddings WHERE id = 1;
-- Returns: [0.1, 0.2, 0.3, ...]
```

## Persistence

- **Vector data** is fully persisted through WAL and snapshots, like any other data type
- **HNSW index definitions** are persisted in WAL (index metadata includes m, ef_construction, ef_search, and metric)
- **HNSW graph structure** is serialized to binary files during snapshots for fast recovery, avoiding expensive graph rebuilds on restart
- On recovery, if a graph file is found, it is loaded directly; otherwise the graph is rebuilt from table data

## Performance

### Brute-Force Search

For tables without an HNSW index, Stoolap uses parallel brute-force search:
- Vectors are scanned in parallel using Rayon work-stealing
- A fused distance-computation + min-heap selection avoids materializing all distances
- Suitable for small-to-medium datasets (up to ~100K vectors)

### HNSW Index Search

For tables with an HNSW index:
- Sub-linear average-case approximate search with tunable recall
- Graph structure is cached in memory for fast access
- `ef_search` controls the recall/speed tradeoff at query time

### Tips

- Use `VECTOR(N)` with the exact dimension count for your embeddings
- Create an HNSW index for datasets larger than a few thousand rows
- Match the HNSW `metric` parameter to the distance function you use in queries
- Increase `ef_search` if you need higher recall at the cost of latency
- Use `ef_construction` >= 200 for good index quality

## Example: Semantic Search

### With Pre-computed Embeddings

```sql
-- Create table for document embeddings
CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    title TEXT NOT NULL,
    body TEXT,
    embedding VECTOR(768)
);

-- Create HNSW index with cosine metric
CREATE INDEX idx_doc_emb ON documents(embedding) USING HNSW
WITH (m = 32, ef_construction = 300, metric = 'cosine');

-- Insert documents with embeddings (from your ML model)
INSERT INTO documents (title, body, embedding)
VALUES ('Introduction to SQL', 'SQL is a language...', '[0.12, -0.34, ...]');

-- Semantic search: find 5 most relevant documents
SELECT id, title, VEC_DISTANCE_COSINE(embedding, '[0.15, -0.28, ...]') AS relevance
FROM documents
ORDER BY relevance
LIMIT 5;
```

### With Built-in EMBED() Function

With the `semantic` feature flag, you can generate embeddings directly in SQL:

```sql
CREATE TABLE docs (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    content TEXT NOT NULL,
    embedding VECTOR(384)
);

CREATE INDEX idx_emb ON docs(embedding) USING HNSW WITH (metric = 'cosine');

-- Insert with auto-generated embeddings
INSERT INTO docs (content, embedding)
VALUES ('How to reset your password', EMBED('How to reset your password'));

-- Semantic search (no pre-computed vectors needed)
SELECT content, VEC_DISTANCE_COSINE(embedding, EMBED('forgot login credentials')) AS dist
FROM docs
ORDER BY dist
LIMIT 5;
```

See [Semantic Search]({% link _docs/data-types/semantic-search.md %}) for full documentation on the `EMBED()` function.
