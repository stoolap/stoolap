---
layout: post
title: "Vector and Semantic Search in SQL: A Practical Guide"
author: Murat Ucok
date: 2026-02-27
category: tutorials
---

I keep seeing teams bolt on a second database just to do semantic search. I get why, but honestly, for a lot of cases, you can stay in SQL and keep things simple.

This is the setup I use in Stoolap when I want vector search without extra infra:

1. Store embeddings in a `VECTOR(N)` column
2. Query nearest neighbors with distance functions
3. Add an `HNSW` index once data grows
4. Use `EMBED()` when I want built-in embeddings
5. Mix semantic ranking with normal SQL filters

## 1. Create a Table for Vectors

Nothing fancy here, just add a vector column:

```sql
CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(384)
);
```

`VECTOR(384)` matches Stoolap's built-in `EMBED()` output, so it is a safe default.

## 2. Insert Data (External Embeddings or Built In)

If you already have embeddings from another model, insert vector literals directly:

```sql
INSERT INTO documents (title, category, content, embedding)
VALUES (
    'Indexing Strategies for OLTP',
    'Database',
    'A practical comparison of hash, B-tree, and bitmap indexes.',
    '[0.12, -0.34, 0.09, ...]'
);
```

If semantic support is enabled, I usually generate embeddings in SQL:

```sql
INSERT INTO documents (title, category, content, embedding)
VALUES (
    'Password Reset Guide',
    'Support',
    'How to recover your account when you forget your password.',
    EMBED('Password Reset Guide. How to recover your account when you forget your password.')
);
```

## 3. Start with Brute Force k-NN

This query shape is the core of the whole thing:

```sql
SELECT
    id,
    title,
    VEC_DISTANCE_COSINE(embedding, EMBED('forgot my login credentials')) AS dist
FROM documents
ORDER BY dist
LIMIT 10;
```

For text embeddings, cosine distance is usually the right metric. Lower distance means better match.

You can also use:

- `VEC_DISTANCE_L2(a, b)` for Euclidean distance
- `VEC_DISTANCE_IP(a, b)` for inner-product search
- `<=>` as shorthand for L2 distance

## 4. Add an HNSW Index When Data Grows

When the table gets bigger, add an HNSW index so you are not scanning every row:

```sql
CREATE INDEX idx_documents_embedding
ON documents(embedding)
USING HNSW
WITH (m = 32, ef_construction = 300, ef_search = 128, metric = 'cosine');
```

After that, keep the same query shape (`ORDER BY distance LIMIT k`). Stoolap will use the HNSW index when metric and query match.

## 5. Avoid Recomputing the Query Embedding

If you use `EMBED()` in multiple places in one query, compute it once with a CTE:

```sql
WITH query AS (
    SELECT EMBED('how do I secure API keys') AS vec
)
SELECT
    id,
    title,
    category,
    VEC_DISTANCE_COSINE(embedding, query.vec) AS dist
FROM documents, query
ORDER BY dist
LIMIT 5;
```

That avoids repeated model inference. It matters more than people think.

## 6. Hybrid Search: Semantic Plus Structured Filters

This is my favorite part. Hybrid retrieval is just SQL:

```sql
WITH query AS (
    SELECT EMBED('database backup best practices') AS vec
)
SELECT
    id,
    title,
    category,
    VEC_DISTANCE_COSINE(embedding, query.vec) AS dist
FROM documents, query
WHERE category = 'Database'
ORDER BY dist
LIMIT 5;
```

You get semantic relevance plus exact filtering in a single plan. Easy to reason about, easy to debug.

## 7. Feature Flag for `EMBED()`

`EMBED()` requires semantic support at build time:

```bash
cargo build --features semantic
```

Or in `Cargo.toml`:

```toml
[dependencies]
stoolap = { version = "0.3", features = ["semantic"] }
```

Without this feature, vector search still works with pre-computed embeddings. Only `EMBED()` is unavailable.

## Practical Checklist

- Match `VECTOR(N)` to your embedding dimension exactly
- Use cosine distance for most sentence-transformer text embeddings
- Create HNSW indexes once your data grows beyond small toy sets
- Keep query shape as `ORDER BY distance LIMIT k`
- Use CTEs to compute query embeddings once per query

## Performance

Wondering how fast HNSW search actually is through the full SQL path? On the [Fashion-MNIST benchmark](/docs/performance/ann-benchmarks/) (60,000 vectors, 784 dimensions, single-core), Stoolap reaches over 10,000 queries per second at 95% recall and over 4,000 QPS at 99.9% recall. In practical profiles p95 latency stays sub-millisecond, and it rises around 1 to 2 ms only at the exact-recall edge.

## Closing

Vector and semantic search does not always need a separate service. In many apps, SQL plus vectors is enough.

If you want to try it end to end, check Stoolap's `semantic_search` and `vector_search_bench` examples in the repo.
