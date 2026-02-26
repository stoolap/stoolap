---
layout: doc
title: Semantic Search
category: Data Types
order: 5
---

# Semantic Search

Stoolap provides built-in semantic search powered by a sentence-transformer model running entirely in Rust. The `EMBED()` function converts text into 384-dimensional vector embeddings. No external APIs, no Python, no Docker. Just SQL.

> **Feature flag required:** Semantic search requires the `semantic` feature flag. Build with `cargo build --features semantic` or add `features = ["semantic"]` to your Cargo.toml dependency.

## The EMBED() Function

`EMBED(text)` converts any text into a 384-dimensional vector embedding using the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) sentence-transformer model.

```sql
-- Generate an embedding from text
SELECT EMBED('How to reset my password');

-- Returns a VECTOR(384) value
```

The model runs entirely in Rust via [candle](https://github.com/huggingface/candle) (Hugging Face's pure Rust ML framework) with zero C/C++ dependencies. On first use, the model (~90MB) is automatically downloaded from Hugging Face Hub and cached at `~/.cache/huggingface/hub/`.

### Supported Input Types

| Input Type | Behavior |
|------------|----------|
| `TEXT` | Converted to embedding |
| `INTEGER` | Converted to string, then embedded |
| `FLOAT` | Converted to string, then embedded |
| `NULL` | Returns NULL |

## Semantic Search Workflow

### 1. Create a Table with a Vector Column

```sql
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT,
    content TEXT NOT NULL,
    embedding VECTOR(384)
);
```

### 2. Insert Documents with Embeddings

Use `EMBED()` to automatically generate embeddings during insertion:

```sql
INSERT INTO documents (id, title, category, content, embedding)
VALUES (
    1,
    'Password Reset Guide',
    'Support',
    'This guide explains how to reset your account password.',
    EMBED('Password Reset Guide. This guide explains how to reset your account password.')
);
```

For best results, concatenate the title and content when generating embeddings. This gives the model more context.

### 3. Build an HNSW Index

Create an HNSW index with cosine distance for efficient similarity search:

```sql
CREATE INDEX idx_doc_embedding ON documents(embedding)
    USING HNSW WITH (metric = 'cosine');
```

Cosine distance is recommended for text embeddings because the model produces normalized vectors where direction (not magnitude) represents meaning.

### 4. Search by Meaning

```sql
SELECT title, category,
       VEC_DISTANCE_COSINE(embedding, EMBED('forgot my login credentials')) AS distance
FROM documents
ORDER BY distance
LIMIT 10;
```

This finds documents semantically similar to the query, even without any keyword overlap. For example, "forgot my login credentials" would match "Password Reset Guide" because the model understands these concepts are related.

## Search Patterns

### Basic Semantic Search

Find documents most similar to a natural language query:

```sql
SELECT title, VEC_DISTANCE_COSINE(embedding, EMBED('machine learning algorithms')) AS dist
FROM documents
ORDER BY dist
LIMIT 10;
```

### Hybrid Search (Semantic + SQL Filters)

Combine semantic similarity with SQL WHERE clauses:

```sql
SELECT title, VEC_DISTANCE_COSINE(embedding, EMBED('data privacy')) AS dist
FROM documents
WHERE category = 'Legal'
ORDER BY dist
LIMIT 5;
```

### Cross-Domain Discovery

Semantic search finds related content across categories. A query like "environmental sustainability" might return results from Science, Business, and Policy categories:

```sql
SELECT title, category,
       VEC_DISTANCE_COSINE(embedding, EMBED('environmental sustainability')) AS dist
FROM documents
ORDER BY dist
LIMIT 10;
```

### Best Match Per Category (Window Functions)

Find the single most relevant document in each category. Use a CTE to compute the query embedding once:

```sql
WITH query AS (
    SELECT EMBED('artificial intelligence') AS vec
)
SELECT title, category, dist
FROM (
    SELECT title, category,
           VEC_DISTANCE_COSINE(embedding, query.vec) AS dist,
           RANK() OVER (PARTITION BY category ORDER BY VEC_DISTANCE_COSINE(embedding, query.vec)) AS rnk
    FROM documents, query
) sub
WHERE rnk = 1
ORDER BY dist;
```

### Question Answering

Use semantic search as a retrieval step for question answering:

```sql
WITH query AS (
    SELECT EMBED('What are the risks of social media for teenagers?') AS vec
)
SELECT title, content, VEC_DISTANCE_COSINE(embedding, query.vec) AS dist
FROM documents, query
ORDER BY dist
LIMIT 3;
```

The top results provide relevant context that can be used with a language model for answer generation (RAG pattern).

### Tip: Reuse Query Embeddings with CTEs

Each `EMBED()` call runs model inference (~30ms). When you reference the same query embedding multiple times in a query (e.g., in both `SELECT` and `ORDER BY`, or in window functions), use a CTE to compute it once:

```sql
-- Bad: EMBED() called twice per row
SELECT title, VEC_DISTANCE_COSINE(embedding, EMBED('query')) AS dist
FROM documents
ORDER BY VEC_DISTANCE_COSINE(embedding, EMBED('query'))
LIMIT 10;

-- Good: EMBED() called once via CTE
WITH query AS (
    SELECT EMBED('query') AS vec
)
SELECT title, VEC_DISTANCE_COSINE(embedding, query.vec) AS dist
FROM documents, query
ORDER BY dist
LIMIT 10;
```

## Model Details

| Property | Value |
|----------|-------|
| Model | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Architecture | BERT (6 layers, 6 attention heads) |
| Output dimensions | 384 |
| Parameters | 22 million |
| Size on disk | ~90 MB |
| Inference | Pure Rust (candle framework) |
| Pooling | Mean pooling with attention mask |
| Normalization | L2-normalized output vectors |

### Performance

| Metric | Value |
|--------|-------|
| Throughput (CPU) | ~25-50 sentences/second |
| First call | Downloads model (~90MB), then loads into memory |
| Subsequent calls | ~20-40ms per embedding |
| Memory | ~200 MB (model weights + runtime) |

Performance varies by CPU and sentence length. Longer sentences require more computation due to the attention mechanism.

### Model Cache

The model is downloaded on first use and cached at:

- **macOS / Linux:** `~/.cache/huggingface/hub/`
- **Windows:** `%USERPROFILE%\.cache\huggingface\hub\`

To pre-download the model, run any query that uses `EMBED()`:

```sql
SELECT EMBED('warmup');
```

## Building with Semantic Search

### Cargo

```toml
[dependencies]
stoolap = { version = "0.3", features = ["semantic"] }
```

### From Source

```bash
# Build with semantic search enabled
cargo build --release --features semantic

# Run the example
cargo run --example semantic_search --release --features semantic
```

### Feature Isolation

The `semantic` feature is fully optional. Without it:
- The `EMBED()` function is not registered
- No candle, tokenizers, or hf-hub dependencies are compiled
- The default build size and compile time are unaffected

## Comparison with External Embedding APIs

| Approach | Stoolap EMBED() | External API (OpenAI, etc.) |
|----------|:---------------:|:---------------------------:|
| Dependencies | None (pure Rust) | HTTP client, API key |
| Latency | ~30ms per embedding | Provider/model/region dependent (typically higher due to network + service overhead) |
| Cost | Free | Per-token pricing |
| Offline | Yes | No |
| Privacy | Data stays local | Data sent to external server |
| Embedding quality | Good for many retrieval workloads | Model dependent; larger hosted models may score higher on semantic benchmarks |
| Dimensions | 384 | Model dependent (for example, 512-3072) |

For applications where data privacy, offline operation, or zero external dependencies matter, built-in `EMBED()` is ideal. For maximum embedding quality, external APIs may produce better results for nuanced queries.

## Complete Example

```sql
-- Create knowledge base
CREATE TABLE kb (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(384)
);

-- Build HNSW index
CREATE INDEX idx_kb_emb ON kb(embedding) USING HNSW WITH (metric = 'cosine');

-- Insert articles with auto-generated embeddings
INSERT INTO kb (title, category, content, embedding)
VALUES (
    'Getting Started with Rust',
    'Programming',
    'Rust is a systems programming language focused on safety and performance.',
    EMBED('Getting Started with Rust. Rust is a systems programming language focused on safety and performance.')
);

INSERT INTO kb (title, category, content, embedding)
VALUES (
    'Python for Data Science',
    'Programming',
    'Python is widely used in data science for its rich ecosystem of libraries like NumPy and Pandas.',
    EMBED('Python for Data Science. Python is widely used in data science for its rich ecosystem of libraries like NumPy and Pandas.')
);

INSERT INTO kb (title, category, content, embedding)
VALUES (
    'Database Indexing Strategies',
    'Databases',
    'Proper indexing is critical for query performance. B-tree indexes support range queries while hash indexes excel at equality lookups.',
    EMBED('Database Indexing Strategies. Proper indexing is critical for query performance. B-tree indexes support range queries while hash indexes excel at equality lookups.')
);

-- Semantic search (CTE computes embedding once)
WITH query AS (
    SELECT EMBED('How do I write fast code?') AS vec
)
SELECT title, category,
       VEC_DISTANCE_COSINE(embedding, query.vec) AS dist
FROM kb, query
ORDER BY dist
LIMIT 5;
-- Expected: "Getting Started with Rust" ranks high (performance focus)

-- Hybrid search with filter
WITH query AS (
    SELECT EMBED('data analysis tools') AS vec
)
SELECT title,
       VEC_DISTANCE_COSINE(embedding, query.vec) AS dist
FROM kb, query
WHERE category = 'Programming'
ORDER BY dist
LIMIT 5;
-- Expected: "Python for Data Science" ranks highest
```

## See Also

- [Vector Search]({% link _docs/data-types/vector-search.md %}): VECTOR data type, distance functions, and HNSW index details
- [Scalar Functions]({% link _docs/functions/scalar-functions.md %}): Complete function reference including EMBED()
