---
layout: doc
title: ANN Benchmarks
description: HNSW vector search performance on public ANN-Benchmarks datasets
category: Performance
order: 6
---

# ANN Benchmarks

This page presents Stoolap's HNSW vector search performance on a public dataset from the [ANN-Benchmarks](https://ann-benchmarks.com/) project. We use the same dataset, the same HNSW parameters, and compute exact ground truth independently, measuring the **full SQL query path** (parsing, planning, index lookup, result assembly) on a single CPU core. See [Methodology](#methodology) for details.

## Dataset

We use [Fashion-MNIST](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/), the same data behind [fashion-mnist-784-euclidean](https://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5) on ANN-Benchmarks:

| Property | Value |
|----------|-------|
| Base vectors | 60,000 |
| Dimensions | 784 |
| Distance metric | L2 (Euclidean) |
| Query vectors | 10,000 (full test set) |
| k (neighbors returned) | 10 |
| Ground truth | Exact brute-force KNN computed independently in Rust |

Each benchmark query executes a full SQL path:

```sql
SELECT id, VEC_DISTANCE_L2(embedding, ?) AS dist
FROM vectors ORDER BY dist LIMIT 10
```

## Methodology

### Metrics

- **Recall@k**: Fraction of true k-nearest neighbors found (ID-set intersection). 99% recall means 9.9 out of 10 true neighbors are returned.
- **QPS**: Queries per second, computed as `query_count / best_run_total_time`. Higher is better.
- **p95 / p99 latency**: 95th and 99th percentile query time in milliseconds (from the best run). Lower is better.
- **Speedup**: HNSW throughput divided by brute-force throughput.

### Differences from ann-benchmarks.com

| Aspect | ann-benchmarks.com | Stoolap benchmark |
|--------|-------------------|-------------------|
| What is timed | Raw index query | Full SQL path (parse, plan, index lookup, result assembly) |
| Threading | Single CPU | Single CPU (`RAYON_NUM_THREADS=1`) |
| Runs | Best of 5 | HNSW: best of 5, brute-force: single run |
| Query count | All 10,000 | All 10,000 |
| Recall method | Distance-threshold | ID-set intersection |
| Ground truth | Pre-computed from dataset | Computed independently via exact brute-force L2 |
| Warmup | Implicit (best-of-N) | Explicit 10-query warmup + best-of-N |
| HNSW parameters | Identical | Identical (ef_construction=500, same m and ef_search grid) |

Our QPS numbers include SQL overhead (parsing, planning, result assembly) and are therefore **not directly comparable** to ann-benchmarks.com results, which measure raw index operations only.

## Results

<object type="image/svg+xml" data="{{ '/assets/img/benchmarks/ann-public-fashion-mnist-784-euclidean-2026-02-28.svg' | relative_url }}" style="width:100%;max-width:1280px" aria-label="Recall vs QPS and p95 latency">Recall vs QPS and p95 latency chart</object>

### Scorecard

Best configuration for each recall target (single-core, 10K queries, HNSW best-of-5, brute-force single run):

| Recall target | Config | Measured recall | QPS | p95 | p99 | Speedup |
|---------------|--------|-----------------|-----|-----|-----|---------|
| >= 95.0% | m=8, ef=20 | 95.02% | 10,410 | 0.12 ms | 0.15 ms | 733x |
| >= 99.0% | m=24, ef=10 | 99.31% | 6,700 | 0.19 ms | 0.22 ms | 472x |
| >= 99.5% | m=16, ef=40 | 99.69% | 5,762 | 0.22 ms | 0.25 ms | 406x |
| >= 99.9% | m=36, ef=40 | 99.91% | 4,159 | 0.33 ms | 0.38 ms | 293x |
| >= 99.99% | m=64, ef=120 | 99.99% | 1,962 | 0.77 ms | 0.90 ms | 138x |
| 100% | m=48, ef=600 | 100.00% | 913 | 1.59 ms | 1.84 ms | 64x |

Brute-force baseline: 14.2 QPS, p95 = 70.96 ms (single-core, full SQL path).

### Recommended Configurations

| Use case | Config | Recall | QPS | Latency (p95) |
|----------|--------|--------|-----|---------------|
| **Low latency** | m=12, ef_search=10 | 98.0% | 8,819 | 0.14 ms |
| **Balanced** | m=36, ef_search=40 | 99.9% | 4,159 | 0.33 ms |
| **High accuracy** | m=48, ef_search=200 | ~100% | 1,676 | 0.89 ms |

For most workloads, m=12 with ef_search between 10 and 40 provides excellent throughput with over 98% recall. Increase m to 36 and ef_search to 40 when 99.9% recall is needed. For perfect 100% recall, m=48 with ef_search=600 achieves 913 QPS.

## Parameter Exploration

How each `m` value behaves across the ef_search sweep:

| m | Build time | Best QPS | Best QPS with recall >= 99.9% | 100% recall |
|---|-----------|----------|-------------------------------|-------------|
| 4 | 18.5 s | 13,566 QPS (ef=10, 82.4%) | n/a | n/a |
| 8 | 25.8 s | 10,410 QPS (ef=20, 95.0%) | 2,100 QPS (ef=400) | n/a |
| 12 | 31.0 s | 8,819 QPS (ef=10, 98.0%) | 3,595 QPS (ef=120) | n/a |
| 16 | 34.3 s | 7,733 QPS (ef=10, 98.8%) | 3,214 QPS (ef=120) | n/a |
| 24 | 38.1 s | 6,700 QPS (ef=10, 99.3%) | 3,478 QPS (ef=80) | n/a |
| 36 | 41.7 s | 5,621 QPS (ef=10, 99.7%) | 4,159 QPS (ef=40) | n/a |
| 48 | 45.2 s | 4,927 QPS (ef=10, 99.8%) | 3,690 QPS (ef=40) | 913 QPS (ef=600) |
| 64 | 49.2 s | 4,301 QPS (ef=10, 99.8%) | 3,244 QPS (ef=40) | 838 QPS (ef=600) |
| 96 | 56.0 s | 3,455 QPS (ef=10, 99.9%) | 2,615 QPS (ef=40) | 725 QPS (ef=600) |

Lower m values (4, 8) give the fastest queries with lower recall. The sweet spot for 99.9% recall throughput is m=36 with ef=40, achieving 4,159 QPS. For 100% recall, m=48 is optimal (913 QPS at ef=600), beating m=64 and m=96 despite lower graph connectivity.

**Note:** The query executor uses `max(2*k, ef_search)` as the effective search beam width. With k=10, ef_search values below 20 all produce an effective ef of 20. This is why ef=10 and ef=20 yield nearly the same recall for each m value.

## Environment

| Item | Value |
|------|-------|
| Date | 2026-02-28 |
| OS | Darwin 25.1.0 arm64 |
| CPU | Apple M4 (single core, `RAYON_NUM_THREADS=1`) |
| RAM | 16 GiB |
| Engine | stoolap v0.3.2 |
| Rust | rustc 1.92.0 |
| Sweep m | 4, 8, 12, 16, 24, 36, 48, 64, 96 |
| Sweep ef_search | 10, 20, 40, 80, 120, 200, 400, 600, 800 |
| ef_construction | 500 (matches ann-benchmarks.com) |
| Queries per configuration | 10,000 |
| Runs per configuration | HNSW: 5 (best-of-5), brute-force: 1 |
| Measured configurations | 81 |

## Limitations

- **Not directly comparable to ann-benchmarks.com**: Our benchmark measures full SQL query time (including parsing and planning), not raw index operations. Our QPS numbers include overhead that pure index benchmarks avoid.
- **Single machine**: All measurements are from a single Apple M4 machine. Results will differ on other hardware.
- **No cross-engine comparison**: This report shows Stoolap's absolute performance on a standard dataset. We have not run other engines with the same harness on identical hardware.

## Reproduce

```bash
# Self-contained: downloads data, computes ground truth, runs benchmark
RAYON_NUM_THREADS=1 cargo run --release --example ann_benchmark \
  --features ann-benchmark -- --sweep --runs 5 --max-queries 10000 \
  --csv sweep-results.csv
```

See [`examples/ann_benchmark.rs`]({{ site.github.repository_url }}/blob/main/examples/ann_benchmark.rs) for the full source.

## References

- [ANN-Benchmarks](https://ann-benchmarks.com/) by Erik Bernhardsson et al.
- Dataset: [Fashion-MNIST](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/) (same data as [fashion-mnist-784-euclidean](https://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5) on ANN-Benchmarks)
- Stoolap benchmark source: [`examples/ann_benchmark.rs`]({{ site.github.repository_url }}/blob/main/examples/ann_benchmark.rs)
- Blog post: [Vector and Semantic Search in SQL](/blog/2026/02/27/vector-and-semantic-search-in-sql/)
