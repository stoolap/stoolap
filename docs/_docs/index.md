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

## Documentation Sections

{% for category in site.data.doc_categories %}
{% assign category_docs = site.docs | where: "category", category.name | sort: "order" %}
{% if category_docs.size > 0 %}
### {{ category.name }}
{% for doc in category_docs %}
* [{{ doc.title }}]({{ doc.url | relative_url }})
{% endfor %}
{% endif %}
{% endfor %}

## Need Help?

If you can't find what you're looking for in the documentation, you can:
* [Open an issue](https://github.com/stoolap/stoolap/issues) on GitHub
* [Join the discussions](https://github.com/stoolap/stoolap/discussions) to ask questions

---

This documentation is under active development. Contributions are welcome!
