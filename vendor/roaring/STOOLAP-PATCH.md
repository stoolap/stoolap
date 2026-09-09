# Stoolap local patch

Upstream: https://github.com/RoaringBitmap/roaring-rs
Crate: roaring 0.11.3 from crates.io (the pre-existing Cargo.lock version).
Crates.io archive SHA-256: `8ba9ce64a8f45d7fc86358410bb1a82e8c987504c0d4900e9141d69a9f26c885`.
License: MIT OR Apache-2.0; both upstream license files are preserved.

The unmodified package source, manifest and tests are retained. Cargo's local
`.cargo-ok` marker and its standalone generated Cargo.lock are omitted.

This patch adds read-only `allocation_for_value(value, inserting)` APIs to
RoaringTreemap and RoaringBitmap. They inspect only the affected high-key group
and container. Private Store/IntervalStore helpers report requested allocation
capacity and a conservative local point-mutation allocation bound. BTreeMap
node storage is intentionally excluded and must be bounded separately by the
caller. No insertion, deletion, compression, iteration or serialization
algorithm changes. Focused regression coverage lives in treemap/inherent.rs.

The API supports incremental point-mutation accounting. It is not a total
bitmap size, a serialized-size estimate, or a general bound for bulk set ops.
