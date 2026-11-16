// Copyright 2025 Stoolap Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Expression simplification, the bloom filter a hash join builds over its
//! build keys, and the row-count corrections EXPLAIN ANALYZE records

pub mod bloom;
pub mod feedback;
pub mod simplify;

pub use feedback::{
    extract_column_from_predicate, fingerprint_predicate, global_feedback_cache,
    CardinalityFeedback, FeedbackCache, DEFAULT_DECAY_FACTOR, MAX_CORRECTION_FACTOR,
    MIN_CORRECTION_FACTOR, MIN_SAMPLE_COUNT,
};

pub use bloom::{BloomFilter, BloomFilterBuilder, RuntimeBloomFilter};

pub use simplify::{simplify_expression, simplify_expression_fixed_point, ExpressionSimplifier};
