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

//! Sealing hot rows into volumes, compaction and truncation

mod arena_slot_reuse_test;
mod captured_version_payload_test;
mod commit_seal_stamp_test;
mod compaction_proportion_test;
mod compaction_publication_order_test;
mod compaction_publication_visibility_test;
mod compaction_tombstone_holders_test;
mod count_inside_seal_test;
mod deduped_count_test;
mod failed_commit_sealed_rows_test;
mod file_volume_eviction_test;
mod limit_seal_consistency_test;
mod maintenance_limit_test;
mod seal_dead_copy_test;
mod seal_deferral_test;
mod seal_fence_dml_test;
mod seal_publication_window_test;
mod sealed_index_visibility_test;
mod sealed_window_columns_test;
mod segment_correctness_test;
mod segmented_count_equal_test;
mod streaming_compaction_test;
mod streaming_seal_test;
mod truncate_reader_window_test;
mod truncate_refusal_test;
mod truncate_safety_test;
mod truncate_walk_and_seal_fetch_test;
mod unique_index_eviction_test;
mod update_fence_metadata_test;
mod v4_rewrite_test;
mod volume_concurrency_test;
mod volume_integration_test;
mod volume_metadata_test;
mod volume_row_order_test;
mod volume_schema_test;
