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

//! Reads of sealed rows: cold scans, zone maps, the group cache and schema changes over volumes

mod cluster_by_test;
mod cold_distinct_null_test;
mod cold_read_consistency_test;
mod cold_unique_equality_test;
mod column_change_close_test;
mod column_type_change_volume_test;
mod decoded_group_cache_test;
mod delete_cold_rounds_test;
mod delete_cold_shadow_test;
mod drop_column_readers_test;
mod drop_column_transition_test;
mod executor_cold_error_test;
mod fallible_group_access_test;
mod fallible_row_count_test;
mod hot_index_top_k_test;
mod hot_limits_test;
mod hot_schema_evolution_test;
mod hot_streaming_scan_test;
mod insert_cold_rounds_test;
mod kept_descriptor_test;
mod keyless_cold_update_test;
mod limit_cold_read_consistency_test;
mod opened_table_access_test;
mod read_table_handle_test;
mod rename_column_index_test;
mod retype_failure_test;
mod retype_protocol_test;
mod retyped_unique_lookup_test;
mod row_group_prune_test;
mod scan_access_test;
mod scanner_group_cache_test;
mod snapshot_cold_fetch_test;
mod snapshot_cold_test;
mod storage_aggregate_admission_test;
mod storage_filter_error_test;
mod targeted_cold_load_test;
mod upsert_cold_dupe_test;
mod zone_map_visibility_test;
mod zonemap_test;
