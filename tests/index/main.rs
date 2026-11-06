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

//! Indexes of every kind and the primary key paths

mod bitmap_index_range_test;
mod bitmap_index_sql_test;
mod bitmap_negative_row_id_test;
mod btree_index_sql_test;
mod bug_unique_index_test;
mod carried_key_identity_test;
mod composite_range_intersection_test;
mod count_with_index_test;
mod crossing_range_bounds_test;
mod duplicate_index_test;
mod empty_index_probe_test;
mod exists_side_file_probe_test;
mod hash_candidate_cap_test;
mod hash_index_sql_test;
mod in_equality_candidates_test;
mod index_count_test;
mod index_join_rule_test;
mod index_name_case_test;
mod index_operations_test;
mod index_optimizer_coverage_test;
mod index_optimizer_test;
mod index_page_boundary_test;
mod index_rebound_test;
mod join_equality_candidates_test;
mod memory_stats_unique_index_test;
mod min_max_index_test;
mod order_by_index_gate_test;
mod order_by_index_nulls_test;
mod pk_conjunct_test;
mod pk_fast_path_limit_test;
mod pk_fast_path_test;
mod pk_in_lookup_test;
mod pk_point_read_columns_test;
mod reinsert_unique_check_test;
mod secondary_index_backfill_test;
mod secondary_index_query_test;
mod secondary_side_file_test;
mod side_index_bytes_test;
mod side_index_dictionary_test;
mod side_index_float_test;
mod side_index_matrix_test;
mod sorted_equality_test;
mod unique_check_lock_order_test;
mod unique_conflict_message_test;
mod unique_get_test;
mod unique_index_test;
mod unique_plan_check_test;
mod view_index_candidates_test;
