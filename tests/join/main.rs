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

//! Joins: their plans, sides, conditions and output

#[macro_use]
#[path = "../oracle/mod.rs"]
mod oracle;

mod anti_join_limit_test;
mod correlated_join_on_test;
mod cross_join_limit_test;
mod cross_type_join_test;
mod hash_join_test;
mod join_aggregate_parts_test;
mod join_algorithm_rule_test;
mod join_bare_outer_key_test;
mod join_clause_matrix_test;
mod join_comprehensive_test;
mod join_consumer_rule_test;
mod join_group_limit_chunks_test;
mod join_group_limit_sources_test;
mod join_grouped_feed_test;
mod join_key_filter_outer_test;
mod join_limit_shortfall_test;
mod join_materialized_side_test;
mod join_nested_side_filter_test;
mod join_nested_sides_test;
mod join_on_blocks_test;
mod join_on_condition_test;
mod join_optimizer_test;
mod join_output_blocks_test;
mod join_output_keys_test;
mod join_output_projection_test;
mod join_path_equivalence_test;
mod join_plan_test;
mod join_side_blocks_test;
mod join_side_error_test;
mod join_side_function_columns_test;
mod join_side_needed_columns_test;
mod join_simple_test;
mod join_small_right_side_test;
mod join_source_sides_test;
mod join_source_subquery_test;
mod join_star_blocks_test;
mod join_star_source_names_test;
mod join_unnamed_leaf_test;
mod join_using_column_test;
mod join_where_blocks_test;
mod join_where_false_test;
mod joined_column_sources_test;
mod joined_using_sides_test;
mod merged_join_output_blocks_test;
mod merged_using_kind_test;
mod nested_join_outer_scope_test;
mod plain_join_blocks_test;
mod small_outer_join_test;
mod small_outer_key_types_test;
