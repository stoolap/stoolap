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

//! Aggregates, GROUP BY, HAVING and grouped output

#[macro_use]
#[path = "../oracle/mod.rs"]
mod oracle;

mod aggregate_alias_group_column_test;
mod aggregate_distinct_on_test;
mod aggregate_expression_names_test;
mod aggregate_functions_test;
mod aggregate_modifier_rule_test;
mod aggregate_modifiers_test;
mod aggregate_nan_test;
mod aggregate_order_test;
mod aggregate_path_equivalence_test;
mod aggregate_set_operation_window_test;
mod aggregate_under_operators_test;
mod aggregation_advanced_test;
mod aggregation_test;
mod boolean_aggregate_test;
mod bug_alias_shadow_group_by_test;
mod bug_count_column_nulls_test;
mod bug_max_no_where_test;
mod count_aggregate_test;
mod count_star_fallback_test;
mod extension_min_max_test;
mod first_last_aggregate_test;
mod global_aggregate_projection_test;
mod global_aggregate_rule_test;
mod group_by_aliased_key_reuse_test;
mod group_by_expression_aggregate_test;
mod group_by_key_rule_test;
mod group_by_order_test;
mod group_by_resize_test;
mod group_by_rule_test;
mod group_key_projection_test;
mod grouped_computed_order_keys_test;
mod grouped_index_join_having_test;
mod grouped_join_limit_parameter_test;
mod grouped_join_stream_test;
mod grouped_output_blocks_test;
mod grouped_sources_blocks_test;
mod grouping_float_keys_test;
mod grouping_identity_rule_test;
mod having_aggregates_test;
mod having_constant_fold_test;
mod having_without_group_by_test;
mod rollup_cube_test;
mod runner_global_aggregate_test;
mod runner_grouped_aggregate_test;
mod runner_source_aggregate_test;
mod streaming_group_by_view_test;
mod string_agg_separator_test;
mod sum_exact_integer_test;
mod ungrouped_column_rule_test;
