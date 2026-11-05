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

//! Subqueries: correlated, EXISTS, IN, scalar, derived tables and their blocks

mod block_cache_identity_test;
mod block_join_sources_test;
mod block_later_arguments_test;
mod block_order_limit_test;
mod block_order_position_test;
mod bug_aggregate_in_subquery_test;
mod correlated_having_test;
mod correlated_over_derived_source_test;
mod correlated_over_view_test;
mod correlated_probe_view_test;
mod correlated_subquery_scope_test;
mod correlated_subquery_test;
mod correlated_tuple_in_test;
mod derived_aggregate_once_test;
mod derived_table_blocks_test;
mod exists_few_outer_rows_test;
mod exists_subquery_test;
mod from_source_sites_test;
mod fromless_statement_blocks_test;
mod in_subquery_cache_test;
mod limit_subquery_prepare_test;
mod limit_subquery_test;
mod limit_subquery_volatile_test;
mod nested_outer_reference_test;
mod nested_plan_slots_test;
mod nested_subquery_binding_test;
mod not_exists_null_scan_test;
mod not_in_null_once_test;
mod not_in_subquery_null_test;
mod quantified_subquery_bounds_test;
mod row_count_rule_test;
mod scalar_subquery_test;
mod statement_form_blocks_test;
mod statement_form_rule_test;
mod subquery_advanced_test;
mod subquery_cache_parameters_test;
mod subquery_cache_stamp_test;
mod subquery_coverage_test;
mod subquery_delete_test;
mod subquery_own_with_test;
mod subquery_plan_nodes_test;
mod subquery_site_rules_test;
mod subquery_update_test;
mod unplanned_subquery_test;
mod values_tvf_subquery_test;
mod volatile_subquery_cache_test;
mod where_subquery_containers_test;
mod where_subquery_shaping_test;
