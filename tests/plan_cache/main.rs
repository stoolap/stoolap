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

//! Plans and caches: query and semantic caches, EXPLAIN, PRAGMA and the optimizer

mod adaptive_execution_test;
mod bug_now_stale_cache_test;
mod cardinality_feedback_test;
mod constant_false_where_test;
mod explain_plan_test;
mod explain_test;
mod expression_pushdown_test;
mod filter_pushdown_test;
mod parallel_execution_tests;
mod parallel_regexp_test;
mod plan_lock_reentry_test;
mod plan_validity_test;
mod planned_route_test;
mod planned_statement_test;
mod planner_explain_test;
mod pragma_test;
mod query_cache_key_test;
mod query_cache_test;
mod query_optimization_paths_test;
mod semantic_cache_clear_test;
mod semantic_cache_cross_handle_test;
mod semantic_cache_join_side_test;
mod semantic_cache_stamp_order_test;
mod semantic_cache_tests;
mod show_commands_test;
mod statistics_test;
