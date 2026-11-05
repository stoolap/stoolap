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

//! Window functions

#[macro_use]
#[path = "../oracle/mod.rs"]
mod oracle;

mod count_windows_test;
mod grouped_window_plan_test;
mod grouped_windows_blocks_test;
mod grouping_sets_window_test;
mod named_window_extension_test;
mod order_by_window_test;
mod planned_window_test;
mod unused_window_test;
mod window_advanced_test;
mod window_argument_once_test;
mod window_argument_registry_test;
mod window_arguments_test;
mod window_column_names_test;
mod window_columns_nulls_test;
mod window_coverage_test;
mod window_function_test;
mod window_named_set_operation_test;
mod window_over_group_by_aggregate_test;
mod window_over_groups_rule_test;
mod window_over_key_test;
mod window_qualified_argument_test;
mod window_quantified_left_test;
mod window_range_null_test;
mod window_sort_keys_test;
mod windowed_blocks_test;
