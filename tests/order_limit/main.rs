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

//! ORDER BY, LIMIT, OFFSET and top-k reads

mod bug_contains_orderby_params_test;
mod general_order_limit_test;
mod limit_volatile_once_test;
mod non_number_order_test;
mod offset_before_limit_test;
mod order_by_alias_binding_test;
mod order_by_hidden_column_test;
mod order_by_hidden_derived_column_test;
mod order_by_keys_bridge_test;
mod order_by_position_test;
mod order_by_subquery_test;
mod order_limit_path_equivalence_test;
mod order_limit_test;
mod ordered_limited_scan_test;
mod outer_order_key_test;
mod range_top_k_test;
mod sort_order_rule_test;
mod top_k_added_column_test;
mod values_sort_key_test;
