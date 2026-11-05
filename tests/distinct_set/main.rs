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

//! DISTINCT, DISTINCT ON and set operations

mod distinct_limit_test;
mod distinct_on_binding_test;
mod distinct_on_key_rule_test;
mod distinct_on_key_volatile_test;
mod distinct_on_set_operation_test;
mod distinct_on_test;
mod distinct_order_key_rule_test;
mod distinct_path_equivalence_test;
mod distinct_set_operation_blocks_test;
mod distinct_test;
mod set_operation_dedup_test;
mod set_operation_leading_distinct_test;
mod set_operation_order_limit_test;
mod set_operation_precedence_test;
mod set_operations_test;
mod simple_distinct_sql_test;
mod simple_distinct_test;
mod view_set_operation_text_test;
