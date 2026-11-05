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

//! Operators and types: casts, comparisons, NULL, IN lists and arithmetic

mod bitwise_operators_test;
mod boolean_predicate_test;
mod cast_column_alias_test;
mod cast_padded_text_test;
mod cast_simple_test;
mod cast_test;
mod cast_where_clause_test;
mod column_range_rule_test;
mod comparison_contract_test;
mod comparison_rule_test;
mod constant_folding_test;
mod data_type_validation_test;
mod family_operators_test;
mod float_int_where_test;
mod from_row_test;
mod in_list_duplicate_members_test;
mod in_list_member_type_test;
mod in_list_mixed_bounds_test;
mod in_list_nan_test;
mod in_list_of_expressions_test;
mod in_member_domain_test;
mod int_float_exact_compare_test;
mod integer_arithmetic_rule_test;
mod integer_conversion_test;
mod is_null_test;
mod long_program_test;
mod nan_total_order_test;
mod not_operator_test;
mod not_over_compound_test;
mod null_comparison_test;
mod null_times_zero_test;
mod row_comparison_rule_test;
mod scan_with_where_test;
mod sql_expressions_test;
mod where_clause_test;
mod where_value_error_test;
