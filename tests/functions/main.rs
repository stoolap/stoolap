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

//! Scalar functions: dates and times, text, JSON, CASE, LIKE and registered functions

mod bug_like_escape_test;
mod bug_timestamp_iso_millis_test;
mod built_string_size_test;
mod case_expression_test;
mod case_operand_test;
mod coalesce_test;
mod collate_function_test;
mod collate_test;
mod date_format_test;
mod date_time_parsing_test;
mod date_time_timestamp_test;
mod generate_series_test;
mod greatest_least_rule_test;
mod hash_functions_test;
mod interval_arithmetic_test;
mod interval_test;
mod json_sql_test;
mod like_pattern_test;
mod new_functions_test;
mod new_scalar_functions_test;
mod pattern_match_rule_test;
mod registered_function_rerun_test;
mod registered_function_test;
mod scalar_function_test;
mod scalar_functions_extended_test;
mod string_boundary_test;
mod text_form_rule_test;
mod time_functions_test;
mod time_trunc_test;
mod timestamp_arithmetic_test;
mod timestamp_range_rule_test;
mod timestamp_text_rule_test;
mod timestamp_zone_rule_test;
mod trim_characters_test;
mod volatile_filter_test;
mod volatility_rule_test;
