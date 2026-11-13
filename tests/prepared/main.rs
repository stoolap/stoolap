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

//! Prepared statements, parameters and held programs

#[macro_use]
#[path = "../oracle/mod.rs"]
mod oracle;

mod api_prepared_paths_test;
mod bug_like_param_test;
mod compiled_positions_test;
mod grouped_output_rows_test;
mod held_program_retry_test;
mod held_programs_rebind_test;
mod insert_plan_cache_test;
mod insert_plan_rename_test;
mod key_read_path_equivalence_test;
mod named_params_test;
mod old_path_ledger_test;
mod param_slots_matrix_test;
mod param_slots_test;
mod parameter_binding_test;
mod params_comprehensive_test;
mod prepare_error_test;
mod prepared_conflict_snapshot_test;
mod prepared_conflict_test;
mod prepared_dml_access_test;
mod prepared_dml_snapshot_test;
mod prepared_dml_test;
mod prepared_exists_test;
mod prepared_fromless_block_test;
mod prepared_in_test;
mod prepared_insert_test;
mod prepared_join_test;
mod prepared_kept_members_test;
mod prepared_literal_twin_test;
mod prepared_order_limit_test;
mod prepared_returning_test;
mod prepared_select_test;
mod prepared_set_branch_test;
mod prepared_set_scan_test;
mod prepared_strategy_test;
mod prepared_where_pushdown_test;
mod projection_parameters_test;
mod projection_paths_test;
mod projection_rules_test;
mod pushdown_parameter_expression_test;
mod runner_held_programs_test;
mod shared_literals_test;
