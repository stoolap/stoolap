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

//! INSERT, UPDATE, DELETE, upserts and the constraints they check

#[macro_use]
#[path = "../oracle/mod.rs"]
mod oracle;

mod add_column_not_null_test;
mod alter_auto_increment_test;
mod auto_increment_test;
mod block_reads_tables_as_found_test;
mod bug_composite_unique_upsert_test;
mod bug_conflict_target_test;
mod bug_insert_select_upsert_dup_test;
mod bug_upsert_no_pk_test;
mod check_rule_test;
mod copy_from_test;
mod delete_judged_version_test;
mod delete_test;
mod delete_where_asked_once_test;
mod delete_without_primary_key_test;
mod dml_function_test;
mod dml_held_programs_test;
mod dml_subquery_rule_test;
mod dml_verification_test;
mod dml_where_context_test;
mod drop_primary_key_test;
mod foreign_key_action_into_changed_column_test;
mod foreign_key_cascade_into_key_test;
mod foreign_key_cascade_key_fence_test;
mod foreign_key_cascade_once_test;
mod foreign_key_cascade_race_test;
mod foreign_key_concurrency_test;
mod foreign_key_cross_type_test;
mod foreign_key_delete_key_column_test;
mod foreign_key_drop_column_test;
mod foreign_key_drop_parent_test;
mod foreign_key_one_column_test;
mod foreign_key_rename_parent_test;
mod foreign_key_self_reference_test;
mod foreign_key_test;
mod foreign_key_unique_index_test;
mod foreign_key_update_same_value_test;
mod insert_path_test;
mod insert_select_dataloss_test;
mod insert_select_test;
mod literal_write_rule_test;
mod on_duplicate_key_update_test;
mod pk_write_where_test;
mod plain_dml_family_test;
mod primary_key_update_test;
mod table_level_primary_key_test;
mod unique_update_sealed_test;
mod update_arithmetic_test;
mod update_correlated_restrict_test;
mod update_counts_test;
mod update_parentheses_test;
mod update_path_test;
mod update_set_default_test;
mod update_test;
mod upsert_foreign_key_test;
mod upsert_row_met_twice_test;
mod upsert_where_test;
mod values_clause_test;
mod write_path_rule_test;
