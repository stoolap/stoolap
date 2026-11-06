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

//! Name resolution: aliases, column names, scopes and views

mod ambiguous_column_test;
mod bug_view_column_alias_test;
mod column_alias_select_test;
mod column_alias_test;
mod column_alias_where_basic_test;
mod column_alias_where_test;
mod column_name_case_test;
mod column_rename_resolution_test;
mod column_sources_test;
mod expression_identity_test;
mod grouped_ambiguous_name_test;
mod keyword_column_test;
mod missing_name_rule_test;
mod output_column_names_test;
mod qualifier_scope_test;
mod quoted_boolean_name_test;
mod stored_expression_text_test;
mod view_column_names_test;
mod view_join_filter_test;
mod view_test;
mod where_alias_operators_test;
