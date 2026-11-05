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

//! The lexer and the parser

#[macro_use]
#[path = "../oracle/mod.rs"]
mod oracle;

mod bug_keyword_column_case_test;
mod keyword_identifiers_test;
mod lexer_tokens_test;
mod nesting_limit_test;
mod operator_chain_test;
mod order_by_parser_test;
mod parser_edge_cases_test;
mod sql_comment_parse_test;
mod sql_literals_test;
mod string_quotes_test;
mod unquoted_keyword_column_test;
