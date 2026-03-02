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

//! View definition types shared across storage backends

/// View definition storing the query that defines the view
#[derive(Debug, Clone)]
pub struct ViewDefinition {
    /// View name (lowercase for case-insensitive lookup)
    pub name: String,
    /// Original view name (preserves case)
    pub original_name: String,
    /// The SQL query string that defines the view
    pub query: String,
}

impl ViewDefinition {
    /// Create a new view definition
    pub fn new(name: &str, query: String) -> Self {
        Self {
            name: name.to_lowercase(),
            original_name: name.to_string(),
            query,
        }
    }
}
