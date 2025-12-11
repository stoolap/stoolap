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

//! Storage traits for Stoolap
//!
//! This module defines the core interfaces (traits) for the storage layer:
//!
//! - [`Engine`] - The main storage engine interface
//! - [`Transaction`] - Transaction operations and DDL/DML
//! - [`Table`] - Table operations (insert, update, delete, scan)
//! - [`Index`] - Index operations (find, range queries)
//! - [`QueryResult`] - Query result iteration
//! - [`Scanner`] - Row scanning interface
//!

pub mod engine;
pub mod index_trait;
pub mod result;
pub mod scanner;
pub mod table;
pub mod transaction;

// Re-export main traits
pub use engine::Engine;
pub use index_trait::Index;
pub use result::{EmptyResult, MemoryResult, QueryResult};
pub use scanner::{EmptyScanner, Scanner, VecScanner};
pub use table::{ScanPlan, Table};
pub use transaction::{TemporalType, Transaction};
