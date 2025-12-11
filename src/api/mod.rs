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

//! Top-level Database API
//!
//! This module provides the high-level database interface for Stoolap.
//!
//! # Quick Start
//!
//! ```ignore
//! use stoolap::{Database, params};
//!
//! // Open an in-memory database
//! let db = Database::open_in_memory()?;
//!
//! // Create a table
//! db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)", ())?;
//!
//! // Insert data with parameters
//! db.execute("INSERT INTO users VALUES ($1, $2, $3)", (1, "Alice", 30))?;
//! db.execute("INSERT INTO users VALUES ($1, $2, $3)", params![2, "Bob", 25])?;
//!
//! // Query data
//! for row in db.query("SELECT * FROM users WHERE age > $1", (20,))? {
//!     let row = row?;
//!     let id: i64 = row.get(0)?;
//!     let name: String = row.get("name")?;
//!     println!("{}: {}", id, name);
//! }
//!
//! // Query single value
//! let count: i64 = db.query_one("SELECT COUNT(*) FROM users", ())?;
//!
//! // Transactions
//! let mut tx = db.begin()?;
//! tx.execute("UPDATE users SET age = age + 1", ())?;
//! tx.commit()?;
//! ```
//!
//! # Parameter Binding
//!
//! Parameters can be passed in several ways:
//!
//! ```ignore
//! // Empty tuple for no parameters
//! db.execute("CREATE TABLE foo (id INTEGER)", ())?;
//!
//! // Tuple syntax for inline parameters
//! db.execute("INSERT INTO foo VALUES ($1, $2)", (1, "Alice"))?;
//!
//! // params! macro for explicit parameter list
//! db.execute("INSERT INTO foo VALUES ($1, $2)", params![1, "Alice"])?;
//!
//! // Optional values
//! let name: Option<&str> = Some("Alice");
//! db.execute("INSERT INTO foo VALUES ($1, $2)", (1, name))?;
//! ```
//!
//! # Prepared Statements
//!
//! ```ignore
//! let stmt = db.prepare("SELECT * FROM users WHERE id = $1")?;
//!
//! // Execute multiple times with different parameters
//! for id in 1..=10 {
//!     for row in stmt.query((id,))? {
//!         // ...
//!     }
//! }
//! ```

pub mod database;
pub mod params;
pub mod rows;
pub mod statement;
pub mod transaction;

pub use database::{Database, FromValue};
pub use params::{NamedParams, Params, ToParam};
pub use rows::{FromRow, ResultRow, Rows};
pub use statement::Statement;
pub use transaction::Transaction;
