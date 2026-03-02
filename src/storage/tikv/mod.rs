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

//! TiKV distributed storage backend
//!
//! This module provides a TiKV-based storage engine implementation that enables
//! distributed, horizontally scalable storage for Stoolap.
//!
//! # Usage
//!
//! Enable the `tikv` feature flag and connect via DSN:
//!
//! ```ignore
//! let db = Database::open("tikv://pd1:2379,pd2:2379,pd3:2379")?;
//! ```

pub mod encoding;
pub mod engine;
pub mod error;
pub mod index;
pub mod result;
pub mod scanner;
pub mod table;
pub mod transaction;

pub use engine::TiKVEngine;
