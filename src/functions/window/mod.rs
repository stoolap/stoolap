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

//! Window Functions
//!
//! This module provides window functions for SQL queries:
//!
//! - [`RowNumberFunction`] - ROW_NUMBER()
//! - [`RankFunction`] - RANK()
//! - [`DenseRankFunction`] - DENSE_RANK()
//! - [`NtileFunction`] - NTILE(n)
//! - [`LeadFunction`] - LEAD(column, offset, default)
//! - [`LagFunction`] - LAG(column, offset, default)
//! - [`FirstValueFunction`] - FIRST_VALUE(column)
//! - [`LastValueFunction`] - LAST_VALUE(column)
//! - [`NthValueFunction`] - NTH_VALUE(column, n)
//! - [`PercentRankFunction`] - PERCENT_RANK()
//! - [`CumeDistFunction`] - CUME_DIST()

mod lead_lag;
mod ntile;
mod rank;
mod row_number;
mod value;

pub use lead_lag::{LagFunction, LeadFunction};
pub use ntile::NtileFunction;
pub use rank::{CumeDistFunction, DenseRankFunction, PercentRankFunction, RankFunction};
pub use row_number::RowNumberFunction;
pub use value::{FirstValueFunction, LastValueFunction, NthValueFunction};
