// Copyright 2026 Stoolap Contributors
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

//! Durable catalog identities and immutable schema history.
//!
//! These metadata types do not activate a storage format, publish DDL or
//! acknowledge WAL coverage. Schema column positions remain separate from the
//! stable identities used to interpret historical rows.

pub mod history;
pub mod identity;

pub use history::{ProjectedRow, ProjectionPlan, SchemaRevision, TableSchemaHistory};
pub use identity::{
    ColumnId, ColumnIdAllocator, Incarnation, TableId, TableIdAllocator, TableIdentity,
};
