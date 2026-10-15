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

//! The index definition a side file entry is built for and read through:
//! the index's identity, the columns its key is made of, and the kind of
//! key the entry stores.

use smallvec::SmallVec;

use crate::core::DataType;

/// The value domain one key component is encoded in
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    Integer,
    Timestamp,
}

impl Domain {
    /// The domain a column of `data_type` is keyed in, if it has one
    pub fn of(data_type: DataType) -> Option<Self> {
        match data_type {
            DataType::Integer => Some(Self::Integer),
            DataType::Timestamp => Some(Self::Timestamp),
            _ => None,
        }
    }
}

/// How an entry stores its keys
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyKind {
    /// One i64 per distinct key, in order
    OrderedI64,
}

/// What a read asks of an entry
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
    Equality,
    Range,
}

impl KeyKind {
    pub fn provides(self, capability: Capability) -> bool {
        match self {
            KeyKind::OrderedI64 => matches!(capability, Capability::Equality | Capability::Range),
        }
    }
}

/// One column of a key, by its schema position
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Component {
    pub column: usize,
    pub domain: Domain,
}

/// An index a side file can hold an entry for
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SideIndexDef {
    /// The LSN of the catalog record that created the index; never zero
    pub identity: u64,
    /// The key's columns in key order, never empty
    pub components: SmallVec<[Component; 2]>,
    pub kind: KeyKind,
}

impl SideIndexDef {
    /// The key's first column
    pub fn leading(&self) -> Option<Component> {
        self.components.first().copied()
    }
}

/// The physical columns and identities of the entries a volume written in
/// the current schema is built with: its physical columns are the schema's
pub fn schema_entries(defs: &[SideIndexDef]) -> Vec<(usize, u64)> {
    defs.iter()
        .filter_map(|def| Some((def.leading()?.column, def.identity)))
        .collect()
}
