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

use std::num::NonZeroU64;

use crate::core::{Error, Result};

macro_rules! identity {
    ($name:ident, $description:literal) => {
        #[doc = $description]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(transparent)]
        pub struct $name(NonZeroU64);

        impl $name {
            pub fn new(value: u64) -> Result<Self> {
                NonZeroU64::new(value)
                    .map(Self)
                    .ok_or_else(|| Error::internal(concat!(stringify!($name), " must be nonzero")))
            }

            pub const fn from_nonzero(value: NonZeroU64) -> Self {
                Self(value)
            }

            pub const fn get(self) -> u64 {
                self.0.get()
            }

            pub const fn as_nonzero(self) -> NonZeroU64 {
                self.0
            }
        }

        impl TryFrom<u64> for $name {
            type Error = Error;

            fn try_from(value: u64) -> Result<Self> {
                Self::new(value)
            }
        }

        impl From<$name> for u64 {
            fn from(value: $name) -> Self {
                value.get()
            }
        }
    };
}

identity!(
    TableId,
    "A stable table ID within one durable database catalog."
);
identity!(
    Incarnation,
    "A table's data incarnation, advanced by destructive replacement."
);
identity!(
    ColumnId,
    "A stable column ID within a TableId, independent of its schema position."
);

impl Incarnation {
    pub const FIRST: Self = Self(NonZeroU64::MIN);

    pub fn checked_next(self) -> Result<Self> {
        self.get()
            .checked_add(1)
            .ok_or_else(|| Error::internal("table incarnation exhausted"))
            .and_then(Self::new)
    }
}

/// Complete identity for row data and work prepared against a table.
/// A rename preserves this pair; a new incarnation preserves only TableId.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TableIdentity {
    pub table_id: TableId,
    pub incarnation: Incarnation,
}

impl TableIdentity {
    pub const fn new(table_id: TableId, incarnation: Incarnation) -> Self {
        Self {
            table_id,
            incarnation,
        }
    }

    pub fn checked_next_incarnation(self) -> Result<Self> {
        Ok(Self::new(self.table_id, self.incarnation.checked_next()?))
    }
}

macro_rules! allocator {
    ($name:ident, $id:ident, $description:literal) => {
        #[doc = $description]
        ///
        /// One catalog owner serializes allocation and persists this high-water
        /// mark, including IDs reserved for operations that later fail. Restore
        /// must use that mark, not the maximum currently live ID. The allocator
        /// is deliberately not Clone; dropping an object never returns its ID.
        #[derive(Debug, Default)]
        pub struct $name {
            high_water_mark: u64,
        }

        impl $name {
            pub const fn new() -> Self {
                Self { high_water_mark: 0 }
            }

            /// Restore a previously persisted allocation high-water mark.
            /// u64::MAX represents an exhausted sequence, not a wrapped ID.
            pub const fn from_high_water_mark(high_water_mark: u64) -> Self {
                Self { high_water_mark }
            }

            pub const fn high_water_mark(&self) -> u64 {
                self.high_water_mark
            }

            pub fn allocate(&mut self) -> Result<$id> {
                let next = self.high_water_mark.checked_add(1).ok_or_else(|| {
                    Error::internal(concat!(stringify!($id), " allocation exhausted"))
                })?;
                let id = $id::new(next)?;
                self.high_water_mark = next;
                Ok(id)
            }
        }
    };
}

allocator!(
    TableIdAllocator,
    TableId,
    "Monotonic table IDs; DROP/recreate does not reuse an ID."
);
allocator!(
    ColumnIdAllocator,
    ColumnId,
    "Monotonic column IDs; DROP/re-add and TRUNCATE do not reset this sequence."
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonzero_identities_and_checked_incarnation() {
        assert!(TableId::new(0).is_err());
        assert!(ColumnId::try_from(0).is_err());
        assert!(Incarnation::new(0).is_err());
        let identity = TableIdentity::new(TableId::new(7).unwrap(), Incarnation::FIRST);
        let next = identity.checked_next_incarnation().unwrap();
        assert_eq!(next.table_id, identity.table_id);
        assert_eq!(next.incarnation.get(), 2);
        assert!(Incarnation::new(u64::MAX).unwrap().checked_next().is_err());
        assert_eq!(std::mem::size_of::<Option<TableId>>(), 8);
        assert_eq!(std::mem::size_of::<TableIdentity>(), 16);
    }

    #[test]
    fn allocation_restores_reserved_ids_and_never_wraps() {
        let mut tables = TableIdAllocator::new();
        assert_eq!(tables.allocate().unwrap().get(), 1);
        // ID2 was reserved by a failed CREATE; no live descriptor need exist.
        assert_eq!(tables.allocate().unwrap().get(), 2);
        let mut restored = TableIdAllocator::from_high_water_mark(tables.high_water_mark());
        assert_eq!(restored.allocate().unwrap().get(), 3);
        let mut columns = ColumnIdAllocator::from_high_water_mark(u64::MAX - 1);
        assert_eq!(columns.allocate().unwrap().get(), u64::MAX);
        for _ in 0..2 {
            assert!(columns.allocate().is_err());
            assert_eq!(columns.high_water_mark(), u64::MAX);
        }
        assert!(TableIdAllocator::from_high_water_mark(u64::MAX)
            .allocate()
            .is_err());
    }
}
