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

//! Core type definitions for Stoolap
//!
//! This module defines the fundamental types: DataType, Operator, IndexType, IsolationLevel

use std::fmt;
use std::str::FromStr;

use super::error::Error;

/// SQL data types supported by Stoolap
///

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum DataType {
    /// NULL data type, used for unknown/unspecified types

    #[default]
    Null = 0,

    /// 64-bit signed integer
    Integer = 1,

    /// 64-bit floating point number
    Float = 2,

    /// UTF-8 text string
    Text = 3,

    /// Boolean true/false
    Boolean = 4,

    /// Timestamp with timezone (stored as UTC)
    Timestamp = 5,

    /// JSON document
    Json = 6,
}

impl DataType {
    /// Returns true if this type is numeric (INTEGER or FLOAT)
    pub fn is_numeric(&self) -> bool {
        matches!(self, DataType::Integer | DataType::Float)
    }

    /// Returns true if this type can be compared for ordering
    pub fn is_orderable(&self) -> bool {
        !matches!(self, DataType::Json)
    }

    /// Returns the type ID as u8 for serialization
    pub fn as_u8(&self) -> u8 {
        *self as u8
    }

    /// Create DataType from u8
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(DataType::Null),
            1 => Some(DataType::Integer),
            2 => Some(DataType::Float),
            3 => Some(DataType::Text),
            4 => Some(DataType::Boolean),
            5 => Some(DataType::Timestamp),
            6 => Some(DataType::Json),
            _ => None,
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::Null => write!(f, "NULL"),
            DataType::Integer => write!(f, "INTEGER"),
            DataType::Float => write!(f, "FLOAT"),
            DataType::Text => write!(f, "TEXT"),
            DataType::Boolean => write!(f, "BOOLEAN"),
            DataType::Timestamp => write!(f, "TIMESTAMP"),
            DataType::Json => write!(f, "JSON"),
        }
    }
}

impl FromStr for DataType {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "NULL" => Ok(DataType::Null),
            "INTEGER" | "INT" | "BIGINT" | "SMALLINT" | "TINYINT" => Ok(DataType::Integer),
            "FLOAT" | "DOUBLE" | "REAL" | "DECIMAL" | "NUMERIC" => Ok(DataType::Float),
            "TEXT" | "VARCHAR" | "CHAR" | "STRING" => Ok(DataType::Text),
            "BOOLEAN" | "BOOL" => Ok(DataType::Boolean),
            "TIMESTAMP" | "DATETIME" | "DATE" | "TIME" => Ok(DataType::Timestamp),
            "JSON" | "JSONB" => Ok(DataType::Json),
            _ => Err(Error::InvalidColumnType),
        }
    }
}

/// Comparison operators for expressions
///

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Operator {
    /// Equality (=)
    Eq = 0,

    /// Inequality (!=)
    Ne = 1,

    /// Greater than (>)
    Gt = 2,

    /// Greater than or equal (>=)
    Gte = 3,

    /// Less than (<)
    Lt = 4,

    /// Less than or equal (<=)
    Lte = 5,

    /// Pattern matching (LIKE)
    Like = 6,

    /// Value in set (IN)
    In = 7,

    /// Value not in set (NOT IN)
    NotIn = 8,

    /// IS NULL check
    IsNull = 9,

    /// IS NOT NULL check
    IsNotNull = 10,
}

impl Operator {
    /// Returns true if this operator needs a right-hand value
    pub fn needs_value(&self) -> bool {
        !matches!(self, Operator::IsNull | Operator::IsNotNull)
    }

    /// Returns true if this operator is a null check
    pub fn is_null_check(&self) -> bool {
        matches!(self, Operator::IsNull | Operator::IsNotNull)
    }

    /// Returns the negation of this operator, if applicable
    pub fn negate(&self) -> Option<Self> {
        match self {
            Operator::Eq => Some(Operator::Ne),
            Operator::Ne => Some(Operator::Eq),
            Operator::Gt => Some(Operator::Lte),
            Operator::Gte => Some(Operator::Lt),
            Operator::Lt => Some(Operator::Gte),
            Operator::Lte => Some(Operator::Gt),
            Operator::In => Some(Operator::NotIn),
            Operator::NotIn => Some(Operator::In),
            Operator::IsNull => Some(Operator::IsNotNull),
            Operator::IsNotNull => Some(Operator::IsNull),
            Operator::Like => None, // LIKE doesn't have a simple negation
        }
    }

    /// Returns the type ID as u8 for serialization
    pub fn as_u8(&self) -> u8 {
        *self as u8
    }
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operator::Eq => write!(f, "="),
            Operator::Ne => write!(f, "!="),
            Operator::Gt => write!(f, ">"),
            Operator::Gte => write!(f, ">="),
            Operator::Lt => write!(f, "<"),
            Operator::Lte => write!(f, "<="),
            Operator::Like => write!(f, "LIKE"),
            Operator::In => write!(f, "IN"),
            Operator::NotIn => write!(f, "NOT IN"),
            Operator::IsNull => write!(f, "IS NULL"),
            Operator::IsNotNull => write!(f, "IS NOT NULL"),
        }
    }
}

impl FromStr for Operator {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "=" | "==" => Ok(Operator::Eq),
            "!=" | "<>" => Ok(Operator::Ne),
            ">" => Ok(Operator::Gt),
            ">=" => Ok(Operator::Gte),
            "<" => Ok(Operator::Lt),
            "<=" => Ok(Operator::Lte),
            "LIKE" => Ok(Operator::Like),
            "IN" => Ok(Operator::In),
            "NOT IN" | "NOTIN" => Ok(Operator::NotIn),
            "IS NULL" | "ISNULL" => Ok(Operator::IsNull),
            "IS NOT NULL" | "ISNOTNULL" => Ok(Operator::IsNotNull),
            _ => Err(Error::parse(format!("unknown operator: {}", s))),
        }
    }
}

/// Index types supported by the storage engine
///

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexType {
    /// Bitmap index for low-cardinality columns (< 1000 distinct values)
    /// Best for: BOOLEAN, status fields, categories
    /// Uses RoaringBitmap for O(n/64) AND/OR operations
    Bitmap,

    /// B-tree index for range queries and ordered access
    /// Best for: INTEGER, TIMESTAMP, DATE columns
    /// Supports: <, <=, >, >=, BETWEEN, ORDER BY
    BTree,

    /// Hash index for fast O(1) equality lookups
    /// Best for: TEXT, VARCHAR, UUID columns
    /// Uses ahash - avoids O(strlen) comparisons
    /// Note: Does NOT support range queries
    Hash,

    /// Multi-column composite index for queries on multiple columns
    /// Hybrid: Hash for exact lookups (O(1)), lazy BTree for range queries
    /// Best for: WHERE col1 = x AND col2 = y AND col3 = z
    MultiColumn,
}

impl IndexType {
    /// Returns the string representation used in SQL
    pub fn as_str(&self) -> &'static str {
        match self {
            IndexType::Bitmap => "bitmap",
            IndexType::BTree => "btree",
            IndexType::Hash => "hash",
            IndexType::MultiColumn => "multicolumn",
        }
    }
}

impl fmt::Display for IndexType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for IndexType {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "bitmap" => Ok(IndexType::Bitmap),
            "btree" | "b-tree" => Ok(IndexType::BTree),
            "hash" => Ok(IndexType::Hash),
            "multicolumn" | "multi-column" | "composite" => Ok(IndexType::MultiColumn),
            _ => Err(Error::parse(format!("unknown index type: {}", s))),
        }
    }
}

/// Transaction isolation levels
///

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum IsolationLevel {
    /// Read committed: transactions see only committed data

    #[default]
    ReadCommitted = 0,

    /// Snapshot isolation (equivalent to Repeatable Read):
    /// transactions see a consistent snapshot from the start
    SnapshotIsolation = 1,
}

impl IsolationLevel {
    /// Returns the type ID as u8 for serialization
    pub fn as_u8(&self) -> u8 {
        *self as u8
    }

    /// Create IsolationLevel from u8
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(IsolationLevel::ReadCommitted),
            1 => Some(IsolationLevel::SnapshotIsolation),
            _ => None,
        }
    }
}

impl fmt::Display for IsolationLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IsolationLevel::ReadCommitted => write!(f, "READ COMMITTED"),
            IsolationLevel::SnapshotIsolation => write!(f, "SNAPSHOT ISOLATION"),
        }
    }
}

impl FromStr for IsolationLevel {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "READ COMMITTED" | "READCOMMITTED" => Ok(IsolationLevel::ReadCommitted),
            "SNAPSHOT ISOLATION" | "SNAPSHOTSISOLATION" | "REPEATABLE READ" | "REPEATABLEREAD" => {
                Ok(IsolationLevel::SnapshotIsolation)
            }
            _ => Err(Error::parse(format!("unknown isolation level: {}", s))),
        }
    }
}

/// Index entry representing a result from an index lookup
///

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IndexEntry {
    /// Row ID in the table
    pub row_id: i64,
    /// Reference ID in the index
    pub ref_id: i64,
}

impl IndexEntry {
    /// Create a new index entry
    pub fn new(row_id: i64, ref_id: i64) -> Self {
        Self { row_id, ref_id }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // DataType tests
    // =========================================================================

    #[test]
    fn test_datatype_display() {
        assert_eq!(DataType::Null.to_string(), "NULL");
        assert_eq!(DataType::Integer.to_string(), "INTEGER");
        assert_eq!(DataType::Float.to_string(), "FLOAT");
        assert_eq!(DataType::Text.to_string(), "TEXT");
        assert_eq!(DataType::Boolean.to_string(), "BOOLEAN");
        assert_eq!(DataType::Timestamp.to_string(), "TIMESTAMP");
        assert_eq!(DataType::Json.to_string(), "JSON");
    }

    #[test]
    fn test_datatype_from_str() {
        assert_eq!("INTEGER".parse::<DataType>().unwrap(), DataType::Integer);
        assert_eq!("INT".parse::<DataType>().unwrap(), DataType::Integer);
        assert_eq!("BIGINT".parse::<DataType>().unwrap(), DataType::Integer);
        assert_eq!("float".parse::<DataType>().unwrap(), DataType::Float);
        assert_eq!("TEXT".parse::<DataType>().unwrap(), DataType::Text);
        assert_eq!("VARCHAR".parse::<DataType>().unwrap(), DataType::Text);
        assert_eq!("BOOLEAN".parse::<DataType>().unwrap(), DataType::Boolean);
        assert_eq!("BOOL".parse::<DataType>().unwrap(), DataType::Boolean);
        assert_eq!(
            "TIMESTAMP".parse::<DataType>().unwrap(),
            DataType::Timestamp
        );
        assert_eq!("JSON".parse::<DataType>().unwrap(), DataType::Json);
        assert!("UNKNOWN".parse::<DataType>().is_err());
    }

    #[test]
    fn test_datatype_is_numeric() {
        assert!(DataType::Integer.is_numeric());
        assert!(DataType::Float.is_numeric());
        assert!(!DataType::Text.is_numeric());
        assert!(!DataType::Boolean.is_numeric());
        assert!(!DataType::Timestamp.is_numeric());
        assert!(!DataType::Json.is_numeric());
        assert!(!DataType::Null.is_numeric());
    }

    #[test]
    fn test_datatype_is_orderable() {
        assert!(DataType::Integer.is_orderable());
        assert!(DataType::Float.is_orderable());
        assert!(DataType::Text.is_orderable());
        assert!(DataType::Boolean.is_orderable());
        assert!(DataType::Timestamp.is_orderable());
        assert!(!DataType::Json.is_orderable());
    }

    #[test]
    fn test_datatype_u8_conversion() {
        for (i, dt) in [
            DataType::Null,
            DataType::Integer,
            DataType::Float,
            DataType::Text,
            DataType::Boolean,
            DataType::Timestamp,
            DataType::Json,
        ]
        .iter()
        .enumerate()
        {
            assert_eq!(dt.as_u8(), i as u8);
            assert_eq!(DataType::from_u8(i as u8), Some(*dt));
        }
        assert_eq!(DataType::from_u8(100), None);
    }

    // =========================================================================
    // Operator tests
    // =========================================================================

    #[test]
    fn test_operator_display() {
        assert_eq!(Operator::Eq.to_string(), "=");
        assert_eq!(Operator::Ne.to_string(), "!=");
        assert_eq!(Operator::Gt.to_string(), ">");
        assert_eq!(Operator::Gte.to_string(), ">=");
        assert_eq!(Operator::Lt.to_string(), "<");
        assert_eq!(Operator::Lte.to_string(), "<=");
        assert_eq!(Operator::Like.to_string(), "LIKE");
        assert_eq!(Operator::In.to_string(), "IN");
        assert_eq!(Operator::NotIn.to_string(), "NOT IN");
        assert_eq!(Operator::IsNull.to_string(), "IS NULL");
        assert_eq!(Operator::IsNotNull.to_string(), "IS NOT NULL");
    }

    #[test]
    fn test_operator_from_str() {
        assert_eq!("=".parse::<Operator>().unwrap(), Operator::Eq);
        assert_eq!("==".parse::<Operator>().unwrap(), Operator::Eq);
        assert_eq!("!=".parse::<Operator>().unwrap(), Operator::Ne);
        assert_eq!("<>".parse::<Operator>().unwrap(), Operator::Ne);
        assert_eq!(">".parse::<Operator>().unwrap(), Operator::Gt);
        assert_eq!(">=".parse::<Operator>().unwrap(), Operator::Gte);
        assert_eq!("<".parse::<Operator>().unwrap(), Operator::Lt);
        assert_eq!("<=".parse::<Operator>().unwrap(), Operator::Lte);
        assert_eq!("LIKE".parse::<Operator>().unwrap(), Operator::Like);
        assert_eq!("IN".parse::<Operator>().unwrap(), Operator::In);
        assert_eq!("NOT IN".parse::<Operator>().unwrap(), Operator::NotIn);
        assert_eq!("IS NULL".parse::<Operator>().unwrap(), Operator::IsNull);
        assert_eq!(
            "IS NOT NULL".parse::<Operator>().unwrap(),
            Operator::IsNotNull
        );
    }

    #[test]
    fn test_operator_needs_value() {
        assert!(Operator::Eq.needs_value());
        assert!(Operator::Ne.needs_value());
        assert!(Operator::Gt.needs_value());
        assert!(Operator::Like.needs_value());
        assert!(!Operator::IsNull.needs_value());
        assert!(!Operator::IsNotNull.needs_value());
    }

    #[test]
    fn test_operator_negate() {
        assert_eq!(Operator::Eq.negate(), Some(Operator::Ne));
        assert_eq!(Operator::Ne.negate(), Some(Operator::Eq));
        assert_eq!(Operator::Gt.negate(), Some(Operator::Lte));
        assert_eq!(Operator::Gte.negate(), Some(Operator::Lt));
        assert_eq!(Operator::Lt.negate(), Some(Operator::Gte));
        assert_eq!(Operator::Lte.negate(), Some(Operator::Gt));
        assert_eq!(Operator::In.negate(), Some(Operator::NotIn));
        assert_eq!(Operator::NotIn.negate(), Some(Operator::In));
        assert_eq!(Operator::IsNull.negate(), Some(Operator::IsNotNull));
        assert_eq!(Operator::IsNotNull.negate(), Some(Operator::IsNull));
        assert_eq!(Operator::Like.negate(), None);
    }

    // =========================================================================
    // IndexType tests
    // =========================================================================

    #[test]
    fn test_indextype_display() {
        assert_eq!(IndexType::Bitmap.to_string(), "bitmap");
        assert_eq!(IndexType::BTree.to_string(), "btree");
        assert_eq!(IndexType::Hash.to_string(), "hash");
    }

    #[test]
    fn test_indextype_from_str() {
        assert_eq!("bitmap".parse::<IndexType>().unwrap(), IndexType::Bitmap);
        assert_eq!("btree".parse::<IndexType>().unwrap(), IndexType::BTree);
        assert_eq!("b-tree".parse::<IndexType>().unwrap(), IndexType::BTree);
        assert_eq!("hash".parse::<IndexType>().unwrap(), IndexType::Hash);
        assert!("unknown".parse::<IndexType>().is_err());
    }

    // =========================================================================
    // IsolationLevel tests
    // =========================================================================

    #[test]
    fn test_isolationlevel_display() {
        assert_eq!(IsolationLevel::ReadCommitted.to_string(), "READ COMMITTED");
        assert_eq!(
            IsolationLevel::SnapshotIsolation.to_string(),
            "SNAPSHOT ISOLATION"
        );
    }

    #[test]
    fn test_isolationlevel_from_str() {
        assert_eq!(
            "READ COMMITTED".parse::<IsolationLevel>().unwrap(),
            IsolationLevel::ReadCommitted
        );
        assert_eq!(
            "SNAPSHOT ISOLATION".parse::<IsolationLevel>().unwrap(),
            IsolationLevel::SnapshotIsolation
        );
        assert_eq!(
            "REPEATABLE READ".parse::<IsolationLevel>().unwrap(),
            IsolationLevel::SnapshotIsolation
        );
    }

    #[test]
    fn test_isolationlevel_u8_conversion() {
        assert_eq!(IsolationLevel::ReadCommitted.as_u8(), 0);
        assert_eq!(IsolationLevel::SnapshotIsolation.as_u8(), 1);
        assert_eq!(
            IsolationLevel::from_u8(0),
            Some(IsolationLevel::ReadCommitted)
        );
        assert_eq!(
            IsolationLevel::from_u8(1),
            Some(IsolationLevel::SnapshotIsolation)
        );
        assert_eq!(IsolationLevel::from_u8(2), None);
    }

    // =========================================================================
    // IndexEntry tests
    // =========================================================================

    #[test]
    fn test_index_entry() {
        let entry = IndexEntry::new(42, 100);
        assert_eq!(entry.row_id, 42);
        assert_eq!(entry.ref_id, 100);
    }

    #[test]
    fn test_default_values() {
        assert_eq!(DataType::default(), DataType::Null);
        assert_eq!(IsolationLevel::default(), IsolationLevel::ReadCommitted);
    }
}
