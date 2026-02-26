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

//! Error types for Stoolap
//!
//! This module defines all error types used throughout the storage engine.

use thiserror::Error;

/// Result type alias for Stoolap operations
pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for Stoolap storage operations
///
/// This enum covers all error cases including both sentinel errors
/// and structured errors with context.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum Error {
    // =========================================================================
    // Table errors
    // =========================================================================
    /// Table not found in the database
    #[error("table '{0}' not found")]
    TableNotFound(String),

    /// Table already exists when trying to create
    #[error("table '{0}' already exists")]
    TableAlreadyExists(String),

    /// Table has been closed and cannot be used
    #[error("table closed")]
    TableClosed,

    /// Table column count mismatch
    #[error("table columns don't match, expected {expected}, got {got}")]
    TableColumnsNotMatch { expected: usize, got: usize },

    /// Cannot truncate table because other transactions hold uncommitted writes
    #[error("cannot truncate table: active transactions have uncommitted changes")]
    TableHasActiveTransactions,

    // =========================================================================
    // Column errors
    // =========================================================================
    /// Column not found in table schema
    #[error("column '{0}' not found")]
    ColumnNotFound(String),

    /// Invalid column type for operation
    #[error("invalid column type")]
    InvalidColumnType,

    /// Vector dimension mismatch
    #[error("Vector dimension mismatch: expected {expected}, got {got}")]
    VectorDimensionMismatch { expected: u16, got: u16 },

    /// Duplicate column name in schema
    #[error("duplicate column")]
    DuplicateColumn,

    // =========================================================================
    // Value errors
    // =========================================================================
    /// Invalid value for operation
    #[error("invalid value")]
    InvalidValue,

    /// Invalid argument for function
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// Value exceeds maximum length
    #[error("value for column {column} is too long, max {max}, got {got}")]
    ValueTooLong {
        column: String,
        max: usize,
        got: usize,
    },

    // =========================================================================
    // Constraint errors
    // =========================================================================
    /// NOT NULL constraint violation
    #[error("not null constraint failed for column {column}")]
    NotNullConstraint { column: String },

    /// Primary key constraint violation
    #[error("primary key constraint failed with {row_id} already exists in this table")]
    PrimaryKeyConstraint { row_id: i64 },

    /// Unique constraint violation
    #[error("unique constraint failed for index {index} on column {column} with value {value}")]
    UniqueConstraint {
        index: String,
        column: String,
        value: String,
    },

    /// CHECK constraint violation
    #[error("CHECK constraint failed for column {column}: {expression}")]
    CheckConstraintViolation { column: String, expression: String },

    /// Foreign key constraint violation
    #[error("foreign key constraint violation: column '{column}' in table '{table}' references '{ref_table}({ref_column})' — {detail}")]
    ForeignKeyViolation {
        table: String,
        column: String,
        ref_table: String,
        ref_column: String,
        detail: String,
    },

    // =========================================================================
    // Transaction errors
    // =========================================================================
    /// Transaction has not been started
    #[error("transaction not started")]
    TransactionNotStarted,

    /// Transaction has already been started
    #[error("transaction already started")]
    TransactionAlreadyStarted,

    /// Transaction has already ended (committed or rolled back)
    #[error("transaction already ended")]
    TransactionEnded,

    /// Transaction was aborted
    #[error("transaction aborted")]
    TransactionAborted,

    /// Transaction has already been committed
    #[error("transaction already committed")]
    TransactionCommitted,

    /// Transaction has been closed
    #[error("transaction already closed")]
    TransactionClosed,

    // =========================================================================
    // Index errors
    // =========================================================================
    /// Index not found
    #[error("index '{0}' not found")]
    IndexNotFound(String),

    /// Index already exists
    #[error("index '{0}' already exists")]
    IndexAlreadyExists(String),

    /// Column for index not found
    #[error("index column not found")]
    IndexColumnNotFound,

    /// Index is closed
    #[error("index is closed")]
    IndexClosed,

    // =========================================================================
    // Engine errors
    // =========================================================================
    /// Engine is not open
    #[error("engine is not open")]
    EngineNotOpen,

    /// Engine is already open
    #[error("engine is already open")]
    EngineAlreadyOpen,

    // =========================================================================
    // View errors
    // =========================================================================
    /// View already exists
    #[error("view '{0}' already exists")]
    ViewAlreadyExists(String),

    /// View not found
    #[error("view '{0}' not found")]
    ViewNotFound(String),

    // =========================================================================
    // Lock errors
    // =========================================================================
    /// Failed to acquire lock
    #[error("failed to acquire lock: {0}")]
    LockAcquisitionFailed(String),

    // =========================================================================
    // Query result errors
    // =========================================================================
    /// Query returned no rows
    #[error("query returned no rows")]
    NoRowsReturned,

    /// No statements to execute
    #[error("no statements to execute")]
    NoStatementsToExecute,

    /// Column index out of bounds
    #[error("column index {index} out of bounds")]
    ColumnIndexOutOfBounds { index: usize },

    // =========================================================================
    // WAL errors
    // =========================================================================
    /// WAL manager is not running
    #[error("WAL manager is not running")]
    WalNotRunning,

    /// WAL file is closed
    #[error("WAL file is closed")]
    WalFileClosed,

    /// WAL not initialized
    #[error("WAL not initialized")]
    WalNotInitialized,

    // =========================================================================
    // Database errors
    // =========================================================================
    /// Database is locked by another process
    #[error("database is locked by another process")]
    DatabaseLocked,

    /// Cannot drop primary key column
    #[error("cannot drop primary key column")]
    CannotDropPrimaryKey,

    // =========================================================================
    // Comparison errors
    // =========================================================================
    /// Cannot compare NULL with non-NULL value
    #[error("cannot compare NULL with non-NULL value")]
    NullComparison,

    /// Cannot compare incompatible types
    #[error("cannot compare incompatible types")]
    IncomparableTypes,

    // =========================================================================
    // Other errors
    // =========================================================================
    /// Operation not supported
    #[error("not supported: {0}")]
    NotSupported(String),

    /// Segment not found (internal storage error)
    #[error("segment not found")]
    SegmentNotFound,

    /// Expression evaluation failed
    #[error("expression evaluation failed")]
    ExpressionEvaluation,

    /// Expression evaluation failed with message
    #[error("expression evaluation failed: {message}")]
    ExpressionEvaluationWithMessage { message: String },

    /// Type conversion error
    #[error("type conversion error: cannot convert {from} to {to}")]
    TypeConversion { from: String, to: String },

    /// Parse error
    #[error("parse error: {0}")]
    Parse(String),

    /// IO error (wrapped)
    #[error("IO error: {message}")]
    Io { message: String },

    /// Internal error for unexpected conditions
    #[error("{message}")]
    Internal { message: String },

    // =========================================================================
    // Executor errors
    // =========================================================================
    /// Table or view not found (with name)
    #[error("table or view '{0}' not found")]
    TableOrViewNotFound(String),

    /// Type error
    #[error("type error: {0}")]
    Type(String),

    /// Division by zero
    #[error("division by zero")]
    DivisionByZero,

    /// Query cancelled
    #[error("query cancelled")]
    QueryCancelled,
}

impl Error {
    /// Create a new TableColumnsNotMatch error
    pub fn table_columns_not_match(expected: usize, got: usize) -> Self {
        Error::TableColumnsNotMatch { expected, got }
    }

    /// Create a new ValueTooLong error
    pub fn value_too_long(column: impl Into<String>, max: usize, got: usize) -> Self {
        Error::ValueTooLong {
            column: column.into(),
            max,
            got,
        }
    }

    /// Create a new NotNullConstraint error
    pub fn not_null_constraint(column: impl Into<String>) -> Self {
        Error::NotNullConstraint {
            column: column.into(),
        }
    }

    /// Create a new PrimaryKeyConstraint error
    pub fn primary_key_constraint(row_id: i64) -> Self {
        Error::PrimaryKeyConstraint { row_id }
    }

    /// Create a new UniqueConstraint error
    pub fn unique_constraint(
        index: impl Into<String>,
        column: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Error::UniqueConstraint {
            index: index.into(),
            column: column.into(),
            value: value.into(),
        }
    }

    /// Create a new ForeignKeyViolation error
    pub fn foreign_key_violation(
        table: impl Into<String>,
        column: impl Into<String>,
        ref_table: impl Into<String>,
        ref_column: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Error::ForeignKeyViolation {
            table: table.into(),
            column: column.into(),
            ref_table: ref_table.into(),
            ref_column: ref_column.into(),
            detail: detail.into(),
        }
    }

    /// Create a new TypeConversion error
    pub fn type_conversion(from: impl Into<String>, to: impl Into<String>) -> Self {
        Error::TypeConversion {
            from: from.into(),
            to: to.into(),
        }
    }

    /// Create a new Parse error
    pub fn parse(message: impl Into<String>) -> Self {
        Error::Parse(message.into())
    }

    /// Create a new IO error
    pub fn io(message: impl Into<String>) -> Self {
        Error::Io {
            message: message.into(),
        }
    }

    /// Create a new Internal error
    pub fn internal(message: impl Into<String>) -> Self {
        Error::Internal {
            message: message.into(),
        }
    }

    /// Create a new ExpressionEvaluationWithMessage error
    pub fn expression_evaluation(message: impl Into<String>) -> Self {
        Error::ExpressionEvaluationWithMessage {
            message: message.into(),
        }
    }

    /// Create a new InvalidArgument error
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Error::InvalidArgument(message.into())
    }

    /// Check if this is a "not found" type error
    pub fn is_not_found(&self) -> bool {
        matches!(
            self,
            Error::TableNotFound(_)
                | Error::ColumnNotFound(_)
                | Error::IndexNotFound(_)
                | Error::IndexColumnNotFound
                | Error::SegmentNotFound
                | Error::ViewNotFound(_)
                | Error::TableOrViewNotFound(_)
        )
    }

    /// Check if this is a constraint violation error
    pub fn is_constraint_violation(&self) -> bool {
        matches!(
            self,
            Error::NotNullConstraint { .. }
                | Error::PrimaryKeyConstraint { .. }
                | Error::UniqueConstraint { .. }
                | Error::ForeignKeyViolation { .. }
        )
    }

    /// Check if this is a transaction-related error
    pub fn is_transaction_error(&self) -> bool {
        matches!(
            self,
            Error::TransactionNotStarted
                | Error::TransactionAlreadyStarted
                | Error::TransactionEnded
                | Error::TransactionAborted
                | Error::TransactionCommitted
                | Error::TransactionClosed
        )
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io {
            message: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        assert_eq!(
            Error::TableNotFound("users".to_string()).to_string(),
            "table 'users' not found"
        );
        assert_eq!(
            Error::TableAlreadyExists("users".to_string()).to_string(),
            "table 'users' already exists"
        );
        assert_eq!(
            Error::ColumnNotFound("email".to_string()).to_string(),
            "column 'email' not found"
        );
        assert_eq!(Error::InvalidValue.to_string(), "invalid value");
        assert_eq!(
            Error::TransactionNotStarted.to_string(),
            "transaction not started"
        );
        assert_eq!(
            Error::IndexNotFound("idx_email".to_string()).to_string(),
            "index 'idx_email' not found"
        );
        assert_eq!(
            Error::NullComparison.to_string(),
            "cannot compare NULL with non-NULL value"
        );
    }

    #[test]
    fn test_structured_error_display() {
        let err = Error::table_columns_not_match(5, 3);
        assert_eq!(
            err.to_string(),
            "table columns don't match, expected 5, got 3"
        );

        let err = Error::value_too_long("name", 100, 150);
        assert_eq!(
            err.to_string(),
            "value for column name is too long, max 100, got 150"
        );

        let err = Error::not_null_constraint("email");
        assert_eq!(
            err.to_string(),
            "not null constraint failed for column email"
        );

        let err = Error::primary_key_constraint(42);
        assert_eq!(
            err.to_string(),
            "primary key constraint failed with 42 already exists in this table"
        );

        let err = Error::unique_constraint("idx_email", "email", "test@example.com");
        assert_eq!(
            err.to_string(),
            "unique constraint failed for index idx_email on column email with value test@example.com"
        );
    }

    #[test]
    fn test_error_classification() {
        assert!(Error::TableNotFound("t".to_string()).is_not_found());
        assert!(Error::ColumnNotFound("c".to_string()).is_not_found());
        assert!(Error::IndexNotFound("i".to_string()).is_not_found());
        assert!(!Error::InvalidValue.is_not_found());

        assert!(Error::not_null_constraint("col").is_constraint_violation());
        assert!(Error::primary_key_constraint(1).is_constraint_violation());
        assert!(Error::unique_constraint("idx", "col", "val").is_constraint_violation());
        assert!(!Error::TableNotFound("t".to_string()).is_constraint_violation());

        assert!(Error::TransactionNotStarted.is_transaction_error());
        assert!(Error::TransactionCommitted.is_transaction_error());
        assert!(!Error::TableNotFound("t".to_string()).is_transaction_error());
    }

    #[test]
    fn test_error_equality() {
        assert_eq!(
            Error::TableNotFound("t".to_string()),
            Error::TableNotFound("t".to_string())
        );
        assert_ne!(
            Error::TableNotFound("t".to_string()),
            Error::TableAlreadyExists("t".to_string())
        );

        let err1 = Error::table_columns_not_match(5, 3);
        let err2 = Error::table_columns_not_match(5, 3);
        let err3 = Error::table_columns_not_match(5, 4);
        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::Io { .. }));
        assert!(err.to_string().contains("file not found"));
    }
}
