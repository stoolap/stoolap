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

//! The one rule for a value read as a condition: BOOLEAN, an INTEGER (true
//! when not zero) or NULL; any other type is an error

use super::{DataType, Error, Value};

/// A value of this type was read as a condition; small, so the answer of
/// `condition_truth` stays in a register until it becomes an [`Error`]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NotACondition(pub DataType);

impl From<NotACondition> for Error {
    #[cold]
    #[inline(never)]
    fn from(refused: NotACondition) -> Error {
        not_a_condition(refused.0)
    }
}

/// `value` read as a condition: TRUE, FALSE, or None for unknown
#[inline(always)]
pub fn condition_truth(value: &Value) -> std::result::Result<Option<bool>, NotACondition> {
    match value {
        Value::Boolean(b) => Ok(Some(*b)),
        Value::Integer(i) => Ok(Some(*i != 0)),
        Value::Null(_) => Ok(None),
        other => Err(NotACondition(other.data_type())),
    }
}

/// Whether a value of `data_type` may be read as a condition
pub fn is_condition_type(data_type: DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean | DataType::Integer | DataType::Null
    )
}

/// The error for a value of `data_type` read as a condition
#[cold]
#[inline(never)]
pub fn not_a_condition(data_type: DataType) -> Error {
    Error::Type(not_a_condition_message(data_type))
}

/// What `not_a_condition` says, for an error raised before any row is read
pub fn not_a_condition_message(data_type: DataType) -> String {
    format!("a condition must be BOOLEAN or INTEGER, got {data_type}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boolean_integer_and_null_are_conditions_and_nothing_else() {
        let truth = |v: Value| condition_truth(&v).ok();
        assert_eq!(truth(Value::Boolean(true)), Some(Some(true)));
        assert_eq!(truth(Value::Integer(0)), Some(Some(false)));
        assert_eq!(truth(Value::Integer(-3)), Some(Some(true)));
        assert_eq!(truth(Value::null_unknown()), Some(None));
        for (value, kind) in [
            (Value::Float(0.5), "FLOAT"),
            (Value::text("x"), "TEXT"),
            (Value::text(""), "TEXT"),
        ] {
            let err = Error::from(condition_truth(&value).unwrap_err());
            assert_eq!(
                err.to_string(),
                format!("type error: a condition must be BOOLEAN or INTEGER, got {kind}")
            );
        }
    }
}
