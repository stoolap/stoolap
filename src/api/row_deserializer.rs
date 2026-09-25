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

//! Deserialize result rows into any type that implements `serde::Deserialize`.
//!
//! A row is offered to serde as a map from column name to value, so a struct
//! that derives `Deserialize` fills its fields by column name, and as a
//! sequence, so a tuple or a `Vec` takes the columns by position. Column
//! values map to serde's data model as follows:
//!
//! | Stoolap value | serde |
//! |---|---|
//! | `INTEGER` | integer (any width that holds the value) |
//! | `FLOAT` | `f64` (an `INTEGER` also deserializes into a float) |
//! | `TEXT` | string, or a unit enum variant named by the text |
//! | `BOOLEAN` | bool |
//! | `TIMESTAMP` | owned RFC 3339 string, so `chrono::DateTime<Utc>` deserializes directly |
//! | `JSON` | the parsed document, so nested structs, maps and `serde_json::Value` work; a `String` gets the raw text; the document `null` is `None` for an `Option` |
//! | `VECTOR(n)` | sequence of exactly `n` `f32` |
//! | `NULL` | `None` for an `Option`, otherwise an error |
//!
//! Columns the target type does not name are ignored; a field with no column
//! is an error unless serde has a default for it. A single-column row also
//! deserializes into one bare value (an integer, a `String`, an `Option`, a
//! newtype, an enum), so `query_as_serde::<i64>("SELECT COUNT(*) FROM t", ())`
//! works. A `Vec` or a tuple always takes the columns by position, so a
//! vector column on its own is read as `(Vec<f32>,)`.
//!
//! ```
//! use serde::Deserialize;
//! use stoolap::Database;
//!
//! #[derive(Deserialize)]
//! struct User {
//!     id: i64,
//!     name: String,
//!     email: Option<String>,
//! }
//!
//! # fn main() -> stoolap::Result<()> {
//! let db = Database::open_in_memory()?;
//! db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)", ())?;
//! db.execute("INSERT INTO users VALUES (1, 'Alice', NULL)", ())?;
//!
//! let users: Vec<User> = db.query_as_serde("SELECT id, name, email FROM users", ())?;
//! assert_eq!(users[0].name, "Alice");
//! assert!(users[0].email.is_none());
//!
//! for row in db.query("SELECT id, name FROM users", ())? {
//!     let (id, name): (i64, String) = row?.deserialize()?;
//!     assert_eq!((id, name.as_str()), (1, "Alice"));
//! }
//! # Ok(())
//! # }
//! ```

use std::fmt;

use serde::de::{
    self, DeserializeOwned, DeserializeSeed, Deserializer, IntoDeserializer, MapAccess, SeqAccess,
    Visitor,
};
use serde::forward_to_deserialize_any;

use super::database::Database;
use super::params::{NamedParams, Params};
use super::rows::{ResultRow, Rows};
use crate::core::{DataType, Error, Result, Value};

/// The error serde's visitors report while a row is deserialized. It becomes
/// an [`Error::InvalidArgument`] naming the target type at the API boundary.
#[derive(Debug)]
pub struct RowDeserializeError(String);

impl fmt::Display for RowDeserializeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RowDeserializeError {}

impl de::Error for RowDeserializeError {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        RowDeserializeError(msg.to_string())
    }
}

impl From<serde_json::Error> for RowDeserializeError {
    fn from(e: serde_json::Error) -> Self {
        RowDeserializeError(format!("in JSON column: {}", e))
    }
}

fn into_error<T>(e: RowDeserializeError) -> Error {
    Error::invalid_argument(format!(
        "cannot deserialize row into {}: {}",
        std::any::type_name::<T>(),
        e.0
    ))
}

impl ResultRow {
    /// Deserialize this row into any `serde::Deserialize` type.
    ///
    /// A struct is filled by column name, a tuple or sequence by position.
    /// Text columns can be borrowed: `T` may hold `&str` fields that live as
    /// long as the row. See the [module documentation](self) for how each
    /// column type maps to serde's data model.
    pub fn deserialize<'a, T: de::Deserialize<'a>>(&'a self) -> Result<T> {
        T::deserialize(RowDeserializer { row: self }).map_err(into_error::<T>)
    }
}

impl Rows {
    /// Deserialize every remaining row into `T`, keeping the rows' order and
    /// surfacing query errors in place.
    pub fn deserialize<T: DeserializeOwned>(self) -> impl Iterator<Item = Result<T>> {
        self.map(|row| row.and_then(|row| row.deserialize()))
    }
}

impl Database {
    /// Execute a query and deserialize every row into `T` with serde.
    ///
    /// ```
    /// # use serde::Deserialize;
    /// # use stoolap::Database;
    /// #[derive(Deserialize)]
    /// struct Total {
    ///     n: i64,
    /// }
    /// # fn main() -> stoolap::Result<()> {
    /// let db = Database::open_in_memory()?;
    /// let totals: Vec<Total> = db.query_as_serde("SELECT 3 AS n", ())?;
    /// assert_eq!(totals[0].n, 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn query_as_serde<T: DeserializeOwned, P: Params>(
        &self,
        sql: &str,
        params: P,
    ) -> Result<Vec<T>> {
        self.query(sql, params)?.deserialize().collect()
    }

    /// Execute a query with named parameters and deserialize every row into
    /// `T` with serde.
    pub fn query_as_serde_named<T: DeserializeOwned>(
        &self,
        sql: &str,
        params: NamedParams,
    ) -> Result<Vec<T>> {
        self.query_named(sql, params)?.deserialize().collect()
    }
}

/// Offers a row as a map keyed by column name, or as a sequence of its values.
struct RowDeserializer<'a> {
    row: &'a ResultRow,
}

impl<'a> RowDeserializer<'a> {
    /// The only column of a single-column row, for a target that is one value
    /// (an integer, a `String`, a newtype, an enum) rather than a struct or tuple.
    fn single_value(self) -> std::result::Result<ValueDeserializer<'a>, RowDeserializeError> {
        match self.row.get_value(0) {
            Some(value) if self.row.len() == 1 => Ok(ValueDeserializer { value }),
            _ => Err(<RowDeserializeError as de::Error>::custom(format_args!(
                "a row of {} columns is not a single value; deserialize it into a tuple or a struct",
                self.row.len()
            ))),
        }
    }
}

/// Routes a request for one scalar value to the row's only column.
macro_rules! single_value_forward {
    ($($method:ident)*) => {
        $(
            fn $method<V: Visitor<'a>>(
                self,
                visitor: V,
            ) -> std::result::Result<V::Value, Self::Error> {
                self.single_value()?.$method(visitor)
            }
        )*
    };
}

impl<'a> Deserializer<'a> for RowDeserializer<'a> {
    type Error = RowDeserializeError;

    fn deserialize_any<V: Visitor<'a>>(
        self,
        visitor: V,
    ) -> std::result::Result<V::Value, Self::Error> {
        visitor.visit_map(RowMapAccess {
            row: self.row,
            index: 0,
        })
    }

    fn deserialize_seq<V: Visitor<'a>>(
        self,
        visitor: V,
    ) -> std::result::Result<V::Value, Self::Error> {
        visitor.visit_seq(RowSeqAccess {
            row: self.row,
            index: 0,
        })
    }

    fn deserialize_tuple<V: Visitor<'a>>(
        self,
        _len: usize,
        visitor: V,
    ) -> std::result::Result<V::Value, Self::Error> {
        self.deserialize_seq(visitor)
    }

    fn deserialize_tuple_struct<V: Visitor<'a>>(
        self,
        _name: &'static str,
        _len: usize,
        visitor: V,
    ) -> std::result::Result<V::Value, Self::Error> {
        self.deserialize_seq(visitor)
    }

    fn deserialize_newtype_struct<V: Visitor<'a>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> std::result::Result<V::Value, Self::Error> {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_option<V: Visitor<'a>>(
        self,
        visitor: V,
    ) -> std::result::Result<V::Value, Self::Error> {
        match self.row.get_value(0) {
            // A lone NULL column is an absent value; anything else is a row
            Some(value) if self.row.len() == 1 && ValueDeserializer { value }.is_absent() => {
                visitor.visit_none()
            }
            _ => visitor.visit_some(self),
        }
    }

    fn deserialize_enum<V: Visitor<'a>>(
        self,
        name: &'static str,
        variants: &'static [&'static str],
        visitor: V,
    ) -> std::result::Result<V::Value, Self::Error> {
        self.single_value()?
            .deserialize_enum(name, variants, visitor)
    }

    fn deserialize_unit_struct<V: Visitor<'a>>(
        self,
        name: &'static str,
        visitor: V,
    ) -> std::result::Result<V::Value, Self::Error> {
        self.single_value()?.deserialize_unit_struct(name, visitor)
    }

    fn deserialize_ignored_any<V: Visitor<'a>>(
        self,
        visitor: V,
    ) -> std::result::Result<V::Value, Self::Error> {
        visitor.visit_unit()
    }

    single_value_forward! {
        deserialize_bool deserialize_i8 deserialize_i16 deserialize_i32 deserialize_i64
        deserialize_i128 deserialize_u8 deserialize_u16 deserialize_u32 deserialize_u64
        deserialize_u128 deserialize_f32 deserialize_f64 deserialize_char deserialize_str
        deserialize_string deserialize_bytes deserialize_byte_buf deserialize_unit
    }

    forward_to_deserialize_any! {
        <V: Visitor<'a>>
        map struct identifier
    }
}

struct RowMapAccess<'a> {
    row: &'a ResultRow,
    index: usize,
}

impl<'a> MapAccess<'a> for RowMapAccess<'a> {
    type Error = RowDeserializeError;

    fn next_key_seed<K: DeserializeSeed<'a>>(
        &mut self,
        seed: K,
    ) -> std::result::Result<Option<K::Value>, Self::Error> {
        match self.row.columns().get(self.index) {
            Some(name) => seed
                .deserialize(de::value::BorrowedStrDeserializer::new(name))
                .map(Some),
            None => Ok(None),
        }
    }

    fn next_value_seed<V: DeserializeSeed<'a>>(
        &mut self,
        seed: V,
    ) -> std::result::Result<V::Value, Self::Error> {
        let value = self
            .row
            .get_value(self.index)
            .ok_or_else(|| <RowDeserializeError as de::Error>::custom("column without a value"))?;
        self.index += 1;
        seed.deserialize(ValueDeserializer { value })
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.row.len().saturating_sub(self.index))
    }
}

struct RowSeqAccess<'a> {
    row: &'a ResultRow,
    index: usize,
}

impl<'a> SeqAccess<'a> for RowSeqAccess<'a> {
    type Error = RowDeserializeError;

    fn next_element_seed<T: DeserializeSeed<'a>>(
        &mut self,
        seed: T,
    ) -> std::result::Result<Option<T::Value>, Self::Error> {
        match self.row.get_value(self.index) {
            Some(value) => {
                self.index += 1;
                seed.deserialize(ValueDeserializer { value }).map(Some)
            }
            None => Ok(None),
        }
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.row.len().saturating_sub(self.index))
    }
}

/// Offers one column value in serde's data model.
struct ValueDeserializer<'a> {
    value: &'a Value,
}

impl<'a> ValueDeserializer<'a> {
    fn mismatch(&self, expected: &dyn de::Expected) -> RowDeserializeError {
        <RowDeserializeError as de::Error>::invalid_type(self.unexpected(), expected)
    }

    fn unexpected(&self) -> de::Unexpected<'_> {
        match self.value {
            Value::Null(_) => de::Unexpected::Other("null"),
            Value::Integer(i) => de::Unexpected::Signed(*i),
            Value::Float(f) => de::Unexpected::Float(*f),
            Value::Text(s) => de::Unexpected::Str(s.as_str()),
            Value::Boolean(b) => de::Unexpected::Bool(*b),
            Value::Timestamp(_) => de::Unexpected::Other("timestamp"),
            Value::Extension(_) => match self.value.data_type() {
                DataType::Json => de::Unexpected::Other("JSON"),
                DataType::Vector => de::Unexpected::Other("vector"),
                _ => de::Unexpected::Other("extension value"),
            },
        }
    }

    /// NULL, or a JSON column holding the document `null`.
    fn is_absent(&self) -> bool {
        matches!(self.value, Value::Null(_)) || self.json().map(str::trim) == Some("null")
    }

    fn json(&self) -> Option<&'a str> {
        match self.value {
            Value::Extension(_) if self.value.data_type() == DataType::Json => self.value.as_json(),
            _ => None,
        }
    }

    fn deserialize_json<V: Visitor<'a>>(
        json: &'a str,
        visitor: V,
    ) -> std::result::Result<V::Value, RowDeserializeError> {
        let mut de = serde_json::Deserializer::from_str(json);
        let value = de.deserialize_any(visitor)?;
        de.end()?;
        Ok(value)
    }
}

impl<'a> Deserializer<'a> for ValueDeserializer<'a> {
    type Error = RowDeserializeError;

    fn deserialize_any<V: Visitor<'a>>(
        self,
        visitor: V,
    ) -> std::result::Result<V::Value, Self::Error> {
        match self.value {
            Value::Null(_) => {
                // Only a unit-like target accepts NULL; when it is refused,
                // say "null" rather than serde's "unit value".
                let expected = (&visitor as &dyn de::Expected).to_string();
                visitor.visit_unit::<RowDeserializeError>().map_err(|_| {
                    <RowDeserializeError as de::Error>::custom(format_args!(
                        "invalid type: null, expected {}",
                        expected
                    ))
                })
            }
            Value::Integer(i) => visitor.visit_i64(*i),
            Value::Float(f) => visitor.visit_f64(*f),
            Value::Text(s) => visitor.visit_borrowed_str(s.as_str()),
            Value::Boolean(b) => visitor.visit_bool(*b),
            Value::Timestamp(ts) => visitor.visit_string(ts.to_rfc3339()),
            Value::Extension(_) => {
                if let Some(json) = self.json() {
                    Self::deserialize_json(json, visitor)
                } else if let Some(vector) = self.value.as_vector_f32() {
                    // Through the deserializer so that the length is checked
                    de::value::SeqDeserializer::new(vector.into_iter()).deserialize_any(visitor)
                } else {
                    Err(self.mismatch(&"a column value"))
                }
            }
        }
    }

    fn deserialize_option<V: Visitor<'a>>(
        self,
        visitor: V,
    ) -> std::result::Result<V::Value, Self::Error> {
        if self.is_absent() {
            visitor.visit_none()
        } else {
            visitor.visit_some(self)
        }
    }

    /// A JSON column read as a string yields the raw document text, as
    /// `FromValue for String` does, so parsing stays opt-in.
    fn deserialize_str<V: Visitor<'a>>(
        self,
        visitor: V,
    ) -> std::result::Result<V::Value, Self::Error> {
        match self.json() {
            Some(json) => visitor.visit_borrowed_str(json),
            None => self.deserialize_any(visitor),
        }
    }

    fn deserialize_string<V: Visitor<'a>>(
        self,
        visitor: V,
    ) -> std::result::Result<V::Value, Self::Error> {
        self.deserialize_str(visitor)
    }

    fn deserialize_unit<V: Visitor<'a>>(
        self,
        visitor: V,
    ) -> std::result::Result<V::Value, Self::Error> {
        match self.value {
            Value::Null(_) => visitor.visit_unit(),
            _ => Err(self.mismatch(&visitor)),
        }
    }

    fn deserialize_unit_struct<V: Visitor<'a>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> std::result::Result<V::Value, Self::Error> {
        self.deserialize_unit(visitor)
    }

    fn deserialize_newtype_struct<V: Visitor<'a>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> std::result::Result<V::Value, Self::Error> {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_enum<V: Visitor<'a>>(
        self,
        name: &'static str,
        variants: &'static [&'static str],
        visitor: V,
    ) -> std::result::Result<V::Value, Self::Error> {
        match self.value {
            // A text column names a unit variant
            Value::Text(s) => visitor.visit_enum(s.as_str().into_deserializer()),
            _ => {
                if let Some(json) = self.json() {
                    let mut de = serde_json::Deserializer::from_str(json);
                    let value = de.deserialize_enum(name, variants, visitor)?;
                    de.end()?;
                    Ok(value)
                } else {
                    Err(self.mismatch(&visitor))
                }
            }
        }
    }

    fn deserialize_ignored_any<V: Visitor<'a>>(
        self,
        visitor: V,
    ) -> std::result::Result<V::Value, Self::Error> {
        visitor.visit_unit()
    }

    forward_to_deserialize_any! {
        <V: Visitor<'a>>
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char bytes
        byte_buf seq tuple tuple_struct map struct identifier
    }
}
