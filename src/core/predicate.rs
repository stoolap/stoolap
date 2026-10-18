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

//! The comparison an SQL predicate makes between two values. It is kept
//! apart from `Value`'s equality, hash and ordering: a number against a TEXT
//! reads the TEXT as a number, so `'01' = 1` and `1 = '1'` hold while
//! `'01' = '1'` does not, which no key equality can follow.

use std::cmp::Ordering;

use super::value::{compare_floats, total_cmp_i64_f64, Value};
use super::ValueSet;
use crate::common::CompactArc;

/// A TEXT read as a number by a comparison: `[+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?`.
/// An INTEGER when it has no fraction or exponent and fits i64, a FLOAT
/// otherwise; None when it is not of that form or the FLOAT is not finite
pub fn parse_sql_number(text: &str) -> Option<Value> {
    let bytes = text.as_bytes();
    let digits = |from: usize| {
        bytes[from..]
            .iter()
            .take_while(|b| b.is_ascii_digit())
            .count()
    };
    let mut at = usize::from(matches!(bytes.first(), Some(b'+' | b'-')));
    let whole = digits(at);
    if whole == 0 {
        return None;
    }
    at += whole;
    let mut integral = true;
    if bytes.get(at) == Some(&b'.') {
        let fraction = digits(at + 1);
        if fraction == 0 {
            return None;
        }
        at += 1 + fraction;
        integral = false;
    }
    if matches!(bytes.get(at), Some(b'e' | b'E')) {
        at += 1;
        if matches!(bytes.get(at), Some(b'+' | b'-')) {
            at += 1;
        }
        let exponent = digits(at);
        if exponent == 0 {
            return None;
        }
        at += exponent;
        integral = false;
    }
    if at != bytes.len() {
        return None;
    }
    if integral {
        if let Ok(i) = text.parse::<i64>() {
            return Some(Value::Integer(i));
        }
    }
    let f: f64 = text.parse().ok()?;
    f.is_finite().then_some(Value::Float(f))
}

fn is_number(value: &Value) -> bool {
    matches!(value, Value::Integer(_) | Value::Float(_))
}

/// Two numbers ordered exactly: an INTEGER is never rounded to f64 first
fn compare_numbers(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => Some(x.cmp(y)),
        (Value::Float(x), Value::Float(y)) => Some(compare_floats(*x, *y)),
        (Value::Integer(i), Value::Float(f)) => Some(total_cmp_i64_f64(*i, *f)),
        (Value::Float(f), Value::Integer(i)) => Some(total_cmp_i64_f64(*i, *f).reverse()),
        _ => None,
    }
}

/// The number a comparison reads from each side, when one side is a number
/// and the other a TEXT; None for any other pair. The inner None is a TEXT
/// that is not a number
pub fn number_against_text(a: &Value, b: &Value) -> Option<Option<Ordering>> {
    match (a, b) {
        (Value::Text(text), number) if is_number(number) => {
            Some(parse_sql_number(text).and_then(|parsed| compare_numbers(&parsed, number)))
        }
        (number, Value::Text(text)) if is_number(number) => {
            Some(parse_sql_number(text).and_then(|parsed| compare_numbers(number, &parsed)))
        }
        _ => None,
    }
}

/// How `a` orders against `b` in a predicate; None is UNKNOWN: a NULL, a
/// TEXT that does not read as the other side's type, a BOOLEAN or TIMESTAMP
/// against a number, a JSON or VECTOR, or two values of other domains
pub fn sql_compare(a: &Value, b: &Value) -> Option<Ordering> {
    if a.is_null() || b.is_null() || non_number_against_number(a, b) {
        return None;
    }
    if let Some(ordering) = number_against_text(a, b) {
        return ordering;
    }
    let (a, b) = read_text_across(a, b)?;
    match (a.as_ref(), b.as_ref()) {
        (Value::Extension(_), _) | (_, Value::Extension(_)) => None,
        (a, b) if a.data_type() == b.data_type() || (is_number(a) && is_number(b)) => {
            a.partial_cmp(b)
        }
        _ => None,
    }
}

/// A BOOLEAN or TIMESTAMP on one side and a number on the other: no order
fn non_number_against_number(a: &Value, b: &Value) -> bool {
    let other = |v: &Value| matches!(v, Value::Boolean(_) | Value::Timestamp(_));
    (is_number(a) && other(b)) || (other(a) && is_number(b))
}

/// The pair with a TEXT read as the other side's type when that is a
/// BOOLEAN, by the CAST rule, or a TIMESTAMP; None when the TEXT does not
/// read. Any other pair as it is
fn read_text_across<'v>(
    a: &'v Value,
    b: &'v Value,
) -> Option<(std::borrow::Cow<'v, Value>, std::borrow::Cow<'v, Value>)> {
    use std::borrow::Cow;
    let read = |text: &str, like: &Value| match like {
        Value::Boolean(_) => Some(super::value::text_as_boolean(text).map(Value::Boolean)),
        Value::Timestamp(_) => {
            Some(super::value::parse_timestamp_for_compare(text).map(Value::Timestamp))
        }
        _ => None,
    };
    match (a, b) {
        (Value::Text(text), other) => match read(text, other) {
            Some(read) => read.map(|a| (Cow::Owned(a), Cow::Borrowed(b))),
            None => Some((Cow::Borrowed(a), Cow::Borrowed(b))),
        },
        (other, Value::Text(text)) => match read(text, other) {
            Some(read) => read.map(|b| (Cow::Borrowed(a), Cow::Owned(b))),
            None => Some((Cow::Borrowed(a), Cow::Borrowed(b))),
        },
        _ => Some((Cow::Borrowed(a), Cow::Borrowed(b))),
    }
}

/// `a = b` in a predicate; None is UNKNOWN: a NULL, a TEXT that does not
/// read as the other side's type, or a JSON or VECTOR against another type.
/// Two values of other domains, a BOOLEAN or TIMESTAMP against a number
/// among them, are not equal
pub fn sql_eq(a: &Value, b: &Value) -> Option<bool> {
    if a.is_null() || b.is_null() {
        return None;
    }
    if let Some(ordering) = number_against_text(a, b) {
        return ordering.map(|o| o == Ordering::Equal);
    }
    let (a, b) = read_text_across(a, b)?;
    match (a.as_ref(), b.as_ref()) {
        (Value::Extension(x), Value::Extension(y)) if x.first() == y.first() => Some(x == y),
        (Value::Extension(_), _) | (_, Value::Extension(_)) => None,
        (a, b) => Some(a == b),
    }
}

/// `a IS DISTINCT FROM b`: two NULLs are not distinct, a NULL and a value
/// are, and two values are unless `=` holds between them
pub fn sql_distinct(a: &Value, b: &Value) -> bool {
    match (a.is_null(), b.is_null()) {
        (true, true) => false,
        (true, false) | (false, true) => true,
        (false, false) => sql_eq(a, b) != Some(true),
    }
}

/// The members of a constant `IN` list or set, kept for the predicate
/// comparison: a number finds the TEXT members that read as it, a TEXT finds
/// the numeric members it reads as, without a scan
#[derive(Debug, Clone)]
pub struct PredicateSet {
    members: CompactArc<ValueSet>,
    /// The numbers the TEXT members read as
    numbers_in_text: ValueSet,
    has_null: bool,
    has_numbers: bool,
    /// A member that is neither TEXT nor NULL
    has_non_text: bool,
    /// A TEXT member that is not a number: UNKNOWN against any number
    has_unreadable_text: bool,
    /// The kinds of the members that are not NULL, as `PredicateSet::INTEGER` and the others
    kinds: u8,
}

impl PredicateSet {
    pub const INTEGER: u8 = 1;
    pub const FLOAT: u8 = 2;
    pub const TEXT: u8 = 4;
    pub const BOOLEAN: u8 = 8;
    pub const TIMESTAMP: u8 = 16;
    /// Any other kind: JSON, VECTOR and the like
    pub const OTHER: u8 = 32;

    pub fn new(members: CompactArc<ValueSet>, has_null: bool) -> Self {
        let mut numbers_in_text = ValueSet::default();
        let mut has_numbers = false;
        let mut has_non_text = false;
        let mut has_unreadable_text = false;
        let mut kinds = 0;
        for member in members.iter() {
            kinds |= match member {
                Value::Null(_) => 0,
                Value::Integer(_) => Self::INTEGER,
                Value::Float(_) => Self::FLOAT,
                Value::Text(_) => Self::TEXT,
                Value::Boolean(_) => Self::BOOLEAN,
                Value::Timestamp(_) => Self::TIMESTAMP,
                _ => Self::OTHER,
            };
            match member {
                Value::Integer(_) | Value::Float(_) => has_numbers = true,
                Value::Text(text) => match parse_sql_number(text) {
                    Some(number) => {
                        numbers_in_text.insert(number);
                    }
                    None => has_unreadable_text = true,
                },
                _ => {}
            }
            has_non_text |= !matches!(member, Value::Text(_) | Value::Null(_));
        }
        Self {
            members,
            numbers_in_text,
            has_null,
            has_numbers,
            has_non_text,
            has_unreadable_text,
            kinds,
        }
    }

    /// The kinds of the members that are not NULL
    pub fn kinds(&self) -> u8 {
        self.kinds
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// The set of `members`, with a NULL among them as its NULL member
    pub fn of(members: CompactArc<ValueSet>) -> Self {
        let has_null = members.contains(&super::value::NULL_VALUE);
        Self::new(members, has_null)
    }

    pub fn has_null(&self) -> bool {
        self.has_null
    }

    /// Whether a member is a TEXT
    pub fn has_text(&self) -> bool {
        self.has_unreadable_text || !self.numbers_in_text.is_empty()
    }

    /// Whether a member is neither TEXT nor NULL
    pub fn has_non_text(&self) -> bool {
        self.has_non_text
    }

    /// The members as stored, for callers that read them as values
    pub fn members(&self) -> &CompactArc<ValueSet> {
        &self.members
    }

    /// `probe IN (members)`; None is UNKNOWN. A set with no member at all,
    /// NULL included, holds nothing, so it is false even for a NULL probe
    pub fn contains(&self, probe: &Value) -> Option<bool> {
        if self.members.is_empty() && !self.has_null {
            return Some(false);
        }
        if probe.is_null() {
            return None;
        }
        if self.reads_across(probe) {
            return self.scan(probe);
        }
        if self.members.contains(probe) {
            return Some(true);
        }
        let mut unknown = self.has_null;
        match probe {
            Value::Integer(_) | Value::Float(_) => {
                if self.numbers_in_text.contains(probe) {
                    return Some(true);
                }
                unknown |= self.has_unreadable_text;
            }
            Value::Text(text) if self.has_numbers => match parse_sql_number(text) {
                Some(number) if self.members.contains(&number) => return Some(true),
                Some(_) => {}
                None => unknown = true,
            },
            _ => {}
        }
        if unknown {
            None
        } else {
            Some(false)
        }
    }

    /// Whether `probe` meets a member that `=` reads as another type, a
    /// BOOLEAN or TIMESTAMP against a TEXT, or a JSON or VECTOR on either side
    fn reads_across(&self, probe: &Value) -> bool {
        let text_reads = Self::BOOLEAN | Self::TIMESTAMP;
        match probe {
            Value::Integer(_) | Value::Float(_) => self.kinds & Self::OTHER != 0,
            Value::Text(_) => self.kinds & (text_reads | Self::OTHER) != 0,
            Value::Boolean(_) | Value::Timestamp(_) => self.kinds & (Self::TEXT | Self::OTHER) != 0,
            _ => true,
        }
    }

    /// `probe IN (members)` by `=` against each member
    fn scan(&self, probe: &Value) -> Option<bool> {
        let mut unknown = self.has_null;
        for member in self.members.iter() {
            match sql_eq(probe, member) {
                Some(true) => return Some(true),
                Some(false) => {}
                None => unknown = true,
            }
        }
        if unknown {
            None
        } else {
            Some(false)
        }
    }
}
