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
#[inline(always)]
pub fn sql_compare(a: &Value, b: &Value) -> Option<Ordering> {
    if let Some(ordering) = a.plain_cmp(b) {
        return Some(ordering);
    }
    // An INTEGER against a FLOAT orders by exact value, as `compare_across`
    // would answer it, without the checks for the other pairs
    if let (Value::Integer(_), Value::Float(_)) | (Value::Float(_), Value::Integer(_)) = (a, b) {
        return compare_numbers(a, b);
    }
    compare_across(a, b)
}

/// `sql_compare` for a pair that is not of one plain type, kept out of line
/// so the same-type answer costs no more than the comparison itself
#[inline(never)]
fn compare_across(a: &Value, b: &Value) -> Option<Ordering> {
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
    let read = |text: &str, like: &Value| match TextReading::of(like) {
        Some(reading @ (TextReading::Boolean | TextReading::Timestamp)) => Some(reading.read(text)),
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
#[inline(always)]
pub fn sql_eq(a: &Value, b: &Value) -> Option<bool> {
    if let Some(eq) = a.plain_eq(b) {
        return Some(eq);
    }
    // An INTEGER and a FLOAT are equal by exact value, as `eq_across` would
    // answer them
    if let (Value::Integer(_), Value::Float(_)) | (Value::Float(_), Value::Integer(_)) = (a, b) {
        return compare_numbers(a, b).map(Ordering::is_eq);
    }
    eq_across(a, b)
}

/// `sql_eq` for a pair that is not of one plain type, kept out of line
#[inline(never)]
fn eq_across(a: &Value, b: &Value) -> Option<bool> {
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

/// The kinds of the values a column or set holds, as one bit each
pub mod kind {
    use super::super::DataType;
    use super::Value;

    pub const INTEGER: u8 = 1;
    pub const FLOAT: u8 = 2;
    pub const TEXT: u8 = 4;
    pub const BOOLEAN: u8 = 8;
    pub const TIMESTAMP: u8 = 16;
    pub const JSON: u8 = 32;
    pub const VECTOR: u8 = 64;
    /// Any other kind
    pub const OTHER: u8 = 128;
    /// The kinds a value of which is equal only to a value of its own kind
    pub const EXTENSIONS: u8 = JSON | VECTOR | OTHER;
    /// The kinds a TEXT is read as when compared with them
    pub const READ_FROM_TEXT: u8 = INTEGER | FLOAT | BOOLEAN | TIMESTAMP;

    /// The kind of a value; 0 for a NULL
    pub fn of(value: &Value) -> u8 {
        match value {
            Value::Null(_) => 0,
            Value::Integer(_) => INTEGER,
            Value::Float(_) => FLOAT,
            Value::Text(_) => TEXT,
            Value::Boolean(_) => BOOLEAN,
            Value::Timestamp(_) => TIMESTAMP,
            other => of_type(other.data_type()),
        }
    }

    /// The kind of the values of a column of `data_type`
    pub fn of_type(data_type: DataType) -> u8 {
        match data_type {
            DataType::Null => 0,
            DataType::Integer => INTEGER,
            DataType::Float => FLOAT,
            DataType::Text => TEXT,
            DataType::Boolean => BOOLEAN,
            DataType::Timestamp => TIMESTAMP,
            DataType::Json => JSON,
            DataType::Vector => VECTOR,
        }
    }

    /// Whether a value of kinds `a` and one of kinds `b` can meet as a TEXT
    /// the comparison reads as the other side's type, which no key equality follows
    pub fn crosses(a: u8, b: u8) -> bool {
        (a & TEXT != 0 && b & READ_FROM_TEXT != 0) || (b & TEXT != 0 && a & READ_FROM_TEXT != 0)
    }
}

/// A type a comparison reads a TEXT as, the other side being of that type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextReading {
    Number,
    Boolean,
    Timestamp,
}

impl TextReading {
    /// The reading a TEXT gets against `value`; None for any other value
    pub fn of(value: &Value) -> Option<Self> {
        match value {
            Value::Integer(_) | Value::Float(_) => Some(Self::Number),
            Value::Boolean(_) => Some(Self::Boolean),
            Value::Timestamp(_) => Some(Self::Timestamp),
            _ => None,
        }
    }

    /// `text` read as this type; None when it does not read
    pub fn read(self, text: &str) -> Option<Value> {
        match self {
            Self::Number => parse_sql_number(text),
            Self::Boolean => super::value::text_as_boolean(text).map(Value::Boolean),
            Self::Timestamp => {
                super::value::parse_timestamp_for_compare(text).map(Value::Timestamp)
            }
        }
    }
}

/// The side of a cross a key column of one kind is on: TEXT, or a type a
/// TEXT key is read as
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeySide {
    Text,
    Read(TextReading),
}

impl KeySide {
    /// The side of a column holding `kinds` when they are all on one side
    pub fn of_kinds(kinds: u8) -> Option<Self> {
        match kinds {
            kind::TEXT => Some(Self::Text),
            kind::BOOLEAN => Some(Self::Read(TextReading::Boolean)),
            kind::TIMESTAMP => Some(Self::Read(TextReading::Timestamp)),
            k if k != 0 && k & !(kind::INTEGER | kind::FLOAT) == 0 => {
                Some(Self::Read(TextReading::Number))
            }
            _ => None,
        }
    }

    /// Whether `key` meets a column on this side as a TEXT read as the other
    /// side's type
    pub fn crosses(self, key: &Value) -> bool {
        match self {
            Self::Text => TextReading::of(key).is_some(),
            Self::Read(_) => matches!(key, Value::Text(_)),
        }
    }
}

/// The values the TEXT members of a set read as, for one reading
#[derive(Debug, Clone, Default)]
struct TextReads {
    values: ValueSet,
    /// A TEXT member that does not read: UNKNOWN against any probe of that type
    unreadable: bool,
}

/// The members of a constant `IN` list or set, kept for the predicate
/// comparison: a number, BOOLEAN or TIMESTAMP finds the TEXT members that read
/// as it, a TEXT finds the members it reads as, without a scan
#[derive(Debug, Clone)]
pub struct PredicateSet {
    members: CompactArc<ValueSet>,
    /// The numbers the TEXT members read as, None without one; shared, as
    /// the readings are, by a clone
    numbers_in_text: Option<CompactArc<ValueSet>>,
    /// The BOOLEAN and TIMESTAMP readings of the TEXT members, taken when a
    /// probe of that type first meets them
    text_reads: [std::sync::OnceLock<CompactArc<TextReads>>; 2],
    has_null: bool,
    has_numbers: bool,
    /// A member that is neither TEXT nor NULL
    has_non_text: bool,
    /// A TEXT member that is not a number: UNKNOWN against any number
    has_unreadable_text: bool,
    /// The kinds of the members that are not NULL, as `kind::INTEGER` and the others
    kinds: u8,
}

impl PredicateSet {
    pub fn new(members: CompactArc<ValueSet>, has_null: bool) -> Self {
        let mut numbers_in_text = ValueSet::default();
        let mut has_numbers = false;
        let mut has_non_text = false;
        let mut has_unreadable_text = false;
        let mut kinds = 0;
        for member in members.iter() {
            kinds |= kind::of(member);
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
            numbers_in_text: (!numbers_in_text.is_empty())
                .then(|| CompactArc::new(numbers_in_text)),
            text_reads: Default::default(),
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
        self.has_unreadable_text || self.numbers_in_text.is_some()
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
    #[inline]
    pub fn contains(&self, probe: &Value) -> Option<bool> {
        if self.members.is_empty() && !self.has_null {
            return Some(false);
        }
        if probe.is_null() {
            return None;
        }
        if self.members.contains(probe) {
            return Some(true);
        }
        // Members all of the probe's kind meet it only as the lookup found
        if !self.has_null && self.kinds == kind::of(probe) {
            return Some(false);
        }
        self.missed(probe)
    }

    /// `contains` for a probe the lookup did not find, among members of
    /// other kinds or a NULL
    #[inline(never)]
    fn missed(&self, probe: &Value) -> Option<bool> {
        // A JSON or VECTOR member is UNKNOWN against any other kind
        let mut unknown = self.has_null || self.kinds & kind::EXTENSIONS != 0;
        let read_kinds = kind::BOOLEAN | kind::TIMESTAMP;
        match probe {
            Value::Integer(_) | Value::Float(_) => {
                if self
                    .numbers_in_text
                    .as_ref()
                    .is_some_and(|numbers| numbers.contains(probe))
                {
                    return Some(true);
                }
                unknown |= self.has_unreadable_text;
            }
            Value::Boolean(_) | Value::Timestamp(_) if self.kinds & kind::TEXT != 0 => {
                let reads = self.text_reads(matches!(probe, Value::Boolean(_)));
                if reads.values.contains(probe) {
                    return Some(true);
                }
                unknown |= reads.unreadable;
            }
            Value::Text(text) if self.has_numbers || self.kinds & read_kinds != 0 => {
                if self.has_numbers {
                    match parse_sql_number(text) {
                        Some(number) if self.members.contains(&number) => return Some(true),
                        Some(_) => {}
                        None => unknown = true,
                    }
                }
                for (bit, reading) in [
                    (kind::BOOLEAN, TextReading::Boolean),
                    (kind::TIMESTAMP, TextReading::Timestamp),
                ] {
                    if self.kinds & bit != 0 {
                        match reading.read(text) {
                            Some(read) if self.members.contains(&read) => return Some(true),
                            Some(_) => {}
                            None => unknown = true,
                        }
                    }
                }
            }
            // Equal only to a member of its own kind, which the lookup finds
            Value::Extension(_) => {
                unknown = self.has_null || self.kinds & !kind::of(probe) != 0;
            }
            _ => {}
        }
        if unknown {
            None
        } else {
            Some(false)
        }
    }

    /// The readings of the TEXT members as BOOLEAN (`boolean`) or TIMESTAMP,
    /// taken on the first call for that type
    fn text_reads(&self, boolean: bool) -> &TextReads {
        let (slot, reading) = if boolean {
            (&self.text_reads[0], TextReading::Boolean)
        } else {
            (&self.text_reads[1], TextReading::Timestamp)
        };
        slot.get_or_init(|| {
            let mut reads = TextReads::default();
            for member in self.members.iter() {
                if let Value::Text(text) = member {
                    match reading.read(text) {
                        Some(read) => {
                            reads.values.insert(read);
                        }
                        None => reads.unreadable = true,
                    }
                }
            }
            CompactArc::new(reads)
        })
    }
}
