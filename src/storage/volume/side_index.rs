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

use crate::core::value::cmp_i64_f64;
use crate::core::{DataType, IndexType, Operator, Value};

/// The value domain one key component is encoded in
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    Integer,
    Timestamp,
    Float,
    Boolean,
    Text,
}

impl Domain {
    /// The domain a column of `data_type` is keyed in, if it has one
    pub fn of(data_type: DataType) -> Option<Self> {
        match data_type {
            DataType::Integer => Some(Self::Integer),
            DataType::Timestamp => Some(Self::Timestamp),
            DataType::Float => Some(Self::Float),
            DataType::Boolean => Some(Self::Boolean),
            DataType::Text => Some(Self::Text),
            _ => None,
        }
    }

    /// The domain's byte in a version 6 directory entry
    pub fn code(self) -> u8 {
        match self {
            Self::Integer => 1,
            Self::Timestamp => 2,
            Self::Float => 3,
            Self::Boolean => 4,
            Self::Text => 5,
        }
    }

    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Integer),
            2 => Some(Self::Timestamp),
            3 => Some(Self::Float),
            4 => Some(Self::Boolean),
            5 => Some(Self::Text),
            _ => None,
        }
    }
}

/// How an entry stores its keys
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyKind {
    /// One i64 per distinct key, in order: a single INTEGER, TIMESTAMP,
    /// FLOAT or BOOLEAN component
    OrderedI64,
    /// One ordered byte string per distinct key: a TEXT component, or
    /// several components of any domain
    OrderedBytes,
}

impl KeyKind {
    /// The kind's byte in a version 6 directory entry
    pub fn code(self) -> u8 {
        match self {
            Self::OrderedI64 => 1,
            Self::OrderedBytes => 2,
        }
    }

    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::OrderedI64),
            2 => Some(Self::OrderedBytes),
            _ => None,
        }
    }
}

/// What the build of an entry came to, as the file records it
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The entry holds its keys and serves reads
    Built,
    /// The index cannot be held for this volume: no pages, never served,
    /// not built again until the volume or the definition changes
    Unsupported,
}

impl Outcome {
    pub fn code(self) -> u8 {
        match self {
            Self::Built => 0,
            Self::Unsupported => 1,
        }
    }

    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Built),
            1 => Some(Self::Unsupported),
            _ => None,
        }
    }
}

/// The i64 key of a FLOAT value, ordered as the floats are: 0.0 and -0.0
/// are one key. NaN has no key; the caller keeps it out
pub fn float_key(f: f64) -> i64 {
    debug_assert!(!f.is_nan());
    let bits = if f == 0.0 { 0 } else { f.to_bits() as i64 };
    if bits < 0 {
        bits ^ i64::MAX
    } else {
        bits
    }
}

/// The FLOAT value a key stands for; the key of -0.0 stands for -0.0
pub fn float_of_key(key: i64) -> f64 {
    let bits = if key < 0 { key ^ i64::MAX } else { key };
    f64::from_bits(bits as u64)
}

// Ordered bytes: each component a tag, then its value's bytes, so that the
// bytes of two keys compare as the keys do and a component ends where the
// next begins. NULL sorts first, NaN after every number
const TAG_NULL: u8 = 0x00;
const TAG_VALUE: u8 = 0x01;
const TAG_NAN: u8 = 0x02;
const SIGN: u64 = 1 << 63;

pub fn push_null(out: &mut Vec<u8>) {
    out.push(TAG_NULL);
}

/// An INTEGER or TIMESTAMP component, or a FLOAT one by its key
pub fn push_i64(out: &mut Vec<u8>, key: i64) {
    out.push(TAG_VALUE);
    out.extend_from_slice(&(key as u64 ^ SIGN).to_be_bytes());
}

pub fn push_f64(out: &mut Vec<u8>, f: f64) {
    if f.is_nan() {
        out.push(TAG_NAN);
    } else {
        push_i64(out, float_key(f));
    }
}

pub fn push_bool(out: &mut Vec<u8>, b: bool) {
    out.push(TAG_VALUE);
    out.push(b as u8);
}

/// A TEXT component: its bytes with each 0x00 escaped as 0x00 0xFF, ended
/// by 0x00 0x00, so a shorter string sorts before its extensions
pub fn push_text(out: &mut Vec<u8>, text: &[u8]) {
    out.push(TAG_VALUE);
    for &byte in text {
        out.push(byte);
        if byte == 0 {
            out.push(0xFF);
        }
    }
    out.extend_from_slice(&[0, 0]);
}

/// Whether `push_text` writes `text` within the longest key a key page
/// holds, found without encoding it
pub fn text_key_fits(text: &[u8]) -> bool {
    let limit = super::secondary::MAX_BYTE_KEY;
    text.len() + 3 <= limit && text.len() + 3 + text.iter().filter(|&&b| b == 0).count() <= limit
}

/// A TEXT key and after it the bound just above it, so that `[key, bound)`
/// holds that key alone; returns where the bound starts
pub fn push_text_range(out: &mut Vec<u8>, text: &[u8]) -> usize {
    out.clear();
    push_text(out, text);
    let split = out.len();
    out.extend_from_within(..split);
    // The key ends in its 0x00 0x00 terminator
    if let Some(last) = out.last_mut() {
        *last = 1;
    }
    split
}

/// A component holding the resolved key `key` of `domain`
pub fn push_key(out: &mut Vec<u8>, domain: Domain, key: i64) {
    match domain {
        Domain::Boolean => push_bool(out, key != 0),
        _ => push_i64(out, key),
    }
}

/// How a query value went into a key component
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoded {
    Pushed,
    /// No value of the domain equals it
    Empty,
    /// The keys cannot answer it
    Fallback,
}

/// Appends the component a `domain` column holds when `column = value`
pub fn push_equal(domain: Domain, value: &Value, out: &mut Vec<u8>) -> Encoded {
    if domain == Domain::Text {
        return match value {
            Value::Text(text) => {
                push_text(out, text.as_bytes());
                Encoded::Pushed
            }
            _ => Encoded::Fallback,
        };
    }
    match resolve(domain, Operator::Eq, value) {
        Resolved::Key(key) => {
            push_key(out, domain, key);
            Encoded::Pushed
        }
        Resolved::Empty => Encoded::Empty,
        _ => Encoded::Fallback,
    }
}

/// The bytes a `domain` component holding `value` takes, at most; zero for
/// a value the component cannot hold
pub fn encoded_len(domain: Domain, value: &Value) -> usize {
    match (domain, value) {
        (Domain::Text, Value::Text(text)) => {
            let bytes = text.as_bytes();
            3 + bytes.len() + bytes.iter().filter(|&&b| b == 0).count()
        }
        (Domain::Text, _) => 0,
        (Domain::Boolean, _) => 2,
        _ => 9,
    }
}

/// A key range `[lower, upper)` in one buffer: the lower bound, then the
/// upper one, absent when no key is above the range
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ByteRange {
    bytes: Vec<u8>,
    split: usize,
    bounded: bool,
}

impl ByteRange {
    pub fn lower(&self) -> &[u8] {
        &self.bytes[..self.split]
    }

    pub fn upper(&self) -> Option<&[u8]> {
        self.bounded.then(|| &self.bytes[self.split..])
    }

    /// A range no key falls in
    pub fn empty() -> Self {
        Self {
            bytes: vec![0xFF],
            split: 1,
            bounded: true,
        }
    }

    /// The keys that start with `prefix`, built in its own buffer, which
    /// holds twice its bytes when it is allocated so
    pub fn starting_with(mut prefix: Vec<u8>) -> Self {
        let split = prefix.len();
        prefix.extend_from_within(..split);
        let bounded = successor_from(&mut prefix, split);
        Self {
            bytes: prefix,
            split,
            bounded,
        }
    }
}

/// `successor` of the bytes past `from`, leaving those before it
fn successor_from(bytes: &mut Vec<u8>, from: usize) -> bool {
    while bytes.len() > from {
        let at = bytes.len() - 1;
        if bytes[at] < 0xFF {
            bytes[at] += 1;
            return true;
        }
        bytes.pop();
    }
    false
}

/// The keys that start with `prefix` and whose next component, of `domain`,
/// meets every comparison the keys can answer, built on the prefix's own
/// buffer. The prefix back when no comparison can be answered, an empty
/// range when they leave no value. NULL and NaN components fall outside
/// every range, as no comparison holds for them
pub fn component_range<'v>(
    domain: Domain,
    mut prefix: Vec<u8>,
    comparisons: impl Iterator<Item = (Operator, &'v Value)>,
) -> Result<ByteRange, Vec<u8>> {
    let at = prefix.len();
    // The range is chosen first and encoded once
    let (lower, upper): (Edge<'v>, Edge<'v>);
    if domain == Domain::Text {
        let mut low: Option<(&[u8], bool)> = None;
        let mut high: Option<(&[u8], bool)> = None;
        let mut bounded = false;
        for (op, value) in comparisons {
            let Value::Text(text) = value else {
                continue;
            };
            let text = text.as_bytes();
            match op {
                Operator::Gt | Operator::Gte => {
                    let bound = (text, op == Operator::Gt);
                    low = Some(low.map_or(bound, |l| l.max(bound)));
                }
                Operator::Lt | Operator::Lte => {
                    let bound = (text, op == Operator::Lte);
                    high = Some(high.map_or(bound, |h| h.min(bound)));
                }
                _ => continue,
            }
            bounded = true;
        }
        if !bounded {
            return Err(prefix);
        }
        lower = low.map(|(text, past)| (Bound::Text(text), past));
        upper = high.map(|(text, past)| (Bound::Text(text), past));
    } else {
        let (mut low, mut high) = match domain {
            Domain::Boolean => (0, 1),
            _ => (i64::MIN, i64::MAX),
        };
        let mut bounded = false;
        for (op, value) in comparisons {
            if op == Operator::Eq {
                continue;
            }
            let (l, h) = match resolve(domain, op, value) {
                Resolved::Key(k) => (k, k),
                Resolved::Range(l, h) => (l, h),
                Resolved::Empty => return Ok(ByteRange::empty()),
                Resolved::Fallback => continue,
            };
            bounded = true;
            low = low.max(l);
            high = high.min(h);
        }
        if !bounded {
            return Err(prefix);
        }
        if low > high {
            return Ok(ByteRange::empty());
        }
        lower = Some((Bound::Key(low), false));
        upper = Some((Bound::Key(high), true));
    }
    let push = |out: &mut Vec<u8>, bound: Option<(Bound<'_>, bool)>, empty: u8| -> bool {
        match bound {
            None => {
                out.push(empty);
                false
            }
            Some((Bound::Text(text), past)) => {
                push_text(out, text);
                past
            }
            Some((Bound::Key(key), past)) => {
                push_key(out, domain, key);
                past
            }
        }
    };
    // A missing bound is every value: from the first value tag to NaN
    if push(&mut prefix, lower, TAG_VALUE) {
        let past = successor_from(&mut prefix, at);
        debug_assert!(past, "a value's key has a successor");
    }
    let split = prefix.len();
    prefix.extend_from_within(..at);
    let bounded = !push(&mut prefix, upper, TAG_NAN) || successor_from(&mut prefix, split + at);
    Ok(ByteRange {
        bytes: prefix,
        split,
        bounded,
    })
}

/// A chosen range bound before it is encoded
#[derive(Clone, Copy)]
enum Bound<'v> {
    Text(&'v [u8]),
    Key(i64),
}

/// A range's end, if bounded: the bound, and whether the end lies just
/// above it, past every key equal to its value
type Edge<'v> = Option<(Bound<'v>, bool)>;

/// Turns `prefix` into the least byte string above every string it starts:
/// the last byte below 0xFF goes up by one and what follows it goes. False
/// when no such string exists, every byte being 0xFF
pub fn successor(prefix: &mut Vec<u8>) -> bool {
    while let Some(last) = prefix.last_mut() {
        if *last < 0xFF {
            *last += 1;
            return true;
        }
        prefix.pop();
    }
    false
}

/// What a comparison of a key column with a value asks of the entry
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolved {
    Key(i64),
    /// Every key in `[low, high]`
    Range(i64, i64),
    /// No row's value meets the comparison
    Empty,
    /// The keys cannot answer it; the scan does
    Fallback,
}

/// The keys of a `domain` column meeting `column op value`, decided with
/// the executor's exact comparison: a bound with no exact key in the domain
/// moves to the nearest key on the side the comparison keeps
pub fn resolve(domain: Domain, op: Operator, value: &Value) -> Resolved {
    let range = |low: Option<i64>, high: Option<i64>| match (low, high) {
        (Some(low), Some(high)) if low <= high => Resolved::Range(low, high),
        _ => Resolved::Empty,
    };
    // The keys meeting `key op k` for an exact key k
    let exact = |k: i64| match op {
        Operator::Eq => Resolved::Key(k),
        Operator::Lt => range(Some(i64::MIN), k.checked_sub(1)),
        Operator::Lte => range(Some(i64::MIN), Some(k)),
        Operator::Gt => range(k.checked_add(1), Some(i64::MAX)),
        Operator::Gte => range(Some(k), Some(i64::MAX)),
        _ => Resolved::Fallback,
    };
    match (domain, value) {
        (Domain::Integer, Value::Integer(i)) => exact(*i),
        (Domain::Integer, Value::Float(f)) => integer_against_float(op, *f),
        (Domain::Timestamp, Value::Timestamp(ts)) => match ts.timestamp_nanos_opt() {
            Some(nanos) => exact(nanos),
            None => Resolved::Fallback,
        },
        (Domain::Float, Value::Float(f)) if f.is_nan() => Resolved::Fallback,
        (Domain::Float, Value::Float(f)) => exact(float_key(*f)),
        (Domain::Float, Value::Integer(i)) => float_against_integer(op, *i),
        (Domain::Boolean, Value::Boolean(b)) => exact(*b as i64),
        _ => Resolved::Fallback,
    }
}

/// `int_column op f`: the integers meeting it, by `cmp_i64_f64`
fn integer_against_float(op: Operator, f: f64) -> Resolved {
    if f.is_nan() {
        return Resolved::Fallback;
    }
    // The least integer >= f and the greatest integer <= f, when they exist
    const TWO_63: f64 = 9_223_372_036_854_775_808.0;
    let ceil = if f < -TWO_63 {
        Some(i64::MIN)
    } else if f.ceil() >= TWO_63 {
        None
    } else {
        Some(f.ceil() as i64)
    };
    let floor = if f >= TWO_63 {
        Some(i64::MAX)
    } else if f < -TWO_63 {
        None
    } else {
        Some(f.floor() as i64)
    };
    let integral = ceil.zip(floor).is_some_and(|(c, l)| c == l);
    let range = |low: Option<i64>, high: Option<i64>| match (low, high) {
        (Some(low), Some(high)) if low <= high => Resolved::Range(low, high),
        _ => Resolved::Empty,
    };
    match op {
        Operator::Eq if integral => Resolved::Key(floor.unwrap_or_default()),
        Operator::Eq => Resolved::Empty,
        Operator::Lt if integral => range(Some(i64::MIN), floor.and_then(|k| k.checked_sub(1))),
        Operator::Lt | Operator::Lte => range(Some(i64::MIN), floor),
        Operator::Gt if integral => range(ceil.and_then(|k| k.checked_add(1)), Some(i64::MAX)),
        Operator::Gt | Operator::Gte => range(ceil, Some(i64::MAX)),
        _ => Resolved::Fallback,
    }
}

/// `float_column op i`: the float keys meeting it. `i as f64` rounds to the
/// nearest float, and `cmp_i64_f64` says which side of `i` it landed on;
/// adjacent floats have adjacent keys
fn float_against_integer(op: Operator, i: i64) -> Resolved {
    use std::cmp::Ordering;
    let f = i as f64;
    let k = float_key(f);
    // How `f` stands to `i`: Less when it rounded down
    let Some(side) = cmp_i64_f64(i, f).map(Ordering::reverse) else {
        return Resolved::Fallback;
    };
    let range = |low: Option<i64>, high: Option<i64>| match (low, high) {
        (Some(low), Some(high)) if low <= high => Resolved::Range(low, high),
        _ => Resolved::Empty,
    };
    match (op, side) {
        (Operator::Eq, Ordering::Equal) => Resolved::Key(k),
        (Operator::Eq, _) => Resolved::Empty,
        (Operator::Lt, Ordering::Less) | (Operator::Lte, Ordering::Less | Ordering::Equal) => {
            range(Some(i64::MIN), Some(k))
        }
        (Operator::Lt | Operator::Lte, _) => range(Some(i64::MIN), k.checked_sub(1)),
        (Operator::Gt, Ordering::Greater)
        | (Operator::Gte, Ordering::Greater | Ordering::Equal) => range(Some(k), Some(i64::MAX)),
        (Operator::Gt | Operator::Gte, _) => range(k.checked_add(1), Some(i64::MAX)),
        _ => Resolved::Fallback,
    }
}

/// What a read asks of an entry
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
    Equality,
    Range,
    /// Equality on the key's leading components, whole components only
    Prefix,
}

impl KeyKind {
    pub fn provides(self, capability: Capability) -> bool {
        match self {
            KeyKind::OrderedI64 => matches!(capability, Capability::Equality | Capability::Range),
            KeyKind::OrderedBytes => true,
        }
    }
}

/// One column of a key, by its schema position
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Component {
    pub column: u32,
    pub domain: Domain,
}

/// An index a side file can hold an entry for
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SideIndexDef {
    /// The LSN of the catalog record that created the index; never zero
    pub identity: u64,
    /// The key's columns in key order, never empty
    pub components: SmallVec<[Component; 4]>,
    pub kind: KeyKind,
    pub index_type: IndexType,
}

/// The kind a side file holds an index of `index_type` over `components`
/// in: a B-tree on one numeric or time column, a TEXT B-tree or hash, or a
/// key of several keyed columns. None for any other index
pub fn kind_for(index_type: IndexType, components: &[Component]) -> Option<KeyKind> {
    match components {
        [] => None,
        [one] => match (one.domain, index_type) {
            (Domain::Text, IndexType::BTree | IndexType::Hash) => Some(KeyKind::OrderedBytes),
            (Domain::Integer | Domain::Timestamp | Domain::Float, IndexType::BTree) => {
                Some(KeyKind::OrderedI64)
            }
            _ => None,
        },
        _ => matches!(
            index_type,
            IndexType::BTree | IndexType::Hash | IndexType::MultiColumn
        )
        .then_some(KeyKind::OrderedBytes),
    }
}

impl SideIndexDef {
    /// The key's first column
    pub fn leading(&self) -> Option<Component> {
        self.components.first().copied()
    }
}

/// An entry to build into one volume's side file: the index's identity, its
/// key kind, and its key's physical columns with their domains, in key order
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SideTarget {
    pub identity: u64,
    pub kind: KeyKind,
    pub components: SmallVec<[(usize, Domain); 2]>,
}

/// The entries a volume written in the current schema is built with: its
/// physical columns are the schema's
pub fn schema_entries(defs: &[SideIndexDef]) -> Vec<SideTarget> {
    defs.iter()
        .map(|def| SideTarget {
            identity: def.identity,
            kind: def.kind,
            components: def
                .components
                .iter()
                .map(|c| (c.column as usize, c.domain))
                .collect(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    const P53: f64 = 9_007_199_254_740_992.0;
    const P63: f64 = 9_223_372_036_854_775_808.0;

    fn floats() -> Vec<f64> {
        let mut out = vec![
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            f64::from_bits(1),
            -f64::from_bits(1),
            1.5,
            -1.5,
            5.0,
            5.5,
            P53,
            P53 + 2.0,
            -P53,
            P63,
            -P63,
            f64::MAX,
            f64::MIN,
        ];
        out.extend((-20..=20).map(|i| i as f64 * 0.75));
        out
    }

    fn integers() -> Vec<i64> {
        let mut out = vec![
            i64::MIN,
            i64::MIN + 1,
            i64::MAX,
            i64::MAX - 1,
            9_007_199_254_740_992,
            9_007_199_254_740_993,
            -9_007_199_254_740_993,
        ];
        out.extend(-20..=20);
        out
    }

    fn queries() -> Vec<Value> {
        let mut out: Vec<Value> = floats().into_iter().map(Value::Float).collect();
        out.push(Value::Float(f64::NAN));
        out.extend(integers().into_iter().map(Value::Integer));
        out
    }

    const OPS: [Operator; 5] = [
        Operator::Eq,
        Operator::Lt,
        Operator::Lte,
        Operator::Gt,
        Operator::Gte,
    ];

    fn holds(op: Operator, ordering: Ordering) -> bool {
        match op {
            Operator::Eq => ordering == Ordering::Equal,
            Operator::Lt => ordering == Ordering::Less,
            Operator::Lte => ordering != Ordering::Greater,
            Operator::Gt => ordering == Ordering::Greater,
            Operator::Gte => ordering != Ordering::Less,
            _ => unreachable!(),
        }
    }

    /// The keys a resolution selects among `keys`, or None for a fallback
    fn selected(resolved: Resolved, keys: &[i64]) -> Option<Vec<bool>> {
        Some(match resolved {
            Resolved::Key(k) => keys.iter().map(|&key| key == k).collect(),
            Resolved::Range(low, high) => {
                keys.iter().map(|&key| low <= key && key <= high).collect()
            }
            Resolved::Empty => vec![false; keys.len()],
            Resolved::Fallback => return None,
        })
    }

    #[test]
    fn float_keys_keep_the_order_and_merge_the_zeros() {
        let mut values = floats();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for pair in values.windows(2) {
            let (a, b) = (float_key(pair[0]), float_key(pair[1]));
            match pair[0].partial_cmp(&pair[1]).unwrap() {
                Ordering::Equal => assert_eq!(a, b, "{pair:?}"),
                _ => assert!(a < b, "{pair:?}"),
            }
        }
        assert_eq!(float_key(-0.0), float_key(0.0));
        for f in floats() {
            if f != 0.0 {
                assert_eq!(float_of_key(float_key(f)).to_bits(), f.to_bits());
            }
        }
        // Adjacent floats have adjacent keys, so a key one below is the
        // float just below
        assert_eq!(float_key(1.0) - 1, float_key(1.0f64.next_down()));
    }

    #[test]
    fn a_float_column_resolves_as_the_exact_comparison() {
        let column = floats();
        let keys: Vec<i64> = column.iter().map(|&f| float_key(f)).collect();
        for value in queries() {
            for op in OPS {
                let resolved = resolve(Domain::Float, op, &value);
                let Some(got) = selected(resolved, &keys) else {
                    assert!(
                        matches!(value, Value::Float(f) if f.is_nan()),
                        "only NaN falls back: {op:?} {value:?}"
                    );
                    continue;
                };
                let want: Vec<bool> = column
                    .iter()
                    .map(|&f| {
                        let ordering = match value {
                            Value::Float(q) => f.partial_cmp(&q).unwrap(),
                            Value::Integer(q) => cmp_i64_f64(q, f).unwrap().reverse(),
                            _ => unreachable!(),
                        };
                        holds(op, ordering)
                    })
                    .collect();
                assert_eq!(got, want, "f {op:?} {value:?}: {resolved:?}");
            }
        }
    }

    #[test]
    fn an_integer_column_resolves_as_the_exact_comparison() {
        let column = integers();
        for value in queries() {
            for op in OPS {
                let resolved = resolve(Domain::Integer, op, &value);
                let Some(got) = selected(resolved, &column) else {
                    assert!(
                        matches!(value, Value::Float(f) if f.is_nan()),
                        "only NaN falls back: {op:?} {value:?}"
                    );
                    continue;
                };
                let want: Vec<bool> = column
                    .iter()
                    .map(|&i| {
                        let ordering = match value {
                            Value::Float(q) => cmp_i64_f64(i, q).unwrap(),
                            Value::Integer(q) => i.cmp(&q),
                            _ => unreachable!(),
                        };
                        holds(op, ordering)
                    })
                    .collect();
                assert_eq!(got, want, "i {op:?} {value:?}: {resolved:?}");
            }
        }
    }

    #[test]
    fn values_the_keys_cannot_answer_fall_back() {
        let text = Value::text("5");
        assert_eq!(
            resolve(Domain::Integer, Operator::Eq, &text),
            Resolved::Fallback
        );
        assert_eq!(
            resolve(Domain::Timestamp, Operator::Eq, &text),
            Resolved::Fallback
        );
        assert_eq!(
            resolve(Domain::Float, Operator::Eq, &Value::Float(f64::NAN)),
            Resolved::Fallback
        );
        assert_eq!(
            resolve(Domain::Integer, Operator::Ne, &Value::Integer(5)),
            Resolved::Fallback
        );
        assert_eq!(
            resolve(Domain::Boolean, Operator::Eq, &Value::Boolean(true)),
            Resolved::Key(1)
        );
    }
}
