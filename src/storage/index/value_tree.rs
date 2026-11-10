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

//! An ordered map from `Value` keys to row ids: one B+tree per type rank,
//! wide nodes searched first by each key's order word, past the bytes every
//! TEXT key of the node shares, and then by `cmp`; a key's single row id is
//! kept in its slot.

use std::cmp::Ordering;
use std::mem;
use std::ops::Bound;

use crate::common::smart_string::cmp_bytes;
use crate::core::value::float_word;
use crate::core::{DataType, Value};

use super::id_list::{GroupIds, IdList};

/// Keys of a leaf, and separators of an inner node, before it splits
const LEAF_MAX: usize = 64;
const INNER_MAX: usize = 64;

/// A node below a quarter of its width merges into a sibling it fits beside
fn underfull(len: usize, max: usize) -> bool {
    len < max / 4
}

/// The sizes of the fewest nodes of at most `per` that hold `n`, as even as
/// they come: none of them below half of `per` unless there is one
fn spread(n: usize, per: usize) -> impl Iterator<Item = usize> {
    let count = n.div_ceil(per);
    let (base, extra) = match count {
        0 => (0, 0),
        _ => (n / count, n % count),
    };
    (0..count).map(move |i| base + usize::from(i < extra))
}

/// The row ids of one key, a single one inline
#[derive(Debug)]
pub enum Ids {
    One(i64),
    Many(Box<IdList>),
}

impl Ids {
    pub fn len(&self) -> usize {
        match self {
            Ids::One(_) => 1,
            Ids::Many(list) => list.len(),
        }
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = i64> + '_ {
        let (one, many) = match self {
            Ids::One(id) => (Some(*id), None),
            Ids::Many(list) => (None, Some(list.iter())),
        };
        one.into_iter().chain(many.into_iter().flatten())
    }

    /// The ids as a group a caller visits whole
    pub fn as_group(&self) -> GroupIds<'_> {
        match self {
            Ids::One(id) => GroupIds::Slice(std::slice::from_ref(id)),
            Ids::Many(list) => GroupIds::List(list),
        }
    }

    pub fn copy_into(&self, out: &mut Vec<i64>) {
        match self {
            Ids::One(id) => out.push(*id),
            Ids::Many(list) => list.copy_into(out),
        }
    }

    /// As [`IdList::copy_past_into`]
    pub fn copy_past_into(
        &self,
        after: Option<i64>,
        ascending: bool,
        max: usize,
        out: &mut Vec<i64>,
    ) {
        match self {
            Ids::One(id) => {
                let past = after.is_none_or(|a| if ascending { *id > a } else { *id < a });
                if past && max > 0 {
                    out.push(*id);
                }
            }
            Ids::Many(list) => list.copy_past_into(after, ascending, max, out),
        }
    }

    /// Whether `id` was not there
    fn insert(&mut self, id: i64) -> bool {
        match self {
            Ids::One(one) if *one == id => false,
            Ids::One(one) => {
                let pair = if *one < id {
                    vec![*one, id]
                } else {
                    vec![id, *one]
                };
                *self = Ids::Many(Box::new(IdList::from_sorted(pair)));
                true
            }
            Ids::Many(list) => list.insert(id),
        }
    }

    /// Removes the sorted `ids`: how many were there, and whether none is left
    fn remove_sorted(&mut self, ids: &[i64]) -> (usize, bool) {
        match self {
            Ids::One(one) => {
                let found = ids.binary_search(one).is_ok();
                (usize::from(found), found)
            }
            Ids::Many(list) => {
                let removed = list.remove_sorted(ids);
                let left = list.len();
                // The one id left is read only when one is left
                let last = if left == 1 { list.iter().next() } else { None };
                if left == 0 {
                    return (removed, true);
                }
                if let Some(last) = last {
                    *self = Ids::One(last);
                }
                (removed, false)
            }
        }
    }
}

/// The position of `key` among keys in `Value::cmp` order, `key_at` the
/// key at a position and `words` their order words: the words narrow it to
/// the keys sharing `key`'s word, `cmp` decides there
#[inline]
fn search<'a>(
    words: &[u64],
    key_at: impl Fn(usize) -> &'a Value,
    word: u64,
    key: &Value,
) -> Result<usize, usize> {
    let lo = lower_bound(words, word);
    let mut hi = lo;
    while hi < words.len() && words[hi] == word {
        hi += 1;
    }
    let (mut from, mut to) = (lo, hi);
    let found = loop {
        if from == to {
            break Err(from);
        }
        let mid = from + (to - from) / 2;
        match key_at(mid).cmp(key) {
            Ordering::Less => from = mid + 1,
            Ordering::Greater => to = mid,
            Ordering::Equal => break Ok(mid),
        }
    };
    debug_assert!(
        match found {
            Ok(i) => key_at(i).cmp(key) == Ordering::Equal,
            Err(i) => {
                (i == 0 || key_at(i - 1).cmp(key) == Ordering::Less)
                    && (i == words.len() || key.cmp(key_at(i)) == Ordering::Less)
            }
        },
        "an order word disagrees with Value::cmp"
    );
    found
}

/// The first index whose word is not below `word`, the halving free of
/// branches on the words
#[inline]
fn lower_bound(words: &[u64], word: u64) -> usize {
    let mut base = 0;
    let mut size = words.len();
    while size > 1 {
        let half = size / 2;
        base = if words[base + half] < word {
            base + half
        } else {
            base
        };
        size -= half;
    }
    base + usize::from(size == 1 && words[base] < word)
}

/// Leading bytes of a node's TEXT keys skipped at most, kept in the node
const PREFIX: usize = 48;

/// How many leading bytes, up to `PREFIX`, all keys from `first` to `last`
/// share: those the two share, when both are TEXT
fn shared_prefix(first: &Value, last: &Value) -> usize {
    let (Value::Text(first), Value::Text(last)) = (first, last) else {
        return 0;
    };
    let n = first.len().min(last.len()).min(PREFIX);
    let (a, b) = (
        &first.as_str().as_bytes()[..n],
        &last.as_str().as_bytes()[..n],
    );
    // Eight bytes at a time: the first differing byte is the leading zero
    // bytes of the words' difference
    let (a_words, a_tail) = a.as_chunks::<8>();
    let (b_words, b_tail) = b.as_chunks::<8>();
    for (i, (x, y)) in a_words.iter().zip(b_words).enumerate() {
        let diff = u64::from_be_bytes(*x) ^ u64::from_be_bytes(*y);
        if diff != 0 {
            return i * 8 + diff.leading_zeros() as usize / 8;
        }
    }
    let same = a_tail.iter().zip(b_tail).take_while(|(x, y)| x == y);
    a_words.len() * 8 + same.count()
}

fn placeholder() -> Value {
    Value::Null(DataType::Null)
}

/// A key searched for, with its order word when no node's prefix changes
/// it: only a TEXT word depends on the node it is taken in
#[derive(Clone, Copy)]
struct Probe<'a> {
    key: &'a Value,
    word: Option<u64>,
}

impl<'a> Probe<'a> {
    fn new(key: &'a Value) -> Self {
        let word = match key {
            Value::Text(_) => None,
            _ => Some(key.order_word_at(0)),
        };
        Probe { key, word }
    }
}

/// The value an empty slot holds
trait Blank {
    fn blank() -> Self;
}

impl Blank for Ids {
    fn blank() -> Self {
        Ids::One(0)
    }
}

impl Blank for () {
    fn blank() -> Self {}
}

/// The keys of a node, each in a slot with its value. The slots stay where
/// their keys came; the keys' order is a list of slots with their order
/// words beside it, so a write moves those two and no key. The words are
/// taken past the `skip` bytes the node's TEXT keys share, which it keeps a
/// copy of. `CAP` holds one key over the node's width until it splits
struct Keys<const CAP: usize, V> {
    len: usize,
    skip: usize,
    prefix: [u8; PREFIX],
    words: [u64; CAP],
    slots: [u8; CAP],
    keys: [Value; CAP],
    values: [V; CAP],
}

impl<const CAP: usize, V: Blank> Keys<CAP, V> {
    fn new() -> Self {
        Keys {
            len: 0,
            skip: 0,
            prefix: [0; PREFIX],
            words: [0; CAP],
            slots: [0; CAP],
            keys: std::array::from_fn(|_| placeholder()),
            values: std::array::from_fn(|_| V::blank()),
        }
    }

    /// The key at position `i` of the order
    #[inline]
    fn key(&self, i: usize) -> &Value {
        &self.keys[usize::from(self.slots[i])]
    }

    #[inline]
    fn value(&self, i: usize) -> &V {
        &self.values[usize::from(self.slots[i])]
    }

    #[inline]
    fn value_mut(&mut self, i: usize) -> &mut V {
        &mut self.values[usize::from(self.slots[i])]
    }

    fn first(&self) -> &Value {
        self.key(0)
    }

    /// Where the probed key is: a TEXT key outside the shared prefix sorts
    /// before or after all of them, one inside it is found by its word past
    /// the prefix
    #[inline]
    fn locate(&self, probe: &Probe<'_>) -> Result<usize, usize> {
        let key = probe.key;
        if self.skip > 0 {
            if let Value::Text(text) = key {
                let text = text.as_str().as_bytes();
                let n = text.len().min(self.skip);
                let order = cmp_bytes(&text[..n], &self.prefix[..n]).then(n.cmp(&self.skip));
                match order {
                    Ordering::Less => return Err(0),
                    Ordering::Greater => return Err(self.len),
                    Ordering::Equal => {}
                }
            }
        }
        let word = probe.word.unwrap_or_else(|| key.order_word_at(self.skip));
        search(&self.words[..self.len], |i| self.key(i), word, key)
    }

    /// Enters `key` at position `i` in the next free slot. Room for one more
    /// is always there: a node over its width splits at once
    fn insert(&mut self, i: usize, key: Value, value: V) {
        let n = self.len;
        self.words.copy_within(i..n, i + 1);
        self.words[i] = key.order_word_at(self.skip);
        self.slots.copy_within(i..n, i + 1);
        self.slots[i] = n as u8;
        self.keys[n] = key;
        self.values[n] = value;
        self.len += 1;
        if i == 0 || i == n {
            self.refresh();
        }
    }

    /// Appends `key`, greater than every key here, without its word: a
    /// build appends a node's keys and then rewrites it once
    fn push(&mut self, key: Value, value: V) {
        let n = self.len;
        self.slots[n] = n as u8;
        self.keys[n] = key;
        self.values[n] = value;
        self.len += 1;
    }

    /// Takes out the key at position `i`; the last slot moves into its slot
    fn remove(&mut self, i: usize) -> (Value, V) {
        let (n, slot) = (self.len, usize::from(self.slots[i]));
        let last = n - 1;
        let key = mem::replace(&mut self.keys[slot], placeholder());
        let value = mem::replace(&mut self.values[slot], V::blank());
        if slot != last {
            self.keys.swap(slot, last);
            self.values.swap(slot, last);
            if let Some(moved) = self.slots[..n].iter().position(|&s| usize::from(s) == last) {
                self.slots[moved] = slot as u8;
            }
        }
        self.words.copy_within(i + 1..n, i);
        self.slots.copy_within(i + 1..n, i);
        self.len -= 1;
        if i == 0 || i == last {
            self.refresh();
        }
        (key, value)
    }

    /// Moves the keys at positions `at..` into `right`, empty, and the keys
    /// left here into the slots below `at`
    fn split_off(&mut self, at: usize, right: &mut Self) {
        let n = self.len;
        let mut taken = [false; CAP];
        for (j, p) in (at..n).enumerate() {
            let slot = usize::from(self.slots[p]);
            right.keys[j] = mem::replace(&mut self.keys[slot], placeholder());
            right.values[j] = mem::replace(&mut self.values[slot], V::blank());
            right.slots[j] = j as u8;
            taken[slot] = true;
        }
        let mut free = (0..at).filter(|&s| taken[s]);
        for p in 0..at {
            let slot = usize::from(self.slots[p]);
            if slot >= at {
                if let Some(to) = free.next() {
                    self.keys.swap(slot, to);
                    self.values.swap(slot, to);
                    self.slots[p] = to as u8;
                }
            }
        }
        right.len = n - at;
        self.len = at;
        self.rewrite();
        right.rewrite();
    }

    /// Appends the keys of `right`, which fit, in slots after these, leaving
    /// it empty
    fn append(&mut self, right: &mut Self) {
        let at = self.len;
        for p in 0..right.len {
            let slot = usize::from(right.slots[p]);
            self.keys[at + p] = mem::replace(&mut right.keys[slot], placeholder());
            self.values[at + p] = mem::replace(&mut right.values[slot], V::blank());
            self.slots[at + p] = (at + p) as u8;
        }
        self.len += right.len;
        right.len = 0;
        self.rewrite();
    }

    fn shared(&self) -> usize {
        match self.len {
            0 | 1 => 0,
            n => shared_prefix(self.key(0), self.key(n - 1)),
        }
    }

    /// After the first or last key changed: the shared prefix may have
    fn refresh(&mut self) {
        if self.shared() != self.skip {
            self.rewrite();
        }
    }

    fn rewrite(&mut self) {
        self.skip = self.shared();
        if self.skip > 0 {
            if let Value::Text(first) = &self.keys[usize::from(self.slots[0])] {
                let prefix = &first.as_str().as_bytes()[..self.skip];
                self.prefix[..self.skip].copy_from_slice(prefix);
            }
        }
        for i in 0..self.len {
            let word = self.key(i).order_word_at(self.skip);
            self.words[i] = word;
        }
    }
}

struct Leaf {
    keys: Keys<{ LEAF_MAX + 1 }, Ids>,
}

impl Leaf {
    fn new() -> Box<Self> {
        Box::new(Leaf { keys: Keys::new() })
    }

    /// Moves the upper half out: its first key separates the two
    fn split(&mut self) -> (Value, Node) {
        let mut right = Leaf::new();
        self.keys.split_off(self.keys.len / 2, &mut right.keys);
        (right.keys.first().clone(), Node::Leaf(right))
    }
}

/// The children of an inner node, kept in it: a descent reads no list
/// apart from the node
struct Children {
    len: usize,
    nodes: [Option<Node>; INNER_MAX + 2],
}

impl Children {
    fn new() -> Self {
        Children {
            len: 0,
            nodes: std::array::from_fn(|_| None),
        }
    }

    #[inline]
    fn get(&self, i: usize) -> Option<&Node> {
        self.nodes[i].as_ref()
    }

    #[inline]
    fn get_mut(&mut self, i: usize) -> Option<&mut Node> {
        self.nodes[i].as_mut()
    }

    /// Children `from..to`, in order
    fn range(&self, from: usize, to: usize) -> impl DoubleEndedIterator<Item = &Node> {
        self.nodes[from..to].iter().flatten()
    }

    /// Children `left` and `left + 1`
    fn pair_mut(&mut self, left: usize) -> Option<(&mut Node, &mut Node)> {
        let (lower, upper) = self.nodes.split_at_mut(left + 1);
        lower[left].as_mut().zip(upper[0].as_mut())
    }

    fn insert(&mut self, i: usize, node: Node) {
        let n = self.len;
        self.nodes[i..=n].rotate_right(1);
        self.nodes[i] = Some(node);
        self.len += 1;
    }

    fn remove(&mut self, i: usize) -> Option<Node> {
        let n = self.len;
        let node = self.nodes[i].take();
        self.nodes[i..n].rotate_left(1);
        self.len -= 1;
        node
    }

    fn push(&mut self, node: Node) {
        self.nodes[self.len] = Some(node);
        self.len += 1;
    }

    fn pop(&mut self) -> Option<Node> {
        self.len = self.len.checked_sub(1)?;
        self.nodes[self.len].take()
    }

    /// Moves children `at..` out
    fn split_off(&mut self, at: usize) -> Children {
        let mut right = Children::new();
        let n = self.len - at;
        right.nodes[..n].swap_with_slice(&mut self.nodes[at..self.len]);
        right.len = n;
        self.len = at;
        right
    }

    fn append(&mut self, other: &mut Children) {
        let (at, n) = (self.len, other.len);
        self.nodes[at..at + n].swap_with_slice(&mut other.nodes[..n]);
        self.len += n;
        other.len = 0;
    }
}

struct Inner {
    /// Separators; there is one child more
    keys: Keys<{ INNER_MAX + 1 }, ()>,
    children: Children,
}

impl Inner {
    fn new(children: Children) -> Box<Self> {
        Box::new(Inner {
            keys: Keys::new(),
            children,
        })
    }

    /// The child a key belongs to: one equal to a separator, right of it
    #[inline]
    fn child(&self, probe: &Probe<'_>) -> usize {
        match self.keys.locate(probe) {
            Ok(i) => i + 1,
            Err(i) => i,
        }
    }

    /// Enters the separator of a child split at `i`, `right` its new sibling
    fn insert_at(&mut self, i: usize, key: Value, right: Node) {
        self.keys.insert(i, key, ());
        self.children.insert(i + 1, right);
    }

    /// Takes out separator `i` and the child right of it
    fn remove_at(&mut self, i: usize) -> (Value, Option<Node>) {
        (self.keys.remove(i).0, self.children.remove(i + 1))
    }

    /// Moves the separators above the middle one out with their children;
    /// the middle one goes up between the two
    fn split(&mut self) -> (Value, Node) {
        let mid = self.keys.len / 2;
        let mut right = Inner::new(self.children.split_off(mid + 1));
        self.keys.split_off(mid + 1, &mut right.keys);
        let (up, ()) = self.keys.remove(mid);
        (up, Node::Inner(right))
    }

    /// Appends separator `key` and then `right`, which fits
    fn absorb(&mut self, key: Value, right: &mut Inner) {
        self.keys.insert(self.keys.len, key, ());
        self.keys.append(&mut right.keys);
        self.children.append(&mut right.children);
    }

    /// Brings child `c`, underfull, back up: merged with a sibling when
    /// both fit in one node, otherwise given one entry of that sibling
    fn rebalance_child(&mut self, c: usize) {
        if self.keys.len == 0 {
            return;
        }
        let left = if c < self.keys.len { c } else { c - 1 };
        let fits = match (self.children.get(left), self.children.get(left + 1)) {
            (Some(Node::Leaf(l)), Some(Node::Leaf(r))) => l.keys.len + r.keys.len <= LEAF_MAX,
            (Some(Node::Inner(l)), Some(Node::Inner(r))) => {
                l.keys.len + 1 + r.keys.len <= INNER_MAX
            }
            _ => false,
        };
        if fits {
            let (key, right) = self.remove_at(left);
            match (self.children.get_mut(left), right) {
                (Some(Node::Leaf(l)), Some(Node::Leaf(mut r))) => l.keys.append(&mut r.keys),
                (Some(Node::Inner(l)), Some(Node::Inner(mut r))) => l.absorb(key, &mut r),
                _ => {}
            }
            return;
        }
        let down = self.keys.key(left).clone();
        let Some(pair) = self.children.pair_mut(left) else {
            return;
        };
        let separator = match pair {
            (Node::Leaf(l), Node::Leaf(r)) => {
                if c == left {
                    let (key, ids) = r.keys.remove(0);
                    l.keys.insert(l.keys.len, key, ids);
                } else {
                    let (key, ids) = l.keys.remove(l.keys.len - 1);
                    r.keys.insert(0, key, ids);
                }
                r.keys.first().clone()
            }
            (Node::Inner(l), Node::Inner(r)) => {
                if c == left {
                    let (up, ()) = r.keys.remove(0);
                    l.keys.insert(l.keys.len, down, ());
                    if let Some(child) = r.children.remove(0) {
                        l.children.push(child);
                    }
                    up
                } else {
                    let (up, ()) = l.keys.remove(l.keys.len - 1);
                    r.keys.insert(0, down, ());
                    if let Some(child) = l.children.pop() {
                        r.children.insert(0, child);
                    }
                    up
                }
            }
            _ => return,
        };
        self.keys.remove(left);
        self.keys.insert(left, separator, ());
    }
}

enum Node {
    Leaf(Box<Leaf>),
    Inner(Box<Inner>),
}

impl Node {
    fn underfull(&self) -> bool {
        match self {
            Node::Leaf(leaf) => underfull(leaf.keys.len, LEAF_MAX),
            Node::Inner(inner) => underfull(inner.keys.len, INNER_MAX),
        }
    }
}

/// What an insert did below a node
struct Inserted {
    id: bool,
    key: bool,
    split: Option<(Value, Node)>,
}

/// The key exists under another row id and the insert was unique
#[derive(Debug, PartialEq, Eq)]
pub struct KeyTaken;

/// Keys in `Value::cmp` order, each with its row ids
pub struct ValueTree {
    roots: [Option<Node>; Value::TYPE_RANKS],
    keys: usize,
}

impl Default for ValueTree {
    fn default() -> Self {
        Self::new()
    }
}

impl ValueTree {
    pub fn new() -> Self {
        ValueTree {
            roots: Default::default(),
            keys: 0,
        }
    }

    /// A tree of `entries` in any order. They are sorted by rank, order word
    /// and id, a TEXT word taken past the bytes every TEXT key here shares
    /// and a FLOAT word exact when no INTEGER is here, so the sort reads no
    /// value; a run of one word holding more than one key is sorted by `cmp`
    pub fn from_entries(entries: Vec<(&Value, i64)>) -> Self {
        let mut first_text: Option<&Value> = None;
        let mut skip = PREFIX;
        let mut integers = false;
        for &(value, _) in &entries {
            match value {
                Value::Text(_) if skip > 0 => {
                    let first = *first_text.get_or_insert(value);
                    skip = skip.min(shared_prefix(first, value));
                }
                Value::Integer(_) => integers = true,
                _ => {}
            }
        }
        let word = |value: &Value| match value {
            Value::Float(f) if !integers => float_word(*f),
            _ => value.order_word_at(skip),
        };
        let mut sorted: Vec<(u8, u64, &Value, i64)> = entries
            .into_iter()
            .map(|(value, id)| (value.type_rank(), word(value), value, id))
            .collect();
        sorted.sort_unstable_by_key(|&(rank, word, _, id)| (rank, word, id));
        for run in sorted.chunk_by_mut(|a, b| (a.0, a.1) == (b.0, b.1)) {
            if run
                .windows(2)
                .any(|w| w[0].2.cmp(w[1].2) != Ordering::Equal)
            {
                run.sort_unstable_by(|a, b| a.2.cmp(b.2).then(a.3.cmp(&b.3)));
            }
        }
        let pairs: Vec<(&Value, i64)> = sorted.iter().map(|&(_, _, v, id)| (v, id)).collect();
        Self::from_sorted(&pairs)
    }

    /// A tree of `entries`, sorted by key in `Value::cmp` order and then by
    /// id, built bottom-up: each node filled once, three quarters full, with
    /// no search and no shift per key
    fn from_sorted(entries: &[(&Value, i64)]) -> Self {
        debug_assert!(entries
            .windows(2)
            .all(|w| w[0].0.cmp(w[1].0).then(w[0].1.cmp(&w[1].1)) != Ordering::Greater));
        let mut tree = ValueTree::new();
        let mut keys: Vec<(Value, Ids)> = Vec::new();
        for of_rank in entries.chunk_by(|a, b| a.0.type_rank() == b.0.type_rank()) {
            keys.clear();
            for run in of_rank.chunk_by(|a, b| a.0.cmp(b.0) == Ordering::Equal) {
                let one = run[0].1;
                let ids = if run.iter().all(|&(_, id)| id == one) {
                    Ids::One(one)
                } else {
                    let mut ids: Vec<i64> = run.iter().map(|&(_, id)| id).collect();
                    ids.dedup();
                    Ids::Many(Box::new(IdList::from_sorted(ids)))
                };
                keys.push((run[0].0.clone(), ids));
            }
            tree.keys += keys.len();
            tree.roots[usize::from(of_rank[0].0.type_rank())] = Self::build(&mut keys);
        }
        tree
    }

    /// One rank's tree of `keys`, sorted, which it takes
    fn build(keys: &mut Vec<(Value, Ids)>) -> Option<Node> {
        let mut level: Vec<(Value, Node)> = Vec::new();
        let mut taken = keys.drain(..);
        for size in spread(taken.len(), LEAF_MAX * 3 / 4) {
            let mut leaf = Leaf::new();
            for (key, ids) in taken.by_ref().take(size) {
                leaf.keys.push(key, ids);
            }
            leaf.keys.rewrite();
            level.push((leaf.keys.first().clone(), Node::Leaf(leaf)));
        }
        // Each node above is entered by its least key
        while level.len() > 1 {
            let mut nodes = mem::take(&mut level).into_iter();
            for size in spread(nodes.len(), INNER_MAX * 3 / 4 + 1) {
                let mut group = nodes.by_ref().take(size);
                let (least, first) = group.next()?;
                let mut children = Children::new();
                children.push(first);
                let mut inner = Inner::new(children);
                for (key, node) in group {
                    inner.keys.push(key, ());
                    inner.children.push(node);
                }
                inner.keys.rewrite();
                level.push((least, Node::Inner(inner)));
            }
        }
        level.pop().map(|(_, node)| node)
    }

    /// Distinct keys
    pub fn len(&self) -> usize {
        self.keys
    }

    pub fn clear(&mut self) {
        self.roots = Default::default();
        self.keys = 0;
    }

    pub fn get(&self, key: &Value) -> Option<&Ids> {
        let probe = Probe::new(key);
        let mut node = self.roots[usize::from(key.type_rank())].as_ref()?;
        loop {
            match node {
                Node::Inner(inner) => node = inner.children.get(inner.child(&probe))?,
                Node::Leaf(leaf) => {
                    return leaf.keys.locate(&probe).ok().map(|i| leaf.keys.value(i));
                }
            }
        }
    }

    /// Enters `id` under `key`, whether it was not there. A `unique` insert
    /// refuses a key another id holds, changing nothing
    pub fn insert(&mut self, key: &Value, id: i64, unique: bool) -> Result<bool, KeyTaken> {
        let root = &mut self.roots[usize::from(key.type_rank())];
        let Some(node) = root else {
            let mut leaf = Leaf::new();
            leaf.keys.insert(0, key.clone(), Ids::One(id));
            *root = Some(Node::Leaf(leaf));
            self.keys += 1;
            return Ok(true);
        };
        let inserted = Self::insert_in(node, &Probe::new(key), id, unique)?;
        if let Some((separator, right)) = inserted.split {
            if let Some(left) = root.take() {
                let mut children = Children::new();
                children.push(left);
                let mut top = Inner::new(children);
                top.insert_at(0, separator, right);
                *root = Some(Node::Inner(top));
            }
        }
        self.keys += usize::from(inserted.key);
        Ok(inserted.id)
    }

    fn insert_in(
        node: &mut Node,
        probe: &Probe<'_>,
        id: i64,
        unique: bool,
    ) -> Result<Inserted, KeyTaken> {
        match node {
            Node::Leaf(leaf) => match leaf.keys.locate(probe) {
                Ok(i) => {
                    let ids = leaf.keys.value_mut(i);
                    if unique && ids.iter().any(|other| other != id) {
                        return Err(KeyTaken);
                    }
                    Ok(Inserted {
                        id: ids.insert(id),
                        key: false,
                        split: None,
                    })
                }
                Err(i) => {
                    leaf.keys.insert(i, probe.key.clone(), Ids::One(id));
                    let split = (leaf.keys.len > LEAF_MAX).then(|| leaf.split());
                    Ok(Inserted {
                        id: true,
                        key: true,
                        split,
                    })
                }
            },
            Node::Inner(inner) => {
                let c = inner.child(probe);
                let Some(child) = inner.children.get_mut(c) else {
                    return Ok(Inserted {
                        id: false,
                        key: false,
                        split: None,
                    });
                };
                let mut inserted = Self::insert_in(child, probe, id, unique)?;
                if let Some((separator, right)) = inserted.split.take() {
                    inner.insert_at(c, separator, right);
                    inserted.split = (inner.keys.len > INNER_MAX).then(|| inner.split());
                }
                Ok(inserted)
            }
        }
    }

    /// Removes the sorted `ids` from `key`, the key with its last one: how
    /// many were there
    pub fn remove_sorted(&mut self, key: &Value, ids: &[i64]) -> usize {
        let root = &mut self.roots[usize::from(key.type_rank())];
        let Some(node) = root else {
            return 0;
        };
        let (removed, gone) = Self::remove_in(node, &Probe::new(key), ids);
        let collapsed = match node {
            Node::Leaf(leaf) if leaf.keys.len == 0 => Some(None),
            Node::Inner(inner) if inner.keys.len == 0 => inner.children.pop().map(Some),
            _ => None,
        };
        if let Some(next) = collapsed {
            *root = next;
        }
        self.keys -= usize::from(gone);
        removed
    }

    pub fn remove(&mut self, key: &Value, id: i64) -> bool {
        self.remove_sorted(key, std::slice::from_ref(&id)) == 1
    }

    fn remove_in(node: &mut Node, probe: &Probe<'_>, ids: &[i64]) -> (usize, bool) {
        match node {
            Node::Leaf(leaf) => match leaf.keys.locate(probe) {
                Ok(i) => {
                    let (removed, empty) = leaf.keys.value_mut(i).remove_sorted(ids);
                    if empty {
                        leaf.keys.remove(i);
                    }
                    (removed, empty)
                }
                Err(_) => (0, false),
            },
            Node::Inner(inner) => {
                let c = inner.child(probe);
                let Some(child) = inner.children.get_mut(c) else {
                    return (0, false);
                };
                let (removed, gone) = Self::remove_in(child, probe, ids);
                if gone && child.underfull() {
                    inner.rebalance_child(c);
                }
                (removed, gone)
            }
        }
    }

    /// Visits keys in order from `from`, up or down, until `visit` returns
    /// false
    pub fn walk(
        &self,
        from: Bound<&Value>,
        ascending: bool,
        visit: &mut dyn FnMut(&Value, &Ids) -> bool,
    ) {
        let start = match from {
            Bound::Included(key) | Bound::Excluded(key) => usize::from(key.type_rank()),
            Bound::Unbounded if ascending => 0,
            Bound::Unbounded => Value::TYPE_RANKS - 1,
        };
        let from = from.map(Probe::new);
        let mut rank = start;
        loop {
            let bound = if rank == start {
                from
            } else {
                Bound::Unbounded
            };
            let more = match &self.roots[rank] {
                Some(root) if ascending => walk_up(root, bound, visit),
                Some(root) => walk_down(root, bound, visit),
                None => true,
            };
            let next = if ascending {
                rank + 1
            } else {
                rank.wrapping_sub(1)
            };
            if !more || next >= Value::TYPE_RANKS {
                return;
            }
            rank = next;
        }
    }
}

fn walk_up(
    node: &Node,
    from: Bound<Probe<'_>>,
    visit: &mut dyn FnMut(&Value, &Ids) -> bool,
) -> bool {
    match node {
        Node::Leaf(leaf) => {
            let start = match &from {
                Bound::Unbounded => 0,
                Bound::Included(probe) => match leaf.keys.locate(probe) {
                    Ok(i) | Err(i) => i,
                },
                Bound::Excluded(probe) => match leaf.keys.locate(probe) {
                    Ok(i) => i + 1,
                    Err(i) => i,
                },
            };
            (start..leaf.keys.len).all(|i| visit(leaf.keys.key(i), leaf.keys.value(i)))
        }
        Node::Inner(inner) => {
            let c = match &from {
                Bound::Unbounded => 0,
                Bound::Included(probe) | Bound::Excluded(probe) => inner.child(probe),
            };
            inner
                .children
                .get(c)
                .is_none_or(|child| walk_up(child, from, visit))
                && inner
                    .children
                    .range(c + 1, inner.children.len)
                    .all(|child| walk_up(child, Bound::Unbounded, visit))
        }
    }
}

fn walk_down(
    node: &Node,
    to: Bound<Probe<'_>>,
    visit: &mut dyn FnMut(&Value, &Ids) -> bool,
) -> bool {
    match node {
        Node::Leaf(leaf) => {
            let end = match &to {
                Bound::Unbounded => leaf.keys.len,
                Bound::Included(probe) => match leaf.keys.locate(probe) {
                    Ok(i) => i + 1,
                    Err(i) => i,
                },
                Bound::Excluded(probe) => match leaf.keys.locate(probe) {
                    Ok(i) | Err(i) => i,
                },
            };
            (0..end)
                .rev()
                .all(|i| visit(leaf.keys.key(i), leaf.keys.value(i)))
        }
        Node::Inner(inner) => {
            let c = match &to {
                Bound::Unbounded => inner.keys.len,
                Bound::Included(probe) | Bound::Excluded(probe) => inner.child(probe),
            };
            inner
                .children
                .get(c)
                .is_none_or(|child| walk_down(child, to, visit))
                && inner
                    .children
                    .range(0, c)
                    .rev()
                    .all(|child| walk_down(child, Bound::Unbounded, visit))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    /// A deterministic stream of numbers for the random operations
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// Keys of every rank, with ties across INTEGER and FLOAT, NaN, signed
    /// zeros, and text sharing long prefixes
    fn key(rng: &mut Lcg, spread: u64) -> Value {
        let n = rng.below(spread) as i64;
        match rng.below(12) {
            0 => Value::Null(DataType::Integer),
            1 => Value::Boolean(n % 2 == 0),
            2 | 3 => Value::Integer(n - spread as i64 / 2),
            4 => Value::Float(n as f64 / 4.0 - 7.0),
            5 => Value::Float(
                [
                    f64::NAN,
                    -0.0,
                    0.0,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    i64::MAX as f64,
                ][(n % 6) as usize],
            ),
            6 => Value::Integer([i64::MIN, i64::MAX, -1, 0][(n % 4) as usize]),
            7 | 8 => Value::text(format!("k{n:05}")),
            9 => Value::text(format!("key-{n:036}")),
            10 => Value::text(["", "a", "a\0", "ab", "\u{7f}"][(n % 5) as usize]),
            _ => Value::Timestamp(
                chrono::DateTime::from_timestamp(n * 1000 - 50_000, (n % 3) as u32 * 500)
                    .unwrap_or_default(),
            ),
        }
    }

    type Model = BTreeMap<Value, BTreeSet<i64>>;

    fn walked(tree: &ValueTree, from: Bound<&Value>, ascending: bool) -> Vec<(Value, Vec<i64>)> {
        let mut out = Vec::new();
        tree.walk(from, ascending, &mut |key, ids| {
            out.push((key.clone(), ids.iter().collect()));
            true
        });
        out
    }

    fn expected(model: &Model, from: Bound<&Value>, ascending: bool) -> Vec<(Value, Vec<i64>)> {
        let entry = |(k, v): (&Value, &BTreeSet<i64>)| (k.clone(), v.iter().copied().collect());
        if ascending {
            model.range((from, Bound::Unbounded)).map(entry).collect()
        } else {
            model
                .range((Bound::Unbounded, from))
                .rev()
                .map(entry)
                .collect()
        }
    }

    /// Order, words and shared prefix of a node's keys, all inside its
    /// range, and slots that are the first `len` ones, each once
    fn check_keys<const CAP: usize, V: Blank>(
        keys: &Keys<CAP, V>,
        low: Option<&Value>,
        high: Option<&Value>,
    ) {
        let within = |k: &Value| {
            low.is_none_or(|l| l.cmp(k) != Ordering::Greater)
                && high.is_none_or(|h| k.cmp(h) == Ordering::Less)
        };
        assert!(keys.len < CAP);
        let mut slots: Vec<u8> = keys.slots[..keys.len].to_vec();
        slots.sort_unstable();
        assert!(slots.iter().enumerate().all(|(i, &s)| usize::from(s) == i));
        let shared = match (keys.len > 1, keys.first(), keys.key(keys.len.max(1) - 1)) {
            (true, Value::Text(a), Value::Text(b)) => {
                let (a, b) = (a.as_str().as_bytes(), b.as_str().as_bytes());
                let same = a.iter().zip(b).take_while(|(x, y)| x == y).count();
                same.min(PREFIX)
            }
            _ => 0,
        };
        assert_eq!(
            keys.skip, shared,
            "the shared prefix of the first and last key"
        );
        if let (true, Value::Text(first)) = (keys.len > 0, keys.first()) {
            assert_eq!(
                &keys.prefix[..keys.skip],
                &first.as_str().as_bytes()[..keys.skip]
            );
        }
        for i in 0..keys.len {
            assert_eq!(keys.words[i], keys.key(i).order_word_at(keys.skip));
            assert!(within(keys.key(i)));
            assert!(i == 0 || keys.key(i - 1).cmp(keys.key(i)) == Ordering::Less);
        }
    }

    /// Node sizes, key order inside nodes, separators bounding each child
    fn check_node(node: &Node, low: Option<&Value>, high: Option<&Value>, root: bool) -> usize {
        match node {
            Node::Leaf(leaf) => {
                check_keys(&leaf.keys, low, high);
                assert!(root || !underfull(leaf.keys.len, LEAF_MAX));
                assert!((0..leaf.keys.len).all(|i| leaf.keys.value(i).len() > 0));
                1
            }
            Node::Inner(inner) => {
                let n = inner.keys.len;
                check_keys(&inner.keys, low, high);
                assert!(n > 0 && (root || !underfull(n, INNER_MAX)));
                assert_eq!(inner.children.len, n + 1);
                assert_eq!(inner.children.range(0, n + 1).count(), n + 1);
                let mut depth = None;
                for c in 0..=n {
                    let lo = if c == 0 {
                        low
                    } else {
                        Some(inner.keys.key(c - 1))
                    };
                    let hi = if c == n {
                        high
                    } else {
                        Some(inner.keys.key(c))
                    };
                    let child = inner.children.get(c).expect("a child per separator");
                    let d = check_node(child, lo, hi, false);
                    assert!(depth.is_none_or(|x| x == d), "uneven depth");
                    depth = Some(d);
                }
                depth.unwrap_or(0) + 1
            }
        }
    }

    fn check_structure(tree: &ValueTree) {
        for root in tree.roots.iter().flatten() {
            check_node(root, None, None, true);
        }
    }

    fn check(tree: &ValueTree, model: &Model, probe: &mut dyn FnMut() -> Value) {
        check_structure(tree);
        assert_eq!(tree.len(), model.len());
        assert_eq!(
            walked(tree, Bound::Unbounded, true),
            expected(model, Bound::Unbounded, true)
        );
        assert_eq!(
            walked(tree, Bound::Unbounded, false),
            expected(model, Bound::Unbounded, false)
        );
        for _ in 0..20 {
            let probe = probe();
            let got = tree.get(&probe).map(|ids| ids.iter().collect::<Vec<_>>());
            let want = model
                .get(&probe)
                .map(|ids| ids.iter().copied().collect::<Vec<_>>());
            assert_eq!(got, want, "get {probe:?}");
            for ascending in [true, false] {
                for from in [Bound::Included(&probe), Bound::Excluded(&probe)] {
                    let got: Vec<_> = walked(tree, from, ascending).into_iter().take(30).collect();
                    let want: Vec<_> = expected(model, from, ascending)
                        .into_iter()
                        .take(30)
                        .collect();
                    assert_eq!(got, want, "walk from {from:?}, ascending {ascending}");
                }
            }
        }
    }

    #[test]
    fn random_operations_match_a_btreemap() {
        for (seed, spread) in [(1u64, 40u64), (2, 3_000), (3, 60_000)] {
            let mut rng = Lcg(seed);
            let mut probes = Lcg(seed + 100);
            let mut probe = || key(&mut probes, spread);
            let mut tree = ValueTree::new();
            let mut model = Model::new();
            for step in 0..40_000 {
                let k = key(&mut rng, spread);
                let id = rng.below(8) as i64 + (step % 3) * 1000;
                match rng.below(10) {
                    0..=5 => {
                        let unique = rng.below(4) == 0 && !k.is_null();
                        let holder = model.get(&k);
                        let taken =
                            unique && holder.is_some_and(|ids| ids.iter().any(|&o| o != id));
                        let got = tree.insert(&k, id, unique);
                        if taken {
                            assert_eq!(got, Err(KeyTaken));
                        } else {
                            assert_eq!(got, Ok(model.entry(k).or_default().insert(id)));
                        }
                    }
                    6..=8 => {
                        let mut ids: Vec<i64> = (0..rng.below(4) + 1)
                            .map(|_| rng.below(8) as i64 + rng.below(3) as i64 * 1000)
                            .collect();
                        ids.sort_unstable();
                        ids.dedup();
                        let removed = model
                            .get_mut(&k)
                            .map_or(0, |set| ids.iter().filter(|id| set.remove(id)).count());
                        if model.get(&k).is_some_and(BTreeSet::is_empty) {
                            model.remove(&k);
                        }
                        assert_eq!(tree.remove_sorted(&k, &ids), removed);
                    }
                    _ => {
                        if let Some(first) = model
                            .keys()
                            .nth(rng.below(model.len() as u64 + 1) as usize)
                            .cloned()
                        {
                            let ids: Vec<i64> =
                                model.remove(&first).into_iter().flatten().collect();
                            assert_eq!(tree.remove_sorted(&first, &ids), ids.len());
                        }
                    }
                }
                if step % 4_000 == 0 {
                    check(&tree, &model, &mut probe);
                } else if step % 50 == 0 && tree.len() < 3_000 {
                    check_structure(&tree);
                }
            }
            check(&tree, &model, &mut probe);
            let keys: Vec<Value> = model.keys().cloned().collect();
            for (n, k) in keys.into_iter().enumerate() {
                let ids: Vec<i64> = model.remove(&k).into_iter().flatten().collect();
                assert_eq!(tree.remove_sorted(&k, &ids), ids.len());
                if n % 50 == 0 && tree.len() < 3_000 {
                    check_structure(&tree);
                }
            }
            assert!(tree.len() == 0 && tree.roots.iter().all(Option::is_none));
        }
    }

    #[test]
    fn a_tree_built_from_entries_matches_the_model_and_takes_more_writes() {
        for (seed, n, spread) in [
            (11u64, 0usize, 10u64),
            (12, 1, 10),
            (13, 47, 1_000),
            (14, 48, 1_000),
            (15, 49, 1_000),
            (16, 5_000, 40),
            (17, 60_000, 60_000),
        ] {
            let mut rng = Lcg(seed);
            let mut probes = Lcg(seed + 100);
            let mut probe = || key(&mut probes, spread);
            let mut model = Model::new();
            let keys: Vec<Value> = (0..n).map(|_| key(&mut rng, spread)).collect();
            let entries: Vec<(&Value, i64)> = keys
                .iter()
                .zip(0..)
                .map(|(k, id)| (k, id % 7_000))
                .collect();
            for &(k, id) in &entries {
                model.entry(k.clone()).or_default().insert(id);
            }
            let mut tree = ValueTree::from_entries(entries);
            check(&tree, &model, &mut probe);
            for step in 0..4_000i64 {
                let k = key(&mut rng, spread);
                let id = step % 9;
                if rng.below(2) == 0 {
                    assert_eq!(
                        tree.insert(&k, id, false),
                        Ok(model.entry(k).or_default().insert(id))
                    );
                } else {
                    let removed = model.get_mut(&k).is_some_and(|ids| ids.remove(&id));
                    if model.get(&k).is_some_and(BTreeSet::is_empty) {
                        model.remove(&k);
                    }
                    assert_eq!(tree.remove(&k, id), removed);
                }
            }
            check(&tree, &model, &mut probe);
        }
    }

    #[test]
    fn a_tree_of_floats_alone_built_from_entries_matches_the_model() {
        let float = |rng: &mut Lcg| {
            let n = rng.below(400);
            match n % 8 {
                0 => Value::Float(
                    [f64::NAN, -0.0, 0.0, f64::INFINITY, f64::NEG_INFINITY][n as usize % 5],
                ),
                _ => Value::Float(n as f64 * 0.37 - 70.0),
            }
        };
        let mut rng = Lcg(21);
        let mut probes = Lcg(22);
        let mut probe = || float(&mut probes);
        let keys: Vec<Value> = (0..20_000).map(|_| float(&mut rng)).collect();
        let mut model = Model::new();
        let entries: Vec<(&Value, i64)> = keys.iter().zip(0..).collect();
        for &(k, id) in &entries {
            model.entry(k.clone()).or_default().insert(id);
        }
        let tree = ValueTree::from_entries(entries);
        check(&tree, &model, &mut probe);
    }

    #[test]
    fn a_leap_second_sorts_between_its_minute_and_the_next() {
        let ts = |secs: i64, nanos: u32| {
            Value::Timestamp(
                chrono::DateTime::from_timestamp(secs, nanos).expect("a valid timestamp"),
            )
        };
        let (before, leap, midnight, after) = (
            ts(1_483_228_799, 0),
            ts(1_483_228_799, 1_500_000_000),
            ts(1_483_228_800, 0),
            ts(1_483_228_801, 0),
        );
        let mut tree = ValueTree::new();
        for (id, key) in [&midnight, &after, &before, &leap].into_iter().enumerate() {
            assert_eq!(tree.insert(key, id as i64, false), Ok(true));
        }
        check_structure(&tree);
        let walked: Vec<Value> = walked(&tree, Bound::Included(&leap), true)
            .into_iter()
            .map(|(key, _)| key)
            .collect();
        assert_eq!(walked, [leap.clone(), midnight.clone(), after]);
        assert_eq!(tree.get(&midnight).map(|ids| ids.len()), Some(1));
    }

    #[test]
    fn a_scrambled_fill_then_a_drain_in_any_order_keeps_every_node_balanced() {
        const KEYS: i64 = 60_000;
        let drains: [fn(i64) -> i64; 3] = [|n| n * 104_729 % KEYS, |n| n, |n| KEYS - 1 - n];
        let kinds: [fn(i64) -> Value; 2] =
            [Value::Integer, |n| Value::text(format!("key-{n:036}"))];
        for kind in kinds {
            for drain in drains {
                let (mut tree, mut model) = (ValueTree::new(), Model::new());
                let mut probes = Lcg(7);
                let mut probe = || kind(probes.below(KEYS as u64 + 10) as i64 - 5);
                for i in 0..KEYS {
                    let key = kind(i * 7_919 % KEYS);
                    assert_eq!(tree.insert(&key, i, false), Ok(true));
                    model.entry(key).or_default().insert(i);
                }
                check(&tree, &model, &mut probe);
                for n in 0..KEYS {
                    let key = kind(drain(n));
                    let ids: Vec<i64> = model.remove(&key).into_iter().flatten().collect();
                    assert_eq!(tree.remove_sorted(&key, &ids), 1);
                    if n % 500 == 0 {
                        check_structure(&tree);
                    }
                    if n % 10_000 == 0 {
                        check(&tree, &model, &mut probe);
                    }
                }
                assert!(tree.len() == 0 && tree.roots.iter().all(Option::is_none));
            }
        }
    }
}
