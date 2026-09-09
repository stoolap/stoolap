// Copyright 2026 Stoolap Contributors
// Licensed under the Apache License, Version 2.0.

//! Capacity owners for transaction buffers. These measure allocation lifetime;
//! admission and fallible reservation are added with the pressure-seal stages.

use std::ops::{Deref, DerefMut};

use smallvec::{Array, SmallVec};

use crate::common::memory::{MemoryAccount, MemoryCharge};

pub(crate) struct RetainedSmallVec<A: Array> {
    values: SmallVec<A>,
    // Fields drop in order: the buffer must be freed before its charge.
    charge: MemoryCharge,
}

impl<A: Array> RetainedSmallVec<A> {
    pub(crate) fn new(account: &MemoryAccount) -> Self {
        Self::from_inner(SmallVec::new(), account)
    }

    pub(crate) fn from_inner(values: SmallVec<A>, account: &MemoryAccount) -> Self {
        let bytes = Self::bytes(&values);
        Self {
            values,
            charge: MemoryCharge::new(account, bytes),
        }
    }

    fn bytes(values: &SmallVec<A>) -> usize {
        if values.spilled() {
            values.capacity() * std::mem::size_of::<A::Item>()
        } else {
            0
        }
    }

    pub(crate) fn push(&mut self, value: A::Item) {
        if self.values.len() == self.values.capacity() {
            let capacity = self
                .values
                .capacity()
                .checked_mul(2)
                .expect("transaction buffer capacity overflow")
                .max(if self.values.capacity() == 0 { 4 } else { 1 });
            let replacement = SmallVec::with_capacity(capacity);
            let new_bytes = Self::bytes(&replacement);
            self.charge.resize(self.charge.bytes() + new_bytes);
            let old = std::mem::replace(&mut self.values, replacement);
            self.values.extend(old);
            self.charge.resize(new_bytes);
        }
        self.values.push(value);
    }

    pub(crate) fn pop(&mut self) -> Option<A::Item> {
        self.values.pop()
    }
    pub(crate) fn clear(&mut self) {
        self.values.clear();
    }
    pub(crate) fn as_slice(&self) -> &[A::Item] {
        self.values.as_slice()
    }
    #[cfg(test)]
    pub(crate) fn allocation_size(&self) -> usize {
        self.charge.bytes()
    }
}

impl<A: Array> Deref for RetainedSmallVec<A> {
    type Target = [A::Item];
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl<A: Array> DerefMut for RetainedSmallVec<A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}

impl<A: Array> Extend<A::Item> for RetainedSmallVec<A> {
    fn extend<T: IntoIterator<Item = A::Item>>(&mut self, values: T) {
        for value in values {
            self.push(value);
        }
    }
}

impl<'a, A: Array> IntoIterator for &'a RetainedSmallVec<A> {
    type Item = &'a A::Item;
    type IntoIter = std::slice::Iter<'a, A::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

impl<'a, A: Array> IntoIterator for &'a mut RetainedSmallVec<A> {
    type Item = &'a mut A::Item;
    type IntoIter = std::slice::IterMut<'a, A::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.values.iter_mut()
    }
}

pub(crate) struct RetainedSmallVecIntoIter<A: Array> {
    values: smallvec::IntoIter<A>,
    _charge: MemoryCharge,
}

impl<A: Array> IntoIterator for RetainedSmallVec<A> {
    type Item = A::Item;
    type IntoIter = RetainedSmallVecIntoIter<A>;
    fn into_iter(self) -> Self::IntoIter {
        RetainedSmallVecIntoIter {
            values: self.values.into_iter(),
            _charge: self.charge,
        }
    }
}

impl<A: Array> Iterator for RetainedSmallVecIntoIter<A> {
    type Item = A::Item;
    fn next(&mut self) -> Option<Self::Item> {
        self.values.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.values.size_hint()
    }
}

impl<A: Array> ExactSizeIterator for RetainedSmallVecIntoIter<A> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_growth_clear_and_owned_iterator_keep_physical_capacity_charged() {
        let account = MemoryAccount::new();
        let mut values = RetainedSmallVec::<[u64; 1]>::new(&account);
        values.push(1);
        assert_eq!(account.snapshot().retained_bytes, 0);
        values.push(2);
        assert_eq!(account.snapshot().retained_bytes, 16);
        values.push(3);
        assert_eq!(account.snapshot().retained_bytes, 32);
        values.extend([4, 5]);
        let base = account.snapshot().conservative_bytes;
        assert_eq!(account.snapshot().peak_accounted_bytes, base + 96);
        values.clear();
        assert_eq!(account.snapshot().retained_bytes, 64);
        values.extend([4, 5]);
        let mut iter = values.into_iter();
        assert_eq!(iter.next(), Some(4));
        assert_eq!(account.snapshot().retained_bytes, 64);
        drop(iter);
        assert_eq!(account.snapshot().retained_bytes, 0);
    }
}
