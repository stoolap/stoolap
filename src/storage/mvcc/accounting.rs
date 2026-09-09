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
    charge: Option<MemoryCharge>,
}

impl<A: Array> RetainedSmallVec<A> {
    pub(crate) fn new() -> Self {
        Self {
            values: SmallVec::new(),
            charge: None,
        }
    }

    pub(crate) fn from_inner(values: SmallVec<A>, account: &MemoryAccount) -> Self {
        let bytes = Self::bytes(&values);
        Self {
            values,
            charge: (bytes != 0).then(|| MemoryCharge::new(account, bytes)),
        }
    }

    fn bytes(values: &SmallVec<A>) -> usize {
        if values.spilled() {
            values.capacity() * std::mem::size_of::<A::Item>()
        } else {
            0
        }
    }

    /// Inline storage belongs to the enclosing owner and retains no account.
    /// The first spill selects an origin; later growth keeps that same origin.
    pub(crate) fn push(&mut self, value: A::Item, account: &MemoryAccount) {
        if self.values.len() == self.values.capacity() {
            let capacity = self
                .values
                .capacity()
                .checked_mul(2)
                .expect("transaction buffer capacity overflow")
                .max(if self.values.capacity() == 0 { 4 } else { 1 });
            let replacement = SmallVec::with_capacity(capacity);
            let new_bytes = Self::bytes(&replacement);
            let charge = self
                .charge
                .get_or_insert_with(|| MemoryCharge::new(account, 0));
            charge.resize(charge.bytes() + new_bytes);
            let old = std::mem::replace(&mut self.values, replacement);
            self.values.extend(old);
            charge.resize(new_bytes);
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
    pub(crate) fn extend<T: IntoIterator<Item = A::Item>>(
        &mut self,
        values: T,
        account: &MemoryAccount,
    ) {
        for value in values {
            self.push(value, account);
        }
    }
    #[cfg(test)]
    pub(crate) fn allocation_size(&self) -> usize {
        self.charge.as_ref().map_or(0, MemoryCharge::bytes)
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
    _charge: Option<MemoryCharge>,
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
    fn inline_values_do_not_keep_the_supplied_origin_alive() {
        let root = MemoryAccount::new();
        let empty = root.snapshot().accounted_bytes;
        let origin = root.child();
        let mut values = RetainedSmallVec::<[u64; 2]>::new();
        values.push(7, &origin);
        values.extend([9], &origin);
        let copied = RetainedSmallVec::from_inner(SmallVec::<[u64; 2]>::from_slice(&[11]), &origin);
        drop(origin);
        // A zero-byte MemoryCharge would keep the child's own allocation alive.
        // Inline values need neither that owner nor a separately charged buffer.
        assert_eq!(root.snapshot().accounted_bytes, empty);
        let mut iter = values.into_iter();
        assert_eq!(iter.next(), Some(7));
        assert_eq!(iter.next(), Some(9));
        assert_eq!(copied.as_slice(), &[11]);
        drop((iter, copied));
        assert_eq!(root.snapshot().accounted_bytes, empty);
    }

    #[test]
    fn spilled_buffer_and_iterator_keep_the_first_origin_through_final_drop() {
        let root = MemoryAccount::new();
        let empty = root.snapshot().accounted_bytes;
        let origin = root.child();
        let origin_bytes = root.snapshot().accounted_bytes - empty;
        let other_root = MemoryAccount::new();
        let other_empty = other_root.snapshot();
        let mut values = RetainedSmallVec::<[u64; 1]>::new();
        values.extend([7, 9], &origin);
        drop(origin);
        assert_eq!(root.snapshot().accounted_bytes, empty + origin_bytes + 16);
        // Moving the vector does not reparent its existing allocation when a
        // later caller supplies a different account at the growth seam.
        values.push(11, &other_root);
        assert_eq!(other_root.snapshot(), other_empty);
        assert_eq!(root.snapshot().accounted_bytes, empty + origin_bytes + 32);
        let mut iter = values.into_iter();
        assert_eq!(iter.next(), Some(7));
        assert_eq!(root.snapshot().accounted_bytes, empty + origin_bytes + 32);
        drop(iter);
        assert_eq!(root.snapshot().accounted_bytes, empty);
    }

    #[test]
    fn interrupted_extend_keeps_initialized_values_and_spill_ownership() {
        let root = MemoryAccount::new();
        let empty = root.snapshot().accounted_bytes;
        let mut values = RetainedSmallVec::<[u64; 1]>::new();
        let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            values.extend(
                (0..3).inspect(|&value| {
                    assert!(value < 2, "input iterator failed");
                }),
                &root,
            );
        }));
        assert!(failure.is_err());
        assert_eq!(values.as_slice(), &[0, 1]);
        assert_eq!(root.snapshot().accounted_bytes, empty + 16);
        drop(values);
        assert_eq!(root.snapshot().accounted_bytes, empty);
    }

    #[test]
    fn inline_growth_clear_and_owned_iterator_keep_physical_capacity_charged() {
        let account = MemoryAccount::new();
        let mut values = RetainedSmallVec::<[u64; 1]>::new();
        values.push(1, &account);
        assert_eq!(account.snapshot().retained_bytes, 0);
        values.push(2, &account);
        assert_eq!(account.snapshot().retained_bytes, 16);
        values.push(3, &account);
        assert_eq!(account.snapshot().retained_bytes, 32);
        values.extend([4, 5], &account);
        let base = account.snapshot().conservative_bytes;
        assert_eq!(account.snapshot().peak_accounted_bytes, base + 96);
        values.clear();
        assert_eq!(account.snapshot().retained_bytes, 64);
        values.extend([4, 5], &account);
        let mut iter = values.into_iter();
        assert_eq!(iter.next(), Some(4));
        assert_eq!(account.snapshot().retained_bytes, 64);
        drop(iter);
        assert_eq!(account.snapshot().retained_bytes, 0);
    }
}
