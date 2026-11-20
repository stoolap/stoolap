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

//! The engine's hasher for in-memory keys. Every integer word hashes as
//! `I64Map` places an integer key; bytes hash as one word of wide
//! multiplies, read at once up to 8 bytes and 16 at a time beyond. Not for
//! anything written to disk

use std::hash::Hasher;

const MIX: u64 = 0x517cc1b727220a95;
const SEED: [u64; 2] = [0x243f6a8885a308d3, 0x13198a2e03707344];

/// One multiply, its top half folded down so low bits see every input bit
#[inline(always)]
fn mix(word: u64) -> u64 {
    let word = word.wrapping_mul(MIX);
    word ^ (word >> 32)
}

/// Multiplication carries entropy only upwards, so a key's bits above 16,
/// mixed, are added to it: the keys of one 2^16 block share that term and
/// keep distinct slots, while every block gets a term of its own
#[inline(always)]
fn spread(key: u64) -> u64 {
    key.wrapping_add(mix(key >> 16))
}

/// The hash of an integer key alone, whose low bits are its slot
#[inline(always)]
pub fn slot_hash(key: i64) -> u64 {
    spread(key as u64).wrapping_mul(MIX)
}

/// Both halves of the full product, folded into one word
#[inline(always)]
fn folded_multiply(a: u64, b: u64) -> u64 {
    let full = (a as u128).wrapping_mul(b as u128);
    (full as u64) ^ ((full >> 64) as u64)
}

#[inline(always)]
fn read_u64(bytes: &[u8], at: usize) -> u64 {
    let mut word = [0u8; 8];
    word.copy_from_slice(&bytes[at..at + 8]);
    u64::from_le_bytes(word)
}

#[inline(always)]
fn read_u32(bytes: &[u8], at: usize) -> u64 {
    let mut word = [0u8; 4];
    word.copy_from_slice(&bytes[at..at + 4]);
    u32::from_le_bytes(word) as u64
}

/// Up to 8 bytes as one word, its length in the top bits; reads overlap
#[inline(always)]
fn short_word(bytes: &[u8]) -> u64 {
    let len = bytes.len();
    let word = if len >= 4 {
        read_u32(bytes, 0) | (read_u32(bytes, len - 4) << 32)
    } else if len > 0 {
        let ends = ((bytes[0] as u64) << 16) | ((bytes[len / 2] as u64) << 8);
        ends | bytes[len - 1] as u64
    } else {
        0
    };
    word ^ ((len as u64) << 59)
}

/// More than 8 bytes as one mixed word: 16 at a time in two streams, the
/// last 16 read where they end
#[inline(always)]
fn long_word(bytes: &[u8]) -> u64 {
    let len = bytes.len();
    let (mut s0, mut s1) = (SEED[0], SEED[1]);
    if len <= 16 {
        s0 ^= read_u64(bytes, 0);
        s1 ^= read_u64(bytes, len - 8);
    } else {
        let mut at = 0;
        while at + 16 < len {
            let mixed = folded_multiply(s0 ^ read_u64(bytes, at), MIX ^ read_u64(bytes, at + 8));
            (s0, s1) = (s1, mixed);
            at += 16;
        }
        s0 ^= read_u64(bytes, len - 16);
        s1 ^= read_u64(bytes, len - 8);
    }
    folded_multiply(s0, s1) ^ len as u64
}

/// Fx's step over words that are spread first
#[derive(Clone, Copy, Default)]
pub struct KeyHasher {
    hash: u64,
}

impl KeyHasher {
    #[inline(always)]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(MIX);
    }
}

impl Hasher for KeyHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        if bytes.len() <= 8 {
            self.add(folded_multiply(short_word(bytes) ^ SEED[0], MIX));
        } else {
            self.add(long_word(bytes));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.write_u64(i as u64);
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.write_u64(i as u64);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.write_u64(i as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(spread(i));
    }

    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.write_u64(i as u64);
    }

    #[inline]
    fn write_u128(&mut self, i: u128) {
        self.write_u64(i as u64);
        self.write_u64((i >> 64) as u64);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

/// For a `u64` key that already is a mixed hash: the key is its own hash.
/// Any other input goes through `KeyHasher`
#[derive(Clone, Copy, Default)]
pub struct HashedKey {
    hash: u64,
}

impl Hasher for HashedKey {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut hasher = KeyHasher { hash: self.hash };
        hasher.write(bytes);
        self.hash = hasher.finish();
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.hash = self.hash.rotate_left(5) ^ i;
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{BuildHasher, BuildHasherDefault};

    fn hash_of<K: std::hash::Hash>(key: K) -> u64 {
        BuildHasherDefault::<KeyHasher>::default().hash_one(key)
    }

    /// Distinct buckets among 2^17 taken from the low bits, as hashbrown does
    fn buckets<K: std::hash::Hash>(keys: impl Iterator<Item = K>) -> usize {
        let seen: std::collections::HashSet<u64> =
            keys.map(|k| hash_of(k) & ((1 << 17) - 1)).collect();
        seen.len()
    }

    #[test]
    fn an_integer_hashes_as_i64_map_places_it() {
        for key in [0, 1, -1, 7, 1 << 32, i64::MIN, i64::MAX, 1_700_000_000_000] {
            assert_eq!(hash_of(key), slot_hash(key), "{key}");
            assert_eq!(hash_of(key as u64), slot_hash(key), "{key} as u64");
            let word = key as usize;
            assert_eq!(hash_of(word), slot_hash(word as i64), "{key} as usize");
        }
    }

    #[test]
    fn a_hashed_key_is_its_own_hash() {
        let state = BuildHasherDefault::<HashedKey>::default();
        for hash in [0u64, 1, u64::MAX, 0x9e37_79b9_7f4a_7c15] {
            assert_eq!(state.hash_one(hash), hash);
        }
    }

    #[test]
    fn keys_that_differ_only_high_up_never_collapse_into_few_buckets() {
        let n: i64 = 1 << 16;
        let floor = n as usize / 8;
        for (shape, got) in [
            ("dense", buckets(0..n)),
            ("multiples of 2^16", buckets((0..n).map(|k| k << 16))),
            ("multiples of 2^32", buckets((0..n).map(|k| k << 32))),
            ("n * (2^32 + 1)", buckets((0..n).map(|k| k * 4_294_967_297))),
            ("n * (2^16 + 1)", buckets((0..n).map(|k| k * 65_537))),
            ("n * (2^8 + 1)", buckets((0..n).map(|k| k * 257))),
            (
                "n * (2^32 + 2^48)",
                buckets((0..n).map(|k| k.wrapping_mul(0x0001_0001_0000_0000))),
            ),
            (
                "a clock on top",
                buckets((0..n).map(|ms| (1_700_000_000_000 + ms) << 22 | 5)),
            ),
            (
                "pairs differing high up",
                buckets((0..n).map(|k| (7, k << 40))),
            ),
        ] {
            assert!(got > floor, "{shape}: {got} buckets for {n} keys");
        }
    }

    #[test]
    fn texts_get_distinct_buckets_at_every_length() {
        let n = 1 << 16;
        let half = n / 2;
        for len in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100] {
            let texts = (0..n).map(|i| {
                let mut text = vec![b'c'; len];
                for (at, byte) in format!("{i:x}").bytes().rev().enumerate() {
                    if at < len {
                        text[len - 1 - at] = byte;
                    }
                }
                text
            });
            let distinct: std::collections::HashSet<Vec<u8>> = texts.clone().collect();
            let got = buckets(texts);
            assert!(
                got > half.min(distinct.len() / 2),
                "{len} bytes: {got} buckets for {} texts",
                distinct.len()
            );
        }
    }

    #[test]
    fn texts_that_share_a_prefix_or_differ_in_length_hash_apart() {
        let names = ["col_a", "col_b", "col_", "col_a\0", "a", "a\0", "", "\0"];
        let hashes: std::collections::HashSet<u64> = names.iter().map(hash_of).collect();
        assert_eq!(hashes.len(), names.len(), "{names:?}");
        let long = "x".repeat(40);
        let mut other = long.clone().into_bytes();
        other[20] = b'y';
        assert_ne!(hash_of(long.as_bytes()), hash_of(other.as_slice()));
    }
}
