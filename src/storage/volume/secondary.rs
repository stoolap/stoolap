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

//! Prototype: a secondary index over one integer column of a sealed volume.
//! Sorted distinct keys, and for each key the ascending row positions that
//! hold it, kept in one positions array sliced by an offsets array. Built
//! from the volume's column at seal or compaction, written next to the
//! volume as a side file, loaded whole on first use after a reopen.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rustc_hash::FxHashMap;

use super::writer::FrozenVolume;

const MAGIC: [u8; 4] = *b"STSX";
const VERSION: u32 = 1;
const KEY_I64: u8 = 1;
pub const SIDE_EXT: &str = "sidx";

/// Counters the prototype's measurement reads; every lookup adds to them.
pub struct Counters {
    /// Volumes a lookup looked at after the bloom filter
    pub volumes_examined: AtomicU64,
    /// Index probes made (a volume with an index for the column)
    pub probes: AtomicU64,
    /// Candidate positions the probes returned
    pub candidates: AtomicU64,
    /// Candidates that passed visibility and the residual filter
    pub visible: AtomicU64,
    /// Row groups the candidate fetch decoded
    pub groups_fetched: AtomicU64,
    /// Volumes read through the scan because they had no index for the column
    pub unindexed_volumes: AtomicU64,
    /// Indexes loaded from a side file
    pub loads: AtomicU64,
    /// Indexes built from a column
    pub builds: AtomicU64,
    /// Nanoseconds spent building, including the sort
    pub build_ns: AtomicU64,
    /// The largest sort workspace one build reserved, in bytes
    pub build_workspace_peak: AtomicU64,
    /// Bytes of index structures resident right now
    pub resident_bytes: AtomicU64,
    /// Nanoseconds the seal's write fence was held, over all seals
    pub seal_fence_ns: AtomicU64,
    /// Seals that held the fence
    pub seal_fences: AtomicU64,
}

pub static COUNTERS: Counters = Counters {
    seal_fence_ns: AtomicU64::new(0),
    seal_fences: AtomicU64::new(0),
    volumes_examined: AtomicU64::new(0),
    probes: AtomicU64::new(0),
    candidates: AtomicU64::new(0),
    visible: AtomicU64::new(0),
    groups_fetched: AtomicU64::new(0),
    unindexed_volumes: AtomicU64::new(0),
    loads: AtomicU64::new(0),
    builds: AtomicU64::new(0),
    build_ns: AtomicU64::new(0),
    build_workspace_peak: AtomicU64::new(0),
    resident_bytes: AtomicU64::new(0),
};

impl Counters {
    pub fn snapshot(&self) -> [(&'static str, u64); 13] {
        [
            ("seal_fence_ns", self.seal_fence_ns.load(Ordering::Relaxed)),
            ("seal_fences", self.seal_fences.load(Ordering::Relaxed)),
            (
                "volumes_examined",
                self.volumes_examined.load(Ordering::Relaxed),
            ),
            ("probes", self.probes.load(Ordering::Relaxed)),
            ("candidates", self.candidates.load(Ordering::Relaxed)),
            ("visible", self.visible.load(Ordering::Relaxed)),
            (
                "groups_fetched",
                self.groups_fetched.load(Ordering::Relaxed),
            ),
            (
                "unindexed_volumes",
                self.unindexed_volumes.load(Ordering::Relaxed),
            ),
            ("loads", self.loads.load(Ordering::Relaxed)),
            ("builds", self.builds.load(Ordering::Relaxed)),
            ("build_ns", self.build_ns.load(Ordering::Relaxed)),
            (
                "build_workspace_peak",
                self.build_workspace_peak.load(Ordering::Relaxed),
            ),
            (
                "resident_bytes",
                self.resident_bytes.load(Ordering::Relaxed),
            ),
        ]
    }
}

/// One column's index of one volume.
pub struct ColdColumnIndex {
    keys: Vec<i64>,
    /// `offsets[i]..offsets[i + 1]` slices `positions` for `keys[i]`
    offsets: Vec<u32>,
    positions: Vec<u32>,
}

impl ColdColumnIndex {
    /// Builds from the column's non-null values, `(key, position)` pairs
    /// sorted once; the pair vector is the build's workspace.
    pub fn build(values: impl Iterator<Item = (u32, i64)>) -> Self {
        let started = std::time::Instant::now();
        let mut pairs: Vec<(i64, u32)> = values.map(|(pos, key)| (key, pos)).collect();
        let workspace = (pairs.capacity() * std::mem::size_of::<(i64, u32)>()) as u64;
        COUNTERS
            .build_workspace_peak
            .fetch_max(workspace, Ordering::Relaxed);
        pairs.sort_unstable();
        let mut keys: Vec<i64> = Vec::new();
        let mut offsets: Vec<u32> = vec![0];
        let mut positions: Vec<u32> = Vec::with_capacity(pairs.len());
        for (key, pos) in pairs {
            if keys.last() != Some(&key) {
                keys.push(key);
                offsets.push(positions.len() as u32);
            }
            positions.push(pos);
            *offsets.last_mut().expect("offsets holds the open key") = positions.len() as u32;
        }
        let index = Self {
            keys,
            offsets,
            positions,
        };
        COUNTERS.builds.fetch_add(1, Ordering::Relaxed);
        COUNTERS
            .build_ns
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        COUNTERS
            .resident_bytes
            .fetch_add(index.bytes() as u64, Ordering::Relaxed);
        index
    }

    /// From one column of a loaded volume, group by group.
    pub fn build_from_volume(volume: &FrozenVolume, col_idx: usize) -> std::io::Result<Self> {
        let mut pairs: Vec<(u32, i64)> = Vec::with_capacity(volume.meta.row_count);
        if let Some(column) = volume.columns.resident(col_idx) {
            for row in 0..column.len() {
                if !column.is_null(row) {
                    pairs.push((row as u32, column.get_i64(row)));
                }
            }
        } else {
            let store = volume.columns.compressed_store().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "column data is not loaded")
            })?;
            let groups = store.num_groups(col_idx);
            for group in 0..groups {
                let column = store.group_column(col_idx, group)?;
                let base = group * super::column::ROW_GROUP_SIZE;
                for row in 0..column.len() {
                    if !column.is_null(row) {
                        pairs.push(((base + row) as u32, column.get_i64(row)));
                    }
                }
            }
        }
        Ok(Self::build(pairs.into_iter()))
    }

    /// The positions holding `key`, ascending.
    pub fn equal(&self, key: i64) -> &[u32] {
        match self.keys.binary_search(&key) {
            Ok(i) => &self.positions[self.offsets[i] as usize..self.offsets[i + 1] as usize],
            Err(_) => &[],
        }
    }

    /// The positions of every key in `[low, high]`, in key order and ascending
    /// within a key; the caller sorts when it needs volume order.
    pub fn range(&self, low: i64, high: i64) -> &[u32] {
        if low > high || self.keys.is_empty() {
            return &[];
        }
        let first = self.keys.partition_point(|&k| k < low);
        let last = self.keys.partition_point(|&k| k <= high);
        if first >= last {
            return &[];
        }
        &self.positions[self.offsets[first] as usize..self.offsets[last] as usize]
    }

    pub fn distinct_keys(&self) -> usize {
        self.keys.len()
    }

    pub fn bytes(&self) -> usize {
        self.keys.len() * 8 + self.offsets.len() * 4 + self.positions.len() * 4
    }

    fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.keys.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.positions.len() as u64).to_le_bytes());
        for k in &self.keys {
            out.extend_from_slice(&k.to_le_bytes());
        }
        for o in &self.offsets {
            out.extend_from_slice(&o.to_le_bytes());
        }
        for p in &self.positions {
            out.extend_from_slice(&p.to_le_bytes());
        }
    }

    fn read_from(data: &[u8], pos: &mut usize) -> std::io::Result<Self> {
        let n_keys = read_u64(data, pos)? as usize;
        let n_pos = read_u64(data, pos)? as usize;
        let mut keys = Vec::with_capacity(n_keys);
        for _ in 0..n_keys {
            keys.push(read_u64(data, pos)? as i64);
        }
        let mut offsets = Vec::with_capacity(n_keys + 1);
        for _ in 0..=n_keys {
            offsets.push(read_u32(data, pos)?);
        }
        let mut positions = Vec::with_capacity(n_pos);
        for _ in 0..n_pos {
            positions.push(read_u32(data, pos)?);
        }
        if offsets.first() != Some(&0) || offsets.last().copied() != Some(n_pos as u32) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "side index offsets do not cover its positions",
            ));
        }
        let index = Self {
            keys,
            offsets,
            positions,
        };
        COUNTERS
            .resident_bytes
            .fetch_add(index.bytes() as u64, Ordering::Relaxed);
        Ok(index)
    }
}

impl Drop for ColdColumnIndex {
    fn drop(&mut self) {
        COUNTERS
            .resident_bytes
            .fetch_sub(self.bytes() as u64, Ordering::Relaxed);
    }
}

fn read_u64(data: &[u8], pos: &mut usize) -> std::io::Result<u64> {
    let end = *pos + 8;
    let bytes = data.get(*pos..end).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "side index truncated")
    })?;
    *pos = end;
    Ok(u64::from_le_bytes(bytes.try_into().expect("8 bytes")))
}

fn read_u32(data: &[u8], pos: &mut usize) -> std::io::Result<u32> {
    let end = *pos + 4;
    let bytes = data.get(*pos..end).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "side index truncated")
    })?;
    *pos = end;
    Ok(u32::from_le_bytes(bytes.try_into().expect("4 bytes")))
}

/// The side file next to a volume file.
pub fn side_path(volume_path: &Path) -> PathBuf {
    volume_path.with_extension(SIDE_EXT)
}

/// Writes every index of a volume into its side file: header, one entry per
/// column, a CRC over everything before it. Written whole into a temporary
/// file and renamed into place.
pub fn write_side_file(path: &Path, indexes: &[(usize, &ColdColumnIndex)]) -> std::io::Result<()> {
    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(indexes.len() as u32).to_le_bytes());
    for (col_idx, index) in indexes {
        out.extend_from_slice(&(*col_idx as u32).to_le_bytes());
        out.push(KEY_I64);
        index.write_to(&mut out);
    }
    let crc = crc32fast::hash(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    let tmp = path.with_extension(format!("{SIDE_EXT}.tmp"));
    {
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(&out)?;
        file.sync_all()?;
    }
    std::fs::rename(&tmp, path)
}

/// Reads a side file whole; every index of the volume comes back.
pub fn read_side_file(path: &Path) -> std::io::Result<FxHashMap<usize, ColdColumnIndex>> {
    let mut data = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut data)?;
    if data.len() < 16 || data[..4] != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not a side index file",
        ));
    }
    let body = &data[..data.len() - 4];
    let stored = u32::from_le_bytes(data[data.len() - 4..].try_into().expect("4 bytes"));
    if crc32fast::hash(body) != stored {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "side index CRC mismatch",
        ));
    }
    let mut pos = 4;
    let version = read_u32(body, &mut pos)?;
    if version != VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "side index version unsupported",
        ));
    }
    let count = read_u32(body, &mut pos)? as usize;
    let mut indexes = FxHashMap::default();
    for _ in 0..count {
        let col_idx = read_u32(body, &mut pos)? as usize;
        let tag = *body.get(pos).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "side index truncated")
        })?;
        pos += 1;
        if tag != KEY_I64 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "side index key type unsupported",
            ));
        }
        indexes.insert(col_idx, ColdColumnIndex::read_from(body, &mut pos)?);
    }
    COUNTERS.loads.fetch_add(1, Ordering::Relaxed);
    Ok(indexes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_group_their_positions_in_order() {
        let index = ColdColumnIndex::build(
            [(0u32, 5i64), (1, 3), (2, 5), (3, 9), (4, 3), (5, 5)].into_iter(),
        );
        assert_eq!(index.equal(5), &[0, 2, 5]);
        assert_eq!(index.equal(3), &[1, 4]);
        assert_eq!(index.equal(4), &[] as &[u32]);
        assert_eq!(index.range(3, 5), &[1, 4, 0, 2, 5]);
        assert_eq!(index.range(6, 8), &[] as &[u32]);
        assert_eq!(index.range(9, 100), &[3]);
        assert_eq!(index.distinct_keys(), 3);
    }

    #[test]
    fn a_side_file_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vol_1.sidx");
        let a = ColdColumnIndex::build([(0u32, 7i64), (1, 7), (2, 1)].into_iter());
        let b = ColdColumnIndex::build([(0u32, -4i64)].into_iter());
        write_side_file(&path, &[(1, &a), (2, &b)]).unwrap();
        let read = read_side_file(&path).unwrap();
        assert_eq!(read[&1].equal(7), &[0, 1]);
        assert_eq!(read[&1].equal(1), &[2]);
        assert_eq!(read[&2].equal(-4), &[0]);
        assert_eq!(read.len(), 2);
    }
}
