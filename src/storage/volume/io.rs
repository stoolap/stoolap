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

//! File I/O for frozen volumes.
//!
//! Handles writing volumes to disk and reading them back.
//! Volumes are written atomically (write to .tmp, then rename) to prevent
//! corruption from crashes during writes.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};

use crate::core::Result;

use super::column::ROW_GROUP_SIZE;
use super::format::{deserialize_volume_metadata, serialize_volume_metadata};
use super::writer::{CompressedBlockStore, FrozenVolume, LazyColumns};

/// Immutable identity plus a managed location, with no idle file descriptor.
/// Residency transitions share this allocation. Only managed directory renames
/// may change its location; every open still validates the captured identity.
pub(crate) struct VolumeFile {
    path: parking_lot::RwLock<PathBuf>,
    identity: FileIdentity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct FileIdentity {
    device: u64,
    #[cfg(not(windows))]
    file: u64,
    #[cfg(windows)]
    file: [u8; 16],
    length: u64,
}

impl FileIdentity {
    fn capture(file: &std::fs::File) -> std::io::Result<Self> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let metadata = file.metadata()?;
            Ok(Self {
                device: metadata.dev(),
                file: metadata.ino(),
                length: metadata.len(),
            })
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            use windows_sys::Win32::Storage::FileSystem::{
                FileIdInfo, GetFileInformationByHandleEx, FILE_ID_INFO,
            };
            let mut information = FILE_ID_INFO::default();
            // SAFETY: File owns a valid live handle, and information points to
            // writable, correctly aligned storage of exactly the advertised
            // size for FileIdInfo. The synchronous API does not retain it.
            let success = unsafe {
                GetFileInformationByHandleEx(
                    file.as_raw_handle(),
                    FileIdInfo,
                    (&mut information as *mut FILE_ID_INFO).cast(),
                    std::mem::size_of::<FILE_ID_INFO>() as u32,
                )
            };
            if success == 0 {
                return Err(std::io::Error::last_os_error());
            }
            // ReFS needs the full 128-bit ID. Unsupported/unknown IDs must
            // never collapse distinct files onto the same fallback identity.
            let file_id = information.FileId.Identifier;
            if file_id == [0; 16] || file_id == [u8::MAX; 16] {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "filesystem did not provide a stable volume file identity",
                ));
            }
            Ok(Self {
                device: information.VolumeSerialNumber,
                file: file_id,
                length: file.metadata()?.len(),
            })
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = file;
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "persistent volumes require stable file identities on this platform",
            ))
        }
    }
}

impl VolumeFile {
    fn from_file(path: &Path, file: &std::fs::File) -> std::io::Result<Arc<Self>> {
        Ok(Arc::new(Self {
            path: parking_lot::RwLock::new(path.to_path_buf()),
            identity: FileIdentity::capture(file)?,
        }))
    }

    fn open_reader(self: &Arc<Self>) -> std::io::Result<VolumeReadLease> {
        // A managed rename holds the write guard through rename + path update.
        // The active file handle then remains valid independently of the path.
        let path = self.path.read();
        let file = std::fs::File::open(&*path)?;
        if FileIdentity::capture(&file)? != self.identity {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "captured volume file identity changed",
            ));
        }
        Ok(VolumeReadLease {
            file,
            backing: self.clone(),
        })
    }
}

/// One decoder owns one open handle. Its backing owner also prevents a
/// retirement sweep from unlinking the file until this read has completed.
struct VolumeReadLease {
    file: std::fs::File,
    backing: Arc<VolumeFile>,
}

struct RetiredVolume {
    backing: Weak<VolumeFile>,
    path: PathBuf,
    identity: FileIdentity,
}

/// Fully allocated before a destructive WAL operation. Linking this node after
/// success is allocation-free; dropping an uncommitted node never schedules IO.
pub(crate) struct PreparedRetirement {
    entries: Vec<RetiredVolume>,
    pub(crate) omitted_segment_ids: Vec<u64>,
    pub(crate) next: Option<Box<PreparedRetirement>>,
}

impl PreparedRetirement {
    /// Allocate every replacement before the directory is renamed. Parked
    /// batches are owned by the manager until durable omission grants cleanup.
    pub(crate) fn prepare_relocation(&self, old: &Path, new: &Path) -> Vec<Option<PathBuf>> {
        self.entries
            .iter()
            .map(|entry| {
                entry
                    .path
                    .strip_prefix(old)
                    .ok()
                    .map(|suffix| new.join(suffix))
            })
            .collect()
    }

    pub(crate) fn apply_relocation(&mut self, replacements: Vec<Option<PathBuf>>) {
        debug_assert_eq!(self.entries.len(), replacements.len());
        for (entry, replacement) in self.entries.iter_mut().zip(replacements) {
            if let Some(replacement) = replacement {
                entry.path = replacement;
            }
        }
    }
}

impl Drop for PreparedRetirement {
    fn drop(&mut self) {
        // Repeated failed manifest writes can retain several exact batches.
        // Release only metadata here, iteratively, without stack growth or IO.
        let mut next = self.next.take();
        while let Some(mut batch) = next {
            next = batch.next.take();
        }
    }
}

#[derive(Default)]
struct FileCatalog {
    // One inline Weak for the normal one-descriptor case. Independent decodes
    // registered in this engine remain aliases of the same physical identity.
    live: rustc_hash::FxHashMap<FileIdentity, smallvec::SmallVec<[Weak<VolumeFile>; 1]>>,
    pending: Vec<RetiredVolume>,
    prepared: Option<Box<PreparedRetirement>>,
}

impl FileCatalog {
    fn track(&mut self, backing: &Arc<VolumeFile>) {
        let aliases = self.live.entry(backing.identity).or_default();
        aliases.retain(|alias| alias.strong_count() != 0);
        let weak = Arc::downgrade(backing);
        if !aliases.iter().any(|alias| Weak::ptr_eq(alias, &weak)) {
            aliases.push(weak);
        }
    }
}

/// Engine-owned deferred cleanup. Enqueue and sweep happen outside transfer
/// fences; neither a generation nor a file descriptor performs I/O on Drop.
/// Dropping the engine with live readers may leave an orphan for startup cleanup.
#[derive(Default)]
pub(crate) struct VolumeRetirementQueue {
    catalog: parking_lot::Mutex<FileCatalog>,
}

impl VolumeRetirementQueue {
    /// Register before publishing a volume into any manager generation. This
    /// authority is independent of whether its file is eligible for deletion.
    pub(crate) fn track(&self, backing: &Arc<VolumeFile>) {
        self.catalog.lock().track(backing);
    }

    pub(crate) fn retire(&self, backing: &Arc<VolumeFile>) {
        let mut catalog = self.catalog.lock();
        catalog.track(backing);
        catalog.pending.push(RetiredVolume {
            backing: Arc::downgrade(backing),
            path: backing.path.read().clone(),
            identity: backing.identity,
        });
    }

    pub(crate) fn prepare_retirement<'a>(
        &self,
        backings: impl Iterator<Item = &'a Arc<VolumeFile>>,
    ) -> Box<PreparedRetirement> {
        let mut catalog = self.catalog.lock();
        let entries = backings
            .map(|backing| {
                catalog.track(backing);
                RetiredVolume {
                    backing: Arc::downgrade(backing),
                    path: backing.path.read().clone(),
                    identity: backing.identity,
                }
            })
            .collect();
        Box::new(PreparedRetirement {
            entries,
            omitted_segment_ids: Vec::new(),
            next: None,
        })
    }

    /// Exact batch eligibility is granted after truncate WAL and a durable
    /// manifest image omitting its old identities. Storage already exists.
    pub(crate) fn commit_retirement(&self, mut batch: Box<PreparedRetirement>) {
        let mut catalog = self.catalog.lock();
        batch.next = catalog.prepared.take();
        catalog.prepared = Some(batch);
    }

    pub(crate) fn sweep(&self) {
        let mut catalog = self.catalog.lock();
        let FileCatalog {
            live,
            pending,
            prepared,
        } = &mut *catalog;
        let keep = |entry: &RetiredVolume| {
            if entry.backing.strong_count() != 0
                || live
                    .get(&entry.identity)
                    .is_some_and(|aliases| aliases.iter().any(|alias| alias.strong_count() != 0))
            {
                return true;
            }
            // No strong owner can reappear after the count reaches zero.
            // Never knowingly unlink a different file placed at the old path.
            let file = match std::fs::File::open(&entry.path) {
                Ok(file) => file,
                Err(error) => return error.kind() != std::io::ErrorKind::NotFound,
            };
            let identity = FileIdentity::capture(&file);
            drop(file);
            match identity {
                Ok(identity) if identity == entry.identity => {
                    if let Err(error) = std::fs::remove_file(&entry.path) {
                        return error.kind() != std::io::ErrorKind::NotFound;
                    }
                    if let Some(parent) = entry.path.parent() {
                        let _ = std::fs::remove_dir(parent);
                    }
                    false
                }
                Ok(_) => false,
                Err(_) => true,
            }
        };
        pending.retain(keep);
        let mut batches = prepared.take();
        let mut retained = None;
        while let Some(mut batch) = batches {
            batches = batch.next.take();
            batch.entries.retain(keep);
            if !batch.entries.is_empty() {
                batch.next = retained;
                retained = Some(batch);
            }
        }
        *prepared = retained;
        live.retain(|_, aliases| {
            aliases.retain(|alias| alias.strong_count() != 0);
            !aliases.is_empty()
        });
        if pending.is_empty() {
            pending.shrink_to_fit();
        }
    }

    /// Serialize a managed parent rename with reader opens and retirement.
    /// The caller serializes seal/compaction before collecting current volumes.
    pub(crate) fn rename_directory(
        &self,
        old: &Path,
        new: &Path,
        current: &[Arc<FrozenVolume>],
    ) -> std::io::Result<()> {
        let mut catalog = self.catalog.lock();
        for volume in current {
            if let Some(backing) = volume.backing.get() {
                catalog.track(backing);
            }
        }
        let mut backings: Vec<_> = current
            .iter()
            .filter_map(|volume| volume.backing.get().cloned())
            .chain(
                catalog
                    .live
                    .values()
                    .flat_map(|aliases| aliases.iter().filter_map(Weak::upgrade)),
            )
            .collect();
        backings.sort_unstable_by_key(Arc::as_ptr);
        backings.dedup_by(|left, right| Arc::ptr_eq(left, right));
        let mut locations: Vec<_> = backings.iter().map(|file| file.path.write()).collect();
        // Allocate replacement paths before changing anything on disk.
        let replacements: Vec<_> = locations
            .iter()
            .map(|path| path.strip_prefix(old).ok().map(|suffix| new.join(suffix)))
            .collect();
        let FileCatalog {
            pending, prepared, ..
        } = &mut *catalog;
        let mut entries: Vec<&mut RetiredVolume> = pending.iter_mut().collect();
        let mut batch = prepared.as_mut();
        while let Some(current) = batch {
            entries.extend(current.entries.iter_mut());
            batch = current.next.as_mut();
        }
        let retired_replacements: Vec<_> = entries
            .iter()
            .map(|entry| {
                entry
                    .path
                    .strip_prefix(old)
                    .ok()
                    .map(|suffix| new.join(suffix))
            })
            .collect();
        std::fs::rename(old, new)?;
        for (path, replacement) in locations.iter_mut().zip(replacements) {
            if let Some(replacement) = replacement {
                **path = replacement;
            }
        }
        for (entry, replacement) in entries.into_iter().zip(retired_replacements) {
            if let Some(replacement) = replacement {
                entry.path = replacement;
            }
        }
        Ok(())
    }
}

/// Volume file extension
const VOLUME_EXT: &str = "vol";

/// Magic bytes for V4 per-column per-group compressed format.
const V4_MAGIC: [u8; 4] = *b"STV4";

/// V4 format version. Bump when the metadata or block layout changes.
const V4_VERSION: u32 = 1;

/// Volume catalog filename
const CATALOG_FILE: &str = "volumes.catalog";

/// Write a frozen volume to disk atomically (V4 format, LZ4 compressed).
pub fn write_volume_to_disk(
    dir: &Path,
    table_name: &str,
    volume_id: u64,
    volume: &FrozenVolume,
) -> Result<PathBuf> {
    let (path, _store) = write_volume_to_disk_opts(dir, table_name, volume_id, volume, true)?;
    Ok(path)
}

/// Write a frozen volume to disk atomically, with optional LZ4 compression.
///
/// Always writes V4 format (per-column per-group blocks with CRC32).
/// When `compress` is true, blocks are LZ4-compressed (blocks that don't
/// compress well are stored raw automatically). When false, all blocks
/// are stored raw (same V4 layout, no LZ4 overhead).
/// Returns (path, CompressedBlockStore).
pub fn write_volume_to_disk_opts(
    dir: &Path,
    table_name: &str,
    volume_id: u64,
    volume: &FrozenVolume,
    compress: bool,
) -> Result<(PathBuf, CompressedBlockStore)> {
    let table_dir = dir.join(table_name);
    std::fs::create_dir_all(&table_dir)
        .map_err(|e| crate::core::Error::internal(format!("failed to create volume dir: {}", e)))?;

    let filename = format!("vol_{:016x}.{}", volume_id, VOLUME_EXT);
    let final_path = table_dir.join(&filename);
    let tmp_path = table_dir.join(format!("{}.tmp", filename));

    let (data, store) = serialize_v4_opts(volume, compress)
        .map_err(|e| crate::core::Error::internal(format!("V4 serialize failed: {}", e)))?;

    let file = {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)
            .map_err(|e| {
                crate::core::Error::internal(format!("failed to create volume tmp file: {}", e))
            })?;
        f.write_all(&data).map_err(|e| {
            crate::core::Error::internal(format!("failed to write volume file: {}", e))
        })?;
        f.sync_all().map_err(|e| {
            crate::core::Error::internal(format!("failed to fsync volume tmp file: {}", e))
        })?;
        f
    };
    drop(data);

    std::fs::rename(&tmp_path, &final_path).map_err(|e| {
        crate::core::Error::internal(format!("failed to rename volume file: {}", e))
    })?;

    #[cfg(not(windows))]
    if let Ok(d) = std::fs::File::open(&table_dir) {
        d.sync_all().map_err(|e| {
            crate::core::Error::internal(format!("failed to fsync volume directory: {}", e))
        })?;
    }

    // Capture the exact written handle; never reopen the published path here.
    let backing = VolumeFile::from_file(&final_path, &file).map_err(|e| {
        crate::core::Error::internal(format!("failed to bind written volume: {}", e))
    })?;
    drop(file);
    // A volume is immutable. A later copy to another path cannot replace the
    // backing already held by existing captured readers.
    let _ = volume.backing.set(backing);

    Ok((final_path, store))
}

/// Serialize a FrozenVolume to V4 format.
///
/// Layout:
/// ```text
/// [STV4 (4)] [version (4)] [col_count (4)] [num_groups (4)] [meta_compressed_len (4)]
/// [LZ4(metadata)]
/// [block_index: (compressed_len: u64, decompressed_len: u64) * col_count * num_groups]
/// [LZ4 blocks: col_0_grp_0, col_0_grp_1, ..., col_N_grp_G]
/// [CRC32 (4)]
/// ```
/// Serialize a volume to V4 format bytes with LZ4 compression.
pub fn serialize_v4_public(vol: &FrozenVolume) -> std::io::Result<(Vec<u8>, CompressedBlockStore)> {
    serialize_v4_opts(vol, true)
}

/// Returns (file_bytes, CompressedBlockStore). The store can be used to register
/// a lazy volume without re-reading from disk.
fn serialize_v4_opts(
    vol: &FrozenVolume,
    compress: bool,
) -> std::io::Result<(Vec<u8>, CompressedBlockStore)> {
    use std::io::Write;

    let col_count = vol.columns.len();
    let group_size = ROW_GROUP_SIZE;
    let num_groups = if vol.meta.row_count == 0 {
        0
    } else {
        vol.meta.row_count.div_ceil(group_size)
    };

    // 1. Serialize + compress metadata
    let meta_raw = serialize_volume_metadata(vol)?;
    let meta_compressed = lz4_flex::compress_prepend_size(&meta_raw);
    drop(meta_raw);

    // 2. Build CompressedBlockStore (compresses all column blocks when enabled)
    let store = CompressedBlockStore::compress_columns_opts(
        &vol.columns,
        &vol.meta.column_types,
        vol.meta.row_count,
        compress,
    )?;

    // 3. Compute total size for pre-allocation
    let all_blocks = store.raw_blocks();
    let all_decomp_lens = store.decompressed_lens();
    let block_lens_size = col_count * num_groups * 16;
    let total_block_bytes: usize = all_blocks
        .iter()
        .flat_map(|c| c.iter())
        .map(|b| b.len())
        .sum();
    let total_size = 20 + meta_compressed.len() + block_lens_size + total_block_bytes + 4;
    let mut buf = Vec::with_capacity(total_size);

    // 4. Fixed header (20 bytes)
    buf.write_all(&V4_MAGIC)?;
    buf.write_all(&V4_VERSION.to_le_bytes())?;
    buf.write_all(&(col_count as u32).to_le_bytes())?;
    buf.write_all(&(num_groups as u32).to_le_bytes())?;
    buf.write_all(&(meta_compressed.len() as u32).to_le_bytes())?;

    // 5. Compressed metadata
    buf.write_all(&meta_compressed)?;
    drop(meta_compressed);

    // 6. Block index: (compressed_len: u64, decompressed_len: u64) pairs
    for (col_blocks, col_decomp) in all_blocks.iter().zip(all_decomp_lens.iter()) {
        for (block, &decomp_len) in col_blocks.iter().zip(col_decomp.iter()) {
            buf.write_all(&(block.len() as u64).to_le_bytes())?;
            buf.write_all(&(decomp_len as u64).to_le_bytes())?;
        }
    }

    // 7. Block data
    for col_blocks in all_blocks {
        for block in col_blocks {
            buf.write_all(block)?;
        }
    }

    // 8. Trailing CRC32
    let crc = crc32fast::hash(&buf);
    buf.write_all(&crc.to_le_bytes())?;

    Ok((buf, store))
}

/// Read a V4 volume via streaming I/O. Never holds the full file in memory.
/// CRC32 is computed incrementally as sections are read.
/// Blocks are read one at a time into a reusable buffer and decompressed
/// directly into final column vectors. No intermediate compressed storage.
pub(crate) fn read_volume_file(backing: Arc<VolumeFile>) -> Result<FrozenVolume> {
    let lease = backing.open_reader().map_err(|e| {
        crate::core::Error::internal(format!("failed to open captured volume: {}", e))
    })?;
    read_volume_lease(lease)
}

fn read_volume_lease(lease: VolumeReadLease) -> Result<FrozenVolume> {
    use std::io::Read;

    let inv = |msg: &str| crate::core::Error::internal(format!("V4: {}", msg));

    let file_len = usize::try_from(lease.backing.identity.length)
        .map_err(|_| inv("file length exceeds address space"))?;
    if file_len < 24 {
        return Err(inv("file too small"));
    }

    let VolumeReadLease { file, backing } = lease;
    // Each decode owns its File, so the seek position cannot be shared.
    let mut reader = std::io::BufReader::new(file);
    let mut hasher = crc32fast::Hasher::new();

    // Helper: read exact bytes and feed to CRC
    macro_rules! crc_read {
        ($buf:expr) => {{
            reader
                .read_exact($buf)
                .map_err(|e| crate::core::Error::internal(format!("V4 read: {}", e)))?;
            hasher.update($buf);
        }};
    }

    // 1. Fixed header (20 bytes)
    let mut header = [0u8; 20];
    crc_read!(&mut header);

    if header[0..4] != V4_MAGIC {
        return Err(inv("bad magic"));
    }
    let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
    if version != V4_VERSION {
        return Err(inv(&format!("unsupported version {}", version)));
    }
    let col_count = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
    let num_groups = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
    let meta_len = u32::from_le_bytes(header[16..20].try_into().unwrap()) as usize;

    let total_blocks = col_count
        .checked_mul(num_groups)
        .ok_or_else(|| inv("block count overflow"))?;
    let index_bytes = total_blocks
        .checked_mul(16)
        .ok_or_else(|| inv("block index overflow"))?;
    let data_start = 20usize
        .checked_add(meta_len)
        .and_then(|n| n.checked_add(index_bytes))
        .filter(|&n| n <= file_len - 4)
        .ok_or_else(|| inv("metadata/index exceeds file bounds"))?;
    if meta_len < 4 {
        return Err(inv("metadata too short for LZ4 size prefix"));
    }

    // 2. Compressed metadata (read into temp buffer, decompress, drop)
    let mut meta_compressed = vec![0u8; meta_len];
    crc_read!(&mut meta_compressed);

    // Parse prepended uncompressed size (4 bytes LE), then decompress_into
    // to avoid lz4_flex::decompress_size_prepended allocating a fresh Vec.
    let meta_raw = if meta_compressed.len() >= 4 {
        let uncomp_size = u32::from_le_bytes(meta_compressed[..4].try_into().unwrap()) as usize;
        if uncomp_size
            > meta_compressed
                .len()
                .saturating_sub(4)
                .saturating_mul(255)
                .saturating_add(16)
        {
            return Err(inv("impossible metadata LZ4 decoded size"));
        }
        let mut buf = Vec::new();
        buf.try_reserve_exact(uncomp_size)
            .map_err(|_| inv("metadata allocation failed"))?;
        buf.resize(uncomp_size, 0);
        let written = lz4_flex::decompress_into(&meta_compressed[4..], &mut buf)
            .map_err(|e| inv(&format!("metadata LZ4: {}", e)))?;
        if written != uncomp_size {
            return Err(inv("metadata LZ4 decoded length mismatch"));
        }
        drop(meta_compressed);
        buf
    } else {
        drop(meta_compressed);
        return Err(inv("metadata too short for LZ4 size prefix"));
    };
    let meta = deserialize_volume_metadata(&meta_raw)
        .map_err(|e| crate::core::Error::internal(format!("V4 metadata: {}", e)))?;
    drop(meta_raw);

    if meta.col_type_tags.len() != col_count {
        return Err(inv(&format!(
            "col_count mismatch: header={}, metadata={}",
            col_count,
            meta.col_type_tags.len()
        )));
    }

    if meta.row_count.div_ceil(ROW_GROUP_SIZE) != num_groups {
        return Err(inv("header row group count mismatch"));
    }
    // 3. Block index: (compressed_len: u64, decompressed_len: u64) pairs
    let mut index_buf = vec![0u8; index_bytes];
    crc_read!(&mut index_buf);

    let mut compressed_lens = Vec::with_capacity(total_blocks);
    let mut decompressed_lens_flat = Vec::with_capacity(total_blocks);
    let mut block_end = data_start;
    for i in 0..total_blocks {
        let off = i * 16;
        let stored = usize::try_from(u64::from_le_bytes(
            index_buf[off..off + 8].try_into().unwrap(),
        ))
        .map_err(|_| inv("stored block length exceeds address space"))?;
        let decoded = usize::try_from(u64::from_le_bytes(
            index_buf[off + 8..off + 16].try_into().unwrap(),
        ))
        .map_err(|_| inv("decoded block length exceeds address space"))?;
        block_end = block_end
            .checked_add(stored)
            .filter(|&n| n <= file_len - 4)
            .ok_or_else(|| inv("column block exceeds file bounds"))?;
        if decoded == 0 || stored == 0 || decoded > stored.saturating_mul(255).saturating_add(16) {
            return Err(inv("impossible column block decoded size"));
        }
        let ci = i / num_groups;
        let gi = i % num_groups;
        let rows = (meta.row_count - gi * ROW_GROUP_SIZE).min(ROW_GROUP_SIZE);
        let width = match meta.col_type_tags[ci] {
            super::format::COL_INT64
            | super::format::COL_FLOAT64
            | super::format::COL_TIMESTAMP => Some(9usize),
            super::format::COL_BOOLEAN => Some(2),
            super::format::COL_DICTIONARY => Some(5),
            _ => None,
        };
        if width.is_some_and(|w| rows.checked_mul(w) != Some(decoded)) {
            return Err(inv("fixed-width column block size mismatch"));
        }
        compressed_lens.push(stored);
        decompressed_lens_flat.push(decoded);
    }
    if block_end != file_len - 4 {
        return Err(inv("trailing or missing volume bytes"));
    }
    drop(index_buf);

    let col_data_types = meta.column_types.clone();
    let group_size = ROW_GROUP_SIZE;

    // 5. Read compressed blocks into CompressedBlockStore (deferred mode).
    //    Blocks stay compressed in RAM. Decompression happens on first scan
    //    via the group cache path (~4 GB/s from RAM, ~1ms per column per group).
    //    Each block is read into a buffer then moved into the store.
    let mut all_blocks: Vec<Vec<Vec<u8>>> = Vec::with_capacity(col_count);
    let mut all_decomp_lens: Vec<Vec<usize>> = Vec::with_capacity(col_count);
    let mut block_idx = 0usize;

    for _col_idx in 0..col_count {
        let mut col_blocks = Vec::with_capacity(num_groups);
        let mut col_lens = Vec::with_capacity(num_groups);
        for _gi in 0..num_groups {
            let comp_len = compressed_lens[block_idx];
            let decomp_len = decompressed_lens_flat[block_idx];
            let mut block = vec![0u8; comp_len];
            crc_read!(&mut block);
            block_idx += 1;
            col_blocks.push(block);
            col_lens.push(decomp_len);
        }
        all_blocks.push(col_blocks);
        all_decomp_lens.push(col_lens);
    }

    // 6. Verify CRC32 (computed incrementally over everything we read)
    let mut crc_buf = [0u8; 4];
    reader
        .read_exact(&mut crc_buf)
        .map_err(|e| crate::core::Error::internal(format!("V4 CRC read: {}", e)))?;
    let stored_crc = u32::from_le_bytes(crc_buf);
    if hasher.finalize() != stored_crc {
        return Err(inv("CRC mismatch"));
    }

    // 7. Build CompressedBlockStore + deferred LazyColumns.
    //    Columns start cold (compressed in RAM). First scan decompresses
    //    per-group on demand. After all columns are accessed, is_eager flips
    //    to true (automatic hot promotion).
    let dict_ranges: Vec<(usize, usize, usize)> = {
        let mut ranges = Vec::new();
        let mut offset = 0usize;
        for (i, &count) in meta.col_dict_counts.iter().enumerate() {
            let count = count as usize;
            if count > 0 {
                ranges.push((i, offset, offset + count));
                offset += count;
            }
        }
        ranges
    };
    let store = CompressedBlockStore::from_raw_blocks(
        all_blocks,
        all_decomp_lens,
        meta.col_type_tags.clone(),
        meta.column_types.clone(),
        meta.col_ext_types.clone(),
        meta.shared_dict,
        dict_ranges,
        group_size,
        meta.row_count,
    )?;
    let columns = LazyColumns::deferred(store, col_data_types);

    Ok(FrozenVolume {
        backing: std::sync::OnceLock::from(backing.clone()),
        columns,
        meta: Arc::new(super::writer::VolumeMeta {
            zone_maps: meta.zone_maps,
            bloom_filters: meta.bloom_filters,
            stats: meta.stats,
            row_count: meta.row_count,
            column_names: meta.column_names,
            column_types: meta.column_types,
            row_ids: meta.row_ids,
            sorted_columns: meta.col_sorted,
            column_name_map: meta.column_name_map,
            row_groups: meta.row_groups,
        }),
        unique_indices: std::sync::Arc::new(parking_lot::RwLock::new(
            rustc_hash::FxHashMap::default(),
        )),
        last_access_epoch: std::sync::atomic::AtomicU64::new(
            super::writer::GLOBAL_EVICTION_EPOCH.load(std::sync::atomic::Ordering::Relaxed),
        ),
    })
}

/// Read a frozen volume from disk. Only V4 (STV4) format is supported.
pub fn read_volume_from_disk(path: &Path) -> Result<FrozenVolume> {
    let file = std::fs::File::open(path).map_err(|e| {
        crate::core::Error::internal(format!("failed to open volume {:?}: {}", path, e))
    })?;
    let backing = VolumeFile::from_file(path, &file).map_err(|e| {
        crate::core::Error::internal(format!("failed to bind volume {:?}: {}", path, e))
    })?;
    read_volume_lease(VolumeReadLease { file, backing })
}

/// List all volume files for a table, sorted by volume ID (oldest first).
pub fn list_volumes(dir: &Path, table_name: &str) -> Vec<PathBuf> {
    let table_dir = dir.join(table_name);
    let mut volumes: Vec<PathBuf> = match std::fs::read_dir(&table_dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e == VOLUME_EXT)
                    .unwrap_or(false)
            })
            .collect(),
        Err(_) => return Vec::new(),
    };
    volumes.sort(); // Sorted by filename = sorted by volume ID (hex)
    volumes
}

/// Load all volumes for a table from disk.
pub fn load_all_volumes(dir: &Path, table_name: &str) -> Result<Vec<Arc<FrozenVolume>>> {
    let paths = list_volumes(dir, table_name);
    let mut volumes = Vec::with_capacity(paths.len());
    for path in paths {
        let vol = read_volume_from_disk(&path)?;
        volumes.push(Arc::new(vol));
    }
    Ok(volumes)
}

/// Delete a specific volume file from disk.
pub fn delete_volume(path: &Path) -> Result<()> {
    std::fs::remove_file(path).map_err(|e| {
        crate::core::Error::internal(format!("failed to delete volume {:?}: {}", path, e))
    })
}

/// Delete all volumes for a table.
pub fn delete_all_volumes(dir: &Path, table_name: &str) -> Result<()> {
    let paths = list_volumes(dir, table_name);
    for path in paths {
        delete_volume(&path)?;
    }
    // Remove the table directory if empty
    let table_dir = dir.join(table_name);
    let _ = std::fs::remove_dir(&table_dir); // OK if not empty
    Ok(())
}

/// Generate a new volume ID. Monotonically increasing, unique across calls.
/// Uses microseconds since epoch + CAS loop for uniqueness.
pub fn next_volume_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let micros = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;
    loop {
        let current = COUNTER.load(Ordering::Acquire);
        // New ID is at least micros, or current+1 if clock hasn't advanced
        let candidate = if micros > current {
            micros
        } else {
            current + 1
        };
        match COUNTER.compare_exchange_weak(current, candidate, Ordering::AcqRel, Ordering::Relaxed)
        {
            Ok(_) => return candidate,
            Err(_) => continue,
        }
    }
}

/// Simple volume catalog that tracks which volumes exist for each table.
///
/// This is a lightweight metadata file that allows the engine to know
/// which volumes to load without scanning the filesystem.
#[derive(Debug, Clone)]
pub struct VolumeCatalog {
    /// Volume entries per table: (volume_id, row_count, time_min_micros, time_max_micros)
    pub tables: ahash::AHashMap<String, Vec<VolumeEntry>>,
}

/// Metadata for a single volume.
#[derive(Debug, Clone)]
pub struct VolumeEntry {
    /// Unique volume identifier (timestamp-based)
    pub volume_id: u64,
    /// Number of rows in this volume
    pub row_count: u64,
    /// Minimum timestamp in micros (for time-range pruning without loading)
    pub time_min_micros: i64,
    /// Maximum timestamp in micros
    pub time_max_micros: i64,
}

impl VolumeCatalog {
    /// Create an empty catalog.
    pub fn new() -> Self {
        Self {
            tables: ahash::AHashMap::new(),
        }
    }

    /// Add a volume entry for a table.
    pub fn add_volume(&mut self, table_name: &str, entry: VolumeEntry) {
        self.tables
            .entry(table_name.to_string())
            .or_default()
            .push(entry);
    }

    /// Get volume entries for a table.
    pub fn get_volumes(&self, table_name: &str) -> &[VolumeEntry] {
        self.tables
            .get(table_name)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Serialize the catalog to bytes with trailing CRC32.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"STVC"); // SToolap Volume Catalog
        buf.extend_from_slice(&1u32.to_le_bytes()); // version

        let table_count = self.tables.len() as u32;
        buf.extend_from_slice(&table_count.to_le_bytes());

        for (name, entries) in &self.tables {
            let name_bytes = name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(name_bytes);

            buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
            for entry in entries {
                buf.extend_from_slice(&entry.volume_id.to_le_bytes());
                buf.extend_from_slice(&entry.row_count.to_le_bytes());
                buf.extend_from_slice(&entry.time_min_micros.to_le_bytes());
                buf.extend_from_slice(&entry.time_max_micros.to_le_bytes());
            }
        }
        // Trailing CRC32 for integrity validation on load
        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    fn read_u32(data: &[u8], pos: &mut usize) -> std::io::Result<u32> {
        let end = *pos + 4;
        if end > data.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "truncated volume catalog: expected u32",
            ));
        }
        let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
        *pos = end;
        Ok(v)
    }

    fn read_u64(data: &[u8], pos: &mut usize) -> std::io::Result<u64> {
        let end = *pos + 8;
        if end > data.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "truncated volume catalog: expected u64",
            ));
        }
        let v = u64::from_le_bytes([
            data[*pos],
            data[*pos + 1],
            data[*pos + 2],
            data[*pos + 3],
            data[*pos + 4],
            data[*pos + 5],
            data[*pos + 6],
            data[*pos + 7],
        ]);
        *pos = end;
        Ok(v)
    }

    fn read_i64(data: &[u8], pos: &mut usize) -> std::io::Result<i64> {
        let end = *pos + 8;
        if end > data.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "truncated volume catalog: expected i64",
            ));
        }
        let v = i64::from_le_bytes([
            data[*pos],
            data[*pos + 1],
            data[*pos + 2],
            data[*pos + 3],
            data[*pos + 4],
            data[*pos + 5],
            data[*pos + 6],
            data[*pos + 7],
        ]);
        *pos = end;
        Ok(v)
    }

    /// Deserialize a catalog from bytes.
    pub fn deserialize(data: &[u8]) -> std::io::Result<Self> {
        // Minimum: magic(4) + version(4) + table_count(4) + crc(4) = 16
        if data.len() < 16 || &data[0..4] != b"STVC" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid volume catalog",
            ));
        }
        // Verify trailing CRC32
        let payload = &data[..data.len() - 4];
        let stored_crc = u32::from_le_bytes(data[data.len() - 4..].try_into().map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "truncated catalog CRC")
        })?);
        let computed_crc = crc32fast::hash(payload);
        if stored_crc != computed_crc {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "volume catalog CRC mismatch: stored={:#x} computed={:#x}",
                    stored_crc, computed_crc
                ),
            ));
        }
        let mut pos = 4;

        let _version = Self::read_u32(data, &mut pos)?;
        let table_count = Self::read_u32(data, &mut pos)? as usize;

        let mut tables = ahash::AHashMap::new();

        for _ in 0..table_count {
            let name_len = Self::read_u32(data, &mut pos)? as usize;
            if pos + name_len > data.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "truncated volume catalog: table name",
                ));
            }
            let name = std::str::from_utf8(&data[pos..pos + name_len])
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
                .to_string();
            pos += name_len;

            let entry_count = Self::read_u32(data, &mut pos)? as usize;

            let mut entries = Vec::with_capacity(entry_count);
            for _ in 0..entry_count {
                let volume_id = Self::read_u64(data, &mut pos)?;
                let row_count = Self::read_u64(data, &mut pos)?;
                let time_min = Self::read_i64(data, &mut pos)?;
                let time_max = Self::read_i64(data, &mut pos)?;

                entries.push(VolumeEntry {
                    volume_id,
                    row_count,
                    time_min_micros: time_min,
                    time_max_micros: time_max,
                });
            }
            tables.insert(name, entries);
        }

        Ok(Self { tables })
    }

    /// Write catalog to disk atomically.
    pub fn write_to_disk(&self, dir: &Path) -> Result<()> {
        let data = self.serialize();
        let final_path = dir.join(CATALOG_FILE);
        let tmp_path = dir.join(format!("{}.tmp", CATALOG_FILE));

        // Write to tmp file and fsync BEFORE rename for crash safety.
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp_path).map_err(|e| {
                crate::core::Error::internal(format!("failed to create catalog tmp file: {}", e))
            })?;
            f.write_all(&data).map_err(|e| {
                crate::core::Error::internal(format!("failed to write volume catalog: {}", e))
            })?;
            f.sync_all().map_err(|e| {
                crate::core::Error::internal(format!("failed to fsync volume catalog: {}", e))
            })?;
        }

        std::fs::rename(&tmp_path, &final_path).map_err(|e| {
            crate::core::Error::internal(format!("failed to rename volume catalog: {}", e))
        })?;

        // Fsync directory to ensure the rename is durable.
        // Windows does not support opening directories for fsync;
        // NTFS metadata is flushed with the file's sync_all().
        #[cfg(not(windows))]
        {
            let d = std::fs::File::open(dir).map_err(|e| {
                std::io::Error::other(format!("failed to open dir for fsync: {}", e))
            })?;
            d.sync_all()
                .map_err(|e| std::io::Error::other(format!("failed to fsync dir: {}", e)))?;
        }

        Ok(())
    }

    /// Read catalog from disk.
    pub fn read_from_disk(dir: &Path) -> Result<Self> {
        let path = dir.join(CATALOG_FILE);
        if !path.exists() {
            return Ok(Self::new());
        }
        let data = std::fs::read(&path).map_err(|e| {
            crate::core::Error::internal(format!("failed to read volume catalog: {}", e))
        })?;
        Self::deserialize(&data).map_err(|e| {
            crate::core::Error::internal(format!("failed to parse volume catalog: {}", e))
        })
    }
}

impl Default for VolumeCatalog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::writer::VolumeBuilder;
    use super::*;
    use crate::core::{DataType, Row, SchemaBuilder, Value};

    fn file_lease_fixture(dir: &Path, table: &str, id: u64, value: i64) -> (PathBuf, FrozenVolume) {
        let schema = SchemaBuilder::new(table)
            .column("id", DataType::Integer, false, true)
            .build();
        let mut builder = VolumeBuilder::new(&schema);
        builder.add_row(1, &Row::from_values(vec![Value::Integer(value)]));
        let volume = builder.finish();
        let path = write_volume_to_disk(dir, table, id, &volume).unwrap();
        (path, volume)
    }

    #[test]
    fn captured_file_rejects_replacement_but_active_read_keeps_original() {
        let dir = tempfile::tempdir().unwrap();
        let (path, original) = file_lease_fixture(dir.path(), "captured", 1, 42);
        let backing = original.backing.get().unwrap().clone();
        let active = backing.open_reader().unwrap();
        let (replacement, _) = file_lease_fixture(dir.path(), "replacement", 1, 99);
        std::fs::rename(&replacement, &path).unwrap();
        assert_eq!(
            read_volume_from_disk(&path).unwrap().get_row(0).unwrap()[0],
            Value::Integer(99)
        );
        assert!(read_volume_file(backing).is_err());
        let old = read_volume_lease(active).unwrap();
        assert_eq!(old.get_row(0).unwrap()[0], Value::Integer(42));
    }

    #[test]
    fn captured_file_rejects_external_unlink_without_an_active_read() {
        let dir = tempfile::tempdir().unwrap();
        let (path, volume) = file_lease_fixture(dir.path(), "unlinked", 1, 42);
        std::fs::remove_file(path).unwrap();
        assert!(volume.reload_from_backing().is_err());
    }

    #[test]
    fn retirement_waits_for_last_reader_and_never_performs_io_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let (path, volume) = file_lease_fixture(dir.path(), "retired", 1, 42);
        let queue = VolumeRetirementQueue::default();
        let backing = volume.backing.get().unwrap().clone();
        queue.retire(&backing);
        let active = backing.open_reader().unwrap();
        drop((volume, backing));
        queue.sweep();
        assert!(
            path.exists(),
            "an active decoder retains the identity lease"
        );
        let loaded = read_volume_lease(active).unwrap();
        assert_eq!(loaded.get_row(0).unwrap()[0], Value::Integer(42));
        queue.sweep();
        assert!(
            path.exists(),
            "the decoded volume also retains its identity"
        );
        drop(loaded);
        assert!(path.exists(), "Drop must not perform filesystem I/O");
        queue.sweep();
        assert!(!path.exists());
        assert!(queue.catalog.lock().pending.is_empty());
    }

    #[test]
    fn retirement_never_deletes_a_known_replacement() {
        let dir = tempfile::tempdir().unwrap();
        let (path, volume) = file_lease_fixture(dir.path(), "replaced", 1, 42);
        let original_inode = std::fs::File::open(&path).unwrap();
        let queue = VolumeRetirementQueue::default();
        queue.retire(volume.backing.get().unwrap());
        drop(volume);
        let (replacement, _) = file_lease_fixture(dir.path(), "replacement", 1, 99);
        std::fs::rename(&replacement, &path).unwrap();
        queue.sweep();
        assert_eq!(
            read_volume_from_disk(&path).unwrap().get_row(0).unwrap()[0],
            Value::Integer(99)
        );
        assert!(queue.catalog.lock().pending.is_empty());
        drop(original_inode);
    }

    #[test]
    fn retirement_waits_for_independently_decoded_registered_aliases() {
        let dir = tempfile::tempdir().unwrap();
        let (path, first) = file_lease_fixture(dir.path(), "aliases", 1, 42);
        let second = read_volume_from_disk(&path).unwrap();
        assert!(!Arc::ptr_eq(
            first.backing.get().unwrap(),
            second.backing.get().unwrap()
        ));
        let queue = VolumeRetirementQueue::default();
        queue.track(first.backing.get().unwrap());
        queue.track(second.backing.get().unwrap());
        queue.retire(first.backing.get().unwrap());
        drop(first);
        queue.sweep();
        assert!(path.exists());
        assert_eq!(
            second.reload_from_backing().unwrap().get_row(0).unwrap()[0],
            Value::Integer(42)
        );
        drop(second);
        queue.sweep();
        assert!(!path.exists());
    }

    #[test]
    fn engine_queue_drop_with_live_reader_leaves_a_safe_orphan() {
        let dir = tempfile::tempdir().unwrap();
        let (path, volume) = file_lease_fixture(dir.path(), "orphan", 1, 42);
        let queue = VolumeRetirementQueue::default();
        queue.retire(volume.backing.get().unwrap());
        drop(queue);
        assert_eq!(
            volume.reload_from_backing().unwrap().get_row(0).unwrap()[0],
            Value::Integer(42)
        );
        drop(volume);
        assert!(
            path.exists(),
            "startup can later reclaim an unreferenced file"
        );
    }

    #[test]
    fn managed_directory_rename_updates_current_and_retired_readers() {
        let dir = tempfile::tempdir().unwrap();
        let (_, live) = file_lease_fixture(dir.path(), "old", 1, 42);
        let (_, retired) = file_lease_fixture(dir.path(), "old", 2, 99);
        let live = Arc::new({
            let cold = live.to_cold();
            drop(live);
            cold
        });
        let retired = Arc::new({
            let cold = retired.to_cold();
            drop(retired);
            cold
        });
        let queue = VolumeRetirementQueue::default();
        queue.retire(retired.backing.get().unwrap());
        let old = dir.path().join("old");
        let new = dir.path().join("new");
        std::thread::scope(|scope| {
            for volume in [&live, &retired] {
                scope.spawn(move || {
                    for _ in 0..100 {
                        let loaded = volume.reload_from_backing().unwrap();
                        assert!(Arc::ptr_eq(
                            volume.backing.get().unwrap(),
                            loaded.backing.get().unwrap()
                        ));
                    }
                });
            }
            for _ in 0..10 {
                queue
                    .rename_directory(&old, &new, &[live.clone(), live.clone()])
                    .unwrap();
                queue.rename_directory(&new, &old, &[live.clone()]).unwrap();
            }
        });
        queue.rename_directory(&old, &new, &[live.clone()]).unwrap();
        assert_eq!(
            retired.reload_from_backing().unwrap().get_row(0).unwrap()[0],
            Value::Integer(99)
        );
        drop(retired);
        queue.sweep();
        assert!(!new.join("vol_0000000000000002.vol").exists());
        assert!(new.join("vol_0000000000000001.vol").exists());
    }

    #[test]
    fn failed_managed_rename_keeps_original_locations() {
        let dir = tempfile::tempdir().unwrap();
        let (path, live) = file_lease_fixture(dir.path(), "old", 1, 42);
        let live = Arc::new(live);
        let queue = VolumeRetirementQueue::default();
        queue.retire(live.backing.get().unwrap());
        assert!(queue
            .rename_directory(
                &dir.path().join("old"),
                &dir.path().join("missing/new"),
                &[live.clone()]
            )
            .is_err());
        assert_eq!(*live.backing.get().unwrap().path.read(), path);
        assert_eq!(queue.catalog.lock().pending[0].path, path);
        assert_eq!(
            live.reload_from_backing().unwrap().get_row(0).unwrap()[0],
            Value::Integer(42)
        );
    }

    #[cfg(unix)]
    #[test]
    fn idle_descriptor_count_does_not_grow_open_file_count() {
        const CHILD_MARKER: &str = "STOOLAP_FILE_LEASE_FD_PROBE";
        if std::env::var_os(CHILD_MARKER).is_none() {
            // Other parallel unit tests may open files in this process.
            // The child runs only this probe, so its FD delta is attributable.
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "idle_descriptor_count_does_not_grow_open_file_count",
                    "--test-threads=1",
                    "--nocapture",
                ])
                .env(CHILD_MARKER, "1")
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
            return;
        }
        fn fd_count() -> usize {
            std::fs::read_dir("/dev/fd").unwrap().count()
        }
        let dir = tempfile::tempdir().unwrap();
        let (path, _) = file_lease_fixture(dir.path(), "descriptors", 1, 42);
        let before = fd_count();
        let backings: Vec<_> = (0..512)
            .map(|_| {
                let file = std::fs::File::open(&path).unwrap();
                VolumeFile::from_file(&path, &file).unwrap()
            })
            .collect();
        assert!(
            fd_count() <= before + 1,
            "metadata identities do not retain descriptors"
        );
        let readers: Vec<_> = backings
            .iter()
            .take(8)
            .map(|b| b.open_reader().unwrap())
            .collect();
        assert!(
            fd_count() >= before + 8,
            "only active reads open descriptors"
        );
        drop(readers);
        assert!(fd_count() <= before + 1);
        drop(backings);
    }

    #[test]
    fn residency_reload_preserves_aliases_unique_cache_and_backing_identity() {
        let dir = tempfile::tempdir().unwrap();
        let (_, mut volume) = file_lease_fixture(dir.path(), "aliases", 1, 42);
        volume.merge_column_rename("renamed", "id");
        volume.prebuild_unique_index(&[0]).unwrap();
        let cold = volume.to_cold();
        let loaded = cold.reload_from_backing().unwrap();
        assert!(Arc::ptr_eq(&cold.meta, &loaded.meta));
        assert!(Arc::ptr_eq(&cold.unique_indices, &loaded.unique_indices));
        assert!(Arc::ptr_eq(
            cold.backing.get().unwrap(),
            loaded.backing.get().unwrap()
        ));
        assert_eq!(loaded.column_index("renamed"), Some(0));
        assert!(!loaded.unique_indices.read().is_empty());
        assert_eq!(loaded.get_row(0).unwrap()[0], Value::Integer(42));
    }

    #[test]
    fn malformed_v4_lengths_fail_before_allocating_or_decoding() {
        let schema = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .build();
        let mut builder = VolumeBuilder::new(&schema);
        builder.add_row(1, &Row::from_values(vec![Value::Integer(1)]));
        let volume = builder.finish();
        let (original, _) = serialize_v4_public(&volume).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.vol");
        let meta_len = u32::from_le_bytes(original[16..20].try_into().unwrap()) as usize;
        let index = 20 + meta_len;
        let mut cases = Vec::new();
        let mut bad = original.clone();
        bad[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        cases.push(bad);
        let mut bad = original.clone();
        bad[index..index + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        cases.push(bad);
        let mut bad = original.clone();
        bad[index + 8..index + 16].copy_from_slice(&u64::MAX.to_le_bytes());
        cases.push(bad);
        let mut bad = original.clone();
        bad[20..24].copy_from_slice(&u32::MAX.to_le_bytes());
        cases.push(bad);
        let mut bad = original;
        bad.push(0);
        cases.push(bad);
        for bad in cases {
            std::fs::write(&path, bad).unwrap();
            assert!(read_volume_from_disk(&path).is_err());
        }
    }

    #[test]
    fn v4_metadata_requires_exact_lz4_output_length() {
        let schema = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .build();
        let mut builder = VolumeBuilder::new(&schema);
        builder.add_row(1, &Row::from_values(vec![Value::Integer(1)]));
        let (mut bytes, _) = serialize_v4_public(&builder.finish()).unwrap();
        let length = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        bytes[20..24].copy_from_slice(&(length + 1).to_le_bytes());
        let payload_end = bytes.len() - 4;
        let checksum = crc32fast::hash(&bytes[..payload_end]);
        bytes[payload_end..].copy_from_slice(&checksum.to_le_bytes());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.vol");
        std::fs::write(&path, bytes).unwrap();
        let error = read_volume_from_disk(&path).err().unwrap().to_string();
        assert!(error.contains("decoded length mismatch"), "{error}");
    }

    #[test]
    fn test_write_and_read_volume() {
        let dir = tempfile::tempdir().unwrap();
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .column("name", DataType::Text, false, false)
            .build();

        let mut builder = VolumeBuilder::new(&schema);
        builder.add_row(
            1,
            &Row::from_values(vec![Value::Integer(1), Value::text("hello")]),
        );
        builder.add_row(
            2,
            &Row::from_values(vec![Value::Integer(2), Value::text("world")]),
        );
        let vol = builder.finish();

        let path = write_volume_to_disk(dir.path(), "test_table", 1, &vol).unwrap();
        assert!(path.exists());

        let loaded = read_volume_from_disk(&path).unwrap();
        assert_eq!(loaded.meta.row_count, 2);
        assert_eq!(loaded.columns[0].get_i64(0), 1);
        assert_eq!(loaded.columns[1].get_str(1), "world");
    }

    #[test]
    fn test_list_volumes() {
        let dir = tempfile::tempdir().unwrap();
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .build();

        for i in 0..3 {
            let mut builder = VolumeBuilder::new(&schema);
            builder.add_row(i, &Row::from_values(vec![Value::Integer(i)]));
            let vol = builder.finish();
            write_volume_to_disk(dir.path(), "my_table", i as u64, &vol).unwrap();
        }

        let paths = list_volumes(dir.path(), "my_table");
        assert_eq!(paths.len(), 3);
    }

    #[test]
    fn test_load_all_volumes() {
        let dir = tempfile::tempdir().unwrap();
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .build();

        for i in 1..=3 {
            let mut builder = VolumeBuilder::new(&schema);
            builder.add_row(i, &Row::from_values(vec![Value::Integer(i)]));
            let vol = builder.finish();
            write_volume_to_disk(dir.path(), "t", i as u64, &vol).unwrap();
        }

        let volumes = load_all_volumes(dir.path(), "t").unwrap();
        assert_eq!(volumes.len(), 3);
        assert_eq!(volumes[0].meta.row_count, 1);
    }

    #[test]
    fn test_catalog_roundtrip() {
        let mut catalog = VolumeCatalog::new();
        catalog.add_volume(
            "candlesticks_t1m",
            VolumeEntry {
                volume_id: 1000,
                row_count: 500_000,
                time_min_micros: 1_700_000_000_000_000,
                time_max_micros: 1_700_100_000_000_000,
            },
        );
        catalog.add_volume(
            "candlesticks_t1m",
            VolumeEntry {
                volume_id: 2000,
                row_count: 300_000,
                time_min_micros: 1_700_100_000_000_000,
                time_max_micros: 1_700_200_000_000_000,
            },
        );
        catalog.add_volume(
            "tickers",
            VolumeEntry {
                volume_id: 3000,
                row_count: 100,
                time_min_micros: 0,
                time_max_micros: 0,
            },
        );

        let data = catalog.serialize();
        let loaded = VolumeCatalog::deserialize(&data).unwrap();

        assert_eq!(loaded.get_volumes("candlesticks_t1m").len(), 2);
        assert_eq!(loaded.get_volumes("tickers").len(), 1);
        assert_eq!(loaded.get_volumes("nonexistent").len(), 0);
        assert_eq!(loaded.get_volumes("candlesticks_t1m")[0].row_count, 500_000);
    }

    #[test]
    fn test_catalog_disk_roundtrip() {
        let dir = tempfile::tempdir().unwrap();

        let mut catalog = VolumeCatalog::new();
        catalog.add_volume(
            "t1",
            VolumeEntry {
                volume_id: 42,
                row_count: 1000,
                time_min_micros: 100,
                time_max_micros: 200,
            },
        );

        catalog.write_to_disk(dir.path()).unwrap();
        let loaded = VolumeCatalog::read_from_disk(dir.path()).unwrap();

        assert_eq!(loaded.get_volumes("t1").len(), 1);
        assert_eq!(loaded.get_volumes("t1")[0].volume_id, 42);
    }

    #[test]
    fn test_delete_volumes() {
        let dir = tempfile::tempdir().unwrap();
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .build();

        let mut builder = VolumeBuilder::new(&schema);
        builder.add_row(1, &Row::from_values(vec![Value::Integer(1)]));
        let vol = builder.finish();
        write_volume_to_disk(dir.path(), "t", 1, &vol).unwrap();

        assert_eq!(list_volumes(dir.path(), "t").len(), 1);
        delete_all_volumes(dir.path(), "t").unwrap();
        assert_eq!(list_volumes(dir.path(), "t").len(), 0);
    }

    #[test]
    fn test_v4_roundtrip_basic() {
        let dir = tempfile::tempdir().unwrap();
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .column("name", DataType::Text, false, false)
            .column("price", DataType::Float, false, false)
            .build();

        let mut builder = VolumeBuilder::with_capacity(&schema, 3);
        builder.add_row(
            1,
            &Row::from_values(vec![
                Value::Integer(1),
                Value::text("apple"),
                Value::Float(1.50),
            ]),
        );
        builder.add_row(
            2,
            &Row::from_values(vec![
                Value::Integer(2),
                Value::text("banana"),
                Value::Float(0.75),
            ]),
        );
        builder.add_row(
            3,
            &Row::from_values(vec![
                Value::Integer(3),
                Value::text("apple"),
                Value::Float(3.00),
            ]),
        );
        let vol = builder.finish();

        // write_volume_to_disk with compress=true produces V4
        let path = write_volume_to_disk(dir.path(), "t", 1, &vol).unwrap();
        // Verify STV4 magic
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..4], b"STV4");

        // Read back and verify eager loading
        let loaded = read_volume_from_disk(&path).unwrap();
        assert_eq!(loaded.meta.row_count, 3);

        // Access columns triggers decompression from RAM
        assert_eq!(loaded.columns[0].get_i64(0), 1);
        assert_eq!(loaded.columns[0].get_i64(2), 3);
        assert_eq!(loaded.columns[1].get_str(0), "apple");
        assert_eq!(loaded.columns[1].get_str(1), "banana");
        assert_eq!(loaded.columns[2].get_f64(1), 0.75);

        // Zone maps survived
        assert_eq!(loaded.meta.zone_maps[0].min, Value::Integer(1));
        assert_eq!(loaded.meta.zone_maps[0].max, Value::Integer(3));

        // Stats survived
        assert_eq!(loaded.meta.stats.count_star(), 3);
        assert_eq!(loaded.meta.stats.sum(2), 5.25);

        // Sorted flags survived
        assert!(loaded.meta.sorted_columns[0]);

        // Row IDs survived
        assert_eq!(loaded.meta.row_ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_v4_roundtrip_with_nulls() {
        let dir = tempfile::tempdir().unwrap();
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .column("value", DataType::Float, true, false)
            .build();

        let mut builder = VolumeBuilder::new(&schema);
        builder.add_row(
            1,
            &Row::from_values(vec![Value::Integer(1), Value::Float(10.0)]),
        );
        builder.add_row(
            2,
            &Row::from_values(vec![Value::Integer(2), Value::Null(DataType::Float)]),
        );
        builder.add_row(
            3,
            &Row::from_values(vec![Value::Integer(3), Value::Float(30.0)]),
        );
        let vol = builder.finish();

        let path = write_volume_to_disk(dir.path(), "t", 1, &vol).unwrap();
        let loaded = read_volume_from_disk(&path).unwrap();

        assert_eq!(loaded.meta.row_count, 3);
        assert!(!loaded.columns[1].is_null(0));
        assert!(loaded.columns[1].is_null(1));
        assert!(!loaded.columns[1].is_null(2));
        assert_eq!(loaded.columns[1].get_f64(0), 10.0);
        assert_eq!(loaded.columns[1].get_f64(2), 30.0);
    }

    #[test]
    fn test_v4_roundtrip_multiple_row_groups() {
        let dir = tempfile::tempdir().unwrap();
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .column("label", DataType::Text, false, false)
            .build();

        // Create > ROW_GROUP_SIZE rows to exercise multi-group path
        let n = 70_000; // > 65536 (ROW_GROUP_SIZE)
        let mut builder = VolumeBuilder::with_capacity(&schema, n);
        for i in 0..n {
            builder.add_row(
                i as i64,
                &Row::from_values(vec![
                    Value::Integer(i as i64),
                    Value::text(if i % 2 == 0 { "even" } else { "odd" }),
                ]),
            );
        }
        let vol = builder.finish();

        let path = write_volume_to_disk(dir.path(), "t", 1, &vol).unwrap();
        let loaded = read_volume_from_disk(&path).unwrap();

        assert_eq!(loaded.meta.row_count, n);

        // Check first, middle, and last rows
        assert_eq!(loaded.columns[0].get_i64(0), 0);
        assert_eq!(loaded.columns[0].get_i64(n / 2), (n / 2) as i64);
        assert_eq!(loaded.columns[0].get_i64(n - 1), (n - 1) as i64);
        assert_eq!(loaded.columns[1].get_str(0), "even");
        assert_eq!(loaded.columns[1].get_str(1), "odd");
        assert_eq!(loaded.columns[1].get_str(n - 1), "odd");

        // Row groups present
        assert!(!loaded.meta.row_groups.is_empty());
    }

    #[test]
    fn test_v4_roundtrip_timestamp_boolean() {
        let dir = tempfile::tempdir().unwrap();
        let schema = SchemaBuilder::new("test")
            .column("time", DataType::Timestamp, false, false)
            .column("flag", DataType::Boolean, false, false)
            .build();

        let ts = chrono::Utc::now();
        let mut builder = VolumeBuilder::new(&schema);
        builder.add_row(
            1,
            &Row::from_values(vec![Value::Timestamp(ts), Value::Boolean(true)]),
        );
        builder.add_row(
            2,
            &Row::from_values(vec![
                Value::Timestamp(ts + chrono::Duration::minutes(1)),
                Value::Boolean(false),
            ]),
        );
        let vol = builder.finish();

        let path = write_volume_to_disk(dir.path(), "t", 1, &vol).unwrap();
        let loaded = read_volume_from_disk(&path).unwrap();

        assert_eq!(loaded.meta.row_count, 2);
        // Timestamp nanosecond precision
        if let Value::Timestamp(loaded_ts) = loaded.columns[0].get_value(0) {
            assert_eq!(loaded_ts.timestamp_nanos_opt(), ts.timestamp_nanos_opt());
        } else {
            panic!("expected Timestamp");
        }
        assert!(loaded.columns[1].get_bool(0));
        assert!(!loaded.columns[1].get_bool(1));
    }

    #[test]
    fn test_v4_get_row_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .column("name", DataType::Text, false, false)
            .build();

        let mut builder = VolumeBuilder::new(&schema);
        builder.add_row(
            1,
            &Row::from_values(vec![Value::Integer(42), Value::text("test")]),
        );
        let vol = builder.finish();

        let path = write_volume_to_disk(dir.path(), "t", 1, &vol).unwrap();
        let loaded = read_volume_from_disk(&path).unwrap();

        let row = loaded.get_row(0).unwrap();
        assert_eq!(row.get(0), Some(&Value::Integer(42)));
        assert_eq!(row.get(1), Some(&Value::text("test")));
    }
}
