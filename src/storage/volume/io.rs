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
use std::sync::Arc;

use crate::core::Result;

use super::format::{deserialize_volume, serialize_volume};
use super::writer::FrozenVolume;

/// Volume file extension
const VOLUME_EXT: &str = "vol";

/// Magic bytes for LZ4-compressed volumes. Uncompressed volumes start with
/// b"STVL" (the format magic). On read we check the first 4 bytes: STVZ
/// means decompress first, STVL means read directly. This keeps compression
/// fully transparent to the format module.
const COMPRESSED_MAGIC: [u8; 4] = *b"STVZ";

/// Volume catalog filename
const CATALOG_FILE: &str = "volumes.catalog";

/// Write a frozen volume to disk atomically.
///
/// Writes to a .tmp file first, then renames to the final name.
/// This prevents partial files from crashes during writes.
pub fn write_volume_to_disk(
    dir: &Path,
    table_name: &str,
    volume_id: u64,
    volume: &FrozenVolume,
) -> Result<PathBuf> {
    write_volume_to_disk_opts(dir, table_name, volume_id, volume, true)
}

/// Write a frozen volume to disk atomically, with optional LZ4 compression.
///
/// When `compress` is true, the serialized bytes are LZ4-compressed and
/// prefixed with the STVZ magic. When false (or compression doesn't shrink
/// the data), the raw STVL format is written.
pub fn write_volume_to_disk_opts(
    dir: &Path,
    table_name: &str,
    volume_id: u64,
    volume: &FrozenVolume,
    compress: bool,
) -> Result<PathBuf> {
    let table_dir = dir.join(table_name);
    std::fs::create_dir_all(&table_dir)
        .map_err(|e| crate::core::Error::internal(format!("failed to create volume dir: {}", e)))?;

    let filename = format!("vol_{:016x}.{}", volume_id, VOLUME_EXT);
    let final_path = table_dir.join(&filename);
    let tmp_path = table_dir.join(format!("{}.tmp", filename));

    let raw = serialize_volume(volume)
        .map_err(|e| crate::core::Error::internal(format!("failed to serialize volume: {}", e)))?;

    // LZ4 compress when enabled and it actually shrinks the data.
    // Wrap with STVZ magic so the reader can detect compressed vs uncompressed.
    // Drop raw before building output to avoid holding both in memory.
    let data = if compress {
        let raw_len = raw.len();
        let compressed = lz4_flex::compress_prepend_size(&raw);
        if COMPRESSED_MAGIC.len() + compressed.len() < raw_len {
            drop(raw);
            let mut out = Vec::with_capacity(COMPRESSED_MAGIC.len() + compressed.len());
            out.extend_from_slice(&COMPRESSED_MAGIC);
            out.extend_from_slice(&compressed);
            out
        } else {
            raw
        }
    } else {
        raw
    };

    // Write to tmp file, fsync BEFORE rename for crash safety.
    // Without fsync-before-rename, a power failure after rename could leave
    // the file with zeros (metadata journaled, data not yet flushed).
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&tmp_path).map_err(|e| {
            crate::core::Error::internal(format!("failed to create volume tmp file: {}", e))
        })?;
        f.write_all(&data).map_err(|e| {
            crate::core::Error::internal(format!("failed to write volume file: {}", e))
        })?;
        f.sync_all().map_err(|e| {
            crate::core::Error::internal(format!("failed to fsync volume tmp file: {}", e))
        })?;
    }

    std::fs::rename(&tmp_path, &final_path).map_err(|e| {
        crate::core::Error::internal(format!("failed to rename volume file: {}", e))
    })?;

    // Fsync the parent directory to ensure the rename is durable.
    // Windows does not support opening directories for fsync.
    #[cfg(not(windows))]
    if let Ok(d) = std::fs::File::open(&table_dir) {
        d.sync_all().map_err(|e| {
            crate::core::Error::internal(format!("failed to fsync volume directory: {}", e))
        })?;
    }

    Ok(final_path)
}

/// Read a frozen volume from disk.
///
/// Detects LZ4-compressed files (STVZ magic) and decompresses transparently.
/// Uncompressed files (STVL magic) are read directly.
pub fn read_volume_from_disk(path: &Path) -> Result<FrozenVolume> {
    let data = std::fs::read(path).map_err(|e| {
        crate::core::Error::internal(format!("failed to read volume file {:?}: {}", path, e))
    })?;

    if data.len() >= 4 && data[..4] == COMPRESSED_MAGIC {
        let raw = lz4_flex::decompress_size_prepended(&data[4..]).map_err(|e| {
            crate::core::Error::internal(format!("failed to decompress volume {:?}: {}", path, e))
        })?;
        deserialize_volume(&raw).map_err(|e| {
            crate::core::Error::internal(format!("failed to deserialize volume {:?}: {}", path, e))
        })
    } else {
        deserialize_volume(&data).map_err(|e| {
            crate::core::Error::internal(format!("failed to deserialize volume {:?}: {}", path, e))
        })
    }
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
        assert_eq!(loaded.row_count, 2);
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
        assert_eq!(volumes[0].row_count, 1);
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
}
