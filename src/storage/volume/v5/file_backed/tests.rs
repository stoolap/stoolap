// Copyright 2026 Stoolap Contributors
// Licensed under the Apache License, Version 2.0.

use std::io::{Seek, SeekFrom, Write};
use std::num::NonZeroU64;

use super::*;
use crate::storage::volume::io::VolumeRetirementQueue;
use crate::storage::volume::v5::directory::{
    DirectoryKey, DirectoryRoot, Layout, LeafEntry, RowBounds, Section, VolumeShape, KEY_REQUIRED,
};
use crate::storage::volume::v5::envelope::Codec;
use crate::storage::volume::v5::page_io::PageWriter;

fn identity() -> FileIdentity {
    FileIdentity {
        table_id: NonZeroU64::new(11).unwrap(),
        incarnation: NonZeroU64::new(12).unwrap(),
        volume_id: NonZeroU64::new(13).unwrap(),
    }
}
fn limits() -> ReadLimits {
    ReadLimits {
        root_stored_bytes: 1024,
        root_decoded_bytes: 128,
        page_stored_bytes: 1 << 20,
        page_decoded_bytes: 1 << 20,
    }
}
fn fixture(path: &Path, codec: Codec) -> (FinishedVolume, LeafEntry) {
    let mut file = File::create(path).unwrap();
    let mut writer = PageWriter::new(&mut file, Header::new(identity()), limits()).unwrap();
    // Raw payload bytes exercise the file access layer, not typed row decoding.
    let data = [37u8; 8192];
    let compressed = lz4_flex::block::compress(&data);
    let stored = match codec {
        Codec::Raw => data.as_slice(),
        Codec::Lz4Block => compressed.as_slice(),
    };
    let page = writer
        .append_stored(codec, stored, data.len() as u64)
        .unwrap();
    let entry = LeafEntry {
        key: DirectoryKey {
            section: Section::ColumnBlocks as u16,
            flags: KEY_REQUIRED,
            column: 0,
            ordinal: 0,
        },
        page,
    };
    let directory = writer.append_leaf(&[entry], &mut [0; 8192]).unwrap();
    let summary = VolumeShape {
        layout: Layout::RowId,
        row_count: 1,
        column_count: 1,
        group_count: 1,
        rows: Some(RowBounds { min: 7, max: 7 }),
        window: None,
    }
    .into_root(
        1,
        Some(DirectoryRoot {
            depth: 1,
            page: directory,
        }),
        None,
    )
    .unwrap();
    let finished = writer.finish(&summary).unwrap();
    file.flush().unwrap();
    (finished, entry)
}
fn attach(path: &Path, finished: &FinishedVolume) -> FileBackedVolume {
    FileBackedVolume::from_finished(path, finished, limits(), &mut [0; 1024], &mut [0; 128])
        .unwrap()
}

#[test]
fn bounded_open_and_leased_raw_or_compressed_page_roundtrip() {
    for codec in [Codec::Raw, Codec::Lz4Block] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.vol");
        let (finished, entry) = fixture(&path, codec);
        let volume = attach(&path, &finished);
        assert_eq!(volume.header().identity, identity());
        assert_eq!(*volume.summary(), finished.summary);
        assert_eq!(volume.directory_buffers().unwrap().lookup_node_bytes, 8192);
        let lease = volume.lease().unwrap();
        let mut node = [0; 8192];
        let mut compressed_node = [0; 8192];
        let mut directory = lease.directory(&mut node, &mut compressed_node).unwrap();
        assert_eq!(directory.find(entry.key.into()).unwrap(), Some(entry));
        let page = lease.page(entry.page).unwrap();
        assert_eq!(page.stored_buffer_len(), entry.page.stored_len as usize);
        assert_eq!(
            page.decoded_buffer_len(),
            if codec == Codec::Raw { 0 } else { 8192 }
        );
        let mut stored = [0; 8192];
        let mut decoded = [0; 8192];
        assert_eq!(
            page.read_into(&mut stored, &mut decoded).unwrap(),
            &[37; 8192]
        );
    }
}

#[test]
fn open_defers_payload_and_directory_corruption_until_access() {
    for corrupt_directory in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.vol");
        let (finished, entry) = fixture(&path, Codec::Raw);
        let offset = if corrupt_directory {
            finished.summary.directory.unwrap().page.offset
        } else {
            entry.page.offset
        };
        let mut file = File::options().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.write_all(&[0xff]).unwrap();
        drop(file);
        // Neither unrelated payloads nor all directory nodes are loaded at open.
        let volume = attach(&path, &finished);
        let lease = volume.lease().unwrap();
        if corrupt_directory {
            let mut node = [0; 8192];
            let mut stored = [0; 8192];
            let mut directory = lease.directory(&mut node, &mut stored).unwrap();
            assert!(directory.find(entry.key.into()).is_err());
            assert!(directory.find(entry.key.into()).is_err());
        } else {
            let plan = lease.page(entry.page).unwrap();
            assert!(plan.read_into(&mut [0; 8192], &mut []).is_err());
        }
    }
}

#[test]
fn identity_completion_and_capacity_fail_closed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.vol");
    let (finished, entry) = fixture(&path, Codec::Raw);
    let mut wrong = identity();
    wrong.incarnation = NonZeroU64::new(22).unwrap();
    assert!(matches!(
        FileBackedVolume::open(&path, wrong, limits(), &mut [0; 1024], &mut [0; 128]),
        Err(FileVolumeError::Page(PageIoError::IdentityMismatch))
    ));
    let mut completion = finished;
    completion.summary.rows = Some(RowBounds { min: 8, max: 8 });
    assert!(matches!(
        FileBackedVolume::from_finished(
            &path,
            &completion,
            limits(),
            &mut [0; 1024],
            &mut [0; 128]
        ),
        Err(FileVolumeError::CompletedFileMismatch)
    ));
    assert!(matches!(
        FileBackedVolume::open(&path, identity(), limits(), &mut [0; 127], &mut [0; 128]),
        Err(FileVolumeError::Page(PageIoError::BufferTooSmall))
    ));
    let volume = attach(&path, &finished);
    let lease = volume.lease().unwrap();
    let plan = lease.page(entry.page).unwrap();
    let mut short = [91; 8191];
    assert!(matches!(
        plan.read_into(&mut short, &mut []),
        Err(PageIoError::BufferTooSmall)
    ));
    assert_eq!(short, [91; 8191]);
    let mut outside = entry.page;
    outside.offset = finished.footer.root.offset;
    assert!(lease.page(outside).is_err());
}

#[test]
fn same_length_path_replacement_does_not_rebind_an_existing_handle() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.vol");
    let moved = dir.path().join("original.vol");
    let (finished, entry) = fixture(&path, Codec::Raw);
    let volume = attach(&path, &finished);
    let lease = volume.lease().unwrap();
    std::fs::rename(&path, &moved).unwrap();
    let (replacement, _) = fixture(&path, Codec::Raw);
    assert_eq!(replacement.footer, finished.footer);
    assert_eq!(
        volume.lease().err().unwrap().kind(),
        std::io::ErrorKind::InvalidData
    );
    // A live FD still refers to the old identity even after the path is reused.
    assert_eq!(
        lease
            .page(entry.page)
            .unwrap()
            .read_into(&mut [0; 8192], &mut [])
            .unwrap(),
        &[37; 8192]
    );
}

#[test]
fn managed_rename_and_retirement_share_the_existing_file_owner() {
    let dir = tempfile::tempdir().unwrap();
    let old = dir.path().join("old");
    let new = dir.path().join("new");
    std::fs::create_dir(&old).unwrap();
    let path = old.join("data.vol");
    let (finished, entry) = fixture(&path, Codec::Raw);
    let volume = attach(&path, &finished);
    let alias = volume.clone();
    let queue = VolumeRetirementQueue::default();
    queue.track(volume.backing());
    queue.retire(volume.backing());
    let lease = volume.lease().unwrap();
    let lease = match queue.rename_directory(&old, &new, &[]) {
        Ok(()) => lease,
        #[cfg(windows)]
        Err(error) => {
            // Windows can reject a directory rename while a child has an
            // active file handle. The failed rename must retain the old alias;
            // releasing that short lease permits the same operation to retry.
            assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
            assert!(old.join("data.vol").exists());
            assert!(!new.exists());
            assert!(alias.lease().is_ok());
            drop(lease);
            queue.rename_directory(&old, &new, &[]).unwrap();
            alias.lease().unwrap()
        }
        #[cfg(not(windows))]
        Err(error) => panic!("managed directory rename failed: {error}"),
    };
    assert_eq!(
        lease
            .page(entry.page)
            .unwrap()
            .read_into(&mut [0; 8192], &mut [])
            .unwrap(),
        &[37; 8192]
    );
    drop(lease);
    assert!(alias.lease().is_ok());
    drop(volume);
    queue.sweep();
    assert!(new.join("data.vol").exists());
    drop(alias);
    queue.sweep();
    assert!(!new.join("data.vol").exists());
}
