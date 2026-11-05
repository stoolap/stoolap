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

//! A WAL whose last write a crash cut, or whose records were damaged,
//! opens with every whole commit and keeps the commits made after it
//! across the reopens that follow

use std::path::{Path, PathBuf};
use stoolap::Database;

const COMMIT_MARKER: u8 = 1 << 1;

/// Every test here takes the shared failpoint guard when failpoints are on:
/// one of them fails WAL reads process-wide
#[cfg(feature = "test-failpoints")]
fn serial() -> stoolap::test_failpoints::FailpointGuard {
    stoolap::test_failpoints::FailpointGuard::new()
}

#[cfg(not(feature = "test-failpoints"))]
fn serial() {}

fn dsn(dir: &Path) -> String {
    format!(
        "file://{}?sync_mode=full&checkpoint_on_close=off&checkpoint_interval=0",
        dir.display()
    )
}

fn ids(db: &Database) -> Vec<i64> {
    db.query("SELECT id FROM t ORDER BY id", ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect()
}

fn wal_file(dir: &Path) -> PathBuf {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir.join("wal"))
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("log"))
        .collect();
    assert_eq!(files.len(), 1, "one WAL file: {files:?}");
    files.pop().unwrap()
}

/// Each record's start, length and flags
fn records(bytes: &[u8]) -> Vec<(usize, usize, u8)> {
    let mut out = Vec::new();
    let mut at = 0;
    while at + 32 <= bytes.len() {
        let entry_size = u32::from_le_bytes(bytes[at + 24..at + 28].try_into().unwrap()) as usize;
        let len = 32 + entry_size + 4;
        out.push((at, len, bytes[at + 5]));
        at += len;
    }
    out
}

/// Whether the file is whole records end to end, each with its checksum
fn whole_records(bytes: &[u8]) -> bool {
    let mut at = 0;
    for (start, len, _) in records(bytes) {
        let magic = u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
        if start != at || magic != 0x454C_4157 || start + len > bytes.len() {
            return false;
        }
        let crc = u32::from_le_bytes(bytes[start + len - 4..start + len].try_into().unwrap());
        if crc != crc32fast::hash(&bytes[start + 32..start + len - 4]) {
            return false;
        }
        at = start + len;
    }
    at == bytes.len()
}

/// Rows 1 to 6, one commit each, left in the WAL
fn fixture(dir: &Path) {
    let db = Database::open(&dsn(dir)).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    for id in 1..=6 {
        db.execute(&format!("INSERT INTO t VALUES ({id}, {id})"), ())
            .unwrap();
    }
    db.close().unwrap();
}

/// Opens, commits `more`, and opens twice more: every read after the
/// first holds what the first read held and every commit made since
fn reopen_and_write(dir: &Path, expected: &[i64]) {
    let db = Database::open(&dsn(dir)).unwrap();
    assert_eq!(ids(&db), expected, "the first open");
    db.execute("INSERT INTO t VALUES (10, 10)", ()).unwrap();
    db.execute("INSERT INTO t VALUES (11, 11)", ()).unwrap();
    let mut with_new = expected.to_vec();
    with_new.extend([10, 11]);
    assert_eq!(ids(&db), with_new, "the rows written after the open");
    db.close().unwrap();
    drop(db);
    let db = Database::open(&dsn(dir)).unwrap();
    assert_eq!(ids(&db), with_new, "the second open");
    db.execute("INSERT INTO t VALUES (20, 20)", ()).unwrap();
    db.close().unwrap();
    drop(db);
    let db = Database::open(&dsn(dir)).unwrap();
    with_new.push(20);
    assert_eq!(ids(&db), with_new, "the third open");
    db.close().unwrap();
}

/// A reopen appends after the last whole record, never after torn bytes
fn assert_whole_records(dir: &Path) {
    assert!(
        whole_records(&std::fs::read(wal_file(dir)).unwrap()),
        "the WAL holds whole records only"
    );
}

/// The last commit marker cut short commits nothing, at the first open
/// and at every one after
#[test]
fn a_cut_last_record_commits_nothing_and_later_commits_stay() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    let wal = wal_file(dir.path());
    let len = std::fs::metadata(&wal).unwrap().len();
    let file = std::fs::OpenOptions::new().write(true).open(&wal).unwrap();
    file.set_len(len - 7).unwrap();
    drop(file);
    reopen_and_write(dir.path(), &[1, 2, 3, 4, 5]);
    assert_whole_records(dir.path());
}

/// A record header after the last record, its body never written: it is
/// no record, and the records written after it are read from their start
#[test]
fn a_header_without_its_body_is_dropped_and_later_commits_stay() {
    let _serial = serial();
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    let wal = wal_file(dir.path());
    let bytes = std::fs::read(&wal).unwrap();
    // The last record's header with a longer body claimed and 10 bytes of it
    let (start, _, _) = *records(&bytes).last().unwrap();
    let mut header = bytes[start..start + 32].to_vec();
    header[5] = 0;
    header[24..28].copy_from_slice(&200u32.to_le_bytes());
    let mut file = std::fs::OpenOptions::new().append(true).open(&wal).unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&[0xAB; 10]).unwrap();
    drop(file);
    reopen_and_write(dir.path(), &[1, 2, 3, 4, 5, 6]);
    assert_whole_records(dir.path());
}

/// A commit marker cut short in the middle of the file, the records
/// after it whole: its transaction commits nothing, and the record its
/// claimed size reaches into is read all the same
#[test]
fn a_marker_cut_in_the_middle_loses_no_record_after_it() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    let wal = wal_file(dir.path());
    let mut bytes = std::fs::read(&wal).unwrap();
    let markers: Vec<(usize, usize, u8)> = records(&bytes)
        .into_iter()
        .filter(|(_, _, flags)| flags & COMMIT_MARKER != 0)
        .collect();
    // Row 3's commit marker loses its last 7 bytes; row 4's data follows
    let (start, len, _) = markers[3];
    bytes.drain(start + len - 7..start + len);
    std::fs::write(&wal, &bytes).unwrap();
    reopen_and_write(dir.path(), &[1, 2, 4, 5, 6]);
}

/// A commit marker whose checksum fails commits nothing, whatever its
/// transaction wrote before it
#[test]
fn a_damaged_commit_marker_commits_nothing() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    let wal = wal_file(dir.path());
    let mut bytes = std::fs::read(&wal).unwrap();
    let markers: Vec<(usize, usize, u8)> = records(&bytes)
        .into_iter()
        .filter(|(_, _, flags)| flags & COMMIT_MARKER != 0)
        .collect();
    // Row 4's marker: its last byte before the checksum flipped
    let (start, len, _) = markers[4];
    bytes[start + len - 5] ^= 0xFF;
    std::fs::write(&wal, &bytes).unwrap();
    reopen_and_write(dir.path(), &[1, 2, 3, 5, 6]);
}

/// A write that is the first thing after the reopen takes a transaction id
/// no retained record carries: the cut commit's row stays out after it
#[test]
fn a_first_write_after_the_reopen_does_not_commit_the_cut_transaction() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    let wal = wal_file(dir.path());
    let len = std::fs::metadata(&wal).unwrap().len();
    let file = std::fs::OpenOptions::new().write(true).open(&wal).unwrap();
    file.set_len(len - 7).unwrap();
    drop(file);
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute("INSERT INTO t VALUES (10, 10)", ()).unwrap();
    db.close().unwrap();
    drop(db);
    let db = Database::open(&dsn(dir.path())).unwrap();
    assert_eq!(ids(&db), vec![1, 2, 3, 4, 5, 10]);
    db.close().unwrap();
}

/// A damaged record larger than a megabyte is passed over whole: the
/// commit after it is read
#[test]
fn a_damaged_record_past_a_megabyte_loses_no_commit_after_it() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)", ())
        .unwrap();
    // Two megabytes of hex from a fixed sequence: nothing LZ4 can shorten
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut big = String::with_capacity(2 * 1024 * 1024);
    while big.len() < 2 * 1024 * 1024 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        big.push_str(&format!("{state:016x}"));
    }
    db.execute("INSERT INTO t VALUES (1, $1)", (big,)).unwrap();
    db.execute("INSERT INTO t VALUES (2, 'small')", ()).unwrap();
    db.close().unwrap();
    drop(db);
    let wal = wal_file(dir.path());
    let mut bytes = std::fs::read(&wal).unwrap();
    let (start, len, _) = records(&bytes)
        .into_iter()
        .find(|(_, len, _)| *len > 1024 * 1024)
        .expect("a record past a megabyte");
    bytes[start + len / 2] ^= 0xFF;
    std::fs::write(&wal, &bytes).unwrap();
    let db = Database::open(&dsn(dir.path())).unwrap();
    assert_eq!(ids(&db), vec![2]);
    db.close().unwrap();
}

/// A commit marker in the middle claiming more bytes than the file holds
/// is passed over, not taken for the end of the WAL
#[test]
fn a_size_past_the_end_of_the_file_loses_no_record_after_it() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    let wal = wal_file(dir.path());
    let mut bytes = std::fs::read(&wal).unwrap();
    let markers: Vec<(usize, usize, u8)> = records(&bytes)
        .into_iter()
        .filter(|(_, _, flags)| flags & COMMIT_MARKER != 0)
        .collect();
    let (start, _, _) = markers[3];
    let past_the_end = (bytes.len() as u32) * 2;
    bytes[start + 24..start + 28].copy_from_slice(&past_the_end.to_le_bytes());
    std::fs::write(&wal, &bytes).unwrap();
    reopen_and_write(dir.path(), &[1, 2, 4, 5, 6]);
}

/// A read that fails while looking for the record after a damaged one
/// stops the open and leaves the WAL as it was: nothing is cut on a check
/// that did not finish
#[cfg(feature = "test-failpoints")]
#[test]
fn a_read_error_while_checking_the_wal_stops_the_open_and_cuts_nothing() {
    use std::sync::atomic::Ordering;
    use stoolap::test_failpoints;
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    let wal = wal_file(dir.path());
    let mut bytes = std::fs::read(&wal).unwrap();
    let markers: Vec<(usize, usize, u8)> = records(&bytes)
        .into_iter()
        .filter(|(_, _, flags)| flags & COMMIT_MARKER != 0)
        .collect();
    let (start, len, _) = markers[3];
    bytes[start + len - 5] ^= 0xFF;
    std::fs::write(&wal, &bytes).unwrap();
    test_failpoints::WAL_SCAN_READ_FAIL.store(true, Ordering::Release);
    let opened = Database::open(&dsn(dir.path()));
    test_failpoints::WAL_SCAN_READ_FAIL.store(false, Ordering::Release);
    assert!(
        opened.is_err(),
        "the open stops: {:?}",
        opened.as_ref().err()
    );
    drop(opened);
    assert_eq!(std::fs::read(&wal).unwrap(), bytes, "the WAL is as it was");
    let db = Database::open(&dsn(dir.path())).unwrap();
    assert_eq!(ids(&db), vec![1, 2, 4, 5, 6]);
    db.close().unwrap();
}

/// An open that cannot take the database's lock leaves the WAL alone: the
/// bytes after its last whole record may be a write the owner is making
#[test]
fn an_open_without_the_lock_cuts_nothing() {
    let _serial = serial();
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    let owner = Database::open(&dsn(dir.path())).unwrap();
    let wal = wal_file(dir.path());
    let bytes = std::fs::read(&wal).unwrap();
    // The owner's record in flight: its header and part of its body
    let (start, _, _) = *records(&bytes).last().unwrap();
    let mut file = std::fs::OpenOptions::new().append(true).open(&wal).unwrap();
    file.write_all(&bytes[start..start + 40]).unwrap();
    drop(file);
    let before = std::fs::read(&wal).unwrap();
    let second = Database::open(&format!(
        "file://{}?sync_mode=normal&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ));
    assert!(second.is_err(), "the lock is the owner's");
    drop(second);
    assert_eq!(
        std::fs::read(&wal).unwrap(),
        before,
        "the WAL is as the owner left it"
    );
    drop(owner);
}

/// A commit whose sync fails after the open cut a torn tail is taken back
/// to the end the cut left, not to the length the cut bytes gave the file
#[cfg(feature = "test-failpoints")]
#[test]
fn a_commit_that_fails_after_the_cut_is_taken_back() {
    use std::io::Write;
    use std::sync::atomic::Ordering;
    use stoolap::test_failpoints;
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(wal_file(dir.path()))
        .unwrap();
    file.write_all(&[0xAB; 8192]).unwrap();
    drop(file);
    let db = Database::open(&dsn(dir.path())).unwrap();
    test_failpoints::WAL_SYNC_FAIL.store(true, Ordering::Release);
    let rejected = db.execute("INSERT INTO t VALUES (10, 10)", ());
    test_failpoints::WAL_SYNC_FAIL.store(false, Ordering::Release);
    assert!(rejected.is_err(), "the commit's sync failed");
    drop(db);
    let db = Database::open(&dsn(dir.path())).unwrap();
    assert_eq!(ids(&db), vec![1, 2, 3, 4, 5, 6]);
    db.close().unwrap();
}
