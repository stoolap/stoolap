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

//! The engine owns a volume's secondary index side file: built when the
//! volume is sealed or compacted, opened before the volume is published,
//! moved with a rename, retired with the volume, attached again on reopen.
//! Its builds are admitted against a process-wide budget, and a build the
//! budget refuses leaves the volume uncovered rather than unsealed.

use std::collections::BTreeSet;
use std::path::Path;

use stoolap::core::types::IndexType;
use stoolap::storage::volume::secondary::{definition_of, IndexFile};
use stoolap::Database;

/// The budgets and counters are process-wide, so the tests run one at a
/// time whatever the runner does.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn open(dir: &Path, extra: &str) -> Database {
    Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0{extra}",
        dir.display()
    ))
    .unwrap()
}

/// A table with an index on an integer column, one on a text column
/// (never a side file's), and none on the timestamp column
fn create(db: &Database) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, name TEXT, ts TIMESTAMP)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    db.execute("CREATE INDEX idx_t_name ON t(name)", ())
        .unwrap();
}

/// Inserts `rows` rows with ids from `from`, key `id % 7`, and seals them
fn seal_rows(db: &Database, from: i64, rows: i64) {
    let mut values = String::new();
    for id in from..from + rows {
        values.push_str(&format!(
            "({id},{},'n{}','2026-01-01 00:00:00'),",
            id % 7,
            id % 3
        ));
    }
    values.pop();
    db.execute(&format!("INSERT INTO t VALUES {values}"), ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
}

/// The file stems in a table's volume directory with `ext`
fn files(dir: &Path, table: &str, ext: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    if let Ok(entries) = std::fs::read_dir(dir.join("volumes").join(table)) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some(ext) {
                names.insert(
                    path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap()
                        .to_string(),
                );
            }
        }
    }
    names
}

fn leftovers(dir: &Path, table: &str) -> Vec<String> {
    std::fs::read_dir(dir.join("volumes").join(table))
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|n| n.contains(".build-"))
                .collect()
        })
        .unwrap_or_default()
}

/// One figure of `PRAGMA INDEX_STATS`
fn index_stat(db: &Database, ledger: &str, column: &str) -> i64 {
    let rows = db.query("PRAGMA INDEX_STATS", ()).unwrap();
    let at = rows
        .columns()
        .iter()
        .position(|c| c == column)
        .unwrap_or_else(|| panic!("no column {column}"));
    for row in rows {
        let row = row.unwrap();
        let name: String = row.get(0).unwrap();
        if name == ledger {
            return row.get(at).unwrap();
        }
    }
    panic!("no ledger {ledger}")
}

fn side_of(dir: &Path, table: &str, stem: &str) -> IndexFile {
    IndexFile::open(
        &dir.join("volumes").join(table).join(format!("{stem}.sidx")),
        1,
    )
    .unwrap()
}

#[test]
fn a_seal_writes_a_side_file_of_the_indexed_integer_column_beside_each_volume() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 1), (2, 2)", ())
        .unwrap();
    let builds_before = index_stat(&db, "index_builds", "charged_bytes");
    seal_rows(&db, 1, 5_000);
    seal_rows(&db, 5_001, 5_000);
    let volumes = files(dir.path(), "t", "vol");
    assert_eq!(volumes.len(), 2);
    assert_eq!(
        files(dir.path(), "t", "sidx"),
        volumes,
        "one side file beside each volume"
    );
    assert!(
        files(dir.path(), "u", "sidx").is_empty(),
        "a table without a B-tree index on an integer column gets none"
    );
    assert!(leftovers(dir.path(), "t").is_empty());
    assert_eq!(
        index_stat(&db, "index_builds", "charged_bytes"),
        builds_before,
        "the builds released their workspaces"
    );
    assert!(
        index_stat(&db, "index_directories", "charged_bytes") > 0,
        "the published segments hold their side files' directories"
    );
    // The side file indexes column 1 (k) alone, under its definition, and
    // its positions are the rows with that key
    for stem in &volumes {
        let side = side_of(dir.path(), "t", stem);
        assert_eq!(side.directory().columns.len(), 1);
        assert!(side.covers(1, definition_of("k", IndexType::BTree, false)));
        assert!(!side.covers(2, definition_of("name", IndexType::BTree, false)));
        let (start, end) = side.equal(1, 3).unwrap().unwrap();
        let in_volume: i64 = 5_000;
        let with_key = (0..in_volume).filter(|i| i % 7 == 3).count();
        assert!(
            (end - start) as usize == with_key || (end - start) as usize == with_key + 1,
            "{} positions of key 3 in {stem}",
            end - start
        );
    }
    db.close().unwrap();
}

#[test]
fn a_compaction_replaces_the_inputs_side_files_with_the_outputs_and_a_refused_build_leaves_a_volume_uncovered(
) {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "&compact_threshold=100");
    create(&db);
    // Three sub-target volumes, one over the threshold of two
    seal_rows(&db, 1, 2_000);
    seal_rows(&db, 2_001, 2_000);
    seal_rows(&db, 4_001, 2_000);
    let inputs = files(dir.path(), "t", "vol");
    assert_eq!(inputs.len(), 3);
    assert_eq!(files(dir.path(), "t", "sidx"), inputs);
    db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let outputs = files(dir.path(), "t", "vol");
    assert!(outputs.is_disjoint(&inputs), "the inputs were rewritten");
    assert_eq!(
        files(dir.path(), "t", "sidx"),
        outputs,
        "the inputs' side files went with them and each output has its own"
    );
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 6_000);

    // No budget for builds: the next seal publishes its volume without a
    // side file, counts the refusal, and the rows are there all the same
    db.execute("PRAGMA COMPACT_THRESHOLD = 100", ()).unwrap();
    let refused_before = index_stat(&db, "index_builds", "refused");
    db.execute("PRAGMA INDEX_BUILD_MB = 0", ()).unwrap();
    seal_rows(&db, 6_001, 1_000);
    let after = files(dir.path(), "t", "vol");
    assert_eq!(after.len(), outputs.len() + 1);
    assert_eq!(
        files(dir.path(), "t", "sidx"),
        outputs,
        "the refused volume is uncovered"
    );
    assert_eq!(
        index_stat(&db, "index_builds", "refused"),
        refused_before + 1
    );
    assert_eq!(index_stat(&db, "index_builds", "builds_failed"), 0);
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 7_000);
    let value: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE k = 3", ())
        .unwrap();
    assert_eq!(value, (1..=7_000).filter(|i| i % 7 == 3).count() as i64);
    // With the budget back, the next seals are covered and a compaction
    // covers the refused volume again
    db.execute("PRAGMA INDEX_BUILD_MB = 64", ()).unwrap();
    seal_rows(&db, 7_001, 1_000);
    seal_rows(&db, 8_001, 1_000);
    assert_eq!(files(dir.path(), "t", "sidx").len(), outputs.len() + 2);
    db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let covered = files(dir.path(), "t", "vol");
    assert!(
        covered.is_disjoint(&after),
        "the sub-target volumes were merged"
    );
    assert_eq!(files(dir.path(), "t", "sidx"), covered);
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 9_000);
    assert!(leftovers(dir.path(), "t").is_empty());
    db.close().unwrap();
}

#[test]
fn drop_table_truncate_and_rename_take_the_side_files_with_their_volumes() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    seal_rows(&db, 1, 2_000);
    assert_eq!(files(dir.path(), "t", "sidx").len(), 1);
    db.execute("ALTER TABLE t RENAME TO t2", ()).unwrap();
    assert!(files(dir.path(), "t", "sidx").is_empty());
    assert_eq!(
        files(dir.path(), "t2", "sidx"),
        files(dir.path(), "t2", "vol"),
        "the rename moved the side file with the volume"
    );
    db.execute("ALTER TABLE t2 RENAME TO t", ()).unwrap();
    db.execute("TRUNCATE TABLE t", ()).unwrap();
    assert!(
        files(dir.path(), "t", "sidx").is_empty(),
        "TRUNCATE removed the side files"
    );
    seal_rows(&db, 1, 2_000);
    assert_eq!(files(dir.path(), "t", "sidx").len(), 1);
    db.execute("DROP TABLE t", ()).unwrap();
    assert!(
        files(dir.path(), "t", "sidx").is_empty(),
        "DROP TABLE removed the side files"
    );
    db.close().unwrap();
}

#[test]
fn reopen_attaches_the_side_files_and_removes_one_it_cannot_read() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    seal_rows(&db, 1, 2_000);
    seal_rows(&db, 2_001, 2_000);
    let sides = files(dir.path(), "t", "sidx");
    assert_eq!(sides.len(), 2);
    db.close().unwrap();
    drop(db);

    let db = open(dir.path(), "");
    assert!(
        index_stat(&db, "index_directories", "charged_bytes") > 0,
        "the reopened segments hold their side files' directories"
    );
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 4_000);
    db.close().unwrap();
    drop(db);

    // One side file is cut short: it is removed at reopen, the other stays
    let broken = sides.iter().next().unwrap().clone();
    let broken_path = dir
        .path()
        .join("volumes")
        .join("t")
        .join(format!("{broken}.sidx"));
    let bytes = std::fs::read(&broken_path).unwrap();
    std::fs::write(&broken_path, &bytes[..bytes.len() - 7]).unwrap();
    let db = open(dir.path(), "");
    let remaining = files(dir.path(), "t", "sidx");
    assert_eq!(remaining.len(), 1);
    assert!(!remaining.contains(&broken));
    assert_eq!(files(dir.path(), "t", "vol").len(), 2, "both volumes stay");
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 4_000);
    db.close().unwrap();
}

/// A reader that took its view before a compaction holds the inputs' side
/// files, retired but not removed, until it lets go.
#[cfg(feature = "test-failpoints")]
#[test]
fn a_captured_view_keeps_a_retired_side_file_until_it_lets_go() {
    use stoolap::storage::traits::Engine;

    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "&compact_threshold=3");
    create(&db);
    seal_rows(&db, 1, 2_000);
    seal_rows(&db, 2_001, 2_000);
    seal_rows(&db, 4_001, 2_000);
    let inputs = files(dir.path(), "t", "sidx");
    assert_eq!(inputs.len(), 3);
    // A fourth seal inside the reader's window takes the count over the
    // threshold: the reader's three volumes are merged while it reads
    // them, the fourth stays behind its snapshot
    let root = dir.path().to_path_buf();
    let held = inputs.clone();
    let other = db.clone();
    stoolap::test_failpoints::after_cold_volumes_taken(move || {
        seal_rows(&other, 6_001, 2_000);
        let published: BTreeSet<String> = other
            .query("PRAGMA VOLUME_STATS", ())
            .unwrap()
            .map(|row| {
                let id: i64 = row.unwrap().get(1).unwrap();
                format!("vol_{id:016x}")
            })
            .collect();
        assert!(
            published.is_disjoint(&held),
            "the compaction merged the reader's volumes"
        );
        let present = files(&root, "t", "sidx");
        assert!(
            held.is_subset(&present),
            "the reader's side files must survive the compaction while it reads"
        );
        assert!(
            published.is_subset(&present),
            "the outputs' side files are there"
        );
    });
    let mut tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("t").unwrap();
    let rows = table.collect_all_rows(None).unwrap();
    tx.rollback().unwrap();
    assert_eq!(rows.len(), 6_000, "the reader answers its own volumes");
    drop(rows);
    drop(table);
    let after = files(dir.path(), "t", "sidx");
    assert!(
        after.is_disjoint(&inputs),
        "the retired side files went once the reader let go"
    );
    assert_eq!(after, files(dir.path(), "t", "vol"));
    db.close().unwrap();
}

/// Index DDL between a build and its publication makes the side file
/// stale: the volume is published without it, and the discard is counted.
#[cfg(feature = "test-failpoints")]
#[test]
fn index_ddl_during_the_preparation_discards_the_prepared_side_file() {
    let _serial = serial();
    for compaction in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open(dir.path(), "&compact_threshold=100");
        create(&db);
        if compaction {
            seal_rows(&db, 1, 2_000);
            seal_rows(&db, 2_001, 2_000);
            seal_rows(&db, 4_001, 2_000);
            db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
        } else {
            db.execute(
                "INSERT INTO t VALUES (1, 1, 'a', '2026-01-01 00:00:00')",
                (),
            )
            .unwrap();
        }
        let discarded_before = index_stat(&db, "index_builds", "sides_discarded");
        let other = db.clone();
        stoolap::test_failpoints::after_side_files_built(move || {
            other.execute("DROP INDEX idx_t_k ON t", ()).unwrap();
        });
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        let volumes = files(dir.path(), "t", "vol");
        assert_eq!(volumes.len(), 1, "compaction={compaction}");
        assert!(
            files(dir.path(), "t", "sidx").is_empty(),
            "the side file built under the dropped index was discarded, compaction={compaction}"
        );
        assert_eq!(
            index_stat(&db, "index_builds", "sides_discarded"),
            discarded_before + 1,
            "compaction={compaction}"
        );
        let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
        assert_eq!(count, if compaction { 6_000 } else { 1 });
        db.close().unwrap();
    }
}
