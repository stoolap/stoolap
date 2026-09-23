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

use stoolap::storage::volume::secondary::IndexFile;
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
        let identity = db.engine().index_identity("t", "idx_t_k").unwrap();
        assert!(side.covers(1, identity));
        assert!(!side.covers(1, identity + 1), "another index's identity");
        assert!(
            !side.covers(2, identity),
            "the text column has no side entry"
        );
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
fn reopen_attaches_the_side_files_and_leaves_one_it_cannot_read_in_place() {
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

    // One side file is cut short: its volume reopens uncovered, the file
    // stays where it is (a failed open does not show it is bad), the other
    // is attached
    let broken = sides.iter().next().unwrap().clone();
    let broken_path = dir
        .path()
        .join("volumes")
        .join("t")
        .join(format!("{broken}.sidx"));
    let bytes = std::fs::read(&broken_path).unwrap();
    std::fs::write(&broken_path, &bytes[..bytes.len() - 7]).unwrap();
    let db = open(dir.path(), "");
    assert_eq!(
        files(dir.path(), "t", "sidx"),
        sides,
        "no side file was removed"
    );
    assert_eq!(files(dir.path(), "t", "vol").len(), 2, "both volumes stay");
    let directories = index_stat(&db, "index_directories", "charged_bytes");
    assert!(directories > 0, "the readable side file is attached");
    let healthy = sides.iter().nth(1).unwrap();
    assert_eq!(
        directories as usize,
        side_of(dir.path(), "t", healthy).directory().bytes(),
        "only the readable side file's directory is resident"
    );
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
        // The DROP itself discards the inputs' files, the compaction its
        // prepared output's
        assert_eq!(
            index_stat(&db, "index_builds", "sides_discarded"),
            discarded_before + if compaction { 4 } else { 1 },
            "compaction={compaction}"
        );
        let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
        assert_eq!(count, if compaction { 6_000 } else { 1 });
        db.close().unwrap();
    }
}

fn identity(db: &Database, index: &str) -> u64 {
    db.engine()
        .index_identity("t", index)
        .unwrap_or_else(|| panic!("{index} has no identity"))
}

/// An index dropped and created again with the same definition is a new
/// index: its identity is new, the checkpoint's catalog copy and the
/// reopen keep it, and the old side file, attached at reopen, does not
/// cover the column for it.
#[test]
fn an_index_recreated_with_the_same_definition_gets_a_new_identity_and_the_old_side_file_goes() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    seal_rows(&db, 1, 2_000);
    let old = identity(&db, "idx_t_k");
    let stem = files(dir.path(), "t", "vol").into_iter().next().unwrap();
    assert!(side_of(dir.path(), "t", &stem).covers(1, old));
    db.execute("DROP INDEX idx_t_k ON t", ()).unwrap();
    assert!(db.engine().index_identity("t", "idx_t_k").is_none());
    assert!(
        files(dir.path(), "t", "sidx").is_empty(),
        "the old file goes with the index it stood for"
    );
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    let new = identity(&db, "idx_t_k");
    assert!(new > old, "the identity only grows: {old} then {new}");
    // The checkpoint truncates the WAL and copies the catalog; the copy
    // carries the identity issued at creation
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(identity(&db, "idx_t_k"), new);
    db.close().unwrap();
    drop(db);
    let db = open(dir.path(), "");
    assert_eq!(identity(&db, "idx_t_k"), new, "the reopen reads the copy");
    assert!(
        files(dir.path(), "t", "sidx").is_empty(),
        "the reopen attaches no file to the old volume"
    );
    // The next seal covers the new index; a backfill pass or a compaction
    // covers the old volume for it
    seal_rows(&db, 2_001, 2_000);
    let covered = files(dir.path(), "t", "vol")
        .into_iter()
        .filter(|s| side_of(dir.path(), "t", s).covers(1, new))
        .count();
    assert_eq!(covered, 1);
    db.close().unwrap();
}

/// Index names are keyed as the catalog keys them: two indexes whose names
/// differ in case are two definitions with two identities, and dropping
/// one leaves the other's.
#[test]
fn index_names_that_differ_in_case_keep_their_own_identities() {
    let db = Database::open("memory://secondary_side_file_test_name_case").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX Mixed ON t(k)", ()).unwrap();
    db.execute("CREATE INDEX mixed ON t(v)", ()).unwrap();
    let upper = identity(&db, "Mixed");
    let lower = identity(&db, "mixed");
    assert_ne!(upper, lower, "two definitions, two identities");
    db.execute("DROP INDEX Mixed ON t", ()).unwrap();
    assert!(db.engine().index_identity("t", "Mixed").is_none());
    assert_eq!(
        identity(&db, "mixed"),
        lower,
        "the other index keeps its identity"
    );
}

/// A memory engine issues distinct identities from its own counter.
#[test]
fn a_memory_engine_issues_distinct_identities() {
    let db = Database::open("memory://secondary_side_file_test_identities").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    db.execute("CREATE INDEX idx_t_v ON t(v)", ()).unwrap();
    let k = identity(&db, "idx_t_k");
    let v = identity(&db, "idx_t_v");
    assert_ne!(k, v);
    db.execute("DROP INDEX idx_t_k ON t", ()).unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    assert!(identity(&db, "idx_t_k") > v);
}

/// Records written before identities carry none: each index takes its own
/// record's LSN at replay, so several legacy indexes stay distinct, and the
/// next checkpoint's copy keeps what replay resolved. A record's first
/// write today is the same shape, so the reopen resolves the identity
/// creation issued.
#[test]
fn legacy_records_resolve_distinct_identities_at_replay_and_keep_them_through_a_checkpoint() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    db.execute("CREATE INDEX idx_t_v ON t(v)", ()).unwrap();
    let (k, v) = (identity(&db, "idx_t_k"), identity(&db, "idx_t_v"));
    assert_ne!(k, v);
    // No checkpoint: the reopen replays the two first-write records
    db.close().unwrap();
    drop(db);
    let db = open(dir.path(), "");
    assert_eq!(
        identity(&db, "idx_t_k"),
        k,
        "replay resolves the record's LSN"
    );
    assert_eq!(identity(&db, "idx_t_v"), v);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.close().unwrap();
    drop(db);
    let db = open(dir.path(), "");
    assert_eq!(identity(&db, "idx_t_k"), k, "the catalog copy keeps it");
    assert_eq!(identity(&db, "idx_t_v"), v);
    db.close().unwrap();
}

/// A restore from a snapshot whose DDL file predates identities creates the
/// indexes without any; each takes its identity from its own catalog copy
/// after the restore, distinct, and the reopen keeps it.
#[test]
fn a_legacy_restore_gives_each_index_its_own_identity() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1, 1), (2, 2, 2)", ())
        .unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    db.execute("CREATE INDEX idx_t_v ON t(v)", ()).unwrap();
    db.execute("PRAGMA SNAPSHOT", ()).unwrap();
    // The snapshot's DDL file is rewritten as a version before identities
    // wrote it: each index entry loses its trailing identity, and the
    // checksum is recomputed
    let snapshot_dir = dir.path().join("snapshots");
    let ddl_path = std::fs::read_dir(&snapshot_dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("ddl-") && n.ends_with(".bin"))
        })
        .expect("a ddl file");
    let data = std::fs::read(&ddl_path).unwrap();
    let payload = &data[..data.len() - 4];
    let mut out = payload[..5].to_vec();
    let mut pos = 5usize;
    let count = u32::from_le_bytes(payload[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    out.extend_from_slice(&(count as u32).to_le_bytes());
    let mut stripped = 0;
    for _ in 0..count {
        let len = u32::from_le_bytes(payload[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let entry = &payload[pos..pos + len];
        pos += len;
        let legacy = &entry[..len - 8];
        stripped += 1;
        out.extend_from_slice(&(legacy.len() as u32).to_le_bytes());
        out.extend_from_slice(legacy);
    }
    assert_eq!(stripped, 2);
    out.extend_from_slice(&payload[pos..]);
    let crc = crc32fast::hash(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    std::fs::write(&ddl_path, &out).unwrap();
    db.execute("PRAGMA RESTORE", ()).unwrap();
    let (k, v) = (identity(&db, "idx_t_k"), identity(&db, "idx_t_v"));
    assert_ne!(k, v, "each index took its own catalog copy's LSN");
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE k = 2", ())
        .unwrap();
    assert_eq!(count, 1);
    db.close().unwrap();
    drop(db);
    let db = open(dir.path(), "");
    assert_eq!(identity(&db, "idx_t_k"), k);
    assert_eq!(identity(&db, "idx_t_v"), v);
    db.close().unwrap();
}

/// DDL that lands between the comparison and the binding waits for the DDL
/// guard: the side file attaches for the index it matched, the recreated
/// index gets a new identity, and the file does not cover the column for
/// it. On the seal path and on the compaction path.
#[cfg(feature = "test-failpoints")]
#[test]
fn index_ddl_between_the_comparison_and_the_binding_does_not_bind_stale_coverage() {
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
        let old = identity(&db, "idx_t_k");
        let other = db.clone();
        let (done, waiter) = std::sync::mpsc::channel();
        stoolap::test_failpoints::after_side_files_compared(move || {
            // The DDL runs on its own thread and waits for the guard the
            // comparison holds; the checkpoint goes on meanwhile
            std::thread::spawn(move || {
                other.execute("DROP INDEX idx_t_k ON t", ()).unwrap();
                other.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
                done.send(()).unwrap();
            });
        });
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        waiter
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("the DDL completed after the guard was released");
        let new = identity(&db, "idx_t_k");
        assert!(new > old, "compaction={compaction}");
        let volumes = files(dir.path(), "t", "vol");
        assert_eq!(volumes.len(), 1, "compaction={compaction}");
        // The file bound for the index it was compared against stood for
        // that index alone, so the DROP that waited for the guard took it
        // out: the recreated index has no file to bind to
        assert!(
            files(dir.path(), "t", "sidx").is_empty(),
            "the recreated index does not bind to the file, compaction={compaction}"
        );
        let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
        assert_eq!(count, if compaction { 6_000 } else { 1 });
        db.close().unwrap();
    }
}

/// A change that replaces the segments between a compaction's preparation
/// and its commit (a column added, which rebuilds every mapping) makes the
/// preparation stale: the commit hands it back, the compaction prepares
/// again outside the guard, and the published table holds the output with
/// its side file, every row once, the new column on every row.
#[cfg(feature = "test-failpoints")]
#[test]
fn a_compaction_whose_preparation_went_stale_prepares_again_and_publishes_whole() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "&compact_threshold=100");
    create(&db);
    seal_rows(&db, 1, 2_000);
    seal_rows(&db, 2_001, 2_000);
    seal_rows(&db, 4_001, 2_000);
    db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
    let other = db.clone();
    stoolap::test_failpoints::after_side_files_built(move || {
        // Outputs built and the publication prepared: the column change
        // lands from another thread before the compaction takes the guard
        std::thread::spawn(move || {
            other
                .execute("ALTER TABLE t ADD COLUMN extra INTEGER DEFAULT 5", ())
                .unwrap();
        })
        .join()
        .unwrap();
    });
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let volumes = files(dir.path(), "t", "vol");
    assert_eq!(volumes.len(), 1, "the compaction's output");
    assert_eq!(files(dir.path(), "t", "sidx"), volumes);
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 6_000);
    let with_key: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE k = 3", ())
        .unwrap();
    assert_eq!(
        with_key,
        (1..=6_000i64).filter(|i| i % 7 == 3).count() as i64
    );
    let with_extra: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE extra = 5", ())
        .unwrap();
    assert_eq!(with_extra, 6_000, "the added column reaches every row");
    db.close().unwrap();
}

/// An eviction that replaces the segments after the comparison under the
/// guard and before the commit takes no DDL guard: the commit finds the
/// preparation stale under its own write locks and hands it back before
/// changing anything, and the compaction prepares again and publishes
/// whole.
#[cfg(feature = "test-failpoints")]
#[test]
fn an_eviction_between_the_comparison_and_the_commit_hands_the_preparation_back() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "&compact_threshold=100");
    create(&db);
    seal_rows(&db, 1, 2_000);
    seal_rows(&db, 2_001, 2_000);
    seal_rows(&db, 4_001, 2_000);
    db.execute("PRAGMA COMPACT_THRESHOLD = 2", ()).unwrap();
    let other = db.clone();
    stoolap::test_failpoints::after_side_files_compared(move || {
        // The inputs are evicted to metadata-only, which publishes a new
        // segments map without any DDL coordination
        let (volumes, cold) = other.engine().cold_volumes_for_test("t");
        assert_eq!((volumes, cold), (3, 3), "the inputs were evicted");
    });
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let volumes = files(dir.path(), "t", "vol");
    assert_eq!(volumes.len(), 1, "the compaction's output");
    assert_eq!(files(dir.path(), "t", "sidx"), volumes);
    let count: i64 = db.query_one("SELECT COUNT(*) FROM t", ()).unwrap();
    assert_eq!(count, 6_000);
    let with_key: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE k = 3", ())
        .unwrap();
    assert_eq!(
        with_key,
        (1..=6_000i64).filter(|i| i % 7 == 3).count() as i64
    );
    db.close().unwrap();
}

/// A CREATE INDEX that arrives while a seal is between its registration and
/// its hot cleanup waits for the guard: the new index holds no entry for a
/// sealed row, and answers the rows from the cold side.
#[cfg(feature = "test-failpoints")]
#[test]
fn create_index_racing_the_seal_s_cleanup_holds_no_entry_for_a_sealed_row() {
    use stoolap::storage::index::BTreeIndex;

    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, k2 INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    let mut values = String::new();
    for id in 1..=5_000i64 {
        values.push_str(&format!("({id},{},{}),", id % 7, id % 11));
    }
    values.pop();
    db.execute(&format!("INSERT INTO t VALUES {values}"), ())
        .unwrap();
    let other = db.clone();
    let (done, waiter) = std::sync::mpsc::channel();
    stoolap::test_failpoints::after_side_files_compared(move || {
        std::thread::spawn(move || {
            other.execute("CREATE INDEX idx_t_k2 ON t(k2)", ()).unwrap();
            done.send(()).unwrap();
        });
    });
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    waiter
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("the CREATE INDEX completed after the seal");
    let store = db.engine().get_version_store("t").unwrap();
    let index = store.get_index("idx_t_k2").expect("the index exists");
    let entries = index
        .as_any()
        .downcast_ref::<BTreeIndex>()
        .expect("a B-tree index")
        .entry_count();
    assert_eq!(entries, 0, "no sealed row is in the new hot index");
    let count: i64 = db
        .query_one("SELECT COUNT(*) FROM t WHERE k2 = 3", ())
        .unwrap();
    assert_eq!(count, (1..=5_000i64).filter(|i| i % 11 == 3).count() as i64);
    db.close().unwrap();
}

/// Closing a scanner lets its side walk's reader and reservation go, as
/// dropping it does.
#[test]
fn closing_a_scanner_releases_its_side_reader() {
    use stoolap::core::{DataType, Row, SchemaBuilder, Value};
    use stoolap::storage::traits::Scanner;
    use stoolap::storage::volume::secondary::{
        build_side_file, next_generation, ColumnInput, SidePlan, INDEX_PAGES,
    };
    use stoolap::storage::volume::{scanner::VolumeScanner, writer::VolumeBuilder};
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.sidx");
    build_side_file(
        &path,
        next_generation(),
        vec![ColumnInput {
            column: 1,
            identity: 77,
            pairs: Box::new((0..20_000u32).map(|p| (p, 1i64))),
        }],
        4 * 1024 * 1024,
    )
    .unwrap();
    let file = std::sync::Arc::new(IndexFile::open(&path, 20_803).unwrap());
    let schema = SchemaBuilder::new("t")
        .column("id", DataType::Integer, false, true)
        .column("k", DataType::Integer, false, false)
        .build();
    let mut builder = VolumeBuilder::new(&schema);
    for id in 0..20_000 {
        builder.add_row(
            id,
            &Row::from_values(vec![Value::Integer(id), Value::Integer(1)]),
        );
    }
    let volume = std::sync::Arc::new(builder.finish().unwrap());
    INDEX_PAGES.clear();
    let idle = INDEX_PAGES.stats().charged_bytes;
    let mut scanner = VolumeScanner::new(volume, Vec::new(), None).unwrap();
    scanner.set_side_plan(SidePlan::new(std::sync::Arc::clone(&file), 1, (0, 20_000)));
    assert!(scanner.next());
    assert!(
        INDEX_PAGES.stats().charged_bytes > idle,
        "the walk's reader holds its reservation"
    );
    scanner.close().unwrap();
    INDEX_PAGES.clear();
    assert_eq!(
        INDEX_PAGES.stats().charged_bytes,
        idle,
        "close let the reservation go"
    );
    drop(scanner);
}

fn count_where(db: &Database, sql: &str) -> i64 {
    db.query(sql, ())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap()
}

/// Dropping the only index a volume's side file serves removes the file
/// from the segment and from the disk, and the queries go on without it
#[test]
fn dropping_the_last_index_a_side_file_serves_removes_the_file() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    seal_rows(&db, 1, 700);
    assert_eq!(files(dir.path(), "t", "sidx").len(), 1);

    db.execute("DROP INDEX idx_t_k ON t", ()).unwrap();
    assert!(
        files(dir.path(), "t", "sidx").is_empty(),
        "the side file goes with its last index"
    );
    assert_eq!(count_where(&db, "SELECT COUNT(*) FROM t WHERE k = 3"), 100);
    assert_eq!(
        count_where(&db, "SELECT COUNT(*) FROM t WHERE name = 'n1'"),
        234
    );
    db.close().unwrap();
    drop(db);

    let db = open(dir.path(), "");
    assert!(files(dir.path(), "t", "sidx").is_empty());
    assert_eq!(count_where(&db, "SELECT COUNT(*) FROM t WHERE k = 3"), 100);
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    seal_rows(&db, 701, 7);
    assert_eq!(files(dir.path(), "t", "sidx").len(), 1);
}

/// A side file serving two indexes stays while either stands, and goes
/// with the second drop
#[test]
fn a_side_file_serving_two_indexes_goes_with_the_second_drop() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    db.execute(
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, m INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_u_k ON u(k)", ()).unwrap();
    db.execute("CREATE INDEX idx_u_m ON u(m)", ()).unwrap();
    let mut values = String::new();
    for id in 1..=700 {
        values.push_str(&format!("({id},{},{}),", id % 7, id % 5));
    }
    values.pop();
    db.execute(&format!("INSERT INTO u VALUES {values}"), ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(files(dir.path(), "u", "sidx").len(), 1);

    db.execute("DROP INDEX idx_u_k ON u", ()).unwrap();
    assert_eq!(
        files(dir.path(), "u", "sidx").len(),
        1,
        "the file still serves the index on m"
    );
    assert_eq!(count_where(&db, "SELECT COUNT(*) FROM u WHERE m = 2"), 140);
    assert_eq!(count_where(&db, "SELECT COUNT(*) FROM u WHERE k = 2"), 100);

    db.execute("DROP INDEX idx_u_m ON u", ()).unwrap();
    assert!(files(dir.path(), "u", "sidx").is_empty());
    assert_eq!(count_where(&db, "SELECT COUNT(*) FROM u WHERE m = 2"), 140);
}

/// A side file the drop did not reach before the process ended is
/// discarded when the log replays the drop at reopen
#[test]
fn a_replayed_drop_discards_the_side_file_it_left_behind() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    seal_rows(&db, 1, 700);
    let stem = files(dir.path(), "t", "sidx").into_iter().next().unwrap();
    let path = dir
        .path()
        .join("volumes")
        .join("t")
        .join(format!("{stem}.sidx"));
    let kept = std::fs::read(&path).unwrap();

    db.execute("DROP INDEX idx_t_k ON t", ()).unwrap();
    assert!(!path.exists());
    db.close().unwrap();
    drop(db);
    // The file as it was before the drop, back where the reopen looks
    std::fs::write(&path, kept).unwrap();

    let db = open(dir.path(), "");
    assert!(
        files(dir.path(), "t", "sidx").is_empty(),
        "the replayed drop discards the file"
    );
    assert_eq!(count_where(&db, "SELECT COUNT(*) FROM t WHERE k = 3"), 100);
}

/// A DROP the log replays at reopen is not the catalog's last word: the
/// side file a later CREATE and backfill gave the volume stays, and the
/// reopened index probes it
#[test]
fn a_replayed_drop_leaves_the_file_of_the_index_created_after_it() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    create(&db);
    seal_rows(&db, 1, 700);
    db.execute("DROP INDEX idx_t_k ON t", ()).unwrap();
    assert!(files(dir.path(), "t", "sidx").is_empty());
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    db.execute("PRAGMA INDEX_BACKFILL", ()).unwrap();
    let new = identity(&db, "idx_t_k");
    // The backfill writes the file under its generation's name
    let sides = files(dir.path(), "t", "sidx");
    assert_eq!(sides.len(), 1);
    let side = sides.into_iter().next().unwrap();
    assert!(side_of(dir.path(), "t", &side).covers(1, new));
    db.close().unwrap();
    drop(db);

    let db = open(dir.path(), "");
    assert_eq!(identity(&db, "idx_t_k"), new);
    assert_eq!(
        files(dir.path(), "t", "sidx")
            .into_iter()
            .collect::<Vec<_>>(),
        vec![side.clone()],
        "the file survives the replay"
    );
    assert!(side_of(dir.path(), "t", &side).covers(1, new));
    assert_eq!(count_where(&db, "SELECT COUNT(*) FROM t WHERE k = 3"), 100);
    db.close().unwrap();
}

/// Whether a file still serves an index is decided at the position the
/// volume holds the column, so a column dropped before it does not make
/// the file look uncovered
#[test]
fn dropping_an_index_after_a_column_drop_keeps_the_file_of_the_other_index() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    db.execute(
        "CREATE TABLE u (id INTEGER PRIMARY KEY, padding INTEGER, k INTEGER, m INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_u_k ON u(k)", ()).unwrap();
    db.execute("CREATE INDEX idx_u_m ON u(m)", ()).unwrap();
    let mut values = String::new();
    for id in 1..=700 {
        values.push_str(&format!("({id},0,{},{}),", id % 7, id % 5));
    }
    values.pop();
    db.execute(&format!("INSERT INTO u VALUES {values}"), ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let k = db.engine().index_identity("u", "idx_u_k").unwrap();
    let stem = files(dir.path(), "u", "vol").into_iter().next().unwrap();
    db.execute("ALTER TABLE u DROP COLUMN padding", ()).unwrap();
    // k is the schema's first column now and the volume's second
    assert!(side_of(dir.path(), "u", &stem).covers(2, k));
    db.execute("DROP INDEX idx_u_m ON u", ()).unwrap();
    assert_eq!(
        files(dir.path(), "u", "sidx").len(),
        1,
        "the file still serves k"
    );
    assert!(side_of(dir.path(), "u", &stem).covers(2, k));
    assert_eq!(count_where(&db, "SELECT COUNT(*) FROM u WHERE k = 3"), 100);
    db.execute("DROP INDEX idx_u_k ON u", ()).unwrap();
    assert!(files(dir.path(), "u", "sidx").is_empty());
    db.close().unwrap();
}

/// A column dropped and checkpointed before a reopen moves the indexed
/// column's schema position: the reopen judges the file through the
/// mappings it restores, and keeps it
#[test]
fn a_reopen_after_a_checkpointed_column_drop_keeps_the_side_file() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path(), "");
    db.execute(
        "CREATE TABLE u (id INTEGER PRIMARY KEY, padding INTEGER, k INTEGER, m INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_u_k ON u(k)", ()).unwrap();
    let mut values = String::new();
    for id in 1..=700 {
        values.push_str(&format!("({id},0,{},{}),", id % 7, id % 5));
    }
    values.pop();
    db.execute(&format!("INSERT INTO u VALUES {values}"), ())
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let k = db.engine().index_identity("u", "idx_u_k").unwrap();
    db.execute("ALTER TABLE u DROP COLUMN padding", ()).unwrap();
    // The checkpoint takes the drop into the catalog copy, so the reopen
    // restores the final schema without replaying the ALTER
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    assert_eq!(files(dir.path(), "u", "sidx").len(), 1);
    db.close().unwrap();
    drop(db);

    let db = open(dir.path(), "");
    assert_eq!(db.engine().index_identity("u", "idx_u_k"), Some(k));
    let sides = files(dir.path(), "u", "sidx");
    assert_eq!(sides.len(), 1, "the file survives the reopen");
    let side = sides.into_iter().next().unwrap();
    assert!(side_of(dir.path(), "u", &side).covers(2, k));
    assert_eq!(count_where(&db, "SELECT COUNT(*) FROM u WHERE k = 3"), 100);
    db.close().unwrap();
}
