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

//! An index name is matched without regard to case, as table and column
//! names are: DROP INDEX finds the index by any spelling, CREATE INDEX
//! refuses a name taken under another spelling, and a database that already
//! holds two such names keeps both and drops only the one named exactly.

use std::path::Path;

use stoolap::Database;

fn setup(db: &Database) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER, w INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10, 1, 1), (2, 20, 2, 2)", ())
        .unwrap();
    db.execute("CREATE INDEX IdxK ON t(k)", ()).unwrap();
}

/// `(index, column)` pairs of `t`, as SHOW INDEXES lists them
fn indexes(db: &Database) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = db
        .query("SHOW INDEXES FROM t", ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get(1).unwrap(), r.get(2).unwrap())
        })
        .collect();
    out.sort();
    out
}

fn pairs(list: &[(&str, &str)]) -> Vec<(String, String)> {
    list.iter()
        .map(|(i, c)| (i.to_string(), c.to_string()))
        .collect()
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    let mut ids: Vec<i64> = db
        .query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get(0).unwrap())
        .collect();
    ids.sort_unstable();
    ids
}

fn log_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir.join("wal"))
        .unwrap()
        .map(|entry| entry.unwrap().metadata().unwrap().len())
        .sum()
}

fn copy_dir(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let to = dst.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &to);
        } else {
            std::fs::copy(entry.path(), &to).unwrap();
        }
    }
}

#[test]
fn drop_index_finds_the_index_by_another_spelling() {
    let db = Database::open("memory://index_name_case_drop").unwrap();
    setup(&db);
    db.execute("DROP INDEX idxk ON t", ()).unwrap();
    assert!(indexes(&db).is_empty());
    db.execute("CREATE INDEX IDXK ON t(k)", ()).unwrap();
    db.execute("DROP INDEX IF EXISTS idxK ON t", ()).unwrap();
    assert!(indexes(&db).is_empty());
}

#[test]
fn create_index_refuses_a_name_taken_under_another_spelling() {
    let db = Database::open("memory://index_name_case_create").unwrap();
    setup(&db);
    assert!(db.execute("CREATE INDEX idxk ON t(v)", ()).is_err());
    db.execute("CREATE INDEX IF NOT EXISTS IDXK ON t(v)", ())
        .unwrap();
    assert_eq!(indexes(&db), pairs(&[("IdxK", "k")]));
}

#[test]
fn a_non_ascii_index_name_is_matched_as_the_schema_matches_names() {
    let db = Database::open("memory://index_name_case_unicode").unwrap();
    setup(&db);
    db.execute("CREATE INDEX \"iä\" ON t(v)", ()).unwrap();
    assert_eq!(indexes(&db), pairs(&[("IdxK", "k"), ("iä", "v")]));
    assert!(db.execute("CREATE INDEX \"IÄ\" ON t(w)", ()).is_err());
    db.execute("DROP INDEX \"IÄ\" ON t", ()).unwrap();
    assert_eq!(indexes(&db), pairs(&[("IdxK", "k")]));
}

#[test]
fn a_refused_create_writes_no_record() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}?sync_mode=full", dir.path().display())).unwrap();
    setup(&db);
    let logged = log_bytes(dir.path());
    assert!(db.execute("CREATE INDEX idxk ON t(v)", ()).is_err());
    db.execute("CREATE INDEX IF NOT EXISTS IDXK ON t(v)", ())
        .unwrap();
    assert_eq!(log_bytes(dir.path()), logged);
    assert_eq!(indexes(&db), pairs(&[("IdxK", "k")]));
}

#[test]
fn a_drop_by_another_spelling_survives_replay() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    );
    {
        let db = Database::open(&dsn).unwrap();
        setup(&db);
        db.execute("DROP INDEX idxk ON t", ()).unwrap();
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    assert!(indexes(&db).is_empty(), "after the log is replayed");
    db.execute("CREATE INDEX idxk ON t(k)", ()).unwrap();
}

/// Opens a copy of a database written by a release that accepted `IdxK` on
/// `a` and `idxk` on `b` in one table, from a checkpoint or from the log
#[test]
fn an_existing_pair_of_spellings_is_kept_and_dropped_one_by_one() {
    let both = pairs(&[("IdxK", "a"), ("idxk", "b")]);
    for fixture in ["index_case_pair_checkpoint", "index_case_pair_wal"] {
        let dir = tempfile::tempdir().unwrap();
        copy_dir(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/testdata")
                .join(fixture),
            dir.path(),
        );
        let dsn = format!("file://{}?sync_mode=full", dir.path().display());
        {
            let db = Database::open(&dsn).unwrap();
            assert_eq!(indexes(&db), both, "{fixture}: opened");
            db.close().unwrap();
        }
        let db = Database::open(&dsn).unwrap();
        assert_eq!(indexes(&db), both, "{fixture}: after a checkpoint of ours");
        assert_eq!(ids(&db, "SELECT id FROM t WHERE a = 10"), [1, 3]);
        assert_eq!(ids(&db, "SELECT id FROM t WHERE b = 200"), [2]);

        let logged = log_bytes(dir.path());
        for sql in ["DROP INDEX IDXK ON t", "DROP INDEX IF EXISTS IDXK ON t"] {
            assert!(
                db.execute(sql, ()).is_err(),
                "{fixture}: {sql} is ambiguous"
            );
        }
        assert_eq!(indexes(&db), both, "{fixture}: after the ambiguous drops");
        assert_eq!(log_bytes(dir.path()), logged, "{fixture}: no record");

        let identity = |name: &str| db.engine().index_identity("t", name);
        let lower = identity("idxk");
        assert!(lower.is_some() && identity("IdxK").is_some(), "{fixture}");
        assert_ne!(identity("IdxK"), lower, "{fixture}: two identities");
        db.execute("DROP INDEX IdxK ON t", ()).unwrap();
        assert_eq!(indexes(&db), pairs(&[("idxk", "b")]), "{fixture}");
        assert_eq!(identity("IdxK"), None, "{fixture}");
        assert_eq!(
            identity("idxk"),
            lower,
            "{fixture}: the other keeps its own"
        );
        assert_eq!(ids(&db, "SELECT id FROM t WHERE a = 10"), [1, 3]);
        assert_eq!(ids(&db, "SELECT id FROM t WHERE b = 200"), [2]);
        db.close().unwrap();

        let db = Database::open(&dsn).unwrap();
        assert_eq!(indexes(&db), pairs(&[("idxk", "b")]), "{fixture}: reopened");
        db.execute("DROP INDEX IDXK ON t", ()).unwrap();
        assert!(indexes(&db).is_empty(), "{fixture}");
    }
}
