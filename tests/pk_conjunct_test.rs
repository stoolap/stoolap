// Copyright 2026 Stoolap Contributors
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

//! A WHERE that ANDs `pk = k` with other conditions reads the one row the
//! key names, hot or sealed, and the other conditions still judge it

use stoolap::Database;

const ROWS: i64 = 1000;

fn fill(db: &Database) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, age INTEGER NOT NULL, v INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_t_age ON t(age)", ()).unwrap();
    db.execute(
        &format!(
            "INSERT INTO t SELECT g.value, g.value % 50, g.value FROM generate_series(1, {ROWS}) g"
        ),
        (),
    )
    .unwrap();
}

fn hot(name: &str) -> Database {
    let db = Database::open(&format!("memory://pk_conjunct_{name}")).unwrap();
    fill(&db);
    db
}

fn sealed(dir: &std::path::Path) -> Database {
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.display()
    ))
    .unwrap();
    fill(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect()
}

fn v_of(db: &Database, id: i64) -> i64 {
    db.query_one(&format!("SELECT v FROM t WHERE id = {id}"), ())
        .unwrap()
}

fn answers_alike(db: &Database) {
    // Row 42 has age 42
    for (sql, want) in [
        ("SELECT id FROM t WHERE id = 42 AND age > 0", vec![42]),
        ("SELECT id FROM t WHERE age > 0 AND id = 42", vec![42]),
        ("SELECT id FROM t WHERE id = 42 AND age > 42", vec![]),
        ("SELECT id FROM t WHERE id = 42 AND age = 42", vec![42]),
        ("SELECT id FROM t WHERE id = 42 AND age = 41", vec![]),
        ("SELECT id FROM t WHERE id = 42 AND v > 0", vec![42]),
        ("SELECT id FROM t WHERE id = 42 AND v < 0", vec![]),
        (
            "SELECT id FROM t WHERE id = 42 AND age > 0 AND v = 42",
            vec![42],
        ),
        (
            "SELECT id FROM t WHERE (id = 42 AND age > 0) AND v > 0",
            vec![42],
        ),
        ("SELECT id FROM t WHERE id = 4242 AND age > 0", vec![]),
        ("SELECT COUNT(*) FROM t WHERE id = 42 AND age > 0", vec![1]),
        ("SELECT COUNT(*) FROM t WHERE id = 42 AND age > 42", vec![0]),
    ] {
        assert_eq!(ids(db, sql), want, "{sql}");
    }
}

fn updates_alike(db: &Database) {
    let n = db
        .execute("UPDATE t SET v = v + 1 WHERE id = 42 AND age > 42", ())
        .unwrap();
    assert_eq!(
        (n, v_of(db, 42)),
        (0, 42),
        "a false conjunct changes no row"
    );
    let n = db
        .execute("UPDATE t SET v = v + 1 WHERE id = 42 AND age > 0", ())
        .unwrap();
    assert_eq!((n, v_of(db, 42)), (1, 43));
    // Once hot, the same row again
    let n = db
        .execute("UPDATE t SET v = v + 1 WHERE id = 42 AND v = 43", ())
        .unwrap();
    assert_eq!((n, v_of(db, 42)), (1, 44));
    // The conditions read the row before the SET
    let n = db
        .execute("UPDATE t SET age = age + 1 WHERE id = 43 AND age = 43", ())
        .unwrap();
    assert_eq!(n, 1);
    assert_eq!(ids(db, "SELECT age FROM t WHERE id = 43"), vec![44]);
    let n = db
        .execute("UPDATE t SET age = age + 1 WHERE id = 43 AND age = 43", ())
        .unwrap();
    assert_eq!(n, 0, "the row no longer has age 43");
    let n = db
        .execute("DELETE FROM t WHERE id = 44 AND age > 44", ())
        .unwrap();
    assert_eq!(n, 0);
    let n = db
        .execute("DELETE FROM t WHERE id = 44 AND age = 44", ())
        .unwrap();
    assert_eq!(n, 1);
    assert_eq!(ids(db, "SELECT id FROM t WHERE id = 44"), Vec::<i64>::new());
    assert_eq!(
        ids(db, "SELECT COUNT(*) FROM t"),
        vec![ROWS - 1],
        "no other row changed"
    );
}

#[test]
fn a_hot_pk_conjunct_answers_as_its_conditions_say() {
    let db = hot("hot_answers");
    answers_alike(&db);
    updates_alike(&db);
}

#[test]
fn a_sealed_pk_conjunct_answers_as_its_conditions_say() {
    let dir = tempfile::tempdir().unwrap();
    let db = sealed(dir.path());
    answers_alike(&db);
    updates_alike(&db);
}

#[test]
fn a_pk_conjunct_sees_the_transactions_own_rows() {
    let db = hot("own_rows");
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO t VALUES (5000, 7, 1)", ()).unwrap();
    let n = tx
        .execute("UPDATE t SET v = 2 WHERE id = 5000 AND age > 0", ())
        .unwrap();
    assert_eq!(n, 1);
    let got: Vec<i64> = tx
        .query("SELECT v FROM t WHERE id = 5000 AND age = 7", ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert_eq!(got, vec![2]);
    tx.rollback().unwrap();
}

#[test]
fn explain_names_the_pk_for_a_pk_conjunct() {
    let db = hot("explain");
    let plan: Vec<String> = db
        .query("EXPLAIN SELECT id FROM t WHERE id = 42 AND age > 0", ())
        .unwrap()
        .map(|r| r.unwrap().get::<String>(0).unwrap())
        .collect();
    let plan = plan.join("\n");
    assert!(
        plan.contains("PkGet t id = 42") && !plan.contains("idx_t_age"),
        "{plan}"
    );
}

#[cfg(feature = "test-failpoints")]
mod reads {
    use super::*;
    use stoolap::test_failpoints as fp;

    #[test]
    fn a_hot_pk_conjunct_reads_no_index_range() {
        let db = hot("hot_reads");
        for sql in [
            "SELECT id FROM t WHERE id = 42 AND age > 0",
            "SELECT COUNT(*) FROM t WHERE id = 42 AND age > 0",
            "UPDATE t SET v = v + 1 WHERE id = 42 AND age > 0",
            "DELETE FROM t WHERE id = 42 AND age > 1000",
        ] {
            let before = fp::hot_index_ids_read();
            db.execute(sql, ()).unwrap();
            assert_eq!(fp::hot_index_ids_read() - before, 0, "{sql}");
        }
    }

    #[test]
    fn a_sealed_pk_conjunct_update_walks_no_cold_row() {
        let dir = tempfile::tempdir().unwrap();
        let db = sealed(dir.path());
        let before = fp::cold_update_rows_walked();
        let n = db
            .execute("UPDATE t SET v = v + 1 WHERE id = 42 AND age > 0", ())
            .unwrap();
        assert_eq!(n, 1);
        assert_eq!(fp::cold_update_rows_walked() - before, 0);
    }
}
