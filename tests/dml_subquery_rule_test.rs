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

//! A subquery in an UPDATE or DELETE sees the tables as the statement
//! found them, the target included, whatever the statement writes.

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, b INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30), (4, 40)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 1), (2, 2), (3, 3)", ())
        .unwrap();
    db
}

fn rows(db: &Database) -> Vec<String> {
    db.query("SELECT id, b FROM t ORDER BY id", ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            let b: Option<i64> = r.get(1).unwrap();
            format!(
                "{}:{}",
                r.get::<i64>(0).unwrap(),
                b.map_or("NULL".into(), |b| b.to_string())
            )
        })
        .collect()
}

/// Each statement reading its own target, with the rows it leaves when its
/// subqueries see the table as the statement found it
const SELF_READS: &[(&str, &str)] = &[
    (
        "DELETE FROM t WHERE EXISTS (SELECT 1 FROM t t2 WHERE t2.id = t.id - 1)",
        "1:10",
    ),
    (
        "UPDATE t SET b = b + 1 WHERE b = (SELECT t2.b FROM t t2 WHERE t2.id = t.id - 1) + 10",
        "1:10 2:21 3:31 4:41",
    ),
    (
        "UPDATE t SET b = (SELECT t2.b FROM t t2 WHERE t2.id = t.id - 1)",
        "1:NULL 2:10 3:20 4:30",
    ),
    (
        "UPDATE t SET b = (SELECT SUM(t2.b) FROM t t2 WHERE t2.id < t.id) WHERE b > (SELECT MIN(b) FROM t)",
        "1:10 2:10 3:30 4:60",
    ),
];

#[test]
fn a_subquery_reads_the_target_as_the_statement_found_it() {
    let mut wrong = Vec::new();
    for (i, (sql, want)) in SELF_READS.iter().enumerate() {
        for explicit in [false, true] {
            let db = setup(&format!("dml_self_read_{i}_{explicit}"));
            if explicit {
                let mut tx = db.begin().unwrap();
                tx.execute(sql, ()).unwrap();
                tx.commit().unwrap();
            } else {
                db.execute(sql, ()).unwrap();
            }
            let got = rows(&db).join(" ");
            if got != *want {
                wrong.push(format!(
                    "{sql} (transaction: {explicit}): {got}, want {want}"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_set_may_mix_correlated_and_uncorrelated_subqueries() {
    let db = setup("dml_mixed_set");
    db.execute(
        "CREATE TABLE w (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO w VALUES (1, 0, 0), (2, 0, 0), (4, 0, 0)", ())
        .unwrap();
    db.execute(
        "UPDATE w SET a = (SELECT MAX(x) FROM u), b = (SELECT COUNT(*) FROM u WHERE u.id = w.id)",
        (),
    )
    .unwrap();
    let got: Vec<String> = db
        .query("SELECT id, a, b FROM w ORDER BY id", ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            let (id, a, b): (i64, i64, i64) =
                (r.get(0).unwrap(), r.get(1).unwrap(), r.get(2).unwrap());
            format!("{id}:{a}:{b}")
        })
        .collect();
    assert_eq!(got.join(" "), "1:3:1 2:3:1 4:3:0");
}

#[test]
fn a_scalar_subquery_of_two_rows_fails_the_statement_and_writes_nothing() {
    let db = setup("dml_two_rows");
    for sql in [
        "UPDATE t SET b = (SELECT x FROM u WHERE u.id >= t.id - 1)",
        "DELETE FROM t WHERE b > (SELECT x FROM u WHERE u.id <= t.id)",
    ] {
        assert!(db.execute(sql, ()).is_err(), "{sql} was accepted");
        assert_eq!(rows(&db).join(" "), "1:10 2:20 3:30 4:40", "{sql} wrote");
    }
}

#[test]
fn a_prepared_dml_statement_reads_each_execution_s_parameters() {
    let db = setup("dml_params");
    let update = db
        .prepare("UPDATE t SET b = (SELECT x FROM u WHERE u.id = $1) + t.id WHERE id = $2")
        .unwrap();
    update.execute((1i64, 1i64)).unwrap();
    update.execute((3i64, 2i64)).unwrap();
    let delete = db
        .prepare("DELETE FROM t WHERE b > (SELECT MAX(x) FROM u WHERE u.id < $1)")
        .unwrap();
    delete.execute((3i64,)).unwrap();
    assert_eq!(rows(&db).join(" "), "1:2");
}

#[test]
fn a_cached_dml_statement_follows_a_schema_change() {
    let db = setup("dml_schema_change");
    let sql = "UPDATE t SET b = (SELECT MAX(x) FROM u WHERE u.id <= t.id)";
    db.execute(sql, ()).unwrap();
    db.execute("ALTER TABLE u ADD COLUMN y INTEGER DEFAULT 0", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (4, 9, 0)", ()).unwrap();
    db.execute(sql, ()).unwrap();
    assert_eq!(rows(&db).join(" "), "1:1 2:2 3:3 4:9");
}
