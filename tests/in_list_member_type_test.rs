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

//! A literal IN list whose members are of another type than the column
//! answers as the same list does without an index on the column

use stoolap::Database;

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    out.sort_unstable();
    out
}

/// Column type, a stored value, and lists of members of other types
const CASES: [(&str, &str, &[&str]); 4] = [
    (
        "TIMESTAMP",
        "'2024-01-01 00:00:00'",
        &[
            "'2024-01-01'",
            "'2024-01-01 00:00:00'",
            "'2024-01-01T00:00:00'",
        ],
    ),
    ("BOOLEAN", "TRUE", &["'true'", "1", "'1'"]),
    ("INTEGER", "1", &["TRUE", "'1'", "1.0"]),
    ("TEXT", "'true'", &["TRUE"]),
];

fn check(db: &Database, phase: &str) {
    let mut wrong = Vec::new();
    for (n, (ty, _, members)) in CASES.iter().enumerate() {
        for member in *members {
            let sql = format!("SELECT id FROM {{t}}{n} WHERE k IN ({member})");
            let want = ids(db, &sql.replace("{t}", "plain"));
            for table in ["btree", "hash"] {
                let got = ids(db, &sql.replace("{t}", table));
                if got != want {
                    wrong.push(format!(
                        "{phase}, {table}, {ty}: k IN ({member}) gave {got:?}, want {want:?}"
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

fn create(db: &Database) {
    for (n, (ty, stored, _)) in CASES.iter().enumerate() {
        for (table, index) in [("plain", ""), ("btree", "BTREE"), ("hash", "HASH")] {
            db.execute(
                &format!("CREATE TABLE {table}{n} (id INTEGER PRIMARY KEY, k {ty})"),
                (),
            )
            .unwrap();
            if !index.is_empty() {
                db.execute(
                    &format!("CREATE INDEX {table}{n}_k ON {table}{n}(k) USING {index}"),
                    (),
                )
                .unwrap();
            }
            db.execute(&format!("INSERT INTO {table}{n} VALUES (1, {stored})"), ())
                .unwrap();
        }
    }
}

#[test]
fn members_of_another_type_answer_as_without_an_index_in_memory() {
    let db = Database::open("memory://members_of_another_type_in_memory").unwrap();
    create(&db);
    check(&db, "memory");
}

#[test]
fn members_of_another_type_answer_as_without_an_index_hot_and_sealed() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    create(&db);
    check(&db, "hot");
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    check(&db, "sealed");
}

#[test]
fn a_member_with_no_reading_or_a_json_column_answers_as_without_an_index() {
    let dir = tempfile::tempdir().unwrap();
    let file = format!("file://{}", dir.path().display());
    for dsn in ["memory://a_member_with_no_reading", file.as_str()] {
        let db = Database::open(dsn).unwrap();
        for (table, ty, index, stored) in [
            ("ints_plain", "INTEGER", "", "1"),
            ("ints_btree", "INTEGER", "BTREE", "1"),
            ("ints_hash", "INTEGER", "HASH", "1"),
            ("jsons_plain", "JSON", "", "'{}'"),
            ("jsons_hash", "JSON", "HASH", "'{}'"),
        ] {
            db.execute(
                &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, k {ty})"),
                (),
            )
            .unwrap();
            if !index.is_empty() {
                db.execute(
                    &format!("CREATE INDEX {table}_k ON {table}(k) USING {index}"),
                    (),
                )
                .unwrap();
            }
            db.execute(&format!("INSERT INTO {table} VALUES (1, {stored})"), ())
                .unwrap();
        }
        for (sql, tables) in [
            (
                "SELECT id FROM {t} WHERE k IN (TRUE, 'bad')",
                &["ints_btree", "ints_hash"][..],
            ),
            (
                "SELECT id FROM {t} WHERE k IN ('{}') AND id % 2 = 1",
                &["jsons_hash"][..],
            ),
        ] {
            let plain = if sql.contains("'{}'") {
                "jsons_plain"
            } else {
                "ints_plain"
            };
            let want = ids(&db, &sql.replace("{t}", plain));
            for table in tables {
                assert_eq!(
                    ids(&db, &sql.replace("{t}", table)),
                    want,
                    "{dsn}: {}",
                    sql.replace("{t}", table)
                );
            }
        }
    }
}
