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

//! An equality on a VECTOR column with an HNSW index answers as without
//! the index: the index finds neighbours, not equal keys

use stoolap::{Database, Value};

fn rows(db: &Database, sql: &str, params: Vec<Value>) -> Vec<Vec<Value>> {
    db.query(sql, params)
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| row.get::<Value>(i).unwrap())
                .collect()
        })
        .collect()
}

fn create(db: &Database) {
    for (t, s, index) in [("t", "s", true), ("pt", "ps", false)] {
        for sql in [
            format!("CREATE TABLE {t} (id INTEGER PRIMARY KEY, v VECTOR(3))"),
            format!("CREATE TABLE {s} (id INTEGER PRIMARY KEY, v VECTOR(3))"),
            format!(
                "INSERT INTO {t} VALUES (1, '[1,2,3]'), (2, '[4,5,6]'), (3, '[1,2,3]'), (4, '[7,8,9]')"
            ),
            format!("INSERT INTO {s} VALUES (1, '[1,2,3]'), (2, '[4,5,6]'), (3, '[0,0,1]')"),
        ] {
            db.execute(&sql, ()).unwrap();
        }
        if index {
            db.execute("CREATE INDEX t_v ON t(v) USING HNSW", ())
                .unwrap();
            db.execute("CREATE INDEX s_v ON s(v) USING HNSW", ())
                .unwrap();
        }
    }
}

fn check(db: &Database, phase: &str) {
    let members = || {
        vec![
            Value::vector(vec![1.0, 2.0, 3.0]),
            Value::vector(vec![4.0, 5.0, 6.0]),
        ]
    };
    for (sql, params) in [
        ("SELECT id FROM {t} WHERE v = $1 ORDER BY id", members()[..1].to_vec()),
        ("SELECT id FROM {t} WHERE v IN ($1) ORDER BY id", members()[..1].to_vec()),
        ("SELECT id FROM {t} WHERE v IN ($1, $2) ORDER BY id", members()),
        ("SELECT id FROM {t} WHERE v IN ($1, $2) LIMIT 10", members()),
        ("SELECT id FROM {t} WHERE v IN ($1, $2) AND id > 0 ORDER BY id", members()),
        ("SELECT {s}.id, {t}.id FROM {s} JOIN {t} ON {t}.v = {s}.v ORDER BY 1, 2", vec![]),
        ("SELECT {s}.id, {t}.id FROM {s} LEFT JOIN {t} ON {t}.v = {s}.v ORDER BY 1, 2", vec![]),
        ("SELECT {s}.id, {t}.id FROM {s} JOIN {t} ON {t}.v = {s}.v LIMIT 10", vec![]),
        ("SELECT id FROM {s} WHERE EXISTS (SELECT 1 FROM {t} WHERE {t}.v = {s}.v) ORDER BY id", vec![]),
        ("SELECT id FROM {s} WHERE NOT EXISTS (SELECT 1 FROM {t} WHERE {t}.v = {s}.v) ORDER BY id", vec![]),
        ("SELECT id FROM {s} WHERE v IN (SELECT v FROM {t}) ORDER BY id", vec![]),
    ] {
        let mut want = rows(db, &sql.replace("{t}", "pt").replace("{s}", "ps"), params.clone());
        let mut got = rows(db, &sql.replace("{t}", "t").replace("{s}", "s"), params);
        want.sort_by_key(|row| format!("{row:?}"));
        got.sort_by_key(|row| format!("{row:?}"));
        assert!(!want.is_empty(), "{phase}: {sql} reads some rows");
        assert_eq!(got, want, "{phase}: {sql}");
    }
}

#[test]
fn an_hnsw_index_answers_equality_as_without_it_in_memory() {
    let db = Database::open("memory://an_hnsw_index_answers_equality_as_without_it").unwrap();
    create(&db);
    check(&db, "memory");
}

#[test]
fn an_hnsw_index_answers_equality_as_without_it_hot_and_sealed() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    create(&db);
    check(&db, "hot");
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    check(&db, "sealed");
}
