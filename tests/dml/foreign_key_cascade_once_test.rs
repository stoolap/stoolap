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

use stoolap::Database;

fn rows(db: &Database, sql: &str) -> Vec<Vec<Option<i64>>> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| row.get::<Option<i64>>(i).unwrap())
                .collect()
        })
        .collect()
}

/// Each layout the rows are read from: the hot store, then volumes
fn each_layout(name: &str, check: impl Fn(&Database)) {
    let tables = |dsn: &str, sealed: bool| {
        let db = Database::open(dsn).unwrap();
        for sql in [
            "CREATE TABLE p (id INTEGER PRIMARY KEY, u INTEGER UNIQUE)",
            "CREATE TABLE c (id INTEGER PRIMARY KEY, pu INTEGER UNIQUE REFERENCES p(u) ON UPDATE CASCADE)",
            "CREATE TABLE g (id INTEGER PRIMARY KEY, cu INTEGER REFERENCES c(pu) ON UPDATE CASCADE)",
            "CREATE TABLE n (id INTEGER PRIMARY KEY, pu INTEGER REFERENCES p(u) ON UPDATE SET NULL)",
            "INSERT INTO p VALUES (1, 10), (2, 20), (3, 30)",
            "INSERT INTO c VALUES (1, 10), (2, 20), (3, 30)",
            "INSERT INTO g VALUES (1, 10), (2, 20), (3, 30)",
            "INSERT INTO n VALUES (1, 10), (2, 20)",
        ] {
            db.execute(sql, ()).unwrap();
        }
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
            // A sealed row's UNIQUE value is checked against the values the
            // statement frees only once hot: the parents are moved out and back
            db.execute("UPDATE p SET u = u + 100", ()).unwrap();
            db.execute("UPDATE p SET u = u - 100", ()).unwrap();
        }
        db
    };
    check(&tables(&format!("memory://fk_cascade_once_{name}"), false));
    let dir = tempfile::tempdir().unwrap();
    check(&tables(&format!("file://{}", dir.path().display()), true));
}

fn column(db: &Database, sql: &str) -> Vec<Option<i64>> {
    rows(db, sql).into_iter().map(|row| row[0]).collect()
}

#[test]
fn a_child_follows_the_value_its_parent_had_when_the_statement_began() {
    each_layout("shift_up", |db| {
        db.execute("UPDATE p SET u = u + 10", ()).unwrap();
        let shifted = vec![Some(20), Some(30), Some(40)];
        assert_eq!(column(db, "SELECT u FROM p ORDER BY id"), shifted);
        assert_eq!(column(db, "SELECT pu FROM c ORDER BY id"), shifted);
        assert_eq!(column(db, "SELECT cu FROM g ORDER BY id"), shifted);
        assert_eq!(column(db, "SELECT pu FROM n ORDER BY id"), [None, None]);
    });
}

#[test]
fn a_shift_down_cascades_each_child_once_too() {
    each_layout("shift_down", |db| {
        db.execute("UPDATE p SET u = u - 10", ()).unwrap();
        let shifted = vec![Some(0), Some(10), Some(20)];
        assert_eq!(column(db, "SELECT pu FROM c ORDER BY id"), shifted);
        assert_eq!(column(db, "SELECT cu FROM g ORDER BY id"), shifted);
    });
}
