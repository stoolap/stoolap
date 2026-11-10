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
            "CREATE TABLE c (id INTEGER PRIMARY KEY, pu INTEGER REFERENCES p(u) ON UPDATE CASCADE, pid INTEGER REFERENCES p(id))",
            "CREATE TABLE n (id INTEGER PRIMARY KEY, pu INTEGER REFERENCES p(u) ON UPDATE SET NULL)",
            "CREATE TABLE r (id INTEGER PRIMARY KEY, pu INTEGER REFERENCES p(u))",
            "INSERT INTO p VALUES (1, 10), (2, 20), (3, 30)",
            "INSERT INTO c VALUES (1, 10, 1), (2, 20, 2)",
            "INSERT INTO n VALUES (1, 10)",
            "INSERT INTO r VALUES (1, 30)",
        ] {
            db.execute(sql, ()).unwrap();
        }
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        db
    };
    check(&tables(&format!("memory://upsert_fk_{name}"), false));
    let dir = tempfile::tempdir().unwrap();
    check(&tables(&format!("file://{}", dir.path().display()), true));
}

fn refused(db: &Database, sql: &str) {
    let error = db.execute(sql, ()).unwrap_err().to_string();
    assert!(
        error.contains("foreign key"),
        "{sql} is refused by a foreign key: {error}"
    );
}

#[test]
fn an_upsert_cannot_point_a_child_at_a_missing_parent() {
    each_layout("missing_parent", |db| {
        let upsert = "INSERT INTO c VALUES (1, 10, 1) ON CONFLICT (id) DO UPDATE SET";
        refused(db, &format!("{upsert} pid = 99"));
        refused(db, &format!("{upsert} pu = 77"));
        refused(db, &format!("{upsert} pid = (SELECT 99)"));
        let error = db
            .execute(
                "INSERT INTO c VALUES ($1, 10, 1) ON CONFLICT (id) DO UPDATE SET pid = $2",
                (1, 99),
            )
            .unwrap_err();
        assert!(error.to_string().contains("foreign key"), "{error}");
        db.execute(&format!("{upsert} pid = 3"), ()).unwrap();
        assert_eq!(
            rows(db, "SELECT * FROM c ORDER BY id"),
            [[Some(1), Some(10), Some(3)], [Some(2), Some(20), Some(2)]]
        );
    });
}

#[test]
fn an_upsert_that_changes_a_referenced_value_runs_its_update_actions() {
    each_layout("actions", |db| {
        db.execute(
            "INSERT INTO p VALUES (1, 0), (2, 0) ON CONFLICT (id) DO UPDATE SET u = u + 5",
            (),
        )
        .unwrap();
        assert_eq!(
            rows(db, "SELECT u FROM p ORDER BY id"),
            [[Some(15)], [Some(25)], [Some(30)]]
        );
        assert_eq!(
            rows(db, "SELECT pu FROM c ORDER BY id"),
            [[Some(15)], [Some(25)]]
        );
        assert_eq!(rows(db, "SELECT pu FROM n"), [[None]]);
    });
}

#[test]
fn an_upsert_restricted_by_a_child_changes_nothing() {
    each_layout("restrict", |db| {
        let upsert = "INSERT INTO p VALUES (3, 0) ON CONFLICT (id) DO UPDATE SET u = 35";
        refused(db, upsert);
        db.execute("BEGIN", ()).unwrap();
        db.execute("INSERT INTO p VALUES (4, 40)", ()).unwrap();
        refused(db, upsert);
        db.execute("COMMIT", ()).unwrap();
        assert_eq!(
            rows(db, "SELECT u FROM p ORDER BY id"),
            [[Some(10)], [Some(20)], [Some(30)], [Some(40)]]
        );
        assert_eq!(rows(db, "SELECT pu FROM r"), [[Some(30)]]);
    });
}
