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

//! A qualified column or `q.*` names a FROM entry of the query it is written
//! in or of an enclosing one; a qualifier that names none fails the statement

use stoolap::Database;

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO u VALUES (1, 7), (2, 8), (3, 9)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, b INTEGER)",
        "INSERT INTO s VALUES (5, 9)",
        "CREATE INDEX ux ON u(x)",
        "CREATE VIEW v AS SELECT id, x FROM u",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

fn state(db: &Database) -> String {
    db.query_one("SELECT COUNT(*) || '/' || SUM(x) FROM u", ())
        .unwrap()
}

const UNKNOWN: &[&str] = &[
    "SELECT zz.x FROM u",
    "SELECT x FROM u WHERE zz.id = 1",
    "SELECT x FROM u WHERE s.id = 1",
    "SELECT x FROM u ORDER BY zz.id DESC",
    "SELECT COUNT(*) FROM u GROUP BY zz.x",
    "SELECT x, COUNT(*) FROM u GROUP BY x HAVING zz.x > 7",
    "SELECT x, ROW_NUMBER() OVER (ORDER BY zz.id) FROM u",
    "SELECT x, ROW_NUMBER() OVER w FROM u WINDOW w AS (ORDER BY zz.id)",
    "SELECT SUM(zz.x) FROM u",
    "SELECT DISTINCT zz.x FROM u",
    "SELECT zz.* FROM u",
    "SELECT a.x FROM u AS a WHERE zz.id = 1",
    "SELECT u.x FROM u JOIN s ON zz.id < s.id",
    "SELECT (SELECT zz.x FROM u WHERE u.id = 1)",
    "SELECT EXISTS (SELECT 1 FROM u WHERE zz.id = 1)",
    "SELECT id FROM s WHERE EXISTS (SELECT 1 FROM u WHERE u.id = zz.id)",
    "WITH c AS (SELECT id, x FROM u) SELECT zz.x FROM c",
    "WITH c AS (SELECT id, x FROM u) SELECT x FROM c WHERE u.id = 1",
    "SELECT zz.x FROM v",
    "SELECT * FROM v WHERE u.id = 1",
    "SELECT zz.x FROM (SELECT id, x FROM u) AS d",
    "SELECT x AS q FROM u ORDER BY q.x",
    "SELECT x FROM u a UNION ALL SELECT x FROM u b ORDER BY b.x",
    "SELECT x FROM u a UNION ALL SELECT x FROM u b ORDER BY a.x",
    "EXPLAIN SELECT zz.x FROM u",
    "UPDATE u SET x = 0 WHERE zz.id = 2",
    "UPDATE u SET x = zz.id WHERE id = 1",
    "UPDATE u SET x = 0 WHERE zz.x > 7",
    "DELETE FROM u WHERE zz.id = 3",
    "DELETE FROM u WHERE zz.x = 9",
    "DELETE FROM u WHERE id IN (SELECT zz.id FROM s)",
    "INSERT INTO u SELECT zz.id + 10, x FROM u",
    "INSERT INTO u VALUES (1, 1) ON CONFLICT (id) DO UPDATE SET x = zz.x",
    "UPDATE u SET x = 0 WHERE id = 1 RETURNING zz.x",
    "INSERT INTO u VALUES (4, 4) RETURNING zz.id",
    "DELETE FROM u WHERE id = 3 RETURNING zz.x",
    "CREATE TABLE c1 (id INTEGER PRIMARY KEY, y INTEGER CHECK (zz.y > 0))",
    "CREATE VIEW w AS SELECT zz.x FROM u",
];

const NOT_VISIBLE: &[&str] = &[
    "SELECT x FROM u AS a WHERE u.id = 1",
    "SELECT u.* FROM u AS a",
    "SELECT a.x FROM u a GROUP BY a.x ORDER BY u.x",
    "SELECT x FROM u a JOIN s b ON a.x = b.b ORDER BY s.id",
    "SELECT id FROM s AS t WHERE EXISTS (SELECT 1 FROM u WHERE u.x = s.b)",
    "WITH c AS (SELECT id, x FROM u) SELECT c.x FROM c AS k",
    "SELECT generate_series.x FROM generate_series(1, 3) AS g(x)",
    "SELECT u.id, d.b FROM u, (SELECT s.b FROM s WHERE s.id = u.id) d",
    "DELETE FROM u AS a WHERE u.id = 3",
    "DELETE FROM u AS a WHERE a.id = 3 RETURNING u.x",
    "INSERT INTO u VALUES (4, u.x)",
    "INSERT INTO u SELECT s.id, s.b FROM s WHERE s.b > u.x",
    "INSERT INTO u VALUES (1, 5) ON CONFLICT (id) DO UPDATE SET x = excluded.x RETURNING excluded.x",
];

const ACCEPTED: &[&str] = &[
    "SELECT u.x FROM u WHERE u.id = 1",
    "SELECT a.x FROM u AS a WHERE a.id = 1",
    "SELECT u.* FROM u",
    "SELECT id FROM s WHERE EXISTS (SELECT 1 FROM u WHERE u.x = s.b)",
    "SELECT x FROM u WHERE u.id IN (SELECT u.id FROM s)",
    "SELECT * FROM s WHERE b IN (SELECT x FROM (SELECT x FROM u WHERE u.x = s.b) d)",
    "SELECT c.x FROM (SELECT id, x FROM u) AS c WHERE c.id = 1",
    "WITH c AS (SELECT id, x FROM u) SELECT c.x FROM c WHERE c.id = 1",
    "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT r.n + 1 FROM r WHERE r.n < 3) SELECT r.n FROM r",
    "SELECT v.x FROM v WHERE v.id = 1",
    "SELECT u.x, s.b FROM u JOIN s ON u.x = s.b",
    "SELECT u.x FROM u JOIN s ON s.id = u.id JOIN v ON v.id = s.id",
    "SELECT x FROM u WHERE x > 0 GROUP BY x HAVING MAX(u.id) > 0 ORDER BY u.x",
    "SELECT x, SUM(x) OVER (PARTITION BY u.id ORDER BY u.x) FROM u",
    "SELECT x FROM u ORDER BY (SELECT s.b FROM s WHERE s.id = u.id)",
    "SELECT (SELECT u.x FROM s LIMIT 1) FROM u",
    "SELECT g.x FROM generate_series(1, 3) AS g(x)",
    "SELECT d.x FROM (VALUES (1), (2)) AS d(x) WHERE d.x > 1",
    "SELECT x FROM u AS U1 WHERE u1.id = 1",
    "SELECT x FROM u AS \"A b\" WHERE \"A b\".id = 1",
    "UPDATE u SET x = u.x + 1 WHERE u.id = 1 RETURNING u.x",
    "UPDATE u SET x = (SELECT s.b FROM s WHERE s.id = u.id) WHERE u.id = 1",
    "DELETE FROM u WHERE u.id = 3",
    "DELETE FROM u AS a WHERE a.id = 3 RETURNING a.x",
    "INSERT INTO u VALUES (4, 4) RETURNING u.x",
    "INSERT INTO u VALUES (1, 5) ON CONFLICT (id) DO UPDATE SET x = excluded.x WHERE excluded.x > 0",
    "INSERT INTO u VALUES (1, 5) ON CONFLICT (id) DO UPDATE SET x = u.x + 1 WHERE u.x > 0",
    "INSERT INTO u VALUES (1, 5) ON CONFLICT (id) DO UPDATE SET x = (SELECT MAX(b) FROM s WHERE s.b > excluded.x)",
    "CREATE TABLE c2 (id INTEGER PRIMARY KEY, y INTEGER CHECK (c2.y > 0))",
];

#[test]
fn a_qualifier_that_names_no_from_entry_fails_the_statement() {
    let mut wrong = Vec::new();
    for (shapes, message) in [
        (UNKNOWN, "unknown table or alias"),
        (NOT_VISIBLE, "is not visible here"),
    ] {
        for sql in shapes {
            let db = setup("qualifier_scope_refused");
            let before = state(&db);
            match db.execute(sql, ()) {
                Ok(_) => wrong.push(format!("{sql}: accepted")),
                Err(e) if !e.to_string().contains(message) => {
                    wrong.push(format!("{sql}: {e}"));
                }
                Err(_) => {}
            }
            if state(&db) != before {
                wrong.push(format!("{sql}: changed a row"));
            }
            if db.prepare(sql).is_ok() {
                wrong.push(format!("{sql}: prepared"));
            }
            let mut tx = db.begin().unwrap();
            if tx.execute(sql, ()).is_ok() {
                wrong.push(format!("{sql}: accepted in a transaction"));
            }
            tx.rollback().unwrap();
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_qualifier_that_names_a_visible_entry_is_read() {
    let mut wrong = Vec::new();
    for sql in ACCEPTED {
        let db = setup("qualifier_scope_accepted");
        if let Err(e) = db.execute(sql, ()) {
            wrong.push(format!("{sql}: {e}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_qualifier_reads_the_entry_it_names() {
    let db = setup("qualifier_scope_reads");
    let read = |sql: &str| -> i64 { db.query_one(sql, ()).unwrap() };
    assert_eq!(read("SELECT a.x FROM u AS a WHERE a.id = 2"), 8);
    assert_eq!(
        read("SELECT COUNT(*) FROM s WHERE EXISTS (SELECT 1 FROM u WHERE u.x = s.b)"),
        1
    );
    assert_eq!(
        read("SELECT x FROM u WHERE u.id IN (SELECT u.id FROM s)"),
        7
    );
    db.execute("DELETE FROM u AS a WHERE a.id = 3", ()).unwrap();
    assert_eq!(state(&db), "2/15");
}
