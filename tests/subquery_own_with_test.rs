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

//! A subquery or a derived table may carry its own WITH: its CTEs are read
//! by it and the statements inside it alone, ahead of any other name, and
//! may read the enclosing query.

use stoolap::Database;

/// The same rows hot in memory and sealed into a volume
fn tables(name: &str) -> [(Database, tempfile::TempDir); 2] {
    let open = |dsn: &str, dir: tempfile::TempDir, seal: bool| {
        let db = Database::open(dsn).unwrap();
        for sql in [
            "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
            "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER, k INTEGER)",
            "INSERT INTO t VALUES (1, 1, 10), (2, 2, 20), (3, NULL, 30), (4, 5, 40)",
            "INSERT INTO u VALUES (1, 1, 1), (2, 2, 1), (3, 3, 2), (4, NULL, 2)",
        ] {
            db.execute(sql, ()).unwrap();
        }
        if seal {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        (db, dir)
    };
    let dir = tempfile::tempdir().unwrap();
    let sealed = format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    );
    [
        open(
            &format!("memory://{name}"),
            tempfile::tempdir().unwrap(),
            false,
        ),
        open(&sealed, dir, true),
    ]
}

fn column(db: &Database, sql: &str, at: usize) -> Vec<Option<i64>> {
    db.query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| row.unwrap().get::<Option<i64>>(at).unwrap())
        .collect()
}

fn error(db: &Database, sql: &str) -> String {
    match db.query(sql, ()) {
        Err(e) => e.to_string(),
        Ok(rows) => rows
            .map(|row| row.map(|_| ()))
            .collect::<Result<Vec<()>, _>>()
            .expect_err(sql)
            .to_string(),
    }
}

const BY_A: [Option<i64>; 4] = [Some(1), Some(2), None, Some(3)];

#[test]
fn a_subquery_reads_its_own_ctes() {
    for (db, _dir) in tables("own_with_blocks") {
        for (sql, want) in [
            (
                "SELECT id, (WITH c AS (SELECT x FROM u WHERE x > 1) SELECT MIN(x) FROM c) \
                 FROM t ORDER BY id",
                vec![Some(2); 4],
            ),
            (
                "SELECT id, (WITH c AS (SELECT x FROM u) SELECT MAX(x) FROM c WHERE x <= t.a) \
                 FROM t ORDER BY id",
                BY_A.to_vec(),
            ),
            (
                "SELECT id, (WITH c AS (SELECT x FROM u WHERE x <= t.a) SELECT MAX(x) FROM c) \
                 FROM t ORDER BY id",
                BY_A.to_vec(),
            ),
            (
                "SELECT id, (WITH c(z) AS (SELECT x FROM u) SELECT MAX(z) FROM c WHERE z <= t.a) \
                 FROM t ORDER BY id",
                BY_A.to_vec(),
            ),
            (
                "SELECT id, (WITH c AS (SELECT x FROM u), d AS (SELECT x FROM c WHERE x > 1) \
                 SELECT COUNT(*) FROM d) FROM t ORDER BY id",
                vec![Some(2); 4],
            ),
            (
                "SELECT id, (WITH t AS (SELECT x AS a FROM u WHERE x = 3) SELECT a FROM t) \
                 FROM t ORDER BY id",
                vec![Some(3); 4],
            ),
            (
                "SELECT id, (WITH u AS (SELECT 9 AS x) SELECT COUNT(*) FROM u \
                 WHERE x IN (SELECT x FROM u)) FROM t ORDER BY id",
                vec![Some(1); 4],
            ),
            (
                "SELECT id, (WITH RECURSIVE r(n) AS (SELECT t.a UNION ALL \
                 SELECT n + 1 FROM r WHERE n < 3) SELECT COUNT(*) FROM r) FROM t ORDER BY id",
                vec![Some(3), Some(2), Some(1), Some(1)],
            ),
            (
                "SELECT id, (SELECT MAX(x) FROM (WITH c AS (SELECT x FROM u WHERE x <= t.a) \
                 SELECT x FROM c) s) FROM t ORDER BY id",
                BY_A.to_vec(),
            ),
            (
                "WITH c AS (SELECT x FROM u) SELECT id, \
                 (WITH d AS (SELECT x FROM c WHERE x > 1) SELECT MIN(x) FROM d) FROM t ORDER BY id",
                vec![Some(2); 4],
            ),
            (
                "WITH c AS (SELECT x FROM u) SELECT id, \
                 (WITH c AS (SELECT a AS x FROM t) SELECT MAX(x) FROM c) FROM t ORDER BY id",
                vec![Some(5); 4],
            ),
        ] {
            assert_eq!(column(&db, sql, 1), want, "{sql}");
        }
        for (sql, want) in [
            (
                "SELECT id FROM t WHERE a IN (WITH c AS (SELECT x FROM u) SELECT x FROM c) \
                 ORDER BY id",
                vec![Some(1), Some(2)],
            ),
            (
                "SELECT id FROM t WHERE EXISTS (WITH c AS (SELECT x FROM u) \
                 SELECT 1 FROM c WHERE c.x = t.a) ORDER BY id",
                vec![Some(1), Some(2)],
            ),
            (
                "SELECT id FROM t WHERE a = ANY (WITH c AS (SELECT x FROM u) SELECT x + 1 FROM c) \
                 ORDER BY id",
                vec![Some(2)],
            ),
            (
                "SELECT s.x FROM (WITH c AS (SELECT x FROM u WHERE x > 1) SELECT x FROM c) s \
                 ORDER BY 1",
                vec![Some(2), Some(3)],
            ),
        ] {
            assert_eq!(column(&db, sql, 0), want, "{sql}");
        }
        // Outside its statement the name reads nothing of it
        for (sql, want) in [
            (
                "SELECT (WITH c AS (SELECT 1 AS x) SELECT x FROM c), (SELECT COUNT(*) FROM c)",
                "table or view 'c' not found",
            ),
            (
                "SELECT id, (WITH c(z) AS (SELECT x FROM u) SELECT MAX(x) FROM c) FROM t",
                "column 'x' not found",
            ),
        ] {
            assert!(error(&db, sql).contains(want), "{sql}: {}", error(&db, sql));
        }
    }
}

#[test]
fn a_with_inside_hides_and_reads_the_ctes_around_it_on_every_path() {
    for (db, _dir) in tables("own_with_scopes") {
        // An unused named window leaves the statement to the general path
        for window in ["", " WINDOW unused AS ()"] {
            for (sql, at, want) in [
                (
                    "WITH c AS (SELECT 1 AS x) \
                     SELECT s.x FROM (WITH c AS (SELECT 2 AS x) SELECT x FROM c) s",
                    0,
                    vec![Some(2)],
                ),
                (
                    "WITH c(x) AS (SELECT 1) SELECT (WITH d(y) AS (SELECT x FROM c) SELECT y FROM d)",
                    0,
                    vec![Some(1)],
                ),
                (
                    "WITH c(x) AS (SELECT 1) \
                     SELECT s.y FROM (WITH d(y) AS (SELECT x FROM c) SELECT y FROM d) s",
                    0,
                    vec![Some(1)],
                ),
                (
                    "SELECT id FROM t WHERE EXISTS (WITH u AS (SELECT 99 AS x) \
                     SELECT 1 FROM u WHERE u.x = t.a) ORDER BY id",
                    0,
                    vec![],
                ),
                (
                    "SELECT id FROM t WHERE NOT EXISTS (WITH u AS (SELECT 99 AS x) \
                     SELECT 1 FROM u WHERE u.x = t.a) ORDER BY id",
                    0,
                    vec![Some(1), Some(2), Some(3), Some(4)],
                ),
                (
                    "SELECT id, (WITH u AS (SELECT 99 AS x) SELECT COUNT(*) FROM u WHERE u.x = t.a) \
                     FROM t ORDER BY id",
                    1,
                    vec![Some(0); 4],
                ),
                (
                    "SELECT (WITH c AS (SELECT 1 AS x) SELECT c.x FROM c \
                     JOIN (SELECT 1 AS y) d ON c.x = d.y WHERE c.x IN (SELECT x FROM c))",
                    0,
                    vec![Some(1)],
                ),
                (
                    "SELECT (WITH c AS (SELECT 1 AS x) SELECT (SELECT MAX(x) FROM c) FROM c \
                     JOIN (SELECT 1 AS y) d ON c.x = d.y)",
                    0,
                    vec![Some(1)],
                ),
                (
                    "SELECT (WITH c AS (SELECT 1 AS x) SELECT c.x FROM c \
                     JOIN (SELECT 1 AS y) d ON c.x = d.y \
                     WHERE c.x IN (SELECT x FROM (SELECT x FROM c) s))",
                    0,
                    vec![Some(1)],
                ),
                (
                    "SELECT (WITH c AS (SELECT 1 AS x) SELECT c.x FROM c \
                     JOIN (SELECT 1 AS y) d ON c.x = d.y \
                     WHERE EXISTS (SELECT 1 FROM (SELECT x FROM c) s WHERE s.x = d.y))",
                    0,
                    vec![Some(1)],
                ),
                (
                    "SELECT (WITH c AS (SELECT 1 AS x) SELECT c.x FROM c \
                     JOIN (SELECT 1 AS y) d ON c.x = d.y AND c.x IN (SELECT x FROM (SELECT x FROM c) s))",
                    0,
                    vec![Some(1)],
                ),
            ] {
                let at_order = sql.rfind(" ORDER BY ").unwrap_or(sql.len());
                let sql = format!("{}{window}{}", &sql[..at_order], &sql[at_order..]);
                assert_eq!(column(&db, &sql, at), want, "{sql}");
            }
        }
    }
}

#[test]
fn a_subquery_s_own_ctes_read_parameters_and_serve_every_statement() {
    for (db, _dir) in tables("own_with_statements") {
        let prepared = db
            .prepare(
                "SELECT id, (WITH c AS (SELECT x FROM u WHERE x > $1) SELECT MIN(x) FROM c) \
                 FROM t ORDER BY id",
            )
            .unwrap();
        for (param, want) in [(0, 1), (2, 3)] {
            let got: Vec<i64> = prepared
                .query((param,))
                .unwrap()
                .map(|row| row.unwrap().get::<i64>(1).unwrap())
                .collect();
            assert_eq!(got, vec![want; 4], "x > {param}");
        }
        db.execute(
            "CREATE VIEW vs AS SELECT id, \
             (WITH c AS (SELECT x FROM u WHERE x <= t.a) SELECT MAX(x) FROM c) AS m FROM t",
            (),
        )
        .unwrap();
        assert_eq!(column(&db, "SELECT m FROM vs ORDER BY id", 0), BY_A);
        db.execute(
            "UPDATE t SET b = (WITH c AS (SELECT x FROM u WHERE x <= t.a) SELECT MAX(x) FROM c) \
             WHERE id < 4",
            (),
        )
        .unwrap();
        assert_eq!(
            column(&db, "SELECT b FROM t ORDER BY id", 0),
            [Some(1), Some(2), None, Some(40)]
        );
        db.execute(
            "DELETE FROM u WHERE x IN (WITH c AS (SELECT a FROM t WHERE a > 4) SELECT a - 2 FROM c)",
            (),
        )
        .unwrap();
        assert_eq!(
            column(&db, "SELECT id FROM u ORDER BY id", 0),
            [Some(1), Some(2), Some(4)]
        );
    }
}
