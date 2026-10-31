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

use std::sync::{Mutex, MutexGuard};

use stoolap::{Database, Value};

// The general path counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn setup(name: &str, dir: &tempfile::TempDir, sealed: bool) -> Database {
    let dsn = match sealed {
        true => format!("file://{}/db", dir.path().display()),
        false => format!("memory://{name}"),
    };
    let db = Database::open(&dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, k INTEGER)",
        "INSERT INTO t VALUES (1, 30, 1), (2, 10, 2), (3, NULL, 1), (4, 20, 3)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "INSERT INTO u VALUES (1, 1, 5), (2, 2, 7), (3, 1, NULL), (4, 4, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn rows(db: &Database, sql: &str) -> Result<Vec<Vec<Value>>, String> {
    db.query(sql, ())
        .and_then(|rows| {
            rows.map(|row| row.map(|row| (0..row.len()).map(|i| row.get(i).unwrap()).collect()))
                .collect()
        })
        .map_err(|e| e.to_string())
}

/// Each statement with its twin, the block's star written as its columns
const SHAPES: &[(&str, &str)] = &[
    (
        "SELECT id FROM t WHERE (t.id, t.k) IN \
         (SELECT * FROM (SELECT id FROM u) a1 JOIN (SELECT k FROM u) b1 ON a1.id = b1.k) ORDER BY id",
        "SELECT id FROM t WHERE (t.id, t.k) IN \
         (SELECT a1.id, b1.k FROM (SELECT id FROM u) a1 JOIN (SELECT k FROM u) b1 ON a1.id = b1.k) \
         ORDER BY id",
    ),
    (
        "SELECT id FROM t WHERE (t.id, t.k, 5, t.id, t.a) IN \
         (SELECT * FROM u JOIN (SELECT id AS x, a FROM t) t2 ON t2.x = u.id) ORDER BY id",
        "SELECT id FROM t WHERE (t.id, t.k, 5, t.id, t.a) IN \
         (SELECT u.id, u.k, u.v, t2.x, t2.a FROM u \
         JOIN (SELECT id AS x, a FROM t) t2 ON t2.x = u.id) ORDER BY id",
    ),
    (
        "SELECT id FROM t WHERE (t.k, t.a) NOT IN \
         (SELECT * FROM (SELECT k FROM u) a1 JOIN (SELECT v FROM u WHERE v IS NOT NULL) b1 ON TRUE) \
         ORDER BY id",
        "SELECT id FROM t WHERE (t.k, t.a) NOT IN \
         (SELECT a1.k, b1.v FROM (SELECT k FROM u) a1 \
         JOIN (SELECT v FROM u WHERE v IS NOT NULL) b1 ON TRUE) ORDER BY id",
    ),
    (
        "SELECT id FROM t WHERE EXISTS \
         (SELECT * FROM u JOIN t t2 ON t2.id = u.id WHERE u.k = t.k AND t2.a > 15) ORDER BY id",
        "SELECT id FROM t WHERE EXISTS \
         (SELECT u.id FROM u JOIN t t2 ON t2.id = u.id WHERE u.k = t.k AND t2.a > 15) ORDER BY id",
    ),
];

#[test]
fn a_block_selecting_star_over_a_join_reads_its_sources_columns() {
    let _guard = serial();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = setup("join_star_blocks_answers", &dir, sealed);
        let mut wrong = Vec::new();
        for (sql, twin) in SHAPES {
            let want = rows(&db, twin).unwrap_or_else(|e| panic!("{twin}: {e}"));
            match rows(&db, sql) {
                Ok(got) if got == want => {}
                got => wrong.push(format!(
                    "sealed {sealed}: {sql}\n  got {got:?}\n  want {want:?}"
                )),
            }
        }
        assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
    }
}

#[test]
fn a_star_over_a_join_counts_every_column_of_its_sources() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = setup("join_star_blocks_width", &dir, false);
    for (sql, want) in [
        (
            "SELECT id, (SELECT * FROM (SELECT MAX(v) AS m FROM u WHERE u.k = t.k) a1 \
             JOIN (SELECT 1 AS z) b1 ON FALSE) FROM t",
            "a subquery selects 2 values where its place takes 1",
        ),
        (
            "SELECT id FROM t WHERE (t.id, t.k, t.id) IN \
             (SELECT * FROM u JOIN (SELECT id AS x FROM t) t2 ON t2.x = u.id)",
            "a subquery selects 4 values where its place takes 3",
        ),
    ] {
        let got = rows(&db, sql).expect_err(sql);
        assert!(got.contains(want), "{sql}: {got}");
    }
}

#[test]
fn a_star_over_a_join_that_merges_columns_stays_unprepared() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = setup("join_star_blocks_merged", &dir, false);
    let sql = "SELECT id FROM t WHERE (t.id, t.k, t.a) IN (SELECT * FROM u JOIN t t2 USING (id))";
    let plan = rows(&db, &format!("EXPLAIN {sql}")).unwrap();
    let line = plan
        .iter()
        .filter_map(|row| match row.first() {
            Some(Value::Text(text)) => Some(text.to_string()),
            _ => None,
        })
        .find(|line| line.trim_start().starts_with("Prepared:"))
        .unwrap_or_default();
    assert_eq!(
        line.trim(),
        "Prepared: no (a subquery block selecting * over a join that merges columns)"
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_block_selecting_star_over_a_join_runs_no_general_path_mechanism() {
    use stoolap::test_failpoints::general_path_total;
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = setup("join_star_blocks_general_path", &dir, false);
    for (sql, _) in SHAPES {
        let before = general_path_total();
        rows(&db, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
        assert_eq!(general_path_total(), before, "{sql}");
    }
}
