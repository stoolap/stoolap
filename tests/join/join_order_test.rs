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

//! A run of INNER joins over base tables joins from the table estimated to
//! keep the fewest rows, through the tables its conditions reach, when that
//! is estimated at most half the cost of the written order; the answer is
//! the written statement's

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::{join_order_keys, join_orders};
use stoolap::Database;

/// `t` 5,000 rows; `u` 5,000, its `tid` the id of a `t` row and `sid` the id
/// of an `s` row, both indexed; `s` 100 rows, `x = 1` on two
fn tables(db: &Database) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)",
        "INSERT INTO t SELECT g.value, g.value % 7 FROM generate_series(1, 5000) g",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, tid INTEGER, sid INTEGER)",
        "INSERT INTO u SELECT g.value, g.value, g.value % 100 FROM generate_series(1, 5000) g",
        "CREATE INDEX u_tid ON u (tid)",
        "CREATE INDEX u_sid ON u (sid)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO s SELECT g.value - 1, CASE WHEN (g.value - 1) % 50 = 0 THEN 1 ELSE 0 END \
         FROM generate_series(1, 100) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

fn plan(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .collect::<Vec<_>>()
        .join("\n")
}

fn rows(db: &Database, sql: &str) -> Vec<Vec<i64>> {
    let result = db.query(sql, ()).unwrap();
    let width = result.columns().len();
    let mut out = Vec::new();
    for row in result {
        let row = row.unwrap();
        out.push((0..width).map(|at| row.get::<i64>(at).unwrap()).collect());
    }
    out
}

/// Whether `sql` ran in a reordered join, and whether its order was decided
fn ran(db: &Database, sql: &str) -> (Vec<Vec<i64>>, bool, bool) {
    let (decided, reordered) = join_orders();
    let answer = rows(db, sql);
    let now = join_orders();
    (answer, now.1 > reordered, now.0 > decided)
}

const TRAP: &str = "FROM t JOIN u ON t.id = u.tid JOIN s ON u.sid = s.id WHERE s.x = 1";

#[test]
fn a_selective_table_written_last_is_joined_first() {
    let db = Database::open("memory://join_order_trap").unwrap();
    tables(&db);
    let sql = format!("SELECT COUNT(*) {TRAP}");
    let explained = plan(&db, &sql);
    assert!(
        explained.contains("Join order: s, u, t (written t, u, s)"),
        "{explained}"
    );
    let (answer, reordered, decided) = ran(&db, &sql);
    assert_eq!(answer, vec![vec![100]]);
    assert!(
        reordered && decided,
        "the first run decides the order and takes it"
    );
    let (answer, reordered, decided) = ran(&db, &sql);
    assert_eq!(answer, vec![vec![100]]);
    assert!(
        reordered && !decided,
        "a later run takes the order its plan holds"
    );
}

#[test]
fn a_reordered_join_answers_as_written() {
    let db = Database::open("memory://join_order_answers").unwrap();
    tables(&db);
    db.execute("CREATE TABLE w (id INTEGER PRIMARY KEY, tag INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO w SELECT g.value, g.value % 3 FROM generate_series(1, 5000) g",
        (),
    )
    .unwrap();
    let cases: Vec<(String, Vec<Vec<i64>>)> = vec![
        (
            format!("SELECT t.id, s.id {TRAP} ORDER BY t.id LIMIT 3"),
            vec![vec![50, 50], vec![100, 0], vec![150, 50]],
        ),
        (
            format!("SELECT s.id, COUNT(*) {TRAP} GROUP BY s.id ORDER BY s.id"),
            vec![vec![0, 50], vec![50, 50]],
        ),
        (format!("SELECT SUM(t.v) {TRAP}"), vec![vec![297]]),
        (
            format!("SELECT DISTINCT s.id {TRAP} ORDER BY s.id"),
            vec![vec![0], vec![50]],
        ),
        (
            "SELECT COUNT(*) FROM t JOIN u ON t.id = u.tid JOIN w ON w.id = t.id \
             JOIN s ON u.sid = s.id WHERE s.x = 1"
                .to_string(),
            vec![vec![100]],
        ),
        (
            "SELECT COUNT(*), COUNT(w.id) FROM t JOIN u ON t.id = u.tid \
             JOIN s ON u.sid = s.id LEFT JOIN w ON w.id = t.id AND w.tag = 9 WHERE s.x = 1"
                .to_string(),
            vec![vec![100, 0]],
        ),
    ];
    for (sql, want) in cases {
        let explained = plan(&db, &sql);
        assert!(
            explained.contains("Join order: s, u,"),
            "{sql}\n{explained}"
        );
        let (answer, reordered, _) = ran(&db, &sql);
        assert_eq!(answer, want, "{sql}");
        assert!(reordered, "{sql} ran reordered");
    }
}

#[test]
fn a_prepared_join_with_a_parameter_runs_reordered() {
    let db = Database::open("memory://join_order_prepared").unwrap();
    tables(&db);
    let stmt = db
        .prepare(
            "SELECT COUNT(*) FROM t JOIN u ON t.id = u.tid JOIN s ON u.sid = s.id \
             WHERE s.x = $1",
        )
        .unwrap();
    for (x, want) in [(1, 100), (0, 4900)] {
        let (_, before) = join_orders();
        let got: i64 = stmt
            .query((x,))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(got, want, "x = {x}");
        assert!(join_orders().1 > before, "x = {x} ran reordered");
    }
}

#[test]
fn the_written_order_is_kept_where_the_rule_does_not_reach() {
    let db = Database::open("memory://join_order_barriers").unwrap();
    tables(&db);
    for sql in [
        "CREATE TABLE s2 (sid INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO s2 SELECT id, x FROM s",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let cases: Vec<(String, i64)> = vec![
        // The star's columns follow the tables' order
        (format!("SELECT * {TRAP}"), 100),
        (
            "SELECT COUNT(*) FROM t JOIN u ON t.id = u.tid JOIN s2 USING (sid) WHERE s2.x = 1"
                .to_string(),
            100,
        ),
        (
            "SELECT COUNT(*) FROM t, u, s WHERE t.id = u.tid AND u.sid = s.id AND s.x = 1"
                .to_string(),
            100,
        ),
        (
            "SELECT COUNT(*) FROM t JOIN u ON t.id = u.tid \
             JOIN (SELECT id, x FROM s) s ON u.sid = s.id WHERE s.x = 1"
                .to_string(),
            100,
        ),
        (
            "SELECT COUNT(*) FROM t JOIN u ON t.id = u.tid \
             JOIN s ON u.sid >= s.id AND u.sid <= s.id WHERE s.x = 1"
                .to_string(),
            100,
        ),
        // A subquery the LIMIT holds
        (format!("SELECT COUNT(*) {TRAP} LIMIT (SELECT 1)"), 100),
        // Two tables, and a written order already the cheapest
        (
            "SELECT COUNT(*) FROM u JOIN s ON u.sid = s.id WHERE s.x = 1".to_string(),
            100,
        ),
        (
            "SELECT COUNT(*) FROM s JOIN u ON u.sid = s.id JOIN t ON t.id = u.tid \
             WHERE s.x = 1"
                .to_string(),
            100,
        ),
    ];
    for (sql, want) in cases {
        let explained = plan(&db, &sql);
        assert!(!explained.contains("Join order"), "{sql}\n{explained}");
        let (answer, reordered, _) = ran(&db, &sql);
        if sql.starts_with("SELECT *") {
            assert_eq!(answer.len() as i64, want, "{sql}");
        } else {
            assert_eq!(answer, vec![vec![want]], "{sql}");
        }
        assert!(!reordered, "{sql} ran as written");
    }
}

#[test]
fn the_order_is_decided_again_when_its_facts_move() {
    let db = Database::open("memory://join_order_memo").unwrap();
    tables(&db);
    let sql = format!("SELECT COUNT(*) {TRAP}");
    assert!(ran(&db, &sql).2, "the first run decides");
    assert!(!ran(&db, &sql).2, "the second takes the order held");
    db.execute("ANALYZE s", ()).unwrap();
    assert!(ran(&db, &sql).2, "ANALYZE moves the statistics epoch");
    db.execute("CREATE INDEX t_v ON t (v)", ()).unwrap();
    assert!(ran(&db, &sql).2, "an index created moves the schema epoch");
    assert!(!ran(&db, &sql).2);
    db.execute(
        "INSERT INTO s SELECT g.value + 100, 0 FROM generate_series(1, 1000) g",
        (),
    )
    .unwrap();
    let (answer, _, decided) = ran(&db, &sql);
    assert_eq!(answer, vec![vec![100]]);
    assert!(decided, "s grew from 100 rows to 1,100, past two buckets");
}

#[test]
fn a_moved_on_term_reads_the_column_its_written_join_saw() {
    let db = Database::open("memory://join_order_on_scope").unwrap();
    tables(&db);
    db.execute("ALTER TABLE s ADD COLUMN v INTEGER DEFAULT 9", ())
        .unwrap();
    // As written, the first ON sees t and u alone, so its v is t.v
    let sql = "SELECT COUNT(*) FROM t JOIN u ON t.id = u.tid AND v = 1 \
               JOIN s ON u.sid = s.id WHERE s.x = 1";
    let explained = plan(&db, sql);
    assert!(explained.contains("Join order: s, u, t"), "{explained}");
    let (answer, reordered, _) = ran(&db, sql);
    assert_eq!(answer, vec![vec![15]], "the rows of t.v = 1, not s.v");
    assert!(reordered, "{sql} ran reordered");
}

#[test]
fn an_on_term_naming_a_column_two_written_tables_hold_stays_ambiguous() {
    let db = Database::open("memory://join_order_on_ambiguous").unwrap();
    tables(&db);
    // The first ON sees t and u, which both hold id
    let sql = "SELECT COUNT(*) FROM t JOIN u ON t.id = u.tid AND id = 50 \
               JOIN s ON u.sid = s.id WHERE s.x = 1";
    let error = db.query(sql, ()).err().map(|e| e.to_string());
    assert!(
        error.as_deref().is_some_and(|e| e.contains("ambiguous")),
        "{error:?}"
    );
}

#[test]
fn a_held_order_of_nine_tables_is_checked_without_collecting_a_key() {
    let db = Database::open("memory://join_order_nine_tables").unwrap();
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY)",
        "INSERT INTO p VALUES (1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut sql = String::from("SELECT COUNT(*) FROM p a1");
    for at in 2..=9 {
        sql.push_str(&format!(" JOIN p a{at} ON a{at}.id = a{}.id", at - 1));
    }
    let stmt = db.prepare(&sql).unwrap();
    let keys = || {
        let before = join_order_keys();
        assert_eq!(stmt.query(()).unwrap().count(), 1, "{sql}");
        join_order_keys() - before
    };
    assert_eq!(keys(), 1, "the first run decides, under a key it collects");
    assert_eq!(keys(), 0, "a later run reads the held key as it stands");
    assert_eq!(keys(), 0);
}

#[test]
fn explain_analyze_keeps_the_subqueries_of_a_limit() {
    let db = Database::open("memory://join_order_limit_subquery").unwrap();
    tables(&db);
    let explained = plan(
        &db,
        &format!("ANALYZE SELECT COUNT(*) {TRAP} LIMIT (SELECT 1)"),
    );
    assert!(explained.contains("Subqueries:"), "{explained}");
    assert!(!explained.contains("Join order"), "{explained}");
}

#[test]
fn explain_analyze_prints_the_order_its_run_took() {
    let db = Database::open("memory://join_order_explain_analyze").unwrap();
    tables(&db);
    for sql in ["CREATE INDEX s_x ON s (x)", "CREATE INDEX t_v ON t (v)"] {
        db.execute(sql, ()).unwrap();
    }
    let sql = "SELECT COUNT(*) FROM t JOIN u ON t.id = u.tid JOIN s ON u.sid = s.id \
               WHERE s.x = 1 AND t.v = 3";
    let analyzed = || plan(&db, &format!("ANALYZE {sql}"));
    let first = analyzed();
    assert!(first.contains("Join order: s, u, t"), "{first}");
    // Every s row now has x = 1 and one t row v = 3: the counts would start
    // from t, as written, while the epochs and the row buckets stay
    db.execute("UPDATE s SET x = 1", ()).unwrap();
    db.execute("UPDATE t SET v = 0 WHERE v = 3 AND id > 10", ())
        .unwrap();
    let fresh = plan(&db, sql);
    assert!(!fresh.contains("Join order"), "{fresh}");
    let second = analyzed();
    assert!(
        second.contains("Join order: s, u, t (written t, u, s)"),
        "the run took the order its plan held: {second}"
    );
}

#[test]
fn a_sealed_join_runs_reordered() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    tables(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let sql = format!("SELECT COUNT(*) {TRAP}");
    let (answer, reordered, _) = ran(&db, &sql);
    assert_eq!(answer, vec![vec![100]]);
    assert!(reordered, "{}", plan(&db, &sql));
}
