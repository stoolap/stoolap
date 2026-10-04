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

//! An INNER join without a LIMIT whose outer side is small and held in
//! memory reads the inner table by its candidates, and otherwise answers
//! as the standard join does

use std::sync::{Mutex, MutexGuard};

use stoolap::Database;

// The index page ledger is process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// `t` has k = 777 for every tenth id, else id % 1000, sealed into a
/// volume whose side file covers k, and more rows than a query result
/// keeps in memory, so its scan streams; `u` holds the same rows without
/// an index; `s` is the small outer side
fn sealed(dir: &tempfile::TempDir) -> Database {
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    for table in ["t", "u"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)"),
            (),
        )
        .unwrap();
        db.execute(
            &format!(
                "INSERT INTO {table} SELECT g.value, \
                 CASE WHEN g.value % 10 = 0 THEN 777 ELSE g.value % 1000 END, g.value \
                 FROM generate_series(1, 120000) g"
            ),
            (),
        )
        .unwrap();
    }
    db.execute("CREATE INDEX t_k ON t(k)", ()).unwrap();
    db.execute(
        "CREATE TABLE s (id INTEGER PRIMARY KEY, k INTEGER, w INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO s VALUES (1, 1, 10), (2, 2, 20), (3, 5, 30)",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db
}

fn rows(db: &Database, sql: &str) -> Vec<Vec<i64>> {
    let mut out: Vec<Vec<i64>> = db
        .query(sql, ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (0..r.len())
                .map(|i| r.get::<i64>(i).unwrap_or(-1))
                .collect()
        })
        .collect();
    out.sort_unstable();
    out
}

/// The rows of `sql` on `t` equal its rows on the unindexed copy
fn same_as_control(db: &Database, sql: &str) -> Vec<Vec<i64>> {
    let found = rows(db, sql);
    let control = rows(db, &sql.replace(" t ", " u ").replace("t.", "u."));
    assert_eq!(found, control, "{sql}");
    found
}

#[test]
fn a_join_without_an_index_answers_as_its_control() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed(&dir);
    for sql in [
        "WITH c AS (SELECT k FROM s) SELECT t.id FROM c JOIN t ON t.k = c.k",
        "WITH c AS (SELECT k, w FROM s) SELECT t.id, c.w FROM c JOIN t ON t.k = c.k AND t.v > 5000",
        "WITH c AS (SELECT k FROM s) SELECT t.id FROM c JOIN t ON t.k = c.k WHERE t.v % 2 = 0",
        "WITH c AS (SELECT k, COUNT(*) AS n FROM s GROUP BY k) SELECT t.id, c.n FROM c JOIN t ON t.k = c.k",
        "SELECT t.id FROM t JOIN (SELECT k FROM s) c ON t.k = c.k WHERE t.v > 100",
    ] {
        assert!(!same_as_control(&db, sql).is_empty(), "{sql}");
    }
}

#[test]
fn null_keys_moved_keys_and_own_rows_answer_as_the_control() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = sealed(&dir);
    db.execute("INSERT INTO s VALUES (4, NULL, 40)", ())
        .unwrap();
    let join = "WITH c AS (SELECT k FROM s) SELECT t.id FROM c JOIN t ON t.k = c.k";
    same_as_control(&db, join);
    for table in ["t", "u"] {
        db.execute(
            &format!("UPDATE {table} SET k = 999999 WHERE id = 1001"),
            (),
        )
        .unwrap();
        db.execute(&format!("UPDATE {table} SET k = 5 WHERE id = 1002"), ())
            .unwrap();
    }
    let moved = same_as_control(&db, join);
    assert!(!moved.contains(&vec![1001]) && moved.contains(&vec![1002]));
    db.execute("BEGIN", ()).unwrap();
    for table in ["t", "u"] {
        db.execute(&format!("INSERT INTO {table} VALUES (190001, 2, 0)"), ())
            .unwrap();
    }
    assert!(same_as_control(&db, join).contains(&vec![190001]));
    db.execute("ROLLBACK", ()).unwrap();
}

#[test]
fn a_timestamp_key_answers_as_its_control() {
    let _serial = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    for table in ["t", "u"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, ts TIMESTAMP)"),
            (),
        )
        .unwrap();
        db.execute(
            &format!(
                "INSERT INTO {table} SELECT g.value, CASE WHEN g.value % 100 = 0 \
                 THEN TIMESTAMP '2024-01-01 03:00:00' ELSE TIMESTAMP '2024-01-02 00:00:00' END \
                 FROM generate_series(1, 20000) g"
            ),
            (),
        )
        .unwrap();
    }
    db.execute("CREATE INDEX t_ts ON t(ts)", ()).unwrap();
    db.execute("CREATE TABLE s (id INTEGER PRIMARY KEY, ts TIMESTAMP)", ())
        .unwrap();
    db.execute(
        "INSERT INTO s VALUES (1, TIMESTAMP '2024-01-01 03:00:00')",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let joined = same_as_control(&db, "SELECT t.id FROM s JOIN t ON t.ts = s.ts");
    assert_eq!(joined.len(), 200);
}

#[cfg(feature = "test-failpoints")]
mod counted {
    use super::*;
    use stoolap::test_failpoints::{in_member_reads, join_side_runs, small_outer_joins};

    struct Counts {
        /// Joins whose small outer narrowed the inner side
        narrowed: usize,
        /// Join sides executed
        sides: usize,
        /// Inner reads by candidates and by a scan of the members
        reads: (usize, usize),
    }

    fn run(db: &Database, sql: &str) -> (Vec<Vec<i64>>, Counts) {
        let (narrowed, sides, reads) = (small_outer_joins(), join_side_runs(), in_member_reads());
        let found = rows(db, sql);
        let after = in_member_reads();
        let counts = Counts {
            narrowed: small_outer_joins() - narrowed,
            sides: join_side_runs() - sides,
            reads: (after.0 - reads.0, after.1 - reads.1),
        };
        (found, counts)
    }

    #[test]
    fn a_small_outer_reads_the_inner_side_by_candidates() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let db = sealed(&dir);
        for sql in [
            "WITH c AS (SELECT k FROM s) SELECT t.id FROM c JOIN t ON t.k = c.k",
            "SELECT t.id, s.w FROM s JOIN t ON t.k = s.k WHERE s.w > 10",
            "SELECT t.id, c.w FROM t JOIN (SELECT k, w FROM s) c \
             ON t.k = c.k AND t.v > 1000 WHERE t.v % 3 = 0",
        ] {
            let (found, counts) = run(&db, sql);
            assert_eq!(counts.narrowed, 1, "the small outer did not narrow: {sql}");
            assert_eq!(
                counts.reads,
                (1, 0),
                "the inner side was not read by candidates: {sql}"
            );
            assert_eq!(counts.sides, 2, "a side ran more than once: {sql}");
            assert_eq!(found, same_as_control(&db, sql), "{sql}");
        }
    }

    #[test]
    fn a_refused_key_scans_the_inner_side_with_the_outer_keys() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let db = sealed(&dir);
        db.execute("INSERT INTO s VALUES (9, 777, 90)", ()).unwrap();
        for sql in [
            "WITH c AS (SELECT k FROM s) SELECT t.id FROM c JOIN t ON t.k = c.k",
            "SELECT t.id, c.w FROM t JOIN (SELECT k, w FROM s) c \
             ON t.k = c.k AND t.v > 1000 WHERE t.v % 3 = 0",
        ] {
            let (found, counts) = run(&db, sql);
            assert_eq!(counts.narrowed, 1, "{sql}");
            assert_eq!(
                counts.reads,
                (0, 0),
                "the refused keys took the candidates: {sql}"
            );
            assert_eq!(counts.sides, 2, "a side ran more than once: {sql}");
            assert_eq!(found, same_as_control(&db, sql), "{sql}");
        }
    }

    #[test]
    fn a_large_outer_keeps_the_standard_join() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let db = sealed(&dir);
        db.execute(
            "INSERT INTO s SELECT 100 + g.value, g.value % 1000, 0 FROM generate_series(1, 5000) g",
            (),
        )
        .unwrap();
        let sql = "WITH c AS (SELECT k FROM s) SELECT t.id FROM c JOIN t ON t.k = c.k";
        let (found, counts) = run(&db, sql);
        assert_eq!(counts.narrowed, 0, "a large outer narrowed the inner side");
        assert_eq!(counts.reads, (0, 0));
        assert_eq!(counts.sides, 2, "a side ran more than once");
        assert_eq!(found, same_as_control(&db, sql));
    }

    #[test]
    fn explain_names_the_conditional_strategy() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let db = sealed(&dir);
        let plan: Vec<String> = db
            .query(
                "EXPLAIN WITH c AS (SELECT k FROM s) SELECT t.id FROM c JOIN t ON t.k = c.k",
                (),
            )
            .unwrap()
            .map(|r| r.unwrap().get::<String>(0).unwrap())
            .collect();
        assert!(
            plan.iter().any(|line| line.contains(
                "Standard Join (conditional: a small outer narrows the inner side by its index)"
            )),
            "{plan:#?}"
        );
    }

    #[test]
    fn an_equality_inside_the_inner_table_narrows_nothing() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let db = sealed(&dir);
        for table in ["p", "q"] {
            db.execute(
                &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)"),
                (),
            )
            .unwrap();
            db.execute(&format!("INSERT INTO {table} VALUES (1, 1, 999)"), ())
                .unwrap();
        }
        for sql in [
            // The first equality names two inner columns; the outer side is a
            // join, so no left alias qualifies the outer column
            "SELECT t.id FROM p JOIN q USING (k, v) JOIN t ON t.k = t.v AND t.k = p.k",
            // An unqualified name the inner table also has
            "SELECT t.id FROM p JOIN q USING (k, v) JOIN t ON t.k = v",
        ] {
            let (found, counts) = run(&db, sql);
            assert_eq!(counts.narrowed, 0, "an unsure key narrowed: {sql}");
            assert_eq!(found, same_as_control(&db, sql), "{sql}");
        }
    }

    #[test]
    fn a_dotted_inner_alias_keeps_its_identity() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let db = sealed(&dir);
        let control = rows(
            &db,
            "SELECT \"i.n\".id FROM s JOIN u AS \"i.n\" ON \"i.n\".k = s.k",
        );
        let (found, counts) = run(
            &db,
            "SELECT \"i.n\".id FROM s JOIN t AS \"i.n\" ON \"i.n\".k = s.k",
        );
        assert_eq!(counts.narrowed, 1, "the small outer did not narrow");
        assert_eq!(found, control);
    }

    #[test]
    fn explain_names_the_strategy_only_where_it_runs() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let db = sealed(&dir);
        // The plain table on the right is never taken as the outer side
        let sql = "SELECT t.id FROM t JOIN s ON t.k = s.k";
        let plan: Vec<String> = db
            .query(&format!("EXPLAIN {sql}"), ())
            .unwrap()
            .map(|r| r.unwrap().get::<String>(0).unwrap())
            .collect();
        let (found, counts) = run(&db, sql);
        let named = plan
            .iter()
            .any(|line| line.contains("conditional: a small outer"));
        assert_eq!(named, counts.narrowed == 1, "{plan:#?}");
        assert_eq!(found, same_as_control(&db, sql));
    }
}
