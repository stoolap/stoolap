// Copyright 2026 Stoolap Contributors
// SPDX-License-Identifier: Apache-2.0

use stoolap::Database;

#[test]
fn returning_reports_affected_rows_for_dml_and_prepared_insert() {
    let db = Database::open_in_memory().unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)", ())
        .unwrap();
    assert_eq!(
        db.execute(
            "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30) RETURNING id",
            ()
        )
        .unwrap(),
        3
    );
    assert_eq!(
        db.prepare("INSERT INTO t VALUES ($1, $2) RETURNING id")
            .unwrap()
            .execute((4i64, 40i64))
            .unwrap(),
        1
    );
    let mut tx = db.begin().unwrap();
    assert_eq!(
        tx.execute("UPDATE t SET n=n+1 WHERE id<=3 RETURNING id, n", ())
            .unwrap(),
        3
    );
    let mut rows = tx
        .query("DELETE FROM t WHERE id<=3 RETURNING id", ())
        .unwrap();
    assert_eq!(rows.rows_affected(), 3);
    assert_eq!(
        rows.by_ref()
            .collect::<stoolap::Result<Vec<_>>>()
            .unwrap()
            .len(),
        3
    );
    assert_eq!(rows.rows_affected(), 3);
    drop(rows);
    assert_eq!(
        tx.execute("DELETE FROM t WHERE id<=3 RETURNING id", ())
            .unwrap(),
        0
    );
    tx.rollback().unwrap();
    assert_eq!(
        db.execute(
            "INSERT INTO t VALUES (1, 100) ON CONFLICT DO NOTHING RETURNING id",
            ()
        )
        .unwrap(),
        0
    );
    assert_eq!(
        db.query_one::<i64, _>("SELECT COUNT(*) FROM t", ())
            .unwrap(),
        4
    );
}

#[test]
fn returning_projection_failure_rolls_back_before_commit_and_reopen() {
    for cold in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let dsn = format!(
            "file://{}?checkpoint_interval=3600&sync_mode=full",
            dir.path().display()
        );
        let db = Database::open(&dsn).unwrap();
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
            .unwrap();
        if cold {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        for explicit in [false, true] {
            for sql in [
                "INSERT INTO t VALUES (4, 40), (5, 50) RETURNING 'x' REGEXP CASE WHEN id=5 THEN '[' ELSE 'x' END",
                "INSERT INTO t VALUES (4, 40), (5, 50) ON CONFLICT DO NOTHING RETURNING 'x' REGEXP CASE WHEN id=5 THEN '[' ELSE 'x' END",
                "INSERT INTO t SELECT id+3, n FROM t WHERE id<=2 RETURNING 'x' REGEXP CASE WHEN id=5 THEN '[' ELSE 'x' END",
                "INSERT INTO t SELECT id+3, n FROM t WHERE id<=2 ON CONFLICT DO NOTHING RETURNING 'x' REGEXP CASE WHEN id=5 THEN '[' ELSE 'x' END",
                "UPDATE t SET n=n+100 WHERE id<=3 RETURNING 'x' REGEXP CASE WHEN id=3 THEN '[' ELSE 'x' END",
                "DELETE FROM t WHERE id<=3 RETURNING 'x' REGEXP CASE WHEN id=3 THEN '[' ELSE 'x' END",
            ] {
                if explicit {
                    let mut tx = db.begin().unwrap();
                    tx.execute("INSERT INTO t VALUES (99, 99)", ()).unwrap();
                    assert!(tx.execute(sql, ()).is_err(), "{sql}");
                    assert_eq!(
                        tx.query_one::<i64, _>("SELECT COUNT(*) FROM t", ())
                            .unwrap(),
                        4
                    );
                    assert_eq!(
                        tx.query_one::<i64, _>("SELECT SUM(n) FROM t WHERE id<=3", ())
                            .unwrap(),
                        60
                    );
                    tx.commit().unwrap();
                    assert_eq!(
                        db.query_one::<i64, _>("SELECT n FROM t WHERE id=99", ())
                            .unwrap(),
                        99
                    );
                    db.execute("DELETE FROM t WHERE id=99", ()).unwrap();
                } else {
                    assert!(db.execute(sql, ()).is_err(), "{sql}");
                }
                assert_eq!(
                    db.query_one::<i64, _>("SELECT COUNT(*) FROM t", ())
                        .unwrap(),
                    3
                );
                assert_eq!(
                    db.query_one::<i64, _>("SELECT SUM(n) FROM t", ()).unwrap(),
                    60
                );
            }
        }
        // A compile-time RETURNING error is also pre-commit.
        assert!(db
            .execute("DELETE FROM t RETURNING missing_column", ())
            .is_err());
        drop(db);
        let reopened = Database::open(&dsn).unwrap();
        assert_eq!(
            reopened
                .query_one::<i64, _>("SELECT COUNT(*) FROM t", ())
                .unwrap(),
            3
        );
        assert_eq!(
            reopened
                .query_one::<i64, _>("SELECT SUM(n) FROM t", ())
                .unwrap(),
            60
        );
    }
}
