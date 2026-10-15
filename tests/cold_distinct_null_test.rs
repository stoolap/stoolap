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

//! DISTINCT, COUNT(DISTINCT) and window partitions see the NULLs of sealed
//! rows the way they see hot ones: a visible NULL counts, a hidden one does not

use stoolap::core::Value;
use stoolap::Database;

/// Six rows, two of them NULL in f, s and i, all NULL in n, sealed when `seal`
fn open(dir: &std::path::Path, indexed: bool, seal: bool) -> Database {
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, f FLOAT, s TEXT, i INTEGER, n INTEGER)",
        (),
    )
    .unwrap();
    if indexed {
        for c in ["f", "s", "i", "n"] {
            db.execute(&format!("CREATE INDEX t_{c} ON t({c})"), ())
                .unwrap();
        }
    }
    db.execute(
        "INSERT INTO t (id, f, s, i) VALUES (1, 1.5, 'a', 1), (2, NULL, NULL, NULL), (3, 2.5, 'b', 2), \
         (4, 1.5, 'a', 1), (5, NULL, NULL, NULL), (6, 2.5, 'b', 2)",
        (),
    )
    .unwrap();
    if seal {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn render(v: &Value) -> String {
    match v {
        Value::Null(_) => "NULL".into(),
        Value::Float(x) => format!("{x:?}"),
        Value::Integer(x) => x.to_string(),
        Value::Text(s) => s.to_string(),
        other => format!("{other:?}"),
    }
}

/// The first column of every row, sorted
fn column(db: &Database, sql: &str) -> Vec<String> {
    let mut out: Vec<String> = db
        .query(sql, ())
        .unwrap()
        .map(|row| render(row.unwrap().get_value(0).unwrap()))
        .collect();
    out.sort();
    out
}

fn count(db: &Database, sql: &str) -> i64 {
    db.query_one::<i64, _>(sql, ()).unwrap()
}

fn each_layout(check: impl Fn(&Database, &str)) {
    for indexed in [false, true] {
        for seal in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let db = open(dir.path(), indexed, seal);
            check(&db, &format!("indexed={indexed} sealed={seal}"));
        }
    }
}

#[test]
fn distinct_keeps_the_null_of_sealed_rows() {
    each_layout(|db, at| {
        for (c, want) in [
            ("f", ["1.5", "2.5", "NULL"]),
            ("s", ["NULL", "a", "b"]),
            ("i", ["1", "2", "NULL"]),
        ] {
            assert_eq!(
                column(db, &format!("SELECT DISTINCT {c} FROM t")),
                want,
                "{at}: DISTINCT {c}"
            );
            assert_eq!(
                count(db, &format!("SELECT COUNT(DISTINCT {c}) FROM t")),
                2,
                "{at}: COUNT(DISTINCT {c})"
            );
        }
    });
}

#[test]
fn a_null_stays_while_one_null_row_is_visible() {
    each_layout(|db, at| {
        db.execute("DELETE FROM t WHERE id = 2", ()).unwrap();
        assert_eq!(
            column(db, "SELECT DISTINCT f FROM t"),
            ["1.5", "2.5", "NULL"],
            "{at}"
        );
        assert_eq!(
            column(db, "SELECT DISTINCT s FROM t"),
            ["NULL", "a", "b"],
            "{at}"
        );
    });
}

#[test]
fn deleted_null_rows_leave_no_null() {
    each_layout(|db, at| {
        db.execute("DELETE FROM t WHERE f IS NULL", ()).unwrap();
        assert_eq!(
            column(db, "SELECT DISTINCT f FROM t"),
            ["1.5", "2.5"],
            "{at}"
        );
        assert_eq!(column(db, "SELECT DISTINCT s FROM t"), ["a", "b"], "{at}");
        assert_eq!(column(db, "SELECT DISTINCT i FROM t"), ["1", "2"], "{at}");
    });
}

#[test]
fn null_rows_updated_to_values_leave_no_null() {
    each_layout(|db, at| {
        db.execute("UPDATE t SET f = 9.5, s = 'z', i = 9 WHERE f IS NULL", ())
            .unwrap();
        assert_eq!(
            column(db, "SELECT DISTINCT f FROM t"),
            ["1.5", "2.5", "9.5"],
            "{at}"
        );
        assert_eq!(
            column(db, "SELECT DISTINCT s FROM t"),
            ["a", "b", "z"],
            "{at}"
        );
        assert_eq!(
            column(db, "SELECT DISTINCT i FROM t"),
            ["1", "2", "9"],
            "{at}"
        );
    });
}

#[test]
fn an_added_column_keeps_its_null_default() {
    each_layout(|db, at| {
        db.execute("ALTER TABLE t ADD COLUMN g INTEGER", ())
            .unwrap();
        db.execute("CREATE INDEX t_g ON t(g)", ()).unwrap();
        assert_eq!(column(db, "SELECT DISTINCT g FROM t"), ["NULL"], "{at}");
        assert_eq!(count(db, "SELECT COUNT(DISTINCT g) FROM t"), 0, "{at}");
        assert_eq!(
            column(db, "SELECT id FROM t WHERE g IS NULL"),
            ["1", "2", "3", "4", "5", "6"],
            "{at}"
        );
        let ids = column(
            db,
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY g) FROM t LIMIT 100",
        );
        assert_eq!(ids, ["1", "2", "3", "4", "5", "6"], "{at}: window LIMIT");
        let ids = column(db, "SELECT id, ROW_NUMBER() OVER (PARTITION BY g) FROM t");
        assert_eq!(ids, ["1", "2", "3", "4", "5", "6"], "{at}: window");
        db.execute("ALTER TABLE t ADD COLUMN h TEXT", ()).unwrap();
        assert_eq!(column(db, "SELECT DISTINCT h FROM t"), ["NULL"], "{at}: h");
        assert_eq!(count(db, "SELECT COUNT(DISTINCT h) FROM t"), 0, "{at}: h");
    });
}

#[test]
fn sealed_and_hot_nulls_share_one_window_partition() {
    each_layout(|db, at| {
        db.execute("INSERT INTO t VALUES (7, NULL, NULL, NULL, NULL)", ())
            .unwrap();
        for c in ["f", "s", "i"] {
            for limit in ["", " LIMIT 100"] {
                let sql = format!("SELECT id, COUNT(*) OVER (PARTITION BY {c}) FROM t{limit}");
                let mut sizes: Vec<(i64, i64)> = db
                    .query(&sql, ())
                    .unwrap()
                    .map(|row| {
                        let row = row.unwrap();
                        (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap())
                    })
                    .collect();
                sizes.sort_unstable();
                assert_eq!(
                    sizes,
                    [(1, 2), (2, 3), (3, 2), (4, 2), (5, 3), (6, 2), (7, 3)],
                    "{at}: {sql}"
                );
            }
        }
    });
}

#[test]
fn an_all_null_column_keeps_its_rows() {
    each_layout(|db, at| {
        assert_eq!(column(db, "SELECT DISTINCT n FROM t"), ["NULL"], "{at}");
        assert_eq!(count(db, "SELECT COUNT(DISTINCT n) FROM t"), 0, "{at}");
        for limit in ["", " LIMIT 100"] {
            let sql = format!("SELECT id, ROW_NUMBER() OVER (PARTITION BY n) FROM t{limit}");
            assert_eq!(
                column(db, &sql),
                ["1", "2", "3", "4", "5", "6"],
                "{at}: {sql}"
            );
        }
    });
}

#[test]
fn count_distinct_skips_a_hot_null_beside_sealed_rows() {
    each_layout(|db, at| {
        db.execute("INSERT INTO t VALUES (7, NULL, NULL, NULL, NULL)", ())
            .unwrap();
        for c in ["f", "s", "i"] {
            assert_eq!(
                count(db, &format!("SELECT COUNT(DISTINCT {c}) FROM t")),
                2,
                "{at}: COUNT(DISTINCT {c})"
            );
        }
        assert_eq!(
            column(db, "SELECT DISTINCT f FROM t"),
            ["1.5", "2.5", "NULL"],
            "{at}"
        );
    });
}

#[test]
fn window_partitions_keep_sealed_null_rows() {
    each_layout(|db, at| {
        for c in ["f", "s", "i"] {
            for limit in ["", " LIMIT 100"] {
                let sql = format!("SELECT id, ROW_NUMBER() OVER (PARTITION BY {c}) FROM t{limit}");
                assert_eq!(
                    column(db, &sql),
                    ["1", "2", "3", "4", "5", "6"],
                    "{at}: {sql}"
                );
            }
        }
    });
}
