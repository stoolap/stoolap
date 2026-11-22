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

//! ALTER TABLE MODIFY COLUMN that changes a column's type judges every
//! CHECK of the table that reads the column over the converted values

use stoolap::Database;

fn text(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| row.get_value(i).unwrap().to_string())
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect()
}

/// `t (id, v, w, u)` as `columns` declares it, holding `rows`, on the hot
/// store and then sealed
fn each_layout(columns: &str, rows: &str, check: impl Fn(&Database, &str)) {
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let dsn = format!("file://{}?checkpoint_interval=0", dir.path().display());
        let db = Database::open(&dsn).unwrap();
        db.execute(
            &format!("CREATE TABLE t (id INTEGER PRIMARY KEY, {columns})"),
            (),
        )
        .unwrap();
        db.execute(&format!("INSERT INTO t VALUES {rows}"), ())
            .unwrap();
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        check(&db, if sealed { "sealed" } else { "hot" });
    }
}

/// The type change is refused with `names` in its error, and the table and
/// its rows stay as they were
fn assert_refused(columns: &str, rows: &str, alter: &str, names: &str) {
    each_layout(columns, rows, |db, layout| {
        let before = (
            text(db, "SHOW CREATE TABLE t"),
            text(db, "SELECT * FROM t ORDER BY id"),
        );
        let got = match db.execute(alter, ()) {
            Ok(_) => "accepted".to_string(),
            Err(error) => error.to_string(),
        };
        assert!(
            got.contains("CHECK constraint failed") && got.contains(names),
            "{layout}: {alter}: {got}"
        );
        let after = (
            text(db, "SHOW CREATE TABLE t"),
            text(db, "SELECT * FROM t ORDER BY id"),
        );
        assert_eq!(after, before, "{layout}: {alter}");
    });
}

#[test]
fn a_type_change_is_refused_when_a_converted_value_fails_the_columns_check() {
    assert_refused(
        "v FLOAT CHECK (v <> 7), w INTEGER, u INTEGER",
        "(1, 3.5, 1, 1), (2, 7.4, 2, 2)",
        "ALTER TABLE t MODIFY COLUMN v INTEGER",
        "(v <> 7)",
    );
}

#[test]
fn a_type_change_is_refused_when_a_converted_value_fails_another_columns_check() {
    assert_refused(
        "v FLOAT, w INTEGER CHECK (w < v), u INTEGER",
        "(1, 3.5, 1, 1), (2, 8.5, 8, 2)",
        "ALTER TABLE t MODIFY COLUMN v INTEGER",
        "(w < v)",
    );
}

#[test]
fn a_type_change_is_refused_when_a_converted_value_fails_a_table_check() {
    assert_refused(
        "v FLOAT, w INTEGER, u INTEGER, CHECK (u + v > 13)",
        "(1, 30.5, 1, 1), (2, 8.5, 2, 5)",
        "ALTER TABLE t MODIFY COLUMN v INTEGER",
        "((u + v) > 13)",
    );
}

#[test]
fn a_type_change_judges_the_check_written_with_it_and_those_kept() {
    assert_refused(
        "v FLOAT, w INTEGER CHECK (w < v), u INTEGER",
        "(1, 3.5, 1, 1), (2, 8.5, 8, 2)",
        "ALTER TABLE t MODIFY COLUMN v INTEGER CHECK (v < 100)",
        "(w < v)",
    );
}

#[test]
fn a_type_change_is_taken_when_every_converted_row_passes() {
    each_layout(
        "v FLOAT CHECK (v <> 7), w INTEGER CHECK (w < v), u TEXT, CHECK (LENGTH(u) < v)",
        "(1, 3.5, 1, 'a'), (2, 8.5, 7, 'b'), (3, 9.5, NULL, NULL)",
        |db, layout| {
            for sql in [
                "ALTER TABLE t MODIFY COLUMN v INTEGER",
                // A value the new type cannot hold becomes NULL, which passes
                "ALTER TABLE t MODIFY COLUMN u INTEGER",
            ] {
                db.execute(sql, ())
                    .unwrap_or_else(|e| panic!("{layout}: {sql}: {e}"));
            }
            assert_eq!(
                text(db, "SELECT id, v, w, u FROM t ORDER BY id"),
                ["1|3|1|NULL", "2|8|7|NULL", "3|9|NULL|NULL"],
                "{layout}"
            );
        },
    );
}
