// Copyright 2026 Stoolap Contributors
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

//! An ON UPDATE action that reaches a column the statement itself changes
//! writes that column, and keys that form a cycle still end

use stoolap::Database;

/// The rows of `table` in id order, each column NULL as None
fn rows(db: &Database, table: &str) -> Vec<Vec<Option<i64>>> {
    db.query(&format!("SELECT * FROM {table} ORDER BY id"), ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| row.get::<Option<i64>>(i).unwrap())
                .collect()
        })
        .collect()
}

/// Opens `dsn` with `s`: b references a and c references b, b's key by
/// `action`
fn chain(dsn: &str, action: &str) -> Database {
    let db = Database::open(dsn).unwrap();
    for sql in [
        &format!(
            "CREATE TABLE s (id INTEGER PRIMARY KEY, a INTEGER UNIQUE, \
             b INTEGER UNIQUE REFERENCES s(a) ON UPDATE {action}, \
             c INTEGER REFERENCES s(b) ON UPDATE CASCADE)"
        ),
        "INSERT INTO s VALUES (1, 1, NULL, NULL), (2, 2, 2, NULL), (3, 3, 1, NULL), (4, 4, NULL, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

#[test]
fn an_update_of_a_and_of_b_carries_a_s_change_into_another_row_s_b() {
    let db = chain("memory://fk_into_changed_column_cascade", "CASCADE");
    db.execute("UPDATE s SET a = a + 10, b = b + 1 WHERE id IN (1, 2)", ())
        .unwrap();
    assert_eq!(
        rows(&db, "s"),
        [
            [Some(1), Some(11), None, None],
            [Some(2), Some(12), Some(3), None],
            [Some(3), Some(3), Some(11), None],
            [Some(4), Some(4), None, Some(11)],
        ],
        "row 3's b follows a from 1 to 11, and row 4's c follows that b"
    );
}

#[test]
fn an_update_of_a_and_of_b_clears_another_row_s_b_by_set_null() {
    let db = chain("memory://fk_into_changed_column_set_null", "SET NULL");
    db.execute("UPDATE s SET a = a + 10, b = b + 1 WHERE id IN (1, 2)", ())
        .unwrap();
    assert_eq!(
        rows(&db, "s"),
        [
            [Some(1), Some(11), None, None],
            [Some(2), Some(12), Some(3), None],
            [Some(3), Some(3), None, None],
            [Some(4), Some(4), None, None],
        ],
        "row 3's b is cleared, and row 4's c follows it"
    );
}

#[test]
fn an_upsert_changing_a_and_b_on_two_rows_carries_a_s_change() {
    let db = chain("memory://fk_into_changed_column_upsert", "CASCADE");
    db.execute(
        "INSERT INTO s VALUES (1, 5, NULL, NULL), (2, 2, 3, NULL) \
         ON CONFLICT (id) DO UPDATE SET a = excluded.a, b = excluded.b",
        (),
    )
    .unwrap();
    assert_eq!(
        rows(&db, "s"),
        [
            [Some(1), Some(5), None, None],
            [Some(2), Some(2), Some(3), None],
            [Some(3), Some(3), Some(5), None],
            [Some(4), Some(4), None, Some(5)],
        ]
    );
}

#[test]
fn a_cascade_round_a_cycle_of_keys_ends_where_it_reaches_no_row() {
    let db = Database::open("memory://fk_into_changed_column_cycles").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER UNIQUE REFERENCES t(b) ON UPDATE CASCADE, \
         b INTEGER UNIQUE REFERENCES t(a) ON UPDATE CASCADE)",
        "INSERT INTO t VALUES (1, NULL, NULL)",
        "UPDATE t SET a = 1, b = 1 WHERE id = 1",
        "INSERT INTO t VALUES (1, 3, 3) ON CONFLICT (id) DO UPDATE SET a = excluded.a",
        "CREATE TABLE w (id INTEGER PRIMARY KEY, a INTEGER UNIQUE REFERENCES w(c) ON UPDATE CASCADE, \
         b INTEGER UNIQUE REFERENCES w(a) ON UPDATE CASCADE, \
         c INTEGER UNIQUE REFERENCES w(b) ON UPDATE CASCADE)",
        "INSERT INTO w VALUES (1, NULL, NULL, NULL)",
        "UPDATE w SET a = 1, b = 1, c = 1 WHERE id = 1",
        "INSERT INTO w VALUES (1, 3, 3, 3) ON CONFLICT (id) DO UPDATE SET a = excluded.a",
    ] {
        db.execute(sql, ()).unwrap();
    }
    assert_eq!(rows(&db, "t"), [[Some(1), Some(3), Some(3)]]);
    assert_eq!(rows(&db, "w"), [[Some(1), Some(3), Some(3), Some(3)]]);
}
