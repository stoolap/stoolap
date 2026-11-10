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

fn rows(db: &Database, sql: &str) -> Vec<Vec<i64>> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len()).map(|i| row.get::<i64>(i).unwrap()).collect()
        })
        .collect()
}

/// Each layout the rows are read from: the hot store, then volumes
fn each_layout(name: &str, check: impl Fn(&Database)) {
    let tables = |dsn: &str, sealed: bool| {
        let db = Database::open(dsn).unwrap();
        for sql in [
            "CREATE TABLE p (id INTEGER PRIMARY KEY, u INTEGER UNIQUE)",
            "CREATE TABLE c (id INTEGER PRIMARY KEY REFERENCES p(u) ON UPDATE CASCADE, x INTEGER)",
            "INSERT INTO p VALUES (1, 10), (2, 20), (3, 30)",
            "INSERT INTO c VALUES (10, 100), (20, 200)",
        ] {
            db.execute(sql, ()).unwrap();
        }
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        db
    };
    check(&tables(
        &format!("memory://fk_cascade_into_key_{name}"),
        false,
    ));
    let dir = tempfile::tempdir().unwrap();
    check(&tables(&format!("file://{}", dir.path().display()), true));
}

#[test]
fn a_cascade_into_a_child_key_moves_the_child_to_its_new_key() {
    each_layout("moved", |db| {
        db.execute("UPDATE p SET u = 40 WHERE id = 1", ()).unwrap();
        assert_eq!(
            rows(db, "SELECT id, x FROM c ORDER BY id"),
            [[20, 200], [40, 100]]
        );
        assert_eq!(rows(db, "SELECT x FROM c WHERE id = 40"), [[100]]);
        assert!(rows(db, "SELECT x FROM c WHERE id = 10").is_empty());
        assert!(db.execute("INSERT INTO c VALUES (40, 9)", ()).is_err());
        db.execute("UPDATE p SET u = u + 100", ()).unwrap();
        assert_eq!(
            rows(db, "SELECT id, x FROM c ORDER BY id"),
            [[120, 200], [140, 100]]
        );
        assert_eq!(rows(db, "SELECT x FROM c WHERE id = 120"), [[200]]);
    });
}

#[test]
fn an_update_that_leaves_a_named_value_as_it_is_leaves_the_child_in_place() {
    each_layout("unchanged", |db| {
        for sql in [
            "UPDATE p SET u = u WHERE id = 1",
            "UPDATE p SET u = 10 WHERE id = 1",
            "UPDATE p SET u = id * 10 WHERE id <= 2",
            "UPDATE p SET u = u + 0",
            "UPDATE p SET u = CASE WHEN id = 3 THEN 31 ELSE u END",
        ] {
            db.execute(sql, ()).unwrap_or_else(|e| panic!("{sql}: {e}"));
        }
        db.execute("UPDATE p SET u = $1 WHERE id = 2", (20,))
            .unwrap();
        let stmt = db.prepare("UPDATE p SET u = $1 WHERE id = $2").unwrap();
        stmt.execute((10, 1)).unwrap();
        assert_eq!(
            rows(db, "SELECT id, u FROM p ORDER BY id"),
            [[1, 10], [2, 20], [3, 31]]
        );
        assert_eq!(rows(db, "SELECT x FROM c WHERE id = 10"), [[100]]);
        stmt.execute((11, 1)).unwrap();
        assert_eq!(rows(db, "SELECT x FROM c WHERE id = 11"), [[100]]);
    });
}

static TICKS: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
/// The counter is process-wide: the tests that read it take turns
static TICK_TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// 1 on its first call, then 2, 3 and on; registered as answering anew per call
#[derive(Default)]
struct KeyCascadeTick;

impl stoolap::functions::ScalarFunction for KeyCascadeTick {
    fn name(&self) -> &str {
        "KEY_CASCADE_TICK"
    }
    fn info(&self) -> stoolap::functions::FunctionInfo {
        use stoolap::functions::{FunctionDataType, FunctionInfo, FunctionSignature, FunctionType};
        FunctionInfo::new(
            "KEY_CASCADE_TICK",
            FunctionType::Scalar,
            "test counter",
            FunctionSignature::new(FunctionDataType::Integer, vec![], 0, 0),
        )
        .non_deterministic()
    }
    fn evaluate(
        &self,
        _args: &[stoolap::core::Value],
    ) -> stoolap::core::Result<stoolap::core::Value> {
        let tick = TICKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        Ok(stoolap::core::Value::Integer(tick))
    }
    fn clone_box(&self) -> Box<dyn stoolap::functions::ScalarFunction> {
        Box::new(KeyCascadeTick)
    }
}

#[test]
fn an_update_over_a_key_cascade_asks_its_where_once_per_row() {
    stoolap::functions::global_registry()
        .register_scalar::<KeyCascadeTick>()
        .unwrap();
    let db = Database::open("memory://fk_cascade_into_key_where_once").unwrap();
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY, u INTEGER UNIQUE)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY REFERENCES p(u) ON UPDATE CASCADE)",
        "INSERT INTO p VALUES (1, 10)",
        "INSERT INTO c VALUES (10)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let _ticks = TICK_TURN.lock().unwrap_or_else(|e| e.into_inner());
    TICKS.store(0, std::sync::atomic::Ordering::Relaxed);
    let updated = db
        .execute("UPDATE p SET u = 11 WHERE KEY_CASCADE_TICK() = 1", ())
        .unwrap();
    assert_eq!(updated, 1, "the row the WHERE chose is the row written");
    assert_eq!(rows(&db, "SELECT u FROM p"), [[11]]);
    assert_eq!(rows(&db, "SELECT id FROM c"), [[11]]);
    assert_eq!(TICKS.load(std::sync::atomic::Ordering::Relaxed), 1);
}

#[test]
fn a_parent_value_no_child_names_is_updated() {
    each_layout("unreferenced", |db| {
        db.execute("UPDATE p SET u = 31 WHERE id = 3", ()).unwrap();
        assert_eq!(rows(db, "SELECT u FROM p WHERE id = 3"), [[31]]);
    });
}

#[test]
fn a_moved_child_follows_the_transaction_it_was_moved_in() {
    each_layout("txn", |db| {
        db.execute("BEGIN", ()).unwrap();
        db.execute("UPDATE p SET u = 40 WHERE id = 1", ()).unwrap();
        assert!(db.execute("UPDATE p SET u = 20 WHERE id = 1", ()).is_err());
        assert_eq!(rows(db, "SELECT x FROM c WHERE id = 40"), [[100]]);
        db.execute("ROLLBACK", ()).unwrap();
        assert_eq!(
            rows(db, "SELECT id, x FROM c ORDER BY id"),
            [[10, 100], [20, 200]]
        );
        db.execute("BEGIN", ()).unwrap();
        db.execute("UPDATE p SET u = 40 WHERE id = 1", ()).unwrap();
        db.execute("COMMIT", ()).unwrap();
        assert_eq!(
            rows(db, "SELECT id, u FROM p ORDER BY id"),
            [[1, 40], [2, 20], [3, 30]]
        );
        assert_eq!(rows(db, "SELECT x FROM c WHERE id = 40"), [[100]]);
    });
}

#[test]
fn a_cascade_reaching_a_grandchild_key_moves_it_too() {
    let db = Database::open("memory://fk_cascade_into_key_grandchild").unwrap();
    for sql in [
        "CREATE TABLE g (id INTEGER PRIMARY KEY, a INTEGER UNIQUE)",
        "CREATE TABLE h (k INTEGER PRIMARY KEY, ga INTEGER UNIQUE REFERENCES g(a) ON UPDATE CASCADE)",
        "CREATE TABLE i (z INTEGER PRIMARY KEY REFERENCES h(ga) ON UPDATE CASCADE)",
        "INSERT INTO g VALUES (1, 5)",
        "INSERT INTO h VALUES (1, 5)",
        "INSERT INTO i VALUES (5)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db.execute("UPDATE g SET a = 6 WHERE id = 1", ()).unwrap();
    assert_eq!(rows(&db, "SELECT a FROM g"), [[6]]);
    assert_eq!(rows(&db, "SELECT ga FROM h"), [[6]]);
    assert_eq!(rows(&db, "SELECT z FROM i WHERE z = 6"), [[6]]);
    assert!(rows(&db, "SELECT z FROM i WHERE z = 5").is_empty());
}

#[test]
fn a_cascade_refuses_a_key_another_foreign_key_of_the_child_does_not_find() {
    let db = Database::open("memory://fk_cascade_into_key_other_parent").unwrap();
    for sql in [
        "CREATE TABLE p (id INTEGER PRIMARY KEY)",
        "CREATE TABLE q (id INTEGER PRIMARY KEY)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, \
         FOREIGN KEY (id) REFERENCES p(id) ON UPDATE CASCADE, \
         FOREIGN KEY (id) REFERENCES q(id))",
        "INSERT INTO p VALUES (1)",
        "INSERT INTO q VALUES (1)",
        "INSERT INTO c VALUES (1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let error = db
        .execute("UPDATE p SET id = 2 WHERE id = 1", ())
        .unwrap_err()
        .to_string();
    assert!(error.contains("does not exist"), "{error}");
    db.execute("INSERT INTO p VALUES (2)", ()).unwrap();
    let error = db
        .execute("UPDATE c SET id = 2 WHERE id = 1", ())
        .unwrap_err()
        .to_string();
    assert!(error.contains("does not exist"), "{error}");
    db.execute("DELETE FROM p WHERE id = 2", ()).unwrap();
    assert_eq!(rows(&db, "SELECT id FROM c"), [[1]]);
    assert_eq!(rows(&db, "SELECT id FROM p"), [[1]]);
}

#[test]
fn a_child_key_two_cascades_reach_is_checked_once_both_are_done() {
    let db = Database::open("memory://fk_cascade_into_key_two_paths").unwrap();
    for sql in [
        "CREATE TABLE r (id INTEGER PRIMARY KEY, u INTEGER UNIQUE)",
        "CREATE TABLE p (id INTEGER PRIMARY KEY REFERENCES r(id) ON UPDATE CASCADE)",
        "CREATE TABLE q (id INTEGER PRIMARY KEY REFERENCES r(u) ON UPDATE CASCADE)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, \
         FOREIGN KEY (id) REFERENCES p(id) ON UPDATE CASCADE, \
         FOREIGN KEY (id) REFERENCES q(id) ON UPDATE CASCADE)",
        "INSERT INTO r VALUES (1, 1)",
        "INSERT INTO p VALUES (1)",
        "INSERT INTO q VALUES (1)",
        "INSERT INTO c VALUES (1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db.execute("UPDATE r SET id = 2, u = 2", ()).unwrap();
    db.execute(
        "INSERT INTO r VALUES (2, 9) ON CONFLICT (id) DO UPDATE SET id = 5, u = 5",
        (),
    )
    .unwrap();
    for table in ["p", "q", "c"] {
        assert_eq!(
            rows(&db, &format!("SELECT id FROM {table}")),
            [[5]],
            "{table}"
        );
    }
    let error = db
        .execute("UPDATE r SET id = 6", ())
        .unwrap_err()
        .to_string();
    assert!(error.contains("does not exist"), "{error}");
    assert_eq!(rows(&db, "SELECT id FROM c"), [[5]]);
}

#[test]
fn a_child_key_two_cascades_reach_moves_once() {
    let db = Database::open("memory://fk_cascade_into_key_two_paths_once").unwrap();
    for sql in [
        "CREATE TABLE r (id INTEGER PRIMARY KEY)",
        "CREATE TABLE p (id INTEGER PRIMARY KEY REFERENCES r(id) ON UPDATE CASCADE)",
        "CREATE TABLE q (id INTEGER PRIMARY KEY REFERENCES r(id) ON UPDATE CASCADE)",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, x INTEGER, \
         FOREIGN KEY (id) REFERENCES p(id) ON UPDATE CASCADE, \
         FOREIGN KEY (id) REFERENCES q(id) ON UPDATE CASCADE)",
        "INSERT INTO r VALUES (1), (2), (3)",
        "INSERT INTO p VALUES (1), (2), (3)",
        "INSERT INTO q VALUES (1), (2), (3)",
        "INSERT INTO c VALUES (1, 10), (3, 30)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db.execute("UPDATE r SET id = id + 1", ()).unwrap();
    assert_eq!(
        rows(&db, "SELECT id, x FROM c ORDER BY id"),
        [[2, 10], [4, 30]]
    );
    db.execute("UPDATE r SET id = id - 1", ()).unwrap();
    assert_eq!(
        rows(&db, "SELECT id, x FROM c ORDER BY id"),
        [[1, 10], [3, 30]]
    );
}

fn open(name: &str, sql: &[&str]) -> Database {
    let db = Database::open(&format!("memory://fk_cascade_graph_{name}")).unwrap();
    for sql in sql {
        db.execute(sql, ()).unwrap();
    }
    db
}

#[test]
fn a_row_two_columns_of_cascades_reach_takes_each_value_once() {
    let db = open(
        "two_columns",
        &[
            "CREATE TABLE r (id INTEGER PRIMARY KEY, a INTEGER UNIQUE, b INTEGER UNIQUE, d INTEGER UNIQUE)",
            "INSERT INTO r VALUES (1, 1, 1, 1), (2, 2, 2, 2)",
            "CREATE TABLE p (k INTEGER PRIMARY KEY REFERENCES r(a) ON UPDATE CASCADE)",
            "CREATE TABLE q (k INTEGER PRIMARY KEY REFERENCES r(d) ON UPDATE CASCADE)",
            "INSERT INTO p VALUES (1), (2)",
            "INSERT INTO q VALUES (1), (2)",
            "CREATE TABLE c (id INTEGER PRIMARY KEY, k INTEGER UNIQUE, \
             FOREIGN KEY (id) REFERENCES r(b) ON UPDATE CASCADE, \
             FOREIGN KEY (k) REFERENCES p(k) ON UPDATE CASCADE, \
             FOREIGN KEY (k) REFERENCES q(k) ON UPDATE CASCADE)",
            "INSERT INTO c VALUES (1, 1)",
        ],
    );
    db.execute("UPDATE r SET a = a + 1, b = b + 1, d = d + 1", ())
        .unwrap();
    assert_eq!(rows(&db, "SELECT id, k FROM c"), [[2, 2]]);
}

#[test]
fn a_set_null_and_a_cascade_into_one_column_leave_it_null() {
    let db = open(
        "set_null",
        &[
            "CREATE TABLE r (id INTEGER PRIMARY KEY, a INTEGER UNIQUE, b INTEGER UNIQUE)",
            "INSERT INTO r VALUES (1, 1, 1), (2, 2, 2)",
            "CREATE TABLE p (id INTEGER PRIMARY KEY REFERENCES r(a) ON UPDATE CASCADE)",
            "INSERT INTO p VALUES (1)",
            "CREATE TABLE c (id INTEGER PRIMARY KEY, k INTEGER UNIQUE, \
             FOREIGN KEY (k) REFERENCES p(id) ON UPDATE CASCADE, \
             FOREIGN KEY (k) REFERENCES r(b) ON UPDATE SET NULL)",
            "INSERT INTO c VALUES (1, 1)",
        ],
    );
    db.execute("UPDATE r SET a = a + 1, b = b + 2", ()).unwrap();
    let k: Option<i64> = db.query_one("SELECT k FROM c", ()).unwrap();
    assert_eq!(k, None);
}

#[test]
fn a_change_no_child_row_held_at_the_start_is_not_cascaded() {
    let db = open(
        "start_values",
        &[
            "CREATE TABLE r (id INTEGER PRIMARY KEY, a INTEGER UNIQUE, b FLOAT UNIQUE)",
            "INSERT INTO r VALUES (1, 1, 1.0), (2, 2, 2.0)",
            "CREATE TABLE p (id INTEGER PRIMARY KEY REFERENCES r(a) ON UPDATE CASCADE)",
            "CREATE TABLE q (id INTEGER PRIMARY KEY, k FLOAT UNIQUE REFERENCES r(b) ON UPDATE CASCADE)",
            "INSERT INTO p VALUES (1), (2)",
            "INSERT INTO q VALUES (1, 1.0), (2, 2.0)",
            "CREATE TABLE c (id INTEGER PRIMARY KEY, \
             FOREIGN KEY (id) REFERENCES p(id) ON UPDATE CASCADE, \
             FOREIGN KEY (id) REFERENCES q(k) ON UPDATE CASCADE)",
            "INSERT INTO c VALUES (1)",
        ],
    );
    db.execute(
        "UPDATE r SET a = a + 1, b = CASE WHEN id = 1 THEN 2.0 ELSE 2.5 END",
        (),
    )
    .unwrap();
    assert_eq!(rows(&db, "SELECT id FROM c"), [[2]]);
}

#[test]
fn two_cascades_that_give_a_row_different_values_refuse_the_statement() {
    let db = open(
        "disagree",
        &[
            "CREATE TABLE r (id INTEGER PRIMARY KEY, a INTEGER UNIQUE)",
            "INSERT INTO r VALUES (1, 1)",
            "CREATE TABLE p (id INTEGER PRIMARY KEY REFERENCES r(id) ON UPDATE CASCADE)",
            "CREATE TABLE q (id INTEGER PRIMARY KEY REFERENCES r(a) ON UPDATE CASCADE)",
            "INSERT INTO p VALUES (1)",
            "INSERT INTO q VALUES (1)",
            "CREATE TABLE c (id INTEGER PRIMARY KEY, \
             FOREIGN KEY (id) REFERENCES p(id) ON UPDATE CASCADE, \
             FOREIGN KEY (id) REFERENCES q(id) ON UPDATE CASCADE)",
            "INSERT INTO c VALUES (1)",
        ],
    );
    let error = db
        .execute("UPDATE r SET id = 2, a = 3", ())
        .unwrap_err()
        .to_string();
    assert!(error.contains("cannot cascade both"), "{error}");
    assert_eq!(rows(&db, "SELECT id FROM c"), [[1]]);
    assert_eq!(rows(&db, "SELECT id, a FROM r"), [[1, 1]]);
}

#[test]
fn a_set_null_reaches_the_rows_that_reference_the_nulled_column() {
    let db = open(
        "set_null_beneath",
        &[
            "CREATE TABLE r (id INTEGER PRIMARY KEY)",
            "INSERT INTO r VALUES (1)",
            "CREATE TABLE c (id INTEGER PRIMARY KEY, k INTEGER UNIQUE REFERENCES r(id) ON UPDATE SET NULL)",
            "INSERT INTO c VALUES (1, 1)",
            "CREATE TABLE g (id INTEGER PRIMARY KEY, x INTEGER REFERENCES c(k) ON UPDATE CASCADE)",
            "INSERT INTO g VALUES (1, 1)",
        ],
    );
    db.execute("UPDATE r SET id = 2", ()).unwrap();
    let k: Option<i64> = db.query_one("SELECT k FROM c", ()).unwrap();
    let x: Option<i64> = db.query_one("SELECT x FROM g", ()).unwrap();
    assert_eq!((k, x), (None, None));
}

#[test]
fn a_set_null_leaves_a_column_null_whatever_the_cascades_beside_it_give() {
    for (null_on, name) in [("a", "null_on_a"), ("b", "null_on_b"), ("c", "null_on_c")] {
        let key = |column: &str| {
            let action = if column == null_on {
                "SET NULL"
            } else {
                "CASCADE"
            };
            format!("FOREIGN KEY (k) REFERENCES r({column}) ON UPDATE {action}")
        };
        let db = open(
            name,
            &[
                "CREATE TABLE r (id INTEGER PRIMARY KEY, a INTEGER UNIQUE, b INTEGER UNIQUE, c INTEGER UNIQUE)",
                "INSERT INTO r VALUES (1, 1, 1, 1)",
                &format!(
                    "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER UNIQUE, {}, {}, {})",
                    key("a"),
                    key("b"),
                    key("c")
                ),
                "INSERT INTO t VALUES (1, 1)",
            ],
        );
        db.execute("UPDATE r SET a = 4, b = 2, c = 3", ()).unwrap();
        let k: Option<i64> = db.query_one("SELECT k FROM t", ()).unwrap();
        assert_eq!(k, None, "SET NULL on {null_on}");
    }
}

#[test]
fn a_set_null_leaves_a_column_null_where_a_cascade_beside_it_cannot_be_stored() {
    for (null_on, name) in [("a", "unstored_null_on_a"), ("b", "unstored_null_on_b")] {
        let (float, int) = if null_on == "a" {
            ("b", "a")
        } else {
            ("a", "b")
        };
        let db = open(
            name,
            &[
                &format!(
                    "CREATE TABLE r (id INTEGER PRIMARY KEY, {float} FLOAT UNIQUE, {int} INTEGER UNIQUE)"
                ),
                &format!("INSERT INTO r (id, {float}, {int}) VALUES (1, 1.0, 1)"),
                &format!(
                    "CREATE TABLE c (id INTEGER PRIMARY KEY, k INTEGER UNIQUE, \
                     FOREIGN KEY (k) REFERENCES r({float}) ON UPDATE CASCADE, \
                     FOREIGN KEY (k) REFERENCES r({int}) ON UPDATE SET NULL)"
                ),
                "INSERT INTO c VALUES (1, 1)",
            ],
        );
        db.execute(&format!("UPDATE r SET {float} = 2.5, {int} = 2"), ())
            .unwrap();
        let k: Option<i64> = db.query_one("SELECT k FROM c", ()).unwrap();
        assert_eq!(k, None, "SET NULL on {null_on}");
    }
}
