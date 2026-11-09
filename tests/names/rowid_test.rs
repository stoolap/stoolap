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

//! `_rowid_` reads a row's identity: the primary key's value in a table with
//! one, the id the engine assigned in a table without one. No column and no
//! alias takes the name.

use stoolap::core::Value;
use stoolap::Database;

fn int(v: i64) -> Value {
    Value::Integer(v)
}

fn text(v: &str) -> Value {
    Value::text(v)
}

fn rows(db: &Database, sql: &str) -> Vec<Vec<Value>> {
    let result = db.query(sql, ()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    result
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| row.get_value(i).unwrap().clone())
                .collect()
        })
        .collect()
}

fn names(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ()).unwrap().columns().to_vec()
}

fn error(db: &Database, sql: &str) -> String {
    match db.query(sql, ()) {
        Ok(result) => match result.into_iter().find_map(|row| row.err()) {
            Some(e) => e.to_string(),
            None => panic!("{sql}: no error"),
        },
        Err(e) => e.to_string(),
    }
}

/// A database with `k`, keyed by `id`, and `n`, without a primary key, each
/// holding three rows, and `r`, with a column of its own named rowid; sealed
/// into volumes when `sealed`
fn tables(dsn: &str, sealed: bool) -> Database {
    let db = Database::open(dsn).unwrap();
    for sql in [
        "CREATE TABLE k (id INTEGER PRIMARY KEY, a INTEGER, b TEXT)",
        "CREATE TABLE n (a INTEGER, b TEXT)",
        "CREATE TABLE r (rowid INTEGER, a INTEGER)",
        "INSERT INTO k VALUES (10, 1, 'x'), (20, 2, 'y'), (30, 3, 'z')",
        "INSERT INTO n VALUES (1, 'x'), (2, 'y'), (3, 'z')",
        "INSERT INTO r VALUES (100, 1), (200, 2)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// Each layout the rows are read from: the hot store, then volumes
fn each_layout(name: &str, check: impl Fn(&Database)) {
    check(&tables(&format!("memory://rowid_{name}"), false));
    let dir = tempfile::tempdir().unwrap();
    check(&tables(&format!("file://{}", dir.path().display()), true));
}

/// Every read context over table `t`, whose rows 1, 2 and 3 (column a) have
/// the row ids `ids`
fn reads(db: &Database, t: &str, [i1, i2, i3]: [i64; 3]) {
    let q = |sql: &str| rows(db, &sql.replace("$t", t));
    let sum = i1 + i2 + i3;
    assert_eq!(
        q("SELECT _rowid_, a FROM $t ORDER BY _rowid_"),
        [[int(i1), int(1)], [int(i2), int(2)], [int(i3), int(3)]],
        "select list in {t}"
    );
    assert_eq!(
        names(db, &format!("SELECT _rowid_, {t}._rowid_ FROM {t}")),
        ["_rowid_", "_rowid_"],
        "headers in {t}"
    );
    assert_eq!(
        q(&format!("SELECT b FROM $t WHERE _rowid_ = {i2}")),
        [[text("y")]],
        "equality in {t}"
    );
    assert_eq!(
        q(&format!(
            "SELECT a FROM $t WHERE $t._rowid_ IN ({i3}, {i1}, NULL) ORDER BY a"
        )),
        [[int(1)], [int(3)]],
        "IN in {t}"
    );
    assert_eq!(
        q(&format!(
            "SELECT _rowid_ FROM $t WHERE _rowid_ > {i1} ORDER BY _rowid_ DESC LIMIT 1"
        )),
        [[int(i3)]],
        "range in {t}"
    );
    assert_eq!(
        q(&format!("SELECT a FROM $t WHERE _rowid_ = {i2} AND a = 9")),
        Vec::<Vec<Value>>::new(),
        "equality with another conjunct in {t}"
    );
    let by_param = db
        .query(&format!("SELECT a FROM {t} WHERE _rowid_ = $1"), (i3,))
        .unwrap()
        .map(|row| row.unwrap().get_value(0).unwrap().clone())
        .collect::<Vec<_>>();
    assert_eq!(by_param, [int(3)], "parameter in {t}");
    assert_eq!(
        q("SELECT COUNT(*), MAX(_rowid_), SUM(_rowid_) FROM $t"),
        [[int(3), int(i3), int(sum)]],
        "aggregates in {t}"
    );
    assert_eq!(
        q(&format!(
            "SELECT _rowid_, COUNT(*) FROM $t GROUP BY _rowid_ HAVING _rowid_ > {i1} ORDER BY _rowid_"
        )),
        [[int(i2), int(1)], [int(i3), int(1)]],
        "GROUP BY and HAVING in {t}"
    );
    assert_eq!(
        q(&format!(
            "SELECT CASE WHEN _rowid_ = {i2} THEN 'hit' ELSE 'miss' END FROM $t ORDER BY a"
        )),
        [[text("miss")], [text("hit")], [text("miss")]],
        "CASE in {t}"
    );
    assert_eq!(
        q("SELECT a, ROW_NUMBER() OVER (ORDER BY _rowid_ DESC) FROM $t ORDER BY a"),
        [[int(1), int(3)], [int(2), int(2)], [int(3), int(1)]],
        "window in {t}"
    );
    assert_eq!(
        q("SELECT (SELECT COUNT(*) FROM $t i WHERE i._rowid_ <= o._rowid_) FROM $t o ORDER BY o.a"),
        [[int(1)], [int(2)], [int(3)]],
        "correlated subquery in {t}"
    );
    assert_eq!(
        q("SELECT a FROM $t o WHERE EXISTS (SELECT 1 FROM $t i WHERE i._rowid_ > o._rowid_) ORDER BY a"),
        [[int(1)], [int(2)]],
        "EXISTS in {t}"
    );
    assert_eq!(
        q("SELECT a FROM $t WHERE _rowid_ IN (SELECT _rowid_ FROM $t WHERE a > 1) ORDER BY a"),
        [[int(2)], [int(3)]],
        "IN subquery in {t}"
    );
    assert_eq!(
        q("SELECT (SELECT i._rowid_ FROM $t i ORDER BY i._rowid_ DESC LIMIT 1)"),
        [[int(i3)]],
        "a subquery run as a statement in {t}"
    );
    assert_eq!(
        q("SELECT (SELECT _rowid_ FROM $t WHERE a = 2 UNION SELECT -1 ORDER BY 1 DESC LIMIT 1)"),
        [[int(i2)]],
        "a set operation inside a subquery in {t}"
    );
    assert_eq!(
        q("SELECT a FROM $t o WHERE EXISTS (SELECT DISTINCT i._rowid_ FROM $t i WHERE i._rowid_ = o._rowid_ AND i.a > 1) ORDER BY a"),
        [[int(2)], [int(3)]],
        "EXISTS correlated by rowid in {t}"
    );
    assert_eq!(
        q("SELECT _rowid_ FROM $t WHERE a = 1 UNION ALL SELECT _rowid_ FROM $t WHERE a = 3"),
        [[int(i1)], [int(i3)]],
        "set operation in {t}"
    );
    assert_eq!(
        q("SELECT * FROM (SELECT _rowid_ FROM $t) d ORDER BY 1"),
        [[int(i1)], [int(i2)], [int(i3)]],
        "a derived table's output column in {t}"
    );
    assert_eq!(
        q(&format!("SELECT * FROM $t WHERE _rowid_ = {i1}"))[0].len(),
        if t == "k" { 3 } else { 2 },
        "a star leaves rowid out in {t}"
    );
    assert_eq!(
        q("SELECT \"_rowid_\" FROM $t WHERE a = 1"),
        [[text("_rowid_")]],
        "a quoted name is no rowid in {t}"
    );
    assert!(
        error(db, &format!("SELECT _rowid_ FROM (SELECT * FROM {t}) d")).contains("not found"),
        "a derived table has no rowid in {t}"
    );
    assert_eq!(
        q("SELECT *, _rowid_ AS r FROM $t ORDER BY r + 0 DESC")[0].last(),
        Some(&int(i3)),
        "an alias after a star in {t}"
    );
    assert_eq!(
        q("SELECT _rowid_, COUNT(*) FROM $t GROUP BY GROUPING SETS ((_rowid_), ()) ORDER BY 1"),
        [
            [int(i1), int(1)],
            [int(i2), int(1)],
            [int(i3), int(1)],
            [Value::null_unknown(), int(3)]
        ],
        "GROUPING SETS in {t}"
    );
    assert_eq!(
        q("SELECT a, SUM(a) OVER w FROM $t WINDOW w AS (ORDER BY _rowid_) ORDER BY a"),
        [[int(1), int(1)], [int(2), int(3)], [int(3), int(6)]],
        "a named window in {t}"
    );
    assert_eq!(
        q("SELECT SUM((SELECT o._rowid_)) FROM $t o"),
        [[int(sum)]],
        "a subquery inside an aggregate in {t}"
    );
    assert_eq!(
        q("SELECT (SELECT x FROM (VALUES (o._rowid_)) v(x)) FROM $t o ORDER BY o.a"),
        [[int(i1)], [int(i2)], [int(i3)]],
        "a VALUES row reading the outer rowid in {t}"
    );
    assert_eq!(
        q("SELECT a FROM $t o ORDER BY (SELECT o._rowid_) DESC"),
        [[int(3)], [int(2)], [int(1)]],
        "an ORDER BY key reading rowid through a subquery in {t}"
    );
    assert_eq!(
        q("SELECT SUM(_rowid_) OVER () + 0, (SELECT 1) FROM $t"),
        [[int(sum), int(1)], [int(sum), int(1)], [int(sum), int(1)]],
        "a grouped window over rowid in {t}"
    );
    assert_eq!(
        q("SELECT *, SUM(_rowid_) OVER () FROM $t WHERE a = 1")[0].len(),
        if t == "k" { 4 } else { 3 },
        "a star beside a window leaves rowid out in {t}"
    );
    assert!(
        error(db, &format!("SELECT STRING_AGG(b, _rowid_) FROM {t}"))
            .contains("must not read the row"),
        "rowid as an aggregate's later argument reads the row in {t}"
    );
    let lagged = db
        .query(
            &format!("SELECT *, LAG(_rowid_, $1) OVER (ORDER BY _rowid_) FROM {t} ORDER BY a"),
            (1,),
        )
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| row.get_value(i).unwrap().clone())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        lagged
            .iter()
            .map(|row| row.last().cloned())
            .collect::<Vec<_>>(),
        [Some(Value::null_unknown()), Some(int(i1)), Some(int(i2))],
        "a window planned at its run in {t}"
    );
    assert_eq!(
        lagged[0].len(),
        if t == "k" { 4 } else { 3 },
        "a star beside a window planned at its run in {t}"
    );
    assert_eq!(
        q("SELECT a FROM $t ORDER BY SUM(_rowid_) OVER (), a"),
        [[int(1)], [int(2)], [int(3)]],
        "a window reading rowid in ORDER BY in {t}"
    );
    assert_eq!(
        q("SELECT a, SUM(_rowid_) OVER () AS s FROM $t ORDER BY s, a"),
        [[int(1), int(sum)], [int(2), int(sum)], [int(3), int(sum)]],
        "a window reading rowid ordered by its alias in {t}"
    );
}

/// Every clause over table `t`, whose rows 30, 20 and 10 (column a) have the
/// row ids `ids`, ascending, so an order by a reads them backwards
fn clauses(db: &Database, t: &str, [i1, i2, i3]: [i64; 3]) {
    let q = |sql: &str| rows(db, &sql.replace("$t", t));
    let cases: [(&str, Vec<Vec<Value>>); 9] = [
        ("SELECT a FROM $t ORDER BY _rowid_", vec![vec![int(30)], vec![int(20)], vec![int(10)]]),
        ("SELECT a FROM $t ORDER BY _rowid_ DESC LIMIT 1", vec![vec![int(10)]]),
        ("SELECT a FROM $t ORDER BY $t._rowid_ LIMIT 1 OFFSET 1", vec![vec![int(20)]]),
        ("SELECT a AS id FROM $t ORDER BY _rowid_ LIMIT 1", vec![vec![int(30)]]),
        ("SELECT a FROM $t ORDER BY _rowid_ + 0", vec![vec![int(30)], vec![int(20)], vec![int(10)]]),
        ("SELECT a FROM $t GROUP BY _rowid_, a ORDER BY _rowid_", vec![vec![int(30)], vec![int(20)], vec![int(10)]]),
        (
            "SELECT SUM(a) FROM $t GROUP BY _rowid_ HAVING _rowid_ > (SELECT MIN(_rowid_) FROM $t) ORDER BY _rowid_",
            vec![vec![int(20)], vec![int(10)]],
        ),
        ("SELECT a FROM $t WHERE _rowid_ > 0 ORDER BY _rowid_ DESC", vec![vec![int(10)], vec![int(20)], vec![int(30)]]),
        ("SELECT COUNT(*), SUM(_rowid_) FROM $t", vec![vec![int(3), int(i1 + i2 + i3)]]),
    ];
    for (sql, answer) in cases {
        assert_eq!(q(sql), answer, "{sql} in {t}");
    }
    for sql in [
        "SELECT a FROM {t} WHERE _rowid_ = (SELECT a FROM {t})",
        "DELETE FROM {t} WHERE _rowid_ = (SELECT a FROM {t})",
    ] {
        let sql = sql.replace("{t}", t);
        assert!(error(db, &sql).contains("more than one row"), "{sql}");
    }
    assert_eq!(
        q("SELECT COUNT(*) FROM $t"),
        [[int(3)]],
        "nothing deleted in {t}"
    );
    let by_id = db
        .query(&format!("SELECT a FROM {t} WHERE _rowid_ = $1"), (i2,))
        .unwrap()
        .map(|row| row.unwrap().get_value(0).unwrap().clone())
        .collect::<Vec<_>>();
    assert_eq!(by_id, [int(20)], "a parameter in {t}");
}

#[test]
fn a_statement_shaped_like_one_without_rowid_still_reads_it() {
    let db = tables("memory://rowid_shaped_alike", false);
    assert_eq!(rows(&db, "SELECT SUM(a) FROM k"), [[int(6)]]);
    assert_eq!(rows(&db, "SELECT SUM(_rowid_) FROM k"), [[int(60)]]);
    assert_eq!(rows(&db, "SELECT SUM(a) FROM n"), [[int(6)]]);
    assert_eq!(rows(&db, "SELECT SUM(_rowid_) FROM n"), [[int(6)]]);
}

#[test]
fn rowid_reads_alike_in_every_clause() {
    each_layout("clauses", |db| {
        for sql in [
            "CREATE TABLE kr (id INTEGER PRIMARY KEY, a INTEGER UNIQUE)",
            "CREATE TABLE nr (a INTEGER UNIQUE)",
            "INSERT INTO kr VALUES (1, 30), (2, 20), (3, 10)",
            "INSERT INTO nr VALUES (30), (20), (10)",
        ] {
            db.execute(sql, ()).unwrap();
        }
        clauses(db, "kr", [1, 2, 3]);
        clauses(db, "nr", [1, 2, 3]);
        for t in ["kr", "nr"] {
            assert_eq!(
                rows(
                    db,
                    &format!(
                        "SELECT _rowid_, a AS id FROM {t} UNION ALL SELECT 9, 0 ORDER BY _rowid_"
                    )
                ),
                [
                    [int(1), int(30)],
                    [int(2), int(20)],
                    [int(3), int(10)],
                    [int(9), int(0)]
                ],
                "a set operation's ORDER BY names its output in {t}"
            );
            assert_eq!(
                rows(
                    db,
                    &format!(
                        "SELECT SUM(_rowid_) AS s FROM {t} UNION ALL SELECT 0 ORDER BY SUM(_rowid_)"
                    )
                ),
                [[int(0)], [int(6)]],
                "a set operation's ORDER BY reading an item's expression in {t}"
            );
            assert_eq!(
                rows(
                    db,
                    &format!(
                        "SELECT _rowid_ + 0 FROM {t} UNION ALL SELECT 9 ORDER BY _rowid_ + 0 DESC"
                    )
                ),
                [[int(9)], [int(3)], [int(2)], [int(1)]],
                "a set operation's ORDER BY reading an item's rowid expression in {t}"
            );
            assert_eq!(
                rows(
                    db,
                    &format!(
                        "SELECT 2, 1, _rowid_ FROM {t} UNION ALL SELECT 3, 4, 0 ORDER BY 1, 3"
                    )
                ),
                [
                    [int(2), int(1), int(1)],
                    [int(2), int(1), int(2)],
                    [int(2), int(1), int(3)],
                    [int(3), int(4), int(0)]
                ],
                "a set operation's ORDER BY position beside items alike to it in {t}"
            );
            // A star before the item counts every column it shows
            let other = if t == "kr" { "0, 99, 0" } else { "99, 0" };
            let first = if t == "kr" {
                vec![int(0), int(99), int(0)]
            } else {
                vec![int(99), int(0)]
            };
            for star in ["*", &format!("{t}.*")] {
                let sql = format!(
                    "SELECT {star}, _rowid_ + 0 AS r FROM {t} UNION ALL SELECT {other} ORDER BY _rowid_ + 0"
                );
                let answer = rows(db, &sql);
                assert_eq!(answer[0], first, "{sql}");
                assert_eq!(
                    answer
                        .iter()
                        .map(|row| row.last().cloned())
                        .collect::<Vec<_>>(),
                    [Some(int(0)), Some(int(1)), Some(int(2)), Some(int(3))],
                    "{sql}"
                );
            }
            let sql = format!(
                "SELECT \"Ö\".*, _rowid_ + 0 AS r FROM {t} AS \"Ö\" UNION ALL SELECT {other} ORDER BY _rowid_ + 0"
            );
            assert_eq!(rows(db, &sql)[0], first, "{sql}");
            let sql = format!(
                "SELECT \"ΟΣ\".*, _rowid_ + 0 AS r FROM {t} AS \"ΟΣ\" UNION ALL SELECT {other} ORDER BY _rowid_ + 0"
            );
            assert_eq!(rows(db, &sql)[0], first, "{sql}");
        }
        assert_eq!(
            rows(db, "SELECT _rowid_, _rowid_ + 0 FROM nr WHERE a = 10"),
            [[int(3), int(3)]]
        );
        assert_eq!(
            rows(
                db,
                "SELECT SUM(kr._rowid_), COUNT(*) FROM kr JOIN k ON kr._rowid_ = k.id / 10"
            ),
            [[int(6), int(3)]]
        );
        assert_eq!(
            rows(
                db,
                "SELECT kr.a FROM kr JOIN k ON kr._rowid_ = k.id / 10 ORDER BY kr._rowid_"
            ),
            [[int(30)], [int(20)], [int(10)]]
        );
        assert_eq!(
            rows(
                db,
                "INSERT INTO nr VALUES (10) ON CONFLICT (a) DO UPDATE SET a = 10 RETURNING nr._rowid_"
            ),
            [[int(3)]]
        );
        assert_eq!(rows(db, "SELECT COUNT(*) FROM nr"), [[int(3)]]);
    });
}

#[test]
fn a_set_operation_over_a_join_keeps_the_rowid_of_its_side() {
    each_layout("using", |db| {
        for sql in [
            "CREATE TABLE ua (id INTEGER PRIMARY KEY)",
            "CREATE TABLE ub (id INTEGER PRIMARY KEY)",
            "INSERT INTO ua VALUES (1)",
            "INSERT INTO ub VALUES (2)",
        ] {
            db.execute(sql, ()).unwrap();
        }
        assert_eq!(
            rows(
                db,
                "SELECT ub._rowid_ FROM ua LEFT JOIN ub USING (id) UNION ALL SELECT -1"
            ),
            [[Value::null_unknown()], [int(-1)]]
        );
    });
}

#[test]
fn a_primary_key_named_rowid_or_like_a_literal_reads_as_itself() {
    each_layout("named_key", |db| {
        for sql in [
            "CREATE TABLE p1 (rowid INTEGER PRIMARY KEY)",
            "CREATE TABLE p2 (\"true\" INTEGER PRIMARY KEY)",
            "INSERT INTO p1 VALUES (7)",
            "INSERT INTO p2 VALUES (8)",
        ] {
            db.execute(sql, ()).unwrap();
        }
        assert_eq!(
            rows(db, "SELECT SUM(_rowid_), rowid FROM p1 GROUP BY rowid"),
            [[int(7), int(7)]]
        );
        assert_eq!(
            rows(db, "SELECT CAST(_rowid_ AS INTEGER), p2._rowid_ FROM p2"),
            [[int(8), int(8)]]
        );
        assert!(error(db, "SELECT SUM(_rowid_) FROM p2").contains("not found"));
        assert_eq!(
            rows(
                db,
                "INSERT INTO p2 VALUES (9) RETURNING CAST(_rowid_ AS INTEGER)"
            ),
            [[int(9)]]
        );
    });
}

#[test]
fn a_nearest_read_by_an_index_carries_the_rowid() {
    each_layout("nearest", |db| {
        for sql in [
            "CREATE TABLE h (v VECTOR(2))",
            "INSERT INTO h VALUES ('[3,3]'), ('[1,1]')",
            "CREATE INDEX h_v ON h (v) USING HNSW",
        ] {
            db.execute(sql, ()).unwrap();
        }
        assert_eq!(
            rows(
                db,
                "SELECT _rowid_ FROM h WHERE _rowid_ > 0 ORDER BY VEC_DISTANCE_L2(v, '[0,0]') LIMIT 1"
            ),
            [[int(2)]]
        );
    });
}

#[test]
fn an_index_ordered_read_carries_the_rowid() {
    each_layout("ordered", |db| {
        for sql in [
            "CREATE TABLE o (a INTEGER)",
            "INSERT INTO o VALUES (20), (10)",
            "CREATE INDEX o_a ON o (a)",
        ] {
            db.execute(sql, ()).unwrap();
        }
        assert_eq!(
            rows(db, "SELECT _rowid_ FROM o ORDER BY a LIMIT 1"),
            [[int(2)]]
        );
        assert_eq!(
            rows(db, "SELECT _rowid_, a FROM o ORDER BY a DESC LIMIT 1"),
            [[int(1), int(20)]]
        );
        // The ids order the rows against their values here
        assert_eq!(
            rows(db, "SELECT a FROM o ORDER BY (SELECT o._rowid_) DESC"),
            [[int(10)], [int(20)]]
        );
    });
}

#[test]
fn a_returned_row_carries_its_rowid() {
    each_layout("returned", |db| {
        assert_eq!(
            names(db, "INSERT INTO n VALUES (4, 'w') RETURNING *, _rowid_"),
            ["a", "b", "_rowid_"]
        );
        assert_eq!(
            rows(
                db,
                "INSERT INTO n VALUES (5, 'v') RETURNING _rowid_, (SELECT 1)"
            ),
            [[int(5), int(1)]]
        );
        let mut deleted = rows(db, "DELETE FROM n RETURNING _rowid_");
        deleted.sort_by_key(|row| row[0].as_int64());
        assert_eq!(deleted, [[int(1)], [int(2)], [int(3)], [int(4)], [int(5)]]);
    });
}

#[test]
fn rowid_reads_the_primary_key_of_a_keyed_table() {
    each_layout("keyed", |db| reads(db, "k", [10, 20, 30]));
}

#[test]
fn rowid_reads_the_assigned_id_of_a_table_without_a_key() {
    each_layout("keyless", |db| reads(db, "n", [1, 2, 3]));
}

#[test]
fn a_column_named_rowid_is_an_ordinary_column() {
    each_layout("plain_rowid", |db| {
        assert_eq!(
            rows(db, "SELECT rowid, _rowid_ FROM r ORDER BY a"),
            [[int(100), int(1)], [int(200), int(2)]]
        );
        db.execute("UPDATE r SET rowid = 300 WHERE a = 1", ())
            .unwrap();
        assert_eq!(
            rows(db, "SELECT a AS rowid FROM k ORDER BY rowid DESC"),
            [[int(3)], [int(2)], [int(1)]]
        );
        assert!(error(db, "SELECT rowid FROM n").contains("not found"));
    });
}

#[test]
fn no_column_and_no_alias_takes_the_name_of_the_row_identity() {
    let db = tables("memory://rowid_reserved", false);
    for sql in [
        "CREATE TABLE x (_rowid_ INTEGER)",
        "CREATE TABLE x (a INTEGER, _ROWID_ TEXT)",
        "ALTER TABLE n ADD COLUMN _rowid_ INTEGER",
        "ALTER TABLE n RENAME COLUMN a TO _rowid_",
        "CREATE TABLE x AS SELECT _rowid_ FROM n",
        "SELECT a AS _rowid_ FROM n",
        "SELECT a _rowid_ FROM n",
        "WITH c (_rowid_) AS (SELECT 1) SELECT * FROM c",
        "SELECT * FROM (VALUES (1)) v (_rowid_)",
        "SELECT (SELECT a AS _rowid_ FROM n WHERE a = 1)",
        "UPDATE n SET a = a RETURNING a AS _rowid_",
        "DELETE FROM n WHERE a = 0 RETURNING a AS _rowid_",
        "INSERT INTO n VALUES (9, 'q') RETURNING a AS _rowid_",
    ] {
        let message = match db.execute(sql, ()) {
            Ok(_) => panic!("{sql}: no error"),
            Err(e) => e.to_string(),
        };
        assert!(message.contains("row's identity"), "{sql}: {message}");
    }
    assert_eq!(rows(&db, "SELECT COUNT(*) FROM n"), [[int(3)]]);
    // The storage API changes a table's columns by the same rule
    use stoolap::storage::traits::Engine;
    let mut tx = db.engine().begin_transaction().unwrap();
    let column = stoolap::core::SchemaColumn::new(
        2,
        "_rowid_",
        stoolap::core::DataType::Integer,
        true,
        false,
    );
    for refused in [
        tx.add_table_column("n", column),
        tx.rename_table_column("n", "a", "_rowid_"),
    ] {
        assert!(refused.unwrap_err().to_string().contains("row's identity"));
    }
    tx.rollback().unwrap();
    assert_eq!(names(&db, "SELECT * FROM n"), ["a", "b"]);
}

#[test]
fn rowid_is_read_by_writes_and_never_written() {
    each_layout("writes", |db| {
        for (t, i2, i3) in [("k", 20, 30), ("n", 2, 3)] {
            assert_eq!(
                rows(
                    db,
                    &format!("UPDATE {t} SET a = a + 10 WHERE _rowid_ = {i2} RETURNING _rowid_, a")
                ),
                [[int(i2), int(12)]],
                "UPDATE in {t}"
            );
            assert_eq!(
                rows(
                    db,
                    &format!("UPDATE {t} SET a = _rowid_ WHERE a = 1 RETURNING a")
                ),
                [[int(if t == "k" { 10 } else { 1 })]],
                "SET reading rowid in {t}"
            );
            assert_eq!(
                rows(
                    db,
                    &format!("DELETE FROM {t} WHERE _rowid_ = {i3} RETURNING _rowid_, b")
                ),
                [[int(i3), text("z")]],
                "DELETE in {t}"
            );
            for sql in [
                format!("UPDATE {t} SET _rowid_ = 5"),
                format!("INSERT INTO {t} (_rowid_, a) VALUES (5, 5)"),
            ] {
                assert!(
                    db.execute(&sql, ())
                        .unwrap_err()
                        .to_string()
                        .contains("row's identity"),
                    "{sql}"
                );
            }
        }
        assert_eq!(
            rows(db, "INSERT INTO k VALUES (40, 4, 'w') RETURNING _rowid_"),
            [[int(40)]]
        );
        assert_eq!(
            rows(
                db,
                "INSERT INTO n VALUES (4, 'w'), (5, 'v') RETURNING _rowid_, a"
            ),
            [[int(4), int(4)], [int(5), int(5)]]
        );
        assert_eq!(
            rows(db, "SELECT _rowid_, a FROM n ORDER BY _rowid_"),
            [
                [int(1), int(1)],
                [int(2), int(12)],
                [int(4), int(4)],
                [int(5), int(5)]
            ]
        );
    });
}

#[test]
fn a_join_reads_a_keyed_side_rowid_and_refuses_a_keyless_one() {
    each_layout("join", |db| {
        assert_eq!(
            rows(
                db,
                "SELECT k._rowid_, n.a FROM k JOIN n ON k.a = n.a ORDER BY 1"
            ),
            [[int(10), int(1)], [int(20), int(2)], [int(30), int(3)]]
        );
        assert!(error(db, "SELECT n._rowid_ FROM k JOIN n ON k.a = n.a").contains("not found"));
        assert!(error(db, "SELECT _rowid_ FROM k JOIN n ON k.a = n.a").contains("ambiguous"));
        assert!(error(db, "SELECT _rowid_ FROM k x JOIN k y ON x.a = y.a").contains("ambiguous"));
    });
}

#[test]
fn a_bare_rowid_counts_a_derived_table_column_with_the_tables_beside_it() {
    each_layout("derived", |db| {
        for sql in [
            "SELECT _rowid_ FROM k CROSS JOIN (SELECT _rowid_ FROM n) d",
            "SELECT (SELECT _rowid_) FROM k CROSS JOIN (SELECT _rowid_ FROM n) d",
            "WITH c AS (SELECT _rowid_ FROM n) SELECT _rowid_ FROM k CROSS JOIN c",
            "SELECT _rowid_ FROM (SELECT x._rowid_, y._rowid_ FROM k x, k y) d",
        ] {
            assert!(error(db, sql).contains("ambiguous"), "{sql}");
        }
        for sql in [
            "SELECT _rowid_ FROM k CROSS JOIN (SELECT a AS x FROM n WHERE a = 1) d ORDER BY 1",
            "SELECT (SELECT _rowid_) FROM k CROSS JOIN (SELECT a AS x FROM n WHERE a = 1) d ORDER BY 1",
            "SELECT _rowid_ FROM k CROSS JOIN (VALUES (7)) v (x) ORDER BY 1",
        ] {
            assert_eq!(
                rows(db, sql),
                [[int(10)], [int(20)], [int(30)]],
                "{sql}"
            );
        }
        assert_eq!(
            rows(
                db,
                "SELECT d._rowid_, k._rowid_ FROM k CROSS JOIN (SELECT _rowid_ FROM n WHERE a = 2) d WHERE k.a = 1"
            ),
            [[int(2), int(10)]]
        );
        assert_eq!(
            rows(
                db,
                "SELECT SUM((SELECT _rowid_)), COUNT(*) FROM (SELECT _rowid_ FROM n) d"
            ),
            [[int(6), int(3)]]
        );
    });
}

#[test]
fn an_upsert_reads_the_rowid_of_the_row_met_and_never_writes_it() {
    each_layout("upsert", |db| {
        let upsert = "INSERT INTO k VALUES (10, 5, 'q') ON CONFLICT (id) DO UPDATE SET";
        let sql = format!("{upsert} _rowid_ = 2, a = 99");
        let refused = db.execute(&sql, ()).unwrap_err().to_string();
        assert!(refused.contains("row's identity"), "{sql}: {refused}");
        for (set, a) in [
            ("a = _rowid_ + 1 WHERE _rowid_ = 10", 11),
            ("a = (SELECT _rowid_) + 2 WHERE (SELECT _rowid_) = 10", 12),
            ("a = k._rowid_ + 3", 13),
            ("a = 0 WHERE _rowid_ = 20", 13),
        ] {
            db.execute(&format!("{upsert} {set}"), ())
                .unwrap_or_else(|e| panic!("{set}: {e}"));
            assert_eq!(
                rows(db, "SELECT id, a FROM k WHERE id = 10"),
                [[int(10), int(a)]],
                "{set}"
            );
        }
        let sql = format!("{upsert} a = excluded._rowid_");
        let refused = db.execute(&sql, ()).unwrap_err().to_string();
        assert!(refused.contains("not found"), "{sql}: {refused}");
        for sql in [
            "CREATE TABLE u (k INTEGER UNIQUE, v INTEGER)",
            "INSERT INTO u VALUES (5, 0), (6, 0)",
            "CREATE TABLE p (\"true\" INTEGER PRIMARY KEY, v INTEGER)",
            "INSERT INTO p VALUES (7, 0)",
        ] {
            db.execute(sql, ()).unwrap();
        }
        for (t, upsert, id) in [
            (
                "u",
                "INSERT INTO u VALUES (6, 1) ON CONFLICT (k) DO UPDATE SET",
                2,
            ),
            (
                "p",
                "INSERT INTO p VALUES (7, 1) ON CONFLICT (\"true\") DO UPDATE SET",
                7,
            ),
        ] {
            for (set, v) in [
                (format!("v = _rowid_ + 1 WHERE _rowid_ = {id}"), id + 1),
                (
                    format!("v = (SELECT _rowid_) + 2 WHERE (SELECT _rowid_) = {id}"),
                    id + 2,
                ),
                (String::from("v = 0 WHERE _rowid_ = 99"), id + 2),
            ] {
                db.execute(&format!("{upsert} {set}"), ())
                    .unwrap_or_else(|e| panic!("{t} {set}: {e}"));
                assert_eq!(
                    rows(db, &format!("SELECT v FROM {t} WHERE _rowid_ = {id}")),
                    [[int(v)]],
                    "{t} {set}"
                );
            }
            for (set, v) in [
                (format!("v = {t}._rowid_ + 3"), id + 3),
                (String::from("v = (SELECT _rowid_) + 4"), id + 4),
            ] {
                assert_eq!(
                    rows(db, &format!("{upsert} {set} RETURNING _rowid_, v")),
                    [[int(id), int(v)]],
                    "{t} {set}"
                );
            }
        }
    });
}

#[test]
fn distinct_matches_an_order_by_rowid_to_the_list_as_written() {
    each_layout("distinct", |db| {
        assert_eq!(
            rows(db, "SELECT DISTINCT _rowid_ FROM k ORDER BY _rowid_ DESC"),
            [[int(30)], [int(20)], [int(10)]]
        );
        let sql = "SELECT DISTINCT id FROM k ORDER BY _rowid_";
        assert!(error(db, sql).contains("SELECT DISTINCT list"), "{sql}");
    });
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_keyless_table_reads_the_rows_its_rowid_names_by_id() {
    use stoolap::test_failpoints::take_read_paths;
    let dir = tempfile::tempdir().unwrap();
    let db = tables(&format!("file://{}", dir.path().display()), true);
    for sql in [
        "SELECT a FROM n WHERE _rowid_ = 2",
        "SELECT a FROM n WHERE _rowid_ IN (1, 3) ORDER BY a",
    ] {
        take_read_paths();
        assert!(!rows(&db, sql).is_empty(), "{sql}");
        let paths = take_read_paths();
        assert!(
            paths.contains(&"seg.fetch_rows_by_ids_into")
                && !paths.contains(&"seg.scan_cold_and_hot"),
            "{sql} read {paths:?}"
        );
    }
}

#[test]
fn a_keyless_table_keeps_its_row_ids_across_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().display());
    {
        let db = tables(&dsn, true);
        db.execute("DELETE FROM n WHERE a = 1", ()).unwrap();
        db.execute("INSERT INTO n VALUES (4, 'w')", ()).unwrap();
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    assert_eq!(
        rows(&db, "SELECT _rowid_, a FROM n ORDER BY _rowid_"),
        [[int(2), int(2)], [int(3), int(3)], [int(4), int(4)]]
    );
    db.execute("INSERT INTO n VALUES (5, 'v')", ()).unwrap();
    assert_eq!(rows(&db, "SELECT _rowid_ FROM n WHERE a = 5"), [[int(5)]]);
}
