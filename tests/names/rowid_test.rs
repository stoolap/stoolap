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

//! `rowid` and `_rowid_` read a row's identity: the primary key's value in a
//! table with one, the id the engine assigned in a table without one.

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
/// holding three rows, and `r`, whose own column is named rowid; sealed into
/// volumes when `sealed`
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
    assert_eq!(
        q("SELECT rowid, a FROM $t ORDER BY rowid"),
        [[int(i1), int(1)], [int(i2), int(2)], [int(i3), int(3)]],
        "select list in {t}"
    );
    assert_eq!(
        names(db, &format!("SELECT rowid, _rowid_, {t}.rowid FROM {t}")),
        ["rowid", "_rowid_", "rowid"],
        "headers in {t}"
    );
    assert_eq!(
        q(&format!("SELECT b FROM $t WHERE rowid = {i2}")),
        [[text("y")]],
        "equality in {t}"
    );
    assert_eq!(
        q(&format!(
            "SELECT a FROM $t WHERE $t.rowid IN ({i3}, {i1}, NULL) ORDER BY a"
        )),
        [[int(1)], [int(3)]],
        "IN in {t}"
    );
    assert_eq!(
        q(&format!(
            "SELECT rowid FROM $t WHERE rowid > {i1} ORDER BY rowid DESC LIMIT 1"
        )),
        [[int(i3)]],
        "range in {t}"
    );
    assert_eq!(
        q(&format!("SELECT a FROM $t WHERE rowid = {i2} AND a = 9")),
        Vec::<Vec<Value>>::new(),
        "equality with another conjunct in {t}"
    );
    let by_param = db
        .query(&format!("SELECT a FROM {t} WHERE rowid = $1"), (i3,))
        .unwrap()
        .map(|row| row.unwrap().get_value(0).unwrap().clone())
        .collect::<Vec<_>>();
    assert_eq!(by_param, [int(3)], "parameter in {t}");
    assert_eq!(
        q("SELECT COUNT(*), MAX(rowid), SUM(rowid) FROM $t"),
        [[int(3), int(i3), int(i1 + i2 + i3)]],
        "aggregates in {t}"
    );
    assert_eq!(
        q(&format!(
            "SELECT rowid, COUNT(*) FROM $t GROUP BY rowid HAVING rowid > {i1} ORDER BY rowid"
        )),
        [[int(i2), int(1)], [int(i3), int(1)]],
        "GROUP BY and HAVING in {t}"
    );
    assert_eq!(
        q(&format!(
            "SELECT CASE WHEN rowid = {i2} THEN 'hit' ELSE 'miss' END FROM $t ORDER BY a"
        )),
        [[text("miss")], [text("hit")], [text("miss")]],
        "CASE in {t}"
    );
    assert_eq!(
        q("SELECT a, ROW_NUMBER() OVER (ORDER BY rowid DESC) FROM $t ORDER BY a"),
        [[int(1), int(3)], [int(2), int(2)], [int(3), int(1)]],
        "window in {t}"
    );
    assert_eq!(
        q("SELECT (SELECT COUNT(*) FROM $t i WHERE i.rowid <= o.rowid) FROM $t o ORDER BY o.a"),
        [[int(1)], [int(2)], [int(3)]],
        "correlated subquery in {t}"
    );
    assert_eq!(
        q("SELECT a FROM $t o WHERE EXISTS (SELECT 1 FROM $t i WHERE i.rowid > o.rowid) ORDER BY a"),
        [[int(1)], [int(2)]],
        "EXISTS in {t}"
    );
    assert_eq!(
        q("SELECT a FROM $t WHERE rowid IN (SELECT rowid FROM $t WHERE a > 1) ORDER BY a"),
        [[int(2)], [int(3)]],
        "IN subquery in {t}"
    );
    assert_eq!(
        q("SELECT (SELECT i.rowid FROM $t i ORDER BY i.rowid DESC LIMIT 1)"),
        [[int(i3)]],
        "a subquery run as a statement in {t}"
    );
    assert_eq!(
        q("SELECT (SELECT rowid FROM $t WHERE a = 2 UNION SELECT -1 ORDER BY 1 DESC LIMIT 1)"),
        [[int(i2)]],
        "a set operation inside a subquery in {t}"
    );
    assert_eq!(
        q("SELECT a FROM $t o WHERE EXISTS (SELECT DISTINCT i.rowid FROM $t i WHERE i.rowid = o.rowid AND i.a > 1) ORDER BY a"),
        [[int(2)], [int(3)]],
        "EXISTS correlated by rowid in {t}"
    );
    assert_eq!(
        q("SELECT rowid FROM $t WHERE a = 1 UNION ALL SELECT rowid FROM $t WHERE a = 3"),
        [[int(i1)], [int(i3)]],
        "set operation in {t}"
    );
    assert_eq!(
        q("SELECT * FROM (SELECT rowid FROM $t) d ORDER BY 1"),
        [[int(i1)], [int(i2)], [int(i3)]],
        "a derived table's output column in {t}"
    );
    assert_eq!(
        q(&format!("SELECT * FROM $t WHERE rowid = {i1}"))[0].len(),
        if t == "k" { 3 } else { 2 },
        "a star leaves rowid out in {t}"
    );
    assert_eq!(
        q("SELECT a AS rowid FROM $t ORDER BY rowid DESC"),
        [[int(3)], [int(2)], [int(1)]],
        "an alias wins in ORDER BY in {t}"
    );
    assert_eq!(
        q("SELECT \"rowid\" FROM $t WHERE a = 1"),
        [[text("rowid")]],
        "a quoted name is no rowid in {t}"
    );
    assert!(
        error(db, &format!("SELECT rowid FROM (SELECT * FROM {t}) d")).contains("not found"),
        "a derived table has no rowid in {t}"
    );
    assert_eq!(
        q("SELECT *, rowid AS r FROM $t ORDER BY r + 0 DESC")[0].last(),
        Some(&int(i3)),
        "an alias after a star in {t}"
    );
    assert_eq!(
        q("SELECT MAX(rowid) AS rowid FROM $t"),
        [[int(i3)]],
        "an alias over its own rowid in {t}"
    );
    assert_eq!(
        q("SELECT rowid, COUNT(*) FROM $t GROUP BY GROUPING SETS ((rowid), ()) ORDER BY 1"),
        [
            [int(i1), int(1)],
            [int(i2), int(1)],
            [int(i3), int(1)],
            [Value::null_unknown(), int(3)]
        ],
        "GROUPING SETS in {t}"
    );
    assert_eq!(
        q("SELECT a, SUM(a) OVER w FROM $t WINDOW w AS (ORDER BY rowid) ORDER BY a"),
        [[int(1), int(1)], [int(2), int(3)], [int(3), int(6)]],
        "a named window in {t}"
    );
    assert_eq!(
        q("SELECT SUM((SELECT o.rowid)) FROM $t o"),
        [[int(i1 + i2 + i3)]],
        "a subquery inside an aggregate in {t}"
    );
    assert_eq!(
        q("SELECT (SELECT x FROM (VALUES (o.rowid)) v(x)) FROM $t o ORDER BY o.a"),
        [[int(i1)], [int(i2)], [int(i3)]],
        "a VALUES row reading the outer rowid in {t}"
    );
    assert_eq!(
        q("SELECT a FROM $t o ORDER BY (SELECT o.rowid) DESC"),
        [[int(3)], [int(2)], [int(1)]],
        "an ORDER BY key reading rowid through a subquery in {t}"
    );
    assert_eq!(
        q("SELECT SUM(rowid) OVER () + 0, (SELECT 1) FROM $t"),
        [
            [int(i1 + i2 + i3), int(1)],
            [int(i1 + i2 + i3), int(1)],
            [int(i1 + i2 + i3), int(1)]
        ],
        "a grouped window over rowid in {t}"
    );
    assert_eq!(
        q("SELECT *, SUM(rowid) OVER () FROM $t WHERE a = 1")[0].len(),
        if t == "k" { 4 } else { 3 },
        "a star beside a window leaves rowid out in {t}"
    );
    assert_eq!(
        q("SELECT MAX(rowid) AS rowid FROM $t HAVING MAX(rowid) > 0"),
        [[int(i3)]],
        "an alias named rowid in HAVING's aggregate in {t}"
    );
    assert!(
        error(db, &format!("SELECT STRING_AGG(b, rowid) FROM {t}"))
            .contains("must not read the row"),
        "rowid as an aggregate's later argument reads the row in {t}"
    );
    let lagged = db
        .query(
            &format!("SELECT *, LAG(rowid, $1) OVER (ORDER BY rowid) FROM {t} ORDER BY a"),
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
        q("SELECT a AS rowid, SUM(rowid) OVER () FROM $t ORDER BY 1"),
        [
            [int(1), int(i1 + i2 + i3)],
            [int(2), int(i1 + i2 + i3)],
            [int(3), int(i1 + i2 + i3)]
        ],
        "an alias named rowid beside a window reading rowid in {t}"
    );
    assert!(
        db.query(
            &format!("SELECT a AS rowid FROM {t} ORDER BY SUM(rowid) OVER ()"),
            ()
        )
        .is_err(),
        "a window reading rowid in ORDER BY is refused in {t}"
    );
    assert_eq!(
        q("SELECT a AS rowid, COUNT(*) FROM $t GROUP BY _rowid_, a ORDER BY 1"),
        [[int(1), int(1)], [int(2), int(1)], [int(3), int(1)]],
        "an alias named rowid beside _rowid_ in {t}"
    );
}

#[test]
fn a_primary_key_named_like_a_pseudo_column_or_a_literal_reads_as_itself() {
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
            rows(db, "SELECT CAST(rowid AS INTEGER), p2.rowid FROM p2"),
            [[int(8), int(8)]]
        );
        for sql in [
            "SELECT SUM(rowid) FROM p2",
            "INSERT INTO p2 VALUES (9) RETURNING CAST(rowid AS INTEGER)",
        ] {
            assert!(error(db, sql).contains("not found"), "{sql}");
        }
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
                "SELECT rowid FROM h WHERE rowid > 0 ORDER BY VEC_DISTANCE_L2(v, '[0,0]') LIMIT 1"
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
            rows(db, "SELECT rowid FROM o ORDER BY a LIMIT 1"),
            [[int(2)]]
        );
        assert_eq!(
            rows(db, "SELECT rowid, a FROM o ORDER BY a DESC LIMIT 1"),
            [[int(1), int(20)]]
        );
        // The ids order the rows against their values here
        assert_eq!(
            rows(db, "SELECT a FROM o ORDER BY (SELECT o.rowid) DESC"),
            [[int(10)], [int(20)]]
        );
    });
}

#[test]
fn a_returned_row_carries_its_rowid() {
    each_layout("returned", |db| {
        assert_eq!(
            names(db, "INSERT INTO r VALUES (300, 3) RETURNING *, rowid"),
            ["rowid", "a", "rowid"]
        );
        assert_eq!(
            rows(db, "INSERT INTO r VALUES (400, 4) RETURNING *, rowid"),
            [[int(400), int(4), int(400)]]
        );
        assert_eq!(
            rows(
                db,
                "INSERT INTO n VALUES (4, 'w') RETURNING rowid, (SELECT 1)"
            ),
            [[int(4), int(1)]]
        );
        let mut deleted = rows(db, "DELETE FROM n RETURNING rowid");
        deleted.sort_by_key(|row| row[0].as_int64());
        assert_eq!(deleted, [[int(1)], [int(2)], [int(3)], [int(4)]]);
        db.execute("CREATE TABLE q (\"true\" INTEGER PRIMARY KEY)", ())
            .unwrap();
        db.execute("INSERT INTO q VALUES (7)", ()).unwrap();
        assert_eq!(rows(db, "SELECT CAST(rowid AS INTEGER) FROM q"), [[int(7)]]);
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
fn a_column_named_rowid_wins_over_the_row_identity() {
    each_layout("real_column", |db| {
        assert_eq!(
            rows(db, "SELECT rowid, _rowid_ FROM r ORDER BY a"),
            [[int(100), int(1)], [int(200), int(2)]]
        );
        db.execute("UPDATE r SET rowid = 300 WHERE a = 1", ())
            .unwrap();
        assert_eq!(
            rows(db, "SELECT rowid FROM r ORDER BY a"),
            [[int(300)], [int(200)]]
        );
    });
}

#[test]
fn rowid_is_read_by_writes_and_never_written() {
    each_layout("writes", |db| {
        for (t, i2, i3) in [("k", 20, 30), ("n", 2, 3)] {
            assert_eq!(
                rows(
                    db,
                    &format!("UPDATE {t} SET a = a + 10 WHERE rowid = {i2} RETURNING rowid, a")
                ),
                [[int(i2), int(12)]],
                "UPDATE in {t}"
            );
            assert_eq!(
                rows(
                    db,
                    &format!("UPDATE {t} SET a = rowid WHERE a = 1 RETURNING a")
                ),
                [[int(if t == "k" { 10 } else { 1 })]],
                "SET reading rowid in {t}"
            );
            assert_eq!(
                rows(
                    db,
                    &format!("DELETE FROM {t} WHERE rowid = {i3} RETURNING rowid, b")
                ),
                [[int(i3), text("z")]],
                "DELETE in {t}"
            );
            for sql in [
                format!("UPDATE {t} SET rowid = 5"),
                format!("UPDATE {t} SET _rowid_ = 5"),
                format!("INSERT INTO {t} (rowid, a) VALUES (5, 5)"),
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
            rows(db, "INSERT INTO k VALUES (40, 4, 'w') RETURNING rowid"),
            [[int(40)]]
        );
        assert_eq!(
            rows(
                db,
                "INSERT INTO n VALUES (4, 'w'), (5, 'v') RETURNING rowid, a"
            ),
            [[int(4), int(4)], [int(5), int(5)]]
        );
        assert_eq!(
            rows(db, "SELECT rowid, a FROM n ORDER BY rowid"),
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
                "SELECT k.rowid, n.a FROM k JOIN n ON k.a = n.a ORDER BY 1"
            ),
            [[int(10), int(1)], [int(20), int(2)], [int(30), int(3)]]
        );
        assert!(error(db, "SELECT n.rowid FROM k JOIN n ON k.a = n.a").contains("not found"));
        assert!(error(db, "SELECT rowid FROM k JOIN n ON k.a = n.a").contains("ambiguous"));
        assert!(error(db, "SELECT rowid FROM k x JOIN k y ON x.a = y.a").contains("ambiguous"));
    });
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_keyless_table_reads_the_rows_its_rowid_names_by_id() {
    use stoolap::test_failpoints::take_read_paths;
    let dir = tempfile::tempdir().unwrap();
    let db = tables(&format!("file://{}", dir.path().display()), true);
    for sql in [
        "SELECT a FROM n WHERE rowid = 2",
        "SELECT a FROM n WHERE rowid IN (1, 3) ORDER BY a",
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
        rows(&db, "SELECT rowid, a FROM n ORDER BY rowid"),
        [[int(2), int(2)], [int(3), int(3)], [int(4), int(4)]]
    );
    db.execute("INSERT INTO n VALUES (5, 'v')", ()).unwrap();
    assert_eq!(rows(&db, "SELECT rowid FROM n WHERE a = 5"), [[int(5)]]);
}
