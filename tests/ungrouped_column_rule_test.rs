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

//! A query that aggregates reads a column of its rows only through a
//! GROUP BY item or an aggregate call; any other read of one is an error,
//! and so is a GROUP BY position the SELECT list does not have.

use stoolap::Database;

fn open(db: Database, seal: bool) -> Database {
    for sql in [
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "INSERT INTO u VALUES (1, 1, 1), (2, 1, 10), (3, 2, NULL), (4, NULL, 4), (5, 4, 4)",
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, x INTEGER)",
        "INSERT INTO t VALUES (1, 1, 1), (2, 2, 2), (4, 4, 4)",
        "CREATE TABLE e (id INTEGER PRIMARY KEY, s TEXT, w TEXT)",
        "INSERT INTO e VALUES (1, 'a', '!'), (2, 'b', '!')",
        "CREATE TABLE q (id INTEGER PRIMARY KEY, \"a.b\" INTEGER)",
        "INSERT INTO q VALUES (1, 5), (2, 6)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if seal {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn answer(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len())
                .map(|i| format!("{}", row.get_value(i).unwrap()))
                .collect::<Vec<_>>()
                .join("|"),
        );
    }
    Ok(out)
}

/// Statements reading `v` outside every GROUP BY item and aggregate call
const REFUSED: &[&str] = &[
    "SELECT v, COUNT(*) FROM u GROUP BY k",
    "SELECT u.v, COUNT(*) FROM u GROUP BY k",
    "SELECT \"v\", COUNT(*) FROM u GROUP BY k",
    "SELECT \"v\" || 'x', COUNT(*) FROM u GROUP BY k",
    "SELECT s LIKE 'a' ESCAPE \"w\", COUNT(*) FROM e GROUP BY s",
    "SELECT \"a.b\", COUNT(*) FROM q",
    "SELECT k, COUNT(*) FROM u GROUP BY k HAVING \"v\" > 1",
    "SELECT k, COUNT(*) FROM u GROUP BY k ORDER BY \"v\"",
    "SELECT v AS w, COUNT(*) FROM u GROUP BY k",
    "SELECT v + 1, COUNT(*) FROM u GROUP BY k",
    "SELECT CASE WHEN v > 1 THEN 1 END, COUNT(*) FROM u GROUP BY k",
    "SELECT ABS(v), COUNT(*) FROM u GROUP BY k",
    "SELECT k, COUNT(*) FROM u GROUP BY k HAVING v > 1",
    "SELECT k, COUNT(*) FROM u GROUP BY k ORDER BY v",
    "SELECT v, COUNT(*) FROM u",
    "SELECT v, MAX(k) FROM u WHERE id = 1",
    "SELECT v FROM u GROUP BY k",
    "SELECT DISTINCT v FROM u GROUP BY k",
    "SELECT k, SUM(v) OVER () FROM u GROUP BY k",
    "SELECT COUNT(*) FROM u GROUP BY 0",
    "SELECT COUNT(*) FROM u GROUP BY 2",
    "SELECT id FROM t WHERE x = (SELECT v FROM u WHERE u.k = t.k GROUP BY k)",
    "SELECT id FROM t WHERE x = (SELECT v FROM u WHERE u.k = t.k HAVING COUNT(*) > 0)",
    "SELECT id FROM t WHERE x IN (SELECT v FROM u GROUP BY k)",
    "SELECT id FROM t WHERE EXISTS (SELECT v FROM u WHERE u.k = t.k GROUP BY k)",
    "SELECT id FROM t WHERE x = (SELECT MAX(v) FROM u WHERE u.k = t.k HAVING v > 0)",
    "SELECT id, (SELECT v FROM u WHERE u.k = t.k GROUP BY k) FROM t",
    "SELECT id, EXISTS (SELECT v FROM u WHERE u.k = t.k GROUP BY k) FROM t",
    "SELECT id FROM t WHERE x = (SELECT COUNT(*) FROM u WHERE u.k = t.k GROUP BY 3)",
];

/// Statements reading `u.v` outside its groups inside a subquery, which
/// fail as the same read outside one does
const SUBQUERY_READS: &[&str] = &[
    "SELECT k, (SELECT COUNT(*) FROM t WHERE t.id = u.v) FROM u GROUP BY k",
    "SELECT k FROM u GROUP BY k HAVING (SELECT COUNT(*) FROM t WHERE t.id = u.v) > 0",
    "SELECT k FROM u GROUP BY k ORDER BY (SELECT COUNT(*) FROM t WHERE t.id = u.v)",
    "SELECT k FROM u GROUP BY k ORDER BY (SELECT (SELECT u.v) FROM t WHERE t.id = 1)",
    // A block aliasing u hides its name, so `u.v` is still the outer row's
    "SELECT k, (SELECT u2.v FROM u u2 WHERE u2.id = u.v) FROM u GROUP BY k",
    "SELECT k FROM u GROUP BY k HAVING (SELECT COUNT(*) FROM u u2 WHERE u2.id = u.v) > 0",
    "SELECT k FROM u GROUP BY k ORDER BY (SELECT u2.v FROM u u2 WHERE u2.id = u.v), k",
    "SELECT id FROM t WHERE EXISTS (SELECT k FROM u WHERE u.k = t.k GROUP BY k \
     ORDER BY (SELECT u2.v FROM u u2 WHERE u2.id = u.v))",
    "SELECT k, (SELECT COUNT(*) FROM t WHERE t.id = u.v + 1) FROM u GROUP BY k",
    "SELECT DISTINCT ON (k) k FROM u GROUP BY k ORDER BY k, (SELECT COUNT(*) FROM t WHERE t.id = u.v)",
    "SELECT COUNT(*) FROM u ORDER BY (SELECT COUNT(*) FROM t WHERE t.id = u.v)",
    "SELECT t.k FROM t JOIN u ON u.k = t.k GROUP BY t.k ORDER BY (SELECT COUNT(*) FROM e WHERE e.id = u.v)",
    "SELECT id FROM t WHERE EXISTS (SELECT k FROM u WHERE u.k = t.k GROUP BY k \
     ORDER BY (SELECT COUNT(*) FROM e WHERE e.id = u.v))",
    "SELECT id FROM t WHERE EXISTS (SELECT k, (SELECT COUNT(*) FROM e WHERE e.id = u.v) \
     FROM u WHERE u.k = t.k GROUP BY k)",
    "SELECT id FROM t WHERE EXISTS (SELECT k, (SELECT MAX(u.v)) FROM u WHERE u.k = t.k GROUP BY k)",
    "SELECT id FROM t WHERE EXISTS (SELECT k, (SELECT (SELECT u.v)) FROM u WHERE u.k = t.k GROUP BY k)",
    "SELECT id FROM t WHERE EXISTS (SELECT u.k, (SELECT u.v) AS bad, (SELECT 1) FROM u GROUP BY u.k)",
    "SELECT id FROM t WHERE EXISTS (SELECT k, (SELECT u.v) AS bad FROM u WHERE u.k = t.k GROUP BY k)",
    "SELECT id FROM t WHERE EXISTS (SELECT k, (SELECT u.v) > ANY (SELECT 1) FROM u GROUP BY k)",
    "SELECT id FROM t WHERE EXISTS (SELECT k, 1 > ANY (SELECT u.v) FROM u GROUP BY k)",
    "SELECT id FROM t WHERE EXISTS (SELECT k, SUM(v) + (SELECT u.v) FROM u GROUP BY k)",
    "SELECT id FROM t WHERE EXISTS (SELECT COUNT(*), (SELECT COUNT(*) FROM e WHERE e.id = u.v) \
     FROM u WHERE u.k = t.k)",
    "SELECT id FROM t WHERE EXISTS (SELECT k FROM u WHERE u.k = t.k GROUP BY k \
     HAVING (SELECT COUNT(*) FROM e WHERE e.id = u.v) >= 0)",
    "SELECT id FROM t WHERE k IN (SELECT k FROM u GROUP BY k \
     ORDER BY (SELECT COUNT(*) FROM e WHERE e.id = u.v) LIMIT 2)",
    "SELECT id FROM t WHERE k IN (SELECT (SELECT COUNT(*) FROM e WHERE e.id = u.v) FROM u GROUP BY k)",
    "SELECT id, (SELECT k FROM u GROUP BY k ORDER BY (SELECT COUNT(*) FROM e WHERE e.id = u.v) LIMIT 1) \
     FROM t",
    "SELECT * FROM (SELECT k, (SELECT COUNT(*) FROM e WHERE e.id = u.v) AS c FROM u GROUP BY k) d",
    "SELECT * FROM (SELECT k FROM u GROUP BY k ORDER BY (SELECT COUNT(*) FROM e WHERE e.id = u.v) LIMIT 2) d",
];

/// Statements reading their rows only through items and calls, with the
/// rows each gives, sorted
const ANSWERED: &[(&str, &[&str])] = &[
    (
        "SELECT k, COUNT(*), SUM(v) FROM u GROUP BY k ORDER BY k",
        &["NULL|1|4", "1|2|11", "2|1|NULL", "4|1|4"],
    ),
    (
        "SELECT k + 1, MAX(v) FROM u WHERE k = 1 GROUP BY k + 1",
        &["2|10"],
    ),
    (
        "SELECT k AS g, COUNT(*) FROM u WHERE k > 1 GROUP BY 1 ORDER BY 1",
        &["2|1", "4|1"],
    ),
    ("SELECT 7, COUNT(*) FROM u", &["7|5"]),
    (
        "SELECT \"w\", COUNT(*) FROM u WHERE k = 1 GROUP BY k",
        &["w|2"],
    ),
    (
        "SELECT \"a.b\", COUNT(*) FROM q GROUP BY \"a.b\"",
        &["5|1", "6|1"],
    ),
    (
        "SELECT id, 11 IN (SELECT MAX(v) + t.k FROM u) FROM t",
        &["1|true", "2|false", "4|false"],
    ),
    (
        "SELECT id FROM t WHERE x + 1 = (SELECT u.k + t.k FROM u WHERE u.k = 1 GROUP BY 1)",
        &["1", "2", "4"],
    ),
    (
        "SELECT id FROM t WHERE 3 = (SELECT u.k + t.k AS s FROM u WHERE u.k = 1 GROUP BY s)",
        &["2"],
    ),
    (
        "SELECT id FROM t WHERE x < (SELECT MAX(v) + t.k FROM u WHERE u.k = t.k) ORDER BY id",
        &["1", "4"],
    ),
    (
        "SELECT id FROM t WHERE x = (SELECT k FROM u WHERE u.k = t.k GROUP BY k) ORDER BY id",
        &["1", "2", "4"],
    ),
    (
        "SELECT id, (SELECT COUNT(*) FROM u WHERE u.k = t.k GROUP BY k) FROM t ORDER BY id",
        &["1|2", "2|1", "4|1"],
    ),
    (
        "SELECT k FROM u GROUP BY k ORDER BY (SELECT COUNT(*) FROM u u2 WHERE u2.k = u.k), k",
        &["NULL", "1", "2", "4"],
    ),
    (
        "SELECT v AS w FROM u GROUP BY w ORDER BY (SELECT COUNT(*) FROM t WHERE t.id = u.v)",
        &["1", "10", "4", "NULL"],
    ),
    (
        "SELECT v FROM u GROUP BY 1 HAVING (SELECT COUNT(*) FROM t WHERE t.id = u.v) > 0",
        &["1", "4"],
    ),
    (
        "SELECT t.k, (SELECT COUNT(*) FROM u u2 WHERE u2.k = t.k) FROM t JOIN u ON u.k = t.k \
         GROUP BY t.k",
        &["1|2", "2|1", "4|1"],
    ),
    (
        "SELECT id FROM t WHERE EXISTS (SELECT k, (SELECT COUNT(*) FROM u u2 WHERE u2.k = u.k) \
         FROM u WHERE u.k = t.k GROUP BY k)",
        &["1", "2", "4"],
    ),
    (
        "SELECT id FROM t WHERE EXISTS (SELECT v AS w, (SELECT COUNT(*) FROM e WHERE e.id = u.v) \
         FROM u WHERE u.k = t.k GROUP BY w)",
        &["1", "2", "4"],
    ),
    (
        "SELECT id FROM t WHERE EXISTS (SELECT v, (SELECT COUNT(*) FROM e WHERE e.id = u.v) \
         FROM u WHERE u.k = t.k GROUP BY 1)",
        &["1", "2", "4"],
    ),
    (
        "SELECT id FROM t WHERE EXISTS (SELECT k, (SELECT COUNT(*) FROM e WHERE e.id = t.x) \
         FROM u WHERE u.k = t.k GROUP BY k)",
        &["1", "2", "4"],
    ),
    // An aggregate's input and a GROUP BY item read the rows before they group
    (
        "SELECT id FROM t WHERE EXISTS (SELECT SUM((SELECT u.v)) FROM u WHERE u.k = t.k)",
        &["1", "2", "4"],
    ),
    (
        "SELECT id FROM t WHERE EXISTS (SELECT (SELECT u.v) FROM u WHERE u.k = t.k GROUP BY 1)",
        &["1", "2", "4"],
    ),
    (
        "SELECT id FROM t WHERE EXISTS (SELECT SUM((SELECT u.v)) > ANY (SELECT 1) FROM u \
         WHERE u.k = t.k)",
        &["1", "2", "4"],
    ),
    (
        "SELECT id FROM t WHERE EXISTS (SELECT k, COUNT(*) FILTER (WHERE (SELECT u.v) > 0) \
         FROM u WHERE u.k = t.k GROUP BY k)",
        &["1", "2", "4"],
    ),
    (
        "SELECT id FROM t WHERE EXISTS (SELECT k, STRING_AGG(CAST(v AS TEXT), ',' \
         ORDER BY (SELECT u.v)) FROM u WHERE u.k = t.k GROUP BY k)",
        &["1", "2", "4"],
    ),
];

fn check(db: &Database, place: &str) {
    for sql in REFUSED {
        let got = answer(db, sql);
        assert!(got.is_err(), "{place}: {sql} answered {got:?}");
        let prepared = db
            .prepare(sql)
            .and_then(|stmt| stmt.query(()).map(|rows| rows.count()));
        assert!(
            prepared.is_err(),
            "{place}: {sql} prepared answered {prepared:?}"
        );
    }
    for sql in SUBQUERY_READS {
        let got = answer(db, sql);
        assert!(
            matches!(&got, Err(e) if e.contains("Column 'u.v' not found")),
            "{place}: {sql} gave {got:?}"
        );
        let prepared = db
            .prepare(sql)
            .and_then(|stmt| stmt.query(()).map(|rows| rows.count()))
            .map_err(|e| e.to_string());
        assert!(
            matches!(&prepared, Err(e) if e.contains("Column 'u.v' not found")),
            "{place}: {sql} prepared gave {prepared:?}"
        );
    }
    for (sql, want) in ANSWERED {
        let mut got = answer(db, sql).unwrap_or_else(|e| panic!("{place}: {sql}: {e}"));
        got.sort();
        let mut want: Vec<String> = want.iter().map(|s| s.to_string()).collect();
        want.sort();
        assert_eq!(got, want, "{place}: {sql}");
    }
    let parameter = db.query("SELECT v, COUNT(*) FROM u WHERE k = $1 GROUP BY k", (1,));
    assert!(parameter.is_err(), "{place}: a parameter keeps the rule");
}

#[test]
fn test_an_ungrouped_column_is_an_error_wherever_it_is_read() {
    let db = open(
        Database::open("memory://ungrouped_column_rule").unwrap(),
        false,
    );
    check(&db, "hot");
}

#[test]
fn test_an_ungrouped_column_is_an_error_over_sealed_rows() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_interval=0", dir.path().display());
    let db = open(Database::open(&dsn).unwrap(), true);
    check(&db, "sealed");
}
