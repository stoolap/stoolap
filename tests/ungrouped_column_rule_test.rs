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
