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

//! An aggregate result answers with one column per SELECT item, under each
//! item's own name, however many items name the same aggregate, and an
//! expression over a group item or aggregate reads it from its slot

use stoolap::Database;

fn open(db: Database, seal: bool) -> Database {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, a INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO t VALUES (1, 1, 1), (2, 1, 2), (3, 2, 10)", ())
        .unwrap();
    if seal {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// The column names and the one row, as text
fn answer(db: &Database, sql: &str) -> Result<(Vec<String>, Vec<String>), String> {
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    let columns = rows.columns().to_vec();
    let mut values = Vec::new();
    for row in rows {
        let row = row.map_err(|e| e.to_string())?;
        for i in 0..row.len() {
            values.push(format!("{}", row.get_value(i).unwrap()));
        }
    }
    Ok((columns, values))
}

fn check(db: &Database, ctx: &str, wrong: &mut Vec<String>) {
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    for (sql, columns, values) in [
        (
            "SELECT SUM(a), SUM(a), SUM(a) FROM t",
            s(&["SUM(a)", "SUM(a)", "SUM(a)"]),
            s(&["13", "13", "13"]),
        ),
        (
            "SELECT SUM(a) AS x, SUM(a) AS y FROM t",
            s(&["x", "y"]),
            s(&["13", "13"]),
        ),
        (
            "SELECT COUNT(*), COUNT(*) FROM t",
            s(&["COUNT(*)", "COUNT(*)"]),
            s(&["3", "3"]),
        ),
        (
            "SELECT SUM(a), COUNT(*), SUM(a) FROM t",
            s(&["SUM(a)", "COUNT(*)", "SUM(a)"]),
            s(&["13", "3", "13"]),
        ),
        (
            "SELECT MAX(a) AS hi, MIN(a), MAX(a) AS top FROM t",
            s(&["hi", "MIN(a)", "top"]),
            s(&["10", "1", "10"]),
        ),
        (
            "SELECT SUM(a), SUM(a) FROM t WHERE a > 0",
            s(&["SUM(a)", "SUM(a)"]),
            s(&["13", "13"]),
        ),
        (
            "SELECT SUM(a) AS x, COUNT(*), SUM(a) AS y FROM t WHERE a > 1",
            s(&["x", "COUNT(*)", "y"]),
            s(&["12", "2", "12"]),
        ),
        (
            "SELECT SUM(a) FROM t ORDER BY COUNT(*)",
            s(&["SUM(a)"]),
            s(&["13"]),
        ),
        (
            "SELECT sum(A) * 2 AS x, SUM(t.a) + 1 AS y, SUM(a) FROM t",
            s(&["x", "y", "SUM(a)"]),
            s(&["26", "14", "13"]),
        ),
        (
            "SELECT sum(A) * 2 AS x, SUM(a) FROM t WHERE a > 0",
            s(&["x", "SUM(a)"]),
            s(&["26", "13"]),
        ),
        (
            "SELECT SUM(a) FROM t WHERE a > 0 ORDER BY COUNT(*)",
            s(&["SUM(a)"]),
            s(&["13"]),
        ),
    ] {
        let got = answer(db, sql);
        let want = Ok((columns, values));
        if got != want {
            wrong.push(format!("{ctx}: {sql}: got {got:?}, want {want:?}"));
        }
    }
}

#[test]
fn every_select_item_of_a_global_aggregate_has_its_column() {
    let mut wrong = Vec::new();
    check(
        &open(
            Database::open("memory://global_aggregate_projection").unwrap(),
            false,
        ),
        "memory",
        &mut wrong,
    );
    for seal in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open(
            Database::open(&format!(
                "file://{}?checkpoint_interval=0",
                dir.path().display()
            ))
            .unwrap(),
            seal,
        );
        check(&db, &format!("file sealed {seal}"), &mut wrong);
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn items_sharing_an_alias_keep_their_own_values() {
    let db = Database::open("memory://shared_alias_projection").unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, g INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 7)", ()).unwrap();
    let mut wrong = Vec::new();
    for (sql, want) in [
        ("SELECT g AS n, COUNT(*) AS n FROM u GROUP BY g", ["7", "1"]),
        (
            "SELECT g AS n, COUNT(*) AS n FROM u WHERE id + 0 = id GROUP BY g",
            ["7", "1"],
        ),
        (
            "SELECT COUNT(*) AS n, g AS n FROM u WHERE id + 0 = id GROUP BY g",
            ["1", "7"],
        ),
        ("SELECT SUM(g) AS n, COUNT(*) AS n FROM u", ["7", "1"]),
        ("SELECT g AS n, COUNT(*) AS g FROM u GROUP BY g", ["7", "1"]),
        (
            "SELECT g AS n, COUNT(*) AS g FROM u WHERE id + 0 = id GROUP BY g",
            ["7", "1"],
        ),
        (
            "SELECT COUNT(*) AS g, g AS n FROM u WHERE id + 0 = id GROUP BY g",
            ["1", "7"],
        ),
    ] {
        let got = answer(&db, sql).map(|(_, values)| values);
        if got != Ok(want.iter().map(|s| s.to_string()).collect()) {
            wrong.push(format!("{sql}: got {got:?}, want {want:?}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn an_expression_over_a_group_item_reads_its_slot() {
    let db = open(
        Database::open("memory://group_item_projection").unwrap(),
        false,
    );
    let mut wrong = Vec::new();
    for (sql, want) in [
        (
            "SELECT (g + 1) * 10 FROM t GROUP BY g + 1",
            vec!["20", "30"],
        ),
        (
            "SELECT (G+1) * 10, COUNT(*) FROM t GROUP BY g + 1",
            vec!["20|2", "30|1"],
        ),
        (
            "SELECT UPPER(CAST(g AS TEXT)) || 'x' FROM t GROUP BY UPPER(CAST(g AS TEXT))",
            vec!["1x", "2x"],
        ),
        (
            "SELECT g + 1 FROM t GROUP BY g + 1 HAVING (g + 1) * 10 > 25",
            vec!["3"],
        ),
    ] {
        let got = db.query(sql, ()).map_err(|e| e.to_string()).map(|rows| {
            let mut out: Vec<String> = rows
                .map(|row| {
                    let row = row.unwrap();
                    (0..row.len())
                        .map(|i| format!("{}", row.get_value(i).unwrap()))
                        .collect::<Vec<_>>()
                        .join("|")
                })
                .collect();
            out.sort();
            out
        });
        if got != Ok(want.iter().map(|s| s.to_string()).collect()) {
            wrong.push(format!("{sql}: got {got:?}, want {want:?}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
