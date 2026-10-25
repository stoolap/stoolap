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

//! An ORDER BY with a LIMIT answers the rows the whole sort puts at those
//! places, whatever its keys compute, over a table, a derived table and a view

use stoolap::{Database, Params};

fn setup(db: &Database) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b TEXT, c FLOAT)",
        "INSERT INTO t SELECT value, CASE WHEN value % 11 = 0 THEN NULL ELSE value % 13 END, \
         CASE WHEN value % 17 = 0 THEN NULL ELSE 'b' || (value % 7) END, \
         (value * 37 % 100) / 10.0 FROM generate_series(1, 600)",
        "CREATE VIEW v AS SELECT id, a, b, c FROM t",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

/// The column names, then each row's values with their types
fn rows(db: &Database, sql: &str, params: impl Params) -> Vec<String> {
    let rows = db
        .query(sql, params)
        .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = vec![rows.columns().join(", ")];
    out.extend(rows.map(|row| {
        let row = row.unwrap();
        format!(
            "{:?}",
            (0..row.len()).map(|i| row.get_value(i)).collect::<Vec<_>>()
        )
    }));
    out
}

/// Each statement against its rows without the LIMIT and OFFSET, cut to
/// them: every ORDER BY ends in `id`, so the sort has one answer
fn check(db: &Database, phase: &str, wrong: &mut Vec<String>) {
    for (sql, limit, offset) in [
        (
            "SELECT id, a FROM {src} WHERE c > 2 ORDER BY a + 0, id",
            7,
            0,
        ),
        (
            "SELECT id, a FROM {src} ORDER BY a + 0 DESC NULLS FIRST, id",
            7,
            3,
        ),
        (
            "SELECT id, a + 1 AS a1 FROM {src} ORDER BY a1 NULLS LAST, id DESC",
            5,
            0,
        ),
        (
            "SELECT id, a + 1 AS a1 FROM {src} WHERE c < 5 ORDER BY a1 DESC, id",
            9,
            4,
        ),
        ("SELECT id FROM {src} ORDER BY b, a, id", 9, 0),
        ("SELECT id FROM {src} ORDER BY b DESC NULLS LAST, id", 6, 2),
        (
            "SELECT id, b || 'x' AS bx FROM {src} ORDER BY bx DESC, id",
            4,
            2,
        ),
        (
            "SELECT a * 2 AS d, id FROM {src} WHERE c > 1 ORDER BY 1, 2",
            6,
            0,
        ),
        ("SELECT id FROM {src} ORDER BY a * -1, id", 5, 0),
        (
            "SELECT id, COALESCE(a, -1) AS k FROM {src} ORDER BY k, id",
            8,
            0,
        ),
        (
            "SELECT id, CASE WHEN a > 6 THEN 'hi' ELSE 'lo' END AS h FROM {src} \
             ORDER BY h, c, id",
            8,
            1,
        ),
        (
            "SELECT id, a + (SELECT 1) AS x FROM {src} ORDER BY x, id",
            5,
            0,
        ),
        ("SELECT id, a FROM {src} ORDER BY a + 0, id", 0, 0),
        ("SELECT id, a FROM {src} ORDER BY a + 0, id", 5, 1000),
        ("SELECT id, a FROM {src} ORDER BY a + 0, id", 1000, 0),
    ] {
        for src in ["t", "(SELECT id, a, b, c FROM t) s", "v"] {
            let all = sql.replace("{src}", src);
            let mut whole = rows(db, &all, ()).into_iter();
            let names = whole.next();
            let want: Vec<String> = names
                .into_iter()
                .chain(whole.skip(offset).take(limit))
                .collect();
            let cut = format!("{all} LIMIT {limit} OFFSET {offset}");
            let bound = format!("{all} LIMIT $1 OFFSET $2");
            for got in [
                (&cut, rows(db, &cut, ())),
                (&bound, rows(db, &bound, (limit as i64, offset as i64))),
            ] {
                if got.1 != want {
                    let (sql, got) = got;
                    wrong.push(format!("{phase}: {sql}\n  got  {got:?}\n  want {want:?}"));
                }
            }
        }
    }
}

#[test]
fn an_order_by_limit_answers_the_places_the_whole_sort_gives() {
    let mut wrong = Vec::new();
    let db = Database::open("memory://general_order_limit_hot").unwrap();
    setup(&db);
    check(&db, "hot", &mut wrong);

    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    setup(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("UPDATE t SET a = 5 WHERE id % 40 = 0", ())
        .unwrap();
    check(&db, "sealed with hot updates", &mut wrong);
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn tied_keys_keep_the_values_the_whole_sort_puts_first() {
    let db = Database::open("memory://general_order_limit_ties").unwrap();
    setup(&db);
    for (sql, limit) in [
        ("SELECT a + 0 AS k, id FROM t ORDER BY a + 0", 20),
        (
            "SELECT a + 1 AS k, id FROM t ORDER BY k DESC NULLS FIRST",
            30,
        ),
        ("SELECT b AS k, id FROM t ORDER BY b, a + 0", 25),
    ] {
        let key = |row: &String| row.split(", ").next().unwrap().to_string();
        // The column names, then the first rows' keys
        let want: Vec<String> = rows(&db, sql, ()).iter().take(limit + 1).map(key).collect();
        let got: Vec<String> = rows(&db, &format!("{sql} LIMIT {limit}"), ())
            .iter()
            .map(key)
            .collect();
        assert_eq!(got, want, "{sql} LIMIT {limit}");
    }
}
