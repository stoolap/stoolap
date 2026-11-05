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

use stoolap::{Database, Value};

fn setup(name: &str, dir: &tempfile::TempDir, sealed: bool) -> Database {
    let dsn = match sealed {
        true => format!("file://{}/db", dir.path().display()),
        false => format!("memory://{name}"),
    };
    let db = Database::open(&dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, s TEXT, sep TEXT)",
        "INSERT INTO t VALUES (1, 1, 'a', '-'), (2, 1, 'b', '+'), (3, 2, 'c', NULL), \
         (4, 2, 'e', 'x'), (5, 3, 'd', '/')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn answers(db: &Database, sql: &str, params: impl stoolap::Params) -> Vec<(i64, Option<String>)> {
    db.query(sql, params)
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            let text = match row.get::<Value>(1).unwrap() {
                Value::Text(text) => Some(text.to_string()),
                _ => None,
            };
            (row.get::<i64>(0).unwrap(), text)
        })
        .collect()
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect()
}

fn prepared(db: &Database, sql: &str) -> bool {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .any(|row| row.unwrap().get::<String>(0).unwrap().trim() == "Prepared: yes")
}

/// Each row's id and its subquery's text, None for NULL
type Answers<'a> = &'a [(i64, Option<&'a str>)];

fn texts(expected: Answers<'_>) -> Vec<(i64, Option<String>)> {
    expected
        .iter()
        .map(|(id, text)| (*id, text.map(str::to_string)))
        .collect()
}

#[test]
fn a_separator_reading_the_enclosing_row_is_read_for_each_call() {
    let shapes: &[(&str, Answers<'_>)] = &[
        (
            "SELECT id, (SELECT STRING_AGG(t2.s, t.sep ORDER BY t2.id) FROM t t2 \
             WHERE t2.k = t.k) FROM t ORDER BY id",
            &[
                (1, Some("a-b")),
                (2, Some("a+b")),
                (3, None),
                (4, Some("cxe")),
                (5, Some("d")),
            ],
        ),
        (
            "SELECT id, (SELECT STRING_AGG(t2.s, t.sep || t.sep ORDER BY t2.id) FROM t t2 \
             WHERE t2.k = t.k GROUP BY t2.k) FROM t ORDER BY id",
            &[
                (1, Some("a--b")),
                (2, Some("a++b")),
                (3, None),
                (4, Some("cxxe")),
                (5, Some("d")),
            ],
        ),
        (
            "SELECT id, (SELECT STRING_AGG(t2.s, t.sep ORDER BY t2.id) FROM t t2 \
             WHERE t2.k = t.k HAVING COUNT(*) > 1) FROM t ORDER BY id",
            &[
                (1, Some("a-b")),
                (2, Some("a+b")),
                (3, None),
                (4, Some("cxe")),
                (5, None),
            ],
        ),
    ];
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = setup("block_later_arguments_each_call", &dir, sealed);
        for (sql, expected) in shapes {
            assert!(prepared(&db, sql), "sealed {sealed}: not prepared: {sql}");
            assert_eq!(
                answers(&db, sql, ()),
                texts(expected),
                "sealed {sealed}: {sql}"
            );
        }
        let exists = "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM t t2 WHERE t2.k = t.k \
                      HAVING STRING_AGG(t2.s, t.sep ORDER BY t2.id) = 'a-b') ORDER BY id";
        assert!(
            prepared(&db, exists),
            "sealed {sealed}: not prepared: {exists}"
        );
        assert_eq!(ids(&db, exists), vec![1], "sealed {sealed}: {exists}");
        let members = "SELECT id FROM t WHERE 'a+b' IN (SELECT STRING_AGG(t2.s, t.sep \
                       ORDER BY t2.id) FROM t t2 WHERE t2.k = t.k) ORDER BY id";
        assert!(
            prepared(&db, members),
            "sealed {sealed}: not prepared: {members}"
        );
        assert_eq!(ids(&db, members), vec![2], "sealed {sealed}: {members}");
    }
}

#[test]
fn a_parameter_separator_is_read_for_each_execution() {
    let dir = tempfile::tempdir().unwrap();
    let db = setup("block_later_arguments_parameter", &dir, false);
    let sql = "SELECT id, (SELECT STRING_AGG(t2.s, $1 ORDER BY t2.id) FROM t t2 \
               WHERE t2.k = t.k) FROM t ORDER BY id";
    assert!(prepared(&db, sql), "not prepared: {sql}");
    for sep in ["/", "*"] {
        let pair = |a: &str, b: &str| Some(format!("{a}{sep}{b}"));
        assert_eq!(
            answers(&db, sql, (sep,)),
            vec![
                (1, pair("a", "b")),
                (2, pair("a", "b")),
                (3, pair("c", "e")),
                (4, pair("c", "e")),
                (5, Some("d".to_string())),
            ],
            "separator {sep}"
        );
    }
}

#[test]
fn a_separator_reading_the_block_row_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let db = setup("block_later_arguments_own_row", &dir, false);
    let sql = "SELECT id, (SELECT STRING_AGG(t2.s, t2.sep) FROM t t2 WHERE t2.k = t.k) FROM t";
    let error = db
        .query(sql, ())
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("an aggregate argument after the first must not read the row"),
        "{error}"
    );
}

#[test]
fn an_error_from_an_extra_argument_is_the_call_s_on_a_compiled_aggregate() {
    let db = Database::open("memory://block_later_arguments_compiled_error").unwrap();
    for sql in [
        "CREATE TABLE a (id INTEGER PRIMARY KEY, k INTEGER, x INTEGER)",
        "CREATE TABLE b (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "INSERT INTO a VALUES (1, 1, -9223372036854775808)",
        "INSERT INTO b VALUES (1, 1, 5)",
        "BEGIN",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let sql = "SELECT a.id, (SELECT SUM(b.v, CASE WHEN CURRENT_TRANSACTION_ID() IS NULL \
               THEN 0 ELSE ABS(a.x) END) FROM b WHERE b.k = a.k) FROM a";
    let answer = db
        .query(sql, ())
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>());
    db.execute("ROLLBACK", ()).unwrap();
    let error = answer
        .expect_err("the extra argument's overflow")
        .to_string();
    assert!(error.to_lowercase().contains("overflow"), "{error}");
}
