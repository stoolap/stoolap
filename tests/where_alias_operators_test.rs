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

//! A SELECT alias a WHERE names is read as its expression under any
//! operator: CAST, an IN list, a row, CASE, LIKE, a call, hot and sealed,
//! prepared, joined, and on the general path, as ORDER BY and HAVING read it

use stoolap::Database;

const CASES: &[(&str, &str)] = &[
    ("SELECT a + 1 AS z FROM t WHERE z = 2", "2"),
    ("SELECT a + 1 AS z FROM t WHERE CAST(z AS TEXT) = '2'", "2"),
    ("SELECT a + 1 AS z FROM t WHERE 2 IN (z, 5)", "2"),
    ("SELECT a + 1 AS z FROM t WHERE z IN (2, 5)", "2"),
    (
        "SELECT a + 1 AS z FROM t WHERE z NOT IN (2, 3) ORDER BY z",
        "4",
    ),
    ("SELECT a + 1 AS z FROM t WHERE (z, a) = (2, 1)", "2"),
    (
        "SELECT a + 1 AS z FROM t WHERE (z, a) IN ((3, 2), (9, 9))",
        "3",
    ),
    ("SELECT a + 1 AS z FROM t WHERE z BETWEEN 3 AND 3", "3"),
    ("SELECT a + 1 AS z FROM t WHERE NOT (z > 2)", "2"),
    ("SELECT a + 1 AS z FROM t WHERE ABS(z) = 3", "3"),
    (
        "SELECT a + 1 AS z FROM t WHERE CASE z WHEN 4 THEN TRUE END",
        "4",
    ),
    ("SELECT s || 'q' AS w FROM t WHERE w LIKE 'xq%'", "xq"),
    (
        "SELECT s || 'q' AS w FROM t WHERE CAST(w AS TEXT) LIKE 'xzq'",
        "xzq",
    ),
    (
        "SELECT a + 1 AS z FROM t WHERE CAST(CAST(z AS TEXT) AS INTEGER) IN (3, 4) ORDER BY z",
        "3;4",
    ),
    (
        "SELECT t.a + 1 AS z FROM t JOIN u ON u.id = t.id WHERE CAST(z AS TEXT) = '3'",
        "3",
    ),
    (
        "SELECT t.a + 1 AS z FROM t JOIN u ON u.id = t.id WHERE (z, u.k) = (2, 10)",
        "2",
    ),
    ("SELECT a + 1 AS a FROM t WHERE CAST(a AS TEXT) = '2'", "3"),
    (
        "SELECT a + 1 AS z FROM t ORDER BY CAST(z AS TEXT) DESC",
        "4;3;2",
    ),
    (
        "SELECT a + 1 AS z FROM t ORDER BY (z IN (2, 3)), z",
        "4;2;3",
    ),
    (
        "SELECT a AS z, COUNT(*) FROM t GROUP BY a HAVING CAST(z AS TEXT) = '2'",
        "2|1",
    ),
    (
        "SELECT a AS z, COUNT(*) FROM t GROUP BY a HAVING (z, 1) = (3, 1)",
        "3|1",
    ),
];

/// (query, parameter, answer)
const PREPARED: &[(&str, i64, &str)] = &[
    (
        "SELECT a + 1 AS z FROM t WHERE CAST(z AS INTEGER) = $1",
        3,
        "3",
    ),
    ("SELECT a + 1 AS z FROM t WHERE $1 IN (z, 0)", 4, "4"),
    ("SELECT a + 1 AS z FROM t WHERE (z, a) = ($1, 1)", 2, "2"),
];

fn open(dir: &tempfile::TempDir, sealed: bool, name: &str) -> Database {
    let db = match sealed {
        true => Database::open(&format!("file://{}", dir.path().display())).unwrap(),
        false => Database::open(&format!("memory://{name}")).unwrap(),
    };
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, s TEXT)",
        "INSERT INTO t VALUES (1, 1, 'x'), (2, 2, 'y'), (3, 3, 'xz')",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER)",
        "INSERT INTO u VALUES (1, 10), (2, 20), (3, 30)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn answer(rows: stoolap::Result<stoolap::Rows>) -> String {
    let Ok(rows) = rows else {
        return "error".to_string();
    };
    let mut out = Vec::new();
    for row in rows {
        let Ok(row) = row else {
            return "error".to_string();
        };
        let values: Vec<String> = (0..row.len())
            .map(|i| row.get_value(i).unwrap().to_string())
            .collect();
        out.push(values.join("|"));
    }
    out.join(";")
}

fn check(db: &Database, context: &str) {
    for (sql, want) in CASES {
        assert_eq!(answer(db.query(sql, ())), *want, "{context}: {sql}");
    }
    for (sql, param, want) in PREPARED {
        let stmt = db.prepare(sql).unwrap();
        assert_eq!(
            answer(stmt.query((*param,))),
            *want,
            "{context}: {sql} with {param}"
        );
    }
}

// The families' switch is process-wide
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn a_where_reads_an_alias_under_any_operator_hot_and_sealed() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir, sealed, "where_alias_operators_runner");
        check(&db, &format!("sealed {sealed}"));
    }
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_where_reads_an_alias_under_any_operator_on_the_general_path() {
    use std::sync::atomic::Ordering::SeqCst;
    use stoolap::test_failpoints::PREPARED_FAMILIES_OFF;
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            PREPARED_FAMILIES_OFF.store(false, SeqCst);
        }
    }
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    PREPARED_FAMILIES_OFF.store(true, SeqCst);
    let _restore = Restore;
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir, sealed, "where_alias_operators_general");
        check(&db, &format!("general, sealed {sealed}"));
    }
}
