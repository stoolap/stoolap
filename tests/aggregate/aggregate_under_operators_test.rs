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

//! An aggregate call is computed wherever an operator holds it: a CASE
//! value, BETWEEN, IN, LIKE or a row, in the select list, HAVING and ORDER
//! BY, hot and sealed, prepared, and on the general path

use stoolap::Database;

const CASES: &[(&str, &str)] = &[
    (
        "SELECT CASE SUM(a) WHEN 7 THEN 'x' ELSE 'y' END FROM t",
        "x",
    ),
    ("SELECT SUM(a) BETWEEN 1 AND 9 FROM t", "true"),
    ("SELECT 5 BETWEEN MIN(a) AND SUM(a) FROM t", "true"),
    ("SELECT COUNT(*) IN (1, 3) FROM t", "true"),
    ("SELECT 3 IN (COUNT(*), 0) FROM t", "true"),
    ("SELECT MIN(s) LIKE 'x%' FROM t", "true"),
    ("SELECT 'xz' LIKE MIN(s) || '%' FROM t", "true"),
    ("SELECT (SUM(a), COUNT(*)) = (7, 3) FROM t", "true"),
    ("SELECT COUNT(*) FROM t HAVING SUM(a) BETWEEN 1 AND 9", "3"),
    ("SELECT CASE COUNT(*) WHEN 1 THEN 'one' END", "one"),
    (
        "SELECT CASE COUNT(*) WHEN 100 THEN 1 ELSE 0 END FROM generate_series(1, 100) LIMIT 1",
        "1",
    ),
    (
        "SELECT g, SUM(a) BETWEEN 4 AND 9 FROM t GROUP BY g ORDER BY g",
        "1|false;2|true",
    ),
    (
        "SELECT g, CASE COUNT(*) WHEN 2 THEN 'two' END FROM t GROUP BY g ORDER BY g",
        "1|two;2|NULL",
    ),
    (
        "SELECT g FROM t GROUP BY g HAVING SUM(a) BETWEEN 4 AND 9",
        "2",
    ),
    ("SELECT g FROM t GROUP BY g HAVING COUNT(*) IN (2)", "1"),
    ("SELECT g FROM t GROUP BY g HAVING MIN(s) LIKE 'xz'", "2"),
    (
        "SELECT g FROM t GROUP BY g HAVING CASE SUM(a) WHEN 3 THEN TRUE ELSE FALSE END",
        "1",
    ),
    (
        "SELECT g FROM t GROUP BY g ORDER BY CASE SUM(a) WHEN 3 THEN 1 ELSE 0 END",
        "2;1",
    ),
    ("SELECT id FROM t WHERE SUM(a) BETWEEN 1 AND 9", "error"),
];

/// (query, parameter, answer)
const PREPARED: &[(&str, i64, &str)] = &[
    (
        "SELECT CASE SUM(a) WHEN $1 THEN 'x' ELSE 'y' END FROM t",
        7,
        "x",
    ),
    (
        "SELECT CASE SUM(a) WHEN $1 THEN 'x' ELSE 'y' END FROM t",
        6,
        "y",
    ),
    ("SELECT SUM(a) BETWEEN $1 AND 9 FROM t", 8, "false"),
    (
        "SELECT g FROM t GROUP BY g HAVING SUM(a) BETWEEN $1 AND 9",
        4,
        "2",
    ),
    (
        "SELECT g FROM t GROUP BY g HAVING COUNT(*) IN ($1, 5)",
        1,
        "2",
    ),
];

fn open(dir: &tempfile::TempDir, sealed: bool, name: &str) -> Database {
    let db = match sealed {
        true => Database::open(&format!("file://{}", dir.path().display())).unwrap(),
        false => Database::open(&format!("memory://{name}")).unwrap(),
    };
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, a INTEGER, s TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 1, 1, 'x'), (2, 1, 2, 'y'), (3, 2, 4, 'xz')",
        (),
    )
    .unwrap();
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

#[test]
fn an_aggregate_under_an_operator_is_computed_hot_and_sealed() {
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = open(&dir, sealed, "aggregate_under_operators_runner");
        check(&db, &format!("sealed {sealed}"));
    }
}
