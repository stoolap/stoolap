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

use stoolap::core::Value;
use stoolap::{Database, Params};

/// `t` and `u`, ids 1..=10 with `a = id`, an index on `t.a`
fn tables(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for table in ["t", "u"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, a INTEGER)"),
            (),
        )
        .unwrap();
        for i in 1..=10 {
            db.execute(&format!("INSERT INTO {table} VALUES ({i}, {i})"), ())
                .unwrap();
        }
    }
    db.execute("CREATE INDEX ta ON t(a)", ()).unwrap();
    db
}

fn ids<P: Params>(db: &Database, sql: &str, params: P) -> Result<Vec<i64>, String> {
    let rows = db.query(sql, params).map_err(|e| e.to_string())?;
    rows.map(|row| {
        row.map_err(|e| e.to_string())?
            .get::<i64>(0)
            .map_err(|e| e.to_string())
    })
    .collect()
}

/// Every path that applies a LIMIT and OFFSET, `{}` where they go: the rows
/// it returns are the ids 1..=10 in order, or for an unordered path a count
const PATHS: &[(&str, bool)] = &[
    ("SELECT id FROM t ORDER BY id {}", true),
    ("SELECT id FROM t {}", false),
    ("SELECT id FROM t WHERE a > 0 {}", false),
    (
        "WITH c AS (SELECT id FROM t) SELECT id FROM c ORDER BY id {}",
        true,
    ),
    (
        "WITH a AS (SELECT id FROM t), b AS (SELECT id FROM a ORDER BY id {}) \
         SELECT id FROM b ORDER BY id",
        true,
    ),
    ("SELECT a FROM t GROUP BY a ORDER BY a {}", true),
    ("SELECT a FROM t GROUP BY a {}", false),
    (
        "SELECT id FROM t WHERE id IN (1, 2, 3, 4, 5, 6, 7, 8, 9, 10) {}",
        false,
    ),
    (
        "SELECT id FROM t WHERE a IN (1, 2, 3, 4, 5, 6, 7, 8, 9, 10) {}",
        false,
    ),
    ("SELECT id FROM t WHERE id IN (SELECT id FROM u) {}", false),
    (
        "SELECT id FROM t WHERE id NOT IN (SELECT id FROM u WHERE id > 10) {}",
        false,
    ),
    ("SELECT id FROM t WHERE id > 0 ORDER BY id {}", true),
    ("SELECT id FROM t ORDER BY a {}", true),
    (
        "SELECT t.id FROM t JOIN u ON t.id = u.id ORDER BY t.id {}",
        true,
    ),
    ("SELECT t.id FROM t JOIN u ON t.id = u.id {}", false),
    (
        "SELECT id FROM (SELECT id, ROW_NUMBER() OVER (PARTITION BY a) AS r FROM t) w \
         ORDER BY id {}",
        true,
    ),
    (
        "SELECT id FROM t UNION ALL SELECT id FROM u WHERE id > 10 {}",
        false,
    ),
];

fn check(db: &Database, clause: &str, want: Result<&[i64], ()>, params: impl Fn() -> Vec<Value>) {
    for (path, ordered) in PATHS {
        let sql = path.replace("{}", clause);
        let got = ids(db, &sql, params());
        match want {
            Err(()) => assert!(got.is_err(), "{sql} is an error, got {got:?}"),
            Ok(rows) if *ordered => assert_eq!(got.as_deref(), Ok(rows), "{sql}"),
            Ok(rows) => assert_eq!(
                got.as_ref().map(Vec::len),
                Ok(rows.len()),
                "{sql} returns {} rows: {got:?}",
                rows.len()
            ),
        }
    }
}

#[test]
fn test_every_path_reads_a_count_alike() {
    let db = tables("row_count_paths");
    let none = Vec::new;
    for (clause, want) in [
        ("LIMIT 3", Ok(&[1, 2, 3][..])),
        ("LIMIT 1 + 2", Ok(&[1, 2, 3][..])),
        ("LIMIT 3.0", Ok(&[1, 2, 3][..])),
        ("LIMIT 3 OFFSET 2", Ok(&[3, 4, 5][..])),
        ("LIMIT 3 OFFSET 2.0", Ok(&[3, 4, 5][..])),
        ("LIMIT 0", Ok(&[][..])),
        ("LIMIT 3 OFFSET 20", Ok(&[][..])),
        ("LIMIT 2.5", Err(())),
        ("LIMIT 3 OFFSET 1.5", Err(())),
        ("LIMIT -1", Err(())),
        ("LIMIT 3 OFFSET -1", Err(())),
        ("LIMIT NULL", Err(())),
        ("LIMIT 'x'", Err(())),
        ("LIMIT 0.0 / 0.0", Err(())),
    ] {
        check(&db, clause, want, none);
    }
}

#[test]
fn test_a_parameter_count_is_read_as_a_literal_one() {
    let db = tables("row_count_parameters");
    for (value, want) in [
        (Value::Integer(3), Ok(&[1, 2, 3][..])),
        (Value::Float(3.0), Ok(&[1, 2, 3][..])),
        (Value::Float(2.5), Err(())),
        (Value::Integer(-1), Err(())),
        (Value::Null(stoolap::core::DataType::Integer), Err(())),
    ] {
        check(&db, "LIMIT $1", want, || vec![value.clone()]);
    }
}
