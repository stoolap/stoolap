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

/// `w (id, g)` with rows (1, 1), (2, 1), (3, 2), (4, 2); `g` indexed when asked
fn groups(name: &str, indexed: bool) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE w (id INTEGER PRIMARY KEY, g INTEGER)", ())
        .unwrap();
    if indexed {
        db.execute("CREATE INDEX wg ON w(g)", ()).unwrap();
    }
    db.execute("INSERT INTO w VALUES (1, 1), (2, 1), (3, 2), (4, 2)", ())
        .unwrap();
    db
}

#[test]
fn test_a_window_limit_comes_after_distinct() {
    for indexed in [false, true] {
        let db = groups(&format!("row_count_window_distinct_{indexed}"), indexed);
        for clause in ["LIMIT 2", "LIMIT 1 + 1"] {
            let sql =
                format!("SELECT DISTINCT g, COUNT(*) OVER (PARTITION BY g) AS n FROM w {clause}");
            assert_eq!(
                ids(&db, &sql, ()),
                Ok(vec![1, 2]),
                "{sql}, index on g: {indexed}"
            );
        }
    }
}

#[test]
fn test_a_window_limit_path_reads_the_parameters() {
    for indexed in [false, true] {
        let db = groups(&format!("row_count_window_params_{indexed}"), indexed);
        for clause in ["LIMIT 2", "LIMIT 1 + 1"] {
            let sql = format!(
                "SELECT $1 AS p, ROW_NUMBER() OVER (PARTITION BY g ORDER BY id) AS n FROM w {clause}"
            );
            assert_eq!(
                ids(&db, &sql, (42i64,)),
                Ok(vec![42, 42]),
                "{sql}, index on g: {indexed}"
            );
        }
    }
}

#[test]
fn test_a_count_that_needs_a_parameter_is_no_kept_shortcut() {
    let db = tables("row_count_kept_shortcut");
    for _ in 0..2 {
        assert_eq!(
            ids(
                &db,
                "SELECT id FROM t WHERE id = 1 LIMIT COALESCE($1, 1)",
                (0i64,)
            ),
            Ok(vec![]),
            "LIMIT 0 by its parameter, on every run"
        );
    }
    let db = Database::open("memory://row_count_cte_hint").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1), (2), (3)", ())
        .unwrap();
    assert_eq!(
        ids(
            &db,
            "WITH c AS (SELECT id FROM t GROUP BY id) \
             SELECT c.id FROM c LEFT JOIN u ON u.id = c.id LIMIT COALESCE($1, 1)",
            (3i64,)
        )
        .map(|rows| rows.len()),
        Ok(3),
        "the CTE is not cut to the LIMIT the parameter-free reading gives"
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn test_explain_evaluates_no_count() {
    use stoolap::test_failpoints::random_draws;
    let db = tables("row_count_explain");
    db.execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, emb VECTOR(3))",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO docs VALUES (1, '[1,0,0]')", ())
        .unwrap();
    for sql in [
        "EXPLAIN SELECT id FROM docs ORDER BY VEC_DISTANCE_L2(emb, '[1,0,0]') \
         LIMIT CAST(RANDOM() * 0 AS INTEGER) + 1",
        "EXPLAIN SELECT t.id FROM t JOIN u ON t.id = u.id LIMIT CAST(RANDOM() * 0 AS INTEGER) + 1",
    ] {
        let before = random_draws();
        let rows = db.query(sql, ()).unwrap();
        for row in rows {
            row.unwrap();
        }
        assert_eq!(random_draws() - before, 0, "EXPLAIN drew for {sql}");
    }
}
