// Copyright 2026 Stoolap Contributors
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

//! A subquery without FROM reads one row with no columns, and answers as
//! the same subquery over a table holding that one row would

use stoolap::Database;

fn setup(db: &Database) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO t VALUES (1, 1), (2, 2), (3, 3), (4, NULL)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO u VALUES (1, 1), (2, 2)",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

fn answer(rows: stoolap::Result<stoolap::Rows>) -> Result<Vec<String>, String> {
    rows.map_err(|e| e.to_string())?
        .map(|row| {
            let row = row.map_err(|e| e.to_string())?;
            Ok((0..row.len())
                .map(|i| format!("{:?}", row.get::<stoolap::core::Value>(i).unwrap()))
                .collect::<Vec<_>>()
                .join(","))
        })
        .collect()
}

/// Each WHERE with a subquery without FROM, and the same WHERE with what
/// the subquery answers written out
const WHERE_TWINS: &[(&str, &str)] = &[
    ("a = (SELECT 2)", "a = 2"),
    ("a = (SELECT t.a)", "a = a"),
    (
        "a = (SELECT 2 WHERE t.id > 1)",
        "a = CASE WHEN id > 1 THEN 2 END",
    ),
    ("EXISTS (SELECT 1 WHERE t.a = 2)", "a = 2"),
    ("NOT EXISTS (SELECT 1 WHERE t.a = 2)", "a IS NULL OR a <> 2"),
    ("a IN (SELECT 2)", "a IN (2)"),
    ("a NOT IN (SELECT t.id + 1)", "a <> id + 1"),
    ("a > ANY (SELECT 1)", "a > 1"),
    ("(SELECT COUNT(*)) = 1", "1 = 1"),
    ("(SELECT COUNT(*) WHERE t.a = 2) = 1", "a = 2"),
    ("a = (SELECT (SELECT MAX(x) FROM u))", "a = 2"),
    (
        "a = (SELECT MAX(x) FROM u WHERE u.x = (SELECT t.a))",
        "a IN (1, 2)",
    ),
    ("id = (SELECT 2)", "id = 2"),
    ("id IN (SELECT 3)", "id IN (3)"),
    ("EXISTS (SELECT 1 HAVING t.a > 1)", "a > 1"),
    ("EXISTS (SELECT COUNT(*) WHERE t.a > 5)", "1 = 1"),
    ("a = (SELECT a)", "a = a"),
    ("a = (SELECT a AS z WHERE z > 1)", "a > 1"),
];

#[test]
fn a_block_without_from_answers_as_one_row() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = if sealed {
            Database::open(&format!("file://{}", dir.path().display())).unwrap()
        } else {
            Database::open("memory://prepared_fromless_where").unwrap()
        };
        setup(&db);
        if sealed {
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        for (with_block, written) in WHERE_TWINS {
            let sql = format!("SELECT id FROM t WHERE {with_block} ORDER BY id");
            let want =
                answer(db.query(&format!("SELECT id FROM t WHERE {written} ORDER BY id"), ()));
            let plan = answer(db.query(&format!("EXPLAIN {sql}"), ())).unwrap_or_default();
            if !plan.iter().any(|line| line.contains("Prepared: yes")) {
                wrong.push(format!("{sql} (sealed: {sealed}): not prepared: {plan:?}"));
            }
            let prepared = db.prepare(&sql).map_err(|e| e.to_string());
            for context in ["autocommit", "transaction", "prepared", "prepared again"] {
                let got = match context {
                    "autocommit" => answer(db.query(&sql, ())),
                    "transaction" => {
                        let mut tx = db.begin().unwrap();
                        let got = answer(tx.query(&sql, ()));
                        tx.rollback().unwrap();
                        got
                    }
                    _ => prepared.clone().and_then(|p| answer(p.query(()))),
                };
                if got != want {
                    wrong.push(format!(
                        "{sql} (sealed: {sealed}, {context}): {got:?}, want {want:?}"
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_block_without_from_writes_as_one_row() {
    let mut wrong = Vec::new();
    for (statement, want) in [
        (
            "INSERT INTO t VALUES (10, (SELECT 2))",
            "1:1 2:2 3:3 4:NULL 10:2",
        ),
        (
            "INSERT INTO t VALUES (11, (SELECT COUNT(*))), (12, (SELECT 2 WHERE 1 = 0))",
            "1:1 2:2 3:3 4:NULL 11:1 12:NULL",
        ),
        (
            "UPDATE t SET a = (SELECT t.a + 10) WHERE id = 1",
            "1:11 2:2 3:3 4:NULL",
        ),
        (
            "UPDATE t SET a = (SELECT 5) WHERE id = (SELECT 2)",
            "1:1 2:5 3:3 4:NULL",
        ),
        ("DELETE FROM t WHERE id = (SELECT 3)", "1:1 2:2 4:NULL"),
    ] {
        for context in ["autocommit", "transaction"] {
            let db = Database::open("memory://prepared_fromless_writes").unwrap();
            setup(&db);
            let read = "SELECT id, a FROM t ORDER BY id";
            let got = if context == "autocommit" {
                db.execute(statement, ())
                    .map_err(|e| e.to_string())
                    .and_then(|_| answer(db.query(read, ())))
            } else {
                let mut tx = db.begin().unwrap();
                let got = tx
                    .execute(statement, ())
                    .map_err(|e| e.to_string())
                    .and_then(|_| answer(tx.query(read, ())));
                tx.commit().unwrap();
                got
            };
            let got = got.map(|rows| {
                rows.iter()
                    .map(|r| {
                        r.replace("Integer(", "")
                            .replace("Null(Integer)", "NULL")
                            .replace(')', "")
                            .replacen(',', ":", 1)
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            });
            if got.as_deref() != Ok(want) {
                wrong.push(format!("{statement} ({context}): {got:?}, want {want:?}"));
            }
            db.execute("DROP TABLE t", ()).unwrap();
            db.execute("DROP TABLE u", ()).unwrap();
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_block_without_from_reads_again_the_tables_of_its_nested_blocks() {
    let db = Database::open("memory://prepared_fromless_kept").unwrap();
    setup(&db);
    let sql = "SELECT id FROM t WHERE a = (SELECT (SELECT MAX(x) FROM u)) ORDER BY id";
    assert_eq!(
        answer(db.query(sql, ())),
        Ok(vec!["Integer(2)".to_string()])
    );
    db.execute("INSERT INTO u VALUES (3, 3)", ()).unwrap();
    assert_eq!(
        answer(db.query(sql, ())),
        Ok(vec!["Integer(3)".to_string()])
    );
}
