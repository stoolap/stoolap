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

//! A prepared first branch of a set operation that removes duplicates keeps
//! one row of each value as it reads, and the set operation answers as it
//! does over the general path's branch

use stoolap::Database;

mod oracle;

/// Each statement's answer, recorded when the general path gave the same
const RECORDED: oracle::Answers = answers!("prepared_set_branch");

const PREPARED: &str = "(SELECT v FROM u WHERE u.id = 1)";

fn setup(db: &Database) {
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b TEXT, c FLOAT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE w (id INTEGER PRIMARY KEY, a FLOAT, b TEXT)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE u (id INTEGER PRIMARY KEY, v FLOAT)", ())
        .unwrap();
    db.execute("INSERT INTO u VALUES (1, 0.5)", ()).unwrap();
    db.execute(
        "INSERT INTO t SELECT value, CASE WHEN value % 7 = 0 THEN NULL ELSE value % 5 END, \
         'b' || (value % 3), (value * 37 % 100) / 10.0 FROM generate_series(1, 400)",
        (),
    )
    .unwrap();
    // FLOAT values equal to INTEGER ones are one key with them
    db.execute(
        "INSERT INTO w SELECT value, (value % 7) * 1.0, 'b' || (value % 4) \
         FROM generate_series(1, 60)",
        (),
    )
    .unwrap();
}

/// The rows, sorted unless the statement orders them: an ORDER BY past the
/// last parenthesis, which the subquery's own is not
fn answer(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let rows = db.query(sql, (2,)).map_err(|e| e.to_string())?;
    let mut found = rows
        .map(|row| {
            let row = row.map_err(|e| e.to_string())?;
            Ok(format!(
                "{:?}",
                (0..row.len()).map(|i| row.get_value(i)).collect::<Vec<_>>()
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    if !sql
        .rsplit(')')
        .next()
        .is_some_and(|tail| tail.contains("ORDER BY"))
    {
        found.sort();
    }
    Ok(found)
}

fn check(db: &Database, phase: &str, wrong: &mut Vec<String>) {
    for shape in [
        "SELECT a FROM t WHERE c > {sq} UNION SELECT a FROM t WHERE id < 10",
        "SELECT a, b FROM t WHERE c > {sq} UNION SELECT a, b FROM t WHERE id < 30",
        "SELECT a FROM t WHERE c > {sq} INTERSECT SELECT a FROM t WHERE id < 12",
        "SELECT a, b FROM t WHERE c > {sq} EXCEPT SELECT a, b FROM t WHERE id < 40",
        "SELECT a FROM t WHERE c > {sq} EXCEPT SELECT a FROM t WHERE a > 2",
        "SELECT a FROM t WHERE c > {sq} UNION SELECT a FROM w",
        "SELECT a FROM t WHERE c > {sq} INTERSECT SELECT a FROM w",
        "SELECT a FROM t WHERE c > {sq} UNION SELECT a FROM t WHERE id < 5 \
         UNION ALL SELECT a FROM t WHERE id < 5",
        "SELECT a FROM t WHERE c > {sq} UNION SELECT a FROM t WHERE id < 5 \
         EXCEPT ALL SELECT a FROM t WHERE id < 3",
        "SELECT a FROM t WHERE c > {sq} UNION ALL SELECT a FROM t WHERE id < 5",
        "SELECT a FROM t WHERE c > {sq} EXCEPT ALL SELECT a FROM t WHERE id < 50",
        "SELECT a, b FROM t WHERE c > {sq} UNION SELECT a, b FROM t WHERE id < 10 \
         ORDER BY 1, 2 LIMIT 5",
        "SELECT a, b FROM t WHERE c > {sq} UNION SELECT a, b FROM t WHERE id < 10 \
         ORDER BY 2 DESC, 1 NULLS FIRST LIMIT 4 OFFSET 2",
        "SELECT a FROM t WHERE c > {sq} UNION SELECT a FROM t WHERE id < 10 LIMIT 3",
        "SELECT a FROM t WHERE c > {sq} INTERSECT SELECT a FROM t WHERE a > 3 LIMIT 1",
        "SELECT a FROM t WHERE c > {sq} EXCEPT SELECT a FROM t WHERE a < 3 LIMIT 2",
        "SELECT DISTINCT a FROM t WHERE c > {sq} UNION SELECT 1 UNION ALL SELECT 1",
        "SELECT DISTINCT a FROM t WHERE c > {sq} EXCEPT SELECT 2 UNION ALL SELECT a FROM w",
        "SELECT a FROM t WHERE c > {sq} AND id > $1 INTERSECT SELECT a FROM t WHERE id < 20",
        "SELECT a + 1 AS a1 FROM t WHERE c > {sq} UNION SELECT a FROM t WHERE id < 10",
    ] {
        let prepared = shape.replace("{sq}", PREPARED);
        // A LIMIT without ORDER BY may keep any rows of the whole answer,
        // which is the one recorded
        let unordered_limit = prepared
            .rsplit_once(" LIMIT ")
            .filter(|_| !shape.contains("ORDER BY"));
        let whole = unordered_limit.map_or(prepared.as_str(), |(all, _)| all);
        let all = answer(db, whole);
        let key = format!("{phase}: {whole}");
        let general = || oracle::general_answer(whole, |twin| answer(db, twin));
        if let Some(why) = oracle::check_confirmed(&RECORDED, &key, &all, general) {
            wrong.push(why);
        }
        if let Some((_, limit)) = unordered_limit {
            let limit: usize = limit.parse().unwrap();
            match (&answer(db, &prepared), &all) {
                (Ok(got), Ok(all))
                    if got.len() == limit.min(all.len()) && got.iter().all(|r| all.contains(r)) => {
                }
                (got, _) => wrong.push(format!("{phase}: {shape}\n  got  {got:?}")),
            }
        }
    }
}

#[test]
fn a_deduplicating_set_operation_answers_as_over_the_general_branch() {
    let mut wrong = Vec::new();
    let db = Database::open("memory://prepared_set_branch_hot").unwrap();
    setup(&db);
    check(&db, "hot", &mut wrong);

    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_on_close=off", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    setup(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("UPDATE t SET a = 9 WHERE id % 50 = 0", ())
        .unwrap();
    check(&db, "sealed with hot updates", &mut wrong);
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
