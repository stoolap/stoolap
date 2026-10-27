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

//! A join whose side is a NATURAL or USING join runs its subqueries as the
//! plan's blocks and reads that side as a join at the top does: `t.id` and
//! `u.id` are each their table's column, the bare `id` the merged one, and a
//! star shows the pair once

use stoolap::Database;

const SETUP: &[&str] = &[
    "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
    "INSERT INTO t VALUES (1, 1, 10), (2, 2, 20), (3, 4, 30), (4, NULL, 40), (5, 3, 50), (6, 1, 60)",
    "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
    "INSERT INTO u VALUES (1, 7), (2, 8), (3, 9), (5, NULL), (9, 1)",
    "CREATE TABLE w (x INTEGER, y INTEGER)",
    "INSERT INTO w VALUES (1, 5), (2, 6), (9, 7)",
];

fn hot(name: &str) -> Database {
    let db = Database::open(&format!("memory://joined_using_sides_{name}")).unwrap();
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    db
}

fn sealed(dir: &tempfile::TempDir) -> Database {
    let path = dir.path().join("db");
    let db = Database::open(&format!("file://{}?checkpoint_interval=0", path.display())).unwrap();
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db
}

/// The joined side, as written and as the ON join it stands for, with the
/// kinds of join it is written as
const MERGING: &[(&str, &str, &[&str])] = &[
    (
        "t {j} u USING (id)",
        "t {j} u ON u.id = t.id",
        &["JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN"],
    ),
    (
        "t NATURAL {j} u",
        "t {j} u ON u.id = t.id",
        &["JOIN", "LEFT JOIN", "RIGHT JOIN"],
    ),
];

/// The join around the side, `{m}` the side, and the columns its other
/// tables add to a star
const AROUND: &[(&str, &str)] = &[
    ("{m} JOIN u AS r ON r.id = t.a", "r.id, r.x"),
    ("{m} LEFT JOIN u AS r ON r.id = t.a", "r.id, r.x"),
    (
        "{m} LEFT JOIN u AS r ON r.id = t.a AND r.id IN (SELECT x FROM w)",
        "r.id, r.x",
    ),
    (
        "{m} LEFT JOIN u AS r ON r.id = t.a LEFT JOIN u AS p ON p.id = t.id",
        "r.id, r.x, p.id, p.x",
    ),
];

/// Each statement over `{f}`, `{id}` the merged column; the bare name stands
/// for the left table's column, as at the top
const SHAPES: &[&str] = &[
    "SELECT {id}, t.id, u.id, u.x, r.x FROM {f} WHERE COALESCE(u.x, 0) < (SELECT MAX(y) FROM w) + 100",
    "SELECT * FROM {f} WHERE COALESCE(t.b, 0) < (SELECT MAX(y) FROM w) * 100",
    "SELECT u.*, t.* FROM {f} WHERE COALESCE(t.b, 0) < (SELECT MAX(y) FROM w) * 100",
    "SELECT {id}, COUNT(*), SUM(t.b) FROM {f} WHERE COALESCE(u.x, 0) < (SELECT MAX(y) FROM w) + 100 GROUP BY {id}",
    "SELECT u.id, COUNT(*) FROM {f} WHERE COALESCE(u.x, 0) < (SELECT MAX(y) FROM w) + 100 GROUP BY u.id",
    "SELECT {id}, t.id, u.id FROM {f} WHERE EXISTS (SELECT 1 FROM w WHERE w.x = u.id OR w.x = t.a)",
    "SELECT {id}, u.id, r.x FROM {f} WHERE r.x IS NULL OR r.x < (SELECT MAX(y) FROM w) + 100 ORDER BY u.id, {id}",
];

fn answer(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let rows = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len())
                .map(|i| match row.get::<Option<String>>(i) {
                    Ok(value) => value.unwrap_or_else(|| "NULL".into()),
                    Err(e) => format!("<{e}>"),
                })
                .collect::<Vec<_>>()
                .join("|"),
        );
    }
    out.sort();
    Ok(out)
}

fn prepared_line(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .find(|line| line.trim_start().starts_with("Prepared:"))
        .map(|line| line.trim().to_string())
        .unwrap_or_default()
}

fn check(db: &Database, context: &str, wrong: &mut Vec<String>) -> usize {
    let mut count = 0;
    for (merging, as_on, kinds) in MERGING {
        for kind in kinds.iter() {
            let (merging, as_on) = (merging.replace("{j}", kind), as_on.replace("{j}", kind));
            for (around, others) in AROUND {
                let from = around.replace("{m}", &merging);
                let from_on = around.replace("{m}", &as_on);
                for shape in SHAPES {
                    let sql = shape.replace("{f}", &from).replace("{id}", "id");
                    let star = format!("t.id, t.a, t.b, u.x, {others}");
                    let want = shape
                        .replace("SELECT *", &format!("SELECT {star}"))
                        .replace("{f}", &from_on)
                        .replace("{id}", "t.id");
                    count += 1;
                    let (got, want) = (answer(db, &sql), answer(db, &want));
                    if got.is_err() || got != want {
                        wrong.push(format!("{context}: {sql}: {got:?}, want {want:?}"));
                    }
                    let line = prepared_line(db, &sql);
                    if line != "Prepared: yes" {
                        wrong.push(format!("{context}: {sql}: EXPLAIN {line}"));
                    }
                }
            }
        }
    }
    count
}

#[test]
fn a_joined_side_that_merges_columns_runs_its_blocks_and_answers_as_at_the_top() {
    let mut wrong = Vec::new();
    let count = check(&hot("matrix"), "hot", &mut wrong);
    let dir = tempfile::tempdir().unwrap();
    check(&sealed(&dir), "sealed", &mut wrong);
    assert!(count > 150, "{count} statements");
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_star_over_a_joined_side_that_merges_names_the_pair_once() {
    let db = hot("names");
    let sql = "SELECT * FROM t LEFT JOIN u USING (id) JOIN u AS r ON r.id = t.a \
               WHERE t.b < (SELECT MAX(y) FROM w) * 100";
    assert_eq!(prepared_line(&db, sql), "Prepared: yes");
    let columns: Vec<String> = db.query(sql, ()).unwrap().columns().to_vec();
    assert_eq!(columns, ["id", "t.a", "t.b", "u.x", "r.id", "r.x"]);
}
