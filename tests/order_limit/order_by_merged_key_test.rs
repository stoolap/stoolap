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

//! A sort key no SELECT item carries sorts by the input column it names,
//! though an item aliases another column of that name: a join's merged
//! column beside an aliased column of its pair

use stoolap::Database;

fn fill(db: &Database) {
    for sql in [
        "CREATE TABLE t (k INTEGER, a INTEGER)",
        "INSERT INTO t VALUES (1, 30), (2, 20), (5, 10)",
        "CREATE TABLE u (k INTEGER, b INTEGER)",
        "INSERT INTO u VALUES (2, 200), (3, 300)",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

/// Each row as its columns' values, NULL as None
fn rows(db: &Database, sql: &str, prepared: bool) -> Result<Vec<Vec<Option<i64>>>, String> {
    let result = match prepared {
        true => db.prepare(sql).and_then(|p| p.query((0,))),
        false => db.query(sql, (0,)),
    }
    .map_err(|e| e.to_string())?;
    let width = result.columns().len();
    result
        .map(|row| {
            let row = row.map_err(|e| e.to_string())?;
            (0..width)
                .map(|i| row.get::<Option<i64>>(i).map_err(|e| e.to_string()))
                .collect()
        })
        .collect()
}

const N: Option<i64> = None;

fn s(v: i64) -> Option<i64> {
    Some(v)
}

#[test]
fn a_sort_key_no_item_carries_sorts_by_its_input_column() {
    let dir = tempfile::tempdir().unwrap();
    let sealed = Database::open(&format!(
        "file://{}?checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    fill(&sealed);
    sealed.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let hot = Database::open("memory://order_by_merged_key").unwrap();
    fill(&hot);
    let cases: Vec<(&str, Vec<Vec<Option<i64>>>)> = vec![
        // The merged column, which no item reads
        (
            "SELECT u.k AS uk, a FROM t LEFT JOIN u USING (k) WHERE a > $1 ORDER BY k DESC",
            vec![vec![N, s(10)], vec![s(2), s(20)], vec![N, s(30)]],
        ),
        (
            "SELECT u.k AS uk, a FROM t LEFT JOIN u USING (k) WHERE a > $1 \
             ORDER BY k DESC LIMIT 2",
            vec![vec![N, s(10)], vec![s(2), s(20)]],
        ),
        (
            "SELECT t.k AS tk, b FROM t RIGHT JOIN u USING (k) WHERE b > $1 ORDER BY k DESC",
            vec![vec![N, s(300)], vec![s(2), s(200)]],
        ),
        (
            "SELECT t.k AS tk, u.k AS uk FROM t FULL JOIN u USING (k) WHERE $1 = 0 \
             ORDER BY k DESC",
            vec![
                vec![s(5), N],
                vec![N, s(3)],
                vec![s(2), s(2)],
                vec![s(1), N],
            ],
        ),
        (
            "SELECT u.k AS uk, a FROM t LEFT JOIN u USING (k) WHERE a > $1 ORDER BY t.k DESC",
            vec![vec![N, s(10)], vec![s(2), s(20)], vec![N, s(30)]],
        ),
        // An item that reads the merged column carries it
        (
            "SELECT t.k AS tk, a FROM t LEFT JOIN u USING (k) WHERE a > $1 ORDER BY k DESC",
            vec![vec![s(5), s(10)], vec![s(2), s(20)], vec![s(1), s(30)]],
        ),
        // One table: the aliased column, and an alias over another column
        (
            "SELECT a AS x FROM t WHERE a > $1 ORDER BY a DESC",
            vec![vec![s(30)], vec![s(20)], vec![s(10)]],
        ),
        (
            "SELECT k AS a FROM t WHERE a > $1 ORDER BY a",
            vec![vec![s(1)], vec![s(2)], vec![s(5)]],
        ),
    ];
    let mut wrong = Vec::new();
    for (context, db) in [("hot", &hot), ("sealed", &sealed)] {
        for (sql, want) in &cases {
            for prepared in [false, true] {
                let got = rows(db, sql, prepared);
                if got.as_ref() != Ok(want) {
                    wrong.push(format!(
                        "{context}, prepared {prepared}: {sql}\n  got  {got:?}\n  want {want:?}"
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
