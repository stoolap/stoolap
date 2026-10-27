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

//! A view's query reads the tables and views it names, never a CTE of the
//! statement that reads the view, whichever statement read the view first

use stoolap::Database;

/// t: 1, 2. u: 1. Each view reads u where a CTE named u would differ
const SETUP: &[&str] = &[
    "CREATE TABLE t (id INTEGER PRIMARY KEY)",
    "CREATE TABLE u (id INTEGER PRIMARY KEY)",
    "INSERT INTO t VALUES (1), (2)",
    "INSERT INTO u VALUES (1)",
    "CREATE VIEW v_from AS SELECT id FROM u",
    "CREATE VIEW v_in AS SELECT id FROM t WHERE id IN (SELECT id FROM u)",
    "CREATE VIEW v_exists AS SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.id = t.id)",
    "CREATE VIEW v_join AS SELECT t.id FROM t JOIN u ON u.id = t.id \
     WHERE t.id <= (SELECT MAX(id) FROM u)",
];

const VIEWS: &[&str] = &["v_from", "v_in", "v_exists", "v_join"];

/// Each way a statement reads `{v}`, with a CTE u of the row 2
const READS: &[&str] = &[
    "WITH u(id) AS (SELECT 2) SELECT id FROM {v}",
    "WITH u(id) AS (SELECT 2) SELECT x.id FROM t JOIN {v} AS x ON x.id = t.id",
    "WITH u(id) AS (SELECT 2) SELECT id FROM t WHERE EXISTS (SELECT 1 FROM {v} AS x WHERE x.id = t.id)",
    "WITH u(id) AS (SELECT 2) SELECT id FROM t WHERE id IN (SELECT id FROM {v}) \
     AND id > (SELECT MIN(id) - 2 FROM u)",
];

fn ids(db: &Database, sql: &str, prepared: bool) -> Result<Vec<i64>, String> {
    let rows = match prepared {
        true => db
            .prepare(sql)
            .and_then(|statement| statement.query(()))
            .map_err(|e| e.to_string())?,
        false => db.query(sql, ()).map_err(|e| e.to_string())?,
    };
    let mut out = rows
        .map(|row| {
            row.and_then(|row| row.get::<i64>(0))
                .map_err(|e| e.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    out.sort();
    Ok(out)
}

fn open(name: &str, sealed: bool, dir: &tempfile::TempDir) -> Database {
    let db = match sealed {
        true => Database::open(&format!("file://{}/{name}", dir.path().display())).unwrap(),
        false => Database::open(&format!("memory://view_cte_scope_{name}")).unwrap(),
    };
    for sql in SETUP {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

#[test]
fn a_view_reads_its_own_tables_under_a_cte_of_the_same_name() {
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        for prepared in [false, true] {
            for read_first in [true, false] {
                for (index, view) in VIEWS.iter().enumerate() {
                    let dir = tempfile::tempdir().unwrap();
                    let name = format!("{sealed}_{prepared}_{read_first}_{index}");
                    let db = open(&name, sealed, &dir);
                    let plain = format!("SELECT id FROM {view}");
                    if read_first {
                        ids(&db, &plain, prepared).unwrap();
                    }
                    for read in READS {
                        let sql = read.replace("{v}", view);
                        let got = ids(&db, &sql, prepared);
                        if got != Ok(vec![1]) {
                            wrong.push(format!(
                                "sealed {sealed}, prepared {prepared}, read first {read_first}: \
                                 {sql}: {got:?}"
                            ));
                        }
                    }
                    let got = ids(&db, &plain, prepared);
                    if got != Ok(vec![1]) {
                        wrong.push(format!(
                            "sealed {sealed}, prepared {prepared}, read first {read_first}: \
                             {plain} after: {got:?}"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
