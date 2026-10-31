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

use stoolap::Database;

fn setup(name: &str, dir: &tempfile::TempDir, sealed: bool) -> Database {
    let dsn = match sealed {
        true => format!("file://{}/db", dir.path().display()),
        false => format!("memory://{name}"),
    };
    let db = Database::open(&dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
        "INSERT INTO t VALUES (1, 2), (2, 3), (3, 4), (4, 5)",
        "INSERT INTO u VALUES (1, 1, 5), (2, 2, 7), (3, 3, 1), (4, 5, 2)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn rows(db: &Database, sql: &str) -> Result<Vec<Vec<i64>>, String> {
    let result = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in result {
        let row = row.map_err(|e| e.to_string())?;
        out.push((0..row.len()).map(|i| row.get::<i64>(i).unwrap()).collect());
    }
    Ok(out)
}

/// Each shape and its rows, a CTE body or recursive member holding a
/// subquery over the CTE it reads
const SHAPES: &[(&str, &[&[i64]])] = &[
    (
        "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c \
         WHERE n < (SELECT COUNT(*) FROM u)) SELECT n FROM c ORDER BY n",
        &[&[1], &[2], &[3], &[4]],
    ),
    (
        "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c \
         WHERE EXISTS (SELECT 1 FROM u WHERE u.k = c.n)) SELECT n FROM c ORDER BY n",
        &[&[1], &[2], &[3], &[4]],
    ),
    (
        "WITH RECURSIVE c(n, m) AS (SELECT 1, 0 UNION ALL SELECT n + 1, \
         (SELECT MAX(v) FROM u WHERE u.k = c.n) FROM c WHERE n < 4) SELECT n, m FROM c ORDER BY n",
        &[&[1, 0], &[2, 5], &[3, 7], &[4, 1]],
    ),
    (
        "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c \
         WHERE n IN (SELECT k FROM u)) SELECT n FROM c ORDER BY n",
        &[&[1], &[2], &[3], &[4]],
    ),
    (
        "WITH a AS (SELECT id, k FROM t), b AS (SELECT id FROM a \
         WHERE k > (SELECT MIN(v) FROM u WHERE v > 1)) SELECT id FROM b ORDER BY id",
        &[&[2], &[3], &[4]],
    ),
    (
        "WITH a AS (SELECT id, k FROM t), b AS (SELECT id, \
         (SELECT COUNT(*) FROM u WHERE u.k = a.k) AS c FROM a) SELECT id, c FROM b ORDER BY id",
        &[&[1, 1], &[2, 1], &[3, 0], &[4, 1]],
    ),
];

#[test]
fn a_cte_body_holding_a_subquery_answers_over_its_cte() {
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = setup("cte_body_blocks_answers", &dir, sealed);
        for (sql, want) in SHAPES {
            let want: Vec<Vec<i64>> = want.iter().map(|row| row.to_vec()).collect();
            assert_eq!(rows(&db, sql), Ok(want), "sealed {sealed}: {sql}");
        }
        let error = "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM c \
                     WHERE n < (SELECT k FROM u)) SELECT n FROM c";
        let got = rows(&db, error).unwrap_err();
        assert!(got.contains("more than one row"), "sealed {sealed}: {got}");
    }
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_cte_body_holding_a_subquery_runs_no_general_path_mechanism() {
    use stoolap::test_failpoints::general_path_total;
    let dir = tempfile::tempdir().unwrap();
    let db = setup("cte_body_blocks_general_path", &dir, false);
    for (sql, _) in SHAPES {
        let before = general_path_total();
        rows(&db, sql).unwrap();
        assert_eq!(general_path_total(), before, "{sql}");
    }
}
