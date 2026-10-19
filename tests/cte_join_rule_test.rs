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

//! A join with a CTE on a side answers as the same join with a derived
//! table there: USING and NATURAL, joins inside a CTE body that read
//! earlier CTEs, and the column order of `*`

use stoolap::Database;

fn tables(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();
    db.execute("CREATE TABLE r (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO r VALUES (1), (3)", ()).unwrap();
    db
}

fn answer(db: &Database, sql: &str) -> Result<(Vec<String>, Vec<String>), String> {
    let result = db.query(sql, ()).map_err(|e| e.to_string())?;
    let columns = result.columns().to_vec();
    let mut rows: Vec<String> = result
        .map(|row| format!("{:?}", row.unwrap().as_row().as_slice()))
        .collect();
    rows.sort();
    Ok((columns, rows))
}

#[test]
fn a_cte_join_answers_as_the_derived_table_join() {
    let db = tables("cte_join_rule_derived");
    let cte = "WITH c(id) AS (SELECT id FROM r) ";
    let derived = "(SELECT id FROM r) c";
    let mut wrong = Vec::new();
    for shape in [
        "SELECT * FROM {c} JOIN t USING (id)",
        "SELECT * FROM {c} NATURAL JOIN t",
        "SELECT * FROM {c} LEFT JOIN t USING (id)",
        "SELECT * FROM t JOIN {c} ON c.id = t.id",
        "SELECT * FROM t LEFT JOIN {c} ON c.id = t.id",
        "SELECT COUNT(*) FROM {c} JOIN t ON c.id = t.id AND t.v > 10",
    ] {
        let with_cte = format!("{cte}{}", shape.replace("{c}", "c"));
        let with_derived = shape.replace("{c}", derived);
        let (got, want) = (answer(&db, &with_cte), answer(&db, &with_derived));
        if got != want {
            wrong.push(format!("{with_cte}\n  got  {got:?}\n  want {want:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_cte_join_returns_the_stated_rows() {
    let db = tables("cte_join_rule_stated");
    let mut wrong = Vec::new();
    for (sql, columns, rows) in [
        (
            "WITH c(id) AS (SELECT id FROM r) SELECT * FROM c JOIN t USING (id)",
            vec!["id", "t.v"],
            vec!["[Integer(1), Integer(10)]", "[Integer(3), Integer(30)]"],
        ),
        (
            "WITH c(id) AS (SELECT id FROM r) SELECT * FROM t JOIN c ON c.id = t.id",
            vec!["t.id", "t.v", "c.id"],
            vec![
                "[Integer(1), Integer(10), Integer(1)]",
                "[Integer(3), Integer(30), Integer(3)]",
            ],
        ),
        (
            "WITH a(id) AS (SELECT 1), a2(id) AS (SELECT 1), \
             b AS (SELECT a.id, t.v FROM a JOIN t ON a.id = t.id JOIN a2 ON a2.id = t.id) \
             SELECT * FROM b",
            vec!["id", "v"],
            vec!["[Integer(1), Integer(10)]"],
        ),
        (
            "WITH a(id) AS (SELECT 1), a2(id) AS (SELECT 1), \
             b AS (SELECT (SELECT COUNT(*) FROM a2) AS n, t.v FROM a JOIN t ON a.id = t.id) \
             SELECT * FROM b",
            vec!["n", "v"],
            vec!["[Integer(1), Integer(10)]"],
        ),
    ] {
        let want: Result<(Vec<String>, Vec<String>), String> = Ok((
            columns.iter().map(|c| c.to_string()).collect(),
            rows.iter().map(|r| r.to_string()).collect(),
        ));
        let got = answer(&db, sql);
        if got != want {
            wrong.push(format!("{sql}\n  got  {got:?}\n  want {want:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
