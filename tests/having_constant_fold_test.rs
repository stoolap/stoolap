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

//! A HAVING compiled for one execution folds the parameters and constants
//! it compares a group with, and answers as a WHERE over the same groups

use stoolap::{Database, Value};

fn fill(db: &Database) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        "INSERT INTO t SELECT g.value, g.value % 7, g.value % 13 FROM generate_series(1, 120) g",
        "INSERT INTO t VALUES (500, 9, NULL)",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

fn rows(db: &Database, sql: &str, params: Vec<Value>) -> Result<Vec<String>, String> {
    let result = db.query(sql, params).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in result {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len())
                .map(|i| format!("{:?}", row.get_value(i).unwrap()))
                .collect::<Vec<_>>()
                .join("|"),
        );
    }
    Ok(out)
}

#[test]
fn a_having_comparison_answers_as_a_where_over_its_groups() {
    let db = Database::open("memory://having_constant_fold_where").unwrap();
    fill(&db);
    let operands: &[(&str, Option<Value>)] = &[
        ("100", None),
        ("100.5", None),
        ("'100'", None),
        ("NULL", None),
        ("$1", Some(Value::Integer(100))),
        ("$1", Some(Value::Float(100.5))),
        ("$1", Some(Value::text("100"))),
        ("$1", Some(Value::null_unknown())),
        ("$1 * 10", Some(Value::Integer(10))),
        ("$1 + 0.5", Some(Value::Integer(100))),
    ];
    let mut wrong = Vec::new();
    for op in [">", "=", "<>", "<="] {
        for (operand, param) in operands {
            let params: Vec<Value> = param.iter().cloned().collect();
            for test in [
                format!("SUM(b) {op} {operand}"),
                format!("{operand} {op} SUM(b)"),
            ] {
                let having = format!("SELECT a, SUM(b) FROM t GROUP BY a HAVING {test} ORDER BY a");
                let where_ = format!(
                    "SELECT a, s FROM (SELECT a, SUM(b) AS s FROM t GROUP BY a) g \
                     WHERE {} ORDER BY a",
                    test.replace("SUM(b)", "s")
                );
                let got = rows(&db, &having, params.clone());
                let want = rows(&db, &where_, params.clone());
                if got != want {
                    wrong.push(format!("{having} {params:?}: {got:?}, the WHERE {want:?}"));
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

#[test]
fn a_folded_having_raises_its_error_only_for_a_group_and_reads_each_execution_s_parameters() {
    let db = Database::open("memory://having_constant_fold_errors").unwrap();
    fill(&db);
    let sql = "SELECT a, SUM(b) FROM t WHERE a < $2 GROUP BY a HAVING SUM(b) > $1 + 1 ORDER BY a";
    let max = Value::Integer(i64::MAX);
    assert_eq!(
        rows(&db, sql, vec![max.clone(), Value::Integer(0)]),
        Ok(Vec::new()),
        "no group reaches the overflow"
    );
    let got = rows(&db, sql, vec![max, Value::Integer(9)]);
    assert!(
        got.as_ref().is_err_and(|e| e.contains("overflow")),
        "a group reaches it: {got:?}"
    );
    let low = rows(&db, sql, vec![Value::Integer(0), Value::Integer(9)]).unwrap();
    let high = rows(&db, sql, vec![Value::Integer(110), Value::Integer(9)]).unwrap();
    assert_eq!(low.len(), 7, "{low:?}");
    assert!(high.len() < low.len(), "{high:?}");
}

#[test]
fn a_fold_keeps_the_group_item_it_holds() {
    let db = Database::open("memory://having_constant_fold_rollup").unwrap();
    db.execute("CREATE TABLE r (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO r VALUES (1)", ()).unwrap();
    let seven = || vec![Value::Integer(7)];
    for (sql, params) in [
        (
            "SELECT ($1 + 0) + 1, COUNT(*) FROM r GROUP BY ROLLUP($1 + 0) ORDER BY 1",
            seven(),
        ),
        (
            "SELECT (7 + 0) + 1, COUNT(*) FROM r GROUP BY ROLLUP(7 + 0) ORDER BY 1",
            Vec::new(),
        ),
    ] {
        let mut got = rows(&db, sql, params).unwrap();
        got.sort();
        assert!(
            got.len() == 2
                && got[0] == "Integer(8)|Integer(1)"
                && got[1].starts_with("Null(")
                && got[1].ends_with("|Integer(1)"),
            "{sql}: the total row reads the rolled up item as NULL, got {got:?}"
        );
    }
    let sql = "SELECT COUNT(*) FROM r GROUP BY ROLLUP($1 + 0) HAVING ($1 + 0) + 1 IS NULL";
    assert_eq!(
        rows(&db, sql, seven()),
        Ok(vec!["Integer(1)".to_string()]),
        "{sql}: only the total row"
    );
}
