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

//! Every comparison of FLOAT values follows one total order: NaN equals NaN
//! and is greater than every number; NULL is unknown. Each operator answers
//! alike in every context a predicate is read in

use std::cmp::Ordering;

use stoolap::core::Value;
use stoolap::Database;

const NAN: &str = "CAST('NaN' AS FLOAT)";

/// (id, p, q); None is NULL
const ROWS: &[(i64, Option<f64>, Option<f64>)] = &[
    (1, Some(f64::NAN), Some(1.0)),
    (2, Some(1.0), Some(f64::NAN)),
    (3, Some(2.0), Some(1.0)),
    (4, Some(f64::NAN), Some(f64::NAN)),
    (5, None, Some(1.0)),
    (6, Some(-1.0), Some(2.0)),
];

const OPS: &[&str] = &["=", "<>", "<", "<=", ">", ">="];

fn total(a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => a.partial_cmp(&b).unwrap(),
    }
}

fn holds(op: &str, a: Option<f64>, b: Option<f64>) -> bool {
    let (Some(a), Some(b)) = (a, b) else {
        return false;
    };
    let o = total(a, b);
    match op {
        "=" => o == Ordering::Equal,
        "<>" => o != Ordering::Equal,
        "<" => o == Ordering::Less,
        "<=" => o != Ordering::Greater,
        ">" => o == Ordering::Greater,
        _ => o != Ordering::Less,
    }
}

fn lit(c: f64) -> String {
    if c.is_nan() {
        NAN.to_string()
    } else {
        format!("{c:.1}")
    }
}

fn where_ids(f: impl Fn(&(i64, Option<f64>, Option<f64>)) -> bool) -> Vec<i64> {
    ROWS.iter().filter(|r| f(r)).map(|r| r.0).collect()
}

/// (predicate over `p` and `q`, the ids the total order selects)
fn predicates() -> Vec<(String, Vec<i64>)> {
    let mut out = Vec::new();
    for c in [1.0, f64::NAN] {
        for op in OPS {
            out.push((
                format!("p {op} {}", lit(c)),
                where_ids(|r| holds(op, r.1, Some(c))),
            ));
            out.push((
                format!("{} {op} p", lit(c)),
                where_ids(|r| holds(op, Some(c), r.1)),
            ));
        }
        out.push((
            format!("p BETWEEN 0.0 AND {}", lit(c)),
            where_ids(|r| holds(">=", r.1, Some(0.0)) && holds("<=", r.1, Some(c))),
        ));
        out.push((
            format!("p NOT BETWEEN 0.0 AND {}", lit(c)),
            where_ids(|r| {
                r.1.is_some() && !(holds(">=", r.1, Some(0.0)) && holds("<=", r.1, Some(c)))
            }),
        ));
        out.push((
            format!("p IN ({}, 5.0)", lit(c)),
            where_ids(|r| holds("=", r.1, Some(c)) || holds("=", r.1, Some(5.0))),
        ));
        out.push((
            format!("p NOT IN ({}, 5.0)", lit(c)),
            where_ids(|r| {
                r.1.is_some() && !holds("=", r.1, Some(c)) && !holds("=", r.1, Some(5.0))
            }),
        ));
    }
    for op in OPS {
        out.push((format!("p {op} q"), where_ids(|r| holds(op, r.1, r.2))));
    }
    out
}

fn ids(db: &Database, sql: &str, params: Vec<Value>) -> Result<Vec<i64>, String> {
    let mut v = Vec::new();
    for r in db.query(sql, params).map_err(|e| e.to_string())? {
        v.push(
            r.map_err(|e| e.to_string())?
                .get::<i64>(0)
                .map_err(|e| e.to_string())?,
        );
    }
    v.sort_unstable();
    Ok(v)
}

fn open(db: Database, index: &str, seal: bool) -> Database {
    db.execute(
        "CREATE TABLE f (id INTEGER PRIMARY KEY, p FLOAT, q FLOAT, g INTEGER)",
        (),
    )
    .unwrap();
    if !index.is_empty() {
        db.execute(&format!("CREATE INDEX f_p ON f(p) USING {index}"), ())
            .unwrap();
    }
    for (id, p, q) in ROWS {
        let v = |x: &Option<f64>| x.map_or("NULL".to_string(), lit);
        db.execute(
            &format!(
                "INSERT INTO f VALUES ({id}, {}, {}, {})",
                v(p),
                v(q),
                id % 2
            ),
            (),
        )
        .unwrap();
    }
    if seal {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

/// Every predicate in every form on `db`, against the total order
fn check(db: &Database, ctx: &str, all_forms: bool, wrong: &mut Vec<String>) {
    for (pred, want) in predicates() {
        let mut forms = vec![format!("SELECT id FROM f WHERE {pred}")];
        if all_forms {
            forms.push(format!(
                "SELECT id FROM f WHERE {}",
                pred.replace("p ", "(p + 0.0) ").replace(" p", " (p + 0.0)")
            ));
            forms.push(format!(
                "SELECT id FROM f WHERE CASE WHEN {pred} THEN 1 ELSE 0 END = 1"
            ));
            forms.push(format!(
                "SELECT id FROM (SELECT id, ({pred}) AS hit FROM f) s WHERE hit"
            ));
        }
        for sql in forms {
            let got = ids(db, &sql, vec![]);
            if got.as_ref() != Ok(&want) {
                wrong.push(format!("{ctx}: {sql}: got {got:?}, want {want:?}"));
            }
            if sql.contains(NAN) {
                let sql = sql.replace(NAN, "$1");
                let got = ids(db, &sql, vec![Value::Float(f64::NAN)]);
                if got.as_ref() != Ok(&want) {
                    wrong.push(format!("{ctx}: {sql} with NaN: got {got:?}, want {want:?}"));
                }
            }
        }
    }
}

#[test]
fn every_comparison_follows_the_total_order_in_memory() {
    let mut wrong = Vec::new();
    for (n, index) in ["", "BTREE", "HASH"].into_iter().enumerate() {
        let db = open(
            Database::open(&format!("memory://nan_total_order_{n}")).unwrap(),
            index,
            false,
        );
        check(
            &db,
            &format!("memory {index}"),
            index.is_empty(),
            &mut wrong,
        );
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn every_comparison_follows_the_total_order_hot_and_sealed() {
    let mut wrong = Vec::new();
    for seal in [false, true] {
        for index in ["", "BTREE", "HASH"] {
            let dir = tempfile::tempdir().unwrap();
            let db = open(
                Database::open(&format!(
                    "file://{}?checkpoint_interval=0",
                    dir.path().display()
                ))
                .unwrap(),
                index,
                seal,
            );
            check(
                &db,
                &format!("file {index} sealed {seal}"),
                index.is_empty(),
                &mut wrong,
            );
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
fn row_groups_and_volumes_keep_the_nan_rows_their_bounds_leave_out() {
    const ROWS: i64 = 140_000;
    const ALL_NAN: i64 = 65_536;
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE f (id INTEGER PRIMARY KEY, p FLOAT, g INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        &format!(
            "INSERT INTO f SELECT g.value, CASE WHEN g.value <= {ALL_NAN} \
             OR g.value % 1000 = 0 THEN {NAN} ELSE g.value * 1.0 END, g.value % 2 \
             FROM generate_series(1, {ROWS}) g"
        ),
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let nan = |g: i64| g <= ALL_NAN || g % 1000 == 0;
    let count = |f: &dyn Fn(i64) -> bool| (1..=ROWS).filter(|&g| f(g)).count() as i64;
    let mut wrong = Vec::new();
    for (pred, want) in [
        ("p > 1e9".to_string(), count(&nan)),
        (format!("p >= {NAN}"), count(&nan)),
        (format!("p = {NAN}"), count(&nan)),
        (format!("p <= {NAN}"), ROWS),
        (format!("p < {NAN}"), count(&|g| !nan(g))),
        (
            "p > 139000.0".to_string(),
            count(&|g| nan(g) || g > 139_000),
        ),
        (format!("p BETWEEN 0.0 AND {NAN}"), ROWS),
        ("p NOT BETWEEN 0.0 AND 1e9".to_string(), count(&nan)),
        ("p < 70000.5".to_string(), count(&|g| !nan(g) && g < 70_001)),
    ] {
        for sql in [
            format!("SELECT COUNT(*) FROM f WHERE {pred}"),
            format!("SELECT SUM(g) + COUNT(*) - SUM(g) FROM f WHERE {pred}"),
            format!("SELECT COUNT(*) FROM (SELECT id FROM f WHERE {pred}) s"),
        ] {
            let got = db
                .query(&sql, ())
                .map_err(|e| e.to_string())
                .and_then(|mut r| {
                    r.next()
                        .unwrap()
                        .map_err(|e| e.to_string())?
                        .get::<i64>(0)
                        .map_err(|e| e.to_string())
                });
            if got != Ok(want) {
                wrong.push(format!("{sql}: got {got:?}, want {want}"));
            }
        }
    }
    for (op, want) in [
        ("=", count(&nan)),
        (">=", count(&nan)),
        ("<=", ROWS),
        ("<", count(&|g| !nan(g))),
    ] {
        for sql in [
            format!("SELECT COUNT(*) FROM f WHERE p {op} $1"),
            format!("SELECT COUNT(*) FROM (SELECT id FROM f WHERE p {op} $1) s"),
        ] {
            let got = db
                .query(&sql, vec![Value::Float(f64::NAN)])
                .map_err(|e| e.to_string())
                .and_then(|mut r| {
                    r.next()
                        .unwrap()
                        .map_err(|e| e.to_string())?
                        .get::<i64>(0)
                        .map_err(|e| e.to_string())
                });
            if got != Ok(want) {
                wrong.push(format!("{sql} with NaN: got {got:?}, want {want}"));
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
fn sealed_indexes_answer_as_the_scan_of_an_unindexed_twin() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    for (table, index) in [
        ("t", Some("CREATE INDEX t_gp ON t(g, p)")),
        ("s", Some("CREATE INDEX s_p ON s(p)")),
        ("u", None),
    ] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, g INTEGER, p FLOAT)"),
            (),
        )
        .unwrap();
        if let Some(index) = index {
            db.execute(index, ()).unwrap();
        }
        db.execute(
            &format!(
                "INSERT INTO {table} SELECT g.value, g.value % 3, CASE WHEN g.value % 50 = 0 \
                 THEN {NAN} ELSE g.value * 0.5 END FROM generate_series(1, 3000) g"
            ),
            (),
        )
        .unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let mut wrong = Vec::new();
    for pred in [
        "g = 1 AND p > 1e300",
        "g = 1 AND p >= 1400.0",
        "g = 1 AND p = $1",
        "g = 1 AND p >= $1",
        "g = 1 AND p < $1 AND p > 1400.0",
        "g = 1 AND p BETWEEN 1400.0 AND $1",
        "p > 1e300",
        "p >= 1499.0",
        "p = $1",
        "p > 1400.0 AND p <= $1",
    ] {
        let params = if pred.contains("$1") {
            vec![Value::Float(f64::NAN)]
        } else {
            vec![]
        };
        let want = ids(
            &db,
            &format!("SELECT id FROM u WHERE {pred}"),
            params.clone(),
        );
        for table in ["t", "s"] {
            let sql = format!("SELECT id FROM {table} WHERE {pred}");
            let got = ids(&db, &sql, params.clone());
            if got != want {
                wrong.push(format!("{sql}: got {got:?}, want {want:?}"));
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
fn parameters_constants_and_having_follow_the_total_order() {
    let db = open(
        Database::open("memory://nan_total_order_params").unwrap(),
        "",
        false,
    );
    let mut wrong = Vec::new();
    for op in OPS {
        let want = where_ids(|r| holds(op, r.1, Some(f64::NAN)));
        let sql = format!("SELECT id FROM f WHERE p {op} $1");
        let got = ids(&db, &sql, vec![Value::Float(f64::NAN)]);
        if got.as_ref() != Ok(&want) {
            wrong.push(format!("{sql} with NaN: got {got:?}, want {want:?}"));
        }
    }
    for (sql, want) in [
        (format!("SELECT {NAN} = {NAN}"), true),
        (format!("SELECT {NAN} <> {NAN}"), false),
        (format!("SELECT {NAN} > 1.0"), true),
        (format!("SELECT {NAN} >= {NAN}"), true),
        (format!("SELECT {NAN} < 1.0"), false),
        (format!("SELECT 1.0 < {NAN}"), true),
        (format!("SELECT {NAN} BETWEEN 0.0 AND {NAN}"), true),
        (format!("SELECT {NAN} IN (1.0, {NAN})"), true),
    ] {
        let got = db
            .query(&sql, ())
            .map_err(|e| e.to_string())
            .map(|mut r| r.next().unwrap().unwrap().get::<bool>(0).unwrap());
        if got != Ok(want) {
            wrong.push(format!("{sql}: got {got:?}, want {want}"));
        }
    }
    // g = 1 holds ids 1, 3, 5: MAX(p) is NaN; g = 0 holds 2, 4, 6: MAX(p) is NaN
    for (sql, want) in [
        ("SELECT g FROM f GROUP BY g HAVING MAX(p) > 1.0", vec![0, 1]),
        ("SELECT g FROM f GROUP BY g HAVING MIN(p) > 1.0", vec![1]),
        (
            "SELECT g FROM f GROUP BY g HAVING MAX(p) = MAX(p)",
            vec![0, 1],
        ),
        (
            "SELECT id FROM f WHERE (SELECT MAX(p) FROM f) > 1.0",
            vec![1, 2, 3, 4, 5, 6],
        ),
        (
            "SELECT id FROM f WHERE (SELECT MAX(p) FROM f) = (SELECT MAX(q) FROM f)",
            vec![1, 2, 3, 4, 5, 6],
        ),
        (
            "SELECT id FROM f WHERE (SELECT MAX(p) FROM f) < 1.0",
            vec![],
        ),
        (
            "SELECT id FROM f WHERE 2.5 < ANY (SELECT p FROM f WHERE p IS NOT NULL)",
            vec![1, 2, 3, 4, 5, 6],
        ),
        (
            "SELECT id FROM f WHERE p >= ALL (SELECT q FROM f)",
            vec![1, 4],
        ),
        (
            "SELECT id FROM f WHERE p = ANY (SELECT q FROM f)",
            vec![1, 2, 3, 4],
        ),
        (
            "SELECT id FROM f WHERE p > ANY (SELECT q FROM f)",
            vec![1, 3, 4],
        ),
    ] {
        let got = ids(&db, sql, vec![]);
        if got.as_ref() != Ok(&want) {
            wrong.push(format!("{sql}: got {got:?}, want {want:?}"));
        }
    }
    for (sql, want) in [
        (
            "SELECT STRING_AGG(CAST(id AS TEXT), ',' ORDER BY p) FROM f WHERE id <> 4",
            "6,2,3,1,5",
        ),
        ("SELECT CAST(MEDIAN(q) AS TEXT) FROM f WHERE id <> 1", "2"),
    ] {
        let got = db
            .query(sql, ())
            .map_err(|e| e.to_string())
            .and_then(|mut r| {
                r.next()
                    .unwrap()
                    .map_err(|e| e.to_string())?
                    .get::<String>(0)
                    .map_err(|e| e.to_string())
            });
        if got.as_deref() != Ok(want) {
            wrong.push(format!("{sql}: got {got:?}, want {want}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
