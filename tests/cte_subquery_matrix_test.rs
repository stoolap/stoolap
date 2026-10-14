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

//! Subqueries and CTEs give the rows computed here from the tables, under
//! each outer ORDER BY, LIMIT, OFFSET and DISTINCT, for a parameter run
//! again with other values, by a query and by a prepared statement

use std::collections::{BTreeMap, BTreeSet};

use stoolap::{Database, Value};

const L_ROWS: i64 = 30;
const R_ROWS: i64 = 50;

#[derive(Clone, Copy)]
struct L {
    id: i64,
    k: i64,
    g: i64,
}

#[derive(Clone, Copy)]
struct R {
    id: i64,
    k: i64,
    w: i64,
    /// NULL every eleventh row
    n: Option<i64>,
}

fn ls() -> Vec<L> {
    (1..=L_ROWS)
        .map(|id| L {
            id,
            k: id % 7,
            g: (id - 1) / 10,
        })
        .collect()
}

fn rs() -> Vec<R> {
    (1..=R_ROWS)
        .map(|id| R {
            id,
            k: id % 7,
            w: id % 5,
            n: (id % 11 != 0).then_some(id % 7),
        })
        .collect()
}

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE l (id INTEGER PRIMARY KEY, k INTEGER, g INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE r (id INTEGER PRIMARY KEY, k INTEGER, w INTEGER, n INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX r_k ON r (k)", ()).unwrap();
    db.execute("CREATE INDEX l_k ON l (k)", ()).unwrap();
    db.execute(
        &format!(
            "INSERT INTO l SELECT g.value, g.value % 7, (g.value - 1) / 10 \
             FROM generate_series(1, {L_ROWS}) g"
        ),
        (),
    )
    .unwrap();
    db.execute(
        &format!(
            "INSERT INTO r SELECT g.value, g.value % 7, g.value % 5, \
             CASE WHEN g.value % 11 = 0 THEN NULL ELSE g.value % 7 END \
             FROM generate_series(1, {R_ROWS}) g"
        ),
        (),
    )
    .unwrap();
    db
}

/// SQL's three-valued truth: None is UNKNOWN
type Truth = Option<bool>;

/// `x IN (members)`
fn in_set(x: i64, members: &[Option<i64>]) -> Truth {
    if members.contains(&Some(x)) {
        Some(true)
    } else if members.contains(&None) {
        None
    } else {
        Some(false)
    }
}

fn not(t: Truth) -> Truth {
    t.map(|b| !b)
}

/// A WHERE over l's rows, with a parameter: the SQL text using `$1`, and
/// its truth for one l row at one value of the parameter
struct Predicate {
    sql: &'static str,
    truth: fn(&L, &[R], i64) -> Truth,
}

fn r_where(r: &[R], c: i64) -> impl Iterator<Item = &R> {
    r.iter().filter(move |r| r.w < c)
}

const PREDICATES: &[Predicate] = &[
    Predicate {
        sql: "l.k IN (SELECT r.k FROM r WHERE r.w < $1)",
        truth: |l, r, c| in_set(l.k, &r_where(r, c).map(|r| Some(r.k)).collect::<Vec<_>>()),
    },
    Predicate {
        sql: "l.id IN (SELECT r.k FROM r WHERE r.w < $1)",
        truth: |l, r, c| in_set(l.id, &r_where(r, c).map(|r| Some(r.k)).collect::<Vec<_>>()),
    },
    Predicate {
        sql: "l.g IN (SELECT r.w FROM r WHERE r.w < $1)",
        truth: |l, r, c| in_set(l.g, &r_where(r, c).map(|r| Some(r.w)).collect::<Vec<_>>()),
    },
    Predicate {
        sql: "l.k NOT IN (SELECT r.k FROM r WHERE r.w < $1)",
        truth: |l, r, c| {
            not(in_set(
                l.k,
                &r_where(r, c).map(|r| Some(r.k)).collect::<Vec<_>>(),
            ))
        },
    },
    Predicate {
        sql: "l.id NOT IN (SELECT r.k FROM r WHERE r.w < $1)",
        truth: |l, r, c| {
            not(in_set(
                l.id,
                &r_where(r, c).map(|r| Some(r.k)).collect::<Vec<_>>(),
            ))
        },
    },
    Predicate {
        sql: "l.k NOT IN (SELECT r.n FROM r WHERE r.w < $1)",
        truth: |l, r, c| not(in_set(l.k, &r_where(r, c).map(|r| r.n).collect::<Vec<_>>())),
    },
    Predicate {
        sql: "l.k IN (SELECT r.n FROM r WHERE r.w < $1)",
        truth: |l, r, c| in_set(l.k, &r_where(r, c).map(|r| r.n).collect::<Vec<_>>()),
    },
    Predicate {
        sql: "EXISTS (SELECT 1 FROM r WHERE r.k = l.k AND r.w < $1)",
        truth: |l, r, c| Some(r_where(r, c).any(|r| r.k == l.k)),
    },
    Predicate {
        sql: "NOT EXISTS (SELECT 1 FROM r WHERE r.k = l.k AND r.w < $1)",
        truth: |l, r, c| Some(!r_where(r, c).any(|r| r.k == l.k)),
    },
    Predicate {
        sql: "EXISTS (SELECT 1 FROM r WHERE r.id = l.k AND r.w < $1)",
        truth: |l, r, c| Some(r_where(r, c).any(|r| r.id == l.k)),
    },
    Predicate {
        sql: "NOT EXISTS (SELECT 1 FROM r WHERE r.id = l.k AND r.w < $1)",
        truth: |l, r, c| Some(!r_where(r, c).any(|r| r.id == l.k)),
    },
    Predicate {
        sql: "EXISTS (SELECT 1 FROM r WHERE r.w = l.g AND r.w < $1)",
        truth: |l, r, c| Some(r_where(r, c).any(|r| r.w == l.g)),
    },
    Predicate {
        sql: "NOT EXISTS (SELECT 1 FROM r WHERE r.n = l.k AND r.w < $1)",
        truth: |l, r, c| Some(!r_where(r, c).any(|r| r.n == Some(l.k))),
    },
    Predicate {
        sql: "l.g < (SELECT MAX(r.w) FROM r WHERE r.k = l.k AND r.w < $1)",
        truth: |l, r, c| {
            r_where(r, c)
                .filter(|r| r.k == l.k)
                .map(|r| r.w)
                .max()
                .map(|max| l.g < max)
        },
    },
    Predicate {
        sql: "l.k > (SELECT MIN(r.k) FROM r WHERE r.w < $1)",
        truth: |l, r, c| r_where(r, c).map(|r| r.k).min().map(|min| l.k > min),
    },
    Predicate {
        sql: "(SELECT COUNT(*) FROM r WHERE r.k = l.k AND r.w < $1) > 2",
        truth: |l, r, c| Some(r_where(r, c).filter(|r| r.k == l.k).count() > 2),
    },
];

/// An outer query over l's rows that pass a WHERE: the SQL around it, and
/// the rows it gives from those l rows; true when its order is fixed, and
/// the LIMIT to check a subset by when it is not
struct Outer {
    select: &'static str,
    tail: &'static str,
    ordered: bool,
    limit: Option<usize>,
    rows: fn(&[L]) -> Vec<String>,
}

const OUTERS: &[Outer] = &[
    Outer {
        select: "l.id",
        tail: "",
        ordered: false,
        limit: None,
        rows: |ls| ls.iter().map(|l| l.id.to_string()).collect(),
    },
    Outer {
        select: "l.id",
        tail: " ORDER BY l.id LIMIT 4 OFFSET 1",
        ordered: true,
        limit: None,
        rows: |ls| {
            ls.iter()
                .skip(1)
                .take(4)
                .map(|l| l.id.to_string())
                .collect()
        },
    },
    Outer {
        select: "l.id",
        tail: " ORDER BY l.id DESC LIMIT 3",
        ordered: true,
        limit: None,
        rows: |ls| ls.iter().rev().take(3).map(|l| l.id.to_string()).collect(),
    },
    Outer {
        select: "l.id",
        tail: " LIMIT 2",
        ordered: false,
        limit: Some(2),
        rows: |ls| ls.iter().map(|l| l.id.to_string()).collect(),
    },
    Outer {
        select: "DISTINCT l.g",
        tail: " ORDER BY l.g",
        ordered: true,
        limit: None,
        rows: |ls| {
            let groups: BTreeSet<i64> = ls.iter().map(|l| l.g).collect();
            groups.into_iter().map(|g| g.to_string()).collect()
        },
    },
    Outer {
        select: "l.g, COUNT(*)",
        tail: " GROUP BY l.g ORDER BY l.g",
        ordered: true,
        limit: None,
        rows: |ls| {
            let mut counts: BTreeMap<i64, usize> = BTreeMap::new();
            for l in ls {
                *counts.entry(l.g).or_default() += 1;
            }
            counts
                .into_iter()
                .map(|(g, n)| format!("{g}|{n}"))
                .collect()
        },
    },
];

/// Values of the parameter, run one after the other on one connection
const VALUES: [i64; 5] = [2, 4, 0, 5, 2];

fn cells(row: &stoolap::ResultRow) -> String {
    (0..row.len())
        .map(|i| match row.get_value(i) {
            Some(Value::Integer(n)) => n.to_string(),
            Some(Value::Float(f)) => f.to_string(),
            Some(Value::Null(_)) => "NULL".to_string(),
            other => format!("{other:?}"),
        })
        .collect::<Vec<_>>()
        .join("|")
}

/// Whether `got` is what `expected` and the outer query's order allow
fn agrees(outer: &Outer, expected: &[String], got: &[String]) -> bool {
    if outer.ordered {
        return got == expected;
    }
    let mut pool = expected.to_vec();
    let all_in = got
        .iter()
        .all(|row| match pool.iter().position(|r| r == row) {
            Some(at) => {
                pool.swap_remove(at);
                true
            }
            None => false,
        });
    let wanted = outer
        .limit
        .map_or(expected.len(), |n| n.min(expected.len()));
    all_in && got.len() == wanted
}

#[test]
fn test_subqueries_give_the_rows_of_the_tables() {
    let db = setup("cte_subquery_matrix_subqueries");
    let (l, r) = (ls(), rs());
    let mut found = Vec::new();
    let mut checked = 0;
    for predicate in PREDICATES {
        for outer in OUTERS {
            let sql = format!(
                "SELECT {} FROM l WHERE {}{}",
                outer.select, predicate.sql, outer.tail
            );
            let prepared = db.prepare(&sql).unwrap();
            for c in VALUES {
                let kept: Vec<L> = l
                    .iter()
                    .copied()
                    .filter(|row| (predicate.truth)(row, &r, c) == Some(true))
                    .collect();
                let expected = (outer.rows)(&kept);
                for (how, got) in [
                    (
                        "query",
                        db.query(&sql, (c,))
                            .map_err(|e| e.to_string())
                            .map(|rows| rows.map(|row| cells(&row.unwrap())).collect::<Vec<_>>()),
                    ),
                    (
                        "prepared",
                        prepared
                            .query((c,))
                            .map_err(|e| e.to_string())
                            .map(|rows| rows.map(|row| cells(&row.unwrap())).collect::<Vec<_>>()),
                    ),
                ] {
                    checked += 1;
                    match got {
                        Err(error) => found.push(format!("{sql} [$1 = {c}, {how}]\n    error: {error}")),
                        Ok(got) if !agrees(outer, &expected, &got) => found.push(format!(
                            "{sql} [$1 = {c}, {how}]\n    expected {expected:?}\n    got      {got:?}"
                        )),
                        Ok(_) => {}
                    }
                }
            }
        }
    }
    assert!(
        found.is_empty(),
        "{} of {checked} statements are not the tables' rows:\n{}",
        found.len(),
        found.join("\n")
    );
}

/// A CTE statement over l and r with a parameter, and its rows computed
/// from the tables at one value of it
struct CteCase {
    sql: &'static str,
    ordered: bool,
    rows: fn(&[L], &[R], i64) -> Vec<String>,
}

fn ids(rows: impl Iterator<Item = i64>) -> Vec<String> {
    rows.map(|id| id.to_string()).collect()
}

const CTE_CASES: &[CteCase] = &[
    CteCase {
        sql: "WITH c AS (SELECT k FROM r WHERE w < $1) \
              SELECT l.id FROM l WHERE l.k IN (SELECT k FROM c) ORDER BY l.id",
        ordered: true,
        rows: |l, r, p| {
            let ks: BTreeSet<i64> = r_where(r, p).map(|r| r.k).collect();
            ids(l.iter().filter(|l| ks.contains(&l.k)).map(|l| l.id))
        },
    },
    CteCase {
        sql: "WITH c AS (SELECT k FROM r WHERE w < $1) \
              SELECT l.id FROM l WHERE EXISTS (SELECT 1 FROM c WHERE c.k = l.k) ORDER BY l.id",
        ordered: true,
        rows: |l, r, p| {
            let ks: BTreeSet<i64> = r_where(r, p).map(|r| r.k).collect();
            ids(l.iter().filter(|l| ks.contains(&l.k)).map(|l| l.id))
        },
    },
    CteCase {
        sql: "WITH c AS (SELECT k FROM r WHERE w < $1) \
              SELECT l.id FROM l WHERE NOT EXISTS (SELECT 1 FROM c WHERE c.k = l.k) ORDER BY l.id",
        ordered: true,
        rows: |l, r, p| {
            let ks: BTreeSet<i64> = r_where(r, p).map(|r| r.k).collect();
            ids(l.iter().filter(|l| !ks.contains(&l.k)).map(|l| l.id))
        },
    },
    CteCase {
        sql: "WITH c AS (SELECT n FROM r WHERE w < $1) \
              SELECT l.id FROM l WHERE l.k NOT IN (SELECT n FROM c) ORDER BY l.id",
        ordered: true,
        rows: |l, r, p| {
            let members: Vec<Option<i64>> = r_where(r, p).map(|r| r.n).collect();
            ids(l
                .iter()
                .filter(|l| not(in_set(l.k, &members)) == Some(true))
                .map(|l| l.id))
        },
    },
    CteCase {
        sql: "WITH c AS (SELECT k, COUNT(*) AS n FROM r WHERE w < $1 GROUP BY k) \
              SELECT l.id, c.n FROM l JOIN c ON l.k = c.k ORDER BY l.id",
        ordered: true,
        rows: |l, r, p| {
            let mut counts: BTreeMap<i64, usize> = BTreeMap::new();
            for r in r_where(r, p) {
                *counts.entry(r.k).or_default() += 1;
            }
            l.iter()
                .filter_map(|l| counts.get(&l.k).map(|n| format!("{}|{n}", l.id)))
                .collect()
        },
    },
    CteCase {
        sql: "WITH c AS (SELECT id, k FROM r WHERE w < $1 ORDER BY id LIMIT 5) \
              SELECT l.id, c.id FROM l JOIN c ON l.k = c.k ORDER BY l.id, c.id",
        ordered: true,
        rows: |l, r, p| {
            let first: Vec<&R> = r_where(r, p).take(5).collect();
            let mut out = Vec::new();
            for l in l {
                for r in &first {
                    if r.k == l.k {
                        out.push(format!("{}|{}", l.id, r.id));
                    }
                }
            }
            out
        },
    },
    CteCase {
        sql: "WITH c AS (SELECT id, k FROM r WHERE w < $1) \
              SELECT a.id, b.id FROM c a JOIN c b ON a.k = b.k AND a.id < b.id \
              ORDER BY a.id, b.id LIMIT 6",
        ordered: true,
        rows: |_, r, p| {
            let c: Vec<&R> = r_where(r, p).collect();
            let mut out = Vec::new();
            for a in &c {
                for b in &c {
                    if a.k == b.k && a.id < b.id {
                        out.push((a.id, b.id));
                    }
                }
            }
            out.sort();
            out.into_iter()
                .take(6)
                .map(|(a, b)| format!("{a}|{b}"))
                .collect()
        },
    },
    CteCase {
        sql:
            "WITH a AS (SELECT id, k, w FROM r WHERE w < $1), b AS (SELECT k FROM a WHERE id > 20) \
              SELECT DISTINCT l.g FROM l JOIN b ON l.k = b.k ORDER BY l.g",
        ordered: true,
        rows: |l, r, p| {
            let ks: BTreeSet<i64> = r_where(r, p).filter(|r| r.id > 20).map(|r| r.k).collect();
            let groups: BTreeSet<i64> = l
                .iter()
                .filter(|l| ks.contains(&l.k))
                .map(|l| l.g)
                .collect();
            ids(groups.into_iter())
        },
    },
    CteCase {
        sql: "WITH RECURSIVE s(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM s WHERE n < $1) \
              SELECT l.id FROM l JOIN s ON l.k = s.n ORDER BY l.id",
        ordered: true,
        rows: |l, _, p| {
            let top = p.max(1);
            ids(l.iter().filter(|l| l.k >= 1 && l.k <= top).map(|l| l.id))
        },
    },
    CteCase {
        sql: "WITH c AS (SELECT k, w FROM r WHERE w < $1) \
              SELECT l.g, COUNT(*) FROM l JOIN c ON l.k = c.k GROUP BY l.g ORDER BY l.g",
        ordered: true,
        rows: |l, r, p| {
            let c: Vec<&R> = r_where(r, p).collect();
            let mut counts: BTreeMap<i64, usize> = BTreeMap::new();
            for l in l {
                let n = c.iter().filter(|r| r.k == l.k).count();
                if n > 0 {
                    *counts.entry(l.g).or_default() += n;
                }
            }
            counts
                .into_iter()
                .map(|(g, n)| format!("{g}|{n}"))
                .collect()
        },
    },
    CteCase {
        sql: "WITH c AS (SELECT k FROM r WHERE w < $1) \
              SELECT l.id FROM l JOIN c ON l.k = c.k LIMIT 3",
        ordered: false,
        rows: |l, r, p| {
            let c: Vec<&R> = r_where(r, p).collect();
            let mut out = Vec::new();
            for l in l {
                for r in &c {
                    if r.k == l.k {
                        out.push(l.id.to_string());
                    }
                }
            }
            out
        },
    },
    CteCase {
        sql: "WITH c AS (SELECT id, COUNT(*) AS n FROM r WHERE w < $1 GROUP BY id) \
              SELECT l.id, c.n FROM l JOIN c ON l.id + 40 = c.id LIMIT 3",
        ordered: false,
        rows: |l, r, p| {
            let c: BTreeSet<i64> = r_where(r, p).map(|r| r.id).collect();
            l.iter()
                .filter(|l| c.contains(&(l.id + 40)))
                .map(|l| format!("{}|1", l.id))
                .collect()
        },
    },
    CteCase {
        sql: "WITH c AS (SELECT id, COUNT(*) AS n FROM r WHERE w < $1 GROUP BY id) \
              SELECT l.id, c.n FROM l LEFT JOIN c ON l.id + 40 = c.id LIMIT 3",
        ordered: false,
        rows: |l, r, p| {
            let c: BTreeSet<i64> = r_where(r, p).map(|r| r.id).collect();
            l.iter()
                .map(|l| match c.contains(&(l.id + 40)) {
                    true => format!("{}|1", l.id),
                    false => format!("{}|NULL", l.id),
                })
                .collect()
        },
    },
];

#[test]
fn test_ctes_give_the_rows_of_the_tables() {
    let db = setup("cte_subquery_matrix_ctes");
    let (l, r) = (ls(), rs());
    let mut found = Vec::new();
    let mut checked = 0;
    for case in CTE_CASES {
        let prepared = db.prepare(case.sql).unwrap();
        for c in VALUES {
            let all = (case.rows)(&l, &r, c);
            let outer = Outer {
                select: "",
                tail: "",
                ordered: case.ordered,
                limit: case.sql.contains("LIMIT 3").then_some(3),
                rows: |_| Vec::new(),
            };
            for (how, got) in [
                (
                    "query",
                    db.query(case.sql, (c,))
                        .map_err(|e| e.to_string())
                        .map(|rows| rows.map(|row| cells(&row.unwrap())).collect::<Vec<_>>()),
                ),
                (
                    "prepared",
                    prepared
                        .query((c,))
                        .map_err(|e| e.to_string())
                        .map(|rows| rows.map(|row| cells(&row.unwrap())).collect::<Vec<_>>()),
                ),
            ] {
                checked += 1;
                match got {
                    Err(error) => found.push(format!(
                        "{} [$1 = {c}, {how}]\n    error: {error}",
                        case.sql
                    )),
                    Ok(got) if !agrees(&outer, &all, &got) => found.push(format!(
                        "{} [$1 = {c}, {how}]\n    expected {all:?}\n    got      {got:?}",
                        case.sql
                    )),
                    Ok(_) => {}
                }
            }
        }
    }
    assert!(
        found.is_empty(),
        "{} of {checked} statements are not the tables' rows:\n{}",
        found.len(),
        found.join("\n")
    );
}
