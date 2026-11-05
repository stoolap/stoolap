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

//! Every join path gives the rows the join holds, under each select list,
//! ORDER BY, DISTINCT, window, aggregate, LIMIT and OFFSET, with the build
//! side on either side. The rows are computed here from the tables, and
//! checked against the same join written so that no path takes its keys

use std::collections::{BTreeMap, BTreeSet};

use stoolap::{Database, Value};

const L_ROWS: i64 = 30;
const R_ROWS: i64 = 50;

/// l: id, k = id % 7, g = (id - 1) / 10, so the first joined rows share
/// a g and a LIMIT cut before DISTINCT leaves too few groups
fn l_row(id: i64) -> [i64; 3] {
    [id, id % 7, (id - 1) / 10]
}

/// r: id, k = id % 7, w = id % 5; k is indexed, w is not
fn r_row(id: i64) -> [i64; 3] {
    [id, id % 7, id % 5]
}

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE l (id INTEGER PRIMARY KEY, k INTEGER, g INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE r (id INTEGER PRIMARY KEY, k INTEGER, w INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX r_k ON r (k)", ()).unwrap();
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
            "INSERT INTO r SELECT g.value, g.value % 7, g.value % 5 \
             FROM generate_series(1, {R_ROWS}) g"
        ),
        (),
    )
    .unwrap();
    db
}

/// A join source: its WITH prefix, the FROM clause, the same FROM with
/// keys no path takes, and the key columns (positions in l_row, r_row)
struct Source {
    with: &'static str,
    from: &'static str,
    reference: &'static str,
    l_key: usize,
    r_key: usize,
    left: bool,
}

const CTES: &str = "WITH lx AS (SELECT id, k, g FROM l), rx AS (SELECT id, k, w FROM r) ";
// a CTE reading another CTE is not inlined, so these joins read materialized rows
const MATERIALIZED: &str = "WITH lb AS (SELECT id, k, g FROM l), lx AS (SELECT id, k, g FROM lb), \
     rx AS (SELECT id, k, w FROM r) ";

const SOURCES: [Source; 17] = [
    // materialized CTEs: hash join, index nested loop on the PK, both sides, LEFT
    Source {
        with: MATERIALIZED,
        from: "l JOIN rx r ON l.k = r.w",
        reference: "l JOIN rx r ON l.k + 0 = r.w + 0",
        l_key: 1,
        r_key: 2,
        left: false,
    },
    Source {
        with: MATERIALIZED,
        from: "rx r JOIN l ON r.k = l.id",
        reference: "rx r JOIN l ON r.k + 0 = l.id + 0",
        l_key: 0,
        r_key: 1,
        left: false,
    },
    Source {
        with: MATERIALIZED,
        from: "lx l JOIN rx r ON l.g = r.w",
        reference: "lx l JOIN rx r ON l.g + 0 = r.w + 0",
        l_key: 2,
        r_key: 2,
        left: false,
    },
    Source {
        with: MATERIALIZED,
        from: "lx l LEFT JOIN rx r ON l.g = r.w",
        reference: "lx l LEFT JOIN rx r ON l.g + 0 = r.w + 0",
        l_key: 2,
        r_key: 2,
        left: true,
    },
    // keys at different positions on each side, l.k and r.w
    Source {
        with: "",
        from: "l JOIN r ON l.k = r.w",
        reference: "l JOIN r ON l.k + 0 = r.w + 0",
        l_key: 1,
        r_key: 2,
        left: false,
    },
    Source {
        with: CTES,
        from: "lx l JOIN rx r ON l.k = r.w",
        reference: "lx l JOIN rx r ON l.k + 0 = r.w + 0",
        l_key: 1,
        r_key: 2,
        left: false,
    },
    // index joins: r.k is indexed, r.id is the primary key
    Source {
        with: "",
        from: "l JOIN r ON l.k = r.k",
        reference: "l JOIN r ON l.k + 0 = r.k + 0",
        l_key: 1,
        r_key: 1,
        left: false,
    },
    Source {
        with: "",
        from: "l JOIN r ON l.k = r.id",
        reference: "l JOIN r ON l.k + 0 = r.id + 0",
        l_key: 1,
        r_key: 0,
        left: false,
    },
    Source {
        with: "",
        from: "l LEFT JOIN r ON l.k = r.id",
        reference: "l LEFT JOIN r ON l.k + 0 = r.id + 0",
        l_key: 1,
        r_key: 0,
        left: true,
    },
    Source {
        with: CTES,
        from: "lx l JOIN rx r ON l.k = r.k",
        reference: "lx l JOIN rx r ON l.k + 0 = r.k + 0",
        l_key: 1,
        r_key: 1,
        left: false,
    },
    Source {
        with: "",
        from: "(SELECT id, k, g FROM l) l JOIN (SELECT id, k, w FROM r) r ON l.k = r.k",
        reference:
            "(SELECT id, k, g FROM l) l JOIN (SELECT id, k, w FROM r) r ON l.k + 0 = r.k + 0",
        l_key: 1,
        r_key: 1,
        left: false,
    },
    // hash, streaming and standard joins: r.w has no index
    Source {
        with: "",
        from: "l JOIN r ON l.g = r.w",
        reference: "l JOIN r ON l.g + 0 = r.w + 0",
        l_key: 2,
        r_key: 2,
        left: false,
    },
    Source {
        with: "",
        from: "r JOIN l ON r.w = l.g",
        reference: "r JOIN l ON r.w + 0 = l.g + 0",
        l_key: 2,
        r_key: 2,
        left: false,
    },
    Source {
        with: "",
        from: "l LEFT JOIN r ON l.g = r.w",
        reference: "l LEFT JOIN r ON l.g + 0 = r.w + 0",
        l_key: 2,
        r_key: 2,
        left: true,
    },
    Source {
        with: CTES,
        from: "lx l JOIN rx r ON l.g = r.w",
        reference: "lx l JOIN rx r ON l.g + 0 = r.w + 0",
        l_key: 2,
        r_key: 2,
        left: false,
    },
    Source {
        with: CTES,
        from: "rx r JOIN lx l ON r.w = l.g",
        reference: "rx r JOIN lx l ON r.w + 0 = l.g + 0",
        l_key: 2,
        r_key: 2,
        left: false,
    },
    Source {
        with: "",
        from: "(SELECT id, k, g FROM l) l JOIN (SELECT id, k, w FROM r) r ON l.g = r.w",
        reference:
            "(SELECT id, k, g FROM l) l JOIN (SELECT id, k, w FROM r) r ON l.g + 0 = r.w + 0",
        l_key: 2,
        r_key: 2,
        left: false,
    },
];

/// The joined pairs of a source, computed from the tables
/// An l row and the r row it joined, None for a LEFT JOIN row without one
type Pair = ([i64; 3], Option<[i64; 3]>);

fn joined(source: &Source) -> Vec<Pair> {
    let mut pairs = Vec::new();
    for lid in 1..=L_ROWS {
        let l = l_row(lid);
        let mut matched = false;
        for rid in 1..=R_ROWS {
            let r = r_row(rid);
            if l[source.l_key] == r[source.r_key] {
                pairs.push((l, Some(r)));
                matched = true;
            }
        }
        if source.left && !matched {
            pairs.push((l, None));
        }
    }
    pairs
}

fn cell(value: Option<i64>) -> String {
    value.map_or_else(|| "NULL".to_string(), |v| v.to_string())
}

/// A select list, its ORDER BY clauses (true when the ORDER BY fixes the
/// order of the output rows), and the rows it holds, computed from the
/// joined pairs; None when the computation would need NULL ordering
struct Shape {
    select: &'static str,
    group_by: &'static str,
    orders: &'static [(&'static str, bool)],
    rows: fn(&[Pair]) -> Option<Vec<String>>,
}

fn pair_rows(pairs: &[Pair]) -> Option<Vec<String>> {
    Some(
        pairs
            .iter()
            .map(|(l, r)| format!("{}|{}", l[0], cell(r.map(|r| r[0]))))
            .collect(),
    )
}

fn distinct_g(pairs: &[Pair]) -> Option<Vec<String>> {
    let groups: BTreeSet<i64> = pairs.iter().map(|(l, _)| l[2]).collect();
    Some(groups.into_iter().map(|g| g.to_string()).collect())
}

/// DISTINCT ON (l.g) l.g, r.id: the largest r.id of each g, under either
/// ORDER BY tried (r.id DESC, or r.w then r.id)
fn distinct_on_desc(pairs: &[Pair]) -> Option<Vec<String>> {
    let mut best: BTreeMap<i64, i64> = BTreeMap::new();
    for (l, r) in pairs {
        let r = (*r)?;
        let at = best.entry(l[2]).or_insert(r[0]);
        *at = (*at).max(r[0]);
    }
    Some(
        best.into_iter()
            .map(|(g, id)| format!("{g}|{id}"))
            .collect(),
    )
}

fn distinct_on_by_w(pairs: &[Pair]) -> Option<Vec<String>> {
    let mut best: BTreeMap<i64, (i64, i64)> = BTreeMap::new();
    for (l, r) in pairs {
        let r = (*r)?;
        let at = best.entry(l[2]).or_insert((r[2], r[0]));
        *at = (*at).min((r[2], r[0]));
    }
    Some(
        best.into_iter()
            .map(|(g, (_, id))| format!("{g}|{id}"))
            .collect(),
    )
}

fn count_over(pairs: &[Pair]) -> Option<Vec<String>> {
    let total = pairs.len();
    Some(
        pairs
            .iter()
            .map(|(l, r)| format!("{}|{}|{total}", l[0], cell(r.map(|r| r[0]))))
            .collect(),
    )
}

/// ROW_NUMBER() OVER (ORDER BY l.id, r.id): an unmatched row is alone in
/// its l.id, so its NULL is never compared
fn row_number(pairs: &[Pair]) -> Option<Vec<String>> {
    let mut sorted: Vec<(i64, Option<i64>)> =
        pairs.iter().map(|(l, r)| (l[0], r.map(|r| r[0]))).collect();
    sorted.sort();
    Some(
        sorted
            .into_iter()
            .enumerate()
            .map(|(n, (l, r))| format!("{l}|{}|{}", cell(r), n + 1))
            .collect(),
    )
}

fn group_count(pairs: &[Pair]) -> Option<Vec<String>> {
    let mut counts: BTreeMap<i64, usize> = BTreeMap::new();
    for (l, _) in pairs {
        *counts.entry(l[2]).or_default() += 1;
    }
    Some(
        counts
            .into_iter()
            .map(|(g, n)| format!("{g}|{n}"))
            .collect(),
    )
}

const SHAPES: [Shape; 7] = [
    Shape {
        select: "l.id, r.id",
        group_by: "",
        orders: &[
            ("", false),
            (" ORDER BY l.id, r.id", true),
            (" ORDER BY r.w, l.id, r.id", true),
            (" ORDER BY 1, 2", true),
        ],
        rows: pair_rows,
    },
    Shape {
        select: "DISTINCT l.g",
        group_by: "",
        orders: &[("", false), (" ORDER BY l.g", true), (" ORDER BY 1", true)],
        rows: distinct_g,
    },
    Shape {
        select: "DISTINCT ON (l.g) l.g, r.id",
        group_by: "",
        orders: &[(" ORDER BY l.g, r.id DESC", true)],
        rows: distinct_on_desc,
    },
    Shape {
        select: "DISTINCT ON (l.g) l.g, r.id",
        group_by: "",
        orders: &[(" ORDER BY l.g, r.w, r.id", true)],
        rows: distinct_on_by_w,
    },
    Shape {
        select: "l.id, r.id, COUNT(*) OVER () AS c",
        group_by: "",
        orders: &[("", false), (" ORDER BY l.id, r.id", true)],
        rows: count_over,
    },
    Shape {
        select: "l.id, r.id, ROW_NUMBER() OVER (ORDER BY l.id, r.id) AS rn",
        group_by: "",
        orders: &[("", false), (" ORDER BY l.id, r.id", true)],
        rows: row_number,
    },
    Shape {
        select: "l.g, COUNT(*) AS c",
        group_by: " GROUP BY l.g",
        orders: &[("", false), (" ORDER BY l.g", true)],
        rows: group_count,
    },
];

const CUTS: [(Option<usize>, usize); 7] = [
    (None, 0),
    (Some(0), 0),
    (Some(1), 0),
    (Some(2), 0),
    (Some(3), 0),
    (Some(3), 2),
    (Some(1000), 0),
];

fn rows(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let result = db.query(sql, ()).map_err(|e| e.to_string())?;
    result
        .map(|row| {
            let row = row.map_err(|e| e.to_string())?;
            Ok((0..row.len())
                .map(|i| match row.get_value(i) {
                    Some(Value::Integer(n)) => n.to_string(),
                    Some(Value::Null(_)) => "NULL".to_string(),
                    other => format!("{other:?}"),
                })
                .collect::<Vec<_>>()
                .join("|"))
        })
        .collect()
}

fn sorted(mut rows: Vec<String>) -> Vec<String> {
    rows.sort();
    rows
}

/// The statements whose rows are not the join's, each with why, and the
/// number of statements checked
fn mismatches(db: &Database) -> (Vec<String>, usize) {
    let mut found = Vec::new();
    let mut checked = 0;
    for source in &SOURCES {
        let pairs = joined(source);
        for shape in &SHAPES {
            let computed = (shape.rows)(&pairs);
            for &(order, total) in shape.orders {
                let reference_sql = format!(
                    "{}SELECT {} FROM {}{}{}",
                    source.with, shape.select, source.reference, shape.group_by, order
                );
                let reference = match rows(db, &reference_sql) {
                    Ok(reference) => reference,
                    Err(error) => {
                        found.push(format!("{reference_sql}\n    reference error: {error}"));
                        continue;
                    }
                };
                if let Some(computed) = &computed {
                    if sorted(reference.clone()) != sorted(computed.clone()) {
                        found.push(format!(
                            "{reference_sql}\n    reference {reference:?}\n    computed  {computed:?}"
                        ));
                        continue;
                    }
                }
                for (limit, offset) in CUTS {
                    let mut cut = String::new();
                    if let Some(limit) = limit {
                        cut.push_str(&format!(" LIMIT {limit}"));
                    }
                    if offset > 0 {
                        cut.push_str(&format!(" OFFSET {offset}"));
                    }
                    let sql = format!(
                        "{}SELECT {} FROM {}{}{}{}",
                        source.with, shape.select, source.from, shape.group_by, order, cut
                    );
                    checked += 1;
                    let expected: Vec<String> = reference
                        .iter()
                        .skip(offset)
                        .take(limit.unwrap_or(usize::MAX))
                        .cloned()
                        .collect();
                    match rows(db, &sql) {
                        Err(error) => found.push(format!("{sql}\n    error: {error}")),
                        Ok(got) if total => {
                            if got != expected {
                                found.push(format!(
                                    "{sql}\n    expected {expected:?}\n    got      {got:?}"
                                ));
                            }
                        }
                        Ok(got) => {
                            let mut pool = reference.clone();
                            let outside: Vec<&String> = got
                                .iter()
                                .filter(|row| match pool.iter().position(|r| r == *row) {
                                    Some(at) => {
                                        pool.swap_remove(at);
                                        false
                                    }
                                    None => true,
                                })
                                .collect();
                            if got.len() != expected.len() || !outside.is_empty() {
                                found.push(format!(
                                    "{sql}\n    {} rows, expected {}; not in the join: {outside:?}",
                                    got.len(),
                                    expected.len()
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    (found, checked)
}

#[test]
fn test_every_join_path_gives_the_rows_of_the_join() {
    let db = setup("join_clause_matrix");
    let (mut found, mut checked) = mismatches(&db);
    // Statistics move the build side of a hash join to the smaller table
    db.execute("ANALYZE l", ()).unwrap();
    db.execute("ANALYZE r", ()).unwrap();
    let (analyzed, analyzed_checked) = mismatches(&db);
    found.extend(analyzed.into_iter().map(|m| format!("[analyzed] {m}")));
    checked += analyzed_checked;
    assert!(
        found.is_empty(),
        "{} of {checked} statements are not the join's rows:\n{}",
        found.len(),
        found.join("\n")
    );
}
