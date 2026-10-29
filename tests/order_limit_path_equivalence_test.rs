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

//! Every path that chooses the rows an ORDER BY with a LIMIT or OFFSET
//! keeps answers the places the whole sort gives them, and every path is
//! exercised: a path added without a case here fails the coverage check

#![cfg(feature = "test-failpoints")]

use stoolap::test_failpoints::{
    order_limit_paths, order_limit_paths_taken, reset_order_limit_paths,
};
use stoolap::Database;

/// One statement on a path, against a statement every path agrees on
struct Case {
    ctx: &'static str,
    path: &'static str,
    sql: &'static str,
    /// The same rows sorted whole, with no LIMIT or OFFSET under `cut`; a
    /// DISTINCT that keeps every row turns off a shortcut that runs without
    reference: &'static str,
    /// The LIMIT and OFFSET cut from the reference's rows
    cut: Option<(usize, usize)>,
}

const fn case(
    ctx: &'static str,
    path: &'static str,
    sql: &'static str,
    reference: &'static str,
    cut: Option<(usize, usize)>,
) -> Case {
    Case {
        ctx,
        path,
        sql,
        reference,
        cut,
    }
}

const CASES: &[Case] = &[
    // A set operation's rows reach the statement's sort whole
    case(
        "memory",
        "top_n",
        "SELECT id, a FROM t WHERE id <= 150 UNION ALL SELECT id, a FROM t WHERE id > 150 \
         ORDER BY a, id LIMIT 5 OFFSET 2",
        "SELECT id, a FROM t WHERE id <= 150 UNION ALL SELECT id, a FROM t WHERE id > 150 \
         ORDER BY a, id",
        Some((5, 2)),
    ),
    case(
        "sealed",
        "top_n",
        "SELECT id, a FROM t WHERE id <= 150 UNION ALL SELECT id, a FROM t WHERE id > 150 \
         ORDER BY a DESC NULLS FIRST, id LIMIT 5",
        "SELECT id, a FROM t WHERE id <= 150 UNION ALL SELECT id, a FROM t WHERE id > 150 \
         ORDER BY a DESC NULLS FIRST, id",
        Some((5, 0)),
    ),
    // A derived table's body runs on the general path
    case(
        "memory",
        "projected_top",
        "SELECT * FROM (SELECT id, a * 2 AS d FROM t ORDER BY d, id LIMIT 5 OFFSET 2) s",
        "SELECT * FROM (SELECT DISTINCT id, a * 2 AS d FROM t ORDER BY d, id) s",
        Some((5, 2)),
    ),
    case(
        "sealed",
        "projected_top",
        "SELECT * FROM (SELECT id, a * 2 AS d FROM t ORDER BY d DESC NULLS FIRST, id LIMIT 5) s",
        "SELECT * FROM (SELECT DISTINCT id, a * 2 AS d FROM t ORDER BY d DESC NULLS FIRST, id) s",
        Some((5, 0)),
    ),
    case(
        "memory",
        "projected_top",
        "SELECT * FROM (SELECT id FROM t ORDER BY b DESC, a + 0, id LIMIT 4 OFFSET 3) s",
        "SELECT * FROM (SELECT id FROM t ORDER BY b DESC, a + 0, id) s",
        Some((4, 3)),
    ),
    // A statement over a derived table or a view runs on the runner
    case(
        "sealed",
        "prepared_project_first",
        "SELECT id, a FROM (SELECT id, a, c FROM t) s ORDER BY a + 0, id LIMIT 4 OFFSET 1",
        "SELECT id, a FROM (SELECT id, a, c FROM t) s ORDER BY a + 0, id",
        Some((4, 1)),
    ),
    case(
        "memory",
        "prepared_project_first",
        "SELECT id FROM tv ORDER BY a + 0 DESC, id LIMIT 4",
        "SELECT id FROM tv ORDER BY a + 0 DESC, id",
        Some((4, 0)),
    ),
    case(
        "memory",
        "computed_keys",
        "SELECT * FROM (SELECT * FROM t ORDER BY a + 0, id LIMIT 4 OFFSET 1) s",
        "SELECT * FROM (SELECT * FROM t ORDER BY a + 0, id) s",
        Some((4, 1)),
    ),
    case(
        "sealed",
        "computed_keys",
        "SELECT * FROM (SELECT * FROM t ORDER BY a + 0 DESC, id LIMIT 4) s",
        "SELECT * FROM (SELECT * FROM t ORDER BY a + 0 DESC, id) s",
        Some((4, 0)),
    ),
    case(
        "memory",
        "deferred_projection",
        "SELECT * FROM (SELECT id, b FROM t ORDER BY b, id LIMIT 3 OFFSET 1) s",
        "SELECT * FROM (SELECT DISTINCT id, b FROM t ORDER BY b, id) s",
        Some((3, 1)),
    ),
    case(
        "sealed",
        "deferred_projection",
        "SELECT * FROM (SELECT id, b FROM t ORDER BY b DESC, id LIMIT 3) s",
        "SELECT * FROM (SELECT DISTINCT id, b FROM t ORDER BY b DESC, id) s",
        Some((3, 0)),
    ),
    case(
        "memory",
        "index_ordered",
        "SELECT * FROM (SELECT id, b FROM t ORDER BY c DESC NULLS LAST LIMIT 3 OFFSET 1) s",
        "SELECT * FROM (SELECT id, b FROM t ORDER BY c DESC NULLS LAST, id) s",
        Some((3, 1)),
    ),
    case(
        "sealed",
        "index_ordered",
        "SELECT * FROM (SELECT * FROM t ORDER BY id DESC LIMIT 3 OFFSET 2) s",
        "SELECT * FROM (SELECT DISTINCT * FROM t ORDER BY id DESC) s",
        Some((3, 2)),
    ),
    case(
        "memory",
        "top_k_scan",
        "SELECT * FROM (SELECT * FROM t WHERE g = 1 ORDER BY n DESC LIMIT 3) s",
        "SELECT * FROM (SELECT DISTINCT * FROM t WHERE g = 1 ORDER BY n DESC) s",
        Some((3, 0)),
    ),
    case(
        "sealed",
        "top_k_scan",
        "SELECT * FROM (SELECT * FROM t ORDER BY n LIMIT 3 OFFSET 1) s",
        "SELECT * FROM (SELECT DISTINCT * FROM t ORDER BY n) s",
        Some((3, 1)),
    ),
    // The same statements over the table run on the plan
    case(
        "sealed",
        "prepared_top_rows",
        "SELECT id, b FROM t ORDER BY b DESC, id LIMIT 3",
        "SELECT DISTINCT id, b FROM t ORDER BY b DESC, id",
        Some((3, 0)),
    ),
    case(
        "sealed",
        "prepared_project_first",
        "SELECT * FROM t ORDER BY a + 0 DESC, id LIMIT 4",
        "SELECT * FROM t ORDER BY a + 0 DESC, id",
        Some((4, 0)),
    ),
    case(
        "memory",
        "prepared_index_ordered",
        "SELECT id, b FROM t ORDER BY c DESC NULLS LAST LIMIT 3 OFFSET 1",
        "SELECT id, b FROM t ORDER BY c DESC NULLS LAST, id",
        Some((3, 1)),
    ),
    case(
        "memory",
        "prepared_index_ordered",
        "SELECT * FROM t ORDER BY id DESC LIMIT 3",
        "SELECT DISTINCT * FROM t ORDER BY id DESC",
        Some((3, 0)),
    ),
    case(
        "sealed",
        "prepared_index_ordered",
        "SELECT * FROM t ORDER BY id DESC LIMIT 3 OFFSET 2",
        "SELECT DISTINCT * FROM t ORDER BY id DESC",
        Some((3, 2)),
    ),
    case(
        "memory",
        "prepared_top_k",
        "SELECT * FROM t WHERE g = 1 ORDER BY n DESC LIMIT 3",
        "SELECT DISTINCT * FROM t WHERE g = 1 ORDER BY n DESC",
        Some((3, 0)),
    ),
    case(
        "sealed",
        "prepared_top_k",
        "SELECT * FROM t ORDER BY n LIMIT 3 OFFSET 1",
        "SELECT DISTINCT * FROM t ORDER BY n",
        Some((3, 1)),
    ),
    case(
        "memory",
        "prepared_keyset",
        "SELECT * FROM t WHERE id > 5 ORDER BY id LIMIT 3",
        "SELECT DISTINCT * FROM t WHERE id > 5 ORDER BY id",
        Some((3, 0)),
    ),
    case(
        "memory",
        "vector_search",
        "SELECT id, VEC_DISTANCE_L2(v, '[1.0, 0.0, 0.0]') AS d FROM docs ORDER BY d LIMIT 3",
        "SELECT DISTINCT id, VEC_DISTANCE_L2(v, '[1.0, 0.0, 0.0]') AS d FROM docs ORDER BY d",
        Some((3, 0)),
    ),
    case(
        "memory",
        "keyset_pagination",
        "SELECT * FROM (SELECT * FROM t WHERE id > 5 ORDER BY id LIMIT 3) s",
        "SELECT * FROM (SELECT DISTINCT * FROM t WHERE id > 5 ORDER BY id) s",
        Some((3, 0)),
    ),
    case(
        "memory",
        "join_index_sort",
        "SELECT t.id, u.x FROM t JOIN u ON u.id = t.a ORDER BY u.x DESC, t.id LIMIT 3 OFFSET 1",
        "SELECT DISTINCT t.id, u.x FROM t JOIN u ON u.id = t.a ORDER BY u.x DESC, t.id",
        Some((3, 1)),
    ),
    case(
        "sealed",
        "join_index_sort",
        "SELECT t.id, u.x FROM t JOIN u ON u.id = t.a ORDER BY u.x, t.id LIMIT 4",
        "SELECT DISTINCT t.id, u.x FROM t JOIN u ON u.id = t.a ORDER BY u.x, t.id",
        Some((4, 0)),
    ),
    case(
        "memory",
        "cte_order_limit_pushdown",
        "WITH s(k, total) AS (SELECT a, COUNT(*) FROM t GROUP BY a) \
         SELECT s.k, s.total, u.x FROM s LEFT JOIN u ON u.id = s.k \
         ORDER BY s.total DESC, s.k LIMIT 3",
        "WITH s(k, total) AS (SELECT a, COUNT(*) FROM t GROUP BY a) \
         SELECT DISTINCT s.k, s.total, u.x FROM s LEFT JOIN u ON u.id = s.k \
         ORDER BY s.total DESC, s.k",
        Some((3, 0)),
    ),
    case(
        "sealed",
        "cte_order_limit_pushdown",
        "WITH s(k, total) AS (SELECT a, COUNT(*) FROM t GROUP BY a) \
         SELECT s.k, s.total, u.x FROM s JOIN u ON u.id = s.k \
         ORDER BY s.total, s.k DESC LIMIT 4 OFFSET 2",
        "WITH s(k, total) AS (SELECT a, COUNT(*) FROM t GROUP BY a) \
         SELECT DISTINCT s.k, s.total, u.x FROM s JOIN u ON u.id = s.k \
         ORDER BY s.total, s.k DESC",
        Some((4, 2)),
    ),
    case(
        "memory",
        "prepared_index_ordered",
        "SELECT t.id, (SELECT MAX(x) FROM u WHERE u.k = t.id) AS m FROM t \
         ORDER BY c DESC NULLS LAST LIMIT 2",
        "SELECT t.id, (SELECT MAX(x) FROM u WHERE u.k = t.id) AS m FROM t \
         ORDER BY c DESC NULLS LAST",
        Some((2, 0)),
    ),
    case(
        "memory",
        "prepared_top_rows",
        "SELECT id, a FROM t WHERE a IN (SELECT k FROM u) ORDER BY a DESC, id LIMIT 2 OFFSET 1",
        "SELECT DISTINCT id, a FROM t WHERE a IN (SELECT k FROM u) ORDER BY a DESC, id",
        Some((2, 1)),
    ),
    case(
        "memory",
        "prepared_project_first",
        "SELECT id, a * 2 AS d, (SELECT x FROM u WHERE u.id = t.id) AS s FROM t \
         ORDER BY d, id LIMIT 2",
        "SELECT DISTINCT id, a * 2 AS d, (SELECT x FROM u WHERE u.id = t.id) AS s FROM t \
         ORDER BY d, id",
        Some((2, 0)),
    ),
    case(
        "memory",
        "prepared_block_cut",
        "SELECT t.id, (SELECT x FROM u WHERE u.k = t.a ORDER BY x DESC, u.id LIMIT 1 OFFSET 1) \
         AS y FROM t ORDER BY t.id",
        "SELECT t.id, (SELECT x FROM u WHERE u.k = t.a ORDER BY x DESC, u.id \
         LIMIT 1 + 0 OFFSET 1) AS y FROM t ORDER BY t.id",
        None,
    ),
];

fn setup(db: &Database) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b TEXT, c INTEGER, \
         g INTEGER, n INTEGER NOT NULL)",
        "INSERT INTO t SELECT value, CASE WHEN value % 23 = 0 THEN NULL ELSE value % 17 END, \
         'b' || (value % 11), CASE WHEN value % 50 = 0 THEN NULL ELSE 1000 - value END, \
         value % 5, value FROM generate_series(1, 300)",
        "CREATE INDEX t_a ON t (a)",
        "CREATE INDEX t_c ON t (c)",
        "CREATE INDEX t_gn ON t (g, n)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, x INTEGER)",
        "INSERT INTO u SELECT value, value % 17, (value * 13) % 97 FROM generate_series(1, 100)",
        "CREATE INDEX u_k ON u (k)",
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, v VECTOR(3))",
        "CREATE VIEW tv AS SELECT id, a, b, c FROM t",
    ] {
        db.execute(sql, ()).unwrap();
    }
    for id in 1..=40 {
        db.execute(
            &format!("INSERT INTO docs VALUES ({id}, '[{id}.0, 0.0, 0.0]')"),
            (),
        )
        .unwrap();
    }
}

/// The column names, then each row's values with their types, or the
/// error; and the paths that chose them
fn run(db: &Database, sql: &str) -> (Vec<String>, Vec<&'static str>) {
    reset_order_limit_paths();
    let rows = match db.query(sql, ()) {
        Ok(rows) => rows,
        Err(e) => return (vec![format!("error: {e}")], order_limit_paths_taken()),
    };
    let mut out = vec![rows.columns().join(", ")];
    out.extend(rows.map(|row| {
        let row = row.unwrap();
        format!(
            "{:?}",
            (0..row.len()).map(|i| row.get_value(i)).collect::<Vec<_>>()
        )
    }));
    (out, order_limit_paths_taken())
}

#[test]
fn every_order_limit_path_answers_as_the_whole_sort() {
    let memory = Database::open("memory://order_limit_path_equivalence").unwrap();
    setup(&memory);
    let dir = tempfile::tempdir().unwrap();
    let sealed = Database::open(&format!(
        "file://{}?checkpoint_on_close=off",
        dir.path().display()
    ))
    .unwrap();
    setup(&sealed);
    sealed.execute("PRAGMA CHECKPOINT", ()).unwrap();
    sealed
        .execute("UPDATE t SET b = 'z' WHERE id % 40 = 0", ())
        .unwrap();

    let mut wrong = Vec::new();
    let mut covered: Vec<&str> = Vec::new();
    for case in CASES {
        let db = match case.ctx {
            "sealed" => &sealed,
            _ => &memory,
        };
        let (got, taken) = run(db, case.sql);
        let (whole, reference_taken) = run(db, case.reference);
        let want: Vec<String> = match case.cut {
            Some((limit, offset)) => {
                let mut whole = whole.into_iter();
                whole
                    .next()
                    .into_iter()
                    .chain(whole.skip(offset).take(limit))
                    .collect()
            }
            None => whole,
        };
        if !taken.contains(&case.path) {
            wrong.push(format!(
                "{}: {}: took {taken:?}, expected {}",
                case.ctx, case.sql, case.path
            ));
        }
        let reference_wrong = match case.cut {
            Some(_) => !reference_taken.is_empty(),
            None => reference_taken.contains(&case.path),
        };
        if reference_wrong {
            wrong.push(format!(
                "{}: {}: took {reference_taken:?}",
                case.ctx, case.reference
            ));
        }
        if got != want {
            wrong.push(format!(
                "{}: {}\n  got   {got:?}\n  whole {want:?}",
                case.ctx, case.sql
            ));
        }
        covered.extend(taken);
    }
    for path in order_limit_paths() {
        if !covered.contains(&path) {
            wrong.push(format!("no case takes the {path} path"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
