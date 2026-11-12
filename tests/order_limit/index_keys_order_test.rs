// Copyright 2026 Stoolap Contributors
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

//! ORDER BY of several columns + LIMIT, answered by walking the index they
//! lead, gives the order its rule asks: each column's NULLs where it says,
//! both directions, OFFSET across groups, and the shapes it leaves to a sort

use stoolap::Database;

const ROWS: i64 = 400;

/// `w (id, a, b, c, f)`: `a` is `id % 7`, NULL every 13th row; `b` is `id`,
/// NULL every 11th; `c` is `id % 3`; `f` a FLOAT copy of `a`
fn fill(db: &Database) {
    for sql in [
        "CREATE TABLE w (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, c INTEGER, f FLOAT)"
            .to_string(),
        format!(
            "INSERT INTO w SELECT value, CASE WHEN value % 13 = 0 THEN NULL ELSE value % 7 END, \
             CASE WHEN value % 11 = 0 THEN NULL ELSE value END, value % 3, \
             CASE WHEN value % 13 = 0 THEN NULL ELSE value % 7 END \
             FROM generate_series(1, {ROWS})"
        ),
        "CREATE INDEX w_ab ON w (a, b)".to_string(),
        "CREATE INDEX w_cab ON w (c, a, b)".to_string(),
        "CREATE INDEX w_fb ON w (f, b)".to_string(),
    ] {
        db.execute(&sql, ()).unwrap();
    }
}

/// The columns `fill` gives row `id`, NULL as None
fn row(id: i64) -> (Option<i64>, Option<i64>, i64) {
    let a = (id % 13 != 0).then_some(id % 7);
    let b = (id % 11 != 0).then_some(id);
    (a, b, id % 3)
}

/// Where one key goes: NULLs last ascending and first descending, unless
/// the key says NULLS FIRST or NULLS LAST
fn rank(value: Option<i64>, ascending: bool, nulls: &str) -> (u8, i64) {
    let nulls_first = match nulls {
        "NULLS FIRST" => true,
        "NULLS LAST" => false,
        _ => !ascending,
    };
    match value {
        Some(v) if ascending => (1, v),
        Some(v) => (1, -v),
        None if nulls_first => (0, 0),
        None => (2, 0),
    }
}

fn pairs(db: &Database, sql: &str) -> Vec<(Option<i64>, Option<i64>)> {
    db.query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|r| {
            let r = r.unwrap();
            (r.get::<i64>(0).ok(), r.get::<i64>(1).ok())
        })
        .collect()
}

/// The (a, b) sequence of `ORDER BY a <dir> <nulls a>, b <dir> <nulls b>`
fn expected(ascending: bool, (nulls_a, nulls_b): (&str, &str)) -> Vec<(Option<i64>, Option<i64>)> {
    let mut rows: Vec<(Option<i64>, Option<i64>)> =
        (1..=ROWS).map(|id| (row(id).0, row(id).1)).collect();
    rows.sort_by_key(|&(a, b)| (rank(a, ascending, nulls_a), rank(b, ascending, nulls_b)));
    rows
}

const NULLS: [&str; 3] = ["", "NULLS FIRST", "NULLS LAST"];

#[test]
fn a_walk_of_two_keys_gives_each_columns_nulls_where_it_is_asked() {
    let db = Database::open("memory://index_keys_order_nulls").unwrap();
    fill(&db);
    for ascending in [true, false] {
        let dir = if ascending { "ASC" } else { "DESC" };
        for nulls_a in NULLS {
            for nulls_b in NULLS {
                let all = expected(ascending, (nulls_a, nulls_b));
                for (limit, offset) in [(10, 0), (5, 28), (5, 55), (7, 90)] {
                    let sql = format!(
                        "SELECT a, b FROM w ORDER BY a {dir} {nulls_a}, b {dir} {nulls_b} \
                         LIMIT {limit} OFFSET {offset}"
                    );
                    let want: Vec<_> = all.iter().skip(offset).take(limit).copied().collect();
                    assert_eq!(pairs(&db, &sql), want, "{sql}");
                }
            }
        }
    }
}

#[test]
fn a_walk_reads_the_leading_columns_of_a_longer_index_and_all_of_them() {
    let db = Database::open("memory://index_keys_order_prefix").unwrap();
    fill(&db);
    let mut rows: Vec<(i64, Option<i64>, Option<i64>)> = (1..=ROWS)
        .map(|id| {
            let (a, b, c) = row(id);
            (c, a, b)
        })
        .collect();
    rows.sort_by_key(|&(c, a, b)| (c, rank(a, true, ""), rank(b, true, "")));
    let sql = "SELECT c, a, b FROM w ORDER BY c, a, b LIMIT 6 OFFSET 40";
    let got: Vec<(i64, Option<i64>, Option<i64>)> = db
        .query(sql, ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (
                r.get(0).unwrap(),
                r.get::<i64>(1).ok(),
                r.get::<i64>(2).ok(),
            )
        })
        .collect();
    assert_eq!(got, rows[40..46].to_vec(), "{sql}");
    // Two of the index's three columns: (c, a), ties on b in any order
    let sql = "SELECT c, a FROM w ORDER BY c DESC, a DESC LIMIT 6 OFFSET 40";
    let mut want: Vec<(i64, Option<i64>)> = rows.iter().map(|&(c, a, _)| (c, a)).collect();
    want.sort_by_key(|&(c, a)| (-c, rank(a, false, "")));
    let got: Vec<(i64, Option<i64>)> = db
        .query(sql, ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get(0).unwrap(), r.get::<i64>(1).ok())
        })
        .collect();
    assert_eq!(got, want[40..46].to_vec(), "{sql}");
}

/// Shapes the walk leaves to the sort, each still in the rule's order
#[test]
fn the_shapes_a_walk_declines_are_sorted_in_the_same_order() {
    let db = Database::open("memory://index_keys_order_declined").unwrap();
    fill(&db);
    let all = expected(true, ("", ""));
    // A WHERE, a LIMIT past a quarter of the rows, and the transaction's own rows
    let kept: Vec<_> = all
        .iter()
        .filter(|&&(_, b)| b.is_some_and(|b| b > 100))
        .take(5)
        .copied()
        .collect();
    assert_eq!(
        pairs(
            &db,
            "SELECT a, b FROM w WHERE b > 100 ORDER BY a, b LIMIT 5"
        ),
        kept
    );
    assert_eq!(
        pairs(&db, "SELECT a, b FROM w ORDER BY a, b LIMIT 150"),
        all[..150].to_vec()
    );
    db.execute("BEGIN", ()).unwrap();
    db.execute("UPDATE w SET b = NULL WHERE id = 7", ())
        .unwrap();
    let mut own = all.clone();
    let at = own.iter().position(|&p| p == (Some(0), Some(7))).unwrap();
    own.remove(at);
    let nulls_of_0 = own.iter().rposition(|&(a, _)| a == Some(0)).unwrap() + 1;
    own.insert(nulls_of_0, (Some(0), None));
    assert_eq!(
        pairs(&db, "SELECT a, b FROM w ORDER BY a, b LIMIT 20"),
        own[..20].to_vec()
    );
    db.execute("ROLLBACK", ()).unwrap();
    // Mixed directions
    let mut mixed: Vec<(Option<i64>, Option<i64>)> =
        (1..=ROWS).map(|id| (row(id).0, row(id).1)).collect();
    mixed.sort_by_key(|&(a, b)| (rank(a, true, ""), rank(b, false, "")));
    assert_eq!(
        pairs(
            &db,
            "SELECT a, b FROM w ORDER BY a ASC, b DESC LIMIT 8 OFFSET 30"
        ),
        mixed[30..38].to_vec()
    );
    // A FLOAT key, whose NaN the key order does not place
    let got: Vec<(Option<f64>, Option<i64>)> = db
        .query("SELECT f, b FROM w ORDER BY f, b LIMIT 5", ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get::<f64>(0).ok(), r.get::<i64>(1).ok())
        })
        .collect();
    let want: Vec<(Option<f64>, Option<i64>)> = all[..5]
        .iter()
        .map(|&(a, b)| (a.map(|a| a as f64), b))
        .collect();
    assert_eq!(got, want);
}

#[test]
fn a_walk_over_sealed_rows_is_left_to_the_sort() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}/sealed", dir.path().display())).unwrap();
    fill(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("INSERT INTO w (id, a, b, c) VALUES (1000, 0, 0, 0)", ())
        .unwrap();
    let mut all = expected(true, ("", ""));
    all.insert(0, (Some(0), Some(0)));
    assert_eq!(
        pairs(&db, "SELECT a, b FROM w ORDER BY a, b LIMIT 10 OFFSET 3"),
        all[3..13].to_vec()
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn an_order_of_two_keys_walks_only_the_rows_it_keeps() {
    use stoolap::test_failpoints::hot_top_k_visits;
    let db = Database::open("memory://index_keys_order_walk").unwrap();
    fill(&db);
    for sql in [
        "SELECT a, b FROM w ORDER BY a, b LIMIT 5",
        "SELECT a, b FROM w ORDER BY a DESC, b DESC LIMIT 5",
        "SELECT a, b FROM w ORDER BY a NULLS FIRST, b NULLS FIRST LIMIT 5 OFFSET 10",
    ] {
        let before = hot_top_k_visits();
        pairs(&db, sql);
        let visits = hot_top_k_visits() - before;
        assert!(
            (1..=15).contains(&visits),
            "{sql}: the walk visited {visits} rows"
        );
    }
}
