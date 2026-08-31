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

//! ORDER BY + LIMIT top-K selection and the in-place JOIN reorder must
//! preserve ordering semantics, including NULLS placement and OFFSET.

use stoolap::api::Database;

#[test]
fn order_by_limit_offset_matches_full_sort() {
    let db = Database::open("memory://topk_basic").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, w INTEGER)",
        (),
    )
    .unwrap();
    let mut tx = db.begin().unwrap();
    for i in 0..500i64 {
        // v cycles so there are plenty of ties; w breaks them
        tx.execute("INSERT INTO t VALUES ($1, $2, $3)", (i, i % 7, i))
            .unwrap();
    }
    // A few NULLs in v
    for i in 500..510i64 {
        tx.execute("INSERT INTO t (id, w) VALUES ($1, $2)", (i, i))
            .unwrap();
    }
    tx.commit().unwrap();

    // Deterministic two-column ordering with OFFSET through the top-K path
    let rows = db
        .query(
            "SELECT v, w FROM t ORDER BY v DESC, w ASC LIMIT 10 OFFSET 5",
            (),
        )
        .unwrap()
        .collect_vec()
        .unwrap();
    let expected = db
        .query("SELECT v, w FROM t ORDER BY v DESC, w ASC", ())
        .unwrap()
        .collect_vec()
        .unwrap();
    assert_eq!(rows.len(), 10);
    for (i, r) in rows.iter().enumerate() {
        let (v1, w1): (Option<i64>, i64) = (r.get(0).ok(), r.get(1).unwrap());
        let e = &expected[i + 5];
        let (v2, w2): (Option<i64>, i64) = (e.get(0).ok(), e.get(1).unwrap());
        assert_eq!((v1, w1), (v2, w2), "row {i} diverges from full sort");
    }

    // NULLS placement: default DESC puts NULLs first
    let first = db
        .query("SELECT v FROM t ORDER BY v DESC LIMIT 3", ())
        .unwrap()
        .collect_vec()
        .unwrap();
    for r in &first {
        assert!(r.get::<i64>(0).is_err(), "DESC default is NULLS FIRST");
    }
}

#[test]
fn join_order_by_is_sorted_and_complete() {
    let db = Database::open("memory://topk_join").unwrap();
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE b (id INTEGER PRIMARY KEY, y INTEGER)", ())
        .unwrap();
    let mut tx = db.begin().unwrap();
    for i in 0..300i64 {
        tx.execute("INSERT INTO a VALUES ($1, $2)", (i, 299 - i))
            .unwrap();
        tx.execute("INSERT INTO b VALUES ($1, $2)", (i, i * 3))
            .unwrap();
    }
    tx.commit().unwrap();

    let rows = db
        .query(
            "SELECT a.x, b.y FROM a JOIN b ON a.id = b.id ORDER BY a.x ASC",
            (),
        )
        .unwrap()
        .collect_vec()
        .unwrap();
    assert_eq!(rows.len(), 300, "in-place permutation must keep all rows");
    let mut prev = -1i64;
    for r in &rows {
        let x: i64 = r.get(0).unwrap();
        assert!(x > prev, "rows must be sorted ascending");
        prev = x;
    }
    // Pairing stays intact after the permutation
    let (x0, y0): (i64, i64) = (rows[0].get(0).unwrap(), rows[0].get(1).unwrap());
    assert_eq!(x0, 0);
    assert_eq!(y0, 299 * 3);

    // And with LIMIT via the take-path
    let top = db
        .query(
            "SELECT a.x FROM a JOIN b ON a.id = b.id ORDER BY a.x DESC LIMIT 5",
            (),
        )
        .unwrap()
        .collect_vec()
        .unwrap();
    let xs: Vec<i64> = top.iter().map(|r| r.get(0).unwrap()).collect();
    assert_eq!(xs, vec![299, 298, 297, 296, 295]);
}

#[test]
fn join_order_by_offset_applies_once() {
    let db = Database::open("memory://topk_join_offset").unwrap();
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE b (id INTEGER PRIMARY KEY, y INTEGER)", ())
        .unwrap();
    let mut tx = db.begin().unwrap();
    for i in 0..30i64 {
        tx.execute("INSERT INTO a VALUES ($1, $2)", (i, i)).unwrap();
        tx.execute("INSERT INTO b VALUES ($1, $2)", (i, i)).unwrap();
    }
    tx.commit().unwrap();

    // INL-eligible: PK join key, no WHERE. OFFSET must apply exactly once.
    let rows = db
        .query(
            "SELECT a.x FROM a JOIN b ON a.id = b.id ORDER BY a.x ASC LIMIT 10 OFFSET 5",
            (),
        )
        .unwrap()
        .collect_vec()
        .unwrap();
    let xs: Vec<i64> = rows.iter().map(|r| r.get(0).unwrap()).collect();
    assert_eq!(xs, (5..15).collect::<Vec<i64>>());
}

#[test]
fn complex_order_by_limit_matches_full_sort() {
    let db = Database::open("memory://topk_complex").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    let mut tx = db.begin().unwrap();
    for i in 0..400i64 {
        tx.execute("INSERT INTO t VALUES ($1, $2)", (i, i)).unwrap();
    }
    tx.commit().unwrap();

    // Expression ORDER BY routes through the complex-sort path (the
    // select_nth top-K hook)
    let rows = db
        .query(
            "SELECT id FROM t ORDER BY v % 7 ASC, id DESC LIMIT 5 OFFSET 3",
            (),
        )
        .unwrap()
        .collect_vec()
        .unwrap();
    let expected = db
        .query("SELECT id FROM t ORDER BY v % 7 ASC, id DESC", ())
        .unwrap()
        .collect_vec()
        .unwrap();
    assert_eq!(rows.len(), 5);
    for (i, r) in rows.iter().enumerate() {
        let got: i64 = r.get(0).unwrap();
        let want: i64 = expected[i + 3].get(0).unwrap();
        assert_eq!(got, want, "row {i} diverges from full sort");
    }
}

#[test]
fn join_limit_offset_without_order_by_returns_full_count() {
    let db = Database::open("memory://topk_join_offset_noorder").unwrap();
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE b (id INTEGER PRIMARY KEY, y INTEGER)", ())
        .unwrap();
    let mut tx = db.begin().unwrap();
    for i in 0..30i64 {
        tx.execute("INSERT INTO a VALUES ($1, $2)", (i, i)).unwrap();
        tx.execute("INSERT INTO b VALUES ($1, $2)", (i, i)).unwrap();
    }
    tx.commit().unwrap();

    // Early-termination pushdown must collect OFFSET extra rows.
    // PK join key routes to index nested loop
    let rows = db
        .query(
            "SELECT a.x FROM a JOIN b ON a.id = b.id LIMIT 10 OFFSET 5",
            (),
        )
        .unwrap()
        .collect_vec()
        .unwrap();
    assert_eq!(rows.len(), 10, "INL join undercounts with OFFSET");

    // Non-indexed join key routes to the hash/streaming join executor
    let rows = db
        .query(
            "SELECT a.x FROM a JOIN b ON a.x = b.y LIMIT 10 OFFSET 5",
            (),
        )
        .unwrap()
        .collect_vec()
        .unwrap();
    assert_eq!(rows.len(), 10, "hash join undercounts with OFFSET");
}

#[test]
fn join_order_by_alias_sorts_correctly() {
    let db = Database::open("memory://topk_join_alias").unwrap();
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE b (id INTEGER PRIMARY KEY, y INTEGER)", ())
        .unwrap();
    let mut tx = db.begin().unwrap();
    for i in 0..30i64 {
        tx.execute("INSERT INTO a VALUES ($1, $2)", (i, i)).unwrap();
        tx.execute("INSERT INTO b VALUES ($1, $2)", (i, i)).unwrap();
    }
    tx.commit().unwrap();

    // ORDER BY references a SELECT alias: the join-local sort cannot
    // resolve it, so ordering must fall through to the standard path
    let rows = db
        .query(
            "SELECT a.x AS z FROM a JOIN b ON a.id = b.id ORDER BY z DESC LIMIT 10 OFFSET 5",
            (),
        )
        .unwrap()
        .collect_vec()
        .unwrap();
    let zs: Vec<i64> = rows.iter().map(|r| r.get(0).unwrap()).collect();
    assert_eq!(zs, (15..25).rev().collect::<Vec<i64>>());
}

#[test]
fn join_order_by_ordinal_resolves_to_column() {
    let db = Database::open("memory://topk_join_ordinal").unwrap();
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE b (id INTEGER PRIMARY KEY, y INTEGER)", ())
        .unwrap();
    let mut tx = db.begin().unwrap();
    for i in 0..30i64 {
        tx.execute("INSERT INTO a VALUES ($1, $2)", (i, i)).unwrap();
        tx.execute("INSERT INTO b VALUES ($1, $2)", (i, i)).unwrap();
    }
    tx.commit().unwrap();

    // ORDER BY 1 is positional: it must resolve to the first output
    // column, not the constant integer 1
    let rows = db
        .query(
            "SELECT a.x FROM a JOIN b ON a.id = b.id ORDER BY 1 DESC LIMIT 10 OFFSET 5",
            (),
        )
        .unwrap()
        .collect_vec()
        .unwrap();
    let xs: Vec<i64> = rows.iter().map(|r| r.get(0).unwrap()).collect();
    assert_eq!(xs, (15..25).rev().collect::<Vec<i64>>());
}

#[test]
fn join_order_by_desc_null_placement() {
    let db = Database::open("memory://topk_join_desc_nulls").unwrap();
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, x INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE b (id INTEGER PRIMARY KEY, y INTEGER)", ())
        .unwrap();
    let mut tx = db.begin().unwrap();
    for i in 0..10i64 {
        if i < 2 {
            tx.execute("INSERT INTO a VALUES ($1, NULL)", (i,)).unwrap();
        } else {
            tx.execute("INSERT INTO a VALUES ($1, $2)", (i, i)).unwrap();
        }
        tx.execute("INSERT INTO b VALUES ($1, $2)", (i, i)).unwrap();
    }
    tx.commit().unwrap();

    let fetch = |sql: &str| -> Vec<Option<i64>> {
        db.query(sql, ())
            .unwrap()
            .collect_vec()
            .unwrap()
            .iter()
            .map(|r| r.get(0).unwrap())
            .collect()
    };

    // Default DESC places NULLs first (AST contract)
    let got = fetch("SELECT a.x FROM a JOIN b ON a.id = b.id ORDER BY a.x DESC");
    let mut expected: Vec<Option<i64>> = vec![None, None];
    expected.extend((2..10).rev().map(Some));
    assert_eq!(got, expected, "default DESC");

    // Explicit NULLS LAST under DESC must be honored, not inverted
    let got = fetch("SELECT a.x FROM a JOIN b ON a.id = b.id ORDER BY a.x DESC NULLS LAST");
    let mut expected: Vec<Option<i64>> = (2..10).rev().map(Some).collect();
    expected.extend([None, None]);
    assert_eq!(got, expected, "explicit NULLS LAST");

    // Explicit NULLS FIRST under DESC
    let got = fetch("SELECT a.x FROM a JOIN b ON a.id = b.id ORDER BY a.x DESC NULLS FIRST");
    let mut expected: Vec<Option<i64>> = vec![None, None];
    expected.extend((2..10).rev().map(Some));
    assert_eq!(got, expected, "explicit NULLS FIRST");
}
