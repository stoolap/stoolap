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

//! ORDER BY one column + LIMIT with only bounds on that column, answered by
//! walking the column's own B-tree, matches the full sort: ASC and DESC,
//! OFFSET, NULLs, ties, a predicate the index does not cover, changed rows,
//! the transaction's own rows, and hot rows mixed with sealed ones

use stoolap::Database;

fn column(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|r| format!("{:?}", r.unwrap().get_value(0)))
        .collect()
}

/// The order column's sequence must match the full sort by (t, id); ties
/// leave the order among equal keys to the path
fn check(db: &Database, where_clause: &str, desc: bool, limit: usize, offset: usize) {
    let dir = if desc { "DESC" } else { "ASC" };
    let expected: Vec<String> = column(
        db,
        &format!("SELECT t FROM r WHERE {where_clause} ORDER BY t {dir}, id {dir}"),
    )
    .into_iter()
    .skip(offset)
    .take(limit)
    .collect();
    let query = format!(
        "SELECT t FROM r WHERE {where_clause} ORDER BY t {dir} LIMIT {limit} OFFSET {offset}"
    );
    assert_eq!(column(db, &query), expected, "{query}");
}

fn check_shapes(db: &Database, low: &str, high: &str) {
    for desc in [true, false] {
        for where_clause in [
            format!("t >= {low}"),
            format!("t > {low}"),
            format!("t <= {high}"),
            format!("t < {high}"),
            format!("t > {low} AND t < {high}"),
            format!("t >= {low} AND t <= {high} AND v > 90"),
            format!("t >= {low} AND v = 7"),
        ] {
            check(db, &where_clause, desc, 5, 0);
            check(db, &where_clause, desc, 7, 3);
            check(db, &where_clause, desc, 100_000, 0);
        }
    }
}

/// 3000 rows over 700 keys out of order, so the walk crosses its chunks and
/// keys repeat; with `nullable`, every 50th key is NULL
fn create(db: &Database, ty: &str, nullable: bool) {
    let not_null = if nullable { "" } else { " NOT NULL" };
    db.execute(
        &format!("CREATE TABLE r (id INTEGER PRIMARY KEY, t {ty}{not_null}, v INTEGER)"),
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX r_t ON r(t) USING BTREE", ())
        .unwrap();
    let mut stmt = String::from("INSERT INTO r VALUES ");
    for id in 0..3000i64 {
        if id > 0 {
            stmt.push(',');
        }
        let k = (id * 7919) % 700;
        let t = if nullable && id % 50 == 0 {
            "NULL".to_string()
        } else {
            key(ty, k)
        };
        stmt.push_str(&format!("({id}, {t}, {})", id % 100));
    }
    db.execute(&stmt, ()).unwrap();
}

fn key(ty: &str, k: i64) -> String {
    match ty {
        "TIMESTAMP" => format!("'2024-01-01 00:{:02}:{:02}'", k / 60, k % 60),
        "TEXT" => format!("'k{k:04}'"),
        _ => k.to_string(),
    }
}

fn change(db: &Database) {
    db.execute("DELETE FROM r WHERE id % 7 = 3", ()).unwrap();
    db.execute("UPDATE r SET v = v + 1 WHERE id % 5 = 1", ())
        .unwrap();
    // Keys moved across the walk, both ways
    db.execute(
        "UPDATE r SET t = (SELECT t FROM r WHERE id = 2) WHERE id % 11 = 4",
        (),
    )
    .unwrap();
}

#[test]
fn a_range_top_k_matches_the_full_sort() {
    for ty in ["INTEGER", "TIMESTAMP", "TEXT"] {
        for nullable in [false, true] {
            let db =
                Database::open(&format!("memory://range_top_k_shapes_{ty}_{nullable}")).unwrap();
            create(&db, ty, nullable);
            let (low, high) = (key(ty, 100), key(ty, 600));
            check_shapes(&db, &low, &high);
            change(&db);
            check_shapes(&db, &low, &high);
        }
    }
}

#[test]
fn a_range_top_k_inside_a_transaction_sees_its_own_rows() {
    let db = Database::open("memory://range_top_k_txn").unwrap();
    create(&db, "INTEGER", true);
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO r VALUES (5000, 101, 1), (5001, 599, 2)", ())
        .unwrap();
    db.execute("DELETE FROM r WHERE t = 102", ()).unwrap();
    check_shapes(&db, "100", "600");
    db.execute("COMMIT", ()).unwrap();
    check_shapes(&db, "100", "600");
}

#[test]
fn a_range_top_k_over_hot_and_sealed_rows() {
    for nullable in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let dsn = format!("file://{}", dir.path().display());
        let db = Database::open(&dsn).unwrap();
        create(&db, "INTEGER", nullable);
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.execute(
            "INSERT INTO r VALUES (5000, 101, 1), (5001, 599, 2), (5002, 100, 3)",
            (),
        )
        .unwrap();
        change(&db);
        check_shapes(&db, "100", "600");
        drop(db);
        let db = Database::open(&dsn).unwrap();
        check_shapes(&db, "100", "600");
    }
}

#[test]
fn a_range_top_k_with_bounds_that_hold_nothing_is_empty() {
    let db = Database::open("memory://range_top_k_empty_bounds").unwrap();
    create(&db, "INTEGER", true);
    for where_clause in [
        "t > 600 AND t < 100",
        "t > 100 AND t < 100",
        "t >= 100 AND t < 100",
    ] {
        for dir in ["ASC", "DESC"] {
            let query = format!("SELECT t FROM r WHERE {where_clause} ORDER BY t {dir} LIMIT 5");
            assert!(column(&db, &query).is_empty(), "{query}");
        }
    }
}

#[test]
fn a_range_top_k_with_bounds_of_another_type_matches_the_full_sort() {
    for (ty, bounds) in [
        (
            "INTEGER",
            ["t >= 100.5", "t < 599.5", "t >= '100'", "t < 'zzz'"],
        ),
        ("TEXT", ["t >= 5", "t < 9", "t > 0.5", "t <= 7"]),
        ("TIMESTAMP", ["t >= 5", "t < 1e18", "t > 'x'", "t <= 'zzz'"]),
    ] {
        let db = Database::open(&format!("memory://range_top_k_other_type_{ty}")).unwrap();
        create(&db, ty, false);
        for where_clause in bounds {
            for desc in [true, false] {
                check(&db, where_clause, desc, 5, 0);
            }
        }
    }
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_range_top_k_reads_only_the_rows_near_its_bound() {
    use stoolap::test_failpoints::hot_top_k_visits;
    for ty in ["INTEGER", "TIMESTAMP", "TEXT"] {
        let db = Database::open(&format!("memory://range_top_k_reads_{ty}")).unwrap();
        create(&db, ty, true);
        for (op, dir, k) in [(">=", "", 100), ("<", "DESC", 600), ("<", "", 600)] {
            let query = format!(
                "SELECT t FROM r WHERE t {op} {} ORDER BY t {dir} LIMIT 5",
                key(ty, k)
            );
            let before = hot_top_k_visits();
            assert_eq!(column(&db, &query).len(), 5, "{query}");
            let visits = hot_top_k_visits() - before;
            assert!(
                (5..=20).contains(&visits),
                "{query}: walked {visits} rows for five"
            );
        }
        // Another column's predicate may pass few of the walked rows
        let query = format!(
            "SELECT t FROM r WHERE t >= {} AND id = 77 ORDER BY t LIMIT 5",
            key(ty, 100)
        );
        let before = hot_top_k_visits();
        column(&db, &query);
        assert_eq!(hot_top_k_visits() - before, 0, "{query}");
    }
}

#[cfg(all(feature = "test-failpoints", not(feature = "test-filedb")))]
#[test]
fn a_key_moved_behind_the_walk_is_still_answered() {
    let db = Database::open("memory://range_top_k_moved").unwrap();
    let other = db.clone();
    create(&db, "INTEGER", false);
    db.execute("DELETE FROM r WHERE t < 100", ()).unwrap();
    let fired = std::rc::Rc::new(std::cell::Cell::new(false));
    let fired_in_hook = std::rc::Rc::clone(&fired);
    // The walk starts at 100; the row moves to 50, behind it
    stoolap::test_failpoints::on_hot_top_k_visit(move || {
        other
            .execute("UPDATE r SET t = 50 WHERE id = 1", ())
            .unwrap();
        fired_in_hook.set(true);
    });
    let got = column(&db, "SELECT t FROM r WHERE t >= 40 ORDER BY t LIMIT 1");
    assert!(fired.get(), "the walk visited a row");
    assert_eq!(got, vec!["Some(Integer(50))".to_string()]);
}

/// 30000 rows, 20000 of them on one key, so the walk's chunks stop inside it
fn create_heavy_key(db: &Database) {
    db.execute(
        "CREATE TABLE r (id INTEGER PRIMARY KEY, t INTEGER NOT NULL, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX r_t ON r(t) USING BTREE", ())
        .unwrap();
    db.execute(
        "INSERT INTO r SELECT value, CASE WHEN value % 3 = 0 THEN value % 1000 ELSE 500 END, \
         value % 100 FROM generate_series(1, 30000)",
        (),
    )
    .unwrap();
}

#[test]
fn a_range_top_k_through_a_key_held_by_many_rows_matches_the_full_sort() {
    let db = Database::open("memory://range_top_k_heavy_key").unwrap();
    create_heavy_key(&db);
    db.execute("DELETE FROM r WHERE t = 500 AND id % 7 = 0", ())
        .unwrap();
    for where_clause in ["t >= 500", "t <= 500", "t > 499 AND t < 501", "t >= 0"] {
        for desc in [false, true] {
            for (limit, offset) in [(5, 0), (5000, 100), (25_000, 0)] {
                check(&db, where_clause, desc, limit, offset);
                let dir = if desc { "DESC" } else { "ASC" };
                let query = format!(
                    "SELECT id FROM r WHERE {where_clause} ORDER BY t {dir} LIMIT {limit} OFFSET {offset}"
                );
                let mut got = column(&db, &query);
                let returned = got.len();
                got.sort_unstable();
                got.dedup();
                assert_eq!(got.len(), returned, "duplicate rows: {query}");
            }
        }
    }
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_range_top_k_copies_a_chunk_of_a_key_held_by_many_rows() {
    use stoolap::test_failpoints::take_btree_walk_chunk_max;
    let db = Database::open("memory://range_top_k_heavy_key_copies").unwrap();
    db.execute(
        "CREATE TABLE r (id INTEGER PRIMARY KEY, t INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX r_t ON r(t) USING BTREE", ())
        .unwrap();
    db.execute(
        "INSERT INTO r SELECT value, 7 FROM generate_series(1, 100000)",
        (),
    )
    .unwrap();
    for (dir, limit, most) in [
        ("", 1, 64),
        ("DESC", 1, 64),
        ("", 50_000, 4096),
        ("DESC", 50_000, 4096),
    ] {
        let query = format!("SELECT id FROM r WHERE t >= 7 ORDER BY t {dir} LIMIT {limit}");
        take_btree_walk_chunk_max();
        assert_eq!(column(&db, &query).len(), limit, "{query}");
        let (copied, room) = take_btree_walk_chunk_max();
        assert!(
            copied > 0 && copied <= most,
            "{query}: copied {copied} ids under one lock"
        );
        assert!(room <= most, "{query}: room for {room} ids");
    }
}
