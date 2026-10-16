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

//! A correlated EXISTS or NOT EXISTS over an indexed inner column, with few
//! outer rows against the inner table's, probes the index per outer row and
//! answers as the semi-join over the same table without the index

use stoolap::Database;

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect();
    out.sort_unstable();
    out
}

/// `t` with an index on `c` and its twin `w` without, 3200 rows over 800
/// keys and some NULLs; `few` with 10 outer rows, `many` with 1000
fn create(db: &Database) {
    for (table, index) in [("t", true), ("w", false)] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, c INTEGER, v INTEGER)"),
            (),
        )
        .unwrap();
        if index {
            db.execute("CREATE INDEX t_c ON t(c) USING BTREE", ())
                .unwrap();
        }
        db.execute(
            &format!(
                "INSERT INTO {table} SELECT value, CASE WHEN value % 50 = 0 THEN NULL \
                 ELSE value * 7919 % 800 END, value % 7 FROM generate_series(1, 3200)"
            ),
            (),
        )
        .unwrap();
    }
    for (table, rows) in [("few", 10), ("many", 1000)] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, k INTEGER)"),
            (),
        )
        .unwrap();
        db.execute(
            &format!(
                "INSERT INTO {table} SELECT value, CASE WHEN value = 3 THEN NULL \
                 ELSE value * 97 % 1200 END FROM generate_series(1, {rows})"
            ),
            (),
        )
        .unwrap();
    }
}

fn check(db: &Database, phase: &str) {
    for outer in ["few", "many"] {
        for test in ["EXISTS", "NOT EXISTS"] {
            for extra in ["", " AND {o}.id > 4", " OR {o}.id = 2"] {
                let sql = format!(
                    "SELECT {{o}}.id FROM {{o}} WHERE {test} \
                     (SELECT 1 FROM {{i}} WHERE {{i}}.c = {{o}}.k){extra}"
                )
                .replace("{o}", outer);
                let got = ids(db, &sql.replace("{i}", "t"));
                let want = ids(db, &sql.replace("{i}", "w"));
                assert_eq!(got, want, "{phase}: {}", sql.replace("{i}", "t"));
            }
        }
    }
}

#[test]
fn exists_over_few_outer_rows_answers_as_the_semi_join() {
    let db = Database::open("memory://exists_few_outer_rows").unwrap();
    create(&db);
    check(&db, "memory");
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO t VALUES (9001, 97, 0)", ())
        .unwrap();
    db.execute("INSERT INTO w VALUES (9001, 97, 0)", ())
        .unwrap();
    db.execute("DELETE FROM t WHERE c = 194", ()).unwrap();
    db.execute("DELETE FROM w WHERE c = 194", ()).unwrap();
    check(&db, "transaction");
    db.execute("COMMIT", ()).unwrap();
    check(&db, "committed");
}

#[test]
fn exists_over_few_outer_rows_answers_as_the_semi_join_hot_and_sealed() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    create(&db);
    check(&db, "hot");
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    check(&db, "sealed");
    db.execute("DELETE FROM t WHERE c = 291", ()).unwrap();
    db.execute("DELETE FROM w WHERE c = 291", ()).unwrap();
    check(&db, "sealed with hot deletes");
}

#[cfg(feature = "test-failpoints")]
#[test]
fn exists_over_few_outer_rows_does_not_read_the_inner_table_whole() {
    use stoolap::test_failpoints::inner_table_reads;
    let db = Database::open("memory://exists_few_outer_rows_reads").unwrap();
    create(&db);
    for (outer, inner, reads_whole) in
        [("few", "t", false), ("many", "t", true), ("few", "w", true)]
    {
        for test in ["EXISTS", "NOT EXISTS"] {
            let sql = format!(
                "SELECT {outer}.id FROM {outer} WHERE {test} \
                 (SELECT 1 FROM {inner} WHERE {inner}.c = {outer}.k)"
            );
            let before = inner_table_reads();
            ids(&db, &sql);
            assert_eq!(inner_table_reads() > before, reads_whole, "{sql}");
        }
    }
}

#[test]
fn not_exists_sees_its_own_transaction() {
    for index in [false, true] {
        let db = Database::open(&format!("memory://not_exists_own_transaction_{index}")).unwrap();
        db.execute("CREATE TABLE w (id INTEGER PRIMARY KEY, c INTEGER)", ())
            .unwrap();
        if index {
            db.execute("CREATE INDEX w_c ON w(c) USING BTREE", ())
                .unwrap();
        }
        db.execute("CREATE TABLE s (id INTEGER PRIMARY KEY, k INTEGER)", ())
            .unwrap();
        db.execute("INSERT INTO w VALUES (1, 10), (2, 20)", ())
            .unwrap();
        db.execute("INSERT INTO s VALUES (1, 10), (2, 20), (3, 30)", ())
            .unwrap();
        db.execute("BEGIN", ()).unwrap();
        db.execute("DELETE FROM w WHERE c = 20", ()).unwrap();
        db.execute("INSERT INTO w VALUES (3, 30)", ()).unwrap();
        let not_exists = ids(
            &db,
            "SELECT s.id FROM s WHERE NOT EXISTS (SELECT 1 FROM w WHERE w.c = s.k)",
        );
        let exists = ids(
            &db,
            "SELECT s.id FROM s WHERE EXISTS (SELECT 1 FROM w WHERE w.c = s.k)",
        );
        db.execute("ROLLBACK", ()).unwrap();
        assert_eq!(not_exists, vec![2], "NOT EXISTS, index {index}");
        assert_eq!(exists, vec![1, 3], "EXISTS, index {index}");
        assert_eq!(
            ids(
                &db,
                "SELECT s.id FROM s WHERE NOT EXISTS (SELECT 1 FROM w WHERE w.c = s.k)"
            ),
            vec![3],
            "NOT EXISTS after the rollback, index {index}"
        );
    }
}

/// A key held by more inner rows than a probe copies answers from the
/// semi-join set, built at most once, instead of copying every row of the
/// key per outer row
#[cfg(feature = "test-failpoints")]
#[test]
fn exists_over_a_key_held_by_many_rows_builds_the_semi_join_once() {
    use stoolap::test_failpoints::{
        equality_candidate_ids_copied_count, exists_subquery_runs, inner_table_reads,
    };
    for using in ["BTREE", "HASH", "BITMAP"] {
        let db = heavy_key_tables(&format!("exists_heavy_key_{using}"), using);
        for test in ["EXISTS", "NOT EXISTS"] {
            let sql = format!(
                "SELECT o.id FROM o WHERE {test} (SELECT 1 FROM {{i}} WHERE {{i}}.c = o.k)"
            );
            let case = format!("{using}: {sql}");
            let (copied, reads, runs) = (
                equality_candidate_ids_copied_count(),
                inner_table_reads(),
                exists_subquery_runs(),
            );
            let got = ids(&db, &sql.replace("{i}", "h"));
            let copied = equality_candidate_ids_copied_count() - copied;
            let reads = inner_table_reads() - reads;
            assert_eq!(got, ids(&db, &sql.replace("{i}", "hw")), "{case}");
            assert!(copied <= 1024, "{case}: copied {copied} ids");
            assert!(reads <= 1, "{case}: inner table read whole {reads} times");
            assert_eq!(exists_subquery_runs() - runs, 0, "{case}: subqueries run");
        }
    }
}

/// `h` indexed `using` and its twin `hw`, 8000 rows with 5000 on key 7;
/// `o` asks key 7 on most of its 8 rows
#[cfg(feature = "test-failpoints")]
fn heavy_key_tables(name: &str, using: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for table in ["h", "hw"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, c INTEGER)"),
            (),
        )
        .unwrap();
        db.execute(
            &format!(
                "INSERT INTO {table} SELECT value, CASE WHEN value <= 5000 THEN 7 \
                 ELSE value % 900 END FROM generate_series(1, 8000)"
            ),
            (),
        )
        .unwrap();
    }
    db.execute(&format!("CREATE INDEX h_c ON h(c) USING {using}"), ())
        .unwrap();
    db.execute("CREATE TABLE o (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO o VALUES (1, 7), (2, 7), (3, 7), (4, 7), (5, 600), (6, 9000), (7, 7), (8, 7)",
        (),
    )
    .unwrap();
    db
}

/// The EXISTS cap leaves a correlated COUNT to count a key of any size
/// through the index
#[cfg(feature = "test-failpoints")]
#[test]
fn a_correlated_count_still_counts_a_key_held_by_many_rows_from_the_index() {
    use stoolap::test_failpoints::equality_candidate_ids_copied_count;
    let db = heavy_key_tables("count_heavy_key", "BTREE");
    let sql = "SELECT o.id, (SELECT COUNT(*) FROM {i} WHERE {i}.c = o.k) FROM o \
               ORDER BY o.id LIMIT 1";
    let count = |table: &str| -> i64 {
        db.query(&sql.replace("{i}", table), ())
            .unwrap()
            .map(|r| r.unwrap().get::<i64>(1).unwrap())
            .next()
            .unwrap()
    };
    let want = count("hw");
    let copied = equality_candidate_ids_copied_count();
    assert_eq!(count("h"), want);
    let copied = equality_candidate_ids_copied_count() - copied;
    assert!(
        copied >= 5000,
        "the count read the key's ids from the index: {copied}"
    );
}

/// Inside a transaction that changed the inner table its index answers no
/// probe, so the semi-join is chosen and no probe is asked
#[cfg(feature = "test-failpoints")]
#[test]
fn exists_after_local_changes_asks_no_probe() {
    use stoolap::test_failpoints::{correlated_probes_asked, exists_subquery_runs};
    let db = Database::open("memory://exists_local_changes").unwrap();
    create(&db);
    db.execute("BEGIN", ()).unwrap();
    db.execute("DELETE FROM t WHERE c = 194", ()).unwrap();
    for test in ["EXISTS", "NOT EXISTS"] {
        let sql =
            format!("SELECT few.id FROM few WHERE {test} (SELECT 1 FROM t WHERE t.c = few.k)");
        let (asked, runs) = (correlated_probes_asked(), exists_subquery_runs());
        ids(&db, &sql);
        assert_eq!(correlated_probes_asked() - asked, 0, "{sql}: probes asked");
        assert_eq!(exists_subquery_runs() - runs, 0, "{sql}: subqueries run");
    }
    db.execute("ROLLBACK", ()).unwrap();
}

/// An EXISTS over a recursive CTE sees the rows of the round that asks
#[test]
fn exists_over_a_recursive_cte_sees_each_round() {
    let db = Database::open("memory://exists_recursive_cte").unwrap();
    let rows: Vec<(i64, bool)> = db
        .query(
            "WITH RECURSIVE r(n, seen) AS ( \
               SELECT 1, TRUE \
               UNION ALL \
               SELECT c.n + 1, EXISTS (SELECT 1 FROM r q WHERE q.n = c.n) \
               FROM r c WHERE c.n < 3 \
             ) SELECT n, seen FROM r ORDER BY n",
            (),
        )
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get::<i64>(0).unwrap(), r.get::<bool>(1).unwrap())
        })
        .collect();
    assert_eq!(rows, vec![(1, true), (2, true), (3, true)]);
}
