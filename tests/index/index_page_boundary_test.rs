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

//! Index groups larger than a page of row ids answer as whole groups:
//! through the grouped walk, after deletes and after a seal

use stoolap::core::Value;
use stoolap::storage::traits::Engine;
use stoolap::Database;

fn filled(dsn: &str, groups: &[(i64, i64)]) -> Database {
    let db = Database::open(dsn).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_t_k ON t(k)", ()).unwrap();
    let insert = db.prepare("INSERT INTO t VALUES ($1, $2, $1)").unwrap();
    db.execute("BEGIN", ()).unwrap();
    let mut id = 0;
    for &(k, rows) in groups {
        for _ in 0..rows {
            id += 1;
            insert.execute((id, k)).unwrap();
        }
    }
    db.execute("COMMIT", ()).unwrap();
    db
}

/// The rows of a two-column result, sorted: a GROUP BY without ORDER BY
/// gives its groups in no set order
fn pairs(db: &Database, sql: &str) -> Vec<(i64, i64)> {
    let mut out: Vec<(i64, i64)> = db
        .query(sql, ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get::<i64>(0).unwrap(), r.get::<i64>(1).unwrap())
        })
        .collect();
    out.sort_unstable();
    out
}

/// Each group of the index walk on `k` with its row ids, one entry per call;
/// None when the walk cannot answer
fn walked(db: &Database) -> Option<Vec<(i64, Vec<i64>)>> {
    let tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("t").unwrap();
    let mut groups = Vec::new();
    let answer = table
        .walk_btree_groups("k", 4096, 1024 * 1024, &mut |key, ids| {
            let Value::Integer(k) = key else {
                panic!("an integer key");
            };
            groups.push((*k, ids.iter().collect()));
            Ok(true)
        })
        .unwrap();
    answer.map(|()| groups)
}

fn count(db: &Database, sql: &str) -> i64 {
    db.query_one::<i64, _>(sql, ()).unwrap()
}

#[test]
fn the_group_walk_gives_a_group_of_several_pages_in_one_call() {
    let db = filled(
        "memory://page_boundary_walk",
        &[(1, 1_500), (2, 700), (3, 3)],
    );
    let groups = walked(&db).expect("the memory walk answers");
    let expected = vec![
        (1, (1..=1_500).collect::<Vec<i64>>()),
        (2, (1_501..=2_200).collect()),
        (3, (2_201..=2_203).collect()),
    ];
    assert_eq!(groups, expected, "one call per key, every id in order");

    assert_eq!(
        pairs(
            &db,
            "SELECT k, COUNT(*) FROM t GROUP BY k HAVING COUNT(*) > 600"
        ),
        vec![(1, 1_500), (2, 700)]
    );
    let mut sums: Vec<(i64, f64)> = db
        .query(
            "SELECT k, SUM(v) FROM t GROUP BY k HAVING SUM(v) > 1000000 LIMIT 5",
            (),
        )
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get::<i64>(0).unwrap(), r.get::<f64>(1).unwrap())
        })
        .collect();
    sums.sort_by_key(|(k, _)| *k);
    assert_eq!(sums, vec![(1, 1_125_750.0), (2, 1_295_350.0)]);

    let mut spans: Vec<(i64, f64, f64, f64)> = db
        .query(
            "SELECT k, MIN(v), MAX(v), AVG(v) FROM t GROUP BY k LIMIT 5",
            (),
        )
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (
                r.get::<i64>(0).unwrap(),
                r.get::<f64>(1).unwrap(),
                r.get::<f64>(2).unwrap(),
                r.get::<f64>(3).unwrap(),
            )
        })
        .collect();
    spans.sort_by_key(|(k, ..)| *k);
    assert_eq!(
        spans,
        vec![
            (1, 1.0, 1_500.0, 750.5),
            (2, 1_501.0, 2_200.0, 1_850.5),
            (3, 2_201.0, 2_203.0, 2_202.0)
        ],
        "every page of a group aggregated into one row"
    );
}

#[test]
fn a_group_deleted_across_its_pages_and_back_to_one_answers_by_index() {
    let db = filled("memory://page_boundary_deletes", &[(1, 1_500), (2, 10)]);
    db.execute("DELETE FROM t WHERE k = 1 AND id % 3 = 0", ())
        .unwrap();
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t WHERE k = 1"), 1_000);
    assert_eq!(
        pairs(&db, "SELECT k, COUNT(*) FROM t GROUP BY k"),
        vec![(1, 1_000), (2, 10)]
    );
    let groups = walked(&db).expect("the memory walk answers");
    let kept: Vec<i64> = (1..=1_500).filter(|id| id % 3 != 0).collect();
    assert_eq!(groups[0], (1, kept));
    db.execute("DELETE FROM t WHERE k = 1 AND id > 300", ())
        .unwrap();
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t WHERE k = 1"), 200);
    assert_eq!(count(&db, "SELECT MAX(id) FROM t WHERE k = 1"), 299);
}

#[test]
fn a_group_larger_than_the_capture_falls_back_to_a_whole_answer() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_interval=0&checkpoint_on_close=off",
        dir.path().display()
    );
    let db = filled(&dsn, &[(1, 5_000), (2, 10)]);
    assert_eq!(
        walked(&db),
        None,
        "a first group over the capture's 4,096 rows is left out whole"
    );
    assert_eq!(
        pairs(&db, "SELECT k, COUNT(*) FROM t GROUP BY k"),
        vec![(1, 5_000), (2, 10)]
    );
}

#[test]
fn a_seal_takes_paged_groups_out_of_every_composite_structure() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_interval=0&checkpoint_on_close=off",
        dir.path().display()
    );
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER NOT NULL, b INTEGER NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_t_ab ON t(a, b)", ()).unwrap();
    let insert = db.prepare("INSERT INTO t VALUES ($1, $2, $3)").unwrap();
    let rows = |from: i64, to: i64| {
        db.execute("BEGIN", ()).unwrap();
        for id in from..=to {
            insert.execute((id, id % 2, id % 3)).unwrap();
        }
        db.execute("COMMIT", ()).unwrap();
    };
    let expect = |to: i64, a: i64, b: Option<i64>| {
        (1..=to)
            .filter(|id| id % 2 == a && b.is_none_or(|b| id % 3 == b))
            .count() as i64
    };
    rows(1, 3_000);
    // The prefix, the exact keys, the sorted keys and the walk orders built
    assert_eq!(count(&db, "SELECT COUNT(*) FROM t WHERE a = 1"), 1_500);
    assert_eq!(
        count(&db, "SELECT COUNT(*) FROM t WHERE a = 1 AND b = 2"),
        500
    );
    assert_eq!(
        count(&db, "SELECT COUNT(*) FROM t WHERE a = 1 AND b >= 1"),
        1_000
    );
    assert_eq!(
        count(
            &db,
            "SELECT id FROM t WHERE a = 1 ORDER BY b DESC, id LIMIT 1"
        ),
        5
    );

    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    rows(3_001, 3_600);
    for (a, b) in [(1, None), (0, Some(0)), (1, Some(2))] {
        let sql = match b {
            None => format!("SELECT COUNT(*) FROM t WHERE a = {a}"),
            Some(b) => format!("SELECT COUNT(*) FROM t WHERE a = {a} AND b = {b}"),
        };
        assert_eq!(count(&db, &sql), expect(3_600, a, b), "{sql}");
    }
    assert_eq!(
        count(&db, "SELECT COUNT(*) FROM t WHERE a = 1 AND b >= 1"),
        expect(3_600, 1, Some(1)) + expect(3_600, 1, Some(2))
    );
}
