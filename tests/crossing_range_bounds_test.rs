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

//! Range bounds that cross, or meet with one side open, hold no row through
//! any index form, as through no index

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

fn create(db: &Database) {
    for (table, index) in [
        ("plain", ""),
        ("btree", "CREATE INDEX btree_t ON btree(t) USING BTREE"),
        ("uniq", "CREATE UNIQUE INDEX uniq_t ON uniq(t)"),
        ("pair", "CREATE INDEX pair_a_t ON pair(a, t)"),
    ] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, a INTEGER, t INTEGER)"),
            (),
        )
        .unwrap();
        if !index.is_empty() {
            db.execute(index, ()).unwrap();
        }
        db.execute(
            &format!(
                "INSERT INTO {table} SELECT value, value % 3, value FROM generate_series(1, 900)"
            ),
            (),
        )
        .unwrap();
    }
}

fn check(db: &Database) {
    for where_clause in [
        "t > 600 AND t < 100",
        "t >= 600 AND t <= 100",
        "t BETWEEN 600 AND 100",
        "t > 100 AND t < 100",
        "t >= 100 AND t < 100",
        "t > 100 AND t <= 100",
        "a = 1 AND t > 600 AND t < 100",
        "a = 1 AND t > 100 AND t < 100",
        "a = 1 AND t BETWEEN 600 AND 100",
    ] {
        let sql = format!("SELECT id FROM {{t}} WHERE {where_clause}");
        assert!(
            ids(db, &sql.replace("{t}", "plain")).is_empty(),
            "{where_clause}"
        );
        for table in ["btree", "uniq", "pair"] {
            assert!(
                ids(db, &sql.replace("{t}", table)).is_empty(),
                "{table}: {where_clause}"
            );
        }
    }
}

#[test]
fn crossing_bounds_hold_nothing_in_memory() {
    let db = Database::open("memory://crossing_bounds_hold_nothing").unwrap();
    create(&db);
    check(&db);
}

#[test]
fn crossing_bounds_hold_nothing_hot_and_sealed() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    create(&db);
    check(&db);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    check(&db);
}

/// The index API answers the same bounds as SQL does
#[test]
fn crossing_bounds_hold_nothing_through_the_index_api() {
    use stoolap::core::{DataType, Value};
    use stoolap::storage::index::{BTreeIndex, MultiColumnIndex};
    use stoolap::storage::traits::Index;

    let btree = BTreeIndex::new(
        "b".into(),
        "t".into(),
        0,
        "t".into(),
        DataType::Integer,
        false,
        0,
    );
    let multi = MultiColumnIndex::new(
        "m".into(),
        "t".into(),
        vec!["a".into(), "t".into()],
        vec![0, 1],
        vec![DataType::Integer, DataType::Integer],
        false,
        0,
    );
    for id in 1..=900i64 {
        btree.add(&[Value::Integer(id)], id, id).unwrap();
        multi
            .add(&[Value::Integer(id % 3), Value::Integer(id)], id, id)
            .unwrap();
    }
    let int = |v: i64| Value::Integer(v);
    for (low, high, low_in, high_in) in [
        (600, 100, false, false),
        (600, 100, true, true),
        (100, 100, false, false),
        (100, 100, true, false),
        (100, 100, false, true),
    ] {
        let case = format!("{low} {low_in} .. {high} {high_in}");
        let found = btree.find_range(&[int(low)], &[int(high)], low_in, high_in);
        assert!(found.unwrap().is_empty(), "btree find_range {case}");
        let ids = btree.get_row_ids_in_range(&[int(low)], &[int(high)], low_in, high_in);
        assert!(ids.is_empty(), "btree get_row_ids_in_range {case}");
        let found = multi.find_range(&[int(1), int(low)], &[int(1), int(high)], low_in, high_in);
        assert!(found.unwrap().is_empty(), "multi find_range {case}");
    }
}
