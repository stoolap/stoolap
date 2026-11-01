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

use std::sync::{Mutex, MutexGuard};

use stoolap::{Database, Value};

// The general path counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn setup(name: &str, dir: &tempfile::TempDir, sealed: bool) -> Database {
    let dsn = match sealed {
        true => format!("file://{}/db", dir.path().display()),
        false => format!("memory://{name}"),
    };
    let db = Database::open(&dsn).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, k INTEGER)",
        "INSERT INTO t VALUES (1, 30, 1), (2, 10, 2), (3, NULL, 1), (4, 20, 3), (5, 10, 2)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER, s TEXT)",
        "INSERT INTO u VALUES (1, 1, 5, 'b'), (2, 2, 7, 'a'), (3, 1, NULL, 'c'), (4, 4, 1, 'a'), \
         (5, 2, 3, 'b')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn rows(db: &Database, sql: &str) -> Result<Vec<Vec<Value>>, String> {
    db.query(sql, ())
        .and_then(|rows| {
            rows.map(|row| row.map(|row| (0..row.len()).map(|i| row.get(i).unwrap()).collect()))
                .collect()
        })
        .map_err(|e| e.to_string())
}

/// Each statement with its twin, the block's ORDER BY position written as
/// the item it names
const SHAPES: &[(&str, &str)] = &[
    (
        "SELECT id, (SELECT v FROM u WHERE u.k = t.k ORDER BY 1 DESC LIMIT 1) FROM t ORDER BY id",
        "SELECT id, (SELECT v FROM u WHERE u.k = t.k ORDER BY v DESC LIMIT 1) FROM t ORDER BY id",
    ),
    (
        "SELECT id, (SELECT v FROM u WHERE u.k = t.k ORDER BY 1 NULLS LAST LIMIT 1) FROM t ORDER BY id",
        "SELECT id, (SELECT v FROM u WHERE u.k = t.k ORDER BY v NULLS LAST LIMIT 1) FROM t ORDER BY id",
    ),
    (
        "SELECT id, (SELECT v * 2 AS w FROM u WHERE u.k = t.k ORDER BY 1 LIMIT 1 OFFSET 1) \
         FROM t ORDER BY id",
        "SELECT id, (SELECT v * 2 AS w FROM u WHERE u.k = t.k ORDER BY v * 2 LIMIT 1 OFFSET 1) \
         FROM t ORDER BY id",
    ),
    (
        "SELECT id, (SELECT s FROM u WHERE u.v IS NOT NULL ORDER BY 1, id DESC LIMIT 1) \
         FROM t ORDER BY id",
        "SELECT id, (SELECT s FROM u WHERE u.v IS NOT NULL ORDER BY s, id DESC LIMIT 1) \
         FROM t ORDER BY id",
    ),
    (
        "SELECT id FROM t WHERE a IN (SELECT v * 2 FROM u ORDER BY 1 DESC LIMIT 2) ORDER BY id",
        "SELECT id FROM t WHERE a IN (SELECT v * 2 FROM u ORDER BY v * 2 DESC LIMIT 2) ORDER BY id",
    ),
    (
        "SELECT id FROM t WHERE (t.k, t.a) IN (SELECT k, v * 2 FROM u ORDER BY 2 DESC, 1 LIMIT 3) \
         ORDER BY id",
        "SELECT id FROM t WHERE (t.k, t.a) IN (SELECT k, v * 2 FROM u ORDER BY v * 2 DESC, k LIMIT 3) \
         ORDER BY id",
    ),
    (
        "SELECT id FROM t WHERE a > ALL (SELECT v FROM u WHERE v IS NOT NULL ORDER BY 1 LIMIT 2) \
         ORDER BY id",
        "SELECT id FROM t WHERE a > ALL (SELECT v FROM u WHERE v IS NOT NULL ORDER BY v LIMIT 2) \
         ORDER BY id",
    ),
    (
        "SELECT id, (SELECT k FROM u GROUP BY k ORDER BY 1 DESC LIMIT 1) FROM t ORDER BY id",
        "SELECT id, (SELECT k FROM u GROUP BY k ORDER BY k DESC LIMIT 1) FROM t ORDER BY id",
    ),
    (
        "SELECT id, (SELECT MAX(w) FROM (SELECT v AS w FROM u WHERE u.k = t.k ORDER BY 1 LIMIT 1) d) \
         FROM t ORDER BY id",
        "SELECT id, (SELECT MAX(w) FROM (SELECT v AS w FROM u WHERE u.k = t.k ORDER BY v LIMIT 1) d) \
         FROM t ORDER BY id",
    ),
];

#[test]
fn a_block_ordered_by_position_sorts_by_the_item_it_names() {
    let _guard = serial();
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = setup("block_order_position_answers", &dir, sealed);
        let mut wrong = Vec::new();
        for (sql, twin) in SHAPES {
            let want = rows(&db, twin).unwrap_or_else(|e| panic!("{twin}: {e}"));
            match rows(&db, sql) {
                Ok(got) if got == want => {}
                got => wrong.push(format!(
                    "sealed {sealed}: {sql}\n  got {got:?}\n  want {want:?}"
                )),
            }
        }
        assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
    }
}

#[test]
fn a_null_it_answers_keeps_its_items_type() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = setup("block_order_position_null_type", &dir, false);
    let got = rows(
        &db,
        "SELECT id, (SELECT v FROM u WHERE u.k = t.k ORDER BY 1 DESC LIMIT 1) FROM t WHERE id = 1",
    )
    .unwrap();
    let tags: Vec<String> = got.iter().map(|row| format!("{:?}", row[1])).collect();
    assert_eq!(
        tags,
        vec![format!(
            "{:?}",
            Value::Null(stoolap::core::DataType::Integer)
        )]
    );
}

#[test]
fn a_position_past_its_items_is_an_error() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = setup("block_order_position_past", &dir, false);
    let got = rows(
        &db,
        "SELECT id, (SELECT v FROM u WHERE u.k = t.k ORDER BY 2 LIMIT 1) FROM t",
    )
    .expect_err("a position past the items");
    assert!(got.contains("ORDER BY position 2"), "{got}");
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_block_ordered_by_position_runs_no_general_path_mechanism() {
    use stoolap::test_failpoints::general_path_total;
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let db = setup("block_order_position_general_path", &dir, false);
    for (sql, _) in SHAPES {
        let before = general_path_total();
        rows(&db, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
        assert_eq!(general_path_total(), before, "{sql}");
    }
}
