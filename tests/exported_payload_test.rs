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

use std::sync::Mutex;
use stoolap::Database;

static EXPORT_TESTS: Mutex<()> = Mutex::new(());

fn exported(db: &Database) -> usize {
    db.engine()
        .memory_stats()
        .last()
        .unwrap()
        .exported_payload_bytes
}

fn populate(db: &Database) {
    db.execute(
        "CREATE TABLE items (id INTEGER PRIMARY KEY, label TEXT)",
        (),
    )
    .unwrap();
    let text = "retained payload ".repeat(256);
    for id in 1..=3 {
        db.execute("INSERT INTO items VALUES ($1, $2)", (id, text.as_str()))
            .unwrap();
    }
}

#[test]
fn results_keep_exported_payloads_after_truncate() {
    let _serial = EXPORT_TESTS.lock().unwrap();
    for (shape, sql) in [
        "SELECT * FROM items LIMIT 1",
        "SELECT label FROM items ORDER BY id LIMIT 1",
        "SELECT MIN(label) FROM items",
        "SELECT label, MAX(label) FROM items GROUP BY label",
        "WITH selected AS (SELECT * FROM items) SELECT label FROM selected LIMIT 1",
    ]
    .into_iter()
    .enumerate()
    {
        let db =
            Database::open(&format!("memory://results_keep_exported_payloads_{shape}")).unwrap();
        populate(&db);
        let before = exported(&db);
        let mut rows = db.query(sql, ()).unwrap();
        db.clone().execute("TRUNCATE TABLE items", ()).unwrap();
        let retained = exported(&db);
        assert!(
            retained >= before + 4096,
            "{sql}: exported payload must remain charged after TRUNCATE"
        );
        assert!(rows.next().unwrap().is_ok(), "{sql}");
        drop(rows);
        assert_eq!(
            exported(&db),
            before,
            "{sql}: result destruction must release its charge"
        );
    }
}

#[test]
fn cached_result_exports_outlive_cache_invalidation() {
    let _serial = EXPORT_TESTS.lock().unwrap();
    let db = Database::open("memory://cached_result_exports_outlive_cache_invalidation").unwrap();
    populate(&db);
    let before = exported(&db);
    drop(db.query("SELECT * FROM items WHERE id > 0", ()).unwrap());
    let cached = exported(&db);
    assert!(cached >= before + 4096);
    let rows = db.query("SELECT * FROM items WHERE id > 0", ()).unwrap();
    assert!(
        exported(&db) > cached,
        "cache hit must retain its own export charge"
    );
    db.execute("TRUNCATE TABLE items", ()).unwrap();
    assert!(
        exported(&db) >= before + 4096,
        "cache invalidation must not release the result's payload charge"
    );
    drop(rows);
    assert_eq!(exported(&db), before);
}

fn assert_normalized_default_retention(explicit: bool) {
    let _serial = EXPORT_TESTS.lock().unwrap();
    let db = Database::open(&format!("memory://normalized_schema_defaults_{explicit}")).unwrap();
    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO items VALUES (1)", ()).unwrap();
    let default = "schema default ".repeat(512);
    let before = exported(&db);
    db.execute(
        &format!("ALTER TABLE items ADD COLUMN label TEXT DEFAULT '{default}'"),
        (),
    )
    .unwrap();
    if explicit {
        db.execute("BEGIN", ()).unwrap();
    }
    let statement = db.prepare("SELECT * FROM items WHERE id = $1").unwrap();
    let mut rows = statement.query((1,)).unwrap();
    if explicit {
        db.execute("ROLLBACK", ()).unwrap();
    }
    db.clone().execute("DROP TABLE items", ()).unwrap();
    let retained = exported(&db);
    assert!(
        retained >= before + default.len(),
        "explicit={explicit}: default bytes {}, before {before}, retained {retained}",
        default.len(),
    );
    assert_eq!(
        rows.next().unwrap().unwrap().get::<String>(1).unwrap(),
        default,
    );
    drop(rows);
    drop(statement);
    assert_eq!(exported(&db), before);
}

#[test]
fn normalized_schema_defaults_stay_charged_after_table_drop() {
    assert_normalized_default_retention(false);
}

#[test]
fn transaction_schema_defaults_stay_charged_after_table_drop() {
    assert_normalized_default_retention(true);
}

#[test]
fn transaction_local_result_exports_outlive_rollback() {
    let _serial = EXPORT_TESTS.lock().unwrap();
    let db = Database::open("memory://transaction_local_result_exports_outlive_rollback").unwrap();
    populate(&db);
    let before = exported(&db);
    db.execute("BEGIN", ()).unwrap();
    db.execute(
        "UPDATE items SET label = $1 WHERE id = 1",
        ("new local payload ".repeat(512),),
    )
    .unwrap();
    let rows = db
        .query("SELECT label FROM items WHERE id = 1", ())
        .unwrap();
    db.execute("ROLLBACK", ()).unwrap();
    assert!(exported(&db) >= before + 8192);
    drop(rows);
    assert_eq!(exported(&db), before);
}

#[test]
fn prepared_result_exports_survive_thread_transfer_and_table_drop() {
    let _serial = EXPORT_TESTS.lock().unwrap();
    let db = Database::open("memory://prepared_result_exports_survive_thread_transfer").unwrap();
    populate(&db);
    let before = exported(&db);
    let statement = db.prepare("SELECT label FROM items WHERE id = $1").unwrap();
    let rows = statement.query((1,)).unwrap();
    db.clone().execute("DROP TABLE items", ()).unwrap();
    assert!(exported(&db) >= before + 4096);
    let rows = std::thread::spawn(move || {
        let mut rows = rows;
        assert!(
            rows.next()
                .unwrap()
                .unwrap()
                .get::<String>(0)
                .unwrap()
                .len()
                >= 4096
        );
        rows
    })
    .join()
    .unwrap();
    drop(rows);
    assert_eq!(exported(&db), before);
}
