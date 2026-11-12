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

//! DISTINCT and COUNT(DISTINCT) over sealed volumes answer with the values
//! of the rows the statement sees, whichever rows tombstones, hot copies and
//! moved keys hide

use std::collections::BTreeSet;

use stoolap::core::Value;
use stoolap::Database;

const PER_VOLUME: i64 = 2000;
const VOLUMES: i64 = 3;

#[derive(Clone)]
struct Row {
    id: i64,
    i: Option<i64>,
    f: Option<f64>,
    ts: Option<i64>,
    t: Option<String>,
    b: Option<bool>,
    u: i64,
    /// A value of its own on every row
    h: Option<i64>,
}

fn row(id: i64) -> Row {
    let k = (id * 7919) % 300;
    Row {
        id,
        i: (id % 97 != 0).then_some(k),
        f: match id {
            _ if id % 89 == 0 => Some(f64::NAN),
            _ if id % 83 == 0 => None,
            _ => Some(k as f64 * 0.37 + 0.25),
        },
        ts: (id % 79 != 0).then_some(k),
        t: (id % 71 != 0).then(|| format!("k{k:04}")),
        b: (id % 61 != 0).then_some(k % 2 == 0),
        u: k % 40,
        h: (id % 53 != 0).then_some(id * 13),
    }
}

fn sql_of(r: &Row) -> String {
    let f = match r.f {
        Some(f) if f.is_nan() => "CAST('NaN' AS FLOAT)".to_string(),
        Some(f) => format!("{f:?}"),
        None => "NULL".to_string(),
    };
    let ts = r.ts.map_or("NULL".to_string(), |s| {
        format!(
            "'2024-01-01 {:02}:{:02}:{:02}'",
            s / 3600,
            s / 60 % 60,
            s % 60
        )
    });
    format!(
        "({}, {}, {f}, {ts}, {}, {}, {}, {})",
        r.id,
        r.i.map_or("NULL".to_string(), |i| i.to_string()),
        r.t.as_ref()
            .map_or("NULL".to_string(), |t| format!("'{t}'")),
        r.b.map_or("NULL".to_string(), |b| b.to_string()),
        r.u,
        r.h.map_or("NULL".to_string(), |h| h.to_string())
    )
}

/// The model's values of column `c`, rendered as the engine's are
fn model_values(model: &[Row], c: &str) -> BTreeSet<String> {
    model
        .iter()
        .map(|r| match c {
            "i" => r.i.map_or("NULL".into(), |i| i.to_string()),
            "f" => r.f.map_or("NULL".into(), |f| format!("{f:?}")),
            "ts" => r.ts.map_or("NULL".into(), |s| s.to_string()),
            "t" => r.t.clone().unwrap_or_else(|| "NULL".into()),
            "b" => r.b.map_or("NULL".into(), |b| b.to_string()),
            "u" => r.u.to_string(),
            "h" => r.h.map_or("NULL".into(), |h| h.to_string()),
            _ => unreachable!(),
        })
        .collect()
}

fn render(v: &Value) -> String {
    match v {
        Value::Null(_) => "NULL".into(),
        Value::Float(x) => format!("{x:?}"),
        Value::Integer(x) => x.to_string(),
        Value::Text(s) => s.to_string(),
        Value::Boolean(b) => b.to_string(),
        // Seconds past 2024-01-01, as the model keeps them
        Value::Timestamp(ts) => (ts.timestamp() - 1_704_067_200).to_string(),
        other => format!("{other:?}"),
    }
}

/// Every DISTINCT and COUNT(DISTINCT) of the table against the model
fn wrong_answers(db: &Database, model: &[Row], state: &str) -> Vec<String> {
    let mut wrong = Vec::new();
    for c in ["i", "f", "ts", "t", "b", "u", "h"] {
        let got: BTreeSet<String> = db
            .query(&format!("SELECT DISTINCT {c} FROM w"), ())
            .unwrap()
            .map(|row| render(row.unwrap().get_value(0).unwrap()))
            .collect();
        let want = model_values(model, c);
        if got != want {
            let missing: Vec<_> = want.difference(&got).take(5).collect();
            let extra: Vec<_> = got.difference(&want).take(5).collect();
            wrong.push(format!(
                "{state}: DISTINCT {c} lacks {missing:?}, holds {extra:?}"
            ));
        }
        let counted: i64 = db
            .query_one(&format!("SELECT COUNT(DISTINCT {c}) FROM w"), ())
            .unwrap();
        let non_null = want.iter().filter(|v| *v != "NULL").count() as i64;
        if counted != non_null {
            wrong.push(format!(
                "{state}: COUNT(DISTINCT {c}) {counted}, want {non_null}"
            ));
        }
    }
    wrong
}

/// Three sealed volumes, every column but `u` indexed
fn sealed(dir: &std::path::Path) -> (Database, Vec<Row>) {
    let db = Database::open(&format!(
        "file://{}?checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=100",
        dir.display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE w (id INTEGER PRIMARY KEY, i INTEGER, f FLOAT, ts TIMESTAMP, t TEXT, \
         b BOOLEAN, u INTEGER, h INTEGER)",
        (),
    )
    .unwrap();
    for c in ["i", "f", "ts", "t", "b", "h"] {
        db.execute(&format!("CREATE INDEX w_{c} ON w ({c})"), ())
            .unwrap();
    }
    let model: Vec<Row> = (1..=VOLUMES * PER_VOLUME).map(row).collect();
    for chunk in model.chunks(PER_VOLUME as usize) {
        for part in chunk.chunks(500) {
            let values: Vec<String> = part.iter().map(sql_of).collect();
            db.execute(&format!("INSERT INTO w VALUES {}", values.join(", ")), ())
                .unwrap();
        }
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    (db, model)
}

#[test]
fn a_sealed_distinct_answers_with_the_values_of_the_rows_it_sees() {
    let dir = tempfile::tempdir().unwrap();
    let (db, mut model) = sealed(dir.path());
    let mut wrong = wrong_answers(&db, &model, "sealed");

    // Every row of key 17 goes: its values leave each column
    db.execute("DELETE FROM w WHERE (id * 7919) % 300 = 17", ())
        .unwrap();
    model.retain(|r| (r.id * 7919) % 300 != 17);
    wrong.extend(wrong_answers(&db, &model, "key 17 deleted"));

    // Key 18 moves to 1000 in place in the hot store, its sealed copies hidden
    db.execute(
        "UPDATE w SET i = 1000, f = 1000.5, ts = '2024-01-02 00:00:00', t = 'moved', b = NULL, \
         u = 1000, h = h + 1000000 WHERE (id * 7919) % 300 = 18",
        (),
    )
    .unwrap();
    for r in model.iter_mut().filter(|r| (r.id * 7919) % 300 == 18) {
        r.i = Some(1000);
        r.f = Some(1000.5);
        r.ts = Some(86_400);
        r.t = Some("moved".into());
        r.b = None;
        r.u = 1000;
        r.h = r.h.map(|h| h + 1_000_000);
    }
    wrong.extend(wrong_answers(&db, &model, "key 18 moved"));

    // More rows hidden than a volume holds: each row's id is looked up
    db.execute("DELETE FROM w WHERE id % 3 = 0 OR id > 5000", ())
        .unwrap();
    model.retain(|r| r.id % 3 != 0 && r.id <= 5000);
    wrong.extend(wrong_answers(&db, &model, "most rows deleted"));
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_sealed_distinct_of_a_column_added_after_the_seal_answers_with_its_default() {
    let dir = tempfile::tempdir().unwrap();
    let (db, model) = sealed(dir.path());
    db.execute("ALTER TABLE w ADD COLUMN a INTEGER DEFAULT 7", ())
        .unwrap();
    db.execute("CREATE INDEX w_a ON w (a)", ()).unwrap();
    let values: Vec<i64> = db
        .query("SELECT DISTINCT a FROM w", ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert_eq!(values, [7], "every sealed row holds the default");
    let counted: i64 = db.query_one("SELECT COUNT(DISTINCT a) FROM w", ()).unwrap();
    assert_eq!(counted, 1);
    drop(model);
}

#[test]
fn a_snapshot_distinct_answers_with_the_values_it_began_with() {
    let dir = tempfile::tempdir().unwrap();
    let (db, model) = sealed(dir.path());
    let mut reader = db
        .begin_with_isolation(stoolap::IsolationLevel::SnapshotIsolation)
        .unwrap();
    reader.query("SELECT COUNT(*) FROM w", ()).unwrap();
    db.execute("UPDATE w SET t = 'later' WHERE (id * 7919) % 300 = 5", ())
        .unwrap();
    db.execute("DELETE FROM w WHERE (id * 7919) % 300 = 6", ())
        .unwrap();
    let got: BTreeSet<String> = reader
        .query("SELECT DISTINCT t FROM w", ())
        .unwrap()
        .map(|row| render(row.unwrap().get_value(0).unwrap()))
        .collect();
    assert_eq!(got, model_values(&model, "t"), "the snapshot's values");
    let counted: i64 = reader
        .query_one("SELECT COUNT(DISTINCT t) FROM w", ())
        .unwrap();
    assert_eq!(counted, model_values(&model, "t").len() as i64 - 1);
    reader.rollback().unwrap();
}

#[test]
fn a_sealed_distinct_counts_a_row_its_transaction_wrote_over_a_deleted_id() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    for sql in [
        "CREATE TABLE x (id INTEGER PRIMARY KEY, v INTEGER, s TEXT)",
        "CREATE INDEX x_v ON x (v)",
        "INSERT INTO x VALUES (1, 10, 'a')",
        "PRAGMA CHECKPOINT",
        "INSERT INTO x VALUES (2, 20, 'b')",
        "DELETE FROM x WHERE id = 2",
    ] {
        db.execute(sql, ()).unwrap();
    }
    // The deleted hot version of id 2 is still held when the transaction writes id 2
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO x VALUES (2, 30, 'c')", ()).unwrap();
    for (sql, want) in [
        ("SELECT COUNT(DISTINCT v) FROM x", 2),
        ("SELECT COUNT(DISTINCT s) FROM x", 2),
        ("SELECT COUNT(*) FROM (SELECT DISTINCT v FROM x) AS d", 2),
        ("SELECT COUNT(*) FROM (SELECT DISTINCT s FROM x) AS d", 2),
    ] {
        let got: i64 = tx.query_one(sql, ()).unwrap();
        assert_eq!(got, want, "{sql}");
    }
    tx.rollback().unwrap();
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_sealed_text_distinct_that_hides_no_row_reads_no_block() {
    use stoolap::test_failpoints::block_file_reads;
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=100",
        dir.path().display()
    );
    {
        let (db, _) = sealed(dir.path());
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    db.engine().cold_volumes_for_test("w");
    let before = block_file_reads();
    let got: i64 = db.query_one("SELECT COUNT(DISTINCT t) FROM w", ()).unwrap();
    let read = block_file_reads().0 - before.0;
    assert_eq!(got, 300);
    assert_eq!(read, 0, "the volumes' dictionaries answer");
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_sealed_numeric_distinct_that_hides_no_row_reads_no_block() {
    use stoolap::test_failpoints::block_file_reads;
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=100",
        dir.path().display()
    );
    let mut model = {
        let (db, model) = sealed(dir.path());
        db.execute("CREATE TABLE z (id INTEGER PRIMARY KEY, f FLOAT)", ())
            .unwrap();
        db.execute("CREATE INDEX z_f ON z (f)", ()).unwrap();
        let values: Vec<String> = model
            .iter()
            .map(|r| match r.f {
                Some(f) if !f.is_nan() => format!("({}, {f:?})", r.id),
                _ => format!("({}, NULL)", r.id),
            })
            .collect();
        db.execute(&format!("INSERT INTO z VALUES {}", values.join(", ")), ())
            .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
        model
    };
    let db = Database::open(&dsn).unwrap();
    db.engine().cold_volumes_for_test("w");
    db.engine().cold_volumes_for_test("z");
    for r in model.iter_mut() {
        r.f = r.f.filter(|f| !f.is_nan());
    }
    for (table, c) in [("w", "i"), ("w", "ts"), ("w", "h"), ("z", "f")] {
        let before = block_file_reads();
        let got: BTreeSet<String> = db
            .query(&format!("SELECT DISTINCT {c} FROM {table}"), ())
            .unwrap()
            .map(|row| render(row.unwrap().get_value(0).unwrap()))
            .collect();
        let read = block_file_reads().0 - before.0;
        assert_eq!(got, model_values(&model, c), "DISTINCT {c}");
        assert_eq!(read, 0, "the side files' keys answer DISTINCT {c}");
    }
}
