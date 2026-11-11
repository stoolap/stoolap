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

//! A COUNT(*) over sealed volumes whose WHERE bounds an indexed column
//! counts the rows the side files name, as a scan counts them

use stoolap::Database;

const ROWS: i64 = 6000;
const PER_VOLUME: i64 = 2000;

#[derive(Clone)]
struct Row {
    id: i64,
    i: Option<i64>,
    f: Option<f64>,
    ts: Option<i64>,
    p: i64,
    t: Option<String>,
    h: Option<String>,
}

fn row(id: i64) -> Row {
    let i = (id * 7919) % 3000;
    Row {
        id,
        t: (id % 71 != 0).then(|| format!("k{:04}", i % 500)),
        h: (id % 73 != 0).then(|| format!("h{:04}", i % 300)),
        i: (id % 97 != 0).then_some(i),
        f: match id {
            _ if id % 89 == 0 => Some(f64::NAN),
            _ if id % 83 == 0 => None,
            5 => Some(-0.0),
            _ => Some(i as f64 * 0.5),
        },
        ts: (id % 79 != 0).then_some(i),
        p: id % 10,
    }
}

fn sql_of(r: &Row) -> String {
    let f = match r.f {
        Some(f) if f.is_nan() => "CAST('NaN' AS FLOAT)".to_string(),
        Some(f) if f == 0.0 && f.is_sign_negative() => "CAST('-0.0' AS FLOAT)".to_string(),
        Some(f) => format!("{f:?}"),
        None => "NULL".to_string(),
    };
    let opt = |v: Option<i64>| v.map_or("NULL".to_string(), |v| v.to_string());
    let text = |v: &Option<String>| v.as_ref().map_or("NULL".to_string(), |v| format!("'{v}'"));
    let ts = r.ts.map_or("NULL".to_string(), |s| {
        format!(
            "'2024-01-01 {:02}:{:02}:{:02}'",
            s / 3600,
            s / 60 % 60,
            s % 60
        )
    });
    format!(
        "({}, {}, {f}, {ts}, {}, {}, {})",
        r.id,
        opt(r.i),
        r.p,
        text(&r.t),
        text(&r.h)
    )
}

/// Three sealed volumes, then rows deleted, rows updated in place into the
/// hot store, and rows whose indexed key an update moved
fn sealed(dir: &std::path::Path) -> (Database, Vec<Row>) {
    let db = Database::open(&format!("file://{}?checkpoint_on_close=off", dir.display())).unwrap();
    for sql in [
        "CREATE TABLE w (id INTEGER PRIMARY KEY, i INTEGER, f FLOAT, ts TIMESTAMP, p INTEGER, \
         t TEXT, h TEXT)",
        "CREATE INDEX w_i ON w (i)",
        "CREATE INDEX w_f ON w (f)",
        "CREATE INDEX w_ts ON w (ts)",
        "CREATE INDEX w_t ON w (t) USING BTREE",
        "CREATE INDEX w_h ON w (h) USING HASH",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut model: Vec<Row> = (1..=ROWS).map(row).collect();
    for chunk in model.chunks(PER_VOLUME as usize) {
        for part in chunk.chunks(500) {
            let values: Vec<String> = part.iter().map(sql_of).collect();
            db.execute(&format!("INSERT INTO w VALUES {}", values.join(", ")), ())
                .unwrap();
        }
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db.execute("DELETE FROM w WHERE id % 50 = 0", ()).unwrap();
    model.retain(|r| r.id % 50 != 0);
    db.execute("UPDATE w SET p = p WHERE id % 60 = 1", ())
        .unwrap();
    db.execute(
        "UPDATE w SET i = i + 1, f = f + 0.5, ts = ts + INTERVAL '1 second', t = t || 'x', \
         h = h || 'x' WHERE id % 70 = 2",
        (),
    )
    .unwrap();
    for r in model.iter_mut().filter(|r| r.id % 70 == 2) {
        r.i = r.i.map(|i| i + 1);
        r.f = r.f.map(|f| f + 0.5);
        r.ts = r.ts.map(|s| s + 1);
        r.t = r.t.as_ref().map(|t| format!("{t}x"));
        r.h = r.h.as_ref().map(|h| format!("{h}x"));
    }
    (db, model)
}

type Case = (&'static str, fn(&Row) -> bool);

fn feq(f: Option<f64>, x: f64) -> bool {
    f.is_some_and(|f| f == x || (f.is_nan() && x.is_nan()))
}

fn fgt(f: Option<f64>, x: f64) -> bool {
    f.is_some_and(|f| f.is_nan() || f > x)
}

const CASES: &[Case] = &[
    ("i = 17", |r| r.i == Some(17)),
    ("i = 0", |r| r.i == Some(0)),
    ("i = 18", |r| r.i == Some(18)),
    ("i >= 100 AND i < 200", |r| {
        r.i.is_some_and(|i| (100..200).contains(&i))
    }),
    ("i > 2990", |r| r.i.is_some_and(|i| i > 2990)),
    ("i = 17 AND p = 3", |r| r.i == Some(17) && r.p == 3),
    ("i >= 10 AND i <= 40 AND p < 5", |r| {
        r.i.is_some_and(|i| (10..=40).contains(&i)) && r.p < 5
    }),
    ("i < 0", |r| r.i.is_some_and(|i| i < 0)),
    ("f = 8.5", |r| feq(r.f, 8.5)),
    ("f = 0.0", |r| feq(r.f, 0.0)),
    ("f > 1490.0", |r| fgt(r.f, 1490.0)),
    ("f >= 10.0 AND f < 20.0", |r| {
        r.f.is_some_and(|f| !f.is_nan() && (10.0..20.0).contains(&f))
    }),
    ("ts = '2024-01-01 00:00:17'", |r| r.ts == Some(17)),
    (
        "ts >= '2024-01-01 00:10:00' AND ts < '2024-01-01 00:12:00'",
        |r| r.ts.is_some_and(|s| (600..720).contains(&s)),
    ),
    ("t = 'k0017'", |r| r.t.as_deref() == Some("k0017")),
    ("t = 'k0017' AND p = 3", |r| {
        r.t.as_deref() == Some("k0017") && r.p == 3
    }),
    ("t = 'k0002x'", |r| r.t.as_deref() == Some("k0002x")),
    ("t = 'nowhere'", |r| r.t.as_deref() == Some("nowhere")),
    // Keys of id 2, whose key an update moved, and of id 50, deleted
    ("t = 'k0338'", |r| r.t.as_deref() == Some("k0338")),
    ("t = 'k0450'", |r| r.t.as_deref() == Some("k0450")),
    ("h = 'h0238'", |r| r.h.as_deref() == Some("h0238")),
    ("h = 'h0250'", |r| r.h.as_deref() == Some("h0250")),
    ("h = 'h0042'", |r| r.h.as_deref() == Some("h0042")),
    ("h = 'h0042' AND i < 1500", |r| {
        r.h.as_deref() == Some("h0042") && r.i.is_some_and(|i| i < 1500)
    }),
];

#[test]
fn a_sealed_count_through_side_files_counts_as_the_scan_does() {
    let dir = tempfile::tempdir().unwrap();
    let (db, model) = sealed(dir.path());
    let mut wrong = Vec::new();
    for &(predicate, holds) in CASES {
        let sql = format!("SELECT COUNT(*) FROM w WHERE {predicate}");
        let got: i64 = db.query_one(&sql, ()).unwrap();
        let want = model.iter().filter(|r| holds(r)).count() as i64;
        if got != want {
            wrong.push(format!("{sql}: got {got}, want {want}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The calls a statement makes over the rows its WHERE passes, as text
type Calls = fn(&[&Row]) -> String;

/// The calls, the WHERE, the model's WHERE and the model's answer
type AggregateCase = (&'static str, &'static str, fn(&Row) -> bool, Calls);

fn text_of(v: Option<i64>) -> String {
    v.map_or("NULL".to_string(), |v| v.to_string())
}

#[test]
fn a_sealed_aggregate_through_side_files_answers_as_the_scan_does() {
    let dir = tempfile::tempdir().unwrap();
    let (db, model) = sealed(dir.path());
    let sum_p: Calls = |rows| text_of((!rows.is_empty()).then(|| rows.iter().map(|r| r.p).sum()));
    let min_max_i: Calls = |rows| {
        let keys = rows.iter().filter_map(|r| r.i);
        format!("{}|{}", text_of(keys.clone().min()), text_of(keys.max()))
    };
    let counts: Calls = |rows| {
        let f = rows.iter().filter(|r| r.f.is_some()).count();
        format!("{}|{f}", rows.len())
    };
    let cases: [AggregateCase; 6] = [
        ("SUM(p)", "i = 17", |r| r.i == Some(17), sum_p),
        (
            "SUM(p)",
            "t = 'k0338'",
            |r| r.t.as_deref() == Some("k0338"),
            sum_p,
        ),
        (
            "MIN(i), MAX(i)",
            "h = 'h0238'",
            |r| r.h.as_deref() == Some("h0238"),
            min_max_i,
        ),
        (
            "MIN(i), MAX(i)",
            "i >= 100 AND i < 200 AND p < 5",
            |r| r.i.is_some_and(|i| (100..200).contains(&i)) && r.p < 5,
            min_max_i,
        ),
        (
            "COUNT(*), COUNT(f)",
            "ts >= '2024-01-01 00:10:00' AND ts < '2024-01-01 00:12:00'",
            |r| r.ts.is_some_and(|s| (600..720).contains(&s)),
            counts,
        ),
        (
            "SUM(p)",
            "t = 'nowhere'",
            |r| r.t.as_deref() == Some("nowhere"),
            sum_p,
        ),
    ];
    let mut wrong = Vec::new();
    for (calls, predicate, holds, answer) in cases {
        let sql = format!("SELECT {calls} FROM w WHERE {predicate}");
        let row = db.query(&sql, ()).unwrap().next().unwrap().unwrap();
        let got: Vec<String> = (0..row.len())
            .map(|i| row.get_value(i).unwrap().to_string())
            .collect();
        let passing: Vec<&Row> = model.iter().filter(|r| holds(r)).collect();
        let want = answer(&passing);
        if got.join("|") != want {
            wrong.push(format!("{sql}: got {}, want {want}", got.join("|")));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_sealed_count_of_one_key_reads_the_side_file_and_at_most_one_block() {
    use stoolap::test_failpoints::{block_file_reads, side_file_reads};
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=100",
        dir.path().display()
    );
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE x (id INTEGER PRIMARY KEY, k INTEGER, s TEXT, h TEXT)",
            (),
        )
        .unwrap();
        db.execute("CREATE INDEX x_k ON x(k)", ()).unwrap();
        db.execute("CREATE INDEX x_s ON x(s) USING BTREE", ())
            .unwrap();
        db.execute("CREATE INDEX x_h ON x(h) USING HASH", ())
            .unwrap();
        db.execute(
            "INSERT INTO x SELECT value, (value * 7919) % 200000, \
             'k' || ((value * 7919) % 200000), 'h' || ((value * 7919) % 200000) \
             FROM generate_series(1, 200000)",
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        db.close().unwrap();
    }
    let mut wrong = Vec::new();
    for (sql, want) in [
        ("SELECT COUNT(*) FROM x WHERE k = 1", 1),
        ("SELECT COUNT(*) FROM x WHERE k = 150000", 1),
        ("SELECT COUNT(*) FROM x WHERE s = 'k1'", 1),
        ("SELECT COUNT(*) FROM x WHERE h = 'h150000'", 1),
    ] {
        // Reopened and evicted, no block of the volume is held
        let db = Database::open(&dsn).unwrap();
        db.engine().cold_volumes_for_test("x");
        let (before, before_sides) = (block_file_reads(), side_file_reads());
        let got: i64 = db.query_one(sql, ()).unwrap();
        let read = block_file_reads().0 - before.0;
        let after = side_file_reads();
        let (sides, fenced) = (after.0 - before_sides.0, after.1 - before_sides.1);
        db.close().unwrap();
        if got != want {
            wrong.push(format!("{sql}: got {got}, want {want}"));
        }
        if read > 1 || sides == 0 || fenced > 0 {
            wrong.push(format!(
                "{sql}: read {read} blocks, {sides} side file pages, {fenced} under the seal fence"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
