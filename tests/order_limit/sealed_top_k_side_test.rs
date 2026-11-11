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

//! A top-k read of an indexed column over scrambled sealed volumes, read
//! through the volumes' side files, answers as the whole sort does

use std::cmp::Ordering;

use stoolap::Database;

const PER_VOLUME: i64 = 3000;

/// One column's value in a row of the model
#[derive(Clone, Debug, PartialEq)]
enum Key {
    I(i64),
    T(String),
    F(f64),
    Nan,
    Null,
}

impl Key {
    /// The order of an ascending sort: values, then NaN, then NULL
    fn ascending(&self, other: &Key) -> Ordering {
        let rank = |k: &Key| match k {
            Key::Nan => 1,
            Key::Null => 2,
            _ => 0,
        };
        match (self, other) {
            (Key::I(a), Key::I(b)) => a.cmp(b),
            (Key::T(a), Key::T(b)) => a.cmp(b),
            (Key::F(a), Key::F(b)) => a.total_cmp(b),
            _ => rank(self).cmp(&rank(other)),
        }
    }
}

#[derive(Clone)]
struct Row {
    id: i64,
    i: Key,
    t: Key,
    f: Key,
    ts: Key,
    p: String,
}

impl Row {
    fn key(&self, column: &str) -> &Key {
        match column {
            "i" => &self.i,
            "t" => &self.t,
            "f" => &self.f,
            _ => &self.ts,
        }
    }

    fn values(&self) -> String {
        let i = match &self.i {
            Key::I(v) => v.to_string(),
            _ => "NULL".into(),
        };
        let t = match &self.t {
            Key::T(v) => format!("'{v}'"),
            _ => "NULL".into(),
        };
        let f = match &self.f {
            Key::F(v) => format!("{v:?}"),
            Key::Nan => "CAST('NaN' AS FLOAT)".into(),
            _ => "NULL".into(),
        };
        let ts = match &self.ts {
            Key::I(s) => format!(
                "'2024-01-01 {:02}:{:02}:{:02}'",
                s / 3600,
                s % 3600 / 60,
                s % 60
            ),
            _ => "NULL".into(),
        };
        format!("({}, {i}, {t}, {f}, {ts}, '{}')", self.id, self.p)
    }
}

/// Keys scrambled over the ids, each held by two or three rows across the
/// volumes; the third volume holds `i` on one row in a hundred
fn row(id: i64) -> Row {
    let key = (id * 7919) % 4000;
    let i = if id % 37 == 0 || (id > 2 * PER_VOLUME && id % 100 != 0) {
        Key::Null
    } else {
        Key::I(key)
    };
    let t = if id % 41 == 0 {
        Key::Null
    } else {
        Key::T(format!("k{key}"))
    };
    let f = match id {
        _ if id % 47 == 0 => Key::Null,
        _ if id % 43 == 0 => Key::Nan,
        _ => Key::F(key as f64 * 0.5 - 500.0),
    };
    let ts = if id % 31 == 0 { Key::Null } else { Key::I(key) };
    Row {
        id,
        i,
        t,
        f,
        ts,
        p: format!("p{id}"),
    }
}

fn insert(db: &Database, rows: &[Row]) {
    for chunk in rows.chunks(500) {
        let values: Vec<String> = chunk.iter().map(Row::values).collect();
        db.execute(&format!("INSERT INTO m VALUES {}", values.join(", ")), ())
            .unwrap();
    }
}

/// Three sealed volumes, then a deleted sealed row in fifty-three, two
/// sealed rows updated to the best and the worst keys, and hot rows
fn sealed(dir: &std::path::Path) -> (Database, Vec<Row>) {
    let db = Database::open(&format!(
        "file://{}?checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=100",
        dir.display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE m (id INTEGER PRIMARY KEY, i INTEGER, t TEXT, f FLOAT, ts TIMESTAMP, p TEXT)",
        (),
    )
    .unwrap();
    for column in ["i", "t", "f", "ts"] {
        db.execute(
            &format!("CREATE INDEX m_{column} ON m({column}) USING BTREE"),
            (),
        )
        .unwrap();
    }
    let mut model: Vec<Row> = (1..=3 * PER_VOLUME).map(row).collect();
    for volume in model.chunks(PER_VOLUME as usize) {
        insert(&db, volume);
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db.execute("DELETE FROM m WHERE id % 53 = 0", ()).unwrap();
    model.retain(|r| r.id % 53 != 0);
    db.execute(
        "UPDATE m SET i = -5, t = 'a', f = -9999.0, ts = '2023-12-31 00:00:00' WHERE id = 4321",
        (),
    )
    .unwrap();
    db.execute(
        "UPDATE m SET i = 99999, t = 'zz', f = 99999.0, ts = '2025-01-01 00:00:00' WHERE id = 1234",
        (),
    )
    .unwrap();
    for r in model.iter_mut() {
        if r.id == 4321 {
            (r.i, r.t, r.f, r.ts) = (
                Key::I(-5),
                Key::T("a".into()),
                Key::F(-9999.0),
                Key::I(-86400),
            );
        }
        if r.id == 1234 {
            let late = 366 * 86400;
            (r.i, r.t, r.f, r.ts) = (
                Key::I(99999),
                Key::T("zz".into()),
                Key::F(99999.0),
                Key::I(late),
            );
        }
    }
    let hot: Vec<Row> = (9001..=9010).map(|id| row(id - 9000 + 17)).collect();
    let hot: Vec<Row> = hot
        .into_iter()
        .enumerate()
        .map(|(n, r)| Row {
            id: 9001 + n as i64,
            ..r
        })
        .collect();
    insert(&db, &hot);
    model.extend(hot);
    (db, model)
}

/// The statements, each with its WHERE as the model reads it; `{c}` is the
/// column's name in the statement
type Case = (&'static str, fn(&Row, &str) -> bool, bool, usize, usize);

fn cases() -> Vec<Case> {
    fn any(_: &Row, _: &str) -> bool {
        true
    }
    fn below(r: &Row, c: &str) -> bool {
        r.key(c).ascending(&bound(c)) == Ordering::Less
    }
    fn at_least(r: &Row, c: &str) -> bool {
        r.key(c) != &Key::Null && r.key(c).ascending(&bound(c)) != Ordering::Less
    }
    fn not_null(r: &Row, c: &str) -> bool {
        r.key(c) != &Key::Null
    }
    fn some_ids(r: &Row, _: &str) -> bool {
        r.id % 37 < 2
    }
    fn third_volume(r: &Row, _: &str) -> bool {
        r.id > 2 * PER_VOLUME
    }
    fn later_p(r: &Row, _: &str) -> bool {
        r.p.as_str() >= "p5"
    }
    vec![
        ("ORDER BY {c} LIMIT 7", any, true, 7, 0),
        ("ORDER BY {c} DESC LIMIT 7", any, false, 7, 0),
        ("ORDER BY {c} LIMIT 5 OFFSET 3", any, true, 5, 3),
        (
            "WHERE {c} < {b} ORDER BY {c} DESC LIMIT 6",
            below,
            false,
            6,
            0,
        ),
        (
            "WHERE {c} >= {b} ORDER BY {c} LIMIT 6",
            at_least,
            true,
            6,
            0,
        ),
        (
            "WHERE {c} IS NOT NULL ORDER BY {c} DESC LIMIT 6",
            not_null,
            false,
            6,
            0,
        ),
        (
            "WHERE id % 37 < 2 ORDER BY {c} LIMIT 600",
            some_ids,
            true,
            600,
            0,
        ),
        (
            "WHERE id > 6000 ORDER BY {c} LIMIT 40",
            third_volume,
            true,
            40,
            0,
        ),
        ("WHERE p >= 'p5' ORDER BY {c} LIMIT 9", later_p, true, 9, 0),
        // The side file holds neither NULL nor NaN: a volume is read again
        // for them, past the rows its side file rounds kept
        (
            "WHERE {c} < {l} OR {c} IS NULL ORDER BY {c} LIMIT 30",
            low_or_null,
            true,
            30,
            0,
        ),
        (
            "WHERE {c} < {l} OR {c} IS NULL ORDER BY {c} LIMIT 4000",
            low_or_null,
            true,
            4000,
            0,
        ),
    ]
}

/// A low bound the cases compare a column with, as the model and as SQL
fn low(column: &str) -> (Key, &'static str) {
    match column {
        "i" => (Key::I(30), "30"),
        "t" => (Key::T("k104".into()), "'k104'"),
        "f" => (Key::F(-485.0), "-485.0"),
        _ => (Key::I(30), "'2024-01-01 00:00:30'"),
    }
}

fn low_or_null(r: &Row, c: &str) -> bool {
    let key = r.key(c);
    key == &Key::Null || key.ascending(&low(c).0) == Ordering::Less
}

/// The bound the cases compare a column with, as the model and as SQL
fn bound(column: &str) -> Key {
    match column {
        "i" => Key::I(2000),
        "t" => Key::T("k2".into()),
        "f" => Key::F(0.0),
        _ => Key::I(2000),
    }
}

fn bound_sql(column: &str) -> &'static str {
    match column {
        "i" => "2000",
        "t" => "'k2'",
        "f" => "0.0",
        _ => "'2024-01-01 00:33:20'",
    }
}

/// The model's keys of the rows the statement returns, in order
fn want(model: &[Row], column: &str, case: &Case) -> Vec<Key> {
    let (_, keeps, ascending, limit, offset) = *case;
    let mut keys: Vec<Key> = model
        .iter()
        .filter(|r| keeps(r, column))
        .map(|r| r.key(column).clone())
        .collect();
    keys.sort_by(|a, b| {
        if ascending {
            a.ascending(b)
        } else {
            b.ascending(a)
        }
    });
    keys.into_iter().skip(offset).take(limit).collect()
}

/// Runs every case on `name`, the statement's name of model column
/// `column`; the side file rounds the reads took, when the build counts them
fn check(db: &Database, model: &[Row], name: &str, column: &str) -> usize {
    #[cfg(feature = "test-failpoints")]
    let before = stoolap::test_failpoints::top_k_side_rounds();
    for case in cases() {
        let sql = format!(
            "SELECT id FROM m {}",
            case.0
                .replace("{c}", name)
                .replace("{b}", bound_sql(column))
                .replace("{l}", low(column).1)
        );
        let ids: Vec<i64> = db
            .query(&sql, ())
            .unwrap()
            .map(|row| row.unwrap().get(0).unwrap())
            .collect();
        let mut distinct = ids.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), ids.len(), "{sql}: a row came twice");
        let got: Vec<Key> = ids
            .iter()
            .map(|id| {
                let row = model.iter().find(|r| r.id == *id);
                row.unwrap_or_else(|| panic!("{sql}: row {id} is not in the table"))
                    .key(column)
                    .clone()
            })
            .collect();
        assert_eq!(got, want(model, column, &case), "{sql}");
    }
    #[cfg(feature = "test-failpoints")]
    {
        stoolap::test_failpoints::top_k_side_rounds() - before
    }
    #[cfg(not(feature = "test-failpoints"))]
    {
        0
    }
}

#[test]
fn a_sealed_top_k_through_side_files_answers_as_the_sort_does() {
    let dir = tempfile::tempdir().unwrap();
    let (db, model) = sealed(dir.path());
    for column in ["i", "t", "f", "ts"] {
        let rounds = check(&db, &model, column, column);
        if cfg!(feature = "test-failpoints") {
            assert!(rounds > 0, "{column}: no read went through a side file");
        }
    }
    db.execute("ALTER TABLE m RENAME COLUMN i TO j", ())
        .unwrap();
    let rounds = check(&db, &model, "j", "i");
    if cfg!(feature = "test-failpoints") {
        assert!(rounds > 0, "j: no read went through a side file");
    }
}

#[test]
fn a_sealed_top_k_held_full_by_hot_nulls_still_takes_the_sealed_nans() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=100",
        dir.path().display()
    ))
    .unwrap();
    db.execute("CREATE TABLE n (id INTEGER PRIMARY KEY, f FLOAT)", ())
        .unwrap();
    db.execute("CREATE INDEX n_f ON n(f) USING BTREE", ())
        .unwrap();
    db.execute(
        "INSERT INTO n SELECT value, CASE WHEN value = 7 THEN 2.5 WHEN value = 9 THEN 1.5 \
         ELSE CAST('NaN' AS FLOAT) END FROM generate_series(1, 3000)",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute(
        "INSERT INTO n SELECT value, NULL FROM generate_series(3001, 3010)",
        (),
    )
    .unwrap();
    let rows: Vec<(i64, String)> = db
        .query("SELECT id, f FROM n ORDER BY f LIMIT 10", ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get(0).unwrap(), row.get_value(1).unwrap().to_string())
        })
        .collect();
    let keys: Vec<&str> = rows.iter().map(|(_, f)| f.as_str()).collect();
    assert_eq!(
        keys,
        ["1.5", "2.5", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN"]
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_side_file_round_whose_walk_is_refused_reads_the_volume_once() {
    use stoolap::test_failpoints::{refuse_side_walks, top_k_side_rounds};
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=100",
        dir.path().display()
    ))
    .unwrap();
    db.execute("CREATE TABLE w (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("CREATE INDEX w_k ON w(k) USING BTREE", ())
        .unwrap();
    db.execute(
        "INSERT INTO w SELECT value, (value * 7919) % 200000 FROM generate_series(1, 200000)",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let before = top_k_side_rounds();
    refuse_side_walks(usize::MAX);
    let keys: Vec<i64> = db
        .query(
            "SELECT k FROM w WHERE k IN (49999, 149999) ORDER BY k LIMIT 5",
            (),
        )
        .unwrap()
        .map(|row| row.unwrap().get(0).unwrap())
        .collect();
    refuse_side_walks(0);
    assert_eq!(keys, [49_999, 149_999]);
    assert_eq!(
        top_k_side_rounds() - before,
        1,
        "the round that scanned the volume was followed by more"
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_sealed_top_k_through_a_side_file_loads_few_row_groups() {
    use stoolap::test_failpoints::take_group_column_decodes;
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!(
        "file://{}?checkpoint_on_close=off&checkpoint_interval=0&compact_threshold=100",
        dir.path().display()
    );
    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE g (id INTEGER PRIMARY KEY, k INTEGER, t TEXT)",
            (),
        )
        .unwrap();
        db.execute("CREATE INDEX g_k ON g(k) USING BTREE", ())
            .unwrap();
        db.execute("CREATE INDEX g_t ON g(t) USING BTREE", ())
            .unwrap();
        for volume in 0..2 {
            db.execute(
                &format!(
                    "INSERT INTO g SELECT value, (value * 7919) % 400000, \
                     'k' || ((value * 7919) % 400000) FROM generate_series({}, {})",
                    volume * 200_000 + 1,
                    (volume + 1) * 200_000
                ),
                (),
            )
            .unwrap();
            db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        }
        db.close().unwrap();
    }
    // The statement, its order column and the key it returns
    let cases: [(&str, usize, i64); 4] = [
        (
            "SELECT id FROM g WHERE k > 200000 ORDER BY k LIMIT 1",
            1,
            200_001,
        ),
        (
            "SELECT id FROM g WHERE k < 200000 ORDER BY k DESC LIMIT 1",
            1,
            199_999,
        ),
        ("SELECT id FROM g WHERE t > 'k5' ORDER BY t LIMIT 1", 2, 50),
        (
            "SELECT id FROM g WHERE t < 'k5' ORDER BY t DESC LIMIT 1",
            2,
            49_999,
        ),
    ];
    for (sql, column, key) in cases {
        // Reopened, no block of the volumes is decoded yet
        let db = Database::open(&dsn).unwrap();
        take_group_column_decodes();
        let id: i64 = db.query_one(sql, ()).unwrap();
        let decodes = take_group_column_decodes();
        db.close().unwrap();
        assert_eq!((id * 7919) % 400_000, key, "{sql}");
        let read = decodes.iter().filter(|(c, _)| *c == column).count();
        // Of two volumes of four row groups each, one group of each at most
        assert!(
            read <= 2,
            "{sql}: decoded {read} groups of the order column"
        );
    }
}
