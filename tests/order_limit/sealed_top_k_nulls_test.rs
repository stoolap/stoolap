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

//! A top-k read over sealed and hot rows of a nullable or FLOAT column
//! places NULL and NaN as the whole sort does

use stoolap::Database;

/// Two volumes, the second with `i` all NULL and `f` NaN or NULL, then hot
/// rows, a sealed row updated and a sealed row deleted
fn sealed(dir: &std::path::Path) -> Database {
    let db = Database::open(&format!("file://{}?checkpoint_on_close=off", dir.display())).unwrap();
    db.execute(
        "CREATE TABLE s (id INTEGER PRIMARY KEY, i INTEGER, t TEXT, f FLOAT, ts TIMESTAMP)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX s_i ON s (i)", ()).unwrap();
    let row = |id: i64, i: &str, t: &str, f: &str, ts: &str| {
        format!("INSERT INTO s VALUES ({id}, {i}, {t}, {f}, {ts})")
    };
    let day = |d: i64| format!("'2024-01-{d:02} 00:00:00'");
    for id in 1..=10 {
        let i = if id == 3 {
            "NULL".into()
        } else {
            (id * 10).to_string()
        };
        let t = if id == 4 {
            "NULL".into()
        } else {
            format!("'t{id:02}'")
        };
        let f = match id {
            5 => "CAST('NaN' AS FLOAT)".into(),
            6 => "NULL".into(),
            _ => format!("{id}.5"),
        };
        let ts = if id == 7 { "NULL".into() } else { day(id + 1) };
        db.execute(&row(id, &i, &t, &f, &ts), ()).unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    for id in 11..=15 {
        let f = if id < 13 {
            "NULL"
        } else {
            "CAST('NaN' AS FLOAT)"
        };
        db.execute(&row(id, "NULL", &format!("'t{id}'"), f, &day(id + 1)), ())
            .unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute(&row(16, "5", "'a16'", "0.25", &day(1)), ())
        .unwrap();
    db.execute(
        &row(17, "155", "'t99'", "CAST('NaN' AS FLOAT)", &day(30)),
        (),
    )
    .unwrap();
    db.execute(&row(18, "NULL", "NULL", "NULL", "NULL"), ())
        .unwrap();
    db.execute("UPDATE s SET i = 1 WHERE id = 10", ()).unwrap();
    db.execute("DELETE FROM s WHERE id = 9", ()).unwrap();
    db
}

/// The sort column's values in order, and the ids, sorted: rows tied on
/// the key may come in any order
fn read(db: &Database, sql: &str) -> (Vec<String>, Vec<i64>) {
    let mut keys = Vec::new();
    let mut ids = Vec::new();
    for row in db.query(sql, ()).unwrap() {
        let row = row.unwrap();
        ids.push(row.get::<i64>(0).unwrap());
        keys.push(row.get_value(1).unwrap().to_string());
    }
    ids.sort_unstable();
    (keys, ids)
}

/// Whether the top-k read answered `sql`; None in a build that cannot tell
fn taken(db: &Database, sql: &str) -> Option<bool> {
    #[cfg(feature = "test-failpoints")]
    {
        stoolap::test_failpoints::reset_order_limit_paths();
        db.query(sql, ()).unwrap().for_each(drop);
        Some(stoolap::test_failpoints::order_limit_paths_taken().contains(&"prepared_top_k"))
    }
    #[cfg(not(feature = "test-failpoints"))]
    {
        let _ = (db, sql);
        None
    }
}

/// The statement, whether the top-k read answers it, the sort column's
/// values and the ids
type Case = (&'static str, bool, &'static [&'static str], &'static [i64]);

const CASES: &[Case] = &[
    // NULLs last ascending, first descending, unless written
    (
        "SELECT id, i FROM s ORDER BY i LIMIT 3",
        true,
        &["1", "5", "10"],
        &[1, 10, 16],
    ),
    (
        "SELECT id, i FROM s ORDER BY i DESC LIMIT 9",
        true,
        &[
            "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "155", "80",
        ],
        &[3, 8, 11, 12, 13, 14, 15, 17, 18],
    ),
    (
        "SELECT id, i FROM s ORDER BY i DESC NULLS LAST LIMIT 3",
        true,
        &["155", "80", "70"],
        &[7, 8, 17],
    ),
    (
        "SELECT id, i FROM s ORDER BY i NULLS FIRST LIMIT 9",
        true,
        &[
            "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "1", "5",
        ],
        &[3, 10, 11, 12, 13, 14, 15, 16, 18],
    ),
    // Fewer values than the limit: the NULLs follow
    (
        "SELECT id, i FROM s ORDER BY i LIMIT 17",
        true,
        &[
            "1", "5", "10", "20", "40", "50", "60", "70", "80", "155", "NULL", "NULL", "NULL",
            "NULL", "NULL", "NULL", "NULL",
        ],
        &[1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    ),
    // A WHERE on the column keeps every NULL out
    (
        "SELECT id, i FROM s WHERE i > 15 ORDER BY i DESC LIMIT 3",
        true,
        &["155", "80", "70"],
        &[7, 8, 17],
    ),
    (
        "SELECT id, i FROM s WHERE i > 15 ORDER BY i LIMIT 2 OFFSET 1",
        true,
        &["40", "50"],
        &[4, 5],
    ),
    // NaN after every number, NULL after NaN ascending
    (
        "SELECT id, f FROM s ORDER BY f LIMIT 3",
        true,
        &["0.25", "1.5", "2.5"],
        &[1, 2, 16],
    ),
    (
        "SELECT id, f FROM s ORDER BY f LIMIT 13",
        true,
        &[
            "0.25", "1.5", "2.5", "3.5", "4.5", "7.5", "8.5", "10.5", "NaN", "NaN", "NaN", "NaN",
            "NaN",
        ],
        &[1, 2, 3, 4, 5, 7, 8, 10, 13, 14, 15, 16, 17],
    ),
    (
        "SELECT id, f FROM s ORDER BY f LIMIT 17",
        true,
        &[
            "0.25", "1.5", "2.5", "3.5", "4.5", "7.5", "8.5", "10.5", "NaN", "NaN", "NaN", "NaN",
            "NaN", "NULL", "NULL", "NULL", "NULL",
        ],
        &[1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    ),
    (
        "SELECT id, f FROM s WHERE f > 3 ORDER BY f LIMIT 2",
        true,
        &["3.5", "4.5"],
        &[3, 4],
    ),
    // Descending, NaN comes first: a volume that may hold one is read unbounded
    (
        "SELECT id, f FROM s ORDER BY f DESC LIMIT 9",
        true,
        &[
            "NULL", "NULL", "NULL", "NULL", "NaN", "NaN", "NaN", "NaN", "NaN",
        ],
        &[5, 6, 11, 12, 13, 14, 15, 17, 18],
    ),
    (
        "SELECT id, f FROM s ORDER BY f DESC LIMIT 12",
        true,
        &[
            "NULL", "NULL", "NULL", "NULL", "NaN", "NaN", "NaN", "NaN", "NaN", "10.5", "8.5", "7.5",
        ],
        &[5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18],
    ),
    (
        "SELECT id, f FROM s ORDER BY f DESC NULLS LAST LIMIT 7",
        true,
        &["NaN", "NaN", "NaN", "NaN", "NaN", "10.5", "8.5"],
        &[5, 8, 10, 13, 14, 15, 17],
    ),
    (
        "SELECT id, f FROM s WHERE f < 9 ORDER BY f DESC LIMIT 3",
        true,
        &["8.5", "7.5", "4.5"],
        &[4, 7, 8],
    ),
    (
        "SELECT id, t FROM s ORDER BY t LIMIT 2",
        true,
        &["a16", "t01"],
        &[1, 16],
    ),
    (
        "SELECT id, t FROM s ORDER BY t DESC LIMIT 4",
        true,
        &["NULL", "NULL", "t99", "t15"],
        &[4, 15, 17, 18],
    ),
    (
        "SELECT id, t FROM s WHERE t >= 't12' ORDER BY t DESC LIMIT 2",
        true,
        &["t99", "t15"],
        &[15, 17],
    ),
    (
        "SELECT id, ts FROM s ORDER BY ts DESC LIMIT 3",
        true,
        &["NULL", "NULL", "2024-01-30T00:00:00+00:00"],
        &[7, 17, 18],
    ),
    (
        "SELECT id, ts FROM s ORDER BY ts LIMIT 2",
        true,
        &["2024-01-01T00:00:00+00:00", "2024-01-02T00:00:00+00:00"],
        &[1, 16],
    ),
];

#[test]
fn a_sealed_top_k_places_null_and_nan_as_the_sort_does() {
    let dir = tempfile::tempdir().unwrap();
    let db = sealed(dir.path());
    let mut wrong = Vec::new();
    for &(sql, top_k, keys, ids) in CASES {
        let got = read(&db, sql);
        if got.0 != keys || got.1 != ids {
            wrong.push(format!("{sql}\n  got  {got:?}\n  want ({keys:?}, {ids:?})"));
        }
        if taken(&db, sql).is_some_and(|taken| taken != top_k) {
            wrong.push(format!("{sql}: top-k read taken is not {top_k}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_descending_float_top_k_reads_a_volume_holding_nan_past_its_zone_bound() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_on_close=off",
        dir.path().display()
    ))
    .unwrap();
    for (table, index) in [("m", false), ("n", true)] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, f FLOAT)"),
            (),
        )
        .unwrap();
        if index {
            db.execute(&format!("CREATE INDEX {table}_f ON {table} (f)"), ())
                .unwrap();
        }
        db.execute(
            &format!("INSERT INTO {table} SELECT value, value FROM generate_series(1, 1000)"),
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        // The newer volume's numbers are all below the older one's
        db.execute(
            &format!(
                "INSERT INTO {table} SELECT value, CASE WHEN value = 2000 \
                 THEN CAST('NaN' AS FLOAT) ELSE 0.5 END FROM generate_series(1001, 3000)"
            ),
            (),
        )
        .unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    // One volume of two row groups, the second's numbers below the first's
    db.execute("CREATE TABLE g (id INTEGER PRIMARY KEY, f FLOAT)", ())
        .unwrap();
    db.execute(
        "INSERT INTO g SELECT value, CASE WHEN value = 68000 THEN CAST('NaN' AS FLOAT) \
         WHEN value > 65536 THEN 0.5 ELSE value END FROM generate_series(1, 70000)",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let mut wrong = Vec::new();
    let grouped = (
        "SELECT id, f FROM g ORDER BY f DESC LIMIT 2".to_string(),
        vec!["NaN", "65536"],
        vec![65536, 68000],
    );
    for table in ["m", "n", "g"] {
        let cases = if table == "g" {
            vec![grouped.clone()]
        } else {
            vec![
                (
                    format!("SELECT id, f FROM {table} ORDER BY f DESC LIMIT 2"),
                    vec!["NaN", "1000"],
                    vec![1000, 2000],
                ),
                (
                    format!("SELECT id, f FROM {table} WHERE f > 999 ORDER BY f DESC LIMIT 2"),
                    vec!["NaN", "1000"],
                    vec![1000, 2000],
                ),
                (
                    format!("SELECT id, f FROM {table} WHERE f < 999 ORDER BY f DESC LIMIT 2"),
                    vec!["998", "997"],
                    vec![997, 998],
                ),
            ]
        };
        for (sql, keys, ids) in cases {
            let got = read(&db, &sql);
            if got.0 != keys || got.1 != ids {
                wrong.push(format!("{sql}\n  got  {got:?}\n  want ({keys:?}, {ids:?})"));
            }
            if taken(&db, &sql) == Some(false) {
                wrong.push(format!("{sql}: the top-k read did not answer"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_descending_float_top_k_through_a_side_file_takes_the_nan_its_range_holds() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_on_close=off",
        dir.path().display()
    ))
    .unwrap();
    for sql in [
        "CREATE TABLE d (id INTEGER PRIMARY KEY, f FLOAT)",
        "CREATE INDEX d_f ON d (f)",
        "INSERT INTO d SELECT value, CASE WHEN value = 2000 THEN CAST('NaN' AS FLOAT) \
         ELSE value END FROM generate_series(1, 2000)",
        "PRAGMA CHECKPOINT",
    ] {
        db.execute(sql, ()).unwrap();
    }
    // The statement, its keys and ids, and whether side file rounds answer it
    let cases: [(&str, &[&str], &[i64], bool); 4] = [
        (
            "SELECT id, f FROM d ORDER BY f DESC LIMIT 2",
            &["NaN", "1999"],
            &[1999, 2000],
            false,
        ),
        (
            "SELECT id, f FROM d WHERE f > 1000 ORDER BY f DESC LIMIT 2",
            &["NaN", "1999"],
            &[1999, 2000],
            false,
        ),
        (
            "SELECT id, f FROM d WHERE f < 1500 ORDER BY f DESC LIMIT 2",
            &["1499", "1498"],
            &[1498, 1499],
            true,
        ),
        (
            "SELECT id, f FROM d ORDER BY f LIMIT 2",
            &["1", "2"],
            &[1, 2],
            true,
        ),
    ];
    let mut wrong = Vec::new();
    for (sql, keys, ids, rounds) in cases {
        #[cfg(feature = "test-failpoints")]
        let before = stoolap::test_failpoints::top_k_side_rounds();
        let got = read(&db, sql);
        if got.0 != keys || got.1 != ids {
            wrong.push(format!("{sql}\n  got  {got:?}\n  want ({keys:?}, {ids:?})"));
        }
        #[cfg(feature = "test-failpoints")]
        if (stoolap::test_failpoints::top_k_side_rounds() > before) != rounds {
            wrong.push(format!("{sql}: side file rounds taken is not {rounds}"));
        }
        #[cfg(not(feature = "test-failpoints"))]
        let _ = rounds;
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The ids a statement returns, in order
fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect()
}

#[test]
fn a_sealed_top_k_takes_no_bound_from_a_json_or_vector_zone_map() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_on_close=off",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE x (id INTEGER PRIMARY KEY, j JSON, k JSON NOT NULL, v VECTOR(2))",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO x VALUES (1, '9', '9', '[1.0, 0.0]'), (2, '1', '1', '[0.5, 0.0]')",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("INSERT INTO x VALUES (3, '5', '5', '[9.0, 0.0]')", ())
        .unwrap();
    let cases: [(&str, &[i64]); 6] = [
        ("SELECT id FROM x ORDER BY j LIMIT 1", &[2]),
        ("SELECT id FROM x ORDER BY j DESC LIMIT 1", &[1]),
        ("SELECT id FROM x ORDER BY k LIMIT 1", &[2]),
        ("SELECT id FROM x ORDER BY k DESC LIMIT 1", &[1]),
        ("SELECT id FROM x ORDER BY v LIMIT 1", &[2]),
        ("SELECT id FROM x ORDER BY v DESC LIMIT 1", &[1]),
    ];
    let wrong: Vec<String> = cases
        .iter()
        .filter_map(|&(sql, want)| {
            let got = ids(&db, sql);
            (got != want).then(|| format!("{sql}: got {got:?}, want {want:?}"))
        })
        .collect();
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_hot_top_k_walks_a_prefix_index_when_the_where_bounds_a_nullable_column() {
    let db = Database::open("memory://sealed_top_k_nulls_prefix").unwrap();
    db.execute(
        "CREATE TABLE p (id INTEGER PRIMARY KEY, g INTEGER, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX p_gv ON p (g, v)", ()).unwrap();
    db.execute(
        "INSERT INTO p VALUES (1, 1, 30), (2, 1, NULL), (3, 1, 10), (4, 2, 5), (5, 1, 20)",
        (),
    )
    .unwrap();
    let sql = "SELECT id, v FROM p WHERE g = 1 AND v > 10 ORDER BY v LIMIT 2";
    assert_eq!(
        read(&db, sql),
        (vec!["20".to_string(), "30".to_string()], vec![1, 5])
    );
    assert_ne!(
        taken(&db, sql),
        Some(false),
        "{sql}: the hot index walk answers"
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_sealed_top_k_builds_few_text_or_float_rows_the_heap_cannot_take() {
    use stoolap::test_failpoints::top_k_cold_rows;
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?checkpoint_on_close=off",
        dir.path().display()
    ))
    .unwrap();
    db.execute(
        "CREATE TABLE r (id INTEGER PRIMARY KEY, t TEXT, f FLOAT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO r SELECT value, 'k' || ((value * 7919) % 20000), \
         ((value * 7919) % 20000) * 0.5 FROM generate_series(1, 20000)",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let cases: [(&str, [&str; 10]); 3] = [
        (
            "SELECT id, t FROM r ORDER BY t LIMIT 10",
            [
                "k0", "k1", "k10", "k100", "k1000", "k10000", "k10001", "k10002", "k10003",
                "k10004",
            ],
        ),
        (
            "SELECT id, t FROM r ORDER BY t DESC LIMIT 10",
            [
                "k9999", "k9998", "k9997", "k9996", "k9995", "k9994", "k9993", "k9992", "k9991",
                "k9990",
            ],
        ),
        (
            "SELECT id, f FROM r ORDER BY f LIMIT 10",
            ["0", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5"],
        ),
    ];
    for (sql, want) in cases {
        let before = top_k_cold_rows();
        let (keys, _) = read(&db, sql);
        let built = top_k_cold_rows() - before;
        assert_eq!(keys, want, "{sql}");
        // Of 20,000 scrambled sealed rows, the heap takes a few hundred
        assert!(built <= 1_000, "{sql}: built {built} sealed rows");
    }
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_sealed_top_k_decodes_the_other_columns_of_its_kept_rows_alone() {
    use stoolap::test_failpoints::take_group_column_decodes;
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/kept", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE w (id INTEGER PRIMARY KEY, t TEXT, p TEXT, f INTEGER)",
        (),
    )
    .unwrap();
    // The first three row groups lead on the key and hold no row the WHERE
    // passes; the last holds the answer
    db.execute(
        "INSERT INTO w SELECT value, \
         CASE WHEN value <= 196608 THEN 'a' ELSE 'b' END || value, 'payload ' || value, \
         CASE WHEN value <= 196608 THEN (value % 2) * 2 ELSE 1 END \
         FROM generate_series(1, 200000)",
        (),
    )
    .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    drop(db);
    // Reopened, no block of the volume is decoded yet
    let db = Database::open(&dsn).unwrap();
    take_group_column_decodes();
    let rows: Vec<(i64, String, String)> = db
        .query(
            "SELECT id, t, p FROM w WHERE f > 0 AND f < 2 ORDER BY t LIMIT 1",
            (),
        )
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (
                row.get(0).unwrap(),
                row.get(1).unwrap(),
                row.get(2).unwrap(),
            )
        })
        .collect();
    assert_eq!(
        rows,
        [(196609, "b196609".to_string(), "payload 196609".to_string())]
    );
    let decodes = take_group_column_decodes();
    let groups = |col: usize| {
        let mut groups: Vec<usize> = decodes.iter().filter(|d| d.0 == col).map(|d| d.1).collect();
        groups.sort_unstable();
        groups.dedup();
        groups
    };
    // Every group's key, the payload of the one group holding the row kept
    assert_eq!(groups(1), [0, 1, 2, 3], "key blocks decoded: {decodes:?}");
    assert_eq!(groups(2), [3], "payload blocks decoded: {decodes:?}");
}
