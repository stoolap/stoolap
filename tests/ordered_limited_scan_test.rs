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

//! ORDER BY one column + LIMIT over a segmented table answers the same rows
//! as a full sort, whatever the layout: volumes whose time ranges overlap
//! (a backfill), hot rows older and newer than the sealed ones, rows deleted
//! before and after a checkpoint, ASC and DESC, OFFSET, and a LIMIT past the
//! end.

use stoolap::Database;

struct Lcg(u64, std::collections::HashSet<i64>);

impl Lcg {
    fn next(&mut self, bound: i64) -> i64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % bound as u64) as i64
    }
}

fn ts(secs: i64) -> String {
    chrono::DateTime::from_timestamp(secs, 0)
        .unwrap()
        .format("%Y-%m-%d %H:%M:%S")
        .to_string()
}

/// Timestamps are unique across the table so the full sort's order is total
/// and the comparison does not depend on how ties come out
fn insert_batch(db: &Database, rng: &mut Lcg, count: usize, base: i64, span: i64, keys: i64) {
    let mut stmt = String::from("INSERT INTO c (t, k, v) VALUES ");
    for i in 0..count {
        if i > 0 {
            stmt.push(',');
        }
        let t = loop {
            let t = base + rng.next(span);
            if rng.1.insert(t) {
                break t;
            }
        };
        let k = rng.next(keys);
        stmt.push_str(&format!("('{}', 'k{k}', {})", ts(t), rng.next(100)));
    }
    db.execute(&stmt, ()).unwrap();
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    db.query(sql, ())
        .unwrap()
        .map(|r| r.unwrap().get::<i64>(0).unwrap())
        .collect()
}

/// The reference is the full sort with the same tie-break the top-k uses
fn check(db: &Database, where_clause: &str, desc: bool, limit: usize, offset: usize) {
    let dir = if desc { "DESC" } else { "ASC" };
    let reference = ids(
        db,
        &format!("SELECT id FROM c {where_clause} ORDER BY t {dir}, id {dir}"),
    );
    let expected: Vec<i64> = reference.iter().skip(offset).take(limit).copied().collect();
    let offset_clause = if offset > 0 {
        format!(" OFFSET {offset}")
    } else {
        String::new()
    };
    let actual = ids(
        db,
        &format!("SELECT id FROM c {where_clause} ORDER BY t {dir} LIMIT {limit}{offset_clause}"),
    );
    assert_eq!(
        actual, expected,
        "{where_clause} ORDER BY t {dir} LIMIT {limit}{offset_clause}"
    );
}

fn check_all(db: &Database) {
    for where_clause in [
        "",
        "WHERE k = 'k7'",
        "WHERE k = 'k7' AND t >= '2024-03-01 00:00:00'",
        "WHERE v > 90",
        "WHERE k = 'nope'",
    ] {
        for desc in [true, false] {
            check(db, where_clause, desc, 1, 0);
            check(db, where_clause, desc, 50, 0);
            check(db, where_clause, desc, 50, 30);
            check(db, where_clause, desc, 100_000, 0);
        }
    }
}

fn setup(dsn: &str) -> (Database, Lcg) {
    let db = Database::open(dsn).unwrap();
    db.execute(
        "CREATE TABLE c (id INTEGER PRIMARY KEY AUTO_INCREMENT, t TIMESTAMP NOT NULL, \
         k TEXT NOT NULL, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX idx_c_t ON c(t) USING BTREE", ())
        .unwrap();
    (db, Lcg(7, std::collections::HashSet::new()))
}

#[test]
fn test_top_k_matches_the_full_sort_across_layouts() {
    let dir = tempfile::tempdir().unwrap();
    let (db, mut rng) = setup(&format!("file://{}/topk", dir.path().display()));
    let day = 86400;
    let base = 1_709_251_200; // 2024-03-01

    // Volume 1: March, then delete some rows before sealing
    insert_batch(&db, &mut rng, 20_000, base, 31 * day, 20);
    db.execute("DELETE FROM c WHERE v = 13", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    check_all(&db);

    // Volume 2: a backfill of February, overlapping nothing but older than volume 1
    insert_batch(&db, &mut rng, 20_000, base - 29 * day, 29 * day, 20);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    check_all(&db);

    // Volume 3: April with a slice of March, overlapping volume 1
    insert_batch(&db, &mut rng, 20_000, base + 20 * day, 41 * day, 20);
    db.execute("DELETE FROM c WHERE v = 42", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("DELETE FROM c WHERE v = 77", ()).unwrap();
    check_all(&db);

    // Hot rows: some newer than everything, some older than everything
    insert_batch(&db, &mut rng, 2_000, base + 70 * day, 5 * day, 20);
    insert_batch(&db, &mut rng, 2_000, base - 60 * day, 5 * day, 20);
    db.execute("DELETE FROM c WHERE v = 5", ()).unwrap();
    check_all(&db);

    // Inside a transaction the same rows come back, and a sealed row this
    // transaction deleted or updated away from the filter is not among them
    db.execute("BEGIN", ()).unwrap();
    insert_batch(&db, &mut rng, 500, base + 80 * day, day, 20);
    // The two newest sealed rows of k7 (volume 3 ends before 2024-04-25)
    let sealed = "k = 'k7' AND t < '2024-04-25 00:00:00'";
    let newest_k7 = ids(
        &db,
        &format!("SELECT id FROM c WHERE {sealed} ORDER BY t DESC LIMIT 2"),
    );
    db.execute(&format!("DELETE FROM c WHERE id = {}", newest_k7[0]), ())
        .unwrap();
    db.execute(
        &format!("UPDATE c SET k = 'moved' WHERE id = {}", newest_k7[1]),
        (),
    )
    .unwrap();
    let after = ids(
        &db,
        &format!("SELECT id FROM c WHERE {sealed} ORDER BY t DESC LIMIT 5"),
    );
    assert!(
        !after.contains(&newest_k7[0]) && !after.contains(&newest_k7[1]),
        "{after:?} still holds {newest_k7:?}"
    );
    check_all(&db);
    db.execute("ROLLBACK", ()).unwrap();
    check_all(&db);
}

#[test]
fn test_top_k_over_a_volume_written_out_of_time_order() {
    // Three row groups of 65,536 rows whose time bounds run high, low, medium
    // in physical order: the newest rows sit in the first group
    let dir = tempfile::tempdir().unwrap();
    let (db, mut rng) = setup(&format!("file://{}/unsorted", dir.path().display()));
    let base = 1_709_251_200;
    let day = 86400;
    for start in [base + 300 * day, base, base + 100 * day] {
        for _ in 0..4 {
            insert_batch(&db, &mut rng, 16_384, start, 30 * day, 20);
        }
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let newest = ids(&db, "SELECT id FROM c ORDER BY t DESC LIMIT 1");
    let reference = ids(&db, "SELECT id FROM c ORDER BY t DESC, id DESC");
    assert_eq!(newest, reference[..1]);
    check_all(&db);
}

#[test]
fn test_top_k_over_a_float_column_keeps_nan_in_order() {
    // Zone maps leave NaN out, so a float column is never answered from the bounds
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}/nan", dir.path().display())).unwrap();
    db.execute(
        "CREATE TABLE f (id INTEGER PRIMARY KEY, x FLOAT NOT NULL)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO f VALUES (1, $1), (2, 100.0)", (f64::NAN,))
        .unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.execute("INSERT INTO f VALUES (3, 200.0)", ()).unwrap();
    for dir in ["DESC", "ASC"] {
        assert_eq!(
            ids(&db, &format!("SELECT id FROM f ORDER BY x {dir} LIMIT 1")),
            ids(&db, &format!("SELECT id FROM f ORDER BY x {dir}, id {dir}"))[..1]
        );
    }
}

#[test]
fn test_top_k_keeps_the_select_list_alias_and_qualified_star() {
    let dir = tempfile::tempdir().unwrap();
    let (db, mut rng) = setup(&format!("file://{}/alias", dir.path().display()));
    insert_batch(&db, &mut rng, 20_000, 1_709_251_200, 30 * 86400, 20);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();

    // ORDER BY names the alias of -v, not the column v
    let min_v = ids(&db, "SELECT MIN(v) FROM c WHERE v > 0")[0];
    assert_eq!(
        ids(
            &db,
            "SELECT -v AS v FROM c WHERE v > 0 ORDER BY v DESC LIMIT 1"
        ),
        [-min_v]
    );

    // A qualified star keeps every column
    let expected = ids(
        &db,
        "SELECT id FROM c WHERE k = 'k7' ORDER BY t DESC, id DESC",
    );
    let row = db
        .query(
            "SELECT c.* FROM c WHERE k = 'k7' ORDER BY t DESC LIMIT 1",
            (),
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(row.len(), 4);
    assert_eq!(row.get::<i64>(0).unwrap(), expected[0]);
}

#[test]
fn test_top_k_on_the_memory_engine_and_after_reopen() {
    let (db, mut rng) = setup("memory://ordered_limited_scan_memory");
    insert_batch(&db, &mut rng, 5_000, 1_709_251_200, 30 * 86400, 10);
    check_all(&db);

    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}/reopen", dir.path().display());
    let (db, mut rng) = setup(&dsn);
    insert_batch(&db, &mut rng, 20_000, 1_709_251_200, 30 * 86400, 20);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    insert_batch(
        &db,
        &mut rng,
        20_000,
        1_709_251_200 + 15 * 86400,
        30 * 86400,
        20,
    );
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    drop(db);
    let db = Database::open(&dsn).unwrap();
    check_all(&db);
}
