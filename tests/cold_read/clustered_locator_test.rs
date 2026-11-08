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

//! A clustered volume's file holds its positions in row id order, so the
//! first lookup by id after a reopen reads them instead of sorting the ids.

use stoolap::Database;

/// A closed file database whose table `t`, clustered by a key that cycles
/// with the ids, is sealed into one volume; its DSN and volume file
fn clustered(dir: &std::path::Path) -> (String, std::path::PathBuf) {
    let dsn = format!("file://{}", dir.display());
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER NOT NULL, v INTEGER)",
        (),
    )
    .unwrap();
    db.execute("ALTER TABLE t CLUSTER BY (k)", ()).unwrap();
    for start in (1..=70_000i64).step_by(10_000) {
        let values: Vec<String> = (start..start + 10_000)
            .map(|id| format!("({id}, {}, {})", id % 97, id * 3))
            .collect();
        db.execute(&format!("INSERT INTO t VALUES {}", values.join(",")), ())
            .unwrap();
    }
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    db.close().unwrap();
    let volume = std::fs::read_dir(dir.join("volumes").join("t"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|x| x == "vol"))
        .unwrap();
    (dsn, volume)
}

fn check_lookups(db: &Database) {
    for id in [1i64, 777, 40_001, 70_000] {
        let v: i64 = db
            .query_one("SELECT v FROM t WHERE id = $1", (id,))
            .unwrap();
        assert_eq!(v, id * 3, "row {id}");
    }
    let missing = db
        .query("SELECT v FROM t WHERE id = $1", (70_001i64,))
        .unwrap()
        .count();
    assert_eq!(missing, 0);
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_reopened_clustered_volume_is_looked_up_without_sorting_its_ids() {
    use stoolap::test_failpoints::{self as fp, FailpointGuard};

    let _failpoints = FailpointGuard::new();
    let dir = tempfile::tempdir().unwrap();
    let (dsn, _) = clustered(dir.path());
    let db = Database::open(&dsn).unwrap();
    let built = fp::row_orders_built();
    check_lookups(&db);
    assert_eq!(
        fp::row_orders_built() - built,
        0,
        "a lookup sorted the volume's ids"
    );
}

#[test]
fn a_damaged_locator_is_sorted_around_and_reported_by_verify() {
    let dir = tempfile::tempdir().unwrap();
    let (dsn, volume) = clustered(dir.path());
    let mut bytes = std::fs::read(&volume).unwrap();
    // The locator's place in the section table, kind 11
    let u64_at = |b: &[u8], at: usize| u64::from_le_bytes(b[at..at + 8].try_into().unwrap());
    let table = u64_at(&bytes, bytes.len() - 32) as usize;
    let count = u32::from_le_bytes(bytes[table..table + 4].try_into().unwrap()) as usize;
    let locator = (0..count)
        .map(|i| table + 8 + i * 40)
        .find(|&entry| bytes[entry..entry + 4] == 11u32.to_le_bytes())
        .map(|entry| u64_at(&bytes, entry + 8) as usize)
        .expect("a clustered volume's file holds a locator");
    bytes[locator + 1] ^= 0xFF;
    std::fs::write(&volume, &bytes).unwrap();
    let db = Database::open(&dsn).unwrap();
    check_lookups(&db);
    let (status, detail): (String, Option<String>) = db
        .query("PRAGMA VERIFY", ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get(2).unwrap(), r.get(4).unwrap())
        })
        .next()
        .unwrap();
    assert_eq!(status, "damaged");
    let detail = detail.unwrap();
    assert!(detail.contains("section 11 CRC mismatch"), "{detail}");
}
