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

//! A column only a window reads, in its call, FILTER, partition, order or
//! frame, is decoded from the volumes like any other: a sealed table answers
//! as the same rows hot do

use stoolap::Database;

const STATEMENTS: &[&str] = &[
    "SELECT id, SUM(a) OVER (PARTITION BY p ORDER BY id) FROM t",
    "SELECT id, ROW_NUMBER() OVER (ORDER BY o DESC, id) FROM t",
    "SELECT id, COUNT(*) FILTER (WHERE f > 2) OVER (ORDER BY id) FROM t",
    "SELECT id, SUM(id) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t",
    "SELECT p, SUM(a), RANK() OVER (ORDER BY SUM(o) DESC) FROM t GROUP BY p",
    "SELECT p, COUNT(*), SUM(SUM(f)) OVER (ORDER BY p) FROM t WHERE id > 1 GROUP BY p",
];

fn fill(db: &Database) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, p INTEGER, o INTEGER, f INTEGER, unused TEXT)",
        "INSERT INTO t SELECT value, value * 3, value % 3, (value * 7) % 11, value % 5, 'x' \
         FROM generate_series(1, 40)",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

fn rows(db: &Database, sql: &str) -> Vec<String> {
    let mut rows: Vec<String> = db
        .query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| row.get_value(i).unwrap().to_string())
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect();
    rows.sort();
    rows
}

#[test]
fn a_column_only_a_window_reads_is_decoded_from_the_volumes() {
    let hot = Database::open("memory://sealed_window_columns").unwrap();
    fill(&hot);
    let dir = tempfile::tempdir().unwrap();
    let sealed = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    fill(&sealed);
    sealed.execute("PRAGMA CHECKPOINT", ()).unwrap();
    for sql in STATEMENTS {
        assert_eq!(rows(&sealed, sql), rows(&hot, sql), "{sql}");
    }
}
