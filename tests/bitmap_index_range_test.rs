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

//! A bitmap index answers equalities only: a range on its column, alone,
//! as BETWEEN or beside another column's comparison, answers as the scan
//! of a twin table without indexes

use stoolap::Database;

/// `t` with a bitmap index on c, `u` the same rows without
fn open(name: &str, ty: &str, value: impl Fn(i64) -> String) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for table in ["t", "u"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, a INTEGER, c {ty})"),
            (),
        )
        .unwrap();
    }
    db.execute("CREATE INDEX t_c ON t(c) USING BITMAP", ())
        .unwrap();
    for table in ["t", "u"] {
        let rows: Vec<String> = (0..40)
            .map(|id| format!("({id}, {}, {})", id % 4, value(id)))
            .collect();
        db.execute(
            &format!("INSERT INTO {table} VALUES {}", rows.join(", ")),
            (),
        )
        .unwrap();
    }
    db
}

fn ids(db: &Database, sql: &str) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    out.sort_unstable();
    out
}

fn check(db: &Database, low: &str, high: &str) {
    let predicates = [
        format!("c BETWEEN {low} AND {high}"),
        format!("c NOT BETWEEN {low} AND {high}"),
        format!("c >= {low} AND c <= {high}"),
        format!("a = 1 AND c < {high}"),
        format!("a = 1 AND c <= {high}"),
        format!("a = 1 AND c > {low}"),
        format!("a = 1 AND c >= {low}"),
        format!("c < {high} AND a = 1"),
        format!("a = 1 AND c BETWEEN {low} AND {high}"),
        format!("a = 1 AND c >= {low} AND c < {high}"),
        format!("c < {high}"),
        format!("c = {low}"),
    ];
    for predicate in predicates {
        let want = ids(db, &format!("SELECT id FROM u WHERE {predicate}"));
        assert!(!want.is_empty() || predicate.contains("NOT"), "{predicate}");
        let got = ids(db, &format!("SELECT id FROM t WHERE {predicate}"));
        assert_eq!(got, want, "{predicate}");
    }
}

#[test]
fn bitmap_integer_ranges_answer_as_the_scan() {
    let db = open("bitmap_integer_ranges", "INTEGER", |id| {
        format!("{}", id % 10 - 5)
    });
    check(&db, "-3", "2");
}

#[test]
fn bitmap_float_ranges_answer_as_the_scan() {
    let db = open("bitmap_float_ranges", "FLOAT", |id| {
        format!("{}", (id % 10) as f64 * 0.5)
    });
    check(&db, "1.0", "3.5");
}

#[test]
fn bitmap_text_ranges_answer_as_the_scan() {
    let db = open("bitmap_text_ranges", "TEXT", |id| format!("'k{}'", id % 10));
    check(&db, "'k2'", "'k7'");
}

#[test]
fn bitmap_timestamp_ranges_answer_as_the_scan() {
    let db = open("bitmap_timestamp_ranges", "TIMESTAMP", |id| {
        format!("'2026-01-{:02} 00:00:00'", id % 10 + 1)
    });
    check(&db, "'2026-01-03 00:00:00'", "'2026-01-08 00:00:00'");
}
