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

//! The column a USING or NATURAL join merges reads the left side's column of
//! an inner or left join and the right side's of a right join, wherever a
//! statement reads it

use stoolap::Database;

fn fill(db: &Database) {
    for sql in [
        "CREATE TABLE t (k INTEGER, a INTEGER)",
        "INSERT INTO t VALUES (1, 10), (2, 20)",
        "CREATE TABLE u (k INTEGER, b INTEGER)",
        "INSERT INTO u VALUES (2, 200), (3, 300)",
        "CREATE TABLE w (k INTEGER, c INTEGER)",
        "INSERT INTO w VALUES (3, 3000), (4, 4000)",
        "CREATE TABLE y (k INTEGER)",
        "INSERT INTO y VALUES (3)",
        "CREATE TABLE p1 (k INTEGER)",
        "INSERT INTO p1 VALUES (1)",
        "CREATE TABLE p2 (k INTEGER)",
        "INSERT INTO p2 VALUES (2)",
        "CREATE TABLE p3 (k INTEGER)",
        "INSERT INTO p3 VALUES (3)",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

/// Each row as its columns' values, NULL as None
fn rows(db: &Database, sql: &str, prepared: bool) -> Vec<Vec<Option<i64>>> {
    let result = match prepared {
        true => db.prepare(sql).unwrap().query((0,)),
        false => db.query(sql, (0,)),
    }
    .unwrap_or_else(|e| panic!("{sql}: {e}"));
    let width = result.columns().len();
    result
        .map(|row| {
            let row = row.unwrap();
            (0..width)
                .map(|i| row.get::<Option<i64>>(i).unwrap())
                .collect()
        })
        .collect()
}

fn ints(values: &[&[Option<i64>]]) -> Vec<Vec<Option<i64>>> {
    values.iter().map(|row| row.to_vec()).collect()
}

const N: Option<i64> = None;

fn s(v: i64) -> Option<i64> {
    Some(v)
}

#[test]
fn a_merged_column_reads_the_side_its_join_kind_keeps() {
    let dir = tempfile::tempdir().unwrap();
    let sealed = Database::open(&format!(
        "file://{}?checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    fill(&sealed);
    sealed.execute("PRAGMA CHECKPOINT", ()).unwrap();
    let hot = Database::open("memory://merged_using_kind").unwrap();
    fill(&hot);
    let cases: Vec<(&str, Vec<Vec<Option<i64>>>)> = vec![
        // The kinds that keep the left side read its column
        (
            "SELECT k FROM t JOIN u USING (k) WHERE k > $1 ORDER BY k",
            ints(&[&[s(2)]]),
        ),
        (
            "SELECT k FROM t LEFT JOIN u USING (k) WHERE k > $1 ORDER BY k",
            ints(&[&[s(1)], &[s(2)]]),
        ),
        // A right join reads the right side's
        (
            "SELECT k FROM t RIGHT JOIN u USING (k) WHERE k > $1 ORDER BY k",
            ints(&[&[s(2)], &[s(3)]]),
        ),
        (
            "SELECT k FROM t NATURAL RIGHT JOIN u WHERE k > $1 ORDER BY k",
            ints(&[&[s(2)], &[s(3)]]),
        ),
        (
            "SELECT * FROM t RIGHT JOIN u USING (k) WHERE b > $1 ORDER BY b",
            ints(&[&[s(2), s(20), s(200)], &[s(3), N, s(300)]]),
        ),
        (
            "SELECT t.k, u.k FROM t RIGHT JOIN u USING (k) WHERE b > $1 ORDER BY b",
            ints(&[&[s(2), s(2)], &[N, s(3)]]),
        ),
        (
            "SELECT COUNT(*) FROM t RIGHT JOIN u USING (k) WHERE k = 3 + $1",
            ints(&[&[s(1)]]),
        ),
        (
            "SELECT k, COUNT(*) FROM t RIGHT JOIN u USING (k) WHERE b > $1 GROUP BY k ORDER BY k",
            ints(&[&[s(2), s(1)], &[s(3), s(1)]]),
        ),
        (
            "SELECT k, (SELECT COUNT(*) FROM w) FROM t RIGHT JOIN u USING (k) \
             WHERE b > $1 ORDER BY k",
            ints(&[&[s(2), s(2)], &[s(3), s(2)]]),
        ),
        (
            "SELECT k FROM t RIGHT JOIN u USING (k) \
             WHERE EXISTS (SELECT 1 FROM (SELECT c FROM w) ww WHERE ww.c = k * 1000 + $1) \
             ORDER BY k",
            ints(&[&[s(3)]]),
        ),
        (
            "SELECT k FROM (SELECT * FROM t) x RIGHT JOIN (SELECT * FROM u) y USING (k) \
             WHERE b > $1 ORDER BY k",
            ints(&[&[s(2)], &[s(3)]]),
        ),
        (
            "SELECT COUNT(_rowid_) FROM (SELECT _rowid_ FROM t WHERE k = 1 + $1) a \
             RIGHT JOIN (SELECT _rowid_ FROM t) b USING (_rowid_)",
            ints(&[&[s(2)]]),
        ),
        // A join over a right join's merged column keys on the right side's
        (
            "SELECT k, b, c FROM t RIGHT JOIN u USING (k) JOIN w USING (k) WHERE c > $1",
            ints(&[&[s(3), s(300), s(3000)]]),
        ),
        (
            "SELECT k, b, c FROM t RIGHT JOIN u USING (k) RIGHT JOIN w USING (k) \
             WHERE c > $1 ORDER BY k",
            ints(&[&[s(3), s(300), s(3000)], &[s(4), N, s(4000)]]),
        ),
        // A star over a chain of right joins beside another source
        (
            "SELECT * FROM t RIGHT JOIN u USING (k) RIGHT JOIN w USING (k) \
             CROSS JOIN (SELECT 9 AS z) x WHERE c > $1 ORDER BY c",
            ints(&[
                &[s(3), N, s(300), s(3000), s(9)],
                &[s(4), N, N, s(4000), s(9)],
            ]),
        ),
        (
            "SELECT *, ROW_NUMBER() OVER (ORDER BY c) FROM t RIGHT JOIN u USING (k) \
             RIGHT JOIN w USING (k) CROSS JOIN (SELECT 9 AS z) x WHERE c > $1 ORDER BY c",
            ints(&[
                &[s(3), N, s(300), s(3000), s(9), s(1)],
                &[s(4), N, N, s(4000), s(9), s(2)],
            ]),
        ),
        (
            "SELECT *, ROW_NUMBER() OVER (ORDER BY c) FROM t RIGHT JOIN u USING (k) \
             RIGHT JOIN w USING (k) JOIN y USING (k) WHERE c > $1",
            ints(&[&[s(3), N, s(300), s(3000), s(1)]]),
        ),
        (
            "SELECT *, ROW_NUMBER() OVER () FROM p1 RIGHT JOIN p2 USING (k) \
             RIGHT JOIN p3 USING (k) JOIN y USING (k) WHERE k > $1",
            ints(&[&[s(3), s(1)]]),
        ),
        (
            "SELECT k, a, b FROM w RIGHT JOIN (SELECT * FROM t RIGHT JOIN u USING (k)) x \
             USING (k) WHERE b > $1 ORDER BY k",
            ints(&[&[s(2), s(20), s(200)], &[s(3), N, s(300)]]),
        ),
    ];
    let mut wrong = Vec::new();
    for (context, db) in [("hot", &hot), ("sealed", &sealed)] {
        for (sql, want) in &cases {
            for prepared in [false, true] {
                let got = rows(db, sql, prepared);
                if got != *want {
                    wrong.push(format!(
                        "{context}, prepared {prepared}: {sql}\n  got  {got:?}\n  want {want:?}"
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
