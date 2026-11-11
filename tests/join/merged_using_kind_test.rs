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
        "CREATE TABLE p4 (k INTEGER)",
        "INSERT INTO p4 VALUES (4)",
        "CREATE TABLE ri (id INTEGER PRIMARY KEY)",
        "INSERT INTO ri VALUES (1), (2), (3)",
        "CREATE TABLE tx (k TEXT)",
        "INSERT INTO tx VALUES ('a key longer than inline 1'), ('a key longer than inline 2')",
        "CREATE TABLE ux (k TEXT)",
        "INSERT INTO ux VALUES ('a key longer than inline 2'), ('a key longer than inline 3')",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

/// Each row as its columns' values, NULL as None
fn rows(db: &Database, sql: &str, prepared: bool) -> Result<Vec<Vec<Option<i64>>>, String> {
    let result = match prepared {
        true => db.prepare(sql).and_then(|p| p.query((0,))),
        false => db.query(sql, (0,)),
    }
    .map_err(|e| e.to_string())?;
    let width = result.columns().len();
    result
        .map(|row| {
            let row = row.map_err(|e| e.to_string())?;
            (0..width)
                .map(|i| row.get::<Option<i64>>(i).map_err(|e| e.to_string()))
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
    let cases: Vec<(&str, Vec<Vec<Option<i64>>>)> =
        vec![
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
            "SELECT *, ROW_NUMBER() OVER () FROM p1 RIGHT JOIN p2 USING (k) \
             RIGHT JOIN p3 USING (k) WHERE k > $1",
            ints(&[&[s(3), s(1)]]),
        ),
        (
            "SELECT p3.*, k, ROW_NUMBER() OVER () FROM p1 RIGHT JOIN p2 USING (k) \
             RIGHT JOIN p3 USING (k) RIGHT JOIN p4 USING (k) WHERE k > $1",
            ints(&[&[N, s(4), s(1)]]),
        ),
        (
            "SELECT SUM(k) FROM p1 RIGHT JOIN p2 USING (k) RIGHT JOIN p3 USING (k) \
             RIGHT JOIN p4 USING (k) WHERE k > $1",
            ints(&[&[s(4)]]),
        ),
        (
            "SELECT k, a, b FROM w RIGHT JOIN (SELECT * FROM t RIGHT JOIN u USING (k)) x \
             USING (k) WHERE b > $1 ORDER BY k",
            ints(&[&[s(2), s(20), s(200)], &[s(3), N, s(300)]]),
        ),
        // A full join reads the side that has the row
        (
            "SELECT k FROM t FULL JOIN u USING (k) WHERE k > $1 ORDER BY k",
            ints(&[&[s(1)], &[s(2)], &[s(3)]]),
        ),
        (
            "SELECT * FROM t FULL JOIN u USING (k) WHERE k > $1 ORDER BY k",
            ints(&[&[s(1), s(10), N], &[s(2), s(20), s(200)], &[s(3), N, s(300)]]),
        ),
        (
            "SELECT u.*, k FROM t FULL JOIN u USING (k) WHERE k > $1 ORDER BY k",
            ints(&[&[N, N, s(1)], &[s(2), s(200), s(2)], &[s(3), s(300), s(3)]]),
        ),
        (
            "SELECT t.k, u.k, k FROM t FULL JOIN u USING (k) WHERE k > $1 ORDER BY 3",
            ints(&[&[s(1), N, s(1)], &[s(2), s(2), s(2)], &[N, s(3), s(3)]]),
        ),
        (
            "SELECT COUNT(*) FROM t FULL JOIN u USING (k) WHERE k = 3 + $1",
            ints(&[&[s(1)]]),
        ),
        (
            "SELECT COUNT(*) FROM t FULL JOIN u USING (k) WHERE k IS NULL OR k = $1",
            ints(&[&[s(0)]]),
        ),
        (
            "SELECT k FROM t FULL JOIN u USING (k) WHERE k IN (1, 3) AND k > $1 ORDER BY k",
            ints(&[&[s(1)], &[s(3)]]),
        ),
        (
            "SELECT SUM(k), MAX(k) FROM t FULL JOIN u USING (k) WHERE k > $1",
            ints(&[&[s(6), s(3)]]),
        ),
        (
            "SELECT k, COUNT(*) FROM t FULL JOIN u USING (k) WHERE k > $1 GROUP BY k \
             HAVING k > 1 ORDER BY k",
            ints(&[&[s(2), s(1)], &[s(3), s(1)]]),
        ),
        (
            "SELECT DISTINCT k FROM t FULL JOIN u USING (k) WHERE k > $1 ORDER BY k DESC",
            ints(&[&[s(3)], &[s(2)], &[s(1)]]),
        ),
        (
            "SELECT k, ROW_NUMBER() OVER (PARTITION BY k > 1 ORDER BY k) \
             FROM t FULL JOIN u USING (k) WHERE k > $1 ORDER BY k",
            ints(&[&[s(1), s(1)], &[s(2), s(1)], &[s(3), s(2)]]),
        ),
        (
            "SELECT k, (SELECT ww.c FROM (SELECT k AS kk, c FROM w) ww WHERE ww.kk = k + $1) \
             FROM t FULL JOIN u USING (k) ORDER BY k",
            ints(&[&[s(1), N], &[s(2), N], &[s(3), s(3000)]]),
        ),
        (
            "SELECT k FROM t FULL JOIN u USING (k) \
             WHERE EXISTS (SELECT 1 FROM (SELECT c FROM w) ww WHERE ww.c = k * 1000 + $1) \
             ORDER BY k",
            ints(&[&[s(3)]]),
        ),
        (
            "SELECT k FROM (SELECT * FROM t) x FULL JOIN (SELECT * FROM u) y USING (k) \
             WHERE k > $1 ORDER BY k",
            ints(&[&[s(1)], &[s(2)], &[s(3)]]),
        ),
        (
            "SELECT COUNT(_rowid_) FROM (SELECT _rowid_ FROM t WHERE k = 1 + $1) a \
             FULL JOIN (SELECT _rowid_ FROM t WHERE k = 2) b USING (_rowid_)",
            ints(&[&[s(2)]]),
        ),
        (
            "SELECT COUNT(_rowid_) FROM (SELECT _rowid_ FROM ri WHERE id = 1 + $1) a \
             FULL JOIN (SELECT _rowid_ FROM ri WHERE id = 2) b USING (_rowid_) \
             FULL JOIN (SELECT _rowid_ FROM ri WHERE id = 3) c USING (_rowid_)",
            ints(&[&[s(3)]]),
        ),
        (
            "SELECT COUNT(_rowid_) FROM (SELECT _rowid_ FROM ri WHERE id = 1 + $1) a \
             RIGHT JOIN (SELECT _rowid_ FROM ri WHERE id = 2) b USING (_rowid_) \
             FULL JOIN (SELECT _rowid_ FROM ri WHERE id = 3) c USING (_rowid_)",
            ints(&[&[s(2)]]),
        ),
        (
            "SELECT COUNT(*) FROM tx FULL JOIN ux USING (k) \
             WHERE k = 'a key longer than inline 3' AND $1 = 0",
            ints(&[&[s(1)]]),
        ),
        // A join over a full join's merged column keys on the side that has it
        (
            "SELECT k, a, b, c FROM t FULL JOIN u USING (k) FULL JOIN w USING (k) \
             WHERE k > $1 ORDER BY k",
            ints(&[
                &[s(1), s(10), N, N],
                &[s(2), s(20), s(200), N],
                &[s(3), N, s(300), s(3000)],
                &[s(4), N, N, s(4000)],
            ]),
        ),
        (
            "SELECT * FROM t FULL JOIN u USING (k) FULL JOIN w USING (k) WHERE k > $1 ORDER BY k",
            ints(&[
                &[s(1), s(10), N, N],
                &[s(2), s(20), s(200), N],
                &[s(3), N, s(300), s(3000)],
                &[s(4), N, N, s(4000)],
            ]),
        ),
        (
            "SELECT k, COUNT(*) FROM t FULL JOIN u USING (k) FULL JOIN w USING (k) \
             WHERE k > $1 GROUP BY k ORDER BY k",
            ints(&[&[s(1), s(1)], &[s(2), s(1)], &[s(3), s(1)], &[s(4), s(1)]]),
        ),
        (
            "SELECT k, a, b, c FROM t FULL JOIN u USING (k) LEFT JOIN w USING (k) \
             WHERE $1 = 0 ORDER BY k",
            ints(&[
                &[s(1), s(10), N, N],
                &[s(2), s(20), s(200), N],
                &[s(3), N, s(300), s(3000)],
            ]),
        ),
        (
            "SELECT k, a, b, c FROM t LEFT JOIN u USING (k) FULL JOIN w USING (k) \
             WHERE $1 = 0 ORDER BY k",
            ints(&[
                &[s(1), s(10), N, N],
                &[s(2), s(20), s(200), N],
                &[s(3), N, N, s(3000)],
                &[s(4), N, N, s(4000)],
            ]),
        ),
        (
            "SELECT k, a, b, c FROM t RIGHT JOIN u USING (k) FULL JOIN w USING (k) \
             WHERE $1 = 0 ORDER BY k",
            ints(&[
                &[s(2), s(20), s(200), N],
                &[s(3), N, s(300), s(3000)],
                &[s(4), N, N, s(4000)],
            ]),
        ),
        (
            "SELECT k, a, b, c FROM t FULL JOIN u USING (k) RIGHT JOIN w USING (k) \
             WHERE $1 = 0 ORDER BY k",
            ints(&[&[s(3), N, s(300), s(3000)], &[s(4), N, N, s(4000)]]),
        ),
        (
            "SELECT *, ROW_NUMBER() OVER () FROM t FULL JOIN u USING (k) JOIN y USING (k) \
             WHERE k > $1",
            ints(&[&[s(3), N, s(300), s(1)]]),
        ),
        (
            "SELECT p3.*, k, ROW_NUMBER() OVER (ORDER BY k) FROM p1 FULL JOIN p2 USING (k) \
             FULL JOIN p3 USING (k) FULL JOIN p4 USING (k) WHERE k > 2 + $1 ORDER BY k",
            ints(&[&[s(3), s(3), s(1)], &[N, s(4), s(2)]]),
        ),
        (
            "SELECT SUM(k), COUNT(*) FROM p1 FULL JOIN p2 USING (k) FULL JOIN p3 USING (k) \
             FULL JOIN p4 USING (k) WHERE k > $1",
            ints(&[&[s(10), s(4)]]),
        ),
        (
            "SELECT k, a, b FROM w FULL JOIN (SELECT * FROM t FULL JOIN u USING (k)) x \
             USING (k) WHERE $1 = 0 ORDER BY k",
            ints(&[
                &[s(1), s(10), N],
                &[s(2), s(20), s(200)],
                &[s(3), N, s(300)],
                &[s(4), N, N],
            ]),
        ),
    ];
    let mut wrong = Vec::new();
    for (context, db) in [("hot", &hot), ("sealed", &sealed)] {
        for (sql, want) in &cases {
            for prepared in [false, true] {
                let got = rows(db, sql, prepared);
                if got.as_ref() != Ok(want) {
                    wrong.push(format!(
                        "{context}, prepared {prepared}: {sql}\n  got  {got:?}\n  want {want:?}"
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
