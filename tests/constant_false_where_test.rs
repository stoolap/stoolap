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

//! A WHERE false for every row answers as a WHERE that filters every row
//! out: the same columns, values and types, in every clause around it

use stoolap::Database;

fn answer(db: &Database, sql: &str) -> Result<(Vec<String>, Vec<String>), String> {
    let result = db.query(sql, ()).map_err(|e| e.to_string())?;
    let columns = result.columns().to_vec();
    let rows = result
        .map(|row| row.map(|row| format!("{:?}", row.as_row().as_slice())))
        .collect::<stoolap::Result<Vec<_>>>()
        .map_err(|e| e.to_string())?;
    Ok((columns, rows))
}

#[test]
fn a_constant_false_where_answers_as_an_empty_filter() {
    let db = Database::open("memory://constant_false_where").unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, v FLOAT, s TEXT)",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX t_g ON t (g)", ()).unwrap();
    db.execute(
        "INSERT INTO t SELECT value, value % 3, value * 1.5, 'x' FROM generate_series(1, 50)",
        (),
    )
    .unwrap();
    let mut wrong = Vec::new();
    for select in [
        "SELECT * FROM t",
        "SELECT t.* FROM t",
        "SELECT id, s FROM t",
        "SELECT id + 1, UPPER(s) AS u FROM t",
        "SELECT x.id FROM t x",
        "SELECT COUNT(*) FROM t",
        "SELECT COUNT(*), SUM(v), MIN(id), MAX(s), AVG(v) FROM t",
        "SELECT SUM(v) * 2 FROM t",
        "SELECT g, COUNT(*) FROM t GROUP BY g",
        "SELECT g FROM t GROUP BY g",
        "SELECT g + 1 FROM t GROUP BY g ORDER BY g",
        "SELECT g, SUM(v) FROM t GROUP BY g HAVING SUM(v) > 0",
        "SELECT COUNT(*) FROM t HAVING COUNT(*) = 0",
        "SELECT DISTINCT g FROM t",
        "SELECT id FROM t ORDER BY v DESC LIMIT 3",
        "SELECT id FROM t ORDER BY id LIMIT 5 OFFSET 2",
        "SELECT id, ROW_NUMBER() OVER (ORDER BY id) FROM t",
        "SELECT id, SUM(v) OVER (PARTITION BY g) FROM t",
        "SELECT COUNT(DISTINCT g) FROM t",
        "SELECT STRING_AGG(s, ',') FROM t",
        "SELECT id FROM t WHERE g = 1 AND {never}",
        "SELECT id FROM t WHERE id = 7 AND {never}",
        "SELECT id FROM t WHERE id IN (1, 2) AND {never}",
        "SELECT MIN(id) FROM t WHERE {never}",
    ] {
        let at = |never: &str| {
            if select.contains("{never}") {
                select.replace("{never}", never)
            } else if let Some(pos) = select.find(" GROUP BY") {
                format!("{} WHERE {never}{}", &select[..pos], &select[pos..])
            } else if let Some(pos) = select.find(" HAVING") {
                format!("{} WHERE {never}{}", &select[..pos], &select[pos..])
            } else if let Some(pos) = select.find(" ORDER BY") {
                format!("{} WHERE {never}{}", &select[..pos], &select[pos..])
            } else {
                format!("{select} WHERE {never}")
            }
        };
        for never in ["1 = 0", "FALSE", "NULL", "1 = 0 AND id > 0"] {
            let sql = at(never);
            let (got, want) = (answer(&db, &sql), answer(&db, &at("ABS(id) < 0")));
            if got != want {
                wrong.push(format!("{sql}\n  got  {got:?}\n  want {want:?}"));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn a_scan_its_zone_maps_prune_answers_as_an_empty_filter() {
    let db = Database::open("memory://constant_false_zone_maps").unwrap();
    db.execute("CREATE TABLE z (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO z SELECT value, value * 10 FROM generate_series(1, 100)",
        (),
    )
    .unwrap();
    db.query("ANALYZE z", ()).unwrap().count();
    let mut wrong = Vec::new();
    for select in [
        "SELECT * FROM z WHERE {w}",
        "SELECT COUNT(*) FROM z WHERE {w}",
        "SELECT SUM(v), MAX(id) FROM z WHERE {w}",
        "SELECT v, COUNT(*) FROM z WHERE {w} GROUP BY v",
        "SELECT v FROM z WHERE {w} GROUP BY v",
        "SELECT id, ROW_NUMBER() OVER (ORDER BY id) FROM z WHERE {w}",
        "SELECT COUNT(*), ROW_NUMBER() OVER () FROM z WHERE {w}",
    ] {
        let sql = select.replace("{w}", "v > 5000");
        let want = answer(&db, &select.replace("{w}", "v + 0 > 5000"));
        let got = answer(&db, &sql);
        if got != want {
            wrong.push(format!("{sql}\n  got  {got:?}\n  want {want:?}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn a_where_that_keeps_no_row_reads_no_row() {
    use stoolap::test_failpoints::random_draws;
    let db = Database::open("memory://constant_false_reads").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO t SELECT value, value % 3 FROM generate_series(1, 50)",
        (),
    )
    .unwrap();
    let mut wrong = Vec::new();
    for sql in [
        "SELECT * FROM t WHERE RANDOM() < 2 AND NULL",
        "SELECT id FROM t WHERE RANDOM() < 2 AND $1 = 0",
        "SELECT COUNT(*) FROM t WHERE RANDOM() < 2 AND $1 = 0",
        "SELECT g, COUNT(*) FROM t WHERE RANDOM() < 2 AND $1 = 0 GROUP BY g",
        "SELECT id, ROW_NUMBER() OVER (ORDER BY id) FROM t WHERE RANDOM() < 2 AND $1 = 0",
        "SELECT id FROM t WHERE RANDOM() < 2 AND $1 = 0 ORDER BY g LIMIT 3",
        "SELECT DISTINCT g FROM t WHERE RANDOM() < 2 AND $1 = 0",
    ] {
        let before = random_draws();
        let rows = db.query(sql, (1,)).map(|rows| rows.count());
        let draws = random_draws() - before;
        if draws != 0 {
            wrong.push(format!("{sql} drew {draws} values, rows {rows:?}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
