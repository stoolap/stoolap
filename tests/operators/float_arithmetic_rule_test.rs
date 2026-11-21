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

//! `+`, `-`, `*` and `/` on finite operands, either of them a FLOAT, whose
//! result is not finite fail the statement with `Float overflow`, wherever
//! the expression stands; an operand already infinite is not checked

use stoolap::Database;

fn setup(name: &str, dir: &tempfile::TempDir, sealed: bool) -> Database {
    let dsn = match sealed {
        true => format!("file://{}/db", dir.path().display()),
        false => format!("memory://{name}"),
    };
    let db = Database::open(&dsn).unwrap();
    db.execute(
        "CREATE TABLE o (id INTEGER PRIMARY KEY, g INTEGER, d FLOAT, i INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO o VALUES (1, 1, 1e308, 9223372036854775807), (2, 1, 1.0, 2), (3, 2, 2.5, 3)",
        (),
    )
    .unwrap();
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn rows(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    db.query(sql, ())
        .map_err(|e| e.to_string())?
        .map(|row| {
            let row = row.map_err(|e| e.to_string())?;
            Ok((0..row.len())
                .map(|i| row.get_value(i).unwrap().to_string())
                .collect::<Vec<_>>()
                .join("|"))
        })
        .collect()
}

fn overflow(x: &str, y: &str) -> String {
    format!("type error: Float overflow in arithmetic operation: {x} and {y}")
}

#[test]
fn an_overflowing_float_operation_fails_wherever_it_stands() {
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = setup("float_arithmetic_rule", &dir, sealed);
        for (sql, x, y) in [
            ("SELECT id, d * 10 FROM o", "1e308", "10"),
            ("SELECT id, d + d FROM o", "1e308", "1e308"),
            ("SELECT id, -d - d FROM o", "-1e308", "1e308"),
            ("SELECT id, d / 1e-10 FROM o", "1e308", "1e-10"),
            (
                "SELECT id, i * 1e300 FROM o",
                "9.223372036854776e18",
                "1e300",
            ),
            ("SELECT 1e308 * 10", "1e308", "10"),
            ("SELECT id FROM o WHERE d * 10 > 0", "1e308", "10"),
            (
                "SELECT id, CASE WHEN d * 10 > 0 THEN 1 END FROM o",
                "1e308",
                "10",
            ),
            ("SELECT id FROM o ORDER BY d * 10", "1e308", "10"),
            ("SELECT g, SUM(d * 2) FROM o GROUP BY g", "1e308", "2"),
            (
                "SELECT g FROM o GROUP BY g HAVING MAX(d) * 10 > 0",
                "1e308",
                "10",
            ),
            (
                "SELECT id, SUM(d * 2) OVER (ORDER BY id) FROM o",
                "1e308",
                "2",
            ),
        ] {
            let err = rows(&db, sql).unwrap_err();
            assert!(
                err.ends_with(&overflow(x, y)),
                "sealed {sealed}: {sql}: {err}"
            );
        }

        let err = db.execute("UPDATE o SET d = d * 10", ()).unwrap_err();
        assert_eq!(err.to_string(), overflow("1e308", "10"), "sealed {sealed}");
        let err = db
            .execute(
                "INSERT INTO o SELECT 4, g, d * 10, i FROM o WHERE id = 1",
                (),
            )
            .unwrap_err();
        assert_eq!(err.to_string(), overflow("1e308", "10"), "sealed {sealed}");
        let statement = db.prepare("SELECT id FROM o WHERE d * $1 > 0").unwrap();
        let err = statement
            .query((10,))
            .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
            .unwrap_err();
        assert_eq!(err.to_string(), overflow("1e308", "10"), "sealed {sealed}");

        assert_eq!(
            rows(
                &db,
                "SELECT id, d * 10, d / 0, d % 0, d - 1 FROM o WHERE id > 1 ORDER BY id"
            ),
            Ok(vec![
                "2|10|NULL|NULL|0".to_string(),
                "3|25|NULL|NULL|1.5".to_string(),
            ]),
            "sealed {sealed}"
        );
        assert_eq!(
            rows(&db, "SELECT id, d FROM o ORDER BY id"),
            Ok(vec![
                "1|1e308".to_string(),
                "2|1".to_string(),
                "3|2.5".to_string(),
            ]),
            "sealed {sealed}: the failed UPDATE and INSERT left the rows alone"
        );
    }
}

#[test]
fn an_infinite_operand_is_not_an_overflow() {
    let db = Database::open("memory://float_arithmetic_infinite_operand").unwrap();
    assert_eq!(
        rows(
            &db,
            "SELECT CAST('inf' AS FLOAT) * 2, CAST('-inf' AS FLOAT) + 1"
        ),
        Ok(vec!["Infinity|-Infinity".to_string()])
    );
}
