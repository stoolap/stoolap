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

//! A statement's foreign key actions write once its RETURNING and ON
//! CONFLICT blocks have read, so every block reads the tables as the
//! statement found them

use stoolap::core::Value;
use stoolap::Database;

const TABLES: [&str; 7] = [
    "CREATE TABLE p (id INTEGER PRIMARY KEY, code INTEGER UNIQUE, n INTEGER)",
    "CREATE TABLE cd (id INTEGER PRIMARY KEY, pd INTEGER REFERENCES p(id) ON DELETE CASCADE)",
    "CREATE TABLE cu (id INTEGER PRIMARY KEY, pc INTEGER REFERENCES p(code) ON UPDATE CASCADE ON DELETE SET NULL)",
    "CREATE VIEW vcd AS SELECT * FROM cd",
    "INSERT INTO p VALUES (1, 5, 0), (2, 6, 0), (3, 7, 0)",
    "INSERT INTO cd VALUES (1, 1), (2, 1), (3, 2)",
    "INSERT INTO cu VALUES (1, 7), (2, 7), (3, 6)",
];

/// A statement, its parameters, the rows it returns, and what holds after it
struct Case {
    sql: &'static str,
    params: &'static [i64],
    returned: &'static [&'static [i64]],
    after: &'static [(&'static str, i64)],
}

const CASES: &[Case] = &[
    Case {
        sql: "DELETE FROM p WHERE id IN (1, 2) RETURNING id, \
              (SELECT COUNT(*) FROM cd WHERE cd.pd = p.id), (SELECT COUNT(*) FROM cd)",
        params: &[],
        returned: &[&[1, 2, 3], &[2, 1, 3]],
        after: &[
            ("SELECT COUNT(*) FROM cd", 0),
            ("SELECT COUNT(*) FROM cu WHERE pc IS NULL", 1),
        ],
    },
    Case {
        sql: "DELETE FROM p WHERE id = $1 RETURNING id, \
              (SELECT COUNT(*) FROM cd WHERE cd.pd = p.id)",
        params: &[1],
        returned: &[&[1, 2]],
        after: &[("SELECT COUNT(*) FROM cd", 1)],
    },
    Case {
        sql: "DELETE FROM p WHERE id = 1 RETURNING (SELECT COUNT(*) FROM vcd WHERE vcd.pd = p.id)",
        params: &[],
        returned: &[&[2]],
        after: &[("SELECT COUNT(*) FROM vcd", 1)],
    },
    Case {
        sql: "DELETE FROM p WHERE id = 3 RETURNING (SELECT COUNT(*) FROM cu WHERE cu.pc = p.code)",
        params: &[],
        returned: &[&[2]],
        after: &[("SELECT COUNT(*) FROM cu WHERE pc IS NULL", 2)],
    },
    Case {
        sql: "UPDATE p SET code = $1 WHERE id = $2 RETURNING \
              (SELECT COUNT(*) FROM cu WHERE cu.pc = p.code), (SELECT COUNT(*) FROM cu WHERE pc = 7)",
        params: &[70, 3],
        returned: &[&[0, 2]],
        after: &[("SELECT COUNT(*) FROM cu WHERE pc = 70", 2)],
    },
    Case {
        sql: "INSERT INTO p VALUES (3, 70, 0) ON CONFLICT (id) DO UPDATE SET code = excluded.code \
              RETURNING (SELECT COUNT(*) FROM cu WHERE cu.pc = p.code)",
        params: &[],
        returned: &[&[0]],
        after: &[("SELECT COUNT(*) FROM cu WHERE pc = 70", 2)],
    },
    Case {
        sql: "INSERT INTO p VALUES (3, 70, 0), (2, 6, 0) ON CONFLICT (id) DO UPDATE \
              SET code = excluded.code, n = (SELECT COUNT(*) FROM cu WHERE cu.pc = excluded.code + 64)",
        params: &[],
        returned: &[],
        after: &[
            ("SELECT n FROM p WHERE id = 2", 0),
            ("SELECT COUNT(*) FROM cu WHERE pc = 70", 2),
        ],
    },
    Case {
        sql: "INSERT INTO p VALUES (3, 70, 0), (3, 700, 0) ON CONFLICT (id) DO UPDATE \
              SET code = excluded.code",
        params: &[],
        returned: &[],
        after: &[
            ("SELECT COUNT(*) FROM cu WHERE pc = 700", 2),
            ("SELECT COUNT(*) FROM cu WHERE pc = 70", 0),
        ],
    },
];

fn ints(rows: stoolap::Rows) -> Vec<Vec<i64>> {
    rows.map(|row| {
        let row = row.unwrap();
        (0..row.len()).map(|i| row.get::<i64>(i).unwrap()).collect()
    })
    .collect()
}

fn scalar(db: &Database, sql: &str) -> i64 {
    db.query_one::<i64, _>(sql, ()).unwrap()
}

/// Opens the tables hot, or sealed in a file database
fn tables(dsn: &str, sealed: bool) -> Database {
    let db = Database::open(dsn).unwrap();
    for sql in TABLES {
        db.execute(sql, ()).unwrap();
    }
    if sealed {
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    }
    db
}

fn run(db: &Database, case: &Case, prepared: bool, in_transaction: bool) -> Vec<Vec<i64>> {
    let params: Vec<Value> = case.params.iter().map(|&v| Value::Integer(v)).collect();
    if in_transaction {
        db.execute("BEGIN", ()).unwrap();
    }
    let got = match prepared {
        true => ints(db.prepare(case.sql).unwrap().query(params).unwrap()),
        false => ints(db.query(case.sql, params).unwrap()),
    };
    if in_transaction {
        db.execute("COMMIT", ()).unwrap();
    }
    got
}

#[test]
fn every_block_reads_the_tables_before_the_statement_s_foreign_key_actions() {
    let mut wrong = Vec::new();
    for (at, case) in CASES.iter().enumerate() {
        for sealed in [false, true] {
            for prepared in [false, true] {
                for in_transaction in [false, true] {
                    let dir = tempfile::tempdir().unwrap();
                    let dsn = match sealed {
                        true => format!("file://{}", dir.path().display()),
                        false => format!(
                            "memory://fk_actions_after_blocks_{at}_{prepared}_{in_transaction}"
                        ),
                    };
                    let db = tables(&dsn, sealed);
                    let cell = format!(
                        "{} (sealed {sealed}, prepared {prepared}, in a transaction {in_transaction})",
                        case.sql
                    );
                    let expected: Vec<Vec<i64>> =
                        case.returned.iter().map(|row| row.to_vec()).collect();
                    let got = run(&db, case, prepared, in_transaction);
                    if got != expected {
                        wrong.push(format!("{cell}: {got:?}, not {expected:?}"));
                    }
                    for &(check, value) in case.after {
                        let got = scalar(&db, check);
                        if got != value {
                            wrong.push(format!("{check} after {cell}: {got}, not {value}"));
                        }
                    }
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_block_sees_the_transaction_s_earlier_writes_but_not_the_cascade() {
    let db = tables("memory://fk_actions_after_blocks_earlier_write", false);
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO cd VALUES (9, 1)", ()).unwrap();
    let got = ints(
        db.query(
            "DELETE FROM p WHERE id = 1 RETURNING (SELECT COUNT(*) FROM cd WHERE cd.pd = p.id)",
            (),
        )
        .unwrap(),
    );
    db.execute("COMMIT", ()).unwrap();
    assert_eq!(
        got,
        [[3]],
        "the two children found and the one this transaction added"
    );
    assert_eq!(scalar(&db, "SELECT COUNT(*) FROM cd"), 1);
}
