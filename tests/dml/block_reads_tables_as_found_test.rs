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

//! A block a DML statement reads, in RETURNING or ON CONFLICT, reads the
//! tables as the statement found them: the transaction's earlier writes,
//! not the statement's own or those of its foreign key actions

use stoolap::core::Value;
use stoolap::Database;

const TABLES: [&str; 11] = [
    "CREATE TABLE p (id INTEGER PRIMARY KEY, code INTEGER UNIQUE, n INTEGER)",
    "CREATE TABLE cd (id INTEGER PRIMARY KEY, pd INTEGER REFERENCES p(id) ON DELETE CASCADE)",
    "CREATE TABLE cdd (id INTEGER PRIMARY KEY, pdd INTEGER REFERENCES cd(id) ON DELETE CASCADE)",
    "CREATE TABLE cu (id INTEGER PRIMARY KEY, pc INTEGER REFERENCES p(code) ON UPDATE CASCADE ON DELETE SET NULL)",
    "CREATE TABLE cn (id INTEGER PRIMARY KEY, pc INTEGER REFERENCES p(code) ON UPDATE SET NULL ON DELETE SET NULL)",
    "CREATE VIEW vcd AS SELECT * FROM cd",
    "INSERT INTO p VALUES (1, 5, 0), (2, 6, 0), (3, 7, 0)",
    "INSERT INTO cd VALUES (1, 1), (2, 1), (3, 2)",
    "INSERT INTO cdd VALUES (1, 1), (2, 3)",
    "INSERT INTO cu VALUES (1, 7), (2, 7), (3, 6)",
    "INSERT INTO cn VALUES (1, 5)",
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
    Case {
        sql: "INSERT INTO p VALUES (3, 8, 0), (4, 7, 0), (4, 9, 0), (3, 7, 0) \
              ON CONFLICT (id) DO UPDATE SET code = excluded.code",
        params: &[],
        returned: &[],
        after: &[
            ("SELECT COUNT(*) FROM cu WHERE pc = 7", 2),
            ("SELECT COUNT(*) FROM cu WHERE pc = 9", 0),
        ],
    },
    Case {
        sql: "INSERT INTO p VALUES (1, 50, 0), (1, 5, 0) ON CONFLICT (id) DO UPDATE \
              SET code = excluded.code",
        params: &[],
        returned: &[],
        after: &[("SELECT COUNT(*) FROM cn WHERE pc IS NULL", 1)],
    },
    Case {
        sql: "INSERT INTO p VALUES (3, 70, 0), (4, 700, 0) ON CONFLICT (id) DO UPDATE \
              SET id = excluded.id + 1, code = excluded.code",
        params: &[],
        returned: &[],
        after: &[
            ("SELECT COUNT(*) FROM cu WHERE pc = 700", 2),
            ("SELECT COUNT(*) FROM p WHERE id = 5 AND code = 700", 1),
        ],
    },
    Case {
        sql: "INSERT INTO p VALUES (3, 70, 0), (3, 700, 0) ON CONFLICT (id) DO UPDATE \
              SET code = excluded.code RETURNING (SELECT COUNT(*) FROM cu WHERE pc = p.id + 4)",
        params: &[],
        returned: &[&[2], &[2]],
        after: &[("SELECT COUNT(*) FROM cu WHERE pc = 700", 2)],
    },
    Case {
        sql: "DELETE FROM p WHERE id = 1 RETURNING \
              (SELECT COUNT(*) FROM cdd WHERE cdd.pdd IN (SELECT id FROM cd WHERE cd.pd = p.id))",
        params: &[],
        returned: &[&[1]],
        after: &[("SELECT COUNT(*) FROM cdd", 1)],
    },
    Case {
        sql: "DELETE FROM p WHERE id = 1 RETURNING \
              CASE WHEN EXISTS (SELECT 1 FROM cd WHERE cd.pd = p.id) THEN 1 ELSE 0 END, \
              CASE WHEN p.id IN (SELECT pd FROM cd) THEN 1 ELSE 0 END",
        params: &[],
        returned: &[&[1, 1]],
        after: &[("SELECT COUNT(*) FROM cd WHERE pd = 1", 0)],
    },
    Case {
        sql: "DELETE FROM p WHERE id = 1 RETURNING \
              (SELECT COUNT(*) FROM cd JOIN cu ON cu.id = cd.id WHERE cd.pd = p.id), \
              (SELECT COUNT(*) FROM (SELECT pd FROM cd) AS d WHERE d.pd = p.id)",
        params: &[],
        returned: &[&[2, 2]],
        after: &[("SELECT COUNT(*) FROM cd", 1)],
    },
];

/// How the statement runs: in autocommit, or in a transaction of an isolation
const MODES: [(&str, Option<&str>); 3] = [
    ("autocommit", None),
    ("read_committed", Some("BEGIN")),
    (
        "snapshot",
        Some("BEGIN TRANSACTION ISOLATION LEVEL SNAPSHOT"),
    ),
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

fn run(db: &Database, case: &Case, prepared: bool, begin: Option<&str>) -> Vec<Vec<i64>> {
    let params: Vec<Value> = case.params.iter().map(|&v| Value::Integer(v)).collect();
    if let Some(begin) = begin {
        db.execute(begin, ()).unwrap();
    }
    let got = match prepared {
        true => ints(db.prepare(case.sql).unwrap().query(params).unwrap()),
        false => ints(db.query(case.sql, params).unwrap()),
    };
    if begin.is_some() {
        db.execute("COMMIT", ()).unwrap();
    }
    got
}

/// Runs `CASES[at]` hot and sealed, in every mode, and checks each cell
fn check(at: usize) {
    let case = &CASES[at];
    let mut wrong = Vec::new();
    for sealed in [false, true] {
        // Sealed cells run prepared only: both forms read storage alike
        for prepared in [false, true].into_iter().filter(|&p| p || !sealed) {
            for (mode, begin) in MODES {
                let dir = tempfile::tempdir().unwrap();
                let dsn = match sealed {
                    true => format!("file://{}?sync_mode=none", dir.path().display()),
                    false => format!("memory://blocks_as_found_{at}_{prepared}_{mode}"),
                };
                let db = tables(&dsn, sealed);
                let cell = format!(
                    "{} (sealed {sealed}, prepared {prepared}, {mode})",
                    case.sql
                );
                let expected: Vec<Vec<i64>> =
                    case.returned.iter().map(|row| row.to_vec()).collect();
                let got = run(&db, case, prepared, begin);
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
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

macro_rules! cases {
    ($($name:ident = $at:expr,)*) => {
        $(
            #[test]
            fn $name() {
                check($at);
            }
        )*
    };
}

cases! {
    delete_cascade_over_two_parents = 0,
    delete_with_a_parameter = 1,
    delete_read_through_a_view = 2,
    delete_set_null = 3,
    update_cascade = 4,
    upsert_returning = 5,
    upsert_per_conflict_value = 6,
    upsert_changing_one_row_twice = 7,
    upsert_passing_a_code_to_a_row_it_inserts = 8,
    upsert_set_null_changed_back = 9,
    upsert_moving_a_key = 10,
    upsert_changing_one_row_twice_returning = 11,
    delete_two_level_cascade = 12,
    delete_exists_and_in = 13,
    delete_join_and_derived_table = 14,
}

#[test]
fn every_case_has_a_test() {
    assert_eq!(CASES.len(), 15);
}

/// Runs `earlier` in a transaction, then `sql`, returning its rows; the
/// transaction commits when `sql` succeeds
fn after_earlier_writes(db: &Database, earlier: &[&str], sql: &str) -> Vec<Vec<i64>> {
    db.execute("BEGIN", ()).unwrap();
    for statement in earlier {
        db.execute(statement, ()).unwrap();
    }
    let got = ints(db.query(sql, ()).unwrap());
    db.execute("COMMIT", ()).unwrap();
    got
}

/// The children of p(1) a DELETE's RETURNING counts
const DELETE_ONE: &str =
    "DELETE FROM p WHERE id = 1 RETURNING (SELECT COUNT(*) FROM cd WHERE cd.pd = p.id)";

#[test]
fn a_block_sees_the_transaction_s_earlier_writes_but_not_the_statement_s() {
    let cases: [(&str, &[&str], &str, i64); 5] = [
        ("insert", &["INSERT INTO cd VALUES (9, 1)"], DELETE_ONE, 3),
        ("delete", &["DELETE FROM cd WHERE id = 2"], DELETE_ONE, 1),
        (
            "repointed",
            &["UPDATE cd SET pd = 1 WHERE id = 3"],
            DELETE_ONE,
            3,
        ),
        (
            "savepoint",
            &[
                "INSERT INTO cd VALUES (9, 1)",
                "SAVEPOINT s",
                "INSERT INTO cd VALUES (10, 1)",
                "ROLLBACK TO SAVEPOINT s",
            ],
            DELETE_ONE,
            3,
        ),
        (
            "cascaded_again",
            &["UPDATE cu SET pc = 7 WHERE id = 3"],
            "UPDATE p SET code = 70 WHERE id = 3 RETURNING (SELECT COUNT(*) FROM cu WHERE pc = 7)",
            3,
        ),
    ];
    for sealed in [false, true] {
        for (name, earlier, sql, expected) in cases {
            let dir = tempfile::tempdir().unwrap();
            let dsn = match sealed {
                true => format!("file://{}?sync_mode=none", dir.path().display()),
                false => format!("memory://blocks_as_found_earlier_{name}"),
            };
            let db = tables(&dsn, sealed);
            assert_eq!(
                after_earlier_writes(&db, earlier, sql),
                [[expected]],
                "{name} (sealed {sealed}): {earlier:?} then {sql}"
            );
        }
    }
}

#[test]
fn a_statement_that_fails_after_its_cascade_leaves_the_children() {
    let db = tables("memory://blocks_as_found_failed_statement", false);
    db.execute("BEGIN", ()).unwrap();
    let failed = db.query(
        "DELETE FROM p WHERE id = 1 RETURNING (SELECT id FROM cd WHERE cd.pd = p.id)",
        (),
    );
    assert!(failed
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
        .is_err());
    assert_eq!(scalar(&db, "SELECT COUNT(*) FROM cd WHERE pd = 1"), 2);
    db.execute("COMMIT", ()).unwrap();
    assert_eq!(scalar(&db, "SELECT COUNT(*) FROM p"), 3);
}

#[test]
fn the_next_statement_sees_the_cascade() {
    let db = tables("memory://blocks_as_found_next_statement", false);
    db.execute("BEGIN", ()).unwrap();
    assert_eq!(ints(db.query(DELETE_ONE, ()).unwrap()), [[2]]);
    assert_eq!(scalar(&db, "SELECT COUNT(*) FROM cd WHERE pd = 1"), 0);
    db.execute("COMMIT", ()).unwrap();
}
