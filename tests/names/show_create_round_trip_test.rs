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

//! SHOW CREATE TABLE and SHOW CREATE VIEW print DDL that, run again, makes
//! the same table or view: every name quoted where the parser needs it, a
//! VECTOR's dimension, and each UNIQUE key the table keeps

use stoolap::Database;

fn text(db: &Database, sql: &str, column: usize) -> Vec<String> {
    db.query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| row.unwrap().get_value(column).unwrap().to_string())
        .collect()
}

fn create_table(db: &Database, table: &str) -> String {
    text(db, &format!("SHOW CREATE TABLE {table}"), 1).remove(0)
}

const TABLES: &[(&str, &str, &str)] = &[
    (
        "CREATE TABLE \"my t\" (\"product id\" INTEGER PRIMARY KEY, \"a\"\"b\" TEXT)",
        "\"my t\"",
        "CREATE TABLE \"my t\" (\"product id\" INTEGER PRIMARY KEY, \"a\"\"b\" TEXT)",
    ),
    (
        "CREATE TABLE kw (\"order\" INTEGER PRIMARY KEY, \"select\" TEXT, \"null\" INTEGER, \
         level INTEGER, CHECK (\"null\" > level))",
        "kw",
        "CREATE TABLE kw (\"order\" INTEGER PRIMARY KEY, \"select\" TEXT, \"null\" INTEGER, \
         level INTEGER, CHECK (\"null\" > level))",
    ),
    (
        "CREATE TABLE c (id INTEGER PRIMARY KEY, \"parent id\" INTEGER REFERENCES \"my t\"(\"product id\"))",
        "c",
        "CREATE TABLE c (id INTEGER PRIMARY KEY, \"parent id\" INTEGER, FOREIGN KEY (\"parent id\") \
         REFERENCES \"my t\"(\"product id\") ON DELETE RESTRICT ON UPDATE RESTRICT)",
    ),
    (
        "CREATE TABLE ct (id INTEGER PRIMARY KEY, \"t s\" TIMESTAMP) CLUSTER BY (\"t s\")",
        "ct",
        "CREATE TABLE ct (id INTEGER PRIMARY KEY, \"t s\" TIMESTAMP) CLUSTER BY (\"t s\")",
    ),
    (
        "CREATE TABLE k (id INTEGER PRIMARY KEY, e VECTOR(3))",
        "k",
        "CREATE TABLE k (id INTEGER PRIMARY KEY, e VECTOR(3))",
    ),
    (
        "CREATE TABLE u (a INTEGER PRIMARY KEY, b INTEGER, \"c d\" INTEGER, UNIQUE (b, \"c d\"))",
        "u",
        "CREATE TABLE u (a INTEGER PRIMARY KEY, b INTEGER, \"c d\" INTEGER, UNIQUE (b, \"c d\"))",
    ),
    (
        "CREATE TABLE cp (a INTEGER, b INTEGER, PRIMARY KEY (a, b))",
        "cp",
        "CREATE TABLE cp (a INTEGER NOT NULL, b INTEGER NOT NULL, UNIQUE (a, b))",
    ),
    (
        "CREATE TABLE \"check\" (id INTEGER PRIMARY KEY, \"unique\" INTEGER, \"check\" INTEGER, \
         UNIQUE (\"unique\", \"check\"))",
        "\"check\"",
        "CREATE TABLE check (id INTEGER PRIMARY KEY, \"unique\" INTEGER, \"check\" INTEGER, \
         UNIQUE (unique, check))",
    ),
    (
        "CREATE TABLE n (a INTEGER PRIMARY KEY, \"0\" INTEGER UNIQUE, b INTEGER, c INTEGER, \
         UNIQUE (b, c))",
        "n",
        "CREATE TABLE n (a INTEGER PRIMARY KEY, \"0\" INTEGER UNIQUE, b INTEGER, c INTEGER, \
         UNIQUE (b, c))",
    ),
    (
        "CREATE TABLE ck (id INTEGER PRIMARY KEY AUTO_INCREMENT, \
         age INTEGER NOT NULL DEFAULT 3 CHECK (age >= 0), ok BOOLEAN CHECK (ok), \
         s TEXT CHECK (s IN ('a', 'b)')), j JSON, f FLOAT, \
         CHECK (age > 0 OR ok), CHECK ((age > 1) AND (ok)))",
        "ck",
        "CREATE TABLE ck (id INTEGER PRIMARY KEY AUTO_INCREMENT, \
         age INTEGER NOT NULL DEFAULT 3 CHECK (age >= 0), ok BOOLEAN CHECK (ok), \
         s TEXT CHECK (s IN ('a', 'b)')), j JSON, f FLOAT, \
         CHECK ((age > 0) OR ok), CHECK ((age > 1) AND ok))",
    ),
];

#[test]
fn show_create_table_prints_ddl_that_makes_the_same_table() {
    let db = Database::open("memory://show_create_round_trip").unwrap();
    let again = Database::open("memory://show_create_round_trip_again").unwrap();
    for (create, name, want) in TABLES {
        db.execute(create, ())
            .unwrap_or_else(|e| panic!("{create}: {e}"));
        let shown = create_table(&db, name);
        assert_eq!(shown, *want, "{create}");
        again
            .execute(&shown, ())
            .unwrap_or_else(|e| panic!("{shown}: {e}"));
        assert_eq!(create_table(&again, name), shown, "{shown}");
    }

    again.execute("INSERT INTO u VALUES (1, 1, 1)", ()).unwrap();
    let err = again
        .execute("INSERT INTO u VALUES (2, 1, 1)", ())
        .unwrap_err();
    assert!(err.to_string().contains("unique"), "{err}");
    again.execute("INSERT INTO cp VALUES (1, 1)", ()).unwrap();
    let err = again
        .execute("INSERT INTO cp VALUES (1, 1)", ())
        .unwrap_err();
    assert!(err.to_string().contains("unique"), "{err}");
    let err = again
        .execute("INSERT INTO k VALUES (1, '[1.0, 2.0]')", ())
        .unwrap_err();
    assert!(err.to_string().contains("dimension"), "{err}");
}

#[test]
fn describe_and_show_indexes_print_what_the_table_was_made_with() {
    let db = Database::open("memory://show_create_describe_indexes").unwrap();
    db.execute("CREATE TABLE k (id INTEGER PRIMARY KEY, e VECTOR(3))", ())
        .unwrap();
    db.execute(
        "CREATE INDEX hk ON k(e) USING HNSW \
         WITH (metric = 'l2', m = 8, ef_construction = 64, ef_search = 99)",
        (),
    )
    .unwrap();
    assert_eq!(text(&db, "DESCRIBE k", 1), vec!["INTEGER", "VECTOR(3)"]);
    db.execute(
        "CREATE TABLE ck (id INTEGER PRIMARY KEY AUTO_INCREMENT CHECK (id > 0), \
         age INTEGER CHECK (age >= 0), ok BOOLEAN CHECK (ok), \
         s TEXT CHECK (s IN ('a', 'b)')), f FLOAT, CHECK (age > f))",
        (),
    )
    .unwrap();
    assert_eq!(
        text(&db, "DESCRIBE ck", 1),
        vec!["INTEGER", "INTEGER", "BOOLEAN", "TEXT", "FLOAT"]
    );
    assert_eq!(
        text(&db, "DESCRIBE ck", 5),
        vec![
            "auto_increment, CHECK (id > 0)",
            "CHECK (age >= 0)",
            "CHECK (ok)",
            "CHECK (s IN ('a', 'b)'))",
            "",
        ]
    );
    assert_eq!(
        text(&db, "SHOW INDEXES FROM k", 5),
        vec!["metric=l2, m=8, ef_construction=64, ef_search=99"]
    );
}

#[test]
fn show_create_view_prints_a_view_that_runs_again() {
    let db = Database::open("memory://show_create_view_round_trip").unwrap();
    db.execute(
        "CREATE TABLE kw (\"order\" INTEGER PRIMARY KEY, \"select\" TEXT, \"null\" INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO kw VALUES (1, 'x', 5)", ()).unwrap();
    db.execute(
        "CREATE VIEW \"my v\" AS SELECT \"order\", \"select\", \"null\" + 1 AS n FROM kw",
        (),
    )
    .unwrap();
    let shown = text(&db, "SHOW CREATE VIEW \"my v\"", 1).remove(0);
    assert_eq!(
        shown,
        "CREATE VIEW \"my v\" AS SELECT \"order\", \"select\", (\"null\" + 1) AS n FROM kw"
    );
    db.execute("DROP VIEW \"my v\"", ()).unwrap();
    db.execute(&shown, ()).unwrap();
    assert_eq!(text(&db, "SELECT n FROM \"my v\"", 0), vec!["6"]);
}
