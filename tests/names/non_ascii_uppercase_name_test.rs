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

//! A name whose uppercase letters are all outside ASCII names what its
//! lowercase spelling names, in every statement

use stoolap::Database;

fn tables(db: &Database) -> Vec<String> {
    let mut out: Vec<String> = db
        .query("SHOW TABLES", ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .collect();
    out.sort();
    out
}

fn count(db: &Database, sql: &str) -> i64 {
    db.query_one::<i64, _>(sql, ()).unwrap()
}

fn run(db: &Database, sql: &str) {
    if let Err(e) = db.execute(sql, ()) {
        panic!("{sql}: {e}");
    }
}

fn ddl_and_dml_by_the_uppercase_spelling(db: &Database) {
    run(db, "CREATE TABLE \"äö\" (id INTEGER PRIMARY KEY)");
    run(db, "INSERT INTO \"ÄÖ\" VALUES (1)");
    run(db, "ALTER TABLE \"ÄÖ\" ADD COLUMN v INTEGER");
    run(db, "ALTER TABLE \"ÄÖ\" RENAME COLUMN v TO \"üv\"");
    run(db, "UPDATE \"ÄÖ\" SET \"ÜV\" = 5");
    run(db, "CREATE INDEX i_uv ON \"ÄÖ\" (\"ÜV\")");
    assert_eq!(count(db, "SELECT COUNT(*) FROM \"äö\" WHERE \"üv\" = 5"), 1);
    run(db, "DROP INDEX i_uv ON \"ÄÖ\"");
    run(db, "ALTER TABLE \"ÄÖ\" DROP COLUMN \"ÜV\"");
    run(db, "TRUNCATE TABLE \"ÄÖ\"");
    assert_eq!(count(db, "SELECT COUNT(*) FROM \"äö\""), 0);
    run(db, "ALTER TABLE \"ÄÖ\" RENAME TO \"ÖÄ\"");
    assert_eq!(tables(db), ["öä"], "the renamed table is stored lowercased");

    run(db, "CREATE TABLE \"σοφός\" (id INTEGER PRIMARY KEY)");
    run(
        db,
        "CREATE TABLE ch (id INTEGER PRIMARY KEY, p INTEGER REFERENCES \"ΣΟΦΌΣ\"(id))",
    );
    run(db, "INSERT INTO \"ΣΟΦΌΣ\" VALUES (1)");
    run(db, "INSERT INTO ch VALUES (1, 1)");
    assert!(
        db.execute("DELETE FROM \"ΣΟΦΌΣ\"", ()).is_err(),
        "the foreign key restricts the parent it names in uppercase"
    );
    assert!(
        db.execute("DROP TABLE \"ΣΟΦΌΣ\"", ()).is_err(),
        "a table a foreign key names is not dropped"
    );
    run(db, "DROP TABLE ch");
    run(db, "DROP TABLE \"ΣΟΦΌΣ\"");
    run(db, "DROP TABLE IF EXISTS \"ÖÄ\"");
    assert_eq!(tables(db), Vec::<String>::new());
    run(db, "CREATE TABLE \"ÄÖ\" (id INTEGER PRIMARY KEY)");
    assert_eq!(tables(db), ["äö"]);
}

#[test]
fn ddl_and_dml_find_a_table_by_its_non_ascii_uppercase_name() {
    let db = Database::open("memory://non_ascii_uppercase_ddl").unwrap();
    ddl_and_dml_by_the_uppercase_spelling(&db);
}

#[test]
fn a_reopened_database_finds_a_table_by_its_non_ascii_uppercase_name() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}", dir.path().display());
    {
        let db = Database::open(&dsn).unwrap();
        run(
            &db,
            "CREATE TABLE \"äö\" (id INTEGER PRIMARY KEY, v INTEGER)",
        );
        run(&db, "INSERT INTO \"ÄÖ\" VALUES (1, 1), (2, 2)");
        run(&db, "CREATE INDEX i_v ON \"ÄÖ\" (v)");
        db.close().unwrap();
    }
    let db = Database::open(&dsn).unwrap();
    assert_eq!(count(&db, "SELECT COUNT(*) FROM \"ÄÖ\" WHERE v = 2"), 1);
    run(&db, "DROP INDEX i_v ON \"ÄÖ\"");
    run(&db, "ALTER TABLE \"ÄÖ\" ADD COLUMN w INTEGER");
    run(&db, "DROP TABLE \"ÄÖ\"");
    assert_eq!(tables(&db), Vec::<String>::new());
}

#[test]
fn names_whose_lowercase_changes_length_or_context_find_their_table() {
    let db = Database::open("memory://non_ascii_uppercase_edges").unwrap();
    run(&db, "CREATE TABLE \"İ1\" (id INTEGER PRIMARY KEY)");
    run(&db, "INSERT INTO \"İ1\" VALUES (1)");
    run(&db, "ALTER TABLE \"İ1\" ADD COLUMN v INTEGER");
    run(&db, "CREATE TABLE \"ÜP\" (id INTEGER PRIMARY KEY)");
    run(&db, "DROP TABLE \"üp\"");
    run(&db, "CREATE TABLE \"straße\" (id INTEGER PRIMARY KEY)");
    run(&db, "CREATE TABLE \"STRASSE\" (id INTEGER PRIMARY KEY)");
    assert_eq!(
        tables(&db),
        ["i\u{307}1", "strasse", "straße"],
        "İ lowercases to two characters, ß stays and SS is another name"
    );
    run(&db, "DROP TABLE \"İ1\"");
    run(&db, "DROP TABLE \"STRASSE\"");
    run(&db, "DROP TABLE \"STRAßE\"");
    assert_eq!(tables(&db), Vec::<String>::new());
}

fn binding_table(dsn: &str) -> Database {
    let db = Database::open(dsn).unwrap();
    run(
        &db,
        "CREATE TABLE t (id INTEGER PRIMARY KEY, \"äb\" INTEGER DEFAULT 9, c INTEGER)",
    );
    run(&db, "INSERT INTO t VALUES (1, 1, 1), (2, 1, 2)");
    db
}

#[test]
fn update_set_default_names_its_column_in_non_ascii_uppercase() {
    let db = binding_table("memory://non_ascii_uppercase_set_default");
    run(&db, "UPDATE t SET \"ÄB\" = DEFAULT WHERE id = 2");
    assert_eq!(count(&db, "SELECT \"äb\" FROM t WHERE id = 2"), 9);
}

#[test]
fn a_window_name_binds_in_non_ascii_uppercase() {
    let db = binding_table("memory://non_ascii_uppercase_window");
    let sums: Vec<i64> = db
        .query(
            "SELECT SUM(c) OVER \"ÄW\" FROM t WINDOW \"äw\" AS (ORDER BY id)",
            (),
        )
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert_eq!(sums, [1, 3]);
}

#[test]
fn an_aggregate_alias_is_no_group_by_key_in_non_ascii_uppercase() {
    let db = binding_table("memory://non_ascii_uppercase_alias_key");
    assert!(
        db.execute("SELECT COUNT(*) AS \"çc\" FROM t GROUP BY \"Çc\"", ())
            .is_err(),
        "an aggregate's alias is no GROUP BY key in any case"
    );
}

fn sealed_table(dir: &tempfile::TempDir) -> Database {
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    run(
        &db,
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, \"Ab\" INTEGER UNIQUE, \
         \"Äc\" INTEGER UNIQUE)",
    );
    run(&db, "INSERT INTO t VALUES (1, 11, 1, 1), (2, 22, 2, 2)");
    run(&db, "PRAGMA CHECKPOINT");
    db
}

#[test]
fn sealed_rows_keep_a_column_renamed_to_non_ascii_uppercase() {
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_table(&dir);
    run(&db, "ALTER TABLE t RENAME COLUMN a TO \"Äd\"");
    assert_eq!(count(&db, "SELECT SUM(\"äd\") FROM t"), 33);
    assert_eq!(count(&db, "SELECT id FROM t WHERE \"ÄD\" = 22"), 2);
}

#[test]
fn sealed_rows_keep_a_unique_column_with_uppercase_letters_unique() {
    let dir = tempfile::tempdir().unwrap();
    let db = sealed_table(&dir);
    assert!(
        db.execute("INSERT INTO t VALUES (3, 0, 1, 3)", ()).is_err(),
        "a sealed row holds \"Ab\" = 1"
    );
    assert!(
        db.execute("INSERT INTO t VALUES (4, 0, 4, 2)", ()).is_err(),
        "a sealed row holds \"Äc\" = 2"
    );
}

#[test]
fn a_commit_rechecks_a_unique_column_with_uppercase_letters_against_sealed_rows() {
    let dir = tempfile::tempdir().unwrap();
    let dsn = format!("file://{}?checkpoint_interval=3600", dir.path().display());
    let db = Database::open(&dsn).unwrap();
    run(
        &db,
        "CREATE TABLE t (id INTEGER PRIMARY KEY, \"Ab\" INTEGER UNIQUE)",
    );
    run(&db, "INSERT INTO t VALUES (1, 10)");
    run(&db, "PRAGMA CHECKPOINT");
    let mut txn = db.begin().unwrap();
    txn.execute("INSERT INTO t VALUES (2, 7)", ()).unwrap();
    run(&db, "INSERT INTO t VALUES (3, 7)");
    run(&db, "PRAGMA CHECKPOINT");
    let committed = txn.commit();
    assert!(
        matches!(
            committed,
            Err(stoolap::core::Error::UniqueConstraint { row_id: 3, .. })
        ),
        "{committed:?}"
    );
}

#[test]
fn on_conflict_names_a_unique_column_with_uppercase_letters() {
    let db = Database::open("memory://on_conflict_name_case").unwrap();
    run(
        &db,
        "CREATE TABLE u (id INTEGER PRIMARY KEY, \"Ab\" INTEGER UNIQUE, \"Äc\" INTEGER UNIQUE)",
    );
    run(&db, "INSERT INTO u VALUES (1, 1, 1)");
    for (target, row) in [
        ("\"Ab\"", "2, 1, 2"),
        ("ab", "3, 1, 3"),
        ("\"äC\"", "4, 4, 1"),
    ] {
        run(
            &db,
            &format!("INSERT INTO u VALUES ({row}) ON CONFLICT ({target}) DO NOTHING"),
        );
    }
    assert_eq!(count(&db, "SELECT COUNT(*) FROM u"), 1);
}

#[test]
fn on_conflict_raises_a_violation_of_an_index_repeating_one_target_column() {
    let db = Database::open("memory://on_conflict_repeated_column").unwrap();
    run(
        &db,
        "CREATE TABLE u (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
    );
    run(&db, "CREATE UNIQUE INDEX uaa ON u (a, a)");
    run(&db, "CREATE UNIQUE INDEX uab ON u (a, b)");
    run(&db, "INSERT INTO u VALUES (1, 1, 1)");
    let sql = "INSERT INTO u VALUES (2, 1, 2) ON CONFLICT (a, b) DO NOTHING";
    assert!(
        db.execute(sql, ()).is_err(),
        "the (a, a) violation is outside the target (a, b)"
    );
}

#[test]
fn a_savepoint_name_ignores_case() {
    let db = Database::open("memory://savepoint_name_case").unwrap();
    run(&db, "CREATE TABLE t (id INTEGER PRIMARY KEY)");
    for (created, named) in [("sp", "SP"), ("\"äb\"", "\"ÄB\"")] {
        run(&db, "BEGIN");
        run(&db, &format!("SAVEPOINT {created}"));
        run(&db, "INSERT INTO t VALUES (1)");
        run(&db, &format!("ROLLBACK TO SAVEPOINT {named}"));
        run(&db, &format!("RELEASE SAVEPOINT {named}"));
        run(&db, "COMMIT");
        assert_eq!(
            count(&db, "SELECT COUNT(*) FROM t"),
            0,
            "{created} as {named}"
        );
    }
}

#[test]
fn a_table_primary_key_names_its_column_in_non_ascii_uppercase() {
    let db = Database::open("memory://non_ascii_uppercase_pk").unwrap();
    run(
        &db,
        "CREATE TABLE t (\"äb\" INTEGER, c INTEGER, PRIMARY KEY (\"ÄB\"))",
    );
    run(&db, "INSERT INTO t VALUES (1, 1)");
    assert!(
        db.execute("INSERT INTO t VALUES (1, 2)", ()).is_err(),
        "the column PRIMARY KEY (\"ÄB\") names is the key"
    );
}

#[test]
fn a_composite_key_and_cluster_by_name_columns_in_non_ascii_uppercase() {
    let db = Database::open("memory://non_ascii_uppercase_composite").unwrap();
    run(
        &db,
        "CREATE TABLE u (id INTEGER PRIMARY KEY, \"äb\" INTEGER, c INTEGER, d INTEGER, \
         PRIMARY KEY (\"ÄB\", c)) CLUSTER BY (\"ÄB\")",
    );
    assert!(
        db.execute("INSERT INTO u VALUES (1, NULL, 1, 1)", ())
            .is_err(),
        "a column of a composite PRIMARY KEY (\"ÄB\", c) holds no NULL"
    );
}
