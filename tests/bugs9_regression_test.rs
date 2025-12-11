// Regression tests for Bug Batch 9
// Tests for bugs found during exploratory testing:
// - Bug #7: COALESCE with duplicate aggregate returns NULL
// - Bug #8: Column alias without AS keyword
// - Bug #9: WINDOW clause (named windows)

use stoolap::Database;

fn setup_db() -> Database {
    Database::open_in_memory().expect("Failed to create in-memory database")
}

// =============================================================================
// BUG #7: COALESCE with Same Aggregate in SELECT Returns NULL
// Problem: When SUM(val) and COALESCE(SUM(val), 0) both appear in SELECT,
//          the COALESCE version incorrectly returned NULL for all rows.
// =============================================================================

#[test]
fn test_bugs9_coalesce_with_same_aggregate() {
    let db = setup_db();

    db.execute("CREATE TABLE t_coalesce (grp TEXT, val INTEGER)", ())
        .expect("Failed to create table");
    db.execute(
        "INSERT INTO t_coalesce VALUES ('A', 10), ('A', 20), ('B', NULL), ('C', 5)",
        (),
    )
    .expect("Failed to insert data");

    // Bug fix: both SUM(val) and COALESCE(SUM(val), 0) should work correctly
    let mut rows = db
        .query(
            "SELECT grp, SUM(val) AS raw, COALESCE(SUM(val), 0) AS coalesced FROM t_coalesce GROUP BY grp ORDER BY grp",
            (),
        )
        .expect("Query failed");

    // Group A: SUM(10+20) = 30
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "A");
    assert_eq!(row.get::<i64>(1).unwrap(), 30);
    assert_eq!(row.get::<i64>(2).unwrap(), 30); // Was returning NULL before fix

    // Group B: SUM(NULL) = NULL, COALESCE = 0
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "B");
    assert!(row.is_null(1)); // NULL
    assert_eq!(row.get::<i64>(2).unwrap(), 0); // Was returning NULL before fix

    // Group C: SUM(5) = 5
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "C");
    assert_eq!(row.get::<i64>(1).unwrap(), 5);
    assert_eq!(row.get::<i64>(2).unwrap(), 5); // Was returning NULL before fix
}

#[test]
fn test_bugs9_coalesce_aggregate_only() {
    let db = setup_db();

    db.execute("CREATE TABLE t_coalesce2 (grp TEXT, val INTEGER)", ())
        .expect("Failed to create table");
    db.execute(
        "INSERT INTO t_coalesce2 VALUES ('A', 10), ('A', 20), ('B', NULL), ('C', 5)",
        (),
    )
    .expect("Failed to insert data");

    // When only COALESCE version is used (original behavior that worked)
    let mut rows = db
        .query(
            "SELECT grp, COALESCE(SUM(val), 0) AS coalesced FROM t_coalesce2 GROUP BY grp ORDER BY grp",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(1).unwrap(), 30);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(1).unwrap(), 0); // NULL coalesced to 0

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(1).unwrap(), 5);
}

#[test]
fn test_bugs9_multiple_coalesce_aggregates() {
    let db = setup_db();

    db.execute("CREATE TABLE t_coalesce3 (grp TEXT, val INTEGER)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO t_coalesce3 VALUES ('A', 10), ('B', NULL)", ())
        .expect("Failed to insert data");

    // Multiple different aggregates with COALESCE
    let mut rows = db
        .query(
            "SELECT grp, COALESCE(SUM(val), 0) AS s, COALESCE(AVG(val), 0.0) AS a FROM t_coalesce3 GROUP BY grp ORDER BY grp",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "A");
    assert_eq!(row.get::<i64>(1).unwrap(), 10);
    assert_eq!(row.get::<f64>(2).unwrap(), 10.0);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "B");
    assert_eq!(row.get::<i64>(1).unwrap(), 0);
    assert_eq!(row.get::<f64>(2).unwrap(), 0.0);
}

// =============================================================================
// BUG #8: Column Alias Without AS Keyword Not Supported
// Problem: Standard SQL allows "SELECT col alias" but Stoolap required "SELECT col AS alias"
// =============================================================================

#[test]
fn test_bugs9_alias_without_as_simple() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t_alias (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO t_alias VALUES (1, 'test')", ())
        .expect("Failed to insert data");

    // Simple alias without AS keyword
    let mut rows = db
        .query("SELECT id identifier FROM t_alias", ())
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);

    // Check that we can retrieve by the alias name (column header should be "identifier")
    let cols = rows.columns().to_vec();
    assert!(cols.contains(&"identifier".to_string()));
}

#[test]
fn test_bugs9_alias_without_as_multiple() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t_alias2 (id INTEGER PRIMARY KEY, name TEXT, value FLOAT)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO t_alias2 VALUES (1, 'test', 3.14)", ())
        .expect("Failed to insert data");

    // Multiple aliases without AS
    let mut rows = db
        .query("SELECT id a, name b, value c FROM t_alias2", ())
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);
    assert_eq!(row.get::<String>(1).unwrap(), "test");

    let cols = rows.columns().to_vec();
    assert!(cols.contains(&"a".to_string()));
    assert!(cols.contains(&"b".to_string()));
    assert!(cols.contains(&"c".to_string()));
}

#[test]
fn test_bugs9_alias_without_as_expression() {
    let db = setup_db();

    // Expression alias without AS
    let mut rows = db.query("SELECT 1 + 2 result", ()).expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 3);

    let cols = rows.columns().to_vec();
    assert!(cols.contains(&"result".to_string()));
}

#[test]
fn test_bugs9_alias_mixed_with_and_without_as() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t_alias3 (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO t_alias3 VALUES (1, 'test')", ())
        .expect("Failed to insert data");

    // Mix of alias with AS and without AS
    let mut rows = db
        .query("SELECT id a, name AS b FROM t_alias3", ())
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);
    assert_eq!(row.get::<String>(1).unwrap(), "test");

    let cols = rows.columns().to_vec();
    assert!(cols.contains(&"a".to_string()));
    assert!(cols.contains(&"b".to_string()));
}

// =============================================================================
// BUG #9: WINDOW Clause (Named Windows) Parse Error
// Problem: WINDOW w AS (...) syntax was not parsed correctly
// =============================================================================

#[test]
fn test_bugs9_window_clause_simple() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t_win (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO t_win VALUES (1, 10), (2, 20), (3, 30)", ())
        .expect("Failed to insert data");

    // Named window with ORDER BY
    let mut rows = db
        .query(
            "SELECT id, val, RANK() OVER w FROM t_win WINDOW w AS (ORDER BY val)",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);
    assert_eq!(row.get::<i64>(1).unwrap(), 10);
    assert_eq!(row.get::<i64>(2).unwrap(), 1);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(2).unwrap(), 2);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(2).unwrap(), 3);
}

#[test]
fn test_bugs9_window_clause_multiple_functions() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t_win2 (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO t_win2 VALUES (1, 10), (2, 20), (3, 30)", ())
        .expect("Failed to insert data");

    // Multiple window functions sharing same named window
    let mut rows = db
        .query(
            "SELECT id, ROW_NUMBER() OVER w, SUM(val) OVER w, RANK() OVER w FROM t_win2 WINDOW w AS (ORDER BY val)",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);
    assert_eq!(row.get::<i64>(1).unwrap(), 1); // ROW_NUMBER
    assert_eq!(row.get::<i64>(2).unwrap(), 10); // SUM running total
    assert_eq!(row.get::<i64>(3).unwrap(), 1); // RANK

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(1).unwrap(), 2);
    assert_eq!(row.get::<i64>(2).unwrap(), 30);
    assert_eq!(row.get::<i64>(3).unwrap(), 2);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(1).unwrap(), 3);
    assert_eq!(row.get::<i64>(2).unwrap(), 60);
    assert_eq!(row.get::<i64>(3).unwrap(), 3);
}

#[test]
fn test_bugs9_window_clause_with_partition() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t_win3 (id INTEGER PRIMARY KEY, grp TEXT, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO t_win3 VALUES (1, 'A', 10), (2, 'A', 20), (3, 'B', 30), (4, 'B', 40)",
        (),
    )
    .expect("Failed to insert data");

    // Named window with PARTITION BY and ORDER BY
    let mut rows = db
        .query(
            "SELECT id, grp, val, ROW_NUMBER() OVER w FROM t_win3 WINDOW w AS (PARTITION BY grp ORDER BY val)",
            (),
        )
        .expect("Query failed");

    // Group A: row numbers 1, 2
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(1).unwrap(), "A");
    assert_eq!(row.get::<i64>(3).unwrap(), 1);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(1).unwrap(), "A");
    assert_eq!(row.get::<i64>(3).unwrap(), 2);

    // Group B: row numbers reset to 1, 2
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(1).unwrap(), "B");
    assert_eq!(row.get::<i64>(3).unwrap(), 1);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(1).unwrap(), "B");
    assert_eq!(row.get::<i64>(3).unwrap(), 2);
}

#[test]
fn test_bugs9_inline_window_still_works() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t_win4 (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO t_win4 VALUES (1, 10), (2, 20), (3, 30)", ())
        .expect("Failed to insert data");

    // Inline window spec should still work
    let mut rows = db
        .query("SELECT id, val, RANK() OVER (ORDER BY val) FROM t_win4", ())
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(2).unwrap(), 1);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(2).unwrap(), 2);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(2).unwrap(), 3);
}

#[test]
fn test_bugs9_unknown_window_name_error() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t_win5 (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO t_win5 VALUES (1, 10)", ())
        .expect("Failed to insert data");

    // Reference to undefined window should error
    let result = db.query("SELECT id, RANK() OVER w FROM t_win5", ());

    match result {
        Ok(_) => panic!("Expected error for unknown window name"),
        Err(e) => {
            let err = e.to_string();
            assert!(
                err.contains("Unknown window name"),
                "Expected 'Unknown window name' error, got: {}",
                err
            );
        }
    }
}
