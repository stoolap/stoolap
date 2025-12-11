// Regression tests for Bug Batch 8
// Tests for bugs found during exploratory testing (keywords after dot, DEFAULT in VALUES)

use stoolap::Database;

fn setup_db() -> Database {
    Database::open_in_memory().expect("Failed to create in-memory database")
}

// =============================================================================
// BUG: Keywords after dot in qualified identifiers (t.level, t.type)
// Problem: Using reserved keywords as column names with table qualifier failed
//          e.g., "SELECT t.level FROM t" returned parse error
// =============================================================================

#[test]
fn test_bugs8_keyword_column_level() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t_level (id INTEGER PRIMARY KEY, level INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO t_level VALUES (1, 100), (2, 200)", ())
        .expect("Failed to insert data");

    // Using keyword 'level' as column name with table alias
    let mut rows = db
        .query("SELECT t.level FROM t_level t ORDER BY t.id", ())
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 100);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 200);

    assert!(rows.next().is_none());
}

#[test]
fn test_bugs8_keyword_column_type() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t_type (id INTEGER PRIMARY KEY, type TEXT)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO t_type VALUES (1, 'premium'), (2, 'basic')", ())
        .expect("Failed to insert data");

    // Using keyword 'type' as column name with table alias
    let mut rows = db
        .query("SELECT t.type FROM t_type t ORDER BY t.id", ())
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "premium");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "basic");

    assert!(rows.next().is_none());
}

#[test]
fn test_bugs8_keyword_column_in_expression() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t_expr (id INTEGER PRIMARY KEY, level INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO t_expr VALUES (1, 100)", ())
        .expect("Failed to insert data");

    // Keyword column in expression with table alias
    let mut rows = db
        .query("SELECT t.level + 10 AS boosted FROM t_expr t", ())
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 110);
}

#[test]
fn test_bugs8_multiple_keyword_columns() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t_keywords (id INTEGER PRIMARY KEY, level INTEGER, type TEXT, value FLOAT)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO t_keywords VALUES (1, 100, 'A', 1.5)", ())
        .expect("Failed to insert data");

    // Multiple keyword columns in same query
    let mut rows = db
        .query("SELECT t.level, t.type, t.value FROM t_keywords t", ())
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 100);
    assert_eq!(row.get::<String>(1).unwrap(), "A");
    assert_eq!(row.get::<f64>(2).unwrap(), 1.5);
}

// =============================================================================
// BUG: DEFAULT keyword in VALUES clause returns NULL
// Problem: INSERT INTO t VALUES (1, DEFAULT) returned NULL instead of
//          using the column's default value
// =============================================================================

#[test]
fn test_bugs8_default_in_values_text() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t_default1 (id INTEGER PRIMARY KEY, name TEXT DEFAULT 'anonymous')",
        (),
    )
    .expect("Failed to create table");

    // Use DEFAULT keyword for text column
    db.execute("INSERT INTO t_default1 VALUES (1, DEFAULT)", ())
        .expect("Insert failed");

    let mut rows = db
        .query("SELECT * FROM t_default1", ())
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);
    assert_eq!(row.get::<String>(1).unwrap(), "anonymous");
}

#[test]
fn test_bugs8_default_in_values_integer() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t_default2 (id INTEGER PRIMARY KEY, score INTEGER DEFAULT 100)",
        (),
    )
    .expect("Failed to create table");

    // Use DEFAULT keyword for integer column
    db.execute("INSERT INTO t_default2 VALUES (1, DEFAULT)", ())
        .expect("Insert failed");

    let mut rows = db
        .query("SELECT * FROM t_default2", ())
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);
    assert_eq!(row.get::<i64>(1).unwrap(), 100);
}

#[test]
fn test_bugs8_default_mixed_with_values() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t_default3 (id INTEGER PRIMARY KEY, name TEXT DEFAULT 'guest', score INTEGER DEFAULT 50)",
        (),
    )
    .expect("Failed to create table");

    // Mix explicit values with DEFAULT
    db.execute("INSERT INTO t_default3 VALUES (1, 'Alice', DEFAULT)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t_default3 VALUES (2, DEFAULT, 200)", ())
        .expect("Insert failed");
    db.execute("INSERT INTO t_default3 VALUES (3, DEFAULT, DEFAULT)", ())
        .expect("Insert failed");

    let mut rows = db
        .query("SELECT * FROM t_default3 ORDER BY id", ())
        .expect("Query failed");

    // Row 1: explicit name, default score
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);
    assert_eq!(row.get::<String>(1).unwrap(), "Alice");
    assert_eq!(row.get::<i64>(2).unwrap(), 50);

    // Row 2: default name, explicit score
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 2);
    assert_eq!(row.get::<String>(1).unwrap(), "guest");
    assert_eq!(row.get::<i64>(2).unwrap(), 200);

    // Row 3: all defaults
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 3);
    assert_eq!(row.get::<String>(1).unwrap(), "guest");
    assert_eq!(row.get::<i64>(2).unwrap(), 50);
}

#[test]
fn test_bugs8_default_no_default_defined() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t_default4 (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed to create table");

    // DEFAULT on column without default should give NULL
    db.execute("INSERT INTO t_default4 VALUES (1, DEFAULT)", ())
        .expect("Insert failed");

    let mut rows = db
        .query("SELECT * FROM t_default4", ())
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);
    assert!(row.is_null(1)); // NULL when no default defined
}

#[test]
fn test_bugs8_default_with_named_columns() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t_default5 (id INTEGER PRIMARY KEY, a TEXT DEFAULT 'A', b TEXT DEFAULT 'B', c TEXT DEFAULT 'C')",
        (),
    )
    .expect("Failed to create table");

    // DEFAULT with named columns
    db.execute("INSERT INTO t_default5 (id, b) VALUES (1, DEFAULT)", ())
        .expect("Insert failed");

    let mut rows = db
        .query("SELECT * FROM t_default5", ())
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);
    assert_eq!(row.get::<String>(1).unwrap(), "A"); // default for a (not specified)
    assert_eq!(row.get::<String>(2).unwrap(), "B"); // DEFAULT keyword used
    assert_eq!(row.get::<String>(3).unwrap(), "C"); // default for c (not specified)
}

// =============================================================================
// BUG (FIXED): ALTER TABLE ADD COLUMN with DEFAULT now applies to existing rows
// Previously: ALTER TABLE t ADD COLUMN c TEXT DEFAULT 'x' left NULL for existing rows
// Now: Existing rows get the default value applied during schema evolution
// =============================================================================

#[test]
fn test_bugs8_alter_table_add_column_default_text() {
    let db = setup_db();

    db.execute("CREATE TABLE t_alter1 (id INTEGER PRIMARY KEY)", ())
        .expect("Failed to create table");
    db.execute("INSERT INTO t_alter1 VALUES (1), (2)", ())
        .expect("Failed to insert data");

    // Add column with DEFAULT
    db.execute(
        "ALTER TABLE t_alter1 ADD COLUMN status TEXT DEFAULT 'active'",
        (),
    )
    .expect("Failed to alter table");

    // Insert new row - should use default
    db.execute("INSERT INTO t_alter1 (id) VALUES (3)", ())
        .expect("Failed to insert data");

    let mut rows = db
        .query("SELECT * FROM t_alter1 ORDER BY id", ())
        .expect("Query failed");

    // Existing rows now get the default value (schema evolution)
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);
    assert_eq!(row.get::<String>(1).unwrap(), "active");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 2);
    assert_eq!(row.get::<String>(1).unwrap(), "active");

    // New row should have default value
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 3);
    assert_eq!(row.get::<String>(1).unwrap(), "active");
}

#[test]
fn test_bugs8_alter_table_add_column_default_integer() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t_alter2 (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO t_alter2 VALUES (1, 'Alice')", ())
        .expect("Failed to insert data");

    // Add integer column with DEFAULT
    db.execute(
        "ALTER TABLE t_alter2 ADD COLUMN score INTEGER DEFAULT 100",
        (),
    )
    .expect("Failed to alter table");

    // Insert new row - should use default
    db.execute("INSERT INTO t_alter2 (id, name) VALUES (2, 'Bob')", ())
        .expect("Failed to insert data");

    let mut rows = db
        .query("SELECT * FROM t_alter2 ORDER BY id", ())
        .expect("Query failed");

    // Existing row now gets default value (schema evolution)
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);
    assert_eq!(row.get::<i64>(2).unwrap(), 100);

    // New row with default
    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 2);
    assert_eq!(row.get::<i64>(2).unwrap(), 100);
}

#[test]
fn test_bugs8_alter_table_add_column_default_with_explicit_value() {
    let db = setup_db();

    db.execute("CREATE TABLE t_alter3 (id INTEGER PRIMARY KEY)", ())
        .expect("Failed to create table");

    // Add column with DEFAULT
    db.execute(
        "ALTER TABLE t_alter3 ADD COLUMN level INTEGER DEFAULT 1",
        (),
    )
    .expect("Failed to alter table");

    // Insert with DEFAULT keyword
    db.execute("INSERT INTO t_alter3 VALUES (1, DEFAULT)", ())
        .expect("Insert with DEFAULT failed");

    // Insert with explicit value (overriding default)
    db.execute("INSERT INTO t_alter3 VALUES (2, 99)", ())
        .expect("Insert with explicit value failed");

    let mut rows = db
        .query("SELECT * FROM t_alter3 ORDER BY id", ())
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);
    assert_eq!(row.get::<i64>(1).unwrap(), 1); // DEFAULT keyword

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 2);
    assert_eq!(row.get::<i64>(1).unwrap(), 99); // explicit value
}

// =============================================================================
// BUG: CTE UNION ALL referencing other CTEs only returns first SELECT
// Problem: WITH a AS (...), b AS (...), c AS (SELECT FROM a UNION ALL SELECT FROM b)
//          only returned rows from 'a', not from 'b'
// =============================================================================

#[test]
fn test_bugs8_cte_union_all_referencing_ctes() {
    let db = setup_db();

    // CTE c references both a and b with UNION ALL
    let mut rows = db
        .query(
            "WITH
                a AS (SELECT 1 as n),
                b AS (SELECT 2 as n),
                c AS (SELECT n FROM a UNION ALL SELECT n FROM b)
            SELECT * FROM c ORDER BY n",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 2);

    assert!(rows.next().is_none());
}

#[test]
fn test_bugs8_cte_union_referencing_ctes() {
    let db = setup_db();

    // CTE with UNION (not UNION ALL) - should deduplicate
    let mut rows = db
        .query(
            "WITH
                a AS (SELECT 1 as n),
                b AS (SELECT 1 as n),
                c AS (SELECT n FROM a UNION SELECT n FROM b)
            SELECT * FROM c",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);

    // Should only have one row due to UNION deduplication
    assert!(rows.next().is_none());
}

#[test]
fn test_bugs8_cte_complex_union_all() {
    let db = setup_db();

    // More complex scenario with three CTEs
    let mut rows = db
        .query(
            "WITH
                x AS (SELECT 10 as val),
                y AS (SELECT 20 as val),
                z AS (SELECT 30 as val),
                combined AS (
                    SELECT val FROM x
                    UNION ALL SELECT val FROM y
                    UNION ALL SELECT val FROM z
                )
            SELECT * FROM combined ORDER BY val",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 10);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 20);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 30);

    assert!(rows.next().is_none());
}

#[test]
fn test_bugs8_cte_union_with_table() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t_union (id INTEGER PRIMARY KEY, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute("INSERT INTO t_union VALUES (1, 100), (2, 200)", ())
        .expect("Failed to insert data");

    // CTE union with actual table
    let mut rows = db
        .query(
            "WITH
                cte_vals AS (SELECT 50 as val)
            SELECT val FROM cte_vals
            UNION ALL
            SELECT val FROM t_union
            ORDER BY val",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 50);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 100);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 200);

    assert!(rows.next().is_none());
}
