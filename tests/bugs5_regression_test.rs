// Regression tests for Bug Batch 5
// Tests for bugs discovered during exploratory testing session 3

use stoolap::Database;

fn setup_db() -> Database {
    Database::open_in_memory().expect("Failed to create in-memory database")
}

// =============================================================================
// BUG 35: ORDER BY aggregate not in SELECT list was ignored
// Problem: SELECT cat FROM t GROUP BY cat ORDER BY SUM(val) DESC
//          didn't sort by the aggregate value
// =============================================================================

#[test]
fn test_bugs5_order_by_hidden_aggregate() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t35 (id INTEGER PRIMARY KEY, cat TEXT, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO t35 VALUES (1, 'A', 100), (2, 'A', 200), (3, 'B', 50)",
        (),
    )
    .expect("Failed to insert data");

    // A has sum=300, B has sum=50
    // ORDER BY SUM(val) DESC should give A first, then B
    let mut rows = db
        .query(
            "SELECT cat FROM t35 GROUP BY cat ORDER BY SUM(val) DESC",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "A"); // A has higher sum

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "B"); // B has lower sum
}

#[test]
fn test_bugs5_order_by_hidden_aggregate_asc() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t35b (id INTEGER PRIMARY KEY, cat TEXT, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO t35b VALUES (1, 'X', 10), (2, 'X', 20), (3, 'Y', 100)",
        (),
    )
    .expect("Failed to insert data");

    // X has sum=30, Y has sum=100
    // ORDER BY SUM(val) ASC should give X first, then Y
    let mut rows = db
        .query(
            "SELECT cat FROM t35b GROUP BY cat ORDER BY SUM(val) ASC",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "X"); // X has lower sum

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "Y"); // Y has higher sum
}

#[test]
fn test_bugs5_order_by_count_hidden() {
    let db = setup_db();

    db.execute("CREATE TABLE t35c (id INTEGER PRIMARY KEY, cat TEXT)", ())
        .expect("Failed to create table");
    db.execute(
        "INSERT INTO t35c VALUES (1, 'A'), (2, 'A'), (3, 'A'), (4, 'B'), (5, 'B')",
        (),
    )
    .expect("Failed to insert data");

    // A has count=3, B has count=2
    // ORDER BY COUNT(*) DESC should give A first, then B
    let mut rows = db
        .query(
            "SELECT cat FROM t35c GROUP BY cat ORDER BY COUNT(*) DESC",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "A"); // A has more rows

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "B"); // B has fewer rows
}

#[test]
fn test_bugs5_order_by_avg_hidden() {
    let db = setup_db();

    db.execute(
        "CREATE TABLE t35d (id INTEGER PRIMARY KEY, cat TEXT, val INTEGER)",
        (),
    )
    .expect("Failed to create table");
    db.execute(
        "INSERT INTO t35d VALUES (1, 'P', 10), (2, 'P', 20), (3, 'Q', 100)",
        (),
    )
    .expect("Failed to insert data");

    // P has avg=15, Q has avg=100
    // ORDER BY AVG(val) DESC should give Q first, then P
    let mut rows = db
        .query(
            "SELECT cat FROM t35d GROUP BY cat ORDER BY AVG(val) DESC",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "Q"); // Q has higher avg

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "P"); // P has lower avg
}

// =============================================================================
// BUG 36: RANK/DENSE_RANK returned 1 for all rows with JOIN + GROUP BY + aggregate ORDER BY
// Problem: Pattern matching for aggregate expressions only generated unqualified patterns
// =============================================================================

#[test]
fn test_bugs5_rank_with_join_group_by_aggregate() {
    let db = setup_db();

    db.execute("CREATE TABLE t36a (id INTEGER PRIMARY KEY, grp TEXT)", ())
        .expect("Failed to create table t36a");
    db.execute(
        "CREATE TABLE t36b (id INTEGER PRIMARY KEY, t36a_id INTEGER, val INTEGER)",
        (),
    )
    .expect("Failed to create table t36b");

    db.execute("INSERT INTO t36a VALUES (1, 'A'), (2, 'A'), (3, 'B')", ())
        .expect("Failed to insert into t36a");
    db.execute(
        "INSERT INTO t36b VALUES (1, 1, 100), (2, 2, 50), (3, 3, 75)",
        (),
    )
    .expect("Failed to insert into t36b");

    // A=150, B=75, so RANK should be A=1, B=2
    let mut rows = db
        .query(
            "SELECT t36a.grp, SUM(t36b.val) as total, \
             RANK() OVER (ORDER BY SUM(t36b.val) DESC) as rnk \
             FROM t36a JOIN t36b ON t36a.id = t36b.t36a_id \
             GROUP BY t36a.grp \
             ORDER BY rnk",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "A");
    assert_eq!(row.get::<i64>(1).unwrap(), 150);
    assert_eq!(row.get::<i64>(2).unwrap(), 1);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "B");
    assert_eq!(row.get::<i64>(1).unwrap(), 75);
    assert_eq!(row.get::<i64>(2).unwrap(), 2);
}

#[test]
fn test_bugs5_dense_rank_with_join_group_by_aggregate() {
    let db = setup_db();

    db.execute("CREATE TABLE t36c (id INTEGER PRIMARY KEY, grp TEXT)", ())
        .expect("Failed to create table t36c");
    db.execute(
        "CREATE TABLE t36d (id INTEGER PRIMARY KEY, t36c_id INTEGER, val INTEGER)",
        (),
    )
    .expect("Failed to create table t36d");

    db.execute(
        "INSERT INTO t36c VALUES (1, 'X'), (2, 'X'), (3, 'Y'), (4, 'Z')",
        (),
    )
    .expect("Failed to insert into t36c");
    db.execute(
        "INSERT INTO t36d VALUES (1, 1, 50), (2, 2, 50), (3, 3, 80), (4, 4, 20)",
        (),
    )
    .expect("Failed to insert into t36d");

    // X=100, Y=80, Z=20, so DENSE_RANK should be X=1, Y=2, Z=3
    let mut rows = db
        .query(
            "SELECT t36c.grp, SUM(t36d.val) as total, \
             DENSE_RANK() OVER (ORDER BY SUM(t36d.val) DESC) as dr \
             FROM t36c JOIN t36d ON t36c.id = t36d.t36c_id \
             GROUP BY t36c.grp \
             ORDER BY dr",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "X");
    assert_eq!(row.get::<i64>(1).unwrap(), 100);
    assert_eq!(row.get::<i64>(2).unwrap(), 1);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "Y");
    assert_eq!(row.get::<i64>(1).unwrap(), 80);
    assert_eq!(row.get::<i64>(2).unwrap(), 2);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "Z");
    assert_eq!(row.get::<i64>(1).unwrap(), 20);
    assert_eq!(row.get::<i64>(2).unwrap(), 3);
}

#[test]
fn test_bugs5_row_number_with_join_still_works() {
    let db = setup_db();

    db.execute("CREATE TABLE t36e (id INTEGER PRIMARY KEY, grp TEXT)", ())
        .expect("Failed to create table t36e");
    db.execute(
        "CREATE TABLE t36f (id INTEGER PRIMARY KEY, t36e_id INTEGER, val INTEGER)",
        (),
    )
    .expect("Failed to create table t36f");

    db.execute("INSERT INTO t36e VALUES (1, 'A'), (2, 'B')", ())
        .expect("Failed to insert into t36e");
    db.execute("INSERT INTO t36f VALUES (1, 1, 100), (2, 2, 50)", ())
        .expect("Failed to insert into t36f");

    // ROW_NUMBER should always work
    let mut rows = db
        .query(
            "SELECT t36e.grp, SUM(t36f.val) as total, \
             ROW_NUMBER() OVER (ORDER BY SUM(t36f.val) DESC) as rn \
             FROM t36e JOIN t36f ON t36e.id = t36f.t36e_id \
             GROUP BY t36e.grp \
             ORDER BY rn",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "A");
    assert_eq!(row.get::<i64>(2).unwrap(), 1);

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "B");
    assert_eq!(row.get::<i64>(2).unwrap(), 2);
}

#[test]
fn test_bugs5_rank_with_tied_values() {
    let db = setup_db();

    db.execute("CREATE TABLE t36g (id INTEGER PRIMARY KEY, grp TEXT)", ())
        .expect("Failed to create table t36g");
    db.execute(
        "CREATE TABLE t36h (id INTEGER PRIMARY KEY, t36g_id INTEGER, val INTEGER)",
        (),
    )
    .expect("Failed to create table t36h");

    db.execute("INSERT INTO t36g VALUES (1, 'A'), (2, 'B'), (3, 'C')", ())
        .expect("Failed to insert into t36g");
    db.execute(
        "INSERT INTO t36h VALUES (1, 1, 100), (2, 2, 100), (3, 3, 50)",
        (),
    )
    .expect("Failed to insert into t36h");

    // A=100, B=100 (tied), C=50
    // RANK should be: A=1, B=1 (tied), C=3 (gap after tie)
    let mut rows = db
        .query(
            "SELECT t36g.grp, SUM(t36h.val) as total, \
             RANK() OVER (ORDER BY SUM(t36h.val) DESC) as rnk \
             FROM t36g JOIN t36h ON t36g.id = t36h.t36g_id \
             GROUP BY t36g.grp \
             ORDER BY total DESC, grp",
            (),
        )
        .expect("Query failed");

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(1).unwrap(), 100);
    assert_eq!(row.get::<i64>(2).unwrap(), 1); // First tied value

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(1).unwrap(), 100);
    assert_eq!(row.get::<i64>(2).unwrap(), 1); // Second tied value

    let row = rows.next().unwrap().unwrap();
    assert_eq!(row.get::<i64>(1).unwrap(), 50);
    assert_eq!(row.get::<i64>(2).unwrap(), 3); // Gap after tie
}
