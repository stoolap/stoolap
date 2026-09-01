use stoolap::Database;
fn main() {
    let db = Database::open("memory://gbtrace").unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1,7),(2,1),(3,7),(4,5)", ())
        .unwrap();
    let _ = db
        .query("SELECT SUM(id) FROM t GROUP BY v ORDER BY v", ())
        .unwrap()
        .collect_vec()
        .unwrap();
    println!("--- query A done");
    let _ = db
        .query(
            "SELECT COUNT(*) FROM t GROUP BY v ORDER BY SUM(id) DESC",
            (),
        )
        .unwrap()
        .collect_vec()
        .unwrap();
    println!("--- query with hidden agg done");
}
