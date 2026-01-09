#![no_main]

use libfuzzer_sys::fuzz_target;
use stoolap::api::Database;

fuzz_target!(|data: &[u8]| {
    // Convert bytes to string, skip invalid UTF-8
    if let Ok(sql) = std::str::from_utf8(data) {
        // Skip empty strings
        if sql.trim().is_empty() {
            return;
        }

        // Create a fresh in-memory database for each test
        if let Ok(db) = Database::open_in_memory() {
            // Create a test table that queries can operate on
            let _ = db.execute(
                "CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT, value FLOAT, active BOOLEAN)",
                (),
            );

            // Insert some test data
            let _ = db.execute(
                "INSERT INTO test VALUES (1, 'alice', 1.5, true), (2, 'bob', 2.5, false), (3, 'charlie', 3.5, true)",
                (),
            );

            // Try to execute the fuzzed SQL - should never panic
            let _ = db.execute(sql, ());

            // Also try as a query
            let _ = db.query(sql, ());
        }
    }
});
