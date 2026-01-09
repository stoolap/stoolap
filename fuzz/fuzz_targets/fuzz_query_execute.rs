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
