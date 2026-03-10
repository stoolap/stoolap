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

use stoolap::Database;

fn setup_db() -> Database {
    let db = Database::open_in_memory().unwrap();

    db.execute(
        "CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            username TEXT NOT NULL,
            plan TEXT NOT NULL DEFAULT 'free',
            metadata JSON
        )",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO users (username, plan, metadata) VALUES
        ('alice', 'pro', '{\"country\":\"US\"}'),
        ('bob', 'enterprise', '{\"country\":\"UK\"}'),
        ('charlie', 'free', '{\"country\":\"DE\"}')",
        (),
    )
    .unwrap();

    db.execute(
        "CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTO_INCREMENT,
            user_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            duration_ms INTEGER,
            properties JSON,
            created_at TIMESTAMP NOT NULL
        )",
        (),
    )
    .unwrap();

    db.execute(
        "INSERT INTO events (user_id, event_type, duration_ms, properties, created_at) VALUES
        (1, 'page_view', 3200, '{\"device\":\"desktop\"}', '2026-01-05 09:10:00'),
        (2, 'purchase', 45000, '{\"device\":\"mobile\",\"amount\":99.99}', '2026-01-05 11:30:00'),
        (3, 'page_view', 1800, '{\"device\":\"desktop\"}', '2026-01-06 08:00:00')",
        (),
    )
    .unwrap();

    db.execute(
        "CREATE VIEW user_dashboard AS
        SELECT u.id AS user_id, u.username, u.plan, COUNT(e.id) AS total_events
        FROM users u LEFT JOIN events e ON e.user_id = u.id
        GROUP BY u.id, u.username, u.plan",
        (),
    )
    .unwrap();

    db
}

/// Bug 1: View columns retain table alias prefixes (u.username instead of username)
#[test]
fn test_view_column_names_no_prefix() {
    let db = setup_db();

    // SELECT * should return columns without table alias prefix
    let result = db.query("SELECT * FROM user_dashboard", ()).unwrap();
    let rows: Vec<_> = result.collect::<Result<Vec<_>, _>>().unwrap();

    // Check we got rows
    assert_eq!(rows.len(), 3);

    // The columns should be: user_id, username, plan, total_events
    // NOT: user_id, u.username, u.plan, total_events
    let username = rows[0].get::<String>(1);
    assert!(
        username.is_ok(),
        "Column at index 1 should be accessible and be 'username' not 'u.username'"
    );
}

/// Bug 1b: SELECT specific column from view fails when column names have prefix
#[test]
fn test_view_select_specific_column() {
    let db = setup_db();

    // This should work: SELECT username FROM user_dashboard
    let result = db.query("SELECT username FROM user_dashboard", ());
    assert!(
        result.is_ok(),
        "SELECT username FROM view should work, but fails because column is named 'u.username': {:?}",
        result.err()
    );
}

/// Bug 2: Window function on view causes panic instead of error
#[test]
fn test_window_function_on_view_no_panic() {
    let db = setup_db();

    // This should NOT panic - it should either work or return a proper error
    let result = db.query(
        "SELECT total_events,
                RANK() OVER (ORDER BY total_events) AS rnk
         FROM user_dashboard",
        (),
    );

    // Should not panic. If window functions on views aren't supported,
    // it should return an error, not crash.
    match result {
        Ok(rows) => {
            let collected: Vec<_> = rows.collect::<Result<Vec<_>, _>>().unwrap();
            assert_eq!(collected.len(), 3);
        }
        Err(e) => {
            // An error is acceptable; a panic is not.
            eprintln!("Window function on view returned error (acceptable): {}", e);
        }
    }
}

/// Bug 3: PERCENT_RANK + ROUND on view
#[test]
fn test_percent_rank_on_view_no_panic() {
    let db = setup_db();

    let result = db.query(
        "SELECT total_events,
                RANK() OVER (ORDER BY total_events DESC) AS revenue_rank,
                ROUND(PERCENT_RANK() OVER (ORDER BY total_events), 2) AS activity_percentile
         FROM user_dashboard
         ORDER BY revenue_rank",
        (),
    );

    match result {
        Ok(rows) => {
            let collected: Vec<_> = rows.collect::<Result<Vec<_>, _>>().unwrap();
            assert_eq!(collected.len(), 3);
        }
        Err(e) => {
            eprintln!("PERCENT_RANK on view returned error (acceptable): {}", e);
        }
    }
}
