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

//! Closing a database wakes its background thread instead of waiting for
//! the thread's next step

use std::time::{Duration, Instant};

use stoolap::Database;

#[test]
fn closing_a_database_does_not_wait_for_the_background_step() {
    let fastest = (0..3)
        .map(|i| {
            let db = Database::open(&format!("memory://close_wakes_cleanup_{i}")).unwrap();
            db.query_one::<i64, _>("SELECT 1", ()).unwrap();
            // The background thread is inside its 100 ms step by now
            std::thread::sleep(Duration::from_millis(20));
            let start = Instant::now();
            db.close().unwrap();
            start.elapsed()
        })
        .min()
        .unwrap();
    assert!(
        fastest < Duration::from_millis(50),
        "a close 20 ms after open took {fastest:?}; the background step is 100 ms"
    );
}
