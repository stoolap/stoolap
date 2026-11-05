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

/// Set operations whose ORDER BY subquery no prepared plan runs, in a form
/// the compiler would fold away
const DROPPED: &[&str] = &[
    "SELECT DISTINCT ON (1) 1 UNION ALL SELECT 2 ORDER BY (SELECT 1 UNION ALL SELECT 2) IN ()",
    "SELECT DISTINCT ON (1) 1 UNION ALL SELECT 2 ORDER BY (SELECT 1 UNION ALL SELECT 2) NOT IN ()",
];

#[test]
fn a_subquery_no_plan_runs_is_refused_even_where_its_value_is_never_read() {
    let db = Database::open("memory://unplanned_subquery_dropped").unwrap();
    for sql in DROPPED {
        let got = db
            .query(sql, ())
            .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
            .map(|rows| rows.len());
        let error = got.expect_err(sql).to_string();
        assert!(error.contains("outside a prepared plan"), "{sql}: {error}");
    }
}
