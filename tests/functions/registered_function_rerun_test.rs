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

//! Once a function is registered, a SELECT a truncate overtakes is not run
//! again. The only test of this binary: it registers a function process-wide

#![cfg(feature = "test-failpoints")]

use std::sync::Arc;

use stoolap::functions::global_registry;
use stoolap::test_failpoints as fp;
use stoolap::{Database, Error};

#[test]
fn a_select_after_a_function_is_registered_is_not_run_again() {
    global_registry()
        .register_tvf(
            "MY_SERIES",
            Arc::new(|| Box::new(stoolap::functions::tvf::GenerateSeriesFunction)),
        )
        .unwrap();
    let db = Database::open("memory://registered_function_rerun").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)",
        "INSERT INTO t VALUES (1, 10), (2, 20)",
        "CREATE TABLE other (id INTEGER PRIMARY KEY)",
        "INSERT INTO other VALUES (1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let other = db.clone();
    fp::after_statement_view_taken(move || {
        other.execute("TRUNCATE TABLE other", ()).unwrap();
    });
    assert!(matches!(
        db.query("SELECT v FROM t ORDER BY v", ()),
        Err(Error::ViewOvertaken)
    ));
}
