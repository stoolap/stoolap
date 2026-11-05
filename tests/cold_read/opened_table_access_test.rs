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

//! A statement reads the table it opened by that table's own indexes: a
//! plan whose indexes were chosen for another table of the same name, made
//! after the open, is not asked of the opened one

#![cfg(feature = "test-failpoints")]

use std::cell::Cell;
use std::rc::Rc;

use stoolap::Database;

/// The ids `sql` answers, or its error
fn ids(result: stoolap::Result<stoolap::Rows>) -> Result<Vec<i64>, String> {
    let rows = result.map_err(|e| e.to_string())?;
    rows.map(|row| {
        row.map(|row| row.get::<i64>(0).unwrap())
            .map_err(|e| e.to_string())
    })
    .collect()
}

#[test]
fn a_select_reads_the_table_it_opened_by_its_own_indexes() {
    let sql = "SELECT id FROM t WHERE k = 7";
    for run in ["unprepared", "prepared", "prepared again"] {
        let db = Database::open(&format!("memory://opened_table_access_{run}")).unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
            (),
        )
        .unwrap();
        db.execute("CREATE INDEX t_k ON t(k)", ()).unwrap();
        db.execute("INSERT INTO t VALUES (1, 7, 0)", ()).unwrap();
        let statement = db.prepare(sql).unwrap();
        if run == "prepared again" {
            assert_eq!(ids(statement.query(())), Ok(vec![1]));
        }
        let other = db.clone();
        let fired = Rc::new(Cell::new(false));
        let hook_fired = Rc::clone(&fired);
        stoolap::test_failpoints::after_select_table_opened(move || {
            for sql in [
                "ALTER TABLE t RENAME TO old_t",
                "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, v INTEGER)",
                "CREATE INDEX t_k2 ON t(k)",
                "INSERT INTO t VALUES (2, 7, 0)",
            ] {
                other.execute(sql, ()).unwrap();
            }
            hook_fired.set(true);
        });
        let answer = match run {
            "unprepared" => ids(db.query(sql, ())),
            _ => ids(statement.query(())),
        };
        assert!(
            fired.get(),
            "{run}: the rename ran between the open and the read"
        );
        // The opened table's row, the new table's, or a refusal; never no row
        assert!(
            matches!(answer.as_deref(), Ok([1]) | Ok([2]) | Err(_)),
            "{run}: {answer:?}"
        );
    }
}
