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

#[test]
fn failed_statement_restores_prior_cold_validation_generation() {
    for failed_statement in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&format!(
            "file://{}?checkpoint_interval=3600",
            dir.path().display()
        ))
        .unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, u INTEGER UNIQUE)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO t VALUES (1, 1)", ()).unwrap();
        let mut first = db.begin().unwrap();
        first.execute("INSERT INTO t VALUES (4, 100)", ()).unwrap();
        db.execute("INSERT INTO t VALUES (5, 100)", ()).unwrap();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        if failed_statement {
            assert!(first
                .execute("INSERT INTO t VALUES (6, 600), (1, 1)", ())
                .is_err());
        } else {
            first.execute("INSERT INTO t VALUES (6, 600)", ()).unwrap();
        }
        assert!(first.commit().is_err(), "failed statement advanced seal generation for prior unvalidated writes, allowing duplicate cold UNIQUE value");
        assert_eq!(
            db.query_one::<i64, _>("SELECT COUNT(*) FROM t WHERE u = 100", ())
                .unwrap(),
            1
        );
    }
}
