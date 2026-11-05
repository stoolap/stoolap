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

//! A read whose rows a side file's walk names does not search the volume's
//! TEXT dictionary for its filter; the plain scan still does, and a walk
//! refused its reader falls back to the scan with the same rows

#![cfg(feature = "test-failpoints")]

use std::sync::Mutex;

use stoolap::core::Value;
use stoolap::test_failpoints::{dictionary_searches, refuse_side_walks, rows_materialized};
use stoolap::Database;

/// The counters and the refusal are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

const ROWS: i64 = 4000;

fn open(dir: &std::path::Path) -> Database {
    Database::open(&format!(
        "file://{}?checkpoint_interval=3600",
        dir.display()
    ))
    .unwrap()
}

/// `t` with a TEXT B-tree on s and a composite on (a, s), `u` without; s
/// holds four rows per value, and `note` keeps two of them
fn create(db: &Database) {
    for table in ["t", "u"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, a INTEGER, s TEXT, note TEXT)"),
            (),
        )
        .unwrap();
    }
    db.execute("CREATE INDEX t_s ON t(s) USING BTREE", ())
        .unwrap();
    db.execute("CREATE INDEX t_as ON t(a, s)", ()).unwrap();
    db.execute("BEGIN", ()).unwrap();
    for table in ["t", "u"] {
        let insert = format!("INSERT INTO {table} VALUES ($1, $2, $3, $4)");
        for id in 0..ROWS {
            let note = if id / 1000 % 2 == 0 { "keep" } else { "drop" };
            db.execute(
                &insert,
                vec![
                    Value::Integer(id),
                    Value::Integer(id % 1000 % 7),
                    Value::text(format!("s{:04}", id % 1000)),
                    Value::text(note),
                ],
            )
            .unwrap();
        }
    }
    db.execute("COMMIT", ()).unwrap();
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
}

fn ids(db: &Database, sql: &str, params: &[Value]) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .query(sql, params.to_vec())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    out.sort_unstable();
    out
}

fn stat(db: &Database, name: &str) -> i64 {
    let rows = db.query("PRAGMA INDEX_READ_STATS", ()).unwrap();
    let at = rows.columns().iter().position(|c| c == name).unwrap();
    let row = rows.into_iter().next().unwrap().unwrap();
    row.get::<i64>(at).unwrap()
}

/// The equalities a side file serves, each with a TEXT condition outside
/// the index that keeps two of the four candidates, plain and limited
fn served_cases() -> Vec<(String, Vec<Value>)> {
    let s = || Value::text("s0123");
    let keep = || Value::text("keep");
    let mut cases = Vec::new();
    for limit in ["", " LIMIT 100"] {
        cases.push((
            format!("SELECT id FROM t WHERE s = $1 AND note = $2{limit}"),
            vec![s(), keep()],
        ));
        cases.push((
            format!("SELECT id FROM t WHERE a = $1 AND s = $2 AND note = $3{limit}"),
            vec![Value::Integer(123 % 7), s(), keep()],
        ));
    }
    cases
}

fn check(db: &Database, phase: &str) {
    for (sql, params) in served_cases() {
        let want = ids(db, &sql.replace(" t ", " u "), &params);
        assert_eq!(want.len(), 2, "{sql}");
        let (probes, searches) = (stat(db, "probes"), dictionary_searches());
        let built = rows_materialized();
        assert_eq!(ids(db, &sql, &params), want, "{phase}: {sql}");
        assert!(
            rows_materialized() - built <= want.len(),
            "{phase}: a candidate the TEXT condition drops is not built: {sql}"
        );
        assert!(
            stat(db, "probes") > probes,
            "{phase}: side file read: {sql}"
        );
        assert_eq!(
            dictionary_searches(),
            searches,
            "{phase}: no dictionary search under a walk: {sql}"
        );
        // Refused its reader once the plan and the filter are set, the
        // walk leaves the volume to the scan
        let refused = stat(db, "refused");
        refuse_side_walks(1);
        let got = ids(db, &sql, &params);
        refuse_side_walks(0);
        assert_eq!(got, want, "{phase}: refused walk: {sql}");
        assert!(
            stat(db, "refused") > refused,
            "{phase}: walk refused: {sql}"
        );
    }
    check_collected(db, phase);
    let plain = "SELECT id FROM t WHERE note = $1 AND id < $2";
    let params = [Value::text("keep"), Value::Integer(100)];
    let searches = dictionary_searches();
    assert_eq!(
        ids(db, plain, &params),
        ids(db, &plain.replace(" t ", " u "), &params)
    );
    assert!(
        dictionary_searches() > searches,
        "{phase}: the plain scan searches the dictionary"
    );
}

/// The same equalities through a table's whole collect, not its scanner
fn check_collected(db: &Database, phase: &str) {
    use stoolap::core::Operator;
    use stoolap::storage::expression::{AndExpr, ComparisonExpr, Expression};
    use stoolap::storage::traits::Engine;
    let text = |column: &str, value: &str| -> Box<dyn Expression> {
        Box::new(ComparisonExpr::new(
            column,
            Operator::Eq,
            Value::text(value),
        ))
    };
    let lead: Box<dyn Expression> = Box::new(ComparisonExpr::new(
        "a",
        Operator::Eq,
        Value::Integer(123 % 7),
    ));
    for filter in [
        vec![text("s", "s0123"), text("note", "keep")],
        vec![lead, text("s", "s0123"), text("note", "keep")],
    ] {
        let tx = db.engine().begin_transaction().unwrap();
        let table = tx.get_table("t").unwrap();
        let mut filter = AndExpr::new(filter);
        filter.prepare_for_schema(table.schema());
        let collect = || -> Vec<i64> {
            let mut ids: Vec<i64> = table
                .collect_all_rows(Some(&filter))
                .unwrap()
                .iter()
                .map(|(id, _)| *id)
                .collect();
            ids.sort_unstable();
            ids
        };
        let (probes, searches) = (stat(db, "probes"), dictionary_searches());
        assert_eq!(collect(), [123, 2123], "{phase}: collected");
        assert!(
            stat(db, "probes") > probes,
            "{phase}: collected through the side file"
        );
        assert_eq!(
            dictionary_searches(),
            searches,
            "{phase}: collected without a search"
        );
        let refused = stat(db, "refused");
        refuse_side_walks(1);
        let got = collect();
        refuse_side_walks(0);
        assert_eq!(got, [123, 2123], "{phase}: collected after a refused walk");
        assert!(
            stat(db, "refused") > refused,
            "{phase}: collect walk refused"
        );
    }
}

#[test]
fn a_walk_skips_the_dictionary_after_the_seal_and_after_a_reopen() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    create(&db);
    check(&db, "sealed");
    drop(db);
    let db = open(dir.path());
    check(&db, "reopened");
}
