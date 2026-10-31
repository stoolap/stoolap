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

//! An IN list of members of more than one kind, handed to the storage scan,
//! gives sealed rows the answer hot rows get

use stoolap::core::Value;
use stoolap::storage::expression::{Expression, InListExpr};
use stoolap::storage::traits::{Engine, ScanAccess};
use stoolap::Database;

/// The second column of every row the scan of `t` keeps under `s IN list`
fn scan(db: &Database, list: &[Value]) -> Vec<String> {
    let tx = db.engine().begin_transaction().unwrap();
    let table = tx.get_table("t").unwrap();
    let mut expr = InListExpr::new("s", list.to_vec());
    expr.prepare_for_schema(table.schema());
    let plan = table.index_plan(&expr);
    let access = plan.as_ref().map_or(ScanAccess::Seq, ScanAccess::Index);
    let mut scanner = table
        .scan_access(access, &[0, 1], None, Some(&expr))
        .unwrap();
    let mut out = Vec::new();
    while scanner.next() {
        out.push(format!("{:?}", scanner.take_row().get(1).unwrap()));
    }
    assert!(scanner.err().is_none());
    out.sort();
    out
}

#[test]
fn mixed_members_answer_sealed_rows_as_hot_rows() {
    let lists = [
        vec![Value::Integer(2), Value::Integer(10), Value::text("15")],
        vec![Value::Integer(9), Value::text("10")],
    ];
    let tables: [(&str, &[Value]); 4] = [
        (
            "TEXT",
            &[
                Value::text("0"),
                Value::text("2"),
                Value::text("10"),
                Value::text("15"),
                Value::text("z"),
            ],
        ),
        ("TEXT", &[Value::text("10")]),
        ("INTEGER", &[Value::Integer(10)]),
        ("INTEGER", &[Value::Integer(2)]),
    ];
    for (ty, rows) in tables {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
        db.execute(
            &format!("CREATE TABLE t (id INTEGER PRIMARY KEY, s {ty})"),
            (),
        )
        .unwrap();
        for (id, value) in rows.iter().enumerate() {
            db.execute(
                "INSERT INTO t VALUES ($1, $2)",
                vec![Value::Integer(id as i64), value.clone()],
            )
            .unwrap();
        }
        let hot: Vec<Vec<String>> = lists.iter().map(|list| scan(&db, list)).collect();
        db.execute("PRAGMA CHECKPOINT", ()).unwrap();
        for (list, want) in lists.iter().zip(&hot) {
            assert_eq!(&scan(&db, list), want, "{ty} {rows:?} IN {list:?}");
        }
    }
}
