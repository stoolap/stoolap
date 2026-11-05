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

//! ORDER BY one indexed column + LIMIT, with no WHERE, answers as the same
//! query on the column without an index: every index form, NULLs in the
//! column, ASC and DESC, default and explicit NULLS FIRST / LAST

use stoolap::Database;

fn column(db: &Database, sql: &str) -> Vec<String> {
    db.query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|r| format!("{:?}", r.unwrap().get_value(0)))
        .collect()
}

const FORMS: [(&str, &str); 4] = [
    ("btree", "CREATE INDEX btree_c ON btree(c) USING BTREE"),
    ("hash", "CREATE INDEX hash_c ON hash(c) USING HASH"),
    ("bitmap", "CREATE INDEX bitmap_c ON bitmap(c) USING BITMAP"),
    ("uniq", "CREATE UNIQUE INDEX uniq_c ON uniq(c)"),
];

/// 2000 rows over `distinct` keys, every 97th NULL; the unique table holds
/// each key once
fn create(db: &Database, ty: &str, distinct: i64, nulls: bool) {
    for table in ["plain", "btree", "hash", "bitmap", "uniq", "plain_uniq"] {
        db.execute(
            &format!("CREATE TABLE {table} (id INTEGER PRIMARY KEY, c {ty})"),
            (),
        )
        .unwrap();
    }
    for (table, index) in FORMS {
        db.execute(index, ())
            .unwrap_or_else(|e| panic!("{table}: {e}"));
    }
    let key = |k: i64| match ty {
        "TEXT" => format!("'k{k:05}'"),
        "FLOAT" => format!("{}.5", k - distinct / 2),
        _ => format!("{}", k - distinct / 2),
    };
    for table in ["plain", "btree", "hash", "bitmap"] {
        let mut stmt = format!("INSERT INTO {table} VALUES ");
        for id in 0..2000i64 {
            if id > 0 {
                stmt.push(',');
            }
            let c = if nulls && id % 97 == 0 {
                "NULL".to_string()
            } else {
                key((id * 7919) % distinct)
            };
            stmt.push_str(&format!("({id}, {c})"));
        }
        db.execute(&stmt, ()).unwrap();
    }
    for table in ["uniq", "plain_uniq"] {
        let mut stmt = format!("INSERT INTO {table} VALUES ");
        for id in 0..2000i64 {
            if id > 0 {
                stmt.push(',');
            }
            let c = if nulls && id % 97 == 0 {
                "NULL".to_string()
            } else {
                key((id * 7919) % 2000)
            };
            stmt.push_str(&format!("({id}, {c})"));
        }
        db.execute(&stmt, ()).unwrap();
    }
}

fn check(db: &Database, ty: &str, wrong: &mut Vec<String>) {
    for order in [
        "ASC",
        "DESC",
        "ASC NULLS FIRST",
        "ASC NULLS LAST",
        "DESC NULLS FIRST",
        "DESC NULLS LAST",
    ] {
        for limit in [5, 40, 3000] {
            let sql = format!("SELECT c FROM {{t}} ORDER BY c {order} LIMIT {limit}");
            for (table, _) in FORMS {
                let twin = if table == "uniq" {
                    "plain_uniq"
                } else {
                    "plain"
                };
                let want = column(db, &sql.replace("{t}", twin));
                let got = column(db, &sql.replace("{t}", table));
                if got != want {
                    wrong.push(format!("{ty} {table}: {}", sql.replace("{t}", table)));
                }
            }
        }
    }
}

#[test]
fn order_by_an_indexed_column_with_nulls_matches_no_index() {
    let mut wrong = Vec::new();
    for ty in ["INTEGER", "TEXT", "FLOAT"] {
        for distinct in [10, 1000] {
            for nulls in [false, true] {
                let db = Database::open(&format!(
                    "memory://order_by_index_nulls_{ty}_{distinct}_{nulls}"
                ))
                .unwrap();
                create(&db, ty, distinct, nulls);
                check(&db, &format!("{ty} D={distinct} nulls={nulls}"), &mut wrong);
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn order_by_an_indexed_column_with_nulls_matches_no_index_hot_and_sealed() {
    let mut wrong = Vec::new();
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    create(&db, "INTEGER", 1000, true);
    check(&db, "hot", &mut wrong);
    db.execute("PRAGMA CHECKPOINT", ()).unwrap();
    check(&db, "sealed", &mut wrong);
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
