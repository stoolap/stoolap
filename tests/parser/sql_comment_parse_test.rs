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

//! A comment inside a statement is whitespace: it neither ends the statement
//! nor hides a `;`, and an unterminated block comment is a parse error

use stoolap::Database;

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)", ())
        .unwrap();
    db
}

fn rows(db: &Database) -> Vec<(i64, i64)> {
    db.query("SELECT id, v FROM t ORDER BY id", ())
        .unwrap()
        .map(|r| {
            let r = r.unwrap();
            (r.get(0).unwrap(), r.get(1).unwrap())
        })
        .collect()
}

#[test]
fn a_comment_inside_an_expression_keeps_the_rest_of_it() {
    let db = open("sql_comment_expression");
    for sql in [
        "SELECT 1 -- note\n+ 1",
        "SELECT 1 + -- note\n 1",
        "SELECT 1 /* note */ + 1",
        "SELECT 1 # note\n+ 1",
    ] {
        let got: i64 = db.query_one(sql, ()).unwrap();
        assert_eq!(got, 2, "{sql:?}");
    }
}

#[test]
fn an_update_with_a_comment_before_where_changes_only_its_row() {
    let db = open("sql_comment_update");
    db.execute("UPDATE t SET v = v + 5 -- one row\n WHERE id = 2", ())
        .unwrap();
    db.execute("UPDATE t /* one row */ SET v = 0 WHERE id = 3", ())
        .unwrap();
    assert_eq!(rows(&db), vec![(1, 10), (2, 25), (3, 0)]);
}

#[test]
fn a_delete_with_a_comment_before_where_removes_only_its_row() {
    let db = open("sql_comment_delete");
    db.execute("DELETE FROM t /* only one */ WHERE id = 1", ())
        .unwrap();
    db.execute("DELETE FROM t -- only one\n WHERE id = 3", ())
        .unwrap();
    assert_eq!(rows(&db), vec![(2, 20)]);
}

#[test]
fn a_semicolon_inside_a_comment_does_not_split_the_statement() {
    let db = open("sql_comment_semicolon");
    let got: i64 = db.query_one("SELECT 1 /* ; */ + 1", ()).unwrap();
    assert_eq!(got, 2);
    db.execute("UPDATE t SET v = 7 -- ; DELETE FROM t\n WHERE id = 1", ())
        .unwrap();
    assert_eq!(rows(&db), vec![(1, 7), (2, 20), (3, 30)]);
}

#[test]
fn statements_separated_by_semicolons_all_run() {
    let db = open("sql_comment_program");
    db.execute(
        "/* first */ UPDATE t SET v = 1 WHERE id = 1; -- second\n UPDATE t SET v = 2 WHERE id = 2 /* end */;",
        (),
    )
    .unwrap();
    assert_eq!(rows(&db), vec![(1, 1), (2, 2), (3, 30)]);
}

#[test]
fn parameters_keep_their_order_across_comments() {
    let db = open("sql_comment_parameters");
    let got: i64 = db.query_one("SELECT $1 /* a */ - $2", (10, 3)).unwrap();
    assert_eq!(got, 7);
    db.execute(
        "UPDATE t SET v = ? -- value\n WHERE id = ? /* key */",
        (99, 2),
    )
    .unwrap();
    assert_eq!(rows(&db), vec![(1, 10), (2, 99), (3, 30)]);
}

#[test]
fn an_unterminated_block_comment_fails_before_anything_is_written() {
    let db = open("sql_comment_unterminated");
    assert!(db.execute("UPDATE t SET v = 0 /* no end", ()).is_err());
    assert!(db
        .execute("INSERT INTO t VALUES (4, 40); /* no end", ())
        .is_err());
    assert!(db.query("SELECT 1 /* no end", ()).is_err());
    assert_eq!(rows(&db), vec![(1, 10), (2, 20), (3, 30)]);
}
