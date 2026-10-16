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

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute("CREATE TABLE one (id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO one VALUES (1)", ()).unwrap();
    db
}

/// The members of table `m`, one column `v` of `kind`
fn members(db: &Database, kind: &str, values: &[&str]) {
    db.execute("DROP TABLE IF EXISTS m", ()).unwrap();
    db.execute(
        &format!("CREATE TABLE m (id INTEGER PRIMARY KEY, v {kind})"),
        (),
    )
    .unwrap();
    for (i, v) in values.iter().enumerate() {
        db.execute(&format!("INSERT INTO m VALUES ({i}, {v})"), ())
            .unwrap();
    }
}

/// Whether the one row passes `predicate`, its negation, or neither
fn tri(db: &Database, predicate: &str) -> Option<bool> {
    let passes = |p: String| {
        db.query(&format!("SELECT id FROM one WHERE {p}"), ())
            .unwrap()
            .count()
            == 1
    };
    let truth = passes(predicate.to_string());
    let negated = passes(format!("NOT ({predicate})"));
    assert!(
        !(truth && negated),
        "{predicate} and its negation both pass"
    );
    if truth {
        Some(true)
    } else if negated {
        Some(false)
    } else {
        None
    }
}

#[test]
fn test_numeric_text_members_compare_as_numbers() {
    let db = setup("quantified_numeric_text");
    members(&db, "TEXT", &["'9'", "'10'"]);
    for (predicate, expected) in [
        ("9 < ANY (SELECT v FROM m)", Some(true)),
        ("9 <= ANY (SELECT v FROM m)", Some(true)),
        ("10 > ANY (SELECT v FROM m)", Some(true)),
        ("9 >= ALL (SELECT v FROM m)", Some(false)),
        ("9 > ALL (SELECT v FROM m)", Some(false)),
        ("11 > ALL (SELECT v FROM m)", Some(true)),
        ("9 <= ALL (SELECT v FROM m)", Some(true)),
        ("10 < ALL (SELECT v FROM m)", Some(false)),
    ] {
        assert_eq!(tri(&db, predicate), expected, "{predicate} over '9', '10'");
    }
}

#[test]
fn test_text_against_text_members_keeps_text_order() {
    let db = setup("quantified_text_text");
    members(&db, "TEXT", &["'9'", "'10'"]);
    for (predicate, expected) in [
        ("'9' < ANY (SELECT v FROM m)", Some(false)),
        ("'9' >= ALL (SELECT v FROM m)", Some(true)),
        ("'0' < ALL (SELECT v FROM m)", Some(true)),
    ] {
        assert_eq!(
            tri(&db, predicate),
            expected,
            "{predicate}: '10' sorts before '9'"
        );
    }
}

#[test]
fn test_unknown_members_decide_only_when_nothing_else_does() {
    let db = setup("quantified_unknown");
    members(&db, "TEXT", &["'9'", "'10'", "NULL"]);
    for (predicate, expected) in [
        ("9 < ANY (SELECT v FROM m)", Some(true)),
        ("9 > ALL (SELECT v FROM m)", Some(false)),
        ("20 > ALL (SELECT v FROM m)", None),
        ("20 < ANY (SELECT v FROM m)", None),
    ] {
        assert_eq!(
            tri(&db, predicate),
            expected,
            "{predicate} over '9', '10', NULL"
        );
    }
    members(&db, "TEXT", &["'abc'", "'10'"]);
    for (predicate, expected) in [
        ("9 < ANY (SELECT v FROM m)", Some(true)),
        ("9 > ALL (SELECT v FROM m)", Some(false)),
        ("20 > ALL (SELECT v FROM m)", None),
        ("20 < ANY (SELECT v FROM m)", None),
    ] {
        assert_eq!(
            tri(&db, predicate),
            expected,
            "{predicate} over 'abc', '10'"
        );
    }
}

#[test]
fn test_numeric_members_keep_their_answers() {
    let db = setup("quantified_numeric");
    members(&db, "INTEGER", &["9", "2"]);
    for (predicate, expected) in [
        ("'10' > ALL (SELECT v FROM m)", Some(true)),
        ("'5' < ANY (SELECT v FROM m)", Some(true)),
        ("1 > ANY (SELECT v FROM m)", Some(false)),
        ("'abc' > ALL (SELECT v FROM m)", None),
    ] {
        assert_eq!(tri(&db, predicate), expected, "{predicate} over 9, 2");
    }
    members(&db, "FLOAT", &["9007199254740992.0", "1.5"]);
    for (predicate, expected) in [
        ("9007199254740993 > ALL (SELECT v FROM m)", Some(true)),
        ("9007199254740992 > ALL (SELECT v FROM m)", Some(false)),
        ("2 < ANY (SELECT v FROM m)", Some(true)),
    ] {
        assert_eq!(tri(&db, predicate), expected, "{predicate} over 2^53, 1.5");
    }
}
