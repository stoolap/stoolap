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

#[test]
fn test_a_left_value_of_another_type_reads_each_member_by_the_contract() {
    let db = setup("quantified_timestamp_left");
    members(&db, "INTEGER", &["9", "10"]);
    let left = "CAST('2025-01-01' AS TIMESTAMP)";
    for (op, quantifier, joined) in [
        ("<", "ANY", "OR"),
        ("<", "ALL", "AND"),
        (">", "ANY", "OR"),
        (">", "ALL", "AND"),
    ] {
        let quantified = format!("{left} {op} {quantifier} (SELECT v FROM m)");
        let one_by_one = format!("({left} {op} 9) {joined} ({left} {op} 10)");
        assert_eq!(
            tri(&db, &quantified),
            tri(&db, &one_by_one),
            "{quantified} answers as {one_by_one}"
        );
    }
}

#[test]
fn test_a_large_text_set_compares_without_a_deep_expression() {
    let db = setup("quantified_large_text");
    db.execute("CREATE TABLE m (id INTEGER PRIMARY KEY, v TEXT)", ())
        .unwrap();
    db.execute(
        "INSERT INTO m SELECT value, CAST(value AS TEXT) FROM generate_series(1, 50000)",
        (),
    )
    .unwrap();
    assert_eq!(tri(&db, "0 < ANY (SELECT v FROM m)"), Some(true));
    assert_eq!(tri(&db, "0 > ALL (SELECT v FROM m)"), Some(false));
    assert_eq!(tri(&db, "50001 > ALL (SELECT v FROM m)"), Some(true));
}

#[cfg(feature = "test-failpoints")]
#[test]
fn test_the_left_value_is_evaluated_once_per_row() {
    let db = setup("quantified_volatile_left");
    // RANDOM() is never below '0', so the second member is reached too
    members(&db, "TEXT", &["'0'", "'2'"]);
    let draws = stoolap::test_failpoints::random_draws();
    let found = db
        .query(
            "SELECT id FROM one WHERE RANDOM() < ANY (SELECT v FROM m)",
            (),
        )
        .unwrap()
        .count();
    assert_eq!(found, 1, "RANDOM() is below '2'");
    assert_eq!(
        stoolap::test_failpoints::random_draws() - draws,
        1,
        "one draw for the one row, compared with every member"
    );
}
