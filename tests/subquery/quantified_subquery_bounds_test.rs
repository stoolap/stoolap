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

#[test]
fn test_a_join_filter_on_the_right_table_reads_the_right_table() {
    let db = Database::open("memory://quantified_join_side").unwrap();
    for sql in [
        "CREATE TABLE a (id INTEGER PRIMARY KEY, v INTEGER)",
        "CREATE TABLE b (id INTEGER PRIMARY KEY, v INTEGER)",
        "CREATE TABLE m (id INTEGER PRIMARY KEY, v INTEGER)",
        "INSERT INTO a VALUES (1, 0)",
        "INSERT INTO b VALUES (1, 10)",
        "INSERT INTO m VALUES (1, 5)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    for quantifier in ["ANY", "ALL"] {
        let found: Vec<i64> = db
            .query(
                &format!(
                    "SELECT a.id FROM a JOIN b ON a.id = b.id WHERE b.v > {quantifier} (SELECT v FROM m)"
                ),
                (),
            )
            .unwrap()
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect();
        assert_eq!(
            found,
            vec![1],
            "b.v is 10, above 5, whatever a.v holds ({quantifier})"
        );
    }
}

#[cfg(feature = "test-failpoints")]
#[test]
fn test_a_numeric_text_left_value_compares_with_the_bound_once() {
    use stoolap::test_failpoints::quantified_member_comparisons;
    let db = setup("quantified_numeric_text_left");
    db.execute(
        "CREATE TABLE k (id INTEGER PRIMARY KEY, s TEXT, ts TIMESTAMP)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO k VALUES (1, '1001', '2025-01-01 00:00:00'), (2, '5', '2025-01-01 00:00:00'), (3, 'abc', '2025-01-01 00:00:00')",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE m (id INTEGER PRIMARY KEY, v INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO m SELECT value, value FROM generate_series(1, 1000)",
        (),
    )
    .unwrap();
    let before = quantified_member_comparisons();
    let found: Vec<i64> = db
        .query(
            "SELECT id FROM k WHERE s < ANY (SELECT v FROM m) ORDER BY id",
            (),
        )
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert_eq!(
        found,
        vec![2],
        "'5' is below 1000, '1001' is not, 'abc' is UNKNOWN"
    );
    assert_eq!(
        quantified_member_comparisons() - before,
        0,
        "a TEXT read as a number takes the bound, not every member"
    );
    let before = quantified_member_comparisons();
    db.query("SELECT id FROM k WHERE ts < ANY (SELECT v FROM m)", ())
        .unwrap()
        .for_each(drop);
    assert!(
        quantified_member_comparisons() > before,
        "a TIMESTAMP is not a number: it is compared with the members"
    );
}

fn rows(db: &Database, sql: &str) -> Vec<Vec<String>> {
    db.query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| format!("{:?}", row.get_value(i).unwrap()))
                .collect()
        })
        .collect()
}

#[test]
fn test_an_aggregate_on_the_left_is_computed() {
    let db = Database::open("memory://quantified_aggregate_left").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, v INTEGER)",
        "CREATE TABLE m (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO t VALUES (1, 1, 2, 10), (2, 2, 1, 20), (3, 0, 0, 30)",
        "INSERT INTO m VALUES (1, 5)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    assert_eq!(
        rows(
            &db,
            "SELECT COUNT(*), SUM(v) > ANY (SELECT x FROM m), SUM(v) < ALL (SELECT x FROM m) FROM t"
        ),
        rows(&db, "SELECT 3, TRUE, FALSE"),
        "SUM(v) is 60, above 5"
    );
    assert_eq!(
        rows(
            &db,
            "SELECT a + b AS g, COUNT(*), a + b > ANY (SELECT x FROM m) FROM t GROUP BY a + b ORDER BY g"
        ),
        rows(&db, "SELECT 0, 1, FALSE UNION ALL SELECT 3, 2, FALSE"),
        "the group key a + b is rebound inside the comparison"
    );
}

#[test]
fn test_a_nan_left_value_answers_alike_on_join_and_single_table() {
    let db = Database::open("memory://quantified_nan_left").unwrap();
    for sql in [
        "CREATE TABLE a (id INTEGER PRIMARY KEY, v FLOAT)",
        "CREATE TABLE b (id INTEGER PRIMARY KEY)",
        "CREATE TABLE m (id INTEGER PRIMARY KEY, v FLOAT)",
        "INSERT INTO a VALUES (1, CAST('NaN' AS FLOAT))",
        "INSERT INTO b VALUES (1)",
        "INSERT INTO m VALUES (1, 1.0)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    for (predicate, literal, passes) in [
        ("a.v > ANY (SELECT v FROM m)", "a.v > 1.0", true),
        ("a.v >= ALL (SELECT v FROM m)", "a.v >= 1.0", true),
        ("a.v < ALL (SELECT v FROM m)", "a.v < 1.0", false),
        ("a.v <= ANY (SELECT v FROM m)", "a.v <= 1.0", false),
        ("a.v = ALL (SELECT v FROM m)", "a.v = 1.0", false),
        ("a.v <> ANY (SELECT v FROM m)", "a.v <> 1.0", true),
        (
            "NOT (a.v > ANY (SELECT v FROM m))",
            "NOT (a.v > 1.0)",
            false,
        ),
    ] {
        let expected = if passes {
            vec![vec!["Integer(1)".to_string()]]
        } else {
            vec![]
        };
        for sql in [
            format!("SELECT a.id FROM a WHERE {predicate}"),
            format!("SELECT a.id FROM a JOIN b ON a.id = b.id WHERE {predicate}"),
            format!("SELECT a.id FROM a WHERE {literal}"),
        ] {
            assert_eq!(
                rows(&db, &sql),
                expected,
                "{sql}: NaN is above 1.0, as against the literal 1.0"
            );
        }
    }
}

#[test]
fn test_two_sets_keep_two_aggregates() {
    let db = Database::open("memory://quantified_aggregate_identity").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, s TEXT)",
        "CREATE TABLE m1 (id INTEGER PRIMARY KEY, x INTEGER)",
        "CREATE TABLE m2 (id INTEGER PRIMARY KEY, x INTEGER)",
        "CREATE TABLE mt (id INTEGER PRIMARY KEY, x TEXT)",
        "INSERT INTO t VALUES (1, 10, '10')",
        "INSERT INTO m1 VALUES (1, 5)",
        "INSERT INTO m2 VALUES (1, 15)",
        "INSERT INTO mt VALUES (1, '5')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    assert_eq!(
        rows(
            &db,
            "SELECT SUM(CASE WHEN v > ANY (SELECT x FROM m1) THEN 1 ELSE 0 END), \
             SUM(CASE WHEN v > ANY (SELECT x FROM m2) THEN 1 ELSE 0 END) FROM t"
        ),
        rows(&db, "SELECT 1, 0"),
        "10 is above 5 and not above 15"
    );
    assert_eq!(
        rows(
            &db,
            "SELECT SUM(CASE WHEN s > ANY (SELECT x FROM m1) THEN 1 ELSE 0 END), \
             SUM(CASE WHEN s > ANY (SELECT x FROM mt) THEN 1 ELSE 0 END) FROM t"
        ),
        rows(&db, "SELECT 1, 0"),
        "'10' reads as 10 against 5, and sorts below '5' as text"
    );
}

#[cfg(feature = "test-failpoints")]
#[test]
fn test_a_volatile_left_value_is_not_frozen_by_dml_pushdown() {
    let db = Database::open("memory://quantified_volatile_dml").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, active BOOLEAN, hit INTEGER)",
        "CREATE TABLE m (id INTEGER PRIMARY KEY, v FLOAT)",
        "INSERT INTO t SELECT value, TRUE, 0 FROM generate_series(1, 100)",
        "INSERT INTO m VALUES (1, 0.5)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let draws = stoolap::test_failpoints::random_draws();
    db.execute(
        "UPDATE t SET hit = 1 WHERE active = NOT (RANDOM() < ANY (SELECT v FROM m))",
        (),
    )
    .unwrap();
    assert!(
        stoolap::test_failpoints::random_draws() - draws >= 100,
        "RANDOM() is drawn for each of the 100 rows, not once for the statement"
    );
}

#[test]
fn test_a_parameter_on_the_left_keeps_a_join_side_out_of_the_cache() {
    let db = Database::open("memory://quantified_parameter_join").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, a INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, tid INTEGER)",
        "CREATE TABLE w (id INTEGER PRIMARY KEY, v INTEGER)",
        "INSERT INTO t VALUES (1, 10, 1), (2, 20, 2)",
        "INSERT INTO u VALUES (1, 1), (2, 2)",
        "INSERT INTO w VALUES (1, 5)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    assert_eq!(rows(&db, "SELECT * FROM t WHERE x > 5").len(), 2);
    let found: Vec<i64> = db
        .query(
            "SELECT t.id FROM t JOIN u ON t.id = u.tid WHERE t.x > 5 AND t.a + $1 > ANY (SELECT v FROM w) ORDER BY t.id",
            (10,),
        )
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert_eq!(found, vec![1, 2], "a + 10 is above 5 on both rows");
}

#[test]
fn test_a_window_on_the_left_beside_another_window_is_computed() {
    let db = Database::open("memory://quantified_window_left").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
        "CREATE TABLE m (id INTEGER PRIMARY KEY, x INTEGER)",
        "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30)",
        "INSERT INTO m VALUES (1, 1)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    assert_eq!(
        rows(
            &db,
            "SELECT a, ROW_NUMBER() OVER (ORDER BY a) AS rn, \
             ROW_NUMBER() OVER (ORDER BY a) > ANY (SELECT x FROM m) FROM t ORDER BY a"
        ),
        rows(
            &db,
            "SELECT 10, 1, FALSE UNION ALL SELECT 20, 2, TRUE UNION ALL SELECT 30, 3, TRUE"
        ),
        "the row number above 1 passes"
    );
}

#[test]
fn test_sets_differing_in_letter_case_keep_two_aggregates() {
    let db = Database::open("memory://quantified_aggregate_case").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, v TEXT)",
        "CREATE TABLE m1 (id INTEGER PRIMARY KEY, x TEXT)",
        "CREATE TABLE m2 (id INTEGER PRIMARY KEY, x TEXT)",
        "INSERT INTO t VALUES (1, 1, 'B')",
        "INSERT INTO m1 VALUES (1, 'a'), (2, 'B')",
        "INSERT INTO m2 VALUES (1, 'A'), (2, 'b')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    assert_eq!(
        rows(
            &db,
            "SELECT SUM(CASE WHEN v > ANY (SELECT x FROM m1 ORDER BY id) THEN 1 ELSE 0 END), \
             SUM(CASE WHEN v > ANY (SELECT x FROM m2 ORDER BY id) THEN 1 ELSE 0 END) FROM t"
        ),
        rows(&db, "SELECT 0, 1"),
        "'B' is above neither 'a' nor 'B', and above 'A'"
    );
    assert_eq!(
        rows(
            &db,
            "SELECT g, SUM(CASE WHEN v > ANY (SELECT x FROM m1 ORDER BY id) THEN 1 ELSE 0 END) + 0, \
             SUM(CASE WHEN v > ANY (SELECT x FROM m2 ORDER BY id) THEN 1 ELSE 0 END) + 0 FROM t GROUP BY g"
        ),
        rows(&db, "SELECT 1, 0, 1"),
        "an aggregate inside an expression reads its own slot"
    );
    assert_eq!(
        rows(&db, "SELECT MAX(v || 'a'), MAX(v || 'A') FROM t"),
        rows(&db, "SELECT 'Ba', 'BA'"),
        "a TEXT literal keeps its case in the aggregate identity"
    );
    assert_eq!(
        rows(
            &db,
            "SELECT g FROM t GROUP BY g HAVING MAX(v || 'a') = 'Ba' AND MAX(v || 'A') = 'BA'"
        ),
        rows(&db, "SELECT 1"),
        "HAVING reads each aggregate from its own slot"
    );
    db.execute(
        "CREATE TABLE u (id INTEGER PRIMARY KEY, g INTEGER, v TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO u VALUES (1, 1, 'B'), (2, 2, 'b')", ())
        .unwrap();
    assert_eq!(
        rows(
            &db,
            "SELECT g, SUM(CASE WHEN v > 'A' THEN 1 ELSE 0 END), SUM(CASE WHEN v > 'a' THEN 1 ELSE 0 END) \
             FROM u GROUP BY g ORDER BY SUM(CASE WHEN v > 'a' THEN 1 ELSE 0 END) DESC, g"
        ),
        rows(&db, "SELECT 2, 1, 1 UNION ALL SELECT 1, 1, 0"),
        "ORDER BY reads the aggregate over 'a', which only 'b' passes"
    );
    assert_eq!(
        rows(&db, "SELECT g, COUNT(V), count(v) FROM t GROUP BY G"),
        rows(&db, "SELECT 1, 1, 1"),
        "identifiers still match in any case"
    );
}

#[test]
fn test_a_name_holding_an_apostrophe_still_reads_its_aggregate() {
    let db = Database::open("memory://quantified_apostrophe_names").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, v INTEGER, \"a'B\" INTEGER, ab INTEGER)",
        "INSERT INTO t VALUES (1, 1, 7, 3, 3), (2, 2, 9, 4, 4)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    for (quoted, plain) in [
        (
            "SELECT g, SUM(v) AS \"x'Y\" FROM t GROUP BY g ORDER BY g",
            "SELECT g, SUM(v) AS xy FROM t GROUP BY g ORDER BY g",
        ),
        (
            "SELECT g + 1 AS \"x'Y\", COUNT(*) FROM t GROUP BY g + 1 ORDER BY 1",
            "SELECT g + 1 AS xy, COUNT(*) FROM t GROUP BY g + 1 ORDER BY 1",
        ),
        (
            "SELECT g, SUM(v) AS \"x'Y\" FROM t GROUP BY g ORDER BY \"x'Y\" DESC",
            "SELECT g, SUM(v) AS xy FROM t GROUP BY g ORDER BY xy DESC",
        ),
        (
            "SELECT g, SUM(\"a'B\") FROM t GROUP BY g ORDER BY g",
            "SELECT g, SUM(ab) FROM t GROUP BY g ORDER BY g",
        ),
        (
            "SELECT \"a'B\", COUNT(*) FROM t GROUP BY \"a'B\" ORDER BY 1",
            "SELECT ab, COUNT(*) FROM t GROUP BY ab ORDER BY 1",
        ),
    ] {
        assert_eq!(
            rows(&db, quoted),
            rows(&db, plain),
            "{quoted}: a name with an apostrophe reads as one without"
        );
    }
    db.execute("CREATE TABLE s (id INTEGER PRIMARY KEY, v TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO s VALUES (1, 'B')", ()).unwrap();
    assert_eq!(
        rows(
            &db,
            "SELECT MAX(v || 'A'), MAX(v || 'a') AS lo, COALESCE(MAX(v || 'a'), '') FROM s"
        ),
        rows(&db, "SELECT 'BA', 'Ba', 'Ba'"),
        "the aggregate over 'a' is found by its own case, not through the one over 'A'"
    );
    for (quoted, plain) in [
        (
            "SELECT t.\"a'B\", SUM(v + 0) FROM t GROUP BY t.\"a'B\" ORDER BY 1",
            "SELECT t.ab, SUM(v + 0) FROM t GROUP BY t.ab ORDER BY 1",
        ),
        (
            "SELECT \"a'b\" + 1, COUNT(*) FROM t GROUP BY \"a'B\" + 1 ORDER BY 1",
            "SELECT ab + 1, COUNT(*) FROM t GROUP BY AB + 1 ORDER BY 1",
        ),
        (
            "SELECT \"a'B\" + 1, COUNT(*) FROM t GROUP BY \"a'B\" + 1 HAVING \"a'b\" + 1 > 0 ORDER BY 1",
            "SELECT ab + 1, COUNT(*) FROM t GROUP BY ab + 1 HAVING AB + 1 > 0 ORDER BY 1",
        ),
    ] {
        assert_eq!(
            rows(&db, quoted),
            rows(&db, plain),
            "{quoted}: a grouped name with an apostrophe reads in any case"
        );
    }
}
