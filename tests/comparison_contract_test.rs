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

use stoolap::{Database, Value};

const T: Option<bool> = Some(true);
const F: Option<bool> = Some(false);
/// UNKNOWN
const U: Option<bool> = None;

fn open(name: &str) -> Database {
    Database::open(&format!("memory://{name}")).unwrap()
}

/// The truth value of a one-cell SELECT
fn truth(db: &Database, sql: &str, params: Vec<Value>) -> Option<bool> {
    let mut rows = db.query(sql, params).unwrap();
    let row = rows.next().expect("one row").unwrap();
    match row.get_value(0) {
        Some(Value::Boolean(b)) => Some(*b),
        Some(Value::Null(_)) => None,
        other => panic!("{sql}: not a truth value: {other:?}"),
    }
}

fn ids(db: &Database, sql: &str, params: Vec<Value>) -> Vec<i64> {
    db.query(sql, params)
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect()
}

/// `number op 'text'` written with literals, with the text as a parameter,
/// with the number as a parameter, and with the operands swapped
fn check_number_text(db: &Database, number: &str, op: &str, text: &str, expected: Option<bool>) {
    let flipped = match op {
        "<" => ">",
        ">" => "<",
        "<=" => ">=",
        ">=" => "<=",
        other => other,
    };
    let as_value = |n: &str| -> Value {
        n.parse::<i64>()
            .map(Value::Integer)
            .unwrap_or_else(|_| Value::Float(n.parse::<f64>().unwrap()))
    };
    let forms: [(String, Vec<Value>); 4] = [
        (format!("SELECT {number} {op} '{text}'"), vec![]),
        (format!("SELECT {number} {op} $1"), vec![Value::text(text)]),
        (format!("SELECT $1 {op} '{text}'"), vec![as_value(number)]),
        (format!("SELECT '{text}' {flipped} {number}"), vec![]),
    ];
    for (sql, params) in forms {
        assert_eq!(
            truth(db, &sql, params.clone()),
            expected,
            "{sql} with {params:?}"
        );
    }
}

#[test]
fn test_number_against_text_reads_the_text_as_a_number() {
    let db = open("contract_number_text");
    for (number, op, text, expected) in [
        ("1", "=", "1", T),
        ("1", "=", "01", T),
        ("1", "=", "+1", T),
        ("1", "=", "1.0", T),
        ("1", "=", "1e0", T),
        ("100", "=", "1E2", T),
        ("1", "=", "1.5", F),
        ("1", "<", "1.5", T),
        ("9", "<", "10", T),
        ("10", ">", "9", T),
        ("5", ">=", "5", T),
        ("1.5", "=", "1.5", T),
        ("-2", "=", "-2", T),
    ] {
        check_number_text(&db, number, op, text, expected);
    }
}

#[test]
fn test_text_that_is_not_a_number_is_unknown() {
    let db = open("contract_not_a_number");
    for text in [
        "abc", "", " 1", "1 ", ".5", "1.", "inf", "NaN", "0x1", "1e400", "1,5",
    ] {
        for op in ["=", "<>", "<", "<=", ">", ">="] {
            check_number_text(&db, "1", op, text, U);
        }
    }
}

#[test]
fn test_both_text_compare_as_text() {
    let db = open("contract_text_text");
    assert_eq!(truth(&db, "SELECT '01' = '1'", vec![]), F);
    assert_eq!(
        truth(
            &db,
            "SELECT $1 = $2",
            vec![Value::text("01"), Value::text("1")]
        ),
        F
    );
    assert_eq!(truth(&db, "SELECT '10' < '9'", vec![]), T);
}

#[test]
fn test_large_integers_compare_exactly() {
    let db = open("contract_large");
    for (number, op, text, expected) in [
        ("9007199254740993", "=", "9007199254740993", T),
        ("9007199254740993", "=", "9007199254740992.0", F),
        ("9007199254740993", ">", "9007199254740992.0", T),
        ("9223372036854775807", "=", "9223372036854775807", T),
        ("9223372036854775807", "=", "9223372036854775808", F),
        ("9223372036854775807", "<", "9223372036854775808", T),
        ("-9223372036854775807", "=", "-9223372036854775807", T),
    ] {
        check_number_text(&db, number, op, text, expected);
    }
}

#[test]
fn test_null_and_the_outer_operators() {
    let db = open("contract_null");
    for (sql, expected) in [
        ("SELECT 1 = NULL", U),
        ("SELECT NOT (1 = 'abc')", U),
        ("SELECT (1 = 'abc') = TRUE", U),
        ("SELECT NOT (1 = '1')", F),
        ("SELECT (1 = '01') = TRUE", T),
    ] {
        assert_eq!(truth(&db, sql, vec![]), expected, "{sql}");
    }
    assert_eq!(
        truth(&db, "SELECT NOT ($1 = 'abc')", vec![Value::Integer(1)]),
        U
    );
}

#[test]
fn test_is_distinct_from() {
    let db = open("contract_distinct");
    for (sql, expected) in [
        ("SELECT 1 IS DISTINCT FROM '1'", F),
        ("SELECT 1 IS NOT DISTINCT FROM '01'", T),
        ("SELECT 1 IS DISTINCT FROM 'abc'", T),
        ("SELECT 1 IS NOT DISTINCT FROM 'abc'", F),
        ("SELECT 'abc' IS DISTINCT FROM 1", T),
        ("SELECT 1 IS DISTINCT FROM NULL", T),
        ("SELECT NULL IS DISTINCT FROM NULL", F),
        ("SELECT NULL IS NOT DISTINCT FROM NULL", T),
        ("SELECT '01' IS DISTINCT FROM '1'", T),
    ] {
        assert_eq!(truth(&db, sql, vec![]), expected, "{sql}");
    }
    assert_eq!(
        truth(
            &db,
            "SELECT $1 IS DISTINCT FROM $2",
            vec![Value::Integer(1), Value::text("01")]
        ),
        F
    );
}

#[test]
fn test_in_and_between_follow_the_comparison() {
    let db = open("contract_in_between");
    for (sql, expected) in [
        ("SELECT 1 IN ('abc', 2)", U),
        ("SELECT 1 NOT IN ('abc', 2)", U),
        ("SELECT 1 IN ('abc', 1)", T),
        ("SELECT 1 IN ('01', 2)", T),
        ("SELECT '01' IN (1, 2)", T),
        ("SELECT '01' IN ('1', 2)", F),
        ("SELECT 1 BETWEEN 'abc' AND 0", F),
        ("SELECT 1 NOT BETWEEN 'abc' AND 0", T),
        ("SELECT 5 BETWEEN '1' AND '10'", T),
        ("SELECT 1 BETWEEN 'abc' AND 2", U),
    ] {
        assert_eq!(truth(&db, sql, vec![]), expected, "{sql}");
    }
    assert_eq!(
        truth(&db, "SELECT 1 IN ($1, 2)", vec![Value::text("1")]),
        T,
        "a parameter in the list"
    );
    assert_eq!(
        truth(&db, "SELECT $1 IN ('abc', 2)", vec![Value::Integer(1)]),
        U,
        "the probe as a parameter"
    );
    let nan = Value::Float(f64::NAN);
    for (sql, params, expected) in [
        ("SELECT $1 NOT IN ('1')", vec![nan.clone()], T),
        ("SELECT $1 IN ('1', '2')", vec![nan.clone()], F),
        (
            "SELECT $1 NOT IN ($2)",
            vec![nan.clone(), Value::text("1")],
            T,
        ),
        ("SELECT '1' IN (SELECT $1)", vec![nan.clone()], F),
        ("SELECT '1' NOT IN ($1)", vec![nan.clone()], T),
        ("SELECT $1 > '1'", vec![nan.clone()], T),
    ] {
        assert_eq!(truth(&db, sql, params), expected, "{sql} with NaN");
    }
}

fn table(name: &str) -> Database {
    let db = open(name);
    db.execute(
        "CREATE TABLE q (id INTEGER PRIMARY KEY, k INTEGER, s TEXT, f FLOAT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO q VALUES (1, 1, '01', 1.0), (2, 9, '9', 9.0), (3, 10, 'abc', 10.0), \
         (4, 9007199254740993, '1', 0.5)",
        (),
    )
    .unwrap();
    db
}

/// Each WHERE in every form: as written, kept in memory, and through an index
fn check_rows(db: &Database, condition: &str, params: Vec<Value>, expected: &[i64]) {
    let forms = [
        format!("SELECT id FROM q WHERE {condition} ORDER BY id"),
        format!("SELECT id FROM q WHERE ({condition}) = TRUE ORDER BY id"),
        format!("SELECT id FROM q WHERE id + 0 > 0 AND ({condition}) ORDER BY id"),
    ];
    for sql in forms {
        assert_eq!(
            ids(db, &sql, params.clone()),
            expected,
            "{sql} with {params:?}"
        );
    }
}

fn check_rows_with_and_without_indexes(db: &Database, cases: &[(&str, Vec<Value>, Vec<i64>)]) {
    for (condition, params, expected) in cases {
        check_rows(db, condition, params.clone(), expected);
    }
    db.execute("CREATE INDEX q_k ON q (k)", ()).unwrap();
    db.execute("CREATE INDEX q_s ON q (s)", ()).unwrap();
    db.execute("CREATE INDEX q_f ON q (f)", ()).unwrap();
    for (condition, params, expected) in cases {
        check_rows(db, condition, params.clone(), expected);
    }
}

#[test]
fn test_columns_follow_the_comparison() {
    let db = table("contract_columns");
    check_rows_with_and_without_indexes(
        &db,
        &[
            ("k = '01'", vec![], vec![1]),
            ("k = $1", vec![Value::text("01")], vec![1]),
            ("k < '1.5'", vec![], vec![1]),
            ("k < $1", vec![Value::text("10")], vec![1, 2]),
            ("k = 'abc'", vec![], vec![]),
            ("k <> 'abc'", vec![], vec![]),
            ("NOT (k = 'abc')", vec![], vec![]),
            ("k = '9007199254740993'", vec![], vec![4]),
            ("k = '9007199254740992.0'", vec![], vec![]),
            ("s = 1", vec![], vec![1, 4]),
            ("s = $1", vec![Value::Integer(9)], vec![2]),
            ("s = '1'", vec![], vec![4]),
            ("s > 5", vec![], vec![2]),
            ("f = '1'", vec![], vec![1]),
            ("f < '.5'", vec![], vec![]),
            ("k IN ('01', 'abc')", vec![], vec![1]),
            ("k IN ($1, 2)", vec![Value::text("9")], vec![2]),
            ("s IN (1, 9)", vec![], vec![1, 2, 4]),
            ("k NOT IN ('abc', 2)", vec![], vec![]),
            ("k BETWEEN '1' AND '9'", vec![], vec![1, 2]),
            (
                "k BETWEEN $1 AND $2",
                vec![Value::text("2"), Value::text("10")],
                vec![2, 3],
            ),
            ("k IS DISTINCT FROM '01'", vec![], vec![2, 3, 4]),
            ("s IS NOT DISTINCT FROM 1", vec![], vec![1, 4]),
            ("ABS(k) = '01'", vec![], vec![1]),
            ("ABS(k) < '9.5'", vec![], vec![1, 2]),
            ("'01' = ABS(k)", vec![], vec![1]),
            ("LENGTH(s) = '2'", vec![], vec![1]),
            ("ABS(k) <> 'abc'", vec![], vec![]),
            ("NOT (ABS(k) <> 'abc')", vec![], vec![]),
            ("NOT (ABS(k) <> 'abc' AND id > 0)", vec![], vec![]),
            ("(ABS(k) = 1) XOR (ABS(k) <> 'abc')", vec![], vec![]),
            ("NOT (ABS(k) = 9)", vec![], vec![1, 3, 4]),
            ("k NOT BETWEEN NULL AND 0", vec![], vec![1, 2, 3, 4]),
            ("k BETWEEN NULL AND 0", vec![], vec![]),
            ("k NOT BETWEEN 0 AND NULL", vec![], vec![]),
            (
                "k NOT BETWEEN $1 AND 0",
                vec![Value::Null(stoolap::DataType::Integer)],
                vec![1, 2, 3, 4],
            ),
        ],
    );
}

#[test]
fn test_a_negated_function_comparison_keeps_a_null_unknown() {
    let db = open("contract_negated_function_null");
    db.execute("CREATE TABLE n (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO n VALUES (1, 1), (2, NULL)", ())
        .unwrap();
    for condition in ["NOT (ABS(k) = 5)", "NOT (ABS(k) = 5 OR id > 5)"] {
        assert_eq!(
            ids(&db, &format!("SELECT id FROM n WHERE {condition}"), vec![]),
            vec![1],
            "{condition}"
        );
    }
}

#[test]
fn test_joins_on_a_number_and_a_text_follow_the_comparison() {
    let db = open("contract_join");
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE b (id INTEGER PRIMARY KEY, s TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO a VALUES (1, 1), (2, 9), (3, 10)", ())
        .unwrap();
    db.execute(
        "INSERT INTO b VALUES (10, '01'), (11, '1'), (12, 'abc'), (13, '10')",
        (),
    )
    .unwrap();
    let pairs = |sql: &str| -> Vec<(i64, i64)> {
        db.query(sql, ())
            .unwrap()
            .map(|row| {
                let row = row.unwrap();
                (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap())
            })
            .collect()
    };
    let expected = vec![(1, 10), (1, 11), (3, 13)];
    for sql in [
        "SELECT a.id, b.id FROM a JOIN b ON a.k = b.s ORDER BY a.id, b.id",
        "SELECT a.id, b.id FROM a JOIN b ON b.s = a.k ORDER BY a.id, b.id",
        "SELECT a.id, b.id FROM a, b WHERE a.k = b.s ORDER BY a.id, b.id",
        "SELECT a.id, b.id FROM a JOIN b ON a.k = b.s AND a.id > 0 ORDER BY a.id, b.id",
    ] {
        assert_eq!(pairs(sql), expected, "{sql}");
    }
    db.execute("CREATE INDEX b_s ON b (s)", ()).unwrap();
    assert_eq!(
        pairs("SELECT a.id, b.id FROM a JOIN b ON a.k = b.s ORDER BY a.id, b.id"),
        expected,
        "with an index on the TEXT side"
    );
    let left: Vec<(i64, i64)> = db
        .query(
            "SELECT a.id, b.id FROM a LEFT JOIN b ON a.k = b.s ORDER BY a.id, b.id",
            (),
        )
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap_or(-1))
        })
        .collect();
    assert_eq!(left, vec![(1, 10), (1, 11), (2, -1), (3, 13)], "LEFT JOIN");
}

/// The two ids of each joined row, sorted, a NULL as -1
fn sorted_pairs(db: &Database, sql: &str) -> Vec<(i64, i64)> {
    let mut pairs: Vec<(i64, i64)> = db
        .query(sql, ())
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            (
                row.get::<i64>(0).unwrap_or(-1),
                row.get::<i64>(1).unwrap_or(-1),
            )
        })
        .collect();
    pairs.sort_unstable();
    pairs
}

fn join_tables(name: &str) -> Database {
    let db = open(name);
    db.execute(
        "CREATE TABLE a (id INTEGER PRIMARY KEY, v TEXT, n INTEGER)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE b (id INTEGER PRIMARY KEY, n INTEGER)", ())
        .unwrap();
    db.execute(
        "INSERT INTO a VALUES (1, '01', 1), (2, '02', 2), (3, '03', 3), (4, 'x', 4)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO b VALUES (1, 1), (2, 2), (3, 3), (5, 5)", ())
        .unwrap();
    db
}

#[test]
fn test_a_join_key_that_turns_text_late_is_compared() {
    let db = join_tables("contract_join_late_text");
    let derived = "(SELECT id, CASE WHEN id = 3 THEN v ELSE id END AS key FROM a)";
    let expected = vec![(1, 1), (2, 2), (3, 3)];
    for sql in [
        format!("SELECT d.id, b.id FROM {derived} d JOIN b ON d.key = b.n"),
        format!("SELECT d.id, b.id FROM {derived} d JOIN b ON d.key = b.n LIMIT 10"),
        format!("SELECT d.id, b.id FROM {derived} d JOIN b ON d.key = b.id"),
        format!("SELECT d.id, b.id FROM {derived} d JOIN b ON d.key = b.id LIMIT 10"),
        format!("SELECT b.id, d.id FROM b JOIN {derived} d ON b.n = d.key LIMIT 10"),
    ] {
        let found = sorted_pairs(&db, &sql);
        let found = if sql.starts_with("SELECT b.id") {
            found.into_iter().map(|(b, d)| (d, b)).collect()
        } else {
            found
        };
        assert_eq!(found, expected, "{sql}");
    }
}

#[test]
fn test_a_composite_join_key_drops_only_the_mixed_pair() {
    let db = join_tables("contract_join_composite");
    for sql in [
        "SELECT a.id, b.id FROM a JOIN b ON a.id = b.id AND a.v = b.n",
        "SELECT a.id, b.id FROM a JOIN b ON a.v = b.n AND a.n = b.n",
        "SELECT a.id, b.id FROM a JOIN b ON a.v = b.n AND a.id = b.id LIMIT 10",
    ] {
        assert_eq!(
            sorted_pairs(&db, sql),
            vec![(1, 1), (2, 2), (3, 3)],
            "{sql}"
        );
    }
}

#[test]
fn test_a_merge_join_is_chosen_again_without_the_mixed_pair() {
    let db = open("contract_join_merge");
    db.execute(
        "CREATE TABLE a (id INTEGER PRIMARY KEY, x INTEGER, y INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE b (id INTEGER PRIMARY KEY, x TEXT, y INTEGER)",
        (),
    )
    .unwrap();
    let rows = 300;
    let a: Vec<String> = (1..=rows)
        .map(|i| format!("({i}, {i}, {})", rows - i))
        .collect();
    let b: Vec<String> = (1..=rows)
        .flat_map(|i| {
            [
                format!("({}, '{i:04}', {})", 2 * i, rows - i),
                format!("({}, '{i:04}x', 1000)", 2 * i + 1),
            ]
        })
        .collect();
    db.execute(&format!("INSERT INTO a VALUES {}", a.join(", ")), ())
        .unwrap();
    db.execute(&format!("INSERT INTO b VALUES {}", b.join(", ")), ())
        .unwrap();
    let count: i64 = db
        .query(
            "SELECT COUNT(*) FROM a JOIN b ON a.x = b.x AND a.y = b.y",
            (),
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(count, rows, "sorted on both keys, not on the one left");
}

#[test]
fn test_outer_joins_on_a_number_and_a_text() {
    let db = join_tables("contract_join_outer");
    assert_eq!(
        sorted_pairs(&db, "SELECT a.id, b.id FROM a LEFT JOIN b ON a.v = b.n"),
        vec![(1, 1), (2, 2), (3, 3), (4, -1)]
    );
    assert_eq!(
        sorted_pairs(&db, "SELECT a.id, b.id FROM a RIGHT JOIN b ON a.v = b.n"),
        vec![(-1, 5), (1, 1), (2, 2), (3, 3)]
    );
    assert_eq!(
        sorted_pairs(&db, "SELECT a.id, b.id FROM a FULL JOIN b ON a.v = b.n"),
        vec![(-1, 5), (1, 1), (2, 2), (3, 3), (4, -1)]
    );
    assert_eq!(
        sorted_pairs(&db, "SELECT a.id, b.id FROM a LEFT JOIN b ON a.v = b.id"),
        vec![(1, 1), (2, 2), (3, 3), (4, -1)],
        "by the primary key"
    );
}

#[test]
fn test_index_joins_on_a_number_and_a_text() {
    let db = join_tables("contract_join_index");
    db.execute("CREATE INDEX b_n ON b (n)", ()).unwrap();
    let expected = vec![(1, 1), (2, 2), (3, 3)];
    for sql in [
        "SELECT a.id, b.id FROM a JOIN b ON a.v = b.n",
        "SELECT a.id, b.id FROM a JOIN b ON a.v = b.n LIMIT 10",
        "SELECT a.id, b.id FROM a JOIN b ON a.v = b.id",
        "SELECT a.id, b.id FROM a JOIN b ON a.v = b.id LIMIT 10",
        "WITH c AS (SELECT id, v FROM a) SELECT c.id, b.id FROM c JOIN b ON c.v = b.id",
        "WITH c AS (SELECT id, v FROM a) SELECT c.id, b.id FROM c JOIN b ON c.v = b.n",
        "WITH c AS (SELECT id, v FROM a), d AS (SELECT * FROM c) \
         SELECT c.id, b.id FROM c JOIN b ON c.v = b.id",
        "WITH c AS (SELECT id, v FROM a), d AS (SELECT * FROM c) \
         SELECT c.id, b.id FROM c JOIN b ON c.v = b.n",
    ] {
        assert_eq!(sorted_pairs(&db, sql), expected, "{sql}");
    }
    db.execute("CREATE INDEX a_v ON a (v)", ()).unwrap();
    for sql in [
        "SELECT a.id, b.id FROM b JOIN a ON b.n = a.v",
        "SELECT a.id, b.id FROM b JOIN a ON b.n = a.v LIMIT 10",
        "SELECT a.id, d.id FROM (SELECT id, n FROM b) d JOIN a ON d.n = a.v",
        "SELECT a.id, d.id FROM (SELECT id, n FROM b) d JOIN a ON d.n = a.v LIMIT 10",
    ] {
        assert_eq!(sorted_pairs(&db, sql), expected, "{sql}");
    }
}

#[test]
fn test_an_inner_filter_on_a_text_key_is_not_moved_to_a_numeric_key() {
    let db = open("contract_join_filter_rewrite");
    db.execute("CREATE TABLE o (id INTEGER PRIMARY KEY, n INTEGER)", ())
        .unwrap();
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO o VALUES (1, 9), (2, 2)", ())
        .unwrap();
    db.execute("INSERT INTO t VALUES (10, '9'), (20, '2')", ())
        .unwrap();
    db.execute("CREATE INDEX t_s ON t (s)", ()).unwrap();
    for sql in [
        "SELECT o.id, t.id FROM o JOIN t ON o.n = t.s WHERE t.s > '10'",
        "SELECT o.id, t.id FROM o JOIN t ON o.n = t.s WHERE t.s > '10' LIMIT 10",
    ] {
        assert_eq!(sorted_pairs(&db, sql), vec![(1, 10), (2, 20)], "{sql}");
    }
}

#[test]
fn test_number_joins_keep_their_answers() {
    let db = join_tables("contract_join_numbers");
    for sql in [
        "SELECT a.id, b.id FROM a JOIN b ON a.n = b.n",
        "SELECT a.id, b.id FROM a JOIN b ON a.n = b.n LIMIT 10",
        "SELECT a.id, b.id FROM a JOIN b ON a.n = b.id LIMIT 10",
    ] {
        assert_eq!(
            sorted_pairs(&db, sql),
            vec![(1, 1), (2, 2), (3, 3)],
            "{sql}"
        );
    }
}

/// An outer table with a TEXT, a NULL and a numeric key, and an inner one
/// whose keys hold a match beside an UNKNOWN member, a NULL and duplicates
/// after reading: '1' and '01' are both 1
fn semi_tables(name: &str) -> Database {
    let db = open(name);
    db.execute(
        "CREATE TABLE o (id INTEGER PRIMARY KEY, v TEXT, n INTEGER)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, s TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO o VALUES (1, '01', 1), (2, 'x', 2), (3, NULL, 3), (4, '4', 4)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (10, 1, '1'), (11, 1, '01'), (12, NULL, 'abc'), (13, 5, NULL)",
        (),
    )
    .unwrap();
    db
}

/// Each query as written and, without an ORDER BY, under a LIMIT and an
/// OFFSET, which the key paths apply themselves
fn check_semi(db: &Database, query: &str, expected: &[i64]) {
    let mut all = ids(db, query, vec![]);
    all.sort_unstable();
    assert_eq!(all, expected, "{query}");
    let mut limited = ids(db, &format!("{query} LIMIT 10"), vec![]);
    limited.sort_unstable();
    assert_eq!(limited, expected, "{query} LIMIT 10");
    let shifted = ids(db, &format!("{query} LIMIT 2 OFFSET 1"), vec![]);
    let want: Vec<i64> = expected.iter().copied().skip(1).take(2).collect();
    assert_eq!(shifted, want, "{query} LIMIT 2 OFFSET 1");
}

#[test]
fn test_exists_on_a_number_and_a_text() {
    let db = semi_tables("contract_semi_exists");
    for (query, expected) in [
        (
            "SELECT id FROM o WHERE EXISTS (SELECT 1 FROM t WHERE t.k = o.v)",
            vec![1],
        ),
        (
            "SELECT id FROM o WHERE NOT EXISTS (SELECT 1 FROM t WHERE t.k = o.v)",
            vec![2, 3, 4],
        ),
        (
            "SELECT id FROM o WHERE EXISTS (SELECT 1 FROM t WHERE t.s = o.id)",
            vec![1],
        ),
        (
            "SELECT id FROM o WHERE NOT EXISTS (SELECT 1 FROM t WHERE t.s = o.id)",
            vec![2, 3, 4],
        ),
        (
            "SELECT id FROM o WHERE NOT EXISTS (SELECT 1 FROM t WHERE t.s = o.n)",
            vec![2, 3, 4],
        ),
        (
            "SELECT id FROM o WHERE NOT EXISTS \
             (SELECT 1 FROM t WHERE t.s = o.id AND t.id > 100)",
            vec![1, 2, 3, 4],
        ),
        (
            "SELECT id FROM o WHERE EXISTS (SELECT 1 FROM t WHERE t.s = o.id AND t.id > 100)",
            vec![],
        ),
        (
            "SELECT id FROM o WHERE NOT EXISTS (SELECT 1 FROM t WHERE t.k = o.id)",
            vec![2, 3, 4],
        ),
    ] {
        check_semi(&db, query, &expected);
    }
    db.execute("CREATE INDEX t_s ON t (s)", ()).unwrap();
    db.execute("CREATE INDEX t_k ON t (k)", ()).unwrap();
    for (query, expected) in [
        (
            "SELECT id FROM o WHERE EXISTS (SELECT 1 FROM t WHERE t.s = o.id)",
            vec![1],
        ),
        (
            "SELECT id FROM o WHERE NOT EXISTS (SELECT 1 FROM t WHERE t.s = o.id)",
            vec![2, 3, 4],
        ),
        (
            "SELECT id FROM o WHERE NOT EXISTS (SELECT 1 FROM t WHERE t.k = o.v)",
            vec![2, 3, 4],
        ),
    ] {
        check_semi(&db, query, &expected);
    }
}

#[test]
fn test_the_inner_where_of_an_anti_join_follows_the_comparison() {
    let db = open("contract_anti_join_where");
    for sql in [
        "CREATE TABLE o (id INTEGER PRIMARY KEY)",
        "CREATE TABLE b (id INTEGER PRIMARY KEY, k INTEGER, s TEXT)",
        "INSERT INTO o VALUES (1), (2)",
        "INSERT INTO b VALUES (1, 1, '01'), (2, 5, 'x')",
    ] {
        db.execute(sql, ()).unwrap();
    }
    for (inner, expected) in [
        ("b.k = '01'", vec![2]),
        ("b.s = 1", vec![2]),
        ("b.k IN ('01', 'abc')", vec![2]),
        ("ABS(b.k) = 5", vec![1]),
        ("b.k = 5", vec![1]),
    ] {
        let sql = format!(
            "SELECT id FROM o WHERE NOT EXISTS (SELECT 1 FROM b WHERE b.id = o.id AND {inner})"
        );
        assert_eq!(ids(&db, &sql, vec![]), expected, "{sql}");
    }
}

#[test]
fn test_a_vector_search_filter_follows_the_comparison() {
    let db = open("contract_vector_filter");
    db.execute(
        "CREATE TABLE v (id INTEGER PRIMARY KEY, k INTEGER, e VECTOR(2))",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO v VALUES (1, 1, '[0.0, 0.0]'), (2, 2, '[1.0, 1.0]')",
        (),
    )
    .unwrap();
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM v WHERE k = '01' ORDER BY VEC_DISTANCE_L2(e, '[0.0, 0.0]') LIMIT 5",
            vec![]
        ),
        vec![1]
    );
}

#[test]
fn test_semi_joins_over_sealed_rows() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(&format!(
        "file://{}?sync_mode=none&checkpoint_on_close=off&checkpoint_interval=0",
        dir.path().display()
    ))
    .unwrap();
    for sql in [
        "CREATE TABLE o (id INTEGER PRIMARY KEY, v TEXT, n INTEGER)",
        "CREATE TABLE t (id INTEGER PRIMARY KEY, k INTEGER, s TEXT)",
        "INSERT INTO o VALUES (1, '01', 1), (2, 'x', 2), (3, NULL, 3), (4, '4', 4)",
        "INSERT INTO t VALUES (10, 1, '1'), (11, 1, '01'), (12, NULL, 'abc'), (13, 5, NULL)",
        "CREATE INDEX o_n ON o (n)",
        "CREATE INDEX t_k ON t (k)",
        "CREATE INDEX t_s ON t (s)",
        "PRAGMA CHECKPOINT",
    ] {
        db.execute(sql, ()).unwrap();
    }
    for (query, expected) in [
        (
            "SELECT id FROM o WHERE EXISTS (SELECT 1 FROM t WHERE t.s = o.id)",
            vec![1],
        ),
        (
            "SELECT id FROM o WHERE NOT EXISTS (SELECT 1 FROM t WHERE t.k = o.v)",
            vec![2, 3, 4],
        ),
        (
            "SELECT id FROM o WHERE NOT EXISTS (SELECT 1 FROM t WHERE t.s = o.n)",
            vec![2, 3, 4],
        ),
        (
            "SELECT id FROM o WHERE n IN (SELECT s FROM t WHERE s <> 'abc')",
            vec![1],
        ),
        (
            "SELECT id FROM o WHERE n NOT IN (SELECT s FROM t WHERE s <> 'abc')",
            vec![2, 3, 4],
        ),
    ] {
        check_semi(&db, query, &expected);
    }
}

#[test]
fn test_in_a_subquery_on_a_number_and_a_text() {
    let db = semi_tables("contract_semi_in");
    for (query, expected) in [
        ("SELECT id FROM o WHERE id IN (SELECT s FROM t)", vec![1]),
        ("SELECT id FROM o WHERE id NOT IN (SELECT s FROM t)", vec![]),
        (
            "SELECT id FROM o WHERE id NOT IN (SELECT s FROM t WHERE s IS NOT NULL)",
            vec![],
        ),
        (
            "SELECT id FROM o WHERE id NOT IN (SELECT s FROM t WHERE s <> 'abc')",
            vec![2, 3, 4],
        ),
        (
            "SELECT id FROM o WHERE id IN (SELECT s FROM t WHERE s <> 'abc')",
            vec![1],
        ),
        ("SELECT id FROM o WHERE v IN (SELECT k FROM t)", vec![1]),
        (
            "SELECT id FROM o WHERE v NOT IN (SELECT k FROM t WHERE k IS NOT NULL)",
            vec![4],
        ),
    ] {
        check_semi(&db, query, &expected);
    }
    db.execute("CREATE INDEX o_v ON o (v)", ()).unwrap();
    for (query, expected) in [
        ("SELECT id FROM o WHERE v IN (SELECT k FROM t)", vec![1]),
        (
            "SELECT id FROM o WHERE v IN (SELECT k FROM t) AND id > 0",
            vec![1],
        ),
    ] {
        check_semi(&db, query, &expected);
    }
}

#[cfg(feature = "test-failpoints")]
#[test]
fn test_an_index_join_on_a_number_and_a_text_runs_each_side_once() {
    use stoolap::test_failpoints::join_side_runs;
    let db = join_tables("contract_join_index_sides");
    db.execute("CREATE INDEX b_n ON b (n)", ()).unwrap();
    let expected = vec![(1, 1), (2, 2), (3, 3)];
    for sql in [
        "SELECT a.id, b.id FROM a JOIN b ON a.v = b.id",
        "SELECT a.id, b.id FROM a JOIN b ON a.v = b.id LIMIT 10",
        "SELECT a.id, b.id FROM a JOIN b ON a.v = b.n",
        "SELECT d.id, b.id FROM (SELECT id, v FROM a) d JOIN b ON d.v = b.id",
        "SELECT d.id, b.id FROM (SELECT id, v FROM a) d JOIN b ON d.v = b.id LIMIT 10",
        "SELECT d.id, b.id FROM (SELECT id, v FROM a) d JOIN b ON d.v = b.n",
    ] {
        let before = join_side_runs();
        assert_eq!(sorted_pairs(&db, sql), expected, "{sql}");
        let runs = join_side_runs() - before;
        assert!(runs <= 2, "a side ran more than once, {runs} runs: {sql}");
    }
    let sql =
        "SELECT x.id, COUNT(y.id) FROM a x LEFT JOIN b y ON x.v = y.id GROUP BY x.id LIMIT 10";
    let before = join_side_runs();
    assert_eq!(
        sorted_pairs(&db, sql),
        vec![(1, 1), (2, 1), (3, 1), (4, 0)],
        "{sql}"
    );
    let runs = join_side_runs() - before;
    assert!(runs <= 2, "a side ran more than once, {runs} runs: {sql}");
}

#[cfg(feature = "test-failpoints")]
#[test]
fn test_an_index_join_answers_either_family_while_it_streams() {
    use stoolap::test_failpoints::index_join_outer_rows;
    let db = open("contract_join_index_streams");
    db.execute(
        "CREATE TABLE big (id INTEGER PRIMARY KEY, n INTEGER, t TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO big SELECT g.value, g.value % 100, CAST(g.value % 100 AS TEXT) \
         FROM generate_series(1, 5000) g",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE small (id INTEGER PRIMARY KEY, s TEXT)", ())
        .unwrap();
    db.execute(
        "INSERT INTO small SELECT g.value, CAST(g.value AS TEXT) FROM generate_series(0, 99) g",
        (),
    )
    .unwrap();
    db.execute("CREATE INDEX small_s ON small (s)", ()).unwrap();
    // The file engine plans a hash join over the secondary TEXT index
    let by_text_index = !cfg!(feature = "test-filedb");
    for (sql, streams) in [
        (
            "SELECT d.id, small.id FROM (SELECT id, n FROM big) d JOIN small ON d.n = small.id LIMIT 10",
            true,
        ),
        (
            "SELECT d.id, small.id FROM (SELECT id, t FROM big) d JOIN small ON d.t = small.id LIMIT 10",
            true,
        ),
        (
            "SELECT d.id, small.id FROM (SELECT id, n FROM big) d JOIN small ON d.n = small.s LIMIT 10",
            by_text_index,
        ),
        (
            "SELECT big.id, small.id FROM big JOIN small ON big.t = small.id LIMIT 10",
            true,
        ),
        (
            "SELECT big.id, small.id FROM big JOIN small ON big.n = small.s LIMIT 10",
            by_text_index,
        ),
    ] {
        let before = index_join_outer_rows();
        let pairs = sorted_pairs(&db, sql);
        let read = index_join_outer_rows() - before;
        assert_eq!(pairs.len(), 10, "{sql}");
        assert!(pairs.iter().all(|(d, s)| d % 100 == *s), "{sql}: {pairs:?}");
        if streams {
            assert!(
                read > 0 && read < 200,
                "{sql}: the index join read {read} outer rows"
            );
        }
    }
}

const CTE_SIDES: &str = "WITH x AS (SELECT id, v, n FROM a), y AS (SELECT id, n FROM b) ";

#[test]
fn test_a_limit_join_of_ctes_on_a_number_and_a_text() {
    let db = join_tables("contract_cte_limit_join");
    for (join, expected) in [
        ("FROM x JOIN y ON x.v = y.n", vec![(1, 1), (2, 2), (3, 3)]),
        ("FROM y JOIN x ON y.n = x.v", vec![(1, 1), (2, 2), (3, 3)]),
        (
            "FROM x LEFT JOIN y ON x.v = y.n",
            vec![(1, 1), (2, 2), (3, 3), (4, -1)],
        ),
        (
            "FROM y LEFT JOIN x ON y.n = x.v",
            vec![(1, 1), (2, 2), (3, 3), (5, -1)],
        ),
        (
            "FROM x JOIN y ON x.v = y.n AND x.id = y.id",
            vec![(1, 1), (2, 2), (3, 3)],
        ),
    ] {
        let first = join.split_whitespace().nth(1).unwrap();
        let second = if first == "x" { "y" } else { "x" };
        let sql = format!("{CTE_SIDES}SELECT {first}.id, {second}.id {join} LIMIT 10");
        assert_eq!(sorted_pairs(&db, &sql), expected, "{sql}");
    }
}

#[cfg(feature = "test-failpoints")]
#[test]
fn test_a_limit_join_of_ctes_streams_over_one_key() {
    use stoolap::test_failpoints::streaming_joins_streamed;
    let db = join_tables("contract_cte_limit_join_streams");
    for (join, streams) in [
        ("FROM x JOIN y ON x.n = y.n", true),
        ("FROM x JOIN y ON x.v = y.n", true),
        ("FROM y JOIN x ON y.n = x.v", true),
        ("FROM x JOIN y ON x.v = y.n AND x.id = y.id", false),
    ] {
        let sql = format!("{CTE_SIDES}SELECT x.id, y.id {join} LIMIT 10");
        let before = streaming_joins_streamed();
        sorted_pairs(&db, &sql);
        assert_eq!(
            streaming_joins_streamed() - before == 1,
            streams,
            "{sql}: streamed {}",
            streaming_joins_streamed() - before
        );
    }
}

#[test]
fn test_a_limit_join_with_the_build_side_on_the_left_reads_each_side_s_key() {
    let db = open("contract_limit_join_build_left");
    db.execute("CREATE TABLE a (id INTEGER PRIMARY KEY, k INTEGER)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE b (id INTEGER PRIMARY KEY, s TEXT, n INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO a VALUES (1, 7)", ()).unwrap();
    // The one row it knows of makes `a` the build side, on the left
    db.execute("ANALYZE a", ()).unwrap();
    db.execute(
        "INSERT INTO b SELECT g.value, CASE WHEN g.value = 1 THEN '07' ELSE 'x' END, \
         CASE WHEN g.value = 1 THEN 7 ELSE 0 END FROM generate_series(1, 300) g",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE c (n INTEGER, id INTEGER PRIMARY KEY)", ())
        .unwrap();
    db.execute("INSERT INTO c VALUES (7, 1), (0, 2)", ())
        .unwrap();
    for sql in [
        "SELECT a.id FROM a JOIN (SELECT s FROM b) x ON a.k = x.s LIMIT 10",
        "SELECT a.id FROM a JOIN (SELECT n FROM b) x ON a.k = x.n LIMIT 10",
        "SELECT a.id FROM a JOIN c ON a.k = c.n LIMIT 10",
    ] {
        assert_eq!(ids(&db, sql, vec![]), vec![1], "{sql}");
    }
}

#[test]
fn test_a_limit_join_of_ctes_with_a_window_function() {
    let db = join_tables("contract_cte_limit_join_window");
    for sql in [
        format!(
            "{CTE_SIDES}SELECT x.id, ROW_NUMBER() OVER (ORDER BY x.id) AS rn \
             FROM x JOIN y ON x.v = y.n LIMIT 10"
        ),
        "SELECT a.id, ROW_NUMBER() OVER (ORDER BY a.id) AS rn \
         FROM a JOIN b ON a.n = b.n LIMIT 10"
            .to_string(),
    ] {
        let rows: Vec<(i64, i64)> = db
            .query(&sql, ())
            .unwrap()
            .map(|row| {
                let row = row.unwrap();
                (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap())
            })
            .collect();
        assert_eq!(rows, vec![(1, 1), (2, 2), (3, 3)], "{sql}");
    }
}

#[test]
fn test_a_join_limit_below_its_matches_counts_every_match_first() {
    let db = join_tables("contract_join_limit_below_matches");
    db.execute("CREATE TABLE d (id INTEGER PRIMARY KEY, g INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO d VALUES (1, 1), (2, 1), (3, 2)", ())
        .unwrap();
    for (sql, expected) in [
        (
            "SELECT COUNT(*) OVER () FROM a JOIN b ON a.n = b.n LIMIT 1",
            vec![3],
        ),
        (
            "SELECT COUNT(*) OVER () FROM a JOIN b ON a.n = b.id LIMIT 1",
            vec![3],
        ),
        (
            "SELECT DISTINCT d.g FROM b JOIN d ON b.n = d.id LIMIT 2",
            vec![1, 2],
        ),
        (
            "SELECT DISTINCT d.g FROM b JOIN d ON b.n = d.id ORDER BY d.g LIMIT 2",
            vec![1, 2],
        ),
        (
            "SELECT DISTINCT d.g FROM b JOIN d ON b.n = d.id ORDER BY d.g LIMIT 1 OFFSET 1",
            vec![2],
        ),
        (
            "WITH x AS (SELECT id, g FROM d), y AS (SELECT n FROM b) \
             SELECT DISTINCT x.g FROM x JOIN y ON x.id = y.n ORDER BY x.g LIMIT 2",
            vec![1, 2],
        ),
        (
            "WITH x AS (SELECT id, n FROM a), y AS (SELECT id, n FROM b) \
             SELECT COUNT(*) OVER () FROM x JOIN y ON x.n = y.n LIMIT 1",
            vec![3],
        ),
        (
            "SELECT DISTINCT d.g FROM d JOIN b ON d.id = b.n ORDER BY 1 LIMIT 2",
            vec![1, 2],
        ),
        (
            "SELECT DISTINCT d.g FROM d JOIN b ON d.id = b.n LIMIT 2",
            vec![1, 2],
        ),
        (
            "WITH x AS (SELECT id, g FROM d), y AS (SELECT n FROM b) \
             SELECT DISTINCT x.g FROM x JOIN y ON x.id = y.n LIMIT 2",
            vec![1, 2],
        ),
    ] {
        let mut found = ids(&db, sql, vec![]);
        found.sort_unstable();
        assert_eq!(found, expected, "{sql}");
    }
}

#[test]
fn test_a_distinct_on_join_orders_by_a_column_it_does_not_select() {
    let db = join_tables("contract_join_distinct_on_hidden_order");
    db.execute("CREATE TABLE d (id INTEGER PRIMARY KEY, g INTEGER)", ())
        .unwrap();
    db.execute("INSERT INTO d VALUES (1, 1), (2, 1), (3, 2)", ())
        .unwrap();
    for (order, expected) in [
        ("ASC", vec![(1, 1), (2, 3)]),
        ("DESC", vec![(1, 2), (2, 3)]),
    ] {
        for limit in ["", " LIMIT 5"] {
            let sql = format!(
                "SELECT DISTINCT ON (d.g) d.g, b.n AS picked FROM b JOIN d ON b.n = d.id \
                 ORDER BY d.g, d.id {order}{limit}"
            );
            assert_eq!(sorted_pairs(&db, &sql), expected, "{sql}");
        }
    }
}
