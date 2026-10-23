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
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, a INTEGER, b INTEGER, v TEXT, \"a'B\" INTEGER, ab INTEGER)",
        "INSERT INTO t VALUES (1, 1, 1, 5, 'B', 3, 3), (2, 1, 2, 1, 'b', 3, 3), (3, 2, 10, 0, 'B', 4, 4)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

fn rows(db: &Database, sql: &str) -> Vec<Vec<String>> {
    db.query(sql, ())
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|row| {
            let row = row.unwrap();
            (0..row.len())
                .map(|i| format!("{:?}", row.get_value(i).unwrap()))
                .collect()
        })
        .collect()
}

fn same(db: &Database, sql: &str, expected: &str, why: &str) {
    assert_eq!(rows(db, sql), rows(db, expected), "{sql}: {why}");
}

#[test]
fn test_aggregates_differing_in_literal_case_keep_two_slots() {
    let db = setup("identity_aggregate_literal_case");
    same(
        &db,
        "SELECT SUM(CASE WHEN v > 'a' THEN 1 ELSE 0 END), SUM(CASE WHEN v > 'A' THEN 1 ELSE 0 END) FROM t",
        "SELECT 1, 3",
        "only 'b' sorts above 'a', all three above 'A'",
    );
    same(
        &db,
        "SELECT MAX(v || 'A'), MAX(v || 'a') AS lo, COALESCE(MAX(v || 'a'), '') FROM t",
        "SELECT 'bA', 'ba', 'ba'",
        "each MAX appends its own letter",
    );
    same(
        &db,
        "SELECT g, MAX(v || 'a'), MAX(v || 'A') FROM t GROUP BY g ORDER BY g",
        "SELECT 1, 'ba', 'bA' UNION ALL SELECT 2, 'Ba', 'BA'",
        "per group, each MAX appends its own letter",
    );
    same(
        &db,
        "SELECT g FROM t GROUP BY g HAVING MAX(v || 'a') = 'ba' AND MAX(v || 'A') = 'bA'",
        "SELECT 1",
        "HAVING reads each aggregate from its own slot",
    );
}

#[test]
fn test_group_keys_differing_in_literal_case_stay_apart() {
    let db = setup("identity_group_literal_case");
    same(
        &db,
        "SELECT v || 'a', v || 'A', COUNT(*) FROM t GROUP BY v || 'a', v || 'A' ORDER BY 2",
        "SELECT 'Ba', 'BA', 2 UNION ALL SELECT 'ba', 'bA', 1",
        "each group key keeps its own letter",
    );
    same(
        &db,
        "SELECT v, v || 'a', COUNT(*) FROM t GROUP BY v, v || 'A' ORDER BY 1",
        "SELECT 'B', 'Ba', 2 UNION ALL SELECT 'b', 'ba', 1",
        "v || 'a' is computed from v, not read from the key v || 'A'",
    );
    same(
        &db,
        "SELECT v, COUNT(*) FROM t GROUP BY v, v || 'A' HAVING v || 'a' = 'Ba'",
        "SELECT 'B', 2",
        "HAVING computes v || 'a' from v",
    );
}

#[test]
fn test_aggregates_with_other_extra_arguments_keep_two_slots() {
    let db = setup("identity_extra_arguments");
    same(
        &db,
        "SELECT g, STRING_AGG(v, ','), STRING_AGG(v, ';') FROM t GROUP BY g ORDER BY g",
        "SELECT 1, 'B,b', 'B;b' UNION ALL SELECT 2, 'B', 'B'",
        "each STRING_AGG joins with its own separator",
    );
}

#[test]
fn test_order_by_an_aggregate_reads_that_aggregate() {
    let db = setup("identity_order_by_aggregate");
    same(
        &db,
        "SELECT g, SUM(b * 2) AS t2 FROM t GROUP BY g ORDER BY SUM(a + 1) DESC",
        "SELECT 2, 0 UNION ALL SELECT 1, 12",
        "SUM(a + 1) is 11 for group 2 and 5 for group 1",
    );
    same(
        &db,
        "SELECT g, SUM(b * 2) AS t2, SUM(a + 1) AS t1 FROM t GROUP BY g ORDER BY SUM(a + 1) DESC",
        "SELECT 2, 0, 11 UNION ALL SELECT 1, 12, 5",
        "the ORDER BY aggregate is the one aliased t1",
    );
    same(
        &db,
        "SELECT g FROM t GROUP BY g ORDER BY MAX(v || 'A') DESC",
        "SELECT 1 UNION ALL SELECT 2",
        "an aggregate over an expression the SELECT list does not hold",
    );
    same(
        &db,
        "SELECT g, MAX(v || 'a') FROM t GROUP BY g ORDER BY MAX(v || 'A') DESC",
        "SELECT 1, 'ba' UNION ALL SELECT 2, 'Ba'",
        "the hidden aggregate differs from the visible one only in literal case",
    );
    same(
        &db,
        "SELECT g FROM t GROUP BY g ORDER BY SUM(a) + 1 DESC",
        "SELECT 2 UNION ALL SELECT 1",
        "an expression over a hidden aggregate",
    );
}

#[test]
fn test_a_window_over_aggregates_reads_the_right_one() {
    let db = setup("identity_window_over_aggregates");
    same(
        &db,
        "SELECT g, RANK() OVER (ORDER BY SUM(CASE WHEN v > 'a' THEN 1 ELSE 0 END) DESC, g), \
         SUM(CASE WHEN v > 'A' THEN 1 ELSE 0 END) FROM t GROUP BY g ORDER BY g",
        "SELECT 1, 1, 2 UNION ALL SELECT 2, 2, 1",
        "the window orders by the aggregate over 'a'",
    );
}

#[test]
fn test_names_with_an_apostrophe_read_as_names_without() {
    let db = setup("identity_apostrophe_names");
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
            "SELECT g, SUM(a) AS \"x'Y\" FROM t GROUP BY g ORDER BY \"x'Y\" DESC",
            "SELECT g, SUM(a) AS xy FROM t GROUP BY g ORDER BY xy DESC",
        ),
        (
            "SELECT t.\"a'B\", SUM(a + 0) FROM t GROUP BY t.\"a'B\" ORDER BY 1",
            "SELECT t.ab, SUM(a + 0) FROM t GROUP BY t.ab ORDER BY 1",
        ),
        (
            "SELECT \"a'b\" + 1, COUNT(*) FROM t GROUP BY \"a'B\" + 1 ORDER BY 1",
            "SELECT ab + 1, COUNT(*) FROM t GROUP BY AB + 1 ORDER BY 1",
        ),
        (
            "SELECT \"a'B\" + 1, COUNT(*) FROM t GROUP BY \"a'B\" + 1 HAVING \"a'b\" + 1 > 0 ORDER BY 1",
            "SELECT ab + 1, COUNT(*) FROM t GROUP BY ab + 1 HAVING AB + 1 > 0 ORDER BY 1",
        ),
        (
            "SELECT g, SUM(\"a'B\") FROM t GROUP BY g ORDER BY g",
            "SELECT g, SUM(ab) FROM t GROUP BY g ORDER BY g",
        ),
    ] {
        assert_eq!(
            rows(&db, quoted),
            rows(&db, plain),
            "{quoted}: a name with an apostrophe reads as one without"
        );
    }
}

#[test]
fn test_identifiers_match_in_any_case() {
    let db = setup("identity_identifier_case");
    same(
        &db,
        "SELECT g, COUNT(V), count(v), SUM(A) FROM t GROUP BY G ORDER BY SUM(a) DESC",
        "SELECT g, COUNT(v), COUNT(v), SUM(a) FROM t GROUP BY g ORDER BY SUM(a) DESC",
        "identifiers match whatever their case",
    );
    same(
        &db,
        "SELECT G + 1, COUNT(*) FROM t GROUP BY g + 1 HAVING SUM(A) > 3 ORDER BY 1",
        "SELECT 3, 1",
        "a group expression and HAVING match in any case",
    );
}

#[test]
fn test_dml_reads_each_aggregate_by_identity() {
    let db = setup("identity_dml");
    db.execute(
        "CREATE TABLE out (g INTEGER PRIMARY KEY, lo TEXT, hi TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO out SELECT g, MAX(v || 'a'), MAX(v || 'A') FROM t GROUP BY g",
        (),
    )
    .unwrap();
    same(
        &db,
        "SELECT g, lo, hi FROM out ORDER BY g",
        "SELECT 1, 'ba', 'bA' UNION ALL SELECT 2, 'Ba', 'BA'",
        "INSERT ... SELECT keeps each aggregate's own slot",
    );
    db.execute(
        "UPDATE out SET lo = (SELECT MAX(v || 'A') FROM t WHERE t.g = out.g) \
         WHERE g IN (SELECT g FROM t GROUP BY g HAVING MAX(v || 'a') = 'ba')",
        (),
    )
    .unwrap();
    same(
        &db,
        "SELECT g, lo FROM out ORDER BY g",
        "SELECT 1, 'bA' UNION ALL SELECT 2, 'Ba'",
        "UPDATE reads the HAVING aggregate over 'a' and sets the one over 'A'",
    );
    db.execute(
        "DELETE FROM out WHERE g IN (SELECT g FROM t GROUP BY g HAVING MAX(v || 'A') = 'BA')",
        (),
    )
    .unwrap();
    same(
        &db,
        "SELECT g FROM out ORDER BY g",
        "SELECT 1",
        "DELETE removes only the group whose MAX over 'A' is 'BA'",
    );
}

#[test]
fn test_order_by_an_expression_reads_that_expression() {
    let db = setup("identity_order_by_expression");
    same(
        &db,
        "SELECT id, REPLACE(v, 'B', 'z') FROM t ORDER BY REPLACE(v, 'b', 'z'), id",
        "SELECT 1, 'z' UNION ALL SELECT 3, 'z' UNION ALL SELECT 2, 'b'",
        "the sort key replaces 'b', not the selected column's 'B'",
    );
    same(
        &db,
        "SELECT t.id, REPLACE(t.v, 'B', 'z') FROM t JOIN t u ON t.id = u.id ORDER BY REPLACE(t.v, 'b', 'z'), t.id",
        "SELECT 1, 'z' UNION ALL SELECT 3, 'z' UNION ALL SELECT 2, 'b'",
        "the join sorts by its own key too",
    );
    same(
        &db,
        "SELECT DISTINCT ON (REPLACE(v, 'b', 'z')) id, REPLACE(v, 'B', 'z') FROM t ORDER BY REPLACE(v, 'b', 'z'), id",
        "SELECT 1, 'z' UNION ALL SELECT 2, 'b'",
        "DISTINCT ON keys on the expression it names",
    );
}

#[test]
fn test_a_sort_key_over_a_union_reads_each_branch() {
    let db = setup("identity_union_sort");
    same(
        &db,
        "SELECT 'a' AS kind, id FROM t WHERE g = 1 UNION ALL SELECT 'b', id FROM t WHERE g = 2 \
         ORDER BY CASE WHEN kind = 'a' THEN 1 ELSE 0 END, id",
        "SELECT 'b', 3 UNION ALL SELECT 'a', 1 UNION ALL SELECT 'a', 2",
        "the literal in the sort key is a constant, not the kind column",
    );
    same(
        &db,
        "SELECT 0 AS src, id FROM t WHERE g = 1 UNION ALL SELECT 5, id FROM t WHERE g = 2 \
         ORDER BY id + 0 DESC",
        "SELECT 5, 3 UNION ALL SELECT 0, 2 UNION ALL SELECT 0, 1",
        "0 in the sort key is a constant, not the src column",
    );
}

#[test]
fn test_a_name_in_a_sort_expression_reads_the_table_column_first() {
    let db = setup("identity_sort_output_name");
    same(
        &db,
        "SELECT a AS x, b AS a FROM t ORDER BY x + a, x",
        "SELECT 1, 5 UNION ALL SELECT 2, 1 UNION ALL SELECT 10, 0",
        "a in the sort key is the table's a; x, which no table has, is the alias",
    );
}

fn pairs(result: stoolap::Result<stoolap::Rows>) -> Result<Vec<(i64, i64)>, String> {
    let mut out = Vec::new();
    for row in result.map_err(|e| e.to_string())? {
        let row = row.map_err(|e| e.to_string())?;
        out.push((row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap()));
    }
    Ok(out)
}

#[test]
fn test_a_parameter_is_one_expression_by_its_slot() {
    let db = setup("identity_parameter_slots");
    let grouped = Ok(vec![(11, 2), (12, 1)]);
    assert_eq!(
        pairs(db.query(
            "SELECT g + $1, COUNT(*) FROM t GROUP BY g + $1 ORDER BY 1",
            (10,)
        )),
        grouped,
        "g + $1 is the group key"
    );
    assert_eq!(
        pairs(db.query_named(
            "SELECT g + :n, COUNT(*) FROM t GROUP BY g + :n ORDER BY 1",
            stoolap::named_params! { n: 10 },
        )),
        grouped,
        "a name written twice is one slot"
    );
    let two = pairs(db.query(
        "SELECT g + ?, COUNT(*) FROM t GROUP BY g + ? ORDER BY 1",
        (10, 10),
    ));
    assert!(
        two.as_ref().is_err_and(|e| e.contains("'g'")),
        "two ? are two expressions, so g is not grouped: {two:?}"
    );
}

#[test]
fn test_two_anonymous_parameters_are_two_expressions() {
    let rows_of = |db: &Database, sql: &str, params: (i64, i64)| -> Vec<Vec<String>> {
        db.query(sql, params)
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
            .map(|row| {
                let row = row.unwrap();
                (0..row.len())
                    .map(|i| format!("{:?}", row.get_value(i).unwrap()))
                    .collect()
            })
            .collect()
    };
    let cases: &[(&str, (i64, i64), &str)] = &[
        (
            "SELECT g + ? AS x, g + ? AS y, COUNT(*) FROM t GROUP BY 1, 2 ORDER BY 1, 2",
            (1, 2),
            "SELECT g + 1 AS x, g + 2 AS y, COUNT(*) FROM t GROUP BY 1, 2 ORDER BY 1, 2",
        ),
        (
            "SELECT g + ? AS x, g + ? AS y, COUNT(*) FROM t GROUP BY 1, 2 HAVING y > 3 ORDER BY 1",
            (1, 2),
            "SELECT g + 1 AS x, g + 2 AS y, COUNT(*) FROM t GROUP BY 1, 2 HAVING y > 3 ORDER BY 1",
        ),
        (
            "SELECT SUM(a + ?), SUM(a + ?) FROM t",
            (1, 2),
            "SELECT SUM(a + 1), SUM(a + 2) FROM t",
        ),
        (
            "SELECT g, SUM(a * ?), SUM(a * ?) FROM t GROUP BY g ORDER BY g",
            (1, 2),
            "SELECT g, SUM(a * 1), SUM(a * 2) FROM t GROUP BY g ORDER BY g",
        ),
        (
            "SELECT g * ? AS x, a FROM t ORDER BY g * ?, a",
            (1, -1),
            "SELECT g * 1 AS x, a FROM t ORDER BY g * -1, a",
        ),
        (
            "SELECT g * ?, a FROM t ORDER BY g * ?, a",
            (1, -1),
            "SELECT g * 1, a FROM t ORDER BY g * -1, a",
        ),
        (
            "SELECT a FROM t ORDER BY g * ?, g * ?, a DESC",
            (0, -1),
            "SELECT a FROM t ORDER BY g * 0, g * -1, a DESC",
        ),
        (
            "SELECT a, SUM(a) OVER (PARTITION BY g * ?), SUM(a) OVER (PARTITION BY g * ?) \
             FROM t ORDER BY a",
            (1, 0),
            "SELECT a, SUM(a) OVER (PARTITION BY g * 1), SUM(a) OVER (PARTITION BY g * 0) \
             FROM t ORDER BY a",
        ),
        (
            "SELECT g + (SELECT ?) AS x, g + (SELECT ?) AS y, COUNT(*) FROM t GROUP BY 1, 2 \
             HAVING y > (SELECT 3) ORDER BY 1",
            (1, 2),
            "SELECT g + 1 AS x, g + 2 AS y, COUNT(*) FROM t GROUP BY 1, 2 HAVING y > 3 ORDER BY 1",
        ),
    ];
    for sealed in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let db = match sealed {
            true => {
                let db = Database::open(&format!("file://{}", dir.path().display())).unwrap();
                for sql in [
                    "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, a INTEGER)",
                    "INSERT INTO t VALUES (1, 1, 1), (2, 1, 2), (3, 2, 10)",
                    "PRAGMA CHECKPOINT",
                ] {
                    db.execute(sql, ()).unwrap();
                }
                db
            }
            false => setup("identity_two_anonymous_parameters"),
        };
        for (sql, params, literal) in cases {
            assert_eq!(
                rows_of(&db, sql, *params),
                rows(&db, literal),
                "sealed {sealed}: {sql} with {params:?}"
            );
        }
    }
}

#[test]
fn test_a_window_over_a_derived_aggregate_column() {
    let db = setup("identity_window_derived");
    same(
        &db,
        "SELECT g, RANK() OVER (ORDER BY SUM(a)) FROM (SELECT g, SUM(a) FROM t GROUP BY g) s ORDER BY g",
        "SELECT 1, 1 UNION ALL SELECT 2, 2",
        "SUM(a) names the derived column",
    );
}

#[test]
fn test_an_aggregate_in_having_and_order_by_only() {
    let db = setup("identity_having_order_by_only");
    same(
        &db,
        "SELECT g FROM t GROUP BY g HAVING SUM(a) > 0 ORDER BY SUM(a) DESC",
        "SELECT 2 UNION ALL SELECT 1",
        "the ORDER BY aggregate computed for HAVING is carried to the sort",
    );
}

#[test]
fn test_distinct_on_keys_differing_in_literal_case_stay_apart() {
    let db = setup("identity_distinct_on_case");
    db.execute("CREATE TABLE s (id INTEGER PRIMARY KEY, v TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO s VALUES (1, 'a'), (2, 'A'), (3, 'b')", ())
        .unwrap();
    same(
        &db,
        "SELECT DISTINCT ON (v = 'A', v = 'a') id FROM s ORDER BY v = 'A', v = 'a', id",
        "SELECT 3 UNION ALL SELECT 1 UNION ALL SELECT 2",
        "each row has its own pair of keys",
    );
}

#[test]
fn test_a_window_argument_over_a_derived_aggregate_column() {
    let db = setup("identity_window_derived_argument");
    same(
        &db,
        "SELECT g, FIRST_VALUE(SUM(a)) OVER (ORDER BY g) FROM (SELECT g, SUM(a) FROM t GROUP BY g) s ORDER BY g",
        "SELECT g, FIRST_VALUE(x) OVER (ORDER BY g) FROM (SELECT g, SUM(a) AS x FROM t GROUP BY g) s ORDER BY g",
        "SUM(a) as an argument names the derived column",
    );
    same(
        &db,
        "SELECT g, RANK() OVER (ORDER BY SUM(a + 1)) FROM (SELECT g, SUM(a + 1) FROM t GROUP BY g) s ORDER BY g",
        "SELECT g, RANK() OVER (ORDER BY x) FROM (SELECT g, SUM(a + 1) AS x FROM t GROUP BY g) s ORDER BY g",
        "an aggregate over an expression names the derived column too",
    );
}

#[test]
fn test_every_alias_of_a_shared_aggregate_names_it() {
    let db = setup("identity_shared_aggregate_aliases");
    same(
        &db,
        "SELECT SUM(a) FILTER (WHERE a > 0) AS x, SUM(a) FILTER (WHERE a > 0) AS y FROM t HAVING y > 0",
        "SELECT 13, 13",
        "y names the same aggregate as x",
    );
    same(
        &db,
        "SELECT g, SUM(a) AS x, SUM(a) AS y FROM t GROUP BY g HAVING y > 5 ORDER BY g",
        "SELECT g, SUM(a), SUM(a) FROM t GROUP BY g HAVING SUM(a) > 5 ORDER BY g",
        "an alias of a deduplicated aggregate reaches HAVING",
    );
}

#[test]
fn test_a_constant_sort_key_over_a_union() {
    let db = setup("identity_union_constant_key");
    same(
        &db,
        "SELECT 'a', 2 AS id FROM t WHERE id = 1 UNION ALL SELECT 'b', 1 FROM t WHERE id = 1 ORDER BY 'a', id",
        "SELECT 'b', 1 UNION ALL SELECT 'a', 2",
        "the first key is a constant, so id orders",
    );
}

#[test]
fn test_an_aggregate_sort_key_over_a_union() {
    let db = setup("identity_union_aggregate_key");
    same(
        &db,
        "SELECT SUM(a) AS s FROM t WHERE id = 1 UNION ALL SELECT SUM(a) FROM t WHERE id = 3 ORDER BY SUM(a) DESC",
        "SELECT SUM(a) FROM t WHERE id = 3 UNION ALL SELECT SUM(a) FROM t WHERE id = 1",
        "SUM(a) names the first column of every branch",
    );
}

#[test]
fn test_a_bare_alias_of_a_shared_aggregate_in_select() {
    let db = setup("identity_bare_shared_alias");
    same(
        &db,
        "SELECT SUM(a) AS x, SUM(a) AS y, y FROM t",
        "SELECT 13, 13, 13",
        "y names the aggregate x shares",
    );
}

#[test]
fn test_a_derived_aggregate_over_a_name_holding_an_apostrophe() {
    let db = setup("identity_derived_apostrophe");
    same(
        &db,
        "SELECT g, RANK() OVER (ORDER BY SUM(\"a'B\")) FROM (SELECT g, SUM(\"a'B\") FROM t GROUP BY g) s ORDER BY g",
        "SELECT g, RANK() OVER (ORDER BY x) FROM (SELECT g, SUM(\"a'B\") AS x FROM t GROUP BY g) s ORDER BY g",
        "the apostrophe belongs to a column name, not a literal",
    );
    same(
        &db,
        "SELECT g, FIRST_VALUE(SUM(\"a'B\")) OVER (ORDER BY g) FROM (SELECT g, SUM(\"a'B\") FROM t GROUP BY g) s ORDER BY g",
        "SELECT g, FIRST_VALUE(x) OVER (ORDER BY g) FROM (SELECT g, SUM(\"a'B\") AS x FROM t GROUP BY g) s ORDER BY g",
        "a window argument reads it too",
    );
}

#[test]
fn test_a_constant_call_sort_key_over_a_union() {
    let db = setup("identity_union_constant_call");
    same(
        &db,
        "SELECT ABS(1), 2 AS id FROM t WHERE id = 1 UNION ALL SELECT ABS(2), 1 FROM t WHERE id = 1 ORDER BY ABS(1), id",
        "SELECT 2, 1 UNION ALL SELECT 1, 2",
        "ABS(1) is a constant, so id orders",
    );
}

#[test]
fn test_a_derived_aggregate_over_a_numeric_case() {
    let db = setup("identity_derived_case");
    same(
        &db,
        "SELECT g, RANK() OVER (ORDER BY SUM(CASE WHEN a > 0 THEN a ELSE 0 END)) \
         FROM (SELECT g, SUM(CASE WHEN a > 0 THEN a ELSE 0 END) FROM t GROUP BY g) s ORDER BY g",
        "SELECT g, RANK() OVER (ORDER BY x) \
         FROM (SELECT g, SUM(CASE WHEN a > 0 THEN a ELSE 0 END) AS x FROM t GROUP BY g) s ORDER BY g",
        "a CASE without a TEXT literal names the derived column",
    );
}

#[test]
fn test_a_derived_aggregate_over_a_parameter() {
    let db = setup("identity_derived_parameter");
    let ranks = |sql: &str| -> Vec<(i64, i64)> {
        db.query(sql, (1,))
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
            .map(|row| {
                let row = row.unwrap();
                (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap())
            })
            .collect()
    };
    assert_eq!(
        ranks(
            "SELECT g, RANK() OVER (ORDER BY SUM(a + $1)) FROM (SELECT g, SUM(a + $1) FROM t GROUP BY g) s ORDER BY g"
        ),
        ranks(
            "SELECT g, RANK() OVER (ORDER BY x) FROM (SELECT g, SUM(a + $1) AS x FROM t GROUP BY g) s ORDER BY g"
        ),
        "a parameter is no TEXT literal in the derived column's name"
    );
}
