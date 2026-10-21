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

//! Every parameter is a slot: `$n` and `?` by position, each distinct
//! `:name` by its first appearance. A slot without a value is an error on
//! every path, never NULL

use stoolap::{named_params, Database, NamedParams, Value};

fn open(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    db.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER, s TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO t VALUES (1, 10, 'a'), (2, 20, 'b'), (3, 30, 'c')",
        (),
    )
    .unwrap();
    db
}

fn rows(result: stoolap::Result<stoolap::Rows>) -> stoolap::Result<Vec<Vec<Value>>> {
    let mut out = Vec::new();
    for row in result? {
        let row = row?;
        out.push(
            (0..row.len())
                .map(|i| row.get_value(i).unwrap().clone())
                .collect(),
        );
    }
    Ok(out)
}

/// The answer of `sql` with `params` on each named path: ad hoc twice (the
/// first call parses, the second takes the cached plan), prepared, prepared
/// in a transaction, ad hoc in a transaction and by plan
fn named_paths(
    db: &Database,
    sql: &str,
    params: impl Fn() -> NamedParams,
) -> Vec<(&'static str, stoolap::Result<Vec<Vec<Value>>>)> {
    // A session of its own has an empty plan cache
    let db = &db.clone();
    let mut out = vec![
        ("ad hoc, parsed", rows(db.query_named(sql, params()))),
        ("ad hoc, cached", rows(db.query_named(sql, params()))),
    ];
    match db.prepare(sql) {
        Ok(stmt) => {
            out.push(("prepared", rows(stmt.query_named(params()))));
            let mut tx = db.begin().unwrap();
            out.push((
                "prepared in tx",
                rows(stmt.query_named_in(&mut tx, params())),
            ));
            out.push(("ad hoc in tx", rows(tx.query_named(sql, params()))));
            tx.rollback().unwrap();
        }
        Err(e) => out.push(("prepared", Err(e))),
    }
    if let Ok(plan) = db.cached_plan(sql) {
        out.push(("plan", rows(db.query_named_plan(&plan, params()))));
    }
    out
}

/// The same for positional parameters
fn positional_paths(
    db: &Database,
    sql: &str,
    params: &[Value],
) -> Vec<(&'static str, stoolap::Result<Vec<Vec<Value>>>)> {
    let params = || params.to_vec();
    let db = &db.clone();
    let mut out = vec![
        ("ad hoc, parsed", rows(db.query(sql, params()))),
        ("ad hoc, cached", rows(db.query(sql, params()))),
    ];
    match db.prepare(sql) {
        Ok(stmt) => {
            out.push(("prepared", rows(stmt.query(params()))));
            let mut tx = db.begin().unwrap();
            out.push(("prepared in tx", rows(stmt.query_in(&mut tx, params()))));
            out.push(("ad hoc in tx", rows(tx.query(sql, params()))));
            tx.rollback().unwrap();
        }
        Err(e) => out.push(("prepared", Err(e))),
    }
    if let Ok(plan) = db.cached_plan(sql) {
        out.push(("plan", rows(db.query_plan(&plan, params()))));
    }
    out
}

fn assert_all_ok(
    sql: &str,
    paths: Vec<(&str, stoolap::Result<Vec<Vec<Value>>>)>,
    want: &[Vec<Value>],
) {
    for (path, got) in paths {
        match got {
            Ok(got) => assert_eq!(got, want, "{path}: {sql}"),
            Err(e) => panic!("{path}: {sql}: {e}"),
        }
    }
}

fn assert_all_err(sql: &str, paths: Vec<(&str, stoolap::Result<Vec<Vec<Value>>>)>, says: &str) {
    for (path, got) in paths {
        match got {
            Ok(got) => panic!("{path}: {sql}: answered {got:?}"),
            Err(e) => assert!(e.to_string().contains(says), "{path}: {sql}: {e}"),
        }
    }
}

fn int(i: i64) -> Value {
    Value::Integer(i)
}

#[test]
fn a_named_parameter_binds_by_name_and_a_repeated_name_reads_one_value() {
    let db = open("param_slots_named");
    let sql = "SELECT :b, :a, :b + :a";
    assert_all_ok(
        sql,
        named_paths(&db, sql, || named_params! { a: 1, b: 2 }),
        &[vec![int(2), int(1), int(3)]],
    );
    let sql = "SELECT id FROM t WHERE v > :low AND v < :low + 15 ORDER BY id";
    assert_all_ok(
        sql,
        named_paths(&db, sql, || named_params! { low: 5 }),
        &[vec![int(1)]],
    );
}

#[test]
fn a_named_point_lookup_answers_as_the_positional_one_on_every_path() {
    let db = open("param_slots_point");
    for id in [1, 2, 99] {
        let want = rows(db.query("SELECT id, v, s FROM t WHERE id = $1", (id,))).unwrap();
        let sql = "SELECT id, v, s FROM t WHERE id = :id";
        assert_all_ok(
            sql,
            named_paths(&db, sql, || named_params! { id: id }),
            &want,
        );
    }
}

#[test]
fn a_name_the_call_does_not_give_is_an_error_on_every_path() {
    let db = open("param_slots_named_missing");
    for sql in [
        "SELECT :a",
        "SELECT id FROM t WHERE id = :id",
        "SELECT id FROM t WHERE v > :v",
        "SELECT id FROM t WHERE id IN (SELECT id FROM t WHERE v = :v)",
        "SELECT id FROM t ORDER BY id LIMIT :n",
        "WITH c AS (SELECT id FROM t WHERE v >= :v) SELECT id FROM c",
    ] {
        assert_all_err(sql, named_paths(&db, sql, NamedParams::new), "has no value");
    }
    // Names are matched as written
    let sql = "SELECT id FROM t WHERE id = :ID";
    assert_all_err(
        sql,
        named_paths(&db, sql, || named_params! { id: 1 }),
        ":ID has no value",
    );
}

#[test]
fn a_name_the_statement_does_not_read_is_ignored() {
    let db = open("param_slots_named_extra");
    let sql = "SELECT id FROM t WHERE id = :id";
    assert_all_ok(
        sql,
        named_paths(&db, sql, || named_params! { id: 2, other: 5 }),
        &[vec![int(2)]],
    );
}

#[test]
fn a_position_the_call_does_not_give_is_an_error_on_every_path() {
    let db = open("param_slots_positional_missing");
    for (sql, given) in [
        ("SELECT $1", 0),
        ("SELECT ?", 0),
        ("SELECT id FROM t WHERE id = $1", 0),
        ("SELECT id FROM t WHERE id = $1 AND v = $2", 1),
        ("SELECT id FROM t WHERE id = $1 ORDER BY id LIMIT $2", 1),
    ] {
        assert_all_err(
            sql,
            positional_paths(&db, sql, &[int(1)][..given]),
            "parameters but only",
        );
    }
}

#[test]
fn a_write_missing_a_value_writes_nothing() {
    let db = open("param_slots_writes");
    let before = rows(db.query("SELECT id, v, s FROM t ORDER BY id", ())).unwrap();
    let positional: [(&str, Vec<Value>); 4] = [
        ("UPDATE t SET v = $1 WHERE id = $2", vec![int(5)]),
        ("UPDATE t SET v = $2 WHERE s = $1", vec![Value::text("a")]),
        ("DELETE FROM t WHERE id = $1", vec![]),
        ("INSERT INTO t VALUES ($1, $2, $3)", vec![int(9), int(9)]),
    ];
    for (sql, params) in positional {
        let db = &db.clone();
        for run in 0..2 {
            let error = db.execute(sql, params.clone()).expect_err(sql).to_string();
            assert!(
                error.contains("parameters but only"),
                "{sql} {run}: {error}"
            );
        }
        let stmt = db.prepare(sql).unwrap();
        assert!(stmt.execute(params.clone()).is_err(), "prepared {sql}");
        let mut tx = db.begin().unwrap();
        assert!(
            stmt.execute_in(&mut tx, params.clone()).is_err(),
            "in tx {sql}"
        );
        tx.rollback().unwrap();
    }
    let named: [(&str, NamedParams); 4] = [
        (
            "UPDATE t SET v = :v WHERE id = :id",
            named_params! { id: 1 },
        ),
        ("UPDATE t SET v = :v WHERE s = :s", named_params! { s: "a" }),
        ("DELETE FROM t WHERE id = :id", named_params! {}),
        (
            "INSERT INTO t VALUES (:id, :v, :s)",
            named_params! { id: 9, v: 9 },
        ),
    ];
    for (sql, params) in named {
        let db = &db.clone();
        for run in 0..2 {
            let error = db
                .execute_named(sql, params.clone())
                .expect_err(sql)
                .to_string();
            assert!(error.contains("has no value"), "{sql} {run}: {error}");
        }
        let stmt = db.prepare(sql).unwrap();
        assert!(
            stmt.execute_named(params.clone()).is_err(),
            "prepared {sql}"
        );
        let mut tx = db.begin().unwrap();
        assert!(
            stmt.execute_named_in(&mut tx, params.clone()).is_err(),
            "in tx {sql}"
        );
        tx.rollback().unwrap();
    }
    assert_eq!(
        rows(db.query("SELECT id, v, s FROM t ORDER BY id", ())).unwrap(),
        before
    );
}

#[test]
fn every_statement_of_a_program_is_bound_before_any_runs() {
    let db = open("param_slots_program");
    let sql = "UPDATE t SET v = :v WHERE id = 1; UPDATE t SET v = :w WHERE id = 2";
    let error = db
        .execute_named(sql, named_params! { v: 7 })
        .expect_err("second statement has no :w")
        .to_string();
    assert!(error.contains(":w has no value"), "{error}");
    let positional = "UPDATE t SET v = $1 WHERE id = 1; UPDATE t SET v = $2 WHERE id = 2";
    assert!(db.execute(positional, (7,)).is_err());
    assert_eq!(
        rows(db.query("SELECT v FROM t WHERE id <= 2 ORDER BY id", ())).unwrap(),
        vec![vec![int(10)], vec![int(20)]],
        "nothing written"
    );

    db.execute_named(sql, named_params! { v: 7, w: 8 }).unwrap();
    assert_eq!(
        rows(db.query("SELECT v FROM t WHERE id <= 2 ORDER BY id", ())).unwrap(),
        vec![vec![int(7)], vec![int(8)]]
    );
}

#[test]
fn a_statement_takes_positional_or_named_parameters_not_both() {
    let db = open("param_slots_mixed");
    for sql in [
        "SELECT id FROM t WHERE id = $1 AND v = :v",
        "SELECT id FROM t WHERE v = :v AND id = ?",
        "SELECT $0",
    ] {
        assert!(db.prepare(sql).is_err(), "prepare {sql}");
        assert!(db.query(sql, (1, 2)).is_err(), "positional {sql}");
        assert!(
            db.query_named(sql, named_params! { v: 1 }).is_err(),
            "named {sql}"
        );
    }
}

#[test]
fn named_writes_take_the_compiled_point_paths_as_positional_ones_do() {
    let db = open("param_slots_named_writes");
    let update = db.prepare("UPDATE t SET v = :v WHERE id = :id").unwrap();
    let delete = db.prepare("DELETE FROM t WHERE id = :id").unwrap();
    for run in 0..3 {
        assert_eq!(
            update
                .execute_named(named_params! { v: run, id: 1 })
                .unwrap(),
            1
        );
    }
    assert_eq!(
        update
            .execute_named(named_params! { v: 0, id: 99 })
            .unwrap(),
        0
    );
    let mut tx = db.begin().unwrap();
    assert_eq!(
        update
            .execute_named_in(&mut tx, named_params! { v: 50, id: 2 })
            .unwrap(),
        1
    );
    assert_eq!(
        delete
            .execute_named_in(&mut tx, named_params! { id: 3 })
            .unwrap(),
        1
    );
    tx.commit().unwrap();
    assert_eq!(
        rows(db.query("SELECT id, v FROM t ORDER BY id", ())).unwrap(),
        vec![vec![int(1), int(2)], vec![int(2), int(50)]]
    );
}

#[test]
fn explain_reads_no_value_and_explain_analyze_needs_them_all() {
    let db = open("param_slots_explain");
    for sql in [
        "EXPLAIN SELECT id FROM t WHERE id = $1",
        "EXPLAIN SELECT id FROM t ORDER BY id LIMIT 2 OFFSET $1",
        "EXPLAIN SELECT id FROM t WHERE v = :v",
    ] {
        let plan = rows(db.query(sql, ())).unwrap();
        assert!(!plan.is_empty(), "{sql}");
    }
    let error = rows(db.query("EXPLAIN ANALYZE SELECT id FROM t WHERE id = $1", ()))
        .expect_err("EXPLAIN ANALYZE runs the statement")
        .to_string();
    assert!(error.contains("parameters but only"), "{error}");
    let error = rows(db.query_named(
        "EXPLAIN ANALYZE SELECT id FROM t WHERE v = :v",
        NamedParams::new(),
    ))
    .expect_err("EXPLAIN ANALYZE runs the statement")
    .to_string();
    assert!(error.contains(":v has no value"), "{error}");
    assert!(rows(db.query("EXPLAIN ANALYZE SELECT id FROM t WHERE id = $1", (1,))).is_ok());
}

#[test]
fn a_view_cannot_take_parameters() {
    let db = open("param_slots_view");
    for sql in [
        "CREATE VIEW v AS SELECT :x AS n",
        "CREATE VIEW v AS SELECT id FROM t WHERE id = $1",
        "CREATE VIEW v AS SELECT id FROM t WHERE id = ?",
    ] {
        let error = db.execute(sql, ()).expect_err(sql).to_string();
        assert!(
            error.contains("a view cannot take parameters"),
            "{sql}: {error}"
        );
    }
    // A view stored before the rule: its own :x would take the caller's slot 1
    db.engine()
        .create_view("old_v", "SELECT :x AS n".to_string(), false)
        .unwrap();
    let error = rows(db.query_named("SELECT :y AS a, n FROM old_v", named_params! { x: 7, y: 1 }))
        .expect_err("the old view's parameter")
        .to_string();
    assert!(error.contains("a view cannot take parameters"), "{error}");
    // Run directly first, so the view's text is in the plan cache with its slots
    db.engine()
        .create_view("old_w", "SELECT $1 AS n".to_string(), false)
        .unwrap();
    assert_eq!(
        rows(db.query("SELECT $1 AS n", (5,))).unwrap(),
        vec![vec![int(5)]]
    );
    assert!(rows(db.query("SELECT n FROM old_w", ())).is_err(), "cached");
    db.execute("DROP VIEW old_v", ()).unwrap();
    db.execute("DROP VIEW old_w", ()).unwrap();
}

#[test]
fn a_parsed_statement_run_by_its_ast_checks_every_slot_before_it_writes() {
    let db = open("param_slots_raw_ast");
    let parse = |sql: &str| {
        stoolap::parser::Parser::new(sql)
            .parse_program()
            .unwrap()
            .statements
            .remove(0)
    };
    let insert = parse("INSERT INTO t VALUES (9, CASE WHEN TRUE THEN 1 ELSE $1 END, 'x')");
    let select = parse("SELECT id FROM t WHERE id = 1 OR v = CASE WHEN TRUE THEN 0 ELSE $1 END");
    let mut tx = db.begin().unwrap();
    let error = tx
        .execute_prepared(&insert, ())
        .expect_err("$1 has no value");
    assert!(error.to_string().contains("parameters but only"), "{error}");
    assert!(tx.query_prepared(&select, ()).is_err(), "query_prepared");
    assert_eq!(tx.execute_prepared(&insert, (0,)).unwrap(), 1, "given");
    tx.rollback().unwrap();
    assert_eq!(
        rows(db.query("SELECT COUNT(*) FROM t WHERE id = 9", ())).unwrap(),
        vec![vec![int(0)]]
    );
}

#[test]
fn a_statement_takes_at_most_65535_parameter_slots() {
    let db = open("param_slots_limit");
    let named = |n: usize| {
        let list: Vec<String> = (1..=n).map(|i| format!(":p{i}")).collect();
        format!("SELECT {}", list.join(", "))
    };
    let error = db.prepare(&named(65_536)).err().expect("one slot too many");
    assert!(
        error.to_string().contains("at most 65535 parameters"),
        "{error}"
    );
    let error = db.prepare("SELECT $65536").err().expect("$65536");
    assert!(
        error.to_string().contains("at most 65535 parameters"),
        "{error}"
    );
    // The last slot reads its own value, not slot 1's
    let last = db.prepare("SELECT $65535, $1").unwrap();
    let values: Vec<Value> = (1..=65_535).map(Value::Integer).collect();
    assert_eq!(
        rows(last.query(values)).unwrap(),
        vec![vec![int(65_535), int(1)]]
    );
}
