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

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{prepared_exists_counts, prepared_scalar_runs, random_draws};
use stoolap::{Database, Params};

// The counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn setup(name: &str) -> Database {
    let db = Database::open(&format!("memory://{name}")).unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO t VALUES (1, 10), (2, 20), (3, 30), (4, NULL)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, x INTEGER)",
        "INSERT INTO u VALUES (1, 1, 5), (2, 1, 9), (3, 1, NULL), (4, 2, 7), (5, 2, 3), \
         (6, 3, NULL), (7, 3, 4), (8, 3, 4)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db
}

/// The column each row of `sql` gives at `i`
fn column<P: Params>(db: &Database, sql: &str, i: usize, params: P) -> Vec<Option<i64>> {
    db.query(sql, params)
        .unwrap()
        .map(|row| row.unwrap().get::<Option<i64>>(i).unwrap())
        .collect()
}

fn error(db: &Database, sql: &str) -> String {
    match db.query(sql, ()) {
        Err(e) => e.to_string(),
        Ok(rows) => rows
            .map(|row| row.map(|_| ()))
            .collect::<Result<Vec<()>, _>>()
            .expect_err("expected an error")
            .to_string(),
    }
}

fn prepared_line(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .find(|line| line.trim_start().starts_with("Prepared:"))
        .map(|line| line.trim().to_string())
        .unwrap_or_default()
}

fn scalar(block: &str) -> String {
    format!("SELECT id, ({block}) FROM t ORDER BY id")
}

#[test]
fn a_scalar_block_answers_from_the_rows_its_order_by_limit_and_offset_keep() {
    let _guard = serial();
    let db = setup("block_order_scalar");
    let n = None;
    for (block, want) in [
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY x LIMIT 1",
            [Some(5), Some(3), Some(4), n],
        ),
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY x DESC LIMIT 1",
            [n, Some(7), n, n],
        ),
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY x DESC NULLS LAST LIMIT 1",
            [Some(9), Some(7), Some(4), n],
        ),
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY x NULLS FIRST LIMIT 1",
            [n, Some(3), n, n],
        ),
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY x LIMIT 1 OFFSET 1",
            [Some(9), Some(7), Some(4), n],
        ),
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY x DESC LIMIT 1 OFFSET 2",
            [Some(5), n, Some(4), n],
        ),
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY x OFFSET 2",
            [n, n, n, n],
        ),
        ("SELECT x FROM u WHERE u.k = t.id LIMIT 0", [n, n, n, n]),
        (
            "SELECT u.id FROM u WHERE u.k = t.id ORDER BY x, u.id DESC LIMIT 1",
            [Some(1), Some(5), Some(8), n],
        ),
        (
            "SELECT u.id FROM u WHERE u.k = t.id ORDER BY x LIMIT 1",
            [Some(1), Some(5), Some(7), n],
        ),
        // Rows 7 and 8 tie on x: they stay in the order read
        (
            "SELECT u.id FROM u WHERE u.k = t.id ORDER BY x LIMIT 1 OFFSET 1",
            [Some(2), Some(4), Some(8), n],
        ),
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY ABS(x - t.a) LIMIT 1",
            [Some(9), Some(7), Some(4), n],
        ),
        (
            "SELECT x AS y FROM u WHERE u.k = t.id ORDER BY y DESC NULLS LAST LIMIT 1",
            [Some(9), Some(7), Some(4), n],
        ),
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY 1 / (x - 9) LIMIT 1",
            [Some(5), Some(7), Some(4), n],
        ),
        (
            "SELECT u.x FROM u WHERE u.k = t.id \
             AND u.x > (SELECT MIN(x) FROM u AS w WHERE w.k = u.k) ORDER BY u.x LIMIT 1",
            [Some(9), Some(7), n, n],
        ),
        (
            "SELECT x FROM (VALUES (1, 5), (1, 2), (2, 8)) AS v(k, x) \
             WHERE v.k = t.id ORDER BY x LIMIT 1",
            [Some(2), Some(8), n, n],
        ),
    ] {
        let sql = scalar(block);
        assert_eq!(prepared_line(&db, &sql), "Prepared: yes", "{sql}");
        let runs = prepared_scalar_runs();
        assert_eq!(column(&db, &sql, 1, ()), want, "{sql}");
        // The nested MIN runs for each of the eight candidates too
        let per_call = if block.contains("AS w") { 12 } else { 4 };
        assert_eq!(
            prepared_scalar_runs() - runs,
            per_call,
            "{sql} runs once per row"
        );
    }
    let once = scalar("SELECT x FROM u WHERE u.k = 1 ORDER BY x DESC LIMIT 1");
    let runs = prepared_scalar_runs();
    assert_eq!(column(&db, &once, 1, ()), [n, n, n, n]);
    assert_eq!(
        prepared_scalar_runs() - runs,
        1,
        "an uncorrelated block runs once"
    );
    let params = scalar("SELECT x FROM u WHERE u.k = t.id ORDER BY x LIMIT $1 OFFSET $2");
    assert_eq!(
        column(&db, &params, 1, (1, 1)),
        [Some(9), Some(7), Some(4), n],
        "LIMIT and OFFSET as parameters"
    );
    let ordered = "SELECT id FROM t \
                   ORDER BY (SELECT x FROM u WHERE u.k = t.id ORDER BY x DESC NULLS LAST LIMIT 1), id";
    assert_eq!(prepared_line(&db, ordered), "Prepared: yes");
    assert_eq!(
        column(&db, ordered, 0, ()),
        [Some(3), Some(2), Some(1), Some(4)]
    );
}

#[test]
fn a_scalar_block_cut_to_more_than_one_row_or_by_a_bad_count_fails() {
    let _guard = serial();
    let db = setup("block_order_errors");
    for (block, want) in [
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY x LIMIT 2",
            "scalar subquery returned more than one row",
        ),
        (
            "SELECT x FROM u WHERE u.k = 2 ORDER BY x LIMIT 2",
            "scalar subquery returned more than one row",
        ),
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY x LIMIT -1",
            "LIMIT must be non-negative, got -1",
        ),
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY x LIMIT 1 OFFSET -1",
            "OFFSET must be non-negative, got -1",
        ),
    ] {
        let sql = scalar(block);
        assert_eq!(prepared_line(&db, &sql), "Prepared: yes", "{sql}");
        let err = error(&db, &sql);
        assert!(err.contains(want), "{sql}: {err}");
    }
}

#[test]
fn exists_and_member_blocks_answer_from_the_rows_kept() {
    let _guard = serial();
    let db = setup("block_order_members");
    for (sql, want) in [
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.id LIMIT 0) ORDER BY id",
            &[][..],
        ),
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.id LIMIT 1 OFFSET 2) \
             ORDER BY id",
            &[1, 3],
        ),
        (
            "SELECT id FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.k = t.id OFFSET 2) \
             ORDER BY id",
            &[2, 4],
        ),
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.id ORDER BY x LIMIT 1) \
             ORDER BY id",
            &[1, 2, 3],
        ),
        (
            "SELECT id FROM t WHERE a / 10 IN (SELECT k FROM u ORDER BY x DESC LIMIT 2) ORDER BY id",
            &[1, 3],
        ),
        (
            "SELECT id FROM t WHERE a / 10 NOT IN (SELECT k FROM u ORDER BY x LIMIT 3) ORDER BY id",
            &[1],
        ),
        (
            "SELECT id FROM t WHERE a / 10 IN (SELECT k FROM u ORDER BY x NULLS FIRST, u.id LIMIT 2) \
             ORDER BY id",
            &[1, 3],
        ),
        (
            "SELECT id FROM t WHERE 5 IN (SELECT x FROM u WHERE u.k = t.id ORDER BY x LIMIT 1) \
             ORDER BY id",
            &[1],
        ),
        (
            "SELECT id FROM t WHERE 4 = ANY \
             (SELECT x FROM u WHERE u.k = t.id ORDER BY x DESC LIMIT 1 OFFSET 1) ORDER BY id",
            &[3],
        ),
        (
            "SELECT id FROM t WHERE a > ALL \
             (SELECT x FROM u WHERE u.k = t.id ORDER BY x DESC LIMIT 1) ORDER BY id",
            &[2, 4],
        ),
    ] {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
        let want: Vec<_> = want.iter().map(|&id| Some(id)).collect();
        assert_eq!(column(&db, sql, 0, ()), want, "{sql}");
    }
    let sql = "SELECT id, EXISTS (SELECT 1 FROM u WHERE u.k = t.id LIMIT 1 OFFSET 1) FROM t \
               ORDER BY id";
    let probes = prepared_exists_counts().0;
    let found: Vec<bool> = db
        .query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<bool>(1).unwrap())
        .collect();
    assert_eq!(found, [true, true, true, false]);
    assert_eq!(prepared_exists_counts().0 - probes, 4, "one probe per row");
}

#[test]
fn a_cut_block_outside_the_family_says_why() {
    let _guard = serial();
    let db = setup("block_order_refused");
    // A grouped block with ORDER BY and LIMIT runs as a statement block
    let grouped = scalar("SELECT MAX(x) FROM u WHERE u.k = t.id ORDER BY 1 LIMIT 1");
    assert_eq!(prepared_line(&db, &grouped), "Prepared: yes", "{grouped}");
    for (block, reason) in [
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY 1 LIMIT 1",
            "an ORDER BY position in a subquery block",
        ),
        (
            "SELECT x FROM u WHERE u.k = t.id ORDER BY x LIMIT t.a",
            "a derived table's LIMIT or OFFSET naming a column",
        ),
    ] {
        let sql = scalar(block);
        assert_eq!(
            prepared_line(&db, &sql),
            format!("Prepared: no ({reason})"),
            "{sql}"
        );
    }
}

#[test]
fn an_order_by_name_binds_as_the_general_path_binds_it() {
    let _guard = serial();
    let db = setup("block_order_binding");
    let n = None;
    for (block, want) in [
        // A bare alias sorts by the SELECT item, over the column it shadows
        (
            "SELECT -u.id AS id FROM u WHERE u.k = t.id ORDER BY id LIMIT 1",
            [Some(-3), Some(-5), Some(-8), n],
        ),
        // Inside an expression the column comes first
        (
            "SELECT -u.id AS id FROM u WHERE u.k = t.id ORDER BY id + 0 LIMIT 1",
            [Some(-1), Some(-4), Some(-6), n],
        ),
        (
            "SELECT -u.x AS k FROM u WHERE u.k = t.id ORDER BY k LIMIT 1",
            [Some(-9), Some(-7), Some(-4), n],
        ),
        (
            "SELECT -u.x AS y FROM u WHERE u.k = t.id ORDER BY -y LIMIT 1",
            [Some(-5), Some(-3), Some(-4), n],
        ),
        (
            "SELECT x * 2 FROM u WHERE u.k = t.id ORDER BY x * 2 DESC LIMIT 1",
            [n, Some(14), n, n],
        ),
    ] {
        let sql = scalar(block);
        assert_eq!(prepared_line(&db, &sql), "Prepared: yes", "{sql}");
        assert_eq!(column(&db, &sql, 1, ()), want, "{sql}");
    }
    let member = "SELECT id FROM t \
                  WHERE -5 IN (SELECT -u.id AS id FROM u WHERE u.k = t.id ORDER BY id LIMIT 1)";
    assert_eq!(column(&db, member, 0, ()), [Some(2)]);
    db.execute(
        "CREATE TABLE w (id INTEGER PRIMARY KEY, x INTEGER, \"w.x\" INTEGER)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO w VALUES (1, 10, 20), (2, 20, 10)", ())
        .unwrap();
    // The item as written is matched before its alias, which here names
    // the enclosing column too, is read inside the key
    let sql = "SELECT (SELECT ABS(u.x - a) AS a FROM u WHERE u.k = 1 \
               ORDER BY ABS(u.x - a) LIMIT 1) FROM t WHERE t.id = 1";
    assert_eq!(prepared_line(&db, sql), "Prepared: yes");
    assert_eq!(column(&db, sql, 0, ()), [Some(1)]);
    // A column named w.x is not the item w.x, though the two read alike
    for (key, want) in [("\"w.x\", x", 20), ("w.x", 10)] {
        let sql =
            format!("SELECT (SELECT w.x FROM w ORDER BY {key} LIMIT 1) FROM t WHERE t.id = 1");
        assert_eq!(prepared_line(&db, &sql), "Prepared: yes", "{sql}");
        assert_eq!(column(&db, &sql, 0, ()), [Some(want)], "{sql}");
    }
}

#[test]
fn a_key_that_is_the_select_item_answers_with_the_value_it_sorted() {
    let _guard = serial();
    let db = setup("block_order_item_value");
    let sql = "SELECT (SELECT RANDOM() AS r FROM u ORDER BY r LIMIT 1) FROM t";
    assert_eq!(prepared_line(&db, sql), "Prepared: yes");
    let draws = random_draws();
    let found = db.query(sql, ()).unwrap().collect::<Result<Vec<_>, _>>();
    assert_eq!(found.unwrap().len(), 4);
    assert_eq!(
        random_draws() - draws,
        8,
        "one run, one draw per inner row and none for the answer"
    );
    // The item's computation over the same columns, however they are spelled
    for key in ["RANDOM() + x", "RANDOM() + \"x\"", "random() + U.X"] {
        let sql = format!("SELECT (SELECT RANDOM() + x AS r FROM u ORDER BY {key} LIMIT 1) FROM t");
        assert_eq!(prepared_line(&db, &sql), "Prepared: yes", "{sql}");
        let draws = random_draws();
        let found = db.query(&sql, ()).unwrap().collect::<Result<Vec<_>, _>>();
        assert_eq!(found.unwrap().len(), 4);
        assert_eq!(
            random_draws() - draws,
            8,
            "{sql} sorts by the value it answers"
        );
    }
    // A key computed from a volatile alias runs as a statement block
    let statement = "SELECT (SELECT RANDOM() AS r FROM u ORDER BY r + 0 LIMIT 1) FROM t";
    assert_eq!(prepared_line(&db, statement), "Prepared: yes");
    let found = db
        .query(statement, ())
        .unwrap()
        .collect::<Result<Vec<_>, _>>();
    assert_eq!(found.unwrap().len(), 4);
}

#[test]
fn a_row_block_binds_its_order_by_names_as_a_value_block_does() {
    let _guard = serial();
    let db = setup("block_order_row_binding");
    for (item, key) in [
        ("-u.id AS id", "id"),
        ("-u.id AS id", "id + 0"),
        ("-u.x AS k", "k"),
        ("-u.x AS y", "-y"),
        ("x * 2", "x * 2 DESC"),
    ] {
        let value = format!("(SELECT {item} FROM u WHERE u.k = t.id ORDER BY {key} LIMIT 1)");
        for (row, block) in [
            (format!("({value}, 0)"), format!("{item}, 0")),
            (format!("(0, {value})"), format!("0, {item}")),
        ] {
            let block = format!("SELECT {block} FROM u WHERE u.k = t.id ORDER BY {key} LIMIT 1");
            for (sql, want) in [
                (
                    format!("SELECT id FROM t WHERE {row} IS NOT DISTINCT FROM ({block})"),
                    vec![Some(1), Some(2), Some(3)],
                ),
                (
                    format!("SELECT id FROM t WHERE {row} IN ({block})"),
                    column(
                        &db,
                        &format!("SELECT id FROM t WHERE {value} IS NOT NULL"),
                        0,
                        (),
                    ),
                ),
            ] {
                assert_eq!(prepared_line(&db, &sql), "Prepared: yes", "{sql}");
                assert_eq!(column(&db, &sql, 0, ()), want, "{sql}");
            }
        }
    }
}

#[test]
fn a_row_block_key_that_is_a_value_answers_with_the_value_it_sorted() {
    let _guard = serial();
    let db = setup("block_order_row_value");
    for sql in [
        "SELECT (0.5, 0) < (SELECT RANDOM() AS r, 0 FROM u ORDER BY r LIMIT 1) FROM t",
        "SELECT (0.5, 0) > ANY (SELECT 0 + RANDOM(), 0 FROM u ORDER BY 0 + RANDOM() LIMIT 1) \
         FROM t",
        "SELECT (0, 1) IN (SELECT ROUND(RANDOM()) AS r, id FROM u ORDER BY r, r, id LIMIT 1) \
         FROM t",
        "SELECT (0, 1) IN (SELECT 0 + RANDOM(), id FROM u \
         ORDER BY 0 + RANDOM(), 0 + RANDOM() DESC, id LIMIT 1) FROM t",
    ] {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
        let draws = random_draws();
        let found = db.query(sql, ()).unwrap().collect::<Result<Vec<_>, _>>();
        assert_eq!(found.unwrap().len(), 4);
        assert_eq!(
            random_draws() - draws,
            8,
            "{sql}: one run, one draw per inner row and none for the answer"
        );
    }
    // A key computed from a volatile alias runs as a statement block
    let statement =
        "SELECT (0.5, 0) < (SELECT RANDOM() AS r, 0 FROM u ORDER BY r + 0 LIMIT 1) FROM t";
    assert_eq!(prepared_line(&db, statement), "Prepared: yes");
    let found = db
        .query(statement, ())
        .unwrap()
        .collect::<Result<Vec<_>, _>>();
    assert_eq!(found.unwrap().len(), 4);
}

#[test]
fn an_exists_reads_no_row_past_the_first_its_offset_leaves() {
    let _guard = serial();
    let db = setup("block_order_exists_stops");
    // Rows past the second raise an error, which no answer reaches
    let fails = "CASE WHEN u.id > 2 THEN ABS('a') ELSE 1 END = 1";
    for tail in ["OFFSET 1", "LIMIT 5 OFFSET 1"] {
        let sql = format!("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE {fails} {tail})");
        assert_eq!(prepared_line(&db, &sql), "Prepared: yes", "{sql}");
        assert_eq!(column(&db, &sql, 0, ()).len(), 4, "{sql}");
    }
    let sql = format!("SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE {fails} OFFSET 2)");
    assert!(error(&db, &sql).contains("ABS"), "{sql}");
}

#[test]
fn a_table_function_block_generates_the_rows_its_limit_keeps() {
    let _guard = serial();
    let db = setup("block_order_function");
    let series = "generate_series(1, 10000001)";
    for (sql, want) in [
        (
            scalar(&format!("SELECT value FROM {series} LIMIT 1 OFFSET 2")),
            [Some(3), Some(3), Some(3), Some(3)],
        ),
        (
            scalar(&format!("SELECT value + t.id FROM {series} LIMIT 1")),
            [Some(2), Some(3), Some(4), Some(5)],
        ),
    ] {
        assert_eq!(prepared_line(&db, &sql), "Prepared: yes", "{sql}");
        assert_eq!(column(&db, &sql, 1, ()), want, "{sql}");
    }
    let exists = format!("SELECT id FROM t WHERE EXISTS (SELECT value FROM {series} LIMIT 1)");
    assert_eq!(column(&db, &exists, 0, ()).len(), 4);
    let member = format!("SELECT id FROM t WHERE id + 2 IN (SELECT value FROM {series} LIMIT 3)");
    assert_eq!(column(&db, &member, 0, ()), [Some(1)]);
}

#[test]
fn an_order_by_that_chooses_no_row_is_still_bound() {
    let _guard = serial();
    let db = setup("block_order_unused");
    for (sql, want) in [
        (
            scalar("SELECT x FROM u WHERE u.id = 1 ORDER BY missing"),
            "missing",
        ),
        (
            scalar("SELECT x FROM u WHERE u.id = 1 ORDER BY u.missing"),
            "u.missing",
        ),
        (
            scalar("SELECT x FROM u WHERE u.id = 1 ORDER BY t.missing"),
            "t.missing",
        ),
        (
            scalar("SELECT x FROM u WHERE u.id = 1 ORDER BY nosuchfn(x)"),
            "NOSUCHFN",
        ),
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u ORDER BY missing)".to_string(),
            "missing",
        ),
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u ORDER BY missing LIMIT 1)".to_string(),
            "missing",
        ),
        (
            "SELECT id FROM t WHERE a / 10 IN (SELECT k FROM u ORDER BY missing)".to_string(),
            "missing",
        ),
    ] {
        assert_eq!(prepared_line(&db, &sql), "Prepared: yes", "{sql}");
        let err = error(&db, &sql);
        assert!(err.contains(want), "{sql}: {err}");
    }
    let sql = scalar("SELECT x FROM u WHERE u.id = 1 ORDER BY ABS(x - t.a)");
    assert_eq!(prepared_line(&db, &sql), "Prepared: yes");
    let runs = prepared_scalar_runs();
    assert_eq!(column(&db, &sql, 1, ()), [Some(5); 4]);
    assert_eq!(
        prepared_scalar_runs() - runs,
        1,
        "a key that chooses no row reads no outer value"
    );
    let refused = scalar("SELECT x FROM u WHERE u.id = 1 ORDER BY (SELECT 1)");
    assert_eq!(
        prepared_line(&db, &refused),
        "Prepared: no (a subquery block's ORDER BY key holding a subquery without LIMIT)"
    );
}

#[test]
fn a_key_looked_at_only_takes_no_slot_of_its_own() {
    use stoolap::test_failpoints::prepared_slot_count;
    let _guard = serial();
    let db = setup("block_order_slots");
    let slots = |sql: &str| {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
        column(&db, sql, 0, (1, 1, 1, 1, 1, 1, 1));
        prepared_slot_count()
    };
    // The key matched against the item first, and a key that chooses no row
    for tail in ["LIMIT $7", "LIMIT 1 OFFSET $7"] {
        let sql = |key: &str| format!("SELECT (SELECT x FROM u ORDER BY {key} {tail}) FROM t");
        assert_eq!(
            slots(&sql("ABS(x - t.a)")),
            slots(&sql("x")),
            "{tail}: matching the key with the item takes no slot of its own"
        );
    }
    let unused =
        |key: &str| format!("SELECT (SELECT x FROM u WHERE u.id = $7 ORDER BY {key}) FROM t");
    assert_eq!(
        slots(&unused("ABS(x - t.a)")),
        slots(&unused("x")),
        "a key that chooses no row reads no enclosing value"
    );
}

#[test]
fn a_block_in_a_key_that_chooses_no_row_is_not_kept_to_run() {
    use stoolap::test_failpoints::prepared_block_count;
    let _guard = serial();
    let db = setup("block_order_dropped_blocks");
    let blocks = |sql: &str| {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
        column(&db, sql, 0, ());
        prepared_block_count()
    };
    for from in [
        "u WHERE u.id = t.id",
        "(SELECT x FROM u WHERE u.id = t.id) q",
    ] {
        let sql = |key: &str| {
            format!(
                "SELECT id FROM t WHERE EXISTS (SELECT (SELECT MAX(x) FROM u) AS k, x \
                 FROM {from} ORDER BY {key}) ORDER BY id"
            )
        };
        assert_eq!(blocks(&sql("k")), blocks(&sql("x")), "{from}");
    }
}

#[test]
fn a_grouped_distinct_on_block_keeps_a_group_whose_aggregate_is_null() {
    let db = Database::open("memory://block_order_limit_grouped_distinct_on_null").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER, y INTEGER)",
        "INSERT INTO t VALUES (1, 1), (2, 2), (3, 3), (4, 0), (5, 1), (6, 2), (7, 3), (8, 0)",
        "INSERT INTO u VALUES (1, 1, 1), (2, 2, 2), (3, 0, 3), (4, 1, 4), (5, 2, 5), (6, 0, NULL)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    // Row 6's block has one group, whose MAX(y) is NULL: NOT IN over it is unknown
    let sql = "SELECT id FROM t WHERE a NOT IN (SELECT DISTINCT ON (x) MAX(y) FROM u \
               WHERE id = t.id GROUP BY x ORDER BY x) ORDER BY id";
    let ids: Vec<i64> = db
        .query(sql, ())
        .unwrap()
        .map(|row| row.unwrap().get::<i64>(0).unwrap())
        .collect();
    assert_eq!(ids, [4, 5, 7, 8], "{sql}");
}

#[test]
fn a_block_key_s_subquery_answers_whatever_it_reads() {
    let db = Database::open("memory://block_order_limit_unchosen_alias").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, x INTEGER)",
        "CREATE TABLE v (id INTEGER PRIMARY KEY)",
        "INSERT INTO t VALUES (1, 1), (2, 2), (3, 7)",
        "INSERT INTO u VALUES (1, 1), (2, 2)",
        "INSERT INTO v VALUES (1), (2)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    for (sql, want) in [
        // The key reads the SELECT alias z inside its subquery
        (
            "SELECT id FROM t WHERE a IN (SELECT x AS z FROM u ORDER BY (SELECT z + u.x)) \
             ORDER BY id",
            vec![1, 2],
        ),
        (
            "SELECT id FROM t WHERE a IN (SELECT x AS z FROM u ORDER BY (SELECT z + u.x) DESC \
             LIMIT 1 + 0) ORDER BY id",
            vec![2],
        ),
        // The inner DISTINCT ON reads u.z, the alias, not a column of u
        (
            "SELECT id FROM t WHERE a IN (SELECT x AS z FROM u \
             ORDER BY (SELECT DISTINCT ON (u.z) u.x) DESC LIMIT 1 + 0) ORDER BY id",
            vec![2],
        ),
        // The same key one statement deeper, in a block a computed LIMIT sends
        (
            "SELECT id FROM t WHERE a IN (SELECT (SELECT x AS z FROM u \
             ORDER BY (SELECT z + u.x) DESC LIMIT 1 + 0) FROM v LIMIT 1 + 0) ORDER BY id",
            vec![2],
        ),
        // An EXISTS computes no key: u.id * 9223372036854775807 overflows for 2
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.id = t.id AND u.id > 0 \
             GROUP BY u.id ORDER BY u.id * 9223372036854775807) ORDER BY id",
            vec![1, 2],
        ),
        // An EXISTS orders no row: (SELECT id FROM v), two rows, is never read
        (
            "SELECT id FROM t WHERE EXISTS (SELECT 1 FROM u WHERE u.id = t.id AND u.id > 0 \
             GROUP BY u.id ORDER BY (SELECT id FROM v)) ORDER BY id",
            vec![1, 2],
        ),
    ] {
        let ids: Vec<i64> = db
            .query(sql, ())
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
            .map(|row| row.unwrap().get::<i64>(0).unwrap())
            .collect();
        assert_eq!(ids, want, "{sql}");
    }
}
