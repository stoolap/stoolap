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

//! A grouped statement holding subqueries runs them through prepared blocks
//! over any source and with any grouping modifier: it answers as the same
//! statement over a table does, or as its twin written without them

#![cfg(feature = "test-failpoints")]

use std::sync::{Mutex, MutexGuard};

use stoolap::test_failpoints::{
    general_path_hits, prepared_strategy_runs, tvf_rows, GeneralPath, PreparedStrategy,
};
use stoolap::Database;

// The general path counters are process-wide
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// t: id 1..200, a = id % 9, b = id % 13; u: id 1..40, k = id % 11,
/// x = id % 7; w: id 1..200, z = id % 4; v and c read t as it is
fn fill(db: &Database) {
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, x INTEGER)",
        "CREATE TABLE w (id INTEGER PRIMARY KEY, z INTEGER)",
        "CREATE TABLE s (value INTEGER PRIMARY KEY)",
        "INSERT INTO t SELECT g.value, g.value % 9, g.value % 13 FROM generate_series(1, 200) g",
        "INSERT INTO u SELECT g.value, g.value % 11, g.value % 7 FROM generate_series(1, 40) g",
        "INSERT INTO w SELECT g.value, g.value % 4 FROM generate_series(1, 200) g",
        "INSERT INTO s SELECT g.value FROM generate_series(1, 60) g",
        "CREATE VIEW v AS SELECT id, a, b FROM t",
    ] {
        db.execute(sql, ()).unwrap();
    }
}

/// t's rows written as a VALUES list
fn values_source() -> String {
    let rows: Vec<String> = (1..=200)
        .map(|id| format!("({id}, {}, {})", id % 9, id % 13))
        .collect();
    format!("(VALUES {}) t(id, a, b)", rows.join(", "))
}

/// A source reading t's rows under the name t
struct Source {
    name: &'static str,
    from: String,
    /// The WITH the source needs
    with: &'static str,
    /// A join whose WHERE subqueries the join blocks run: two tables
    where_blocks: bool,
}

fn sources() -> Vec<Source> {
    let source = |name, from: &str, with, where_blocks| Source {
        name,
        from: from.to_string(),
        with,
        where_blocks,
    };
    vec![
        source("table", "t", "", true),
        source("derived", "(SELECT id, a, b FROM t) t", "", true),
        source("view", "v t", "", true),
        source("cte", "c t", "WITH c AS (SELECT id, a, b FROM t) ", true),
        source("values", &values_source(), "", true),
        source("join", "t JOIN w ON w.id = t.id", "", true),
        source("join right", "w JOIN t ON t.id = w.id", "", true),
        source(
            "left join",
            "t LEFT JOIN w ON w.id = t.id AND w.z < 0",
            "",
            true,
        ),
        source(
            "three-way join",
            "t JOIN w ON w.id = t.id JOIN w w2 ON w2.id = t.id",
            "",
            false,
        ),
        source(
            "join with a derived side",
            "(SELECT id, a, b FROM t) t JOIN w ON w.id = t.id",
            "",
            false,
        ),
    ]
}

/// Each statement over `{t}`, which every source answers as t does
const SITES: &[&str] = &[
    "SELECT a, SUM(b), (SELECT MAX(u.x) FROM u) FROM {t} GROUP BY a ORDER BY a",
    "SELECT a, (SELECT COUNT(*) FROM u WHERE u.k = t.a) FROM {t} GROUP BY a ORDER BY a",
    "SELECT a, SUM(b) FROM {t} GROUP BY a HAVING SUM(b) > (SELECT MAX(u.x) FROM u) * 20 ORDER BY a",
    "SELECT a, SUM(b) FROM {t} GROUP BY a \
     HAVING EXISTS (SELECT 1 FROM u WHERE u.k = t.a AND u.x > 3) ORDER BY a",
    "SELECT a, SUM((SELECT COUNT(*) FROM u WHERE u.k = t.b)) FROM {t} GROUP BY a ORDER BY a",
    "SELECT COUNT(*), SUM(b) FROM {t} GROUP BY (SELECT COUNT(*) FROM u WHERE u.k = t.a) \
     ORDER BY 1, 2",
    "SELECT (SELECT COUNT(*) FROM u WHERE u.k = t.a) AS g, COUNT(*) FROM {t} GROUP BY g \
     ORDER BY 1",
    "SELECT a, COUNT(*) FILTER (WHERE b IN (SELECT u.x FROM u WHERE u.k < 3)) FROM {t} \
     GROUP BY a ORDER BY a",
    "SELECT a, SUM(b) FROM {t} WHERE t.b > (SELECT MIN(u.x) FROM u) + 2 GROUP BY a \
     HAVING SUM(b) > (SELECT MAX(u.x) FROM u) ORDER BY a",
    "SELECT a, (SELECT COUNT(*) FROM u WHERE u.k = t.a) FROM {t} \
     WHERE EXISTS (SELECT 1 FROM u WHERE u.k = t.b) GROUP BY a ORDER BY a",
    "SELECT SUM(b), (SELECT MAX(u.x) FROM u), COUNT(*) FROM {t}",
    "SELECT a, SUM(b), (SELECT COUNT(*) FROM u WHERE u.k = t.a) FROM {t} GROUP BY a \
     ORDER BY a LIMIT 4 OFFSET 2",
];

/// Each statement over `{t}` with a grouping modifier, DISTINCT or a set
/// operation, and its twin over t written without them
const SHAPES: &[(&str, &str)] = &[
    (
        "SELECT a, b, COUNT(*), (SELECT MAX(u.x) FROM u) FROM {t} GROUP BY ROLLUP(a, b) \
         ORDER BY 1, 2",
        "SELECT a, b, COUNT(*), 6 FROM t GROUP BY ROLLUP(a, b) ORDER BY 1, 2",
    ),
    (
        "SELECT a, COUNT(*), (SELECT COUNT(*) FROM u WHERE u.k = t.a) FROM {t} GROUP BY CUBE(a) \
         ORDER BY 1",
        "SELECT r.a, r.c, (SELECT COUNT(*) FROM u WHERE u.k = r.a) FROM \
         (SELECT a, COUNT(*) AS c FROM t GROUP BY CUBE(a)) r ORDER BY 1",
    ),
    (
        "SELECT a, b, SUM(b) FROM {t} GROUP BY GROUPING SETS ((a), (b), ()) \
         HAVING SUM(b) > (SELECT MAX(u.x) FROM u) * 10 ORDER BY 1, 2",
        "SELECT a, b, SUM(b) FROM t GROUP BY GROUPING SETS ((a), (b), ()) \
         HAVING SUM(b) > 60 ORDER BY 1, 2",
    ),
    (
        "SELECT a, SUM((SELECT COUNT(*) FROM u WHERE u.k = t.b)) FROM {t} GROUP BY ROLLUP(a) \
         ORDER BY 1",
        "SELECT a, SUM(v) FROM (SELECT a, (SELECT COUNT(*) FROM u WHERE u.k = t.b) AS v FROM t) d \
         GROUP BY ROLLUP(a) ORDER BY 1",
    ),
    (
        "SELECT (SELECT COUNT(*) FROM u WHERE u.k = t.a) AS g, COUNT(*) FROM {t} \
         GROUP BY ROLLUP(g) ORDER BY 1, 2",
        "SELECT g, COUNT(*) FROM (SELECT (SELECT COUNT(*) FROM u WHERE u.k = t.a) AS g FROM t) d \
         GROUP BY ROLLUP(g) ORDER BY 1, 2",
    ),
    (
        "SELECT a, GROUPING(a), (SELECT MAX(u.x) FROM u) FROM {t} GROUP BY ROLLUP(a) ORDER BY 1",
        "SELECT a, GROUPING(a), 6 FROM t GROUP BY ROLLUP(a) ORDER BY 1",
    ),
    (
        "SELECT DISTINCT (SELECT COUNT(*) FROM u WHERE u.k = t.a) FROM {t} GROUP BY a ORDER BY 1",
        "SELECT DISTINCT c FROM (SELECT (SELECT COUNT(*) FROM u WHERE u.k = t.a) AS c FROM t) d \
         ORDER BY 1",
    ),
    (
        "SELECT DISTINCT SUM(b) > (SELECT MAX(u.x) FROM u) * 20 FROM {t} GROUP BY a ORDER BY 1",
        "SELECT DISTINCT SUM(b) > 120 FROM t GROUP BY a ORDER BY 1",
    ),
    (
        "SELECT a, SUM((SELECT COUNT(*) FROM u WHERE u.k = t.b)) FROM {t} GROUP BY a \
         UNION ALL SELECT -1, 0 ORDER BY 1",
        "SELECT a, SUM(v) FROM (SELECT a, (SELECT COUNT(*) FROM u WHERE u.k = t.b) AS v FROM t) d \
         GROUP BY a UNION ALL SELECT -1, 0 ORDER BY 1",
    ),
    (
        "SELECT a, (SELECT COUNT(*) FROM u WHERE u.k = t.a) FROM {t} GROUP BY a \
         UNION SELECT a, 0 FROM t GROUP BY a ORDER BY 1, 2 LIMIT 5 OFFSET 3",
        "SELECT d.a, (SELECT COUNT(*) FROM u WHERE u.k = d.a) FROM (SELECT a FROM t GROUP BY a) d \
         UNION SELECT a, 0 FROM t GROUP BY a ORDER BY 1, 2 LIMIT 5 OFFSET 3",
    ),
    (
        "SELECT a, SUM(b) FROM {t} GROUP BY a HAVING SUM(b) > (SELECT MAX(u.x) FROM u) * 20 \
         INTERSECT SELECT a, SUM(b) FROM t GROUP BY a ORDER BY 1",
        "SELECT a, SUM(b) FROM t GROUP BY a HAVING SUM(b) > 120 \
         INTERSECT SELECT a, SUM(b) FROM t GROUP BY a ORDER BY 1",
    ),
    (
        "SELECT a, COUNT(*) FROM {t} GROUP BY a HAVING COUNT(*) > (SELECT MIN(u.x) FROM u) + 21 \
         EXCEPT SELECT 0, 22 ORDER BY 1",
        "SELECT a, COUNT(*) FROM t GROUP BY a HAVING COUNT(*) > 21 EXCEPT SELECT 0, 22 ORDER BY 1",
    ),
];

/// A table function's grouped statements, and each over a table of its rows
const FUNCTION_SITES: &[(&str, &str)] = &[
    (
        "SELECT value % 3, COUNT(*), (SELECT MAX(u.x) FROM u) FROM generate_series(1, 60) \
         GROUP BY value % 3 ORDER BY 1",
        "SELECT value % 3, COUNT(*), (SELECT MAX(u.x) FROM u) FROM s GROUP BY value % 3 ORDER BY 1",
    ),
    (
        "SELECT SUM((SELECT COUNT(*) FROM u WHERE u.k = g.value % 11)) FROM generate_series(1, 60) g",
        "SELECT SUM((SELECT COUNT(*) FROM u WHERE u.k = g.value % 11)) FROM s g",
    ),
    (
        "SELECT COUNT(*) FROM generate_series(1, 60) g \
         GROUP BY (SELECT COUNT(*) FROM u WHERE u.k = g.value % 11) ORDER BY 1",
        "SELECT COUNT(*) FROM s g GROUP BY (SELECT COUNT(*) FROM u WHERE u.k = g.value % 11) \
         ORDER BY 1",
    ),
];

/// The mechanisms a statement the blocks run never reaches
const GENERAL: &[GeneralPath] = &[
    GeneralPath::LiftedSubqueries,
    GeneralPath::SelectFold,
    GeneralPath::CorrelatedWhere,
    GeneralPath::OuterBinding,
    GeneralPath::OuterSubstitution,
    GeneralPath::ExistsSubquery,
    GeneralPath::InSubquery,
];

/// Those a grouped statement's own subqueries never reach, when the WHERE
/// of a join outside the join blocks reads its subqueries the general way
const GROUPED: &[GeneralPath] = &[GeneralPath::LiftedSubqueries, GeneralPath::SelectFold];

/// The mechanisms `sql` over `source` must not reach
fn general_for(source: &Source, site: &str) -> &'static [GeneralPath] {
    match source.where_blocks || !site.contains("{t} WHERE") {
        true => GENERAL,
        false => GROUPED,
    }
}

fn general_hits(general: &[GeneralPath]) -> Vec<usize> {
    general
        .iter()
        .map(|path| general_path_hits(*path))
        .collect()
}

/// Each row, without the column names
fn rows(db: &Database, sql: &str) -> Result<Vec<String>, String> {
    let result = db.query(sql, ()).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in result {
        let row = row.map_err(|e| e.to_string())?;
        out.push(
            (0..row.len())
                .map(|i| format!("{:?}", row.get_value(i).unwrap()))
                .collect::<Vec<_>>()
                .join("|"),
        );
    }
    Ok(out)
}

fn prepared_line(db: &Database, sql: &str) -> String {
    db.query(&format!("EXPLAIN {sql}"), ())
        .unwrap()
        .map(|row| row.unwrap().get::<String>(0).unwrap())
        .find(|line| line.trim_start().starts_with("Prepared:"))
        .map(|line| line.trim().to_string())
        .unwrap_or_default()
}

fn databases(dir: &tempfile::TempDir, name: &str) -> [(bool, Database); 2] {
    let hot = Database::open(&format!("memory://{name}")).unwrap();
    fill(&hot);
    let sealed = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    fill(&sealed);
    sealed.execute("PRAGMA CHECKPOINT", ()).unwrap();
    [(false, hot), (true, sealed)]
}

/// Runs `sql` and checks it answers `want` without reaching `general`
fn check(
    db: &Database,
    sql: &str,
    want: &[String],
    general: &[GeneralPath],
    wrong: &mut Vec<String>,
) {
    let line = prepared_line(db, sql);
    let before = general_hits(general);
    let got = rows(db, sql);
    let general: Vec<String> = general
        .iter()
        .zip(general_hits(general).iter().zip(&before))
        .filter(|(_, (after, before))| after != before)
        .map(|(path, _)| format!("{path:?}"))
        .collect();
    if got.as_deref() != Ok(want) || line != "Prepared: yes" || !general.is_empty() {
        wrong.push(format!(
            "{sql}\n  [{line}] general {general:?}\n  got {got:?}\n  want {want:?}"
        ));
    }
}

#[test]
fn every_source_answers_a_grouped_statement_as_its_table_does() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for (sealed, db) in databases(&dir, "grouped_sources_sites") {
        for site in SITES {
            let want = rows(&db, &site.replace("{t}", "t"))
                .unwrap_or_else(|e| panic!("{site} over t: {e}"));
            for source in sources() {
                let sql = format!("{}{}", source.with, site.replace("{t}", &source.from));
                let mut found = Vec::new();
                check(&db, &sql, &want, general_for(&source, site), &mut found);
                let name = source.name;
                wrong.extend(
                    found
                        .into_iter()
                        .map(|w| format!("sealed {sealed}, {name}: {w}")),
                );
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn every_source_answers_a_grouping_modifier_and_a_set_operation_as_their_twins() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for (sealed, db) in databases(&dir, "grouped_sources_shapes") {
        for (shape, twin) in SHAPES {
            let want = rows(&db, twin).unwrap_or_else(|e| panic!("{twin}: {e}"));
            for source in sources() {
                let sql = format!("{}{}", source.with, shape.replace("{t}", &source.from));
                let mut found = Vec::new();
                check(&db, &sql, &want, general_for(&source, shape), &mut found);
                let name = source.name;
                wrong.extend(
                    found
                        .into_iter()
                        .map(|w| format!("sealed {sealed}, {name}: {w}")),
                );
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_held_plan_reads_each_execution_s_parameters_over_every_source() {
    let _guard = serial();
    let db = Database::open("memory://grouped_sources_parameters").unwrap();
    fill(&db);
    for source in sources() {
        let sql = format!(
            "{}SELECT a, SUM(b) + $1, (SELECT COUNT(*) FROM u WHERE u.k = t.a + $2) FROM {} \
             GROUP BY a HAVING SUM(b) > (SELECT MAX(u.x) FROM u) * $3 ORDER BY a",
            source.with, source.from
        );
        let table = "SELECT a, SUM(b) + $1, (SELECT COUNT(*) FROM u WHERE u.k = t.a + $2) FROM t \
                     GROUP BY a HAVING SUM(b) > (SELECT MAX(u.x) FROM u) * $3 ORDER BY a";
        let read = |sql: &str, params: (i64, i64, i64)| -> Vec<String> {
            db.query(sql, params)
                .unwrap()
                .map(|row| {
                    let row = row.unwrap();
                    (0..row.len())
                        .map(|i| format!("{:?}", row.get_value(i).unwrap()))
                        .collect::<Vec<_>>()
                        .join("|")
                })
                .collect()
        };
        assert_eq!(prepared_line(&db, &sql), "Prepared: yes", "{}", source.name);
        for params in [(0, 0, 20), (100, 1, 21), (7, 2, 0)] {
            let before = general_hits(GROUPED);
            let got = read(&sql, params);
            assert_eq!(general_hits(GROUPED), before, "{}: {params:?}", source.name);
            assert_eq!(got, read(table, params), "{}: {params:?}", source.name);
        }
    }
}

#[test]
fn a_grouped_join_reads_the_transaction_s_own_writes() {
    let _guard = serial();
    let db = Database::open("memory://grouped_sources_transaction").unwrap();
    fill(&db);
    let mut tx = db.begin().unwrap();
    tx.execute("INSERT INTO u VALUES (100, 4, 99)", ()).unwrap();
    for sql in [
        "SELECT t.a, (SELECT MAX(u.x) FROM u WHERE u.k = t.a) FROM t JOIN w ON w.id = t.id \
         WHERE t.a = 4 GROUP BY t.a",
        "SELECT t.a, MAX((SELECT MAX(u.x) FROM u WHERE u.k = t.a)) FROM (SELECT id, a FROM t) t \
         WHERE t.a = 4 GROUP BY t.a",
    ] {
        assert_eq!(prepared_line(&db, sql), "Prepared: yes", "{sql}");
        let before = general_hits(GROUPED);
        let mut seen = tx.query(sql, ()).unwrap();
        let row = seen.next().unwrap().unwrap();
        assert_eq!(row.get::<i64>(1).unwrap(), 99, "{sql}");
        assert_eq!(general_hits(GROUPED), before, "{sql}");
    }
    tx.rollback().unwrap();
}

#[test]
fn a_grouped_statement_beside_a_with_it_does_not_read_runs_its_blocks() {
    let _guard = serial();
    let db = Database::open("memory://grouped_sources_with").unwrap();
    fill(&db);
    let sql = "WITH c AS (SELECT 1 AS one) SELECT a, SUM(b), \
               (SELECT COUNT(*) FROM u WHERE u.k = t.a) FROM t GROUP BY a ORDER BY a";
    let want = rows(
        &db,
        "SELECT a, SUM(b), (SELECT COUNT(*) FROM u WHERE u.k = t.a) FROM t GROUP BY a ORDER BY a",
    )
    .unwrap();
    let mut wrong = Vec::new();
    check(&db, sql, &want, GENERAL, &mut wrong);
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_key_from_an_untyped_column_reads_the_keyed_groups_only_when_its_type_matches() {
    let _guard = serial();
    let db = Database::open("memory://grouped_sources_untyped_keys").unwrap();
    fill(&db);
    let values = "(VALUES (1, 1), (2.0, 2), ('3', 3), (NULL, 4), (5, 5)) d(a, n)";
    let literals = ["1", "2.0", "'3'", "NULL", "5"];
    let sorted = |sql: &str| {
        let mut got = rows(&db, sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
        got.sort();
        got
    };
    let union = |each: &dyn Fn(usize, &str) -> String| {
        literals
            .iter()
            .enumerate()
            .map(|(i, lit)| each(i + 1, lit))
            .collect::<Vec<_>>()
            .join(" UNION ALL ")
    };
    let keyed_before = prepared_strategy_runs(PreparedStrategy::KeyedGroups);
    let grouped_before = prepared_strategy_runs(PreparedStrategy::Grouped);
    assert_eq!(
        sorted(&format!(
            "SELECT d.n, (SELECT COUNT(*) FROM u WHERE u.k = d.a) FROM {values} GROUP BY d.n, d.a"
        )),
        sorted(&union(&|n, lit| format!(
            "SELECT {n}, (SELECT COUNT(*) FROM u WHERE u.k = {lit})"
        ))),
        "an aggregate block"
    );
    assert!(
        prepared_strategy_runs(PreparedStrategy::KeyedGroups) > keyed_before,
        "an INTEGER key reads the keyed groups"
    );
    assert!(
        prepared_strategy_runs(PreparedStrategy::Grouped) > grouped_before,
        "a FLOAT or TEXT key is asked without the keys"
    );
    assert_eq!(
        sorted(&format!(
            "SELECT d.n FROM {values} WHERE EXISTS (SELECT 1 FROM u WHERE u.k = d.a)"
        )),
        sorted(&union(&|n, lit| format!(
            "SELECT {n} WHERE EXISTS (SELECT 1 FROM u WHERE u.k = {lit})"
        ))),
        "an EXISTS block"
    );
    assert_eq!(
        sorted(&format!(
            "SELECT d.n FROM {values} WHERE d.n IN (SELECT u.x FROM u WHERE u.k = d.a)"
        )),
        sorted(&union(&|n, lit| format!(
            "SELECT {n} WHERE {n} IN (SELECT u.x FROM u WHERE u.k = {lit})"
        ))),
        "an IN block"
    );
}

#[test]
fn a_group_by_position_reads_the_source_column_it_names() {
    let _guard = serial();
    let db = Database::open("memory://grouped_sources_position").unwrap();
    fill(&db);
    let want = rows(
        &db,
        "SELECT a, (SELECT MAX(u.x) FROM u WHERE u.k = t.a) FROM t WHERE a >= 0 GROUP BY 1 \
         ORDER BY 1",
    )
    .unwrap();
    let mut wrong = Vec::new();
    for from in [
        "(SELECT a, id FROM t) t",
        "(SELECT a AS a FROM t) t",
        "t JOIN w ON w.id = t.id",
    ] {
        let sql = format!(
            "SELECT t.a, (SELECT MAX(u.x) FROM u WHERE u.k = t.a) FROM {from} WHERE t.a >= 0 \
             GROUP BY 1 ORDER BY 1"
        );
        check(&db, &sql, &want, GENERAL, &mut wrong);
    }
    let want = rows(
        &db,
        "SELECT g.value, (SELECT MAX(u.x) FROM u WHERE u.k = g.value) FROM s g GROUP BY 1 \
         ORDER BY 1",
    )
    .unwrap();
    check(
        &db,
        "SELECT g.value, (SELECT MAX(u.x) FROM u WHERE u.k = g.value) \
         FROM generate_series(1, 60) g GROUP BY 1 ORDER BY 1",
        &want,
        GENERAL,
        &mut wrong,
    );
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_source_read_for_blocks_that_cannot_run_them_is_not_read_again() {
    let _guard = serial();
    let db = Database::open("memory://grouped_sources_read_once").unwrap();
    fill(&db);
    // The block reads g.value, which the GROUP BY does not name
    let grouped = "GROUP BY g.value % 7 ORDER BY 1";
    let select = "SELECT g.value % 7, (SELECT MAX(u.x) FROM u WHERE u.k = g.value)";
    let want = rows(&db, &format!("{select} FROM s g {grouped}"));
    let sql = format!("{select} FROM generate_series(1, 60) g {grouped}");
    for run in 0..2 {
        let before = tvf_rows();
        assert_eq!(rows(&db, &sql), want, "run {run}");
        assert_eq!(
            tvf_rows() - before,
            60,
            "run {run}: the series is generated once"
        );
    }
}

#[test]
fn a_join_scope_keeps_a_column_whose_name_holds_a_dot() {
    let _guard = serial();
    let db = Database::open("memory://grouped_sources_dotted").unwrap();
    fill(&db);
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY, \"x.y\" INTEGER, k INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY)",
        "INSERT INTO l SELECT g.value, g.value * 2, g.value % 11 FROM generate_series(1, 30) g",
        "INSERT INTO r SELECT g.value FROM generate_series(1, 30) g",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let want = rows(
        &db,
        "SELECT SUM((SELECT u.x FROM u WHERE u.id = l.k + 1)) FROM l",
    )
    .unwrap();
    let mut wrong = Vec::new();
    check(
        &db,
        "SELECT SUM((SELECT u.x FROM u WHERE u.id = l.k + 1)) FROM l JOIN r ON l.id = r.id",
        &want,
        GENERAL,
        &mut wrong,
    );
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_group_column_named_like_a_qualified_one_reads_its_own_cell() {
    let _guard = serial();
    let db = Database::open("memory://grouped_sources_qualified_name").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, \"t.a\" INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, x INTEGER)",
        "INSERT INTO t VALUES (1, 1, 2)",
        "INSERT INTO r VALUES (1)",
        "INSERT INTO u VALUES (1, 1, 10), (2, 2, 20)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut wrong = Vec::new();
    for read in ["t.\"t.a\"", "\"t.a\""] {
        check(
            &db,
            &format!(
                "SELECT t.a, (SELECT MAX(u.x) FROM u WHERE u.k = {read}) FROM t \
                 JOIN r ON r.id = t.id GROUP BY t.a, t.\"t.a\""
            ),
            &["Integer(1)|Integer(20)".to_string()],
            GENERAL,
            &mut wrong,
        );
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_join_scope_keeps_an_alias_another_alias_begins_with() {
    let _guard = serial();
    let db = Database::open("memory://grouped_sources_alias_prefix").unwrap();
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE TABLE q (id INTEGER PRIMARY KEY)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO l VALUES (1)",
        "INSERT INTO r VALUES (1, 7)",
        "INSERT INTO q VALUES (1)",
        "INSERT INTO s VALUES (7, 42)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    let mut wrong = Vec::new();
    check(
        &db,
        "SELECT SUM((SELECT s.a FROM s WHERE s.id = \"a.b\".k)) \
         FROM l a CROSS JOIN r \"a.b\" CROSS JOIN q c",
        &["Integer(42)".to_string()],
        GENERAL,
        &mut wrong,
    );
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_rolled_up_column_reads_the_cell_of_the_item_naming_it() {
    let _guard = serial();
    let db = Database::open("memory://grouped_sources_rollup_alias").unwrap();
    for sql in [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER)",
        "CREATE TABLE u (id INTEGER PRIMARY KEY, k INTEGER, x INTEGER)",
        "INSERT INTO t VALUES (1, 1)",
        "INSERT INTO u VALUES (1, 1, 10)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    // A subtotal reads the cell the output reads for t.a, NULL where it rolled up
    for (sets, tens) in [
        ("ROLLUP(a, g)", 2),
        ("ROLLUP(g, a)", 1),
        ("ROLLUP(a, t.a)", 1),
        ("ROLLUP(t.a, a)", 2),
    ] {
        let mut want = rows(
            &db,
            &format!(
                "SELECT a AS g, COUNT(*), CASE WHEN t.a = 1 THEN 10 END FROM t GROUP BY {sets}"
            ),
        )
        .unwrap();
        want.sort();
        let sql = format!(
            "SELECT a AS g, COUNT(*), (SELECT MAX(u.x) FROM u WHERE u.k = t.a) FROM t \
             GROUP BY {sets}"
        );
        let mut got = rows(&db, &sql).unwrap();
        got.sort();
        assert_eq!(got, want, "{sql}");
        let read = got
            .iter()
            .filter(|row| row.ends_with("Integer(10)"))
            .count();
        assert_eq!(read, tens, "{sql}: the rows whose a is 1");
        assert_eq!(prepared_line(&db, &sql), "Prepared: yes");
    }
    // A computed correlation reads the item its column reads alone
    db.execute("INSERT INTO u VALUES (2, 2, 20)", ()).unwrap();
    for sets in [
        "ROLLUP(a, t.a)",
        "ROLLUP(t.a, a)",
        "ROLLUP(a, g)",
        "ROLLUP(g, a)",
    ] {
        let mut want = rows(
            &db,
            &format!(
                "SELECT a AS g, COUNT(*), CASE WHEN t.a + 1 = 2 THEN 20 END FROM t GROUP BY {sets}"
            ),
        )
        .unwrap();
        want.sort();
        for test in ["u.k = t.a + 1", "u.k - 1 = t.a"] {
            let sql = format!(
                "SELECT a AS g, COUNT(*), (SELECT MAX(u.x) FROM u WHERE {test}) FROM t \
                 GROUP BY {sets}"
            );
            let mut got = rows(&db, &sql).unwrap();
            got.sort();
            assert_eq!(got, want, "{sql}");
        }
    }
    // A bare name reads the item grouping the expression it names, through
    // an alias or a position, before one that names its column
    for sets in ["ROLLUP(g, t.a)", "ROLLUP(1, t.a)"] {
        let mut want = rows(
            &db,
            &format!("SELECT a AS g, COUNT(*), CASE WHEN a = 1 THEN 10 END FROM t GROUP BY {sets}"),
        )
        .unwrap();
        want.sort();
        let sql = format!(
            "SELECT a AS g, COUNT(*), (SELECT MAX(u.x) FROM u WHERE u.k = a) FROM t \
             GROUP BY {sets}"
        );
        let mut got = rows(&db, &sql).unwrap();
        got.sort();
        assert_eq!(got, want, "{sql}");
    }
}

#[test]
fn a_join_of_two_derived_tables_groups_by_a_subquery() {
    let _guard = serial();
    let db = Database::open("memory://grouped_sources_two_derived").unwrap();
    for sql in [
        "CREATE TABLE l (id INTEGER PRIMARY KEY, k INTEGER)",
        "CREATE TABLE r (id INTEGER PRIMARY KEY)",
        "CREATE TABLE s (id INTEGER PRIMARY KEY, a INTEGER)",
        "INSERT INTO l VALUES (1, 7)",
        "INSERT INTO r VALUES (1)",
        "INSERT INTO s VALUES (7, 42)",
    ] {
        db.execute(sql, ()).unwrap();
    }
    db.execute("INSERT INTO s VALUES (1, 41)", ()).unwrap();
    let mut wrong = Vec::new();
    check(
        &db,
        "SELECT COUNT(*) FROM (SELECT id, k FROM l) a CROSS JOIN (SELECT id FROM r) b \
         GROUP BY (SELECT s.a FROM s WHERE s.id = a.k)",
        &["Integer(1)".to_string()],
        GENERAL,
        &mut wrong,
    );
    // Dotted names the text splits two ways; the join read its sides apart
    check(
        &db,
        "SELECT MAX(\"a.b\".id), (SELECT s.a FROM s WHERE s.id = a.id) \
         FROM (SELECT id, k AS \"b.k\" FROM l) a CROSS JOIN (SELECT id FROM r) \"a.b\" \
         GROUP BY (SELECT s.a FROM s WHERE s.id = a.id)",
        &["Integer(1)|Integer(41)".to_string()],
        GENERAL,
        &mut wrong,
    );
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_grouped_output_outside_the_blocks_beside_join_where_blocks_runs() {
    let _guard = serial();
    let db = Database::open("memory://grouped_sources_join_where").unwrap();
    fill(&db);
    let tail = "WHERE EXISTS (SELECT 1 FROM u WHERE u.id = t.id)";
    let select = "SELECT SUM((SELECT d.x FROM (SELECT k, x FROM u) d WHERE d.k = t.a LIMIT 1))";
    assert_eq!(
        rows(
            &db,
            &format!("{select} FROM t JOIN w ON w.id = t.id {tail}")
        ),
        rows(&db, &format!("{select} FROM t {tail}")),
    );
}

#[test]
fn a_table_function_answers_a_grouped_statement_as_a_table_of_its_rows() {
    let _guard = serial();
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for (sealed, db) in databases(&dir, "grouped_sources_function") {
        for (sql, twin) in FUNCTION_SITES {
            let want = rows(&db, twin).unwrap_or_else(|e| panic!("{twin}: {e}"));
            let mut found = Vec::new();
            check(&db, sql, &want, GENERAL, &mut found);
            wrong.extend(found.into_iter().map(|w| format!("sealed {sealed}: {w}")));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
