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

//! An argument a call reads at a row reads that row: NTILE's bucket count
//! at the first row of each partition, LAG and LEAD's offset and
//! NTH_VALUE's n at each row, an aggregate's arguments after the first at
//! each input row; a frame offset is read once and may not read the row

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Mutex, MutexGuard};

use stoolap::core::{Result, Value};
use stoolap::functions::registry::global_registry;
use stoolap::functions::{
    FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, ScalarFunction,
};
use stoolap::Database;

static TICKS: AtomicI64 = AtomicI64::new(0);
static SERIAL: Mutex<()> = Mutex::new(());

/// 1 on its first call, then 2, 3 and on, answering anew per call
#[derive(Default)]
struct Tick;

impl ScalarFunction for Tick {
    fn name(&self) -> &str {
        "ROW_ARGUMENTS_TICK"
    }
    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "ROW_ARGUMENTS_TICK",
            FunctionType::Scalar,
            "test counter",
            FunctionSignature::new(FunctionDataType::Integer, vec![], 0, 0),
        )
        .non_deterministic()
    }
    fn evaluate(&self, _args: &[Value]) -> Result<Value> {
        Ok(Value::Integer(TICKS.fetch_add(1, Ordering::Relaxed) + 1))
    }
    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(Tick)
    }
}

/// t as rows of (id, g, x, s, n), in memory and sealed into a volume
fn databases(name: &str, dir: &tempfile::TempDir) -> [Database; 2] {
    let hot = Database::open(&format!("memory://row_arguments_{name}")).unwrap();
    let sealed = Database::open(&format!("file://{}", dir.path().display())).unwrap();
    for db in [&hot, &sealed] {
        for sql in [
            "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, x TEXT, s TEXT, n INTEGER)",
            "INSERT INTO t VALUES (1, 1, 'a', '-', 1), (2, 1, 'b', '+', 2), \
             (3, 1, 'c', NULL, 1), (4, 2, 'd', '-', 3), (5, 2, 'e', '+', 1), \
             (6, 2, 'f', '+', 2)",
        ] {
            db.execute(sql, ()).unwrap();
        }
    }
    sealed.execute("PRAGMA CHECKPOINT", ()).unwrap();
    [hot, sealed]
}

/// The rows `rows` gives, each its cells joined by `|`, NULL as written
fn shown(rows: stoolap::Result<stoolap::Rows>) -> std::result::Result<String, String> {
    let mut out = Vec::new();
    for row in rows.map_err(|e| e.to_string())? {
        let row = row.map_err(|e| e.to_string())?;
        let cells: Vec<String> = (0..row.len())
            .map(|i| match row.get_value(i).unwrap() {
                Value::Null(_) => "NULL".to_string(),
                Value::Text(text) => text.to_string(),
                other => other.to_string(),
            })
            .collect();
        out.push(cells.join("|"));
    }
    Ok(out.join(" "))
}

/// Each way `sql` runs with `params`: as a query, and twice as one
/// prepared statement
fn runs(db: &Database, sql: &str, params: &[Value]) -> Vec<std::result::Result<String, String>> {
    let mut answers = vec![shown(db.query(sql, params))];
    match db.prepare(sql) {
        Ok(stmt) => {
            for _ in 0..2 {
                answers.push(shown(stmt.query(params)));
            }
        }
        Err(e) => answers.push(Err(e.to_string())),
    }
    answers
}

const STATEMENTS: &[(&str, &str)] = &[
    (
        "SELECT id, NTILE(n) OVER (PARTITION BY g ORDER BY id) FROM t ORDER BY id",
        "1|1 2|1 3|1 4|1 5|2 6|3",
    ),
    (
        "SELECT id, NTILE((SELECT COUNT(*) FROM t u WHERE u.g = t.g AND u.id <= t.id) + 1) \
         OVER (PARTITION BY g ORDER BY id) FROM t ORDER BY id",
        "1|1 2|1 3|2 4|1 5|1 6|2",
    ),
    (
        "SELECT id, LAG(x, n) OVER (ORDER BY id) FROM t ORDER BY id",
        "1|NULL 2|NULL 3|b 4|a 5|d 6|d",
    ),
    (
        "SELECT id, LEAD(x, n, s) OVER (ORDER BY id) FROM t ORDER BY id",
        "1|b 2|d 3|d 4|- 5|f 6|+",
    ),
    (
        "SELECT id, NTH_VALUE(x, n) OVER (PARTITION BY g ORDER BY id \
         ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) FROM t ORDER BY id",
        "1|a 2|b 3|a 4|f 5|d 6|e",
    ),
    (
        "SELECT id, NTH_VALUE(x, n) OVER (ORDER BY id) FROM t ORDER BY id",
        "1|a 2|b 3|a 4|c 5|a 6|b",
    ),
    (
        "SELECT g, NTILE(MAX(n)) OVER (ORDER BY g) FROM t GROUP BY g ORDER BY g",
        "1|1 2|2",
    ),
    (
        "SELECT g, LAG(g, MIN(n)) OVER (ORDER BY g) FROM t GROUP BY g ORDER BY g",
        "1|NULL 2|1",
    ),
    (
        "SELECT g, STRING_AGG(x, s ORDER BY id) FROM t GROUP BY g ORDER BY g",
        "1|a+bc 2|d+e+f",
    ),
    (
        "SELECT g, STRING_AGG(x, s ORDER BY id DESC) FROM t GROUP BY g ORDER BY g",
        "1|c+b-a 2|f+e-d",
    ),
    (
        "SELECT g, STRING_AGG(x, CAST(g AS TEXT) ORDER BY id) FROM t GROUP BY g ORDER BY g",
        "1|a1b1c 2|d2e2f",
    ),
    ("SELECT STRING_AGG(x, s ORDER BY id) FROM t", "a+bc-d+e+f"),
    ("SELECT STRING_AGG(x, NULL ORDER BY id) FROM t", "abcdef"),
    (
        "SELECT STRING_AGG(x, s ORDER BY id) FILTER (WHERE id <> 2) FROM t",
        "ac-d+e+f",
    ),
    (
        "SELECT STRING_AGG(DISTINCT CAST(g AS TEXT), s) FROM t",
        "1+11-2+2",
    ),
    (
        "SELECT g FROM t GROUP BY g HAVING STRING_AGG(x, s ORDER BY id) = 'a+bc'",
        "1",
    ),
    (
        "SELECT id, (SELECT STRING_AGG(u.x, u.s ORDER BY u.id) FROM t u WHERE u.g = t.g) \
         FROM t ORDER BY id",
        "1|a+bc 2|a+bc 3|a+bc 4|d+e+f 5|d+e+f 6|d+e+f",
    ),
    (
        "SELECT id, STRING_AGG(x, s) OVER (PARTITION BY g ORDER BY id) FROM t ORDER BY id",
        "1|a 2|a+b 3|a+bc 4|d 5|d+e 6|d+e+f",
    ),
    (
        "SELECT id, STRING_AGG(x, s) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) \
         FROM t ORDER BY id",
        "1|a 2|a+b 3|bc 4|c-d 5|d+e 6|e+f",
    ),
    (
        "SELECT id, STRING_AGG(x, s) FILTER (WHERE id <> 2) OVER (ORDER BY id) FROM t ORDER BY id",
        "1|a 2|a 3|ac 4|ac-d 5|ac-d+e 6|ac-d+e+f",
    ),
    (
        "SELECT id, STRING_AGG(x, s ORDER BY id DESC) OVER (PARTITION BY g) FROM t ORDER BY id",
        "1|c+b-a 2|c+b-a 3|c+b-a 4|f+e-d 5|f+e-d 6|f+e-d",
    ),
    (
        "SELECT id, SUM(n) OVER (ORDER BY id ROWS BETWEEN n PRECEDING AND CURRENT ROW) FROM t",
        "error: a frame offset must not read the row",
    ),
    (
        "SELECT id, LAG(x, n - 2) OVER (ORDER BY id) FROM t",
        "error: the offset of LAG must be an integer of at least 0, got -1",
    ),
];

/// Statements with one parameter, each with the value bound and the answer
fn with_parameter() -> [(&'static str, Value, &'static str); 3] {
    [
        (
            "SELECT g, STRING_AGG(x, $1 || s ORDER BY id) FROM t GROUP BY g ORDER BY g",
            Value::text("*"),
            "1|a*+bc 2|d*+e*+f",
        ),
        (
            "SELECT g, STRING_AGG(x, $1 || s ORDER BY id) FROM t GROUP BY g ORDER BY g",
            Value::text("1"),
            "1|a1+bc 2|d1+e1+f",
        ),
        (
            "SELECT id, LAG(x, n - $1) OVER (ORDER BY id) FROM t ORDER BY id",
            Value::Integer(1),
            "1|a 2|a 3|c 4|b 5|e 6|e",
        ),
    ]
}

/// Whether `got` is the written answer `want`: an error holding the text
/// after `error: `, or the rows as shown
fn is(got: &std::result::Result<String, String>, want: &str) -> bool {
    match (got, want.strip_prefix("error")) {
        (Err(e), Some(text)) => e.contains(text.trim_start_matches(": ")),
        (Ok(rows), None) => rows == want,
        _ => false,
    }
}

#[test]
fn an_argument_read_at_a_row_reads_that_row() {
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for (db, at) in databases("answers", &dir).iter().zip(["hot", "sealed"]) {
        for (sql, want) in STATEMENTS {
            for got in runs(db, sql, &[]) {
                if !is(&got, want) {
                    wrong.push(format!("{at}: {sql}\n  got {got:?}\n  want {want}"));
                }
            }
        }
        for (sql, param, want) in with_parameter() {
            for got in runs(db, sql, std::slice::from_ref(&param)) {
                if !is(&got, want) {
                    wrong.push(format!(
                        "{at}: {sql} with {param}\n  got {got:?}\n  want {want}"
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Each statement with the counter, its answer and the calls one run makes
const DRAWS: &[(&str, &str, i64)] = &[
    (
        "SELECT id, NTILE(ROW_ARGUMENTS_TICK()) OVER (PARTITION BY g ORDER BY id) \
         FROM t ORDER BY id",
        "1|1 2|1 3|1 4|1 5|1 6|2",
        2,
    ),
    (
        "SELECT id, LAG(x, ROW_ARGUMENTS_TICK() % 2) OVER (ORDER BY id) FROM t ORDER BY id",
        "1|NULL 2|b 3|b 4|d 5|d 6|f",
        6,
    ),
    (
        "SELECT id, NTH_VALUE(x, ROW_ARGUMENTS_TICK()) OVER (ORDER BY id \
         ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) FROM t ORDER BY id",
        "1|a 2|b 3|c 4|d 5|e 6|f",
        6,
    ),
    (
        "SELECT id, COUNT(*) OVER (ORDER BY id \
         ROWS BETWEEN ROW_ARGUMENTS_TICK() PRECEDING AND CURRENT ROW) FROM t ORDER BY id",
        "1|1 2|2 3|2 4|2 5|2 6|2",
        1,
    ),
    (
        "SELECT g, STRING_AGG(x, CAST(ROW_ARGUMENTS_TICK() AS TEXT) ORDER BY id) \
         FROM t GROUP BY g ORDER BY g",
        "1|a2b3c 2|d5e6f",
        6,
    ),
    (
        "SELECT STRING_AGG(x, CAST(ROW_ARGUMENTS_TICK() AS TEXT) ORDER BY id) \
         FILTER (WHERE g = 2) FROM t",
        "d2e3f",
        3,
    ),
    (
        "SELECT id, STRING_AGG(x, CAST(ROW_ARGUMENTS_TICK() AS TEXT) ORDER BY id) \
         FILTER (WHERE g = 2) OVER () FROM t ORDER BY id",
        "1|d2e3f 2|d2e3f 3|d2e3f 4|d2e3f 5|d2e3f 6|d2e3f",
        3,
    ),
];

#[test]
fn a_function_drawing_per_call_draws_where_its_argument_is_read() {
    let _guard: MutexGuard<'_, ()> = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    global_registry().register_scalar::<Tick>().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mut wrong = Vec::new();
    for (db, at) in databases("draws", &dir).iter().zip(["hot", "sealed"]) {
        for (sql, want, draws) in DRAWS {
            let stmt = db.prepare(sql).unwrap();
            for run in 0..3 {
                TICKS.store(0, Ordering::Relaxed);
                let got = match run {
                    0 => shown(db.query(sql, ())),
                    _ => shown(stmt.query(())),
                };
                let ticks = TICKS.load(Ordering::Relaxed);
                if !is(&got, want) || ticks != *draws {
                    wrong.push(format!(
                        "{at} run {run}: {sql}\n  got {got:?} after {ticks} calls, \
                         want {want} after {draws}"
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
