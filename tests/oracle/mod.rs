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

//! Recorded answers for tests whose statements no other path answers: each
//! statement's row count and a digest of its rows in the order given, one
//! per line of a `.answers` file here; a test that ignores order sorts its
//! rows first. `STOOLAP_RECORD_ORACLE=1` rewrites the files from the
//! answers given, so a meant change shows in their diff

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::Mutex;

use stoolap::Database;

/// A test's recorded answers: the file's text as built, and its path. Each
/// test has a file of its own, since recording rewrites it per process
pub struct Answers {
    pub text: &'static str,
    pub path: &'static str,
}

/// The `Answers` of `tests/oracle/<name>.answers`
#[macro_export]
macro_rules! answers {
    ($name:literal) => {
        oracle::Answers {
            text: include_str!(concat!("oracle/", $name, ".answers")),
            path: concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/oracle/",
                $name,
                ".answers"
            ),
        }
    };
}

/// The row count and the FNV-1a of the rows in order, each ended by a
/// newline: no Rust version changes it
pub fn digest(rows: &[String]) -> (usize, u64) {
    let bytes = rows.iter().flat_map(|row| row.bytes().chain(*b"\n"));
    let hash = bytes.fold(0xcbf2_9ce4_8422_2325, |hash: u64, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    });
    (rows.len(), hash)
}

/// Whether answers are recorded rather than checked
pub fn recording() -> bool {
    std::env::var("STOOLAP_RECORD_ORACLE").is_ok_and(|v| v == "1")
}

/// The answers recorded so far in this process, by file
static RECORDED: Mutex<BTreeMap<&'static str, BTreeMap<String, String>>> =
    Mutex::new(BTreeMap::new());

/// One file line: the count, the digest and the statement
fn line(sql: &str, (count, sum): (usize, u64)) -> String {
    format!("{count}\t{sum:016x}\t{sql}")
}

/// What is wrong with `rows` as the answer to `sql` that `answers` records,
/// None when it matches; when recording, the answer is written instead
pub fn check(answers: &Answers, sql: &str, rows: &[String]) -> Option<String> {
    assert!(!sql.contains(['\t', '\n']), "{sql:?} cannot be one line");
    let given = line(sql, digest(rows));
    if recording() {
        let mut recorded = RECORDED.lock().unwrap_or_else(|e| e.into_inner());
        let file = recorded.entry(answers.path).or_default();
        if let Some(earlier) = file.insert(sql.to_string(), given.clone()) {
            assert_eq!(earlier, given, "{sql} answered two ways while recording");
        }
        let text: String = file.values().map(|line| format!("{line}\n")).collect();
        std::fs::write(answers.path, text).unwrap();
        return None;
    }
    let recorded = answers
        .text
        .lines()
        .find(|line| line.splitn(3, '\t').nth(2) == Some(sql));
    match recorded {
        Some(recorded) if recorded == given => None,
        Some(recorded) => Some(format!(
            "{sql}: gave {rows:?}, recorded as {}",
            &recorded[..recorded.len() - sql.len()]
        )),
        None => Some(format!("{sql}: no recorded answer, gave {rows:?}")),
    }
}

/// `check` of an answer that may be an error, which is recorded as its text
pub fn check_answer(
    answers: &Answers,
    sql: &str,
    answer: &Result<Vec<String>, String>,
) -> Option<String> {
    match answer {
        Ok(rows) => check(answers, sql, rows),
        Err(e) => check(answers, sql, &[format!("error: {e}")]),
    }
}

/// `check_answer`, recording `answer` only when `confirm`, another path's
/// answer to the same statement, gives the same rows
pub fn check_confirmed(
    answers: &Answers,
    sql: &str,
    answer: &Result<Vec<String>, String>,
    confirm: impl FnOnce() -> Result<Vec<String>, String>,
) -> Option<String> {
    if recording() {
        let other = confirm();
        assert_eq!(
            &other, answer,
            "{sql}: another path disagrees, not recorded"
        );
    }
    check_answer(answers, sql, answer)
}

/// Where `keyword` stands in `sql` outside every parenthesis and quote
fn top_level(sql: &str, keyword: &str) -> Option<usize> {
    let (mut depth, mut quoted) = (0i32, false);
    for (at, c) in sql.char_indices() {
        match c {
            '\'' => quoted = !quoted,
            '(' if !quoted => depth += 1,
            ')' if !quoted => depth -= 1,
            _ if !quoted && depth == 0 && sql[at..].starts_with(keyword) => return Some(at),
            _ => {}
        }
    }
    None
}

/// `sql` with an unused named window, which no prepared family takes: the
/// general path answers it for as long as it lives
pub fn general_twin(sql: &str) -> String {
    let at = [" ORDER BY ", " LIMIT ", " OFFSET "]
        .iter()
        .filter_map(|keyword| top_level(sql, keyword))
        .min()
        .unwrap_or(sql.len());
    format!("{} WINDOW oracle_unused AS (){}", &sql[..at], &sql[at..])
}

/// A value written as a literal that reads back as the same value
fn literal(value: &stoolap::core::Value) -> String {
    use stoolap::core::Value;
    match value {
        Value::Null(_) => "NULL".to_string(),
        Value::Integer(n) => n.to_string(),
        Value::Float(f) => format!("{f:?}"),
        Value::Boolean(b) => b.to_string(),
        Value::Text(s) => format!("'{}'", s.replace('\'', "''")),
        other => panic!("no literal for {other:?}"),
    }
}

/// `sql` with the uncorrelated subquery `subquery`, written in it as
/// `(subquery)`, replaced by its answer: the one value it gives, NULL for
/// none, or with `list` every value as an `IN` list, `()` for none
pub fn literal_twin(db: &Database, sql: &str, subquery: &str, list: bool) -> String {
    let values: Vec<String> = db
        .query(subquery, ())
        .unwrap_or_else(|e| panic!("{subquery}: {e}"))
        .map(|row| literal(row.unwrap().get_value(0).unwrap()))
        .collect();
    let written = format!("({subquery})");
    assert!(sql.contains(&written), "{sql} holds no {written}");
    let answer = match (list, values.as_slice()) {
        (true, values) => format!("({})", values.join(", ")),
        (false, []) => "(NULL)".to_string(),
        (false, [value]) => format!("({value})"),
        (false, _) => panic!("{subquery} gives more than one row"),
    };
    sql.replace(&written, &answer)
}
