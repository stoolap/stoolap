// Copyright 2026 Stoolap Contributors
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

//! Agent mode: the build line, one JSON line per statement, the summary and
//! the exit code

use std::io::Write;
use std::process::{Command, Stdio};

/// stdout, stderr and the exit code of the CLI run with `args`, `env` and
/// `input`
fn run(args: &[&str], env: &[(&str, &str)], input: &str) -> (String, String, i32) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_stoolap"))
        .args(args)
        .env_remove("STOOLAP_AGENT")
        .envs(env.iter().copied())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    (
        String::from_utf8(out.stdout).unwrap(),
        String::from_utf8(out.stderr).unwrap(),
        out.status.code().unwrap(),
    )
}

/// The lines after the build line
fn agent(args: &[&str], input: &str) -> (Vec<String>, i32) {
    let mut all = vec!["--agent"];
    all.extend_from_slice(args);
    let (out, err, code) = run(&all, &[], input);
    assert_eq!(err, "", "nothing on stderr");
    let lines = out.lines().skip(1).map(str::to_string).collect();
    (lines, code)
}

#[test]
fn the_first_line_names_the_build() {
    let (out, _, code) = run(&["--agent", "-e", "SELECT 1"], &[], "");
    let build: serde_json::Value = serde_json::from_str(out.lines().next().unwrap()).unwrap();
    let kind = match cfg!(debug_assertions) {
        true => "debug",
        false => "release",
    };
    assert_eq!(
        (
            build["stoolap"].as_str(),
            build["build"].as_str(),
            build["dsn"].as_str(),
            build["commit"].is_string(),
            build["features"].is_array(),
            code
        ),
        (
            Some(env!("CARGO_PKG_VERSION")),
            Some(kind),
            Some("memory://"),
            true,
            true,
            0
        ),
        "{out}"
    );
}

#[test]
fn every_statement_gives_one_line_and_the_run_a_summary() {
    let script =
        "CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT, f FLOAT, j JSON, ts TIMESTAMP);\n\
                  INSERT INTO t VALUES (1, 'NULL', 0.1, '{\"a\":[1,2]}', '2024-01-01 10:00:00'),\n\
                  (2, NULL, CAST('NaN' AS FLOAT), NULL, NULL);\n\
                  SELECT * FROM t ORDER BY id;\n\
                  SELECT CASE WHEN id = 1 THEN 1 ELSE 'x' END AS m, NULL AS n FROM t ORDER BY id;\n\
                  SELECT nope FROM t;\n\
                  SELECT 1; SELECT * FROM t WHERE;\n";
    let (lines, code) = agent(&[], script);
    assert_eq!(
        (lines, code),
        (
            vec![
                r#"{"stmt":1,"line":1,"ok":true,"rows_affected":0}"#.to_string(),
                r#"{"stmt":2,"line":2,"ok":true,"rows_affected":2}"#.to_string(),
                r#"{"stmt":3,"line":4,"ok":true,"columns":["id","s","f","j","ts"],"types":["INTEGER","TEXT","FLOAT","JSON","TIMESTAMP"],"rows":[[1,"NULL",0.1,{"a":[1,2]},"2024-01-01T10:00:00Z"],[2,null,"NaN",null,null]],"row_count":2}"#.to_string(),
                r#"{"stmt":4,"line":5,"ok":true,"columns":["m","n"],"types":["MIXED",null],"rows":[[{"t":"INTEGER","v":1},null],[{"t":"TEXT","v":"x"},null]],"row_count":2}"#.to_string(),
                r#"{"stmt":5,"line":6,"ok":false,"error":{"kind":"column_not_found","message":"column 'nope' not found"}}"#.to_string(),
                r#"{"stmt":6,"line":7,"ok":true,"columns":["1"],"types":["INTEGER"],"rows":[[1]],"row_count":1}"#.to_string(),
                r#"{"stmt":7,"line":7,"ok":false,"error":{"kind":"parse","message":"parse error: expected expression after WHERE at position line 1, column 22","line":7,"column":32}}"#.to_string(),
                r#"{"done":true,"statements":7,"failed":2,"first_failed":5,"open_transaction":false}"#.to_string(),
            ],
            1
        )
    );
}

#[test]
fn a_cut_says_what_it_left_out() {
    let (lines, _) = agent(
        &["-l", "2"],
        "SELECT * FROM GENERATE_SERIES(1, 5) g;\nSELECT REPEAT('x', 1001) AS s;\n",
    );
    assert_eq!(
        lines[0],
        r#"{"stmt":1,"line":1,"ok":true,"columns":["value"],"types":["INTEGER"],"rows":[[1],[2]],"row_count":5,"shown":2,"truncated":true}"#
    );
    let cut = format!(
        r#"{{"stmt":2,"line":2,"ok":true,"columns":["s"],"types":["TEXT"],"rows":[["{}"]],"row_count":1,"cells_truncated":true}}"#,
        "x".repeat(1000)
    );
    assert_eq!(lines[1], cut);
    let (full, _) = agent(&["--full"], "SELECT REPEAT('x', 1001) AS s;\n");
    assert_eq!(
        full[0],
        format!(
            r#"{{"stmt":1,"line":1,"ok":true,"columns":["s"],"types":["TEXT"],"rows":[["{}"]],"row_count":1}}"#,
            "x".repeat(1001)
        )
    );
}

#[test]
fn a_plan_is_one_string() {
    let (lines, _) = agent(
        &["-l", "1"],
        "CREATE TABLE t (id INTEGER PRIMARY KEY);\nEXPLAIN SELECT * FROM t WHERE id = 1;\n",
    );
    let plan: serde_json::Value = serde_json::from_str(&lines[1]).unwrap();
    assert!(
        plan["plan"].as_str().is_some_and(|p| p.lines().count() > 1) && plan.get("rows").is_none(),
        "{}",
        lines[1]
    );
}

#[test]
fn stop_on_error_skips_the_rest_and_counts_it() {
    let script = "SELECT 1 AS a;\nSELECT nope;\nSELECT 3 AS c;\nSELECT 4 AS d;\n";
    assert_eq!(
        agent(&["--on-error", "stop"], script),
        (
            vec![
                r#"{"stmt":1,"line":1,"ok":true,"columns":["a"],"types":["INTEGER"],"rows":[[1]],"row_count":1}"#.to_string(),
                r#"{"stmt":2,"line":2,"ok":false,"error":{"kind":"column_not_found","message":"column 'nope' not found"}}"#.to_string(),
                r#"{"done":true,"statements":4,"failed":1,"first_failed":2,"skipped":2,"open_transaction":false}"#.to_string(),
            ],
            1
        )
    );
}

#[test]
fn the_summary_says_a_transaction_was_left_open() {
    let (lines, code) = agent(&[], "BEGIN;\nSELECT 1 AS a;\n");
    assert_eq!(
        (lines.last().unwrap().as_str(), code),
        (
            r#"{"done":true,"statements":2,"failed":0,"open_transaction":true}"#,
            0
        )
    );
}

#[test]
fn agent_mode_is_on_only_when_asked() {
    let table = ("a\n----\n1\n".to_string(), "".to_string(), 0);
    assert_eq!(run(&["-q", "-e", "SELECT 1 AS a"], &[], ""), table);
    assert_eq!(
        run(
            &["-q", "--no-agent", "-e", "SELECT 1 AS a"],
            &[("STOOLAP_AGENT", "1")],
            ""
        ),
        table
    );
    let (out, _, _) = run(&["-e", "SELECT 1 AS a"], &[("STOOLAP_AGENT", "1")], "");
    assert!(out.starts_with("{\"stoolap\":"), "{out}");
}

#[test]
fn agent_mode_bounds_every_statement_by_default() {
    let timeout = |args: &[&str]| {
        let (lines, _) = agent(args, "PRAGMA query_timeout;\n");
        lines[0].clone()
    };
    let line = |ms: u64| {
        format!(
            r#"{{"stmt":1,"line":1,"ok":true,"columns":["query_timeout"],"types":["INTEGER"],"rows":[[{ms}]],"row_count":1}}"#
        )
    };
    assert_eq!(timeout(&[]), line(30000));
    assert_eq!(timeout(&["-t", "0"]), line(0));
    assert_eq!(timeout(&["-d", "memory://?query_timeout=1234"]), line(1234));
    let (lines, _) = agent(
        &["-t", "50"],
        "BEGIN;\nSELECT COUNT(*) FROM GENERATE_SERIES(1, 3000) a, GENERATE_SERIES(1, 3000) b;\n",
    );
    assert_eq!(
        lines[1],
        r#"{"stmt":2,"line":2,"ok":false,"error":{"kind":"query_cancelled","message":"query cancelled"}}"#
    );
}
