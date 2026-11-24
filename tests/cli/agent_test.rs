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

/// The lines after the build line, as written
fn agent_lines(args: &[&str], input: &str) -> (Vec<String>, i32) {
    let mut all = vec!["--agent"];
    all.extend_from_slice(args);
    let (out, err, code) = run(&all, &[], input);
    assert_eq!(err, "", "nothing on stderr");
    let lines = out.lines().skip(1).map(str::to_string).collect();
    (lines, code)
}

/// The lines after the build line without each result's hash and order,
/// which the hash tests check
fn agent(args: &[&str], input: &str) -> (Vec<String>, i32) {
    let (lines, code) = agent_lines(args, input);
    (lines.iter().map(|l| without_hash(l)).collect(), code)
}

fn without_hash(line: &str) -> String {
    let Some(at) = line.find(",\"hash\":\"") else {
        return line.to_string();
    };
    let rest = &line[at..];
    let end = rest
        .find("ordered\"")
        .map_or(rest.len(), |i| i + "ordered\"".len());
    format!("{}{}", &line[..at], &rest[end..])
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
fn a_json_cell_is_valid_json_on_one_line() {
    let (lines, _) = agent(
        &[],
        "SELECT CAST('{\n  \"n\": 18446744073709551617,\n  \"s\": \"x  y\"\n}' AS JSON) AS j, \
         CAST(CAST('NaN' AS FLOAT) AS JSON) AS n;\n",
    );
    assert_eq!(
        lines[0],
        r#"{"stmt":1,"line":1,"ok":true,"columns":["j","n"],"types":["JSON","JSON"],"rows":[[{   "n": 18446744073709551617,   "s": "x  y" },"NaN"]],"row_count":1}"#
    );
}

#[test]
fn a_parse_error_column_counts_characters() {
    let (lines, _) = agent(&[], "SELECT 'é'; SELECT * FROM t WHERE;\n");
    assert_eq!(
        lines[1],
        r#"{"stmt":2,"line":1,"ok":false,"error":{"kind":"parse","message":"parse error: expected expression after WHERE at position line 1, column 22","line":1,"column":34}}"#
    );
}

#[test]
fn maintenance_does_not_run_in_agent_mode() {
    let refused = |args: &[&str], env: &[(&str, &str)]| {
        let (out, err, code) = run(args, env, "");
        out.is_empty() && code == 2 && err.contains("do not run in agent mode")
    };
    assert!(refused(&["--agent", "--snapshot"], &[]));
    assert!(refused(&["--reset-volumes"], &[("STOOLAP_AGENT", "1")]));
    assert!(refused(&["--restore"], &[("STOOLAP_AGENT", "1")]));
}

#[test]
fn a_plan_is_its_lines() {
    let (lines, _) = agent(
        &["-l", "1"],
        "CREATE TABLE t (id INTEGER PRIMARY KEY);\nEXPLAIN SELECT id FROM t WHERE id > 5;\n",
    );
    assert_eq!(
        lines[1],
        r#"{"stmt":2,"line":2,"ok":true,"plan":["Project id","  IndexScan t (id > 5)","    Index __pk_t_id (id range)"]}"#
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
    assert_eq!(timeout(&["-d", "memory://?query_timeout=0"]), line(0));
    assert_eq!(timeout(&["-d", "memory://query_timeout=0"]), line(30000));
    let slow = "SELECT COUNT(*) FROM GENERATE_SERIES(1, 3000) a, GENERATE_SERIES(1, 3000) b;\n";
    let (lines, _) = agent(
        &["-t", "50"],
        &format!("BEGIN;\n{slow}ROLLBACK;\nPRAGMA query_timeout = 40;\n{slow}"),
    );
    let cancelled = |stmt: u32, ms: u32| {
        format!(
            r#"{{"stmt":{stmt},"line":{stmt},"ok":false,"error":{{"kind":"query_cancelled","message":"query cancelled","timeout_ms":{ms},"hint":"the statement timeout of {ms} ms was reached; -t MS sets it, -t 0 turns it off"}}}}"#
        )
    };
    assert_eq!(
        (&lines[1], &lines[4]),
        (&cancelled(2, 50), &cancelled(5, 40))
    );
}

#[test]
fn no_rows_leaves_the_rows_out_and_keeps_the_rest() {
    let (lines, code) = agent_lines(
        &["--no-rows", "-l", "1"],
        "SELECT * FROM GENERATE_SERIES(1, 100) g;\nSELECT 1 AS a, 'x' AS b;\n",
    );
    assert_eq!(
        (lines, code),
        (
            vec![
                r#"{"stmt":1,"line":1,"ok":true,"columns":["value"],"types":["INTEGER"],"row_count":100,"hash":"f84e4355aeca41cc","order":"unordered"}"#.to_string(),
                r#"{"stmt":2,"line":2,"ok":true,"columns":["a","b"],"types":["INTEGER","TEXT"],"row_count":1,"hash":"797a862bc8256c7a","order":"unordered"}"#.to_string(),
                r#"{"done":true,"statements":2,"failed":0,"open_transaction":false}"#.to_string(),
            ],
            0
        )
    );
}

/// Each result's hash and order, by statement
fn hashes(args: &[&str], input: &str) -> Vec<(String, String)> {
    let (lines, _) = agent_lines(args, input);
    lines
        .iter()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter(|v| v.get("hash").is_some())
        .map(|v| {
            (
                v["hash"].as_str().unwrap().into(),
                v["order"].as_str().unwrap().into(),
            )
        })
        .collect()
}

#[test]
fn a_result_hash_is_the_same_in_every_build() {
    let pinned = |hash: &str| (hash.to_string(), "unordered".to_string());
    assert_eq!(
        hashes(
            &[],
            "SELECT 1 AS x;\nSELECT * FROM GENERATE_SERIES(1, 100) g;\n"
        ),
        vec![pinned("ff4ea0efc5072510"), pinned("f84e4355aeca41cc")]
    );
}

#[test]
fn a_hash_tells_equal_answers_from_different_ones() {
    let script = "CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT);\n\
                  INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, NULL);\n\
                  SELECT id, v FROM t;\n\
                  SELECT id, v FROM (SELECT * FROM t ORDER BY id DESC) s;\n\
                  SELECT id, v FROM t ORDER BY id;\n\
                  SELECT id, v FROM t ORDER BY id DESC;\n\
                  SELECT 1 AS x;\nSELECT '1' AS x;\nSELECT NULL AS x;\nSELECT 'NULL' AS x;\n";
    let h = hashes(&["-l", "1"], script);
    let order: Vec<&str> = h.iter().map(|(_, o)| o.as_str()).collect();
    assert_eq!(
        order,
        [
            "unordered",
            "unordered",
            "ordered",
            "ordered",
            "unordered",
            "unordered",
            "unordered",
            "unordered"
        ]
    );
    assert_eq!(
        h[0].0, h[1].0,
        "the same rows in another order, no ORDER BY"
    );
    assert_ne!(h[2].0, h[3].0, "ORDER BY in the two directions");
    let typed: std::collections::HashSet<&str> = h[4..].iter().map(|(x, _)| x.as_str()).collect();
    assert_eq!(typed.len(), 4, "1, '1', NULL and 'NULL' differ: {h:?}");
}

#[test]
fn the_hash_and_the_types_cover_the_rows_the_cut_left_out() {
    let sql = "SELECT CASE WHEN g.value = 1 THEN 1 ELSE 'x' END AS m FROM GENERATE_SERIES(1, 2) g";
    let (cut, _) = agent_lines(&["-l", "1", "-e", sql], "");
    let (all, _) = agent_lines(&["-l", "0", "-e", sql], "");
    assert_eq!(
        cut[0],
        r#"{"stmt":1,"line":1,"ok":true,"columns":["m"],"types":["MIXED"],"rows":[[{"t":"INTEGER","v":1}]],"row_count":2,"hash":"2acdec8968fb1c27","order":"unordered","shown":1,"truncated":true}"#
    );
    assert_eq!(
        all[0],
        r#"{"stmt":1,"line":1,"ok":true,"columns":["m"],"types":["MIXED"],"rows":[[{"t":"INTEGER","v":1}],[{"t":"TEXT","v":"x"}]],"row_count":2,"hash":"2acdec8968fb1c27","order":"unordered"}"#
    );
}

#[test]
fn each_case_runs_in_a_database_of_its_own() {
    let script = "CREATE TABLE t (id INTEGER PRIMARY KEY);\n\
                  -- @case first\n\
                  CREATE TABLE t (id INTEGER PRIMARY KEY);\n\
                  INSERT INTO t VALUES (1);\n\
                  -- @case second\n\
                  SELECT COUNT(*) AS n FROM t;\n\
                  BEGIN;\n";
    assert_eq!(
        agent(&[], script),
        (
            vec![
                r#"{"stmt":1,"line":1,"ok":true,"rows_affected":0}"#.to_string(),
                r#"{"stmt":2,"case":"first","line":3,"ok":true,"rows_affected":0}"#.to_string(),
                r#"{"stmt":3,"case":"first","line":4,"ok":true,"rows_affected":1}"#.to_string(),
                r#"{"stmt":4,"case":"second","line":6,"ok":false,"error":{"kind":"table_or_view_not_found","message":"table or view 't' not found"}}"#.to_string(),
                r#"{"stmt":5,"case":"second","line":7,"ok":true,"rows_affected":0}"#.to_string(),
                r#"{"done":true,"statements":5,"failed":1,"first_failed":4,"open_transaction":true,"cases":[{"name":"first","statements":2,"failed":0,"open_transaction":false},{"name":"second","statements":2,"failed":1,"open_transaction":true}]}"#.to_string(),
            ],
            1
        )
    );
}

#[test]
fn stop_on_error_skips_the_rest_of_the_case_only() {
    let script = "-- @case a\nSELECT nope;\nSELECT 2 AS b;\n-- @case b\nSELECT 3 AS c;\n";
    assert_eq!(
        agent(&["--on-error", "stop"], script),
        (
            vec![
                r#"{"stmt":1,"case":"a","line":2,"ok":false,"error":{"kind":"column_not_found","message":"column 'nope' not found"}}"#.to_string(),
                r#"{"stmt":3,"case":"b","line":5,"ok":true,"columns":["c"],"types":["INTEGER"],"rows":[[3]],"row_count":1}"#.to_string(),
                r#"{"done":true,"statements":3,"failed":1,"first_failed":1,"skipped":1,"open_transaction":false,"cases":[{"name":"a","statements":2,"failed":1,"open_transaction":false},{"name":"b","statements":1,"failed":0,"open_transaction":false}]}"#.to_string(),
            ],
            1
        )
    );
}

#[test]
fn a_time_past_the_nanosecond_range_keeps_its_fraction_in_the_hash() {
    let h = hashes(
        &[],
        "SELECT TIMESTAMP '2500-01-01 00:00:00.1' AS t;\nSELECT TIMESTAMP '2500-01-01 00:00:00.2' AS t;\n",
    );
    assert_eq!(h.len(), 2);
    assert_ne!(h[0].0, h[1].0, "{h:?}");
}

#[test]
fn a_case_runs_with_the_timeout_of_the_run() {
    let timeouts = |args: &[&str]| -> Vec<String> {
        let (lines, _) = agent(
            args,
            "PRAGMA query_timeout;\n-- @case a\nPRAGMA query_timeout;\n",
        );
        lines[..2]
            .iter()
            .map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).unwrap();
                v["rows"][0][0].to_string()
            })
            .collect()
    };
    assert_eq!(timeouts(&[]), ["30000", "30000"]);
    assert_eq!(timeouts(&["-t", "50"]), ["50", "50"]);
    assert_eq!(
        timeouts(&["-d", "memory://?query_timeout=1234"]),
        ["1234", "1234"]
    );
    assert_eq!(timeouts(&["-d", "memory://?query_timeout=0"]), ["0", "0"]);
}

#[test]
fn repeat_runs_a_reading_statement_again_and_gives_the_spread() {
    let (lines, code) = agent_lines(
        &["--repeat", "4", "--warmup", "2"],
        "CREATE TABLE t (id INTEGER PRIMARY KEY);\nINSERT INTO t VALUES (1);\nSELECT id FROM t;\nEXPLAIN SELECT id FROM t;\n",
    );
    let parsed: Vec<serde_json::Value> = lines
        .iter()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(code, 0);
    let skipped = |why: &str| serde_json::json!({"runs": 1, "skipped": why});
    assert_eq!(
        [
            &parsed[0]["repeat"],
            &parsed[1]["repeat"],
            &parsed[3]["repeat"]
        ],
        [
            &skipped("not_read_only"),
            &skipped("not_read_only"),
            &skipped("explain")
        ]
    );
    let repeat = &parsed[2]["repeat"];
    let us = |key: &str| repeat[key].as_f64().unwrap();
    assert!(
        lines[2].contains(&format!("\"first_us\":{:.3},", us("first_us"))),
        "microseconds to the nanosecond: {}",
        lines[2]
    );
    assert_eq!(
        (&repeat["runs"], &repeat["warmup"]),
        (&4.into(), &2.into()),
        "{}",
        lines[2]
    );
    assert!(
        us("min_us") <= us("p50_us")
            && us("p50_us") <= us("p90_us")
            && us("p90_us") <= us("max_us"),
        "{}",
        lines[2]
    );
    assert!(repeat.get("first_us").is_some() && repeat.get("failed").is_none());
    assert_eq!(parsed[2]["rows"], serde_json::json!([[1]]), "printed once");
}

#[test]
fn repeat_in_the_table_and_json_outputs() {
    let (out, _, code) = run(
        &[
            "-q",
            "--repeat",
            "3",
            "--warmup",
            "1",
            "-e",
            "SELECT 1 AS a",
        ],
        &[],
        "",
    );
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!((&lines[..3], code), (&["a", "----", "1"][..], 0), "{out}");
    assert!(
        lines[3].starts_with("Ran 3 times after 1 warmup run: first ")
            && lines[3].contains(", then min ")
            && lines[3].contains(", p90 "),
        "{out}"
    );
    let (out, _, _) = run(
        &["-q", "-j", "--repeat", "3", "-e", "SELECT 1 AS a"],
        &[],
        "",
    );
    let json: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
    assert_eq!(
        (
            json["rows"].clone(),
            json["count"].clone(),
            json["repeat"]["runs"].clone()
        ),
        (
            serde_json::json!([[1]]),
            serde_json::json!(1),
            serde_json::json!(3)
        )
    );
}

#[test]
fn schema_gives_every_table_in_one_line() {
    let script = "CREATE TABLE \"my t\" (id INTEGER PRIMARY KEY, a INTEGER UNIQUE, b TEXT);\n\
                  CREATE INDEX ib ON \"my t\"(b);\n\
                  INSERT INTO \"my t\" VALUES (1, 2, 'x');\n\
                  CREATE VIEW v AS SELECT id FROM \"my t\";\n\
                  .schema\n\
                  .schema nope\n\
                  .tables\n";
    let (lines, code) = agent(&[], script);
    assert_eq!(
        (&lines[4..], code),
        (
            &[
                r#"{"stmt":5,"line":5,"ok":true,"command":".schema","tables":[{"table":"my t","create":"CREATE TABLE \"my t\" (id INTEGER PRIMARY KEY, a INTEGER UNIQUE, b TEXT)","indexes":[{"table_name":"my t","index_name":"unique_my t_a","column_name":"a","index_type":"BTREE","is_unique":true,"options":""},{"table_name":"my t","index_name":"ib","column_name":"b","index_type":"BTREE","is_unique":false,"options":""}],"row_count":1}],"views":["v"]}"#.to_string(),
                r#"{"stmt":6,"line":6,"ok":false,"error":{"kind":"table_not_found","message":"table 'nope' not found"}}"#.to_string(),
                r#"{"stmt":7,"line":7,"ok":false,"error":{"kind":"unknown_command","message":"unknown command: .tables"}}"#.to_string(),
                r#"{"done":true,"statements":7,"failed":2,"first_failed":6,"open_transaction":false}"#.to_string(),
            ][..],
            1
        )
    );
}

#[test]
fn stats_gives_the_three_statistics() {
    let (lines, _) = agent(&[], "CREATE TABLE t (id INTEGER PRIMARY KEY);\n.stats\n");
    let stats: serde_json::Value = serde_json::from_str(&lines[1]).unwrap();
    assert!(
        stats["ok"] == true
            && stats["memory_stats"].is_array()
            && stats["volume_stats"].is_array()
            && stats["index_stats"].is_array(),
        "{}",
        lines[1]
    );
}

#[test]
fn a_command_line_is_one_only_between_statements() {
    let script = "-- @case c\n.schema\nCREATE TABLE u (id INTEGER PRIMARY KEY);\n\
                  /* x\n.schema\n*/ SELECT 2 AS c;\n";
    let (lines, _) = agent(&[], script);
    assert_eq!(
        &lines[..3],
        [
            r#"{"stmt":1,"case":"c","line":2,"ok":true,"command":".schema","tables":[],"views":[]}"#,
            r#"{"stmt":2,"case":"c","line":3,"ok":true,"rows_affected":0}"#,
            r#"{"stmt":3,"case":"c","line":6,"ok":true,"columns":["c"],"types":["INTEGER"],"rows":[[2]],"row_count":1}"#,
        ]
    );
}

/// A run of `args` whose first run of a SLEEP that fails half the time
/// succeeded, so its failure, nearly sure among 29 more, is a repeat's
fn first_run_passing(args: &[&str]) -> (String, String, i32) {
    let sql = "SELECT SLEEP(CASE WHEN RANDOM() < 0.5 THEN 0 ELSE -1 END) AS s";
    let mut all = args.to_vec();
    all.extend_from_slice(&["--repeat", "30", "-e", sql]);
    (0..20)
        .map(|_| run(&all, &[], ""))
        .find(|(out, _, _)| out.contains("\"s\"") || out.starts_with("s\n"))
        .expect("a first run passed")
}

#[test]
fn a_repeat_that_fails_fails_the_statement() {
    let (out, _, code) = first_run_passing(&["--agent"]);
    let lines: Vec<serde_json::Value> = out
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let failed = lines[1]["repeat"]["failed"].as_str().unwrap_or("");
    assert!(
        lines[1]["ok"] == true
            && failed.contains("SLEEP duration cannot be negative")
            && lines[2]["failed"] == 1
            && lines[2]["first_failed"] == 1
            && code == 1,
        "{out}"
    );
    let (out, err, code) = first_run_passing(&["-q"]);
    assert!(
        out.starts_with("s\n----\n0\nRan ")
            && out.contains("; stopped: invalid argument: SLEEP duration cannot be negative")
            && err.starts_with("Error (statement 1, line 1): ")
            && code == 1,
        "{out}{err}"
    );
}

#[test]
fn a_command_follows_a_block_comment_closed_on_a_later_line() {
    let (lines, _) = agent(&[], "/* ;\n*/\n.schema\n");
    assert_eq!(
        lines[0],
        r#"{"stmt":1,"line":3,"ok":true,"command":".schema","tables":[],"views":[]}"#
    );
}

#[test]
fn a_warmup_that_fails_counts_only_the_runs_that_passed() {
    let (out, _, _) = first_run_passing(&["--agent", "--warmup", "10"]);
    let line: serde_json::Value = serde_json::from_str(out.lines().nth(1).unwrap()).unwrap();
    let repeat = &line["repeat"];
    // With no measured run, the failure came within the ten warmup runs
    if repeat.get("min_us").is_none() {
        assert!(repeat["warmup"].as_u64().unwrap() < 10, "{out}");
    }
}

const GUIDE: &str = include_str!("../../src/bin/stoolap/agent_guide.txt");

#[test]
fn help_in_agent_mode_gives_the_agent_guide() {
    let (out, err, code) = run(&["--agent", "--help"], &[], "");
    assert_eq!((out.as_str(), err.as_str(), code), (GUIDE, "", 0));
    let (out, _, _) = run(&["-h"], &[("STOOLAP_AGENT", "1")], "");
    assert_eq!(out, GUIDE, "the variable turns it on");
    let (out, _, code) = run(&["--help"], &[], "");
    assert!(
        out.contains("run `stoolap --agent --help` for agent mode")
            && !out.contains("COMPARE ANSWERS")
            && code == 0,
        "{out}"
    );
    let (out, _, _) = run(&["--agent", "--no-agent", "--help"], &[], "");
    assert!(!out.contains("COMPARE ANSWERS"), "{out}");
}

#[test]
fn the_agent_guide_names_only_flags_that_exist() {
    let (help, _, _) = run(&["--help"], &[], "");
    let flags: Vec<&str> = GUIDE
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '-'))
        .filter(|w| w.starts_with("--") && w.len() > 2)
        .collect();
    assert!(flags.len() >= 8, "{flags:?}");
    for flag in flags {
        assert!(
            help.contains(&format!("{flag} ")) || help.contains(&format!("{flag}\n")),
            "{flag} is not a flag"
        );
    }
}

#[test]
fn the_agent_guide_example_gives_the_lines_it_shows() {
    let (out, _, code) = run(&["--agent", "-e", "SELECT 1 AS a; SELECT nope"], &[], "");
    let shown: Vec<&str> = GUIDE
        .lines()
        .skip_while(|l| !l.starts_with("OUTPUT"))
        .skip(2)
        .take(3)
        .map(str::trim)
        .collect();
    assert_eq!((out.lines().skip(1).collect::<Vec<_>>(), code), (shown, 1));
}

#[test]
fn a_reader_that_stops_reading_ends_the_run_quietly() {
    for args in [&["--agent"][..], &["-q"], &["-q", "-j"], &[], &["-j"]] {
        let mut child = Command::new(env!("CARGO_BIN_EXE_stoolap"))
            .args(args)
            .args([
                "-l",
                "0",
                "-e",
                "SELECT * FROM GENERATE_SERIES(1, 1000000) g; SELECT 1",
            ])
            .env_remove("STOOLAP_AGENT")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut first = [0u8; 16];
        std::io::Read::read_exact(child.stdout.as_mut().unwrap(), &mut first).unwrap();
        drop(child.stdout.take());
        let out = child.wait_with_output().unwrap();
        assert_eq!(
            (
                String::from_utf8_lossy(&out.stderr).as_ref(),
                out.status.code()
            ),
            ("", Some(1)),
            "{args:?}"
        );
    }
}

#[test]
fn a_params_list_is_read_as_sql_values() {
    let (lines, code) = agent(
        &[],
        "SELECT $1 AS a, $2 AS b, $3 AS c; -- PARAMS: 3, 'x, y', 'it''s'\n\
         SELECT $1 AS a; -- PARAMS: alice\n",
    );
    assert_eq!(
        (&lines[..2], code),
        (
            &[
                r#"{"stmt":1,"line":1,"ok":true,"columns":["a","b","c"],"types":["INTEGER","TEXT","TEXT"],"rows":[[3,"x, y","it's"]],"row_count":1}"#.to_string(),
                r#"{"stmt":2,"line":2,"ok":false,"error":{"kind":"params","message":"PARAMS, read as SQL values: column 'alice' not found"}}"#.to_string(),
            ][..],
            1
        )
    );
}
