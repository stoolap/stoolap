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

//! What the CLI prints for each statement of `-e`, `-f` and piped input,
//! and its exit code

use std::io::Write;
use std::process::{Command, Stdio};

/// stdout, stderr and the exit code of the CLI run with `args` and `input`
fn cli(args: &[&str], input: &str) -> (String, String, i32) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_stoolap"))
        .args(args)
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

#[test]
fn every_statement_of_execute_prints_its_result() {
    assert_eq!(
        cli(&["-q", "-e", "SELECT 1 AS a; SELECT 2 AS b"], ""),
        ("a\n----\n1\nb\n----\n2\n".into(), "".into(), 0)
    );
}

#[test]
fn a_failed_statement_names_itself_and_fails_the_run() {
    let script = "SELECT 1 AS a;\nSELECT nope;\nSELECT 3 AS c;\n";
    let expected = (
        "a\n----\n1\nc\n----\n3\n".to_string(),
        "Error (statement 2, line 2): column 'nope' not found\n".to_string(),
        1,
    );
    assert_eq!(cli(&["-q"], script), expected, "piped");

    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("script.sql");
    std::fs::write(&file, script).unwrap();
    assert_eq!(
        cli(&["-q", "-f", file.to_str().unwrap()], ""),
        expected,
        "file"
    );

    let line = script.replace('\n', " ");
    let (out, err, code) = cli(&["-q", "-e", &line], "");
    assert_eq!(
        (out, err, code),
        (
            expected.0,
            "Error (statement 2, line 1): column 'nope' not found\n".to_string(),
            1
        ),
        "execute"
    );
}

#[test]
fn values_that_read_alike_print_apart() {
    let sql = "SELECT NULL AS a, 'NULL' AS b, 1 AS c, '1' AS d, 'x|y' AS e, \
               0.1 + 0.2 AS f, 2.0 AS g, '' AS h, 'it''s' AS i";
    assert_eq!(
        cli(&["-q", "-e", sql], "").0,
        "a | b | c | d | e | f | g | h | i\n\
         -----+------+------+------+------+------+------+------+-----\n\
         NULL | 'NULL' | 1 | '1' | 'x|y' | 0.30000000000000004 | 2.0 | '' | it's\n"
    );
}

#[test]
fn a_cut_keeps_the_first_rows_and_counts_the_rest() {
    let (out, _, code) = cli(
        &["-q", "-l", "2"],
        "SELECT * FROM GENERATE_SERIES(1, 5) g;\n",
    );
    assert_eq!(
        (out.as_str(), code),
        ("value\n----\n1\n2\n(3 more rows, 5 in total)\n", 0)
    );
}

#[test]
fn json_follows_the_limit_and_writes_nan_as_text() {
    assert_eq!(
        cli(
            &["-q", "-j", "-l", "2"],
            "SELECT * FROM GENERATE_SERIES(1, 4) g;\nSELECT CAST('NaN' AS FLOAT) AS n;\n"
        )
        .0,
        "{\"columns\":[\"value\"],\"rows\":[[1],[2]],\"count\":4,\"shown\":2}\n\
         {\"columns\":[\"n\"],\"rows\":[[\"NaN\"]],\"count\":1}\n"
    );
}

#[test]
fn a_statement_without_rows_prints_its_count() {
    let script = "CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT);\n\
                  INSERT INTO t VALUES (1, 'x RETURNING y'), (2, 'b');\n\
                  UPDATE t SET s = 'c' WHERE id = 2 RETURNING id, s;\n";
    assert_eq!(
        cli(&["-q"], script).0,
        "0 rows affected\n2 rows affected\nid | s\n-----+-----\n2 | c\n"
    );
}

#[test]
fn a_params_comment_after_the_semicolon_gives_the_statement_its_parameters() {
    for sql in [
        "SELECT $1 AS v; -- PARAMS: 7",
        "SELECT $1 AS v; /* note */ -- PARAMS: 7",
    ] {
        assert_eq!(
            cli(&["-q", "-e", sql], ""),
            ("v\n----\n7\n".into(), "".into(), 0),
            "{sql}"
        );
    }
    assert_eq!(
        cli(&["-q", "-e", "SELECT 'x -- PARAMS: y' AS v"], "").0,
        "v\n----\nx -- PARAMS: y\n"
    );
}

#[test]
fn a_block_comment_the_input_leaves_open_fails_the_run() {
    for (sql, out) in [
        ("/* never closed", ""),
        ("SELECT 1 AS a; /* never closed", "a\n----\n1\n"),
    ] {
        let (stdout, stderr, code) = cli(&["-q", "-e", sql], "");
        assert!(
            stdout == out && code == 1 && stderr.contains("unterminated block comment"),
            "{sql}: {stdout:?} {stderr:?} {code}"
        );
    }
}

#[test]
fn nothing_inside_an_open_block_comment_runs() {
    for opening in ["\n/* start;", " /* start;"] {
        let script = format!(
            "CREATE TABLE t (id INTEGER PRIMARY KEY);\n\
             INSERT INTO t VALUES (1);{opening}\n\
             SELECT 0;\n\
             DROP TABLE t;\n\
             */\n\
             SELECT COUNT(*) AS n FROM t;\n"
        );
        assert_eq!(
            cli(&["-q"], &script),
            (
                "0 rows affected\n1 row affected\nn\n----\n1\n".into(),
                "".into(),
                0
            ),
            "{script}"
        );
    }
}

#[test]
fn a_carriage_return_in_a_string_reaches_the_engine() {
    assert_eq!(
        cli(&["-q", "-e", "SELECT LENGTH('a\r\nb') AS n"], "").0,
        "n\n----\n4\n"
    );
    assert_eq!(
        cli(&["-q"], "SELECT LENGTH('a\r\nb') AS n;\r\n").0,
        "n\n----\n4\n"
    );
}

#[test]
fn only_explain_is_shown_past_the_limit() {
    assert_eq!(
        cli(
            &[
                "-q",
                "-l",
                "1",
                "-e",
                "SELECT value AS plan FROM GENERATE_SERIES(1, 3) g"
            ],
            ""
        )
        .0,
        "plan\n----\n1\n(2 more rows, 3 in total)\n"
    );
    let (plan, _, code) = cli(
        &[
            "-q",
            "-l",
            "1",
            "-e",
            "EXPLAIN SELECT * FROM GENERATE_SERIES(1, 3) g",
        ],
        "",
    );
    assert!(
        code == 0 && plan.lines().count() > 3 && !plan.contains("more rows"),
        "{plan}"
    );
}

#[test]
fn without_quiet_the_open_each_statement_and_the_close_are_timed_apart() {
    let note = match cfg!(debug_assertions) {
        true => " [debug build: times are not those of a release build]",
        false => "",
    };
    let (out, _, code) = cli(&["-e", "SELECT 1 AS a"], "");
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!((lines.len(), code), (7, 0), "{out}");
    assert!(
        lines[0].starts_with("Connected to database: memory:// (opened in ")
            && lines[0].ends_with(&format!("){note}")),
        "{out}"
    );
    assert_eq!(&lines[1..4], ["a", "----", "1"], "{out}");
    assert_eq!(lines[4], "1 row in set", "{out}");
    assert!(
        lines[5].starts_with("Query executed in ") && lines[5].ends_with(note),
        "{out}"
    );
    assert!(
        lines[6].starts_with("Database closed in ") && lines[6].ends_with(note),
        "{out}"
    );
}

#[test]
fn a_semicolon_in_a_string_or_comment_does_not_end_the_statement() {
    let script = "SELECT 'a;b' AS v; -- c; d\nSELECT 2 /* ; */ AS w # e; f\n;\n";
    assert_eq!(
        cli(&["-q"], script),
        ("v\n----\na;b\nw\n----\n2\n".into(), "".into(), 0)
    );
}
