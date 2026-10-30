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

//! The lexer reads every text into the tokens recorded for it: their type,
//! text, line, column, byte offset, quoting and keyword

use stoolap::parser::{token::TokenType, Lexer};

mod oracle;

const RECORDED: oracle::Answers = answers!("lexer_tokens");

/// Texts that reach every kind of token, every way a token ends and every
/// way a line and a column count
const TEXTS: &[&str] = &[
    "",
    " ",
    "\0",
    "a\0b",
    "SELECT\r\n1",
    "x\r\ny\rz\n\rw",
    "  \t\n  SELECT",
    "SELECT\t1\n\n\n  ,2",
    "select Select SELECT sElEcT",
    "SELECT * FROM users WHERE id = 1234",
    "SELECT id, name FROM users WHERE age = 17 AND balance > 4521 AND name <> 'x'",
    "UPDATE u1 SET v = 1.5 WHERE k = 8638",
    "INSERT INTO t (a, b) VALUES ($1, $2) ON CONFLICT (a) DO UPDATE SET b = excluded.b",
    "DELETE FROM t WHERE id IN (SELECT id FROM u) RETURNING *",
    "SELECT users.id, u1.v FROM users JOIN u1 ON u1.id = users.id WHERE users.id = 4321",
    "WITH c AS (SELECT 1 AS x) SELECT x FROM c ORDER BY 1 DESC LIMIT 2 OFFSET 1",
    "SELECT 'é' AS ü, \"naïve\" FROM t\n-- c\nWHERE x = 1",
    "SÉLECT ñame FROM tàble",
    "ä",
    "€ 1",
    "SELECT 1e5, 1.5e-3, 2E+7, .5, 5., 1.2.3, 0x1F, 1e, 1e+",
    "SELECT 123abc, abc123, _x, x$y, $$, 1_000",
    "SELECT $1, $12, ?, :name, :_x, $, :1",
    "a ->> b -> c #> d #>> e @> f <@ g ?| h ?& i <=> j",
    "x <> y != z >= w <= v << u >> s || r = q < p > o",
    "@ ~ ^ & | ! % + - / *",
    "a.b.c, t.*, f(x)[1]; g:h",
    "/* open",
    "'unterminated",
    "\"open",
    "`open",
    "`back`",
    "SELECT \"a\"\"b\", `c``d`",
    "SELECT /* nested /* */ 1",
    "SELECT /* a\nb */ 2",
    "SELECT 'it''s', 'a\\b', E'x', ''",
    "SELECT 'two\nlines', 'é''é'",
    "SELECT 1 # comment\n2",
    "SELECT -1, - -2, --x\n3",
    "SELECT 1 -- tail",
    "TIMESTAMP '2024-01-01 10:00:00', DATE '2024-01-01', TIME '10:00'",
    "CREATE TABLE t (id INTEGER PRIMARY KEY, v VECTOR(3), j JSON, ts TIMESTAMP)",
    "SELECT v <=> '[1.0, 2.0, 3.0]' FROM t",
    "SELECT j->>'a', j->'b'->0 FROM t WHERE j @> '{\"a\":1}'",
    "SELECT CASE WHEN a THEN 'x' ELSE \"y\" END",
    "select \u{feff}1",
    "SELECT \u{3b1}\u{3b2} FROM \u{4e2d}\u{6587}",
];

#[test]
fn every_text_lexes_into_its_recorded_tokens() {
    let mut wrong = Vec::new();
    for text in TEXTS {
        let mut lexer = Lexer::new(text);
        let mut tokens = Vec::new();
        loop {
            let token = lexer.next_token();
            tokens.push(format!(
                "{:?} {:?} {}:{}@{} {} {:?}",
                token.token_type,
                token.literal.as_str(),
                token.position.line,
                token.position.column,
                token.position.offset,
                token.quoted,
                token.keyword
            ));
            if token.token_type == TokenType::Eof || tokens.len() > 1_000 {
                break;
            }
        }
        if let Some(e) = oracle::check(&RECORDED, &format!("{text:?}"), &tokens) {
            wrong.push(e);
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
