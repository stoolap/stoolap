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

//! SQL text run by the CLI: cut into statements by the engine's lexer, each
//! statement run once, its rows streamed to the output

use std::io::{self, Write};
use std::time::{Duration, Instant};

use stoolap::parser::lexer::Lexer;
use stoolap::parser::token::{Keyword, Token, TokenType};
use stoolap::{ApiTransaction, Database, Row, Rows, Value};

/// A statement of a script, the line its first token is on, and the
/// parameters a `-- PARAMS: 1, 'a'` comment gives it
pub struct Statement {
    pub line: usize,
    pub sql: String,
    pub params: Vec<Value>,
}

/// Cuts SQL text into statements at the semicolons the engine's lexer
/// reads, so quotes and comments follow the parser's own rules. Text is
/// lexed once: a cut resumes where the last one stopped
pub struct Splitter {
    /// Text of the statement being read, from its first byte
    pending: String,
    /// `pending` before this offset is lexed and holds no semicolon
    resume: usize,
    /// Line of the byte at `resume`, from 1
    resume_line: usize,
    /// Line of the first token of the statement being read
    first_line: Option<usize>,
    /// Parameters a comment in the statement being read gave
    params: Option<Vec<Value>>,
    /// What the text is left open in at `resume`, and whether a line since
    /// closed it
    open: Option<Open>,
    closed: bool,
    /// A line with the quote of an open string before it, lexed alone
    probe: String,
}

/// A string, quoted name or block comment that runs to the end of the text
#[derive(Clone, Copy)]
enum Open {
    Quote(char),
    Comment,
    Other,
}

impl Default for Splitter {
    fn default() -> Self {
        Self {
            pending: String::new(),
            resume: 0,
            resume_line: 1,
            first_line: None,
            params: None,
            open: None,
            closed: false,
            probe: String::new(),
        }
    }
}

impl Splitter {
    /// Adds one line of input, without its `\n`, and hands out the
    /// statements it completes
    pub fn push_line(&mut self, line: &str, out: &mut Vec<Statement>) {
        self.pending.push_str(line);
        self.pending.push('\n');
        if let Some(open) = self.open
            && !self.closed
        {
            self.closed = match open {
                // A line starts inside the string with nothing pending, so
                // the line lexed after the quote alone says whether it ends
                // the string, by the lexer's own escapes
                Open::Quote(quote) => {
                    self.probe.clear();
                    self.probe.push(quote);
                    self.probe.push_str(line);
                    self.probe.push('\n');
                    let mut lexer = Lexer::new(&self.probe);
                    lexer.next_token().token_type != TokenType::Error
                        || lexer.next_token().token_type != TokenType::Eof
                }
                Open::Comment => line.contains("*/"),
                Open::Other => true,
            };
        }
        if line.contains(';') && (self.open.is_none() || self.closed) {
            self.cut(false, out);
        }
    }

    /// Whether text other than blanks is waiting for its semicolon
    pub fn is_pending(&self) -> bool {
        !self.pending.trim().is_empty()
    }

    /// Hands out the statement the input ends with, if any
    pub fn finish(&mut self, out: &mut Vec<Statement>) {
        self.cut(true, out);
    }

    fn cut(&mut self, at_end: bool, out: &mut Vec<Statement>) {
        let base = self.resume;
        let mut lexer = Lexer::new(&self.pending[base..]);
        let mut held: Option<Token> = None;
        // A statement cut at a semicolon, and that semicolon's line: a
        // PARAMS comment after it on that line is its own
        let mut cut: Option<(Statement, usize)> = None;
        let mut start = 0;
        self.open = None;
        self.closed = false;
        loop {
            let token = held.take().unwrap_or_else(|| lexer.next_token());
            let at = base + token.position.offset as usize;
            let line = self.resume_line + token.position.line as usize - 1;
            if let Some((mut statement, semicolon_line)) = cut.take() {
                // Comments closed on the semicolon's line are passed over,
                // and the first PARAMS comment among them is the statement's;
                // any other token, an open block comment among them, is read
                // as below
                if token.token_type == TokenType::Comment
                    && line == semicolon_line
                    && !open_block(&token.literal)
                {
                    match params_of(&token.literal) {
                        Some(params) => {
                            statement.params = params;
                            out.push(statement);
                        }
                        None => cut = Some((statement, semicolon_line)),
                    }
                    continue;
                }
                out.push(statement);
            }
            match token.token_type {
                TokenType::Eof => {
                    self.resume = self.pending.len();
                    self.resume_line = line;
                    break;
                }
                // The lexer gives a block comment that runs to the end of
                // the text as a comment, not an error: what follows in later
                // lines is still inside it
                TokenType::Comment if !at_end && open_block(&token.literal) => {
                    self.open = Some(Open::Comment);
                    self.resume = at;
                    self.resume_line = line;
                    break;
                }
                // Still open at the end of the input, it is a statement of
                // its own or the end of one, and the engine reports it
                TokenType::Comment if open_block(&token.literal) => {
                    self.first_line.get_or_insert(line);
                }
                TokenType::Comment => {
                    if let Some(params) = params_of(&token.literal) {
                        self.params = Some(params);
                    }
                }
                TokenType::Punctuator if token.literal == ";" => {
                    let params = self.params.take().unwrap_or_default();
                    if let Some(first) = self.first_line.take() {
                        let sql = self.pending[start..at].trim().to_string();
                        cut = Some((
                            Statement {
                                line: first,
                                sql,
                                params,
                            },
                            line,
                        ));
                    }
                    start = at + 1;
                }
                TokenType::Error if !at_end => {
                    // An unclosed string, name or comment runs to the end of
                    // the text: the statement goes on in a later line, and
                    // is lexed again from here once a line can close it
                    let next = lexer.next_token();
                    if next.token_type == TokenType::Eof {
                        self.open = Some(match self.pending.as_bytes()[at] {
                            quote @ (b'\'' | b'"' | b'`') => Open::Quote(quote as char),
                            _ => Open::Other,
                        });
                        self.resume = at;
                        self.resume_line = line;
                        break;
                    }
                    self.first_line.get_or_insert(line);
                    held = Some(next);
                }
                _ => {
                    self.first_line.get_or_insert(line);
                }
            }
        }
        if at_end {
            if let Some(first) = self.first_line.take() {
                let sql = self.pending[start..].trim().to_string();
                let params = self.params.take().unwrap_or_default();
                out.push(Statement {
                    line: first,
                    sql,
                    params,
                });
            }
            start = self.pending.len();
        }
        self.pending.drain(..start);
        self.resume -= start;
    }
}

/// Whether a comment token is a block comment the text ended inside
fn open_block(comment: &str) -> bool {
    comment.starts_with("/*") && (comment.len() < 4 || !comment.ends_with("*/"))
}

/// `sql` with its comments taken out, as the lexer reads them
fn without_comments(sql: &str) -> String {
    let mut lexer = Lexer::new(sql);
    let mut text = String::with_capacity(sql.len());
    let mut from = 0;
    loop {
        let token = lexer.next_token();
        match token.token_type {
            TokenType::Eof => break,
            TokenType::Comment => {
                let at = token.position.offset as usize;
                text.push_str(&sql[from..at]);
                from = at + token.literal.len();
            }
            _ => {}
        }
    }
    text.push_str(&sql[from..]);
    text
}

/// The parameters a `-- PARAMS: 1, 'a'` comment gives its statement
fn params_of(comment: &str) -> Option<Vec<Value>> {
    let list = comment
        .strip_prefix("--")?
        .trim_start()
        .strip_prefix("PARAMS:")?;
    Some(list.split(',').map(|v| param_value(v.trim())).collect())
}

/// What a statement is, read from its tokens with comments passed over
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Command {
    Begin,
    Commit,
    Rollback,
    Explain,
    Exit,
    Help,
    Other,
}

/// What `sql` is: the transaction commands and EXPLAIN by their keyword
/// token, so a quoted name spelling one is not it, and the CLI's own exit
/// and help by the bare word alone
pub fn command(sql: &str) -> Command {
    let mut lexer = Lexer::new(sql);
    let mut tokens = std::iter::from_fn(|| loop {
        let token = lexer.next_token();
        match token.token_type {
            TokenType::Eof => return None,
            TokenType::Comment => {}
            TokenType::Punctuator if token.literal == ";" => {}
            _ => return Some(token),
        }
    });
    let Some(first) = tokens.next() else {
        return Command::Other;
    };
    let alone = tokens.next().is_none();
    let from = first.position.offset as usize;
    // Written as a word, not quoted with double quotes or backticks
    let bare = alone
        && matches!(first.token_type, TokenType::Identifier | TokenType::Keyword)
        && sql.as_bytes()[from].is_ascii_alphabetic();
    let word = |words: &[&str]| words.iter().any(|w| first.literal.eq_ignore_ascii_case(w));
    // The backslash forms are no SQL tokens: they are read as written, with
    // their comments taken out, when the first token is a backslash
    let slash = |forms: &[&str]| {
        sql.as_bytes()[from] == b'\\' && {
            let text = without_comments(sql);
            let text = text.trim().trim_end_matches(';').trim();
            forms.iter().any(|f| text.eq_ignore_ascii_case(f))
        }
    };
    match first.keyword {
        Some(Keyword::Begin) => Command::Begin,
        Some(Keyword::Explain) => Command::Explain,
        Some(Keyword::Commit) if alone => Command::Commit,
        Some(Keyword::Rollback) if alone => Command::Rollback,
        _ if bare && word(&["exit", "quit"]) || slash(&["\\q"]) => Command::Exit,
        _ if bare && word(&["help"]) || slash(&["\\h", "\\?"]) => Command::Help,
        _ => Command::Other,
    }
}

/// How results are printed
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Table,
    Json,
}

/// Where a script's results go and how much of them is shown
pub struct Output<W: Write> {
    pub out: W,
    pub format: Format,
    /// Rows shown per result, 0 for all
    pub limit: usize,
    pub quiet: bool,
    /// Print the time of each statement after its result
    pub timing: bool,
}

/// What a statement runs against
pub enum Session<'a> {
    Database(&'a Database),
    Transaction(&'a mut ApiTransaction),
}

impl Session<'_> {
    fn query(&mut self, sql: &str, params: Vec<Value>, timeout_ms: u64) -> stoolap::Result<Rows> {
        match self {
            Session::Database(db) if timeout_ms > 0 => {
                db.query_with_timeout(sql, params, timeout_ms)
            }
            Session::Database(db) => db.query(sql, params),
            Session::Transaction(tx) => tx.query(sql, params),
        }
    }
}

/// The time a statement took in the engine, and the time the CLI took to
/// print what it returned
#[derive(Clone, Copy, Default)]
pub struct Timing {
    pub engine: Duration,
    pub output: Duration,
}

/// What a statement gave: rows, of which the sink took `shown`, or the
/// count of a statement without them
pub enum Ran {
    Rows {
        columns: Vec<String>,
        total: u64,
        shown: u64,
    },
    Affected(i64),
}

/// Takes the rows a statement shows
pub trait Sink {
    fn begin(&mut self, columns: &[String]) -> io::Result<()>;
    fn row(&mut self, row: &Row) -> io::Result<()>;
}

/// The single place a statement is run, for every mode of the CLI: hands
/// the first `limit` rows (0 for all) to `sink` and counts the rest, timing
/// only the sink, so a cut result costs its execution and nothing per row
pub fn run(
    session: &mut Session,
    sql: &str,
    params: Vec<Value>,
    timeout_ms: u64,
    limit: usize,
    sink: &mut impl Sink,
) -> Result<(Ran, Timing), String> {
    // A plan is shown whole whatever the limit
    let limit = match command(sql) {
        Command::Explain => 0,
        _ => limit,
    };
    let start = Instant::now();
    let mut rows = session
        .query(sql, params, timeout_ms)
        .map_err(|e| e.to_string())?;
    if rows.columns().is_empty() {
        let timing = Timing {
            engine: start.elapsed(),
            output: Duration::ZERO,
        };
        return Ok((Ran::Affected(rows.rows_affected()), timing));
    }
    let shown = Instant::now();
    let columns = rows.columns().to_vec();
    sink.begin(&columns).map_err(|e| e.to_string())?;
    let mut output = shown.elapsed();
    let mut total = 0u64;
    let mut timed = 0u64;
    while rows.advance() {
        if limit == 0 || total < limit as u64 {
            let shown = Instant::now();
            sink.row(rows.current_row()).map_err(|e| e.to_string())?;
            output += shown.elapsed();
            timed += 1;
        }
        total += 1;
    }
    let elapsed = start.elapsed();
    // A timed row leaves about one clock reading outside its span
    let readings = clock_reading().mul_f64(timed as f64);
    let engine = elapsed.saturating_sub(output).saturating_sub(readings);
    output += readings;
    if let Some(err) = rows.error() {
        return Err(err.to_string());
    }
    let ran = Ran::Rows {
        columns,
        total,
        shown: timed,
    };
    Ok((ran, Timing { engine, output }))
}

/// What one reading of the clock costs, measured once per process
fn clock_reading() -> Duration {
    static COST: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *COST.get_or_init(|| {
        let start = Instant::now();
        for _ in 0..1024 {
            std::hint::black_box(Instant::now());
        }
        start.elapsed() / 1024
    })
}

/// Runs every statement of `lines` as soon as its semicolon arrives;
/// returns the number of statements that failed
pub fn run_script<W: Write>(
    db: &Database,
    lines: impl Iterator<Item = io::Result<String>>,
    out: &mut Output<W>,
    timeout_ms: u64,
) -> Result<usize, String> {
    let mut splitter = Splitter::default();
    let mut ready = Vec::new();
    let mut failed = 0;
    let mut number = 0;
    for line in lines {
        let line = line.map_err(|e| format!("reading input: {e}"))?;
        splitter.push_line(&line, &mut ready);
        if !run_ready(db, &mut ready, &mut number, &mut failed, out, timeout_ms)? {
            return Ok(failed);
        }
    }
    splitter.finish(&mut ready);
    run_ready(db, &mut ready, &mut number, &mut failed, out, timeout_ms)?;
    Ok(failed)
}

/// Runs the statements cut so far; false once a statement asked to exit
fn run_ready<W: Write>(
    db: &Database,
    ready: &mut Vec<Statement>,
    number: &mut usize,
    failed: &mut usize,
    out: &mut Output<W>,
    timeout_ms: u64,
) -> Result<bool, String> {
    for statement in ready.drain(..) {
        *number += 1;
        match command(&statement.sql) {
            Command::Exit => return Ok(false),
            Command::Help => {
                super::print_help_main();
                continue;
            }
            _ => {}
        }
        let printed = print_statement(
            &mut Session::Database(db),
            &statement.sql,
            statement.params,
            out,
            timeout_ms,
        );
        if let Err(message) = printed {
            // The rows printed before the error come first
            out.out.flush().map_err(|e| e.to_string())?;
            *failed += 1;
            eprintln!(
                "Error (statement {}, line {}): {}",
                number, statement.line, message
            );
        }
    }
    Ok(true)
}

/// Runs one statement and prints its result in the output's format
pub fn print_statement<W: Write>(
    session: &mut Session,
    sql: &str,
    params: Vec<Value>,
    out: &mut Output<W>,
    timeout_ms: u64,
) -> Result<Timing, String> {
    let Output {
        out: w,
        format,
        limit,
        quiet,
        timing,
    } = out;
    let ran = match format {
        Format::Table => run(session, sql, params, timeout_ms, *limit, &mut TableSink(w)),
        Format::Json => {
            let mut sink = JsonSink {
                w,
                begun: false,
                rows: 0,
            };
            let ran = run(session, sql, params, timeout_ms, *limit, &mut sink);
            // Rows already written stay readable: the object is closed
            if ran.is_err() && sink.begun {
                let _ = sink.w.write_all(b"]}\n");
            }
            ran
        }
    };
    let (ran, took) = ran?;
    let footer = match (ran, *format) {
        (Ran::Affected(n), Format::Json) => writeln!(w, "{{\"rows_affected\":{n}}}"),
        (Ran::Affected(n), Format::Table) => writeln!(w, "{n} {} affected", plural(n, "row")),
        (Ran::Rows { total, shown, .. }, Format::Json) => match shown < total {
            true => writeln!(w, "],\"count\":{total},\"shown\":{shown}}}"),
            false => writeln!(w, "],\"count\":{total}}}"),
        },
        (Ran::Rows { total, shown, .. }, Format::Table) => {
            let cut = match shown < total {
                true => writeln!(w, "({} more rows, {total} in total)", total - shown),
                false => Ok(()),
            };
            cut.and_then(|()| match *quiet {
                true => Ok(()),
                false => writeln!(w, "{total} {} in set", plural(total as i64, "row")),
            })
        }
    };
    footer.map_err(|e| e.to_string())?;
    if *timing {
        writeln!(w, "{}", timing_line(took)).map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())?;
    Ok(took)
}

/// Said after every time a debug build prints, which runs many times slower
pub const BUILD_NOTE: &str = match cfg!(debug_assertions) {
    true => " [debug build: times are not those of a release build]",
    false => "",
};

/// `Query executed in 1.2 ms`, with the time spent printing when there was any
pub fn timing_line(took: Timing) -> String {
    match took.output.is_zero() {
        true => format!("Query executed in {}{BUILD_NOTE}", duration(took.engine)),
        false => format!(
            "Query executed in {} (output {}){BUILD_NOTE}",
            duration(took.engine),
            duration(took.output)
        ),
    }
}

pub fn duration(d: Duration) -> String {
    let us = d.as_secs_f64() * 1e6;
    if us < 1000.0 {
        format!("{us:.1} µs")
    } else if us < 1e6 {
        format!("{:.3} ms", us / 1e3)
    } else {
        format!("{:.3} s", us / 1e6)
    }
}

pub fn plural(n: i64, word: &str) -> String {
    match n {
        1 => word.to_string(),
        _ => format!("{word}s"),
    }
}

/// Rows as `a | b` lines under a header
struct TableSink<'a, W: Write>(&'a mut W);

impl<W: Write> Sink for TableSink<'_, W> {
    fn begin(&mut self, columns: &[String]) -> io::Result<()> {
        let w = &mut *self.0;
        for (i, column) in columns.iter().enumerate() {
            if i > 0 {
                w.write_all(b" | ")?;
            }
            w.write_all(column.as_bytes())?;
        }
        w.write_all(b"\n")?;
        for i in 0..columns.len() {
            if i > 0 {
                w.write_all(b"-+-")?;
            }
            w.write_all(b"----")?;
        }
        w.write_all(b"\n")
    }

    fn row(&mut self, row: &Row) -> io::Result<()> {
        let w = &mut *self.0;
        for (i, value) in row.as_slice().iter().enumerate() {
            if i > 0 {
                w.write_all(b" | ")?;
            }
            write_value(w, value)?;
        }
        w.write_all(b"\n")
    }
}

/// Rows as one JSON object, written as they come; the caller closes it
struct JsonSink<'a, W: Write> {
    w: &'a mut W,
    begun: bool,
    rows: u64,
}

impl<W: Write> Sink for JsonSink<'_, W> {
    fn begin(&mut self, columns: &[String]) -> io::Result<()> {
        self.begun = true;
        self.w.write_all(b"{\"columns\":")?;
        serde_json::to_writer(&mut *self.w, columns)?;
        self.w.write_all(b",\"rows\":[")
    }

    fn row(&mut self, row: &Row) -> io::Result<()> {
        let w = &mut *self.w;
        if self.rows > 0 {
            w.write_all(b",")?;
        }
        self.rows += 1;
        w.write_all(b"[")?;
        for (i, value) in row.as_slice().iter().enumerate() {
            if i > 0 {
                w.write_all(b",")?;
            }
            write_json_value(w, value)?;
        }
        w.write_all(b"]")
    }
}

/// A value as the table shows it: NULL bare, a TEXT that could be read as
/// another value or splits the row in quotes, everything else as it prints
pub fn write_value(w: &mut impl Write, value: &Value) -> io::Result<()> {
    match value {
        Value::Null(_) => w.write_all(b"NULL"),
        Value::Integer(i) => write!(w, "{i}"),
        Value::Float(f) => write_float(w, *f),
        Value::Text(s) if needs_quotes(s) => write_quoted(w, s),
        Value::Text(s) => w.write_all(s.as_bytes()),
        Value::Boolean(b) => w.write_all(if *b { b"true" } else { b"false" }),
        Value::Timestamp(ts) => write_timestamp(w, ts),
        Value::Extension(data) if data.first() == Some(&(stoolap::DataType::Json as u8)) => {
            w.write_all(&data[1..])
        }
        Value::Extension(data) if data.first() == Some(&(stoolap::DataType::Vector as u8)) => {
            w.write_all(stoolap::core::value::format_vector_bytes(&data[1..]).as_bytes())
        }
        Value::Extension(_) => w.write_all(b"<extension>"),
    }
}

/// A value as `write_value` shows it, for the interactive table
pub fn value_text(value: &Value) -> String {
    let mut text = Vec::new();
    let _ = write_value(&mut text, value);
    String::from_utf8_lossy(&text).into_owned()
}

/// Whether a TEXT shown bare could be read as NULL, a boolean, a number or
/// a quoted text, or would break the row it is in
fn needs_quotes(s: &str) -> bool {
    let Some(&first) = s.as_bytes().first() else {
        return true;
    };
    if s.bytes().any(|b| matches!(b, b'|' | b'\n' | b'\r' | b'\t')) || first == b'\'' {
        return true;
    }
    if ["null", "true", "false"]
        .iter()
        .any(|w| s.eq_ignore_ascii_case(w))
    {
        return true;
    }
    matches!(
        first,
        b'0'..=b'9' | b'+' | b'-' | b'.' | b'i' | b'I' | b'n' | b'N'
    ) && s.parse::<f64>().is_ok()
}

fn write_quoted(w: &mut impl Write, s: &str) -> io::Result<()> {
    w.write_all(b"'")?;
    let mut run = 0;
    for (i, b) in s.bytes().enumerate() {
        let escape: &[u8] = match b {
            b'\'' => b"''",
            b'\\' => b"\\\\",
            b'\n' => b"\\n",
            b'\r' => b"\\r",
            b'\t' => b"\\t",
            _ => continue,
        };
        w.write_all(&s.as_bytes()[run..i])?;
        w.write_all(escape)?;
        run = i + 1;
    }
    w.write_all(&s.as_bytes()[run..])?;
    w.write_all(b"'")
}

/// The shortest text that reads back as the same FLOAT, with `.0` on a
/// whole number so it does not read as an INTEGER
fn write_float(w: &mut impl Write, f: f64) -> io::Result<()> {
    if f.is_nan() {
        w.write_all(b"NaN")
    } else if f.is_infinite() {
        w.write_all(if f > 0.0 { b"Infinity" } else { b"-Infinity" })
    } else {
        write!(w, "{f:?}")
    }
}

/// RFC 3339 in UTC, with the fraction of a second only as long as it needs;
/// a leap second, which chrono keeps as second 59 with a billion
/// nanoseconds or more, is second 60
fn write_timestamp(w: &mut impl Write, ts: &chrono::DateTime<chrono::Utc>) -> io::Result<()> {
    use chrono::{Datelike, Timelike};
    let (second, mut nanos) = match ts.timestamp_subsec_nanos() {
        n if n >= 1_000_000_000 => (ts.second() + 1, n - 1_000_000_000),
        n => (ts.second(), n),
    };
    write!(
        w,
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
        ts.year(),
        ts.month(),
        ts.day(),
        ts.hour(),
        ts.minute(),
        second
    )?;
    if nanos > 0 {
        let mut digits = 9;
        while nanos.is_multiple_of(10) {
            nanos /= 10;
            digits -= 1;
        }
        write!(w, ".{nanos:0digits$}")?;
    }
    w.write_all(b"Z")
}

/// A value in JSON; NaN and the infinities, which JSON has no number for,
/// are strings rather than null
fn write_json_value(w: &mut impl Write, value: &Value) -> io::Result<()> {
    match value {
        Value::Null(_) => w.write_all(b"null"),
        Value::Integer(i) => write!(w, "{i}"),
        Value::Float(f) if f.is_finite() => {
            serde_json::to_writer(&mut *w, f).map_err(io::Error::from)
        }
        Value::Float(f) => {
            w.write_all(b"\"")?;
            write_float(w, *f)?;
            w.write_all(b"\"")
        }
        Value::Text(s) => serde_json::to_writer(&mut *w, s.as_str()).map_err(io::Error::from),
        Value::Boolean(b) => w.write_all(if *b { b"true" } else { b"false" }),
        Value::Timestamp(ts) => {
            w.write_all(b"\"")?;
            write_timestamp(w, ts)?;
            w.write_all(b"\"")
        }
        Value::Extension(data) if data.first() == Some(&(stoolap::DataType::Json as u8)) => {
            let json = std::str::from_utf8(&data[1..]).unwrap_or("");
            serde_json::to_writer(&mut *w, json).map_err(io::Error::from)
        }
        Value::Extension(_) => {
            let text = value_text(value);
            serde_json::to_writer(&mut *w, text.as_str()).map_err(io::Error::from)
        }
    }
}

fn param_value(value: &str) -> Value {
    if let Ok(i) = value.parse::<i64>() {
        return Value::Integer(i);
    }
    if value.contains('.')
        && let Ok(f) = value.parse::<f64>()
    {
        return Value::Float(f);
    }
    match value {
        "true" => Value::Boolean(true),
        "false" => Value::Boolean(false),
        _ if value.eq_ignore_ascii_case("null") => Value::null_unknown(),
        _ => Value::text(value),
    }
}

// The jemalloc feature, on by default, sets the library's own global allocator
#[cfg(all(test, not(feature = "jemalloc")))]
mod alloc_tests {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    use super::*;

    struct ThreadCounting;

    thread_local! {
        static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    }

    unsafe impl GlobalAlloc for ThreadCounting {
        // SAFETY: forwards to the system allocator; the counter is a const
        // thread-local Cell, which never allocates
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOCATIONS.with(|a| a.set(a.get() + 1));
            unsafe { System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            ALLOCATIONS.with(|a| a.set(a.get() + 1));
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    #[global_allocator]
    static COUNTING: ThreadCounting = ThreadCounting;

    /// Allocations of printing `sql` with `limit`, into a buffer with room
    fn allocations(db: &Database, sql: &str, format: Format, limit: usize) -> usize {
        let mut out = Output {
            out: Vec::with_capacity(1 << 22),
            format,
            limit,
            quiet: false,
            timing: false,
        };
        let before = ALLOCATIONS.with(Cell::get);
        print_statement(&mut Session::Database(db), sql, Vec::new(), &mut out, 0).unwrap();
        ALLOCATIONS.with(Cell::get) - before
    }

    #[test]
    fn a_printed_row_allocates_nothing() {
        let db = Database::open("memory://cli_printed_row_allocations").unwrap();
        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, s TEXT, q TEXT, f FLOAT, \
             ts TIMESTAMP, b BOOLEAN, n INTEGER)",
            (),
        )
        .unwrap();
        for id in 1..=2000 {
            db.execute(
                &format!(
                    "INSERT INTO t VALUES ({id}, 'name {id}', '{id}', {id}.25, \
                     '2024-01-01 10:00:00.5', TRUE, NULL)"
                ),
                (),
            )
            .unwrap();
        }
        let sql = "SELECT * FROM t";
        for format in [Format::Table, Format::Json] {
            let one = allocations(&db, sql, format, 1);
            let all = allocations(&db, sql, format, 0);
            assert!(
                all <= one + 2,
                "printing 1,999 more rows allocated {} times more",
                all - one
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(text: &str) -> Vec<(usize, String)> {
        let mut splitter = Splitter::default();
        let mut out = Vec::new();
        for line in text.lines() {
            splitter.push_line(line, &mut out);
        }
        splitter.finish(&mut out);
        out.into_iter().map(|s| (s.line, s.sql)).collect()
    }

    #[test]
    fn statements_end_at_semicolons_the_lexer_reads() {
        assert_eq!(
            split("SELECT 1; SELECT 'a;b';\n-- c; d\nSELECT \"x;y\"\n  FROM t;\nSELECT 2"),
            vec![
                (1, "SELECT 1".to_string()),
                (1, "SELECT 'a;b'".to_string()),
                (3, "-- c; d\nSELECT \"x;y\"\n  FROM t".to_string()),
                (5, "SELECT 2".to_string()),
            ]
        );
    }

    #[test]
    fn a_string_open_across_lines_holds_its_semicolons() {
        assert_eq!(
            split("SELECT 'a;\nb;';\n/* x;\ny */ SELECT 3;"),
            vec![
                (1, "SELECT 'a;\nb;'".to_string()),
                (4, "/* x;\ny */ SELECT 3".to_string()),
            ]
        );
    }

    fn params(text: &str) -> Vec<(String, Vec<Value>)> {
        let mut splitter = Splitter::default();
        let mut out = Vec::new();
        for line in text.split('\n') {
            splitter.push_line(line, &mut out);
        }
        splitter.finish(&mut out);
        out.into_iter().map(|s| (s.sql, s.params)).collect()
    }

    #[test]
    fn a_params_comment_belongs_to_its_statement() {
        assert_eq!(
            params(
                "SELECT $1; -- PARAMS: 7\nSELECT $1 -- PARAMS: 'b', 2\n;\nSELECT 'x -- PARAMS: y'"
            ),
            vec![
                ("SELECT $1".to_string(), vec![Value::Integer(7)]),
                (
                    "-- PARAMS: 7\nSELECT $1 -- PARAMS: 'b', 2".to_string(),
                    vec![Value::text("'b'"), Value::Integer(2)]
                ),
                ("SELECT 'x -- PARAMS: y'".to_string(), vec![]),
            ]
        );
        assert_eq!(
            params("SELECT $1; /* note */ # more\n"),
            vec![("SELECT $1".to_string(), vec![])]
        );
        assert_eq!(
            params("SELECT $1; /* note */ -- PARAMS: 7"),
            vec![("SELECT $1".to_string(), vec![Value::Integer(7)])]
        );
    }

    #[test]
    fn a_carriage_return_inside_a_string_is_kept() {
        assert_eq!(
            params("SELECT 'a\r\nb';\r\nSELECT 2\r"),
            vec![
                ("SELECT 'a\r\nb'".to_string(), vec![]),
                ("SELECT 2".to_string(), vec![])
            ]
        );
    }

    #[test]
    fn a_string_left_open_is_lexed_again_only_once_a_line_can_close_it() {
        let mut splitter = Splitter::default();
        let mut out = Vec::new();
        splitter.push_line("SELECT 1; SELECT 'a;", &mut out);
        let open_at = splitter.resume;
        for line in [
            "text; more;",
            "it''s; a doubled quote;",
            "a \\' quote; escaped",
        ]
        .repeat(500)
        {
            splitter.push_line(line, &mut out);
            assert!(
                !splitter.closed && splitter.resume == open_at,
                "{line} does not close the string"
            );
        }
        splitter.push_line("end';", &mut out);
        let sql: Vec<_> = out.iter().map(|s| s.sql.len()).collect();
        assert_eq!((out.len(), sql[0]), (2, "SELECT 1".len()));
        assert!(!splitter.is_pending());
    }

    #[test]
    fn statements_inside_an_open_block_comment_stay_in_it() {
        assert_eq!(
            split("/* start;\nSELECT 0;\nDROP TABLE t;\n*/ SELECT 1;\nSELECT 2;"),
            vec![
                (
                    4,
                    "/* start;\nSELECT 0;\nDROP TABLE t;\n*/ SELECT 1".to_string()
                ),
                (5, "SELECT 2".to_string()),
            ]
        );
        assert_eq!(
            split("SELECT 1; /* start;\nSELECT 0;\nDROP TABLE t;\n*/ SELECT 2;"),
            vec![
                (1, "SELECT 1".to_string()),
                (
                    4,
                    "/* start;\nSELECT 0;\nDROP TABLE t;\n*/ SELECT 2".to_string()
                ),
            ]
        );
    }

    #[test]
    fn a_command_is_read_past_its_comments() {
        let cases = [
            ("/* done */ COMMIT", Command::Commit),
            ("-- start\nBEGIN", Command::Begin),
            ("BEGIN TRANSACTION", Command::Begin),
            ("rollback", Command::Rollback),
            ("ROLLBACK TO SAVEPOINT s", Command::Other),
            ("/* c */ EXPLAIN SELECT 1", Command::Explain),
            ("SELECT value AS plan FROM t", Command::Other),
            ("exit", Command::Exit),
            ("\\?", Command::Help),
            ("\"COMMIT\"", Command::Other),
            ("\"ROLLBACK\";", Command::Other),
            ("/* note */ help;", Command::Help),
            ("`quit`", Command::Other),
            ("\"exit\";", Command::Other),
            ("/* note */ \\h;", Command::Help),
            ("\\H", Command::Help),
            ("\\Q;", Command::Exit),
            ("\\h /* note */;", Command::Help),
            ("\\q -- bye", Command::Exit),
            ("-- bye\nquit", Command::Exit),
            ("'help'", Command::Other),
        ];
        for (sql, expected) in cases {
            assert!(command(sql) == expected, "{sql}");
        }
    }

    #[test]
    fn a_leap_second_prints_as_second_sixty() {
        let ts = chrono::NaiveDate::from_ymd_opt(2016, 12, 31)
            .unwrap()
            .and_hms_nano_opt(23, 59, 59, 1_500_000_000)
            .unwrap()
            .and_utc();
        assert_eq!(value_text(&Value::Timestamp(ts)), "2016-12-31T23:59:60.5Z");
    }

    #[test]
    fn comments_and_blanks_alone_are_no_statement() {
        assert_eq!(split("-- only a comment;\n\n;\n# hash; comment"), vec![]);
    }

    #[test]
    fn a_text_that_reads_as_another_value_is_quoted() {
        let cases = [
            (Value::Null(stoolap::DataType::Text), "NULL"),
            (Value::text("NULL"), "'NULL'"),
            (Value::text("1"), "'1'"),
            (Value::Integer(1), "1"),
            (Value::text("x|y"), "'x|y'"),
            (Value::text("a\nb"), "'a\\nb'"),
            (Value::text("it's"), "it's"),
            (Value::text("'q'"), "'''q'''"),
            (Value::text(""), "''"),
            (Value::text("true"), "'true'"),
            (Value::text("nan"), "'nan'"),
            (Value::text("name"), "name"),
            (Value::Float(0.1 + 0.2), "0.30000000000000004"),
            (Value::Float(1.0), "1.0"),
            (Value::Float(f64::NAN), "NaN"),
        ];
        for (value, shown) in cases {
            assert_eq!(value_text(&value), shown, "{value:?}");
        }
    }
}
