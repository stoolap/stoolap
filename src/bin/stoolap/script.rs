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
use stoolap::parser::token::{Token, TokenType};
use stoolap::{ApiTransaction, Database, Row, Rows, Value};

/// A statement of a script and the line its first token is on
pub struct Statement {
    pub line: usize,
    pub sql: String,
}

/// Cuts SQL text into statements at the semicolons the engine's lexer
/// reads, so quotes and comments follow the parser's own rules
#[derive(Default)]
pub struct Splitter {
    /// Text read since the last statement was cut
    pending: String,
    /// Lines before the first byte of `pending`
    pending_line: usize,
}

impl Splitter {
    /// Adds one line of input and hands out the statements it completes
    pub fn push_line(&mut self, line: &str, out: &mut Vec<Statement>) {
        self.pending.push_str(line);
        self.pending.push('\n');
        if line.contains(';') {
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
        let mut lexer = Lexer::new(&self.pending);
        let mut held: Option<Token> = None;
        let mut start = 0;
        let mut first_line = None;
        loop {
            let token = held.take().unwrap_or_else(|| lexer.next_token());
            match token.token_type {
                TokenType::Eof => break,
                TokenType::Comment => {}
                TokenType::Punctuator if token.literal == ";" => {
                    let end = token.position.offset as usize;
                    if let Some(line) = first_line.take() {
                        out.push(self.statement(line, start, end));
                    }
                    start = end + 1;
                }
                TokenType::Error if !at_end => {
                    // An unclosed string, name or comment runs to the end of
                    // the text: the statement goes on in the next line
                    let next = lexer.next_token();
                    if next.token_type == TokenType::Eof {
                        break;
                    }
                    first_line.get_or_insert(token.position.line);
                    held = Some(next);
                }
                _ => {
                    first_line.get_or_insert(token.position.line);
                }
            }
        }
        if at_end {
            if let Some(line) = first_line {
                out.push(self.statement(line, start, self.pending.len()));
            }
            start = self.pending.len();
        }
        self.pending_line += self.pending[..start].matches('\n').count();
        self.pending.drain(..start);
    }

    fn statement(&self, line: u32, start: usize, end: usize) -> Statement {
        Statement {
            line: self.pending_line + line as usize,
            sql: self.pending[start..end].trim().to_string(),
        }
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

/// What a statement gave: rows, or the count of a statement without them
pub enum Ran {
    Rows { columns: Vec<String>, total: u64 },
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
    timeout_ms: u64,
    limit: usize,
    sink: &mut impl Sink,
) -> Result<(Ran, Timing), String> {
    let (sql, params) = parse_params(sql);
    let start = Instant::now();
    let mut rows = session
        .query(&sql, params, timeout_ms)
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
    // A plan is shown whole whatever the limit
    let limit = match columns.len() == 1 && columns[0] == "plan" {
        true => 0,
        false => limit,
    };
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
    Ok((Ran::Rows { columns, total }, Timing { engine, output }))
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
        let word = statement.sql.trim_end_matches(';').trim();
        if ["exit", "quit", "\\q"]
            .iter()
            .any(|w| word.eq_ignore_ascii_case(w))
        {
            return Ok(false);
        }
        if ["help", "\\h", "\\?"]
            .iter()
            .any(|w| word.eq_ignore_ascii_case(w))
        {
            super::print_help_main();
            continue;
        }
        let printed = print_statement(&mut Session::Database(db), &statement.sql, out, timeout_ms);
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
        Format::Table => run(session, sql, timeout_ms, *limit, &mut TableSink(w)),
        Format::Json => {
            let mut sink = JsonSink {
                w,
                begun: false,
                rows: 0,
            };
            let ran = run(session, sql, timeout_ms, *limit, &mut sink);
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
        (Ran::Rows { columns, total }, Format::Json) => {
            let shown = shown_rows(&columns, total, *limit);
            match shown < total {
                true => writeln!(w, "],\"count\":{total},\"shown\":{shown}}}"),
                false => writeln!(w, "],\"count\":{total}}}"),
            }
        }
        (Ran::Rows { columns, total }, Format::Table) => {
            let shown = shown_rows(&columns, total, *limit);
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

/// How many of `total` rows a result with these columns showed
fn shown_rows(columns: &[String], total: u64, limit: usize) -> u64 {
    match limit == 0 || (columns.len() == 1 && columns[0] == "plan") {
        true => total,
        false => total.min(limit as u64),
    }
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

/// RFC 3339 in UTC, with the fraction of a second only as long as it needs
fn write_timestamp(w: &mut impl Write, ts: &chrono::DateTime<chrono::Utc>) -> io::Result<()> {
    use chrono::{Datelike, Timelike};
    write!(
        w,
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
        ts.year(),
        ts.month(),
        ts.day(),
        ts.hour(),
        ts.minute(),
        ts.second()
    )?;
    let mut nanos = ts.timestamp_subsec_nanos();
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

/// `SELECT ... -- PARAMS: 1, 'a'` runs the statement with those parameters
fn parse_params(query: &str) -> (String, Vec<Value>) {
    let mut parts = query.splitn(2, " -- PARAMS: ");
    let sql = parts
        .next()
        .unwrap_or_default()
        .trim()
        .trim_end_matches(';')
        .to_string();
    let params = parts
        .next()
        .map(|list| list.split(',').map(|v| param_value(v.trim())).collect())
        .unwrap_or_default();
    (sql, params)
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
        print_statement(&mut Session::Database(db), sql, &mut out, 0).unwrap();
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
