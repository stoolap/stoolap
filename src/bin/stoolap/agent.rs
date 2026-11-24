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

//! Agent mode: a build line, one JSON line per statement and a summary,
//! written over the same run as every other mode of the CLI

use std::io::{self, Write};

use stoolap::common::version::{GIT_COMMIT, VERSION};
use stoolap::{DataType, Database, Row, Value};

use crate::script::{self, Command, Failure, Ran, Session, Sink, Splitter, Statement};

/// A cell longer than this many characters is cut unless `--full` is given
const CELL_LIMIT: usize = 1000;

/// How agent mode runs a script
pub struct Options {
    /// Rows shown per result, 0 for all
    pub limit: usize,
    /// Keep cells whole
    pub full: bool,
    /// Leave every result's rows out
    pub no_rows: bool,
    /// Skip the statements after the first that fails
    pub stop_on_error: bool,
    /// Add each statement's engine time
    pub timing: bool,
    /// Runs of a statement that only reads, the first written, 1 for once,
    /// and the unmeasured runs between the first and the measured ones
    pub repeat: usize,
    pub warmup: usize,
}

/// Opens the database the `n`th case runs in
pub type OpenCase<'a> = &'a dyn Fn(usize) -> Result<Database, String>;

/// Runs a script in agent mode; true when no statement failed. Statements
/// before the first `-- @case` line run in `db`, each case in a database
/// `open_case` gives
pub fn run_script<W: Write>(
    db: &Database,
    dsn: &str,
    open_case: OpenCase,
    lines: impl Iterator<Item = io::Result<String>>,
    w: &mut W,
    options: &Options,
) -> io::Result<bool> {
    write_build_line(w, dsn)?;
    let mut run = Run {
        first: db,
        case_db: None,
        open_case,
        options,
        sink: AgentSink::default(),
        number: 0,
        failed: 0,
        first_failed: None,
        skipped: 0,
        stopped: false,
        exited: false,
        case: None,
        cases: Vec::new(),
    };
    let mut splitter = Splitter::default();
    let mut ready = Vec::new();
    for (number, line) in lines.enumerate() {
        let line = line?;
        // A line starting with a dot between statements is a command of
        // the CLI; the splitter takes an empty line in its place to keep
        // counting lines
        if line.trim_start().starts_with('.') && !splitter.in_statement() {
            if let Some(name) = splitter.take_case() {
                run.start_case(name)?;
            }
            splitter.push_line("", &mut ready);
            run.dot_command(line.trim(), number + 1, w)?;
        } else {
            splitter.push_line(&line, &mut ready);
            run.statements(&mut ready, w)?;
        }
        if run.exited {
            break;
        }
    }
    if !run.exited {
        splitter.finish(&mut ready);
        run.statements(&mut ready, w)?;
    }
    run.summary(w)?;
    Ok(run.failed == 0)
}

fn write_build_line(w: &mut impl Write, dsn: &str) -> io::Result<()> {
    let build = match cfg!(debug_assertions) {
        true => "debug",
        false => "release",
    };
    let features: Vec<&str> = [
        ("parallel", cfg!(feature = "parallel")),
        ("jemalloc", cfg!(feature = "jemalloc")),
        ("semantic", cfg!(feature = "semantic")),
        ("heap-profile", cfg!(feature = "heap-profile")),
    ]
    .into_iter()
    .filter_map(|(name, on)| on.then_some(name))
    .collect();
    write!(w, "{{\"stoolap\":")?;
    serde_json::to_writer(&mut *w, VERSION)?;
    write!(w, ",\"commit\":")?;
    serde_json::to_writer(&mut *w, GIT_COMMIT)?;
    write!(w, ",\"build\":\"{build}\",\"features\":")?;
    serde_json::to_writer(&mut *w, &features)?;
    write!(w, ",\"dsn\":")?;
    serde_json::to_writer(&mut *w, dsn)?;
    writeln!(w, "}}")?;
    w.flush()
}

/// The state of a script run in agent mode
struct Run<'a> {
    /// The database of the statements before the first case
    first: &'a Database,
    /// The database of the case running, closed when the next starts
    case_db: Option<Database>,
    open_case: OpenCase<'a>,
    options: &'a Options,
    sink: AgentSink,
    number: usize,
    failed: usize,
    first_failed: Option<usize>,
    skipped: usize,
    /// Under `--on-error stop`, a statement of the case failed
    stopped: bool,
    exited: bool,
    case: Option<Case>,
    cases: Vec<Case>,
}

/// A case's name and counts, and whether it left a transaction open
struct Case {
    name: String,
    statements: usize,
    failed: usize,
    open_transaction: &'static str,
}

/// Whether `db`'s session has an explicit transaction open, read with
/// CURRENT_TRANSACTION_ID(), NULL outside one: `true`, `false` or `null`
fn open_transaction(db: &Database) -> &'static str {
    match db.query("SELECT CURRENT_TRANSACTION_ID()", ()) {
        Ok(mut rows) => match rows.next() {
            Some(Ok(row)) => match row.get_value(0) {
                Some(Value::Null(_)) | None => "false",
                Some(_) => "true",
            },
            _ => "null",
        },
        Err(_) => "null",
    }
}

impl Run<'_> {
    /// Counts the statement running as failed, in the run and its case,
    /// and under `--on-error stop` skips the rest of the case
    fn count_failure(&mut self) {
        self.failed += 1;
        if let Some(case) = &mut self.case {
            case.failed += 1;
        }
        self.first_failed.get_or_insert(self.number);
        self.stopped = self.options.stop_on_error;
    }

    fn db(&self) -> &Database {
        self.case_db.as_ref().unwrap_or(self.first)
    }

    /// Ends the case running, noting whether it left a transaction open
    fn end_case(&mut self) {
        if let Some(mut case) = self.case.take() {
            case.open_transaction = open_transaction(self.db());
            self.cases.push(case);
        }
    }

    /// Starts case `name` in a database of its own
    fn start_case(&mut self, name: String) -> io::Result<()> {
        self.end_case();
        self.case_db = None;
        let db = (self.open_case)(self.cases.len() + 1).map_err(io::Error::other)?;
        self.case_db = Some(db);
        self.stopped = false;
        self.case = Some(Case {
            name,
            statements: 0,
            failed: 0,
            open_transaction: "false",
        });
        Ok(())
    }

    fn statements(&mut self, ready: &mut Vec<Statement>, w: &mut impl Write) -> io::Result<()> {
        for mut statement in ready.drain(..) {
            if self.exited {
                break;
            }
            if let Some(name) = statement.case.take() {
                self.start_case(name)?;
            }
            self.number += 1;
            if let Some(case) = &mut self.case {
                case.statements += 1;
            }
            if self.stopped {
                self.skipped += 1;
                continue;
            }
            match script::command(&statement.sql) {
                Command::Exit => {
                    self.exited = true;
                    break;
                }
                Command::Help => {
                    writeln!(
                        w,
                        "{{\"stmt\":{},\"line\":{},\"ok\":true,\"ignored\":\"help\"}}",
                        self.number, statement.line
                    )?;
                }
                _ => self.statement(statement, w)?,
            }
            w.flush()?;
        }
        Ok(())
    }

    fn statement(&mut self, statement: Statement, w: &mut impl Write) -> io::Result<()> {
        let explain = script::command(&statement.sql) == Command::Explain;
        let ordered = !explain && script::orders_rows(&statement.sql);
        let repeating = self.options.repeat > 1 && !explain && script::reads_only(&statement.sql);
        // A first run the warm ones are set against is timed apart from
        // the sink's hash and types, as they run without them
        let timed = self.options.timing || repeating;
        self.sink.reset(explain, self.options, ordered, timed);
        let db = self.case_db.as_ref().unwrap_or(self.first);
        let ran = script::run(
            &mut Session::Database(db),
            &statement.sql,
            statement.params.clone(),
            self.options.limit,
            &mut self.sink,
        );
        write!(w, "{{\"stmt\":{},", self.number)?;
        if let Some(case) = &self.case {
            write!(w, "\"case\":")?;
            serde_json::to_writer(&mut *w, &case.name)?;
            write!(w, ",")?;
        }
        write!(w, "\"line\":{},", statement.line)?;
        let took = match ran {
            Err(failure) => {
                write!(w, "\"ok\":false,\"error\":")?;
                write_error(w, &failure, &statement, db)?;
                self.count_failure();
                return writeln!(w, "}}");
            }
            Ok((Ran::Affected(n), took)) => {
                write!(w, "\"ok\":true,\"rows_affected\":{n}")?;
                took
            }
            Ok((Ran::Rows { .. }, took)) if explain => {
                write!(w, "\"ok\":true,\"plan\":")?;
                serde_json::to_writer(&mut *w, &self.sink.plan)?;
                took
            }
            Ok((
                Ran::Rows {
                    columns,
                    total,
                    shown,
                },
                took,
            )) => {
                write!(w, "\"ok\":true,\"columns\":")?;
                serde_json::to_writer(&mut *w, &columns)?;
                self.sink.write_result(w, total, shown)?;
                if repeating {
                    let repeats = script::repeat(
                        &mut Session::Database(db),
                        &statement.sql,
                        &statement.params,
                        took.engine,
                        (self.options.warmup, self.options.repeat - 1),
                    );
                    w.write_all(b",\"repeat\":")?;
                    repeats.write_json(w)?;
                    // A run after the first that failed fails the statement
                    if repeats.failed.is_some() {
                        self.count_failure();
                    }
                }
                took
            }
        };
        if self.options.repeat > 1 && !repeating {
            let why = if explain { "explain" } else { "not_read_only" };
            write!(w, ",\"repeat\":{{\"runs\":1,\"skipped\":\"{why}\"}}")?;
        }
        if self.options.timing {
            write!(w, ",\"elapsed_us\":{}", script::Micros(took.engine))?;
        }
        writeln!(w, "}}")
    }

    fn summary(&mut self, w: &mut impl Write) -> io::Result<()> {
        let open = open_transaction(self.db());
        self.end_case();
        write!(
            w,
            "{{\"done\":true,\"statements\":{},\"failed\":{}",
            self.number, self.failed
        )?;
        if let Some(first) = self.first_failed {
            write!(w, ",\"first_failed\":{first}")?;
        }
        if self.options.stop_on_error {
            write!(w, ",\"skipped\":{}", self.skipped)?;
        }
        write!(w, ",\"open_transaction\":{open}")?;
        if !self.cases.is_empty() {
            w.write_all(b",\"cases\":[")?;
            for (i, case) in self.cases.iter().enumerate() {
                if i > 0 {
                    w.write_all(b",")?;
                }
                w.write_all(b"{\"name\":")?;
                serde_json::to_writer(&mut *w, &case.name)?;
                write!(
                    w,
                    ",\"statements\":{},\"failed\":{},\"open_transaction\":{}}}",
                    case.statements, case.failed, case.open_transaction
                )?;
            }
            w.write_all(b"]")?;
        }
        writeln!(w, "}}")?;
        w.flush()
    }
}

/// `{"kind":...,"message":...}`, with the line and column in the input of
/// a parse error
impl Run<'_> {
    /// `.schema [table]` or `.stats`, written as a statement's line is
    fn dot_command(&mut self, text: &str, line: usize, w: &mut impl Write) -> io::Result<()> {
        self.number += 1;
        if let Some(case) = &mut self.case {
            case.statements += 1;
        }
        if self.stopped {
            self.skipped += 1;
            return Ok(());
        }
        let text = text.trim_end_matches(';').trim();
        let (name, arg) = text.split_once(char::is_whitespace).unwrap_or((text, ""));
        let arg = arg.trim();
        let db = self.case_db.as_ref().unwrap_or(self.first);
        let body = match name {
            ".schema" => schema(db, (!arg.is_empty()).then_some(arg)),
            ".stats" if arg.is_empty() => stats(db),
            _ => Err(("unknown_command", format!("unknown command: {text}"))),
        };
        write!(w, "{{\"stmt\":{},", self.number)?;
        if let Some(case) = &self.case {
            write!(w, "\"case\":")?;
            serde_json::to_writer(&mut *w, &case.name)?;
            write!(w, ",")?;
        }
        write!(w, "\"line\":{line},")?;
        match body {
            Ok(body) => {
                write!(w, "\"ok\":true,\"command\":\"{name}\",")?;
                w.write_all(&body)?;
            }
            Err((kind, message)) => {
                self.count_failure();
                write!(
                    w,
                    "\"ok\":false,\"error\":{{\"kind\":\"{kind}\",\"message\":"
                )?;
                serde_json::to_writer(&mut *w, &message)?;
                w.write_all(b"}")?;
            }
        }
        writeln!(w, "}}")?;
        w.flush()
    }
}

/// A command's failure: its kind and its message
type CommandError = (&'static str, String);

fn engine_error(e: stoolap::Error) -> CommandError {
    (e.kind(), e.to_string())
}

fn output_error(e: io::Error) -> CommandError {
    ("output", e.to_string())
}

/// A name in double quotes, as SQL reads it whatever it holds
fn quoted(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// The first column of every row `sql` returns, as text
fn first_column(db: &Database, sql: &str) -> Result<Vec<String>, CommandError> {
    let rows = db.query(sql, ()).map_err(engine_error)?;
    rows.map(|row| row.and_then(|r| r.get::<String>(0)).map_err(engine_error))
        .collect()
}

/// `"tables":[{"table":..,"create":..,"indexes":[..],"row_count":..}]`,
/// for every table or the one named, and `"views"` for every table
fn schema(db: &Database, table: Option<&str>) -> Result<Vec<u8>, CommandError> {
    let tables = match table {
        Some(table) => vec![table.to_string()],
        None => first_column(db, "SHOW TABLES")?,
    };
    let mut out = Vec::new();
    out.extend_from_slice(b"\"tables\":[");
    for (i, table) in tables.iter().enumerate() {
        let name = quoted(table);
        let create = db
            .query(&format!("SHOW CREATE TABLE {name}"), ())
            .and_then(|mut rows| match rows.next() {
                Some(row) => row.and_then(|r| r.get::<String>(1)),
                None => Ok(String::new()),
            })
            .map_err(engine_error)?;
        let count: i64 = db
            .query_one(&format!("SELECT COUNT(*) FROM {name}"), ())
            .map_err(engine_error)?;
        if i > 0 {
            out.push(b',');
        }
        out.extend_from_slice(b"{\"table\":");
        serde_json::to_writer(&mut out, table).map_err(|e| output_error(e.into()))?;
        out.extend_from_slice(b",\"create\":");
        serde_json::to_writer(&mut out, &create).map_err(|e| output_error(e.into()))?;
        out.extend_from_slice(b",\"indexes\":");
        write_objects(&mut out, db, &format!("SHOW INDEXES FROM {name}"))?;
        write!(out, ",\"row_count\":{count}}}").map_err(output_error)?;
    }
    out.push(b']');
    if table.is_none() {
        out.extend_from_slice(b",\"views\":");
        let views = first_column(db, "SHOW VIEWS")?;
        serde_json::to_writer(&mut out, &views).map_err(|e| output_error(e.into()))?;
    }
    Ok(out)
}

/// The rows `sql` returns as objects keyed by column name
fn write_objects(out: &mut Vec<u8>, db: &Database, sql: &str) -> Result<(), CommandError> {
    let rows = db.query(sql, ()).map_err(engine_error)?;
    let columns = rows.columns().to_vec();
    out.push(b'[');
    for (i, row) in rows.enumerate() {
        let row = row.map_err(engine_error)?;
        if i > 0 {
            out.push(b',');
        }
        out.push(b'{');
        for (c, column) in columns.iter().enumerate() {
            if c > 0 {
                out.push(b',');
            }
            serde_json::to_writer(&mut *out, column).map_err(|e| output_error(e.into()))?;
            out.push(b':');
            let value = row.get_value(c).cloned().unwrap_or(Value::null_unknown());
            script::write_json_value(out, &value).map_err(output_error)?;
        }
        out.push(b'}');
    }
    out.push(b']');
    Ok(())
}

/// The memory, volume and index statistics, each as its rows of objects
fn stats(db: &Database) -> Result<Vec<u8>, CommandError> {
    let mut out = Vec::new();
    for (i, (key, pragma)) in [
        ("memory_stats", "PRAGMA MEMORY_STATS"),
        ("volume_stats", "PRAGMA VOLUME_STATS"),
        ("index_stats", "PRAGMA INDEX_STATS"),
    ]
    .into_iter()
    .enumerate()
    {
        if i > 0 {
            out.push(b',');
        }
        write!(out, "\"{key}\":").map_err(output_error)?;
        write_objects(&mut out, db, pragma)?;
    }
    Ok(out)
}

fn write_error(
    w: &mut impl Write,
    failure: &Failure,
    statement: &Statement,
    db: &Database,
) -> io::Result<()> {
    let kind = match failure {
        Failure::Engine(e) => e.kind(),
        Failure::Output(_) => "output",
    };
    write!(w, "{{\"kind\":\"{kind}\",\"message\":")?;
    serde_json::to_writer(&mut *w, &failure.to_string())?;
    if let Failure::Engine(stoolap::Error::Parse(_)) = failure
        && let Err(errors) = stoolap::parser::parse_sql(&statement.sql)
        && let Some(error) = errors.errors.first()
    {
        let at = (error.position.line as usize, error.position.column as usize);
        let (line, column) = statement.input_position(at);
        write!(w, ",\"line\":{line},\"column\":{column}")?;
    }
    // The connection's timeout now, which the script may have set itself
    if let Failure::Engine(stoolap::Error::QueryCancelled) = failure
        && let Ok(timeout) = db.query_one::<i64, _>("PRAGMA query_timeout", ())
        && timeout > 0
    {
        write!(
            w,
            ",\"timeout_ms\":{timeout},\"hint\":\"the statement timeout of {timeout} ms was \
             reached; -t MS sets it, -t 0 turns it off\""
        )?;
    }
    write!(w, "}}")
}

/// A result's hash, stable from build to build: every row's values and
/// their types, summed for a result in no order, folded in order for one
/// with an ORDER BY, so two builds with the same hash gave the same answer
#[derive(Default)]
struct ResultHash {
    acc: u64,
    ordered: bool,
}

/// "STOOLAP1", the start of every row's hash
const SEED: u64 = 0x5354_4f4f_4c41_5031;

fn mix(h: u64, word: u64) -> u64 {
    (h ^ word)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .rotate_left(29)
}

/// MurmurHash3's 64-bit finalizer
fn finish(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^ (h >> 33)
}

fn mix_bytes(mut h: u64, bytes: &[u8]) -> u64 {
    h = mix(h, bytes.len() as u64);
    for chunk in bytes.chunks(8) {
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        h = mix(h, u64::from_le_bytes(word));
    }
    h
}

/// A value with a tag of its type, so `1` and `'1'`, NULL and `'NULL'`
/// differ; a NaN is one NaN whatever its bits
fn mix_value(h: u64, value: &Value) -> u64 {
    match value {
        Value::Null(_) => mix(h, 0),
        Value::Integer(i) => mix(mix(h, 1), *i as u64),
        Value::Float(f) if f.is_nan() => mix(mix(h, 2), 0x7ff8_0000_0000_0000),
        Value::Float(f) => mix(mix(h, 2), f.to_bits()),
        Value::Text(s) => mix_bytes(mix(h, 3), s.as_bytes()),
        Value::Boolean(b) => mix(mix(h, 4), *b as u64),
        // Seconds and the nanoseconds within them apart: a time past the
        // range of i64 nanoseconds keeps its fraction
        Value::Timestamp(ts) => mix(
            mix(mix(h, 5), ts.timestamp() as u64),
            ts.timestamp_subsec_nanos() as u64,
        ),
        Value::Extension(data) => match data.split_first() {
            Some((&kind, bytes)) => mix_bytes(mix(h, 16 + kind as u64), bytes),
            None => mix(h, 16),
        },
    }
}

impl ResultHash {
    fn add(&mut self, row: &Row) {
        let row = finish(row.as_slice().iter().fold(SEED, mix_value));
        self.acc = match self.ordered {
            true => mix(self.acc, row),
            false => self.acc.wrapping_add(row),
        };
    }

    fn value(&self, rows: u64) -> u64 {
        finish(mix(self.acc, rows))
    }
}

/// The type a column's values have
#[derive(Clone, Copy, PartialEq)]
enum ColumnType {
    /// No value other than NULL yet
    Unknown,
    Single(DataType),
    Mixed,
}

/// The shown rows of a result, written as JSON into one buffer as they
/// come, cell by cell, so a column found to hold values of more than one
/// type can have its cells tagged when the result is written. The buffers
/// are kept from statement to statement
#[derive(Default)]
struct AgentSink {
    /// The cells' JSON, one after another
    cells: Vec<u8>,
    /// Each cell's end in `cells` and its value's type, row after row
    spans: Vec<(usize, DataType)>,
    types: Vec<ColumnType>,
    columns: usize,
    cells_cut: bool,
    full: bool,
    /// Write the rows; without, `--no-rows`, they count in the types and the hash only
    keep_rows: bool,
    /// Whether the statement is EXPLAIN, whose lines go to `plan`
    explain: bool,
    plan: Vec<String>,
    /// Every row, the shown ones and the rest
    hash: ResultHash,
    /// `--timing`: the engine's time is measured apart from the sink's
    timed: bool,
}

impl AgentSink {
    fn reset(&mut self, explain: bool, options: &Options, ordered: bool, timed: bool) {
        self.timed = timed;
        self.cells.clear();
        self.spans.clear();
        self.types.clear();
        self.columns = 0;
        self.cells_cut = false;
        self.full = options.full;
        self.keep_rows = !options.no_rows;
        self.explain = explain;
        self.plan.clear();
        self.hash = ResultHash { acc: 0, ordered };
    }

    /// Notes the types of a row's values: a column holding more than one
    /// type, NULL aside, is MIXED
    fn see_types(&mut self, row: &Row) {
        for (seen, value) in self.types.iter_mut().zip(row.as_slice()) {
            if let Value::Null(_) = value {
                continue;
            }
            let t = value.data_type();
            *seen = match *seen {
                ColumnType::Unknown => ColumnType::Single(t),
                ColumnType::Single(s) if s == t => *seen,
                _ => ColumnType::Mixed,
            };
        }
    }

    fn write_result(&self, w: &mut impl Write, total: u64, shown: u64) -> io::Result<()> {
        write!(w, ",\"types\":[")?;
        for (i, t) in self.types.iter().enumerate() {
            if i > 0 {
                w.write_all(b",")?;
            }
            match t {
                ColumnType::Unknown => w.write_all(b"null")?,
                ColumnType::Single(t) => write!(w, "\"{t}\"")?,
                ColumnType::Mixed => w.write_all(b"\"MIXED\"")?,
            }
        }
        w.write_all(b"]")?;
        debug_assert!(self.keep_rows || self.spans.is_empty());
        if self.keep_rows {
            w.write_all(b",\"rows\":[")?;
        }
        let mut from = 0;
        for (i, &(end, t)) in self.spans.iter().enumerate() {
            let column = i % self.columns;
            if column == 0 {
                if i > 0 {
                    w.write_all(b"],")?;
                }
                w.write_all(b"[")?;
            } else {
                w.write_all(b",")?;
            }
            let cell = &self.cells[from..end];
            from = end;
            match self.types[column] == ColumnType::Mixed && t != DataType::Null {
                true => {
                    write!(w, "{{\"t\":\"{t}\",\"v\":")?;
                    w.write_all(cell)?;
                    w.write_all(b"}")?;
                }
                false => w.write_all(cell)?,
            }
        }
        if !self.spans.is_empty() {
            w.write_all(b"]")?;
        }
        if self.keep_rows {
            w.write_all(b"]")?;
        }
        write!(w, ",\"row_count\":{total}")?;
        let order = match self.hash.ordered {
            true => "ordered",
            false => "unordered",
        };
        let hash = self.hash.value(total);
        write!(w, ",\"hash\":\"{hash:016x}\",\"order\":\"{order}\"")?;
        if self.keep_rows && shown < total {
            write!(w, ",\"shown\":{shown},\"truncated\":true")?;
        }
        if self.cells_cut {
            w.write_all(b",\"cells_truncated\":true")?;
        }
        Ok(())
    }

    /// A value as agent mode writes it: JSON as itself, a vector as an
    /// array, NaN and the infinities as strings, a long text cut
    fn write_value(&mut self, value: &Value) -> io::Result<()> {
        let w = &mut self.cells;
        match value {
            Value::Text(s) => {
                let text = self.cut(s.as_str());
                serde_json::to_writer(&mut self.cells, text)?;
            }
            Value::Extension(data) if data.first() == Some(&(DataType::Json as u8)) => {
                let json = std::str::from_utf8(&data[1..]).unwrap_or("");
                // Written as JSON only when it reads as JSON, NaN does not.
                // Valid JSON holds a newline only between its tokens, never
                // in a string, so a space in its place puts it on one line
                // with every token, numbers and strings, as it was
                let valid = serde_json::from_str::<serde::de::IgnoredAny>(json).is_ok();
                match valid && (self.full || json.chars().count() <= CELL_LIMIT) {
                    true => self.cells.extend(json.bytes().map(|b| match b {
                        b'\n' | b'\r' => b' ',
                        b => b,
                    })),
                    false => {
                        let text = self.cut(json);
                        serde_json::to_writer(&mut self.cells, text)?;
                    }
                }
            }
            Value::Extension(data) if data.first() == Some(&(DataType::Vector as u8)) => {
                let text = stoolap::core::value::format_vector_bytes(&data[1..]);
                match text.contains(|c: char| c.is_ascii_alphabetic()) {
                    // NaN or inf: not JSON numbers
                    true => serde_json::to_writer(&mut *w, &text)?,
                    false => w.extend_from_slice(text.as_bytes()),
                }
            }
            _ => script::write_json_value(w, value)?,
        }
        Ok(())
    }

    /// `text` cut to `CELL_LIMIT` characters unless the output keeps cells
    /// whole, noting the cut
    fn cut<'t>(&mut self, text: &'t str) -> &'t str {
        if self.full {
            return text;
        }
        match text.char_indices().nth(CELL_LIMIT) {
            Some((end, _)) => {
                self.cells_cut = true;
                &text[..end]
            }
            None => text,
        }
    }
}

impl Sink for AgentSink {
    fn begin(&mut self, columns: &[String]) -> io::Result<()> {
        self.columns = columns.len();
        self.types.resize(columns.len(), ColumnType::Unknown);
        Ok(())
    }

    fn row(&mut self, row: &Row) -> io::Result<()> {
        if self.explain {
            if let Some(Value::Text(line)) = row.as_slice().first() {
                self.plan.push(line.to_string());
            }
            return Ok(());
        }
        if !self.keep_rows {
            return self.rest(row);
        }
        for value in row.as_slice() {
            self.write_value(value)?;
            let t = match value {
                Value::Null(_) => DataType::Null,
                _ => value.data_type(),
            };
            self.spans.push((self.cells.len(), t));
        }
        self.see_types(row);
        self.hash.add(row);
        Ok(())
    }

    /// The rows past the limit count in the types and the hash
    fn wants_rest(&self) -> bool {
        !self.explain
    }

    fn rest(&mut self, row: &Row) -> io::Result<()> {
        self.see_types(row);
        self.hash.add(row);
        Ok(())
    }

    /// Without `--timing` the output holds no times, so no row pays for
    /// reading the clock
    fn timed(&self) -> bool {
        self.timed
    }
}
