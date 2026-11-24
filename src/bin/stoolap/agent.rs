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
    /// Skip the statements after the first that fails
    pub stop_on_error: bool,
    /// Add each statement's engine time
    pub timing: bool,
}

/// Runs a script in agent mode; true when no statement failed
pub fn run_script<W: Write>(
    db: &Database,
    dsn: &str,
    lines: impl Iterator<Item = io::Result<String>>,
    w: &mut W,
    options: &Options,
) -> io::Result<bool> {
    write_build_line(w, dsn)?;
    let mut run = Run {
        db,
        options,
        sink: AgentSink::default(),
        number: 0,
        failed: 0,
        first_failed: None,
        skipped: 0,
        stopped: false,
        exited: false,
    };
    let mut splitter = Splitter::default();
    let mut ready = Vec::new();
    for line in lines {
        splitter.push_line(&line?, &mut ready);
        run.statements(&mut ready, w)?;
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
    db: &'a Database,
    options: &'a Options,
    sink: AgentSink,
    number: usize,
    failed: usize,
    first_failed: Option<usize>,
    skipped: usize,
    stopped: bool,
    exited: bool,
}

impl Run<'_> {
    fn statements(&mut self, ready: &mut Vec<Statement>, w: &mut impl Write) -> io::Result<()> {
        for statement in ready.drain(..) {
            if self.exited {
                break;
            }
            self.number += 1;
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
        self.sink.reset(explain, self.options.full);
        let ran = script::run(
            &mut Session::Database(self.db),
            &statement.sql,
            statement.params.clone(),
            self.options.limit,
            &mut self.sink,
        );
        write!(w, "{{\"stmt\":{},\"line\":{},", self.number, statement.line)?;
        let took = match ran {
            Err(failure) => {
                self.failed += 1;
                self.first_failed.get_or_insert(self.number);
                self.stopped = self.options.stop_on_error;
                write!(w, "\"ok\":false,\"error\":")?;
                write_error(w, &failure, &statement)?;
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
                took
            }
        };
        if self.options.timing {
            write!(w, ",\"elapsed_us\":{}", took.engine.as_micros())?;
        }
        writeln!(w, "}}")
    }

    fn summary(&self, w: &mut impl Write) -> io::Result<()> {
        // NULL outside an explicit transaction, its id inside one
        let open = match self.db.query("SELECT CURRENT_TRANSACTION_ID()", ()) {
            Ok(mut rows) => match rows.next() {
                Some(Ok(row)) => match row.get_value(0) {
                    Some(Value::Null(_)) | None => "false",
                    Some(_) => "true",
                },
                _ => "null",
            },
            Err(_) => "null",
        };
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
        writeln!(w, ",\"open_transaction\":{open}}}")?;
        w.flush()
    }
}

/// `{"kind":...,"message":...}`, with the line and column in the input of
/// a parse error
fn write_error(w: &mut impl Write, failure: &Failure, statement: &Statement) -> io::Result<()> {
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
    write!(w, "}}")
}

/// The type a column's shown values have
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
    /// Whether the statement is EXPLAIN, whose lines go to `plan`
    explain: bool,
    plan: String,
}

impl AgentSink {
    fn reset(&mut self, explain: bool, full: bool) {
        self.cells.clear();
        self.spans.clear();
        self.types.clear();
        self.columns = 0;
        self.cells_cut = false;
        self.full = full;
        self.explain = explain;
        self.plan.clear();
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
        w.write_all(b"],\"rows\":[")?;
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
        write!(w, "],\"row_count\":{total}")?;
        if shown < total {
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
                if !self.plan.is_empty() {
                    self.plan.push('\n');
                }
                self.plan.push_str(line);
            }
            return Ok(());
        }
        for (column, value) in row.as_slice().iter().enumerate() {
            self.write_value(value)?;
            let t = match value {
                Value::Null(_) => DataType::Null,
                _ => value.data_type(),
            };
            self.spans.push((self.cells.len(), t));
            if t != DataType::Null {
                let seen = &mut self.types[column];
                *seen = match *seen {
                    ColumnType::Unknown => ColumnType::Single(t),
                    ColumnType::Single(s) if s == t => *seen,
                    _ => ColumnType::Mixed,
                };
            }
        }
        Ok(())
    }
}
