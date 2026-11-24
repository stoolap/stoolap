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

/// A statement of a script, the line its first token is on, and the list
/// of values a `-- PARAMS: 1, 'a'` comment gives it
pub struct Statement {
    pub line: usize,
    pub sql: String,
    pub params: Option<String>,
    /// Line and column (from 1, in characters) of the first byte of `sql`, so a
    /// position inside the statement maps to the input
    pub text_at: (usize, usize),
    /// The name of the case a `-- @case <name>` line before the statement
    /// starts with it
    pub case: Option<String>,
}

impl Statement {
    /// The input's line and column of `(line, column)` inside `sql`
    pub fn input_position(&self, (line, column): (usize, usize)) -> (usize, usize) {
        match line {
            1 => (self.text_at.0, self.text_at.1 + column - 1),
            _ => (self.text_at.0 + line - 1, column),
        }
    }
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
    /// The PARAMS list a comment in the statement being read gave
    params: Option<String>,
    /// What the text is left open in at `resume`, and whether a line since
    /// closed it
    open: Option<Open>,
    closed: bool,
    /// A line with the quote of an open string before it, lexed alone
    probe: String,
    /// Line and column of the first byte of `pending`
    origin: (usize, usize),
    /// A case a `-- @case` line between statements started, for the next
    case: Option<String>,
}

/// A byte of `pending` with its line and column
type Cursor = (usize, (usize, usize));

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
            origin: (1, 1),
            case: None,
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

    /// Whether a statement, or a block comment, has begun and not ended:
    /// what is waiting is more than blanks and closed comments
    pub fn in_statement(&self) -> bool {
        // A construct a later line closed is lexed again below, from its start
        if self.first_line.is_some() || (self.open.is_some() && !self.closed) {
            return true;
        }
        let mut lexer = Lexer::new(&self.pending[self.resume..]);
        loop {
            let token = lexer.next_token();
            match token.token_type {
                TokenType::Eof => return false,
                TokenType::Comment if !open_block(&token.literal) => {}
                _ => return true,
            }
        }
    }

    /// Before a line that is not SQL, with no statement begun: lets the
    /// comments waiting go and gives the case a marker among them started
    pub fn take_case(&mut self) -> Option<String> {
        let mut lexer = Lexer::new(&self.pending[self.resume..]);
        loop {
            let token = lexer.next_token();
            match token.token_type {
                TokenType::Eof => break,
                TokenType::Comment => {
                    if let Some(name) = case_of(&token.literal) {
                        self.case = Some(name);
                    }
                }
                _ => {}
            }
        }
        self.origin = self.advance((0, self.origin), self.pending.len()).1;
        self.pending.clear();
        self.resume = 0;
        self.resume_line = self.origin.0;
        self.open = None;
        self.closed = false;
        self.params = None;
        self.case.take()
    }

    /// The cursor at byte `to` of `pending`, read on from `from`; columns
    /// count characters, as the lexer's positions do
    fn advance(&self, (from, (line, column)): Cursor, to: usize) -> Cursor {
        let text = &self.pending[from..to];
        match text.rfind('\n') {
            None => (to, (line, column + text.chars().count())),
            Some(newline) => (
                to,
                (
                    line + text.matches('\n').count(),
                    text[newline + 1..].chars().count() + 1,
                ),
            ),
        }
    }

    /// The statement whose text, trimmed, is `pending[start..end]`
    fn statement(
        &self,
        cursor: &mut Cursor,
        (start, end): (usize, usize),
        line: usize,
        (params, case): (Option<String>, Option<String>),
    ) -> Statement {
        let raw = &self.pending[start..end];
        let lead = raw.len() - raw.trim_start().len();
        *cursor = self.advance(*cursor, start + lead);
        Statement {
            line,
            sql: raw.trim().to_string(),
            params,
            text_at: cursor.1,
            case,
        }
    }

    fn cut(&mut self, at_end: bool, out: &mut Vec<Statement>) {
        let base = self.resume;
        let mut lexer = Lexer::new(&self.pending[base..]);
        let mut held: Option<Token> = None;
        // A statement cut at a semicolon, and that semicolon's line: a
        // PARAMS comment after it on that line is its own
        let mut cut: Option<(Statement, usize)> = None;
        let mut start = 0;
        let mut cursor: Cursor = (0, self.origin);
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
                            statement.params = Some(params);
                            out.push(statement);
                        }
                        None => {
                            if let Some(name) = case_of(&token.literal) {
                                self.case = Some(name);
                            }
                            cut = Some((statement, semicolon_line));
                        }
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
                    } else if self.first_line.is_none()
                        && let Some(name) = case_of(&token.literal)
                    {
                        self.case = Some(name);
                    }
                }
                TokenType::Punctuator if token.literal == ";" => {
                    let params = self.params.take();
                    if let Some(first) = self.first_line.take() {
                        let with = (params, self.case.take());
                        let statement = self.statement(&mut cursor, (start, at), first, with);
                        cut = Some((statement, line));
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
                let with = (self.params.take(), self.case.take());
                let end = self.pending.len();
                out.push(self.statement(&mut cursor, (start, end), first, with));
            }
            start = self.pending.len();
        }
        self.origin = self.advance(cursor, start).1;
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

/// The name a `-- @case <name>` comment gives the case it starts
fn case_of(comment: &str) -> Option<String> {
    let rest = comment
        .strip_prefix("--")?
        .trim_start()
        .strip_prefix("@case")?;
    let name = rest.trim();
    (rest.starts_with(char::is_whitespace) && !name.is_empty()).then(|| name.to_string())
}

/// The list of values a `-- PARAMS: 1, 'a'` comment gives its statement
fn params_of(comment: &str) -> Option<String> {
    let list = comment
        .strip_prefix("--")?
        .trim_start()
        .strip_prefix("PARAMS:")?;
    Some(list.trim().to_string())
}

/// The values a PARAMS list holds, read as the engine reads `SELECT <list>`:
/// SQL literals, text in single quotes
pub fn param_values(session: &mut Session, list: Option<&str>) -> Result<Vec<Value>, Failure> {
    let Some(list) = list.filter(|list| !list.is_empty()) else {
        return Ok(Vec::new());
    };
    let read = |session: &mut Session| -> stoolap::Result<Vec<Value>> {
        let mut rows = session.query(&format!("SELECT {list}"), Vec::new())?;
        let values = match rows.advance() {
            true => rows.current_row().as_slice().to_vec(),
            false => Vec::new(),
        };
        match rows.error() {
            Some(e) => Err(e),
            None => Ok(values),
        }
    };
    read(session).map_err(Failure::Params)
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

/// Whether `sql` orders the rows it returns: an ORDER BY outside every
/// parenthesis, where a window's, a subquery's or an aggregate's are not
pub fn orders_rows(sql: &str) -> bool {
    let mut lexer = Lexer::new(sql);
    let mut depth = 0i32;
    let mut after_order = false;
    loop {
        let token = lexer.next_token();
        match token.token_type {
            TokenType::Eof => return false,
            TokenType::Comment => continue,
            TokenType::Punctuator if token.literal == "(" => depth += 1,
            TokenType::Punctuator if token.literal == ")" => depth -= 1,
            _ if after_order && token.keyword == Some(Keyword::By) => return true,
            _ if depth == 0 && token.keyword == Some(Keyword::Order) => {
                after_order = true;
                continue;
            }
            _ => {}
        }
        after_order = false;
    }
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
    let command = match first.keyword {
        Some(Keyword::Begin) => Command::Begin,
        Some(Keyword::Explain) => Command::Explain,
        Some(Keyword::Commit) if alone => Command::Commit,
        Some(Keyword::Rollback) if alone => Command::Rollback,
        _ if bare && word(&["exit", "quit"]) || slash(&["\\q"]) => Command::Exit,
        _ if bare && word(&["help"]) || slash(&["\\h", "\\?"]) => Command::Help,
        _ => Command::Other,
    };
    // A command the CLI runs itself is not one while a block comment is
    // still open: the text goes on in later lines
    match command {
        Command::Explain | Command::Other => command,
        _ if ends_in_open_comment(sql) => Command::Other,
        _ => command,
    }
}

/// Whether `sql` ends inside a block comment
fn ends_in_open_comment(sql: &str) -> bool {
    let mut lexer = Lexer::new(sql);
    let mut last = None;
    loop {
        let token = lexer.next_token();
        if token.token_type == TokenType::Eof {
            break;
        }
        last = Some(token);
    }
    last.is_some_and(|t| t.token_type == TokenType::Comment && open_block(&t.literal))
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
    /// Runs of a statement that only reads, the first printed, 1 for once,
    /// and the unmeasured runs between the first and the measured ones
    pub repeat: usize,
    pub warmup: usize,
}

/// A time in microseconds to the nanosecond, `121.375`, as a JSON number
pub struct Micros(pub Duration);

impl std::fmt::Display for Micros {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let nanos = self.0.as_nanos();
        write!(f, "{}.{:03}", nanos / 1000, nanos % 1000)
    }
}

/// The runs of a statement after the one printed, each reading every row
/// and printing none: their engine times, and the error that stopped them
pub struct Repeats {
    pub first: Duration,
    /// Runs after the first that were run, not measured, and passed
    pub warmup: usize,
    /// The measured runs' engine times, sorted
    pub warm: Vec<Duration>,
    pub failed: Option<Failure>,
}

impl Repeats {
    /// The nearest-rank `q` quantile of the measured runs
    fn quantile(&self, q: f64) -> Option<Duration> {
        let rank = (q * self.warm.len() as f64).ceil() as usize;
        self.warm.get(rank.saturating_sub(1)).copied()
    }

    /// The measured runs' min, p50, p90 and max
    fn spread(&self) -> Option<[Duration; 4]> {
        Some([
            *self.warm.first()?,
            self.quantile(0.5)?,
            self.quantile(0.9)?,
            *self.warm.last()?,
        ])
    }

    /// `{"runs":..,"warmup":..,"first_us":..,"min_us":..,"p50_us":..,
    /// "p90_us":..,"max_us":..}`
    pub fn write_json(&self, w: &mut impl Write) -> io::Result<()> {
        write!(
            w,
            "{{\"runs\":{},\"warmup\":{},\"first_us\":{}",
            self.warm.len() + 1,
            self.warmup,
            Micros(self.first)
        )?;
        if let Some([min, p50, p90, max]) = self.spread() {
            write!(
                w,
                ",\"min_us\":{},\"p50_us\":{},\"p90_us\":{},\"max_us\":{}",
                Micros(min),
                Micros(p50),
                Micros(p90),
                Micros(max)
            )?;
        }
        if let Some(failure) = &self.failed {
            w.write_all(b",\"failed\":")?;
            serde_json::to_writer(&mut *w, &failure.to_string())?;
        }
        w.write_all(b"}")
    }

    /// `Ran 10 times after 2 warmup runs: first 1.2 ms, then min ..,
    /// p50 .., p90 .., max ..`
    fn line(&self) -> String {
        let mut line = format!("Ran {} times", self.warm.len() + 1);
        if self.warmup > 0 {
            line += &format!(
                " after {} warmup {}",
                self.warmup,
                plural(self.warmup as i64, "run")
            );
        }
        line += &format!(": first {}", duration(self.first));
        if let Some([min, p50, p90, max]) = self.spread() {
            line += &format!(
                ", then min {}, p50 {}, p90 {}, max {}",
                duration(min),
                duration(p50),
                duration(p90),
                duration(max)
            );
        }
        if let Some(failure) = &self.failed {
            line += &format!("; stopped: {failure}");
        }
        line + BUILD_NOTE
    }
}

/// Takes every row and keeps none, reading no clock
struct Drain;

impl Sink for Drain {
    fn begin(&mut self, _columns: &[String]) -> io::Result<()> {
        Ok(())
    }

    fn row(&mut self, _row: &Row) -> io::Result<()> {
        Ok(())
    }

    fn timed(&self) -> bool {
        false
    }
}

/// After a first run whose engine time was `first`, runs `sql` `warmup`
/// times unmeasured and `times` times measured, each to its last row;
/// stops at a run that fails
pub fn repeat(
    session: &mut Session,
    sql: &str,
    params: &[Value],
    first: Duration,
    (warmup, times): (usize, usize),
) -> Repeats {
    let mut warm = Vec::with_capacity(times);
    let mut failed = None;
    let mut warmed = 0;
    for run_number in 0..warmup + times {
        match run(session, sql, params.to_vec(), 0, &mut Drain) {
            Ok((_, took)) if run_number >= warmup => warm.push(took.engine),
            Ok(_) => warmed += 1,
            Err(failure) => {
                failed = Some(failure);
                break;
            }
        }
    }
    warm.sort();
    Repeats {
        first,
        warmup: warmed,
        warm,
        failed,
    }
}

/// Whether running `sql` again leaves the data as it was: it starts with
/// SELECT, WITH, VALUES, SHOW or DESCRIBE and has no INSERT, UPDATE or
/// DELETE outside parentheses. EXPLAIN is not, as EXPLAIN ANALYZE runs
/// what it explains
pub fn reads_only(sql: &str) -> bool {
    let mut lexer = Lexer::new(sql);
    let mut depth = 0i32;
    let mut first = true;
    loop {
        let token = lexer.next_token();
        match token.token_type {
            TokenType::Eof => return !first,
            TokenType::Comment => continue,
            _ if first => {
                first = false;
                let reads = matches!(
                    token.keyword,
                    Some(
                        Keyword::Select
                            | Keyword::With
                            | Keyword::Values
                            | Keyword::Show
                            | Keyword::Describe
                            | Keyword::Desc
                    )
                );
                if !reads {
                    return false;
                }
            }
            TokenType::Punctuator if token.literal == "(" => depth += 1,
            TokenType::Punctuator if token.literal == ")" => depth -= 1,
            _ if depth == 0
                && matches!(
                    token.keyword,
                    Some(Keyword::Insert | Keyword::Update | Keyword::Delete)
                ) =>
            {
                return false;
            }
            _ => {}
        }
    }
}

/// What a statement runs against
pub enum Session<'a> {
    Database(&'a Database),
    Transaction(&'a mut ApiTransaction),
}

impl Session<'_> {
    /// Runs `sql` under the connection's `query_timeout`, which the CLI's
    /// `--timeout` sets once for every statement
    fn query(&mut self, sql: &str, params: Vec<Value>) -> stoolap::Result<Rows> {
        match self {
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

/// Why a statement gave no result: the engine's error, its PARAMS list's,
/// or the output's
#[derive(Debug)]
pub enum Failure {
    Engine(stoolap::Error),
    Params(stoolap::Error),
    Output(io::Error),
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Failure::Engine(e) => write!(f, "{e}"),
            Failure::Params(e) => write!(f, "PARAMS, read as SQL values: {e}"),
            Failure::Output(e) => write!(f, "writing output: {e}"),
        }
    }
}

impl From<stoolap::Error> for Failure {
    fn from(e: stoolap::Error) -> Self {
        Failure::Engine(e)
    }
}

impl From<io::Error> for Failure {
    fn from(e: io::Error) -> Self {
        Failure::Output(e)
    }
}

/// Takes the rows a statement shows
pub trait Sink {
    fn begin(&mut self, columns: &[String]) -> io::Result<()>;
    fn row(&mut self, row: &Row) -> io::Result<()>;

    /// Whether the sink reads the rows past the limit too, through `rest`
    fn wants_rest(&self) -> bool {
        false
    }

    /// A row past the limit, for a sink that reads every row
    fn rest(&mut self, _row: &Row) -> io::Result<()> {
        Ok(())
    }

    /// Whether the time the sink takes is measured apart from the engine's,
    /// at two clock readings a row it takes
    fn timed(&self) -> bool {
        true
    }
}

/// The single place a statement is run, for every mode of the CLI: hands
/// the first `limit` rows (0 for all) to `sink` and counts the rest, timing
/// only the sink, so a cut result costs its execution and nothing per row
pub fn run(
    session: &mut Session,
    sql: &str,
    params: Vec<Value>,
    limit: usize,
    sink: &mut impl Sink,
) -> Result<(Ran, Timing), Failure> {
    // A plan is shown whole whatever the limit
    let limit = match command(sql) {
        Command::Explain => 0,
        _ => limit,
    };
    let start = Instant::now();
    let mut rows = session.query(sql, params)?;
    if rows.columns().is_empty() {
        let timing = Timing {
            engine: start.elapsed(),
            output: Duration::ZERO,
        };
        return Ok((Ran::Affected(rows.rows_affected()), timing));
    }
    let shown = Instant::now();
    let columns = rows.columns().to_vec();
    sink.begin(&columns)?;
    let mut output = shown.elapsed();
    let mut total = 0u64;
    let mut timed = 0u64;
    let mut shown = 0u64;
    let rest = sink.wants_rest();
    let clocked = sink.timed();
    while rows.advance() {
        let row = rows.current_row();
        let show = limit == 0 || total < limit as u64;
        total += 1;
        if !show && !rest {
            continue;
        }
        let start = clocked.then(Instant::now);
        match show {
            true => {
                sink.row(row)?;
                shown += 1;
            }
            false => sink.rest(row)?,
        }
        if let Some(start) = start {
            output += start.elapsed();
            timed += 1;
        }
    }
    let elapsed = start.elapsed();
    // A timed row leaves about one clock reading outside its span
    let readings = clock_reading().mul_f64(timed as f64);
    let engine = elapsed.saturating_sub(output).saturating_sub(readings);
    output += readings;
    if let Some(err) = rows.error() {
        return Err(err.into());
    }
    let ran = Ran::Rows {
        columns,
        total,
        shown,
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
) -> io::Result<usize> {
    let mut splitter = Splitter::default();
    let mut ready = Vec::new();
    let mut failed = 0;
    let mut number = 0;
    for line in lines {
        let line = line.map_err(|e| io::Error::new(e.kind(), format!("reading input: {e}")))?;
        splitter.push_line(&line, &mut ready);
        if !run_ready(db, &mut ready, &mut number, &mut failed, out)? {
            return Ok(failed);
        }
    }
    splitter.finish(&mut ready);
    run_ready(db, &mut ready, &mut number, &mut failed, out)?;
    Ok(failed)
}

/// Runs the statements cut so far; false once a statement asked to exit
fn run_ready<W: Write>(
    db: &Database,
    ready: &mut Vec<Statement>,
    number: &mut usize,
    failed: &mut usize,
    out: &mut Output<W>,
) -> io::Result<bool> {
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
            statement.params.as_deref(),
            out,
        );
        match printed {
            // Output that cannot be written ends the script
            Err(Failure::Output(e)) => return Err(e),
            Err(message) => {
                // The rows printed before the error come first
                out.out.flush()?;
                *failed += 1;
                eprintln!(
                    "Error (statement {}, line {}): {}",
                    number, statement.line, message
                );
            }
            Ok(_) => {}
        }
    }
    Ok(true)
}

/// Runs one statement and prints its result in the output's format
pub fn print_statement<W: Write>(
    session: &mut Session,
    sql: &str,
    params: Option<&str>,
    out: &mut Output<W>,
) -> Result<Timing, Failure> {
    let params = param_values(session, params)?;
    let Output {
        out: w,
        format,
        limit,
        quiet,
        timing,
        repeat,
        warmup,
    } = out;
    let again = (*repeat > 1 && reads_only(sql)).then(|| params.clone());
    let ran = match format {
        Format::Table => run(session, sql, params, *limit, &mut TableSink(w)),
        Format::Json => {
            let mut sink = JsonSink {
                w,
                begun: false,
                rows: 0,
            };
            let ran = run(session, sql, params, *limit, &mut sink);
            // Rows already written stay readable: the object is closed
            if ran.is_err() && sink.begun {
                let _ = sink.w.write_all(b"]}\n");
            }
            ran
        }
    };
    let (ran, took) = ran?;
    let repeats = match (&ran, again) {
        (Ran::Rows { .. }, Some(params)) => Some(self::repeat(
            session,
            sql,
            &params,
            took.engine,
            (*warmup, *repeat - 1),
        )),
        _ => None,
    };
    let footer = match (ran, *format) {
        (Ran::Affected(n), Format::Json) => writeln!(w, "{{\"rows_affected\":{n}}}"),
        (Ran::Affected(n), Format::Table) => writeln!(w, "{n} {} affected", plural(n, "row")),
        (Ran::Rows { total, shown, .. }, Format::Json) => {
            let counts = match shown < total {
                true => write!(w, "],\"count\":{total},\"shown\":{shown}"),
                false => write!(w, "],\"count\":{total}"),
            };
            counts
                .and_then(|()| match &repeats {
                    Some(repeats) => {
                        w.write_all(b",\"repeat\":")?;
                        repeats.write_json(w)
                    }
                    None => Ok(()),
                })
                .and_then(|()| writeln!(w, "}}"))
        }
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
    footer?;
    if *timing {
        writeln!(w, "{}", timing_line(took))?;
    }
    if let (Format::Table, Some(repeats)) = (*format, &repeats) {
        writeln!(w, "{}", repeats.line())?;
    }
    w.flush()?;
    // A run after the first that failed fails the statement, after its
    // result and the runs before it are printed
    match repeats.and_then(|r| r.failed) {
        Some(failure) => Err(failure),
        None => Ok(took),
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
pub fn write_json_value(w: &mut impl Write, value: &Value) -> io::Result<()> {
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
            repeat: 1,
            warmup: 0,
        };
        let before = ALLOCATIONS.with(Cell::get);
        print_statement(&mut Session::Database(db), sql, None, &mut out).unwrap();
        ALLOCATIONS.with(Cell::get) - before
    }

    /// Allocations of agent mode running `sql` with `limit`, the rows
    /// written or not
    fn agent_allocations(db: &Database, sql: &str, limit: usize, no_rows: bool) -> usize {
        let mut out = Vec::with_capacity(1 << 22);
        let options = crate::agent::Options {
            limit,
            full: false,
            no_rows,
            stop_on_error: false,
            timing: false,
            repeat: 1,
            warmup: 0,
        };
        let lines = std::iter::once(Ok(sql.to_string()));
        let before = ALLOCATIONS.with(Cell::get);
        let no_cases = |_: usize| Err("no cases".to_string());
        crate::agent::run_script(db, "memory://", &no_cases, lines, &mut out, &options).unwrap();
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
        // Agent mode keeps the shown cells in buffers that grow by doubling
        let one = agent_allocations(&db, sql, 1, false);
        let all = agent_allocations(&db, sql, 0, false);
        assert!(
            all <= one + 48,
            "agent mode printing 1,999 more rows allocated {} times more",
            all - one
        );
        let none = agent_allocations(&db, sql, 0, true);
        assert!(
            none <= one,
            "--no-rows over 2,000 rows allocated {none}, one shown row {one}"
        );
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

    fn params(text: &str) -> Vec<(String, Option<String>)> {
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
        let list = |list: &str| Some(list.to_string());
        assert_eq!(
            params(
                "SELECT $1; -- PARAMS: 7\nSELECT $1 -- PARAMS: 'b', 2\n;\nSELECT 'x -- PARAMS: y'"
            ),
            vec![
                ("SELECT $1".to_string(), list("7")),
                (
                    "-- PARAMS: 7\nSELECT $1 -- PARAMS: 'b', 2".to_string(),
                    list("'b', 2")
                ),
                ("SELECT 'x -- PARAMS: y'".to_string(), None),
            ]
        );
        assert_eq!(
            params("SELECT $1; /* note */ # more\n"),
            vec![("SELECT $1".to_string(), None)]
        );
        assert_eq!(
            params("SELECT $1; /* note */ -- PARAMS: 7"),
            vec![("SELECT $1".to_string(), list("7"))]
        );
    }

    #[test]
    fn a_params_list_holds_sql_values() {
        let db = Database::open("memory://cli_params_list_holds_sql_values").unwrap();
        let values = |list: &str| param_values(&mut Session::Database(&db), Some(list));
        assert_eq!(
            values("3, 'x, y', 'it''s', -5, 1.5, TRUE, NULL").unwrap(),
            vec![
                Value::Integer(3),
                Value::text("x, y"),
                Value::text("it's"),
                Value::Integer(-5),
                Value::Float(1.5),
                Value::Boolean(true),
                Value::null_unknown(),
            ]
        );
        assert_eq!(values("").unwrap(), Vec::<Value>::new(), "an empty list");
        for wrong in ["alice", "[3, 'x']", "'a',"] {
            assert!(
                matches!(values(wrong), Err(Failure::Params(_))),
                "{wrong} is not a list of values"
            );
        }
    }

    #[test]
    fn a_carriage_return_inside_a_string_is_kept() {
        assert_eq!(
            params("SELECT 'a\r\nb';\r\nSELECT 2\r"),
            vec![
                ("SELECT 'a\r\nb'".to_string(), None),
                ("SELECT 2".to_string(), None)
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
    fn the_spread_takes_nearest_rank_quantiles() {
        let ms = |n: u64| Duration::from_millis(n);
        let repeats = Repeats {
            first: ms(50),
            warmup: 0,
            warm: (1..=10).map(ms).collect(),
            failed: None,
        };
        assert_eq!(repeats.spread(), Some([ms(1), ms(5), ms(9), ms(10)]));
        let one = Repeats {
            warm: vec![ms(3)],
            ..repeats
        };
        assert_eq!(one.spread(), Some([ms(3); 4]));
        let none = Repeats {
            warm: Vec::new(),
            ..one
        };
        assert_eq!(none.spread(), None);
    }

    #[test]
    fn only_a_statement_that_reads_runs_again() {
        let cases = [
            ("SELECT a FROM t", true),
            ("/* c */ WITH x AS (SELECT 1) SELECT * FROM x", true),
            ("VALUES (1)", true),
            ("SHOW TABLES", true),
            ("DESCRIBE t", true),
            ("SELECT a FROM t WHERE a IN (SELECT b FROM u)", true),
            ("WITH x AS (SELECT 1) INSERT INTO t SELECT * FROM x", false),
            ("INSERT INTO t VALUES (1) RETURNING *", false),
            ("UPDATE t SET a = 1 RETURNING a", false),
            ("EXPLAIN ANALYZE SELECT a FROM t", false),
            ("(SELECT a FROM t)", false),
            ("PRAGMA query_timeout", false),
            ("", false),
        ];
        for (sql, reads) in cases {
            assert_eq!(reads_only(sql), reads, "{sql}");
        }
    }

    #[test]
    fn only_an_order_by_outside_parentheses_orders_the_rows() {
        let cases = [
            ("SELECT a FROM t ORDER BY a", true),
            ("SELECT a FROM t ORDER /* c */ BY a", true),
            ("SELECT a FROM t UNION SELECT b FROM u ORDER BY 1", true),
            ("SELECT a FROM (SELECT a FROM t ORDER BY a) s", false),
            ("SELECT ROW_NUMBER() OVER (ORDER BY a) FROM t", false),
            ("SELECT STRING_AGG(a, ',' ORDER BY a) FROM t", false),
            ("SELECT 'ORDER BY' FROM t", false),
            ("SELECT \"order\", \"by\" FROM t", false),
        ];
        for (sql, ordered) in cases {
            assert_eq!(orders_rows(sql), ordered, "{sql}");
        }
    }

    #[test]
    fn a_case_line_between_statements_starts_a_case() {
        let mut splitter = Splitter::default();
        let mut out = Vec::new();
        let text = "SELECT 1;\n-- @case first one\nSELECT 2;\nSELECT 3; -- @case second\n\
                    SELECT 4 -- @case not here\n;\n--@casex\nSELECT 5;";
        for line in text.split('\n') {
            splitter.push_line(line, &mut out);
        }
        splitter.finish(&mut out);
        let cases: Vec<_> = out.iter().map(|s| s.case.as_deref()).collect();
        assert_eq!(
            cases,
            vec![None, Some("first one"), None, Some("second"), None]
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
            ("COMMIT /* never closed", Command::Other),
            ("ROLLBACK /* never closed", Command::Other),
            ("BEGIN TRANSACTION /* never closed", Command::Other),
            ("\\q /* never closed", Command::Other),
            ("help /* never closed", Command::Other),
            ("COMMIT /* closed */", Command::Commit),
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
