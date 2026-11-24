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

//! Stoolap CLI - Interactive SQL database command-line interface
//!

#[path = "stoolap/agent.rs"]
mod agent;
#[path = "stoolap/script.rs"]
mod script;

use std::fs::File;
use std::io::{self, BufRead, BufReader, IsTerminal};
use std::time::{Duration, Instant};

use clap::Parser;
use comfy_table::{presets::UTF8_FULL_CONDENSED, Cell, ContentArrangement, Table};
use rustyline::error::ReadlineError;
use rustyline::history::DefaultHistory;
use rustyline::{Config, DefaultEditor, EditMode, Editor};

use stoolap::api::{Database, Transaction as ApiTransaction};
use stoolap::common::version::{MAJOR, MINOR, PATCH};
use stoolap::{Row, Value};

use script::{Command, Format, Output, Ran, Session, Sink, Splitter, Timing};

/// Version string constant
const VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION_MAJOR"),
    ".",
    env!("CARGO_PKG_VERSION_MINOR"),
    ".",
    env!("CARGO_PKG_VERSION_PATCH")
);

/// Stoolap SQL Database CLI
#[derive(Parser, Debug)]
#[command(name = "stoolap")]
#[command(author = "Stoolap Contributors")]
#[command(version = VERSION)]
#[command(about = "High-performance embedded SQL database with MVCC")]
#[command(
    long_about = "Stoolap is a high-performance embedded SQL database with MVCC transactions.\n\
This CLI provides an interactive interface to execute SQL queries and manage your database.\n\
A program or an AI agent: run `stoolap --agent --help` for agent mode, JSON lines out.\n\n\
PERSISTENCE DSN PARAMETERS:\n\
  file:///path/to/db?param=value&param2=value2\n\n\
  sync_mode=none|normal|full   Fsync mode (default: normal)\n\
    none: no fsync, durable at checkpoint only\n\
    normal: fsync every 1 second, DDL fsyncs immediately\n\
    full: fsync on every write\n\
  checkpoint_interval=SECS    Checkpoint interval in seconds (default: 60)\n\
  compact_threshold=COUNT     Sub-target volumes per table before merging (default: 4)\n\
  compression=on|off          WAL + volume LZ4 compression (default: on)\n\
  wal_compression=on|off      WAL compression only (default: on)\n\
  volume_compression=on|off   Volume LZ4 compression only (default: on)\n\
  keep_snapshots=COUNT        Backup snapshots per table (default: 3)\n\
  checkpoint_on_close=on|off  Checkpoint on shutdown (default: on)\n\
  wal_max_size=BYTES          Max WAL file size before rotation (default: 67108864)\n\
  wal_buffer_size=BYTES       WAL buffer size (default: 65536)\n\
  wal_flush_trigger=BYTES     Buffer size to trigger flush (default: 32768)\n\
  commit_batch_size=COUNT     Commits to batch before sync (default: 100)\n\
  sync_interval_ms=MS         Min time between syncs in normal mode (default: 1000)\n\
  compression_threshold=BYTES Min size to compress (default: 64)\n\
  target_volume_rows=ROWS     Target rows per cold volume (default: 1048576)\n\n\
EXAMPLES:\n\
  stoolap -d memory://                                    In-memory database\n\
  stoolap -d file:///tmp/mydb                             Persistent database\n\
  stoolap -d file:///tmp/mydb?sync_mode=full               Maximum durability\n\
  stoolap -d file:///tmp/mydb?sync_mode=none&compression=off  Max performance\n\
  stoolap -d file:///tmp/mydb --profile durable           Use durable preset\n\
  stoolap -d file:///tmp/mydb --sync full --compression off\n\n\
BACKUP & RESTORE:\n\
  stoolap -d file:///tmp/mydb --snapshot                  Create backup snapshot\n\
  stoolap -d file:///tmp/mydb --restore                   Restore from latest snapshot\n\
  stoolap -d file:///tmp/mydb --restore 20260315-100000   Restore specific snapshot\n\
  stoolap -d file:///tmp/mydb --reset-volumes --restore   Recovery from corrupted volumes"
)]
struct Args {
    /// Database path (file://<path> or memory://)
    #[arg(short = 'd', long = "db", default_value = "memory://")]
    db_path: String,

    /// Output results in JSON format
    #[arg(short = 'j', long = "json", default_value = "false")]
    json_output: bool,

    /// Suppress connection messages
    #[arg(short = 'q', long = "quiet", default_value = "false")]
    quiet: bool,

    /// Maximum number of rows to display (0 for unlimited)
    #[arg(short = 'l', long = "limit", default_value = "40")]
    limit: usize,

    /// Execute a single SQL statement and exit
    #[arg(short = 'e', long = "execute")]
    execute: Option<String>,

    /// Execute SQL statements from a file
    #[arg(short = 'f', long = "file")]
    file: Option<String>,

    /// WAL sync mode for durability (none, normal, full)
    /// - none: No fsync, data durable at checkpoint only
    /// - normal: Fsync every 1 second, DDL fsyncs immediately (default)
    /// - full: Fsync on every write
    #[arg(short = 's', long = "sync", value_name = "MODE")]
    sync_mode: Option<String>,

    /// Persistence profile preset (fast, normal, durable)
    /// - fast: Optimized for performance, less durable
    /// - normal: Balanced performance and durability (default)
    /// - durable: Maximum durability, slower performance
    #[arg(short = 'p', long = "profile", value_name = "PROFILE")]
    persistence_profile: Option<String>,

    /// Checkpoint interval in seconds (default: 60)
    #[arg(long = "checkpoint-interval", value_name = "SECONDS")]
    checkpoint_interval: Option<u32>,

    /// Sub-target volumes per table before merging (default: 4)
    #[arg(long = "compact-threshold", value_name = "COUNT")]
    compact_threshold: Option<u32>,

    /// Maximum WAL file size in MB before rotation (default: 64)
    #[arg(long = "wal-max-size", value_name = "MB")]
    wal_max_size: Option<u32>,

    /// Enable or disable LZ4 compression for WAL and volumes (default: on)
    #[arg(long = "compression", value_name = "on|off")]
    compression: Option<String>,

    /// Number of backup snapshots to keep per table (default: 3)
    #[arg(long = "keep-snapshots", value_name = "COUNT")]
    keep_snapshots: Option<u32>,

    /// Disable checkpoint on close (for crash simulation in tests)
    #[arg(long = "no-checkpoint-on-close")]
    no_checkpoint_on_close: bool,

    /// Restore database from backup snapshot and exit.
    /// Deletes corrupted volumes/ if needed, loads snapshot data, recreates indexes.
    /// Optionally specify a timestamp: --restore "20260315-100000.000"
    #[arg(long = "restore", value_name = "TIMESTAMP", num_args = 0..=1, default_missing_value = "")]
    restore: Option<String>,

    /// Create a backup snapshot and exit.
    #[arg(long = "snapshot")]
    snapshot: bool,

    /// Force remove volumes/ directory before opening (for recovery from corrupted volumes).
    /// Use with --restore to force snapshot-based recovery.
    #[arg(long = "reset-volumes")]
    reset_volumes: bool,

    /// Statement timeout in milliseconds, set as the connection's
    /// query_timeout so it bounds every statement, inside a transaction too
    /// (0 for none; default: none, or 30000 in agent mode)
    #[arg(short = 't', long = "timeout", value_name = "MS")]
    timeout_ms: Option<u64>,

    /// Agent mode: a build line, one JSON line per statement and a summary,
    /// for a program to read (also STOOLAP_AGENT=1)
    #[arg(long = "agent", conflicts_with = "no_agent")]
    agent: bool,

    /// Turn agent mode off, whatever STOOLAP_AGENT says
    #[arg(long = "no-agent")]
    no_agent: bool,

    /// Agent mode: what a failed statement does to the rest of the script
    #[arg(
        long = "on-error",
        value_name = "continue|stop",
        default_value = "continue"
    )]
    on_error: OnError,

    /// Agent mode: keep cells longer than 1,000 characters whole
    #[arg(long = "full")]
    full: bool,

    /// Agent mode: add each statement's engine time as elapsed_us
    #[arg(long = "timing")]
    timing: bool,

    /// Run each statement that only reads N times, -e, -f, piped input and
    /// agent mode: the first prints, and the spread of the engine times of
    /// the others is given after it
    #[arg(long = "repeat", value_name = "N", default_value = "1", value_parser = clap::value_parser!(u64).range(1..))]
    repeat: u64,

    /// With --repeat: runs after the first, before the measured ones, that
    /// are run and not measured
    #[arg(long = "warmup", value_name = "M", default_value = "0")]
    warmup: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum, Debug)]
enum OnError {
    Continue,
    Stop,
}

/// CLI state for interactive mode
struct Cli {
    db: Database,
    tx: Option<ApiTransaction>,
    in_transaction: bool,
    json_output: bool,
    limit: usize,
    #[allow(dead_code)]
    quiet: bool,
    editor: Editor<(), DefaultHistory>,
    /// The lines of the statement being typed, for the history
    current_query: String,
    splitter: Splitter,
}

/// The first rows of a result, kept to lay out the interactive table
struct Collect(Vec<Vec<Value>>);

impl Sink for Collect {
    fn begin(&mut self, _columns: &[String]) -> io::Result<()> {
        Ok(())
    }

    fn row(&mut self, row: &Row) -> io::Result<()> {
        self.0.push(row.as_slice().to_vec());
        Ok(())
    }
}

impl Cli {
    fn new(db: Database, json_output: bool, limit: usize, quiet: bool) -> io::Result<Self> {
        let config = Config::builder()
            .history_ignore_space(true)
            .edit_mode(EditMode::Emacs)
            .build();

        let mut editor =
            DefaultEditor::with_config(config).map_err(|e| io::Error::other(e.to_string()))?;

        // Load history from file
        if let Some(home) = dirs::home_dir() {
            let history_file = home.join(".stoolap_history");
            let _ = editor.load_history(&history_file);
        }

        Ok(Self {
            db,
            tx: None,
            in_transaction: false,
            json_output,
            limit,
            quiet,
            editor,
            current_query: String::new(),
            splitter: Splitter::default(),
        })
    }

    fn get_prompt(&self) -> &'static str {
        if self.splitter.is_pending() {
            if self.in_transaction {
                "\x1b[1;33m[TXN]->\x1b[0m "
            } else {
                "\x1b[1;36m->\x1b[0m "
            }
        } else if self.in_transaction {
            "\x1b[1;33m[TXN]>\x1b[0m "
        } else {
            "\x1b[1;36m>\x1b[0m "
        }
    }

    fn run(&mut self) -> io::Result<()> {
        println!("Stoolap v{}.{}.{}", MAJOR, MINOR, PATCH);
        println!("Enter SQL commands, 'help' for assistance, or 'exit' to quit.");
        println!("Use Up/Down arrows for history, Ctrl+R to search history.");
        if self.json_output {
            println!("JSON output mode enabled.");
        }
        println!();

        loop {
            let prompt = self.get_prompt();
            match self.editor.readline(prompt) {
                Ok(line) => {
                    let line = line.trim();

                    // Handle empty line
                    if !self.splitter.is_pending() && line.is_empty() {
                        continue;
                    }

                    // Special commands and transaction control run without
                    // a semicolon (only when not in multi-line mode)
                    if !self.splitter.is_pending() {
                        match script::command(line) {
                            Command::Exit => {
                                if self.in_transaction {
                                    eprintln!("\x1b[1;33mWarning: Exiting with active transaction. Rolling back...\x1b[0m");
                                    let _ = self.rollback_transaction();
                                }
                                break;
                            }
                            Command::Help => {
                                self.print_help();
                                continue;
                            }
                            Command::Begin | Command::Commit | Command::Rollback
                                if !line.contains(';') =>
                            {
                                let _ = self.editor.add_history_entry(line);
                                self.run_statement(line, Vec::new());
                                continue;
                            }
                            _ => {}
                        }
                    }

                    // Add line to current query
                    if !self.current_query.is_empty() {
                        self.current_query.push('\n');
                    }
                    self.current_query.push_str(line);

                    let mut statements = Vec::new();
                    self.splitter.push_line(line, &mut statements);
                    if !self.splitter.is_pending() {
                        let history_entry = self.current_query.replace('\n', "\\n");
                        let _ = self.editor.add_history_entry(&history_entry);
                        self.current_query.clear();
                    }
                    for statement in statements {
                        self.run_statement(&statement.sql, statement.params);
                    }
                }
                Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                    if self.in_transaction {
                        eprintln!("\n\x1b[1;33mWarning: Exiting with active transaction. Rolling back...\x1b[0m");
                        let _ = self.rollback_transaction();
                    }
                    break;
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    break;
                }
            }
        }

        // Save history
        if let Some(home) = dirs::home_dir() {
            let history_file = home.join(".stoolap_history");
            let _ = self.editor.save_history(&history_file);
        }

        Ok(())
    }

    /// Runs one statement and prints its result and its time
    fn run_statement(&mut self, sql: &str, params: Vec<Value>) {
        match self.execute_query(sql, params) {
            Ok(took) => println!("\x1b[1;32m{}\x1b[0m", script::timing_line(took)),
            Err(e) => eprintln!("\x1b[1;31mError:\x1b[0m {}", e),
        }
    }

    fn execute_query(&mut self, query: &str, params: Vec<Value>) -> Result<Timing, String> {
        let start = Instant::now();
        let control = match script::command(query) {
            Command::Help => {
                self.print_help();
                return Ok(Timing::default());
            }
            Command::Begin => Some(self.begin_transaction()),
            Command::Commit => Some(self.commit_transaction()),
            Command::Rollback => Some(self.rollback_transaction()),
            _ => None,
        };
        if let Some(done) = control {
            return done.map(|()| Timing {
                engine: start.elapsed(),
                output: Duration::ZERO,
            });
        }

        let mut session = match self.tx.as_mut() {
            Some(tx) if self.in_transaction => Session::Transaction(tx),
            _ => Session::Database(&self.db),
        };
        if self.json_output {
            let mut out = Output {
                out: io::BufWriter::new(io::stdout().lock()),
                format: Format::Json,
                limit: self.limit,
                quiet: true,
                timing: false,
                repeat: 1,
                warmup: 0,
            };
            return script::print_statement(&mut session, query, params, &mut out)
                .map_err(|e| e.to_string());
        }
        let mut shown = Collect(Vec::new());
        let (ran, mut took) = script::run(&mut session, query, params, self.limit, &mut shown)
            .map_err(|e| e.to_string())?;
        let render = Instant::now();
        match ran {
            Ran::Affected(n) => println!(
                "\x1b[1;32m{} {} affected\x1b[0m",
                n,
                script::plural(n, "row")
            ),
            Ran::Rows { columns, total, .. } => self.output_table(&columns, &shown.0, total),
        }
        took.output += render.elapsed();
        Ok(took)
    }

    fn begin_transaction(&mut self) -> Result<(), String> {
        if self.in_transaction {
            return Err("already in a transaction".to_string());
        }

        let tx = self.db.begin().map_err(|e| e.to_string())?;
        self.tx = Some(tx);
        self.in_transaction = true;

        println!("\x1b[1;32mTransaction started\x1b[0m");
        Ok(())
    }

    fn commit_transaction(&mut self) -> Result<(), String> {
        if !self.in_transaction {
            return Err("not in a transaction".to_string());
        }

        if let Some(mut tx) = self.tx.take() {
            tx.commit().map_err(|e| e.to_string())?;
        }

        self.in_transaction = false;
        println!("\x1b[1;32mTransaction committed\x1b[0m");
        Ok(())
    }

    fn rollback_transaction(&mut self) -> Result<(), String> {
        if !self.in_transaction {
            return Err("not in a transaction".to_string());
        }

        if let Some(mut tx) = self.tx.take() {
            tx.rollback().map_err(|e| e.to_string())?;
        }

        self.in_transaction = false;
        println!("\x1b[1;33mTransaction rolled back\x1b[0m");
        Ok(())
    }

    fn output_explain_plan(&self, rows: &[Vec<Value>], row_count: u64) {
        println!();
        for row in rows {
            if let Some(Value::Text(line)) = row.first() {
                println!("{}", colorize_plan_line(line));
            }
        }
        println!();
        let line_text = if row_count == 1 { "line" } else { "lines" };
        println!("\x1b[2m({} {} in plan)\x1b[0m", row_count, line_text);
    }

    /// Lays out the first rows of a result of `total` rows
    fn output_table(&self, columns: &[String], rows: &[Vec<Value>], total: u64) {
        // Render EXPLAIN output without table borders, with color highlighting.
        // Verify first row starts with a known statement keyword to avoid false positives
        // from queries that alias a column as "plan".
        if columns.len() == 1 && columns[0] == "plan" && is_explain_output(rows) {
            return self.output_explain_plan(rows, total);
        }

        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL_CONDENSED)
            .set_content_arrangement(ContentArrangement::Dynamic);
        table.set_header(columns.iter().map(Cell::new));
        for row in rows {
            table.add_row(row.iter().map(|v| Cell::new(script::value_text(v))));
        }
        println!("{table}");

        let row_text = script::plural(total as i64, "row");
        let shown = rows.len() as u64;
        if shown < total {
            println!(
                "\x1b[1;32m{} {} in set (showing the first {}, {} more)\x1b[0m",
                total,
                row_text,
                shown,
                total - shown
            );
        } else {
            println!("\x1b[1;32m{} {} in set\x1b[0m", total, row_text);
        }
    }

    fn print_help(&self) {
        println!("\x1b[1mStoolap SQL CLI Commands:\x1b[0m");
        println!();
        println!("  \x1b[1;33mSQL Commands:\x1b[0m");
        println!("    SELECT ...             Execute a SELECT query");
        println!("    INSERT ...             Insert data into a table");
        println!("    UPDATE ...             Update data in a table");
        println!("    DELETE ...             Delete data from a table");
        println!("    CREATE TABLE ...       Create a new table");
        println!("    CREATE INDEX ...       Create an index on a column");
        println!("    ALTER TABLE ...        Modify table schema");
        println!("    SHOW TABLES            List all tables");
        println!("    SHOW CREATE TABLE ...  Show CREATE TABLE statement for a table");
        println!("    SHOW INDEXES FROM ...  Show indexes for a table");
        println!();
        println!("  \x1b[1;33mTransaction Commands:\x1b[0m");
        println!("    BEGIN                  Start a new transaction");
        println!("    COMMIT                 Commit the current transaction");
        println!("    ROLLBACK               Rollback the current transaction");
        println!();
        println!("  \x1b[1;33mPRAGMA Commands:\x1b[0m");
        println!("    PRAGMA CHECKPOINT      Seal hot data to volumes, compact, truncate WAL");
        println!("    PRAGMA SNAPSHOT        Create backup snapshot (.bin files)");
        println!("    PRAGMA RESTORE         Restore from latest backup snapshot");
        println!(
            "    PRAGMA key=value       Set configuration (sync_mode, checkpoint_interval, ...)"
        );
        println!("    PRAGMA key             Show current configuration value");
        println!();
        println!("  \x1b[1;33mSpecial Commands:\x1b[0m");
        println!("    exit, quit, \\q         Exit the CLI");
        println!("    help, \\h, \\?          Show this help message");
        println!();
        println!("  \x1b[1;33mKeyboard Shortcuts:\x1b[0m");
        println!("    Up/Down arrow keys     Navigate command history");
        println!("    Ctrl+R                 Search command history");
        println!("    Ctrl+A                 Move cursor to beginning of line");
        println!("    Ctrl+E                 Move cursor to end of line");
        println!("    Ctrl+W                 Delete word before cursor");
        println!("    Ctrl+U                 Delete from cursor to beginning of line");
        println!("    Ctrl+K                 Delete from cursor to end of line");
        println!("    Ctrl+L                 Clear screen");
        println!();
    }
}

/// Build DSN with query parameters from CLI args
fn build_dsn(args: &Args) -> String {
    let mut dsn = args.db_path.clone();

    // Only add params for file:// databases
    if !dsn.starts_with("file://") {
        return dsn;
    }

    let mut params = Vec::new();

    // Handle persistence profile presets
    if let Some(ref profile) = args.persistence_profile {
        match profile.to_lowercase().as_str() {
            "fast" => {
                params.push("sync=none".to_string());
            }
            "durable" => {
                params.push("sync=full".to_string());
            }
            "normal" => {
                // Default, no params needed
            }
            _ => {
                eprintln!(
                    "Warning: Unknown profile '{}', using 'normal'. Valid: fast, normal, durable",
                    profile
                );
            }
        }
    }

    // Individual sync mode overrides profile
    if let Some(ref sync) = args.sync_mode {
        // Remove any existing sync param from profile
        params.retain(|p| !p.starts_with("sync="));
        match sync.to_lowercase().as_str() {
            "none" | "off" => params.push("sync=none".to_string()),
            "normal" => params.push("sync=normal".to_string()),
            "full" => params.push("sync=full".to_string()),
            _ => {
                eprintln!(
                    "Warning: Unknown sync mode '{}', using 'normal'. Valid: none, normal, full",
                    sync
                );
            }
        }
    }

    // Checkpoint interval
    if let Some(interval) = args.checkpoint_interval {
        params.push(format!("checkpoint_interval={}", interval));
    }

    // Compact threshold
    if let Some(count) = args.compact_threshold {
        params.push(format!("compact_threshold={}", count));
    }

    // WAL max size (convert MB to bytes)
    if let Some(mb) = args.wal_max_size {
        params.push(format!("wal_max_size={}", mb as u64 * 1024 * 1024));
    }

    // Compression
    if let Some(ref comp) = args.compression {
        match comp.to_lowercase().as_str() {
            "on" | "true" | "1" | "yes" => params.push("compression=on".to_string()),
            "off" | "false" | "0" | "no" => params.push("compression=off".to_string()),
            _ => {
                eprintln!(
                    "Warning: Unknown compression value '{}', using 'on'. Valid: on, off",
                    comp
                );
            }
        }
    }

    // Keep snapshots
    if let Some(count) = args.keep_snapshots {
        params.push(format!("keep_snapshots={}", count));
    }

    // Checkpoint on close
    if args.no_checkpoint_on_close {
        params.push("checkpoint_on_close=off".to_string());
    }

    // Append params to DSN
    if !params.is_empty() {
        let separator = if dsn.contains('?') { "&" } else { "?" };
        dsn.push_str(separator);
        dsn.push_str(&params.join("&"));
    }

    dsn
}

/// Print persistence configuration info
fn print_persistence_info(args: &Args) {
    let sync_mode = if let Some(ref sync) = args.sync_mode {
        sync.to_lowercase()
    } else if let Some(ref profile) = args.persistence_profile {
        match profile.to_lowercase().as_str() {
            "fast" => "none".to_string(),
            "durable" => "full".to_string(),
            _ => "normal".to_string(),
        }
    } else {
        "normal".to_string()
    };

    let sync_desc = match sync_mode.as_str() {
        "none" | "off" => "none (fastest, less durable)",
        "full" => "full (slowest, most durable)",
        _ => "normal (balanced)",
    };

    println!("Persistence: WAL sync mode = {}", sync_desc);

    if let Some(interval) = args.checkpoint_interval {
        println!("Persistence: Checkpoint interval = {}s", interval);
    }

    if let Some(count) = args.compact_threshold {
        println!("Persistence: Compact threshold = {}", count);
    }

    if let Some(count) = args.keep_snapshots {
        println!("Persistence: Keep snapshots = {}", count);
    }

    if let Some(mb) = args.wal_max_size {
        println!("Persistence: WAL max size = {}MB", mb);
    }

    if let Some(ref comp) = args.compression {
        println!("Persistence: Compression = {}", comp);
    }

    if args.no_checkpoint_on_close {
        println!("Persistence: Checkpoint on close = off");
    }
}

/// Agent mode is turned on only by asking, never by what the environment
/// looks like
fn agent_mode(agent: bool, no_agent: bool) -> bool {
    !no_agent && (agent || std::env::var("STOOLAP_AGENT").is_ok_and(|v| v == "1"))
}

fn main() {
    let mut args = match Args::try_parse() {
        Ok(args) => args,
        Err(e) if e.kind() == clap::error::ErrorKind::DisplayHelp => {
            let given = |flag: &str| std::env::args().any(|a| a == flag);
            if agent_mode(given("--agent"), given("--no-agent")) {
                print!("{}", include_str!("stoolap/agent_guide.txt"));
                std::process::exit(0);
            }
            e.exit()
        }
        Err(e) => e.exit(),
    };
    let agent = agent_mode(args.agent, args.no_agent);
    if agent {
        args.quiet = true;
    }
    // Maintenance prints plain text and changes files: refused before either
    if agent && (args.snapshot || args.restore.is_some() || args.reset_volumes) {
        eprintln!("Error: --snapshot, --restore and --reset-volumes do not run in agent mode");
        std::process::exit(2);
    }

    // Build the DSN with optional query parameters
    let db_path = build_dsn(&args);

    // ── Filesystem-level operations (before Database::open) ────────
    // These work directly on disk so they can recover from broken state.

    let db_dir = if let Some(path) = db_path.strip_prefix("file://") {
        Some(std::path::PathBuf::from(
            path.split('?').next().unwrap_or(path),
        ))
    } else {
        None
    };

    // Handle --reset-volumes BEFORE --restore: delete volumes/ (and wal/)
    // so that Database::open() in the restore path doesn't choke on
    // corrupted on-disk state. This makes `--reset-volumes --restore`
    // work as documented.
    if args.reset_volumes {
        if let Some(ref dir) = db_dir {
            let vol_dir = dir.join("volumes");
            if vol_dir.exists() {
                match std::fs::remove_dir_all(&vol_dir) {
                    Ok(()) => {
                        if !args.quiet {
                            eprintln!("Removed volumes/ directory for recovery");
                        }
                    }
                    Err(e) => {
                        eprintln!("Error removing volumes/ directory: {}", e);
                        std::process::exit(1);
                    }
                }
            }
            let wal_dir = dir.join("wal");
            if wal_dir.exists() {
                match std::fs::remove_dir_all(&wal_dir) {
                    Ok(()) => {
                        if !args.quiet {
                            eprintln!("Removed wal/ directory for recovery");
                        }
                    }
                    Err(e) => {
                        eprintln!("Error removing wal/ directory: {}", e);
                        std::process::exit(1);
                    }
                }
            }
            // Also remove the stale lock file so open() doesn't fail
            let lock_file = dir.join("db.lock");
            if lock_file.exists() {
                let _ = std::fs::remove_file(&lock_file);
            }
        } else if !args.quiet {
            eprintln!("Warning: --reset-volumes only applies to file:// databases");
        }
    }

    // Handle --restore: filesystem-level recovery from backup snapshots.
    // If --reset-volumes was used above, volumes/ and wal/ are already gone.
    if args.restore.is_some() {
        let dir = match &db_dir {
            Some(d) => d,
            None => {
                eprintln!("Error: --restore requires a file:// database");
                std::process::exit(1);
            }
        };

        let snapshot_dir = dir.join("snapshots");
        if !snapshot_dir.exists() {
            eprintln!("Error: No snapshots/ directory found in {:?}", dir);
            eprintln!(
                "Create a backup first with: stoolap -d {} --snapshot",
                db_path
            );
            std::process::exit(1);
        }

        // Find snapshot files to validate
        let mut table_count = 0u32;
        if let Ok(entries) = std::fs::read_dir(&snapshot_dir) {
            for entry in entries.flatten() {
                if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                    let table_name = entry.file_name().to_string_lossy().to_string();
                    let has_bin = std::fs::read_dir(entry.path())
                        .ok()
                        .map(|e| {
                            e.flatten().any(|f| {
                                f.path()
                                    .extension()
                                    .map(|ext| ext == "bin")
                                    .unwrap_or(false)
                            })
                        })
                        .unwrap_or(false);
                    if has_bin {
                        table_count += 1;
                        if !args.quiet {
                            eprintln!("[restore] Found snapshot for table '{}'", table_name);
                        }
                    }
                }
            }
        }

        if table_count == 0 {
            eprintln!("Error: No snapshot .bin files found in {:?}", snapshot_dir);
            std::process::exit(1);
        }

        // Timestamp restore: open the database normally and use the engine's
        // restore_snapshot() API, which loads the specific timestamp without
        // deleting other backup files. Other snapshots are preserved.
        if let Some(ref ts) = args.restore
            && !ts.is_empty()
        {
            if !args.quiet {
                eprintln!("[restore] Restoring to timestamp: {}", ts);
            }

            let db = match Database::open(&db_path) {
                Ok(db) => db,
                Err(e) => {
                    eprintln!("Error opening database: {}", e);
                    std::process::exit(1);
                }
            };

            match db.restore_snapshot(Some(ts)) {
                Ok(msg) => {
                    println!("{}", msg);
                }
                Err(e) => {
                    eprintln!("Error restoring snapshot: {}", e);
                    db.close().unwrap_or(());
                    std::process::exit(1);
                }
            }

            db.close().unwrap_or(());
            return;
        }

        // Crash recovery (no timestamp): find the newest complete manifest
        // and use the timestamped restore path for a coherent snapshot set.
        let snap_dir = dir.join("snapshots");

        // Scan manifests newest-first, pick the latest whose table snapshots all exist.
        let restore_ts = {
            let mut manifests: Vec<String> = std::fs::read_dir(&snap_dir)
                .ok()
                .map(|entries| {
                    entries
                        .filter_map(|e| e.ok())
                        .filter_map(|e| {
                            let name = e.file_name().to_string_lossy().to_string();
                            name.strip_prefix("manifest-")
                                .and_then(|s| s.strip_suffix(".json"))
                                .map(|ts| ts.to_string())
                        })
                        .collect()
                })
                .unwrap_or_default();
            manifests.sort();
            manifests.reverse(); // newest first

            let mut chosen: Option<String> = None;
            for ts in &manifests {
                // Read manifest to get table list
                let manifest_path = snap_dir.join(format!("manifest-{}.json", ts));
                let Ok(content) = std::fs::read_to_string(&manifest_path) else {
                    continue;
                };
                let table_names: Vec<String> = if let Some(start) = content.find("\"tables\":[") {
                    let after = &content[start + 10..];
                    // Parse JSON string values properly (handle escaped quotes/backslashes)
                    let mut names = Vec::new();
                    let mut i = 0;
                    let bytes = after.as_bytes();
                    while i < bytes.len() {
                        if bytes[i] == b']' {
                            break;
                        }
                        if bytes[i] == b'"' {
                            i += 1;
                            let mut name = String::new();
                            while i < bytes.len() && bytes[i] != b'"' {
                                if bytes[i] == b'\\' && i + 1 < bytes.len() {
                                    i += 1; // skip backslash, take next char
                                }
                                name.push(bytes[i] as char);
                                i += 1;
                            }
                            if !name.is_empty() {
                                names.push(name);
                            }
                            i += 1; // skip closing quote
                        } else {
                            i += 1;
                        }
                    }
                    names
                } else {
                    Vec::new()
                };

                if table_names.is_empty() {
                    continue;
                }

                // Verify all table snapshots AND DDL blob exist for this timestamp
                let ddl_file = snap_dir.join(format!("ddl-{}.bin", ts));
                let all_exist = ddl_file.exists()
                    && table_names.iter().all(|table| {
                        let snapshot_file =
                            snap_dir.join(table).join(format!("snapshot-{}.bin", ts));
                        snapshot_file.exists()
                    });

                if all_exist {
                    if !args.quiet {
                        eprintln!(
                            "[restore] Found complete manifest for timestamp {} ({} tables)",
                            ts,
                            table_names.len()
                        );
                    }
                    chosen = Some(ts.clone());
                    break;
                } else if !args.quiet {
                    eprintln!(
                        "[restore] Manifest {} has missing table snapshots, trying older...",
                        ts
                    );
                }
            }
            chosen
        };

        if let Some(ts) = restore_ts {
            // Use the timestamped restore path for a coherent snapshot set.
            // This reuses the engine's restore_from_snapshot which handles
            // data + DDL matching correctly.
            if !args.quiet {
                eprintln!("[restore] Restoring from timestamp {}...", ts);
            }

            let db = match Database::open(&db_path) {
                Ok(db) => db,
                Err(e) => {
                    eprintln!("Error opening database: {}", e);
                    std::process::exit(1);
                }
            };

            match db.restore_snapshot(Some(&ts)) {
                Ok(msg) => println!("{}", msg),
                Err(e) => {
                    eprintln!("Error restoring snapshot: {}", e);
                    db.close().unwrap_or(());
                    std::process::exit(1);
                }
            }
            db.close().unwrap_or(());
        } else {
            // No manifest found — legacy fallback: open DB and let load_snapshots
            // pick the newest per table. Warn about potential DDL mismatch.
            if !args.quiet {
                eprintln!("[restore] No snapshot manifest found, using legacy restore path.");
                eprintln!("[restore] Warning: DDL (indexes, views) may not match restored data.");
            }

            // Remove volumes/ and wal/ for clean startup
            let vol_dir = dir.join("volumes");
            if vol_dir.exists()
                && let Err(e) = std::fs::remove_dir_all(&vol_dir)
            {
                eprintln!("Error removing volumes/: {}", e);
                std::process::exit(1);
            }
            let wal_dir = dir.join("wal");
            if wal_dir.exists()
                && let Err(e) = std::fs::remove_dir_all(&wal_dir)
            {
                eprintln!("Error removing wal/: {}", e);
                std::process::exit(1);
            }

            let db = match Database::open(&db_path) {
                Ok(db) => db,
                Err(e) => {
                    eprintln!("Error opening database after restore: {}", e);
                    std::process::exit(1);
                }
            };

            // Verify data loaded
            match db.query("SHOW TABLES", ()) {
                Ok(rows) => {
                    let tables: Vec<String> = rows
                        .filter_map(|r| r.ok())
                        .filter_map(|r| r.get::<String>(0).ok())
                        .collect();
                    println!(
                        "Restore complete (legacy). {} tables recovered: {}",
                        tables.len(),
                        tables.join(", ")
                    );
                }
                Err(_) => {
                    println!(
                        "Restore complete (legacy). {} tables recovered.",
                        table_count
                    );
                }
            }
            db.close().unwrap_or(());
        }
        return;
    }

    // --reset-volumes already handled above (before --restore).

    // Open the database
    let opening = Instant::now();
    let db = match Database::open(&db_path) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("Error opening database: {}", e);
            std::process::exit(1);
        }
    };
    let opened = opening.elapsed();

    // The connection's own timeout bounds every statement it runs, those of
    // a transaction too: -t, else the connection string's, else 30 seconds
    // in agent mode, which never waits on a statement for ever. Resolved
    // once, it is the timeout of every case's database as well
    let dsn_timeout_ms = Database::dsn_query_timeout(&db_path).ok().flatten();
    let run_timeout_ms =
        args.timeout_ms
            .or(dsn_timeout_ms)
            .unwrap_or(if agent { 30_000 } else { 0 });
    // The open already gave the connection the connection string's
    if run_timeout_ms != dsn_timeout_ms.unwrap_or(0)
        && let Err(e) = db.execute(&format!("PRAGMA query_timeout = {run_timeout_ms}"), ())
    {
        eprintln!("Error setting the timeout: {}", e);
        std::process::exit(1);
    }

    if !args.quiet {
        println!(
            "Connected to database: {} (opened in {}){}",
            db_path,
            script::duration(opened),
            script::BUILD_NOTE
        );
        // Show persistence info for file databases
        if db_path.starts_with("file://") {
            print_persistence_info(&args);
        }
    }

    // Handle --snapshot: open DB, create backup, close, exit
    if args.snapshot {
        match db.query("PRAGMA SNAPSHOT", ()) {
            Ok(mut rows) => {
                if let Some(Ok(row)) = rows.next()
                    && let Ok(msg) = row.get::<String>(0)
                {
                    println!("{}", msg);
                }
            }
            Err(e) => {
                eprintln!("Error creating snapshot: {}", e);
                std::process::exit(1);
            }
        }
        db.close().unwrap_or(());
        return;
    }

    // Handle execute flag - run its statements and exit
    let mode = Mode {
        agent,
        dsn: &db_path,
        case_timeout_ms: run_timeout_ms,
    };
    if let Some(ref sql) = args.execute {
        run_script_and_exit(db, sql.split('\n').map(|l| Ok(l.to_string())), &args, mode);
    }

    // Handle file flag - execute SQL from file
    if let Some(ref filename) = args.file {
        let file = match File::open(filename) {
            Ok(file) => file,
            Err(e) => {
                drop(db);
                eprintln!("Error opening file {}: {}", filename, e);
                std::process::exit(1);
            }
        };
        run_script_and_exit(db, lines_of(BufReader::new(file)), &args, mode);
    }

    // Input from a pipe runs as a script, each statement as it arrives, and
    // agent mode reads its input so whatever it comes from: it never prompts
    if agent || !std::io::stdin().is_terminal() {
        run_script_and_exit(db, lines_of(io::stdin().lock()), &args, mode);
    }

    // Interactive mode
    let mut cli = match Cli::new(db, args.json_output, args.limit, args.quiet) {
        Ok(cli) => cli,
        Err(e) => {
            eprintln!("Error initializing CLI: {}", e);
            std::process::exit(1);
        }
    };

    let ran = cli.run();
    // Drop cli (and db) before exiting so close_engine() runs
    close(cli, &args);
    if let Err(e) = ran {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

/// Closes the database `held` owns and, unless quiet, says how long the
/// close took: it seals the rows still in memory, so after writes it can
/// take longer than the statements did
fn close<T>(held: T, args: &Args) {
    let closing = Instant::now();
    drop(held);
    if !args.quiet {
        println!(
            "Database closed in {}{}",
            script::duration(closing.elapsed()),
            script::BUILD_NOTE
        );
    }
}

/// The lines of `input` without their `\n`, keeping a `\r` before it, which
/// may be part of a string literal
fn lines_of(input: impl BufRead) -> impl Iterator<Item = io::Result<String>> {
    input.split(b'\n').map(|line| {
        String::from_utf8(line?).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    })
}

/// How a script's results are written
#[derive(Clone, Copy)]
struct Mode<'a> {
    agent: bool,
    dsn: &'a str,
    /// The timeout of each case's database in agent mode, 0 for none
    case_timeout_ms: u64,
}

/// Runs a script on `db`, closes it, and exits with 1 when a statement failed
fn run_script_and_exit(
    db: Database,
    lines: impl Iterator<Item = io::Result<String>>,
    args: &Args,
    mode: Mode,
) -> ! {
    if mode.agent {
        let options = agent::Options {
            limit: args.limit,
            full: args.full,
            stop_on_error: args.on_error == OnError::Stop,
            timing: args.timing,
            repeat: args.repeat as usize,
            warmup: args.warmup as usize,
        };
        let mut out = io::BufWriter::new(io::stdout().lock());
        // Each case gets a database of its own, named so no other open
        // database shares it, with the run's timeout
        let open_case = |n: usize| -> Result<Database, String> {
            let dsn = format!("memory://stoolap-agent-case-{}-{n}", std::process::id());
            let db = Database::open(&dsn).map_err(|e| e.to_string())?;
            if mode.case_timeout_ms > 0 {
                let pragma = format!("PRAGMA query_timeout = {}", mode.case_timeout_ms);
                db.execute(&pragma, ()).map_err(|e| e.to_string())?;
            }
            Ok(db)
        };
        let result = agent::run_script(&db, mode.dsn, &open_case, lines, &mut out, &options);
        drop(out);
        close(db, args);
        match result {
            Ok(true) => std::process::exit(0),
            Ok(false) => std::process::exit(1),
            Err(e) => {
                eprintln!("Error: {}", e);
                std::process::exit(1)
            }
        }
    }
    let mut output = Output {
        out: io::BufWriter::new(io::stdout().lock()),
        format: match args.json_output {
            true => Format::Json,
            false => Format::Table,
        },
        limit: args.limit,
        quiet: args.quiet,
        timing: !args.json_output && !args.quiet,
        repeat: args.repeat as usize,
        warmup: args.warmup as usize,
    };
    let result = script::run_script(&db, lines, &mut output);
    drop(output);
    close(db, args);
    match result {
        Ok(0) => std::process::exit(0),
        Ok(_) => std::process::exit(1),
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1)
        }
    }
}

// ============================================================================
// EXPLAIN Plan Colorization
// ============================================================================

// ANSI color codes for EXPLAIN plan output (256-color, high contrast)
const C_HEADER: &str = "\x1b[1;4m"; // Bold + underline for headers
const C_DIM: &str = "\x1b[38;5;245m"; // Medium gray for arrows
const C_RESET: &str = "\x1b[0m";
const C_INDEX: &str = "\x1b[1;38;5;46m"; // Bright green for index access
const C_SEQ: &str = "\x1b[1;38;5;208m"; // Orange for seq scan warning
const C_JOIN: &str = "\x1b[1;38;5;75m"; // Light blue for join operators
const C_STATS: &str = "\x1b[1;38;5;196m"; // Bright pure red for timing/row stats
const C_LABEL: &str = "\x1b[38;5;117m"; // Sky blue for labels
const C_SUBIDX: &str = "\x1b[38;5;114m"; // Medium green for sub-index details
const C_CTE: &str = "\x1b[38;5;183m"; // Lavender for CTE/subquery scans

/// Check if rows look like EXPLAIN plan output (not a regular query with a "plan" column)
fn is_explain_output(rows: &[Vec<Value>]) -> bool {
    if let Some(first_row) = rows.first()
        && let Some(Value::Text(line)) = first_row.first()
    {
        let trimmed = line.trim();
        return trimmed.starts_with("SELECT")
            || trimmed.starts_with("INSERT")
            || trimmed.starts_with("UPDATE")
            || trimmed.starts_with("DELETE")
            || trimmed.starts_with("WITH ");
    }
    false
}

/// Colorize a single EXPLAIN plan line with ANSI escape codes
fn colorize_plan_line(line: &str) -> String {
    let trimmed = line.trim_start();
    let indent = &line[..line.len() - trimmed.len()];

    let mut result = String::with_capacity(line.len() + 80);
    result.push_str(indent);

    if trimmed.starts_with("-> ") {
        colorize_node(trimmed, &mut result);
    } else if trimmed.starts_with("SELECT")
        || trimmed.starts_with("INSERT")
        || trimmed.starts_with("UPDATE")
        || trimmed.starts_with("DELETE")
    {
        colorize_header(trimmed, &mut result);
    } else if trimmed.starts_with("WITH ") || trimmed.starts_with("Statement:") {
        result.push_str(C_HEADER);
        result.push_str(trimmed);
        result.push_str(C_RESET);
    } else {
        colorize_detail(trimmed, &mut result);
    }

    result
}

/// Split a line into the main content and trailing stats parenthetical.
/// Stats always start with "(actual " or "(rows=".
fn split_stats(line: &str) -> (&str, Option<&str>) {
    for marker in &["(actual ", "(rows="] {
        if let Some(pos) = line.rfind(marker) {
            return (line[..pos].trim_end(), Some(&line[pos..]));
        }
    }
    (line, None)
}

/// Colorize a node line starting with "-> "
fn colorize_node(line: &str, result: &mut String) {
    let after_arrow = &line[3..]; // skip "-> "

    // Dim arrow
    result.push_str(C_DIM);
    result.push_str("->");
    result.push_str(C_RESET);
    result.push(' ');

    // Indexed access patterns (green)
    let indexed = [
        "IndexScan",
        "Index Scan",
        "PK Lookup",
        "Multi-Index Scan",
        "Composite Index Scan",
        "Index Nested Loop",
    ];
    // Sequential scan patterns (orange warning)
    let sequential = ["Seq Scan", "Parallel Seq Scan"];
    // Join operator patterns (blue)
    let joins = ["Hash Join", "Merge Join", "Nested Loop ("];

    let color = if indexed.iter().any(|p| after_arrow.starts_with(p)) {
        C_INDEX
    } else if sequential.iter().any(|p| after_arrow.starts_with(p)) {
        C_SEQ
    } else if joins.iter().any(|p| after_arrow.starts_with(p)) {
        C_JOIN
    } else if after_arrow.starts_with("Subquery Scan") || after_arrow.starts_with("CTE Scan") {
        C_CTE
    } else {
        // Sub-index detail lines like "idx_cat on category: = Electronics"
        result.push_str(C_SUBIDX);
        result.push_str(after_arrow);
        result.push_str(C_RESET);
        return;
    };

    let (scan_part, stats_part) = split_stats(after_arrow);
    result.push_str(color);
    result.push_str(scan_part);
    result.push_str(C_RESET);
    if let Some(stats) = stats_part {
        result.push(' ');
        result.push_str(C_STATS);
        result.push_str(stats);
        result.push_str(C_RESET);
    }
}

/// Colorize a statement header line (SELECT, INSERT, UPDATE, DELETE)
fn colorize_header(line: &str, result: &mut String) {
    let (scan_part, stats_part) = split_stats(line);
    result.push_str(C_HEADER);
    result.push_str(scan_part);
    result.push_str(C_RESET);
    if let Some(stats) = stats_part {
        result.push(' ');
        result.push_str(C_STATS);
        result.push_str(stats);
        result.push_str(C_RESET);
    }
}

/// Colorize a detail line (labels like Columns:, Filter:, Group By:, etc.)
fn colorize_detail(line: &str, result: &mut String) {
    let labels = [
        "Columns:",
        "Filter:",
        "Index Cond:",
        "Join Cond:",
        "Group By:",
        "Order By:",
        "Having:",
        "Limit:",
        "Offset:",
        "Using:",
        "Alias:",
        "Values:",
        "Set:",
        "Source:",
    ];

    for label in &labels {
        if let Some(rest) = line.strip_prefix(label) {
            result.push_str(C_LABEL);
            result.push_str(label);
            result.push_str(C_RESET);
            result.push_str(rest);
            return;
        }
    }

    // No recognized label — render as-is
    result.push_str(line);
}

fn print_help_main() {
    println!("Stoolap SQL CLI");
    println!();
    println!("  SQL Commands:");
    println!("    SELECT ...             Execute a SELECT query");
    println!("    INSERT ...             Insert data into a table");
    println!("    UPDATE ...             Update data in a table");
    println!("    DELETE ...             Delete data from a table");
    println!("    CREATE TABLE ...       Create a new table");
    println!("    CREATE INDEX ...       Create an index on a column");
    println!("    SHOW TABLES            List all tables");
    println!("    SHOW CREATE TABLE ...  Show CREATE TABLE statement for a table");
    println!("    SHOW INDEXES FROM ...  Show indexes for a table");
    println!();
    println!("  Transaction Commands:");
    println!("    BEGIN                  Start a new transaction");
    println!("    COMMIT                 Commit the current transaction");
    println!("    ROLLBACK               Rollback the current transaction");
    println!();
    println!("  Special Commands:");
    println!("    help, \\h, \\?          Show this help message");
    println!();
}
