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

use std::fs::File;
use std::io::{self, BufRead, BufReader, IsTerminal};
use std::time::Instant;

use clap::Parser;
use comfy_table::{presets::UTF8_FULL_CONDENSED, Cell, ContentArrangement, Table};
use rustyline::error::ReadlineError;
use rustyline::history::DefaultHistory;
use rustyline::{Config, DefaultEditor, EditMode, Editor};

use stoolap::api::{Database, Transaction as ApiTransaction};
use stoolap::common::version::{MAJOR, MINOR, PATCH};
use stoolap::Value;

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
This CLI provides an interactive interface to execute SQL queries and manage your database.\n\n\
PERSISTENCE DSN PARAMETERS:\n\
  file:///path/to/db?param=value&param2=value2\n\n\
  sync=none|normal|full       WAL sync mode (default: normal)\n\
  snapshot_interval=SECS      Snapshot interval in seconds (default: 300)\n\
  keep_snapshots=COUNT        Number of snapshots to keep (default: 5)\n\
  wal_max_size=BYTES          Max WAL file size before rotation (default: 67108864)\n\
  wal_buffer_size=BYTES       WAL buffer size (default: 65536)\n\
  wal_flush_trigger=BYTES     Buffer size to trigger flush (default: 32768)\n\
  commit_batch_size=COUNT     Commits to batch before sync (default: 100)\n\
  sync_interval_ms=MS         Min time between syncs (default: 10)\n\
  compression=on|off          Enable/disable all compression (default: on)\n\
  wal_compression=on|off      WAL compression only (default: on)\n\
  snapshot_compression=on|off Snapshot compression only (default: on)\n\
  compression_threshold=BYTES Min size to compress (default: 64)\n\n\
EXAMPLES:\n\
  stoolap -d memory://                                    In-memory database\n\
  stoolap -d file:///tmp/mydb                             Persistent database\n\
  stoolap -d file:///tmp/mydb?sync=full                   Maximum durability\n\
  stoolap -d file:///tmp/mydb?sync=none&compression=off   Maximum performance\n\
  stoolap -d file:///tmp/mydb --profile durable           Use durable preset\n\
  stoolap -d file:///tmp/mydb --sync full --compression off"
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
    /// - none: Fastest but least durable - doesn't force syncs
    /// - normal: Syncs on transaction commits (default)
    /// - full: Forces syncs on every WAL write - slowest but most durable
    #[arg(short = 's', long = "sync", value_name = "MODE")]
    sync_mode: Option<String>,

    /// Persistence profile preset (fast, normal, durable)
    /// - fast: Optimized for performance, less durable
    /// - normal: Balanced performance and durability (default)
    /// - durable: Maximum durability, slower performance
    #[arg(short = 'p', long = "profile", value_name = "PROFILE")]
    persistence_profile: Option<String>,

    /// Snapshot interval in seconds (default: 300)
    #[arg(long = "snapshot-interval", value_name = "SECONDS")]
    snapshot_interval: Option<u32>,

    /// Number of snapshots to keep (default: 5)
    #[arg(long = "keep-snapshots", value_name = "COUNT")]
    keep_snapshots: Option<u32>,

    /// Maximum WAL file size in MB before rotation (default: 64)
    #[arg(long = "wal-max-size", value_name = "MB")]
    wal_max_size: Option<u32>,

    /// Enable or disable compression for WAL and snapshots (default: on)
    #[arg(long = "compression", value_name = "on|off")]
    compression: Option<String>,
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
    current_query: String,
    in_multi_line: bool,
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
            in_multi_line: false,
        })
    }

    fn get_prompt(&self) -> &'static str {
        if self.in_multi_line {
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
                    if !self.in_multi_line && line.is_empty() {
                        continue;
                    }

                    // Handle special commands (only when not in multi-line mode)
                    if !self.in_multi_line {
                        match line.to_lowercase().as_str() {
                            "exit" | "quit" | "\\q" => {
                                if self.in_transaction {
                                    eprintln!("\x1b[1;33mWarning: Exiting with active transaction. Rolling back...\x1b[0m");
                                    let _ = self.rollback_transaction();
                                }
                                break;
                            }
                            "help" | "\\h" | "\\?" => {
                                self.print_help();
                                continue;
                            }
                            _ => {}
                        }
                    }

                    // Check for transaction control statements
                    let upper_line = line.to_uppercase();
                    if upper_line == "BEGIN"
                        || upper_line == "COMMIT"
                        || upper_line == "ROLLBACK"
                        || upper_line.starts_with("BEGIN ")
                    {
                        let _ = self.editor.add_history_entry(line);
                        let start = Instant::now();
                        if let Err(e) = self.execute_query(line) {
                            eprintln!("\x1b[1;31mError:\x1b[0m {}", e);
                        } else {
                            println!("\x1b[1;32mQuery executed in {:?}\x1b[0m", start.elapsed());
                        }
                        continue;
                    }

                    // Add line to current query
                    if !self.current_query.is_empty() {
                        self.current_query.push('\n');
                    }
                    self.current_query.push_str(line);

                    // Check if query ends with semicolon
                    let full_query = self.current_query.trim().to_string();
                    if full_query.ends_with(';') {
                        // Add to history
                        let history_entry = full_query.replace('\n', "\\n");
                        let _ = self.editor.add_history_entry(&history_entry);

                        self.in_multi_line = false;

                        // Split and execute statements
                        let statements = split_sql_statements(&full_query);
                        for stmt in statements {
                            let stmt = stmt.trim();
                            if stmt.is_empty() {
                                continue;
                            }

                            let start = Instant::now();
                            if let Err(e) = self.execute_query(stmt) {
                                eprintln!("\x1b[1;31mError:\x1b[0m {}", e);
                            } else {
                                println!(
                                    "\x1b[1;32mQuery executed in {:?}\x1b[0m",
                                    start.elapsed()
                                );
                            }
                        }

                        self.current_query.clear();
                    } else {
                        self.in_multi_line = true;
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

    fn execute_query(&mut self, query: &str) -> Result<(), String> {
        let upper_query = query.to_uppercase();
        let upper_query = upper_query.trim();

        // Handle special commands
        match upper_query {
            "HELP" | "\\H" | "\\?" => {
                self.print_help();
                return Ok(());
            }
            _ => {}
        }

        // Handle transaction commands
        if upper_query.starts_with("BEGIN") {
            return self.begin_transaction();
        } else if upper_query == "COMMIT" {
            return self.commit_transaction();
        } else if upper_query == "ROLLBACK" {
            return self.rollback_transaction();
        }

        // Check if it's a query that returns rows
        // Include statements with RETURNING clause (INSERT/UPDATE/DELETE RETURNING)
        if upper_query.starts_with("SELECT")
            || upper_query.starts_with("WITH")
            || upper_query.starts_with("SHOW")
            || upper_query.starts_with("DESCRIBE")
            || upper_query.starts_with("DESC ")
            || upper_query.starts_with("EXPLAIN")
            || (upper_query.starts_with("PRAGMA") && !upper_query.contains('='))
            || upper_query.contains(" RETURNING ")
            || upper_query.ends_with(" RETURNING")
        {
            self.execute_read_query(query)
        } else {
            self.execute_write_query(query)
        }
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

    fn execute_read_query(&mut self, query: &str) -> Result<(), String> {
        let rows_result = if self.in_transaction {
            if let Some(ref mut tx) = self.tx {
                tx.query(query, ()).map_err(|e| e.to_string())?
            } else {
                return Err("Transaction not available".to_string());
            }
        } else {
            self.db.query(query, ()).map_err(|e| e.to_string())?
        };

        let columns: Vec<String> = rows_result.columns().to_vec();

        // Collect all rows
        let mut all_rows: Vec<Vec<Value>> = Vec::new();
        for row_result in rows_result {
            let row = row_result.map_err(|e| e.to_string())?;
            let mut values = Vec::new();
            for i in 0..row.len() {
                values.push(row.get_value(i).cloned().unwrap_or(Value::null_unknown()));
            }
            all_rows.push(values);
        }

        let row_count = all_rows.len();

        if self.json_output {
            self.output_json(&columns, &all_rows, row_count)?;
        } else {
            self.output_table(&columns, &all_rows, row_count)?;
        }

        Ok(())
    }

    fn execute_write_query(&mut self, query: &str) -> Result<(), String> {
        let rows_affected = if self.in_transaction {
            if let Some(ref mut tx) = self.tx {
                tx.execute(query, ()).map_err(|e| e.to_string())?
            } else {
                return Err("Transaction not available".to_string());
            }
        } else {
            self.db.execute(query, ()).map_err(|e| e.to_string())?
        };

        if self.json_output {
            println!(r#"{{"rows_affected":{}}}"#, rows_affected);
        } else {
            let row_text = if rows_affected == 1 { "row" } else { "rows" };
            println!("\x1b[1;32m{} {} affected\x1b[0m", rows_affected, row_text);
        }

        Ok(())
    }

    fn output_json(
        &self,
        columns: &[String],
        rows: &[Vec<Value>],
        row_count: usize,
    ) -> Result<(), String> {
        let json_rows: Vec<Vec<serde_json::Value>> = rows
            .iter()
            .map(|row| row.iter().map(value_to_json).collect())
            .collect();

        let result = serde_json::json!({
            "columns": columns,
            "rows": json_rows,
            "count": row_count
        });

        println!(
            "{}",
            serde_json::to_string(&result).map_err(|e| e.to_string())?
        );
        Ok(())
    }

    fn output_table(
        &self,
        columns: &[String],
        rows: &[Vec<Value>],
        row_count: usize,
    ) -> Result<(), String> {
        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL_CONDENSED)
            .set_content_arrangement(ContentArrangement::Dynamic);

        // Add header
        table.set_header(columns.iter().map(Cell::new));

        // Smart truncation with limit
        if self.limit > 0 && row_count > self.limit {
            let top_rows = self.limit / 2;
            let bottom_rows = self.limit - top_rows;

            // Add top rows
            for row in rows.iter().take(top_rows) {
                table.add_row(row.iter().map(|v| Cell::new(format_value(v))));
            }

            // Add truncation indicator
            let hidden_rows = row_count - self.limit;
            let mut truncation_row: Vec<Cell> = Vec::new();
            let message = format!("... ({} more rows) ...", hidden_rows);
            for (i, _) in columns.iter().enumerate() {
                if i == columns.len() / 2 {
                    truncation_row.push(Cell::new(&message));
                } else {
                    truncation_row.push(Cell::new(""));
                }
            }
            table.add_row(truncation_row);

            // Add bottom rows
            let start_idx = row_count.saturating_sub(bottom_rows).max(top_rows);
            for row in rows.iter().skip(start_idx) {
                table.add_row(row.iter().map(|v| Cell::new(format_value(v))));
            }
        } else {
            // Add all rows
            for row in rows {
                table.add_row(row.iter().map(|v| Cell::new(format_value(v))));
            }
        }

        println!("{table}");

        // Print summary
        let row_text = if row_count == 1 { "row" } else { "rows" };
        if self.limit > 0 && row_count > self.limit {
            println!(
                "\x1b[1;32m{} {} in set (showing {})\x1b[0m",
                row_count, row_text, self.limit
            );
        } else {
            println!("\x1b[1;32m{} {} in set\x1b[0m", row_count, row_text);
        }

        Ok(())
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
        println!("    SHOW TABLES            List all tables");
        println!("    SHOW CREATE TABLE ...  Show CREATE TABLE statement for a table");
        println!("    SHOW INDEXES FROM ...  Show indexes for a table");
        println!();
        println!("  \x1b[1;33mTransaction Commands:\x1b[0m");
        println!("    BEGIN                  Start a new transaction");
        println!("    COMMIT                 Commit the current transaction");
        println!("    ROLLBACK               Rollback the current transaction");
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

    // Snapshot interval
    if let Some(interval) = args.snapshot_interval {
        params.push(format!("snapshot_interval={}", interval));
    }

    // Keep snapshots
    if let Some(count) = args.keep_snapshots {
        params.push(format!("keep_snapshots={}", count));
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

    if let Some(interval) = args.snapshot_interval {
        println!("Persistence: Snapshot interval = {}s", interval);
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
}

fn main() {
    let args = Args::parse();

    // Build the DSN with optional query parameters
    let db_path = build_dsn(&args);

    // Open the database
    let db = match Database::open(&db_path) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("Error opening database: {}", e);
            std::process::exit(1);
        }
    };

    if !args.quiet {
        println!("Connected to database: {}", db_path);
        // Show persistence info for file databases
        if db_path.starts_with("file://") {
            print_persistence_info(&args);
        }
    }

    // Handle execute flag - run single query and exit
    if let Some(ref sql) = args.execute {
        if let Err(e) =
            execute_query_with_options(&db, sql, args.json_output, args.quiet, args.limit)
        {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
        return;
    }

    // Handle file flag - execute SQL from file
    if let Some(ref filename) = args.file {
        if let Err(e) = execute_from_file(&db, filename, args.json_output, args.quiet, args.limit) {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
        return;
    }

    // Check if we're getting input from a pipe
    let is_pipe = !std::io::stdin().is_terminal();

    if is_pipe {
        if let Err(e) = execute_piped_input(&db, args.json_output, args.quiet, args.limit) {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
        return;
    }

    // Interactive mode
    let mut cli = match Cli::new(db, args.json_output, args.limit, args.quiet) {
        Ok(cli) => cli,
        Err(e) => {
            eprintln!("Error initializing CLI: {}", e);
            std::process::exit(1);
        }
    };

    if let Err(e) = cli.run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn execute_from_file(
    db: &Database,
    filename: &str,
    json_output: bool,
    quiet: bool,
    row_limit: usize,
) -> Result<(), String> {
    let file =
        File::open(filename).map_err(|e| format!("Error opening file {}: {}", filename, e))?;
    let reader = BufReader::new(file);
    let mut current_statement = String::new();

    for line_result in reader.lines() {
        let line = line_result.map_err(|e| format!("Error reading file: {}", e))?;

        // Skip shell-style comment lines
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            continue;
        }

        // Skip SQL-style comment lines
        if trimmed.starts_with("--") || (trimmed.starts_with("/*") && trimmed.ends_with("*/")) {
            continue;
        }

        // If blank line and we have a statement, execute it
        if trimmed.is_empty() && !current_statement.is_empty() {
            let q = current_statement.trim().to_string();
            current_statement.clear();

            if !q.is_empty() {
                let statements = split_sql_statements(&q);
                for stmt in statements {
                    let stmt = stmt.trim();
                    if stmt.is_empty() {
                        continue;
                    }

                    if let Err(e) =
                        execute_query_with_options(db, stmt, json_output, quiet, row_limit)
                    {
                        eprintln!("Error: {}", e);
                    }
                }
            }
        } else {
            current_statement.push_str(&line);
            current_statement.push('\n');
        }
    }

    // Execute any remaining statement
    if !current_statement.is_empty() {
        let q = current_statement.trim().to_string();
        if !q.is_empty() {
            let statements = split_sql_statements(&q);
            for stmt in statements {
                let stmt = stmt.trim();
                if stmt.is_empty() {
                    continue;
                }

                if let Err(e) = execute_query_with_options(db, stmt, json_output, quiet, row_limit)
                {
                    eprintln!("Error: {}", e);
                }
            }
        }
    }

    Ok(())
}

fn execute_piped_input(
    db: &Database,
    json_output: bool,
    quiet: bool,
    row_limit: usize,
) -> Result<(), String> {
    let stdin = io::stdin();
    let reader = stdin.lock();
    let mut current_statement = String::new();

    for line_result in reader.lines() {
        let line = line_result.map_err(|e| format!("Error reading input: {}", e))?;

        // Skip shell-style comment lines
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            continue;
        }

        // Skip SQL-style comment lines
        if trimmed.starts_with("--") || (trimmed.starts_with("/*") && trimmed.ends_with("*/")) {
            continue;
        }

        // If blank line and we have a statement, execute it
        if trimmed.is_empty() && !current_statement.is_empty() {
            let q = current_statement.trim().to_string();
            current_statement.clear();

            if !q.is_empty() {
                let statements = split_sql_statements(&q);
                for stmt in statements {
                    let stmt = stmt.trim();
                    if stmt.is_empty() {
                        continue;
                    }

                    let start = Instant::now();
                    if let Err(e) =
                        execute_query_with_options(db, stmt, json_output, quiet, row_limit)
                    {
                        eprintln!("Error: {}", e);
                    } else if !json_output && !quiet {
                        println!("Query executed in {:?}", start.elapsed());
                    }
                }
            }
        } else {
            current_statement.push_str(&line);
            current_statement.push('\n');
        }
    }

    // Execute any remaining statement
    if !current_statement.is_empty() {
        let q = current_statement.trim().to_string();
        if !q.is_empty() {
            let statements = split_sql_statements(&q);
            for stmt in statements {
                let stmt = stmt.trim();
                if stmt.is_empty() {
                    continue;
                }

                let start = Instant::now();
                if let Err(e) = execute_query_with_options(db, stmt, json_output, quiet, row_limit)
                {
                    eprintln!("Error: {}", e);
                } else if !json_output && !quiet {
                    println!("Query executed in {:?}", start.elapsed());
                }
            }
        }
    }

    Ok(())
}

fn execute_query_with_options(
    db: &Database,
    query: &str,
    json_output: bool,
    quiet: bool,
    row_limit: usize,
) -> Result<(), String> {
    let upper_query = query.to_uppercase();
    let upper_query = upper_query.trim();

    // Handle special commands
    match upper_query {
        "HELP" | "\\H" | "\\?" => {
            print_help_main();
            return Ok(());
        }
        "EXIT" | "QUIT" | "\\Q" => {
            return Err("exit requested".to_string());
        }
        _ => {}
    }

    // Handle parameter syntax: "SELECT ... -- PARAMS: val1, val2"
    let (sql, params) = parse_params(query);

    // Check if it's a query that returns rows
    // Include statements with RETURNING clause (INSERT/UPDATE/DELETE RETURNING)
    if upper_query.starts_with("SELECT")
        || upper_query.starts_with("WITH")
        || upper_query.starts_with("SHOW")
        || upper_query.starts_with("DESCRIBE")
        || upper_query.starts_with("DESC ")
        || upper_query.starts_with("EXPLAIN")
        || (upper_query.starts_with("PRAGMA") && !upper_query.contains('='))
        || upper_query.contains(" RETURNING ")
        || upper_query.ends_with(" RETURNING")
    {
        let rows_result = db.query(&sql, params).map_err(|e| e.to_string())?;

        let columns: Vec<String> = rows_result.columns().to_vec();

        // Collect all rows
        let mut all_rows: Vec<Vec<Value>> = Vec::new();
        for row_result in rows_result {
            let row = row_result.map_err(|e| e.to_string())?;
            let mut values = Vec::new();
            for i in 0..row.len() {
                values.push(row.get_value(i).cloned().unwrap_or(Value::null_unknown()));
            }
            all_rows.push(values);
        }

        let row_count = all_rows.len();

        if json_output {
            output_json(&columns, &all_rows, row_count)?;
        } else {
            output_table(&columns, &all_rows, row_count, row_limit, quiet)?;
        }
    } else {
        // Execute a non-query statement
        let rows_affected = db.execute(&sql, params).map_err(|e| e.to_string())?;

        if json_output {
            println!(r#"{{"rows_affected":{}}}"#, rows_affected);
        } else if !quiet {
            println!("{} rows affected", rows_affected);
        }
    }

    Ok(())
}

fn parse_params(query: &str) -> (String, Vec<Value>) {
    let parts: Vec<&str> = query.split(" -- PARAMS: ").collect();

    if parts.len() <= 1 {
        // Remove trailing semicolon
        let sql = parts[0].trim().trim_end_matches(';').to_string();
        return (sql, Vec::new());
    }

    let sql = parts[0].trim().trim_end_matches(';').to_string();
    let param_string = parts[1].trim();

    let mut params = Vec::new();
    for val in param_string.split(',') {
        params.push(convert_param_value(val.trim()));
    }

    (sql, params)
}

fn convert_param_value(value: &str) -> Value {
    // Try to convert to integer
    if let Ok(i) = value.parse::<i64>() {
        return Value::Integer(i);
    }

    // Try to convert to float (only if contains '.')
    if value.contains('.') {
        if let Ok(f) = value.parse::<f64>() {
            return Value::Float(f);
        }
    }

    // Try to convert to boolean
    if value == "true" {
        return Value::Boolean(true);
    }
    if value == "false" {
        return Value::Boolean(false);
    }

    // Check for null
    if value.eq_ignore_ascii_case("null") {
        return Value::null_unknown();
    }

    // Default to string
    Value::text(value)
}

fn output_json(columns: &[String], rows: &[Vec<Value>], row_count: usize) -> Result<(), String> {
    let json_rows: Vec<Vec<serde_json::Value>> = rows
        .iter()
        .map(|row| row.iter().map(value_to_json).collect())
        .collect();

    let result = serde_json::json!({
        "columns": columns,
        "rows": json_rows,
        "count": row_count
    });

    println!(
        "{}",
        serde_json::to_string(&result).map_err(|e| e.to_string())?
    );
    Ok(())
}

fn output_table(
    columns: &[String],
    rows: &[Vec<Value>],
    row_count: usize,
    row_limit: usize,
    quiet: bool,
) -> Result<(), String> {
    // Print the column names
    for (i, column) in columns.iter().enumerate() {
        if i > 0 {
            print!(" | ");
        }
        print!("{}", column);
    }
    println!();

    // Print a separator
    for (i, _) in columns.iter().enumerate() {
        if i > 0 {
            print!("-+-");
        }
        print!("----");
    }
    println!();

    // Display rows with smart truncation
    if row_limit == 0 || row_count <= row_limit {
        for row in rows {
            for (i, value) in row.iter().enumerate() {
                if i > 0 {
                    print!(" | ");
                }
                print!("{}", format_value(value));
            }
            println!();
        }

        if !quiet {
            println!("{} rows in set", row_count);
        }
    } else {
        // Smart truncation
        let top_rows = row_limit / 2;
        let bottom_rows = row_limit - top_rows;

        // Show top rows
        for row in rows.iter().take(top_rows) {
            for (i, value) in row.iter().enumerate() {
                if i > 0 {
                    print!(" | ");
                }
                print!("{}", format_value(value));
            }
            println!();
        }

        // Show truncation indicator
        let hidden_rows = row_count - row_limit;
        println!();
        println!("    \x1b[2m... ({} more rows) ...\x1b[0m", hidden_rows);
        println!();

        // Show bottom rows
        let start_idx = row_count.saturating_sub(bottom_rows).max(top_rows);
        for row in rows.iter().skip(start_idx) {
            for (i, value) in row.iter().enumerate() {
                if i > 0 {
                    print!(" | ");
                }
                print!("{}", format_value(value));
            }
            println!();
        }

        if !quiet {
            println!("{} rows in set (showing {})", row_count, row_limit);
        }
    }

    Ok(())
}

fn format_value(value: &Value) -> String {
    match value {
        Value::Null(_) => "NULL".to_string(),
        Value::Integer(i) => i.to_string(),
        Value::Float(f) => {
            if *f == f.trunc() {
                format!("{:.1}", f)
            } else {
                format!("{:.4}", f)
                    .trim_end_matches('0')
                    .trim_end_matches('.')
                    .to_string()
            }
        }
        Value::Text(s) => s.to_string(),
        Value::Boolean(b) => if *b { "true" } else { "false" }.to_string(),
        Value::Timestamp(ts) => ts.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        Value::Json(s) => s.to_string(),
    }
}

fn value_to_json(value: &Value) -> serde_json::Value {
    match value {
        Value::Null(_) => serde_json::Value::Null,
        Value::Integer(i) => serde_json::json!(i),
        Value::Float(f) => serde_json::json!(f),
        Value::Text(s) => serde_json::json!(s.as_ref()),
        Value::Boolean(b) => serde_json::json!(b),
        Value::Timestamp(ts) => serde_json::json!(ts.format("%Y-%m-%dT%H:%M:%SZ").to_string()),
        Value::Json(s) => serde_json::json!(s.as_ref()),
    }
}

/// Split SQL statements by semicolons, handling quotes and comments
fn split_sql_statements(input: &str) -> Vec<String> {
    let mut statements = Vec::new();
    let mut current_statement = String::new();

    let mut in_single_quotes = false;
    let mut in_double_quotes = false;
    let mut in_line_comment = false;
    let mut in_block_comment = false;

    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let char = chars[i];

        // Handle end of line comment
        if in_line_comment {
            if char == '\n' {
                in_line_comment = false;
                current_statement.push(char);
            }
            i += 1;
            continue;
        }

        // Handle start of line comment
        // Only treat -- as comment if followed by whitespace/newline/EOF
        // Otherwise, it's double negation (--val or --5)
        if !in_single_quotes
            && !in_double_quotes
            && !in_block_comment
            && char == '-'
            && i + 1 < chars.len()
            && chars[i + 1] == '-'
        {
            // Check what follows the second dash
            let after_second_dash = if i + 2 < chars.len() {
                chars[i + 2]
            } else {
                '\0' // End of input
            };
            // Only treat as comment if followed by whitespace, newline, or EOF
            if after_second_dash == '\0'
                || after_second_dash == ' '
                || after_second_dash == '\t'
                || after_second_dash == '\n'
                || after_second_dash == '\r'
            {
                in_line_comment = true;
                i += 2;
                continue;
            }
            // Otherwise, fall through and treat as two minus operators
        }

        // Handle end of block comment
        if in_block_comment {
            if char == '*' && i + 1 < chars.len() && chars[i + 1] == '/' {
                in_block_comment = false;
                i += 2;
                continue;
            }
            i += 1;
            continue;
        }

        // Handle start of block comment
        if !in_single_quotes
            && !in_double_quotes
            && char == '/'
            && i + 1 < chars.len()
            && chars[i + 1] == '*'
        {
            in_block_comment = true;
            i += 2;
            continue;
        }

        // Handle quotes
        if !in_block_comment && !in_line_comment {
            if char == '\'' && (i == 0 || chars[i - 1] != '\\') {
                in_single_quotes = !in_single_quotes;
            } else if char == '"' && (i == 0 || chars[i - 1] != '\\') {
                in_double_quotes = !in_double_quotes;
            }
        }

        // If we find a semicolon outside of quotes and comments, it's a statement delimiter
        if char == ';'
            && !in_single_quotes
            && !in_double_quotes
            && !in_block_comment
            && !in_line_comment
        {
            statements.push(current_statement.clone());
            current_statement.clear();
        } else {
            current_statement.push(char);
        }

        i += 1;
    }

    // Add any remaining statement
    if !current_statement.is_empty() {
        statements.push(current_statement);
    }

    statements
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
