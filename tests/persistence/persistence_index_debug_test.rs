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

use std::fs;
use stoolap::Database;
use tempfile::tempdir;

// WAL constants from the new format
const WAL_ENTRY_MAGIC: u32 = 0x454C4157; // "WALE" in little-endian

#[test]
fn test_persistence_index_debug() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    eprintln!("=== PHASE 1: Create table, index, and insert ===");

    {
        let db = Database::open(&dsn).unwrap();
        db.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT)",
            (),
        )
        .unwrap();
        eprintln!("Table created");

        db.execute("CREATE UNIQUE INDEX idx_email ON users(email)", ())
            .unwrap();
        eprintln!("Unique index created");

        db.execute(
            "INSERT INTO users (id, email) VALUES (1, 'test@example.com')",
            (),
        )
        .unwrap();
        eprintln!("Row inserted");

        db.close().unwrap();
    }

    // Dump WAL with 32-byte header format
    let wal_dir = db_path.join("wal");
    if wal_dir.exists() {
        for entry in fs::read_dir(&wal_dir).unwrap() {
            let entry = entry.unwrap();
            let content = fs::read(entry.path()).unwrap();
            eprintln!(
                "\nWAL file: {:?} ({} bytes)",
                entry.file_name(),
                content.len()
            );

            let mut pos = 0;
            let mut entry_num = 0;
            while pos + 32 <= content.len() {
                // Parse 32-byte header
                let magic = u32::from_le_bytes(content[pos..pos + 4].try_into().unwrap());
                if magic != WAL_ENTRY_MAGIC {
                    break;
                }

                let flags = content[pos + 5];
                let header_size =
                    u16::from_le_bytes(content[pos + 6..pos + 8].try_into().unwrap()) as usize;
                let lsn = u64::from_le_bytes(content[pos + 8..pos + 16].try_into().unwrap());
                let entry_size =
                    u32::from_le_bytes(content[pos + 24..pos + 28].try_into().unwrap()) as usize;

                pos += header_size;

                if pos + entry_size + 4 > content.len() {
                    break;
                }

                let data = &content[pos..pos + entry_size];
                let mut dpos = 0;
                let txn_id = i64::from_le_bytes(data[dpos..dpos + 8].try_into().unwrap());
                dpos += 8;
                let table_len =
                    u16::from_le_bytes(data[dpos..dpos + 2].try_into().unwrap()) as usize;
                dpos += 2;
                let table_name = String::from_utf8_lossy(&data[dpos..dpos + table_len]);
                dpos += table_len;
                let row_id = i64::from_le_bytes(data[dpos..dpos + 8].try_into().unwrap());
                dpos += 8;
                let op = data[dpos];

                let op_name = match op {
                    1 => "Insert",
                    2 => "Update",
                    3 => "Delete",
                    4 => "Commit",
                    5 => "Rollback",
                    6 => "CreateTable",
                    7 => "DropTable",
                    8 => "AlterTable",
                    9 => "CreateIndex",
                    10 => "DropIndex",
                    _ => "Unknown",
                };
                let commit_flag = if flags & 0x02 != 0 {
                    " [COMMIT_MARKER]"
                } else {
                    ""
                };
                let abort_flag = if flags & 0x04 != 0 {
                    " [ABORT_MARKER]"
                } else {
                    ""
                };
                eprintln!(
                    "  Entry {}: LSN={}, flags={:#04x}, txn={}, table='{}', row={}, op={} ({}){}{}",
                    entry_num,
                    lsn,
                    flags,
                    txn_id,
                    table_name,
                    row_id,
                    op,
                    op_name,
                    commit_flag,
                    abort_flag
                );

                pos += entry_size + 4;
                entry_num += 1;
            }
        }
    }

    eprintln!("\n=== PHASE 2: Reopen and test unique constraint ===");

    let db = Database::open(&dsn).unwrap();

    // Check existing data
    let count: i64 = db.query_one("SELECT COUNT(*) FROM users", ()).unwrap();
    eprintln!("Users count after reopen: {}", count);

    // Try to insert duplicate
    let result = db.execute(
        "INSERT INTO users (id, email) VALUES (2, 'test@example.com')",
        (),
    );
    let is_err = result.is_err();
    match result {
        Ok(_) => eprintln!("FAILURE: Duplicate insert succeeded (should have failed)"),
        Err(e) => eprintln!("Duplicate rejected as expected: {}", e),
    }

    assert!(is_err, "Duplicate should be rejected");
}
