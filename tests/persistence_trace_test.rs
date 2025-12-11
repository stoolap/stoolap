use std::fs;
use stoolap::Database;
use tempfile::tempdir;

#[test]
fn test_persistence_trace() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let dsn = format!("file://{}", db_path.display());

    eprintln!("=== PHASE 1: Create and Insert ===");

    // Phase 1: Create table and insert data
    {
        let db = Database::open(&dsn).unwrap();

        db.execute(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, value INTEGER)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO test (id, value) VALUES (1, 100)", ())
            .unwrap();
        db.execute("INSERT INTO test (id, value) VALUES (2, 200)", ())
            .unwrap();

        let count: i64 = db.query_one("SELECT COUNT(*) FROM test", ()).unwrap();
        eprintln!("Phase 1 count: {}", count);

        // Print WAL content in a more detailed way
        let wal_dir = db_path.join("wal");
        if wal_dir.exists() {
            for entry in fs::read_dir(&wal_dir).unwrap() {
                let entry = entry.unwrap();
                let content = fs::read(&entry.path()).unwrap();
                eprintln!(
                    "\nWAL file: {:?} ({} bytes)",
                    entry.file_name(),
                    content.len()
                );

                // Parse WAL entries manually
                let mut pos = 0;
                let mut entry_num = 0;
                while pos + 16 <= content.len() {
                    let lsn = u64::from_le_bytes(content[pos..pos + 8].try_into().unwrap());
                    let size =
                        u64::from_le_bytes(content[pos + 8..pos + 16].try_into().unwrap()) as usize;
                    pos += 16;

                    if pos + size > content.len() {
                        eprintln!(
                            "  Entry {}: LSN={}, size={} (truncated)",
                            entry_num, lsn, size
                        );
                        break;
                    }

                    // Parse entry
                    let data = &content[pos..pos + size];
                    if data.len() >= 32 {
                        let txn_id = i64::from_le_bytes(data[0..8].try_into().unwrap());
                        let row_id = i64::from_le_bytes(data[8..16].try_into().unwrap());
                        let _timestamp = i64::from_le_bytes(data[16..24].try_into().unwrap());

                        // Table name length
                        if data.len() >= 26 {
                            let table_len =
                                u16::from_le_bytes(data[24..26].try_into().unwrap()) as usize;
                            if data.len() >= 26 + table_len {
                                let table_name = String::from_utf8_lossy(&data[26..26 + table_len]);
                                let op_pos = 26 + table_len;
                                if data.len() > op_pos {
                                    let op = data[op_pos];
                                    let op_name = match op {
                                        1 => "CreateTable",
                                        2 => "DropTable",
                                        3 => "Insert",
                                        4 => "Update",
                                        5 => "Delete",
                                        6 => "Commit",
                                        7 => "Rollback",
                                        9 => "CreateIndex",
                                        10 => "DropIndex",
                                        _ => "Unknown",
                                    };
                                    eprintln!("  Entry {}: LSN={}, txn={}, row={}, table='{}', op={} ({})", 
                                             entry_num, lsn, txn_id, row_id, table_name, op, op_name);
                                }
                            }
                        }
                    }

                    pos += size;
                    entry_num += 1;
                }
            }
        }

        db.close().unwrap();
    }

    eprintln!("\n=== PHASE 2: Reopen and verify ===");

    // Phase 2: Reopen and verify
    {
        let db = Database::open(&dsn).unwrap();

        // Check if table exists
        let table_result = db.query("SELECT * FROM test", ());
        match table_result {
            Ok(rows) => {
                let collected: Vec<_> = rows.collect();
                eprintln!("Table 'test' exists with {} rows:", collected.len());
                for row in collected {
                    match row {
                        Ok(r) => {
                            let id: i64 = r.get(0).unwrap();
                            let value: i64 = r.get(1).unwrap();
                            eprintln!("  id={}, value={}", id, value);
                        }
                        Err(e) => eprintln!("  Error reading row: {}", e),
                    }
                }
            }
            Err(e) => eprintln!("Table 'test' query failed: {}", e),
        }

        let count: i64 = db.query_one("SELECT COUNT(*) FROM test", ()).unwrap();
        eprintln!("\nPhase 2 count: {}", count);

        if count == 2 {
            eprintln!("✅ SUCCESS!");
        } else {
            eprintln!("❌ FAILURE: Expected 2, got {}", count);
        }

        assert_eq!(count, 2);
    }
}
