use std::fs;
use stoolap::Database;
use tempfile::tempdir;

// WAL constants from the new format
const WAL_ENTRY_MAGIC: u32 = 0x454C4157; // "WALE" in little-endian

#[test]
fn test_persistence_wal_dump2() {
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

        db.close().unwrap();
    }

    // Dump WAL with 32-byte header format
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

            let mut pos = 0;
            let mut entry_num = 0;
            while pos + 32 <= content.len() {
                // Parse 32-byte header
                let magic = u32::from_le_bytes(content[pos..pos + 4].try_into().unwrap());
                if magic != WAL_ENTRY_MAGIC {
                    break;
                }

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

                // txn_id (8 bytes)
                let txn_id = i64::from_le_bytes(data[dpos..dpos + 8].try_into().unwrap());
                dpos += 8;

                // table_name_len (2 bytes)
                let table_len =
                    u16::from_le_bytes(data[dpos..dpos + 2].try_into().unwrap()) as usize;
                dpos += 2;

                // table_name
                let table_name = String::from_utf8_lossy(&data[dpos..dpos + table_len]);
                dpos += table_len;

                // row_id (8 bytes)
                let row_id = i64::from_le_bytes(data[dpos..dpos + 8].try_into().unwrap());
                dpos += 8;

                // op (1 byte) - CORRECT mapping from WALOperationType enum
                let op = data[dpos];
                dpos += 1;

                // timestamp (8 bytes)
                dpos += 8;

                // data_len (4 bytes)
                let data_len =
                    u32::from_le_bytes(data[dpos..dpos + 4].try_into().unwrap()) as usize;

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
                eprintln!(
                    "  Entry {}: LSN={}, txn={}, table='{}', row={}, op={} ({}), data_len={}",
                    entry_num, lsn, txn_id, table_name, row_id, op, op_name, data_len
                );

                pos += entry_size + 4;
                entry_num += 1;
            }
            eprintln!("  Total entries: {}", entry_num);
        }
    }

    eprintln!("\n=== PHASE 2: Reopen and verify ===");

    let db = Database::open(&dsn).unwrap();
    let count: i64 = db.query_one("SELECT COUNT(*) FROM test", ()).unwrap();
    eprintln!("Phase 2 count: {}", count);

    assert_eq!(count, 2);
}
