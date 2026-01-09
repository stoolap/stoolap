#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use stoolap::api::Database;

/// Transaction operations
#[derive(Debug, Arbitrary, Clone)]
enum TxOp {
    /// Begin a new transaction
    Begin,

    /// Commit current transaction
    Commit,

    /// Rollback current transaction
    Rollback,

    /// Insert a row
    Insert { id: u16, value: i16 },

    /// Update rows
    Update { target_id: u8, new_value: i16 },

    /// Delete rows
    Delete { target_id: u8 },

    /// Select to verify isolation
    Select { filter_id: u8 },

    /// Select with aggregation
    SelectAggregate,

    /// Select count
    SelectCount,

    /// Create savepoint (if supported)
    Savepoint { name: u8 },

    /// Rollback to savepoint
    RollbackToSavepoint { name: u8 },
}

impl TxOp {
    fn execute(&self, db: &Database, tx_active: &mut bool) {
        match self {
            TxOp::Begin => {
                if !*tx_active {
                    if db.execute("BEGIN", ()).is_ok() {
                        *tx_active = true;
                    }
                }
            }

            TxOp::Commit => {
                if *tx_active {
                    let _ = db.execute("COMMIT", ());
                    *tx_active = false;
                }
            }

            TxOp::Rollback => {
                if *tx_active {
                    let _ = db.execute("ROLLBACK", ());
                    *tx_active = false;
                }
            }

            TxOp::Insert { id, value } => {
                // Use modulo to keep IDs in a reasonable range and avoid too many conflicts
                let actual_id = (*id as i32 % 10000) + 1000;
                let _ = db.execute(
                    &format!(
                        "INSERT INTO txtest (id, value, status) VALUES ({}, {}, 'active')",
                        actual_id, value
                    ),
                    (),
                );
            }

            TxOp::Update { target_id, new_value } => {
                let _ = db.execute(
                    &format!(
                        "UPDATE txtest SET value = {} WHERE id % 100 = {}",
                        new_value,
                        target_id % 100
                    ),
                    (),
                );
            }

            TxOp::Delete { target_id } => {
                let _ = db.execute(
                    &format!("DELETE FROM txtest WHERE id % 50 = {}", target_id % 50),
                    (),
                );
            }

            TxOp::Select { filter_id } => {
                let _ = db.query(
                    &format!("SELECT * FROM txtest WHERE id % 20 = {}", filter_id % 20),
                    (),
                );
            }

            TxOp::SelectAggregate => {
                let _ = db.query("SELECT status, SUM(value), COUNT(*) FROM txtest GROUP BY status", ());
            }

            TxOp::SelectCount => {
                let _ = db.query("SELECT COUNT(*) FROM txtest", ());
            }

            TxOp::Savepoint { name } => {
                if *tx_active {
                    let _ = db.execute(&format!("SAVEPOINT sp_{}", name % 10), ());
                }
            }

            TxOp::RollbackToSavepoint { name } => {
                if *tx_active {
                    let _ = db.execute(&format!("ROLLBACK TO SAVEPOINT sp_{}", name % 10), ());
                }
            }
        }
    }
}

/// Isolation level for transactions
#[derive(Debug, Arbitrary)]
enum IsolationLevel {
    ReadCommitted,
    RepeatableRead,
    Serializable,
}

impl IsolationLevel {
    fn sql(&self) -> &'static str {
        match self {
            IsolationLevel::ReadCommitted => "READ COMMITTED",
            IsolationLevel::RepeatableRead => "REPEATABLE READ",
            IsolationLevel::Serializable => "SERIALIZABLE",
        }
    }
}

/// Test scenario types
#[derive(Debug, Arbitrary)]
enum Scenario {
    /// Simple sequence of operations
    SimpleSequence {
        ops: Vec<TxOp>,
    },

    /// Interleaved reads and writes
    InterleavedReadWrite {
        writes: Vec<TxOp>,
        reads: Vec<TxOp>,
    },

    /// Multiple transactions pattern (simulated sequentially)
    MultipleTransactions {
        tx1_ops: Vec<TxOp>,
        tx2_ops: Vec<TxOp>,
    },

    /// Stress test with many small transactions
    ManySmallTransactions {
        count: u8,
        op_per_tx: u8,
    },

    /// Large transaction with many operations
    LargeTransaction {
        ops: Vec<TxOp>,
    },

    /// Rollback scenarios
    RollbackHeavy {
        ops: Vec<TxOp>,
        rollback_points: Vec<u8>,
    },

    /// Read-only transaction pattern
    ReadOnlyPattern {
        setup_ops: Vec<TxOp>,
        read_ops: Vec<TxOp>,
    },

    /// Write-heavy pattern
    WriteHeavy {
        inserts: Vec<(u16, i16)>,
        updates: Vec<(u8, i16)>,
        deletes: Vec<u8>,
    },
}

impl Scenario {
    fn execute(&self, db: &Database) {
        let mut tx_active = false;

        match self {
            Scenario::SimpleSequence { ops } => {
                for op in ops.iter().take(50) {
                    op.execute(db, &mut tx_active);
                }
                // Cleanup: commit or rollback any pending transaction
                if tx_active {
                    let _ = db.execute("COMMIT", ());
                }
            }

            Scenario::InterleavedReadWrite { writes, reads } => {
                let _ = db.execute("BEGIN", ());
                tx_active = true;

                // Interleave writes and reads
                let max_ops = writes.len().max(reads.len()).min(30);
                for i in 0..max_ops {
                    if let Some(write_op) = writes.get(i) {
                        write_op.execute(db, &mut tx_active);
                    }
                    if let Some(read_op) = reads.get(i) {
                        read_op.execute(db, &mut tx_active);
                    }
                }

                if tx_active {
                    let _ = db.execute("COMMIT", ());
                }
            }

            Scenario::MultipleTransactions { tx1_ops, tx2_ops } => {
                // Execute first transaction
                let _ = db.execute("BEGIN", ());
                tx_active = true;
                for op in tx1_ops.iter().take(20) {
                    op.execute(db, &mut tx_active);
                }
                if tx_active {
                    let _ = db.execute("COMMIT", ());
                    tx_active = false;
                }

                // Execute second transaction
                let _ = db.execute("BEGIN", ());
                tx_active = true;
                for op in tx2_ops.iter().take(20) {
                    op.execute(db, &mut tx_active);
                }
                if tx_active {
                    let _ = db.execute("COMMIT", ());
                }
            }

            Scenario::ManySmallTransactions { count, op_per_tx } => {
                let actual_count = (*count).min(20) as usize;
                let ops_per = (*op_per_tx).max(1).min(10) as usize;

                for i in 0..actual_count {
                    let _ = db.execute("BEGIN", ());
                    tx_active = true;

                    for j in 0..ops_per {
                        let id = ((i * 100 + j) % 5000) as u16;
                        let value = (i as i16 * 10) + (j as i16);
                        TxOp::Insert { id, value }.execute(db, &mut tx_active);
                    }

                    // Randomly commit or rollback based on index
                    if i % 3 == 0 {
                        let _ = db.execute("ROLLBACK", ());
                    } else {
                        let _ = db.execute("COMMIT", ());
                    }
                    tx_active = false;
                }
            }

            Scenario::LargeTransaction { ops } => {
                let _ = db.execute("BEGIN", ());
                tx_active = true;

                for op in ops.iter().take(100) {
                    op.execute(db, &mut tx_active);
                    if !tx_active {
                        // Transaction was committed/rolled back, start a new one
                        let _ = db.execute("BEGIN", ());
                        tx_active = true;
                    }
                }

                if tx_active {
                    let _ = db.execute("COMMIT", ());
                }
            }

            Scenario::RollbackHeavy { ops, rollback_points } => {
                let _ = db.execute("BEGIN", ());
                tx_active = true;

                let rollback_set: std::collections::HashSet<u8> =
                    rollback_points.iter().take(10).map(|r| r % 50).collect();

                for (i, op) in ops.iter().take(50).enumerate() {
                    op.execute(db, &mut tx_active);

                    if rollback_set.contains(&(i as u8)) && tx_active {
                        let _ = db.execute("ROLLBACK", ());
                        tx_active = false;
                        // Start new transaction
                        let _ = db.execute("BEGIN", ());
                        tx_active = true;
                    }
                }

                if tx_active {
                    let _ = db.execute("COMMIT", ());
                }
            }

            Scenario::ReadOnlyPattern { setup_ops, read_ops } => {
                // Setup phase
                let _ = db.execute("BEGIN", ());
                tx_active = true;
                for op in setup_ops.iter().take(20) {
                    if matches!(op, TxOp::Insert { .. } | TxOp::Update { .. }) {
                        op.execute(db, &mut tx_active);
                    }
                }
                if tx_active {
                    let _ = db.execute("COMMIT", ());
                    tx_active = false;
                }

                // Read phase
                let _ = db.execute("BEGIN", ());
                tx_active = true;
                for op in read_ops.iter().take(30) {
                    if matches!(op, TxOp::Select { .. } | TxOp::SelectAggregate | TxOp::SelectCount) {
                        op.execute(db, &mut tx_active);
                    }
                }
                if tx_active {
                    let _ = db.execute("COMMIT", ());
                }
            }

            Scenario::WriteHeavy {
                inserts,
                updates,
                deletes,
            } => {
                let _ = db.execute("BEGIN", ());
                tx_active = true;

                // Batch inserts
                for (id, value) in inserts.iter().take(30) {
                    TxOp::Insert {
                        id: *id,
                        value: *value,
                    }
                    .execute(db, &mut tx_active);
                }

                // Batch updates
                for (target_id, new_value) in updates.iter().take(20) {
                    TxOp::Update {
                        target_id: *target_id,
                        new_value: *new_value,
                    }
                    .execute(db, &mut tx_active);
                }

                // Batch deletes
                for target_id in deletes.iter().take(10) {
                    TxOp::Delete {
                        target_id: *target_id,
                    }
                    .execute(db, &mut tx_active);
                }

                if tx_active {
                    let _ = db.execute("COMMIT", ());
                }
            }
        }
    }
}

fn setup_database() -> Option<Database> {
    let db = Database::open_in_memory().ok()?;

    // Create test table
    db.execute(
        "CREATE TABLE txtest (
            id INTEGER PRIMARY KEY,
            value INTEGER,
            status TEXT
        )",
        (),
    )
    .ok()?;

    // Create index for faster queries
    let _ = db.execute("CREATE INDEX idx_txtest_status ON txtest(status)", ());
    let _ = db.execute("CREATE INDEX idx_txtest_value ON txtest(value)", ());

    // Insert initial data
    for i in 1..=100 {
        let _ = db.execute(
            &format!(
                "INSERT INTO txtest VALUES ({}, {}, '{}')",
                i,
                i * 10,
                if i % 2 == 0 { "active" } else { "inactive" }
            ),
            (),
        );
    }

    Some(db)
}

fuzz_target!(|data: &[u8]| {
    let mut unstructured = Unstructured::new(data);

    if let Ok(scenario) = Scenario::arbitrary(&mut unstructured) {
        if let Some(db) = setup_database() {
            // Execute the scenario - should never panic
            scenario.execute(&db);

            // Verify database is still consistent by running a final query
            let _ = db.query("SELECT COUNT(*) FROM txtest", ());
        }
    }
});
