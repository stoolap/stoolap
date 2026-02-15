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

use stoolap::Database;

fn setup() -> Database {
    Database::open_in_memory().expect("Failed to create in-memory database")
}

fn setup_parent_child(db: &Database) {
    db.execute(
        "CREATE TABLE parents (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO parents VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')",
        (),
    )
    .unwrap();
}

fn count(db: &Database, sql: &str) -> i64 {
    let mut rows = db.query(sql, ()).unwrap();
    if let Some(row) = rows.next() {
        let row = row.unwrap();
        return row.get::<i64>(0).unwrap();
    }
    0
}

// =====================================================================
// INSERT tests
// =====================================================================

#[test]
fn test_fk_insert_valid_parent() {
    let db = setup();
    setup_parent_child(&db);
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id), name TEXT)",
        (),
    ).unwrap();

    // Insert with valid parent reference
    db.execute("INSERT INTO children VALUES (1, 1, 'Child1')", ())
        .unwrap();
    db.execute("INSERT INTO children VALUES (2, 2, 'Child2')", ())
        .unwrap();

    assert_eq!(count(&db, "SELECT COUNT(*) FROM children"), 2);
}

#[test]
fn test_fk_insert_invalid_parent() {
    let db = setup();
    setup_parent_child(&db);
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id), name TEXT)",
        (),
    ).unwrap();

    // Insert with non-existent parent should fail
    let result = db.execute("INSERT INTO children VALUES (1, 999, 'Orphan')", ());
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("foreign key"),
        "Expected FK error, got: {}",
        err
    );
}

#[test]
fn test_fk_insert_null_allowed() {
    let db = setup();
    setup_parent_child(&db);
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id), name TEXT)",
        (),
    ).unwrap();

    // Insert with NULL FK should be allowed (SQL standard)
    db.execute("INSERT INTO children VALUES (1, NULL, 'NoParent')", ())
        .unwrap();
    assert_eq!(count(&db, "SELECT COUNT(*) FROM children"), 1);
}

#[test]
fn test_fk_insert_select_valid() {
    let db = setup();
    setup_parent_child(&db);
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id), name TEXT)",
        (),
    ).unwrap();

    db.execute(
        "CREATE TABLE temp_data (id INTEGER PRIMARY KEY, pid INTEGER, n TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO temp_data VALUES (10, 1, 'Via Select')", ())
        .unwrap();

    let result = db.execute("INSERT INTO children SELECT * FROM temp_data", ());
    assert!(result.is_ok());
}

#[test]
fn test_fk_insert_select_invalid() {
    let db = setup();
    setup_parent_child(&db);
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id), name TEXT)",
        (),
    ).unwrap();

    db.execute(
        "CREATE TABLE temp_data (id INTEGER PRIMARY KEY, pid INTEGER, n TEXT)",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO temp_data VALUES (10, 999, 'Bad Ref')", ())
        .unwrap();

    let result = db.execute("INSERT INTO children SELECT * FROM temp_data", ());
    assert!(result.is_err());
}

// =====================================================================
// DELETE with RESTRICT (default)
// =====================================================================

#[test]
fn test_fk_delete_restrict_blocks() {
    let db = setup();
    setup_parent_child(&db);
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id), name TEXT)",
        (),
    ).unwrap();
    db.execute("INSERT INTO children VALUES (1, 1, 'Child1')", ())
        .unwrap();

    // Deleting parent with existing child should fail (RESTRICT is default)
    let result = db.execute("DELETE FROM parents WHERE id = 1", ());
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("foreign key") || err.contains("referenced"),
        "Expected FK error, got: {}",
        err
    );

    // Parent should still exist
    assert_eq!(count(&db, "SELECT COUNT(*) FROM parents WHERE id = 1"), 1);
}

#[test]
fn test_fk_delete_no_children_ok() {
    let db = setup();
    setup_parent_child(&db);
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id), name TEXT)",
        (),
    ).unwrap();
    db.execute("INSERT INTO children VALUES (1, 1, 'Child1')", ())
        .unwrap();

    // Deleting parent with NO children should succeed
    db.execute("DELETE FROM parents WHERE id = 3", ()).unwrap();
    assert_eq!(count(&db, "SELECT COUNT(*) FROM parents WHERE id = 3"), 0);
}

// =====================================================================
// DELETE with CASCADE
// =====================================================================

#[test]
fn test_fk_delete_cascade() {
    let db = setup();
    setup_parent_child(&db);
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id) ON DELETE CASCADE, item TEXT)",
        (),
    ).unwrap();
    db.execute(
        "INSERT INTO orders VALUES (1, 1, 'Item1'), (2, 1, 'Item2'), (3, 2, 'Item3')",
        (),
    )
    .unwrap();

    // Deleting parent with CASCADE should delete child rows
    db.execute("DELETE FROM parents WHERE id = 1", ()).unwrap();

    assert_eq!(count(&db, "SELECT COUNT(*) FROM parents WHERE id = 1"), 0);
    assert_eq!(
        count(&db, "SELECT COUNT(*) FROM orders WHERE parent_id = 1"),
        0
    );
    assert_eq!(
        count(&db, "SELECT COUNT(*) FROM orders WHERE parent_id = 2"),
        1
    );
}

// =====================================================================
// DELETE with SET NULL
// =====================================================================

#[test]
fn test_fk_delete_set_null() {
    let db = setup();
    setup_parent_child(&db);
    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id) ON DELETE SET NULL, item TEXT)",
        (),
    ).unwrap();
    db.execute(
        "INSERT INTO orders VALUES (1, 1, 'Item1'), (2, 1, 'Item2'), (3, 2, 'Item3')",
        (),
    )
    .unwrap();

    // Deleting parent with SET NULL should set FK to NULL in child rows
    db.execute("DELETE FROM parents WHERE id = 1", ()).unwrap();

    assert_eq!(count(&db, "SELECT COUNT(*) FROM parents WHERE id = 1"), 0);
    assert_eq!(
        count(&db, "SELECT COUNT(*) FROM orders WHERE parent_id IS NULL"),
        2
    );
    assert_eq!(count(&db, "SELECT COUNT(*) FROM orders"), 3);
}

// =====================================================================
// DROP TABLE tests
// =====================================================================

#[test]
fn test_fk_drop_parent_blocked() {
    let db = setup();
    setup_parent_child(&db);
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id), name TEXT)",
        (),
    ).unwrap();
    db.execute("INSERT INTO children VALUES (1, 1, 'Child1')", ())
        .unwrap();

    let result = db.execute("DROP TABLE parents", ());
    assert!(result.is_err());
}

#[test]
fn test_fk_drop_child_ok() {
    let db = setup();
    setup_parent_child(&db);
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id), name TEXT)",
        (),
    ).unwrap();
    db.execute("INSERT INTO children VALUES (1, 1, 'Child1')", ())
        .unwrap();

    db.execute("DROP TABLE children", ()).unwrap();
    assert_eq!(count(&db, "SELECT COUNT(*) FROM parents"), 3);
}

// =====================================================================
// TRUNCATE tests
// =====================================================================

#[test]
fn test_fk_truncate_parent_blocked() {
    let db = setup();
    setup_parent_child(&db);
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id), name TEXT)",
        (),
    ).unwrap();
    db.execute("INSERT INTO children VALUES (1, 1, 'Child1')", ())
        .unwrap();

    let result = db.execute("TRUNCATE TABLE parents", ());
    assert!(result.is_err());
}

#[test]
fn test_fk_truncate_child_ok() {
    let db = setup();
    setup_parent_child(&db);
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id), name TEXT)",
        (),
    ).unwrap();
    db.execute("INSERT INTO children VALUES (1, 1, 'Child1')", ())
        .unwrap();

    db.execute("TRUNCATE TABLE children", ()).unwrap();
}

// =====================================================================
// CREATE TABLE validation tests
// =====================================================================

#[test]
fn test_fk_create_table_invalid_parent() {
    let db = setup();

    let result = db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES nonexistent(id))",
        (),
    );
    assert!(result.is_err());
}

#[test]
fn test_fk_create_table_table_level_constraint() {
    let db = setup();
    setup_parent_child(&db);

    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER, name TEXT, FOREIGN KEY(parent_id) REFERENCES parents(id))",
        (),
    ).unwrap();

    db.execute("INSERT INTO children VALUES (1, 1, 'Child1')", ())
        .unwrap();
    let result = db.execute("INSERT INTO children VALUES (2, 999, 'Bad')", ());
    assert!(result.is_err());
}

#[test]
fn test_fk_create_table_with_actions() {
    let db = setup();
    setup_parent_child(&db);

    db.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, parent_id INTEGER, FOREIGN KEY(parent_id) REFERENCES parents(id) ON DELETE CASCADE ON UPDATE CASCADE)",
        (),
    ).unwrap();

    db.execute("INSERT INTO orders VALUES (1, 1)", ()).unwrap();
    db.execute("DELETE FROM parents WHERE id = 1", ()).unwrap();

    assert_eq!(count(&db, "SELECT COUNT(*) FROM orders"), 0);
}

// =====================================================================
// Multiple FK columns
// =====================================================================

#[test]
fn test_fk_multiple_fk_columns() {
    let db = setup();

    db.execute(
        "CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE managers (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "INSERT INTO departments VALUES (1, 'Engineering'), (2, 'Sales')",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO managers VALUES (10, 'Alice'), (20, 'Bob')", ())
        .unwrap();

    db.execute(
        "CREATE TABLE employees (id INTEGER PRIMARY KEY, dept_id INTEGER REFERENCES departments(id), mgr_id INTEGER REFERENCES managers(id), name TEXT)",
        (),
    ).unwrap();

    // Both FK values valid
    db.execute("INSERT INTO employees VALUES (1, 1, 10, 'Employee1')", ())
        .unwrap();

    // Invalid dept FK
    let result = db.execute("INSERT INTO employees VALUES (2, 999, 10, 'BadDept')", ());
    assert!(result.is_err());

    // Invalid manager FK
    let result = db.execute("INSERT INTO employees VALUES (3, 1, 999, 'BadMgr')", ());
    assert!(result.is_err());

    // One NULL FK (allowed)
    db.execute("INSERT INTO employees VALUES (4, 1, NULL, 'NoMgr')", ())
        .unwrap();
}

// =====================================================================
// WAL persistence test
// =====================================================================

#[test]
fn test_fk_survives_restart() {
    let dir = tempfile::tempdir().unwrap();
    let path = format!("file://{}", dir.path().to_str().unwrap());

    // Create tables with FK in first connection
    {
        let db = Database::open(&path).expect("Failed to open database");
        db.execute(
            "CREATE TABLE parents (id INTEGER PRIMARY KEY, name TEXT)",
            (),
        )
        .unwrap();
        db.execute("INSERT INTO parents VALUES (1, 'Alice')", ())
            .unwrap();
        db.execute(
            "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id), name TEXT)",
            (),
        ).unwrap();
        db.execute("INSERT INTO children VALUES (1, 1, 'Child1')", ())
            .unwrap();
    }

    // Open a new connection - FK constraints should still be enforced
    {
        let db = Database::open(&path).expect("Failed to reopen database");

        db.execute("INSERT INTO children VALUES (2, 1, 'Child2')", ())
            .unwrap();

        let result = db.execute("INSERT INTO children VALUES (3, 999, 'Orphan')", ());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("foreign key"),
            "Expected FK error after restart, got: {}",
            err
        );
    }
}

// =====================================================================
// Transaction rollback tests
// =====================================================================

#[test]
fn test_fk_transaction_rollback() {
    let db = setup();
    setup_parent_child(&db);
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id), name TEXT)",
        (),
    ).unwrap();

    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO children VALUES (1, 1, 'Child1')", ())
        .unwrap();

    let result = db.execute("INSERT INTO children VALUES (2, 999, 'Bad')", ());
    assert!(result.is_err());

    db.execute("ROLLBACK", ()).unwrap();
    assert_eq!(count(&db, "SELECT COUNT(*) FROM children"), 0);
}

// =====================================================================
// ON DELETE NO ACTION test
// =====================================================================

#[test]
fn test_fk_no_action_blocks() {
    let db = setup();
    setup_parent_child(&db);
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id) ON DELETE NO ACTION, name TEXT)",
        (),
    ).unwrap();
    db.execute("INSERT INTO children VALUES (1, 1, 'Child1')", ())
        .unwrap();

    let result = db.execute("DELETE FROM parents WHERE id = 1", ());
    assert!(result.is_err());
}

// =====================================================================
// Non-FK tables unaffected (zero cost)
// =====================================================================

#[test]
fn test_non_fk_table_unaffected() {
    let db = setup();

    db.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    db.execute("INSERT INTO items VALUES (1, 'Item1')", ())
        .unwrap();
    db.execute("INSERT INTO items VALUES (2, 'Item2')", ())
        .unwrap();
    db.execute("UPDATE items SET name = 'Updated' WHERE id = 1", ())
        .unwrap();
    db.execute("DELETE FROM items WHERE id = 2", ()).unwrap();
    db.execute("TRUNCATE TABLE items", ()).unwrap();
    db.execute("DROP TABLE items", ()).unwrap();
}

// =====================================================================
// Default referenced column (PK) test
// =====================================================================

#[test]
fn test_fk_default_references_pk() {
    let db = setup();
    setup_parent_child(&db);

    // References parents without specifying column - defaults to PK
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents, name TEXT)",
        (),
    ).unwrap();

    db.execute("INSERT INTO children VALUES (1, 1, 'Child1')", ())
        .unwrap();
    let result = db.execute("INSERT INTO children VALUES (2, 999, 'Bad')", ());
    assert!(result.is_err());
}

// =====================================================================
// UPDATE FK validation tests
// =====================================================================

#[test]
fn test_fk_update_fk_column_to_valid() {
    let db = setup();
    setup_parent_child(&db);
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id), name TEXT)",
        (),
    ).unwrap();
    db.execute("INSERT INTO children VALUES (1, 1, 'Child1')", ())
        .unwrap();

    // Update FK to another valid parent
    db.execute("UPDATE children SET parent_id = 2 WHERE id = 1", ())
        .unwrap();
    assert_eq!(
        count(&db, "SELECT COUNT(*) FROM children WHERE parent_id = 2"),
        1
    );
}

#[test]
fn test_fk_update_fk_column_to_invalid() {
    let db = setup();
    setup_parent_child(&db);
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id), name TEXT)",
        (),
    ).unwrap();
    db.execute("INSERT INTO children VALUES (1, 1, 'Child1')", ())
        .unwrap();

    // Update FK to non-existent parent should fail
    let result = db.execute("UPDATE children SET parent_id = 999 WHERE id = 1", ());
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("foreign key"),
        "Expected FK error, got: {}",
        err
    );
}

#[test]
fn test_fk_update_non_fk_column_ok() {
    let db = setup();
    setup_parent_child(&db);
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id), name TEXT)",
        (),
    ).unwrap();
    db.execute("INSERT INTO children VALUES (1, 1, 'Child1')", ())
        .unwrap();

    // Updating non-FK column should always succeed (no FK check needed)
    db.execute("UPDATE children SET name = 'Updated' WHERE id = 1", ())
        .unwrap();
}

// =====================================================================
// CASCADE atomicity: rollback undoes cascade effects
// =====================================================================

#[test]
fn test_fk_cascade_rollback_atomicity() {
    let db = setup();
    setup_parent_child(&db);
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id) ON DELETE CASCADE, name TEXT)",
        (),
    ).unwrap();
    db.execute("INSERT INTO children VALUES (1, 1, 'Child1')", ())
        .unwrap();
    db.execute("INSERT INTO children VALUES (2, 1, 'Child2')", ())
        .unwrap();
    db.execute("INSERT INTO children VALUES (3, 2, 'Child3')", ())
        .unwrap();

    // Start explicit transaction, delete parent (CASCADE should delete children 1,2)
    db.execute("BEGIN", ()).unwrap();
    db.execute("DELETE FROM parents WHERE id = 1", ()).unwrap();

    // Rollback — both the parent DELETE and the cascaded child DELETEs must be undone
    db.execute("ROLLBACK", ()).unwrap();

    // All rows should still exist
    assert_eq!(count(&db, "SELECT COUNT(*) FROM parents"), 3);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM children"), 3);
    assert_eq!(
        count(&db, "SELECT COUNT(*) FROM children WHERE parent_id = 1"),
        2
    );
}

// =====================================================================
// FK check sees uncommitted rows (multi-statement transaction)
// =====================================================================

#[test]
fn test_fk_sees_uncommitted_parent_insert() {
    let db = setup();
    db.execute(
        "CREATE TABLE parents (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id), name TEXT)",
        (),
    ).unwrap();

    // Within one transaction: insert parent, then insert child referencing it
    db.execute("BEGIN", ()).unwrap();
    db.execute("INSERT INTO parents VALUES (100, 'NewParent')", ())
        .unwrap();
    // This should succeed because the parent row exists in the current transaction
    db.execute("INSERT INTO children VALUES (1, 100, 'Child')", ())
        .unwrap();
    db.execute("COMMIT", ()).unwrap();

    assert_eq!(
        count(&db, "SELECT COUNT(*) FROM children WHERE parent_id = 100"),
        1
    );
}

// =====================================================================
// CASCADE SET NULL atomicity with rollback
// =====================================================================

#[test]
fn test_fk_set_null_rollback_atomicity() {
    let db = setup();
    setup_parent_child(&db);
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id) ON DELETE SET NULL, name TEXT)",
        (),
    ).unwrap();
    db.execute("INSERT INTO children VALUES (1, 1, 'Child1')", ())
        .unwrap();

    db.execute("BEGIN", ()).unwrap();
    db.execute("DELETE FROM parents WHERE id = 1", ()).unwrap();
    db.execute("ROLLBACK", ()).unwrap();

    // After rollback, the child should still have parent_id = 1 (SET NULL was undone)
    assert_eq!(
        count(&db, "SELECT COUNT(*) FROM children WHERE parent_id = 1"),
        1
    );
    assert_eq!(count(&db, "SELECT COUNT(*) FROM parents WHERE id = 1"), 1);
}

// =====================================================================
// CASCADE UPDATE atomicity with rollback
// =====================================================================

#[test]
fn test_fk_cascade_update_rollback_atomicity() {
    let db = setup();
    setup_parent_child(&db);
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id) ON UPDATE CASCADE, name TEXT)",
        (),
    ).unwrap();
    db.execute("INSERT INTO children VALUES (1, 1, 'Child1')", ())
        .unwrap();

    db.execute("BEGIN", ()).unwrap();
    db.execute("UPDATE parents SET id = 100 WHERE id = 1", ())
        .unwrap();
    db.execute("ROLLBACK", ()).unwrap();

    // After rollback, parent id should still be 1 and child's parent_id should still be 1
    assert_eq!(count(&db, "SELECT COUNT(*) FROM parents WHERE id = 1"), 1);
    assert_eq!(
        count(&db, "SELECT COUNT(*) FROM children WHERE parent_id = 1"),
        1
    );
}

// =====================================================================
// DROP TABLE cleans up orphaned FK constraints in child tables
// =====================================================================

#[test]
fn test_fk_drop_parent_cleans_child_fk() {
    let db = setup();
    setup_parent_child(&db);
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id), name TEXT)",
        (),
    ).unwrap();

    // Delete all child rows so we can drop the parent
    // (no child rows reference the parent — DROP is allowed)
    // Don't insert any children, so the parent has no references

    db.execute("DROP TABLE parents", ()).unwrap();

    // Now insert into child table with any parent_id — FK constraint should be gone
    db.execute("INSERT INTO children VALUES (1, 999, 'Orphan')", ())
        .unwrap();
    assert_eq!(count(&db, "SELECT COUNT(*) FROM children"), 1);
}

// =====================================================================
// Multi-level CASCADE (grandparent → parent → child)
// =====================================================================

#[test]
fn test_fk_multi_level_cascade_delete() {
    let db = setup();

    db.execute(
        "CREATE TABLE grandparents (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE parents (id INTEGER PRIMARY KEY, gp_id INTEGER REFERENCES grandparents(id) ON DELETE CASCADE, name TEXT)", ()).unwrap();
    db.execute("CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id) ON DELETE CASCADE, name TEXT)", ()).unwrap();

    db.execute("INSERT INTO grandparents VALUES (1, 'GP1'), (2, 'GP2')", ())
        .unwrap();
    db.execute(
        "INSERT INTO parents VALUES (10, 1, 'P1'), (20, 1, 'P2'), (30, 2, 'P3')",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO children VALUES (100, 10, 'C1'), (200, 10, 'C2'), (300, 20, 'C3'), (400, 30, 'C4')", ()).unwrap();

    // Delete grandparent 1 → should cascade to parents 10,20 → should cascade to children 100,200,300
    db.execute("DELETE FROM grandparents WHERE id = 1", ())
        .unwrap();

    assert_eq!(count(&db, "SELECT COUNT(*) FROM grandparents"), 1); // GP2 remains
    assert_eq!(count(&db, "SELECT COUNT(*) FROM parents"), 1); // P3 remains
    assert_eq!(count(&db, "SELECT COUNT(*) FROM children"), 1); // C4 remains
    assert_eq!(
        count(&db, "SELECT COUNT(*) FROM children WHERE id = 400"),
        1
    );
}

#[test]
fn test_fk_multi_level_cascade_restrict_blocks() {
    let db = setup();

    // grandparents → parents (CASCADE) → children (RESTRICT)
    db.execute(
        "CREATE TABLE grandparents (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute("CREATE TABLE parents (id INTEGER PRIMARY KEY, gp_id INTEGER REFERENCES grandparents(id) ON DELETE CASCADE, name TEXT)", ()).unwrap();
    db.execute("CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id) ON DELETE RESTRICT, name TEXT)", ()).unwrap();

    db.execute("INSERT INTO grandparents VALUES (1, 'GP1')", ())
        .unwrap();
    db.execute("INSERT INTO parents VALUES (10, 1, 'P1')", ())
        .unwrap();
    db.execute("INSERT INTO children VALUES (100, 10, 'C1')", ())
        .unwrap();

    // Delete grandparent → CASCADE deletes parent 10 → but RESTRICT blocks because child 100 exists
    let result = db.execute("DELETE FROM grandparents WHERE id = 1", ());
    assert!(
        result.is_err(),
        "Should fail because grandchild RESTRICT blocks cascade"
    );
}

// =====================================================================
// DROP TABLE with NULL FK values should not be blocked
// =====================================================================

#[test]
fn test_fk_drop_table_with_null_fk_children() {
    let db = setup();
    setup_parent_child(&db);
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id), name TEXT)",
        (),
    ).unwrap();

    // Insert children with NULL parent_id — these don't reference the parent
    db.execute("INSERT INTO children VALUES (1, NULL, 'Orphan1')", ())
        .unwrap();
    db.execute("INSERT INTO children VALUES (2, NULL, 'Orphan2')", ())
        .unwrap();

    // DROP TABLE should succeed — no child rows actually reference the parent
    db.execute("DROP TABLE parents", ()).unwrap();

    // Children should still exist
    assert_eq!(count(&db, "SELECT COUNT(*) FROM children"), 2);
}

// =====================================================================
// DROP TABLE blocked for CASCADE/SET NULL FK actions (not just RESTRICT)
// =====================================================================

#[test]
fn test_fk_drop_table_blocked_for_cascade_fk() {
    let db = setup();
    db.execute(
        "CREATE TABLE parents (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id) ON DELETE CASCADE, name TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO parents VALUES (1, 'Alice')", ())
        .unwrap();
    db.execute("INSERT INTO children VALUES (10, 1, 'Child1')", ())
        .unwrap();

    // DROP TABLE should be blocked even though child FK is CASCADE —
    // DDL operations don't cascade rows, so child would be orphaned
    let result = db.execute("DROP TABLE parents", ());
    assert!(
        result.is_err(),
        "DROP TABLE should be blocked when child rows with CASCADE FK exist"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("still reference it"),
        "Error should mention referencing rows: {}",
        err
    );

    // After deleting the child rows, DROP should succeed
    db.execute("DELETE FROM children WHERE parent_id = 1", ())
        .unwrap();
    db.execute("DROP TABLE parents", ()).unwrap();
}

#[test]
fn test_fk_drop_table_blocked_for_set_null_fk() {
    let db = setup();
    db.execute(
        "CREATE TABLE parents (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id) ON DELETE SET NULL, name TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO parents VALUES (1, 'Alice')", ())
        .unwrap();
    db.execute("INSERT INTO children VALUES (10, 1, 'Child1')", ())
        .unwrap();

    // DROP TABLE should be blocked even though child FK is SET NULL
    let result = db.execute("DROP TABLE parents", ());
    assert!(
        result.is_err(),
        "DROP TABLE should be blocked when child rows with SET NULL FK exist"
    );

    // Setting FK to NULL manually, then DROP should succeed
    db.execute("UPDATE children SET parent_id = NULL WHERE id = 10", ())
        .unwrap();
    db.execute("DROP TABLE parents", ()).unwrap();
}

// =====================================================================
// DROP TABLE in explicit transaction sees uncommitted child deletes
// =====================================================================

#[test]
fn test_fk_drop_table_sees_uncommitted_child_deletes() {
    let db = setup();
    db.execute(
        "CREATE TABLE parents (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id), name TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO parents VALUES (1, 'Alice')", ())
        .unwrap();
    db.execute("INSERT INTO children VALUES (10, 1, 'Child1')", ())
        .unwrap();

    // In an explicit transaction: delete child rows, then drop parent
    db.execute("BEGIN", ()).unwrap();
    db.execute("DELETE FROM children WHERE parent_id = 1", ())
        .unwrap();

    // DROP TABLE should succeed because the child delete (uncommitted) is visible
    // within the same transaction
    let result = db.execute("DROP TABLE parents", ());
    assert!(
        result.is_ok(),
        "DROP TABLE should succeed after uncommitted child delete in same txn: {:?}",
        result
    );

    db.execute("COMMIT", ()).unwrap();

    // Verify parent is gone
    let result = db.execute("SELECT * FROM parents", ());
    assert!(result.is_err(), "parents table should not exist after DROP");
}

// =====================================================================
// child_rows_exist sees uncommitted child deletes (no false positive)
// =====================================================================

#[test]
fn test_fk_restrict_sees_uncommitted_child_delete() {
    let db = setup();
    db.execute(
        "CREATE TABLE parents (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id) ON DELETE RESTRICT, name TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO parents VALUES (1, 'Alice')", ())
        .unwrap();
    db.execute("INSERT INTO children VALUES (10, 1, 'Child1')", ())
        .unwrap();

    // In an explicit transaction: delete child rows, then delete parent
    db.execute("BEGIN", ()).unwrap();
    db.execute("DELETE FROM children WHERE parent_id = 1", ())
        .unwrap();

    // DELETE parent should succeed because the child delete (uncommitted) is visible
    // within the same transaction — child_rows_exist should return false
    let result = db.execute("DELETE FROM parents WHERE id = 1", ());
    assert!(
        result.is_ok(),
        "DELETE parent should succeed after uncommitted child delete in same txn: {:?}",
        result
    );

    db.execute("COMMIT", ()).unwrap();

    // Verify both are deleted
    assert_eq!(count(&db, "SELECT COUNT(*) FROM parents"), 0);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM children"), 0);
}

// =====================================================================
// parent_row_exists sees uncommitted parent deletes (no false positive)
// =====================================================================

#[test]
fn test_fk_insert_blocked_after_uncommitted_parent_delete() {
    let db = setup();
    db.execute(
        "CREATE TABLE parents (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id), name TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO parents VALUES (1, 'Alice')", ())
        .unwrap();

    // In an explicit transaction: delete parent, then try to insert child
    db.execute("BEGIN", ()).unwrap();
    db.execute("DELETE FROM parents WHERE id = 1", ()).unwrap();

    // INSERT child referencing deleted parent should fail —
    // parent_row_exists should see the uncommitted delete
    let result = db.execute("INSERT INTO children VALUES (10, 1, 'Child1')", ());
    assert!(
        result.is_err(),
        "INSERT child should fail when parent was deleted (uncommitted) in same txn"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("does not exist"),
        "Error should mention parent not existing: {}",
        err
    );

    db.execute("ROLLBACK", ()).unwrap();

    // After rollback, parent is back — insert should succeed
    db.execute("INSERT INTO children VALUES (10, 1, 'Child1')", ())
        .unwrap();
    assert_eq!(count(&db, "SELECT COUNT(*) FROM children"), 1);
}

/// Verifies that an index is automatically created on FK columns at CREATE TABLE time.
#[test]
fn test_fk_auto_index_creation() {
    let db = setup();

    db.execute(
        "CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();

    db.execute(
        "CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            dept_id INTEGER REFERENCES departments(id),
            name TEXT
        )",
        (),
    )
    .unwrap();

    // Check that an FK index was created on the dept_id column
    let rows = db.query("SHOW INDEXES FROM employees", ()).unwrap();
    let mut found_fk_index = false;
    for row in rows {
        let row = row.unwrap();
        let idx_name: String = row.get(1).unwrap(); // index_name is column 1
        if idx_name.contains("fk_") && idx_name.contains("dept_id") {
            found_fk_index = true;
        }
    }
    assert!(
        found_fk_index,
        "FK auto-index should be created on dept_id column"
    );
}

/// Verifies that FK auto-index is NOT created when the FK column already has a UNIQUE constraint.
#[test]
fn test_fk_auto_index_skips_unique_column() {
    let db = setup();

    db.execute(
        "CREATE TABLE parents (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();

    db.execute(
        "CREATE TABLE children (
            id INTEGER PRIMARY KEY,
            parent_id INTEGER UNIQUE REFERENCES parents(id),
            name TEXT
        )",
        (),
    )
    .unwrap();

    // parent_id already has a UNIQUE index — no FK index should be created
    let rows = db.query("SHOW INDEXES FROM children", ()).unwrap();
    let mut found_fk_index = false;
    for row in rows {
        let row = row.unwrap();
        let idx_name: String = row.get(1).unwrap(); // index_name is column 1
        if idx_name.starts_with("fk_") {
            found_fk_index = true;
        }
    }
    assert!(
        !found_fk_index,
        "FK auto-index should NOT be created when column already has UNIQUE index"
    );
}

/// Verifies CASCADE DELETE performance is fast with auto-created FK index.
#[test]
fn test_fk_cascade_delete_performance_with_auto_index() {
    let db = setup();

    db.execute(
        "CREATE TABLE categories (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();

    db.execute(
        "CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            cat_id INTEGER REFERENCES categories(id) ON DELETE CASCADE,
            name TEXT
        )",
        (),
    )
    .unwrap();

    // Insert parent and many children
    for i in 1..=10 {
        db.execute(
            &format!("INSERT INTO categories VALUES ({}, 'Cat{}')", i, i),
            (),
        )
        .unwrap();
    }
    for i in 1..=1000 {
        let cat_id = (i % 10) + 1;
        db.execute(
            &format!(
                "INSERT INTO products VALUES ({}, {}, 'Prod{}')",
                i, cat_id, i
            ),
            (),
        )
        .unwrap();
    }

    assert_eq!(count(&db, "SELECT COUNT(*) FROM products"), 1000);

    // CASCADE DELETE — should be fast with the auto-created FK index
    let start = std::time::Instant::now();
    db.execute("DELETE FROM categories WHERE id = 1", ())
        .unwrap();
    let elapsed = start.elapsed();

    // 100 child rows should be cascaded
    assert_eq!(count(&db, "SELECT COUNT(*) FROM products"), 900);
    assert_eq!(count(&db, "SELECT COUNT(*) FROM categories"), 9);

    // With FK index this should be well under 100ms
    assert!(
        elapsed.as_millis() < 100,
        "CASCADE DELETE took {}ms — expected <100ms with FK index",
        elapsed.as_millis()
    );
}

/// SET NULL on a NOT NULL FK column must be rejected at CREATE TABLE time,
/// not deferred to a confusing runtime error.
#[test]
fn test_fk_set_null_on_not_null_column_rejected() {
    let db = setup();
    setup_parent_child(&db);

    let result = db.execute(
        "CREATE TABLE bad_child (
            id INTEGER PRIMARY KEY,
            parent_id INTEGER NOT NULL REFERENCES parents(id) ON DELETE SET NULL,
            name TEXT
        )",
        (),
    );
    assert!(
        result.is_err(),
        "SET NULL on NOT NULL column must be rejected at CREATE TABLE time"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("SET NULL") && err.contains("NOT NULL"),
        "error should mention SET NULL and NOT NULL, got: {}",
        err
    );
}

/// Multi-level CASCADE + RESTRICT: RESTRICT failure must not leave partial
/// cascade deletes in the transaction state.
#[test]
fn test_fk_cascade_restrict_no_partial_state() {
    let db = setup();

    // grandparent → parent (CASCADE) → child (RESTRICT)
    db.execute("CREATE TABLE gp (id INTEGER PRIMARY KEY, name TEXT)", ())
        .unwrap();
    db.execute(
        "CREATE TABLE par (id INTEGER PRIMARY KEY, gp_id INTEGER REFERENCES gp(id) ON DELETE CASCADE, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE ch (id INTEGER PRIMARY KEY, par_id INTEGER REFERENCES par(id) ON DELETE RESTRICT, name TEXT)",
        (),
    )
    .unwrap();

    db.execute("INSERT INTO gp VALUES (1, 'G1')", ()).unwrap();
    db.execute("INSERT INTO par VALUES (10, 1, 'P1')", ())
        .unwrap();
    db.execute("INSERT INTO ch VALUES (100, 10, 'C1')", ())
        .unwrap();

    // In an explicit transaction, deleting grandparent should fail due to RESTRICT on ch
    db.execute("BEGIN", ()).unwrap();
    let result = db.execute("DELETE FROM gp WHERE id = 1", ());
    assert!(result.is_err(), "RESTRICT on grandchild must block");

    // Crucially: par row must NOT have been deleted in the transaction state
    let par_count = count(&db, "SELECT COUNT(*) FROM par WHERE id = 10");
    assert_eq!(
        par_count, 1,
        "parent row must still exist — RESTRICT failure should prevent cascade deletes"
    );

    db.execute("ROLLBACK", ()).unwrap();
}

/// Verify that UPDATE with invalid constant FK value does NOT leave dirty state
/// in an explicit transaction. The row should remain unchanged after the error.
#[test]
fn test_fk_update_constant_no_dirty_state_in_explicit_txn() {
    let db = Database::open_in_memory().unwrap();
    db.execute(
        "CREATE TABLE parents (id INTEGER PRIMARY KEY, name TEXT)",
        (),
    )
    .unwrap();
    db.execute(
        "CREATE TABLE children (id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES parents(id))",
        (),
    )
    .unwrap();
    db.execute("INSERT INTO parents VALUES (1, 'p1')", ())
        .unwrap();
    db.execute("INSERT INTO children VALUES (1, 1)", ())
        .unwrap();

    // Explicit transaction: UPDATE with invalid FK should fail cleanly
    db.execute("BEGIN", ()).unwrap();

    let result = db.execute("UPDATE children SET parent_id = 999 WHERE id = 1", ());
    assert!(result.is_err(), "FK violation expected");

    // The row should still have the original value — NOT 999
    let parent_id = count(&db, "SELECT parent_id FROM children WHERE id = 1");
    assert_eq!(
        parent_id, 1,
        "row must be unchanged after failed FK update in explicit transaction"
    );

    db.execute("ROLLBACK", ()).unwrap();
}
