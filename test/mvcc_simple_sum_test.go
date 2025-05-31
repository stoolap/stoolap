package test

import (
	"database/sql"
	"testing"

	_ "github.com/stoolap/stoolap/pkg/driver"
)

// TestSimpleSumIsolation tests basic SUM behavior
func TestSimpleSumIsolation(t *testing.T) {
	db, err := sql.Open("stoolap", "memory://")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	// Create simple table
	_, err = db.Exec(`CREATE TABLE test (id INT PRIMARY KEY, val INT)`)
	if err != nil {
		t.Fatal(err)
	}

	// Insert data
	_, err = db.Exec(`INSERT INTO test VALUES (1, 100), (2, 200)`)
	if err != nil {
		t.Fatal(err)
	}

	// Test 1: Simple SUM
	var sum int
	err = db.QueryRow("SELECT SUM(val) FROM test").Scan(&sum)
	if err != nil {
		t.Fatal(err)
	}
	if sum != 300 {
		t.Errorf("Expected sum 300, got %d", sum)
	}

	// Test 2: SUM in transaction
	tx, err := db.Begin()
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback()

	var txSum int
	err = tx.QueryRow("SELECT SUM(val) FROM test").Scan(&txSum)
	if err != nil {
		t.Fatal(err)
	}
	if txSum != 300 {
		t.Errorf("Expected sum 300 in transaction, got %d", txSum)
	}
}

// TestSumAfterUpdate tests SUM after updates
func TestSumAfterUpdate(t *testing.T) {
	db, err := sql.Open("stoolap", "memory://")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	// Setup
	_, err = db.Exec(`CREATE TABLE test (id INT PRIMARY KEY, val INT)`)
	if err != nil {
		t.Fatal(err)
	}
	_, err = db.Exec(`INSERT INTO test VALUES (1, 100), (2, 200)`)
	if err != nil {
		t.Fatal(err)
	}

	// Update one row
	_, err = db.Exec("UPDATE test SET val = 150 WHERE id = 1")
	if err != nil {
		t.Fatal(err)
	}

	// Check sum
	var sum int
	err = db.QueryRow("SELECT SUM(val) FROM test").Scan(&sum)
	if err != nil {
		t.Fatal(err)
	}
	if sum != 350 { // 150 + 200
		t.Errorf("Expected sum 350 after update, got %d", sum)
	}
}
