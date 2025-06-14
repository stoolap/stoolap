/*
Copyright 2025 Stoolap Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package test

import (
	"context"
	"database/sql/driver"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stoolap/stoolap"
	"github.com/stoolap/stoolap/internal/storage"
)

func TestDirtyWriteBatchPrevention(t *testing.T) {
	testCases := []struct {
		name      string
		isolation storage.IsolationLevel
	}{
		{"READ_COMMITTED", storage.ReadCommitted},
		{"SNAPSHOT", storage.SnapshotIsolation},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			db, err := stoolap.Open("memory://")
			if err != nil {
				t.Fatal(err)
			}
			defer db.Close()

			ctx := context.Background()
			err = db.Engine().SetIsolationLevel(tc.isolation)
			if err != nil {
				t.Fatal(err)
			}

			// Create table with unique name to avoid conflicts
			_, err = db.Exec(ctx, "CREATE TABLE dirty_batch_test (id INTEGER PRIMARY KEY, value INTEGER)")
			if err != nil {
				t.Fatal(err)
			}

			// Insert initial rows
			_, err = db.Exec(ctx, "INSERT INTO dirty_batch_test (id, value) VALUES (1, 100), (2, 200), (3, 300)")
			if err != nil {
				t.Fatal(err)
			}

			// Start two transactions
			tx1, err := db.Begin()
			if err != nil {
				t.Fatal(err)
			}

			tx2, err := db.Begin()
			if err != nil {
				t.Fatal(err)
			}

			// Tx1 updates row 2 but doesn't commit
			_, err = tx1.ExecContext(ctx, "UPDATE dirty_batch_test SET value = 250 WHERE id = 2")
			if err != nil {
				t.Fatal(err)
			}
			fmt.Printf("[%s] Tx1 updated row 2 to 250 (not committed)\n", tc.name)

			// Tx2 tries to insert multiple rows, including one that would update row 2
			_, err = tx2.ExecContext(ctx, "INSERT INTO dirty_batch_test (id, value) VALUES (2, 999), (4, 400), (5, 500)")
			if err != nil {
				fmt.Printf("[%s] Tx2 batch insert blocked/failed as expected: %v\n", tc.name, err)
			} else {
				t.Errorf("[%s] DIRTY WRITE: Tx2 was able to insert over uncommitted data!", tc.name)
			}

			// Clean up
			tx1.Rollback()
			tx2.Rollback()
		})
	}
}

func TestDirtyWriteBatchUpdate(t *testing.T) {
	db, err := stoolap.Open("memory://")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	ctx := context.Background()
	err = db.Engine().SetIsolationLevel(storage.SnapshotIsolation)
	if err != nil {
		t.Fatal(err)
	}

	// Create table with unique name to avoid conflicts
	_, err = db.Exec(ctx, "CREATE TABLE dirty_batch_update_test (id INTEGER PRIMARY KEY, value INTEGER)")
	if err != nil {
		t.Fatal(err)
	}

	// Insert initial rows
	_, err = db.Exec(ctx, "INSERT INTO dirty_batch_update_test (id, value) VALUES (1, 100), (2, 200), (3, 300), (4, 400), (5, 500)")
	if err != nil {
		t.Fatal(err)
	}

	// Start two transactions
	tx1, err := db.Begin()
	if err != nil {
		t.Fatal(err)
	}

	tx2, err := db.Begin()
	if err != nil {
		t.Fatal(err)
	}

	// Tx1 updates rows 2 and 3 but doesn't commit
	_, err = tx1.ExecContext(ctx, "UPDATE dirty_batch_update_test SET value = value + 1000 WHERE id IN (2, 3)")
	if err != nil {
		t.Fatal(err)
	}
	fmt.Println("Tx1 updated rows 2,3 (not committed)")

	// Tx2 tries to update rows 3,4,5 (row 3 conflicts)
	_, err = tx2.ExecContext(ctx, "UPDATE dirty_batch_update_test SET value = value + 2000 WHERE id IN (3, 4, 5)")
	if err != nil {
		fmt.Printf("Tx2 batch update failed as expected: %v\n", err)
	} else {
		// Check what values Tx2 sees
		rows, _ := tx2.QueryContext(ctx, "SELECT id, value FROM test WHERE id IN (3, 4, 5) ORDER BY id")
		defer rows.Close()

		fmt.Println("Tx2 sees:")
		for rows.Next() {
			var id, value int
			rows.Scan(&id, &value)
			fmt.Printf("  Row %d: value=%d\n", id, value)
		}

		t.Error("DIRTY WRITE: Tx2 was able to update rows with uncommitted changes!")
	}

	// Clean up
	tx1.Rollback()
	tx2.Rollback()
}

// TestDirtyWriteBatchTransfer tests a simple two-account transfer scenario
func TestDirtyWriteBatchTransfer(t *testing.T) {
	testCases := []struct {
		name      string
		isolation storage.IsolationLevel
	}{
		{"READ_COMMITTED", storage.ReadCommitted},
		{"SNAPSHOT", storage.SnapshotIsolation},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			db, err := stoolap.Open("memory://")
			if err != nil {
				t.Fatal(err)
			}
			defer db.Close()

			ctx := context.Background()
			err = db.Engine().SetIsolationLevel(tc.isolation)
			if err != nil {
				t.Fatal(err)
			}

			// Create table with two accounts - use unique name to avoid conflicts
			_, err = db.Exec(ctx, "CREATE TABLE batch_test_accounts (id INTEGER PRIMARY KEY, balance INTEGER)")
			if err != nil {
				t.Fatal(err)
			}

			// Insert two accounts with 1000 each
			_, err = db.Exec(ctx, "INSERT INTO batch_test_accounts (id, balance) VALUES (1, 1000), (2, 1000)")
			if err != nil {
				t.Fatal(err)
			}

			// Verify initial state
			var totalBalance int
			rows, _ := db.Query(ctx, "SELECT SUM(balance) FROM batch_test_accounts")
			if rows.Next() {
				rows.Scan(&totalBalance)
				rows.Close()
			}
			if totalBalance != 2000 {
				t.Fatalf("Initial balance incorrect: %d", totalBalance)
			}

			// Run concurrent transfers
			var wg sync.WaitGroup
			successCount := 0
			failCount := 0
			var mu sync.Mutex

			for i := 0; i < 10; i++ {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()

					// Start transaction
					tx, err := db.Begin()
					if err != nil {
						mu.Lock()
						failCount++
						mu.Unlock()
						return
					}

					// Try to transfer 100 from account 1 to account 2
					success := true

					// Deduct from account 1
					_, err = tx.ExecContext(ctx,
						"UPDATE batch_test_accounts SET balance = balance - ? WHERE id = ?",
						driver.NamedValue{Ordinal: 1, Value: 100},
						driver.NamedValue{Ordinal: 2, Value: 1})
					if err != nil {
						success = false
						t.Logf("Worker %d: Failed to deduct: %v", id, err)
					}

					// Add delay to increase chance of conflicts
					time.Sleep(10 * time.Millisecond)

					// Add to account 2
					if success {
						_, err = tx.ExecContext(ctx,
							"UPDATE batch_test_accounts SET balance = balance + ? WHERE id = ?",
							driver.NamedValue{Ordinal: 1, Value: 100},
							driver.NamedValue{Ordinal: 2, Value: 2})
						if err != nil {
							success = false
							t.Logf("Worker %d: Failed to credit: %v", id, err)
						}
					}

					// Commit or rollback
					if success {
						err = tx.Commit()
						if err != nil {
							t.Logf("Worker %d: Commit failed: %v", id, err)
							mu.Lock()
							failCount++
							mu.Unlock()
						} else {
							mu.Lock()
							successCount++
							mu.Unlock()
						}
					} else {
						tx.Rollback()
						mu.Lock()
						failCount++
						mu.Unlock()
					}
				}(i)
			}

			wg.Wait()

			// Check final balance
			rows, _ = db.Query(ctx, "SELECT SUM(balance) FROM batch_test_accounts")
			if rows.Next() {
				rows.Scan(&totalBalance)
				rows.Close()
			}

			// Check individual balances
			var bal1, bal2 int
			rows, _ = db.Query(ctx, "SELECT balance FROM batch_test_accounts WHERE id = 1")
			if rows.Next() {
				rows.Scan(&bal1)
				rows.Close()
			}
			rows, _ = db.Query(ctx, "SELECT balance FROM batch_test_accounts WHERE id = 2")
			if rows.Next() {
				rows.Scan(&bal2)
				rows.Close()
			}

			fmt.Printf("[%s] Results:\n", tc.name)
			fmt.Printf("  Successful transfers: %d\n", successCount)
			fmt.Printf("  Failed transfers: %d\n", failCount)
			fmt.Printf("  Account 1 balance: %d\n", bal1)
			fmt.Printf("  Account 2 balance: %d\n", bal2)
			fmt.Printf("  Total balance: %d (expected: 2000)\n", totalBalance)

			if totalBalance != 2000 {
				t.Errorf("Money not conserved! Total balance: %d, expected: 2000", totalBalance)
			}
		})
	}
}
