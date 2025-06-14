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
	"testing"

	"github.com/stoolap/stoolap/internal/sql/executor"
	"github.com/stoolap/stoolap/internal/storage"
	"github.com/stoolap/stoolap/internal/storage/mvcc"
)

// TestDeleteWithINSubquery tests DELETE statements with IN subqueries
func TestDeleteWithINSubquery(t *testing.T) {
	// Create test engine
	cfg := &storage.Config{
		Path: "",
		Persistence: storage.PersistenceConfig{
			Enabled: false,
		},
	}
	eng := mvcc.NewMVCCEngine(cfg)
	if err := eng.Open(); err != nil {
		t.Fatalf("Failed to open MVCC engine: %v", err)
	}
	defer eng.Close()

	// Create executor
	exec := executor.NewExecutor(eng)
	ctx := context.Background()

	// Helper function to execute SQL
	executeSQL := func(query string) error {
		tx, err := eng.BeginTransaction()
		if err != nil {
			return err
		}
		defer tx.Rollback()

		// Execute the query
		_, err = exec.Execute(ctx, tx, query)
		if err != nil {
			return err
		}

		return tx.Commit()
	}

	// Helper function to query and verify
	queryAndVerify := func(query string) ([][]interface{}, error) {
		tx, err := eng.BeginTransaction()
		if err != nil {
			return nil, err
		}
		defer tx.Rollback()

		// Execute the query
		result, err := exec.Execute(ctx, tx, query)
		if err != nil {
			return nil, err
		}
		defer result.Close()

		// Collect results
		var rows [][]interface{}
		for result.Next() {
			row := result.Row()
			values := make([]interface{}, len(row))
			for i, col := range row {
				values[i] = col.AsInterface()
			}
			rows = append(rows, values)
		}

		return rows, nil
	}

	// Create customers table
	err := executeSQL(`
		CREATE TABLE customers (
			id INTEGER PRIMARY KEY,
			name TEXT,
			total_spent FLOAT
		)
	`)
	if err != nil {
		t.Fatalf("Failed to create customers table: %v", err)
	}

	// Create orders table
	err = executeSQL(`
		CREATE TABLE orders (
			id INTEGER PRIMARY KEY,
			customer_id INTEGER,
			amount FLOAT
		)
	`)
	if err != nil {
		t.Fatalf("Failed to create orders table: %v", err)
	}

	// Insert test data - customers
	err = executeSQL(`
		INSERT INTO customers (id, name, total_spent) VALUES 
		(1, 'Alice', 1000.0),
		(2, 'Bob', 500.0),
		(3, 'Charlie', 2000.0),
		(4, 'David', 100.0)
	`)
	if err != nil {
		t.Fatalf("Failed to insert customers: %v", err)
	}

	// Insert test data - orders
	err = executeSQL(`
		INSERT INTO orders (id, customer_id, amount) VALUES 
		(1, 1, 200.0),
		(2, 2, 100.0),
		(3, 3, 500.0),
		(4, 1, 300.0),
		(5, 3, 400.0),
		(6, 4, 50.0)
	`)
	if err != nil {
		t.Fatalf("Failed to insert orders: %v", err)
	}

	// Test 1: Delete orders for customers with total_spent > 1000
	err = executeSQL(`
		DELETE FROM orders 
		WHERE customer_id IN (
			SELECT id FROM customers WHERE total_spent > 1000
		)
	`)
	if err != nil {
		t.Fatalf("Failed to execute DELETE with IN subquery: %v", err)
	}

	// Verify only orders for customer 3 (Charlie) were deleted
	rows, err := queryAndVerify("SELECT id FROM orders ORDER BY id")
	if err != nil {
		t.Fatalf("Failed to query orders after DELETE: %v", err)
	}

	expectedOrderIDs := []int64{1, 2, 4, 6}
	if len(rows) != len(expectedOrderIDs) {
		t.Errorf("Expected %d orders, got %d", len(expectedOrderIDs), len(rows))
	}
	for i, row := range rows {
		if i >= len(expectedOrderIDs) {
			break
		}
		id := row[0].(int64)
		if id != expectedOrderIDs[i] {
			t.Errorf("Expected order id %d, got %d", expectedOrderIDs[i], id)
		}
	}

	// Test 2: Delete customers who have no orders (after previous deletion)
	err = executeSQL(`
		DELETE FROM customers 
		WHERE id NOT IN (
			SELECT DISTINCT customer_id FROM orders
		)
	`)
	if err != nil {
		t.Fatalf("Failed to execute DELETE with NOT IN subquery: %v", err)
	}

	// Verify customer 3 (Charlie) was deleted
	rows, err = queryAndVerify("SELECT id, name FROM customers ORDER BY id")
	if err != nil {
		t.Fatalf("Failed to query customers after DELETE: %v", err)
	}

	expectedCustomers := []struct {
		id   int64
		name string
	}{
		{1, "Alice"},
		{2, "Bob"},
		{4, "David"},
	}

	if len(rows) != len(expectedCustomers) {
		t.Errorf("Expected %d customers, got %d", len(expectedCustomers), len(rows))
	}
	for i, row := range rows {
		if i >= len(expectedCustomers) {
			break
		}
		id := row[0].(int64)
		name := row[1].(string)
		if id != expectedCustomers[i].id || name != expectedCustomers[i].name {
			t.Errorf("Expected customer %d:%s, got %d:%s",
				expectedCustomers[i].id, expectedCustomers[i].name, id, name)
		}
	}
}
